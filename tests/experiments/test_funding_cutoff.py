"""Unit tests for the Matthew-effect funding-cutoff RD experiment.

Covers the prereg_v1 smoke gates that are testable without a live run:
- the application log written by FundingAgency.process_funding_evaluation_results
  reconstructs the cutoff and winners EXACTLY (winners == funded records);
- the RD estimator recovers a known injected effect on synthetic data;
- local-randomization inference behaves under effect / no-effect;
- BH adjustment, KM curve, and the write-path guard.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from utopia.agents.funding_agents import FundingAgency
from utopia.analysis.statistics import benjamini_hochberg
from utopia.analysis.funding_cutoff import guard_write_path, km_curve, local_linear_rd, local_randomization, make_synthetic_rd, cv_bandwidth, placebo_cutoff


# ------------------------------------------------ application log vs winners

def _run_processing(n_apps=10, funding_rate=0.3, lambda_funding=0.0,
                    novelty_penalties=None, log=None):
    apps = [{'applicant_id': f"a{i}"} for i in range(n_apps)]
    ranked = [{'application_id': i, 'applicant_id': f"a{i}", 'rank': i + 1, 'reason': ''}
              for i in range(n_apps)]
    metadata = [{'program_id': 'P', 'apps': apps, 'panel_index': 0}]
    programs = {'P': SimpleNamespace(funding_rate=funding_rate)}
    winners = FundingAgency.process_funding_evaluation_results(
        [(dict(ranked_applications=[dict(r) for r in ranked]), [])], metadata, programs,
        novelty_penalties=novelty_penalties, lambda_funding=lambda_funding,
        application_log=log)
    return winners


def test_application_log_reconstructs_winners_exactly():
    log = []
    winners = _run_processing(n_apps=10, funding_rate=0.3, log=log)
    assert len(log) == 10, 'every application (winners AND losers) must be logged'
    k = max(1, int(10 * 0.3))
    logged_winners = {r['applicant_id'] for r in log if r['funded']}
    assert logged_winners == {w['applicant_id'] for w in winners['P']}
    assert all(r['num_winners'] == k for r in log)
    # positions are complete 1..n and funded == (position <= k)
    assert sorted(r['position'] for r in log) == list(range(1, 11))
    assert all(r['funded'] == (r['position'] <= k) for r in log)
    # running variable has both sides
    xs = [(r['num_winners'] - r['position']) + 0.5 for r in log]
    assert any(x > 0 for x in xs) and any(x < 0 for x in xs)


def test_application_log_matches_intervention_reordering():
    # With a novelty penalty the selection order differs from LLM rank order;
    # the log must reflect the ACTUAL selection order used for the cut.
    log = []
    penalties = {f"a{i}": (1.0 if i < 2 else 0.0) for i in range(6)}
    winners = _run_processing(n_apps=6, funding_rate=0.34, lambda_funding=0.5,
                              novelty_penalties=penalties, log=log)
    logged_winners = {r['applicant_id'] for r in log if r['funded']}
    assert logged_winners == {w['applicant_id'] for w in winners['P']}
    by_pos = sorted(log, key=lambda r: r['position'])
    scores = [r['adjusted_score'] for r in by_pos]
    assert scores == sorted(scores, reverse=True), 'position must follow adjusted score'


def test_application_log_flags_fallback():
    log = []
    apps = [{'applicant_id': f"a{i}"} for i in range(4)]
    metadata = [{'program_id': 'P', 'apps': apps, 'panel_index': 0}]
    programs = {'P': SimpleNamespace(funding_rate=0.5)}
    FundingAgency.process_funding_evaluation_results(
        [(None, [])], metadata, programs, application_log=log)  # LLM failure
    assert len(log) == 4 and all(r['fallback_ranking'] for r in log), \
        'fallback panels must be fully flagged so the analysis can exclude them'


def test_no_logging_when_disabled():
    winners = _run_processing(log=None)
    assert winners['P'], 'behavior unchanged when logging is off'


# ------------------------------------- ranking normalization (crash fix)

def _apps(*applicants):
    return [{'applicant_id': a} for a in applicants]


def _norm(ranked, apps):
    return FundingAgency.normalize_ranked_applications(ranked, apps)


def _assert_complete(normalized, apps):
    assert len(normalized) == len(apps)
    assert sorted(r['application_id'] for r in normalized) == list(range(len(apps)))
    for r in normalized:
        assert r['applicant_id'] == apps[r['application_id']]['applicant_id']


def test_norm_valid_ranking_unchanged():
    apps = _apps('a0', 'a1', 'a2')
    ranked = [{'application_id': i, 'applicant_id': f'a{i}', 'rank': i + 1, 'reason': 'r'}
              for i in (2, 0, 1)]
    normalized, rejects = _norm([dict(r) for r in ranked], apps)
    assert rejects == []
    assert normalized == ranked, 'well-formed input must pass through untouched'


def test_norm_missing_applicant_id_recovered():
    apps = _apps('a0', 'a1')
    normalized, rejects = _norm(
        [{'application_id': 1, 'rank': 1}, {'application_id': 0, 'rank': 2}], apps)
    assert rejects == []
    _assert_complete(normalized, apps)
    assert normalized[0]['applicant_id'] == 'a1'
    assert not any(r.get('imputed_tail') for r in normalized)


def test_norm_missing_application_id_unique_applicant_recovered():
    apps = _apps('a0', 'a1')
    normalized, rejects = _norm(
        [{'applicant_id': 'a1', 'rank': 1}, {'applicant_id': 'a0', 'rank': 2}], apps)
    assert rejects == []
    _assert_complete(normalized, apps)
    assert normalized[0]['application_id'] == 1


def test_norm_missing_application_id_ambiguous_applicant_rejected():
    # a0 submits two applications; an id-less record naming a0 is ambiguous
    apps = _apps('a0', 'a0', 'a1')
    normalized, rejects = _norm(
        [{'applicant_id': 'a0', 'rank': 1},
         {'application_id': 2, 'applicant_id': 'a1', 'rank': 2}], apps)
    assert [r['reason'] for r in rejects] == ['ambiguous_applicant_multiple_applications']
    _assert_complete(normalized, apps)
    # both of a0's applications land in the fallback tail with REAL ids
    tail = [r for r in normalized if r.get('imputed_tail')]
    assert sorted(r['application_id'] for r in tail) == [0, 1]
    assert all(r['application_id'] != -1 for r in normalized)


def test_norm_same_applicant_two_valid_applications_preserved():
    apps = _apps('a0', 'a0', 'a1')
    ranked = [{'application_id': 0, 'applicant_id': 'a0', 'rank': 1},
              {'application_id': 1, 'applicant_id': 'a0', 'rank': 2},
              {'application_id': 2, 'applicant_id': 'a1', 'rank': 3}]
    normalized, rejects = _norm(ranked, apps)
    assert rejects == []
    _assert_complete(normalized, apps)
    assert sum(r['applicant_id'] == 'a0' for r in normalized) == 2
    assert not any(r.get('imputed_tail') for r in normalized)


def test_norm_invalid_application_id():
    apps = _apps('a0', 'a1')
    normalized, rejects = _norm(
        [{'application_id': 99, 'rank': 1}, {'application_id': 'x', 'rank': 1}], apps)
    assert {r['reason'] for r in rejects} == {'application_id_out_of_range',
                                              'no_valid_identifier'}
    _assert_complete(normalized, apps)
    assert all(r['imputed_tail'] for r in normalized)


def test_norm_inconsistent_ids_rejected():
    apps = _apps('a0', 'a1')
    normalized, rejects = _norm(
        [{'application_id': 0, 'applicant_id': 'a1', 'rank': 1},
         {'application_id': 1, 'applicant_id': 'a1', 'rank': 2}], apps)
    assert [r['reason'] for r in rejects] == ['inconsistent_ids']
    _assert_complete(normalized, apps)
    imputed = [r for r in normalized if r.get('imputed_tail')]
    assert [r['application_id'] for r in imputed] == [0]


def test_norm_duplicates_deduped_by_application_id():
    apps = _apps('a0', 'a1')
    normalized, rejects = _norm(
        [{'application_id': 0, 'rank': 3, 'reason': 'worse'},
         {'application_id': 0, 'rank': 1, 'reason': 'best'},
         {'application_id': 0, 'rank': 1, 'reason': 'tie-later'},
         {'application_id': 1, 'rank': 2}], apps)
    assert all(r['reason'] == 'duplicate_application_id' for r in rejects)
    assert len(rejects) == 2
    _assert_complete(normalized, apps)
    kept0 = next(r for r in normalized if r['application_id'] == 0)
    assert kept0['reason'] == 'best', 'best rank wins, then earlier response position'


def test_norm_non_dict_records():
    apps = _apps('a0', 'a1')
    normalized, rejects = _norm(
        ['garbage', None, 3, {'application_id': 0, 'rank': 1}], apps)
    assert sum(r['reason'] == 'non_dict_entry' for r in rejects) == 3
    _assert_complete(normalized, apps)


def test_norm_empty_ranking_pure_fallback_tail():
    apps = _apps('a0', 'a1', 'a2')
    for empty in ([], None):
        normalized, rejects = _norm(empty, apps)
        assert rejects == []
        _assert_complete(normalized, apps)
        assert all(r['imputed_tail'] for r in normalized)
        # deterministic tail: input order, real ids, increasing ranks
        assert [r['application_id'] for r in normalized] == [0, 1, 2]
        assert [r['rank'] for r in normalized] == [1, 2, 3]


def test_norm_fallback_ordering_deterministic_and_after_valid():
    apps = _apps('a0', 'a1', 'a2', 'a3')
    ranked = [{'application_id': 2, 'rank': 5}]
    n1, _ = _norm([dict(r) for r in ranked], apps)
    n2, _ = _norm([dict(r) for r in ranked], apps)
    assert n1 == n2
    tail = [r for r in n1 if r.get('imputed_tail')]
    assert [r['application_id'] for r in tail] == [0, 1, 3], 'tail in input order'
    assert all(r['rank'] > 5 for r in tail), 'tail ranks after all valid ranks'


def test_norm_end_to_end_missing_applicant_id_no_crash():
    # Regression for the year-5 KeyError: entries lacking applicant_id must
    # flow through process_funding_evaluation_results without crashing.
    apps = _apps('a0', 'a1', 'a2', 'a3')
    ranked = [{'application_id': 0, 'rank': 1},        # missing applicant_id
              'garbage',                               # non-dict
              {'application_id': 1, 'applicant_id': 'a1', 'rank': 2}]
    metadata = [{'program_id': 'P', 'apps': apps, 'panel_index': 0}]
    programs = {'P': SimpleNamespace(funding_rate=0.5)}
    log = []
    winners = FundingAgency.process_funding_evaluation_results(
        [(dict(ranked_applications=ranked), [])], metadata, programs,
        application_log=log)
    assert len(log) == 4, 'complete ranking despite malformed entries'
    assert sorted(r['position'] for r in log) == [1, 2, 3, 4]
    assert {w['applicant_id'] for w in winners['P']} == {'a0', 'a1'}
    imputed = {r['applicant_id'] for r in log if r['imputed_tail']}
    assert imputed == {'a2', 'a3'}


# ------------------------------------------------------- estimator recovery

def test_rd_recovers_known_effect():
    df = make_synthetic_rd(n_competitions=150, panel_n=20, k=5, tau=12.0,
                           slope=1.5, noise=4.0, seed=7)
    res = local_linear_rd(df, 'y', h=4)
    assert np.isfinite(res['tau']) and np.isfinite(res['se'])
    assert abs(res['tau'] - 12.0) < 1.5, f"tau={res['tau']} should be near 12"
    assert res['ci_lo'] < 12.0 < res['ci_hi']


def test_rd_null_when_no_effect():
    df = make_synthetic_rd(n_competitions=150, panel_n=20, k=5, tau=0.0, seed=11)
    res = local_linear_rd(df, 'y', h=4)
    assert abs(res['tau']) < 1.5
    assert res['ci_lo'] < 0 < res['ci_hi']


def test_local_randomization_effect_and_null():
    # slope=0 isolates the local-randomization property (with a slope, adjacent
    # ranks genuinely differ, which is a design limitation noted in the report)
    rng = np.random.default_rng(3)
    df = make_synthetic_rd(n_competitions=150, panel_n=20, k=5, tau=12.0,
                           slope=0.0, seed=5)
    res = local_randomization(df, 'y', window=0.5, rng=rng)
    assert res['n_funded'] == res['n_unfunded'] == 150, 'one obs per side per panel'
    assert abs(res['tau'] - 12.0) < 2.5
    assert res['p_fisher'] < 0.01

    df0 = make_synthetic_rd(n_competitions=150, panel_n=20, k=5, tau=0.0,
                            slope=0.0, seed=6)
    res0 = local_randomization(df0, 'y', window=0.5, rng=rng)
    assert res0['p_fisher'] > 0.05


def test_placebo_cutoff_is_null():
    df = make_synthetic_rd(n_competitions=150, panel_n=20, k=6, tau=12.0, seed=9)
    res = placebo_cutoff(df, 'y', c0=-2.0, h=3)
    assert np.isfinite(res['tau'])
    assert abs(res['tau']) < 2.0, 'no jump should exist away from the true cutoff'


def test_cv_bandwidth_returns_candidate():
    df = make_synthetic_rd(n_competitions=60, panel_n=16, k=4, tau=5.0, seed=2)
    df['prior_papers'] = np.random.default_rng(0).poisson(3, len(df)).astype(float)
    h, losses = cv_bandwidth(df, covariate='prior_papers', candidates=[2, 3, 4])
    assert h in (2, 3, 4) and set(losses) == {2, 3, 4}


# ------------------------------------------------------------- small pieces

def test_bh_adjust_known_values():
    ps = np.array([0.01, 0.04, 0.03, 0.005])
    qs = benjamini_hochberg(ps)
    # sorted: .005 .01 .03 .04 -> raw m*p/rank: .02 .02 .04 .04 -> monotone
    assert np.allclose(qs, [0.02, 0.04, 0.04, 0.02])


def test_km_curve_censoring():
    times = np.array([1, 2, 2, 3, 3])
    events = np.array([1, 1, 0, 0, 0])   # censored obs must not count as deaths
    ts, s = km_curve(times, events)
    assert np.isclose(s[0], 0.8)          # 1/5 exit at t=1
    assert np.isclose(s[1], 0.8 * 0.75)   # 1/4 at-risk exit at t=2
    assert np.isclose(s[2], s[1]), 'censoring only at t=3'


# --------------------------------------- cost intervention (funding_cutoff_cost)

class _StubAgent:
    def __init__(self, aid, resources):
        self.id, self.resources = aid, resources

    def update_resources(self, delta):
        assert self.resources + delta >= 0
        self.resources += delta


def _apps_for(agent, program_ids, submit=True):
    return {pid: {'submit': submit, 'author': agent} for pid in program_ids}


def test_cost_zero_is_behaviorally_identical():
    from utopia.simulation import charge_application_costs
    a = _StubAgent('a1', 100)
    apps = [_apps_for(a, ['P1', 'P2'])]
    import copy
    before_keys = [set(d.keys()) for d in apps]
    events = charge_application_costs(apps, 0, year=1)
    assert events == []
    assert [set(d.keys()) for d in apps] == before_keys, 'no application removed'
    assert a.resources == 100, 'no charge at cost 0'


def test_cost_charged_per_application_before_result():
    from utopia.simulation import charge_application_costs
    a = _StubAgent('a1', 50)
    apps = [_apps_for(a, ['P1', 'P2', 'P3'])]
    events = charge_application_costs(apps, 5, year=2)
    assert a.resources == 35, 'charged once PER application (3 x 5)'
    assert len(events) == 3
    assert [e['pre_balance'] for e in events] == [50, 45, 40]
    assert all(e['cost_charged'] == 5 and not e['withdrawn_unaffordable']
               for e in events)
    assert all(e['year'] == 2 for e in events)
    # results are unknown at charge time: no funded/award fields yet
    assert all('funded' not in e for e in events)


def test_cost_unaffordable_application_withdrawn_no_debt():
    from utopia.simulation import charge_application_costs
    a = _StubAgent('a1', 7)
    apps = [_apps_for(a, ['P1', 'P2'])]
    events = charge_application_costs(apps, 5, year=1)
    # first app affordable (7 -> 2), second withdrawn (2 < 5), never negative
    assert a.resources == 2
    assert len(apps[0]) == 1
    withdrawn = [e for e in events if e['withdrawn_unaffordable']]
    assert len(withdrawn) == 1 and withdrawn[0]['cost_charged'] == 0
    assert withdrawn[0]['post_cost_balance'] == 2


def test_cost_skips_non_submitted_applications():
    from utopia.simulation import charge_application_costs
    a = _StubAgent('a1', 100)
    apps = [_apps_for(a, ['P1'], submit=False)]
    events = charge_application_costs(apps, 5, year=1)
    assert events == [] and a.resources == 100 and 'P1' in apps[0]


def test_cost_experiment_id_tag_and_default():
    from utopia.arguments import parse_arguments
    base = ['--experiment_name', 'exploration_vs_exploitation',
            '--experiment_stage', 'funding_cutoff_cost', '--seed', '3200',
            '--population_mode', 'university_only', '--num_institutions', '40',
            '--researchers_per_institution', '5', '--num_years', '5',
            '--num_conferences', '6', '--start_year', '2016',
            '--model', 'Qwen/Qwen3-32B', '--vllm_url', 'http://x/v1',
            '--batch_size', '128', '--industry_funding_mode', 'performance',
            '--funding_allocation_mode', 'fixed']
    a0 = parse_arguments(base)
    import re
    assert a0.funding_application_cost == 0, 'default must be 0'
    assert not re.search(r'_cost\d+$', a0.experiment_id), 'no cost suffix at default'
    a5 = parse_arguments(base + ['--funding_application_cost', '5'])
    assert a5.experiment_id.endswith('_cost5')
    assert a5.experiment_id.startswith('explore_funding_cutoff_cost_')


def test_interaction_rd_recovers_environment_difference():
    from utopia.analysis.funding_cutoff import interaction_rd, paired_seed_differences
    df_base = make_synthetic_rd(n_competitions=120, panel_n=20, k=5, tau=0.0,
                                slope=1.0, noise=4.0, seed=21)
    df_cost = make_synthetic_rd(n_competitions=120, panel_n=20, k=5, tau=8.0,
                                slope=1.0, noise=4.0, seed=22)
    res = interaction_rd(df_base, df_cost, 'y', h=4)
    assert np.isfinite(res['tau_diff']) and np.isfinite(res['se'])
    assert abs(res['tau_diff'] - 8.0) < 2.0, f"diff={res['tau_diff']} should be ~8"
    assert res['ci_lo'] < 8.0 < res['ci_hi']
    ps = paired_seed_differences(df_base, df_cost, 'y', h=4)
    assert np.isfinite(ps['summary']['mean_diff'])


def test_path_guard_allows_cost_dirs():
    assert guard_write_path('data/funding_cutoff_cost/applications.parquet')
    assert guard_write_path('outputs/docs/funding_cutoff_cost_final/report.md')
    assert guard_write_path('outputs/checkpoints/explore_funding_cutoff_cost_qwen3_32b_neutral_i40_n200_y5_seed3200_cost5/x.jsonl')


def test_path_guard_rejects_outside_writes():
    assert guard_write_path('data/funding_cutoff/x.parquet')
    assert guard_write_path('outputs/docs/funding_cutoff_final/report.md')
    assert guard_write_path('outputs/checkpoints/explore_funding_cutoff_qwen3_32b_neutral_i5_n25_y2_seed3100/log.jsonl')
    for bad in ('outputs/docs/explore_final_freeze/final_freeze.json',
                'outputs/checkpoints/explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed1001/x',
                'utopia/experiments/exploration.py', '/tmp/evil.txt', 'data/other/x.csv'):
        with pytest.raises(PermissionError):
            guard_write_path(bad)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
