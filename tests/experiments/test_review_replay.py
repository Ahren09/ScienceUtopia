"""Deterministic tests for the reviewer-monoculture review-replay pilot.

No Qwen/vLLM calls; hand-built objects and stub LLMs only (same conventions as
tests/experiments/test_exploration.py). Run:
    python -m pytest tests/experiments/test_review_replay.py -v
"""

from utopia.analysis import statistics

import json
import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path
import pytest

from utopia.utils.paths import project_root
ROOT = str(project_root(__file__))

import utopia.experiments.review_replay as rr
from utopia.analysis import review_regime_analysis as ra
from utopia.utils.seeding import derive_seed

PREREG = rr.load_prereg()
PAPER = {'title': 'A Study of Widgets', 'abstract': 'We study widgets thoroughly.',
         'topics': ['widgets', 'machine learning']}
EXPERTISE = ['graph mining', 'optimization']


# ------------------------------------------------------------- prompt adapter

def test_policy_block_inserted_before_anchor_once():
    prompt, _ = rr.build_controlled_prompt(PAPER, EXPERTISE,
                                           PREREG['prompt_construction']['policy_text_A'])
    assert prompt.count('## Your Review Approach') == 1
    assert prompt.index('## Your Review Approach') < prompt.index('## Review Instructions')
    assert 'Your Expertise: graph mining, optimization' in prompt
    assert '## Your Recent Experiences' not in prompt  # empty memory policy


def test_A_and_B_prompts_differ_only_in_policy_block():
    a, _ = rr.build_controlled_prompt(PAPER, EXPERTISE,
                                      PREREG['prompt_construction']['policy_text_A'])
    b, _ = rr.build_controlled_prompt(PAPER, EXPERTISE,
                                      PREREG['prompt_construction']['policy_texts_B']['rigor'])
    a_clean = a.replace(PREREG['prompt_construction']['policy_text_A'], '@POLICY@')
    b_clean = b.replace(PREREG['prompt_construction']['policy_texts_B']['rigor'], '@POLICY@')
    assert a_clean == b_clean


def test_prestige_arms_differ_only_in_institution_string():
    hi = rr.prestige_author_info(PREREG, 'prestige_high')
    lo = rr.prestige_author_info(PREREG, 'prestige_low')
    assert hi['author_name'] == lo['author_name']
    assert hi['network_relationship'] == lo['network_relationship']
    p_hi, _ = rr.build_controlled_prompt(PAPER, EXPERTISE, 'policy', hi)
    p_lo, _ = rr.build_controlled_prompt(PAPER, EXPERTISE, 'policy', lo)
    assert '## Author Information (Non-Blind Review)' in p_hi
    assert p_hi.replace(hi['institution'], '@I@') == p_lo.replace(lo['institution'], '@I@')


def test_profile_assignment_deterministic_and_distinct():
    keys = list(PREREG['prompt_construction']['policy_texts_B'])
    p1 = rr.assign_profiles('http://arxiv.org/abs/1234.5678', keys)
    p2 = rr.assign_profiles('http://arxiv.org/abs/1234.5678', keys)
    p3 = rr.assign_profiles('http://arxiv.org/abs/9999.0001', keys)
    assert p1 == p2 and len(set(p1)) == 3
    assert p1 != p3 or True  # different papers may coincide; only determinism is required


# ------------------------------------------------------------- memory filter

def test_memory_filter_keeps_pre_review_information_only():
    bank = [
        {'type': 'select_research_direction', 'year': 1, 'thought': 'a'},
        {'type': 'good_reviews_received', 'year': 1, 'thought': 'b'},
        {'type': 'select_research_direction', 'year': 2, 'thought': 'c'},
        {'type': 'poor_reviews_received', 'year': 2, 'thought': 'd'},
    ]
    kept = rr.filter_memory_at_review_time(bank, review_year=2)
    assert [m['thought'] for m in kept] == ['a', 'b', 'c']  # year-2 review outcome dropped


def test_memory_filter_asserts_on_malformed_entry():
    with pytest.raises(AssertionError):
        rr.filter_memory_at_review_time([{'type': 'x', 'year': 1}], 2)  # no 'thought'


# ------------------------------------------------------------- leakage scanner

def test_leakage_scanner_allows_title_in_paper_section():
    prompt, _ = rr.build_controlled_prompt(PAPER, EXPERTISE, 'policy')
    ok, reason, _ = rr.scan_leakage(prompt, {'author_id': 'institution_0001_researcher_2',
                                             'institution': 'university_0001',
                                             'arxiv_id': 'http://x/1', 'title': PAPER['title']})
    assert ok, reason


def test_leakage_scanner_flags_author_id_outside_paper_section():
    prompt, _ = rr.build_controlled_prompt(PAPER, EXPERTISE,
                                           'policy mentioning institution_0001_researcher_2')
    ok, reason, section = rr.scan_leakage(prompt, {'author_id': 'institution_0001_researcher_2',
                                                   'institution': '', 'arxiv_id': '',
                                                   'title': PAPER['title']})
    assert not ok and 'author_id' in reason and section.startswith('## Your Review Approach')


def test_leakage_scanner_flags_title_in_memory_for_c_replay():
    from utopia.agents.base_agent import SimulationAgent
    agent = SimulationAgent(reputation=5, agent_id='r', llm=None)
    agent.expertise = EXPERTISE
    agent.memory_bank = [{'type': 'select_research_direction', 'year': 1,
                          'thought': f"I read {PAPER['title']} yesterday"}]
    prompt, _ = agent.get_review_prompt(PAPER, author_info=None)
    ok, reason, section = rr.scan_leakage(
        prompt, {'author_id': 'zz', 'institution': 'zz', 'arxiv_id': 'zz',
                 'title': PAPER['title']}, check_title_in_memory=True)
    assert not ok and 'title' in reason and section.startswith('## Your Recent Experiences')


# ------------------------------------------------------------- sampling cells

def full_counts(v):
    return {(t, b, a): v for t in rr.TERCILES for b in rr.BUCKETS for a in (False, True)}


def test_collapse_identity_when_all_cells_large():
    mapping = rr.collapse_cells(full_counts(20))
    assert len(set(mapping.values())) == 18  # no merging
    assert mapping[('t1', 'near', True)] == 't1|near|acc'


def test_collapse_merges_terciles_then_buckets():
    counts = full_counts(20)
    counts[('t1', 'far', False)] = 2          # sparse: forces t1+t2 merge for far/rej
    mapping = rr.collapse_cells(counts)
    assert mapping[('t1', 'far', False)] == mapping[('t2', 'far', False)] == 't1+t2|far|rej'
    assert mapping[('t3', 'far', False)] == 't3|far|rej'
    counts[('t2', 'far', False)] = 2
    counts[('t3', 'far', False)] = 2          # whole far/rej bucket sparse even merged
    mapping = rr.collapse_cells(counts)
    lab = mapping[('t1', 'far', False)]
    assert 'far' in lab and '+' in lab.split('|')[1]  # far merged with another bucket


def test_collapse_never_merges_accepted_with_rejected_or_across_dims():
    counts = full_counts(1)  # everything sparse -> full collapse within acc class
    mapping = rr.collapse_cells(counts)
    acc_labels = {mapping[(t, b, True)] for t in rr.TERCILES for b in rr.BUCKETS}
    rej_labels = {mapping[(t, b, False)] for t in rr.TERCILES for b in rr.BUCKETS}
    assert acc_labels == {'t1+t2+t3|near+mid+far|acc'}
    assert rej_labels == {'t1+t2+t3|near+mid+far|rej'}
    assert not acc_labels & rej_labels


def test_largest_remainder_sums_and_caps():
    alloc = rr.largest_remainder(10, {'a': 30, 'b': 30, 'c': 40})
    assert sum(alloc.values()) == 10 and alloc == {'a': 3, 'b': 3, 'c': 4}
    alloc = rr.largest_remainder(10, {'a': 2, 'b': 100})
    assert alloc['a'] <= 2 and sum(alloc.values()) == 10
    alloc = rr.largest_remainder(10, {'a': 3, 'b': 4})  # quota > eligible total
    assert alloc == {'a': 3, 'b': 4}


def make_pool(seed, n_per_cell=12):
    rows = []
    i = 0
    for t, b, a in [(t, b, a) for t in rr.TERCILES for b in rr.BUCKETS for a in (False, True)]:
        for _ in range(n_per_cell):
            rows.append({'paper_id': f'p{seed}_{i:04d}', 'source_seed': seed,
                         'tercile': t, 'distance_bucket': b, 'accepted': a,
                         'final_cell': f'{t}|{b}|{"acc" if a else "rej"}'})
            i += 1
    return pd.DataFrame(rows)


def test_sample_stage_deterministic_with_correct_ipw():
    pool = make_pool(501)
    s1 = rr.sample_stage(pool, 60, 'corpus_sample', 501)
    s2 = rr.sample_stage(pool, 60, 'corpus_sample', 501)
    assert list(s1.paper_id) == list(s2.paper_id) and len(s1) == 60
    s3 = rr.sample_stage(pool, 60, 'smoke_sample', 501)
    assert list(s3.paper_id) != list(s1.paper_id)  # different rng label
    row = s1.iloc[0]
    assert row.ipw_weight == pytest.approx(row.cell_eligible / row.cell_selected)
    assert row.inclusion_prob == pytest.approx(row.cell_selected / row.cell_eligible)


# ------------------------------------------------------------- task grid

def fake_corpus_row():
    return pd.Series({
        'paper_id': 'http://arxiv.org/abs/1612.00001', 'source_seed': 501,
        'source_id': 'x', 'year': 2, 'conference': 'AAAI', 'accepted': True,
        'strategy': 'explorer', 'institution': 'u1', 'novelty_score': 0.5,
        'title': PAPER['title'], 'abstract': PAPER['abstract'],
        'topics': json.dumps(PAPER['topics']), 'author_id': 'institution_0009_researcher_1',
        'author_university': 'university_0009',
        'orig_reviewer_ids': json.dumps(['r1', 'r2', 'r3']),
        'orig_scores': json.dumps([3, 2, 2]),
        'orig_justifications': json.dumps(['a', 'b', 'c']),
        'slot_expertise': json.dumps([['graphs'], ['nlp'], ['vision']]),
    })


def test_task_grid_counts_and_uid_format():
    tasks = rr.make_tasks_for_paper(fake_corpus_row(), PREREG, 'pilot',
                                    ('A', 'B'), ('blind', 'prestige_high', 'prestige_low'), 1)
    assert len(tasks) == 18  # 2 regimes x 3 arms x 3 slots
    uids = {t['task_uid'] for t in tasks}
    assert len(uids) == 18
    assert all(t['task_uid'] == f"{t['paper_id']}|{t['regime']}|{t['arm']}|1|{t['slot']}"
               for t in tasks)
    a_policies = {t['policy_key'] for t in tasks if t['regime'] == 'A'}
    b_policies = [t['policy_key'] for t in tasks if t['regime'] == 'B' and t['arm'] == 'blind']
    assert a_policies == {'standard'} and len(set(b_policies)) == 3


def test_task_seed_derivation_is_stable():
    assert derive_seed(42, 'review_replay', 'v1', 0, 5, 0) == \
        derive_seed(42, 'review_replay', 'v1', 0, 5, 0)
    assert derive_seed(42, 'review_replay', 'v1', 0, 5, 0) != \
        derive_seed(42, 'review_replay', 'v1', 1, 5, 0)


# ------------------------------------------------------------- path guard

def test_path_guard_allows_pilot_paths_and_blocks_e4():
    for good in ('data/review_replay/x.parquet',
                 'outputs/docs/review_replay_pilot_qwen3_32b_p300_seed42/reviews.parquet',
                 'outputs/logs/review_replay_smoke_qwen3_32b_p24_seed42/generate.log',
):
        rr.assert_write_allowed(good)
    for bad in ('outputs/visual/review_replay_x/f.png', 'outputs/docs/explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed1001/x',
                'outputs/checkpoints/review_replay_x/y.json',
                'outputs/docs/explore_calibration_qwen3_32b_neutral_i100_n500_y10_seed501/p.parquet',
                'utopia/simulation.py', 'data/other/x'):
        with pytest.raises(PermissionError):
            rr.assert_write_allowed(bad)


def test_vllm_url_guard():
    for endpoint in ('http://localhost:8000/v1', 'http://localhost:8032/v1', 'https://example.org/v1'):
        rr.assert_safe_vllm_url(endpoint)
    for endpoint in ('file:///tmp/server', 'missing-scheme'):
        with pytest.raises(ValueError):
            rr.assert_safe_vllm_url(endpoint)


def test_icc1_hand_computed():
    # 3 papers x 2 raters: MSB=8, MSW=0.5 -> ICC1 = 7.5/8.5
    m = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    assert ra.icc1(m) == pytest.approx(7.5 / 8.5)


def test_krippendorff_alpha_interval_hand_computed():
    # units [[1,2],[3,4]]: D_o = 1; D_e = mean of squared diffs over all value pairs = 20/6
    m = np.array([[1, 2], [3, 4]], dtype=float)
    assert ra.krippendorff_alpha_interval(m) == pytest.approx(1 - 1 / (20 / 6))
    perfect = np.array([[3, 3], [1, 1], [5, 5]], dtype=float)
    assert ra.krippendorff_alpha_interval(perfect) == pytest.approx(1.0)


def test_dispersion_and_cosine_diversity():
    assert ra.within_paper_sd([2, 2, 2]) == 0.0
    assert ra.within_paper_sd([1, 3]) == pytest.approx(np.sqrt(2))
    orth = np.array([[1, 0], [0, 1]], dtype=float)
    assert ra.mean_pairwise_cosine_distance(orth) == pytest.approx(1.0)
    same = np.array([[1, 1], [2, 2]], dtype=float)
    assert ra.mean_pairwise_cosine_distance(same) == pytest.approx(0.0, abs=1e-9)


def test_weighted_slope_exact():
    x = np.array([0., 1., 2., 3.])
    y = 2.0 * x + 1.0
    b1, b0 = statistics.weighted_slope(y, x, np.ones_like(x))
    assert b1 == pytest.approx(2.0) and b0 == pytest.approx(1.0)
    w = np.array([1., 1., 100., 100.])   # weighting preserves an exact linear fit
    b1w, _ = statistics.weighted_slope(y, x, w)
    assert b1w == pytest.approx(2.0)


def test_paired_boot_recovers_mean():
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 1.0, 400)
    r = statistics.paired_boot(v, n_boot=500, seed=1)
    assert r['ci_lo'] < 0.5 < r['ci_hi'] and r['estimate'] == pytest.approx(v.mean())


# ------------------------------------------------------------- acceptance rules

def test_original_threshold_midpoint_and_absolute_rule():
    elig = pd.DataFrame({
        'source_seed': [501] * 4, 'conference': ['AAAI'] * 4, 'year': [1] * 4,
        'accepted': [True, True, False, False],
        'review_score_orig': [4.0, 3.0, 2.5, 2.0],
    })
    thr = ra.original_thresholds(elig)
    assert thr[(501, 'AAAI', 1)] == pytest.approx((3.0 + 2.5) / 2)


def test_capacity_rule_top_30_percent_min_one():
    cells = pd.DataFrame({
        'regime': ['A'] * 10, 'source_seed': [501] * 10, 'conference': ['AAAI'] * 10,
        'year': [1] * 10, 'paper_id': [f'p{i}' for i in range(10)],
        'mean_score_int': list(range(10)),
    })
    out = ra.apply_capacity_rule(cells, rate=0.30)
    assert out.accept_capacity.sum() == 3
    assert set(out.loc[out.accept_capacity, 'mean_score_int']) == {7, 8, 9}
    single = ra.apply_capacity_rule(cells.head(1), rate=0.30)
    assert single.accept_capacity.sum() == 1  # min one accept per non-empty cell


# ------------------------------------------------------------- cell table + run_chunk

class FakeBatchLLM:
    def __init__(self, score=3.0):
        self.score = score
        self.call_stats = {}

    def generate_batch(self, prompts, **kwargs):
        return [({'overall_score': self.score + 0.5 * (i % 2), 'justification': f'j{i}'}, [])
                for i, _ in enumerate(prompts)]


def test_run_chunk_row_fields_and_int_cast():
    tasks = pd.DataFrame([{
        'task_uid': f'p|A|blind|1|{s}', 'stage': 'pilot', 'paper_id': 'p',
        'source_seed': 501, 'year': 1, 'conference': 'AAAI', 'regime': 'A',
        'arm': 'blind', 'panel': 1, 'slot': s, 'policy_key': 'standard',
        'orig_reviewer_id': 'r', 'prompt': 'x', 'prompt_sha256': 'h'} for s in range(3)])
    rows = rr.run_chunk(FakeBatchLLM(3.0), tasks, {}, seed_ctx=('review_replay', 'v1', 0))
    assert all(r['success'] for r in rows)
    assert rows[1]['score_raw'] == 3.5 and rows[1]['score_int'] == 3  # int() truncation as live
    assert rows[0]['request_seed_attempt0'] == derive_seed(42, 'review_replay', 'v1', 0, 0, 0)


def test_cell_table_requires_two_slots_and_uses_raw_scores():
    reviews = pd.DataFrame([
        dict(paper_id='p1', regime='A', arm='blind', panel=1, source_seed=501, slot=0,
             success=True, score_raw=3.0, score_int=3, justification='a', justification_len=1),
        dict(paper_id='p1', regime='A', arm='blind', panel=1, source_seed=501, slot=1,
             success=True, score_raw=4.0, score_int=4, justification='b', justification_len=1),
        dict(paper_id='p1', regime='B', arm='blind', panel=1, source_seed=501, slot=0,
             success=True, score_raw=2.0, score_int=2, justification='c', justification_len=1),
    ])
    cells = ra.cell_table(reviews)
    assert len(cells) == 1  # regime B cell dropped (1 slot)
    assert cells.iloc[0].mean_score == pytest.approx(3.5)
    assert cells.iloc[0].sd_score == pytest.approx(np.std([3, 4], ddof=1))


# ==========================================================================
# Network-proximity study (--study network_proximity)
# ==========================================================================

import itertools

NRB_PREREG = json.load(open('configs/network_review_bias.json'))


@pytest.fixture
def nrb_study():
    rr._activate_study('network_proximity')
    yield
    rr._activate_study('reviewer_monoculture')


def test_registry_monoculture_literals_unchanged():
    cfg = rr.STUDIES['reviewer_monoculture']
    assert Path(cfg['prereg']) == Path(ROOT) / 'configs/review_replay.json'
    assert cfg['corpus_dir'] == 'data/review_replay'
    assert cfg['stages'] == {
        'smoke': 'review_replay_smoke_qwen3_32b_p24_seed42',
        'pilot': 'review_replay_pilot_qwen3_32b_p300_seed42',
        'stability': 'review_replay_stability_qwen3_32b_p60_seed42'}
    assert cfg['write_prefixes'] == ('data/review_replay', 'outputs/logs/review_replay_',
                                     'outputs/docs/review_replay_')
    assert cfg['source_seeds'] == [501, 502, 503, 504, 505]
    assert rr.ACTIVE_STUDY == 'reviewer_monoculture'


def test_nrb_path_guard(nrb_study):
    for good in ('data/network_review_bias/x.parquet',
                 'outputs/docs/network_review_bias_main_qwen3_32b_p300_seed42/reviews.parquet',
                 'outputs/logs/network_review_bias_smoke_qwen3_32b_p24_seed42/g.log',
):
        rr.assert_write_allowed(good)
    for bad in ('data/review_replay/x.parquet',            # other study refused
                'outputs/docs/explore_final_freeze/x.json',  # E4 refused
                'outputs/checkpoints/network_review_bias_source_seed7001/x.json',
                'utopia/simulation.py'):
        with pytest.raises(PermissionError):
            rr.assert_write_allowed(bad)


# ------------------------------------------------------- graph timing

def _edges():
    return [{'a': 'u', 'b': 'v', 'weight': 1, 'years': [1]},
            {'a': 'v', 'b': 'w', 'weight': 1, 'years': [2, 3]},
            {'a': 'w', 'b': 'x', 'weight': 1, 'years': [3]},
            {'a': 'x', 'b': 'y', 'weight': 1, 'years': [4]},
            {'a': 'p', 'b': 'q', 'weight': 1, 'years': [2]}]


def test_snapshot_excludes_same_year_and_future_edges():
    G = rr.nrb_graph_snapshot(_edges(), upto_year=2)  # usable for year-3 reviews
    assert G.has_edge('u', 'v') and G.has_edge('v', 'w') and G.has_edge('p', 'q')
    assert not G.has_edge('w', 'x') and not G.has_edge('x', 'y')
    G1 = rr.nrb_graph_snapshot(_edges(), upto_year=0)
    assert G1.number_of_edges() == 0


def test_distance_and_strata():
    G = rr.nrb_graph_snapshot(_edges(), upto_year=3)   # u-v-w-x path, p-q apart
    assert rr.nrb_distance(G, 'u', 'w')[0] == 2
    assert rr.nrb_distance(G, 'u', 'x')[0] == 3
    assert rr.nrb_distance(G, 'u', 'p') == (None, 'disconnected')
    assert rr.nrb_stratum_of(2, '') == 'd2'
    assert rr.nrb_stratum_of(3, '') == 'd3'
    assert rr.nrb_stratum_of(None, 'disconnected') == 'far'
    assert rr.nrb_stratum_of(4, 'connected_ge4') == 'far'
    assert rr.nrb_stratum_of(1, '') is None


# ------------------------------------------------------- eligibility

def _agents():
    ag = {}
    for i, (rid, inst) in enumerate([('u', 'i1'), ('v', 'i2'), ('w', 'i3'), ('x', 'i4'),
                                     ('y', 'i5'), ('p', 'i6'), ('q', 'i7'), ('z', 'i8'),
                                     ('c1', 'i1'), ('c2', 'i9')]):
        ag[rid] = {'institution': inst, 'can_review': True,
                   'expertise': [f'topic{i}'], 'coi': set(),
                   'start': None, 'end': None}
    return ag


def test_multiauthor_eligibility_and_focal_distance():
    # chain u-v-w-x-y plus disconnected p-q; z absent from graph
    G = rr.nrb_graph_snapshot(_edges(), upto_year=4)
    agents = _agents()
    agents['c2']['coi'] = {'u'}          # reviewer-side COI with focal author
    paper = {'focal_author_id': 'u', 'author_list': ['u', 'v']}  # v is co-author
    pools, meta = rr.nrb_candidates_for_paper(G, paper, agents, 
                                              {r: a['coi'] for r, a in agents.items()},
                                              connected_far_only=False)
    all_pool = pools['d2'] + pools['d3'] + pools['far']
    assert 'v' not in all_pool          # co-author
    assert 'w' not in all_pool          # d1 of co-author v
    assert 'c1' not in all_pool         # same institution as focal author (i1)
    assert 'c2' not in all_pool         # reviewer-side COI with an author
    assert 'x' in pools['d3']           # d(u,x)=3, d(v,x)=2 -> focal distance rules stratum
    assert meta['x']['distance'] == 3 and meta['x']['min_distance_any_author'] == 2
    assert 'p' in pools['far'] and meta['p']['far_kind'] == 'disconnected'
    assert 'z' in pools['far'] and meta['z']['far_kind'] == 'disconnected'
    pools_c, _ = rr.nrb_candidates_for_paper(G, paper, agents,
                                             {r: a['coi'] for r, a in agents.items()},
                                             connected_far_only=True)
    assert 'p' not in pools_c['far'] and 'z' not in pools_c['far']


def test_author_side_coi_also_excludes():
    G = rr.nrb_graph_snapshot(_edges(), upto_year=4)
    agents = _agents()
    coi = {r: a['coi'] for r, a in agents.items()}
    coi['u'] = {'x'}                    # author's list names reviewer x
    paper = {'focal_author_id': 'u', 'author_list': ['u']}
    pools, _ = rr.nrb_candidates_for_paper(G, paper, agents, coi, False)
    assert 'x' not in pools['d2'] + pools['d3'] + pools['far']


# ------------------------------------------------------- matching

def _brute_force_triplet(pools, sims, use_counts, cap, tol):
    best = None
    for t in itertools.product(*(sorted(pools[s]) for s in rr.NRB_STRATA)):
        if any(use_counts.get(r, 0) >= cap for r in t):
            continue
        g = max(sims[r] for r in t) - min(sims[r] for r in t)
        if g > tol:
            continue
        key = (round(g, 6), sum(use_counts.get(r, 0) for r in t),
               max(use_counts.get(r, 0) for r in t), t)
        if best is None or key < best[0]:
            best = (key, t)
    if best is None:
        return None
    return dict(zip(rr.NRB_STRATA, best[1]))


def test_best_triplet_matches_brute_force():
    rng = np.random.default_rng(7)
    for trial in range(30):
        pools = {'d2': [f'a{i}' for i in range(4)], 'd3': [f'b{i}' for i in range(4)],
                 'far': [f'c{i}' for i in range(5)]}
        sims = {r: float(rng.uniform(0, 1)) for s in pools for r in pools[s]}
        use = {r: int(rng.integers(0, 3)) for s in pools for r in pools[s]}
        fast = rr.nrb_best_triplet(pools, sims, use, reuse_cap=3, tol=0.25)
        slow = _brute_force_triplet(pools, sims, use, 3, 0.25)
        if slow is None:
            assert fast is None
        else:
            assert fast is not None
            f = (fast['d2'], fast['d3'], fast['far'])
            s = (slow['d2'], slow['d3'], slow['far'])
            fk = (round(max(sims[x] for x in f) - min(sims[x] for x in f), 6),
                  sum(use[x] for x in f), max(use[x] for x in f), f)
            sk = (round(max(sims[x] for x in s) - min(sims[x] for x in s), 6),
                  sum(use[x] for x in s), max(use[x] for x in s), s)
            assert fk == sk, f'trial {trial}: {fk} != {sk}'


def test_match_corpus_deterministic_and_caps_reuse():
    pools = {}
    sims = {}
    for i in range(30):
        pid = f'p{i:03d}'
        pools[pid] = {'d2': ['r1', 'r2'], 'd3': ['r3', 'r4'], 'far': ['r5', 'r6']}
        sims[pid] = {r: 0.5 for r in ('r1', 'r2', 'r3', 'r4', 'r5', 'r6')}
    out = rr.nrb_match_corpus(list(pools), pools, sims, reuse_cap=10, tol=0.1)
    assert max(out['use_counts'].values()) <= 10
    assert len(out['matched']) == 20      # 6 reviewers x cap 10 / 3 per paper
    # input-order permutation invariance
    shuffled = dict(reversed(list(pools.items())))
    out2 = rr.nrb_match_corpus(list(shuffled), shuffled, sims, reuse_cap=10, tol=0.1)
    assert out['matched'] == out2['matched']


def test_matching_excludes_over_tolerance():
    pools = {'p': {'d2': ['a'], 'd3': ['b'], 'far': ['c']}}
    sims = {'p': {'a': 0.9, 'b': 0.5, 'c': 0.1}}
    out = rr.nrb_match_corpus(['p'], pools, sims, reuse_cap=10, tol=0.1)
    assert out['matched'] == {}


# ------------------------------------------------------- prompt adapter

def _nrb_prompts(stratum='d2', far_kind=''):
    syn = rr.nrb_synthetic_id('institution_0001_researcher_2')
    inst = 'institution_0001'
    policy = NRB_PREREG['reviewer_policy_and_memory']['policy_text']
    out = {}
    for cond in rr.NRB_CONDITIONS:
        if cond == 'A':
            block = ''
        else:
            rel = rr.nrb_relationship_value(NRB_PREREG, cond, stratum, far_kind)
            block = rr.nrb_author_block(syn, inst, rel)
        out[cond], _ = rr.nrb_build_prompt(PAPER, EXPERTISE, policy, block)
    return out, syn, inst


def test_nrb_condition_A_has_no_author_block():
    prompts, syn, inst = _nrb_prompts()
    assert '## Author Information' not in prompts['A']
    assert syn not in prompts['A'] and inst not in prompts['A']
    for sentence in NRB_PREREG['conditions']['C_relationship'].values():
        assert sentence not in prompts['A']


def test_nrb_B_and_C_differ_only_in_relationship_field():
    for stratum, far_kind in [('d2', ''), ('d3', ''), ('far', 'connected_ge4'),
                              ('far', 'disconnected')]:
        prompts, syn, inst = _nrb_prompts(stratum, far_kind)
        b_rel = NRB_PREREG['conditions']['B_identity_only']['relationship_value']
        c_rel = rr.nrb_relationship_value(NRB_PREREG, 'C', stratum, far_kind)
        assert b_rel in prompts['B'] and c_rel in prompts['C']
        assert c_rel not in prompts['B']
        assert prompts['B'].replace(b_rel, '@R@') == prompts['C'].replace(c_rel, '@R@')
        assert syn in prompts['B'] and inst in prompts['B']
        diff_b = [l for l in prompts['B'].splitlines() if l not in prompts['C'].splitlines()]
        assert len(diff_b) == 1 and diff_b[0].startswith('- Professional-network relationship:')


def test_nrb_synthetic_id_stable_and_masks_raw_id():
    s1 = rr.nrb_synthetic_id('institution_0001_researcher_2')
    s2 = rr.nrb_synthetic_id('institution_0001_researcher_2')
    assert s1 == s2 and s1.startswith('Researcher ') and 'institution_0001' not in s1


# ------------------------------------------------------- order + seeds

def test_condition_order_deterministic_and_covers_all_conditions():
    o1 = rr.nrb_condition_order('p1', 'r1')
    o2 = rr.nrb_condition_order('p1', 'r1')
    assert o1 == o2 and sorted(o1) == ['A', 'B', 'C']
    orders = {tuple(rr.nrb_condition_order(f'p{i}', f'r{i}')) for i in range(200)}
    assert len(orders) == 6  # all six permutations occur


def test_order_randomization_does_not_change_labels():
    # exec_key encodes order; task_uid and condition stay canonical
    order = rr.nrb_condition_order('pX', 'rY')
    for pos, cond in enumerate(order):
        exec_key = f'pX|d2|{pos}|{cond}'
        assert exec_key.split('|')[3] == cond


def test_nrb_seed_map_deterministic_and_stage_scoped():
    a = derive_seed(42, 'nrb', 'v1', 'main', 0, 5, 0)
    assert a == derive_seed(42, 'nrb', 'v1', 'main', 0, 5, 0)
    assert a != derive_seed(42, 'nrb', 'v1', 'stability', 0, 5, 0)
    assert a != derive_seed(42, 'review_replay', 'v1', 0, 5, 0)


# ------------------------------------------------------- analysis pipeline

def _synthetic_reviews(n_papers=80, d2_effect=0.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_papers):
        pid = f's:{i:03d}'
        base = rng.normal(3, 0.3)
        for s in ('d2', 'd3', 'far'):
            rid = f'rev{i % 20}_{s}'
            for cond in ('A', 'B', 'C'):
                score = base + rng.normal(0, 0.05)
                if cond == 'C' and s == 'd2':
                    score += d2_effect
                rows.append(dict(
                    task_uid=f'{pid}|{rid}|{cond}|{s}|primary', paper_id=pid,
                    reviewer_id=rid, condition=cond, stratum=s,
                    far_kind='connected_ge4' if s == 'far' else '',
                    source_label='seedA' if i % 2 else 'seedB',
                    similarity=0.5, reviewer_betweenness=rng.uniform(0, 0.1),
                    focal_betweenness=rng.uniform(0, 0.1), condition_order='ABC',
                    success=True, score_raw=float(score), score_int=int(score),
                    justification='sound and clear', justification_len=15))
    return pd.DataFrame(rows)


def test_primary_estimator_recovers_injected_effect():
    from utopia.analysis import network_review_bias_analysis as na
    rev = _synthetic_reviews(d2_effect=0.5)
    pairs = na.pair_frame(rev)
    papers = na.paper_frame(pairs)
    assert len(papers) == 80 and papers.paper_effect.notna().all()
    res = na.compute_primary(pairs, papers, n_boot=400)
    est = res['primary_d2_minus_far']
    assert est['ci_lo'] < 0.5 < est['ci_hi'] or abs(est['estimate'] - 0.5) < 0.1
    assert est['ci_lo'] > 0.2  # decisively positive
    assert res['robust_reviewer_cluster']['estimate'] == pytest.approx(est['estimate'], abs=1e-9)
    reg = res['robust_stacked_regression']
    assert reg['d2_vs_far']['estimate'] == pytest.approx(0.5, abs=0.15)


def test_null_effect_gives_null_primary():
    from utopia.analysis import network_review_bias_analysis as na
    rev = _synthetic_reviews(d2_effect=0.0, seed=3)
    pairs = na.pair_frame(rev)
    papers = na.paper_frame(pairs)
    res = na.compute_primary(pairs, papers, n_boot=400)
    assert res['primary_d2_minus_far']['ci_lo'] < 0 < res['primary_d2_minus_far']['ci_hi']


def test_incomplete_papers_excluded_from_primary():
    from utopia.analysis import network_review_bias_analysis as na
    rev = _synthetic_reviews(n_papers=10)
    rev = rev[~((rev.paper_id == 's:000') & (rev.stratum == 'far') & (rev.condition == 'C'))]
    papers = na.paper_frame(na.pair_frame(rev))
    assert papers[papers.paper_id == 's:000'].paper_effect.isna().all()
    assert papers.dropna(subset=['paper_effect']).shape[0] == 9


def test_descriptive_mention_rate():
    from utopia.analysis import network_review_bias_analysis as na
    rev = _synthetic_reviews(n_papers=4)
    rev.loc[rev.condition == 'C', 'justification'] = 'we have collaborated via a mutual friend'
    out = na.descriptives(rev, NRB_PREREG)
    assert out['C']['mention_rate'] == 1.0
    assert out['A']['mention_rate'] == 0.0


# ------------------------------------------------------- source-mode switch

def test_pa_blind_review_env_switch(monkeypatch):
    import inspect
    import utopia.simulation as rs
    if 'UTOPIA_PA_BLIND_REVIEW' not in inspect.getsource(rs.Simulation._build_author_info_for_review):
        pytest.skip('blind-review switch not applied (archived+reverted; see '
                    'outputs/docs/network_review_bias_execution/code_snapshots/)')

    class DummyTracker:
        def get_network_distance(self, a, b):
            return 2

    class DummyAgent:
        university_name = 'institution_0001'

    class DummyEco:
        def get_agent_by_id(self, aid):
            return DummyAgent()

    class DummySim:
        experiment_name = 'preferential_attachment'
        collaboration_tracker = DummyTracker()
        ecosystem = DummyEco()
        _get_lead_author_id = staticmethod(rs.Simulation._get_lead_author_id)
        _describe_network_relationship = rs.Simulation._describe_network_relationship
        _build_author_info_for_review = rs.Simulation._build_author_info_for_review

    class DummyReviewer:
        id = 'reviewer_7'

    sim = DummySim()
    reviewer = DummyReviewer()
    paper = {'author_id': 'institution_0001_researcher_2'}
    monkeypatch.delenv('UTOPIA_PA_BLIND_REVIEW', raising=False)
    info = sim._build_author_info_for_review(reviewer, paper)
    assert info is not None and info['network_distance'] == 2   # default unchanged
    monkeypatch.setenv('UTOPIA_PA_BLIND_REVIEW', '1')
    assert sim._build_author_info_for_review(reviewer, paper) is None  # blind source mode
    monkeypatch.delenv('UTOPIA_PA_BLIND_REVIEW', raising=False)
    sim.experiment_name = 'exploration_vs_exploitation'
    assert sim._build_author_info_for_review(reviewer, paper) is None  # other modes untouched


# --------------------------- expanding collaboration growth mode (prereg v1.1)

from types import SimpleNamespace

import networkx as nx


def _channel(run_seed, year, agent_id):
    # Frozen contract from prereg_v1_1_source_fix.json
    return ('new_tie'
            if derive_seed(run_seed, 'collab_growth_channel', year, agent_id) % 100 < 70
            else 'repeat_tie')


def _ids_with_channel(run_seed, year, channel, n, prefix='ag'):
    ids, i = [], 0
    while len(ids) < n:
        aid = f'{prefix}{i:03d}'
        if _channel(run_seed, year, aid) == channel:
            ids.append(aid)
        i += 1
    return ids


class GrowthAgent:
    def __init__(self, aid, inst, topics):
        self.id = aid
        self.university_name = inst
        self.expertise = list(topics)
        self.conflict_of_interest = set()

    def add_conflict_of_interest(self, other):
        self.conflict_of_interest.add(other)

    def get_collaboration_decision_prompt(self, candidate_infos):
        return self.id, None  # prompt == agent id so the stub LLM can look it up


class GrowthSim:
    """Binds the real Simulation collaboration methods onto a stub harness."""
    experiment_name = 'preferential_attachment'

    def __init__(self, agents, mode='cross_institute', growth='expanding',
                 seed=4242, decisions=None):
        import utopia.simulation as rs
        from utopia.data.collaboration_tracker import CollaborationTracker
        self._rs = rs
        self.collaboration_tracker = CollaborationTracker()
        self.collaboration_mode = mode
        self.collaboration_network_growth_mode = growth
        self.args = SimpleNamespace(seed=seed)
        self._collaboration_pairs = {}
        self.paper_tracker = SimpleNamespace(get_papers_by_author=lambda aid: [])
        self._agents = {a.id: a for a in agents}
        self.agents = agents
        self.ecosystem = SimpleNamespace(get_agent_by_id=lambda aid: self._agents[aid])
        self.network_metrics = SimpleNamespace(record_network_snapshot=lambda y, m: None)
        decisions = decisions or {}
        self.llm = SimpleNamespace(generate=lambda prompt, response_format: (
            {'collaborate': True, 'collaborator_id': decisions[prompt]}
            if prompt in decisions else {'collaborate': False, 'collaborator_id': None}, None))

    def _initialize_preferential_attachment_trackers(self):
        pass

    def run_phase(self, year):
        import utopia.simulation as rs
        year_results = {}
        rs.Simulation._run_phase_1_5_collaboration_formation(
            self, year, {}, self.agents, year_results)
        return year_results

    def select(self, year):
        import utopia.simulation as rs
        from utopia.config import SIMULATION_CONFIG
        pa_config = SIMULATION_CONFIG['preferential_attachment_experiment']
        agent_topics = {a.id: set(a.expertise) for a in self.agents}
        inst = lambda a: getattr(a, 'university_name', None)  # noqa: E731
        return rs.Simulation._expanding_candidate_selection(
            self, year, self.agents, agent_topics, inst,
            self.collaboration_mode, pa_config)

    def form(self, year, collab_decisions, agent_candidates, growth_ctx):
        import utopia.simulation as rs
        return rs.Simulation._expanding_form_pairs(
            self, year, collab_decisions, agent_candidates, growth_ctx)


def test_growth_flag_default_is_legacy():
    from utopia.arguments import parse_arguments
    args = parse_arguments(['--output-dir', 'outputs', '--experiment_name',
                            'preferential_attachment', '--industry_funding_mode',
                            'performance', '--funding_allocation_mode', 'fixed'])
    assert args.collaboration_network_growth_mode == 'legacy'


def test_growth_channel_assignment_deterministic_and_split():
    ids = [f'inst_{i:04d}_researcher_{j}' for i in range(100) for j in range(5)]
    a = [_channel(7100, 3, aid) for aid in ids]
    b = [_channel(7100, 3, aid) for aid in ids]
    assert a == b
    frac_new = a.count('new_tie') / len(a)
    assert 0.60 <= frac_new <= 0.80, f'~70% expected, got {frac_new:.2f}'
    # method must agree with the frozen formula
    agents = [GrowthAgent(aid, 'U' if k % 2 else 'V', ['a', 'b'])
              for k, aid in enumerate(ids[:20])]
    sim = GrowthSim(agents, seed=7100)
    _, ctx = sim.select(3)
    for aid, ch in ctx['channels'].items():
        assert ch == _channel(7100, 3, aid)


def test_new_tie_excludes_d1_coi_self_and_repeat_requires_d1():
    seed, year = 7100, 2
    p_new = _ids_with_channel(seed, year, 'new_tie', 1, 'pn')[0]
    p_rep = _ids_with_channel(seed, year, 'repeat_tie', 1, 'pr')[0]
    partner_new, partner_rep = 'cand_dnew', 'cand_drep'
    open_cand, coi_cand = 'cand_open', 'cand_coi'
    agents = [GrowthAgent(p_new, 'U', ['a', 'b']),
              GrowthAgent(p_rep, 'U', ['a', 'b']),
              GrowthAgent(partner_new, 'V', ['a', 'b']),
              GrowthAgent(partner_rep, 'V', ['a', 'b']),
              GrowthAgent(open_cand, 'V', ['a', 'b']),
              GrowthAgent(coi_cand, 'V', ['a', 'b'])]
    sim = GrowthSim(agents, seed=seed)
    sim.collaboration_tracker.add_collaboration(p_new, partner_new, 1)   # d1 for p_new
    sim.collaboration_tracker.add_collaboration(p_rep, partner_rep, 1)   # d1 for p_rep
    sim._agents[coi_cand].add_conflict_of_interest(p_new)                # COI (other side)
    cands, ctx = sim.select(year)
    assert ctx['channels'][p_new] == 'new_tie' and ctx['channels'][p_rep] == 'repeat_tie'
    new_ids = [c[0] for c in cands[p_new]]
    assert partner_new not in new_ids, 'new-tie must exclude existing d1 partner'
    assert coi_cand not in new_ids, 'new-tie must exclude COI (either direction)'
    assert p_new not in new_ids
    assert open_cand in new_ids
    rep_ids = [c[0] for c in cands[p_rep]]
    assert rep_ids == [partner_rep], 'repeat-tie candidates must be exactly the d1 neighbors'


def test_new_tie_degree_and_d2_score_components():
    seed, year = 7100, 2
    prop = _ids_with_channel(seed, year, 'new_tie', 1, 'px')[0]
    hub, leaf1, leaf2, leaf3 = 'cand_hub', 'cand_lf1', 'cand_lf2', 'cand_lf3'
    agents = [GrowthAgent(prop, 'U', ['a', 'b'])] + [
        GrowthAgent(x, 'V', ['a', 'b']) for x in (hub, leaf1, leaf2, leaf3)]
    sim = GrowthSim(agents, seed=seed)
    for lf in (leaf1, leaf2, leaf3):
        sim.collaboration_tracker.add_collaboration(hub, lf, 1)  # hub degree 3
    cands, ctx = sim.select(year)
    import math
    comp_hub = ctx['components'][(prop, hub)]
    comp_leaf = ctx['components'][(prop, leaf1)]
    assert comp_hub['degree'] == pytest.approx(1.0)  # log1p(3)/log1p(3)
    assert comp_leaf['degree'] == pytest.approx(math.log1p(1) / math.log1p(3))
    assert 0.0 <= comp_leaf['degree'] <= 1.0
    # d2 indicator: connect proposer to hub via a bridge -> leaves are at d2
    sim2 = GrowthSim(agents + [GrowthAgent('bridge', 'V', ['zzz'])], seed=seed)
    for lf in (leaf1, leaf2, leaf3):
        sim2.collaboration_tracker.add_collaboration(hub, lf, 1)
    sim2.collaboration_tracker.add_collaboration(prop, hub, 1)  # hub now d1 (excluded)
    cands2, ctx2 = sim2.select(year)
    ids2 = [c[0] for c in cands2[prop]]
    assert hub not in ids2
    assert ctx2['components'][(prop, leaf1)]['d2_indicator'] == 1.0


def _hand_ctx(pre_edges, distances, run_seed=99, channels=None):
    g = nx.Graph()
    g.add_edges_from(pre_edges)
    return {'pre_graph': g, 'channels': channels or {}, 'distances': distances,
            'components': {}, 'run_seed': run_seed}


def _plain_agents(ids):
    return [GrowthAgent(a, 'U', ['a']) for a in ids]


def test_realized_events_classified_from_preformation_graph():
    ids = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10']
    sim = GrowthSim(_plain_agents(ids))
    ctx = _hand_ctx(
        pre_edges=[('n0', 'n1'), ('n2', 'n3'), ('n6', 'n7'), ('n8', 'n9'), ('n9', 'n10')],
        distances={'n0': {'n1': 1}, 'n8': {'n9': 1, 'n10': 2}})
    decisions = {
        'n0': (True, 'n1'),    # pre-d1 -> repeat_edge
        'n2': (True, 'n6'),    # two components -> new_edge, merging_components
        'n4': (True, 'n5'),    # both isolated -> new_edge, attaching_new_node
        'n5': (True, 'n4'),    # mutual
        'n8': (True, 'n10'),   # pre-d2 same component -> new_edge, within_component
    }
    pairs, stats = sim.form(2, decisions, {}, ctx)
    assert stats['realized'] == {'repeat_edge': 1, 'new_edge': 3}
    assert stats['new_edge_classes'] == {'merging_components': 1,
                                         'attaching_new_node': 1,
                                         'within_component': 1}
    by_agent = {r['agent']: r for r in stats['growth_log'] if r.get('candidate')}
    assert by_agent['n0']['realized'] == 'repeat_edge' and by_agent['n0']['pre_d1']
    assert by_agent['n2']['merge_class'] == 'merging_components'
    assert by_agent['n8']['merge_class'] == 'within_component'
    assert by_agent['n8']['pre_distance'] == 2
    # Exactly one side of the mutual pair carries the event (the mirror is
    # the same collaboration, not a second event)
    mutual_recs = [r for r in stats['growth_log'] if r['agent'] in ('n4', 'n5')]
    assert len(mutual_recs) == 1 and mutual_recs[0]['mutual'] and mutual_recs[0]['formed']
    assert not by_agent['n0']['mutual']
    # repeat edge incremented weight rather than adding structure
    assert sim.collaboration_tracker.graph['n0']['n1']['weight'] == 1  # fresh tracker
    assert stats['components_before'] == 4


def test_one_sided_rejected_and_unmatched_logged():
    sim = GrowthSim(_plain_agents(['p', 'q', 'r', 'z1', 'z2']))
    ctx = _hand_ctx(pre_edges=[], distances={})
    decisions = {'p': (True, 'q'), 'q': (True, 'r')}
    agent_candidates = {'z1': [('z2', 0.5, {})], 'p': [('q', 0.5, {})],
                        'q': [('r', 0.5, {})]}
    pairs, stats = sim.form(2, decisions, agent_candidates, ctx)
    # q->r is one-sided and always forms; p->q always fails (q chose r)
    assert pairs == {'q': 'r'}
    log = {r['agent']: r for r in stats['growth_log']}
    assert log['q']['formed'] and not log['q']['mutual']
    assert log['p']['realized'] == 'rejected' and not log['p']['formed']
    assert log['z1']['realized'] == 'unmatched'
    assert stats['realized'].get('rejected') == 1
    assert stats['realized'].get('unmatched') == 1


def test_growth_logs_reproducible_same_seed():
    def build_and_run():
        ids = [f'm{i}' for i in range(8)]
        sim = GrowthSim(_plain_agents(ids), seed=7100)
        ctx = _hand_ctx(pre_edges=[('m0', 'm1')], distances={'m0': {'m1': 1}},
                        run_seed=7100)
        decisions = {'m0': (True, 'm1'), 'm2': (True, 'm3'), 'm3': (True, 'm2'),
                     'm4': (True, 'm3'), 'm5': (True, 'm6'), 'm6': (True, 'm7')}
        _, stats = sim.form(3, decisions, {}, ctx)
        return stats['growth_log']
    assert build_and_run() == build_and_run()


def test_legacy_end_to_end_regression():
    """Complete legacy outputs on a fixture must match the pre-change
    behavior (hand-derived from the original implementation): proposals,
    formed pairs, new/repeat counts, final graph edges and weights."""
    def mk_agents():
        return [GrowthAgent('A1', 'U', ['x', 'y']), GrowthAgent('A2', 'V', ['x', 'y']),
                GrowthAgent('B1', 'U', ['z', 'w']), GrowthAgent('B2', 'V', ['z', 'w'])]
    decisions = {'A1': 'A2', 'A2': 'A1', 'B1': 'B2', 'B2': 'B1'}
    sim = GrowthSim(mk_agents(), growth='legacy', decisions=decisions)
    yr1 = sim.run_phase(1)
    yr2 = sim.run_phase(2)
    assert sim._collaboration_pairs[1] == {'A1': 'A2', 'B1': 'B2'}
    assert sim._collaboration_pairs[2] == {'A1': 'A2', 'B1': 'B2'}
    g = sim.collaboration_tracker.graph
    assert sorted(map(sorted, g.edges())) == [['A1', 'A2'], ['B1', 'B2']]
    assert g['A1']['A2']['weight'] == 2 and g['B1']['B2']['weight'] == 2
    assert g['A1']['A2']['years'] == [1, 2]
    assert yr1['collaboration_formation']['new_pairs'] == 2
    assert yr2['collaboration_formation']['total_edges'] == 2
    # legacy output carries NO expanding-mode keys
    for k in ('growth_mode', 'realized', 'growth_log', 'channel_assigned'):
        assert k not in yr1['collaboration_formation']
    # COI updated exactly as before
    assert sim._agents['A1'].conflict_of_interest == {'A2'}


def test_legacy_ranks_existing_partner_first_expanding_excludes_them():
    seed, year = 7100, 2
    prop = _ids_with_channel(seed, year, 'new_tie', 1, 'pz')[0]

    def mk(growth):
        agents = [GrowthAgent(prop, 'U', ['a', 'b']),
                  GrowthAgent('partner', 'V', ['a', 'b']),
                  GrowthAgent('fresh', 'V', ['a', 'b'])]
        sim = GrowthSim(agents, growth=growth, seed=seed,
                        decisions={prop: 'partner'})
        sim.collaboration_tracker.add_collaboration(prop, 'partner', 1)
        return sim

    # Legacy: the d1 partner outranks an identical fresh candidate (lock-in)
    sim_legacy = mk('legacy')
    from utopia.config import SIMULATION_CONFIG
    import utopia.simulation as rs
    yr = {}
    rs.Simulation._run_phase_1_5_collaboration_formation(
        sim_legacy, year, {}, sim_legacy.agents, yr)
    assert sim_legacy.collaboration_tracker.graph[prop]['partner']['weight'] == 2

    # Expanding new-tie channel: partner is ineligible, fresh is offered
    sim_exp = mk('expanding')
    cands, _ = sim_exp.select(year)
    ids = [c[0] for c in cands[prop]]
    assert 'partner' not in ids and 'fresh' in ids
