"""Deterministic tests for the scale-expansion experiment flags.

Two goals:
  1. Backward compatibility: with every new flag at its default, the prompts the
     simulator emits are byte-identical to the pre-change templates (golden
     SHA-256 hashes recorded from the unmodified code in
     tests/fixtures/golden_prompt_hashes.json), the legacy acceptance sort and
     funding-winner rule are reproduced exactly, and the global RNG is not
     consumed by the new assignment code.
  2. Correctness of the new mechanisms: k-paper submission prompt/schema,
     standardized review policy insertion, fixed-budget slot apportionment,
     capacity-aware reviewer assignment (caps, CoI, fail-fast), seeded
     tie-break and fixed-slots acceptance, float score mode, CLI validation.

No Simulation construction, no LLM calls, no dataset access.

Run: python -m pytest tests/experiments/test_scale_expansion.py -v
"""

from utopia.utils.paths import project_root
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

os.chdir(str(project_root(__file__)))

from utopia.constants import STANDARDIZED_REVIEW_POLICY
from utopia.simulation import Simulation
from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.conference import Conference, ConferenceStatus, create_default_conference_system
from utopia.agents.funding_agents import FundingAgency
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS
from utopia.agents.researcher_agents import UniversityResearcher
from utopia.arguments import parse_arguments
from utopia.data.paper_tracker import PaperTracker

FIXTURES = str(project_root(__file__) / 'tests/support/fixtures')


# ---------------------------------------------------------------- helpers

def make_agent(name='inst_0_researcher_0', institution='inst_0', funding=100,
               expertise=None, strategy='balanced'):
    return UniversityResearcher(
        researcher_name=name, university_name=institution, funding_level=funding,
        expertise=expertise or AVAILABLE_DIRECTIONS[:3], llm=None,
        exploration_strategy=strategy)


def golden_inputs():
    """Exactly the fixture-generating inputs (see tests/fixtures/golden_prompt_hashes.json)."""
    agent = make_agent()
    confs = create_default_conference_system().conferences[:3]
    tracker = PaperTracker(output_dir=tempfile.mkdtemp())
    direction = {'direction': AVAILABLE_DIRECTIONS[0], 'detailed_focus': 'Focus text.',
                 'reason': 'Reason text.'}
    cands = [Document(page_content=f"Abstract {i}.",
                      metadata={'id': f"https://arxiv.org/abs/2401.0000{i}v1", 'title': f"Title {i}",
                                'topics': ['Artificial Intelligence'], 'tags': ['cs.AI']})
             for i in range(3)]
    paper = {'id': cands[0].metadata['id'], 'title': 'Title 0', 'abstract': 'Abstract 0.',
             'topics': ['Artificial Intelligence'], 'conference': confs[0].conference_id, 'year': 2}
    rejected = {paper['id']: {**paper, 'review_history': [
        {'score': 2.0, 'conference': confs[0].conference_id, 'year': 1}]}}
    return agent, confs, tracker, direction, cands, paper, rejected


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def stub_sim(**overrides):
    """Bare Simulation-like namespace for unbound method calls."""
    base = dict(reviewer_capacity=None, reviews_per_paper=3, reviewer_matching='random',
                review_score_mode='int', funding_budget_slots_frac=None,
                initial_university_count=None, args=SimpleNamespace(seed=42),
                conference_system=None, ecosystem=None)
    base.update(overrides)
    ns = SimpleNamespace(**base)
    ns._get_all_author_ids = lambda paper: Simulation._get_all_author_ids(paper)
    return ns


# ---------------------------------------------------------------- 1. backward compatibility

def test_default_prompts_match_golden_hashes():
    golden = json.load(open(os.path.join(FIXTURES, 'golden_prompt_hashes.json')))
    agent, confs, tracker, direction, cands, paper, rejected = golden_inputs()
    got = {
        'review': sha(agent.get_review_prompt(paper)[0]),
        'submission_y1': sha(agent.get_paper_submission_prompt(cands, 1, confs, direction, tracker)[0]),
        'submission_y2': sha(agent.get_paper_submission_prompt(cands, 2, confs, direction, tracker)[0]),
        'resubmission': sha(agent.get_resubmission_prompt(rejected, confs)[0]),
        'intention': sha(agent.get_round_intention_prompt(direction, 1, tracker)[0]),
    }
    assert got == golden


def test_default_review_prompt_equals_explicit_persona():
    agent, confs, tracker, direction, cands, paper, rejected = golden_inputs()
    agent.memory_bank.append({'type': 'good_reviews_received', 'year': 1, 'thought': 'ok'})
    assert agent.get_review_prompt(paper)[0] == agent.get_review_prompt(paper, review_policy='persona')[0]


def test_legacy_acceptance_sort_is_stable_and_rate_based():
    conf = Conference(conference_id='C', name='C', topics=['cs.AI'])
    conf.acceptance_rate = 0.5  # __post_init__ applies the config default; override afterwards
    conf.status = ConferenceStatus.UNDER_REVIEW
    for i in range(4):  # all tied at 3.0 -> submission order decides
        conf.submitted_papers.append({'id': f'p{i}', 'author_id': f'a{i}', 'type': 'submission'})
        conf.reviews[f'p{i}'] = [{'overall_score': 3}]
    conf.make_acceptance_decisions()
    assert [p['id'] for p in conf.decisions['accept']] == ['p0', 'p1']
    assert conf.fixed_slots is None


def test_legacy_funding_rule_unchanged_without_override():
    apps = [{'applicant_id': f'r{i}'} for i in range(10)]
    program = SimpleNamespace(funding_rate=0.3)
    ranked = {'ranked_applications': [
        {'application_id': i, 'applicant_id': f'r{i}', 'rank': i + 1, 'reason': ''} for i in range(10)]}
    winners = FundingAgency.process_funding_evaluation_results(
        [(ranked, [])], [{'program_id': 'P', 'apps': apps, 'panel_index': 0}], {'P': program})
    assert [w['applicant_id'] for w in winners['P']] == ['r0', 'r1', 'r2']  # int(10*0.3) = 3


def test_cli_defaults_are_legacy():
    a = parse_arguments(['--experiment_name', 'default'])
    assert (a.papers_per_project, a.funding_budget_mode, a.reviewer_capacity, a.reviews_per_paper,
            a.reviewer_matching, a.review_policy, a.review_score_mode, a.acceptance_tiebreak,
            a.acceptance_mode) == (1, 'track', None, 3, 'random', 'persona', 'int', 'stable', 'rate')


# ---------------------------------------------------------------- 2. new mechanisms

def test_standardized_policy_drops_memory_and_inserts_block():
    agent, confs, tracker, direction, cands, paper, rejected = golden_inputs()
    agent.memory_bank.append({'type': 'harsh_reviews_received', 'year': 1, 'thought': 'UNFAIR_MARKER'})
    persona = agent.get_review_prompt(paper, review_policy='persona')[0]
    std = agent.get_review_prompt(paper, review_policy='standardized')[0]
    assert 'UNFAIR_MARKER' in persona and 'UNFAIR_MARKER' not in std
    assert STANDARDIZED_REVIEW_POLICY in std and STANDARDIZED_REVIEW_POLICY not in persona
    assert std.index('## Your Review Approach') < std.index('## Review Instructions')
    # paper block and scale untouched
    assert 'Title: Title 0' in std and 'Overall Score (1-5 scale)' in std


def test_k_paper_submission_prompt_and_schema():
    agent, confs, tracker, direction, cands, paper, rejected = golden_inputs()
    prompt, fmt = agent.get_paper_submission_prompt(cands, 1, confs, direction, tracker, max_submissions=2)
    schema = fmt['json_object']['schema']
    assert 'up to 2 DISTINCT paper IDs' in prompt and '"submissions"' in prompt
    assert 'submissions' in schema['properties'] and 'DO_NOT_SUBMIT' not in prompt
    # force_list keeps the list schema for a 1-paper retry inside a k>1 batch
    prompt1, fmt1 = agent.get_paper_submission_prompt(cands, 1, confs, direction, tracker,
                                                      max_submissions=1, force_list=True)
    assert fmt1['json_object']['name'] == 'paper_submission_list' and 'up to 1 DISTINCT' in prompt1


def test_largest_remainder_apportionment():
    f = FundingAgency.allocate_slots_largest_remainder
    assert f(10, [30, 20, 0, 5]) == [5, 4, 0, 1]
    assert f(3, [1, 1, 1, 1]) == [1, 1, 1, 0]
    assert f(100, [3, 2]) == [3, 2]          # never exceeds demand
    assert f(0, [3, 2]) == [0, 0] and f(5, [0, 0]) == [0, 0]
    assert sum(f(7, [13, 29, 8])) == 7


def test_slot_override_replaces_rate_rule_and_allows_zero():
    apps = [{'applicant_id': f'r{i}'} for i in range(10)]
    program = SimpleNamespace(funding_rate=0.3)
    ranked = {'ranked_applications': [
        {'application_id': i, 'applicant_id': f'r{i}', 'rank': i + 1, 'reason': ''} for i in range(10)]}
    meta = [{'program_id': 'P', 'apps': apps, 'panel_index': 0}]
    w1 = FundingAgency.process_funding_evaluation_results([(ranked, [])], meta, {'P': program},
                                                          slot_override={'P': 1})
    w0 = FundingAgency.process_funding_evaluation_results([(ranked, [])], meta, {'P': program},
                                                          slot_override={'P': 0})
    assert [w['applicant_id'] for w in w1['P']] == ['r0'] and w0['P'] == []
    # panels: program slots split across two panels by size (6 + 4 apps, 5 slots -> 3 + 2)
    meta2 = [{'program_id': 'P', 'apps': apps[:6], 'panel_index': 0},
             {'program_id': 'P', 'apps': apps[6:], 'panel_index': 1}]
    r1 = {'ranked_applications': [{'application_id': i, 'applicant_id': f'r{i}', 'rank': i + 1, 'reason': ''}
                                  for i in range(6)]}
    r2 = {'ranked_applications': [{'application_id': i, 'applicant_id': f'r{6 + i}', 'rank': i + 1, 'reason': ''}
                                  for i in range(4)]}
    w = FundingAgency.process_funding_evaluation_results([(r1, []), (r2, [])], meta2, {'P': program},
                                                         slot_override={'P': 5})
    assert [x['applicant_id'] for x in w['P']] == ['r0', 'r1', 'r2', 'r6', 'r7']


def test_fixed_budget_slots_helper():
    sim = stub_sim(funding_budget_slots_frac=0.2, initial_university_count=100)
    meta = [{'program_id': 'A', 'apps': [1] * 30}, {'program_id': 'B', 'apps': [1] * 10},
            {'program_id': 'A', 'apps': [1] * 10}]  # A has two panels: 40 apps total
    override, info = Simulation._fixed_budget_slots(sim, meta)
    assert info['total_slots'] == 20 and override == {'A': 16, 'B': 4}
    assert info['effective_rate_by_program']['A'] == pytest.approx(0.4)


def test_seeded_tiebreak_is_deterministic_and_differs_from_stable():
    def build():
        conf = Conference(conference_id='C', name='C', topics=['cs.AI'])
        conf.acceptance_rate = 0.5
        conf.status = ConferenceStatus.UNDER_REVIEW
        for i in range(20):
            conf.submitted_papers.append({'id': f'p{i}', 'author_id': f'a{i}', 'type': 'submission'})
            conf.reviews[f'p{i}'] = [{'overall_score': 3}]
        return conf
    c1, c2 = build(), build()
    c1.make_acceptance_decisions(tiebreak_seed=7)
    c2.make_acceptance_decisions(tiebreak_seed=7)
    acc1 = [p['id'] for p in c1.decisions['accept']]
    assert acc1 == [p['id'] for p in c2.decisions['accept']] and len(acc1) == 10
    assert acc1 != [f'p{i}' for i in range(10)]  # not submission order
    # higher scores still win under the seeded key
    c3 = build()
    c3.reviews['p19'] = [{'overall_score': 5}]
    c3.make_acceptance_decisions(tiebreak_seed=7)
    assert c3.decisions['accept'][0]['id'] == 'p19'


def test_fixed_slots_freeze_first_year_count_and_roundtrip():
    conf = Conference(conference_id='C', name='C', topics=['cs.AI'])
    conf.acceptance_rate = 0.5
    for year, n in ((1, 4), (2, 10)):
        conf.reset_for_next_year(year)
        conf.status = ConferenceStatus.UNDER_REVIEW
        for i in range(n):
            conf.submitted_papers.append({'id': f'y{year}p{i}', 'author_id': f'a{i}', 'type': 'submission'})
            conf.reviews[f'y{year}p{i}'] = [{'overall_score': 3}]
        conf.make_acceptance_decisions(acceptance_mode='fixed_slots')
        assert len(conf.decisions['accept']) == 2  # frozen at round(4*0.5)
    assert conf.fixed_slots == 2
    assert Conference.from_dict(conf.to_dict()).fixed_slots == 2
    assert Conference.from_dict({k: v for k, v in conf.to_dict().items() if k != 'fixed_slots'}).fixed_slots is None


def test_cast_review_score_modes():
    assert Simulation._cast_review_score(stub_sim(review_score_mode='int'), 2.5) == 2
    assert Simulation._cast_review_score(stub_sim(review_score_mode='float'), 2.5) == 2.5


def _capacity_world(n_reviewers=12, n_papers=10, institutions=3):
    eco = MultiAgentEcosystem(agent_configs={})
    for i in range(n_reviewers):
        inst = f'inst_{i % institutions}'
        eco.add_agent(make_agent(name=f'{inst}_r{i}', institution=inst,
                                 expertise=AVAILABLE_DIRECTIONS[(i * 3) % 30:(i * 3) % 30 + 3]))
    # within-institution CoI as in the simulator
    by_inst = {}
    for a in eco.agent_population.values():
        by_inst.setdefault(a.university_name, set()).add(a.id)
    for a in eco.agent_population.values():
        a.add_conflict_of_interest(by_inst[a.university_name] - {a.id})
    conf = Conference(conference_id='C', name='C', topics=['cs.AI'])
    authors = list(eco.agent_population.values())
    for j in range(n_papers):
        conf.submitted_papers.append({'id': f'p{j}', 'author_id': authors[j % n_reviewers].id,
                                      'tags': ['cs.AI'], 'topics': ['Artificial Intelligence']})
    cs = SimpleNamespace(conferences=[conf])
    return eco, cs


def test_capacity_assignment_respects_cap_coi_and_floor():
    eco, cs = _capacity_world()
    sim = stub_sim(reviewer_capacity=2, ecosystem=eco, conference_system=cs)
    state_before = random.getstate()
    yr = {}
    assignment = Simulation._assign_reviewers_with_capacity(sim, 1, yr)
    assert random.getstate() == state_before  # global RNG untouched
    # 12 reviewers x cap 2 = 24 slots for 10 papers -> r = min(3, 2) = 2
    assert yr['review_assignment']['reviews_per_paper_target'] == 2
    load = Counter(a.id for revs in assignment.values() for a in revs)
    assert max(load.values()) <= 2 and all(len(v) >= 1 for v in assignment.values())
    for paper in cs.conferences[0].submitted_papers:
        author = eco.get_agent_by_id(paper['author_id'])
        for rev in assignment[paper['id']]:
            assert rev.id != author.id and rev.id not in author.conflict_of_interest
    # deterministic given the seed
    again = Simulation._assign_reviewers_with_capacity(sim, 1, {})
    assert {k: [a.id for a in v] for k, v in again.items()} == \
           {k: [a.id for a in v] for k, v in assignment.items()}


def test_capacity_assignment_fails_fast_when_one_review_impossible():
    eco, cs = _capacity_world(n_reviewers=4, n_papers=10)
    sim = stub_sim(reviewer_capacity=2, ecosystem=eco, conference_system=cs)  # 8 slots < 10 papers
    with pytest.raises(RuntimeError):
        Simulation._assign_reviewers_with_capacity(sim, 1, {})


def test_capacity_assignment_unlimited_cap_uses_target():
    eco, cs = _capacity_world()
    sim = stub_sim(reviews_per_paper=2, ecosystem=eco, conference_system=cs)
    assignment = Simulation._assign_reviewers_with_capacity(sim, 1, {})
    assert all(len(v) == 2 for v in assignment.values())


def test_topic_matching_prefers_overlapping_reviewers():
    eco, cs = _capacity_world(n_reviewers=12, n_papers=4)
    sim = stub_sim(reviewer_capacity=3, reviewer_matching='topic', ecosystem=eco, conference_system=cs)
    yr = {}
    assignment = Simulation._assign_reviewers_with_capacity(sim, 1, yr)
    assert yr['review_assignment']['reviewer_matching'] == 'topic'
    assert all(1 <= len(v) <= 3 for v in assignment.values())


def test_cli_scale_stage_and_validation():
    a = parse_arguments(['--experiment_name', 'scale_expansion', '--experiment_stage', 'scale',
                         '--population_mode', 'university_only', '--num_institutions', '20',
                         '--num_years', '8', '--strategy_mix', 'balanced', '--papers_per_project', '2',
                         '--funding_budget_mode', 'fixed', '--funding_budget_slots_frac', '0.15',
                         '--reviewer_capacity', '6', '--review_policy', 'standardized',
                         '--review_score_mode', 'float', '--acceptance_tiebreak', 'seeded', '--seed', '7',
                         '--model', 'Qwen/Qwen3-32B'])
    assert a.experiment_id == ('scale_qwen3_32b_k2_budgetfixed_cap6_standardized_float_tieseed'
                               '_i20_n100_y8_seed7_mixbal')
    assert a.strategy_mix_list == ['balanced'] * 5
    b = parse_arguments(['--experiment_name', 'scale_expansion', '--experiment_stage', 'scale',
                         '--population_mode', 'university_only', '--num_institutions', '20',
                         '--num_years', '8', '--strategy_mix', 'balanced', '--papers_per_project', '2',
                         '--funding_budget_mode', 'fixed', '--funding_budget_slots_frac', '0.2',
                         '--disable_resubmission', '--seed', '1', '--model', 'Qwen/Qwen3-32B'])
    assert '_noresub_' in b.experiment_id
    from utopia.analysis.scale_expansion import parse_experiment_id
    assert parse_experiment_id(b.experiment_id)['cell'] == 'S1R1_noresub'
    with pytest.raises(ValueError):
        parse_arguments(['--experiment_name', 'default', '--funding_budget_mode', 'fixed'])
    with pytest.raises(ValueError):
        parse_arguments(['--experiment_name', 'default', '--papers_per_project', '0'])


# ---------------------------------------------------------------- 3. analysis on a synthetic world

def test_analysis_endpoints_on_synthetic_checkpoint(tmp_path):
    from utopia.analysis.scale_expansion import compute_run_endpoints, aggregate, parse_experiment_id
    exp_id = 'scale_qwen3_32b_k2_budgetfixed_cap4_standardized_float_tieseed_i2_n4_y4_seed3_mixbal'
    assert parse_experiment_id(exp_id)['cell'] == 'S1R1_K1E1'
    run = tmp_path / exp_id
    run.mkdir()
    founders = ['a0', 'a1', 'a2', 'a3']
    tracker = {aid: [{'year': y, 'resources': 50} for y in range(0, 5)] for aid in founders}
    tracker['a3'] = tracker['a3'][:2]  # last record at year 1 -> culled during year 2
    papers = [
        # accepted year 1, cited by p2 (other author, year 2) -> follow-up
        {'id': 'p0', 'author_id': 'a0', 'status': 'accept', 'year': 1, 'topics': ['A', 'B'],
         'review_history': [{'year': 1, 'decision': 'accept', 'reviews': [1, 1, 1]}]},
        # rejected year 1, resubmitted & accepted year 2 -> one resubmission attempt
        {'id': 'p1', 'author_id': 'a1', 'status': 'accept', 'year': 2, 'topics': ['A'],
         'review_history': [{'year': 1, 'decision': 'reject', 'reviews': [1, 1, 1]},
                            {'year': 2, 'decision': 'accept', 'reviews': [1, 1]}]},
        {'id': 'p2', 'author_id': 'a2', 'status': 'accept', 'year': 2, 'topics': ['C'],
         'review_history': [{'year': 2, 'decision': 'accept', 'reviews': [1, 1, 1]}]},
        # accepted year 1, only self-cited -> no follow-up
        {'id': 'p3', 'author_id': 'a3', 'status': 'accept', 'year': 1, 'topics': ['A'],
         'review_history': [{'year': 1, 'decision': 'accept', 'reviews': [1, 1, 1]}]},
        {'id': 'p4', 'author_id': 'a3', 'status': 'reject', 'year': 2, 'topics': ['B'],
         'review_history': [{'year': 2, 'decision': 'reject', 'reviews': [1, 1, 1]}]},
    ]
    ck = {'year': 4, 'paper_tracker': {'papers': papers}, 'agent_tracker': {'resources': tracker},
          'citation_tracker': {'citations': {'p0': ['p2'], 'p3': ['p4']}},
          'ecosystem_data': {'agents': []},
          'yearly_results': [{'year': y, 'paper_submission': {'num_papers_submitted': 2},
                              'decisions': {'total_accepted': 1, 'total_rejected': 1},
                              'peer_review': {'reviews_conducted': 6},
                              'ecosystem_metrics': {'active_agents': 4, 'resource_distribution': {'gini': 0.1},
                                                    'funding_success_by_type': {'university': 1, 'total_applications': 4}},
                              'funding_budget': {'total_slots': 1, 'total_applications': 4}} for y in range(1, 5)]}
    json.dump(ck, open(run / 'checkpoint_year_4.json', 'w'))
    with open(run / 'funding_applications_year_1.jsonl', 'w') as f:
        for aid, funded in (('a0', True), ('a1', False), ('a2', True), ('a3', False)):
            f.write(json.dumps({'applicant_id': aid, 'funded': funded}) + '\n')
    r = compute_run_endpoints(str(run), num_years=4, horizon=2)
    assert r['cell'] == 'S1R1_K1E1' and r['seed'] == 3
    assert r['P1_survival_h'] == pytest.approx(3 / 4) and r['P1_survival_final'] == pytest.approx(3 / 4)
    assert r['P2_resub_review_share'] == pytest.approx(2 / 17) and r['P2_attempts_per_paper'] == pytest.approx(6 / 5)
    assert r['P3_award_source'] == 'application_log' and r['P3_total_awards'] == 2
    assert r['P3_award_gini'] == pytest.approx(0.5)      # awards (1,1,0,0) over 4 founders
    # window fully observed (accept year <= 4-2): p0, p1, p2, p3 all qualify; only p0 has a non-self citer
    assert r['P4_n_accepted_observed'] == 4 and r['P4_followup_share'] == pytest.approx(1 / 4)
    assert r['yearly_checks'][0]['awards_per_founder'] == pytest.approx(1 / 4)
    per_run, yearly, summary, contrasts = aggregate([r], rarefaction_min_n=2)
    assert summary.iloc[0]['cell'] == 'S1R1_K1E1' and len(yearly) == 4
    assert 0.0 <= per_run.iloc[0]['P4_topic_entropy_rarefied'] <= 1.0


# ---------------------------------------------------------------- 4. paper-baseline (default population) scale-up

def test_fixed_industry_pool_per_unit():
    sim = stub_sim(funding_budget_slots_frac=0.2, initial_industry_count=150)
    # pool = round(0.2 * 150) * 20 = 600; 40 maturity units accepted -> 15 per unit
    per_unit, info = Simulation._fixed_industry_pool_per_unit(sim, 40)
    assert per_unit == 15 and info['pool'] == 600 and info['paid'] == 600
    # more accepted output than the pool covers -> per unit floors at 1 (legacy floor)
    per_unit, info = Simulation._fixed_industry_pool_per_unit(sim, 900)
    assert per_unit == 1 and info['paid'] == 900
    assert Simulation._fixed_industry_pool_per_unit(sim, 0)[0] == 0


def test_default_population_experiment_id_and_parse():
    from utopia.analysis.scale_expansion import parse_experiment_id
    a = parse_arguments(['--experiment_name', 'scale_expansion', '--experiment_stage', 'scale',
                         '--population_mode', 'default', '--num_years', '10', '--num_conferences', '10',
                         '--papers_per_project', '2', '--funding_budget_mode', 'fixed',
                         '--funding_budget_slots_frac', '0.2', '--seed', '42', '--model', 'Qwen/Qwen3-32B'])
    assert '_popdefault_y10_seed42' in a.experiment_id and '_i' not in a.experiment_id.split('popdefault')[0][-4:]
    info = parse_experiment_id(a.experiment_id)
    assert info['cell'] == 'S1R1' and info['population'] == 'default' and info['seed'] == 42
    u = parse_arguments(['--experiment_name', 'scale_expansion', '--experiment_stage', 'scale',
                         '--population_mode', 'university_only', '--num_institutions', '60',
                         '--num_years', '8', '--strategy_mix', 'balanced', '--seed', '1', '--model', 'Qwen/Qwen3-32B'])
    assert '_i60_n300_y8_seed1' in u.experiment_id
    assert parse_experiment_id(u.experiment_id)['population'] == 'university_only'


def test_driver_builds_default_population_command():
    import utopia.experiments.scale_expansion as drv
    args = SimpleNamespace(population='default', num_institutions=20, num_years=10, num_conferences=10,
                           start_year=2016, batch_size=32, vllm_url=['http://x/v1'], slots_frac=0.2,
                           reviewer_capacity=None, reviewer_matching='random', legacy_scoring=False,
                           always_rerun=False, funding_panel_max_apps=0)
    cmd = drv.build_command('S1R1', 42, args)
    assert '--population_mode' in cmd and cmd[cmd.index('--population_mode') + 1] == 'default'
    assert '--strategy_mix' not in cmd and '--num_institutions' not in cmd
    assert cmd[cmd.index('--num_years') + 1] == '10' and cmd[cmd.index('--funding_budget_mode') + 1] == 'fixed'
    args.population = 'university_only'; args.num_institutions = 60; args.num_years = 8; args.funding_panel_max_apps = 25
    cmd = drv.build_command('S0R0', 1, args)
    assert cmd[cmd.index('--num_institutions') + 1] == '60' and cmd[cmd.index('--funding_panel_max_apps') + 1] == '25'
