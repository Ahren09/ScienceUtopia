"""E0 deterministic tests for the exploration-vs-exploitation experiment.

Covers the plan_0708.md section 7 E0 list. No Simulation construction, no
Qwen/vLLM calls — everything runs on hand-built objects and stub LLMs.

Run: python -m pytest tests/experiments/test_exploration.py -v
"""

from utopia.utils.paths import project_root
import os
import sys

import numpy as np
import pytest

os.chdir(str(project_root(__file__)))

from utopia.simulation import build_population_blueprint, check_citation_integrity
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS, create_research_directions_batch
from utopia.agents.funding_agents import FundingAgency
from utopia.arguments import parse_arguments
from utopia.data.paper_tracker import PaperTracker, ArchivedPaper
from utopia.data.tracker import CitationTracker
from utopia.metrics.cd_index import CDIndexCalculator
from utopia.data.keyword_extractor import KeywordExtractor
from utopia.utils.seeding import derive_seed


# ---------------------------------------------------------------- helpers

class FakeBatchLLM:
    """Stub with generate_batch returning canned JSON responses in order."""

    def __init__(self, responses):
        self.responses = responses
        self.model_name = 'fake'

    def generate_batch(self, prompts, **kwargs):
        assert len(prompts) == len(self.responses), \
            f"expected {len(self.responses)} prompts, got {len(prompts)}"
        return [(r, []) for r in self.responses]


class StubAgent:
    def __init__(self, agent_id, expertise, strategy):
        self.id = agent_id
        self.expertise = expertise
        self.exploration_strategy = strategy
        self.memory_bank = []
        self.resources = 100
        self.funding_success_history = {}

    def _get_personality_prompt(self, include_exploration_strategy=False):
        return f"You are researcher {self.id} with strategy {self.exploration_strategy}."

def test_mapped_funding_schema_preserves_order_and_rejects_identifier_mismatch():
    import jsonschema
    apps = [{'applicant_id': 'alice'}, {'applicant_id': 'bob'}]
    schema = FundingAgency.mapped_evaluation_format(apps)['json_object']['schema']
    good = {'ranked_applications': [
        {'application_id': 1, 'applicant_id': 'bob', 'rank': 1, 'reason': 'preferred'},
        {'application_id': 0, 'applicant_id': 'alice', 'rank': 2, 'reason': 'next'},
    ]}
    jsonschema.validate(good, schema)
    good['ranked_applications'][0]['applicant_id'] = 'alice'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(good, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'ranked_applications': []}, schema)


def test_strict_funding_stops_before_awards_when_panel_coverage_is_incomplete():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from utopia.simulation import Simulation
    from utopia.models.models import VLLMServerModel
    simulation = Simulation.__new__(Simulation)
    simulation.args = SimpleNamespace(strict_funding_rankings=True)
    simulation.llm = VLLMServerModel.__new__(VLLMServerModel)
    simulation.llm.generate_batch = Mock(return_value=[(None, [])])
    metadata = [{'program_id': 'program', 'panel_index': 0,
                 'apps': [{'applicant_id': 'alice'}]}]
    with pytest.raises(RuntimeError, match='Incomplete funding ranking'):
        simulation._rank_funding_applications(1, ['panel'], metadata, {})
    assert simulation.llm.generate_batch.call_count == 3
    assert all(call.kwargs['response_formats'][0]['json_object']['schema']
               ['properties']['ranked_applications']['minItems'] == 1
               for call in simulation.llm.generate_batch.call_args_list)


def make_paper(pid, author, year, status='accept'):
    return ArchivedPaper(
        id=pid, title=f"Paper {pid}", abstract=f"Abstract of {pid}",
        author_id=author, author_type='university', year=year,
        conference='conf', status=status)


def build_citation_world(papers, edges):
    """papers: [(pid, author, year)], edges: [(citing, cited, year)]"""
    pt = PaperTracker(output_dir='/tmp')
    for pid, author, year in papers:
        pt.papers_by_id[pid] = make_paper(pid, author, year)
    ct = CitationTracker()
    for citing, cited, year in edges:
        ct.add_citations(citing, cited, year)
    return pt, ct


# ------------------------------------------------- 1-3: population blueprint

def test_analysis_exact_id_excludes_prompt_variants(tmp_path):
    import json
    from utopia.analysis.exploration_cross_seed import discover_runs

    control = 'explore_confirmatory_qwen3_32b_seed42_mix131_panel25'
    variants = [control, control + '_prompt_paraphrase_1',
                control + '_prompt_paraphrase_2']
    for experiment_id in variants:
        docs = tmp_path / 'docs' / experiment_id
        docs.mkdir(parents=True)
        (docs / 'run_manifest.json').write_text(json.dumps({
            'run_seed': 42, 'status': 'complete',
        }))

    exact = discover_runs(outputs_dir=str(tmp_path), experiment_id=control)
    assert [r['experiment_id'] for r in exact] == [control]
    prefixed = discover_runs(control, outputs_dir=str(tmp_path))
    assert {r['experiment_id'] for r in prefixed} == set(variants)
    assert discover_runs(outputs_dir=str(tmp_path),
                         experiment_id=control + '_missing') == []


def test_analysis_accepts_complete_runs_with_different_legacy_hashes(tmp_path):
    from utopia.analysis.exploration_cross_seed import validate_runs

    runs = []
    for index, legacy_hash in enumerate(('old-a', 'old-b', None)):
        docs = tmp_path / str(index)
        docs.mkdir()
        for table in ('agent_year', 'paper'):
            (docs / f'{table}.csv').write_text('id\n1\n')
        runs.append({'docs_dir': str(docs), 'status': 'complete', 'config_hash': legacy_hash})
    valid, excluded = validate_runs(runs)
    assert len(valid) == 3
    assert excluded == []
    runs[-1]['status'] = 'running'
    valid, excluded = validate_runs(runs)
    assert len(valid) == 2
    assert excluded[0]['problems'] == ['status=running']


def test_world_summary_citation_followup_and_founder_denominators():
    import copy
    import pandas as pd
    from utopia.analysis.exploration_cross_seed import summarize_completed_world

    agent_year = pd.DataFrame([
        {'agent_id': aid, 'year': year, 'strategy': 'explorer',
         'is_active': aid == 'a' or year < 4, 'num_papers': int(aid == 'a' and year in (1, 2, 4)),
         'num_accepted': int(aid == 'a' and year in (1, 4)),
         'direction_distance': 0.8, 'topic_switched': False,
         'funding': 10.0 if aid == 'a' else 0.0, 'total_citations': 0}
        for aid in ('a', 'b') for year in range(1, 5)
    ])
    age = pd.DataFrame([{
        'paper_id': 'p1', 'author_id': 'a', 'age': 3,
        'publication_year': 1, 'observation_year': 4, 'citations': 2,
    }])
    ecosystem = pd.DataFrame([
        {'year': year, 'topic_entropy_normalized': 0.75, 'occupied_directions': 3,
         'num_active_researchers': 2 if year < 4 else 1,
         'num_submissions': 1, 'num_accepted': int(year in (1, 4))}
        for year in range(1, 5)
    ])
    checkpoint = {
        'year': 4, 'phase': 5,
        'ecosystem_data': {'agents': [
            {'id': 'a', 'type': 'university', 'is_active': True,
             'funding_success_history': {'1': [{'amount': 20}]}},
            {'id': 'b', 'type': 'university', 'is_active': False},
            {'id': 'agency', 'type': 'funding_agency'},
        ]},
        'paper_tracker': {'papers': [
            {'id': 'p1', 'author_id': 'a', 'year': 1, 'status': 'accept'},
            {'id': 'p2', 'author_id': 'a', 'year': 4, 'status': 'accept'},
            {'id': 'p3', 'author_id': 'a', 'year': 2, 'status': 'reject'},
        ]},
        'citation_tracker': {'citations': {'p1': ['p2', 'background']}},
    }
    result = summarize_completed_world(agent_year, age, ecosystem, checkpoint)
    row = result['strategies'][0]
    assert row['founders'] == 2 and row['active_share_final'] == 0.5
    assert row['researchers_with_submissions'] == 1
    assert row['mean_researcher_acceptance_fraction'] == pytest.approx(2 / 3)
    assert row['citations_to_accepted_papers_at_horizon'] == 2
    assert row['papers_at_publication_age_three'] == 1
    assert row['mean_citations_at_publication_age_three'] == 2
    assert row['mean_cumulative_grants_received'] == 10
    with pytest.raises(ValueError, match='researcher-year coverage'):
        summarize_completed_world(agent_year.iloc[1:], age, ecosystem, checkpoint)
    with pytest.raises(ValueError, match='age-three coverage'):
        summarize_completed_world(agent_year, age.iloc[:0], ecosystem, checkpoint)
    bad = copy.deepcopy(checkpoint)
    bad['citation_tracker']['citations']['p1'].append('p2')
    with pytest.raises(ValueError, match='Duplicate citation'):
        summarize_completed_world(agent_year, age, ecosystem, bad)


def test_blueprint_exact_strategy_mix():
    bp = build_population_blueprint(12, 5, seed=101)
    assert len(bp) == 60
    from collections import Counter
    counts = Counter(row['strategy'] for row in bp)
    assert counts['explorer'] == 12
    assert counts['exploiter'] == 36
    assert counts['cautious_explorer'] == 12
    # exactly 1/3/1 within every institution
    by_inst = {}
    for row in bp:
        by_inst.setdefault(row['institution'], []).append(row['strategy'])
    for inst, strategies in by_inst.items():
        c = Counter(strategies)
        assert (c['explorer'], c['exploiter'], c['cautious_explorer']) == (1, 3, 1), inst


def test_blueprint_same_seed_identical():
    assert build_population_blueprint(10, 5, seed=7) == build_population_blueprint(10, 5, seed=7)


def test_blueprint_different_seed_differs():
    a = build_population_blueprint(10, 5, seed=7)
    b = build_population_blueprint(10, 5, seed=8)
    assert a != b
    # strategy or expertise assignment must actually differ somewhere
    assert any(x['strategy'] != y['strategy'] or x['expertise_indices'] != y['expertise_indices']
               for x, y in zip(a, b))


# --------------------------------- 4: candidate/prompt/validation consistency

def test_direction_validation_uses_shown_candidates():
    """Regression for plan 4.1: an explorer's far pick (outside core expertise)
    must pass validation; a topic NOT shown must fall back to a candidate."""
    expertise = AVAILABLE_DIRECTIONS[:3]
    far_candidates = AVAILABLE_DIRECTIONS[10:13]
    far_topic = far_candidates[0].topic
    assert far_topic not in {d.topic for d in expertise}

    agent = StubAgent('a1', expertise, 'explorer')
    pt = PaperTracker(output_dir='/tmp')

    # Case 1: model picks a shown far candidate -> accepted verbatim
    llm = FakeBatchLLM([{'topic': far_topic, 'detailed_focus': 'x', 'reason': 'y'}])
    counter = {}
    directions = create_research_directions_batch(
        agents=[agent], year=2, paper_tracker=pt, llm=llm,
        average_funding_level=100.0,
        candidate_map={'a1': far_candidates}, fallback_counter=counter)
    assert directions['a1']['direction'].topic == far_topic
    assert counter['n_validation_fallback'] == 0
    assert counter['n_parse_fallback'] == 0

    # Case 2: model picks a topic NOT shown (its own core expertise) -> fallback
    # to the first strategy-valid candidate, counted as validation fallback
    agent2 = StubAgent('a2', expertise, 'explorer')
    llm2 = FakeBatchLLM([{'topic': expertise[0].topic, 'detailed_focus': 'x', 'reason': 'y'}])
    counter2 = {}
    directions2 = create_research_directions_batch(
        agents=[agent2], year=2, paper_tracker=pt, llm=llm2,
        average_funding_level=100.0,
        candidate_map={'a2': far_candidates}, fallback_counter=counter2)
    assert directions2['a2']['direction'].topic == far_candidates[0].topic
    assert counter2['n_validation_fallback'] == 1


# --------------------------------------------- 5-6: embeddings and thresholds

@pytest.fixture(scope='module')
def embedding_tracker(tmp_path_factory):
    from utopia.metrics.embedding_tracker import EmbeddingTracker
    return EmbeddingTracker(cache_dir=str(tmp_path_factory.mktemp('embcache')))


def test_synthetic_near_mid_far_ordering(embedding_tracker):
    t = embedding_tracker
    t.near_threshold, t.far_threshold = 0.3, 0.6
    centroid = np.array([1.0, 0.0, 0.0])
    t.expertise_centroids['author'] = centroid
    cases = {
        'p_near': np.array([1.0, 0.05, 0.0]),   # ~0 distance
        'p_mid': np.array([1.0, 0.9, 0.0]),      # cos = 0.74 -> dist 0.26? adjust
        'p_far': np.array([0.0, 1.0, 0.0]),      # orthogonal -> dist 1.0
    }
    # choose mid vector with distance strictly between 0.3 and 0.6
    v = np.array([1.0, 1.2, 0.0])
    v_dist = 1 - (centroid @ v) / (np.linalg.norm(centroid) * np.linalg.norm(v))
    assert 0.3 < v_dist < 0.6
    cases['p_mid'] = v
    buckets = {}
    for pid, emb in cases.items():
        t.embeddings[pid] = emb
        t.paper_metadata[pid] = {'author_id': 'author', 'year': 1}
        score, bucket = t.compute_novelty_score(pid, 'author')
        buckets[pid] = (score, bucket)
    assert buckets['p_near'][1] == 'near'
    assert buckets['p_mid'][1] == 'mid'
    assert buckets['p_far'][1] == 'far'
    assert buckets['p_near'][0] < buckets['p_mid'][0] < buckets['p_far'][0]


def test_batched_encode_matches_single(embedding_tracker):
    t = embedding_tracker
    texts = ["graph neural networks", "reinforcement learning agents", "quantum error correction"]
    batched = t._encode_cached(texts)
    singles = np.stack([t.model.encode(x, convert_to_numpy=True) for x in texts])
    assert np.allclose(batched, singles, atol=1e-5)


def test_empirical_thresholds_from_direction_distribution(embedding_tracker):
    t = embedding_tracker
    result = t.compute_direction_thresholds(AVAILABLE_DIRECTIONS[:12])
    assert 0.0 < result['near_threshold'] < result['far_threshold'] < 1.5
    assert t.near_threshold == result['near_threshold']


# ------------------------------------------------------- 7: keyword caching

def test_keyword_cache_roundtrip(tmp_path):
    cache_dir = str(tmp_path / 'kw')
    kw_response = {'keywords': [f"kw{i}" for i in range(12)] + ['kw0']}  # 12 + dup
    llm = FakeBatchLLM([kw_response])
    ex = KeywordExtractor(cache_dir=cache_dir, model_id='fake')
    result = ex.extract_keywords_batch(llm, [('p1', 'an abstract about kw0 and kw1')], n_keywords=10)
    assert result['p1']['provenance'] == 'llm'
    assert result['p1']['keywords'] == [f"kw{i}" for i in range(10)]  # order kept, capped, deduped

    # Fresh instance, same cache dir -> served from cache without LLM
    ex2 = KeywordExtractor(cache_dir=cache_dir, model_id='fake')
    result2 = ex2.extract_keywords_batch(FakeBatchLLM([]), [('p1', 'an abstract about kw0 and kw1')])
    assert result2['p1']['provenance'] == 'cache'
    assert result2['p1']['keywords'] == result['p1']['keywords']

    # Failure path -> deterministic fallback, flagged
    class FailingLLM:
        model_name = 'fake'
        def generate_batch(self, prompts, **kw):
            return [(None, []) for _ in prompts]
    ex3 = KeywordExtractor(cache_dir=cache_dir, model_id='fake')
    result3 = ex3.extract_keywords_batch(FailingLLM(), [('p2', 'a longwinded abstract regarding transformers')])
    assert result3['p2']['provenance'] == 'fallback'
    assert len(result3['p2']['keywords']) <= 10


# ------------------------------------------------------------- 8: CD index

def test_cd_pure_disruptive():
    pt, ct = build_citation_world(
        papers=[('R', 'x', 1), ('F', 'x', 2), ('C1', 'y', 3), ('C2', 'y', 3)],
        edges=[('F', 'R', 2), ('C1', 'F', 3), ('C2', 'F', 3)])
    cd = CDIndexCalculator(ct, pt).calculate_cd_index('F')
    assert cd['cd'] == pytest.approx(1.0)
    assert cd['n_disruptive'] == 2 and cd['n_consolidating'] == 0
    assert cd['denominator'] == 2


def test_cd_pure_consolidating():
    pt, ct = build_citation_world(
        papers=[('R', 'x', 1), ('F', 'x', 2), ('C1', 'y', 3), ('C2', 'y', 3)],
        edges=[('F', 'R', 2), ('C1', 'F', 3), ('C1', 'R', 3), ('C2', 'F', 3), ('C2', 'R', 3)])
    cd = CDIndexCalculator(ct, pt).calculate_cd_index('F')
    assert cd['cd'] == pytest.approx(-1.0)
    assert cd['n_consolidating'] == 2


def test_cd_predecessor_only_enlarges_denominator():
    """The old implementation ignored predecessor-only citers entirely; the
    corrected Funk-Owen-Smith formula includes them in the denominator."""
    pt, ct = build_citation_world(
        papers=[('R', 'x', 1), ('F', 'x', 2), ('C1', 'y', 3), ('C2', 'y', 3)],
        edges=[('F', 'R', 2), ('C1', 'F', 3), ('C2', 'R', 3)])
    cd = CDIndexCalculator(ct, pt).calculate_cd_index('F')
    # C1: f=1,b=0 -> +1; C2: f=0,b=1 -> 0; n=2 -> CD = 0.5 (old formula: 1.0)
    assert cd['cd'] == pytest.approx(0.5)
    assert cd['n_predecessor_only'] == 1
    assert cd['denominator'] == 2


def test_cd_min_denominator_and_cutoff():
    pt, ct = build_citation_world(
        papers=[('R', 'x', 1), ('F', 'x', 2), ('C1', 'y', 3), ('C3', 'y', 5)],
        edges=[('F', 'R', 2), ('C1', 'F', 3), ('C3', 'F', 5)])
    calc = CDIndexCalculator(ct, pt)
    assert calc.calculate_cd_index('F', min_denominator=3) is None
    # cutoff at year 3 excludes C3
    cd3 = calc.calculate_cd_index('F', cutoff_year=3)
    assert cd3['denominator'] == 1
    cd5 = calc.calculate_cd_index('F', cutoff_year=5)
    assert cd5['denominator'] == 2
    # focal paper itself (F cites R) is never in the index set
    assert calc.calculate_cd_index('F')['n_predecessor_only'] == 0


# --------------------------------------------- 9: funding penalty reordering

def test_funding_penalty_changes_winners_before_selection():
    from types import SimpleNamespace
    apps = [{'applicant_id': f"a{i}"} for i in range(4)]
    ranked = [{'application_id': i, 'applicant_id': f"a{i}", 'rank': i + 1, 'reason': ''}
              for i in range(4)]
    metadata = [{'program_id': 'P', 'apps': apps, 'panel_index': 0}]
    programs = {'P': SimpleNamespace(funding_rate=0.5)}  # 2 winners of 4

    # Neutral: lambda=0 -> winners are LLM rank 1-2 (a0, a1)
    winners = FundingAgency.process_funding_evaluation_results(
        [(dict(ranked_applications=[dict(r) for r in ranked]), [])], metadata, programs)
    assert [w['applicant_id'] for w in winners['P']] == ['a0', 'a1']
    assert winners['P'][0]['adjusted_score'] == winners['P'][0]['normalized_raw_score']

    # Intervention: a1 highly novel -> adjusted 2/3 - 0.5 = 1/6 drops below
    # a2's 1/3, so low-novelty a2 displaces high-novelty a1 BEFORE selection
    penalties = {'a0': 1.0, 'a1': 1.0, 'a2': 0.0, 'a3': 0.0}
    winners2 = FundingAgency.process_funding_evaluation_results(
        [(dict(ranked_applications=[dict(r) for r in ranked]), [])], metadata, programs,
        novelty_penalties=penalties, lambda_funding=0.5)
    ids2 = [w['applicant_id'] for w in winners2['P']]
    assert 'a2' in ids2 and 'a1' not in ids2
    assert ids2 != ['a0', 'a1']  # winner order actually changed vs neutral
    assert all('adjusted_score' in w and 'normalized_raw_score' in w for w in winners2['P'])


# ----------------------------------- 10-11: citation counts and integrity

def test_citation_counts_equal_in_degree_and_integer():
    pt, ct = build_citation_world(
        papers=[('A', 'x', 1), ('B', 'y', 2), ('C', 'z', 2)],
        edges=[('B', 'A', 2), ('C', 'A', 2)])
    assert ct.get_citation_count('A') == 2
    assert ct.get_citation_count('A') == len(ct.get_papers_citing('A'))
    assert isinstance(ct.get_citation_count('A'), int)
    # a citation-potential multiplier existing elsewhere never alters counts
    multipliers = {'A': 1.8}
    assert ct.get_citation_count('A') == 2
    assert check_citation_integrity(pt, ct) == []


def test_future_citation_detected():
    pt, ct = build_citation_world(
        papers=[('OLD', 'x', 1), ('NEW', 'y', 4)],
        edges=[('OLD', 'NEW', 4)])  # year-1 paper cites year-4 paper
    failures = check_citation_integrity(pt, ct)
    assert any('future citation' in f for f in failures)


# ----------------------------------------------------- 12: canonical paths

def test_canonical_output_paths(tmp_path):
    out = str(tmp_path / 'outputs')
    args = parse_arguments([
        '--experiment_stage', 'smoke', '--seed', '11',
        '--experiment_name', 'exploration_vs_exploitation',
        '--population_mode', 'university_only',
        '--num_institutions', '5', '--num_years', '3',
        '--model', 'Qwen/Qwen3-32B', '--output-dir', out,
    ])
    expected_id = 'explore_smoke_qwen3_32b_neutral_i5_n25_y3_seed11'
    assert args.experiment_id == expected_id
    assert args.log_dir == os.path.join(out, 'logs', expected_id)
    assert args.checkpoint_dir == os.path.join(out, 'checkpoints', expected_id)
    assert args.visual_dir == os.path.join(out, 'visual', expected_id)
    assert args.docs_dir == os.path.join(out, 'docs', expected_id)
    assert args.output_dir == args.checkpoint_dir  # runner writes checkpoints here
    for d in (args.log_dir, args.checkpoint_dir, args.visual_dir, args.docs_dir):
        assert os.path.isdir(d)


def test_checkpoint_rng_roundtrip_preserves_next_decisions():
    import json
    import random
    import torch
    from utopia.utils.seeding import capture_rng_state, restore_rng_state
    original = capture_rng_state()
    try:
        saved = json.loads(json.dumps(original))
        expected = (random.random(), float(np.random.normal()), float(torch.rand(())))
        restore_rng_state(saved)
        actual = (random.random(), float(np.random.normal()), float(torch.rand(())))
        assert actual == expected
        restore_rng_state({'python': saved['python']})
        assert random.random() == expected[0]
    finally:
        restore_rng_state(original)


def test_derive_seed_stable_and_distinct():
    assert derive_seed(11, 'phase1_directions', 3, 0, 0) == derive_seed(11, 'phase1_directions', 3, 0, 0)
    assert derive_seed(11, 'phase1_directions', 3, 0, 0) != derive_seed(11, 'phase1_directions', 3, 0, 1)
    assert derive_seed(11, 'a') != derive_seed(12, 'a')


def test_fulltext_review_uses_complete_indexed_body_without_changing_paper(tmp_path):
    import json
    from utopia.data.rag import FullTextCorpus
    from utopia.simulation import Simulation
    from utopia.agents.base_agent import SimulationAgent
    body = 'Introduction 正文\n' + 'research evidence ' * 6000 + '\nAppendix END'
    prefix = b'other paper\n'
    (tmp_path/'papers.txt').write_bytes(prefix + body.encode())
    (tmp_path/'index.json').write_text(json.dumps({'text_file':'papers.txt','papers':{
        '1606.04422': {'offset':len(prefix),'nbytes':len(body.encode())}}}))
    simulation = Simulation.__new__(Simulation)
    simulation.fulltext_corpus = FullTextCorpus(tmp_path/'index.json')
    paper = dict(id='https://arxiv.org/abs/1606.04422v2', title='Title',
                 abstract='Short abstract',topics=['ML'])
    reviewer = SimulationAgent.__new__(SimulationAgent)
    reviewer.expertise = ['ML']
    reviewer.get_memory_context = lambda: 'Original reviewer memory'
    prompt, _ = reviewer.get_review_prompt(simulation.review_input(paper))
    assert 'Full paper text:\n'+body in prompt
    assert 'Original reviewer memory' in prompt
    assert 'full_text' not in paper
    assert 'Abstract:\nShort abstract' in reviewer.get_review_prompt(paper)[0]
    with pytest.raises(KeyError):
        simulation.review_input(dict(paper,id='missing'))
    (tmp_path/'papers.txt').write_bytes(prefix)
    with pytest.raises(ValueError,match='incomplete full text'):
        simulation.review_input(paper)


@pytest.mark.parametrize('source_case', ['pdf', 'withdrawn_pdf', 'html'])
def test_missing_scievo_body_fetches_all_pdf_pages_and_resumes_offline(tmp_path, monkeypatch, source_case):
    import json
    import pymupdf
    import requests
    from types import SimpleNamespace
    from utopia.data.rag import FullTextCorpus
    with pymupdf.open() as pdf:
        for text in ['Introduction full text', 'Appendix final-page evidence']:
            page = pdf.new_page()
            page.insert_text((72, 72), text)
        source = pdf.tobytes()
    (tmp_path/'index.json').write_text(json.dumps({'text_file': 'base.txt', 'papers': {}}))
    calls = []
    def download(url, timeout):
        calls.append(url)
        if source_case != 'pdf' and not url.endswith('v1'):
            return SimpleNamespace(status_code=404)
        if source_case == 'html':
            if '/pdf/' in url:
                return SimpleNamespace(status_code=404)
            html = '<html><head><style>HIDDEN CSS</style></head><body>Introduction full text<p>Appendix final-page evidence</p><script>HIDDEN JS</script></body></html>'
            return SimpleNamespace(status_code=200, content=html.encode(), text=html, raise_for_status=lambda: None)
        return SimpleNamespace(status_code=200, content=source, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, 'get', download)
    corpus = FullTextCorpus(tmp_path/'index.json', cache_dir=tmp_path/'supplement')
    text = corpus.text('https://arxiv.org/abs/2501.00001v2')
    assert 'Introduction full text' in text
    assert 'Appendix final-page evidence' in text
    if source_case == 'html':
        assert (tmp_path/'supplement/2501.00001.html').is_file()
        assert 'HIDDEN' not in text
    else:
        assert (tmp_path/'supplement/2501.00001.pdf').read_bytes() == source
    resumed = FullTextCorpus(tmp_path/'index.json', cache_dir=tmp_path/'supplement')
    assert resumed.text('2501.00001') == text
    expected = ['https://arxiv.org/pdf/2501.00001']
    if source_case != 'pdf':
        expected.append(expected[0] + 'v1')
    if source_case == 'html':
        expected.append('https://arxiv.org/html/2501.00001v1')
    assert calls == expected
    index = json.loads((tmp_path/'supplement/index.json').read_text())
    assert index['papers']['2501.00001']['source_url'] == expected[-1]
    assert index['papers']['2501.00001']['pages'] == (None if source_case == 'html' else 2)
    assert index['papers']['2501.00001']['nbytes'] == len(text.encode())


@pytest.mark.parametrize('backend', ['bedrock','auto'])
def test_backend_arguments_preserve_explicit_batch_and_output_budget(backend):
    args = parse_arguments(['--backend',backend,'--model','openai.gpt-5.6-luna',
                            '--batch_size','256','--completion_token_floor','32768',
                            '--input_tokens_per_minute','600000',
                            '--output_tokens_per_minute','45000',
                            '--fulltext_index','corpus/index.json'])
    assert args.backend == backend
    assert args.batch_size == 256
    assert args.completion_token_floor == 32768
    assert args.fulltext_index == 'corpus/index.json'
    assert args.input_tokens_per_minute == 600000
    assert args.output_tokens_per_minute == 45000


@pytest.mark.parametrize("query_dtype", ["float16", "float32"])
def test_submission_retrieval_matches_cached_embedding_dtype(query_dtype):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import pandas as pd
    import torch
    from utopia.data.rag import RAG

    rag = RAG.__new__(RAG)
    rag.device = "cpu"
    rag.use_langchain = False
    rag.start_year = 2016
    rag.current_sim_year = 1
    rag.config = SimpleNamespace(normalize_embeddings=True, batch_size=2, num_retrieved_docs=1)
    rag.document_embeddings = torch.tensor([[-1., 0.], [1., 0.], [0., -1.], [0., 1.], [1., 1.]])
    rag.document_status = pd.DataFrame({
        "status": ["accept", "unsubmitted", "reject", "unsubmitted", "unsubmitted"],
        "publication_year": [2016, 2016, 2016, 2016, 2017],
    })
    rag.sentence_transformer_model = Mock()
    rag.sentence_transformer_model.encode.return_value = torch.tensor(
        [[1., 0.], [0., 1.]], dtype=getattr(torch, query_dtype))

    result = rag.batch_retrieve(["first direction", "second direction"])

    assert result["topk_indices"].tolist() == [[1], [3]]
    assert result["similarity_scores"].dtype == torch.float32
    torch.testing.assert_close(result["similarity_scores"], torch.eye(2))


def test_paper_checkpoint_preserves_author_order_and_shared_objects(tmp_path):
    records = [dict(id=pid, title=pid, abstract="full abstract", author_id=authors,
                    author_type="university", year=year, conference="venue", topics=[],
                    tags=[], status="reject", review_history=[])
               for pid, authors, year in [("older", ["a", "b"], 1), ("newer", ["a"], 2)]]
    legacy = {"output_dir": str(tmp_path), "papers": records}
    tracker = PaperTracker.from_dict(legacy)
    tracker.papers_by_author["a"].reverse()
    tracker.papers_by_author = dict(reversed(list(tracker.papers_by_author.items())))
    restored = PaperTracker.from_dict(tracker.to_dict())
    assert list(restored.papers_by_author) == ["b", "a"]
    assert [p.id for p in restored.get_papers_by_author("a")] == ["newer", "older"]
    assert restored.get_papers_by_author("a")[1] is restored.get_papers_by_author("b")[0]
    assert [p.id for p in PaperTracker.from_dict(legacy).get_papers_by_author("a")] == ["older", "newer"]
    assert restored.to_dict()["papers"] == tracker.to_dict()["papers"]


@pytest.mark.parametrize("order", [{"a": ["missing"]}, {"a": ["only", "only"]}, {}])
def test_paper_checkpoint_rejects_lost_or_duplicate_ids(tmp_path, order):
    record = dict(id="only", title="paper", abstract="text", author_id="a",
                  author_type="university", year=1, conference="venue", topics=[],
                  tags=[], status="reject", review_history=[])
    with pytest.raises(ValueError):
        PaperTracker.from_dict({"output_dir": str(tmp_path), "papers": [record],
                                "papers_by_author": order})
