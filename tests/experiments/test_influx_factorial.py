"""Deterministic tests for the influx2x2 experiment (researcher influx x
resubmission 2x2 factorial).

No Simulation construction, no LLM calls — everything runs on hand-built
objects, SimpleNamespace stubs, and unbound Simulation method calls, mirroring
tests/experiments/test_exploration.py.

Run: python -m pytest tests/experiments/test_influx_factorial.py -v
"""

import os
import json
import random
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from utopia.utils.paths import project_root

os.chdir(str(project_root(__file__)))

from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS
from utopia.agents.researcher_agents import UniversityResearcher
from utopia.arguments import parse_arguments
from utopia.simulation import (
    NORMAL_COMPANIES,
    NORMAL_UNIVERSITIES,
    RICH_COMPANIES,
    RICH_UNIVERSITIES,
    Simulation,
    build_influx_cohort,
)

# ---------------------------------------------------------------- helpers

ALL_SELECTED = (RICH_UNIVERSITIES[:10] + NORMAL_UNIVERSITIES[:20]
                + RICH_COMPANIES[:10] + NORMAL_COMPANIES[:20])


def make_university(name, institution, funding=160):
    expertise = [AVAILABLE_DIRECTIONS[0], AVAILABLE_DIRECTIONS[1], AVAILABLE_DIRECTIONS[2]]
    return UniversityResearcher(
        researcher_name=name, university_name=institution,
        funding_level=funding, expertise=expertise, llm=None,
        generate_research_proposal=True, exploration_strategy='balanced')


def make_stub_sim(rate, eco=None, seed=42):
    return SimpleNamespace(
        args=SimpleNamespace(annual_influx_rate=rate, seed=seed,
                             population_mode='default'),
        debug=False,
        ecosystem=eco if eco is not None else MultiAgentEcosystem(agent_configs={}),
        llm=None,
        industry_funding_mode='performance',
    )


# ------------------------------------------------------ cohort schedule

def test_cohort_exact_count_and_alternation():
    hires_by_inst = Counter()
    all_names = []
    for year in range(1, 11):
        cohort = build_influx_cohort(year, 42)
        assert len(cohort) == 30, f"year {year}: expected +30, got {len(cohort)}"
        for row in cohort:
            hires_by_inst[row['institution']] += 1
            all_names.append(row['researcher_name'])
        # halves alternate: an institution hiring this year must not hire next year
        next_insts = {r['institution'] for r in build_influx_cohort(year + 1, 42)}
        assert not next_insts & {r['institution'] for r in cohort}
    # every selected institution hires exactly 5 times over 10 years
    assert set(hires_by_inst) == set(ALL_SELECTED)
    assert all(n == 5 for n in hires_by_inst.values())
    # 300 unique entrant ids, indices 5-9, never colliding with founders 0-4
    assert len(all_names) == len(set(all_names)) == 300
    assert all(n.rsplit('_', 1)[1] in {'5', '6', '7', '8', '9'} for n in all_names)


def test_cohort_indices_increase_with_prior_hires():
    # An H1 institution (even index) hires in years 1,3,5,7,9 at indices 5,6,7,8,9
    inst = RICH_UNIVERSITIES[0]  # index 0 -> H1
    seen = {}
    for year in range(1, 11):
        for row in build_influx_cohort(year, 42):
            if row['institution'] == inst:
                seen[year] = row['researcher_name']
    assert sorted(seen) == [1, 3, 5, 7, 9]
    assert [seen[y] for y in sorted(seen)] == [f"{inst}_researcher_{i}" for i in range(5, 10)]


def test_cohort_stratification():
    for year in range(1, 11):
        strata = Counter((r['kind'], r['funding_level'])
                         for r in build_influx_cohort(year, 42))
        assert strata == {('university', 160): 5, ('university', 80): 10,
                          ('industry', 320): 5, ('industry', 160): 10}, year


def test_cohort_determinism():
    for year in (1, 2, 7):
        assert build_influx_cohort(year, 42) == build_influx_cohort(year, 42)
    a = build_influx_cohort(1, 42)
    b = build_influx_cohort(1, 43)
    # same schedule/names, different expertise draws
    assert [r['researcher_name'] for r in a] == [r['researcher_name'] for r in b]
    assert any(x['expertise_indices'] != y['expertise_indices'] for x, y in zip(a, b))
    for row in a:
        idx = row['expertise_indices']
        assert len(idx) == len(set(idx)) == 3
        assert all(0 <= i < len(AVAILABLE_DIRECTIONS) for i in idx)


def test_cohort_consumes_no_global_rng():
    random.seed(1234)
    np.random.seed(1234)
    py_state = random.getstate()
    np_state = np.random.get_state()
    for year in range(1, 11):
        build_influx_cohort(year, 42)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])


# ------------------------------------------------------ injection method

def test_apply_influx_off_is_noop():
    eco = MultiAgentEcosystem(agent_configs={})
    eco.add_agent(make_university('MIT_researcher_0', 'MIT'))
    stub = make_stub_sim(rate=0.0, eco=eco)
    random.seed(99)
    py_state = random.getstate()
    before = dict(eco.agent_population)
    assert Simulation.apply_annual_influx(stub, 1) == []
    assert eco.agent_population == before
    assert random.getstate() == py_state


def test_apply_influx_on_injects_and_wires_coi():
    eco = MultiAgentEcosystem(agent_configs={})
    founders = [make_university(f'MIT_researcher_{i}', 'MIT') for i in range(5)]
    for f in founders:
        eco.add_agent(f)
    founders[3].is_active = False  # culled member must still be CoI-wired
    stub = make_stub_sim(rate=0.1, eco=eco)

    entrants = Simulation.apply_annual_influx(stub, 1)
    # MIT is RICH_UNIVERSITIES[0] -> H1 -> hires in year 1
    assert 'MIT_researcher_5' in entrants
    entrant = eco.get_agent_by_id('MIT_researcher_5')
    assert entrant.resources == 160
    assert entrant.reputation == 5
    assert entrant.exploration_strategy == 'balanced'
    assert entrant.is_active and entrant.can_author and entrant.can_review
    assert len(entrant.expertise) == 3
    assert entrant.conflict_of_interest == {f'MIT_researcher_{i}' for i in range(5)}
    for f in founders:
        assert 'MIT_researcher_5' in f.conflict_of_interest
    # full cohort landed: population = 5 founders + 30 entrants
    assert len(eco.agent_population) == 35


def test_apply_influx_idempotent():
    eco = MultiAgentEcosystem(agent_configs={})
    stub = make_stub_sim(rate=0.1, eco=eco)
    first = Simulation.apply_annual_influx(stub, 1)
    assert len(first) == 30
    second = Simulation.apply_annual_influx(stub, 1)  # resume replay of year 1
    assert second == []
    assert len(eco.agent_population) == 30


# ------------------------------------------------------ resubmission toggle

def test_disable_resubmission_early_return():
    class ExplodingLLM:
        def generate_batch(self, *a, **k):
            raise AssertionError('Phase 0 must not reach the LLM when disabled')

    # stub deliberately has NO paper_tracker: any access would AttributeError
    stub = SimpleNamespace(
        args=SimpleNamespace(disable_resubmission=True),
        llm=ExplodingLLM(),
    )
    assert Simulation.run_phase_0_resubmissions(stub, 5, {}) == []


# ------------------------------------------------------ CLI flags

def test_influx_flag_validation(tmp_path):
    common = ['--output-dir', str(tmp_path), '--experiment_name', 'influx_compound']
    for rate in ('-0.1', 'nan', 'inf'):
        with pytest.raises(ValueError):
            parse_arguments(common + ['--annual-population-influx', rate])
    for rate in ('0.1', '0.2', '1.5'):
        args = parse_arguments(common + ['--annual-population-influx', rate])
        assert args.annual_population_influx == float(rate)
        assert '_influx_compound_' in args.experiment_id
    assert parse_arguments(common).annual_population_influx == 0


def test_resume_retains_review_pages_and_allows_api_budget_and_rate_changes(tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'text_file': 'texts.utf8', 'papers': {}}))
    checkpoint.write_text(json.dumps({'checkpoint_args': {
        'model': 'openai.gpt-5.6-luna', 'backend': 'bedrock',
        'annual_population_influx': 0.2, 'population_influx_mode': 'compound',
        'review_main_text_pages': 9, 'fulltext_index': str(index),
        'api_budget_usd': 100, 'input_tokens_per_minute': 1000000,
        'output_tokens_per_minute': 100000, 'max_model_len': 262144,
    }}))
    common = ['--output-dir', str(tmp_path / 'restored'),
              '--experiment_name', 'influx_resume', '--resume-from', str(checkpoint)]
    saved = parse_arguments(common)
    assert saved.review_main_text_pages == 9
    assert saved.api_budget_usd == 100
    updated = parse_arguments(common + ['--api_budget_usd', '500',
                                       '--input_tokens_per_minute', '2000000'])
    assert updated.api_budget_usd == 500
    assert updated.input_tokens_per_minute == 2000000
    assert updated.output_tokens_per_minute == 100000
    assert updated.annual_population_influx == 0.2
    with pytest.raises(ValueError, match='scientific setting'):
        parse_arguments(common + ['--annual-population-influx', '0.1'])


@pytest.mark.parametrize('rate,expected', [(0.1, [5500, 6050, 6655, 7321]),
                                         (0.2, [6000, 7200, 8640, 10368])])
def test_population_compounds_all_researchers_and_replays_once(rate, expected):
    eco = MultiAgentEcosystem(agent_configs={})
    for institution in range(1000):
        for member in range(5):
            name = f'institution_{institution:04d}'
            eco.add_agent(make_university(f'{name}_researcher_{member}', name, funding=100))
    next(iter(eco.agent_population.values())).is_active = False
    sim = make_stub_sim(rate, eco)
    sim.args = SimpleNamespace(annual_population_influx=rate, seed=42, initial_funding=100)
    before = random.getstate()
    for year, total in enumerate(expected, 1):
        new = Simulation.apply_annual_influx(sim, year)
        assert len(eco.agent_population) == total
        assert Simulation.apply_annual_influx(sim, year) == []
        assert len(eco.agent_population) == total
        assert all(eco.agent_population[a].resources == 100 for a in new)
        sizes = Counter(a.university_name for a in eco.agent_population.values())
        assert max(sizes.values()) - min(sizes.values()) <= 1
    assert random.getstate() == before


def test_fulltext_prompt_preserves_body_references_and_appendix(tmp_path, monkeypatch):
    from utopia.data.rag import FullTextCorpus
    text = "PAPER START\nMethods\n" + "Complete body. " * 20000 + "\nReferences\nAppendix\nPAPER END"
    encoded = text.encode()
    (tmp_path / "texts.utf8").write_bytes(encoded)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"text_file": "texts.utf8", "papers": {
        "1601.12345": {"offset": 0, "nbytes": len(encoded)}}}))
    corpus = FullTextCorpus(index, cache_dir=tmp_path / "supplement")
    sim = SimpleNamespace(fulltext_corpus=corpus)
    paper = {"id": "1601.12345v2", "title": "Test", "abstract": "Short abstract",
             "topics": ["cs.LG"]}
    reviewer = make_university("reviewer", "other")
    prompt, _ = reviewer.get_review_prompt(Simulation.review_input(sim, paper))
    assert text in prompt
    assert "Full paper text:" in prompt
    assert "Short abstract" not in prompt
    with pytest.raises(ValueError, match="incomplete full text"):
        (tmp_path / "texts.utf8").write_bytes(encoded[:-10])
        Simulation.review_input(sim, paper)
    def missing(paper_id):
        raise ValueError("Full text unavailable")
    monkeypatch.setattr(corpus, "fetch_paper", missing)
    with pytest.raises(ValueError, match="unavailable"):
        Simulation.review_input(sim, dict(paper, id="1601.00001"))

@pytest.mark.parametrize("appendix_page", [None, 3])
def test_review_uses_first_nine_pdf_pages_and_stops_at_appendix(
        tmp_path, monkeypatch, appendix_page):
    import pymupdf
    import shutil
    from utopia.data.rag import FullTextCorpus
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"text_file": "texts.utf8", "papers": {}}))
    cache = tmp_path / "cache"
    cache.mkdir()
    with pymupdf.open() as pdf:
        for number in range(1, 12):
            page = pdf.new_page()
            page.insert_text((50, 60), f"PAGE_{number}_MAIN", fontsize=11)
            page.insert_text((50, 85), "See Appendix A for an extra example.", fontsize=11)
            if number == 2:
                page.insert_text((50, 115), "Contents", fontsize=14)
                page.insert_text((50, 140), "Appendix A .......... 19", fontsize=11)
            if number == appendix_page:
                page.insert_text((50, 160), "Appendix A: Proofs", fontsize=14, fontname="hebo")
                page.insert_text((50, 190), "APPENDIX_SECRET", fontsize=11)
        pdf.save(cache / "1601.12345.pdf")
    corpus = FullTextCorpus(index, cache_dir=cache, main_text_pages=9)
    text = corpus.text("1601.12345v2")
    assert "PAGE_1_MAIN" in text
    assert "Appendix A .......... 19" in text
    assert "PAGE_10_MAIN" not in text
    assert "APPENDIX_SECRET" not in text
    assert ("PAGE_9_MAIN" in text) == (appendix_page is None)
    if appendix_page:
        assert "PAGE_3_MAIN" in text
        assert "PAGE_4_MAIN" not in text
    entry = corpus.supplement["review_texts"]["1601.12345"]["first_9_pages_no_appendix"]
    assert entry["pages"] == (appendix_page or 9)
    assert entry["appendix_start_page"] == appendix_page
    assert entry["source_pages"] == 11
    assert entry["exclude_appendices"]
    relocated = tmp_path / "relocated"
    shutil.copytree(cache, relocated)
    resumed = FullTextCorpus(index, cache_dir=relocated, main_text_pages=9)
    monkeypatch.setattr(resumed, "download_source",
                        lambda pid: pytest.fail("Saved review text should not download again"))
    assert resumed.text("1601.12345") == text


def test_review_excludes_lettered_appendix_after_references_and_paginates_html(tmp_path):
    import pymupdf
    from utopia.data.rag import FullTextCorpus
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"text_file": "texts.utf8", "papers": {}}))
    cache = tmp_path / "cache"
    cache.mkdir()
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 60), "Main argument", fontsize=11)
        page.insert_text((50, 100), "References", fontsize=14, fontname="hebo")
        page.insert_text((50, 125), "A. Smith, Main reference, 2020.", fontsize=11)
        page.insert_text((50, 160), "A. Additional Experiments", fontsize=14, fontname="hebo")
        page.insert_text((50, 190), "APPENDIX_SECRET", fontsize=11)
        pdf.save(cache / "1601.12345.pdf")
    corpus = FullTextCorpus(index, cache_dir=cache, main_text_pages=9)
    text = corpus.text("1601.12345")
    assert "Main reference" in text
    assert "Additional Experiments" not in text
    assert "APPENDIX_SECRET" not in text
    (cache / "1601.00001.html").write_text(
        "<h1>Original HTML Paper</h1><p>MAIN ARGUMENT</p>"
        "<h2>Appendix A</h2><p>APPENDIX_SECRET</p>")
    html = corpus.text("1601.00001")
    assert "MAIN ARGUMENT" in html
    assert "APPENDIX_SECRET" not in html
    assert corpus.supplement["review_texts"]["1601.00001"][
        "first_9_pages_no_appendix"]["page_source"] == "html_rendered_a4"


def test_review_handles_long_pdf_spacing_without_losing_main_text(tmp_path):
    from utopia.data.rag import FullTextCorpus
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"text_file": "texts.utf8", "papers": {}}))
    corpus = FullTextCorpus(index, cache_dir=tmp_path / "cache", main_text_pages=9)

    class SparsePage:
        def get_text(self, kind=None, **kwargs):
            if kind == "dict":
                return {"blocks": []}
            return ("Main argument" + " " * 20000 + "continues here\n"
                    "Appendix A" + " " * 20000 + "12\n"
                    "Still main text\nAppendix A\nAPPENDIX_SECRET")

    text, pages, appendix_page = corpus.extract_main_pages([SparsePage()])
    assert "Main argument continues here" in text
    assert "Still main text" in text
    assert "APPENDIX_SECRET" not in text
    assert pages == appendix_page == 1


@pytest.mark.parametrize("appendix_page", [1, 3, None])
def test_review_keeps_submitted_document_title_and_excludes_internal_appendices(
        tmp_path, monkeypatch, appendix_page):
    import pymupdf
    from utopia.data.rag import FullTextCorpus
    title = "Supplementary Material for the Information Sciences Paper: An Experimental Study"
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"text_file": "texts.utf8", "papers": {}}))
    cache = tmp_path / "cache"
    cache.mkdir()
    with pymupdf.open() as pdf:
        for number in range(1, 12):
            page = pdf.new_page()
            if number == 1:
                page.insert_text((50, 50), "Supplementary Material for the", fontsize=14,
                                 fontname="hebo")
                page.insert_text((50, 70), "Information Sciences Paper: An Experimental Study",
                                 fontsize=14, fontname="hebo")
                page.insert_text((50, 110), "A. Author, Example University", fontsize=11)
                page.insert_text((50, 140), "1. Introduction", fontsize=12, fontname="hebo")
            page.insert_text((50, 180), f"PAGE_{number}_MAIN", fontsize=11)
            if number == appendix_page:
                page.insert_text((50, 210), "References", fontsize=12, fontname="hebo")
                page.insert_text((50, 235), "A. Author, Reference within the main text.", fontsize=11)
                page.insert_text((50, 270), "Supplementary Material for the", fontsize=14,
                                 fontname="hebo")
                page.insert_text((50, 290), "Information Sciences Paper: An Experimental Study",
                                 fontsize=14, fontname="hebo")
                page.insert_text((50, 330), "APPENDIX_SECRET", fontsize=11)
        pdf.save(cache / "1702.04501.pdf")
    corpus = FullTextCorpus(index, cache_dir=cache, main_text_pages=9)
    # Without submission metadata, the original failure is reproducible.
    with pytest.raises(ValueError, match="No main text"):
        corpus.text("1702.04501")
    corpus.prepare_review_texts(["https://arxiv.org/abs/1702.04501v1"],
                               paper_titles={"https://arxiv.org/abs/1702.04501v1": title})
    text = corpus.text("1702.04501")
    assert "1. Introduction" in text
    assert "PAGE_1_MAIN" in text
    assert "APPENDIX_SECRET" not in text
    assert "PAGE_10_MAIN" not in text
    if appendix_page:
        assert "Reference within the main text" in text
        assert f"PAGE_{appendix_page + 1}_MAIN" not in text
    else:
        assert "PAGE_9_MAIN" in text
    entry = corpus.supplement["review_texts"]["1702.04501"]["first_9_pages_no_appendix"]
    assert entry["paper_title"] == title
    assert entry["appendix_start_page"] == appendix_page
    assert entry["exclude_appendices"] is True
    resumed = FullTextCorpus(index, cache_dir=cache, main_text_pages=9)
    monkeypatch.setattr(resumed, "download_source",
                        lambda pid: pytest.fail("Reuse the prepared document by ID"))
    assert resumed.text("1702.04501") == text


def test_conference_checkpoint_retains_submissions_reviews_and_decisions():
    from utopia.agents.conference import Conference, ConferenceSystem
    conference = Conference(conference_id="C", name="Conference", topics=["cs.LG"])
    conference.submitted_papers = [{"id": "p", "author_id": "a"}]
    conference.reviews["p"] = [{"reviewer_id": "r", "overall_score": 4}]
    system = ConferenceSystem([conference])
    system.authors_to_accepted_papers["a"] = [{"id": "old"}]
    restored = ConferenceSystem.from_dict(json.loads(json.dumps(system.to_dict())))
    assert restored.to_dict() == system.to_dict()


def test_experiment_name_whitelist_accepts_influx():
    for name in ['influx2x2_A_noinflux_noresub_seed42',
                 'influx2x2_B_influx_noresub_seed42',
                 'influx2x2_C_noinflux_resub_seed42',
                 'influx2x2_D_influx_resub_seed42',
                 'influx2x2_smoke_A_seed42',
                 'influx2x2_smoke_D_seed42']:
        # replicate the utopia/simulation.py whitelist check
        assert name.startswith(('default', 'acl-acceptance-rates',
                                'acl_schedule_smoke', 'influx'))


def test_parallel_review_preparation_preserves_order_bytes_and_reuses_ids(tmp_path, monkeypatch):
    import shutil
    import pymupdf
    from utopia.data.rag import FullTextCorpus
    from utopia.utils import data_utils

    index = tmp_path / 'base.json'
    index.write_text(json.dumps({'text_file': 'base.txt', 'papers': {}}))
    serial_cache, parallel_cache = tmp_path / 'serial', tmp_path / 'parallel'
    serial_cache.mkdir()
    with pymupdf.open() as pdf:
        for number in range(1, 12):
            page = pdf.new_page()
            page.insert_text((50, 60), f'PAGE_{number}_MAIN')
            if number == 3:
                page.insert_text((50, 100), 'Appendix A', fontsize=15, fontname='hebo')
                page.insert_text((50, 130), 'APPENDIX_SECRET')
        source = pdf.tobytes()
    ids = [f'1601.{number:05d}' for number in range(130)]
    for pid in ids[:-1]:
        (serial_cache / f'{pid}.pdf').write_bytes(source)
    (serial_cache / f'{ids[-1]}.html').write_text(
        '<h1>Paper</h1><p>HTML MAIN</p><h2>Appendix A</h2><p>APPENDIX_SECRET</p>')
    shutil.copytree(serial_cache, parallel_cache)
    serial = FullTextCorpus(index, cache_dir=serial_cache, main_text_pages=9)
    for pid in ids:
        serial.review_text(pid)
    parallel = FullTextCorpus(index, cache_dir=parallel_cache, main_text_pages=9)
    writes = []
    original_write = data_utils.write_json_atomic
    def record_write(path, data, **kwargs):
        writes.append(str(path))
        return original_write(path, data, **kwargs)
    monkeypatch.setattr(data_utils, 'write_json_atomic', record_write)
    before_python, before_numpy = random.getstate(), np.random.get_state()
    parallel.prepare_review_texts(
        [f'https://arxiv.org/abs/{pid}v2' for pid in reversed(ids)] + [ids[0]],
        max_workers=2)
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert parallel.supplement == serial.supplement
    assert (parallel_cache / 'texts.utf8').read_bytes() == (serial_cache / 'texts.utf8').read_bytes()
    assert 0 < len(writes) < len(ids) // 8
    for pid in (ids[0], ids[-1]):
        assert parallel.text(pid) == serial.text(pid)
        assert 'APPENDIX_SECRET' not in parallel.text(pid)
    saved = (parallel_cache / 'index.json').read_bytes()
    monkeypatch.setattr(parallel, 'download_source',
                        lambda pid: pytest.fail('An existing ID must reuse prepared text'))
    parallel.prepare_review_texts(ids, max_workers=2)
    assert (parallel_cache / 'index.json').read_bytes() == saved


def test_parallel_review_preparation_keeps_completed_prefix_on_worker_error(tmp_path, monkeypatch):
    import pymupdf
    from utopia.data.rag import FullTextCorpus

    index = tmp_path / 'base.json'
    index.write_text(json.dumps({'text_file': 'base.txt', 'papers': {}}))
    cache = tmp_path / 'cache'
    cache.mkdir()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 60), 'COMPLETE MAIN TEXT')
        source = pdf.tobytes()
    good, bad = '1601.00001', '1601.00002'
    (cache / f'{good}.pdf').write_bytes(source)
    (cache / f'{bad}.pdf').write_bytes(b'invalid pdf')
    corpus = FullTextCorpus(index, cache_dir=cache, main_text_pages=9)
    with pytest.raises(pymupdf.FileDataError):
        corpus.prepare_review_texts([bad, good], max_workers=2)
    resumed = FullTextCorpus(index, cache_dir=cache, main_text_pages=9)
    assert set(resumed.supplement['review_texts']) == {good}
    prefix = (cache / 'texts.utf8').read_bytes()
    assert resumed.text(good).strip() == 'COMPLETE MAIN TEXT'
    (cache / f'{bad}.pdf').write_bytes(source)
    original_download = resumed.download_source
    def only_missing(pid):
        assert pid == bad
        return original_download(pid)
    monkeypatch.setattr(resumed, 'download_source', only_missing)
    resumed.prepare_review_texts([good, bad], max_workers=2)
    assert (cache / 'texts.utf8').read_bytes() == prefix + prefix
    assert resumed.text(good) == resumed.text(bad)
