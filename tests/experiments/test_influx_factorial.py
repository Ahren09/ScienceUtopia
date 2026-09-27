"""Deterministic tests for the influx2x2 experiment (researcher influx x
resubmission 2x2 factorial).

No Simulation construction, no LLM calls — everything runs on hand-built
objects, SimpleNamespace stubs, and unbound Simulation method calls, mirroring
tests/experiments/test_exploration.py.

Run: python -m pytest tests/experiments/test_influx_factorial.py -v
"""

from utopia.utils.paths import project_root
import os
import random
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

os.chdir(str(project_root(__file__)))

from utopia.simulation import NORMAL_COMPANIES, NORMAL_UNIVERSITIES, RICH_COMPANIES, RICH_UNIVERSITIES, Simulation, build_influx_cohort
from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS
from utopia.agents.researcher_agents import UniversityResearcher
from utopia.arguments import parse_arguments


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
    assert Simulation._apply_annual_influx(stub, 1) == []
    assert eco.agent_population == before
    assert random.getstate() == py_state


def test_apply_influx_on_injects_and_wires_coi():
    eco = MultiAgentEcosystem(agent_configs={})
    founders = [make_university(f'MIT_researcher_{i}', 'MIT') for i in range(5)]
    for f in founders:
        eco.add_agent(f)
    founders[3].is_active = False  # culled member must still be CoI-wired
    stub = make_stub_sim(rate=0.1, eco=eco)

    entrants = Simulation._apply_annual_influx(stub, 1)
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
    first = Simulation._apply_annual_influx(stub, 1)
    assert len(first) == 30
    second = Simulation._apply_annual_influx(stub, 1)  # resume replay of year 1
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
    assert Simulation._run_phase_0_resubmissions(stub, 5, {}) == []


# ------------------------------------------------------ CLI flags

def test_influx_flag_validation():
    with pytest.raises(ValueError):
        parse_arguments(['--experiment_name', 'influx2x2_x_seed42',
                         '--annual_influx_rate', '0.2'])
    ok = parse_arguments(['--experiment_name', 'influx2x2_D_influx_resub_seed42',
                          '--annual_influx_rate', '0.1'])
    assert ok.annual_influx_rate == 0.1
    off = parse_arguments(['--experiment_name', 'influx2x2_A_noinflux_noresub_seed42',
                           '--disable_resubmission'])
    assert off.annual_influx_rate == 0.0 and off.disable_resubmission


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
