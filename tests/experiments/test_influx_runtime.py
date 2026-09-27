"""CPU-only launcher tests: python -B tests/experiments/test_influx_runtime.py.

Launcher tests use only the standard library. With NumPy/Pandas installed, an
additional test runs the unmodified original funding phase on 100 applicants
and multiple panels. Neither suite starts vLLM, queries a GPU or runs an
experiment. All fixture writes are temporary.

Required before production, in the isolated complete EC2 runtime:
  CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=42 WANDB_MODE=disabled \
    UTOPIA_ACTUAL_INFLUX_TESTS=1 python -B tests/experiments/test_influx_runtime.py TestActualInfluxYears -v
This opt-in suite uses the actual simulator/RAG/logger with scripted LLM
transport, all 1202 researcher founders, and complete years 1 and 2 in all
four cells. A small explicit subset of founders remains active to bound CPU
work; every scheduled entrant is active on arrival. These diagnostic fixtures
are not experiment outcomes and cannot satisfy production completion gates.
"""

from utopia.runtime.historical import INFLUX_POPULATION_SEED_NAMESPACE
import contextlib
import ast
import importlib.util
import io
import json
import os
import random
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.dont_write_bytecode = True
import utopia.experiments.influx_factorial as runner
import tests.support.simulation as native_fixture


def population_module():
    """Execute the actual pure blueprint and injection methods, with no LLM imports."""
    class Researcher:
        def __init__(self, researcher_name, **kwargs):
            self.id = researcher_name
            self.__dict__.update(kwargs)
            self.conflict_of_interest = set()
            self.can_author = True
            self.is_active = True

        def add_conflict_of_interest(self, values):
            self.conflict_of_interest.update([values] if isinstance(values, str) else values)

    class UniversityResearcher(Researcher):
        pass

    class IndustryResearcher(Researcher):
        pass

    class FundingAgency:
        def __init__(self, agency_name, **kwargs):
            self.id = agency_name
            self.can_author = False

    ns = {
        'List': list, 'Dict': dict, 'random': random,
        'SIMULATION_CONFIG': {'exploration_experiment': {'institution_strategy_mix': ['balanced'] * 5}},
        'AVAILABLE_DIRECTIONS': [types.SimpleNamespace(topic=f'topic{i}') for i in range(12)],
        'UniversityResearcher': UniversityResearcher, 'IndustryResearcher': IndustryResearcher,
        'FundingAgency': FundingAgency, 'logger': types.SimpleNamespace(info=lambda *args: None),
        'RICH_UNIVERSITY_FUNDING': 160, 'NORMAL_UNIVERSITY_FUNDING': 80,
        'RICH_COMPANY_FUNDING': 320, 'RICH_COMPANIES': ['Google'],
    }
    tree = ast.parse((runner.REPO_ROOT / 'utopia/simulation.py').read_text())
    methods = {'_record_initial_university_count', '_add_funding_agencies', '_apply_annual_influx'}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == 'build_population_blueprint']
    nodes += [node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'Simulation'
              for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    util = ast.parse((runner.REPO_ROOT / 'utopia/utils/seeding.py').read_text())
    nodes += [node for node in util.body if isinstance(node, ast.FunctionDef) and node.name == 'derive_seed']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-population-functions>', 'exec'), ns)
    module = types.SimpleNamespace(**ns)
    ns['build_influx_cohort'] = lambda year, seed: runner.build_fixed_cohort(module, year, seed)
    return module


def fake_simulation(module, cell='A'):
    class Ecosystem:
        def __init__(self):
            self.agent_population = {}

        def add_agent(self, agent):
            self.agent_population[agent.id] = agent

        def get_agent_by_id(self, name):
            return self.agent_population[name]

    sim = types.SimpleNamespace(
        args=types.SimpleNamespace(seed=42, population_mode='default',
                                   annual_influx_rate=0.1 if cell in 'BD' else 0),
        debug=False, llm=None, industry_funding_mode='paper', ecosystem=Ecosystem())
    for name in ('_record_initial_university_count', '_add_funding_agencies', '_apply_annual_influx'):
        setattr(sim, name, types.MethodType(getattr(module, name), sim))
    return sim


def write_request_fixture(out):
    """Closed real RequestAudit with an explicit scripted CPU non-funding request."""
    native_fixture.write_nonfunding_request_fixture(out)


def write_audit_fixtures(out, years=10):
    (out / runner.FUNDING_AUDIT_FILE).write_text('')
    (out / runner.FUNDING_SUMMARY_FILE).write_text(json.dumps({
        **runner.funding_summary_gates(), 'audit_records': 0, 'validation_attempts': 0,
        'invalid_attempts': 0, 'final_panels': 0, 'fixture_only': True}))
    write_request_fixture(out)
    from tests.support.simulation import zero_sequential_fixture
    if not (out / runner.SEQUENTIAL_SUMMARY_FILE).exists():
        zero_sequential_fixture(out, years=years)


def compact_scripted_funding(results, seed_ctx, response_format):
    """Adapt scripted answers to the requested wire schema; never alter rank order."""
    schema = (response_format or {}).get('json_object', {}).get('schema', {})
    if seed_ctx[0] != 'phase5_funding_eval' or 'ranked_application_ids' not in schema.get('properties', {}):
        return results
    return [({'ranked_application_ids': [
        row['application_id'] for row in sorted(result['ranked_applications'], key=lambda row: row['rank'])
    ]}, history) for result, history in results]


class FixtureTokenizer:
    """Fixed 20-token CPU fixture, never a replacement for production tokenization."""
    chat_template = 'TEST_ONLY'

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(20))}


class LauncherTests(unittest.TestCase):




    def test_founders_are_researchers_and_match_across_cells(self):
        module = population_module()
        state = random.getstate()
        founders = runner.build_founders(module)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(len(founders), 1202)
        self.assertEqual(sum(r['kind'] == 'university' for r in founders), 1200)
        self.assertEqual(sum(r['kind'] == 'industry' for r in founders), 2)
        self.assertEqual({r['institution'] for r in founders if r['kind'] == 'industry'}, {'Google'})
        self.assertEqual({r['strategy'] for r in founders}, {'balanced'})
        for cell in 'ABCD':
            sim = fake_simulation(module, cell)
            runner.initialize_population(sim, module, runner.build_founders(module))
            self.assertEqual((sim.initial_university_count, sim.initial_industry_count), (1200, 2))
            self.assertEqual(len(sim.ecosystem.agent_population), 1204)
            self.assertEqual(sum(a.can_author for a in sim.ecosystem.agent_population.values()), 1202)
            self.assertEqual(runner.object_hash(runner.build_founders(module)), runner.object_hash(founders))

    def test_v6_preserves_v2_founder_and_ten_year_cohort_rng_exactly(self):
        # Golden digests captured from the unchanged v2 implementation and this
        # 12-direction CPU fixture before the protocol rename.
        module = population_module()
        self.assertEqual(runner.PROTOCOL, 'influx1202_v6')
        self.assertEqual(runner.POPULATION_RNG_NAMESPACE, INFLUX_POPULATION_SEED_NAMESPACE)
        self.assertEqual(runner.object_hash(runner.build_founders(module)),
                         'f6cf10b3358f2fb42d8c206318d090017e21b5d0982a87cfa958358cc1bc6b2f')
        self.assertEqual(runner.object_hash([runner.build_fixed_cohort(module, y) for y in range(1, 11)]),
                         'd0dcd90351568de4a07a33047656218ceed100f92da75ccf5493e9e993f04f5a')





    def test_growth_reuses_stock_injection_is_idempotent_and_includes_inactive_coi(self):
        module = population_module()
        sim = fake_simulation(module, 'D')
        founders = runner.build_founders(module)
        runner.initialize_population(sim, module, founders)
        sim.ecosystem.agent_population['institution_0000_researcher_0'].is_active = False
        seen = {r['researcher_name'] for r in founders}
        state = random.getstate()
        for year in range(1, 11):
            cohort = runner.build_fixed_cohort(module, year)
            self.assertEqual(len(cohort), 121 if year in (5, 10) else 120)
            self.assertEqual(sum(r['kind'] == 'university' and r['funding_level'] == 160
                                 for r in cohort), 40)
            self.assertFalse(seen.intersection(r['researcher_name'] for r in cohort))
            seen.update(r['researcher_name'] for r in cohort)
            self.assertEqual(len(seen), 1202 + 1202 * year // 10)
            self.assertEqual(len(sim._apply_annual_influx(year)), len(cohort))
            self.assertEqual(sim._apply_annual_influx(year), [])
            self.assertEqual(seen, runner.expected_researcher_ids('D', year))
        self.assertEqual(random.getstate(), state)
        founder = sim.ecosystem.agent_population['institution_0000_researcher_0']
        entrant = sim.ecosystem.agent_population['institution_0000_researcher_5']
        self.assertIn(founder.id, entrant.conflict_of_interest)
        self.assertIn(entrant.id, founder.conflict_of_interest)
        stable = fake_simulation(module, 'C')
        runner.initialize_population(stable, module, founders)
        self.assertEqual(stable._apply_annual_influx(1), [])
        self.assertEqual(len(stable.ecosystem.agent_population), 1204)







































@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('pandas'),
                     'The original funding-phase fixture requires NumPy and Pandas')
class TestPanelizedFundingPhase(unittest.TestCase):
    def test_100_applicants_multiple_panels_match_stock_phase_and_quotas(self):
        """Original phase/selection/prompts/processing; only the LLM is scripted."""
        from collections import Counter
        import tests.support.simulation as fixture
        snapshots = []
        for guarded in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory).resolve()
                sim, agents, agency = fixture.make_simulation(out, base=True, balance=100)
                sim.args.seed = 42
                sim.args.funding_panel_max_apps = 25
                sim.args.log_funding_applications = True
                sim.funding_budget_mode = 'track'
                sim.llm = fixture.SequentialScriptedLLM()
                for i in range(2, 100):
                    agent = fixture.UniversityResearcher(
                        f'founder_{i}', f'institution_{i}', funding_level=100,
                        expertise=[fixture.DirectionStub('algorithms')])
                    agents.append(agent)
                    sim.ecosystem.add_agent(agent)
                agency.funding_programs = {
                    name: types.SimpleNamespace(
                        program_id=name, name=name, topics=['algorithms'],
                        research_directions=[fixture.DirectionStub('algorithms')],
                        funding_rate=rate)
                    for name, rate in (('NSF_THEORY', 0.23), ('DARPA_AI_APPS', 0.10))}
                utility = types.SimpleNamespace(derive_seed=fixture.ORIGINAL['derive_seed'])
                state = random.getstate()
                with mock.patch.dict(sys.modules, {'utopia.utils.seeding': utility}):
                    if guarded:
                        with fixture.fixture_request_scope(out) as request_audit, runner.funding_validation_scope(
                                types.SimpleNamespace(FundingAgency=fixture.FundingAgency,
                                                      VLLMServerModel=fixture.SequentialScriptedLLM,
                                                      Simulation=fixture.Simulation),
                                {'cell': 'A', 'output_dir': str(out)}, out) as handle:
                            sim.llm = fixture.SequentialScriptedLLM(request_audit=request_audit)
                            fixture.run_phase(sim)
                        self.assertTrue(handle.summary()['completion_allowed'])
                        self.assertEqual(handle.summary()['final_panels'], 8)
                        self.assertEqual(handle.summary()['successful_batches'], 1)
                        self.assertEqual(handle.summary()['invalid_attempts'], 0)
                        self.assertEqual(handle.summary()['output_representation'],
                                         'ordered_application_ids_v1')
                        from utopia.funding.feedback import validate_sequential_evidence
                        seq = validate_sequential_evidence(
                            out, handle.summary(), num_years=1, expected_agency_ids=("agency",))
                        self.assertEqual(seq['accepted_steps'], 200)
                        self.assertEqual(seq['processed_panels'], 8)
                    else:
                        fixture.run_phase(sim)
                self.assertEqual(random.getstate(), state)
                log = [json.loads(line) for line in
                       (out / 'funding_applications_year_1.jsonl').read_text().splitlines()]
                self.assertEqual(len(log), 200)
                self.assertEqual({row['n_panel'] for row in log}, {25})
                groups = Counter((row['program_id'], row['panel_index']) for row in log)
                self.assertEqual(len(groups), 8)
                self.assertEqual(set(groups.values()), {25})
                funded = Counter(row['program_id'] for row in log if row['funded'])
                self.assertEqual(funded, {'NSF_THEORY': 20, 'DARPA_AI_APPS': 8})
                # Global quotas would be 23 and 10: cap25 changes rounding.
                self.assertNotEqual(funded['NSF_THEORY'], int(100 * 0.23))
                self.assertNotEqual(funded['DARPA_AI_APPS'], int(100 * 0.10))
                self.assertEqual(sum(a.resources - 100 for a in agents), 28 * 20)
                self.assertTrue(all(not row['fallback_ranking'] and not row['imputed_tail']
                                    for row in log))
                snapshots.append({
                    'resources': [a.resources for a in agents],
                    'history': [a.funding_success_history for a in agents],
                    'application_calls': [call for call in sim.llm.calls
                                          if call[1][0] == 'phase5_funding_apps'],
                    'log': log,
                })
        self.assertEqual(snapshots[0], snapshots[1])




if __name__ == '__main__':
    unittest.main()
