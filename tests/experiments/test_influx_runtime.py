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
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'isolated'
        self.root.mkdir()
        # Source-binding fixtures live only inside this test's temporary root.
        for name in ('utopia/funding/sequential.py', runner.SEQUENTIAL_PROTOCOL_PATH):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((runner.REPO_ROOT / name).read_bytes())

    def args(self, *extra):
        return runner.parse_args(list(extra))

    def finish(self, job, phase=5, status='complete'):
        out = Path(job['output_dir'])
        out.mkdir(parents=True, exist_ok=True)
        (out / f"checkpoint_year_{job['num_years']}.json").write_text(json.dumps(
            {'year': job['num_years'], 'phase': phase}))
        (out / 'run_manifest.json').write_text(json.dumps(
            {'status': status, 'years_completed': job['num_years']}))
        (out / 'influx_result.json').write_text(json.dumps({
            'status': status, 'protocol': runner.PROTOCOL, 'seed': 42, 'cell': job['cell'],
            'founder_researchers': 1202, 'funding_agencies': 2, 'num_years': job['num_years'],
            'funding': runner.funding_spec(),
            'funding_selection_protocol': runner.FUNDING_SELECTION_PROTOCOL,
            'sequential_funding_status': 'complete',
            'funding_application_evidence_valid': True,
            'funding_application_evidence': {'present_sha256': {},
                                             'absent_years': list(range(1, job['num_years'] + 1))},
            'funding_validation_status': 'complete', 'request_audit_status': 'complete',
            'checked_years': list(range(1, job['num_years'] + 1))}))
        (out / 'influx_protocol.json').write_text(json.dumps({
            'common': {'protocol': runner.PROTOCOL, 'seed': 42, 'population': runner.population_spec(),
                       'population_rng_namespace': runner.POPULATION_RNG_NAMESPACE,
                       'output_elicitation_amendment': runner.OUTPUT_ELICITATION_AMENDMENT,
                       'funding': runner.funding_spec(), 'request_audit': runner.request_audit_spec(),
                       'funding_selection_protocol': runner.FUNDING_SELECTION_PROTOCOL,
                       'sequential_funding': runner.sequential_spec(),
                       'sequential_elicitation_amendment': runner.SEQUENTIAL_ELICITATION_AMENDMENT,
                       'structured_output_transport': runner.STRUCTURED_OUTPUT_TRANSPORT},
            'cell': job['cell'], 'experiment_name': job['experiment_name'],
            'num_years': job['num_years']}))
        write_audit_fixtures(out, years=job['num_years'])

    def finish_real(self, job, args=None, *, audit_fixtures=True, validate=True):
        args = args or self.args()
        module = population_module()
        founders = runner.build_founders(module)
        record = runner.cell_provenance(args, job, {'commit': 'f' * 40}, {}, founders, self.root)
        runner.freeze_cell(job, record, founders, self.root)
        sim = fake_simulation(module, job['cell'])
        runner.initialize_population(sim, module, founders)
        out = Path(job['output_dir'])
        for year in range(1, job['num_years'] + 1):
            sim._apply_annual_influx(year)
            agents = []
            for agent in sim.ecosystem.agent_population.values():
                row = {'id': agent.id, 'can_author': agent.can_author}
                for key in ('university_name', 'company_name', 'conflict_of_interest'):
                    if hasattr(agent, key):
                        value = getattr(agent, key)
                        row[key] = sorted(value) if isinstance(value, set) else value
                agents.append(row)
            runner.write_json(out / f'checkpoint_year_{year}.json', {
                'year': year, 'phase': 5, 'initial_university_count': 1200,
                'initial_industry_count': 2, 'ecosystem_data': {'agents': agents},
                'paper_tracker': {'papers': []},
            }, self.root)
        runner.write_json(out / 'run_manifest.json',
                          {'status': 'complete', 'years_completed': job['num_years']}, self.root)
        if audit_fixtures:
            write_audit_fixtures(out, years=job['num_years'])
        if validate:
            runner.write_json(out / 'influx_result.json', runner.validate_results(job, self.root), self.root)
        return record

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

    def test_v6_fresh_paths_leave_prior_artifacts_untouched(self):
        jobs = runner.make_jobs(self.args(), self.root)
        previous = []
        for job in jobs:
            for version in ('influx1202_v2', 'influx1202_v3', 'influx1202_v4', 'influx1202_v5'):
                old = Path(job['output_dir'].replace('influx1202_v6', version))
                old.mkdir(parents=True)
                (old / 'evidence.txt').write_text('preserve prior attempt evidence')
                previous.append(old / 'evidence.txt')
                with self.assertRaisesRegex(ValueError, 'fresh influx1202_v6'):
                    runner.cell_cmd(job['cell'], old.name, 10)
            self.assertFalse(Path(job['output_dir']).exists())
            self.assertFalse(runner.completed(job))
        with self.assertRaisesRegex(ValueError, 'every selected'):
            runner.run_analysis(self.args(), jobs, root=self.root)
        self.assertTrue(all(p.read_text() == 'preserve prior attempt evidence' for p in previous))
        self.assertEqual(runner.PRIOR_ATTEMPT['failure_class'], 'infrastructure_compatibility')
        self.assertEqual(runner.PRIOR_ATTEMPT['completed_worlds'], 0)
        self.assertIn('not outcome-based', runner.PRIOR_ATTEMPT['decision_basis'])
        self.assertEqual(runner.PRIOR_ATTEMPT['stop_requested_at_utc'], '2026-09-19T05:11:39Z')
        self.assertEqual(runner.OUTPUT_ELICITATION_AMENDMENT['predecessor_protocol'], 'influx1202_v3')
        self.assertEqual(runner.OUTPUT_ELICITATION_AMENDMENT['scientific_jobs_started'], 0)
        self.assertEqual(runner.SEQUENTIAL_ELICITATION_AMENDMENT['completed_conditions'], 0)
        self.assertEqual(runner.SEQUENTIAL_ELICITATION_AMENDMENT['predecessor_protocol'], 'influx1202_v5')

    def test_v6_completion_rejects_old_protocol_or_missing_transport_contract(self):
        job = runner.make_jobs(self.args(), self.root)[0]
        self.finish(job)
        self.assertTrue(runner.completed(job))
        path = Path(job['output_dir']) / 'influx_protocol.json'
        correct = json.loads(path.read_text())
        for field, value in (('protocol', 'influx1202_v2'),
                             ('protocol', 'influx1202_v3'),
                             ('structured_output_transport', 'guided_json'),
                             ('population_rng_namespace', 'influx1202_v3'),
                             ('funding', None), ('request_audit', None),
                             ('output_elicitation_amendment', None)):
            record = json.loads(json.dumps(correct))
            record['common'][field] = value
            path.write_text(json.dumps(record))
            self.assertFalse(runner.completed(job), field)

    def test_v6_transport_guard_rejects_legacy_before_simulation(self):
        for field in ('guided_json', 'structured_outputs'):
            def builder(probe, fmt):
                self.assertTrue(probe.enable_thinking)
                schema = fmt['json_object']['schema']
                return {field: schema if field == 'guided_json' else {'json': schema}}
            module = types.SimpleNamespace(VLLMServerModel=types.SimpleNamespace(build_extra_body=builder))
            if field == 'guided_json':
                with self.assertRaisesRegex(ValueError, 'compatibility fix'):
                    runner.validate_structured_transport(module)
            else:
                runner.validate_structured_transport(module)

    def test_current_shared_client_transport_satisfies_v6_contract(self):
        # Exercise Hypatia's actual shared helper without importing GPU/runtime
        # dependencies. This gate intentionally fails if only the wrapper was
        # copied and the shared transport correction was omitted.
        path = runner.REPO_ROOT / 'utopia/models/models.py'
        tree = ast.parse(path.read_text())
        method = next(f for cls in tree.body if isinstance(cls, ast.ClassDef)
                      and cls.name == 'VLLMServerModel' for f in cls.body
                      if isinstance(f, ast.FunctionDef) and f.name == 'build_extra_body')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        helper_path = path.with_name('structured_outputs.py')
        spec = importlib.util.spec_from_file_location('influx_transport_probe_helper', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        namespace = {'build_vllm_extra_body': helper.build_vllm_extra_body}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                     str(path), 'exec'), namespace)
        module = types.SimpleNamespace(VLLMServerModel=types.SimpleNamespace(
            build_extra_body=namespace['build_extra_body']))
        runner.validate_structured_transport(module)

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

    def test_real_result_gates_reject_wrong_population_and_incomplete_year(self):
        job = runner.make_jobs(self.args('--smoke'), self.root)[0]
        self.finish_real(job)
        self.assertEqual(runner.validate_results(job, self.root)['population_by_year']['3']['researchers'], 1202)
        cp_path = Path(job['output_dir']) / 'checkpoint_year_2.json'
        cp = json.loads(cp_path.read_text())
        cp['ecosystem_data']['agents'].pop(0)
        cp_path.write_text(json.dumps(cp))
        with self.assertRaisesRegex(ValueError, 'IDs/counts'):
            runner.validate_results(job, self.root)
        cp['phase'] = 3
        cp_path.write_text(json.dumps(cp))
        with self.assertRaisesRegex(ValueError, 'phase5'):
            runner.validate_results(job, self.root)

    def test_frozen_cell_rejects_old_population_and_changed_science(self):
        args = self.args()
        job = runner.make_jobs(args, self.root)[0]
        module = population_module()
        founders = runner.build_founders(module)
        record = runner.cell_provenance(args, job, {'commit': 'f' * 40}, {}, founders, self.root)
        runner.freeze_cell(job, record, founders, self.root)
        runner.freeze_cell(job, record, founders, self.root)
        changed = json.loads(json.dumps(record))
        changed['common']['population']['founder_researchers'] = 300
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            runner.freeze_cell(job, changed, founders, self.root)
        other = runner.make_jobs(args, self.root)[1]
        self.finish(other)
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            runner.freeze_cell(other, record, founders, self.root)

    def test_default_paper_cells_and_serving(self):
        args = self.args()
        self.assertEqual(args.seeds, [42])
        self.assertEqual(args.tensor_parallel_size, 8)
        self.assertEqual(args.server_gpus, '0,1,2,3,4,5,6,7')
        self.assertEqual(args.client_gpu, '0')
        self.assertEqual(args.port, 8046)
        self.assertEqual(args.max_concurrent_cells, 4)
        jobs = runner.make_jobs(args, self.root)
        self.assertEqual(len(jobs), 4)
        for job in jobs:
            worker = job['command']
            self.assertEqual(worker[worker.index('--run_cell') + 1], job['cell'])
            self.assertIn(runner.PROTOCOL, job['experiment_name'])
            self.assertEqual(job['population']['founder_researchers'], 1202)
            command = job['simulation_command']
            self.assertEqual(job['num_years'], 10)
            self.assertEqual(command[command.index('--seed') + 1], '42')
            self.assertEqual(command[command.index('--model') + 1], 'Qwen/Qwen3-32B')
            self.assertEqual(command[command.index('--batch_size') + 1], '64')
            self.assertEqual(command[command.index('--funding_panel_max_apps') + 1], '25')
            self.assertEqual(command[command.index('--funding_budget_mode') + 1], 'track')
            self.assertIn('--log_funding_applications', command)
            self.assertNotIn('--experiment_stage', command)
            self.assertNotIn('--population_mode', command)  # Preserve legacy default.
            self.assertNotIn('--num_institutions', command)
            self.assertEqual('--disable_resubmission' in command, job['cell'] in 'AB')
            self.assertEqual('--annual_influx_rate' in command, job['cell'] in 'BD')
            self.assertTrue(Path(job['output_dir']).is_relative_to(self.root))
        command = runner.server_cmd(args)
        self.assertIn(runner.REVISION, command)
        self.assertEqual(command[command.index('--dtype') + 1], 'bfloat16')
        self.assertEqual(command[command.index('--seed') + 1], '42')
        self.assertEqual(command[command.index('--max-model-len') + 1], '32768')

    def test_common_funding_flags_reach_actual_parser_and_frozen_provenance(self):
        spec = importlib.util.spec_from_file_location(
            'influx_argument_fixture', runner.REPO_ROOT / 'utopia/arguments.py')
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        args = self.args()
        for job in runner.make_jobs(args, self.root):
            worker = self.args(*job['command'][3:])
            self.assertEqual(worker.funding_panel_max_apps, 25)
            argv = list(job['simulation_command'][3:])
            argv[argv.index('--output-dir') + 1] = str(self.root / 'parser_outputs')
            actual = parser.parse_arguments(argv)
            self.assertTrue(Path(actual.output_dir).is_relative_to(self.root))
            self.assertEqual(actual.funding_panel_max_apps, 25)
            self.assertEqual(actual.funding_budget_mode, 'track')
            self.assertTrue(actual.log_funding_applications)
            self.assertEqual(actual.funding_application_cost, 0)
            self.assertEqual(actual.funding_allocation_mode, 'fixed')
            record = runner.cell_provenance(args, job, {'commit': 'f' * 40}, {}, [], self.root)
            self.assertEqual(record['common']['funding'], runner.funding_spec())
            self.assertFalse(record['common']['funding']['global_ranking_equivalent'])
            self.assertIn('quota rounding', record['limitations'])
        for cap in ('0', '24', '26', '-1'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args('--funding_panel_max_apps', cap)
        hashes = runner.source_hashes()
        for source in ('utopia/funding/validation.py', 'utopia/funding/compact.py',
                       'utopia/funding/sequential.py', runner.SEQUENTIAL_PROTOCOL_PATH,
                       'utopia/models/request_audit.py',
                       'utopia/models/models.py'):
            self.assertEqual(hashes[source], runner.sha256(runner.REPO_ROOT / source))

    def test_audit_gates_reject_missing_failed_or_unprocessed_runtime_evidence(self):
        job = runner.make_jobs(self.args('--smoke'), self.root)[0]
        self.finish_real(job)
        out = Path(job['output_dir'])
        for filename, key, value in (
                (runner.FUNDING_SUMMARY_FILE, 'output_representation', 'legacy_ranked_applications'),
                (runner.FUNDING_SUMMARY_FILE, 'completion_allowed', False),
                (runner.FUNDING_SUMMARY_FILE, 'unprocessed_panels', 1),
                (runner.FUNDING_SUMMARY_FILE, 'processing_failures', 1),
                (runner.REQUEST_SUMMARY_FILE, 'status', 'failed'),
                (runner.REQUEST_SUMMARY_FILE, 'n_clients', 0),
                (runner.REQUEST_SUMMARY_FILE, 'guard_failures', 1),
                (runner.REQUEST_SUMMARY_FILE, 'pending_requests', 1),
                (runner.REQUEST_SUMMARY_FILE, 'n_requests', 0)):
            path = out / filename
            original = path.read_bytes()
            row = json.loads(original)
            row[key] = value
            path.write_text(json.dumps(row))
            self.assertFalse(runner.completed(job), key)
            with self.assertRaises(ValueError):
                runner.validate_results(job, self.root)
            path.write_bytes(original)
        for filename in (runner.FUNDING_AUDIT_FILE, runner.REQUEST_AUDIT_FILE):
            path = out / filename
            original = path.read_bytes()
            path.unlink()
            self.assertFalse(runner.completed(job), filename)
            path.write_bytes(original)
        self.assertTrue(runner.completed(job))

    def test_funding_guard_rejects_unprocessed_panels_and_restores_after_fatal_error(self):
        from tests.support.funding_rankings import apps_for
        from tests.support.simulation import FundingAgency, SequentialScriptedLLM
        from utopia.funding.validation import FundingRankingValidationError
        from utopia.funding.sequential import SequentialFundingError
        for incomplete in (False, True):
            cls = FundingAgency
            original = vars(cls)['validate_funding_result']
            out = self.root / ('pending' if incomplete else 'fatal')
            out.mkdir()
            job = {'output_dir': str(out), 'cell': 'A'}
            module = types.SimpleNamespace(FundingAgency=cls, VLLMServerModel=SequentialScriptedLLM,
                                           Simulation=native_fixture.Simulation)
            with self.assertRaises(FundingRankingValidationError):
                with runner.funding_validation_scope(module, job, self.root):
                    apps = apps_for(2)
                    cls.validate_funding_result({'ranked_application_ids': [0, 1]}, apps)
                    if not incomplete:
                        raise FundingRankingValidationError({'fixture': 'fatal'})
            self.assertIs(vars(cls)['validate_funding_result'], original)
            summary = json.loads((out / runner.FUNDING_SUMMARY_FILE).read_text())
            self.assertEqual(summary['status'], 'failed')
            self.assertFalse(summary['completion_allowed'])
            self.assertFalse(summary['installed'])
            preserved = (out / runner.FUNDING_AUDIT_FILE).read_bytes()
            with self.assertRaises(FundingRankingValidationError):
                with runner.funding_validation_scope(module, job, self.root):
                    self.fail('existing audit must not be adopted')
            self.assertEqual((out / runner.FUNDING_AUDIT_FILE).read_bytes(), preserved)

    def test_single_world_four_cells_have_disjoint_outputs_and_caches(self):
        args = self.args('--seeds', '42')
        self.assertEqual(args.seeds, [42])
        self.assertEqual(args.requested_seeds, [42])
        jobs = runner.make_jobs(args, self.root)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({j['output_dir'] for j in jobs}), 4)
        self.assertEqual(len({j['env_overrides']['UTOPIA_DATA_CACHE_DIR'] for j in jobs}), 4)
        for job in jobs:
            self.assertEqual(job['command'][job['command'].index('--seeds') + 1], '42')
            self.assertEqual(job['env_overrides']['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(runner.server_spec(args)['seed'], 42)

    def test_configurable_hardware_never_changes_seed(self):
        args = self.args('--server_gpus', '0,3,5,6',
                         '--client_gpu', '5', '--port', '8050')
        self.assertEqual(args.tensor_parallel_size, 4)
        self.assertEqual(args.seeds, [42])
        self.assertEqual(args.gpu_memory_utilization, 0.80)
        self.assertEqual(args.server_url, 'http://localhost:8050/v1')
        self.assertEqual(runner.parse_args(['--max_concurrent_cells', '1']).max_concurrent_cells, 1)
        ec2 = self.args('--server_gpus', '0,1,2,3,4,5,6,7',
                        '--external_server', 'http://ec2-server:8046/v1')
        self.assertEqual(ec2.tensor_parallel_size, 8)
        self.assertEqual(ec2.server_url, 'http://ec2-server:8046/v1')
        self.assertEqual(ec2.seeds, [42])
        for pair in ('0,3', '5,6'):
            worker = self.args('--server_gpus', pair, '--client_gpu', '7', '--run_cell', 'A',
                               '--external_server', 'http://localhost:8047/v1')
            self.assertEqual(worker.tensor_parallel_size, 2)
            self.assertEqual(len(runner.make_jobs(worker, self.root)), 1)

    def test_invalid_arguments_fail_before_any_work(self):
        invalid = [
            ['--seeds', '42', '42'], ['--seeds', '-1'], ['--seeds', str(2**31)],
            ['--seeds', '43'], ['--seeds', '42', '43', '44'], ['--run_cell', 'A'],
            ['--server_gpus', '1,1'], ['--server_gpus', '1,01'],
            ['--server_gpus', ''], ['--client_gpu', '1,3'],
            ['--tensor_parallel_size', '4'], ['--server_gpus', '1,2,3'],
            ['--port', '0'], ['--port', '65536'], ['--max_concurrent_cells', '0'],
            ['--gpu_memory_utilization', 'nan'], ['--gpu_memory_utilization', '1'],
            ['--client_headroom_gb', '-2'], ['--timeout_hours', 'inf'],
            ['--external_server', 'file:///tmp/v1'],
            ['--external_server', 'http://user:secret@localhost:8046/v1'],
            ['--external_server_manifest', 'unused.json'],
        ]
        for argv in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.parse_args(argv)

    def test_dry_run_does_not_touch_gpu_git_or_disk(self):
        with mock.patch.object(runner, 'gpu_memory', side_effect=AssertionError('GPU called')), \
                mock.patch.object(runner, 'git_state', side_effect=AssertionError('git called')), \
                mock.patch.object(runner, 'spawn', side_effect=AssertionError('spawn called')), \
                mock.patch.object(runner, 'write_json', side_effect=AssertionError('write called')), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(['--seeds', '42', '--dry_run']), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan['seeds'], [42])
        self.assertEqual(len(plan['jobs']), 4)

    def test_external_server_defaults_and_attestation(self):
        args = self.args('--external_server')
        self.assertEqual(args.server_url, 'http://localhost:8046/v1')
        self.assertIn('operator-declared', runner.external_attestation(args)['verification'])
        path = self.root / 'external.json'
        payload = {**runner.server_spec(args), 'command': runner.server_cmd(args)}
        path.write_text(json.dumps(payload))
        args.external_server_manifest = path
        self.assertEqual(runner.external_attestation(args), payload)
        payload['revision'] = 'different'
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'revision'):
            runner.external_attestation(args)

    def test_gpu_sharing_requires_headroom_before_and_after_server_start(self):
        args = self.args('--server_gpus', '0,3')
        runner.check_gpu_headroom(args, {'0': (0, 81920), '3': (0, 81920)})
        with self.assertRaisesRegex(ValueError, 'headroom'):
            runner.check_gpu_headroom(args, {'0': (12000, 81920), '3': (0, 81920)})
        runner.check_gpu_headroom(args, {'0': (65536, 81920)}, server_running=True)
        with self.assertRaisesRegex(ValueError, 'headroom'):
            runner.check_gpu_headroom(args, {'0': (80000, 81920)}, server_running=True)
        external = self.args('--external_server', 'http://server:8046/v1')
        runner.check_gpu_headroom(external, {'0': (0, 81920)})  # Remote server GPUs not probed.

    def test_redirected_writable_environment_and_offline_model_reads(self):
        env = runner.runtime_env(self.args(), 'seed42_A', self.root)
        with mock.patch.dict(os.environ, {'TMPDIR': '/not/ours', 'WANDB_MODE': 'online'}):
            resolved = runner.child_env(env, self.root)
        self.assertEqual(resolved['WANDB_MODE'], 'disabled')
        self.assertEqual(resolved['HF_HUB_OFFLINE'], '1')
        for key in ('UTOPIA_DATA_CACHE_DIR', 'TMPDIR', 'MPLCONFIGDIR', 'VLLM_CACHE_ROOT',
                    'TORCHINDUCTOR_CACHE_DIR', 'WANDB_DIR', 'HF_DATASETS_CACHE'):
            self.assertTrue(Path(resolved[key]).is_relative_to(self.root), key)
            self.assertTrue(Path(resolved[key]).is_dir(), key)

    def test_output_symlinks_and_hardlinks_are_rejected(self):
        original = Path(self.temp.name) / 'original'
        original.mkdir()
        (self.root / 'outputs').symlink_to(original, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            runner.check_output_isolation(self.root)
        (self.root / 'outputs').unlink()
        (self.root / 'outputs').mkdir()
        source = original / 'result'
        source.write_text('original')
        os.link(source, self.root / 'outputs' / 'result')
        with self.assertRaisesRegex(ValueError, 'hard links'):
            runner.check_output_isolation(self.root)

    def test_explicit_private_storage_links_are_allowed_but_nested_escapes_are_not(self):
        storage = Path(self.temp.name) / 'private_storage'
        for name in ('outputs', 'data'):
            (storage / name).mkdir(parents=True)
            (self.root / name).symlink_to(storage / name, target_is_directory=True)
        with mock.patch.object(runner, 'STORAGE_ROOT', storage):
            runner.check_output_isolation(self.root)
            path = self.root / 'outputs/logs/run.json'
            runner.write_json(path, {'private': True}, self.root)
            self.assertTrue((storage / 'outputs/logs/run.json').exists())
            (storage / 'outputs/escape').symlink_to(self.temp.name, target_is_directory=True)
            with self.assertRaises(ValueError):
                runner.check_output_isolation(self.root)

    def test_cache_symlinks_are_copied_without_original_mutation(self):
        sources = []
        for i, rel in enumerate(runner.SHARED_CACHES):
            source = Path(self.temp.name) / f'original_cache{i}'
            source.write_bytes(f'input-{i}'.encode())
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            sources.append(source)
        runner.check_output_isolation(self.root)
        records = runner.materialize_caches(self.root)
        for i, rel in enumerate(runner.SHARED_CACHES):
            path = self.root / rel
            self.assertFalse(path.is_symlink())
            self.assertEqual(records[rel]['sha256'], runner.sha256(sources[i]))
            path.write_text('isolated rewrite')
            self.assertEqual(sources[i].read_bytes(), f'input-{i}'.encode())

    def test_cache_hardlinks_are_detached(self):
        sources = []
        for i, rel in enumerate(runner.SHARED_CACHES):
            source = Path(self.temp.name) / f'original{i}'
            source.write_text('unchanged')
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, path)
            sources.append(source)
        runner.materialize_caches(self.root)
        for rel, source in zip(runner.SHARED_CACHES, sources):
            (self.root / rel).write_text('local only')
            self.assertEqual(source.read_text(), 'unchanged')

    def test_nested_cache_directory_symlink_rejected(self):
        external = Path(self.temp.name) / 'external'
        external.mkdir()
        (self.root / 'data').mkdir()
        (self.root / 'data/cache').symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            runner.check_output_isolation(self.root)

    def test_branch_guard_rejects_main_and_primary_checkout(self):
        def git_result(command, **kwargs):
            tail = command[1:]
            if tail == ['rev-parse', 'HEAD']:
                return 'a' * 40
            if tail == ['branch', '--show-current']:
                return branch
            if tail == ['status', '--short']:
                return ''
            if tail == ['rev-parse', '--show-toplevel']:
                return str(self.root)
            if tail == ['rev-parse', '--git-dir']:
                return str(self.root / git_dir)
            if tail == ['rev-parse', '--git-common-dir']:
                return str(self.root / '.git')
            raise AssertionError(command)
        branch, git_dir = 'main', '.git/worktrees/llm'
        with mock.patch.object(runner.subprocess, 'check_output', side_effect=git_result):
            with self.assertRaisesRegex(ValueError, 'non-main'):
                runner.git_state(self.root)
            branch, git_dir = 'llm', '.git'
            with self.assertRaisesRegex(ValueError, 'linked worktree'):
                runner.git_state(self.root)
            git_dir = '.git/worktrees/llm'
            self.assertEqual(runner.git_state(self.root)['branch'], 'llm')

    def test_frozen_manifest_cannot_change_or_adopt_historical_results(self):
        jobs = runner.make_jobs(self.args(), self.root)
        plan = {'jobs': jobs, 'seeds': [42], 'tp': 2}
        directory = self.root / 'outputs/logs/manifest'
        runner.freeze_manifest(plan, {}, {}, directory, self.root)
        original = (directory / 'launch_manifest.json').read_bytes()
        runner.freeze_manifest(plan, {}, {}, directory, self.root)
        self.assertEqual((directory / 'launch_manifest.json').read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'differs'):
            runner.freeze_manifest({**plan, 'tp': 4}, {}, {}, directory, self.root)
        self.finish(jobs[0])
        with self.assertRaisesRegex(ValueError, 'existing cell'):
            runner.freeze_manifest(plan, {}, {}, self.root / 'outputs/logs/new', self.root)

    def test_completion_rejects_partial_failed_and_corrupt_final_checkpoints(self):
        job = runner.make_jobs(self.args(), self.root)[0]
        self.finish(job, phase=3)
        self.assertFalse(runner.completed(job))
        self.finish(job, status='running')
        self.assertFalse(runner.completed(job))
        self.finish(job)
        self.assertTrue(runner.completed(job))
        (Path(job['output_dir']) / 'checkpoint_year_10.json').write_text('{')
        self.assertFalse(runner.completed(job))

    def run_fake_queue(self, fail_index=None, timeout=False, server_dead=False):
        args = self.args('--seeds', '42', '--max_concurrent_cells', '2')
        jobs = runner.make_jobs(args, self.root)
        attempt = self.root / 'outputs/logs/attempt'
        attempt.mkdir(parents=True)
        active, started, peak = [], [], [0]
        test = self

        class FakeProcess:
            def __init__(self, index):
                self.index, self.pid, self.returncode = index, index + 100, None
                self.polls = 0

            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                self.returncode = 7 if self.index == fail_index else 0
                if self.returncode == 0:
                    test.finish(jobs[self.index])
                return self.returncode

        def spawn(command, env, log, root):
            proc = FakeProcess(len(started))
            started.append(proc)
            active.append(proc)
            peak[0] = max(peak[0], len(active))
            return proc

        def stop(processes):
            for proc in processes:
                if proc in active:
                    active.remove(proc)

        with mock.patch.object(runner, 'spawn', side_effect=spawn), \
                mock.patch.object(runner, 'stop_processes', side_effect=stop), \
                mock.patch.object(runner.time, 'sleep'), \
                contextlib.redirect_stdout(io.StringIO()):
            server = types.SimpleNamespace(poll=lambda: 1) if server_dead else None
            call = lambda: runner.run_cells(args, jobs, attempt, server_proc=server,
                                           deadline=-1 if timeout else None, root=self.root)
            if timeout:
                with self.assertRaises(TimeoutError):
                    call()
            elif server_dead:
                with self.assertRaisesRegex(RuntimeError, 'server exited'):
                    call()
            else:
                self.assertEqual(call(), fail_index is None)
                self.assertEqual(len(started), 4)
                self.assertLessEqual(peak[0], 2)
            self.assertFalse(active)
        return json.loads((attempt / 'cell_status.json').read_text())

    def test_concurrency_bounded_for_all_preselected_cells(self):
        status = self.run_fake_queue()
        self.assertEqual(len(status), 4)
        self.assertTrue(all(row['state'] == 'complete' for row in status.values()))

    def test_failed_cell_does_not_hide_or_cancel_other_cells(self):
        status = self.run_fake_queue(fail_index=1)
        self.assertEqual(status['42_B']['state'], 'failed')
        self.assertEqual(status['42_D']['state'], 'complete')

    def test_walltime_prevents_launch(self):
        status = self.run_fake_queue(timeout=True)
        self.assertTrue(all(row['state'] == 'not_started' for row in status.values()))

    def test_dead_owned_server_prevents_launch(self):
        self.run_fake_queue(server_dead=True)

    def test_interaction_summary_is_conditional_without_uncertainty(self):
        rows = {42: {'endpoint': {'interaction': -1.0}}}
        summary = runner.summarize_interactions(rows)['endpoint']
        self.assertEqual(summary, {'seed': 42, 'interaction': -1.0, 'status': 'complete'})
        self.assertIn('not independent seed', runner.LIMITATION)
        self.assertIn('not paired controls', runner.LIMITATION)
        with self.assertRaises(ValueError):
            runner.summarize_interactions({**rows, 43: rows[42]})

    def test_missing_endpoint_stays_explicit(self):
        result = runner.summarize_interactions(
            {42: {'endpoint': {'interaction': float('nan')}}})
        self.assertEqual(result['endpoint']['status'], 'missing')
        self.assertNotIn('mean', result['endpoint'])
        self.assertIsNone(result['endpoint']['interaction'])

    def test_undefined_metrics_are_json_null(self):
        path = self.root / 'undefined.json'
        runner.write_json(path, {'undefined': float('nan'), 'nested': [float('inf')]}, self.root)
        self.assertEqual(json.loads(path.read_text()), {'undefined': None, 'nested': [None]})

    def test_analysis_reuses_original_decomposition_without_paper_baseline(self):
        # Load the real pure endpoint/decomposition functions, stubbing only
        # the dependency-heavy checkpoint statistics reader (not its formulas).
        stats_module = types.ModuleType('utopia.analysis.resubmission_stats')
        def compute_stats(path, years):
            self.assertEqual(years, 10)
            name = Path(path).name
            cell = next(key for key in 'ABCD' if name == runner.cell_name(key, 42))
            value = {'A': 10, 'B': 13, 'C': 15, 'D': 21}[cell]
            return {
                'submissions_per_year': {year: value for year in range(1, 11)},
                'reviews_per_active_reviewer_by_year': {10: value / 2},
                'reviews_per_pool_member_by_year': {10: value / 3},
                'recycling_multiplier_by_year': {10: 1.0},
                'active_researchers_by_year': {1: 1202, 10: 1202},
                'population_by_year': {y: len(runner.expected_researcher_ids(cell, y))
                                       for y in range(1, 11)}, 'cascade_depth_mean': 1.0,
                'recycle_rate_pct': 0.0,
            }
        stats_module.compute_stats = compute_stats
        name = 'utopia.analysis.influx_factorial'
        source = runner.REPO_ROOT / 'utopia/analysis/influx_factorial.py'
        spec = importlib.util.spec_from_file_location(name, source)
        factorial = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {
                'utopia.analysis.resubmission_stats': stats_module, name: factorial}):
            spec.loader.exec_module(factorial)
            args = self.args('--seeds', '42')
            jobs = runner.make_jobs(args, self.root)
            for job in jobs:
                self.finish_real(job)
            result = runner.run_analysis(args, jobs, {'server': runner.server_spec(args)}, self.root)
        endpoint = result['interactions']['submissions_yT']
        self.assertEqual(endpoint, {'seed': 42, 'interaction': 3, 'status': 'complete'})
        self.assertEqual(result['endpoints_by_cell']['D']['population_yT'], 2404)
        self.assertNotIn('sample_sd', json.dumps(result))
        self.assertNotIn('standard_error', json.dumps(result))
        self.assertEqual(len(list((self.root / 'outputs/docs').rglob('factorial_summary.json'))), 1)
        self.assertFalse(list((self.root / 'outputs/docs').rglob('across_seed*')))
        records = []
        for job in jobs:
            path = Path(job['output_dir']) / 'influx_protocol.json'
            record = json.loads(path.read_text())
            record['server_url'] = f"http://localhost:{8046 + len(records)}/v1"
            path.write_text(json.dumps(record))
            records.append(record)
        runner.validate_cell_provenance(jobs)  # Endpoints may differ.
        records[-1]['common']['inference']['tensor_parallel_size'] = 2
        path.write_text(json.dumps(records[-1]))
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            runner.validate_cell_provenance(jobs)

    def test_analysis_refuses_incomplete_design(self):
        with self.assertRaisesRegex(ValueError, 'every selected'):
            runner.run_analysis(self.args(), runner.make_jobs(self.args(), self.root), {}, self.root)

    def test_analysis_only_bypasses_family_lock_caches_gpu_and_server(self):
        with mock.patch.object(runner, 'git_state', return_value={'common_dir': '/original/.git'}), \
                mock.patch.object(runner, 'check_output_isolation'), \
                mock.patch.object(runner, 'run_analysis') as analyze, \
                mock.patch.object(runner, 'materialize_caches', side_effect=AssertionError('cache write')), \
                mock.patch.object(runner, 'gpu_memory', side_effect=AssertionError('GPU called')), \
                mock.patch.object(runner, 'spawn', side_effect=AssertionError('spawn called')):
            self.assertEqual(runner.main(['--analysis_only']), 0)
            self.assertEqual(len(analyze.call_args.args[1]), 4)

    def test_cell_commands_and_gates_for_generic_tp2_queue(self):
        for cell, pair in zip('ABCD', ['0,3', '5,6', '0,3', '5,6']):
            command = runner.cell_cmd(
                cell, runner.cell_name(cell, 42), 10, server_url='__SERVER_URL__',
                server_gpus=pair, client_gpu='7', tensor_parallel_size=2)
            self.assertIn('__SERVER_URL__', command)
            self.assertNotIn('--external_server_manifest', command)
            argv = [v.replace('__SERVER_URL__', 'http://localhost:8046/v1') for v in command[3:]]
            args = self.args(*argv)
            job = runner.make_jobs(args, self.root)[0]
            self.assertEqual(job['cell'], cell)
            self.assertEqual(job['env_overrides']['CUDA_VISIBLE_DEVICES'], '7')
            self.assertEqual(job['env_overrides']['PYTHONHASHSEED'], '42')
            gates = job['completion_gates']
            self.assertEqual(gates['influx_result.json']['checked_years'], list(range(1, 11)))
            self.assertEqual(gates['checkpoint_year_10.json']['initial_university_count'], 1200)
            self.assertFalse(runner.completed(job))

    def test_queue_schema_templates_substitute_and_require_all_ten_checkpoints(self):
        private = self.root / 'private'
        jobs = runner.queue_jobs(self.root, private)
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({job['id'] for job in jobs}), 4)
        for job, cell in zip(jobs, 'ABCD'):
            self.assertEqual(job['id'], f'influx_influx1202_v6_{cell}_seed42')
            self.assertTrue(all('influx1202_v6' in path and 'influx1202_v3' not in path
                                and 'influx1202_v2' not in path
                                for path in job['output_roots']))
            self.assertEqual(job['command'][0], str(private / 'env/bin/python'))
            self.assertTrue(job['require_manifest_argument'])
            for pair in ('0,3', '5,6', '1,2', '4,7'):
                replacements = {
                    '__SERVER_URL__': 'http://localhost:8046/v1',
                    '__SERVER_MANIFEST__': str(private / 'server.json'),
                    '__SERVER_GPUS__': pair, '__CLIENT_GPU__': pair.split(',')[0],
                }
                command = [replacements.get(value, value) for value in job['command']]
                args = self.args(*command[3:])
                self.assertEqual(args.tensor_parallel_size, 2)
                self.assertEqual(args.client_gpu, pair.split(',')[0])
                self.assertEqual(args.external_server_manifest, private / 'server.json')
                self.assertEqual(args.storage_root, private)
                self.assertEqual(args.seeds, [42])
            self.assertEqual(sum('checkpoint_year_' in a['path'] for a in job['artifacts']), 10)
            gate = next(a['equals'] for a in job['artifacts'] if a['path'].endswith('influx_result.json'))
            self.assertEqual(gate['/population_by_year/10/researchers'], 2404 if cell in 'BD' else 1202)
            self.assertEqual(gate['/population_by_year/10/all_agents'], 2406 if cell in 'BD' else 1204)
            self.assertTrue(all(not Path(path).exists() for path in job['output_roots']))
            provenance = next(a['equals'] for a in job['artifacts'] if a['path'].endswith('influx_protocol.json'))
            self.assertEqual(provenance['/common/protocol'], 'influx1202_v6')
            self.assertEqual(provenance['/common/structured_output_transport'], 'structured_outputs.json')
            self.assertEqual(provenance['/common/population_rng_namespace'], INFLUX_POPULATION_SEED_NAMESPACE)
            self.assertEqual(provenance['/common/funding'], runner.funding_spec())
            self.assertEqual(provenance['/common/funding']['output_representation'],
                             'ordered_application_ids_v1')
            self.assertEqual(provenance['/common/output_elicitation_amendment'],
                             runner.OUTPUT_ELICITATION_AMENDMENT)
            self.assertEqual(provenance['/common/request_audit'], runner.request_audit_spec())
            for filename, gates in ((runner.FUNDING_SUMMARY_FILE, runner.funding_summary_gates()),
                                    (runner.SEQUENTIAL_SUMMARY_FILE, runner.sequential_summary_gates()),
                                    (runner.REQUEST_SUMMARY_FILE, runner.request_summary_gates())):
                artifact = next(a for a in job['artifacts'] if a['path'].endswith(filename))
                expected = {'/' + k: v for k, v in gates.items()}
                if filename == runner.SEQUENTIAL_SUMMARY_FILE:
                    expected.update({'/years_started': 10, '/years_completed': 10,
                                     '/agency_registrations': 20})
                self.assertEqual(artifact['equals'], expected)
            self.assertTrue(any(a['path'].endswith(runner.REQUEST_AUDIT_FILE) for a in job['artifacts']))
            self.assertTrue(all(a.get('min_bytes', 1) >= 1 for a in job['artifacts']
                                if Path(a['path']).name not in
                                (runner.FUNDING_AUDIT_FILE, runner.SEQUENTIAL_AUDIT_FILE)))
        manifest_args = self.args('--run_cell', 'A', '--external_server',
                                  '--server_gpus', '0,3', '--external_server_manifest',
                                  str(private / 'server.json'))
        private.mkdir()
        payload = {**runner.server_spec(manifest_args),
                   'command': runner.server_cmd(manifest_args), 'endpoint': manifest_args.server_url}
        manifest_args.external_server_manifest.write_text(json.dumps(payload))
        self.assertEqual(runner.external_attestation(manifest_args), payload)
        payload['endpoint'] = 'http://wrong-server:8046/v1'
        manifest_args.external_server_manifest.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            runner.external_attestation(manifest_args)

    def test_queue_attestation_is_recorded_without_raw_simulator_flag(self):
        args = self.args('--external_server')
        manifest = {'queue_verified': True, 'model': runner.MODEL, 'revision': runner.REVISION}
        with mock.patch.dict(os.environ, {'UTOPIA_SERVER_MANIFEST': json.dumps(manifest)}):
            self.assertEqual(runner.external_attestation(args)['queue_attestation'], manifest)
        path = self.root / 'queue_server.json'
        path.write_text(json.dumps(manifest))
        with mock.patch.dict(os.environ, {'UTOPIA_SERVER_MANIFEST': str(path)}):
            self.assertEqual(runner.external_attestation(args)['queue_attestation_sha256'], runner.sha256(path))

    def test_single_worker_calls_patched_simulator_without_server_ownership(self):
        from tests.support.simulation import FundingAgency, SequentialScriptedLLM, make_simulation, run_phase
        from utopia.models import request_audit
        args = self.args('--run_cell', 'A', '--external_server', '--smoke')
        job = runner.make_jobs(args, self.root)[0]
        module = population_module()
        module.VLLMServerModel = SequentialScriptedLLM
        module.Simulation = native_fixture.Simulation
        module.FundingAgency = FundingAgency
        module.build_influx_cohort = lambda year, seed: []
        setup = {'check_cwd': lambda: None}
        exec('def project_setup():\n    check_cwd()\n', setup)
        module.project_setup = setup['project_setup']
        original_initializer = module.Simulation.initialize_agents
        def main():
            self.assertNotEqual(module.Simulation.initialize_agents, original_initializer)
            self.assertEqual(len(module.build_influx_cohort(1, 42)), 120)
            module.project_setup()  # Worktree directory name is permitted.
            self.assertEqual(Path.cwd(), self.root)
            self.assertEqual(sys.argv[0], 'utopia')
            out = Path(job['output_dir'])
            audit = request_audit.active_audit_for(runner.MODEL)
            self.assertIsNotNone(audit)
            response = types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=20, completion_tokens=10),
                choices=[types.SimpleNamespace(finish_reason='stop')])
            audit.create(lambda **kwargs: response, model=runner.MODEL,
                         messages=[{'role': 'user', 'content': 'CPU fixture'}],
                         temperature=0.7, max_tokens=8192)
            sim, _, agency = make_simulation(out, base=True)
            # This worker fixture uses the real two-agency phase boundary.
            from copy import deepcopy
            sim.ecosystem.agent_population.pop(agency.id)
            agency.id = 'NSF'
            agency.funding_programs = {'NSF_THEORY': next(iter(agency.funding_programs.values()))}
            agency.funding_programs['NSF_THEORY'].program_id = 'NSF_THEORY'
            agency.funding_programs['NSF_THEORY'].name = 'NSF_THEORY'
            agency.funding_programs['NSF_THEORY'].funding_rate = .23
            other = deepcopy(agency)
            other.id = 'DARPA'
            other.funding_programs = {'DARPA_AI_APPS': deepcopy(next(iter(agency.funding_programs.values())))}
            other.funding_programs['DARPA_AI_APPS'].program_id = 'DARPA_AI_APPS'
            other.funding_programs['DARPA_AI_APPS'].name = 'DARPA_AI_APPS'
            other.funding_programs['DARPA_AI_APPS'].funding_rate = .10
            sim.ecosystem.add_agent(agency)
            sim.ecosystem.add_agent(other)
            sim.args.seed = 42
            sim.llm = SequentialScriptedLLM(request_audit=audit)
            from tests.support.simulation import panel_seed_import
            with panel_seed_import():
                for year in range(1, job['num_years'] + 1):
                    sim.funding_tracker.start_cycle(year, sim.ecosystem.agent_population)
                    sim._run_phase_5_update_funding(year, [], {})
            self.finish_real(job, args, audit_fixtures=False, validate=False)
        module.main = main
        with mock.patch.dict(sys.modules, {'utopia.simulation': module}), \
                mock.patch('utopia.simulation', module, create=True), \
                mock.patch.dict(os.environ), \
                mock.patch.object(runner, 'materialize_caches', return_value={}), \
                mock.patch.object(runner, 'validate_caches', return_value={}), \
                mock.patch.object(request_audit, '_load_tokenizer', return_value=FixtureTokenizer()), \
                mock.patch.object(runner, 'wait_for_server') as wait, \
                mock.patch.object(runner, 'gpu_memory', return_value={'0': (0, 81920)}), \
                mock.patch.object(runner, 'spawn', side_effect=AssertionError('server spawn')), \
                mock.patch.object(runner, 'stop_processes', side_effect=AssertionError('server stop')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_single_cell(args, job, {'commit': 'f' * 40}, self.root), 0)
            wait.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'single-attempt'):
                runner.run_single_cell(args, job, {'commit': 'f' * 40}, self.root)
            failed_args = self.args('--run_cell', 'D', '--external_server', '--smoke')
            failed_job = runner.make_jobs(failed_args, self.root)[0]
            def fail_before_science():
                raise BaseException('explicit CPU fatal fixture')
            module.main = fail_before_science
            with self.assertRaisesRegex(BaseException, 'explicit CPU fatal fixture'):
                runner.run_single_cell(failed_args, failed_job, {'commit': 'f' * 40}, self.root)
            failed = json.loads((Path(failed_job['output_dir']) / 'run_manifest.json').read_text())
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['wrapper_protocol'], 'influx1202_v6')
            self.assertFalse(runner.completed(failed_job))
        self.assertIs(module.Simulation.initialize_agents, original_initializer)
        self.assertIsNone(request_audit._ACTIVE)
        self.assertTrue(runner.completed(job))
        ledger = Path(job['output_dir']) / 'funding_applications_year_1.jsonl'
        original = ledger.read_text()
        ledger.write_text(' ' + original)  # Same JSON semantics, changed audit input bytes.
        self.assertFalse(runner.completed(job))
        with self.assertRaisesRegex(ValueError, 'runtime audit evidence differs'):
            runner.validate_results(job, self.root)
        ledger.write_text(original)
        self.assertFalse((self.root / 'outputs/logs/influx2x2_launcher/.lock').exists())

    def test_external_main_never_starts_or_stops_server(self):
        args = self.args('--external_server', '--no_analysis', '--skip_preflight_tests')
        jobs = runner.make_jobs(args, self.root)
        safe = runner.safe_path
        write = runner.write_json
        with contextlib.ExitStack() as stack:
            replacements = {
                'REPO_ROOT': self.root, 'STORAGE_ROOT': None,
                'make_jobs': lambda args: jobs,
                'git_state': lambda: {'common_dir': '/original/.git'},
                'check_output_isolation': lambda: None,
                'materialize_caches': lambda: {}, 'validate_caches': lambda: {},
                'frozen_plan': lambda *args: {'server': runner.server_spec(self.args())},
                'freeze_manifest': lambda plan, git, caches, directory:
                    directory.mkdir(parents=True, exist_ok=True),
                'safe_path': lambda path, root=None: safe(path, self.root),
                'write_json': lambda path, value: write(path, value, self.root),
                'gpu_memory': lambda: {'0': (0, 81920)},
                'wait_for_server': lambda *args: [{'id': runner.MODEL}],
                'run_cells': lambda *args: True,
            }
            for key, value in replacements.items():
                stack.enter_context(mock.patch.object(runner, key, value))
            start = stack.enter_context(mock.patch.object(runner, 'spawn'))
            stop = stack.enter_context(mock.patch.object(runner, 'stop_processes'))
            stack.enter_context(mock.patch.object(
                runner.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**12)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(runner.main(['--external_server', '--no_analysis',
                                          '--skip_preflight_tests']), 0)
            start.assert_not_called()
            stop.assert_not_called()


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


@unittest.skipUnless(os.environ.get('UTOPIA_ACTUAL_INFLUX_TESTS') == '1',
                     'Requires opt-in complete CPU runtime and private corpus/embedding caches')
class TestActualInfluxYears(unittest.TestCase):
    """Real constructors, setup, phases 0..5, RAG, logger and checkpoint serialization."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('scripted preflight requires CUDA_VISIBLE_DEVICES empty before Python starts')
        runner.git_state(runner.REPO_ROOT)
        import pickle
        import torch
        import utopia.simulation as actual
        import utopia.agents.conference as conferences
        from collections import Counter
        from copy import deepcopy
        from tests.support.simulation import SequentialScriptedLLM
        from utopia.funding.sequential import DEFAULT_SYSTEM_PROMPT
        cls.actual, cls.torch, cls.conferences = actual, torch, conferences
        runner.validate_structured_transport(actual)
        cls.pristine_conferences = deepcopy(conferences.CONFERENCES_BY_CATEGORY)

        class ScriptedLLM(SequentialScriptedLLM):
            def generate_batch(self, prompts, seed_ctx=None, response_format=None,
                               system_prompt=DEFAULT_SYSTEM_PROMPT,
                               temperature=0.7, max_tokens=2048, **kwargs):
                kwargs.update(response_format=response_format, system_prompt=system_prompt,
                              temperature=temperature, max_tokens=max_tokens)
                if seed_ctx[0] not in ('phase0_resubmit', 'phase2_submissions'):
                    return super().generate_batch(prompts, seed_ctx, **kwargs)
                import re
                self.calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
                answers = []
                for prompt in prompts:
                    if seed_ctx[0] == 'phase2_submissions':
                        choices = re.findall(r'arXiv ID: ([^\n]+)', prompt)
                        chosen = next((p for p in choices if p not in self.chosen), None)
                        if chosen is None:
                            answers.append(({
                                'id': 0, 'arxiv_id': 'DO_NOT_SUBMIT', 'conference': 'DO_NOT_SUBMIT',
                                'reason': 'Finite scripted candidate pool exhausted; exercise supported abstention.',
                            }, []))
                            continue
                        self.chosen.add(chosen)
                        venue = re.search(r'^- "([^"]+)": primary topics:', prompt, re.M)[1]
                        answers.append(({'id': choices.index(chosen) + 1, 'arxiv_id': chosen,
                                         'conference': venue, 'reason': 'Scripted valid selection.'}, []))
                        continue
                    paper = re.search(r'arXiv ID: "([^"]+)"', prompt)[1]
                    venue = re.search(r'^- "([^"]+)": primary topics:', prompt, re.M)[1]
                    answers.append(({'resubmitted_papers': [{'arxiv_id': paper, 'conference': venue}]}, []))
                return answers
        cls.scripted_llm = ScriptedLLM
        base = runner.REPO_ROOT / 'data/cache/influx_scripted_preflight'
        base.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix='cpu-only-', dir=base)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.fixture_root = Path(cls.temporary.name)
        cls.original_cwd = Path.cwd()
        cls.addClassCleanup(os.chdir, cls.original_cwd)
        output = cls.fixture_root / 'outputs'
        (output / 'thenlper_gte-small').mkdir(parents=True)
        with (runner.REPO_ROOT / runner.SHARED_CACHES[0]).open('rb') as stream:
            corpus = pickle.load(stream)
        embeddings = torch.load(runner.REPO_ROOT / runner.SHARED_CACHES[1],
                                map_location='cpu', weights_only=True)
        selected, counts = [], Counter()
        for index, document in enumerate(corpus['documents']):
            year = document.metadata['publication_year']
            if year in (2016, 2017) and counts[year] < 1024:
                selected.append(index)
                counts[year] += 1
        if any(counts[year] < 512 for year in (2016, 2017)):
            raise RuntimeError('scripted fixture needs at least 512 private corpus documents per year')
        documents = [corpus['documents'][index] for index in selected]
        positions, id2year_pos = Counter(), {}
        for document in documents:
            year = document.metadata['publication_year']
            id2year_pos[document.metadata['id']] = (year, positions[year])
            positions[year] += 1
        with (output / 'cache_scievo.pkl').open('wb') as stream:
            pickle.dump({'documents': documents, 'id2year_pos': id2year_pos}, stream)
        torch.save(embeddings[selected], output / 'thenlper_gte-small/cache_embeddings.pt')
        os.chdir(cls.fixture_root)
        cls.founders = runner.build_founders(actual)
        if torch.cuda.is_initialized():
            raise AssertionError('CUDA was initialized by a CPU-only preflight import')

    def test_full_years_all_four_cells_real_population_and_treatment_paths(self):
        from copy import deepcopy
        initial_hashes = []
        initial_rng_states = []
        for cell in 'ABCD':
            with self.subTest(cell=cell):
                name = f'influx_cpu_fixture_{cell}'
                args = self.actual.parse_arguments([
                    '--experiment_name', name, '--output-dir', str(self.fixture_root / 'outputs'),
                    '--seed', '42', '--num_years', '2', '--num_conferences', '10',
                    '--model', runner.MODEL, '--always_rerun',
                    '--funding_panel_max_apps', '25', '--funding_budget_mode', 'track',
                    '--log_funding_applications',
                    *runner.CELLS[cell]['flags'],
                ])
                args.rag_device = 'cpu'
                self.actual.set_seed(42)
                sim = self.actual.Simulation(
                    llm=self.scripted_llm(), args=args, num_years=2, output_dir=args.output_dir,
                    funding_allocation_mode=args.funding_allocation_mode,
                    industry_funding_mode=args.industry_funding_mode,
                    always_rerun=True, experiment_name=name)
                try:
                    runner.initialize_population(sim, self.actual, self.founders)
                    researchers = [a for a in sim.ecosystem.agent_population.values() if a.can_author]
                    self.assertEqual(len(researchers), 1202)
                    self.assertEqual((sim.initial_university_count, sim.initial_industry_count), (1200, 2))
                    initial_hashes.append(runner.object_hash([
                        {'id': agent.id, 'type': agent.get_type(), 'resources': agent.resources,
                         'can_author': agent.can_author, 'can_review': agent.can_review,
                         'strategy': agent.exploration_strategy, 'reputation': agent.reputation,
                         'expertise': [direction.topic for direction in agent.expertise],
                         'coi': sorted(agent.conflict_of_interest)}
                        for agent in sorted(researchers, key=lambda a: a.id)]))
                    initial_rng_states.append(random.getstate())
                    with mock.patch.object(self.conferences, 'CONFERENCES_BY_CATEGORY',
                                           deepcopy(self.pristine_conferences)):
                        sim.conference_system = self.actual.ConferenceSystem(
                            self.actual.select_conferences_for_simulation(researchers, num_conferences=10))
                    # Preserve all founders in the world and all CoI links. Limit
                    # only the initial active workload; stock influx remains real.
                    active = {r['researcher_name'] for r in self.founders[:10]}
                    active.update(f'institution_0080_researcher_{i}' for i in range(5))
                    active.update(f'Google_researcher_{i}' for i in range(2))
                    for agent in researchers:
                        if agent.id not in active:
                            agent.is_active = False
                            agent.resources = 0
                    for agent in sim.ecosystem.agent_population.values():
                        if getattr(agent, 'resources', None) is not None:
                            sim.agent_tracker.record_agent_resources(agent.id, agent.resources, 0)
                    fixture_job = {'cell': cell, 'output_dir': sim.output_dir, 'num_years': 2}
                    fixture_module = types.SimpleNamespace(
                        FundingAgency=self.actual.FundingAgency, VLLMServerModel=self.scripted_llm,
                        Simulation=self.actual.Simulation)
                    with native_fixture.fixture_request_scope(sim.output_dir) as request_audit, \
                            runner.funding_validation_scope(fixture_module, fixture_job, self.fixture_root), \
                            mock.patch.object(self.actual, 'build_influx_cohort',
                                              lambda year, seed: runner.build_fixed_cohort(self.actual, year, seed)):
                        sim.llm.request_audit = request_audit
                        sim.llm.model_name = runner.MODEL
                        for year in (1, 2):
                            result = sim.run_one_round(year)
                            sim.yearly_results.append(result)
                            sim._save_checkpoint(year, phase=5)
                            expected = runner.expected_researcher_ids(cell, year)
                            observed = {a.id for a in sim.ecosystem.agent_population.values() if a.can_author}
                            self.assertEqual(observed, expected)
                            self.assertEqual(len(sim.ecosystem.agent_population), len(expected) + 2)
                    phases = {call[1][0] for call in sim.llm.calls}
                    self.assertIn('phase3_reviews', phases)
                    self.assertIn('phase5_funding_eval', phases)
                    self.assertEqual('phase0_resubmit' in phases, cell in 'CD')
                    self.assertGreater(len(sim.paper_tracker.papers_by_id), 0)
                    # run_one_round is deliberately called directly to run both
                    # full years; its caller normally writes this lifecycle flag.
                    runner.write_json(Path(sim.output_dir) / 'run_manifest.json',
                                      {'status': 'complete', 'years_completed': 2}, self.fixture_root)
                    # Scripted transport is not Qwen; do not bind it to a
                    # production request scope or label it exact-token evidence.
                    checked = runner.validate_results(fixture_job, self.fixture_root)
                    self.assertEqual(checked['population_by_year']['2']['researchers'],
                                     1442 if cell in 'BD' else 1202)
                    self.assertFalse(self.torch.cuda.is_initialized())
                finally:
                    sim.wandb_logger.finish()
        self.assertEqual(len(set(initial_hashes)), 1)
        self.assertTrue(all(state == initial_rng_states[0] for state in initial_rng_states))


if __name__ == '__main__':
    unittest.main()
