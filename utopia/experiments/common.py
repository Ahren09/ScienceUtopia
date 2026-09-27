"""Public experiment configuration and execution, sharing the native simulator."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import importlib
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import traceback

from utopia.utils.paths import project_root
from utopia.utils.data_utils import file_sha256, json_sha256, write_json_atomic

ROOT = project_root(__file__)


def load_config(family, path=None):
    path = Path(path or ROOT / 'configs' / f'{family}.json')
    config = json.loads(path.read_text())
    if config.get('family') != family or not config.get('cases'):
        raise ValueError(f'Expected a {family} configuration with at least one case')
    for key in ('num_years', 'num_institutions', 'researchers_per_institution', 'num_conferences', 'batch_size'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config.get('seed')) is not int:
        raise ValueError('seed must be an integer')
    return config


def simulation_flags(config, cell):
    family = config['family']
    if cell not in config['cases']:
        raise ValueError(f'Unknown case {cell!r}. Choose from {list(config["cases"])}')
    flags = {key: config[key] for key in (
        'model', 'seed', 'num_years', 'num_institutions', 'researchers_per_institution',
        'num_conferences', 'start_year', 'batch_size', 'initial_funding')}
    flags.update(experiment_name=f'default_{family}', experiment_stage='smoke',
                 population_mode='university_only', strategy_mix='balanced',
                 funding_allocation_mode='fixed', industry_funding_mode='performance',
                 rag_device=config.get('rag_device', 'cpu'), always_rerun=True)
    if family in ('exploration', 'funding_cutoff'):
        flags.update(experiment_name='exploration_vs_exploitation', strategy_mix='1,3,1')
        if config['researchers_per_institution'] != 5:
            raise ValueError('Exploration presets use five researchers and a 1/3/1 strategy mix')
        flags['funding_panel_max_apps'] = 25
    if family in ('scale_expansion', 'project_cost'):
        from utopia.experiments.scale_expansion import CELLS
        spec = CELLS[cell]
        flags.update(experiment_name='scale_expansion', papers_per_project=spec['k'],
                     funding_budget_mode=spec['budget'], funding_panel_max_apps=0,
                     review_policy='standardized' if spec['std'] else 'persona',
                     review_score_mode='float', acceptance_tiebreak='seeded',
                     log_funding_applications=True)
        if spec['budget'] == 'fixed':
            flags['funding_budget_slots_frac'] = config['slots_frac']
        if spec['cap']:
            flags['reviewer_capacity'] = config['reviewer_capacity']
        if spec['slots']:
            flags['acceptance_mode'] = 'fixed_slots'
        if spec.get('noresub'):
            flags['disable_resubmission'] = True
        for key in ('production_cost_mode', 'resubmission_cost', 'log_resource_ledger'):
            if key in spec:
                flags[key] = spec[key]
    if family in ('funding_feedback', 'switching_propensity', 'influx_factorial'):
        flags.update(funding_panel_max_apps=25, log_funding_applications=True)
    if family == 'influx_factorial':
        flags['experiment_name'] = 'influx_factorial'
        flags['population_mode'] = 'default'
        if config['num_institutions'] % 6 or config['researchers_per_institution'] != 5 or config.get('industry_researchers') != 2:
            raise ValueError('Influx requires a multiple of six universities, five researchers each, and two industry founders')
    if family == 'resource_size':
        flags['funding_panel_max_apps'] = 0
    if config.get('model_revision'):
        flags['model_revision'] = config['model_revision']
    flags.update(config['cases'][cell])
    return flags


def prepare_args(config, cell, endpoint, output_root='outputs', cache_dir='data/cache'):
    from utopia.arguments import parse_arguments
    flags = simulation_flags(config, cell)
    argv = []
    for key, value in flags.items():
        if isinstance(value, bool):
            if value:
                argv.append('--' + key)
        elif value is not None:
            argv += ['--' + key, str(value)]
    argv += ['--vllm_url', endpoint, '--data-cache-dir', str(cache_dir)]
    args = parse_arguments(argv, create_dirs=False)
    model_tag = args.model.rsplit('/', 1)[-1].replace('.', '_')
    config_hash = json_sha256(config, sort_keys=True)[:12]
    args.experiment_id = f'{config["family"]}_{model_tag}_{cell}_seed{args.seed}_{config_hash}'
    for field, category in [('log_dir', 'logs'), ('docs_dir', 'docs'), ('checkpoint_dir', 'checkpoints')]:
        setattr(args, field, str(Path(output_root).resolve() / category / args.experiment_id))
    args.output_dir = args.checkpoint_dir
    args.data_cache_dir = str(Path(cache_dir).resolve())
    return args


@contextmanager
def module_settings(module, **values):
    previous = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


def simulation_kwargs(args):
    return {key: getattr(args, key) for key in (
        'verbose', 'debug', 'industry_funding_mode', 'funding_allocation_mode',
        'always_rerun', 'use_langchain', 'max_retries', 'weighted_funding_assignment',
        'experiment_name', 'collaboration_mode', 'collaboration_network_growth_mode', 'start_year')}


def execute(config, cell, args, *, initialize=False, initial_choices=None):
    import utopia.simulation as simulator
    import utopia.funding.feedback as feedback
    from utopia.config import SIMULATION_CONFIG
    from utopia.models.request_audit import request_audit_scope
    from utopia.funding.validation import install_funding_validation
    from utopia.runtime.provenance import mark_failed_run_manifest

    family = config['family']
    docs = Path(args.docs_dir)
    for directory in (args.docs_dir, args.log_dir, args.checkpoint_dir):
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'Use a fresh output root. Existing run: {path}')
    for directory in (args.docs_dir, args.log_dir, args.checkpoint_dir, args.data_cache_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)
    manifest_path = docs / 'experiment_manifest.json'
    manifest = {'status': 'running', 'family': family, 'cell': cell, 'config': config,
                'source_files_sha256': {str(p.relative_to(ROOT)): file_sha256(p)
                                        for p in sorted((ROOT / 'utopia').rglob('*.py'))}}
    write_json_atomic(manifest_path, manifest, indent=2)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.StreamHandler(), logging.FileHandler(Path(args.log_dir) / 'simulation.log')])
    simulator.set_seed(args.seed)
    strict_funding = family in ('funding_feedback', 'influx_factorial', 'switching_propensity')
    funding_handle = None
    try:
        with ExitStack() as stack:
            cls, extra = simulator.Simulation, {}
            if family == 'funding_feedback':
                args.funding_feedback = vars(feedback.Mechanisms.from_cell(cell))
                cls = feedback.simulation_class(cls)
                extra['mechanisms'] = feedback.Mechanisms.from_cell(cell)
            elif family == 'influx_factorial':
                module = importlib.import_module('utopia.experiments.influx_factorial')
                stack.enter_context(module_settings(module, SEED=args.seed,
                    UNIVERSITIES=config['num_institutions'],
                    RESEARCHERS_PER_UNIVERSITY=config['researchers_per_institution'],
                    INDUSTRY_RESEARCHERS=config.get('industry_researchers', 2),
                    FOUNDER_RESEARCHERS=config['num_institutions'] * config['researchers_per_institution'] + config.get('industry_researchers', 2),
                    NUM_YEARS=args.num_years))
                founders = module.build_founders(simulator, args.seed)
                class InfluxSimulation(simulator.Simulation):
                    def initialize_agents(self):
                        return module.initialize_population(self, simulator, founders)
                cls = InfluxSimulation
                stack.enter_context(module_settings(simulator, build_influx_cohort=
                    lambda year, seed: module.build_fixed_cohort(simulator, year, seed)))
            elif family == 'resource_size':
                module = importlib.import_module('utopia.experiments.resource_size')
                stack.enter_context(module_settings(module, SEED=args.seed, NUM_YEARS=args.num_years,
                    EXP_ID=args.experiment_id, NUM_CONFERENCES=args.num_conferences,
                    INSTITUTIONS_PER_TIER=config['institutions_per_tier'],
                    NUM_INSTITUTIONS=3 * config['institutions_per_tier'],
                    TIER_SIZES=config['tier_sizes'], LOW_RESOURCES=config['low_resources'],
                    HIGH_RESOURCES=config['high_resources']))
                cls = module.FactorialSimulation
            elif family == 'switching_propensity':
                return execute_switching(config, cell, args, manifest, manifest_path,
                                         initialize=initialize, initial_choices=initial_choices)
            if strict_funding:
                funding_handle = install_funding_validation(simulator.FundingAgency,
                    docs / 'funding_validation.jsonl', compact=True)
                stack.callback(funding_handle.restore)
                sequential = stack.enter_context(feedback.sequential_funding_scope(
                    simulator.FundingAgency, simulator.VLLMServerModel, docs))
                stack.enter_context(feedback.funding_phase_coverage(
                    simulator.Simulation, simulator.FundingAgency, sequential))
            audit = stack.enter_context(request_audit_scope(docs / 'llm_request_audit.jsonl', model_name=args.model, revision=args.model_revision))
            llm = simulator.VLLMServerModel(model_name=args.model, base_url=args.vllm_url,
                    max_concurrent_requests=args.batch_size, run_seed=args.seed)
            if args.model not in {item.id for item in llm.client.models.list().data}:
                raise ValueError('The endpoint does not advertise the requested model')
            sim = cls(llm=llm, args=args, num_years=args.num_years,
                      output_dir=args.output_dir, **simulation_kwargs(args), **extra)
            sim.run()
            if len(sim.yearly_results) != args.num_years:
                raise RuntimeError('Simulation did not complete all requested years')
        manifest.update(status='complete', years_completed=args.num_years,
                        request_audit=audit.summary(), dataset=getattr(sim.rag, 'dataset_identity', None),
                        ordered_documents_sha256=getattr(sim.rag, 'document_identity', None))
        if strict_funding:
            manifest['funding_validation'] = funding_handle.summary()
            manifest['funding_sequential'] = sequential.summary()
            feedback.validate_sequential_evidence(docs, funding_handle.summary(),
                manifest_summary=sequential.summary(), application_dir=args.output_dir,
                num_years=args.num_years)
    except BaseException:
        manifest.update(status='failed', error=traceback.format_exc())
        mark_failed_run_manifest(args.docs_dir, family, manifest['error'])
        raise
    finally:
        if funding_handle is not None:
            write_json_atomic(docs / 'funding_validation.summary.json', funding_handle.summary(), indent=2)
        write_json_atomic(manifest_path, manifest, indent=2, default=str)
    return manifest


def execute_switching(config, cell, args, manifest, manifest_path, *, initialize, initial_choices):
    import utopia.simulation as simulator
    import utopia.agents.research_direction as directions
    import utopia.agents.switching_policy as policy
    import utopia.experiments.switching_propensity as driver
    import utopia.funding.feedback as feedback
    from utopia.models.request_audit import request_audit_scope
    from utopia.funding.validation import install_funding_validation
    from huggingface_hub import snapshot_download

    if not 2 <= args.num_years <= 10:
        raise ValueError('Switching policies require between two and ten calendar years')
    if not initialize and initial_choices is None:
        raise ValueError('Run --initialize first, then pass its --initial-choices file')
    snapshot = snapshot_download(driver.EMBEDDING_MODEL, revision=driver.EMBEDDING_REVISION,
        allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'])
    identity = {'config': config, 'source_files_sha256': manifest['source_files_sha256'],
                'embedding_revision': driver.EMBEDDING_REVISION,
                'model': args.model, 'seed': args.seed, 'dataset_revision': args.dataset_revision}
    cache = None
    cache_hash = None
    if initial_choices is not None:
        cache = json.loads(Path(initial_choices).read_text())
        cache_hash = file_sha256(initial_choices)
        if (cache.get('status') != 'complete' or cache.get('inputs') != identity
                or cache.get('records_sha256') != json_sha256(cache.get('records'), sort_keys=True)):
            raise ValueError('Initialization is incomplete, modified, or belongs to different inputs')
    docs = Path(args.docs_dir)
    n = args.num_institutions * args.researchers_per_institution
    funding_handle = install_funding_validation(simulator.FundingAgency,
        docs / 'funding_validation.jsonl', compact=True)
    try:
        with ExitStack() as stack:
            stack.callback(funding_handle.restore)
            stack.enter_context(module_settings(driver, SEED=args.seed, FOUNDERS=n, YEARS=args.num_years, MODEL=args.model))
            stack.enter_context(module_settings(policy, SEED=args.seed))
            lookup = stack.enter_context(driver.one_year_context(simulator, directions))
            sequential = stack.enter_context(feedback.sequential_funding_scope(
                simulator.FundingAgency, simulator.VLLMServerModel, docs))
            stack.enter_context(feedback.funding_phase_coverage(
                simulator.Simulation, simulator.FundingAgency, sequential))
            audit = stack.enter_context(request_audit_scope(docs / 'llm_request_audit.jsonl', model_name=args.model, revision=args.model_revision))
            simulator.set_seed(args.seed)
            llm = simulator.VLLMServerModel(model_name=args.model, base_url=args.vllm_url,
                max_concurrent_requests=args.batch_size, run_seed=args.seed)
            sim = driver.simulation_class(simulator.Simulation)(
                llm=llm, args=args, num_years=args.num_years, output_dir=args.output_dir,
                cell=None if initialize else cell, initial_cache=cache, embedding_snapshot=snapshot,
                simulator_module=simulator, direction_lookup=lookup, initialize_only=initialize,
                **simulation_kwargs(args))
            stopped = False
            try:
                sim.run()
            except driver.InitialChoicesComplete:
                if not initialize:
                    raise
                stopped = True
                sim.wandb_logger.finish()
                simulator.write_run_manifest(docs, status='initialization_only_complete',
                    extra={'scientific_years_completed': 0, 'years_completed': 0})
            if stopped != initialize or len(sim.initial_records) != n:
                raise RuntimeError('Initial choice completion boundary failed')
            if not initialize and len(sim.yearly_results) != args.num_years:
                raise RuntimeError('Incomplete switching condition')
        if not funding_handle.summary()['completion_allowed']:
            raise RuntimeError('Funding validation failed')
        if initialize:
            cache = {'status': 'complete', 'inputs': identity, 'founder_ids': list(sim.founder_ids),
                     'pre_choice_state_hash': sim.pre_choice_state_hash,
                     'initial_state_hash': sim.initial_state_hash, 'records': sim.initial_records,
                     'records_sha256': json_sha256(sim.initial_records, sort_keys=True)}
            path = docs / 'initial_choices.json'
            write_json_atomic(path, cache, indent=2)
            path.chmod(0o444)
            manifest['initial_choices'] = str(path)
        else:
            if file_sha256(initial_choices) != cache_hash:
                raise RuntimeError('Initialization changed during the condition')
            feedback.validate_sequential_evidence(docs, funding_handle.summary(),
                manifest_summary=sequential.summary(), application_dir=args.output_dir,
                num_years=args.num_years)
        manifest.update(status='complete', initialization_only=initialize,
                        years_completed=0 if initialize else args.num_years,
                        request_audit=audit.summary(), funding_sequential=sequential.summary(),
                        initial_choices_sha256=cache_hash,
                        dataset=getattr(sim.rag, 'dataset_identity', None),
                        ordered_documents_sha256=getattr(sim.rag, 'document_identity', None))
        return manifest
    finally:
        manifest['funding_validation'] = funding_handle.summary()
        write_json_atomic(docs / 'funding_validation.summary.json', funding_handle.summary(), indent=2)


def main_for(family, argv=None):
    parser = argparse.ArgumentParser(description=f'Run fresh {family.replace("_", " ")} experiments.')
    parser.add_argument('--config', type=Path, help='Public experiment JSON configuration.')
    parser.add_argument('--cell', help='Run one named condition. Default: first configured condition.')
    parser.add_argument('--all-cells', action='store_true', help='Run every configured condition sequentially.')
    parser.add_argument('--vllm-url', '--vllm_url', default='http://localhost:8000/v1')
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--cache-dir', type=Path, default=Path('data/cache'))
    parser.add_argument('--model')
    parser.add_argument('--model-revision')
    parser.add_argument('--rag-device', default=None, help='Retrieval device, e.g. cpu or cuda:0.')
    parser.add_argument('--seed', type=int)
    for name in ('num-years', 'num-institutions', 'num-conferences', 'batch-size'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved settings without model calls or file writes.')
    if family == 'switching_propensity':
        parser.add_argument('--initialize', action='store_true')
        parser.add_argument('--initial-choices', type=Path)
    options = parser.parse_args(argv)
    config = load_config(family, options.config)
    for key in ('model', 'model_revision', 'rag_device', 'seed', 'num_years', 'num_institutions', 'num_conferences', 'batch_size'):
        value = getattr(options, key, None)
        if value is not None:
            config[key] = value
    for key in ('num_years', 'num_institutions', 'num_conferences', 'batch_size'):
        if config[key] < 1:
            parser.error(f'{key} must be positive')
    if family == 'resource_size':
        if config['num_institutions'] % 3:
            parser.error('Resource-size designs require a multiple of three institutions')
        config['institutions_per_tier'] = config['num_institutions'] // 3
    if family == 'resource_size' and any(type(n) is not int or n <= 0 or n % 2 for n in config['tier_sizes'].values()):
        parser.error('Resource-size tiers require positive even institution sizes')
    if getattr(options, 'initialize', False) and options.all_cells:
        parser.error('Initialize once; use --all-cells with the saved --initial-choices file')
    if options.cell and options.all_cells:
        parser.error('--cell and --all-cells are mutually exclusive')
    cells = list(config['cases']) if options.all_cells else [options.cell or next(iter(config['cases']))]
    if options.all_cells and not options.dry_run:
        # Each world starts with pristine module state, including mutable venue catalogs.
        original = list(sys.argv[1:] if argv is None else argv)
        original.remove('--all-cells')
        for cell in cells:
            subprocess.run([sys.executable, '-B', '-m', 'utopia.experiments.' + family,
                            *original, '--cell', cell], check=True)
        return
    for cell in cells:
        args = prepare_args(config, cell, options.vllm_url, options.output_root, options.cache_dir)
        if getattr(options, 'initialize', False):
            args.experiment_id += '_initialization'
            for field, category in [('log_dir', 'logs'), ('docs_dir', 'docs'), ('checkpoint_dir', 'checkpoints')]:
                setattr(args, field, str(options.output_root.resolve() / category / args.experiment_id))
            args.output_dir = args.checkpoint_dir
        if options.dry_run:
            print(json.dumps({'cell': cell, 'config': config, 'simulation': vars(args)}, indent=2))
        else:
            execute(config, cell, args, initialize=getattr(options, 'initialize', False),
                    initial_choices=getattr(options, 'initial_choices', None))
