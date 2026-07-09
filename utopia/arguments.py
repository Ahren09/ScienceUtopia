import argparse
import os
import sys


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Academic Society Ecosystem Simulation"
    )
    parser.add_argument(
        '--num_years',
        type=int,
        default=10,
        help='Number of years to simulate (default: 10)'
    )
    parser.add_argument(
        '--start_year',
        type=int,
        default=2016,
        help='Starting real-world year for data partitioning (default: 2016). '
             'Each simulation year maps to a real-world year: Year 1 -> start_year, Year 2 -> start_year+1, etc.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: ./simulation_output)'
    )
    
    

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Verbose mode'
    )
    
    parser.add_argument(
        '--max_retries',
        type=int,
        default=10,
        help='Maximum number of retries for LLM calls (default: 5)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for LLM. OpenAI models (gpt-4o, gpt-4o-mini, etc.) use OpenAI API. '
             'Other models (e.g., Qwen/Qwen3-8B) use vLLM. (default: gpt-4o)'
    )

    # vLLM-specific arguments
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism when using vLLM (default: 1)'
    )
    parser.add_argument(
        '--gpu_memory_utilization',
        type=float,
        default=0.8,
        help='Fraction of GPU memory to use for vLLM (default: 0.8)'
    )
    parser.add_argument(
        '--max_model_len',
        type=int,
        default=8192,
        help='Maximum sequence length for vLLM models (default: 8192)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Maximum batch size for vLLM batch generation (default: 64)'
    )
    parser.add_argument(
        '--vllm_url',
        type=str,
        default=None,
        help='URL of external vLLM server (e.g., http://localhost:8000/v1). '
             'Start server with: vllm serve <model> --gpu-memory-utilization 0.8'
    )

    parser.add_argument(
        '--industry_funding_mode',
        type=str,
        default='performance',
        choices=['performance', 'consistent'],
        help='Funding mode for industry researchers: "performance" (based on papers) or "consistent" (fixed allocation) (default: performance)'
    )

    parser.add_argument(
        '--funding_allocation_mode',
        type=str,
        default='fixed',
        choices=['fixed', 'consumption_based'],
        help='Funding allocation mode: "fixed" (config base budget per winner) or "consumption_based" (derived from consumption history) (default: consumption_based)'
    )

    parser.add_argument(
        '--always_rerun',
        action='store_true',
        default=False,
        help='Ignore all checkpoints and start simulation from scratch (default: resume from latest checkpoint if available)'
    )
    parser.add_argument(
        '--start_phase',
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4, 5],
        help='Start from specific phase (0=resubmissions, 1=directions, 2=submission, 3=review, 4=decisions, 5=funding)'
    )
    parser.add_argument(
        '--use_langchain',
        action='store_true',
        default=False,
        help='Use LangChain for RAG (by default, we use SentenceTransformer)'
    )
    
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='default',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--weighted_funding_assignment',
        action='store_true',
        help='Funding assignment is weighted by the maturity of the paper'
    )
    parser.add_argument(
        '--collaboration_mode',
        type=str,
        default='cross_institute',
        choices=['intra_institute', 'cross_institute'],
        help='Collaboration mode for preferential attachment experiment: '
             '"intra_institute" (same institution only) or "cross_institute" (different institutions only) '
             '(default: cross_institute)'
    )
    parser.add_argument(
        '--collaboration_network_growth_mode',
        type=str,
        default='legacy',
        choices=['legacy', 'expanding'],
        help='Collaboration formation mode for the preferential_attachment experiment. '
             '"legacy" (default) preserves the original behavior exactly; "expanding" assigns each '
             'agent-year a deterministic 70/30 new-tie/repeat-tie proposal channel so the network '
             'grows instead of locking into permanent dyads (prereg_v1_1_source_fix)'
    )

    # Reproducible experiment control (exploration experiment stages)
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Run seed. All other seeds (population, per-request vLLM seeds) are derived from it.'
    )
    parser.add_argument(
        '--experiment_stage',
        type=str,
        default=None,
        choices=['smoke', 'pilot', 'scalecheck', 'calibration', 'confirmatory', 'manyworld', 'mechanism', 'matthew_rd', 'matthew_rd_cost'],
        help='Experiment stage. When set, canonical outputs/{logs,checkpoints,visual,docs}/<experiment_id>/ '
             'paths are used and an experiment_id is derived. When absent, legacy layout is preserved.'
    )
    parser.add_argument(
        '--population_mode',
        type=str,
        default='default',
        choices=['default', 'university_only'],
        help='Population construction: "default" (legacy mixed university+industry) or '
             '"university_only" (balanced equal-funding institutions for the exploration experiment)'
    )
    parser.add_argument(
        '--num_institutions',
        type=int,
        default=None,
        help='Number of institutions in university_only mode (default: config exploration_experiment.num_institutions)'
    )
    parser.add_argument(
        '--researchers_per_institution',
        type=int,
        default=None,
        help='Researchers per institution in university_only mode (default: 5)'
    )
    parser.add_argument(
        '--initial_funding',
        type=float,
        default=None,
        help='Equal initial funding per researcher in university_only mode (default: config value)'
    )
    parser.add_argument(
        '--num_conferences',
        type=int,
        default=None,
        help='Number of conferences to instantiate (default: legacy behavior)'
    )
    parser.add_argument(
        '--acceptance_rate_schedule',
        type=str,
        default=None,
        help='Comma-separated per-year acceptance rates applied to ALL conferences, one per '
             'simulated year (values > 1 are percentages, e.g. "28,26,24.5,..."). Overrides the '
             'fixed config acceptance rate. Default: None (fixed rate for all years).'
    )
    parser.add_argument(
        '--strategy_mix',
        type=str,
        default=None,
        help='Within-institution strategy composition as counts "explorer,exploiter,cautious", '
             'e.g. "1,3,1" (default) or "3,1,1" (explorer-heavy). Must sum to researchers_per_institution. '
             'Used by the E4b composition intervention; appends _mixEXC to the experiment_id.'
    )
    parser.add_argument(
        '--funding_panel_max_apps',
        type=int,
        default=None,
        help='Override funding panel size (config default 25). 0 disables panelization '
             '(global ranking). Used by the panelization fidelity experiment; appends '
             '_panelN to the experiment_id when set.'
    )
    parser.add_argument(
        '--funding_application_cost',
        type=int,
        default=0,
        help='Resource units deducted per GRANT APPLICATION at submission time, before '
             'the funding result is known, for winners and losers alike (matthew_rd_cost '
             'intervention). 0 (default) reproduces existing behavior exactly. Does NOT '
             'affect paper resubmission mechanics.'
    )
    parser.add_argument(
        '--log_funding_applications',
        action='store_true',
        default=False,
        help='Persist the full pre-decision funding ranking (winners AND losers, per panel) '
             'to funding_applications_year_<y>.jsonl in the run output dir. Required by the '
             'Matthew-RD experiment; no effect on funding decisions.'
    )
    parser.add_argument(
        '--funding_intervention',
        action='store_true',
        default=False,
        help='Enable the funding-conservatism intervention (novelty penalty applied BEFORE winner selection). '
             'Off = neutral condition.'
    )
    parser.add_argument(
        '--citation_intervention',
        action='store_true',
        default=False,
        help='Enable the citation risk-return intervention (citation-potential reranking of candidate papers). '
             'Off = neutral condition.'
    )

    args = parser.parse_args(argv)

    # Set default experiment name based on model name
    if args.experiment_name == 'default':
        model_short = args.model.split('/')[-1]
        args.experiment_name = f'default_{model_short}'

    # Parse per-year acceptance-rate schedule into a list of fractions
    if args.acceptance_rate_schedule is not None:
        rates = [float(r) for r in args.acceptance_rate_schedule.split(',')]
        rates = [r / 100 if r > 1 else r for r in rates]
        if len(rates) != args.num_years:
            raise ValueError(
                f'--acceptance_rate_schedule has {len(rates)} entries but --num_years is '
                f'{args.num_years}; provide exactly one rate per simulated year.')
        if not all(0 < r <= 1 for r in rates):
            raise ValueError(f'Acceptance rates must be in (0, 1] after parsing, got {rates}')
        args.acceptance_rate_schedule = rates

    if args.experiment_stage:
        # Canonical layout: one experiment_id across outputs/{logs,checkpoints,visual,docs}
        from utopia.config import SIMULATION_CONFIG
        exp_config = SIMULATION_CONFIG['exploration_experiment']
        if args.num_institutions is None:
            args.num_institutions = exp_config['num_institutions']
        if args.researchers_per_institution is None:
            args.researchers_per_institution = exp_config['researchers_per_institution']
        if args.initial_funding is None:
            args.initial_funding = exp_config['initial_funding']
        if args.num_conferences is None:
            args.num_conferences = exp_config['num_conferences']

        model_tag = args.model.split('/')[-1].lower().replace('-', '_').replace('.', '_')
        if args.funding_intervention or args.citation_intervention:
            mechanism = (f"fund{'015' if args.funding_intervention else '0'}_"
                         f"cite{'on' if args.citation_intervention else 'off'}")
        else:
            mechanism = 'neutral'
        # Composition override (E4b): counts "e,x,c" -> explicit strategy list
        args.strategy_mix_list = None
        mix_tag = ''
        if args.strategy_mix:
            counts = [int(c) for c in args.strategy_mix.split(',')]
            assert len(counts) == 3 and sum(counts) == args.researchers_per_institution, (
                f"--strategy_mix must be 3 counts summing to {args.researchers_per_institution}, "
                f"got {args.strategy_mix}")
            args.strategy_mix_list = (['explorer'] * counts[0] + ['exploiter'] * counts[1]
                                      + ['cautious_explorer'] * counts[2])
            mix_tag = f"_mix{counts[0]}{counts[1]}{counts[2]}"
        if args.funding_panel_max_apps is not None:
            mix_tag += f"_panel{args.funding_panel_max_apps}"
        if args.funding_application_cost:
            mix_tag += f"_cost{args.funding_application_cost}"

        n_researchers = args.num_institutions * args.researchers_per_institution
        args.experiment_id = (
            f"explore_{args.experiment_stage}_{model_tag}_{mechanism}_"
            f"i{args.num_institutions}_n{n_researchers}_y{args.num_years}_seed{args.seed}{mix_tag}"
        )
        base = args.output_dir  # repository-relative 'outputs'
        args.log_dir = os.path.join(base, 'logs', args.experiment_id)
        args.checkpoint_dir = os.path.join(base, 'checkpoints', args.experiment_id)
        args.visual_dir = os.path.join(base, 'visual', args.experiment_id)
        args.docs_dir = os.path.join(base, 'docs', args.experiment_id)
        # Env override lets concurrent runs use disjoint cache dirs (the npz/json
        # caches use non-atomic whole-file flushes and are single-writer only).
        args.data_cache_dir = os.environ.get('UTOPIA_DATA_CACHE_DIR',
                                             os.path.join('data', 'cache'))
        # Existing runner code writes checkpoints/reports to args.output_dir
        args.output_dir = args.checkpoint_dir
        for d in (args.log_dir, args.checkpoint_dir, args.visual_dir, args.docs_dir,
                  args.data_cache_dir):
            os.makedirs(d, exist_ok=True)
        return args

    # Legacy layout (experiment_stage absent): unchanged behavior
    if args.debug:
        args.output_dir = os.path.join(args.output_dir, 'debug', args.experiment_name)

    elif args.experiment_name:
        args.output_dir = os.path.join(args.output_dir, args.experiment_name)

    else:
        raise ValueError("Please specify the `experiment_name` or enable debug mode.")

    args.visual_dir = os.path.join(args.output_dir, 'visual')
    args.log_dir = args.output_dir
    args.docs_dir = args.output_dir
    args.checkpoint_dir = args.output_dir
    args.data_cache_dir = os.path.join('data', 'cache')
    args.experiment_id = args.experiment_name

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.visual_dir, exist_ok=True)


    return args