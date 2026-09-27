import argparse
import os


def parse_arguments(argv=None, *, create_dirs=True):
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
        choices=['smoke', 'pilot', 'scalecheck', 'calibration', 'confirmatory', 'manyworld', 'mechanism', 'funding_cutoff', 'funding_cutoff_cost', 'scale'],
        help='Experiment stage. When set, canonical outputs/{logs,checkpoints,docs}/<experiment_id>/ '
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
             'the funding result is known, for winners and losers alike (funding_cutoff_cost '
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
    parser.add_argument(
        '--annual_influx_rate',
        type=float,
        default=0.0,
        help='Researcher-influx rate for the influx2x2 experiment. 0.0 (default) = OFF, '
             'byte-identical code path to existing runs. Only 0.1 is implemented: the '
             'staggered exact schedule where half of the 60 default-mode institutions '
             'each add 1 junior researcher per year, halves alternating years '
             '(+30/year = 10%% of the initial 300, linear growth). Requires '
             'population_mode=default, non-debug.'
    )
    parser.add_argument(
        '--disable_resubmission',
        action='store_true',
        default=False,
        help='Skip Phase 0 entirely: rejected papers are never resubmitted '
             '(influx2x2 experiment). Default off = existing behavior.'
    )

    # ---- Scale-expansion experiment (all default-off; off = byte-identical legacy path) ----
    parser.add_argument(
        '--papers_per_project',
        type=int,
        default=1,
        help='Papers submitted per completed project (scale experiment factor S). 1 (default) '
             '= legacy single-paper submission. k>1 asks the author to pick up to k distinct '
             'candidate papers in one prompt. Production charges follow --production_cost_mode. '
             'Requires a batching LLM backend (vLLM).'
    )
    parser.add_argument(
        '--production_cost_mode', choices=['per_paper', 'per_project'], default='per_paper',
        help='per_paper preserves the historical submission flow (the configured per-paper '
             'fee is unused in base 2bb7e69). per_project charges that configured fee once '
             'when a completed lead project enters Phase 2, even if it produces zero papers.'
    )
    parser.add_argument(
        '--project_production_cost', type=float, default=None,
        help='Optional finite nonnegative fixed fee for per_project mode. None uses '
             'the existing configured paper fee. 0 permits cost-free production.'
    )
    parser.add_argument(
        '--resubmission_cost', type=float, default=None,
        help='Optional finite nonnegative override of the conference resubmission fee. '
             'Set 0 in BOTH matched k1/k2 cost-control worlds. Positive fees are sensitivity cells.'
    )
    parser.add_argument(
        '--log_resource_ledger', action='store_true',
        help='Persist categorized transactions and all researcher/year balances. Fail on '
             'unexplained residuals. Logging consumes no RNG and requires auditing from year 1.'
    )
    parser.add_argument(
        '--funding_budget_mode',
        type=str,
        default='track',
        choices=['track', 'fixed'],
        help='Agency budget regime (factor R). "track" (default) = legacy: winners per program '
             '= int(n_apps x rate), so agency spending tracks applications. "fixed" = annual '
             'agency slots anchored to the initial university population and split across '
             'programs in proportion to application counts (largest-remainder).'
    )
    parser.add_argument(
        '--funding_budget_slots_frac',
        type=float,
        default=None,
        help='Fraction f of the INITIAL university population funded per year under '
             '--funding_budget_mode fixed (slots = round(f x N0)). Required when mode is fixed.'
    )
    parser.add_argument(
        '--reviewer_capacity',
        type=int,
        default=None,
        help='Per-reviewer annual review cap c (factor K). None (default) = legacy unlimited, '
             '3 random reviewers per paper. When set, reviews per paper = '
             'min(reviews_per_paper, floor(c x N_reviewers / N_submissions)) with a hard floor of 1; '
             'assignment fails fast if any paper would receive zero reviews.'
    )
    parser.add_argument(
        '--reviews_per_paper',
        type=int,
        default=3,
        help='Target reviews per paper (default 3 = legacy).'
    )
    parser.add_argument(
        '--reviewer_matching',
        type=str,
        default='random',
        choices=['random', 'topic'],
        help='Reviewer assignment: "random" (default, legacy) or "topic" (prefer reviewers whose '
             'expertise topics overlap the paper topics; only active with --reviewer_capacity).'
    )
    parser.add_argument(
        '--review_policy',
        type=str,
        default='persona',
        choices=['persona', 'standardized'],
        help='Evaluation scheme (factor E). "persona" (default) = legacy heterogeneous reviewer '
             'prompt (institution persona + memory). "standardized" = one frozen standardized '
             'review policy replaces persona/memory; same paper block, scale, and JSON schema.'
    )
    parser.add_argument(
        '--review_score_mode',
        type=str,
        default='int',
        choices=['int', 'float'],
        help='"int" (default, legacy) truncates the reviewer score with int(); "float" keeps the '
             'decimal score the prompt already allows (e.g. 2.5).'
    )
    parser.add_argument(
        '--acceptance_tiebreak',
        type=str,
        default='stable',
        choices=['stable', 'seeded'],
        help='Tie-breaking among equal mean scores at the acceptance cutoff. "stable" (default, '
             'legacy) keeps submission order; "seeded" uses a deterministic per-(conference, year) '
             'random key derived from --seed.'
    )
    parser.add_argument(
        '--acceptance_mode',
        type=str,
        default='rate',
        choices=['rate', 'fixed_slots'],
        help='"rate" (default, legacy): accept round(n x acceptance_rate) per conference-year. '
             '"fixed_slots": freeze each conference\'s accepted count at its year-1 value, so the '
             'realised acceptance rate falls as submissions grow (venue-capacity sensitivity).'
    )

    parser.add_argument('--model_revision', default=None,
                        help='Immutable model revision for exact request-token accounting.')
    parser.add_argument('--rag_device', default=None,
                        help='Retrieval and embedding device, e.g. cpu or cuda:0.')
    parser.add_argument('--data-cache-dir', default=None,
                        help='Directory for reproducible data and embedding caches.')
    parser.add_argument('--dataset-revision',
                        default='f80fb38a034b2f763dfcbdd2cd187f5b42ddcce6',
                        help='Immutable SciEvo Hugging Face dataset revision.')
    parser.add_argument('--wandb', action='store_true', help='Enable optional scalar W&B logging.')
    args = parser.parse_args(argv)
    for field in ('num_years', 'batch_size', 'num_institutions', 'researchers_per_institution', 'num_conferences'):
        if getattr(args, field) is not None and getattr(args, field) < 1:
            parser.error(f'--{field} must be positive')

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

    if args.annual_influx_rate not in (0.0, 0.1):
        raise ValueError('--annual_influx_rate: only 0.0 (off) and 0.1 (staggered '
                         '+30/year schedule) are implemented')

    if args.papers_per_project < 1:
        raise ValueError('--papers_per_project must be >= 1')
    if args.funding_budget_mode == 'fixed' and not args.funding_budget_slots_frac:
        raise ValueError('--funding_budget_mode fixed requires --funding_budget_slots_frac > 0')
    if args.reviewer_capacity is not None and args.reviewer_capacity < 1:
        raise ValueError('--reviewer_capacity must be >= 1')
    if args.reviews_per_paper < 1:
        raise ValueError('--reviews_per_paper must be >= 1')

    from utopia.funding.accounting import resolve_cost_policy, cost_policy_tag
    from utopia.config import SIMULATION_CONFIG
    cost_tag = cost_policy_tag(resolve_cost_policy(args, SIMULATION_CONFIG))

    if args.experiment_stage:
        # Canonical layout: one experiment_id across outputs/{logs,checkpoints,docs}
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
        if args.strategy_mix == 'balanced':
            # Scale experiment: no exploration-strategy prior for anyone
            args.strategy_mix_list = ['balanced'] * args.researchers_per_institution
            mix_tag = '_mixbal'
        elif args.strategy_mix:
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
        if args.experiment_stage == 'scale':
            # Scale-expansion cells: tag encodes the four design factors (+ sensitivities)
            cell = (f"k{args.papers_per_project}_budget{args.funding_budget_mode}"
                    f"_cap{args.reviewer_capacity if args.reviewer_capacity is not None else 'none'}"
                    f"_{args.review_policy}")
            if args.review_score_mode != 'int':
                cell += '_float'
            if args.acceptance_tiebreak != 'stable':
                cell += '_tieseed'
            if args.acceptance_mode != 'rate':
                cell += '_slots'
            if args.reviewer_matching != 'random':
                cell += '_topicmatch'
            if args.disable_resubmission:
                cell += '_noresub'
            # population tag: synthetic university-only worlds carry i/n; the paper-baseline
            # default population (30 universities + 30 companies x 5) is tagged popdefault
            pop_tag = (f"i{args.num_institutions}_n{n_researchers}" if args.population_mode == 'university_only'
                       else 'popdefault')
            args.experiment_id = (
                f"scale_{model_tag}_{cell}_"
                f"{pop_tag}_y{args.num_years}_seed{args.seed}{mix_tag}"
            )
        else:
            args.experiment_id = (
                f"explore_{args.experiment_stage}_{model_tag}_{mechanism}_"
                f"i{args.num_institutions}_n{n_researchers}_y{args.num_years}_seed{args.seed}{mix_tag}"
            )
        args.experiment_id += cost_tag
        base = args.output_dir  # repository-relative 'outputs'
        args.log_dir = os.path.join(base, 'logs', args.experiment_id)
        args.checkpoint_dir = os.path.join(base, 'checkpoints', args.experiment_id)
        args.docs_dir = os.path.join(base, 'docs', args.experiment_id)
        # Env override lets concurrent runs use disjoint cache dirs (the npz/json
        # caches use non-atomic whole-file flushes and are single-writer only).
        args.data_cache_dir = args.data_cache_dir or os.environ.get('UTOPIA_DATA_CACHE_DIR',
                                             os.path.join('data', 'cache'))
        # Existing runner code writes checkpoints/reports to args.output_dir
        args.output_dir = args.checkpoint_dir
        for d in (args.log_dir, args.checkpoint_dir, args.docs_dir,
                  args.data_cache_dir):
            if create_dirs:
                os.makedirs(d, exist_ok=True)
        return args

    # Default retains the historical layout. Opt-in policies get disjoint paths.
    if args.debug:
        args.output_dir = os.path.join(args.output_dir, 'debug', args.experiment_name + cost_tag)

    elif args.experiment_name:
        args.output_dir = os.path.join(args.output_dir, args.experiment_name + cost_tag)

    else:
        raise ValueError("Please specify the `experiment_name` or enable debug mode.")

    args.log_dir = args.output_dir
    args.docs_dir = args.output_dir
    args.checkpoint_dir = args.output_dir
    # Env override lets concurrent legacy-layout runs use disjoint cache dirs
    # (mirrors the experiment_stage branch above; unset = unchanged behavior).
    args.data_cache_dir = args.data_cache_dir or os.environ.get('UTOPIA_DATA_CACHE_DIR',
                                         os.path.join('data', 'cache'))
    args.experiment_id = args.experiment_name + cost_tag

    if create_dirs:
        os.makedirs(args.output_dir, exist_ok=True)


    return args
