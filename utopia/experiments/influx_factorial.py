#!/usr/bin/env python
"""Bounded single-world influx x resubmission experiment, isolated from old runs.

Protocol influx1202_v6 permits ONLY seed 42 and four 10-year factorial cells.
Each starts with 240 universities x 5 researchers plus Google x 2 researchers:
1202 researcher founders, plus the original two funding agencies (1204 agents).
The university funding tiers retain the original 1:2 rich/normal ratio. Every
researcher uses the original balanced strategy. Influx adds 120 university
researchers/year and one industry researcher in years 5 and 10: cumulative
floor(0.1 * 1202 * year), always based on researcher founders, not agencies.
This is a new population/composition case, not a paired repeat of n=302.

Qwen3-32B remains pinned to the paper revision, BF16, batch size 64. Defaults
target EC2 GPUs 0-7 / TP=8 / client GPU0 / port 8046; GPU/TP lists are configurable.

From the isolated worktree root (never the original checkout):
    python -m utopia.experiments.influx_factorial --seeds 42 --dry_run
    python -m utopia.experiments.influx_factorial --run_cell A --external_server --server_gpus 0,3
    python -m utopia.experiments.influx_factorial --analysis_only

--external_server [URL] uses an existing server without starting/stopping it.
The CLI serving settings are the operator's declaration of that server's
configuration. An optional --external_server_manifest can additionally attest
the actual launch command. /models cannot verify revision, dtype, TP or memory.

The analysis helper and RAG implementation require legacy outputs/<cell>/ and
two shared cache paths under outputs/. These stay inside this worktree. Linked
RAG cache files are copied locally before any child starts; output directory
symlinks, hard-linked results and unprovenanced existing cells are rejected.
--storage_root explicitly allows outputs/data links to that isolated directory.
HF model weights may be read from the existing hub cache in offline mode;
compilation, dataset, temporary, W&B and other writable caches are redirected.

make_jobs(args) and cell_cmd(...) expose standalone commands to a generic queue.
--run_cell never starts/stops a server and installs the population extension
only in its own process, reusing stock constructors, initialization helpers
and influx injection. No simulator source files are edited. Each cell freezes
its provenance before simulation. --analysis_only validates those records,
allowing different endpoints/GPU IDs but requiring matching scientific inputs
and inference settings. Analysis is conditional on one initialized world,
with no across-seed uncertainty or generalization claim.

V3 follows the v2 infrastructure compatibility failure: vLLM 0.12 ignored
the shared client's legacy guided_json field. Main requested the v2 stop at
2026-09-19 05:11:39 UTC, before any complete world, based on the request-schema
audit rather than outcomes. Preserve incomplete v2 A/B artifacts; v3 uses
fresh paths for ALL four cells and never adopts or combines v2 results.
V3 requires the shared transport fix to structured_outputs.json. Population
and cohort RNG namespaces remain those of v2 so the protocol rename does not
change founders, expertise, or the growth schedule.

V4 follows a failed bounded v3 funding-output probe before any v3 scientific
job started. Its prospective output-elicitation amendment requests only
ranked_application_ids in preference order. The shared compact guard requires
a complete permutation, then uses rank positions and the existing applicant
lookup for native award processing, without inventing reasons. This changes
the requested model output and is not a claim of identical elicitation.

All four v6 cells use a PANELIZED funding baseline with at most 25 applications
per panel and full funding-application logging. The stock track budget regime,
NSF/DARPA rates and max(1, floor(panel_size * rate)) rule remain unchanged.
Splitting and shuffling competitions changes competitors and quota rounding;
this is not the original global-ranking baseline. A strict shared funding
validator must reject incomplete rankings before any awards are processed.
V6 prospectively replaces joint ranking elicitation with sequential next-best
choices from remaining application IDs. Every choice is model-produced; the
complete order still passes the compact final guard before native processing.
This changes the elicitation distribution, not just its serialization. Preserve
all failed/interrupted v5 evidence; no v5 condition completed and none is reused.
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import write_json_atomic

from utopia.runtime.historical import INFLUX_POPULATION_SEED_NAMESPACE

from utopia.runtime.commands import module_command

from utopia.utils.data_utils import file_sha256 as sha256

from utopia.utils.paths import project_root
import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
REPO_ROOT = project_root(__file__)
STORAGE_ROOT = None
SEED = 42
PROTOCOL = 'influx1202_v6'
POPULATION_RNG_NAMESPACE = INFLUX_POPULATION_SEED_NAMESPACE
STRUCTURED_OUTPUT_TRANSPORT = 'structured_outputs.json'
PRIOR_ATTEMPT = {
    'protocol': 'influx1202_v2',
    'stop_requested_at_utc': '2026-09-19T05:11:39Z',
    'failure_class': 'infrastructure_compatibility',
    'reason': 'vLLM 0.12 ignored legacy guided_json; intended decoder schema constraints were absent',
    'decision_basis': 'request-schema/server-warning audit before any complete world; not outcome-based',
    'completed_worlds': 0, 'incomplete_cells': ['A', 'B'],
    'artifact_policy': 'preserve v2 artifacts; fresh v3 A/B/C/D; no adoption or mixing',
}
OUTPUT_ELICITATION_AMENDMENT = {
    'predecessor_protocol': 'influx1202_v3',
    'failure_stage': 'bounded_funding_transport_preflight',
    'reason': 'duplicate, missing or mismatched application IDs despite stop finish and valid token budget',
    'decision_basis': 'prospective output-validity amendment; no v3 scientific job started',
    'scientific_jobs_started': 0,
    'new_wire_field': 'ranked_application_ids',
    'new_output_representation': 'ordered_application_ids_v1',
    'validation': 'complete application-ID permutation before native award processing',
    'legacy_expansion': 'rank from array position; applicant from existing lookup; no invented reason',
    'artifact_policy': 'preserve v1-v4 failed startup/probe evidence; fresh v5 A/B/C/D; no adoption or mixing',
}
SEQUENTIAL_ELICITATION_AMENDMENT = {
    'predecessor_protocol': 'influx1202_v5',
    'failure_stage': 'year2_funding_in_cell_B',
    'reason': 'compact ranking failed complete-permutation validation after three native attempts',
    'decision_basis': 'output validity failure, not comparative scientific outcomes',
    'completed_conditions': 0,
    'interrupted_cells': ['A', 'B'],
    'new_funding_selection_protocol': 'sequential_remaining_ids_v1',
    'distribution_equivalence_claimed': False,
    'artifact_policy': 'preserve v5 partial/failure evidence; fresh matched v6 cells; no adoption or resume',
}
UNIVERSITIES = 240
RESEARCHERS_PER_UNIVERSITY = 5
INDUSTRY_RESEARCHERS = 2
FOUNDER_RESEARCHERS = UNIVERSITIES * RESEARCHERS_PER_UNIVERSITY + INDUSTRY_RESEARCHERS
FUNDING_AGENCIES = 2
MODEL = 'Qwen/Qwen3-32B'
REVISION = '9216db5781bf21249d130ec9da846c4624c16137'
PORT = 8046
SERVER_GPUS = '0,1,2,3,4,5,6,7'
CLIENT_GPU = '0'
NUM_YEARS = 10
BATCH_SIZE = 64
FUNDING_PANEL_MAX_APPS = 25
FUNDING_VALIDATION_POLICY = 'complete_permutation_no_fallback_v1'
FUNDING_OUTPUT_REPRESENTATION = 'ordered_application_ids_v1'
FUNDING_AUDIT_FILE = 'funding_validation_audit.jsonl'
FUNDING_SUMMARY_FILE = 'funding_validation_summary.json'
FUNDING_SELECTION_PROTOCOL = 'sequential_remaining_ids_v1'
SEQUENTIAL_AUDIT_FILE = 'funding_sequential.jsonl'
SEQUENTIAL_SUMMARY_FILE = 'funding_sequential.summary.json'
SEQUENTIAL_PROTOCOL_PATH = 'docs/sequential_funding_protocol.md'
REQUEST_AUDIT_FILE = 'llm_request_audit.jsonl'
REQUEST_SUMMARY_FILE = 'llm_request_audit.summary.json'
SMOKE_CELLS = ('A', 'D')
SMOKE_YEARS = 3
CELLS = {
    'A': {'tag': 'noinflux_noresub', 'flags': ['--disable_resubmission']},
    'B': {'tag': 'influx_noresub',
          'flags': ['--annual_influx_rate', '0.1', '--disable_resubmission']},
    'C': {'tag': 'noinflux_resub', 'flags': []},
    'D': {'tag': 'influx_resub', 'flags': ['--annual_influx_rate', '0.1']},
}
SHARED_CACHES = ('outputs/cache_scievo.pkl',
                 'outputs/thenlper_gte-small/cache_embeddings.pt')
LIMITATION = (
    'These are four counterfactual cells conditional on one initialized world '
    '(seed 42). Researchers, papers and years are not independent seed '
    'replications. No across-seed uncertainty, confidence interval or '
    'significance claim is estimated. Historical n=302 seed42 runs are not '
    'paired controls for this new 1202-researcher population/composition. '
    'TP, batching, concurrency and hardware can change LLM outputs. '
    'All cells use a panelized funding baseline (at most 25 applications per '
    'seeded panel), not the original global-ranking baseline. Stock NSF/DARPA '
    'rates and the track budget regime are retained, but competitors and '
    'per-panel quota rounding change; aggregate awards can differ from a '
    'single global competition. V4 prospectively changes funding output '
    'elicitation to a complete ordered application-ID array after a failed '
    'v3 transport probe, before any v3 scientific job started. V6 uses sequential '
    'next-best remaining-ID choices after the v5 ranking-validity failure. This '
    'changes joint elicitation, retry granularity and prefix conditioning; no '
    'distributional equivalence or cross-version/family baseline pooling is claimed.')


def funding_spec():
    """Common scientific condition; panelization is not a performance-only change."""
    return {
        'baseline': 'panelized', 'panel_max_apps': FUNDING_PANEL_MAX_APPS,
        'budget_mode': 'track', 'program_rates': {'NSF': 0.23, 'DARPA': 0.10},
        'quota_rule': 'max(1, floor(panel_size * program_rate))',
        'log_funding_applications': True,
        'validation_policy': FUNDING_VALIDATION_POLICY,
        'output_representation': FUNDING_OUTPUT_REPRESENTATION,
        'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
        'global_ranking_equivalent': False,
    }


def sequential_spec():
    return {
        'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
        'audit_file': SEQUENTIAL_AUDIT_FILE, 'summary_file': SEQUENTIAL_SUMMARY_FILE,
        'protocol_document': SEQUENTIAL_PROTOCOL_PATH,
        'max_sdk_calls_per_step': 3, 'accepted_prefix_reset': False,
        'all_choices_from_remaining_ids': True,
        'seed_context': ['phase5_funding_eval', 'year', FUNDING_SELECTION_PROTOCOL, 'production',
                         'program_id', 'panel_index', 'step'],
        'step_index_base': 0, 'sdk_attempts': [0, 1, 2],
    }


def request_audit_spec():
    return {'required': True, 'model': MODEL, 'revision': REVISION,
            'max_model_len': 32768, 'budget_rule': 'exact_input_tokens + actual_max_tokens <= 32768',
            'n_clients': 1, 'guard_failures': 0}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seeds', type=int, nargs='+', default=[SEED])
    ap.add_argument('--funding_panel_max_apps', type=int, choices=[FUNDING_PANEL_MAX_APPS],
                    default=FUNDING_PANEL_MAX_APPS,
                    help='frozen common funding condition: 25-app panels in all four cells')
    ap.add_argument('--run_cell', choices=tuple(CELLS),
                    help='run only this cell against an external server, for a generic queue')
    ap.add_argument('--server_gpus', default=SERVER_GPUS)
    ap.add_argument('--client_gpu', default=CLIENT_GPU)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--tensor_parallel_size', type=int)
    ap.add_argument('--external_server', nargs='?', const='', default=None,
                    metavar='URL', help='default URL: http://localhost:<port>/v1')
    ap.add_argument('--external_server_manifest', type=Path)
    ap.add_argument('--storage_root', '--private_root', dest='storage_root', type=Path,
                    help='explicit isolated target for worktree outputs/data symlinks')
    ap.add_argument('--max_num_seqs', type=int, default=128)
    ap.add_argument('--max_concurrent_cells', type=int, default=4)
    ap.add_argument('--gpu_memory_utilization', type=float, default=0.80)
    ap.add_argument('--client_headroom_gb', type=float, default=8.0,
                    help='minimum free GiB for the entire concurrent client pool')
    ap.add_argument('--timeout_hours', type=float, default=72.0)
    for flag in ('preflight_only', 'smoke', 'skip_preflight_tests', 'no_analysis',
                 'check_smoke_only', 'dry_run', 'analysis_only'):
        ap.add_argument('--' + flag, action='store_true')
    args = ap.parse_args(argv)
    gpu_ids = args.server_gpus.split(',')
    if (not all(g.isdigit() for g in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids)
            or not args.client_gpu.isdigit()):
        ap.error('GPU IDs must be nonnegative integers; server IDs must be unique')
    args.server_gpus = ','.join(str(int(g)) for g in gpu_ids)
    if len(set(args.server_gpus.split(','))) != len(gpu_ids):
        ap.error('server GPU IDs must be unique')
    args.client_gpu = str(int(args.client_gpu))
    if args.tensor_parallel_size is None:
        args.tensor_parallel_size = len(gpu_ids)
    if args.tensor_parallel_size != len(gpu_ids):
        ap.error('--tensor_parallel_size must equal the number of --server_gpus')
    if (args.tensor_parallel_size < 1 or 64 % args.tensor_parallel_size
            or 8 % args.tensor_parallel_size):
        ap.error('Qwen3-32B TP must divide its 64 attention and 8 KV heads')
    if args.seeds != [SEED]:
        ap.error(f'protocol {PROTOCOL} permits exactly --seeds 42; multiple seeds are disabled')
    args.requested_seeds = list(args.seeds)
    if not 1 <= args.port <= 65535 or min(args.max_concurrent_cells, args.max_num_seqs) < 1:
        ap.error('port must be in [1,65535]; concurrency and max_num_seqs must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        ap.error('--gpu_memory_utilization must be in (0,1)')
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.client_headroom_gb, args.timeout_hours)):
        ap.error('headroom and timeout must be finite and positive')
    args.server_url = f'http://localhost:{args.port}/v1'
    if args.external_server is not None:
        args.server_url = (args.external_server or args.server_url).rstrip('/')
        url = urllib.parse.urlsplit(args.server_url)
        if (url.scheme not in ('http', 'https') or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or url.path != '/v1'):
            ap.error('--external_server must be an http(s) base URL ending in /v1')
    elif args.external_server_manifest:
        ap.error('--external_server_manifest requires --external_server')
    if args.smoke and args.analysis_only:
        ap.error('--analysis_only applies to the full 2x2, not smoke cells')
    if args.run_cell and args.external_server is None:
        ap.error('--run_cell requires --external_server; the queue owns the server')
    if args.run_cell and (args.analysis_only or args.preflight_only or args.check_smoke_only):
        ap.error('--run_cell cannot be combined with analysis/preflight/check-only modes')
    return args


def experiment_id(args):
    mode = 'smoke' if args.smoke or args.check_smoke_only else 'full'
    return f'influx2x2_{PROTOCOL}_{mode}_seed42'


def cell_name(key, seed, smoke=False):
    if seed != SEED:
        raise ValueError('only seed42 is allowed')
    tag = 'smoke_' + key if smoke else key + '_' + CELLS[key]['tag']
    return f'influx2x2_{PROTOCOL}_{tag}_seed{seed}'


def simulation_cmd(cell_key, name, num_years, seed=SEED, server_url=None):
    """Stock simulator arguments; the worker installs the new population first."""
    if seed != SEED or num_years not in (SMOKE_YEARS, NUM_YEARS):
        raise ValueError('only seed42 and the frozen smoke/full horizons are supported')
    return module_command(sys.executable, 'utopia', '--model', MODEL, '--vllm_url', server_url or f'http://localhost:{PORT}/v1', '--seed', str(seed), '--num_years', str(num_years), '--batch_size', str(BATCH_SIZE), '--experiment_name', name, '--output-dir', 'outputs', '--funding_panel_max_apps', str(FUNDING_PANEL_MAX_APPS), '--funding_budget_mode', 'track', '--log_funding_applications') + CELLS[cell_key]['flags']


def cell_cmd(cell_key, name, num_years, seed=SEED, server_url=None, *,
             server_gpus=SERVER_GPUS, client_gpu=CLIENT_GPU, tensor_parallel_size=None,
             gpu_memory_utilization=0.80, client_headroom_gb=8.0, max_num_seqs=128,
             timeout_hours=72.0, storage_root=None, external_server_manifest=None):
    """A standalone queue command. Different cells may target different TP2 servers."""
    if seed != SEED or num_years not in (SMOKE_YEARS, NUM_YEARS):
        raise ValueError('only seed42 and the frozen smoke/full horizons are supported')
    if name != cell_name(cell_key, seed, num_years == SMOKE_YEARS):
        raise ValueError(f'cell name must use the fresh {PROTOCOL} protocol')
    command = module_command(sys.executable, 'utopia.experiments.influx_factorial', '--run_cell', cell_key, '--external_server', server_url or f'http://localhost:{PORT}/v1', '--seeds', str(seed), '--server_gpus', server_gpus, '--funding_panel_max_apps', str(FUNDING_PANEL_MAX_APPS), '--client_gpu', client_gpu, '--gpu_memory_utilization', str(gpu_memory_utilization), '--client_headroom_gb', str(client_headroom_gb), '--max_num_seqs', str(max_num_seqs), '--timeout_hours', str(timeout_hours))
    if tensor_parallel_size is not None:
        command += ['--tensor_parallel_size', str(tensor_parallel_size)]
    if num_years == SMOKE_YEARS:
        command.append('--smoke')
    if storage_root:
        command += ['--storage_root', str(storage_root)]
    if external_server_manifest:
        command += ['--external_server_manifest', str(external_server_manifest)]
    return command


def runtime_env(args, label, root=REPO_ROOT):
    """Separate writable caches; existing HF weights are read offline."""
    base = Path(root) / 'data' / 'cache' / 'influx2x2' / label
    hub = os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
    if not hub:
        hub = str(Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub')
    env = {
        'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONHASHSEED': str(SEED),
        'HF_HUB_CACHE': str(Path(hub).expanduser().resolve()),
        'HUGGINGFACE_HUB_CACHE': str(Path(hub).expanduser().resolve()),
        'SENTENCE_TRANSFORMERS_HOME': str(Path(hub).expanduser().resolve()),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
        'WANDB_MODE': 'disabled', 'WANDB_DISABLED': 'true',
    }
    for key, suffix in {
        'UTOPIA_DATA_CACHE_DIR': 'simulation', 'HF_HOME': 'hf',
        'HF_DATASETS_CACHE': 'datasets', 'HF_ASSETS_CACHE': 'assets',
        'XDG_CACHE_HOME': 'xdg', 'TORCH_HOME': 'torch',
        'TORCH_EXTENSIONS_DIR': 'torch_extensions', 'NUMBA_CACHE_DIR': 'numba',
        'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TRITON_CACHE_DIR': 'triton',
        'CUDA_CACHE_PATH': 'cuda', 'VLLM_CACHE_ROOT': 'vllm',
        'VLLM_CONFIG_ROOT': 'vllm_config', 'HF_MODULES_CACHE': 'hf_modules',
        'MPLCONFIGDIR': 'matplotlib', 'WANDB_DIR': 'wandb',
        'WANDB_CACHE_DIR': 'wandb_cache', 'WANDB_CONFIG_DIR': 'wandb_config',
        'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp', 'RAY_TMPDIR': 'ray',
    }.items():
        env[key] = str(base / suffix)
    env['CUDA_VISIBLE_DEVICES'] = args.server_gpus if label == 'server' else args.client_gpu
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    return env


def make_jobs(args, root=REPO_ROOT):
    jobs = []
    for seed in args.seeds:
        for key in ([args.run_cell] if args.run_cell else SMOKE_CELLS if args.smoke else CELLS):
            name = cell_name(key, seed, args.smoke)
            years = SMOKE_YEARS if args.smoke else NUM_YEARS
            jobs.append({
                'id': f'{seed}_{key}', 'cell': key, 'seed': seed,
                'experiment_name': name, 'num_years': years,
                'command': cell_cmd(
                    key, name, years, seed, args.server_url,
                    server_gpus=args.server_gpus, client_gpu=args.client_gpu,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    client_headroom_gb=args.client_headroom_gb, max_num_seqs=args.max_num_seqs,
                    timeout_hours=args.timeout_hours, storage_root=args.storage_root,
                    external_server_manifest=args.external_server_manifest),
                'simulation_command': simulation_cmd(key, name, years, seed, args.server_url),
                'protocol': PROTOCOL, 'population': population_spec(), 'funding': funding_spec(),
                'env_overrides': runtime_env(args, name, root),
                'output_dir': str(Path(root) / 'outputs' / name),
                'completion_gates': {
                    'influx_result.json': {'status': 'complete', 'protocol': PROTOCOL,
                                          'seed': SEED, 'cell': key, 'num_years': years,
                                          'founder_researchers': 1202, 'funding_agencies': 2,
                                          'funding': funding_spec(),
                                          'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
                                          'sequential_funding_status': 'complete',
                                          'funding_application_evidence_valid': True,
                                          'funding_validation_status': 'complete',
                                          'request_audit_status': 'complete',
                                          'checked_years': list(range(1, years + 1))},
                    'run_manifest.json': {'status': 'complete', 'years_completed': years},
                    FUNDING_SUMMARY_FILE: funding_summary_gates(),
                    SEQUENTIAL_SUMMARY_FILE: {**sequential_summary_gates(),
                                             'years_started': years, 'years_completed': years,
                                             'agency_registrations': 2 * years},
                    REQUEST_SUMMARY_FILE: request_summary_gates(),
                    f'checkpoint_year_{years}.json': {'phase': 5, 'year': years,
                                                     'initial_university_count': 1200,
                                                     'initial_industry_count': 2},
                },
            })
    return jobs


def queue_jobs(worktree, private_root):
    """Four fresh jobs in Fermat's queue schema; no files, subprocesses or GPUs."""
    from utopia.funding.feedback import sequential_source_binding
    worktree, private_root = Path(worktree), Path(private_root)
    if not worktree.is_absolute() or not private_root.is_absolute():
        raise ValueError('queue worktree and private_root must be absolute')
    jobs = []
    for key in CELLS:
        name = cell_name(key, SEED)
        out = private_root / 'outputs' / name
        command = cell_cmd(
            key, name, NUM_YEARS, server_url='__SERVER_URL__',
            server_gpus='__SERVER_GPUS__', client_gpu='__CLIENT_GPU__',
            tensor_parallel_size=2, storage_root=private_root,
            external_server_manifest='__SERVER_MANIFEST__')
        command[0] = str(private_root / 'env/bin/python')
        result_fields = {
            '/status': 'complete', '/protocol': PROTOCOL, '/seed': SEED, '/cell': key,
            '/num_years': NUM_YEARS, '/founder_researchers': FOUNDER_RESEARCHERS,
            '/funding_agencies': FUNDING_AGENCIES, '/checked_years': list(range(1, 11)),
            '/funding': funding_spec(),
            '/funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
            '/sequential_funding_status': 'complete',
            '/funding_application_evidence_valid': True,
            '/funding_validation_status': 'complete', '/request_audit_status': 'complete',
        }
        artifacts = []
        for year in range(1, NUM_YEARS + 1):
            researchers = len(expected_researcher_ids(key, year))
            result_fields[f'/population_by_year/{year}/researchers'] = researchers
            result_fields[f'/population_by_year/{year}/all_agents'] = researchers + FUNDING_AGENCIES
            artifacts.append({
                'path': str(out / f'checkpoint_year_{year}.json'), 'min_bytes': 1,
                'equals': {'/year': year, '/phase': 5, '/initial_university_count': 1200,
                           '/initial_industry_count': 2},
            })
        artifacts += [
            {'path': str(out / 'influx_result.json'), 'min_bytes': 1, 'equals': result_fields},
            {'path': str(out / 'run_manifest.json'), 'min_bytes': 1,
             'equals': {'/status': 'complete', '/years_completed': NUM_YEARS}},
            {'path': str(out / 'influx_protocol.json'), 'min_bytes': 1,
             'equals': {'/common/protocol': PROTOCOL, '/common/seed': SEED, '/cell': key,
                        '/num_years': NUM_YEARS, '/common/population/founder_researchers': 1202,
                        '/common/population/total_initial_agents': 1204,
                        '/common/population_rng_namespace': POPULATION_RNG_NAMESPACE,
                        '/common/structured_output_transport': STRUCTURED_OUTPUT_TRANSPORT,
                        '/common/funding': funding_spec(),
                        '/common/funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
                        '/common/sequential_funding': sequential_spec(),
                        '/common/sequential_elicitation_amendment': SEQUENTIAL_ELICITATION_AMENDMENT,
                        '/common/sequential_source_binding': sequential_source_binding(REPO_ROOT),
                        '/common/output_elicitation_amendment': OUTPUT_ELICITATION_AMENDMENT,
                        '/common/request_audit': request_audit_spec(),
                        '/common/inference/tensor_parallel_size': 2,
                        '/common/inference/revision': REVISION,
                        '/common/inference/dtype': 'bfloat16'}},
            {'path': str(out / 'founder_blueprint.json'), 'min_bytes': 1},
            {'path': str(out / FUNDING_AUDIT_FILE), 'min_bytes': 0},
            {'path': str(out / FUNDING_SUMMARY_FILE), 'min_bytes': 1,
             'equals': {'/' + key: value for key, value in funding_summary_gates().items()}},
            {'path': str(out / SEQUENTIAL_AUDIT_FILE), 'min_bytes': 1},
            {'path': str(out / SEQUENTIAL_SUMMARY_FILE), 'min_bytes': 1,
             'equals': {**{'/' + key: value for key, value in sequential_summary_gates().items()},
                        '/years_started': NUM_YEARS, '/years_completed': NUM_YEARS,
                        '/agency_registrations': 2 * NUM_YEARS}},
            {'path': str(out / REQUEST_AUDIT_FILE), 'min_bytes': 1},
            {'path': str(out / REQUEST_SUMMARY_FILE), 'min_bytes': 1,
             'equals': {'/' + key: value for key, value in request_summary_gates().items()}},
        ]
        jobs.append({
            'id': f'influx_{PROTOCOL}_{key}_seed42', 'family': 'influx', 'seed': SEED,
            'command': command, 'env': {'PYTHONHASHSEED': '42'},
            'require_manifest_argument': True,
            'output_roots': [str(out), str(private_root / 'outputs/logs' / PROTOCOL / name)],
            'artifacts': artifacts,
        })
    return jobs


def population_spec():
    return {
        'founder_researchers': FOUNDER_RESEARCHERS,
        'university_researchers': UNIVERSITIES * RESEARCHERS_PER_UNIVERSITY,
        'universities': UNIVERSITIES, 'researchers_per_university': RESEARCHERS_PER_UNIVERSITY,
        'industry_researchers': INDUSTRY_RESEARCHERS, 'company': 'Google',
        'funding_agencies': FUNDING_AGENCIES, 'total_initial_agents': 1204,
        'university_funding': {'rich_institutions': 80, 'rich': 160, 'normal': 80},
        'industry_funding': 320, 'strategy': 'balanced',
        'growth_denominator': '1202 founder researchers; excludes funding agencies',
        'growth_schedule': '120 university entrants/year; 1 Google entrant in years 5 and 10',
        'cumulative_entrants': 'floor(1202 * year / 10)',
        'population_comparability': 'New population AND composition; not paired with historical n302',
    }


def build_founders(simulator, seed=SEED):
    """Pure blueprint, using the existing seeded university blueprint generator."""
    if seed != SEED:
        raise ValueError('only seed42 is allowed')
    rows = simulator.build_population_blueprint(
        UNIVERSITIES, RESEARCHERS_PER_UNIVERSITY, seed,
        strategy_mix=['balanced'] * RESEARCHERS_PER_UNIVERSITY)
    founders = []
    for index, row in enumerate(rows):
        founders.append({**row, 'kind': 'university',
                         'funding_level': (simulator.RICH_UNIVERSITY_FUNDING
                                           if index // RESEARCHERS_PER_UNIVERSITY < 80
                                           else simulator.NORMAL_UNIVERSITY_FUNDING)})
    if simulator.RICH_COMPANIES[0] != 'Google':
        raise ValueError('original rich company roster changed')
    for index in range(INDUSTRY_RESEARCHERS):
        rng = random.Random(simulator.derive_seed(seed, POPULATION_RNG_NAMESPACE, 'industry_founder', index))
        expertise = rng.sample(range(len(simulator.AVAILABLE_DIRECTIONS)), 3)
        founders.append({
            'institution': 'Google', 'researcher_name': f'Google_researcher_{index}',
            'kind': 'industry', 'strategy': 'balanced',
            'funding_level': simulator.RICH_COMPANY_FUNDING,
            'expertise_indices': expertise,
            'expertise_topics': [simulator.AVAILABLE_DIRECTIONS[i].topic for i in expertise],
        })
    if (len(founders) != FOUNDER_RESEARCHERS
            or {r['funding_level'] for r in founders if r['kind'] == 'university'} != {80, 160}
            or any(r['funding_level'] != 320 for r in founders if r['kind'] == 'industry')):
        raise ValueError('founder population or original funding constants changed')
    return founders


def build_fixed_cohort(simulator, year, seed=SEED):
    """Staggered, tier-balanced growth anchored to researchers, with no global RNG use."""
    if seed != SEED or not 1 <= year <= NUM_YEARS:
        raise ValueError('cohort requires seed42 and years 1..10')
    cohort = []
    for index in range(UNIVERSITIES):
        if index % 2 != (year - 1) % 2:
            continue
        cohort.append({
            'institution': f'institution_{index:04d}', 'kind': 'university',
            'funding_level': (simulator.RICH_UNIVERSITY_FUNDING if index < 80
                              else simulator.NORMAL_UNIVERSITY_FUNDING),
            'researcher_name': f'institution_{index:04d}_researcher_{5 + (year - 1) // 2}',
        })
    if year % 5 == 0:
        cohort.append({
            'institution': 'Google', 'kind': 'industry',
            'funding_level': simulator.RICH_COMPANY_FUNDING,
            'researcher_name': f'Google_researcher_{1 + year // 5}',
        })
    for row in cohort:
        rng = random.Random(simulator.derive_seed(seed, POPULATION_RNG_NAMESPACE, 'influx', year, row['institution']))
        row['expertise_indices'] = rng.sample(range(len(simulator.AVAILABLE_DIRECTIONS)), 3)
    return cohort


def initialize_population(simulation, simulator, founders):
    """Stock researcher constructors, stock funding agencies and stock N0 anchors."""
    if (getattr(simulation.args, 'seed', None) != SEED or simulation.debug
            or getattr(simulation.args, 'population_mode', 'default') != 'default'):
        raise ValueError(f'{PROTOCOL} requires seed42, non-debug default treatment mechanics')
    by_institution = {}
    for row in founders:
        kwargs = {
            'researcher_name': row['researcher_name'], 'funding_level': row['funding_level'],
            'expertise': [simulator.AVAILABLE_DIRECTIONS[i] for i in row['expertise_indices']],
            'llm': simulation.llm, 'exploration_strategy': 'balanced',
        }
        if row['kind'] == 'university':
            agent = simulator.UniversityResearcher(
                **kwargs, university_name=row['institution'], generate_research_proposal=True)
        else:
            agent = simulator.IndustryResearcher(
                **kwargs, company_name=row['institution'], funding_mode=simulation.industry_funding_mode)
        simulation.ecosystem.add_agent(agent)
        by_institution.setdefault(row['institution'], set()).add(agent.id)
    for names in by_institution.values():
        for name in names:
            simulation.ecosystem.get_agent_by_id(name).add_conflict_of_interest(names - {name})
    simulation._record_initial_university_count()
    simulation._add_funding_agencies()
    if (simulation.initial_university_count != 1200 or simulation.initial_industry_count != 2
            or len(simulation.ecosystem.agent_population) != 1204):
        raise ValueError('initialized population differs from frozen 1202-researcher protocol')


def server_spec(args):
    return {
        'model': MODEL, 'revision': REVISION, 'dtype': 'bfloat16',
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_model_len': 32768, 'gpu_memory_utilization': args.gpu_memory_utilization,
        'seed': SEED, 'reasoning_parser': 'qwen3', 'server_gpus': args.server_gpus,
        'max_num_seqs': args.max_num_seqs,
    }


def server_cmd(args):
    return ['vllm', 'serve', MODEL, '--revision', REVISION,
            '--dtype', 'bfloat16', '--tensor-parallel-size', str(args.tensor_parallel_size),
            '--max-model-len', '32768',
            '--gpu-memory-utilization', str(args.gpu_memory_utilization),
            '--max-num-seqs', str(args.max_num_seqs),
            '--port', str(args.port), '--seed', str(SEED), '--reasoning-parser', 'qwen3']





def safe_path(path, root=REPO_ROOT):
    """A writable path must resolve inside this worktree, without hard links."""
    path, root = Path(path).absolute(), Path(root).resolve()
    allowed = [root]
    storage = Path(STORAGE_ROOT).resolve() if STORAGE_ROOT is not None else None
    if storage is not None:
        allowed += [storage / 'outputs', storage / 'data']
    if not any(path.resolve().is_relative_to(parent) for parent in allowed):
        raise ValueError(f'writable path escapes isolated worktree: {path}')
    storage_link = (storage is not None and path.parent == root
                    and path.name in ('outputs', 'data')
                    and path.resolve() == storage / path.name)
    if path.is_symlink() and not storage_link:
        raise ValueError(f'writable path is a symlink: {path}')
    if path.is_file() and path.stat().st_nlink > 1:
        raise ValueError(f'writable file has hard links: {path}')
    return path


def check_output_isolation(root=REPO_ROOT):
    root = Path(root).resolve()
    safe_path(root / 'data', root)
    for directory in (root / 'outputs', root / 'data' / 'cache' / 'influx2x2'):
        safe_path(directory, root)
        if not directory.exists():
            continue
        for parent, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs + files:
                path = Path(parent) / name
                if str(path.relative_to(root)) not in SHARED_CACHES:
                    safe_path(path, root)


def git_state(root=REPO_ROOT):
    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=root, text=True,
                                       stderr=subprocess.PIPE).strip()
    try:
        state = {
            'commit': git('rev-parse', 'HEAD'),
            'branch': git('branch', '--show-current'),
            'dirty_files': git('status', '--short').splitlines(),
            'root': str(Path(git('rev-parse', '--show-toplevel')).resolve()),
            'git_dir': str((Path(root) / git('rev-parse', '--git-dir')).resolve()),
            'common_dir': str((Path(root) / git('rev-parse', '--git-common-dir')).resolve()),
        }
    except subprocess.CalledProcessError as exc:
        raise ValueError('launch requires an isolated Git worktree; staging supports --dry_run') from exc
    if state['branch'] in ('', 'main', 'master'):
        raise ValueError('launch requires a named non-main branch')
    if state['git_dir'] == state['common_dir'] or state['root'] != str(Path(root).resolve()):
        raise ValueError('launch requires a linked worktree, not the original checkout')
    return state


def materialize_caches(root=REPO_ROOT):
    """Copy linked cache inputs, replacing only the local directory entry."""
    root = Path(root).resolve()
    records = {}
    for rel in SHARED_CACHES:
        path = root / rel
        safe_path(path.parent, root)
        if not path.is_file():
            raise ValueError(f'missing shared cache: {path}')
        origin = str(path.resolve())
        if path.is_symlink() or path.stat().st_nlink > 1:
            temp = safe_path(path.with_name(path.name + f'.copy-{os.getpid()}'), root)
            try:
                with path.open('rb') as source, temp.open('xb') as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()
        safe_path(path, root)
        records[rel] = {'sha256': sha256(path), 'bytes': path.stat().st_size,
                        'copied_from': origin if origin != str(path) else None}
    return records


def validate_caches(root=REPO_ROOT):
    """CPU-only validation prevents shared-cache regeneration races."""
    import pickle
    import torch
    with (Path(root) / SHARED_CACHES[0]).open('rb') as stream:
        cache = pickle.load(stream)
    docs = cache.get('documents', [])
    if not docs or 'tags' not in docs[0].metadata or 'id2year_pos' not in cache:
        raise ValueError('stale document cache; prepare a valid isolated cache before launch')
    embeddings = torch.load(Path(root) / SHARED_CACHES[1],
                            map_location='cpu', weights_only=True)
    if embeddings.shape[0] != len(docs) or embeddings.dtype != torch.float16:
        raise ValueError('embedding cache must match document count and paper GPU fp16 dtype')
    return {'documents': len(docs), 'embedding_shape': list(embeddings.shape),
            'embedding_dtype': str(embeddings.dtype),
            'dataset': 'Ahren09/SciEvo', 'configuration': 'arxiv', 'split': 'train'}


def gpu_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
         '--format=csv,noheader,nounits'], text=True)
    return {str(int(i.strip())): (int(used.strip()), int(total.strip()))
            for i, used, total in (line.split(',') for line in output.strip().splitlines())}


def check_gpu_headroom(args, memory, server_running=False):
    required = {args.client_gpu: args.client_headroom_gb * 1024}
    if args.external_server is None and not server_running:
        for gpu in args.server_gpus.split(','):
            if gpu not in memory:
                raise ValueError(f'server GPU {gpu} is unavailable')
            required[gpu] = required.get(gpu, 0) + args.gpu_memory_utilization * memory[gpu][1]
    for gpu, reserve in required.items():
        if gpu not in memory or memory[gpu][1] - memory[gpu][0] < reserve:
            raise ValueError(f'GPU {gpu} lacks reserved headroom ({reserve / 1024:.1f} GiB)')


def external_attestation(args):
    if args.external_server is None:
        return None
    if not args.external_server_manifest:
        result = {'verification': 'operator-declared CLI settings; /models verifies model ID only',
                  'declared_settings': server_spec(args)}
        manifest = os.environ.get('UTOPIA_SERVER_MANIFEST')
        if manifest:
            # The generic queue validates its own attestation schema. Preserve
            # its exact record without inventing an additional simulator flag.
            if manifest.lstrip().startswith('{'):
                result['queue_attestation'] = json.loads(manifest)
            else:
                path = Path(manifest).expanduser()
                result['queue_attestation'] = json.loads(path.read_text())
                result['queue_attestation_file'] = str(path.resolve())
                result['queue_attestation_sha256'] = sha256(path)
        return result
    payload = json.loads(args.external_server_manifest.read_text())
    for key, value in server_spec(args).items():
        if payload.get(key) != value:
            raise ValueError(f'external server attestation mismatch: {key} expected {value!r}')
    if not payload.get('command'):
        raise ValueError('external server attestation must include its actual launch command')
    if payload.get('endpoint', args.server_url) != args.server_url:
        raise ValueError('external server attestation endpoint mismatch')
    return payload


def source_hashes(root=REPO_ROOT):
    sources = {}
    root = Path(root)
    for parent, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in
                         {'outputs', 'data', '.git', '__pycache__', '.venv', 'venv'})
        for name in sorted(files):
            path = Path(parent) / name
            if (path.suffix in ('.py', '.yaml', '.yml', '.toml', '.txt')
                    or str(path.relative_to(root)) == SEQUENTIAL_PROTOCOL_PATH):
                sources[str(path.relative_to(root))] = sha256(path)
    return sources


def package_versions():
    versions = {}
    for name in ('torch', 'vllm', 'transformers', 'sentence-transformers', 'numpy', 'pandas'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def frozen_plan(args, jobs, git, caches, cache_metadata, root=REPO_ROOT):
    from utopia.funding.feedback import sequential_source_binding
    root = Path(root)
    return {
        'experiment_id': experiment_id(args), 'seeds': args.seeds,
        'protocol': PROTOCOL, 'population': population_spec(),
        'population_rng_namespace': POPULATION_RNG_NAMESPACE,
        'structured_output_transport': STRUCTURED_OUTPUT_TRANSPORT,
        'funding': funding_spec(),
        'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
        'sequential_funding': sequential_spec(),
        'sequential_elicitation_amendment': SEQUENTIAL_ELICITATION_AMENDMENT,
        'sequential_source_binding': sequential_source_binding(root),
        'request_audit': request_audit_spec(),
        'prior_attempt': PRIOR_ATTEMPT,
        'output_elicitation_amendment': OUTPUT_ELICITATION_AMENDMENT,
        'server': server_spec(args), 'server_command': server_cmd(args),
        'external_server': args.external_server is not None,
        'external_attestation': external_attestation(args),
        'server_url': args.server_url, 'server_env': runtime_env(args, 'server', root),
        'client_gpu': args.client_gpu, 'client_headroom_gb': args.client_headroom_gb,
        'max_concurrent_cells': args.max_concurrent_cells, 'timeout_hours': args.timeout_hours,
        'scientific_defaults': {'population_mode': PROTOCOL, 'initial_researchers': 1202,
                                'initial_agents_including_agencies': 1204,
                                'conferences': 10, 'years': SMOKE_YEARS if args.smoke else NUM_YEARS,
                                'acceptance_rate': 0.30, 'batch_size': BATCH_SIZE},
        'git_commit': git['commit'], 'source_sha256': source_hashes(root),
        'inputs': {k: {f: v[f] for f in ('sha256', 'bytes')} for k, v in caches.items()},
        'cache_metadata': cache_metadata, 'python': sys.version,
        'python_executable': sys.executable, 'package_versions': package_versions(),
        'hostname': socket.gethostname(), 'jobs': jobs, 'limitations': LIMITATION,
        'storage_root': str(STORAGE_ROOT) if STORAGE_ROOT else None,
        'storage_targets': {name: str((root / name).resolve()) for name in ('outputs', 'data')},
        'inherited_library_path': os.environ.get('LD_LIBRARY_PATH'),
    }


def json_finite(value):
    """Undefined secondary metrics become JSON null, never nonstandard NaN."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, content, root=REPO_ROOT, exclusive=False):
    path = safe_path(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json_finite(content)
    if exclusive:
        write_json_file(content, path, indent=2, allow_nan=False, mode='x')
    else:
        temp = safe_path(path.with_suffix(path.suffix + '.tmp'), root)
        write_json_atomic(path, content, indent=2, allow_nan=False,
                          trailing_newline=False, streaming=True)


def object_hash(value):
    from utopia.utils.data_utils import json_sha256
    return json_sha256(value, sort_keys=True, separators=(',', ':'))


def cell_provenance(args, job, git, caches, founders, root=REPO_ROOT):
    from utopia.funding.feedback import sequential_source_binding
    inference = {key: value for key, value in server_spec(args).items() if key != 'server_gpus'}
    return {
        'common': {
            'protocol': PROTOCOL, 'seed': SEED, 'population': population_spec(),
            'population_rng_namespace': POPULATION_RNG_NAMESPACE,
            'structured_output_transport': STRUCTURED_OUTPUT_TRANSPORT,
            'funding': funding_spec(),
            'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
            'sequential_funding': sequential_spec(),
            'sequential_elicitation_amendment': SEQUENTIAL_ELICITATION_AMENDMENT,
            'sequential_source_binding': sequential_source_binding(root),
            'request_audit': request_audit_spec(),
            'prior_attempt': PRIOR_ATTEMPT,
            'output_elicitation_amendment': OUTPUT_ELICITATION_AMENDMENT,
            'founder_sha256': object_hash(founders), 'git_commit': git['commit'],
            'source_sha256': source_hashes(root),
            'inputs': {key: {field: row[field] for field in ('sha256', 'bytes')}
                       for key, row in caches.items()},
            'inference': inference, 'batch_size': BATCH_SIZE,
            'python': sys.version, 'packages': package_versions(),
        },
        'cell': job['cell'], 'experiment_name': job['experiment_name'],
        'num_years': job['num_years'], 'treatment_flags': CELLS[job['cell']]['flags'],
        'command': job['command'], 'simulation_command': job['simulation_command'],
        'server_url': args.server_url, 'server': server_spec(args),
        'external_attestation': external_attestation(args),
        'environment': job['env_overrides'], 'python_executable': sys.executable,
        'hostname': socket.gethostname(), 'limitations': LIMITATION,
        'inherited_library_path': os.environ.get('LD_LIBRARY_PATH'),
    }


def freeze_cell(job, record, founders, root=REPO_ROOT):
    directory = safe_path(job['output_dir'], root)
    stamp = directory / 'influx_protocol.json'
    blueprint = directory / 'founder_blueprint.json'
    if stamp.exists():
        previous = json.loads(stamp.read_text())
        for key in ('common', 'cell', 'experiment_name', 'num_years', 'treatment_flags'):
            if previous.get(key) != record[key]:
                raise ValueError(f'frozen cell provenance differs: {key}; use fresh isolated outputs')
        if json.loads(blueprint.read_text()) != founders:
            raise ValueError('frozen founder blueprint differs')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f'cannot adopt existing cell without {PROTOCOL} provenance')
        write_json(blueprint, founders, root, exclusive=True)
        write_json(stamp, record, root, exclusive=True)


def expected_researcher_ids(cell, year):
    ids = {f'institution_{index:04d}_researcher_{r}'
           for index in range(UNIVERSITIES) for r in range(RESEARCHERS_PER_UNIVERSITY)}
    ids.update(f'Google_researcher_{r}' for r in range(INDUSTRY_RESEARCHERS))
    if cell in ('B', 'D'):
        for entry_year in range(1, year + 1):
            ids.update(f'institution_{index:04d}_researcher_{5 + (entry_year - 1) // 2}'
                       for index in range(UNIVERSITIES) if index % 2 == (entry_year - 1) % 2)
            if entry_year % 5 == 0:
                ids.add(f'Google_researcher_{1 + entry_year // 5}')
    return ids


def validate_results(job, root=REPO_ROOT):
    """Check real simulation artifacts, all years and researcher denominators."""
    out = Path(job['output_dir'])
    manifest = json.loads((out / 'run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('years_completed') != job['num_years']:
        raise ValueError('simulator run manifest is incomplete')
    audits = validate_runtime_audits(out, job['num_years'])
    from utopia.funding.feedback import funding_ledger_evidence
    ledger = funding_ledger_evidence(out, job['num_years'])
    sequential = json.loads((out / SEQUENTIAL_SUMMARY_FILE).read_text())
    stamp = out / 'influx_protocol.json'
    if stamp.exists():
        binding = json.loads(stamp.read_text())['common']['sequential_source_binding']
        if (sequential['source_sha256'] != binding['utopia/funding/sequential.py']
                or any(sequential['source_files_sha256'].get(key) != value
                       for key, value in binding.items())):
            raise ValueError('sequential helper differs from frozen source')
    previous = out / 'influx_result.json'
    if previous.exists():
        record = json.loads(previous.read_text())
        if (record.get('runtime_audits') != audits or record.get('funding_sequential') != sequential
                or record.get('funding_application_evidence') != ledger):
            raise ValueError('runtime audit evidence differs from completed manifest')
    counts = {}
    for year in range(1, job['num_years'] + 1):
        cp = json.loads((out / f'checkpoint_year_{year}.json').read_text())
        if cp.get('year') != year or cp.get('phase') != 5:
            raise ValueError(f'year {year} checkpoint is not a complete phase5 result')
        if cp.get('initial_university_count') != 1200 or cp.get('initial_industry_count') != 2:
            raise ValueError('checkpoint has wrong founder sector denominators')
        agents = cp['ecosystem_data']['agents']
        expected = expected_researcher_ids(job['cell'], year)
        researchers = [agent for agent in agents if agent.get('can_author')]
        if ({agent['id'] for agent in researchers} != expected
                or len(researchers) != len(expected) or len(agents) != len(expected) + FUNDING_AGENCIES):
            raise ValueError(f'year {year} researcher IDs/counts differ from frozen population')
        by_id = {agent['id']: agent for agent in researchers}
        by_inst = {}
        for agent in researchers:
            institution = agent.get('university_name') or agent.get('company_name')
            if not institution:
                raise ValueError('researcher institution missing')
            by_inst.setdefault(institution, set()).add(agent['id'])
        for names in by_inst.values():
            for name in names:
                if not (names - {name}).issubset(set(by_id[name].get('conflict_of_interest', []))):
                    raise ValueError(f'year {year}: incomplete institution CoI wiring')
        papers = cp['paper_tracker']['papers']
        if job['cell'] in ('A', 'B') and any(len(p.get('review_history', [])) > 1 for p in papers):
            raise ValueError('resubmission found in a disabled-resubmission cell')
        counts[str(year)] = {
            'researchers': len(researchers), 'all_agents': len(agents),
            'entrants_since_initialization': len(researchers) - FOUNDER_RESEARCHERS,
        }
    return {'status': 'complete', 'protocol': PROTOCOL, 'seed': SEED, 'cell': job['cell'],
            'num_years': job['num_years'], 'founder_researchers': FOUNDER_RESEARCHERS,
            'funding_agencies': FUNDING_AGENCIES,
            'funding': funding_spec(),
            'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
            'sequential_funding_status': 'complete', 'funding_sequential': sequential,
            'funding_application_evidence': ledger, 'funding_application_evidence_valid': True,
            'funding_validation_status': 'complete', 'request_audit_status': 'complete',
            'runtime_audits': audits,
            'checked_years': list(range(1, job['num_years'] + 1)), 'population_by_year': counts}


def funding_summary_gates():
    return {'schema': 'scienceutopia.funding_validation.v1', 'status': 'complete',
            'output_representation': FUNDING_OUTPUT_REPRESENTATION,
            'policy': FUNDING_VALIDATION_POLICY, 'installed': False,
            'audit_failures': 0, 'invalid_final_panels': 0, 'failed_batches': 0,
            'processing_failures': 0, 'unprocessed_panels': 0, 'completion_allowed': True,
            'imputed_rankings': 0, 'fallback_rankings': 0, 'restore_conflicts': []}


def sequential_summary_gates():
    from utopia.funding.feedback import sequential_summary_gates as gates
    return gates()


def request_summary_gates():
    return {'status': 'complete', 'n_clients': 1, 'guard_failures': 0,
            'pending_requests': 0, 'model': MODEL, 'model_revision': REVISION,
            'context_tokens': 32768}


def audit_summaries(out):
    """Cheap completion check; the full validator also reads the audit records."""
    summaries = {}
    for name, gates in ((FUNDING_SUMMARY_FILE, funding_summary_gates()),
                        (SEQUENTIAL_SUMMARY_FILE, sequential_summary_gates()),
                        (REQUEST_SUMMARY_FILE, request_summary_gates())):
        record = json.loads((out / name).read_text())
        if not isinstance(record, dict) or any(record.get(k) != v for k, v in gates.items()):
            raise ValueError(f'runtime audit failed or incomplete: {name}')
        summaries[name] = record
    n_requests = summaries[REQUEST_SUMMARY_FILE].get('n_requests')
    if type(n_requests) is not int or n_requests <= 0:
        raise ValueError('LLM request audit has no requests')
    if any(not (out / name).is_file() for name in
           (FUNDING_AUDIT_FILE, SEQUENTIAL_AUDIT_FILE, REQUEST_AUDIT_FILE)):
        raise ValueError('required runtime audit artifact missing')
    if (out / REQUEST_AUDIT_FILE).stat().st_size == 0:
        raise ValueError('LLM request audit is empty')
    return summaries


def validate_runtime_audits(out, num_years):
    """Require strict final rankings, cap25, application logs and request-budget evidence."""
    summaries = audit_summaries(out)
    summary = summaries[FUNDING_SUMMARY_FILE]
    final_panels = attempts = invalid_attempts = applications = records = 0
    with (out / FUNDING_AUDIT_FILE).open() as stream:
        for line in stream:
            row = json.loads(line)
            records += 1
            if (row.get('schema') != summary['schema'] or row.get('sequence') != records
                    or row.get('output_representation') != FUNDING_OUTPUT_REPRESENTATION
                    or row.get('event') not in ('validation_attempt', 'final_panel')):
                raise ValueError('invalid funding audit record or sequence')
            n = row.get('n_expected')
            if type(n) is not int or not 0 <= n <= FUNDING_PANEL_MAX_APPS:
                raise ValueError('funding audit exceeds the frozen panel cap')
            if row['event'] == 'validation_attempt':
                attempts += 1
                invalid_attempts += int(row.get('valid') is not True)
            else:
                final_panels += 1
                if (row.get('valid') is not True or row.get('n_valid') != n
                        or row.get('n_returned') != n
                        or any(row.get(k) != 0 for k in
                               ('n_missing', 'n_rejected', 'n_imputed', 'n_fallback'))):
                    raise ValueError('funding final ranking is incomplete or repaired')
                applications += n
    expected_counts = {'audit_records': records, 'validation_attempts': attempts,
                       'invalid_attempts': invalid_attempts, 'final_panels': final_panels}
    if any(summary.get(key) != value for key, value in expected_counts.items()):
        raise ValueError('funding summary disagrees with its audit')
    logged = 0
    for year in range(1, num_years + 1):
        path = out / f'funding_applications_year_{year}.jsonl'
        if not path.exists():
            continue  # The stock phase writes no file when there are no applications.
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                n = row.get('n_panel')
                if (row.get('year') != year or type(n) is not int
                        or not 1 <= n <= FUNDING_PANEL_MAX_APPS
                        or row.get('fallback_ranking') is not False
                        or row.get('imputed_tail') is not False):
                    raise ValueError('funding application log violates the frozen protocol')
                logged += 1
    if logged != applications:
        raise ValueError('funding application logs do not cover all final-panel applications')
    from utopia.funding.feedback import validate_sequential_evidence
    validate_sequential_evidence(out, summary, num_years=num_years)
    with (out / REQUEST_AUDIT_FILE).open() as stream:
        for line in stream:
            if not isinstance(json.loads(line), dict):
                raise ValueError('malformed LLM request audit')
    names = [FUNDING_AUDIT_FILE, FUNDING_SUMMARY_FILE, SEQUENTIAL_AUDIT_FILE,
             SEQUENTIAL_SUMMARY_FILE, REQUEST_AUDIT_FILE, REQUEST_SUMMARY_FILE]
    names += [f'funding_applications_year_{year}.jsonl' for year in range(1, num_years + 1)
              if (out / f'funding_applications_year_{year}.jsonl').exists()]
    return {name: {'sha256': sha256(out / name)} for name in names}


@contextlib.contextmanager
def funding_validation_scope(simulator, job, root=REPO_ROOT):
    """Install Godel's real guard; preserve failure evidence and restore on every exit."""
    from utopia.funding.validation import install_funding_validation, FundingRankingValidationError
    from utopia.funding.feedback import sequential_funding_scope, funding_phase_coverage
    out = safe_path(job['output_dir'], root)
    handle = install_funding_validation(
        simulator.FundingAgency, safe_path(out / FUNDING_AUDIT_FILE, root), compact=True)
    succeeded = False
    try:
        with sequential_funding_scope(
                simulator.FundingAgency, simulator.VLLMServerModel, out) as sequential, \
                funding_phase_coverage(simulator.Simulation, simulator.FundingAgency, sequential):
            yield handle
            if not handle.summary()['completion_allowed']:
                raise FundingRankingValidationError(handle.summary())
        succeeded = True
    finally:
        try:
            handle.restore()
        finally:
            summary = handle.summary()
            gates = funding_summary_gates()
            clean = all(summary.get(k) == v for k, v in gates.items()
                        if k not in ('status', 'policy'))
            record = {**summary, 'policy': FUNDING_VALIDATION_POLICY,
                      'status': 'complete' if succeeded and clean else 'failed',
                      'protocol': PROTOCOL, 'seed': SEED, 'cell': job['cell']}
            write_json(out / FUNDING_SUMMARY_FILE, record, root)
        if succeeded and not clean:
            raise FundingRankingValidationError(record)


def validate_structured_transport(simulator):
    """CPU-only guard against freezing the superseded shared client."""
    schema = {'type': 'object', 'properties': {'transport_probe': {'type': 'boolean'}},
              'required': ['transport_probe'], 'additionalProperties': False}
    response_format = {'type': 'json_schema', 'json_object': {'schema': schema}}
    # The shared method reads this attribute; do not construct a real client,
    # whose initializer contacts the model server.
    probe = SimpleNamespace(enable_thinking=True)
    body = simulator.VLLMServerModel._build_extra_body(probe, response_format)
    if ('guided_json' in body or body.get('structured_outputs', {}).get('json') != schema):
        raise ValueError('v6 requires the shared client structured_outputs.json compatibility fix')


def run_single_cell(args, job, git, root=REPO_ROOT):
    """Queue worker: never owns a vLLM process and never takes the family lock."""
    directory = safe_path(Path(root) / 'outputs/logs' / PROTOCOL / job['experiment_name'], root)
    directory.mkdir(parents=True, exist_ok=True)
    with safe_path(directory / '.lock', root).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(child_env(job['env_overrides'], root))
        # This short preparation lock only serializes detachment/validation of
        # shared private RAG inputs, never a running cell or a server.
        cache_lock = safe_path(directory.parent / '.cache.lock', root)
        with cache_lock.open('a') as cache_stream:
            fcntl.flock(cache_stream, fcntl.LOCK_EX)
            caches = materialize_caches(root)
            validate_caches(root)
        import utopia.simulation as simulator
        validate_structured_transport(simulator)
        founders = build_founders(simulator)
        record = cell_provenance(args, job, git, caches, founders, root)
        # A formal cell has exactly one attempt, even if its old checkpoint is
        # at phase 5 or complete. Read-only analysis is a separate command.
        out = Path(job['output_dir'])
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'v6 single-attempt cell already exists: {out}; no resume or adoption')
        freeze_cell(job, record, founders, root)
        log_path = directory / f'attempt_{time.time_ns()}.log'
        write_json(log_path.with_suffix('.json'), record, root, exclusive=True)
        with safe_path(log_path, root).open('x') as log, \
                contextlib.redirect_stdout(Tee(sys.stdout, log)), \
                contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print('# ' + shlex.join(job['command']), flush=True)
            wait_for_server(args)
            check_gpu_headroom(args, gpu_memory(), server_running=True)
            initializer = simulator.Simulation.initialize_agents
            cohort = simulator.build_influx_cohort
            setup_globals = simulator.project_setup.__globals__
            previous_cwd_check = setup_globals['check_cwd']
            def check_worktree_cwd():
                if Path.cwd().resolve() != Path(root).resolve():
                    raise ValueError('simulator must run from the isolated worktree root')
            previous_argv = sys.argv
            previous_cwd = Path.cwd()
            previous_term = signal.getsignal(signal.SIGTERM)
            previous_alarm = signal.getsignal(signal.SIGALRM)
            def stop_worker(signum, frame):
                # Do not let ordinary model/retry `except Exception` handlers
                # swallow the one-shot wall-time alarm.
                raise KeyboardInterrupt(f'cell interrupted or time bound reached (signal {signum})')
            try:
                signal.signal(signal.SIGTERM, stop_worker)
                signal.signal(signal.SIGALRM, stop_worker)
                signal.setitimer(signal.ITIMER_REAL, args.timeout_hours * 3600)
                simulator.Simulation.initialize_agents = (
                    lambda self: initialize_population(self, simulator, founders))
                simulator.build_influx_cohort = (
                    lambda year, seed: build_fixed_cohort(simulator, year, seed))
                setup_globals['check_cwd'] = check_worktree_cwd
                sys.argv = ['utopia', *job['simulation_command'][3:]]
                os.chdir(root)
                from utopia.models.request_audit import request_audit_scope
                with funding_validation_scope(simulator, job, root), request_audit_scope(
                        safe_path(Path(job['output_dir']) / REQUEST_AUDIT_FILE, root),
                        model_name=MODEL):
                    simulator.main()
                result = validate_results(job, root)
                write_json(Path(job['output_dir']) / 'influx_result.json', result, root)
                return 0
            except BaseException:
                from utopia.runtime.provenance import mark_failed_run_manifest
                mark_failed_run_manifest(job['output_dir'], PROTOCOL, traceback.format_exc())
                raise
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.signal(signal.SIGTERM, previous_term)
                simulator.Simulation.initialize_agents = initializer
                simulator.build_influx_cohort = cohort
                setup_globals['check_cwd'] = previous_cwd_check
                sys.argv = previous_argv
                os.chdir(previous_cwd)


def freeze_manifest(plan, git, caches, directory, root=REPO_ROOT):
    path = Path(directory) / 'launch_manifest.json'
    if path.exists():
        if json.loads(path.read_text())['plan'] != plan:
            raise ValueError('frozen launch plan differs; use a fresh isolated worktree')
        return
    for job in plan['jobs']:
        out = Path(job['output_dir'])
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'cannot adopt existing cell without a matching manifest: {out}')
    write_json(path, {'plan': plan, 'git': git, 'cache_origins': caches,
                      'frozen_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')},
               root, exclusive=True)


def child_env(overrides, root=REPO_ROOT):
    for key, value in overrides.items():
        if key == 'SENTENCE_TRANSFORMERS_HOME':
            continue  # The shared hub is an offline read input, never mkdir/write here.
        if key.endswith(('_DIR', '_HOME', '_PATH', '_ROOT')) or key in (
                'TMPDIR', 'TMP', 'TEMP', 'MPLCONFIGDIR'):
            safe_path(value, root).mkdir(parents=True, exist_ok=True)
    for key in ('HF_DATASETS_CACHE', 'HF_ASSETS_CACHE', 'HF_MODULES_CACHE'):
        safe_path(overrides[key], root).mkdir(parents=True, exist_ok=True)
    return {**os.environ, **overrides}


def spawn(command, overrides, log_path, root=REPO_ROOT):
    env = child_env(overrides, root)
    path = safe_path(log_path, root)
    with path.open('x') as log:
        log.write('# ' + shlex.join(command) + '\n')
        log.flush()
        return subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def stop_processes(processes):
    # Signal the entire group, even if its parent already exited.
    groups = [proc.pid for proc in processes]
    for group in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for proc in processes:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=max(0, deadline - time.monotonic()))
    for group in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    for proc in processes:
        proc.wait()


def wait_for_server(args, proc=None, timeout_s=1200):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f'vLLM exited with code {proc.returncode}; see server.log')
        try:
            with urllib.request.urlopen(args.server_url + '/models', timeout=5) as response:
                models = json.load(response).get('data', [])
            if any(model.get('id') == MODEL for model in models):
                return models
            raise ValueError(f'endpoint does not advertise {MODEL}')
        except (OSError, TimeoutError):
            if proc is None:
                raise RuntimeError('external server is not healthy')
            time.sleep(5)
    raise TimeoutError('vLLM startup timed out')


def completed(job):
    path = Path(job['output_dir']) / f"checkpoint_year_{job['num_years']}.json"
    # The simulator overwrites this path after individual phases, so filename
    # existence alone can silently admit an unfinished final simulation year.
    try:
        manifest = json.loads((path.parent / 'run_manifest.json').read_text())
        result = json.loads((path.parent / 'influx_result.json').read_text())
        stamp = json.loads((path.parent / 'influx_protocol.json').read_text())
        if (stamp['common']['protocol'] != PROTOCOL
                or stamp['common'].get('population_rng_namespace') != POPULATION_RNG_NAMESPACE
                or stamp['common'].get('structured_output_transport') != STRUCTURED_OUTPUT_TRANSPORT
                or stamp['common'].get('funding') != funding_spec()
                or stamp['common'].get('funding_selection_protocol') != FUNDING_SELECTION_PROTOCOL
                or stamp['common'].get('sequential_funding') != sequential_spec()
                or stamp['common'].get('sequential_elicitation_amendment') != SEQUENTIAL_ELICITATION_AMENDMENT
                or stamp['common'].get('output_elicitation_amendment') != OUTPUT_ELICITATION_AMENDMENT
                or stamp['common'].get('request_audit') != request_audit_spec()
                or stamp['common']['population'] != population_spec()
                or stamp['common']['seed'] != SEED or stamp['cell'] != job['cell']
                or stamp['experiment_name'] != job['experiment_name']
                or stamp['num_years'] != job['num_years']
                or result.get('status') != 'complete' or result.get('protocol') != PROTOCOL
                or result.get('seed') != SEED or result.get('cell') != job['cell']
                or result.get('founder_researchers') != FOUNDER_RESEARCHERS
                or result.get('funding_agencies') != FUNDING_AGENCIES
                or result.get('funding') != funding_spec()
                or result.get('funding_validation_status') != 'complete'
                or result.get('funding_selection_protocol') != FUNDING_SELECTION_PROTOCOL
                or result.get('sequential_funding_status') != 'complete'
                or result.get('funding_application_evidence_valid') is not True
                or result.get('request_audit_status') != 'complete'
                or result.get('num_years') != job['num_years']
                or result.get('checked_years') != list(range(1, job['num_years'] + 1))):
            return False
        if (manifest.get('status') != 'complete'
                or manifest.get('years_completed') != job['num_years']):
            return False
        summaries = audit_summaries(path.parent)
        sequential = summaries[SEQUENTIAL_SUMMARY_FILE]
        if (sequential.get('years_completed') != job['num_years']
                or sequential.get('years_started') != job['num_years']
                or sequential.get('agency_registrations') != 2 * job['num_years']):
            return False
        from utopia.funding.feedback import funding_ledger_evidence
        if result.get('funding_application_evidence') != funding_ledger_evidence(
                path.parent, job['num_years']):
            return False
        with path.open() as stream:
            checkpoint = json.load(stream)
        return (checkpoint.get('phase') == 5
                and checkpoint.get('year') == job['num_years'])
    except (OSError, ValueError, KeyError):
        return False


def run_cells(args, jobs, attempt_dir, server_proc=None, deadline=None, root=REPO_ROOT):
    pending = [job for job in jobs if not completed(job)]
    status = {job['id']: {'state': 'complete', 'resumed_skip': True}
              for job in jobs if completed(job)}
    active = {}
    deadline = deadline or time.monotonic() + args.timeout_hours * 3600
    try:
        while pending or active:
            if time.monotonic() >= deadline:
                raise TimeoutError('bounded experiment wall-time exceeded')
            if server_proc is not None and server_proc.poll() is not None:
                raise RuntimeError('managed vLLM server exited during simulation')
            while pending and len(active) < args.max_concurrent_cells:
                job = pending.pop(0)
                try:
                    proc = spawn(job['command'], job['env_overrides'],
                                 Path(attempt_dir) / (job['id'] + '.log'), root)
                except BaseException:
                    status[job['id']] = {'state': 'launch_failed'}
                    raise
                active[job['id']] = (proc, job)
                status[job['id']] = {'state': 'running', 'pid': proc.pid}
                print(f"[launcher] {job['id']} pid {proc.pid}", flush=True)
            for key, (proc, job) in list(active.items()):
                code = proc.poll()
                if code is not None:
                    final = completed(job)
                    status[key] = {'state': 'complete' if code == 0 and final else 'failed',
                                   'returncode': code, 'final_checkpoint': final}
                    # Clean up descendants now; do not retain old PIDs across a
                    # multi-day run, where PID reuse could target unrelated jobs.
                    stop_processes([proc])
                    del active[key]
                    print(f'[launcher] {key}: {status[key]}', flush=True)
            write_json(Path(attempt_dir) / 'cell_status.json', status, root)
            if active:
                time.sleep(5)
        return all(row['state'] == 'complete' for row in status.values())
    finally:
        stop_processes([proc for proc, job in active.values()])
        for key in active:
            status[key]['state'] = 'interrupted'
        for job in pending:
            status[job['id']] = {'state': 'not_started'}
        write_json(Path(attempt_dir) / 'cell_status.json', status, root)


def check_smoke(args, root=REPO_ROOT):
    """Structural gates only; never select seeds on observed outcomes."""
    failures = []
    for job in make_jobs(args, root):
        try:
            if not completed(job):
                raise ValueError('cell lacks complete protocol-stamped results')
            validate_results(job, root)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{job['id']}: {exc}")
    return failures


def summarize_interactions(per_seed):
    """Conditional contrasts only: a single world cannot estimate seed uncertainty."""
    if set(per_seed) != {SEED}:
        raise ValueError('analysis permits exactly one world: seed42')
    return {
        endpoint: {
            'seed': SEED, 'interaction': json_finite(row.get('interaction')),
            'status': ('complete' if isinstance(row.get('interaction'), (int, float))
                       and math.isfinite(row['interaction']) else 'missing'),
        } for endpoint, row in per_seed[SEED].items()
    }


def validate_cell_provenance(jobs):
    common = None
    records = {}
    for job in jobs:
        out = Path(job['output_dir'])
        record = json.loads((out / 'influx_protocol.json').read_text())
        current = record['common']
        founders = json.loads((out / 'founder_blueprint.json').read_text())
        if (current['protocol'] != PROTOCOL or current['seed'] != SEED
                or current.get('population_rng_namespace') != POPULATION_RNG_NAMESPACE
                or current.get('structured_output_transport') != STRUCTURED_OUTPUT_TRANSPORT
                or current.get('funding') != funding_spec()
                or current.get('funding_selection_protocol') != FUNDING_SELECTION_PROTOCOL
                or current.get('sequential_funding') != sequential_spec()
                or current.get('sequential_elicitation_amendment') != SEQUENTIAL_ELICITATION_AMENDMENT
                or current.get('output_elicitation_amendment') != OUTPUT_ELICITATION_AMENDMENT
                or current.get('request_audit') != request_audit_spec()
                or current['population'] != population_spec()
                or current['founder_sha256'] != object_hash(founders)
                or len(founders) != FOUNDER_RESEARCHERS
                or {r['researcher_name'] for r in founders} != expected_researcher_ids('A', 0)
                or sum(r['kind'] == 'university' for r in founders) != 1200
                or sum(r['kind'] == 'industry' for r in founders) != 2
                or record['cell'] != job['cell'] or record['num_years'] != NUM_YEARS
                or record['experiment_name'] != cell_name(job['cell'], SEED)
                or record['treatment_flags'] != CELLS[job['cell']]['flags']):
            raise ValueError(f"cell {job['cell']} has invalid frozen protocol/founder data")
        binding = current['sequential_source_binding']
        if (set(binding) != {'utopia/funding/sequential.py', SEQUENTIAL_PROTOCOL_PATH}
                or any(not value or current['source_sha256'].get(key) != value
                       for key, value in binding.items())):
            raise ValueError('missing or inconsistent sequential source/protocol binding')
        if common is not None and common != current:
            raise ValueError('cells have incompatible founders, inputs, code or inference settings')
        common = current
        records[job['cell']] = record
    return common, records


def run_analysis(args, jobs, plan=None, root=REPO_ROOT):
    """Reuse the original definitions; avoid the hard-coded seed42 baseline CLI."""
    out = Path(root) / 'outputs' / 'docs' / experiment_id(args)
    if not all(completed(job) for job in jobs):
        raise ValueError('analysis requires every selected cell to finish with protocol-stamped results')
    if (args.seeds != [SEED] or len(jobs) != 4 or {job['cell'] for job in jobs} != set('ABCD')
            or any(job['seed'] != SEED or job['num_years'] != NUM_YEARS for job in jobs)):
        raise ValueError('interaction analysis requires exactly four 10-year seed42 cells')
    common, records = validate_cell_provenance(jobs)
    checked = {job['cell']: validate_results(job, root) for job in jobs}
    from utopia.analysis.influx_factorial import cell_endpoints, compute_stats, decompose
    stats = {job['cell']: compute_stats(job['output_dir'], NUM_YEARS) for job in jobs}
    # The stock helper counts can_author researchers (gross, including inactive
    # researchers), not all serialized agents. Verify instead of subtracting two
    # or reusing historical n302 denominators.
    for cell, values in stats.items():
        for year in range(1, NUM_YEARS + 1):
            population = values['population_by_year']
            if population.get(year, population.get(str(year))) != checked[cell]['population_by_year'][str(year)]['researchers']:
                raise ValueError('analysis population denominator differs from researcher checkpoint count')
    endpoints = {key: cell_endpoints(value, NUM_YEARS) for key, value in stats.items()}
    decomposition = decompose(endpoints)
    for endpoint in next(iter(endpoints.values())):
        decomposition.setdefault(endpoint, {'interaction': None})
    summary = {
        'protocol': PROTOCOL, 'seed': SEED, 'num_years': NUM_YEARS,
        'population': population_spec(), 'common_provenance': common,
        'funding': funding_spec(),
        'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
        'cell_provenance': records, 'checked_results': checked,
        'endpoints_by_cell': endpoints, 'decomposition': decomposition,
        'interactions': summarize_interactions({SEED: decomposition}),
        'interaction_definition': '(D - B) - (C - A)',
        'limitations': LIMITATION,
    }
    write_json(out / 'seed42' / 'factorial_summary.json', summary, root)
    write_json(out / 'conditional_interactions.json', summary, root)
    lines = ['Influx x resubmission: conditional seed42 contrasts', '', LIMITATION, '',
             '| Endpoint | A | B | C | D | Interaction (D-B)-(C-A) |',
             '|---|---|---|---|---|---|']
    for endpoint, row in decomposition.items():
        fmt = lambda v: f'{v:.4g}' if isinstance(v, (int, float)) and math.isfinite(v) else 'missing'
        lines.append(f"| {endpoint} | " + ' | '.join(
            fmt(endpoints[cell].get(endpoint)) for cell in 'ABCD') + f" | {fmt(row['interaction'])} |")
    safe_path(out / 'summary.md', root).write_text('\n'.join(lines) + '\n')
    return summary


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def main(argv=None):
    global STORAGE_ROOT
    args = parse_args(argv)
    jobs = make_jobs(args)
    if args.dry_run:
        print(json.dumps({'protocol': PROTOCOL, 'prior_attempt': PRIOR_ATTEMPT,
                          'output_elicitation_amendment': OUTPUT_ELICITATION_AMENDMENT,
                          'population_rng_namespace': POPULATION_RNG_NAMESPACE,
                          'structured_output_transport': STRUCTURED_OUTPUT_TRANSPORT,
                          'funding': funding_spec(),
                          'request_audit': request_audit_spec(),
                          'seeds': args.seeds, 'requested_seeds': args.requested_seeds,
                          'server': server_spec(args), 'server_command': server_cmd(args),
                          'jobs': jobs, 'limitations': LIMITATION}, indent=2))
        return 0
    git = git_state()  # Before any writes; never launch in a main checkout.
    STORAGE_ROOT = args.storage_root.expanduser().resolve() if args.storage_root else None
    if STORAGE_ROOT is not None:
        original = Path(git['common_dir']).parent
        for name in ('outputs', 'data'):
            target = (REPO_ROOT / name).resolve()
            if target == (original / name).resolve():
                raise ValueError(f'{name} points at the original checkout; use private storage')
            if (REPO_ROOT / name).is_symlink() and target != STORAGE_ROOT / name:
                raise ValueError(f'{name} must link exactly to --storage_root/{name}')
    check_output_isolation()
    if args.run_cell:
        return run_single_cell(args, jobs[0], git)
    if args.analysis_only:
        run_analysis(args, jobs)
        return 0
    if args.check_smoke_only:
        args.smoke = True
        failures = check_smoke(args)
        print('\n'.join(failures) or 'ALL SMOKE GATES PASS')
        return int(bool(failures))
    log_root = safe_path(REPO_ROOT / 'outputs' / 'logs' / 'influx2x2_launcher')
    log_root.mkdir(parents=True, exist_ok=True)
    with safe_path(log_root / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another influx launcher owns this worktree') from exc
        caches = materialize_caches()
        cache_metadata = validate_caches()
        plan = frozen_plan(args, jobs, git, caches, cache_metadata)
        directory = safe_path(log_root / experiment_id(args))
        if not args.skip_preflight_tests:
            subprocess.run([sys.executable, '-B', 'tests/experiments/test_influx_runtime.py'],
                           cwd=REPO_ROOT, check=True)
            subprocess.run([sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
                            'tests/experiments/test_influx_factorial.py', '-q'],
                           cwd=REPO_ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                             'PYTHONDONTWRITEBYTECODE': '1'}, check=True)
        if not args.no_analysis and not args.smoke:
            # Fail before GPU hours are consumed if the original helper is missing.
            pass
        if not args.analysis_only:
            check_gpu_headroom(args, gpu_memory())
            if args.external_server is None:
                with socket.socket() as sock:
                    if sock.connect_ex(('localhost', args.port)) == 0:
                        raise ValueError(f'port {args.port} is already bound')
            if shutil.disk_usage(REPO_ROOT).free < 50 * 1e9:
                raise ValueError('at least 50 GB free disk space is required')
        freeze_manifest(plan, git, caches, directory)
        if args.preflight_only:
            print(f'PREFLIGHT OK; frozen manifest: {directory / "launch_manifest.json"}')
            return 0
        attempt = safe_path(directory / f'attempt_{time.time_ns()}')
        attempt.mkdir()
        with (attempt / 'launcher.log').open('x') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), \
                    contextlib.redirect_stderr(Tee(sys.stderr, log)):
                server_proc = None
                status = {'state': 'running', 'launcher_pid': os.getpid(),
                          'command': [sys.executable, *sys.argv],
                          'requested_seeds': args.requested_seeds}
                write_json(attempt / 'status.json', status)
                previous = signal.getsignal(signal.SIGTERM)
                def terminate(signum, frame):
                    raise KeyboardInterrupt(f'received signal {signum}')
                signal.signal(signal.SIGTERM, terminate)
                try:
                    if not args.analysis_only:
                        deadline = time.monotonic() + args.timeout_hours * 3600
                        if args.external_server is None:
                            server_proc = spawn(server_cmd(args), plan['server_env'],
                                                attempt / 'server.log')
                            status['server_pid'] = server_proc.pid
                            write_json(attempt / 'status.json', status)
                        status['server_models'] = wait_for_server(
                            args, server_proc, min(1200, args.timeout_hours * 3600))
                        check_gpu_headroom(args, gpu_memory(), server_running=True)
                        if not run_cells(args, jobs, attempt, server_proc, deadline):
                            raise RuntimeError('one or more cells failed; analysis was not run')
                        if args.smoke:
                            failures = check_smoke(args)
                            if failures:
                                raise RuntimeError('; '.join(failures))
                    if server_proc is not None:
                        stop_processes([server_proc])
                        server_proc = None
                    if not args.smoke and not args.no_analysis:
                        run_analysis(args, jobs, plan)
                    status['state'] = 'complete'
                    return 0
                except BaseException as exc:
                    status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
                    traceback.print_exc()
                    raise
                finally:
                    if server_proc is not None:
                        stop_processes([server_proc])
                    signal.signal(signal.SIGTERM, previous)
                    write_json(attempt / 'status.json', status)


if __name__ == '__main__':
    sys.exit(main())
