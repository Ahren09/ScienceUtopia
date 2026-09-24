#!/usr/bin/env python3
"""Launcher for the scale-expansion experiment (design v1,
outputs/docs/scale_expansion_design/design_v1.md).

Question: in an AI-mediated research system, how do evaluation and resource
allocation determine the long-run gains and costs of scale expansion?

Four binary design factors (level 0 = legacy behaviour, all default-off flags
of run_simulation.py):
    S  papers per completed project      1            vs  k=2   (--papers_per_project)
    R  agency budget                     track apps   vs  fixed (--funding_budget_mode)
    K  review capacity                   3/paper      vs  cap c (--reviewer_capacity)
    E  evaluation scheme                 persona      vs  standardized (--review_policy)

Worlds: --population university_only (synthetic balanced institutions, every
researcher `balanced`; pilot = 20 x 5 = 100 researchers, 8 years, 6 conferences;
scale-up = 60 x 5 = 300 with --funding_panel_max_apps 25) or --population default
(the paper-baseline roster: 30 universities + 30 companies x 5 = 300, 10 years,
10 conferences; factor R then also fixes the industry pay-per-paper pool).
Qwen/Qwen3-32B via an external vLLM server. New cells also use the two realism
fixes (float review scores, seeded tie-break); --legacy_scoring disables them.

Stages / cells:
    smoke   S0R0 only, 1 year, 5 institutions (pipeline check, runtime estimate)
    a       S x R            : S0R0 S1R0 S0R1 S1R1
    b       K x E under S1R1 : S1R1_K1 S1R1_E1 S1R1_K1E1, plus S0R0_E1
    sens    S1R0_slots (fixed acceptance slots, venue-capacity sensitivity)
Stage b requires --slots_frac (f) and --reviewer_capacity (c); stage a requires f.

Outputs (canonical layout, experiment_stage=scale):
    outputs/{logs,checkpoints,visual,docs}/scale_qwen3_32b_<cell tag>_i20_n100_y8_seed<S>_mixbal/
    outputs/logs/scale_expansion_launcher/launch_summary.json  (commands, env, seeds, timings)

Examples:
    python src/run_scale_expansion.py launch --stage smoke --seeds 42 --vllm_url http://localhost:8036/v1
    python src/run_scale_expansion.py launch --stage a --seeds 1 2 3 4 5 --slots_frac 0.20 \
        --vllm_url http://localhost:8031/v1 http://localhost:8032/v1 http://localhost:8033/v1 \
                   http://localhost:8034/v1 http://localhost:8035/v1 --max_parallel 5
    (five single-GPU FP8 servers, see scripts/run_qwen3_32b_fp8_server.py; RAG on CPU)
    python src/run_scale_expansion.py status
    python src/run_scale_expansion.py analyze --num_years 8
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen3-32B"
DEFAULT_VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8036/v1")
LAUNCHER_DIR = os.path.join(REPO_ROOT, 'outputs', 'logs', 'scale_expansion_launcher')

# cell name -> factor levels (S=k, R=budget, K=capacity on, E=standardized, slots=fixed slots)
CELLS = {
    'S0R0':      dict(k=1, budget='track', cap=False, std=False, slots=False),
    'S1R0':      dict(k=2, budget='track', cap=False, std=False, slots=False),
    'S0R1':      dict(k=1, budget='fixed', cap=False, std=False, slots=False),
    'S1R1':      dict(k=2, budget='fixed', cap=False, std=False, slots=False),
    'S1R1_K1':   dict(k=2, budget='fixed', cap=True,  std=False, slots=False),
    'S1R1_E1':   dict(k=2, budget='fixed', cap=False, std=True,  slots=False),
    'S1R1_K1E1': dict(k=2, budget='fixed', cap=True,  std=True,  slots=False),
    'S0R0_E1':   dict(k=1, budget='track', cap=False, std=True,  slots=False),
    'S1R0_slots': dict(k=2, budget='track', cap=False, std=False, slots=True),
    # counterfactual: cut the resubmission channel in the pressure world (mechanism sufficiency)
    'S1R1_noresub': dict(k=2, budget='fixed', cap=False, std=False, slots=False, noresub=True),
}
# Cost control keeps the actual legacy production debit (zero) and waives the
# per-paper resubmission fee. Resubmission decisions and review workload remain.
# The configured funding_cost_per_paper=15 is unused in the legacy code, so
# activating it here would add a new expense rather than remove a confound.
for _base in ('S0R0', 'S1R0', 'S0R1', 'S1R1'):
    CELLS[_base + '_costcontrol'] = {
        **CELLS[_base], 'production_cost_mode': 'per_paper',
        'resubmission_cost': 0, 'log_resource_ledger': True,
    }
STAGES = {
    'smoke': ['S0R0'],
    'a': ['S0R0', 'S1R0', 'S0R1', 'S1R1'],
    'b': ['S1R1', 'S1R1_K1', 'S1R1_E1', 'S1R1_K1E1', 'S0R0_E1'],
    'sens': ['S1R0_slots'],
    'counterfactual': ['S1R1_noresub'],
    'costcontrol': [cell + '_costcontrol' for cell in ('S0R0', 'S1R0', 'S0R1', 'S1R1')],
    'costpilot': ['S0R0_costcontrol', 'S1R0_costcontrol', 'S0R1_costcontrol', 'S1R1_costcontrol'],
}


def build_command(cell: str, seed: int, args, vllm_url: str = None) -> list:
    spec = CELLS[cell]
    vllm_url = vllm_url or args.vllm_url[0]
    cmd = [
        sys.executable, 'run_simulation.py',
        '--experiment_name', 'scale_expansion',
        '--experiment_stage', 'scale',
        '--seed', str(seed),
        '--population_mode', args.population,
    ]
    if args.population == 'university_only':
        # synthetic balanced institutions, every researcher without an exploration prior
        cmd += ['--num_institutions', str(args.num_institutions),
                '--researchers_per_institution', '5',
                '--strategy_mix', 'balanced']
    # default population: the paper-baseline roster (10 rich + 20 normal universities,
    # 10 rich + 20 normal companies, 5 each = 300); all researchers are 'balanced' there
    cmd += [
        '--num_years', str(args.num_years),
        '--num_conferences', str(args.num_conferences),
        '--start_year', str(args.start_year),
        '--model', MODEL,
        '--vllm_url', vllm_url,
        '--batch_size', str(args.batch_size),
        '--industry_funding_mode', 'performance',
        '--funding_allocation_mode', 'fixed',
        '--funding_panel_max_apps', str(args.funding_panel_max_apps),  # 0 = global ranking
        '--log_funding_applications',         # full pre-decision rankings for P3
        '--papers_per_project', str(spec['k']),
        '--funding_budget_mode', spec['budget'],
        '--review_policy', 'standardized' if spec['std'] else 'persona',
    ]
    if spec['budget'] == 'fixed':
        if args.slots_frac is None:
            sys.exit(f"cell {cell} needs --slots_frac (preregistered f)")
        cmd += ['--funding_budget_slots_frac', str(args.slots_frac)]
    if spec['cap']:
        if args.reviewer_capacity is None:
            sys.exit(f"cell {cell} needs --reviewer_capacity (preregistered c)")
        cmd += ['--reviewer_capacity', str(args.reviewer_capacity)]
        if args.reviewer_matching != 'random':
            cmd += ['--reviewer_matching', args.reviewer_matching]
    if spec['slots']:
        cmd += ['--acceptance_mode', 'fixed_slots']
    if spec.get('noresub'):
        cmd.append('--disable_resubmission')
    if 'production_cost_mode' in spec:
        cmd += ['--production_cost_mode', spec['production_cost_mode'],
                '--resubmission_cost', str(spec['resubmission_cost'])]
    if spec.get('log_resource_ledger'):
        cmd.append('--log_resource_ledger')
    if not args.legacy_scoring:
        cmd += ['--review_score_mode', 'float', '--acceptance_tiebreak', 'seeded']
    if args.always_rerun:
        cmd.append('--always_rerun')
    return cmd


def experiment_id_for(cmd: list) -> str:
    """Resolve the experiment_id exactly as run_simulation will. parse_arguments creates the
    canonical output dirs as a side effect; remove them again if they are still empty so a
    dry run leaves no trace (the real run recreates them)."""
    sys.path.insert(0, REPO_ROOT)
    from utopia.arguments import parse_arguments
    a = parse_arguments(cmd[2:])
    for d in (a.log_dir, a.checkpoint_dir, a.visual_dir, a.docs_dir):
        try:
            os.rmdir(d)  # only succeeds when empty
        except OSError:
            pass
    return a.experiment_id


def launch(args):
    os.makedirs(LAUNCHER_DIR, exist_ok=True)
    git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    jobs = []
    for seed in args.seeds:
        for cell in (args.cells or STAGES[args.stage]):
            # round-robin over the available vLLM endpoints (several single-GPU FP8
            # servers, or several TP=2 servers); per-request seeds make routing irrelevant
            url = args.vllm_url[len(jobs) % len(args.vllm_url)]
            cmd = build_command(cell, seed, args, vllm_url=url)
            exp_id = experiment_id_for(cmd)
            jobs.append({'cell': cell, 'seed': seed, 'experiment_id': exp_id, 'cmd': cmd, 'vllm_url': url})
    # de-duplicate shared cells across stages (e.g. S1R1 in a and b) and skip ids that
    # another launcher is running right now (two writers on one checkpoint dir = corruption)
    seen, unique = set(), []
    for j in jobs:
        if j['experiment_id'] in seen:
            continue
        seen.add(j['experiment_id'])
        # run_simulation's command line does not contain the id, so detect a live run by
        # its manifest status plus a recently written log (heartbeat < 15 min)
        manifest = os.path.join(REPO_ROOT, 'outputs', 'docs', j['experiment_id'], 'run_manifest.json')
        log = os.path.join(REPO_ROOT, 'outputs', 'logs', j['experiment_id'], 'simulation.log')
        if os.path.exists(manifest) and os.path.exists(log):
            st = json.load(open(manifest)).get('status')
            if st == 'running' and time.time() - os.path.getmtime(log) < 900:
                print(f"[launcher] SKIP {j['experiment_id']}: running elsewhere (manifest=running, "
                      f"log written {int(time.time() - os.path.getmtime(log))}s ago)", flush=True)
                continue
            if st == 'complete' and not args.always_rerun:
                print(f"[launcher] SKIP {j['experiment_id']}: already complete", flush=True)
                continue
        unique.append(j)
    jobs = unique

    manifest = {'stage': args.stage, 'git_commit': git_commit, 'vllm_url': args.vllm_url,
                'model': MODEL, 'slots_frac': args.slots_frac, 'reviewer_capacity': args.reviewer_capacity,
                'legacy_scoring': args.legacy_scoring, 'max_parallel': args.max_parallel,
                'client_gpus': args.client_gpus, 'started': time.ctime(),
                'jobs': [{k: v for k, v in j.items()} for j in jobs]}
    with open(os.path.join(LAUNCHER_DIR, f'launch_manifest_{args.stage}_{int(time.time())}.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[launcher] {len(jobs)} runs, max_parallel={args.max_parallel}, commit={git_commit[:12]}")
    if args.dry_run:
        for j in jobs:
            print(j['experiment_id']); print('   ' + ' '.join(j['cmd']))
        return

    running, records = [], []
    queue = list(jobs)
    while queue or running:
        while queue and len(running) < args.max_parallel:
            j = queue.pop(0)
            log_dir = os.path.join(REPO_ROOT, 'outputs', 'logs', j['experiment_id'])
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, 'command.txt'), 'w') as f:
                f.write(' '.join(j['cmd']) + '\n')
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = args.client_gpus  # '' = RAG on CPU (embeddings are cached)
            # disjoint caches: the npz/json caches are single-writer
            env['UTOPIA_DATA_CACHE_DIR'] = os.path.join('data', 'cache', j['experiment_id'])
            log_file = open(os.path.join(log_dir, 'simulation.log'), 'a')
            log_file.write(f"\n===== launch @ {time.ctime()} commit={git_commit} =====\n{' '.join(j['cmd'])}\n")
            log_file.flush()
            proc = subprocess.Popen(j['cmd'], cwd=REPO_ROOT, env=env, stdout=log_file,
                                    stderr=subprocess.STDOUT)
            running.append((j, proc, log_file, time.time(), j.get('_attempt', 1)))
            print(f"[launcher] started {j['experiment_id']} (pid {proc.pid})", flush=True)
        time.sleep(30)
        still = []
        for j, proc, log_file, t0, attempt in running:
            rc = proc.poll()
            if rc is None:
                still.append((j, proc, log_file, t0, attempt)); continue
            log_file.close()
            elapsed = round(time.time() - t0, 1)
            if rc == 0:
                records.append({**{k: v for k, v in j.items() if k != 'cmd'}, 'status': 'complete',
                                'elapsed_seconds': elapsed, 'attempts': attempt})
                print(f"[launcher] complete {j['experiment_id']} in {elapsed/3600:.2f} h", flush=True)
            elif attempt < args.max_attempts:
                print(f"[launcher] {j['experiment_id']} exited {rc}; resuming from checkpoint", flush=True)
                j2 = dict(j); j2['cmd'] = [c for c in j['cmd'] if c != '--always_rerun']
                j2['_attempt'] = attempt + 1  # carried through the queue entry
                queue.insert(0, j2)
            else:
                records.append({**{k: v for k, v in j.items() if k != 'cmd'}, 'status': 'failed',
                                'elapsed_seconds': elapsed, 'attempts': attempt})
        running = still
    summary_path = os.path.join(LAUNCHER_DIR, 'launch_summary.json')
    existing = json.load(open(summary_path)) if os.path.exists(summary_path) else []
    with open(summary_path, 'w') as f:
        json.dump(existing + records, f, indent=2)
    print(json.dumps(records, indent=2))
    if any(r['status'] != 'complete' for r in records):
        sys.exit(1)


def status(args):
    runs = sorted(glob.glob(os.path.join(REPO_ROOT, 'outputs', 'checkpoints', 'scale_*')))
    for r in runs:
        ck = sorted(glob.glob(os.path.join(r, 'checkpoint_year_*.json*')))
        years = [int(os.path.basename(c).split('_')[-1].split('.')[0]) for c in ck]
        manifest = os.path.join(REPO_ROOT, 'outputs', 'docs', os.path.basename(r), 'run_manifest.json')
        st = json.load(open(manifest)).get('status') if os.path.exists(manifest) else 'n/a'
        print(f"{os.path.basename(r):85s} years={max(years) if years else 0:2d} status={st}")


def analyze(args):
    sys.path.insert(0, REPO_ROOT)
    from utopia.analysis.scale_expansion import main as analysis_main
    analysis_main(['--num_years', str(args.num_years), '--out_dir', args.out_dir]
                  + (['--glob', args.glob] if args.glob else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('launch')
    p.add_argument('--stage', required=True, choices=list(STAGES))
    p.add_argument('--cells', nargs='+', choices=list(CELLS), default=None, help='override the stage cell list')
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--population', choices=['university_only', 'default'], default='university_only',
                   help='university_only = synthetic balanced institutions (num_institutions x 5); '
                        'default = paper-baseline roster (300 researchers, universities + companies)')
    p.add_argument('--num_institutions', type=int, default=20, help='university_only worlds only')
    p.add_argument('--num_years', type=int, default=None, help='default 8 (university_only) / 10 (default population)')
    p.add_argument('--num_conferences', type=int, default=None, help='default 6 (university_only) / 10 (default population)')
    p.add_argument('--funding_panel_max_apps', type=int, default=0,
                   help='0 = one global ranking per program (fits ~<=60 applications); use 25 when a '
                        'program receives more (validated panelization, exploration experiment)')
    p.add_argument('--start_year', type=int, default=2016)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--vllm_url', type=str, nargs='+', default=[DEFAULT_VLLM_URL],
                   help='one or more OpenAI-compatible endpoints; runs are assigned round-robin')
    p.add_argument('--slots_frac', type=float, default=None, help='preregistered f for R-fixed cells')
    p.add_argument('--reviewer_capacity', type=int, default=None, help='preregistered c for K cells')
    p.add_argument('--reviewer_matching', choices=['random', 'topic'], default='random')
    p.add_argument('--legacy_scoring', action='store_true',
                   help='keep int-truncated scores and stable tie-break (legacy) in the new cells')
    p.add_argument('--max_parallel', type=int, default=4)
    p.add_argument('--max_attempts', type=int, default=3)
    p.add_argument('--client_gpus', type=str, default='', help='CUDA_VISIBLE_DEVICES for the RAG embedder')
    p.add_argument('--always_rerun', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.set_defaults(func=launch)

    p = sub.add_parser('status'); p.set_defaults(func=status)

    p = sub.add_parser('analyze')
    p.add_argument('--num_years', type=int, default=8)
    p.add_argument('--out_dir', type=str, default=os.path.join('outputs', 'docs', 'scale_expansion_analysis'))
    p.add_argument('--glob', type=str, default=None)
    p.set_defaults(func=analyze)

    args = parser.parse_args()
    if args.command == 'launch':
        if args.num_years is None:
            args.num_years = 10 if args.population == 'default' else 8
        if args.num_conferences is None:
            args.num_conferences = 10 if args.population == 'default' else 6
        if args.stage in ('smoke', 'costpilot'):
            args.num_institutions = min(args.num_institutions, 5)
            args.num_years = min(args.num_years, 2 if args.stage == 'costpilot' else 1)
    args.func(args)


if __name__ == '__main__':
    main()
