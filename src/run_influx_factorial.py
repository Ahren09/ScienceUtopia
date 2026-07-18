#!/usr/bin/env python
"""Launcher for the influx2x2 experiment: researcher influx x resubmission
2x2 factorial, single seed 42, full paper scale, local serving on GPUs 1-5.

Starts a bf16 TP=4 vLLM server on GPUs 1-4 port 8036 (identical to the paper
resubmission-pair serving: Qwen/Qwen3-32B @ revision 9216db57, vLLM seed 42,
max-model-len 32768, reasoning parser qwen3), then launches the four cell
simulations IN PARALLEL against it:

    A  influx2x2_A_noinflux_noresub_seed42   --disable_resubmission
    B  influx2x2_B_influx_noresub_seed42     --annual_influx_rate 0.1 --disable_resubmission
    C  influx2x2_C_noinflux_resub_seed42     (config twin of default_Qwen3-32B_fixed30_paired)
    D  influx2x2_D_influx_resub_seed42       --annual_influx_rate 0.1

Per-cell env: CUDA_VISIBLE_DEVICES=5 (the RAG embedder grabs cuda:0 — GPU 5
keeps retrieval fp16 numerics identical to the paper baseline, which ran RAG
on GPU), UTOPIA_DATA_CACHE_DIR=data/cache/influx2x2_<cell> (disjoint caches
for concurrent runs). Cells use the legacy output layout, matching the paper
pair: outputs/<experiment_name>/ + outputs/<experiment_name>_launch.log.

The launcher records everything CLAUDE.md §10 requires under
outputs/logs/influx2x2_launcher/ (launch_manifest.json: commands, env deltas,
seeds, GPU map, server config, PIDs, git state; server.log; launcher output),
monitors progress every 60 s, kills the server when all cells exit, and (full
run) triggers the analysis pipeline: per-cell resubmission_stats ->
influx_factorial decomposition -> plot_influx_factorial figure.

A crashed cell does not stop the others; rerunning the launcher resumes
crashed cells from their checkpoints (influx injection is idempotent) and
skips cells whose final checkpoint already exists.

Usage (from the repo root):
    python src/run_influx_factorial.py --preflight_only   # checks only
    python src/run_influx_factorial.py --smoke            # A+D, 3 years, + gates
    python src/run_influx_factorial.py                    # full 4-cell run + analysis
"""
import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SEED = 42
MODEL = 'Qwen/Qwen3-32B'
REVISION = '9216db5781bf21249d130ec9da846c4624c16137'  # exact paper revision
PORT = 8036
SERVER_GPUS = '1,2,3,4'
CLIENT_GPU = '5'
NUM_YEARS = 10
BATCH_SIZE = 64
EXP_ID = 'influx2x2_qwen3_32b_seed42'
LAUNCHER_LOG_DIR = os.path.join(REPO_ROOT, 'outputs', 'logs', 'influx2x2_launcher')

CELLS = {
    'A': {'name': f'influx2x2_A_noinflux_noresub_seed{SEED}',
          'flags': ['--disable_resubmission']},
    'B': {'name': f'influx2x2_B_influx_noresub_seed{SEED}',
          'flags': ['--annual_influx_rate', '0.1', '--disable_resubmission']},
    'C': {'name': f'influx2x2_C_noinflux_resub_seed{SEED}',
          'flags': []},
    'D': {'name': f'influx2x2_D_influx_resub_seed{SEED}',
          'flags': ['--annual_influx_rate', '0.1']},
}
SMOKE_CELLS = ('A', 'D')
SMOKE_YEARS = 3


def cell_cmd(cell_key: str, name: str, num_years: int):
    return [sys.executable, 'run_simulation.py',
            '--model', MODEL,
            '--vllm_url', f'http://localhost:{PORT}/v1',
            '--seed', str(SEED),
            '--num_years', str(num_years),
            '--batch_size', str(BATCH_SIZE),
            '--experiment_name', name] + CELLS[cell_key]['flags']


def cell_env(cell_key: str):
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = CLIENT_GPU
    env['UTOPIA_DATA_CACHE_DIR'] = os.path.join('data', 'cache', f'influx2x2_{cell_key}')
    env['PYTHONUNBUFFERED'] = '1'
    return env


# ---------------------------------------------------------------- preflight

def gpu_mem_used():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        text=True)
    return {int(i): int(m) for i, m in
            (line.split(', ') for line in out.strip().splitlines())}


def preflight(skip_tests: bool = False):
    failures = []

    mem = gpu_mem_used()
    for g in [int(x) for x in SERVER_GPUS.split(',')] + [int(CLIENT_GPU)]:
        if mem.get(g, 10 ** 9) > 1000:
            failures.append(f'GPU {g} busy ({mem.get(g)} MiB used)')

    with socket.socket() as s:
        if s.connect_ex(('localhost', PORT)) == 0:
            failures.append(f'port {PORT} already bound')

    free_gb = shutil.disk_usage(os.path.join(REPO_ROOT, 'outputs')).free / 1e9
    if free_gb < 50:
        failures.append(f'only {free_gb:.0f} GB free under outputs/ (need >= 50)')

    # Shared read caches must pre-exist: regenerating them under 4 concurrent
    # writers would race (non-atomic whole-file flushes).
    for rel in ('outputs/cache_scievo.pkl',
                'outputs/thenlper_gte-small/cache_embeddings.pt'):
        if not os.path.exists(os.path.join(REPO_ROOT, rel)):
            failures.append(f'missing shared cache {rel} (would be racily regenerated)')

    for spec in CELLS.values():
        if not spec['name'].startswith(('default', 'acl-acceptance-rates',
                                        'acl_schedule_smoke', 'influx')):
            failures.append(f"experiment name {spec['name']} fails the whitelist")

    if not skip_tests:
        r = subprocess.run([sys.executable, '-m', 'pytest',
                            'tests/test_influx_factorial.py', '-q'],
                           cwd=REPO_ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            failures.append('tests/test_influx_factorial.py failing:\n' + r.stdout[-2000:])

    git = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=REPO_ROOT, text=True).strip(),
        'dirty_files': subprocess.check_output(['git', 'status', '--short'],
                                               cwd=REPO_ROOT, text=True).strip().splitlines(),
    }
    return failures, git


# ---------------------------------------------------------------- server

def start_server():
    os.makedirs(LAUNCHER_LOG_DIR, exist_ok=True)
    cmd = ['vllm', 'serve', MODEL,
           '--revision', REVISION,
           '--tensor-parallel-size', '4',
           '--max-model-len', '32768',
           '--gpu-memory-utilization', '0.90',
           '--port', str(PORT),
           '--seed', str(SEED),
           '--reasoning-parser', 'qwen3']
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = SERVER_GPUS
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    log_path = os.path.join(LAUNCHER_LOG_DIR, 'server.log')
    logf = open(log_path, 'a')
    logf.write(f'# launched {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    logf.write('# CUDA_VISIBLE_DEVICES=%s %s\n' % (SERVER_GPUS, ' '.join(cmd)))
    logf.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=REPO_ROOT, start_new_session=True)
    print(f'[launcher] vLLM server pid {proc.pid} (GPUs {SERVER_GPUS}, port {PORT}); '
          f'log {log_path}')
    return proc, {'command': cmd, 'cuda_visible_devices': SERVER_GPUS,
                  'pid': proc.pid, 'log': log_path}


def wait_for_server(proc, timeout_s: int = 1200):
    url = f'http://localhost:{PORT}/v1/models'
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f'vLLM server exited early (code {proc.returncode}); '
                               f'see {LAUNCHER_LOG_DIR}/server.log')
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f'[launcher] server healthy after {time.time() - t0:.0f}s')
                    return
        except Exception:
            pass
        time.sleep(10)
    kill_server(proc)
    raise RuntimeError(f'vLLM server not healthy after {timeout_s}s')


def kill_server(proc):
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    print('[launcher] vLLM server stopped')


# ---------------------------------------------------------------- cells

def launch_cells(cell_keys, num_years, name_override=None, git=None, server_info=None):
    procs, manifest_cells = {}, {}
    for key in cell_keys:
        name = name_override(key) if name_override else CELLS[key]['name']
        final_ckpt = os.path.join(REPO_ROOT, 'outputs', name,
                                  f'checkpoint_year_{num_years}.json')
        if os.path.exists(final_ckpt):
            print(f'[launcher] cell {key} ({name}) already complete; skipping')
            continue
        cmd = cell_cmd(key, name, num_years)
        env = cell_env(key)
        log_path = os.path.join(REPO_ROOT, 'outputs', f'{name}_launch.log')
        logf = open(log_path, 'a')
        logf.write(f'# launched {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        logf.write('# ' + ' '.join(cmd) + '\n')
        logf.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT)
        procs[key] = (proc, name, log_path)
        manifest_cells[key] = {
            'experiment_name': name, 'command': cmd, 'pid': proc.pid,
            'log': log_path, 'seed': SEED, 'num_years': num_years,
            'env_overrides': {'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'],
                              'UTOPIA_DATA_CACHE_DIR': env['UTOPIA_DATA_CACHE_DIR']},
        }
        print(f'[launcher] cell {key} pid {proc.pid} -> {name}')

    os.makedirs(LAUNCHER_LOG_DIR, exist_ok=True)
    manifest = {
        'experiment_id': EXP_ID,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': MODEL, 'revision': REVISION, 'seed': SEED,
        'server': server_info, 'server_gpus': SERVER_GPUS, 'client_gpu': CLIENT_GPU,
        'git': git, 'cells': manifest_cells,
    }
    with open(os.path.join(LAUNCHER_LOG_DIR, 'launch_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    return procs


def monitor(procs):
    def completed_years(name):
        run_dir = os.path.join(REPO_ROOT, 'outputs', name)
        done_years = [int(m.group(1)) for f in os.listdir(run_dir)
                      if (m := re.match(r'checkpoint_year_(\d+)\.json$', f))] \
            if os.path.isdir(run_dir) else []
        return max(done_years) if done_years else 0

    done = {}
    while len(done) < len(procs):
        time.sleep(60)
        status = []
        for key, (proc, name, log_path) in procs.items():
            if key in done:
                continue
            rc = proc.poll()
            if rc is None:
                status.append(f'{key}:y{completed_years(name)}done')
            else:
                done[key] = rc
                print(f'[launcher] cell {key} exited code {rc} '
                      f'(completed years: {completed_years(name)})')
        if status:
            print(f"[launcher] {time.strftime('%H:%M:%S')} running: {' '.join(status)}")
    return done


# ---------------------------------------------------------------- smoke gates

def check_smoke(num_years=SMOKE_YEARS):
    """Structural validity gates on the smoke cells (never outcome-based)."""
    failures = []
    names = {k: f'influx2x2_smoke_{k}_seed{SEED}' for k in SMOKE_CELLS}

    for key, name in names.items():
        run_dir = os.path.join(REPO_ROOT, 'outputs', name)
        log_path = os.path.join(REPO_ROOT, 'outputs', f'{name}_launch.log')
        log = open(log_path, errors='replace').read() if os.path.exists(log_path) else ''
        if 'Traceback' in log or 'AssertionError' in log:
            failures.append(f'{key}: traceback/assertion in {log_path}')
        if log.count('No candidate papers found'):
            failures.append(f"{key}: {log.count('No candidate papers found')} silent "
                            'Phase-2 skips (corpus exhaustion)')
        ckpt_path = os.path.join(run_dir, f'checkpoint_year_{num_years}.json')
        if not os.path.exists(ckpt_path):
            failures.append(f'{key}: missing {ckpt_path}')
            continue
        cp = json.load(open(ckpt_path))
        agents = cp['ecosystem_data']['agents']
        papers = cp['paper_tracker']['papers']
        entrants = [a for a in agents
                    if re.search(r'_researcher_[5-9]$', a['id'])]

        if key == 'D':
            for y in range(1, num_years + 1):
                if f'[Influx] Year {y}: +30' not in log:
                    failures.append(f'D: missing "[Influx] Year {y}: +30" in log')
            if len(entrants) != 30 * num_years:
                failures.append(f'D: expected {30 * num_years} entrants in checkpoint, '
                                f'got {len(entrants)}')
            if len(agents) != 302 + 30 * num_years:
                failures.append(f'D: population {len(agents)} != {302 + 30 * num_years}')
            # entrants must be CoI-wired both directions with a founder colleague
            by_inst = {}
            for a in agents:
                inst = a.get('university_name') or a.get('company_name')
                if inst:
                    by_inst.setdefault(inst, []).append(a)
            for e in entrants[:10]:
                inst = e.get('university_name') or e.get('company_name')
                founders = [a for a in by_inst[inst]
                            if re.search(r'_researcher_[0-4]$', a['id'])]
                if not all(e['id'] in f.get('conflict_of_interest', []) for f in founders):
                    failures.append(f"D: founders missing CoI for entrant {e['id']}")
                if not all(f['id'] in e.get('conflict_of_interest', []) for f in founders):
                    failures.append(f"D: entrant {e['id']} missing CoI for founders")
            if num_years >= 2 and not any(len(p.get('review_history', [])) > 1 for p in papers):
                failures.append('D: no resubmissions found despite resubmission enabled')
        else:  # A
            if entrants:
                failures.append(f'A: unexpected entrants {[e["id"] for e in entrants[:3]]}')
            if len(agents) != 302:
                failures.append(f'A: population {len(agents)} != 302')
            bad = [p['id'] for p in papers if len(p.get('review_history', [])) > 1]
            if bad:
                failures.append(f'A: {len(bad)} papers have >1 attempt despite '
                                f'--disable_resubmission (e.g. {bad[:3]})')
            for y in range(1, num_years + 1):
                if f'Year {y} Phase 0: resubmissions disabled' not in log:
                    failures.append(f'A: missing Phase-0 disabled log for year {y}')
    return failures


# ---------------------------------------------------------------- analysis

def run_analysis():
    for key, spec in CELLS.items():
        subprocess.run([sys.executable, '-m', 'utopia.analysis.resubmission_stats',
                        '--run_dir', os.path.join('outputs', spec['name']),
                        '--num_years', str(NUM_YEARS),
                        '--out_dir',
                        os.path.join('outputs', 'docs',
                                     f"resubmission_stats_{spec['name']}")],
                       cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'utopia.analysis.influx_factorial'],
                   cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, '-m', 'utopia.visual.plot_influx_factorial'],
                   cwd=REPO_ROOT, check=True)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preflight_only', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help=f'cells {SMOKE_CELLS}, {SMOKE_YEARS} years, + structural gates')
    ap.add_argument('--skip_preflight_tests', action='store_true')
    ap.add_argument('--no_analysis', action='store_true')
    ap.add_argument('--check_smoke_only', action='store_true',
                    help='re-run the smoke gates on existing smoke outputs and exit')
    args = ap.parse_args()

    if args.check_smoke_only:
        failures = check_smoke()
        print('\n'.join(f'GATE FAIL: {f}' for f in failures) or 'ALL SMOKE GATES PASS')
        sys.exit(1 if failures else 0)

    failures, git = preflight(skip_tests=args.skip_preflight_tests)
    if failures:
        print('\n'.join(f'PREFLIGHT FAIL: {f}' for f in failures))
        sys.exit(1)
    print(f"[launcher] preflight OK (commit {git['commit'][:12]}, "
          f"{len(git['dirty_files'])} dirty files)")
    if args.preflight_only:
        return

    server_proc, server_info = start_server()
    try:
        wait_for_server(server_proc)
        if args.smoke:
            name_override = lambda k: f'influx2x2_smoke_{k}_seed{SEED}'
            procs = launch_cells(SMOKE_CELLS, SMOKE_YEARS, name_override,
                                 git=git, server_info=server_info)
            codes = monitor(procs) if procs else {}
            failures = check_smoke()
            print('\n'.join(f'GATE FAIL: {f}' for f in failures) or 'ALL SMOKE GATES PASS')
            if failures or any(codes.values()):
                sys.exit(1)
            # resume/idempotence gate: extend smoke-D by one year from checkpoint 3
            print('[launcher] resume gate: extending smoke-D to year 4')
            procs = launch_cells(('D',), SMOKE_YEARS + 1, name_override,
                                 git=git, server_info=server_info)
            codes = monitor(procs) if procs else {}
            resume_failures = check_smoke(num_years=SMOKE_YEARS + 1)
            resume_failures = [f for f in resume_failures if not f.startswith('A')]
            print('\n'.join(f'RESUME GATE FAIL: {f}' for f in resume_failures)
                  or 'RESUME GATE PASS')
            sys.exit(1 if (resume_failures or any(codes.values())) else 0)
        else:
            procs = launch_cells('ABCD', NUM_YEARS, git=git, server_info=server_info)
            codes = monitor(procs) if procs else {}
            bad = {k: c for k, c in codes.items() if c}
            if bad:
                print(f'[launcher] FAILED cells: {bad} — rerun this launcher to resume them')
                sys.exit(1)
    finally:
        kill_server(server_proc)

    if not args.smoke and not args.no_analysis:
        run_analysis()
    print('[launcher] done')


if __name__ == '__main__':
    main()
