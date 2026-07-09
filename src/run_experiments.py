#!/usr/bin/env python3
"""Multi-seed experiment launcher for the exploration-vs-exploitation study.

Queues sequential run_simulation.py runs against the persistent vLLM server,
one seed at a time, with per-run logs, resume-on-failure, and post-run
integrity validation. This is the single Python orchestration entry point
permitted under src/ by the legacy-layout exception.

Example:
    python src/run_experiments.py --stage pilot --seeds 21 22 23 \
        --num_institutions 6 --num_years 6 --num_conferences 3
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "Qwen/Qwen3-32B"
VLLM_URL = "http://localhost:8032/v1"


def experiment_id(stage, seed, num_institutions, num_years, mechanism):
    n = num_institutions * 5
    return f"explore_{stage}_qwen3_32b_{mechanism}_i{num_institutions}_n{n}_y{num_years}_seed{seed}"


def run_one(stage, seed, args) -> dict:
    mechanism = 'neutral'
    if args.funding_intervention or args.citation_intervention:
        mechanism = (f"fund{'015' if args.funding_intervention else '0'}_"
                     f"cite{'on' if args.citation_intervention else 'off'}")
    exp_id = experiment_id(stage, seed, args.num_institutions, args.num_years, mechanism)
    log_dir = os.path.join(REPO_ROOT, 'outputs', 'logs', exp_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'simulation.log')

    cmd = [
        sys.executable, 'run_simulation.py',
        '--experiment_name', 'exploration_vs_exploitation',
        '--experiment_stage', stage,
        '--seed', str(seed),
        '--population_mode', 'university_only',
        '--num_institutions', str(args.num_institutions),
        '--researchers_per_institution', '5',
        '--num_years', str(args.num_years),
        '--num_conferences', str(args.num_conferences),
        '--start_year', str(args.start_year),
        '--model', MODEL,
        '--vllm_url', VLLM_URL,
        '--batch_size', str(args.batch_size),
        '--industry_funding_mode', 'performance',
        '--funding_allocation_mode', 'fixed',
    ]
    if args.funding_intervention:
        cmd.append('--funding_intervention')
    if args.citation_intervention:
        cmd.append('--citation_intervention')

    with open(os.path.join(log_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(cmd) + '\n')

    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', args.client_gpus)

    record = {'experiment_id': exp_id, 'seed': seed, 'attempts': 0,
              'status': 'failed', 'log': log_path}
    for attempt in range(args.max_attempts):
        record['attempts'] += 1
        t0 = time.time()
        print(f"[launcher] {exp_id} attempt {attempt + 1}/{args.max_attempts} "
              f"-> {log_path}", flush=True)
        with open(log_path, 'a') as log_file:
            log_file.write(f"\n===== attempt {attempt + 1} @ {time.ctime()} =====\n")
            log_file.write(' '.join(cmd) + '\n')
            log_file.flush()
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                  stdout=log_file, stderr=subprocess.STDOUT)
        record['elapsed_seconds'] = round(time.time() - t0, 1)
        if proc.returncode == 0:
            record['status'] = 'complete'
            break
        print(f"[launcher] {exp_id} exited {proc.returncode}; "
              f"{'resuming from checkpoint' if attempt + 1 < args.max_attempts else 'giving up'}",
              flush=True)
        time.sleep(args.retry_wait)

    # Post-run integrity gate
    integrity_path = os.path.join(REPO_ROOT, 'outputs', 'docs', exp_id, 'integrity_checks.json')
    if record['status'] == 'complete' and os.path.exists(integrity_path):
        with open(integrity_path) as f:
            checks = json.load(f)
        record['integrity_passed'] = all(c.get('passed') for c in checks)
        record['integrity_years'] = len(checks)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=['smoke', 'pilot', 'scalecheck', 'calibration',
                                 'confirmatory', 'manyworld', 'mechanism'])
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--num_institutions', type=int, required=True)
    parser.add_argument('--num_years', type=int, required=True)
    parser.add_argument('--num_conferences', type=int, default=6)
    parser.add_argument('--start_year', type=int, default=2016)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_attempts', type=int, default=3,
                        help='Total attempts per seed; retries resume from checkpoints')
    parser.add_argument('--retry_wait', type=int, default=30)
    parser.add_argument('--client_gpus', type=str, default='',
                        help='CUDA_VISIBLE_DEVICES for the simulation client (default: CPU only)')
    parser.add_argument('--funding_intervention', action='store_true')
    parser.add_argument('--citation_intervention', action='store_true')
    args = parser.parse_args()

    records = [run_one(args.stage, seed, args) for seed in args.seeds]

    summary_dir = os.path.join(REPO_ROOT, 'outputs', 'docs', f'explore_{args.stage}_launcher')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'launch_summary.json')
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
    with open(summary_path, 'w') as f:
        json.dump(existing + records, f, indent=2)

    print(json.dumps(records, indent=2))
    if any(r['status'] != 'complete' or not r.get('integrity_passed', True) for r in records):
        sys.exit(1)


if __name__ == '__main__':
    main()
