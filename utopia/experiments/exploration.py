#!/usr/bin/env python3
"""Multi-seed experiment launcher for the exploration-vs-exploitation study.

Queues sequential utopia/simulation.py runs against a persistent vLLM server
(local or Modal — selected purely via --vllm_url / $VLLM_URL), with per-run
logs, resume-on-failure, post-run integrity validation, machine-enforced
stage gates, reproducible-code enforcement, and a paired-conditions mode for
the E4b composition intervention.

Machine gates (checked BEFORE any model request):
  calibration            -> outputs/docs/science_utopia_execution/post_e3_gate.json
                            must have passed: true; more than one seed additionally
                            requires profiling_check_passed: true.
  confirmatory/manyworld -> outputs/docs/explore_final_freeze/final_freeze.json
                            must exist, and its code_commit_hash must equal HEAD.

Reproducibility: refuses to launch from a dirty working tree (CLAUDE.md and
generated artifacts excluded) unless --allow_dirty_with_patch points to an
archived code-state JSON whose patch hash matches the current diff.

Examples:
    python -m utopia.experiments.exploration --stage calibration --seeds 501 \
        --num_institutions 100 --num_years 10
    python -m utopia.experiments.exploration --stage manyworld --seeds 2001 \
        --num_institutions 150 --num_years 10 \
        --paired_conditions "1,3,1;3,1,1;1,4,0;1,1,3"
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.runtime.commands import module_command

from utopia.utils.paths import project_root
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

REPO_ROOT = str(project_root(__file__))
MODEL = "Qwen/Qwen3-32B"
DEFAULT_VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8032/v1")
POST_E3_GATE = os.path.join(REPO_ROOT, 'outputs/docs/science_utopia_execution/post_e3_gate.json')
FINAL_FREEZE_GATE = os.path.join(REPO_ROOT, 'outputs/docs/explore_final_freeze/final_freeze.json')

# Working-tree paths that never count as dirty (docs/artifacts, never executable code)
DIRTY_EXEMPT_PREFIXES = ('CLAUDE.md', 'outputs', 'data', 'wandb', '.vscode', 'tests/__pycache__')


def _git(args_):
    return subprocess.run(['git'] + args_, cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def check_clean_tree(allow_patch_path=None):
    """Hard gate: executable code must be committed (or match an archived patch)."""
    dirty = [l for l in _git(['status', '--porcelain']).splitlines()
             if l[3:] and not l[3:].startswith(DIRTY_EXEMPT_PREFIXES)]
    if not dirty:
        return _git(['rev-parse', 'HEAD'])
    if allow_patch_path:
        snap = json.load(open(allow_patch_path))
        current_patch = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=REPO_ROOT,
                                       capture_output=True).stdout
        current_hash = hashlib.sha256(current_patch).hexdigest()
        if current_hash == snap.get('tracked_patch_sha256'):
            print(f"[gate] dirty tree matches archived patch {current_hash[:12]} — approved profiling exception")
            return _git(['rev-parse', 'HEAD']) + f"+patch:{current_hash[:12]}"
        sys.exit(f"[gate] REFUSED: dirty tree does not match archived patch "
                 f"({current_hash[:12]} != {snap.get('tracked_patch_sha256', '')[:12]})")
    sys.exit(f"[gate] REFUSED: working tree has uncommitted executable changes:\n" + "\n".join(dirty)
             + "\nCommit them or pass --allow_dirty_with_patch <archived snapshot json>.")


def check_stage_gate(stage, n_seeds):
    """Machine-readable gate enforcement; missing/stale/failed gate = hard exit."""
    if stage in ('smoke', 'pilot', 'scalecheck'):
        return
    if stage == 'calibration':
        if not os.path.exists(POST_E3_GATE):
            sys.exit(f"[gate] REFUSED: {POST_E3_GATE} missing — post-E3 audit incomplete")
        gate = json.load(open(POST_E3_GATE))
        if gate.get('passed') is not True:
            sys.exit(f"[gate] REFUSED: post_e3_gate passed={gate.get('passed')}")
        if n_seeds > 1 and gate.get('profiling_check_passed') is not True:
            sys.exit("[gate] REFUSED: only ONE calibration profiling seed allowed until "
                     "profiling_check_passed: true is recorded in post_e3_gate.json")
        return
    if stage in ('confirmatory', 'manyworld', 'mechanism'):
        if not os.path.exists(FINAL_FREEZE_GATE):
            sys.exit(f"[gate] REFUSED: {FINAL_FREEZE_GATE} missing — final freeze not created")
        freeze = json.load(open(FINAL_FREEZE_GATE))
        head = _git(['rev-parse', 'HEAD'])
        for key in ('code_commit_hash', 'scientific_config_hash', 'prompt_hash',
                    'analysis_spec_hash', 'e4a_seed_count', 'e4b_conditions',
                    'e4b_required_paired_sets'):
            if key not in freeze:
                sys.exit(f"[gate] REFUSED: final_freeze.json missing field {key}")
        if freeze['code_commit_hash'] != head:
            sys.exit(f"[gate] REFUSED: HEAD {head[:12]} != frozen commit "
                     f"{freeze['code_commit_hash'][:12]}")
        return freeze


def experiment_id(stage, seed, num_institutions, num_years, mechanism, mix=None, panel=None):
    n = num_institutions * 5
    mix_tag = ''
    if mix:
        mix_tag = '_mix' + mix.replace(',', '')
    if panel is not None:
        mix_tag += f"_panel{panel}"
    return (f"explore_{stage}_qwen3_32b_{mechanism}_i{num_institutions}_n{n}"
            f"_y{num_years}_seed{seed}{mix_tag}")


def run_one(stage, seed, args, mix=None) -> dict:
    mechanism = 'neutral'
    if args.funding_intervention or args.citation_intervention:
        mechanism = (f"fund{'015' if args.funding_intervention else '0'}_"
                     f"cite{'on' if args.citation_intervention else 'off'}")
    exp_id = experiment_id(stage, seed, args.num_institutions, args.num_years, mechanism, mix,
                           panel=args.funding_panel_max_apps)
    log_dir = os.path.join(REPO_ROOT, 'outputs', 'logs', exp_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'simulation.log')

    cmd = module_command(sys.executable, 'utopia', '--experiment_name', 'exploration_vs_exploitation', '--experiment_stage', stage, '--seed', str(seed), '--population_mode', 'university_only', '--num_institutions', str(args.num_institutions), '--researchers_per_institution', '5', '--num_years', str(args.num_years), '--num_conferences', str(args.num_conferences), '--start_year', str(args.start_year), '--model', MODEL, '--vllm_url', args.vllm_url, '--batch_size', str(args.batch_size), '--industry_funding_mode', 'performance', '--funding_allocation_mode', 'fixed')
    if mix:
        cmd += ['--strategy_mix', mix]
    if args.funding_panel_max_apps is not None:
        cmd += ['--funding_panel_max_apps', str(args.funding_panel_max_apps)]
    if args.funding_intervention:
        cmd.append('--funding_intervention')
    if args.citation_intervention:
        cmd.append('--citation_intervention')

    with open(os.path.join(log_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(cmd) + '\n')

    env = dict(os.environ)
    env.setdefault('CUDA_VISIBLE_DEVICES', args.client_gpus)

    record = {'experiment_id': exp_id, 'seed': seed, 'mix': mix, 'attempts': 0,
              'status': 'failed', 'log': log_path, 'launch_commit': args.launch_commit}
    for attempt in range(args.max_attempts):
        record['attempts'] += 1
        t0 = time.time()
        print(f"[launcher] {exp_id} attempt {attempt + 1}/{args.max_attempts} "
              f"-> {log_path}", flush=True)
        with open(log_path, 'a') as log_file:
            log_file.write(f"\n===== attempt {attempt + 1} @ {time.ctime()} "
                           f"commit={args.launch_commit} url={args.vllm_url} =====\n")
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
    parser.add_argument('--vllm_url', type=str, default=DEFAULT_VLLM_URL,
                        help='OpenAI-compatible endpoint (local or Modal); '
                             'default $VLLM_URL or localhost')
    parser.add_argument('--paired_conditions', type=str, default=None,
                        help='Semicolon-separated strategy mixes, e.g. "1,3,1;3,1,1;1,4,0;1,1,3". '
                             'Every seed runs ALL conditions; the seed counts as complete only '
                             'if every condition completed and passed integrity (E4b rule).')
    parser.add_argument('--max_attempts', type=int, default=3,
                        help='Total attempts per run; retries resume from checkpoints')
    parser.add_argument('--retry_wait', type=int, default=30)
    parser.add_argument('--client_gpus', type=str, default='',
                        help='CUDA_VISIBLE_DEVICES for the simulation client (default: CPU only)')
    parser.add_argument('--funding_panel_max_apps', type=int, default=None,
                        help='Panelization override for the fidelity experiment (0 = global ranking)')
    parser.add_argument('--allow_dirty_with_patch', type=str, default=None,
                        help='Archived code-state JSON approving a dirty-tree profiling run')
    parser.add_argument('--funding_intervention', action='store_true')
    parser.add_argument('--citation_intervention', action='store_true')
    args = parser.parse_args()

    args.launch_commit = check_clean_tree(args.allow_dirty_with_patch)
    check_stage_gate(args.stage, len(args.seeds))

    conditions = args.paired_conditions.split(';') if args.paired_conditions else [None]
    records = []
    for seed in args.seeds:
        seed_records = [run_one(args.stage, seed, args, mix=mix) for mix in conditions]
        set_ok = all(r['status'] == 'complete' and r.get('integrity_passed', False)
                     for r in seed_records)
        for r in seed_records:
            r['paired_set_complete'] = set_ok
        if args.paired_conditions and not set_ok:
            print(f"[launcher] seed {seed}: paired set INCOMPLETE — this seed does not count "
                  f"toward the frozen replication requirement and must be replaced", flush=True)
        records.extend(seed_records)

    summary_dir = os.path.join(REPO_ROOT, 'outputs', 'docs', f'explore_{args.stage}_launcher')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'launch_summary.json')
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
    write_json_file(existing + records, summary_path, indent=2)

    print(json.dumps(records, indent=2))
    if any(r['status'] != 'complete' or not r.get('integrity_passed', True) for r in records):
        sys.exit(1)


if __name__ == '__main__':
    main()
