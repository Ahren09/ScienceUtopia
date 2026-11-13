"""Matthew-effect funding-cutoff RD experiment driver.

Stages (all Python, no bash wrappers):
  launch   -- run compact source simulations (funding_cutoff stage) with the
              --log_funding_applications flag so full pre-decision rankings
              are persisted. Smoke world first, then <= 3 main seeds.
  validate -- smoke-validation gates (cutoff reconstruction reproduces
              recorded winners, linkage, follow-up correctness).
  extract  -- build the application-level RD dataset -> data/funding_cutoff/.
  gate     -- sample-size gate report (prereg_v1).
  analyze  -- full preregistered analysis + figures + rd_results.json.

Prereg: outputs/docs/funding_cutoff_prereg/prereg_v1.json (frozen).
Existing E4 launchers/configs/freezes are never touched. Reuses the
simulator verbatim; the only mechanism-side change is the default-off
application logging flag.

Examples:
  python -m utopia.experiments.funding_cutoff launch --smoke --vllm_url http://localhost:8034/v1
  python -m utopia.experiments.funding_cutoff launch --seeds 3001 3002 3003 --vllm_url http://localhost:8034/v1
  python -m utopia.experiments.funding_cutoff validate --experiment_id explore_funding_cutoff_qwen3_32b_neutral_i5_n25_y2_seed3100
  python -m utopia.experiments.funding_cutoff extract --experiment_ids <id1> <id2> <id3> --num_years 6
  python -m utopia.experiments.funding_cutoff analyze
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.runtime.commands import module_command

from utopia.utils.paths import project_root

import argparse
import json
import os
import subprocess
import sys
import time

REPO_ROOT = str(project_root(__file__))
from utopia.analysis.funding_cutoff import build_application_dataset, guard_write_path, guarded_makedirs, load_application_log, load_final_checkpoint, run_full_analysis, sample_size_gate, TREATMENT_MAX_FIRST_APP_YEAR

MODEL = 'Qwen/Qwen3-32B'
MAIN_CONFIG = dict(num_institutions=100, researchers_per_institution=5,
                   num_years=6, num_conferences=6, start_year=2016, batch_size=128)
SMOKE_CONFIG = dict(num_institutions=5, researchers_per_institution=5,
                    num_years=2, num_conferences=2, start_year=2016, batch_size=32)
# C0 cost calibration (funding_cutoff_cost): compact world, disjoint seed, enough
# years to observe 3-year reapplication windows for early treatment years.
CALIB_CONFIG = dict(num_institutions=40, researchers_per_institution=5,
                    num_years=5, num_conferences=6, start_year=2016, batch_size=128)
SMOKE_SEED = 3100
CALIB_SEED = 3200
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'funding_cutoff')
LAUNCH_LOG_DIR = os.path.join(REPO_ROOT, 'outputs', 'logs', 'funding_cutoff_launcher')


def experiment_id(seed: int, cfg: dict, stage: str = 'funding_cutoff', cost: int = 0) -> str:
    n = cfg['num_institutions'] * cfg['researchers_per_institution']
    cost_tag = f"_cost{cost}" if cost else ''
    return (f"explore_{stage}_qwen3_32b_neutral_i{cfg['num_institutions']}"
            f"_n{n}_y{cfg['num_years']}_seed{seed}{cost_tag}")


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=REPO_ROOT).decode().strip()
    except Exception:
        return 'unknown'


def launch(args):
    if args.smoke:
        cfg, seeds = SMOKE_CONFIG, [SMOKE_SEED]
    elif args.calib:
        cfg, seeds = CALIB_CONFIG, [CALIB_SEED]
    else:
        cfg, seeds = MAIN_CONFIG, args.seeds
    assert seeds, 'no seeds given'
    assert len(seeds) <= 3, 'prereg allows at most 3 main source seeds'
    assert args.cost == 0 or args.stage == 'funding_cutoff_cost', \
        'nonzero funding_application_cost requires the funding_cutoff_cost stage'
    guarded_makedirs(LAUNCH_LOG_DIR)

    records = []
    for seed in seeds:
        exp_id = experiment_id(seed, cfg, stage=args.stage, cost=args.cost)
        # per-run simulator logs live in the canonical run log dir
        run_log_dir = os.path.join(REPO_ROOT, 'outputs', 'logs', exp_id)
        guarded_makedirs(run_log_dir)
        log_path = os.path.join(run_log_dir, 'simulation.log')

        cmd = module_command(sys.executable, 'utopia', '--experiment_name', 'exploration_vs_exploitation', '--experiment_stage', args.stage, '--seed', str(seed), '--population_mode', 'university_only', '--num_institutions', str(cfg['num_institutions']), '--researchers_per_institution', str(cfg['researchers_per_institution']), '--num_years', str(cfg['num_years']), '--num_conferences', str(cfg['num_conferences']), '--start_year', str(cfg['start_year']), '--model', MODEL, '--vllm_url', args.vllm_url, '--batch_size', str(cfg['batch_size']), '--industry_funding_mode', 'performance', '--funding_allocation_mode', 'fixed', '--log_funding_applications')
        if args.cost:
            cmd += ['--funding_application_cost', str(args.cost)]
        with open(guard_write_path(os.path.join(run_log_dir, 'command.txt')), 'w') as f:
            f.write(' '.join(cmd) + '\n')

        env = dict(os.environ)
        # simulation client is CPU-only; the vLLM server owns the GPUs
        env['CUDA_VISIBLE_DEVICES'] = args.client_gpus

        rec = {'experiment_id': exp_id, 'seed': seed, 'config': cfg,
               'stage': args.stage, 'funding_application_cost': args.cost,
               'model': MODEL, 'vllm_url': args.vllm_url,
               'launch_commit': git_commit(), 'attempts': 0, 'status': 'failed',
               'command': ' '.join(cmd), 'log': log_path,
               'client_gpus': args.client_gpus}
        for attempt in range(args.max_attempts):
            rec['attempts'] += 1
            t0 = time.time()
            print(f"[funding_cutoff] {exp_id} attempt {attempt + 1}/{args.max_attempts} -> {log_path}",
                  flush=True)
            with open(log_path, 'a') as log_file:
                log_file.write(f"\n===== attempt {attempt + 1} @ {time.ctime()} "
                               f"commit={rec['launch_commit']} url={args.vllm_url} =====\n")
                log_file.write(' '.join(cmd) + '\n')
                log_file.flush()
                proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                      stdout=log_file, stderr=subprocess.STDOUT)
            rec['elapsed_seconds'] = round(time.time() - t0, 1)
            if proc.returncode == 0:
                rec['status'] = 'complete'
                break
            print(f"[funding_cutoff] {exp_id} exited {proc.returncode}; "
                  f"{'resuming from checkpoint' if attempt + 1 < args.max_attempts else 'giving up'}",
                  flush=True)
            time.sleep(args.retry_wait)
        records.append(rec)

    label = 'smoke' if args.smoke else ('calib' if args.calib else 'main')
    if args.cost:
        label += f"_cost{args.cost}"
    summary_path = guard_write_path(os.path.join(
        LAUNCH_LOG_DIR, f"launch_{label}_{int(time.time())}.json"))
    write_json_file(records, summary_path, indent=1)
    print(f"[funding_cutoff] launch summary -> {summary_path}")
    if any(r['status'] != 'complete' for r in records):
        sys.exit(1)


def validate(args):
    """Smoke-validation gates on one completed run (prereg smoke_validation_gates)."""
    exp_id = args.experiment_id
    run_dir = os.path.join(args.outputs_dir, 'checkpoints', exp_id)
    docs_dir = os.path.join(args.outputs_dir, 'docs', exp_id)
    checks = {}

    apps = load_application_log(run_dir)
    ckpt = load_final_checkpoint(run_dir)
    agents = {a['id']: a for a in ckpt['ecosystem_data']['agents']}

    # 1. all applications logged, winners AND losers, with required fields
    required = {'program_id', 'panel_index', 'applicant_id', 'position',
                'normalized_raw_score', 'n_panel', 'num_winners', 'funded', 'year'}
    checks['fields_present'] = required <= set(apps.columns)
    checks['has_losers'] = bool((~apps['funded']).any())
    checks['has_winners'] = bool(apps['funded'].any())

    # 2. positions are a complete 1..n sequence per panel and the funded flag
    #    equals (position <= num_winners)
    ok_seq, ok_flag = True, True
    for _, g in apps.groupby(['year', 'program_id', 'panel_index']):
        n = int(g['n_panel'].iloc[0])
        ok_seq &= sorted(g['position']) == list(range(1, n + 1))
        ok_flag &= bool(((g['position'] <= g['num_winners']) == g['funded']).all())
    checks['positions_complete'] = ok_seq
    checks['funded_matches_cutoff_rule'] = ok_flag

    # 3. cutoff reconstruction exactly reproduces recorded winners per program-year
    mismatches = []
    for (pid, yr), g in apps.groupby(['program_id', 'year']):
        logged = set(g.loc[g['funded'], 'applicant_id'])
        recorded = {aid for aid, a in agents.items()
                    if any(int(e['year']) == int(yr)
                           for e in (a.get('funding_success_history') or {}).get(pid, []))}
        if logged != recorded:
            mismatches.append({'program': pid, 'year': int(yr),
                               'logged_minus_recorded': sorted(logged - recorded),
                               'recorded_minus_logged': sorted(recorded - logged)})
    checks['winner_reconstruction_exact'] = not mismatches
    checks['winner_mismatches'] = mismatches

    # 4. applicant IDs persist / are linkable
    checks['applicants_linkable'] = bool(set(apps['applicant_id']) <= set(agents))

    # 5. no post-treatment variable enters the running variable: positions are
    #    derived from normalized_raw_score ordering (neutral: monotone in rank)
    mono = True
    for _, g in apps.groupby(['year', 'program_id', 'panel_index']):
        gs = g.sort_values('position')
        mono &= bool((gs['normalized_raw_score'].diff().dropna() <= 1e-9).all())
    checks['position_monotone_in_pre_decision_score'] = mono

    # 6. fallback accounting present
    checks['fallback_flag_present'] = 'fallback_ranking' in apps.columns
    checks['n_fallback_records'] = int(apps['fallback_ranking'].sum())

    # 7. downstream reconstruction on a sample: awards visible in history
    sample = apps[apps['funded']].head(5)
    ok_hist = all(
        any(int(e['year']) == int(r['year'])
            for e in (agents[r['applicant_id']].get('funding_success_history') or {})
            .get(r['program_id'], []))
        for _, r in sample.iterrows())
    checks['sample_award_reconstruction'] = ok_hist

    # 8. parquet extracts exist for outcome linkage
    checks['agent_year_parquet'] = os.path.exists(os.path.join(docs_dir, 'agent_year.parquet'))
    checks['paper_parquet'] = os.path.exists(os.path.join(docs_dir, 'paper.parquet'))

    checks['all_passed'] = all(v is True for k, v in checks.items()
                               if isinstance(v, bool))
    out_dir = guarded_makedirs(os.path.join(REPO_ROOT, 'outputs', 'docs', 'funding_cutoff_smoke'))
    out = guard_write_path(os.path.join(out_dir, f'validate_{exp_id}.json'))
    write_json_file(checks, out, indent=1)
    print(json.dumps(checks, indent=1))
    print(f"[funding_cutoff] validation -> {out}")
    if not checks['all_passed']:
        sys.exit(1)


def calibgate(args):
    """C0 mechanism gates for one calibration run (funding_cutoff_cost).

    Deterministic, mechanism-only criteria — the funding RD effect, its sign,
    and its p-value are NEVER computed here (prereg discipline).
    """
    import pandas as pd
    exp_id = args.experiment_id
    cost = args.cost
    run_dir = os.path.join(args.outputs_dir, 'checkpoints', exp_id)
    ny = args.num_years

    df, excl = build_application_dataset([exp_id], outputs_dir=args.outputs_dir,
                                         num_years=ny)
    apps_log = load_application_log(run_dir)
    n_apps_by_year = apps_log.groupby('year').size().to_dict()

    # cost events (affordability diagnostics)
    import glob as _glob
    cost_rows = []
    for p in sorted(_glob.glob(os.path.join(run_dir, 'funding_costs_year_*.jsonl'))):
        with open(p) as f:
            cost_rows.extend(json.loads(l) for l in f if l.strip())
    cost_df = pd.DataFrame(cost_rows)
    n_unaffordable = int(cost_df['withdrawn_unaffordable'].sum()) if len(cost_rows) else 0

    near = df[df['x'].abs() <= 2]
    losers = near[near['funded'] == 0]
    # follow-up-eligible: full 3-year window inside the run
    horizon_ok = near['treatment_year'] <= ny - 3
    near_h = near[horizon_ok]
    losers_h = losers[losers['treatment_year'] <= ny - 3]

    reapply_all = float(near_h['any_application_3y'].mean()) if len(near_h) else float('nan')
    loser_no_reapply = float(1 - losers_h['any_application_3y'].mean()) if len(losers_h) else float('nan')
    attrition = float(near_h['attrition_by_3y'].mean()) if len(near_h) else float('nan')

    agent_year = pd.read_parquet(os.path.join(args.outputs_dir, 'docs', exp_id,
                                              'agent_year.parquet'))
    paper = pd.read_parquet(os.path.join(args.outputs_dir, 'docs', exp_id,
                                         'paper.parquet'))
    res = agent_year['funding'].dropna()
    share_below_paper_thr = float((res < 20).mean())
    share_below_app_thr = float((res < cost).mean())
    share_near_constraints = float((res < 20 + cost).mean())

    # projected near-cutoff obs across three FULL seeds (500 researchers, 6y):
    # near-cutoff first-year application count scales ~ with cohort size.
    calib_researchers = 200
    projected = int(round(len(near) * (500 / calib_researchers) * 3))

    gates = {
        'g1_reapplication_not_structurally_100pct': bool(reapply_all < 1.0),
        'g2_loser_nonreapply_ge_10pct_or_rate_lt_95pct':
            bool((loser_no_reapply >= 0.10) or (reapply_all < 0.95)),
        'g3_attrition_nonzero_or_resources_near_constraints':
            bool((attrition > 0) or (share_near_constraints >= 0.10)),
        'g4_attrition_below_30pct': bool(attrition < 0.30),
        'g5_ecosystem_valid': bool(
            all(v > 0 for v in n_apps_by_year.values())
            and excl['panels_fallback'] == 0
            and excl['panels_winner_mismatch'] == 0
            and len(paper) > 0
            and projected >= 150),
        'g6_no_integrity_problems': bool(
            excl['panels_fallback'] == 0 and excl['panels_winner_mismatch'] == 0),
    }
    result = {
        'experiment_id': exp_id, 'cost': cost, 'num_years': ny,
        'passes_all_gates': all(gates.values()),
        'gates': gates,
        'diagnostics': {
            'near_cutoff_n': int(len(near)),
            'near_cutoff_eligible_n': int(len(near_h)),
            'reapplication_rate_eligible': reapply_all,
            'loser_nonreapplication_share': loser_no_reapply,
            'attrition_3y_near_cutoff': attrition,
            'share_agent_years_below_paper_threshold': share_below_paper_thr,
            'share_agent_years_below_application_cost': share_below_app_thr,
            'share_agent_years_below_paper_plus_cost': share_near_constraints,
            'n_unaffordable_withdrawals': n_unaffordable,
            'applications_by_year': {str(k): int(v) for k, v in n_apps_by_year.items()},
            'papers_total': int(len(paper)),
            'projected_near_cutoff_across_3_full_seeds': projected,
            'exclusions': excl,
        },
        'note': 'funding RD effect/p-value deliberately not computed at calibration',
    }
    out_dir = guarded_makedirs(os.path.join(REPO_ROOT, 'outputs', 'docs',
                                            'funding_cutoff_cost_calibration'))
    out = guard_write_path(os.path.join(out_dir, f'gate_cost{cost}.json'))
    write_json_file(result, out, indent=1)
    print(json.dumps(result, indent=1))
    return result


def _data_dir(args) -> str:
    return os.path.join(REPO_ROOT, args.data_dir)


def extract(args):
    guarded_makedirs(_data_dir(args))
    df, excl = build_application_dataset(args.experiment_ids,
                                         outputs_dir=args.outputs_dir,
                                         num_years=args.num_years)
    name = f'{args.dataset_name}.parquet'
    out_parquet = guard_write_path(os.path.join(_data_dir(args), name))
    df.to_parquet(out_parquet, index=False)
    meta = {
        'experiment_ids': args.experiment_ids, 'num_years': args.num_years,
        'n_rows': int(len(df)), 'exclusions': excl,
        'treatment_max_first_app_year': TREATMENT_MAX_FIRST_APP_YEAR,
        'extract_commit': git_commit(),
    }
    out_meta = guard_write_path(os.path.join(_data_dir(args),
                                             f'{args.dataset_name}_meta.json'))
    write_json_file(meta, out_meta, indent=1)
    print(json.dumps(meta, indent=1))
    print(f"[funding_cutoff] dataset -> {out_parquet}")


def gate(args):
    import pandas as pd
    df = pd.read_parquet(os.path.join(_data_dir(args), f'{args.dataset_name}.parquet'))
    g = sample_size_gate(df)
    out_dir = guarded_makedirs(os.path.join(REPO_ROOT, 'outputs', 'docs', args.out_tag))
    out = guard_write_path(os.path.join(out_dir, 'sample_size_gate.json'))
    write_json_file(g, out, indent=1)
    print(json.dumps(g, indent=1))


def analyze(args):
    import pandas as pd
    df = pd.read_parquet(os.path.join(_data_dir(args), f'{args.dataset_name}.parquet'))
    with open(os.path.join(_data_dir(args), f'{args.dataset_name}_meta.json')) as f:
        meta = json.load(f)
    out_docs = os.path.join(REPO_ROOT, 'outputs', 'docs', args.out_tag)
    out_visual = os.path.join(REPO_ROOT, 'outputs', 'visual', args.out_tag)
    results = run_full_analysis(df, out_docs, out_visual,
                                exclusions=meta.get('exclusions'))
    p = results['primary']
    print(f"[funding_cutoff] PRIMARY {p['endpoint']}: tau={p['tau']:.3f} "
          f"se={p['se']:.3f} CI=[{p['ci_lo']:.3f},{p['ci_hi']:.3f}] p={p['p']:.4g} "
          f"n={p['n']} (h={p['h']})")
    print(f"[funding_cutoff] results -> {out_docs}/rd_results.json, figures -> {out_visual}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('launch')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--calib', action='store_true',
                   help='C0 cost-calibration world (seed 3200, compact config)')
    p.add_argument('--stage', default='funding_cutoff',
                   choices=['funding_cutoff', 'funding_cutoff_cost'])
    p.add_argument('--cost', type=int, default=0,
                   help='funding_application_cost passed to the simulator')
    p.add_argument('--seeds', type=int, nargs='*', default=[])
    p.add_argument('--vllm_url', required=True)
    p.add_argument('--client_gpus', default='', help='CUDA_VISIBLE_DEVICES for the sim client (default: CPU-only)')
    p.add_argument('--max_attempts', type=int, default=3)
    p.add_argument('--retry_wait', type=int, default=60)
    p.set_defaults(func=launch)

    p = sub.add_parser('validate')
    p.add_argument('--experiment_id', required=True)
    p.add_argument('--outputs_dir', default='outputs')
    p.set_defaults(func=validate)

    p = sub.add_parser('extract')
    p.add_argument('--experiment_ids', nargs='+', required=True)
    p.add_argument('--outputs_dir', default='outputs')
    p.add_argument('--num_years', type=int, default=6)
    p.add_argument('--data_dir', default='data/funding_cutoff')
    p.add_argument('--dataset_name', default='applications')
    p.set_defaults(func=extract)

    p = sub.add_parser('calibgate')
    p.add_argument('--experiment_id', required=True)
    p.add_argument('--cost', type=int, required=True)
    p.add_argument('--num_years', type=int, default=5)
    p.add_argument('--outputs_dir', default='outputs')
    p.set_defaults(func=calibgate)

    p = sub.add_parser('gate')
    p.add_argument('--data_dir', default='data/funding_cutoff')
    p.add_argument('--dataset_name', default='applications')
    p.add_argument('--out_tag', default='funding_cutoff_final')
    p.set_defaults(func=gate)

    p = sub.add_parser('analyze')
    p.add_argument('--data_dir', default='data/funding_cutoff')
    p.add_argument('--dataset_name', default='applications')
    p.add_argument('--out_tag', default='funding_cutoff_final')
    p.set_defaults(func=analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
