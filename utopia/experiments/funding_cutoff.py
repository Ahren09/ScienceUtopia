"""Funding cutoff experiment and reusable scientific operations."""

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
# C0 cost calibration (funding_cutoff_cost): compact world, disjoint seed, enough
# years to observe 3-year reapplication windows for early treatment years.


def experiment_id(seed: int, cfg: dict, stage: str = 'funding_cutoff', cost: int = 0) -> str:
    n = cfg['num_institutions'] * cfg['researchers_per_institution']
    cost_tag = f"_cost{cost}" if cost else ''
    return (f"explore_{stage}_qwen3_32b_neutral_i{cfg['num_institutions']}"
            f"_n{n}_y{cfg['num_years']}_seed{seed}{cost_tag}")


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


def main(argv=None):
    from utopia.experiments.common import main_for
    return main_for("funding_cutoff", argv)


if __name__ == "__main__":
    main()
