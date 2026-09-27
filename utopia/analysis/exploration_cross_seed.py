"""Cross-seed analysis for the exploration-vs-exploitation experiment.

The one new analysis module permitted by the legacy-layout exception (the
general checkpoint analyzer in analyze_all_results.py stays unchanged; this
module reads the structured Parquet tables written per run).

Modes:
  --multi_run_prefix explore_pilot_qwen3_32b_neutral   # discover + aggregate runs
  --power_analysis                                      # simulation-based power from calibration runs
  --manipulation_check                                  # E2/E3 go-no-go style report only

Outputs (under --out_dir):
  aggregate_results.csv, primary_hypothesis_tests.csv, manipulation_check.csv,
  balance_check.csv, ecosystem_trajectories.csv, analysis_report.md
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import read_json

from utopia.analysis.statistics import hierarchical_bootstrap
import argparse
import glob
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

STRATEGIES = ['explorer', 'exploiter', 'cautious_explorer']
PRIMARY_CONTRASTS = [('explorer', 'exploiter'),
                     ('cautious_explorer', 'exploiter'),
                     ('explorer', 'cautious_explorer')]


# ------------------------------------------------------------ run discovery

def discover_runs(prefix: str, outputs_dir: str = 'outputs'):
    """Find completed runs whose experiment_id starts with prefix.

    Returns list of dicts {experiment_id, seed, docs_dir, manifest}.
    """
    runs = []
    for docs_dir in sorted(glob.glob(os.path.join(outputs_dir, 'docs', prefix + '*'))):
        if 'invalidated' in os.path.basename(docs_dir):
            continue  # archived pre-commit runs are never analyzed
        manifest_path = os.path.join(docs_dir, 'run_manifest.json')
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        runs.append({
            'experiment_id': os.path.basename(docs_dir),
            'seed': manifest.get('run_seed'),
            'docs_dir': docs_dir,
            'status': manifest.get('status'),
            'config_hash': manifest.get('scientific_config_hash'),
            'manifest': manifest,
        })
    return runs


def validate_runs(runs, require_complete=True):
    """Check completeness and matching config hashes; returns (valid, excluded)."""
    valid, excluded = [], []
    hashes = {r['config_hash'] for r in runs if r['config_hash']}
    for r in runs:
        problems = []
        if require_complete and r['status'] != 'complete':
            problems.append(f"status={r['status']}")
        if len(hashes) > 1 and r['config_hash'] != max(hashes, key=lambda h: sum(
                1 for x in runs if x['config_hash'] == h)):
            problems.append('config_hash differs from majority')
        for table in ('agent_year', 'paper'):
            if not _table_path(r['docs_dir'], table):
                problems.append(f'missing table {table}')
        (excluded if problems else valid).append({**r, 'problems': problems})
    return valid, excluded


def _table_path(docs_dir, name):
    for ext in ('.parquet', '.csv'):
        p = os.path.join(docs_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def load_table(runs, name) -> pd.DataFrame:
    """Concatenate one named table across runs, tagging seed/experiment."""
    frames = []
    for r in runs:
        path = _table_path(r['docs_dir'], name)
        if path is None:
            continue
        df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
        df['seed'] = r['seed']
        df['experiment_id'] = r['experiment_id']
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------- core aggregation

def researcher_level(agent_year: pd.DataFrame) -> pd.DataFrame:
    """Aggregate agent-year rows to one row per researcher per seed."""
    g = agent_year.groupby(['seed', 'agent_id'])
    out = g.agg(
        strategy=('strategy', 'first'),
        institution=('institution', 'first'),
        mean_direction_distance=('direction_distance', 'mean'),
        mean_career_distance=('avg_career_distance', 'mean'),
        switch_rate=('topic_switched', 'mean'),
        total_papers=('num_papers', 'sum'),
        total_accepted=('num_accepted', 'sum'),
        final_funding=('funding', 'last'),
        total_citations=('total_citations', 'sum'),
        hit_papers=('hit_papers', 'sum'),
        years_active=('is_active', 'sum'),
    ).reset_index()
    out['acceptance_rate'] = np.where(out['total_papers'] > 0,
                                      out['total_accepted'] / out['total_papers'], np.nan)
    return out


def within_institution_contrasts(researchers: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Strategy contrasts computed within each (seed, institution) block.

    Includes the pairwise contrasts and the H-CE planned contrast
    cautious − (explorer + exploiter)/2."""
    rows = []
    for (seed, inst), block in researchers.groupby(['seed', 'institution']):
        means = block.groupby('strategy')[outcome].mean()
        for a, b in PRIMARY_CONTRASTS:
            if a in means.index and b in means.index and pd.notna(means[a]) and pd.notna(means[b]):
                rows.append({'seed': seed, 'institution': inst, 'outcome': outcome,
                             'contrast': f'{a}-{b}', 'diff': means[a] - means[b]})
        if all(s in means.index and pd.notna(means[s]) for s in STRATEGIES):
            rows.append({'seed': seed, 'institution': inst, 'outcome': outcome,
                         'contrast': 'cautious-mean(explorer,exploiter)',
                         'diff': means['cautious_explorer']
                                 - (means['explorer'] + means['exploiter']) / 2})
    return pd.DataFrame(rows)


def load_funding_received(experiment_id: str, year: int, outputs_dir: str = 'outputs') -> Dict[str, float]:
    """Cumulative funding RECEIVED (flow) per researcher from the year-Y checkpoint's
    funding_success_history — the frozen H-CE endpoint (not year-end resources)."""
    base = os.path.join(outputs_dir, 'checkpoints', experiment_id, f'checkpoint_year_{year}.json')
    ck = read_json(base if os.path.exists(base) else base + '.gz')
    out = {}
    for a in ck['ecosystem_data']['agents']:
        hist = a.get('funding_success_history') or {}
        total = sum(r['amount'] for recs in hist.values() for r in recs)
        out[a['id']] = float(total)
    return out


def load_review_scores(experiment_id: str, outputs_dir: str = 'outputs') -> pd.DataFrame:
    """Individual reviewer scores for every submission event of every paper.

    Read from the LATEST checkpoint's paper_tracker: review_history
    accumulates across years with per-event year tags, so one checkpoint
    holds the full record. Returns one row per individual review:
    year (of the submission event), paper_id, event_idx (submission
    attempt, 0-based), overall_score (1-5 rubric, decimals allowed).
    """
    ck_dir = os.path.join(outputs_dir, 'checkpoints', experiment_id)
    paths = glob.glob(os.path.join(ck_dir, 'checkpoint_year_*.json*'))
    if not paths:
        raise FileNotFoundError(f'No checkpoints under {ck_dir}')
    year_of = lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0])
    path = max(paths, key=year_of)
    ck = read_json(path)
    rows = []
    for paper in ck['paper_tracker']['papers']:
        for event_idx, event in enumerate(paper.get('review_history') or []):
            for review in event.get('reviews') or []:
                rows.append((event['year'], paper['id'], event_idx,
                             float(review['overall_score'])))
    return pd.DataFrame(rows, columns=['year', 'paper_id', 'event_idx',
                                       'overall_score'])


def review_score_trajectories(valid, out_dir: str, outputs_dir: str = 'outputs'):
    """Per seed x year distribution of individual reviewer scores.

    Tests the score-homogenization question descriptively: does the
    cross-review dispersion shrink over the simulated decade, and where does
    the score mass sit relative to the 1-5 rubric (3 = borderline accept;
    the 2.0-2.5 band reads as borderline reject)? Writes
    review_score_trajectories.csv (seed x year: n, mean, sd, percentiles,
    share in [2.0, 2.5], mean within-event reviewer SD) plus
    review_score_convergence.json (per-seed year-1/year-final values, OLS
    slope vs year, and cross-seed direction agreement for each statistic).
    """
    rows = []
    for r in valid:
        reviews = load_review_scores(r['experiment_id'], outputs_dir)
        for year, sub in reviews.groupby('year'):
            s = sub['overall_score']
            # reviewer disagreement within one submission event (~3 reviews)
            within = (sub.groupby(['paper_id', 'event_idx'])['overall_score']
                      .std(ddof=1).dropna())
            rows.append({
                'seed': r['seed'], 'year': year, 'n_reviews': len(s),
                'mean': s.mean(), 'sd': s.std(ddof=1),
                'p10': s.quantile(.10), 'p25': s.quantile(.25),
                'median': s.quantile(.50), 'p75': s.quantile(.75),
                'p90': s.quantile(.90),
                'share_borderline_reject': s.between(2.0, 2.5).mean(),
                'within_event_sd': within.mean(),
            })
    traj = pd.DataFrame(rows).sort_values(['seed', 'year'])
    csv_path = os.path.join(out_dir, 'review_score_trajectories.csv')
    traj.to_csv(csv_path, index=False)

    summary = {}
    for stat in ('sd', 'within_event_sd', 'mean', 'share_borderline_reject'):
        per_seed = {}
        for seed, sub in traj.groupby('seed'):
            per_seed[int(seed)] = {
                'first_year': float(sub[stat].iloc[0]),
                'final_year': float(sub[stat].iloc[-1]),
                'slope_per_year': float(np.polyfit(sub['year'], sub[stat], 1)[0]),
            }
        slopes = [v['slope_per_year'] for v in per_seed.values()]
        summary[stat] = {'per_seed': per_seed,
                         'seeds_with_negative_slope': int(sum(s < 0 for s in slopes)),
                         'mean_slope_per_year': float(np.mean(slopes))}
        print(f"{stat}: first->final "
              + '; '.join(f"seed {k}: {v['first_year']:.3f}->{v['final_year']:.3f}"
                          for k, v in per_seed.items())
              + f" | mean slope {summary[stat]['mean_slope_per_year']:+.4f}/yr, "
                f"{summary[stat]['seeds_with_negative_slope']}/{len(slopes)} seeds negative")
    json_path = os.path.join(out_dir, 'review_score_convergence.json')
    write_json_file(summary, json_path, indent=2)
    print(f'Wrote {csv_path} and {json_path}')








# ----------------------------------------------------- manipulation / balance

def manipulation_check(researchers: pd.DataFrame) -> pd.DataFrame:
    """Per-seed strategy means of movement metrics + ordering flags (plan 11.3)."""
    rows = []
    for seed, block in researchers.groupby('seed'):
        means = block.groupby('strategy').agg(
            direction_distance=('mean_direction_distance', 'mean'),
            career_distance=('mean_career_distance', 'mean'),
            switch_rate=('switch_rate', 'mean'),
            n=('agent_id', 'count'))
        row = {'seed': seed}
        for s in STRATEGIES:
            if s in means.index:
                row[f'{s}_direction_distance'] = means.loc[s, 'direction_distance']
                row[f'{s}_career_distance'] = means.loc[s, 'career_distance']
                row[f'{s}_switch_rate'] = means.loc[s, 'switch_rate']
                row[f'{s}_n'] = means.loc[s, 'n']
        for metric in ('direction_distance', 'career_distance'):
            try:
                row[f'ordering_ok_{metric}'] = (
                    row[f'explorer_{metric}'] > row[f'cautious_explorer_{metric}']
                    > row[f'exploiter_{metric}'])
            except KeyError:
                row[f'ordering_ok_{metric}'] = None
        # descriptive SMD explorer vs exploiter on career distance
        e = block[block.strategy == 'explorer']['mean_career_distance'].dropna()
        x = block[block.strategy == 'exploiter']['mean_career_distance'].dropna()
        if len(e) > 1 and len(x) > 1:
            pooled = np.sqrt((e.var(ddof=1) + x.var(ddof=1)) / 2)
            row['smd_explorer_vs_exploiter_career'] = float((e.mean() - x.mean()) / pooled) \
                if pooled > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def balance_check(runs) -> pd.DataFrame:
    """Initial balance from population blueprints + first-year funding."""
    rows = []
    for r in runs:
        ckpt_dir = r['docs_dir'].replace(os.sep + 'docs' + os.sep, os.sep + 'checkpoints' + os.sep)
        bp_path = os.path.join(ckpt_dir, 'population_blueprint.csv')
        if not os.path.exists(bp_path):
            continue
        bp = pd.read_csv(bp_path)
        counts = bp['strategy'].value_counts()
        per_inst = bp.groupby('institution')['strategy'].value_counts().unstack(fill_value=0)
        rows.append({
            'seed': r['seed'],
            'n_researchers': len(bp),
            'n_institutions': bp['institution'].nunique(),
            **{f'n_{s}': int(counts.get(s, 0)) for s in STRATEGIES},
            'exact_mix_every_institution': bool(
                (per_inst.get('explorer', 0) == 1).all()
                and (per_inst.get('exploiter', 0) == 3).all()
                and (per_inst.get('cautious_explorer', 0) == 1).all()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ power analysis

def power_analysis(researchers: pd.DataFrame, out_dir: str,
                   institution_grid=(100, 250, 500, 750, 1000, 1500),
                   world_grid=(5, 10, 20, 30), n_sim=2000, alpha=0.05, seed=0):
    """Simulation-based power using institution-level contrast variance estimated
    from calibration runs (plan revision: never from the 12-institution run alone).

    For each preregistered effect size, simulates institution-blocked contrast
    means at each scale and reports the fraction of simulations whose 95% CI
    excludes 0 (agent-level) — plus world-level power over seed-mean contrasts.
    """
    rng = np.random.default_rng(seed)
    effects = {
        'acceptance_rate': 0.02,           # 2 percentage points
        'final_funding': 0.10,             # SMD
        'log1p_citations': 0.10,           # SMD
        'mean_career_distance': 0.10,      # SMD
        'hit_paper_or': np.log(1.15),      # log odds ratio proxy on hit prob
    }
    researchers = researchers.copy()
    researchers['log1p_citations'] = np.log1p(researchers['total_citations'])
    researchers['hit_paper_or'] = (researchers['hit_papers'] > 0).astype(float)

    rows = []
    for outcome, effect in effects.items():
        col = outcome if outcome in researchers.columns else None
        if col is None:
            continue
        contrasts = within_institution_contrasts(researchers, col)
        # H-CE planned contrast powered on the funding endpoint
        contrast_name = ('cautious-mean(explorer,exploiter)'
                         if outcome in ('cumulative_funding_received', 'final_funding')
                         else 'explorer-exploiter')
        base = contrasts[contrasts.contrast == contrast_name]['diff'].dropna().values
        if len(base) < 10:
            rows.append({'outcome': outcome, 'note': 'insufficient calibration blocks'})
            continue
        block_sd = float(np.std(base, ddof=1))
        outcome_sd = float(researchers[col].std(ddof=1))
        target = effect if outcome == 'acceptance_rate' else (
            effect if outcome == 'hit_paper_or' else effect * outcome_sd)
        for n_inst in institution_grid:
            hits = 0
            for _ in range(n_sim):
                sample = rng.normal(target, block_sd, n_inst)
                se = sample.std(ddof=1) / np.sqrt(n_inst)
                if abs(sample.mean()) > 1.96 * se:
                    hits += 1
            rows.append({'outcome': outcome, 'level': 'institution',
                         'scale': n_inst, 'target_effect': target,
                         'block_sd': block_sd, 'power': hits / n_sim})
        # world-level: seed-mean contrast variance
        seed_means = contrasts[contrasts.contrast == contrast_name] \
            .groupby('seed')['diff'].mean().values
        if len(seed_means) >= 2:
            world_sd = float(np.std(seed_means, ddof=1))
            for n_world in world_grid:
                hits = 0
                for _ in range(n_sim):
                    sample = rng.normal(target, world_sd, n_world)
                    se = sample.std(ddof=1) / np.sqrt(n_world)
                    if abs(sample.mean()) > 1.96 * se:
                        hits += 1
                rows.append({'outcome': outcome, 'level': 'world',
                             'scale': n_world, 'target_effect': target,
                             'block_sd': world_sd, 'power': hits / n_sim})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'power_curves.csv'), index=False)
    return df


# ------------------------------------------------------------------- command



# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--multi_run_prefix', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, default='outputs')
    parser.add_argument('--power_analysis', action='store_true')
    parser.add_argument('--funding_horizon_year', type=int, default=None,
                        help='Merge cumulative funding received at this horizon from checkpoints '
                             '(activates the frozen H-CE endpoint)')
    parser.add_argument('--n_boot', type=int, default=10000)
    parser.add_argument('--aggregates_only', action='store_true',
                        help='Regenerate only the deterministic per-seed strategy aggregates '
                             '(aggregate_results.csv) and exit before the bootstrap, '
                             'and report, leaving all other analysis outputs untouched')
    parser.add_argument('--review_scores_only', action='store_true',
                        help='Compute only the per-year review-score distribution trajectories '
                             '(review_score_trajectories.csv + review_score_convergence.json) '
                             'from checkpoints and exit, leaving all other outputs untouched')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = discover_runs(args.multi_run_prefix, args.outputs_dir)
    valid, excluded = validate_runs(runs)
    print(f"Discovered {len(runs)} runs; {len(valid)} valid, {len(excluded)} excluded")
    if not (args.aggregates_only or args.review_scores_only):
        pd.DataFrame([{k: r[k] for k in ('experiment_id', 'seed', 'status', 'problems')}
                      for r in excluded]).to_csv(
            os.path.join(args.out_dir, 'failed_or_excluded_runs.csv'), index=False)
    if not valid:
        raise SystemExit("No valid runs found")

    if args.review_scores_only:
        review_score_trajectories(valid, args.out_dir, args.outputs_dir)
        return

    agent_year = load_table(valid, 'agent_year')
    paper_age = load_table(valid, 'paper_age')
    ecosystem = load_table(valid, 'ecosystem_year')
    researchers = researcher_level(agent_year)
    researchers['log1p_citations'] = np.log1p(researchers['total_citations'])
    # Frozen H4 endpoint: within-seed top-decile citation stock (ITT)
    researchers['ever_hit'] = (researchers['total_citations'] >= researchers.groupby('seed')
                               ['total_citations'].transform(lambda x: x.quantile(0.9))).astype(float)
    # Frozen H-CE endpoint: cumulative funding RECEIVED (flow) from checkpoints
    if args.funding_horizon_year:
        fund = {}
        for r in valid:
            try:
                fr = load_funding_received(r['experiment_id'], args.funding_horizon_year,
                                           args.outputs_dir)
                for aid, v in fr.items():
                    fund[(r['seed'], aid)] = v
            except FileNotFoundError:
                pass
        researchers['cumulative_funding_received'] = [
            fund.get((s, a), np.nan) for s, a in zip(researchers.seed, researchers.agent_id)]

    # Manipulation + balance
    if not args.aggregates_only:
        manip = manipulation_check(researchers)
        manip.to_csv(os.path.join(args.out_dir, 'manipulation_check.csv'), index=False)
        balance = balance_check(valid)
        balance.to_csv(os.path.join(args.out_dir, 'balance_check.csv'), index=False)

    # Strategy aggregates per seed (cumulative funding column only when the
    # frozen H-CE endpoint was merged via --funding_horizon_year)
    agg_cols = dict(
        n=('agent_id', 'count'),
        career_distance=('mean_career_distance', 'mean'),
        direction_distance=('mean_direction_distance', 'mean'),
        switch_rate=('switch_rate', 'mean'),
        acceptance_rate=('acceptance_rate', 'mean'),
        final_funding=('final_funding', 'mean'),
        log1p_citations=('log1p_citations', 'mean'),
        hit_rate=('ever_hit', 'mean'),
    )
    if 'cumulative_funding_received' in researchers:
        agg_cols['cumulative_funding_received'] = ('cumulative_funding_received', 'mean')
    agg = researchers.groupby(['seed', 'strategy']).agg(**agg_cols).reset_index()
    agg.to_csv(os.path.join(args.out_dir, 'aggregate_results.csv'), index=False)
    if args.aggregates_only:
        print(f"Wrote {os.path.join(args.out_dir, 'aggregate_results.csv')} (aggregates only)")
        return

    # Primary contrasts with hierarchical bootstrap (frozen endpoints when
    # funding_horizon_year is set; legacy final_funding otherwise)
    primary_outcomes = ['mean_career_distance', 'acceptance_rate', 'log1p_citations', 'ever_hit']
    primary_outcomes.append('cumulative_funding_received' if args.funding_horizon_year
                            else 'final_funding')
    contrast_frames = [within_institution_contrasts(researchers, o) for o in primary_outcomes]
    contrasts = pd.concat([c for c in contrast_frames if len(c)], ignore_index=True)
    tests = hierarchical_bootstrap(contrasts, n_boot=args.n_boot)
    # bootstrap p-values (two-sided, CI-inversion approximation) + BH correction
    tests['excludes_zero'] = (tests['ci_lo'] > 0) | (tests['ci_hi'] < 0)
    tests.to_csv(os.path.join(args.out_dir, 'primary_hypothesis_tests.csv'), index=False)

    ecosystem.to_csv(os.path.join(args.out_dir, 'ecosystem_trajectories.csv'), index=False)


    if args.power_analysis:
        power_analysis(researchers, args.out_dir)

    # Markdown report
    lines = [f"# Cross-seed analysis: {args.multi_run_prefix}", '',
             f"Valid runs: {len(valid)} ({[r['seed'] for r in valid]}); "
             f"excluded: {len(excluded)}", '',
             '## Balance', balance.to_markdown(index=False), '',
             '## Manipulation check (per seed)', manip.to_markdown(index=False), '',
             '## Strategy aggregates (per seed)', agg.to_markdown(index=False), '',
             '## Primary contrasts (institution-blocked hierarchical bootstrap)',
             tests.drop(columns=['per_seed_means']).to_markdown(index=False), '',
             '_All outcomes are simulated. Uncertainty: 2-level bootstrap (seeds, '
             'institutions within seed). Institution-clustered estimates are '
             'conditional on the simulated worlds; cross-world generalization '
             'requires the many-world experiment (E4b)._']
    with open(os.path.join(args.out_dir, 'analysis_report.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"Analysis written to {args.out_dir}")


if __name__ == '__main__':
    main()
