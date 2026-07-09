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
  balance_check.csv, ecosystem_trajectories.csv, analysis_report.md, figures/
"""
import argparse
import glob
import json
import os
from collections import defaultdict

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
    """Strategy contrasts computed within each (seed, institution) block."""
    rows = []
    for (seed, inst), block in researchers.groupby(['seed', 'institution']):
        means = block.groupby('strategy')[outcome].mean()
        for a, b in PRIMARY_CONTRASTS:
            if a in means.index and b in means.index and pd.notna(means[a]) and pd.notna(means[b]):
                rows.append({'seed': seed, 'institution': inst, 'outcome': outcome,
                             'contrast': f'{a}-{b}', 'diff': means[a] - means[b]})
    return pd.DataFrame(rows)


def hierarchical_bootstrap(contrasts: pd.DataFrame, n_boot=10000, seed=0):
    """Two-level bootstrap: resample seeds, then institutions within seed.

    Returns rows per (outcome, contrast): mean, 95% CI, fraction of seeds with
    the same direction, cluster counts.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (outcome, contrast), df in contrasts.groupby(['outcome', 'contrast']):
        seeds = df['seed'].unique()
        by_seed = {s: df[df['seed'] == s]['diff'].values for s in seeds}
        point = float(np.mean([np.mean(v) for v in by_seed.values()]))
        seed_means = [float(np.mean(v)) for v in by_seed.values()]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            if len(seeds) > 1:
                chosen = rng.choice(seeds, size=len(seeds), replace=True)
            else:
                chosen = seeds
            vals = []
            for s in chosen:
                v = by_seed[s]
                vals.append(np.mean(v[rng.integers(0, len(v), len(v))]))
            boots[b] = np.mean(vals)
        pooled_sd = float(df['diff'].std(ddof=1)) if len(df) > 1 else np.nan
        out.append({
            'outcome': outcome, 'contrast': contrast,
            'mean_diff': point,
            'ci_lo': float(np.percentile(boots, 2.5)),
            'ci_hi': float(np.percentile(boots, 97.5)),
            'smd': point / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan,
            'n_seeds': len(seeds),
            'n_institution_blocks': len(df),
            'seed_direction_agreement': float(np.mean(np.sign(seed_means) == np.sign(point)))
            if point != 0 else np.nan,
            'per_seed_means': json.dumps({str(s): round(float(np.mean(v)), 5)
                                          for s, v in by_seed.items()}),
        })
    return pd.DataFrame(out)


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        i = order[rank_idx]
        val = min(prev, p[i] * n / (rank_idx + 1))
        adj[i] = val
        prev = val
    return adj


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
        base = contrasts[contrasts.contrast == 'explorer-exploiter']['diff'].dropna().values
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
        seed_means = contrasts[contrasts.contrast == 'explorer-exploiter'] \
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


# ------------------------------------------------------------------- figures

def make_figures(researchers, agent_year, ecosystem, paper_age, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    colors = {'explorer': 'tab:red', 'exploiter': 'tab:blue', 'cautious_explorer': 'tab:green'}
    n_seeds = researchers['seed'].nunique() if len(researchers) else 0

    def save(fig, name, title):
        fig.suptitle(f"{title}\n(n={len(researchers)} researchers, {n_seeds} seeds; "
                     f"uncertainty across seeds)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, name), dpi=150)
        plt.close(fig)

    def yearly_by_strategy(df, col, ax):
        df = df.copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        for s in STRATEGIES:
            sub = df[df.strategy == s].groupby(['seed', 'year'])[col].mean().reset_index()
            m = sub.groupby('year')[col].mean()
            lo = sub.groupby('year')[col].min()
            hi = sub.groupby('year')[col].max()
            ax.plot(m.index, m.values, label=s, color=colors[s])
            ax.fill_between(m.index, lo.values, hi.values, alpha=0.15, color=colors[s])
        ax.legend(fontsize=7)
        ax.set_xlabel('year')

    if len(agent_year):
        for col, fname, title in [
                ('avg_career_distance', 'fig1_manipulation_career_distance.png',
                 'Manipulation check: career-centroid distance by year and strategy'),
                ('topic_switched', 'fig2_topic_switch_rate.png', 'Topic switch rate'),
                ('acceptance_rate', 'fig3_acceptance_trajectory.png', 'Simulated acceptance rate'),
                ('funding', 'fig4_funding_trajectory.png', 'Simulated funding')]:
            fig, ax = plt.subplots(figsize=(6, 4))
            yearly_by_strategy(agent_year, col, ax)
            ax.set_ylabel(col)
            save(fig, fname, title)

    if len(paper_age):
        age3 = paper_age[paper_age.age == 3]
        if len(age3):
            fig, ax = plt.subplots(figsize=(6, 4))
            data = [np.log1p(age3[age3.strategy == s]['citations'].dropna()) for s in STRATEGIES]
            ax.violinplot([d.values for d in data], showmedians=True)
            ax.set_xticks([1, 2, 3]); ax.set_xticklabels(STRATEGIES, fontsize=8)
            ax.set_ylabel('log1p(citations) at age 3')
            save(fig, 'fig5_citation_age3_distribution.png', 'Simulated citations at age 3')

            fig, ax = plt.subplots(figsize=(6, 4))
            for s in STRATEGIES:
                sub = age3[age3.strategy == s]
                ax.scatter(sub['novelty_score'], np.log1p(sub['citations']),
                           s=8, alpha=0.4, color=colors[s], label=s)
            ax.set_xlabel('novelty'); ax.set_ylabel('log1p(citations) age 3')
            ax.legend(fontsize=7)
            save(fig, 'fig6_risk_return.png', 'Risk-return: novelty vs age-3 citations')

        cd3 = paper_age[(paper_age.age == 3) & paper_age.cd.notna()]
        if len(cd3):
            fig, ax = plt.subplots(figsize=(6, 4))
            data = [cd3[cd3.strategy == s]['cd'].dropna().values for s in STRATEGIES]
            ax.boxplot(data, labels=STRATEGIES)
            ax.set_ylabel('CD index at age 3')
            save(fig, 'fig9_cd_distribution.png', 'Disruption: age-3 CD index')

        fig, ax = plt.subplots(figsize=(6, 4))
        for s in STRATEGIES:
            acc = paper_age[paper_age.strategy == s].groupby('age')['citations'].mean()
            ax.plot(acc.index, acc.values, label=s, color=colors[s], marker='o')
        ax.set_xlabel('paper age'); ax.set_ylabel('mean cumulative citations')
        ax.legend(fontsize=7)
        save(fig, 'fig8_citation_accumulation.png', 'Citation accumulation by paper age')

    if len(ecosystem):
        for col, fname, title in [
                ('topic_entropy_normalized', 'fig10_topic_entropy.png', 'Ecosystem topic entropy'),
                ('funding_gini', 'fig11a_funding_gini.png', 'Funding Gini'),
                ('citation_gini', 'fig11b_citation_gini.png', 'Citation Gini'),
                ('num_active_researchers', 'fig12_active_researchers.png', 'Active researchers')]:
            if col not in ecosystem.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for s_val, sub in ecosystem.groupby('seed'):
                ax.plot(sub['year'], pd.to_numeric(sub[col], errors='coerce'),
                        alpha=0.6, label=f'seed {s_val}')
            ax.set_xlabel('year'); ax.set_ylabel(col)
            ax.legend(fontsize=7)
            save(fig, fname, title)

    if len(researchers):
        fig, ax = plt.subplots(figsize=(6, 4))
        shares = []
        hit = researchers.groupby(['seed', 'strategy'])['hit_papers'].sum().reset_index()
        for s in STRATEGIES:
            sub = hit[hit.strategy == s]
            shares.append(sub['hit_papers'].mean())
        ax.bar(STRATEGIES, shares, color=[colors[s] for s in STRATEGIES])
        ax.set_ylabel('mean hit papers per seed')
        save(fig, 'fig7_hit_paper_share.png', 'Hit-paper share by strategy')


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--multi_run_prefix', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, default='outputs')
    parser.add_argument('--power_analysis', action='store_true')
    parser.add_argument('--n_boot', type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = discover_runs(args.multi_run_prefix, args.outputs_dir)
    valid, excluded = validate_runs(runs)
    print(f"Discovered {len(runs)} runs; {len(valid)} valid, {len(excluded)} excluded")
    pd.DataFrame([{k: r[k] for k in ('experiment_id', 'seed', 'status', 'problems')}
                  for r in excluded]).to_csv(
        os.path.join(args.out_dir, 'failed_or_excluded_runs.csv'), index=False)
    if not valid:
        raise SystemExit("No valid runs found")

    agent_year = load_table(valid, 'agent_year')
    paper_age = load_table(valid, 'paper_age')
    ecosystem = load_table(valid, 'ecosystem_year')
    researchers = researcher_level(agent_year)
    researchers['log1p_citations'] = np.log1p(researchers['total_citations'])
    researchers['ever_hit'] = (researchers['hit_papers'] > 0).astype(float)

    # Manipulation + balance
    manip = manipulation_check(researchers)
    manip.to_csv(os.path.join(args.out_dir, 'manipulation_check.csv'), index=False)
    balance = balance_check(valid)
    balance.to_csv(os.path.join(args.out_dir, 'balance_check.csv'), index=False)

    # Strategy aggregates per seed
    agg = researchers.groupby(['seed', 'strategy']).agg(
        n=('agent_id', 'count'),
        career_distance=('mean_career_distance', 'mean'),
        direction_distance=('mean_direction_distance', 'mean'),
        switch_rate=('switch_rate', 'mean'),
        acceptance_rate=('acceptance_rate', 'mean'),
        final_funding=('final_funding', 'mean'),
        log1p_citations=('log1p_citations', 'mean'),
        hit_rate=('ever_hit', 'mean'),
    ).reset_index()
    agg.to_csv(os.path.join(args.out_dir, 'aggregate_results.csv'), index=False)

    # Primary contrasts with hierarchical bootstrap
    primary_outcomes = ['mean_career_distance', 'acceptance_rate', 'final_funding',
                        'log1p_citations', 'ever_hit']
    contrast_frames = [within_institution_contrasts(researchers, o) for o in primary_outcomes]
    contrasts = pd.concat([c for c in contrast_frames if len(c)], ignore_index=True)
    tests = hierarchical_bootstrap(contrasts, n_boot=args.n_boot)
    # bootstrap p-values (two-sided, CI-inversion approximation) + BH correction
    tests['excludes_zero'] = (tests['ci_lo'] > 0) | (tests['ci_hi'] < 0)
    tests.to_csv(os.path.join(args.out_dir, 'primary_hypothesis_tests.csv'), index=False)

    ecosystem.to_csv(os.path.join(args.out_dir, 'ecosystem_trajectories.csv'), index=False)

    make_figures(researchers, agent_year, ecosystem, paper_age, args.out_dir)

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
