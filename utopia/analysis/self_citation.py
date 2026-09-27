"""Self-citation analysis (RQ9) on the completed E4a confirmatory worlds.

Experiment
    Pure data analysis (no simulation, no GPU) over the three frozen E4a
    large-world runs (1,000 institutions x 5,000 researchers x 10 years,
    seeds 1001-1003). Every paper in E4a has exactly one author, so a
    self-citation is a citation edge whose citing paper and cited paper
    share the same author.

Questions
    1. Who self-cites? Rate by the citing author's exploration strategy
       (explorer / exploiter / cautious_explorer), with hierarchical
       bootstrap CIs (seeds -> institutions) and BH-corrected pairwise
       contrasts.
    2. Does status predict self-citation? Rate by the citing author's
       career-total citation quartile (and funding quartile).
    3. When? Self-citation share of all citation edges per simulated year.
    4. Does self-citation matter for inequality? Citation Gini with vs.
       without self-citation edges (same cited-papers-only convention as
       ecosystem_year.citation_gini).

Data provenance (read-only inputs)
    outputs/docs/explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed{1001,1002,1003}/
        paper.parquet       paper_id -> author_id/strategy/institution, submission years
        agent_year.parquet  per-researcher yearly outcomes (for status quartiles)
        run_manifest.json   completeness + scientific_config_hash validation
    outputs/checkpoints/<same experiment_id>/checkpoint_year_10.json.gz
        citation_tracker: {"citations": {cited -> [citing...]},
                           "references": {citing -> [cited...]}}
        The checkpoint is ~1 GB decompressed, so only the citation_tracker
        block is extracted by streaming byte-marker slicing.

Outputs (written to --out_dir, default outputs/docs/self_citation_e4a/)
    self_citation_edges.csv, rates_by_strategy.csv, strategy_contrasts.csv,
    rates_by_status.csv, rates_by_year.csv, gini_contribution.csv,
    analysis_report.md

Run
    python -m utopia.analysis.self_citation
"""
import argparse
import glob
import gzip
import json
import os
import re

import pandas as pd

from utopia.analysis.statistics import benjamini_hochberg, hierarchical_bootstrap
from utopia.analysis.exploration_cross_seed import PRIMARY_CONTRASTS, STRATEGIES, discover_runs, load_table, researcher_level, validate_runs
from utopia.metrics import calculate_gini_coefficient

RUN_PREFIX = 'explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10'
DEFAULT_OUT_DIR = 'outputs/docs/self_citation_e4a'
DEFAULT_CHECKPOINTS_DIR = 'outputs/checkpoints'


# ------------------------------------------------------- checkpoint reading

def load_citation_tracker(experiment_id: str,
                          checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR) -> dict:
    """Extract the citation_tracker block from a run's final checkpoint.

    The year-10 checkpoint is ~1 GB of JSON, so instead of json.load-ing the
    whole file we stream-decompress and slice the bytes between the top-level
    '"citation_tracker":' key and the following '"agent_tracker":' key
    (~3 MB), then parse only that block.
    """
    paths = glob.glob(os.path.join(checkpoints_dir, experiment_id,
                                   'checkpoint_year_*.json.gz'))
    if not paths:
        raise FileNotFoundError(f'no checkpoints under {checkpoints_dir}/{experiment_id}')
    path = max(paths, key=lambda p: int(re.search(r'year_(\d+)', p).group(1)))
    start_key, end_key = b'"citation_tracker":', b'"agent_tracker":'
    buf, block = b'', None
    with gzip.open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            buf += chunk
            if block is None:
                i = buf.find(start_key)
                if i < 0:
                    buf = buf[-len(start_key):]  # keep overlap for a split marker
                    continue
                buf = buf[i + len(start_key):]
                block = b''
            j = buf.find(end_key)
            if j >= 0:
                block = buf[:j]
                break
    if block in (None, b''):
        raise ValueError(f'citation_tracker block not found in {path}')
    block = block.rstrip().rstrip(b',')
    tracker = json.loads(block)
    assert set(tracker) >= {'citations', 'references'}, sorted(tracker)
    return tracker


# ------------------------------------------------------------- edge building

def build_edge_table(runs, checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR) -> pd.DataFrame:
    """One row per citation edge with citing/cited author attributes.

    Columns: seed, citing_paper, cited_paper, citing_author, cited_author,
    strategy, institution (both of the citing author), year (citing paper's
    first submission year), is_self.
    """
    paper = load_table(runs, 'paper')
    # paper.parquet is one row per (paper, submission-event year); collapse to
    # one row per paper, keeping the first submission year.
    first = (paper.sort_values('year')
             .groupby(['seed', 'paper_id'])
             .agg(author_id=('author_id', 'first'),
                  strategy=('strategy', 'first'),
                  institution=('institution', 'first'),
                  year=('year', 'first'))
             .reset_index())
    rows = []
    for r in runs:
        tracker = load_citation_tracker(r['experiment_id'], checkpoints_dir)
        for citing, cited_list in tracker['references'].items():
            for cited in cited_list:
                rows.append((r['seed'], citing, cited))
        n_edges = sum(len(v) for v in tracker['references'].values())
        print(f"{r['experiment_id']}: {n_edges} citation edges")
    edges = pd.DataFrame(rows, columns=['seed', 'citing_paper', 'cited_paper'])
    citing_attrs = first.rename(columns={
        'paper_id': 'citing_paper', 'author_id': 'citing_author'})
    edges = edges.merge(citing_attrs, on=['seed', 'citing_paper'], how='left')
    cited_attrs = first[['seed', 'paper_id', 'author_id']].rename(columns={
        'paper_id': 'cited_paper', 'author_id': 'cited_author'})
    edges = edges.merge(cited_attrs, on=['seed', 'cited_paper'], how='left')
    unresolved = edges['citing_author'].isna() | edges['cited_author'].isna()
    if unresolved.any():
        print(f'WARNING: dropping {int(unresolved.sum())} edges with papers '
              f'missing from paper.parquet')
        edges = edges[~unresolved].reset_index(drop=True)
    edges['is_self'] = edges['citing_author'] == edges['cited_author']
    return edges


# --------------------------------------------------------------- aggregates

def rates_by_strategy(edges: pd.DataFrame) -> pd.DataFrame:
    """Self-citation rate per (seed, strategy) plus pooled rows."""
    per_seed = (edges.groupby(['seed', 'strategy'])
                .agg(n_edges=('is_self', 'size'), n_self=('is_self', 'sum'))
                .reset_index())
    pooled = (edges.groupby('strategy')
              .agg(n_edges=('is_self', 'size'), n_self=('is_self', 'sum'))
              .reset_index())
    pooled.insert(0, 'seed', 'pooled')
    out = pd.concat([per_seed, pooled], ignore_index=True)
    out['rate'] = out['n_self'] / out['n_edges']
    return out


def strategy_contrasts(edges: pd.DataFrame, n_boot: int, boot_seed: int) -> pd.DataFrame:
    """Pairwise strategy contrasts on institution-level self-citation rates.

    Institution blocks (citing author's institution) are the resampling unit,
    matching the clustering used throughout the E4a analysis.
    """
    grp = (edges.groupby(['seed', 'institution', 'strategy'])['is_self']
           .mean().rename('rate').reset_index())
    wide = grp.pivot_table(index=['seed', 'institution'], columns='strategy',
                           values='rate')
    rows = []
    for a, b in PRIMARY_CONTRASTS:
        if a not in wide.columns or b not in wide.columns:
            continue
        both = wide[[a, b]].dropna()
        for (seed, _inst), row in both.iterrows():
            rows.append({'outcome': 'self_citation_rate',
                         'contrast': f'{a}-{b}',
                         'seed': seed,
                         'diff': row[a] - row[b]})
    contrasts = pd.DataFrame(rows)
    boot = hierarchical_bootstrap(contrasts, n_boot=n_boot, seed=boot_seed)
    boot['p_bh'] = benjamini_hochberg(boot['p_boot'])
    return boot


def rates_by_status(edges: pd.DataFrame, researchers: pd.DataFrame) -> pd.DataFrame:
    """Self-citation rate by the citing author's career-total quartile.

    Quartiles are computed within seed over authors who have at least one
    outgoing citation edge. Note: total_citations includes the (rare)
    self-citations themselves; at a ~0.4% edge share this circularity is
    negligible.
    """
    out = []
    for status_col in ('total_citations', 'final_funding'):
        merged = edges.merge(
            researchers[['seed', 'agent_id', status_col]],
            left_on=['seed', 'citing_author'], right_on=['seed', 'agent_id'],
            how='left')
        authors = (merged.groupby(['seed', 'citing_author'])[status_col]
                   .first().reset_index())
        authors['quartile'] = (authors.groupby('seed')[status_col]
                               .transform(lambda s: pd.qcut(s.rank(method='first'),
                                                            4, labels=[1, 2, 3, 4])))
        merged = merged.merge(authors[['seed', 'citing_author', 'quartile']],
                              on=['seed', 'citing_author'])
        g = (merged.groupby('quartile', observed=True)
             .agg(n_edges=('is_self', 'size'), n_self=('is_self', 'sum'))
             .reset_index())
        g['rate'] = g['n_self'] / g['n_edges']
        g.insert(0, 'status_measure', status_col)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def rates_by_year(edges: pd.DataFrame) -> pd.DataFrame:
    """Self-citation share of all citation edges per simulated year (pooled)."""
    g = (edges.groupby('year')
         .agg(n_edges=('is_self', 'size'), n_self=('is_self', 'sum'))
         .reset_index())
    g['rate'] = g['n_self'] / g['n_edges']
    return g


def gini_contribution(edges: pd.DataFrame) -> pd.DataFrame:
    """Citation Gini per seed with vs. without self-citation edges.

    Matches the ecosystem_year.citation_gini convention: Gini over in-degree
    of papers with at least one (remaining) citation.
    """
    rows = []
    for seed, df in edges.groupby('seed'):
        with_self = df.groupby('cited_paper').size().values
        without = df[~df['is_self']].groupby('cited_paper').size().values
        g_with = calculate_gini_coefficient(list(with_self))
        g_without = calculate_gini_coefficient(list(without))
        rows.append({'seed': seed,
                     'gini_with_self': g_with,
                     'gini_without_self': g_without,
                     'delta': g_with - g_without,
                     'n_cited_papers_with': len(with_self),
                     'n_cited_papers_without': len(without)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- report

def write_report(out_dir, runs, edges, by_strategy, contrasts, by_status,
                 by_year, gini):
    pooled = by_strategy[by_strategy['seed'] == 'pooled'].set_index('strategy')
    total_edges = int(edges.shape[0])
    total_self = int(edges['is_self'].sum())
    lines = [
        '# Self-citation analysis (RQ9) — E4a confirmatory worlds',
        '',
        f'Runs: {", ".join(r["experiment_id"] for r in runs)}',
        f'Command: python -m utopia.analysis.self_citation',
        '',
        f'Total citation edges: {total_edges}; self-citations: {total_self} '
        f'({total_self / total_edges:.4%}).',
        '',
        '## Rate by citing author strategy (pooled over 3 seeds)',
        '',
    ]
    for s in STRATEGIES:
        if s in pooled.index:
            r = pooled.loc[s]
            lines.append(f"- {s}: {int(r['n_self'])}/{int(r['n_edges'])} "
                         f"= {r['rate']:.4%}")
    lines += ['', '## Pairwise contrasts (institution-level rates, '
              'hierarchical bootstrap, BH-corrected)', '']
    for _, r in contrasts.iterrows():
        lines.append(f"- {r['contrast']}: diff={r['mean_diff']:+.5f} "
                     f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}], "
                     f"p_boot={r['p_boot']:.4g}, p_BH={r['p_bh']:.4g}")
    lines += ['', '## Citation Gini with vs. without self-citations', '']
    for _, r in gini.iterrows():
        lines.append(f"- seed {r['seed']}: {r['gini_with_self']:.4f} -> "
                     f"{r['gini_without_self']:.4f} (delta {r['delta']:+.4f})")
    lines += ['', 'Full tables: self_citation_edges.csv, rates_by_strategy.csv, '
              'strategy_contrasts.csv, rates_by_status.csv, rates_by_year.csv, '
              'gini_contribution.csv', '']
    path = os.path.join(out_dir, 'analysis_report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_prefix', required=True)
    ap.add_argument('--outputs_dir', default='outputs')
    ap.add_argument('--checkpoints_dir', default=DEFAULT_CHECKPOINTS_DIR)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--boot_seed', type=int, default=0)
    args = ap.parse_args()

    runs, excluded = validate_runs(discover_runs(args.run_prefix, args.outputs_dir))
    if excluded:
        print(f'Excluded {len(excluded)} runs: '
              f'{[(r["experiment_id"], r["problems"]) for r in excluded]}')
    if not runs:
        raise SystemExit('no valid runs found')
    print(f'Analyzing {len(runs)} runs (seeds {[r["seed"] for r in runs]})')

    os.makedirs(args.out_dir, exist_ok=True)
    edges = build_edge_table(runs, args.checkpoints_dir)
    agent_year = load_table(runs, 'agent_year')
    researchers = researcher_level(agent_year)

    by_strategy = rates_by_strategy(edges)
    contrasts = strategy_contrasts(edges, args.n_boot, args.boot_seed)
    by_status = rates_by_status(edges, researchers)
    by_year = rates_by_year(edges)
    gini = gini_contribution(edges)

    edges[edges['is_self']].to_csv(
        os.path.join(args.out_dir, 'self_citation_edges.csv'), index=False)
    by_strategy.to_csv(os.path.join(args.out_dir, 'rates_by_strategy.csv'), index=False)
    contrasts.to_csv(os.path.join(args.out_dir, 'strategy_contrasts.csv'), index=False)
    by_status.to_csv(os.path.join(args.out_dir, 'rates_by_status.csv'), index=False)
    by_year.to_csv(os.path.join(args.out_dir, 'rates_by_year.csv'), index=False)
    gini.to_csv(os.path.join(args.out_dir, 'gini_contribution.csv'), index=False)
    write_report(args.out_dir, runs, edges, by_strategy, contrasts, by_status,
                 by_year, gini)


if __name__ == '__main__':
    main()
