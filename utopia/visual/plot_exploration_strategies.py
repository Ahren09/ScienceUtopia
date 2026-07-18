"""Paper figure for the exploration-strategy study ("Cautious Exploration
Balances Novelty and Reward", sec:results-exploration).

Experiment
----------
Three research-strategy archetypes are followed through 10 simulated years of
publication, citation, and funding competition, starting from identical
funding and reputation:
  * explorer          -- picks topics far from prior expertise every year
  * cautious_explorer -- explores, but stays within reach of prior expertise
  * exploiter         -- keeps publishing in its established area
Data come from the calibration panel: 5 independent seeds (501-505), each
with 100 institutions x 5 researchers (1 explorer + 3 exploiters +
1 cautious explorer per institution), run in the neutral mechanism condition
(strategy changes topic choice only; review/funding rules are strategy-blind).

Panels (1x4 strip)
------------------
  (a) Manipulation check: mean career distance -- the embedding distance
      between each year's topic and the agent's prior expertise, averaged
      over a career. Confirms the three strategies actually separate in
      behavior (explorer > cautious > exploiter) before outcomes are compared.
  (b) High-impact papers: share of researchers who ever produce a
      high-impact paper (top citation decile of its year cohort; the
      hit_rate/ever_hit columns in the data), per strategy.
  (c) Acceptance-rate trajectory: how the review system treats each strategy,
      year by year.
  (d) Funding accrual: cumulative funding RECEIVED per strategy (a flow,
      summed from funding_success_history -- the frozen H-CE endpoint of the
      cross-seed analysis). Unlike year-end funding held (a stock, which
      declines for everyone because annual costs exceed the injected pool),
      this rises monotonically and isolates how much the funding system
      actually awards each strategy.

Companion figures (separate scripts, same output directory):
  utopia.visual.plot_exploration_contrasts   -- forest plots of the cautious
      explorer's advantage vs. explorer and vs. exploiter
  utopia.visual.plot_exploration_risk_return -- per-strategy novelty vs.
      citation small multiples

Metric definitions
------------------
  career_distance   embedding distance between chosen topic and the agent's
                    accumulated expertise (0 = same area, ~1 = unrelated)
  novelty_score     embedding distance of a paper from its field centroid
  hit               paper in the top citation decile of its cohort; labeled
                    "high-impact paper" in figure text, and aggregated per
                    researcher by the hit_rate / ever_hit data columns
  log1p_citations   log(1 + citations) at paper age 3 (skew-robust)
  SMD               mean difference / pooled within-seed SD

Data provenance (read-only inputs)
----------------------------------
  outputs/docs/explore_calibration_analysis/aggregate_results.csv (and
      manipulation_check.csv for the SMD note)
      -- cross-seed aggregation by utopia.analysis.exploration_cross_seed
  outputs/docs/explore_calibration_..._seed*/{strategy_year,agent_year}.parquet
      -- per-run tables recorded by utopia.metrics.exploration_metrics
  outputs/checkpoints/explore_calibration_..._seed*/checkpoint_year_*.json.gz
      -- funding_success_history for panel (c), extracted once via
      utopia.analysis.exploration_cross_seed.load_funding_received and then
      cached as a small CSV (--funding_cache); delete the CSV to rebuild.

Run:
    python -m utopia.visual.plot_exploration_strategies
Args: --analysis_dir (cross-seed CSV dir), --runs_glob (per-seed run dirs
for the trajectory/scatter panels), --out_dir (figure destination).
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import (
    GRID, INK, MUTED, nature_style, STRATEGY_PALETTE, STRATEGY_NICE)

DEFAULT_ANALYSIS_DIR = 'outputs/docs/explore_calibration_analysis'
DEFAULT_RUNS_GLOB = ('outputs/docs/'
                     'explore_calibration_qwen3_32b_neutral_i100_n500_y10_seed*')
DEFAULT_OUT_DIR = 'outputs/visual/exploration_strategies_calibration'
DEFAULT_FUNDING_CACHE = 'data/explore_calibration_cumulative_funding.csv'

# Shared three-strategy palette (single source of truth in visual_utils):
# Hiroshige-family warm coral / soft amber / deep steel blue. The cautious
# explorer wears the salient deep blue -- it is the story's protagonist.
STRATEGIES = ['explorer', 'cautious_explorer', 'exploiter']
PALETTE = STRATEGY_PALETTE
LABELS = STRATEGY_NICE

def load_run_tables(runs_glob: str, table: str) -> pd.DataFrame:
    """Concat one parquet table across all completed seed runs.

    Skips invalidated run dirs (same convention as exploration_cross_seed)
    and tags each row with the seed parsed from the directory name.
    """
    frames = []
    for run_dir in sorted(glob.glob(runs_glob)):
        if 'invalidated' in run_dir:
            continue
        path = Path(run_dir) / f'{table}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['seed'] = int(re.search(r'seed(\d+)', run_dir).group(1))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f'no {table}.parquet under {runs_glob}')
    return pd.concat(frames, ignore_index=True)


def load_cumulative_funding(runs_glob: str, cache_path: str) -> pd.DataFrame:
    """Cumulative funding received, as seed x year x strategy means.

    A flow, not a stock: for each yearly checkpoint it sums every award in
    funding_success_history per agent (reusing the frozen H-CE endpoint
    extraction, exploration_cross_seed.load_funding_received), then averages
    over the researchers of each strategy. Inactive researchers keep their
    frozen totals and stay in the cohort average.

    Parsing the ~50 gzipped checkpoints takes ~2 s each, so the aggregated
    frame is cached at cache_path; delete that file to force a rebuild.
    """
    cache = Path(cache_path)
    if cache.exists():
        return pd.read_csv(cache)
    from utopia.analysis.exploration_cross_seed import load_funding_received
    rows = []
    for run_dir in sorted(glob.glob(runs_glob)):
        if 'invalidated' in run_dir:
            continue
        exp_id = Path(run_dir).name
        seed = int(re.search(r'seed(\d+)', run_dir).group(1))
        n_years = int(re.search(r'_y(\d+)_', run_dir).group(1))
        strategy_of = (pd.read_parquet(Path(run_dir) / 'agent_year.parquet',
                                       columns=['agent_id', 'strategy'])
                       .drop_duplicates().set_index('agent_id')['strategy'])
        for year in range(1, n_years + 1):
            received = pd.Series(load_funding_received(exp_id, year))
            means = received.groupby(received.index.map(strategy_of)).mean()
            rows += [{'seed': seed, 'year': year, 'strategy': s,
                      'cum_funding': v} for s, v in means.items()]
        print(f'extracted cumulative funding: {exp_id}')
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def strategy_points_panel(ax, agg: pd.DataFrame, col: str, ylabel: str,
                          title: str, note: str = None):
    """Per-strategy mean bar (across seeds) with the 5 seed values overlaid.

    How to read the marks, per strategy:
      * 5 jittered dots -- that strategy's mean in each seed (501-505). The
        horizontal jitter only prevents overplotting and carries no meaning;
        the tighter the dots, the more stable the result across seeds.
      * thick tick line -- the cross-seed mean of those 5 values; the number
        printed beside it is this mean.
      * translucent bar -- the same mean drawn from zero, purely to make the
        relative heights of the three strategies easy to compare.
    With n = 5 seeds, showing the raw seed values is more honest than
    collapsing them into an error bar.

    What `col` means depends on the caller:
      * career_distance (panel a): two-level average -- each researcher's
        yearly embedding distance between the chosen topic and their prior
        expertise is first averaged over the 10-year career, then averaged
        over all researchers of the strategy (0 = stays in the established
        area, ~1 = jumps to an unrelated field).
      * hit_rate (panel b): share of researchers who ever produce a
        high-impact paper -- one in the top citation decile of its cohort.

    The optional `note` reports SMD, the standardized mean difference:
    (group mean A - group mean B) / pooled SD, i.e. the gap expressed in
    standard-deviation units so it is dimension-free. Cohen's convention
    calls 0.2/0.5/0.8 small/medium/large; the explorer-vs-exploiter career
    distances reach 3.2-3.8, near-disjoint distributions, so the strategy
    manipulation clearly worked.
    """
    means = agg.groupby('strategy')[col].mean()
    for x, strat in enumerate(STRATEGIES):
        ax.bar(x, means[strat], width=0.62, color=PALETTE[strat], alpha=0.25,
               edgecolor='none', zorder=1)
        sns.stripplot(x=[x] * 5, y=agg.loc[agg.strategy == strat, col],
                      ax=ax, color=PALETTE[strat], size=3.2, jitter=0.08,
                      native_scale=True, edgecolor='white', linewidth=0.4,
                      zorder=3)
        ax.hlines(means[strat], x - 0.31, x + 0.31, color=PALETTE[strat],
                  linewidth=1.4, zorder=2)
        # Label sits just right of the mean tick, clear of the jittered dots.
        ax.annotate(f'{means[strat]:.2f}', xy=(x + 0.33, means[strat]),
                    xytext=(2, 0), textcoords='offset points', ha='left',
                    va='center', fontsize=6, color=INK)
    if note:
        # Bottom-right corner: the translucent bars leave it readable, and it
        # cannot collide with the mean labels near the bar tops.
        ax.annotate(note, xy=(0.97, 0.04), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=5.5, color=MUTED)
    ax.set_xticks(range(len(STRATEGIES)),
                  [LABELS[s].replace(' ', '\n') for s in STRATEGIES])
    ax.set_xlim(-0.6, len(STRATEGIES) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def trajectory_panel(ax, sy: pd.DataFrame, value: str, ylabel: str, title: str):
    """Ten-year strategy trajectory, mean with 95% CI band across the 5 seeds.

    The band's replication unit is the seed (n = 5), a coarser interval than
    the institution-block bootstrap used for the headline contrasts.
    """
    sns.lineplot(data=sy, x='year', y=value, hue='strategy',
                 hue_order=STRATEGIES, palette=PALETTE, ax=ax,
                 errorbar=('ci', 95), linewidth=1.4, marker='o',
                 markersize=3, markeredgecolor='white', markeredgewidth=0.5,
                 legend=False, err_kws={'alpha': 0.12, 'linewidth': 0})
    ax.set_xticks(range(2, 11, 2))
    ax.margins(x=0.04)
    ax.set_xlabel('Simulation year')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis_dir', default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument('--runs_glob', default=DEFAULT_RUNS_GLOB)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--funding_cache', default=DEFAULT_FUNDING_CACHE,
                    help='CSV cache of checkpoint-extracted cumulative '
                         'funding; delete to rebuild')
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    agg = pd.read_csv(analysis_dir / 'aggregate_results.csv')
    mc = pd.read_csv(analysis_dir / 'manipulation_check.csv')
    sy = load_run_tables(args.runs_glob, 'strategy_year')
    cum = load_cumulative_funding(args.runs_glob, args.funding_cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Double-column Nature width (180 mm = 7.09 in), 1x4 strip.
    fig, axes = plt.subplots(1, 4, figsize=(7.09, 2.1))
    ax_a, ax_b, ax_c, ax_d = axes

    smd = mc.smd_explorer_vs_exploiter_career
    strategy_points_panel(
        ax_a, agg, 'career_distance', 'Mean career distance',
        'Strategies separate',
        note=f'explorer vs exploiter\nSMD {smd.min():.1f}–{smd.max():.1f}')
    strategy_points_panel(ax_b, agg, 'hit_rate',
                          'P($\\geq$1 high-impact paper)',  # mathtext: Arial has no U+2265
                          'High-impact papers')
    trajectory_panel(ax_c, sy, 'avg_acceptance_rate', 'Acceptance rate',
                     'Review outcomes')
    trajectory_panel(ax_d, cum, 'cum_funding',
                     'Cumulative funding received (units)', 'Funding rewards')

    for letter, ax in zip('abcd', axes.flat):
        ax.text(-0.24, 1.08, letter, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', va='bottom')
    sns.despine(fig)

    handles = [mpl.lines.Line2D([], [], color=PALETTE[s], linewidth=1.4,
                                marker='o', markersize=3.5,
                                markeredgecolor='white', label=LABELS[s])
               for s in STRATEGIES]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.05), columnspacing=1.4, handlelength=1.6)
    fig.tight_layout(w_pad=0.4)

    for ext in {'pdf'}:
        path = out_dir / f'fig_exploration_strategies.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
