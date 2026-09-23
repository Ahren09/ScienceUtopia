"""Paper figure for the exploration-strategy study ("Cautious Exploration
Balances Novelty and Reward", sec:results-exploration), including the career
outcomes formerly shown as the separate fig:career_outcomes.

Experiment
----------
Three research-strategy archetypes are followed through 10 simulated years of
publication, citation, and funding competition, starting from identical
funding and reputation:
  * explorer          -- picks topics far from prior expertise every year
  * cautious_explorer -- explores, but stays within reach of prior expertise
  * exploiter         -- keeps publishing in its established area
Panels (a, b) use the calibration panel: 5 independent seeds (501-505), each
with 100 institutions x 5 researchers (1 explorer + 3 exploiters +
1 cautious explorer per institution), run in the neutral mechanism
condition (strategy changes topic choice only; review/funding rules are
strategy-blind). Panels (c, d) use the three E4a confirmatory worlds
(1000 institutions x 5000 researchers x 10 years, seeds 1001-1003, neutral
mechanisms), where researchers are removed ("attrition") when their funding
resources fall below the survival threshold.

Panels (1x4)
------------
Panels (a) and (b) carry two metrics on twin y-axes; color always encodes
the strategy, so the mark/line style is what decodes metric vs. axis.
  (a) Behavior and high-impact papers.
      Left axis -- manipulation check: mean career distance, the embedding
      distance between each year's topic and the agent's prior expertise,
      averaged over a career (translucent bar + 5 seed dots + mean tick).
      Confirms the three strategies actually separate in behavior
      (explorer > cautious > exploiter) before outcomes are compared.
      Right axis -- high-impact papers: share of researchers who ever
      produce a high-impact paper (top citation decile of its year cohort;
      the hit_rate/ever_hit columns in the data), drawn as a diamond at the
      cross-seed mean with a capped min-max whisker over the 5 seeds, offset
      right of each category.
  (b) Review and funding outcomes.
      Left axis -- acceptance-rate trajectory (solid lines, circle markers):
      how the review system treats each strategy, year by year.
      Right axis -- funding accrual (dashed lines, no markers): cumulative
      funding RECEIVED per strategy (a flow, summed from
      funding_success_history -- the frozen H-CE endpoint of the cross-seed
      analysis). Unlike year-end funding held (a stock, which declines for
      everyone because annual costs exceed the injected pool), this rises
      monotonically and isolates how much the funding system actually awards
      each strategy. Line style decodes the metric (solid = acceptance,
      dashed = funding); color always encodes strategy.
  (c) Forest plot of the cautious explorer's edge over each pure
      strategy (E4a confirmatory worlds). Each outcome row holds two dodged
      estimates: cautious-explorer minus explorer (coral, upper) and
      minus exploiter (amber, lower); positive favors the cautious
      explorer. x = standardized mean difference (SMD = mean difference /
      pooled SD, so the four outcomes share one axis); whiskers = 95%
      hierarchical-bootstrap CIs (institution-block resampling). The signed
      raw-unit effect is annotated past each CI. Open marker = CI crosses
      zero. NOTE: color here marks the COMPARISON strategy the cautious
      explorer is measured against -- the figure caption explains this.
  (d) Survival: share of researchers still active by simulated year and
      strategy (mean across the 3 E4a seeds; translucent per-seed traces).
      The three values quoted in the paper are annotated.

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
      -- funding_success_history for the funding overlay in panel (b),
      extracted once via
      utopia.analysis.exploration_cross_seed.load_funding_received and then
      cached as a small CSV (--funding_cache); delete the CSV to rebuild.
  outputs/docs/explore_e4a_analysis/primary_hypothesis_tests.csv
      -- panel (c): (contrast, outcome, mean_diff, smd, ci_lo, ci_hi).
      Contrasts are stored as explorer-cautious_explorer (sign-flipped here)
      and cautious_explorer-exploiter. Calibration CSVs name the funding
      outcome final_funding; it is normalized to cumulative_funding_received.
  outputs/docs/explore_confirmatory_..._seed100{1-3}/agent_year.parquet
      -- panel (d): (agent_id, year, strategy, is_active).

Run:
    python -m utopia.visual.plot_exploration_strategies
Args: --analysis_dir (cross-seed CSV dir), --runs_glob (per-seed run dirs
for the trajectory panels), --out_dir (figure destination),
--e4a_analysis_dir / --e4a_runs_glob (E4a inputs for panels c/d).
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import (
    GRID, INK, MUTED, nature_style, STRATEGY_PALETTE, STRATEGY_NICE)

DEFAULT_ANALYSIS_DIR = 'outputs/docs/explore_calibration_analysis'
DEFAULT_RUNS_GLOB = ('outputs/docs/'
                     'explore_calibration_qwen3_32b_neutral_i100_n500_y10_seed*')
DEFAULT_OUT_DIR = 'outputs/visual/exploration_strategies_calibration'
DEFAULT_FUNDING_CACHE = 'data/explore_calibration_cumulative_funding.csv'
DEFAULT_E4A_ANALYSIS_DIR = 'outputs/docs/explore_e4a_analysis'
DEFAULT_E4A_RUNS_GLOB = ('outputs/docs/'
                         'explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed*')

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
                          title: str, note: str = None,
                          overlay_col: str = None, overlay_ylabel: str = None):
    """Per-strategy mean bar (across seeds) with the 5 seed values overlaid.

    How to read the marks, per strategy:
      * 5 jittered dots -- that strategy's mean in each seed (501-505). The
        horizontal jitter only prevents overplotting and carries no meaning;
        the tighter the dots, the more stable the result across seeds.
      * thick tick line -- the cross-seed mean of those 5 values; the number
        printed beside it is this mean.
      * translucent bar -- the same mean drawn from zero, purely to make the
        relative heights of the three strategies easy to compare.
      * (with overlay_col) diamond + capped min-max whisker (error-bar
        style) offset right of the category, read on the RIGHT twin axis:
        the diamond is the cross-seed mean of overlay_col and the whisker
        spans the min-max of the 5 seed values. Mean-only on purpose -- the
        left series already shows every seed, and 15 more points would
        double the clutter.
    With n = 5 seeds, showing the raw seed values is more honest than
    collapsing them into an error bar.

    What `col` means depends on the caller:
      * career_distance (left axis): two-level average -- each researcher's
        yearly embedding distance between the chosen topic and their prior
        expertise is first averaged over the 10-year career, then averaged
        over all researchers of the strategy (0 = stays in the established
        area, ~1 = jumps to an unrelated field).
      * hit_rate (overlay_col, right axis): share of researchers who ever
        produce a high-impact paper -- one in the top citation decile of its
        cohort.

    Returns the twin axes when overlay_col is given (the caller must despine
    it keeping the right spine), else None.

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
    # Slightly smaller category labels: three names in a ~1.7 in panel.
    ax.tick_params(axis='x', labelsize=5.2)
    ax.set_xlim(-0.6, len(STRATEGIES) - 0.4)
    ax.set_ylim(0, 0.9)  # pinned so the twin-axis overlay layout stays stable
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)

    axr = None
    if overlay_col is not None:
        # Right twin axis, just inside the mean tick's end (tick spans
        # x +- 0.31) so the diamonds read toward the right spine.
        axr = ax.twinx()
        for x, strat in enumerate(STRATEGIES):
            vals = agg.loc[agg.strategy == strat, overlay_col]
            m = vals.mean()
            # Capped error-bar whisker so the min-max interval reads clearly.
            axr.errorbar(x + 0.30, m, yerr=[[m - vals.min()], [vals.max() - m]],
                         fmt='none', ecolor=PALETTE[strat], elinewidth=1.2,
                         capsize=2.0, capthick=1.2, alpha=0.9, zorder=2)
            axr.plot(x + 0.30, m, marker='D', markersize=3.8,
                     linestyle='none', color=PALETTE[strat],
                     markeredgecolor='white', markeredgewidth=0.5, zorder=3)
            # Label right of the diamond: the whisker passes through the
            # space above it, so 'above' would collide.
            axr.annotate(f'{m:.2f}', xy=(x + 0.30, m),
                         xytext=(3, 0), textcoords='offset points', ha='left',
                         va='center', fontsize=6, color=INK)
        axr.set_ylim(0, 1.0)
        axr.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axr.set_ylabel(overlay_ylabel)
    return axr


def trajectory_panel(ax, sy: pd.DataFrame, value: str, ylabel: str, title: str,
                     twin_data: pd.DataFrame = None, twin_value: str = None,
                     twin_ylabel: str = None):
    """Ten-year strategy trajectory, mean with 95% CI band across the 5 seeds.

    The band's replication unit is the seed (n = 5), a coarser interval than
    the institution-block bootstrap used for the headline contrasts.

    When twin_data is given, twin_value is drawn on a right twin axis as
    dashed lines (no markers, lighter CI band): line style decodes the
    metric (solid = left axis, dashed = right axis) while color keeps
    encoding the strategy. The twin draws no y-grid -- two grids at
    different pitches are unreadable. Returns the twin axes (the caller must
    despine it keeping the right spine), else None.
    """
    sns.lineplot(data=sy, x='year', y=value, hue='strategy',
                 hue_order=STRATEGIES, palette=PALETTE, ax=ax,
                 errorbar=('ci', 95), linewidth=1.4, marker='o',
                 markersize=3, markeredgecolor='white', markeredgewidth=0.5,
                 legend=False, err_kws={'alpha': 0.12, 'linewidth': 0})
    axr = None
    if twin_data is not None:
        axr = ax.twinx()
        sns.lineplot(data=twin_data, x='year', y=twin_value, hue='strategy',
                     hue_order=STRATEGIES, palette=PALETTE, ax=axr,
                     errorbar=('ci', 95), linewidth=1.2, linestyle='--',
                     legend=False, err_kws={'alpha': 0.08, 'linewidth': 0})
        axr.set_ylim(0, None)
        axr.set_ylabel(twin_ylabel)
    # x cosmetics after both lineplots: seaborn autoscaling on the shared-x
    # twin can widen the limits, so these must come last.
    ax.set_xticks(range(2, 11, 2))
    ax.margins(x=0.04)
    ax.set_xlabel('Simulation year')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    return axr


# Forest rows: hypothesis-test outcome column, display label, raw-effect format.
OUTCOMES = [
    ('acceptance_rate', 'Acceptance\nrate', lambda d: f'{d * 100:+.1f}'),
    ('ever_hit', 'Citation-hit\nprob.', lambda d: f'{d * 100:+.1f}'),
    ('log1p_citations', 'log\ncitations', lambda d: f'{d:+.3f}'),
    ('cumulative_funding_received', 'Cumulative\nfunding', lambda d: f'{d:+.1f}'),
]
# (csv contrast, flip sign so it reads cautious-minus-other, comparison
#  strategy, vertical dodge from the outcome's base row)
CONTRASTS = [
    ('explorer-cautious_explorer', True, 'explorer', +0.2),
    ('cautious_explorer-exploiter', False, 'exploiter', -0.2),
]
# Survival values quoted in the paper: (strategy, year, text offset pt, ha, va).
ANNOTATE = [('explorer', 7, (3, 2), 'left', 'bottom'),
            ('exploiter', 7, (-3, -3), 'right', 'top'),
            ('explorer', 10, (-4, 0), 'right', 'center')]


def forest_panel(ax, tests: pd.DataFrame):
    """(c) Cautious-minus-other SMDs with 95% CIs for both comparisons.

    Color marks the COMPARISON strategy the cautious explorer is
    measured against, not the strategy plotted.
    """
    base_ys = np.arange(len(OUTCOMES))[::-1].astype(float)
    any_unresolved = False
    for contrast, flip, other, dodge in CONTRASTS:
        rows = tests[tests.contrast == contrast].set_index('outcome')
        color = PALETTE[other]
        for base_y, (outcome, _, fmt) in zip(base_ys, OUTCOMES):
            r = rows.loc[outcome]
            diff, smd = r.mean_diff, r.smd
            scale = smd / diff  # 1 / pooled SD: rescales raw-unit CI bounds to SMD
            lo, hi = r.ci_lo * scale, r.ci_hi * scale
            if flip:
                diff, smd, lo, hi = -diff, -smd, -hi, -lo
            resolved = lo > 0 or hi < 0
            any_unresolved |= not resolved
            y = base_y + dodge
            ax.errorbar(smd, y, xerr=[[smd - lo], [hi - smd]], fmt='o',
                        color=color, elinewidth=1.0, capsize=1.5, capthick=1.0,
                        markersize=3.2, markeredgewidth=0.8,
                        markerfacecolor=color if resolved else 'white')
            ax.annotate(fmt(diff), xy=(hi, y), xytext=(3, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=5, color=MUTED)

    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_yticks(base_ys, [label for _, label, _ in OUTCOMES])
    ax.set_ylim(-0.5, len(OUTCOMES) + 0.4)  # headroom for the "favors" labels
    ax.set_xlabel('Std. mean difference')
    # Direction labels flank the zero line (mathtext arrows: Arial lacks them).
    for text, offset, ha in [('$\\leftarrow$ favors exploration\nor exploitation', -4, 'right'),
                             ('favors cautious$\\rightarrow$\nexplorers', 4, 'left')]:
        # Semi-transparent box dims the axis and grid lines behind the text.
        ax.annotate(text, xy=(0, 0.985), xycoords=('data', 'axes fraction'),
                    xytext=(offset, 0), textcoords='offset points', ha=ha,
                    va='top', fontsize=5, color=MUTED,
                    bbox={'boxstyle': 'square,pad=0.2', 'facecolor': 'white',
                          'edgecolor': 'none', 'alpha': 0.7})
    if any_unresolved:
        ax.annotate('open marker: 95% CI crosses zero', xy=(0.02, 0.02),
                    xycoords='axes fraction', fontsize=5.5, color=MUTED)
    ax.grid(axis='x', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.margins(x=0.16)  # room for the raw-effect labels past the CI caps
    ax.set_title("Cautious\nexplorer's edge", pad=4)


def survival_panel(ax, agent_year: pd.DataFrame):
    """(d) Percent of researchers still active per year, by strategy."""
    surv = (agent_year.groupby(['strategy', 'seed', 'year'])['is_active'].mean()
            .mul(100).rename('pct').reset_index())
    for (s, _), grp in surv.groupby(['strategy', 'seed']):
        ax.plot(grp['year'], grp['pct'], color=PALETTE[s], alpha=0.18, lw=0.5)
    mean = surv.groupby(['strategy', 'year'])['pct'].mean()
    for s in STRATEGIES:
        ax.plot(mean.loc[s].index, mean.loc[s].to_numpy(), color=PALETTE[s], lw=1.3)
    for s, year, offset, ha, va in ANNOTATE:
        v = float(mean.loc[(s, year)])
        ax.plot(year, v, 'o', ms=2.5, color=PALETTE[s], zorder=3)
        ax.annotate(f'{v:.1f}%', xy=(year, v), xytext=offset,
                    textcoords='offset points', ha=ha, va=va,
                    fontsize=5, color=PALETTE[s])
        print(f'{s} year {year}: {v:.1f}% active')
    ax.set_xlabel('Simulation year')
    # No rotated ylabel: the two-line title carries the quantity and unit,
    # which frees horizontal width for the panels to its left.
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylim(35, 103)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title('%Active\nResearchers', pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis_dir', default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument('--runs_glob', default=DEFAULT_RUNS_GLOB)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--funding_cache', default=DEFAULT_FUNDING_CACHE,
                    help='CSV cache of checkpoint-extracted cumulative '
                         'funding; delete to rebuild')
    ap.add_argument('--e4a_analysis_dir', default=DEFAULT_E4A_ANALYSIS_DIR,
                    help='E4a confirmatory cross-seed analysis dir (panel c)')
    ap.add_argument('--e4a_runs_glob', default=DEFAULT_E4A_RUNS_GLOB,
                    help='E4a confirmatory run dirs for the survival panel')
    args = ap.parse_args()

    analysis_dir = Path(args.analysis_dir)
    agg = pd.read_csv(analysis_dir / 'aggregate_results.csv')
    mc = pd.read_csv(analysis_dir / 'manipulation_check.csv')
    sy = load_run_tables(args.runs_glob, 'strategy_year')
    cum = load_cumulative_funding(args.runs_glob, args.funding_cache)
    tests = pd.read_csv(Path(args.e4a_analysis_dir) / 'primary_hypothesis_tests.csv')
    tests['outcome'] = tests['outcome'].replace(
        {'final_funding': 'cumulative_funding_received'})
    agent_year = load_run_tables(args.e4a_runs_glob, 'agent_year')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # 1x4 strip printed at 0.98\linewidth in the single-column ICLR layout
    # (5.5 in text width -> 5.39 in, scale 0.68, effective base font ~4.8 pt).
    # The dual-axis panels (a, b) get the widest slots so their categorical
    # labels and value annotations stay clear of the twin axes.
    
    # fig, axes = plt.subplots(1, 4, figsize=(7.9, 2.15),
    #                          gridspec_kw={'width_ratios': [1.5, 1.3, 1.0, 0.95]})
    
    fig, axes = plt.subplots(
        1, 4,
        figsize=(7.9, 2.15),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1.5, 1.3, 1.0, 0.95],
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes

    smd = mc.smd_explorer_vs_exploiter_career
    axr_a = strategy_points_panel(
        ax_a, agg, 'career_distance', 'Mean career distance',
        'Behavior and high-impact papers',
        note=f'explorer vs exploiter\nSMD {smd.min():.1f}–{smd.max():.1f}',
        overlay_col='hit_rate',
        overlay_ylabel='P($\\geq$1 high-impact)')  # mathtext: Arial has no U+2265
    axr_b = trajectory_panel(
        ax_b, sy, 'avg_acceptance_rate', 'Acceptance rate',
        'Review and funding outcomes',
        twin_data=cum, twin_value='cum_funding',
        twin_ylabel='Cum. funding received (units)')
    forest_panel(ax_c, tests)
    survival_panel(ax_d, agent_year)

    # Panel c's letter clears its two-line outcome tick labels.
    for letter, ax, xoff in zip('abcd', axes, (-0.20, -0.20, -0.36, -0.20)):
        ax.text(xoff, 1.08, letter, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', va='bottom')
    # Per-axes despine: a figure-level call would also strip the twins'
    # right spines, which carry the second y-axis.
    for base in axes:
        sns.despine(ax=base)
    for twin in (axr_a, axr_b):
        sns.despine(ax=twin, top=True, right=False, left=True, bottom=True)

    handles = [mpl.lines.Line2D([], [], color=PALETTE[s], linewidth=1.4,
                                marker='o', markersize=3.5,
                                markeredgecolor='white', label=LABELS[s])
               for s in STRATEGIES]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.12), columnspacing=1.4, handlelength=1.6)

    # Grey mini-legend decodes metric <-> axis in panel b; color stays
    # strategy-only. Panel a needs none: its bars/dots sit on the left axis
    # and the diamonds lean against the right one, and the caption says so.
    lo, hi = ax_b.get_ylim()
    ax_b.set_ylim(lo, lo + (hi - lo) * 1.15)  # headroom for the mini-legend
    ax_b.legend(handles=[
        mpl.lines.Line2D([], [], color=MUTED, linewidth=1.4, marker='o',
                         markersize=3, markeredgecolor='white',
                         label='Acceptance rate (left)'),
        mpl.lines.Line2D([], [], color=MUTED, linewidth=1.2, linestyle='--',
                         label='Cum. funding (right)')],
        loc='upper left', frameon=False, fontsize=5.5, handlelength=1.8,
        handletextpad=0.5, labelspacing=0.25, borderaxespad=0.2)
    # fig.tight_layout(w_pad=0.1)
    
    fig.get_layout_engine().set(
        w_pad=0.01,
        h_pad=0.02,
        wspace=0.01,
        hspace=0.01,
    )

    for ext in {'pdf'}:
        path = out_dir / f'fig_exploration_strategies.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
