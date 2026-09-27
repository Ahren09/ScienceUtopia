"""Paper figure: novelty vs. citation payoff, one subplot per strategy.

Companion to utopia.visual.plot_exploration_strategies (see its docstring for
the experiment setup). Small multiples of the risk-return relation: each
subplot shows one strategy's accepted papers as a scatter of paper novelty
score (embedding distance from the field centroid) against log(1 + citations)
measured three years after publication, with a distribution-free trend line
(mean citation outcome within novelty-score octiles of that strategy's own
papers).

How to read: the horizontal span of each cloud is where that strategy's
papers actually live on the novelty axis -- exploiters concentrate at low
novelty, explorers at high novelty, cautious explorers in between -- so each
trend line is only drawn over its strategy's support. Comparing panels shows
whether extra novelty pays off within each strategy, not just between
strategies. Axes are shared across panels.

Run:
    python -m utopia.visual.plot_exploration_paper_novelty_scores
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.visual.common import GRID, MUTED, nature_style
from utopia.visual.plot_exploration_strategies import DEFAULT_OUT_DIR, DEFAULT_RUNS_GLOB, LABELS, PALETTE, STRATEGIES, load_run_tables

ANNOT_FS = 8  # in-axes annotation type (n-count, trend-line note)


def risk_return_subplot(ax, sub: pd.DataFrame, strat: str, n_bins: int = 8):
    """One strategy's papers: scatter plus novelty-octile mean trend.

    The dotted trend line is built in three steps:
      1. Split this strategy's own papers into n_bins quantile bins of
         novelty score (pd.qcut) -- equal COUNT per bin, not equal width,
         so all points average the same number of papers and are equally
         reliable even in the thin tails of the novelty distribution.
      2. Within each bin take two means: mean novelty (the point's x) and
         mean log(1+citations) at age 3 (the point's y).
      3. Connect the points.
    It answers: within this strategy, do more novel papers earn more
    citations? A distribution-free payoff curve -- no functional form is
    assumed (unlike a fitted line), and the long-tailed citation noise is
    averaged out (unlike the raw scatter).

    Because the bins are cut from the strategy's own novelty distribution,
    the line spans only that strategy's actual support and never
    extrapolates -- e.g. the explorer's line starts near novelty 0.6.
    The scatter is rasterized to keep the PDF light.
    """
    ax.scatter(sub.novelty_score, sub.log_cit, s=3, alpha=0.25,
               color=PALETTE[strat], linewidths=0, rasterized=True, zorder=1)
    bins = pd.qcut(sub.novelty_score, n_bins, duplicates='drop')
    trend = sub.groupby(bins, observed=True).agg(
        x=('novelty_score', 'mean'), y=('log_cit', 'mean'))
    ax.plot(trend.x, trend.y, color=PALETTE[strat], linewidth=1.4,
            marker='o', markersize=3, markeredgecolor='white',
            markeredgewidth=0.5, zorder=3)
    ax.set_title(LABELS[strat], pad=4, color=PALETTE[strat])
    ax.annotate(f'n = {len(sub):,} papers', xy=(0.03, 0.96),
                xycoords='axes fraction', va='top', fontsize=ANNOT_FS, color=MUTED)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs_glob', default=DEFAULT_RUNS_GLOB)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    pa = load_run_tables(args.runs_glob, 'paper_age')
    pa = pa[pa.age == 3].copy()
    pa['log_cit'] = np.log1p(pa.citations)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # This figure spans the full double-column width but is short, so the shared
    # 6-7 pt type prints too small. Bump type up for THIS figure only (the shared
    # nature_style is untouched) and give the panels more height so the larger
    # labels are not cramped.
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
    })
    # Double-column width, one panel per strategy, shared axes; taller than the
    # original 1.76 in so the enlarged labels breathe.
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.9), sharex=True,
                             sharey=True)
    for ax, strat in zip(axes, STRATEGIES):
        risk_return_subplot(ax, pa[pa.strategy == strat], strat)
    axes[0].set_ylabel('log(1+citations) at age 3')
    axes[1].set_xlabel('Topical departure', labelpad=-0.5)
    # Under the explorer panel's n-count, where the point cloud leaves space.
    # axes[0].annotate('Lines: mean per novelty octile', xy=(0.03, 0.85),
    #                  xycoords='axes fraction', va='top', fontsize=ANNOT_FS,
    #                  color=MUTED)
    sns.despine(fig)
    fig.tight_layout(w_pad=0.3)

    for ext in {'pdf'}:
        path = out_dir / f'fig_exploration_paper_novelty_scores.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
