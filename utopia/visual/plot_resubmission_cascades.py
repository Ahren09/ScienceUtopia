"""Paper figure for the resubmission-cascade study (fixed vs declining acceptance).

Reads the two resubmission_stats.json files produced by
utopia.analysis.resubmission_stats for the paired worlds and renders a single
cascade-depth panel: the mean number of venue attempts a rejected paper makes
before acceptance or abandonment (bootstrap 95% CI), with the resubmission
(recycle) rate annotated per regime.

Layout: the compact left panel (~1.4 in) of the paper's two-panel
"Resubmission cascades" figure (right panel: plot_resubmission_difficulty),
placed side by side in a wrapfigure of about half a line (one-column build)
or a column-width float (two-column build). The panel letter lives in the
title, so the LaTeX side needs no subfigure captions.

Run:
    python -m utopia.visual.plot_resubmission_cascades
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from utopia.visual.common import GRID, INK, MUTED, nature_style

DEFAULT_FIXED = ('outputs/docs/resubmission_stats_default_Qwen3-32B_fixed30_paired/'
                 'resubmission_stats.json')
DEFAULT_DECLINING = ('outputs/docs/resubmission_stats_acl-acceptance-rates_Qwen3-32B/'
                     'resubmission_stats.json')
DEFAULT_OUT_DIR = 'outputs/visual/resubmission_cascades_fixed30_vs_acl'

# Two-regime palette validated for CVD separation and lightness band (dataviz
# skill, slots 1-2).
REGIMES = ['Fixed', 'Declining']
PALETTE = {'Fixed': '#2a78d6', 'Declining': '#1baf7a'}


def cascade_panel(ax, stats: dict):
    """Point estimate with bootstrap 95% CI; recycle rate annotated per regime."""
    xs = range(len(REGIMES))
    bounds = []
    for x, regime in zip(xs, REGIMES):
        s = stats[regime]
        mean, (lo, hi) = s['cascade_depth_mean'], s['cascade_depth_ci95']
        bounds += [lo, hi]
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt='none',
                    ecolor=PALETTE[regime], elinewidth=1.1, capsize=2.5, capthick=1.1)
        sns.scatterplot(x=[x], y=[mean], ax=ax, color=PALETTE[regime], s=24,
                        edgecolor='white', linewidth=0.6, zorder=3, legend=False)
        # Stack both labels above the upper CI cap so nothing collides with
        # the axis or tick labels below.
        ax.annotate(f'{mean:.2f}', xy=(x, hi), xytext=(0, 11),
                    textcoords='offset points', ha='center', fontsize=6, color=INK)
        ax.annotate(f"{s['recycle_rate_pct']:.0f}% resubmitted",
                    xy=(x, hi), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=5, color=MUTED)
    span = max(bounds) - min(bounds)
    ax.set_ylim(min(bounds) - 0.10 * span, max(bounds) + 0.30 * span)
    ax.set_xticks(list(xs), [f'{r} 30%' if r == 'Fixed' else f'{r}'
                             for r in REGIMES])
    ax.set_xlim(-0.75, len(REGIMES) - 0.25)  # air for the centred labels
    ax.set_ylabel('Attempts per rejected paper')
    ax.set_title('a  Cascade depth', pad=2)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixed_stats', default=DEFAULT_FIXED)
    ap.add_argument('--declining_stats', default=DEFAULT_DECLINING)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    # Internal keys must stay aligned with REGIMES/PALETTE.
    stats = {'Fixed': json.load(open(args.fixed_stats)),
             'Declining': json.load(open(args.declining_stats))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Left panel of a side-by-side pair (~1.4 in of the pair's ~3.1 in).
    fig, ax = plt.subplots(figsize=(1.5, 1.6))
    cascade_panel(ax, stats)
    sns.despine(fig)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_resubmission_cascades.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
