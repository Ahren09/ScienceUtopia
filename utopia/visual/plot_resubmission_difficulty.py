"""Single-panel paper figure: the rising acceptance price for resubmitted papers.

For resubmitted papers accepted in year y, plots the mean number of prior
rejections they survived (95% CI band), from the
resub_prior_rejections_by_accept_year field of the two resubmission_stats.json
files (fixed vs declining acceptance regime). A rising curve shows that getting
a recycled paper accepted requires enduring ever more rejections.

Run:
    python -m utopia.visual.plot_resubmission_difficulty
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style
from utopia.visual.plot_resubmission_cascades import (
    DEFAULT_DECLINING, DEFAULT_FIXED, DEFAULT_OUT_DIR, PALETTE, REGIMES)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixed_stats', default=DEFAULT_FIXED)
    ap.add_argument('--declining_stats', default=DEFAULT_DECLINING)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    stats = {'Fixed': json.load(open(args.fixed_stats)),
             'Declining': json.load(open(args.declining_stats))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Single-column Nature width (88 mm = 3.46 in).
    fig, ax = plt.subplots(figsize=(3.2, 2.5))
    ends = {}
    for regime in REGIMES:
        rec = stats[regime]['resub_prior_rejections_by_accept_year']
        df = pd.DataFrame({'year': [int(y) for y in rec],
                           'mean': [v['mean'] for v in rec.values()],
                           'sem': [v['sem'] for v in rec.values()]}
                          ).sort_values('year')
        sns.lineplot(data=df, x='year', y='mean', ax=ax, color=PALETTE[regime],
                     linewidth=1.4, marker='o', markersize=3.5,
                     markeredgecolor='white', markeredgewidth=0.6, label=regime)
        ax.fill_between(df['year'], df['mean'] - 1.96 * df['sem'],
                        df['mean'] + 1.96 * df['sem'],
                        color=PALETTE[regime], alpha=0.12, linewidth=0)
        ends[regime] = df['mean'].iloc[-1]
    
    """
    # Annotate the end value
    for regime, y in ends.items():
        dy = 4 if y >= max(ends.values()) else -4  # dodge converging labels
        ax.annotate(f'{y:.2f}', xy=(df['year'].max(), y), xytext=(3, dy),
                    textcoords='offset points', va='center', fontsize=6,
                    color=INK)
    """
        
    
    ax.set_xticks(range(2, 11, 2))
    ax.margins(x=0.04)
    ax.set_xlabel('Acceptance year')
    ax.set_ylabel('Prior rejections per accepted resubmission')
    # ax.set_title('Rising cost of resubmission acceptance', pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, handlelength=1.2, loc='upper left',title='Acceptance Rate',
              borderaxespad=0.2)
    ax.annotate('Shading: 95% CI', xy=(0.98, 0.04), xycoords='axes fraction',
                ha='right', fontsize=5.5, color=MUTED)
    sns.despine(fig)
    fig.tight_layout()

    for ext in {'pdf'}:
        path = out_dir / f'fig_resubmission_difficulty.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
