"""Render only fig_scale_combined.pdf.

Run from the project root: python -m utopia.visual.plot_scale_replication
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utopia.utils.visual_utils import nature_style

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/docs'
OUTPUT = ROOT / 'outputs/visual/scale_expansion_analysis_uni300/fig_scale_combined.pdf'

# Cell: (legend label, curve color, text color).
CONDITIONS = {
    'S0R0': ('1 paper/project, scaling', '#42B4D6', '#228BA9'),
    'S0R1': ('1 paper/project, fixed', '#2678B8', '#2678B8'),
    'S1R0': ('2 papers/project, scaling', '#F2A541', '#B77719'),
    'S1R1': ('2 papers/project, fixed', '#E64B35', '#D94431'),
}
FIXED = ('S0R1', 'S1R1')
# The final bin includes four or more awards, as defined in the paper caption.
AWARDS = [('never', '#E8B89D'), ('1', '#B9E4D8'), ('2', '#86D5C3'),
          ('3', '#3AB5A0'), ('4', '#008C7D')]
SECTORS = [('university', 'active_researchers_university', '-'),
           ('industry', 'active_researchers_industry', (0, (2.2, 1.4)))]
INK, AXIS, GRID = '#242A30', '#646B73', '#E8EBEE'


def curve_style(cell):
    color = CONDITIONS[cell][1]
    scaling = cell.endswith('R0')
    return dict(color=color, lw=1.7, marker='o' if cell.startswith('S0') else 's',
                ms=3.5, markerfacecolor='white' if scaling else color,
                markeredgecolor=color if scaling else 'white', markeredgewidth=0.65)


def plot_survival(ax, yearly, cells, sectors):
    for cell in cells:
        rows = yearly[yearly['cell'] == cell].sort_values('year')
        for _, column, linestyle in sectors:
            share = rows[column] / rows[column].max()
            ax.plot(rows['year'], share, linestyle=linestyle,
                    solid_capstyle='round', **curve_style(cell))
            ax.annotate(f'{share.iloc[-1]:.2f}',
                        xy=(rows['year'].iloc[-1], share.iloc[-1]), xytext=(4, 0),
                        textcoords='offset points', va='center',
                        fontsize=7.1, color=CONDITIONS[cell][2])
    last_year = int(yearly['year'].max())
    ax.set(xlim=(1, last_year + 1.2), ylim=(0.2, 1.02),
           xticks=range(2, last_year + 1, 2), yticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('Simulation year', fontsize=7.5, color=INK, labelpad=3)
    ax.set_ylabel('Active fraction', fontsize=7.5, color=INK, labelpad=4)


def plot_awards(ax, runs, cells):
    for x, cell in enumerate(cells):
        histogram = json.loads(runs.loc[cell, 'X3_award_histogram'])
        counts = np.zeros(5)
        for awards, researchers in histogram.items():
            counts[min(int(awards), 4)] += researchers
        bottom = 0
        for share, (label, color) in zip(counts / counts.sum(), AWARDS):
            ax.bar(x, share, bottom=bottom, width=0.72, color=color,
                   edgecolor='white', linewidth=0.6, label=label if x == 0 else None)
            bottom += share
    ticks = ['1\nscaling', '1\nfixed', '2\nscaling', '2\nfixed'] if len(cells) == 4 else ['1', '2']
    ax.set(xlim=(-0.55, len(cells) - 0.45), ylim=(0, 1),
           xticks=range(len(cells)), xticklabels=ticks, yticks=[0, 0.5, 1.0])
    ax.set_xlabel('Papers/project', fontsize=7.5, color=INK, labelpad=3)
    ax.set_ylabel('Fraction', fontsize=7.5, color=INK, labelpad=4)


def main():
    nature_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.8, 2.25), dpi=300,
                             gridspec_kw={'width_ratios': [1.75, 1.1, 1.75, 0.7]})
    populations = [
        (0, 'scale_expansion_analysis_uni300', tuple(CONDITIONS),
         [(None, 'active_researchers', '-')]),
        (2, 'scale_expansion_analysis_default300_panel25_interim', FIXED, SECTORS),
    ]
    for start, folder, cells, sectors in populations:
        runs = pd.read_csv(DATA / folder / 'per_run_endpoints.csv').set_index('cell')
        yearly = pd.read_csv(DATA / folder / 'yearly_checks.csv')
        plot_survival(axes[start], yearly, cells, sectors)
        plot_awards(axes[start + 1], runs, cells)

    titles = ['%Active Researchers', 'Grant awards', '%Active Researchers', 'Grant awards']
    for ax, letter, title in zip(axes, 'abcd', titles):
        ax.grid(axis='y', linewidth=0.45, color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.1, colors=AXIS, length=2.5, width=0.55)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ('left', 'bottom'))
            spine.set_color(AXIS)
            spine.set_linewidth(0.55)
        ax.text(0, 1.025, letter, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=9, fontweight='bold', color=INK)
        ax.annotate(title, xy=(0, 1.025), xycoords='axes fraction', xytext=(10, 0),
                    textcoords='offset points', ha='left', va='bottom',
                    fontsize=8, color=INK, annotation_clip=False)
    for ax, cells in [(axes[1], tuple(CONDITIONS)), (axes[3], FIXED)]:
        for tick, cell in zip(ax.get_xticklabels(), cells):
            tick.set_color(CONDITIONS[cell][2])

    sector_handles = [plt.Line2D([], [], color=AXIS, lw=1.5, linestyle=style, label=label)
                      for label, _, style in SECTORS]
    axes[2].legend(handles=sector_handles, loc='lower left', frameon=False, fontsize=7.1,
                   handlelength=1.5, borderaxespad=0.2, labelspacing=0.25,
                   handletextpad=0.5, labelcolor=INK)
    handles, labels = axes[3].get_legend_handles_labels()
    axes[3].legend(handles[::-1], labels[::-1], loc='upper left',
                   bbox_to_anchor=(1.01, 1), frameon=False, fontsize=7.1,
                   handlelength=1, handleheight=1, labelspacing=0.35,
                   borderaxespad=0, labelcolor=INK)

    fig.subplots_adjust(left=0.055, right=0.944, bottom=0.29, top=0.79, wspace=0.29)
    for start, title in [(0, 'University Researchers Only'), (2, 'Mixed Population')]:
        x0, x1 = axes[start].get_position().x0, axes[start + 1].get_position().x1
        fig.text((x0 + x1) / 2, 0.925, title, ha='center', va='center',
                 fontsize=8.3, fontweight='bold', color=INK)

    # Keep the condition legend two points below the lowest axis label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bottom = min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes)
    legend_top = (label_bottom - 2 * fig.dpi / 72) / fig.bbox.height
    handles = [plt.Line2D([], [], label=label, **curve_style(cell))
               for cell, (label, _, _) in CONDITIONS.items()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, legend_top),
               ncol=4, frameon=False, fontsize=7.2, handlelength=1.5,
               columnspacing=1.15, handletextpad=0.5, borderaxespad=0,
               borderpad=0, labelcolor=INK)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight')
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
