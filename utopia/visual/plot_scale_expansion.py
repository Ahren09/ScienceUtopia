"""Paper figures for the scale-expansion experiment (design v1).

Experiment
----------
Compact university-only worlds (20 institutions x 5 balanced researchers, 8
years, 6 conferences, Qwen3-32B) crossing four binary design factors:
S papers per completed project (1 vs 2), R agency budget (tracks applications
vs fixed, anchored to the initial population), K review capacity (3 reviews per
paper vs a per-reviewer annual cap) and E evaluation scheme (persona reviewers
vs one standardized policy). Cells are named S{0,1}R{0,1}[_K1][E1]; 5 seeds per
cell, the simulated world is the unit of replication.

Panels
------
Figure 1 (2x2) primary endpoints, one panel each, cells on the x axis:
  (a) P1 founder survival at year h    (b) P2 share of review slots on resubmissions
  (c) P3 Gini of cumulative awards      (d) P4 share of accepted papers with a
                                            non-self citation within h years
  Grey dots = seeds; coloured marker = cell mean with seed-bootstrap 95% CI.
Figure 2 (2x2) manipulation checks vs year, mean over seeds per cell:
  submissions, university awards per founder, reviews per paper, active
  researchers. These are settings that the factors move mechanically; they are
  shown so readers can see the pressure each cell actually applied.

Data provenance
---------------
outputs/docs/scale_expansion_analysis/{per_run_endpoints,cell_summary,yearly_checks}.csv
written by utopia.analysis.scale_expansion (from final yearly checkpoints and
funding_applications_year_<y>.jsonl logs).

Run
---
python -m utopia.visual.plot_scale_expansion
python -m utopia.visual.plot_scale_expansion --analysis_dir outputs/docs/scale_expansion_analysis \
    --out_dir outputs/visual/scale_expansion_analysis
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.analysis.scale_expansion import CELL_ORDER, PRIMARY
from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

PANEL_TITLES = {
    'P1_survival_h': 'Founder survival at year h',
    'P2_resub_review_share': 'Review slots spent on resubmissions',
    'P3_award_gini': 'Gini of cumulative awards',
    'P4_followup_share': 'Accepted papers with non-self follow-up',
}
CHECKS = [('submissions', 'Submissions per year'),
          ('awards_per_founder', 'University awards per founder'),
          ('reviews_per_paper', 'Reviews per paper'),
          ('active_researchers', 'Active researchers')]


def cell_palette(cells):
    colors = sns.color_palette('colorblind', n_colors=max(3, len(cells)))
    return {c: colors[i] for i, c in enumerate(cells)}


def plot_primary(per_run: pd.DataFrame, summary: pd.DataFrame, out_dir: str):
    cells = [c for c in CELL_ORDER if c in set(summary['cell'])]
    pal = cell_palette(cells)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for ax, metric, letter in zip(axes.ravel(), PRIMARY, 'abcd'):
        for i, cell in enumerate(cells):
            vals = per_run.loc[per_run['cell'] == cell, metric].dropna()
            ax.scatter([i] * len(vals), vals, s=9, color=MUTED, alpha=0.6, zorder=2, linewidths=0)
            row = summary.loc[summary['cell'] == cell].iloc[0]
            mean, lo, hi = row[f'{metric}_mean'], row[f'{metric}_ci_lo'], row[f'{metric}_ci_hi']
            if pd.notna(mean):
                err = None if pd.isna(lo) else [[mean - lo], [hi - mean]]
                ax.errorbar(i, mean, yerr=err, fmt='o', ms=4, color=pal[cell], ecolor=pal[cell],
                            elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels(cells, rotation=35, ha='right')
        ax.set_title(f'{letter}  {PANEL_TITLES[metric]}')
        ax.grid(axis='y', color=GRID, linewidth=0.5)
        sns.despine(ax=ax)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'fig_scale_primary_endpoints.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_checks(yearly: pd.DataFrame, out_dir: str):
    cells = [c for c in CELL_ORDER if c in set(yearly['cell'])]
    pal = cell_palette(cells)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for ax, (col, title), letter in zip(axes.ravel(), CHECKS, 'abcd'):
        for cell in cells:
            g = yearly[yearly['cell'] == cell].groupby('year')[col].mean()
            ax.plot(g.index, g.values, marker='o', ms=2.5, lw=1.0, color=pal[cell], label=cell)
        ax.set_title(f'{letter}  {title}')
        ax.grid(color=GRID, linewidth=0.5)
        sns.despine(ax=ax)
    axes[1, 0].set_xlabel('Year'); axes[1, 1].set_xlabel('Year')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(5, len(cells)), frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'fig_scale_yearly_checks.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analysis_dir', default=os.path.join('outputs', 'docs', 'scale_expansion_analysis'))
    parser.add_argument('--out_dir', default=os.path.join('outputs', 'visual', 'scale_expansion_analysis'))
    args = parser.parse_args()
    nature_style()
    os.makedirs(args.out_dir, exist_ok=True)
    per_run = pd.read_csv(os.path.join(args.analysis_dir, 'per_run_endpoints.csv'))
    summary = pd.read_csv(os.path.join(args.analysis_dir, 'cell_summary.csv'))
    yearly = pd.read_csv(os.path.join(args.analysis_dir, 'yearly_checks.csv'))
    plot_primary(per_run, summary, args.out_dir)
    plot_checks(yearly, args.out_dir)
    print(f'figures written to {args.out_dir}')


if __name__ == '__main__':
    main()
