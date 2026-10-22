"""Paper Figure 1 for the influx2x2 experiment (researcher influx x recycling).

Experiment: 2x2 factorial crossing researcher-population influx (staggered
exact 10%/year of the initial 300, +30 juniors/year, 300 -> 600 by year 10)
with manuscript resubmission (Phase 0 enabled/disabled). Single seed 42,
10 years, fixed 30% acceptance, Qwen3-32B. Cells: A = no influx/no recycling,
B = influx only, C = recycling only (paper-baseline twin), D = both (flagship).

Factorial visual encoding (so the 2x2 design is legible at a glance):
    COLOR     = recycling  (cobalt #1F6FB5 = off, NPG red #E5493A = on)
    LINESTYLE = influx     (dashed = off, solid = on)
Vivid accent trio shared with the review-bias figure (visual_utils.VIVID_*).
No error bars: results are one seed-matched counterfactual set.

Main figure (fig_influx_factorial):
    (a) Annual submission events, 4 curves, year-10 end labels 193/364/489/867.
        A oscillates flat, B grows ~linearly with population, C plateaus,
        D keeps climbing - the (D-B)-(C-A) gap is the influx-recycling
        interaction (+207 events, 31% of the total effect).
    (b) First submissions vs resubmissions in the combined world D (stacked
        bars); by year 10, 61% of submission events are recycled manuscripts.
    (c) Reviews per ACTIVE reviewer (denominator: end-of-year is_active &
        can_review agents). Influx alone leaves burden flat (B lies on A);
        recycling nearly triples it (year-10: 2.25/2.07 vs 6.64/5.48).

Supplementary figure (fig_influx_multiplier, appendix): recycling multiplier
M_t = total submission events_t / first submissions_t for the two recycling
worlds (contemporaneous ratio; identically 1 without recycling).

Data provenance: the four per-cell resubmission_stats.json files produced by
utopia.analysis.resubmission_stats from outputs/influx2x2_{A,B,C,D}_*_seed42/.

Run (after utopia.analysis.influx_factorial):
    python -m scripts.figures.plot_influx_factorial
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.figures.visual_utils import (GRID, INK, MUTED, VIVID_COOL, VIVID_GOLD,
                                       VIVID_WARM, nature_style)

DEFAULT_STATS = {
    'A': 'outputs/docs/resubmission_stats_influx2x2_A_noinflux_noresub_seed42/resubmission_stats.json',
    'B': 'outputs/docs/resubmission_stats_influx2x2_B_influx_noresub_seed42/resubmission_stats.json',
    'C': 'outputs/docs/resubmission_stats_influx2x2_C_noinflux_resub_seed42/resubmission_stats.json',
    'D': 'outputs/docs/resubmission_stats_influx2x2_D_influx_resub_seed42/resubmission_stats.json',
}
DEFAULT_OUT_DIR = 'outputs/visual/influx2x2_qwen3_32b_seed42'

# Factorial encoding: color = recycling, linestyle = influx (see docstring).
COOL, WARM = VIVID_COOL, VIVID_WARM    # cobalt / NPG red
CELL_ORDER = ['A', 'B', 'C', 'D']
CELL_STYLE = {
    'A': {'color': COOL, 'linestyle': (0, (4, 2))},
    'B': {'color': COOL, 'linestyle': '-'},
    'C': {'color': WARM, 'linestyle': (0, (4, 2))},
    'D': {'color': WARM, 'linestyle': '-'},
}
CELL_NICE = {
    'A': 'No growth, no resubmission',
    'B': 'Population growth only',
    'C': 'Resubmission only',
    'D': 'Growth + resubmission',
}
# Panel-b stack: first submissions in gold, resubmissions in the recycling red.
FIRST_COLOR, RESUB_COLOR = VIVID_GOLD, VIVID_WARM


def metric_series(stats: dict, field: str, years) -> list:
    d = stats[field]
    return [d.get(y, d.get(str(y), 0)) for y in years]


def factorial_lines(ax, stats_by_cell, field, years):
    for cell in CELL_ORDER:
        ax.plot(years, metric_series(stats_by_cell[cell], field, years),
                linewidth=1.2, label=CELL_NICE[cell], **CELL_STYLE[cell])


def end_labels(ax, stats_by_cell, field, years, fmt, offsets):
    """Direct year-10 value labels at line ends (D bold)."""
    x = years[-1]
    for cell in CELL_ORDER:
        y = metric_series(stats_by_cell[cell], field, years)[-1]
        ax.annotate(fmt(y), xy=(x, y), xytext=(4, offsets.get(cell, 0)),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=6, color=CELL_STYLE[cell]['color'],
                    fontweight='bold' if cell == 'D' else 'normal')


def volume_panel(ax, stats_by_cell: dict, years):
    factorial_lines(ax, stats_by_cell, 'submissions_per_year', years)
    end_labels(ax, stats_by_cell, 'submissions_per_year', years,
                lambda v: f'{v:,.0f}', {'A': -2, 'B': 0, 'C': 2, 'D': 0})
    ax.set_xlabel('Year', labelpad=1)
    ax.set_ylabel('Submission events')
    ax.set_title('Submissions compound', pad=2)
    ax.set_xticks(years[::3])
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.6)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def composition_panel(ax, stats: dict, years):
    comp = stats['attempt_composition_by_year']
    firsts, resubs = [], []
    for y in years:
        c = comp.get(y, comp.get(str(y), {}))
        firsts.append(c.get('first', 0))
        resubs.append(c.get('resub_1st', 0) + c.get('resub_2plus', 0))
    ax.bar(years, firsts, width=0.72, color=FIRST_COLOR,
           label='First submissions', edgecolor='white', linewidth=0.3)
    ax.bar(years, resubs, bottom=firsts, width=0.72, color=RESUB_COLOR,
           label='Resubmissions', edgecolor='white', linewidth=0.3)
    total10 = firsts[-1] + resubs[-1]
    share = resubs[-1] / total10 * 100

    ax.margins(y=0.12)
    ax.set_xlabel('Year', labelpad=1)
    ax.set_ylabel('Submission events')
    ax.set_title('Resubmissions dominate', pad=2)
    ax.set_xticks(years[::3])
    ax.legend(fontsize=6, frameon=False, loc='upper left')
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def burden_panel(ax, stats_by_cell: dict, years):
    factorial_lines(ax, stats_by_cell, 'reviews_per_active_reviewer_by_year', years)
    end_labels(ax, stats_by_cell, 'reviews_per_active_reviewer_by_year', years,
                lambda v: f'{v:.2f}', {'A': 4, 'B': -4, 'C': 2, 'D': -2})
    ax.set_xlabel('Year', labelpad=1)
    ax.set_ylabel('Reviews per active reviewer')
    ax.set_title('Resubmission drives workload', pad=2)
    ax.set_xticks(years[::3])
    ax.set_xlim(years[0] - 0.3, years[-1] + 1.9)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def multiplier_panel(ax, stats_by_cell: dict, years, *, show_legend=True):
    """Draw the archived multiplier series, also used by the combined appendix."""
    styles = {'C': {'color': COOL, 'linestyle': (0, (4, 2))},
              'D': {'color': WARM, 'linestyle': '-'}}
    for cell in ('C', 'D'):
        records = stats_by_cell[cell]['recycling_multiplier_by_year']
        values = [float(records.get(str(y), records.get(y, float('nan'))))
                  for y in years]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f'Missing or nonfinite multiplier for cell {cell}')
        ax.plot(years, values,
                linewidth=1.2, marker='o' if cell == 'C' else 's', markersize=2.7,
                markeredgecolor='white', markeredgewidth=0.3,
                label=CELL_NICE[cell], **styles[cell])
        ax.annotate(f'{values[-1]:.2f}', xy=(years[-1], values[-1]),
                    xytext=(4, 6 if cell == 'C' else -6),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=6.5, color=styles[cell]['color'])
    ax.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=':')
    ax.annotate('No resubmission', xy=(years[0], 1.0), xytext=(0, 5),
                textcoords='offset points', fontsize=6.2, color=MUTED)
    ax.set_xlabel('Simulation year', labelpad=3)
    ax.set_ylabel('Submissions / first submissions', labelpad=3)
    ax.set_title('Submission multiplier', pad=5)
    ax.set_xticks(years[::3])
    ax.set_xlim(years[0] - 0.25, years[-1] + 1.65)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.85, 4.4)
    if show_legend:
        ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def multiplier_figure(stats_by_cell: dict, years, out_dir: Path):
    """Standalone export retained for callers of the factorial figure script."""
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    multiplier_panel(ax, stats_by_cell, years)
    sns.despine(fig)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_influx_multiplier.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for cell, default in DEFAULT_STATS.items():
        ap.add_argument(f'--stats_{cell}', default=default)
    ap.add_argument('--num_years', type=int, default=10)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    stats_by_cell = {c: json.load(open(getattr(args, f'stats_{c}'))) for c in CELL_ORDER}
    years = list(range(1, args.num_years + 1))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    volume_panel(axes[0], stats_by_cell, years)
    composition_panel(axes[1], stats_by_cell['D'], years)
    burden_panel(axes[2], stats_by_cell, years)
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.16, 1.10, letter, transform=ax.transAxes,
                fontsize=8, fontweight='bold', color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
    sns.despine(fig)
    fig.tight_layout()

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_influx_factorial.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)

    multiplier_figure(stats_by_cell, years, out_dir)


if __name__ == '__main__':
    main()
