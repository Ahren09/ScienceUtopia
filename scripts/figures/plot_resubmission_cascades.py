"""Unified three-panel appendix figure for resubmission dynamics.

(a) Annual submission multiplier in the archived growth/no-growth worlds.
(b) Cascade depth in the separate fixed/declining-acceptance pair, retaining
    the supplied paper-level bootstrap 95% confidence intervals.
(c) Prior rejections of accepted resubmissions in that same acceptance pair,
    retaining the original mean +/- 1.96 SEM bands.

Each condition is one ten-year world. The intervals describe papers within a
world, not variation across independently replicated worlds. No simulation,
resampling, smoothing, or new statistical estimation is performed here.

Run from the ScienceUtopia repository:
    python -m scripts.figures.plot_resubmission_cascades

Exports one PDF, PNG, SVG, and a JSON record of the plotted values and source
hashes. The PDF is authored at the paper's 5.5-inch text width.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from scripts.figures.visual_utils import (
    GRID, MUTED, VIVID_COOL, VIVID_WARM, nature_style,
)
from scripts.figures.plot_influx_factorial import DEFAULT_STATS, multiplier_panel

DEFAULT_FIXED = ('outputs/docs/resubmission_stats_default_Qwen3-32B_fixed30_paired/'
                 'resubmission_stats.json')
DEFAULT_DECLINING = ('outputs/docs/resubmission_stats_acl-acceptance-rates_Qwen3-32B/'
                     'resubmission_stats.json')
DEFAULT_OUT_DIR = 'outputs/visual/resubmission_cascades_fixed30_vs_acl'

REGIMES = ['Fixed', 'Declining']
PALETTE = {'Fixed': VIVID_COOL, 'Declining': VIVID_WARM}
REGIME_STYLE = {
    'Fixed': {'marker': 'o', 'linestyle': '-'},
    'Declining': {'marker': 's', 'linestyle': (0, (4, 2))},
}


def cascade_panel(ax, stats: dict):
    """Show the original point estimates, intervals, and resubmission rates."""
    for x, regime in enumerate(REGIMES):
        record = stats[regime]
        mean = float(record['cascade_depth_mean'])
        lo, hi = map(float, record['cascade_depth_ci95'])
        rate = float(record['recycle_rate_pct'])
        if not all(map(math.isfinite, (mean, lo, hi, rate))) or not lo <= mean <= hi:
            raise ValueError(f'Invalid cascade estimate or interval for {regime}')
        ax.errorbar(
            x, mean, yerr=[[mean - lo], [hi - mean]],
            fmt=REGIME_STYLE[regime]['marker'], color=PALETTE[regime],
            markersize=4.6, markeredgecolor='white', markeredgewidth=0.6,
            elinewidth=1.2, capsize=3, capthick=1.0, zorder=3,
        )
        ax.annotate(f'{mean:.2f}', xy=(x, hi), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=PALETTE[regime])
        ax.text(x, 4.36, f'{rate:.0f}%\nresubmitted', ha='center', va='top',
                fontsize=6.4, color=MUTED, linespacing=1.15)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(3.45, 4.4)
    ax.set_yticks([3.5, 3.75, 4.0, 4.25], ['3.50', '3.75', '4.00', '4.25'])
    ax.set_xticks([0, 1], ['Fixed 30%', 'Declining'])
    ax.set_xlabel('Acceptance rate', labelpad=3)
    ax.set_ylabel('Attempts per rejected paper', labelpad=3)
    ax.set_title('Cascade depth', pad=5)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def acceptance_series(stats: dict) -> dict:
    """Retain the original observations and normal-approximation intervals."""
    series = {}
    for regime in REGIMES:
        records = stats[regime]['resub_prior_rejections_by_accept_year']
        rows = sorted((int(year), record) for year, record in records.items())
        years = np.array([year for year, _ in rows])
        means = np.array([record['mean'] for _, record in rows], dtype=float)
        sems = np.array([record['sem'] for _, record in rows], dtype=float)
        counts = np.array([record['n'] for _, record in rows], dtype=int)
        if (not len(rows) or not np.isfinite(means).all()
                or not np.isfinite(sems).all() or (sems < 0).any()
                or (counts <= 0).any()):
            raise ValueError(f'Invalid acceptance-year statistics for {regime}')
        series[regime] = {
            'year': years, 'mean': means, 'sem': sems, 'n': counts,
            'ci_low': means - 1.96 * sems, 'ci_high': means + 1.96 * sems,
        }
    return series


def acceptance_panel(ax, stats: dict, *, show_legend=True):
    """Draw acceptance-year means with the same bands as the previous figure."""
    series = acceptance_series(stats)
    for regime in REGIMES:
        values = series[regime]
        ax.fill_between(values['year'], values['ci_low'], values['ci_high'],
                        color=PALETTE[regime], alpha=0.13, linewidth=0)
        ax.plot(values['year'], values['mean'], color=PALETTE[regime],
                linewidth=1.2, markersize=2.7, markeredgecolor='white',
                markeredgewidth=0.3, label=regime, **REGIME_STYLE[regime])
    years = sorted({int(year) for values in series.values()
                    for year in values['year']})
    ax.set_xticks([year for year in years
                   if year % 2 == 0 or year in (years[0], years[-1])])
    ax.set_xlim(years[0] - 0.25, years[-1] + 0.35)
    ax.set_ylim(0.8, 4.0)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlabel('Acceptance year', labelpad=3)
    ax.set_ylabel('Prior rejections per\naccepted resubmission', labelpad=3)
    ax.set_title('Acceptance price', pad=5)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(frameon=False, loc='upper left', fontsize=6.5)


def make_figure(multiplier_stats: dict, acceptance_stats: dict, years):
    """Lay out all panels once, at their final size in the manuscript."""
    nature_style()
    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
        'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    }):
        fig, axes = plt.subplots(
            1, 3, figsize=(5.5, 2.4),
            gridspec_kw={'left': 0.095, 'right': 0.985, 'bottom': 0.17,
                         'top': 0.735, 'wspace': 0.60},
        )
        multiplier_panel(axes[0], multiplier_stats, years, show_legend=False)
        cascade_panel(axes[1], acceptance_stats)
        acceptance_panel(axes[2], acceptance_stats, show_legend=False)

        for letter, ax in zip('abc', axes):
            ax.set_title(f'{letter}  {ax.get_title(loc="left")}', pad=6)
        sns.despine(fig)

        first = axes[0].get_position()
        middle = axes[1].get_position()
        last = axes[2].get_position()
        fig.text(first.x0, 0.98, 'Population-growth comparison',
                 ha='left', va='top', fontsize=7, color=MUTED)
        fig.text((middle.x0 + last.x1) / 2, 0.98, 'Acceptance-rate comparison',
                 ha='center', va='top', fontsize=7, color=MUTED)
        for left, right in [(first.x0, first.x1), (middle.x0, last.x1)]:
            fig.add_artist(Line2D([left, right], [0.925, 0.925],
                                 transform=fig.transFigure, color=GRID,
                                 linewidth=0.6))

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left',
                   bbox_to_anchor=(first.x0, 0.913), frameon=False,
                   fontsize=6.5, handlelength=2, borderaxespad=0,
                   labelspacing=0.35, handletextpad=0.5)
        handles = [
            Line2D([], [], color=PALETTE[regime], linewidth=1.2,
                   markersize=3.5, **REGIME_STYLE[regime])
            for regime in REGIMES
        ]
        fig.legend(handles, ['Fixed 30%', 'Declining (ACL)'],
                   loc='upper center',
                   bbox_to_anchor=((middle.x0 + last.x1) / 2, 0.89),
                   ncol=2, frameon=False, fontsize=6.5, handlelength=2,
                   columnspacing=1.4, handletextpad=0.5, borderaxespad=0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats_C', default=DEFAULT_STATS['C'])
    parser.add_argument('--stats_D', default=DEFAULT_STATS['D'])
    parser.add_argument('--fixed_stats', default=DEFAULT_FIXED)
    parser.add_argument('--declining_stats', default=DEFAULT_DECLINING)
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    paths = {
        'C': Path(args.stats_C), 'D': Path(args.stats_D),
        'Fixed': Path(args.fixed_stats), 'Declining': Path(args.declining_stats),
    }
    stats = {name: json.loads(path.read_text()) for name, path in paths.items()}
    multiplier_stats = {cell: stats[cell] for cell in ('C', 'D')}
    acceptance_stats = {regime: stats[regime] for regime in REGIMES}
    years = list(range(1, 11))
    fig = make_figure(multiplier_stats, acceptance_stats, years)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / 'fig_resubmission_cascades'
    for extension in ('pdf', 'png', 'svg'):
        path = stem.with_suffix(f'.{extension}')
        # A fixed canvas keeps the embedded font sizes unchanged at \linewidth.
        fig.savefig(path, dpi=300, facecolor='white')
        print(f'Saved {path}')
    plt.close(fig)

    record = {
        'sources': {
            name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in paths.items()
        },
        'panel_a': {
            cell: multiplier_stats[cell]['recycling_multiplier_by_year']
            for cell in ('C', 'D')
        },
        'panel_b': {
            regime: {key: acceptance_stats[regime][key] for key in
                     ('cascade_depth_mean', 'cascade_depth_ci95', 'recycle_rate_pct',
                      'n_rejected_papers')}
            for regime in REGIMES
        },
        'panel_c': {
            regime: {key: values.tolist() for key, values in series.items()}
            for regime, series in acceptance_series(acceptance_stats).items()
        },
        'intervals': {'b': 'Supplied paper-level bootstrap 95% CI',
                      'c': 'Mean +/- 1.96 SEM; no new resampling'},
    }
    stem.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
