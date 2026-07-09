"""Paper figure for the resubmission-cascade study (fixed vs declining acceptance).

Reads the two resubmission_stats.json files produced by
utopia.analysis.resubmission_stats for the paired worlds and renders one
six-panel, Nature-style figure:
  (a) submissions per year          (b) mean reviews per active reviewer
  (c) mean review score (1-5)       (d) first-attempt acceptance rate
  (e) cascade depth (mean, 95% CI)  (f) low-volume venue share, year 1 -> 10

Run:
    python -m utopia.visual.plot_resubmission_cascades
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

DEFAULT_FIXED = ('outputs/docs/resubmission_stats_default_Qwen3-32B_fixed30_paired/'
                 'resubmission_stats.json')
DEFAULT_DECLINING = ('outputs/docs/resubmission_stats_acl-acceptance-rates_Qwen3-32B/'
                     'resubmission_stats.json')
DEFAULT_OUT_DIR = 'outputs/visual/resubmission_cascades_fixed30_vs_acl'

# Two-regime palette validated for CVD separation and lightness band (dataviz
# skill, slots 1-2); the aqua is sub-3:1 on white, so every line carries a
# direct end label as the relief channel.
REGIMES = ['Fixed', 'Declining']
PALETTE = {'Fixed': '#2a78d6', 'Declining': '#1baf7a'}


def yearly_frame(stats: dict, key: str, value_name: str) -> pd.DataFrame:
    """Long-form frame of a per-year dict for both regimes."""
    rows = []
    for regime, s in stats.items():
        series = s[key]
        for year, v in series.items():
            v = v['rate_pct'] if isinstance(v, dict) else v  # first-attempt entries
            rows.append({'Year': int(year), value_name: v, 'Regime': regime})
    return pd.DataFrame(rows)


def line_panel(ax, df: pd.DataFrame, value: str, ylabel: str, title: str,
               end_fmt: str = '{:.0f}', ylim: tuple = None):
    sns.lineplot(data=df, x='Year', y=value, hue='Regime', hue_order=REGIMES,
                 palette=PALETTE, ax=ax, linewidth=1.4, marker='o',
                 markersize=3.5, markeredgecolor='white', markeredgewidth=0.6,
                 legend=False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    last = df[df['Year'] == df['Year'].max()]
    ends = {r: last.loc[last['Regime'] == r, value].iloc[0] for r in REGIMES}
    # Dodge end labels vertically when the lines converge at the right edge;
    # judge convergence against the drawn axis range, not the data range.
    axis_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    dodge = abs(ends[REGIMES[0]] - ends[REGIMES[1]]) < 0.07 * axis_span
    for regime, y in ends.items():
        dy = 0
        if dodge:
            is_upper = (y > min(ends.values())) or (
                ends[REGIMES[0]] == ends[REGIMES[1]] and regime == REGIMES[0])
            dy = 4 if is_upper else -4
        ax.annotate(end_fmt.format(y), xy=(df['Year'].max(), y),
                    xytext=(3, dy), textcoords='offset points',
                    va='center', fontsize=6, color=INK)
    ax.set_xticks(range(2, 11, 2))
    ax.set_xlabel('Simulation year')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.margins(x=0.04)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def cascade_panel(ax, stats: dict):
    """Point estimate with bootstrap 95% CI; recycle rate annotated per regime."""
    xs = range(len(REGIMES))
    bounds = []
    for x, regime in zip(xs, REGIMES):
        s = stats[regime]
        mean, (lo, hi) = s['cascade_depth_mean'], s['cascade_depth_ci95']
        bounds += [lo, hi]
        ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt='none',
                    ecolor=PALETTE[regime], elinewidth=1.2, capsize=3, capthick=1.2)
        sns.scatterplot(x=[x], y=[mean], ax=ax, color=PALETTE[regime], s=28,
                        edgecolor='white', linewidth=0.6, zorder=3, legend=False)
        # Stack both labels above the upper CI cap so nothing collides with
        # the axis or tick labels below.
        ax.annotate(f'{mean:.2f}', xy=(x, hi), xytext=(0, 13),
                    textcoords='offset points', ha='center', fontsize=6, color=INK)
        ax.annotate(f"{s['recycle_rate_pct']:.0f}% resubmitted",
                    xy=(x, hi), xytext=(0, 4), textcoords='offset points',
                    ha='center', fontsize=5.5, color=MUTED)
    span = max(bounds) - min(bounds)
    ax.set_ylim(min(bounds) - 0.10 * span, max(bounds) + 0.30 * span)
    ax.set_xticks(list(xs), REGIMES)
    ax.set_xlim(-0.6, len(REGIMES) - 0.4)
    ax.set_ylabel('Venue attempts per rejected paper')
    ax.set_title('Cascade depth (mean, 95% CI)', pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def venue_share_panel(ax, stats: dict):
    """Slope chart: low-volume venue share of all submissions, year 1 -> 10."""
    ends = {}
    for regime in REGIMES:
        s, ny = stats[regime], stats[regime]['num_years']
        low = s['venue_volume_split']['low']
        y0, y1 = low['share_year1_pct'], low[f'share_year{ny}_pct']
        ends[regime] = y1
        df = pd.DataFrame({'endpoint': [0, 1], 'share': [y0, y1]})
        sns.lineplot(data=df, x='endpoint', y='share', ax=ax,
                     color=PALETTE[regime], linewidth=1.4, marker='o',
                     markersize=4.5, markeredgecolor='white',
                     markeredgewidth=0.6, legend=False)
    for regime, y1 in ends.items():
        dy = 4 if y1 >= max(ends.values()) else -4  # dodge converging labels
        ax.annotate(f'{y1:.1f}%', xy=(1, y1), xytext=(4, dy),
                    textcoords='offset points', va='center', fontsize=6, color=INK)
    y0 = stats[REGIMES[0]]['venue_volume_split']['low']['share_year1_pct']
    ax.annotate(f'{y0:.1f}%', xy=(0, y0), xytext=(-4, 0),
                textcoords='offset points', va='center', ha='right',
                fontsize=6, color=INK)
    ax.set_xticks([0, 1], ['Year 1', 'Year 10'])
    ax.set_xlim(-0.35, 1.35)
    ax.set_xlabel('')
    ax.set_ylabel('Share of all submissions (%)')
    ax.set_title('Low-volume venue share', pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixed_stats', default=DEFAULT_FIXED)
    ap.add_argument('--declining_stats', default=DEFAULT_DECLINING)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    # Internal keys must stay aligned with REGIMES/PALETTE; the richer
    # display text lives only in the legend labels below.
    stats = {'Fixed': json.load(open(args.fixed_stats)),
             'Declining': json.load(open(args.declining_stats))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Double-column Nature width (180 mm = 7.09 in), 2x3 panels.
    fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.3))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    line_panel(ax_a, yearly_frame(stats, 'submissions_per_year', 'v'),
               'v', 'Submissions (all venue attempts)', 'Submission inflation')
    line_panel(ax_b, yearly_frame(stats, 'mean_reviews_per_reviewer_year', 'v'),
               'v', 'Reviews per active reviewer', 'Reviewer burden', '{:.1f}')
    line_panel(ax_c, yearly_frame(stats, 'mean_review_score_by_year', 'v'),
               'v', 'Mean review score (1–5 scale)', 'Score stability', '{:.2f}',
               ylim=(1, 5))  # full review scale: flatness shown without axis zoom
    line_panel(ax_d, yearly_frame(stats, 'first_attempt_acceptance_by_year', 'v'),
               'v', 'First-attempt acceptance (%)', 'First-attempt acceptance',
               '{:.0f}%')
    cascade_panel(ax_e, stats)
    venue_share_panel(ax_f, stats)

    for letter, ax in zip('abcdef', axes.flat):
        ax.text(-0.24, 1.08, letter, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', va='bottom')
    sns.despine(fig)

    legend_labels = {'Fixed': 'Fixed 30%', 'Declining': 'Declining (ACL)'}
    handles = [mpl.lines.Line2D([], [], color=PALETTE[r], linewidth=1.4,
                                marker='o', markersize=3.5,
                                markeredgecolor='white', label=legend_labels[r])
               for r in REGIMES]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.045), columnspacing=1.6, handlelength=1.6)
    fig.tight_layout(w_pad=1.6, h_pad=1.8)

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_resubmission_cascades.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
