"""Paper figure for the funding-cutoff study ("Early Funding Cutoffs Do Not
Generate Matthew Amplification", sec:results-matthew).

Experiment
----------
A regression-discontinuity (RD) design on simulated grant panels: each panel
ranks its n applicants and funds the top k. The running variable is the rank
distance to the funding cutoff, x = (k - rank) + 0.5, so x > 0 means the
applicant narrowly WON and x < 0 narrowly LOST; treatment (funded) switches
deterministically at x = 0 (a sharp design). Comparing applicants just above
vs. just below the cutoff isolates the causal effect of winning an early
grant on later outcomes -- the "Matthew effect" hypothesis predicts a
positive jump. Data: three 500-agent 6-year worlds (seeds 3001-3003),
2,604 applications across 285 competitions.

Panels (1x2)
------------
  (a) Primary outcome: cumulative funding received in years t+1..t+3, as
      binned means (per half-integer rank distance, +-95% CI). No jump is
      visible at the cutoff: tau = -1.50 units (95% CI [-14.5, +11.5],
      p = 0.82) -- a null. The sharpness of the design (treatment is
      deterministic in rank; awards reproduced in all 763 panels) is
      reported in the text rather than plotted.
  (b) Robustness: the RD estimate tau with 95% CI across bandwidths
      h = 2 (preregistered main, LOCO-CV selected) through 8. Every CI
      straddles zero, so the null is not a bandwidth artifact.

Key terms
---------
  tau              the RD estimate: the jump in the outcome at x = 0 implied
                   by triangular-kernel local-linear fits on each side of
                   the cutoff (computed by utopia.analysis.matthew_rd_analysis;
                   annotated on the figure, not drawn)
  bandwidth h      how many rank positions on each side of the cutoff enter
                   the local fit; small h = less bias, fewer observations
  binned means     mean outcome per discrete x value (ranks are half-integer
                   offsets), error bars +-1.96 SE within the bin

Data provenance (read-only inputs)
----------------------------------
  data/matthew_rd/applications.parquet
      -- application-level dataset built by utopia.analysis.matthew_rd_analysis
  outputs/docs/matthew_rd_final/rd_results.json
      -- primary estimate and bandwidth_sensitivity from the same module
The binned-means and local-linear-fit computations are imported from
utopia.analysis.matthew_rd_analysis (single source of truth with the
quick-look figures).

Run:
    python -m utopia.visual.plot_matthew_rd
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.analysis.matthew_rd_analysis import _binned_means
from utopia.utils.visual_utils import GRID, MUTED, nature_style

DEFAULT_APPLICATIONS = 'data/matthew_rd/applications.parquet'
DEFAULT_RESULTS = 'outputs/docs/matthew_rd_final/rd_results.json'
DEFAULT_OUT_DIR = 'outputs/visual/matthew_rd_final'

BLUE = '#2a78d6'  # series accent, shared with the exploration figures


def rd_panel(ax, df: pd.DataFrame, y: str, ylabel: str, title: str,
             max_x: float = 8):
    """Binned means vs. running variable.

    Each point is the mean outcome at one discrete rank distance
    (half-integer x), error bar = +-1.96 x within-bin SE. The RD estimate
    tau (the implied jump at x = 0) is annotated by the caller from
    rd_results.json rather than drawn as fit lines.
    """
    xs, ms, ses = _binned_means(df, y, max_x=max_x)
    ax.errorbar(xs, ms, yerr=1.96 * ses, fmt='o', ms=3.2, color=BLUE,
                ecolor=BLUE, elinewidth=0.9, capsize=1.6, capthick=0.9,
                markeredgecolor='white', markeredgewidth=0.4)
    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    # Only the cutoff matters on this axis; distances are read qualitatively.
    # The cutoff is named at the TOP of the dashed line (above the won/lost
    # arrows), so the x axis carries no ticks at all.
    ax.set_xticks([])
    ax.annotate('funding line', xy=(0, 1.02), xycoords=('data', 'axes fraction'),
                ha='center', va='bottom', fontsize=6, color=MUTED)
    ax.set_xlabel('Applicant rank')
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=0)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def direction_notes(ax):
    """Won/lost labels flanking the cutoff line (mathtext arrows: Arial
    lacks the arrow glyphs)."""
    ax.annotate('$\\leftarrow$\nnarrowly\nlost', xy=(0, 0.985),
                xycoords=('data', 'axes fraction'), xytext=(-4, 0),
                textcoords='offset points', ha='right', va='top',
                fontsize=5.5, color=MUTED)
    ax.annotate('$\\rightarrow$\nnarrowly\nwon', xy=(0, 0.985),
                xycoords=('data', 'axes fraction'), xytext=(4, 0),
                textcoords='offset points', ha='left', va='top',
                fontsize=5.5, color=MUTED)


def bandwidth_forest_panel(ax, results: dict):
    """RD estimate tau with 95% CI per bandwidth, main bandwidth flagged.

    Follows the series-wide forest convention: open (white-filled) marker =
    95% CI crosses zero. Cluster-robust CIs come precomputed from
    rd_results.json (bandwidth_sensitivity); n annotated per row.
    """
    sens = results['robustness']['bandwidth_sensitivity']
    h_main = results['primary']['h']
    hs = sorted(sens, key=float)
    ys = range(len(hs) - 1, -1, -1)
    for y, h_key in zip(ys, hs):
        r = sens[h_key]
        resolved = r['ci_lo'] > 0 or r['ci_hi'] < 0
        ax.errorbar(r['tau'], y,
                    xerr=[[r['tau'] - r['ci_lo']], [r['ci_hi'] - r['tau']]],
                    fmt='o', color=BLUE, ecolor=BLUE, elinewidth=1.2,
                    capsize=2, capthick=1.2, markersize=4,
                    markerfacecolor=BLUE if resolved else 'white',
                    markeredgecolor=BLUE, markeredgewidth=0.8)
        ax.annotate(f"n = {r['n']}", xy=(r['ci_hi'], y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=5.5, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    labels = [f'h={h}' for h in hs]
    ax.set_yticks(list(ys), labels)
    ax.set_ylim(-0.6, len(hs) - 0.4)
    ax.set_xlabel('RD estimate $\\tau$ (funding units)')
    # ax.set_title('Null across bandwidths', pad=4)
    ax.grid(axis='x', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.margins(x=0.18)  # room for the n labels past the CI caps
    ax.annotate('open marker: 95% CI crosses zero', xy=(0.03, 0.03),
                xycoords='axes fraction', fontsize=5.5, color=MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--applications', default=DEFAULT_APPLICATIONS)
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = pd.read_parquet(args.applications)
    results = json.load(open(args.results))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Single-column width, stacked vertically: the RD scatter needs the
    # full width for its 21 bins; the 5-row forest sits below, shorter.
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.5, 3.5), height_ratios=[1.35, 1])

    r = results['primary']
    # max_x=16 doubles the plotted range vs. the quick-look default of 8
    # while keeping every bin at n >= 110 applications; beyond -16 the bins
    # thin out (n = 16-80) and their CIs would dominate the panel.
    rd_panel(ax_a, df, 'cumulative_subsequent_funding_3y',
             'Funding in next 3 years',
             '', max_x=16)
    direction_notes(ax_a)
    ax_a.annotate(f"$\\tau$ = {r['tau']:.2f}\n"
                  f"95% CI [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]\n"
                  f"p = {r['p']:.2f}",
                  xy=(0.03, 0.03), xycoords='axes fraction', va='bottom',
                  fontsize=5.5, color=MUTED)
    bandwidth_forest_panel(ax_b, results)

    for letter, ax in zip('ab', (ax_a, ax_b)):
        ax.text(-0.02, .96, letter, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', va='bottom')
    sns.despine(fig)
    fig.tight_layout(h_pad=1.0)

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_matthew_rd.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
