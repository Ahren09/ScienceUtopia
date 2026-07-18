"""Paper figure for the funding-cutoff study ("Does Early Funding Success
Manifest Matthew Effects?", sec:results-matthew).

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

Panels (1x2, double-column)
---------------------------
  (a) No jump at the funding cutoff. Mean funding received in years t+1..t+3
      per half-integer rank distance X (|X| <= 5 shown, the funded side ends
      at +4.5 because panels fund at most five). Per-side triangular-kernel
      local-linear fits with 95% bands are drawn over the preregistered
      h = 2 window (shaded). The fits omit the competition fixed effects used
      for tau (the level of a one-sided fit is not identified under FE); the
      script asserts that their gap at X = 0 equals the no-FE estimator. The
      annotated tau = -1.50 units (95% CI [-14.5, +11.5]) is the preregistered
      FE + two-way-clustered estimate from rd_results.json.
  (b) No detectable downstream effect. The same h = 2 estimate for later
      funding (primary) and five secondary outcomes, standardized by the SD
      of the narrow-loser group inside the window (Glass delta). Open marker
      = 95% CI crosses zero (all six). Raw-unit estimates are annotated after
      each interval so they map onto the numbers in the text.

Key terms
---------
  tau              the RD estimate: the jump in the outcome at x = 0 from
                   triangular-kernel local-linear fits with competition FE
                   (computed by utopia.analysis.matthew_rd_analysis;
                   annotated on the figure, not drawn)
  bandwidth h      how many rank positions on each side of the cutoff enter
                   the local fit; h = 2 is the preregistered, LOCO-CV-selected
                   main bandwidth (two rank positions per side)
  binned means     mean outcome per discrete x value (ranks are half-integer
                   offsets), error bars +-1.96 SE within the bin

Data provenance (read-only inputs)
----------------------------------
  /workingdir/yjin328/ScienceUtopia/data/matthew_rd/applications.parquet
      -- application-level dataset built by utopia.analysis.matthew_rd_analysis
         (data/ inside the code repo is a symlink, hence the absolute default)
  outputs/docs/matthew_rd_final/rd_results.json
      -- primary and secondary estimates from the same module
Binned means and the one-sided fit bands are imported from
utopia.analysis.matthew_rd_analysis (single source of truth with the
quick-look figures).

中文说明(这张图在讲什么)
    问题:在模拟的经费评审中,刚好跨过资助线(narrow winner)相比刚好没过线
    (narrow loser)的申请人,之后是否累积更多优势(马太效应)?
    为什么没有放大效应(四条通道均未激活)只在正文里说明,不再画成面板。
    · 面板 a 标准 RD 图:横轴是相对资助线的名次 X(虚线即资助线),纵轴是之后三年获得的经费,
      两侧各有局部线性拟合与 95% 置信带。资助线处没有跳跃。
    · 面板 b 森林图:同一估计量应用于六个下游结果(标准化为 SD 单位),
      全部置信区间跨零;区间右侧标注原始单位的估计值。

Run:
    python -m utopia.visual.plot_matthew_rd
    (writes the PDF straight into the paper's figures/ directory and a PNG
    preview under outputs/visual/matthew_rd_final/)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba

from utopia.analysis.matthew_rd_analysis import (_binned_means,
                                                 local_linear_fit_band,
                                                 local_linear_rd)
from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

DEFAULT_APPLICATIONS = ('/workingdir/yjin328/ScienceUtopia/data/matthew_rd/'
                        'applications.parquet')
DEFAULT_RESULTS = 'outputs/docs/matthew_rd_final/rd_results.json'
DEFAULT_OUT = ('/nethome/yjin328/Workspace/Paper/Paper_ScienceUtopia/figures/'
               'fig_matthew_rd.pdf')
DEFAULT_PREVIEW_DIR = 'outputs/visual/matthew_rd_final'

Y = 'cumulative_subsequent_funding_3y'
X_MAX = 5           # bins shown per side; the funded side ends at +4.5
BLUE = '#2a78d6'    # funded side and forest markers (repo series accent)
UNFUNDED = MUTED    # unfunded side: neutral grey, a null needs no red/green
# Type sizes one step above nature_style(), since the figure is scaled down to
# the paper's text width.
LABEL_FS, TICK_FS, NOTE_FS, LETTER_FS = 8.5, 7.5, 7, 10

# Forest rows: (rd_results key, parquet column, row label, raw-unit label).
OUTCOMES = [
    ('primary', Y, 'Later funding', lambda t: f'{t:+.1f} units'),
    ('pubs_written_3y', 'pubs_written_3y', 'Papers written',
     lambda t: f'{t:+.2f} papers'),
    ('pubs_accepted_3y', 'pubs_accepted_3y', 'Papers accepted',
     lambda t: f'{t:+.2f} papers'),
    ('citations_age2_matched', 'citations_age2_matched', 'Citations',
     lambda t: f'{t:+.1f} cites'),
    ('any_award_3y', 'any_award_3y', 'Any later award',
     lambda t: f'{100 * t:+.0f} pp'),
    ('exploration_distance_3y', 'exploration_distance_3y', 'Exploration',
     lambda t: f'{t:+.2f}'),
]


def rd_panel(ax, df: pd.DataFrame, primary: dict):
    """Binned means, per-side local-linear fits with 95% bands, cutoff line.

    面板 a:两侧各画分箱均值 + 局部线性拟合 + 置信带;拟合只画在预注册的
    h = 2 窗口内(灰色底纹),资助线处没有跳跃。
    """
    h = primary['h']
    ax.axvspan(-h, h, color=GRID, alpha=0.45, linewidth=0, zorder=0)
    fits = {}
    for side, color in ((-1, UNFUNDED), (1, BLUE)):
        sub = df[np.sign(df['x']) == side]
        xs, ms, ses = _binned_means(sub, Y, max_x=X_MAX)
        ax.errorbar(xs, ms, yerr=1.96 * ses, fmt='o', ms=3.2, color=color,
                    ecolor=to_rgba(color, 0.5), elinewidth=0.6, capsize=1.2,
                    capthick=0.6, markeredgecolor='white',
                    markeredgewidth=0.4, zorder=3)
        grid, yhat, lo, hi = local_linear_fit_band(df, Y, h, side)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15, linewidth=0,
                        zorder=1)
        ax.plot(grid, yhat, color=color, linewidth=1.2,
                solid_capstyle='butt', zorder=2)
        fits[side] = yhat
    # The drawn gap is exactly the no-FE local-linear estimator: assert it so
    # the picture and the analysis module cannot drift apart.
    gap = fits[1][0] - fits[-1][-1]
    tau_nofe = local_linear_rd(df, Y, h=h, fe=False)['tau']
    assert abs(gap - tau_nofe) < 1e-6, (gap, tau_nofe)

    # Cutoff line runs from above the two-line estimate block at the bottom to
    # just below its "Funding cutoff" label at the top.
    ax.axvline(0, ymin=0.21, ymax=0.9, color=INK, linewidth=0.8,
               linestyle=(0, (3, 2)), zorder=2)
    ax.set_xlim(-X_MAX - 0.4, X_MAX + 0.4)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xticks(range(-X_MAX, X_MAX + 1), minor=True)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel('Rank relative to funding cutoff ($X$)')
    ax.set_ylabel('Funding in next 3 years (units)')
    daf = ('data', 'axes fraction')
    ax.annotate('Funding cutoff', xy=(0, 1.0), xycoords=daf, ha='center',
                va='top', fontsize=NOTE_FS, color=INK)
    ax.annotate('Not funded', xy=(0, 0.88), xycoords=daf, xytext=(-4, 0),
                textcoords='offset points', ha='right', va='top',
                fontsize=NOTE_FS, color=UNFUNDED)
    ax.annotate('Funded', xy=(0, 0.88), xycoords=daf, xytext=(4, 0),
                textcoords='offset points', ha='left', va='top',
                fontsize=NOTE_FS, color=BLUE)
    ax.annotate(f"$\\tau$ = {primary['tau']:.2f}, 95% CI "
                f"[{primary['ci_lo']:.1f}, {primary['ci_hi']:.1f}]",
                xy=(0, 0.11), xycoords=daf, ha='center', va='bottom',
                fontsize=NOTE_FS, color=MUTED)
    ax.annotate(f"fit window $|X| \\leq {h}$, N = {primary['n']}",
                xy=(0, 0.02), xycoords=daf, ha='center', va='bottom',
                fontsize=NOTE_FS, color=MUTED)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def standardized_rows(df: pd.DataFrame, results: dict) -> list:
    """Glass-delta rows: tau and CI divided by the narrow-loser SD in-window."""
    h = results['primary']['h']
    ctrl = df[(df['x'] < 0) & (df['x'] >= -h)]
    rows = []
    for key, col, label, fmt in OUTCOMES:
        r = results['primary'] if key == 'primary' else results['secondary'][key]
        sd = float(ctrl[col].std())
        rows.append({'label': label, 'tau': r['tau'], 'sd': sd,
                     'est': r['tau'] / sd, 'lo': r['ci_lo'] / sd,
                     'hi': r['ci_hi'] / sd, 'raw': fmt(r['tau']),
                     'p_bh': r.get('p_bh', np.nan)})
    return rows


def outcome_forest_panel(ax, rows: list):
    """Standardized RD estimates per outcome; open marker = CI crosses zero.

    面板 b:六个结果的标准化效应森林图,空心点表示 95% 区间跨零,
    区间右侧标注原始单位的估计值。
    """
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    for i, (y, r) in enumerate(zip(ys, rows)):
        resolved = r['lo'] > 0 or r['hi'] < 0
        ax.errorbar(r['est'], y, xerr=[[r['est'] - r['lo']], [r['hi'] - r['est']]],
                    fmt='o', color=BLUE, ecolor=BLUE, elinewidth=1.2,
                    capsize=2, capthick=1.2, markersize=4.6 if i == 0 else 4,
                    markerfacecolor=BLUE if resolved else 'white',
                    markeredgecolor=BLUE, markeredgewidth=0.9)
        ax.annotate(r['raw'], xy=(r['hi'], y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=NOTE_FS, color=MUTED)
    # Primary endpoint above the rule, the secondary family below it.
    ax.axhline(ys[0] - 0.5, color=MUTED, linewidth=0.5, linestyle=(0, (1, 2)))
    ax.set_yticks(ys, [r['label'] for r in rows])
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-0.8, 1.2)  # room for the raw-unit labels past the CI caps
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_xlabel('Effect of narrowly winning (SD units)')
    ax.grid(axis='x', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def panel_letters(fig, axes):
    """Bold a/b above each panel's top-left corner (panels carry no titles).

    Uses fig.text (not ax.text) so the letters stay out of ax.texts and the
    assert_texts_inside overflow check. The x offset clears each panel's
    y-tick labels (panel b's outcome names are wide).
    """
    for letter, ax, dx in zip('ab', axes, (-0.17, -0.33)):
        fig.text(dx, 1.04, letter, transform=ax.transAxes,
                 fontsize=LETTER_FS, fontweight='bold', ha='left', va='baseline')


def assert_texts_inside(fig, ax, tol_pt: float = 1.0):
    """Fail loudly if any in-axes text spills past the axes box.

    Guards against silent overflow when a wider fallback font is substituted
    for Arial/Helvetica (DejaVu Sans is ~15% wider).
    """
    ren = fig.canvas.get_renderer()
    box = ax.get_window_extent(ren)
    tol = tol_pt * fig.dpi / 72
    for t in ax.texts:
        e = t.get_window_extent(ren)
        inside = (e.x0 >= box.x0 - tol and e.x1 <= box.x1 + tol
                  and e.y0 >= box.y0 - tol and e.y1 <= box.y1 + tol)
        assert inside, (f'text overflows panel: {t.get_text()!r} '
                        f'extent=({e.x0:.0f},{e.x1:.0f}) axes=({box.x0:.0f},{box.x1:.0f})')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--applications', default=DEFAULT_APPLICATIONS)
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='PDF path, written straight into the paper repo')
    ap.add_argument('--preview_dir', default=DEFAULT_PREVIEW_DIR,
                    help='directory for the PNG preview')
    args = ap.parse_args()

    df = pd.read_parquet(args.applications)
    results = json.load(open(args.results))
    primary = results['primary']

    nature_style()
    plt.rcParams.update({'axes.labelsize': LABEL_FS, 'xtick.labelsize': TICK_FS,
                         'ytick.labelsize': TICK_FS})
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.9, 2.0), gridspec_kw={'width_ratios': [1.2, 1.0]})
    rd_panel(ax_a, df, primary)
    rows = standardized_rows(df, results)
    outcome_forest_panel(ax_b, rows)
    sns.despine(fig)
    fig.tight_layout(w_pad=1.6)
    panel_letters(fig, (ax_a, ax_b))
    for ax in (ax_a, ax_b):
        assert_texts_inside(fig, ax)

    # Numbers the figure encodes, for checking against the prose.
    near = df[df['x'].abs() <= primary['h']].groupby('funded')[Y].mean()
    print(f"primary: tau = {primary['tau']:.2f} "
          f"[{primary['ci_lo']:.1f}, {primary['ci_hi']:.1f}], p = {primary['p']:.2f}, "
          f"n = {primary['n']}, h = {primary['h']}")
    print(f"near-cutoff means: funded {near[1]:.2f} vs. unfunded {near[0]:.2f}")
    for r in rows:
        print(f"  {r['label']:<16} tau = {r['tau']:+.3f} ({r['raw']}), "
              f"sd = {r['sd']:.3f}, std = {r['est']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}], "
              f"p_bh = {r['p_bh']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f'✅ Saved figure to: {out}')
    preview = Path(args.preview_dir)
    preview.mkdir(parents=True, exist_ok=True)
    png = preview / 'fig_matthew_rd.png'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    print(f'✅ Saved preview to: {png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
