"""Paper figure: the cautious explorer's advantage over each pure strategy.

Companion to utopia.visual.plot_exploration_strategies (see its docstring for
the experiment setup). A single-axes forest plot over four outcomes; each
outcome row holds two vertically dodged estimates:
  * upper, explorer red:   cautious explorer - explorer -- how much moderate
    exploration beats unconstrained exploration.
  * lower, exploiter ochre: cautious explorer - exploiter -- whether moderate
    exploration also beats pure exploitation, or merely matches it.
Positive = the cautious explorer does better. Points wear the color of the
COMPARISON strategy, matching the palette of the companion figures.

How to read: dot = effect estimate, horizontal line = 95%
hierarchical-bootstrap CI (institution-block resampling from
primary_hypothesis_tests.csv). The x-axis is the standardized mean
difference (SMD = mean difference / pooled SD, dimension-free so the four
outcomes share one axis); the raw-unit effect is annotated at the CI's right
end. An open (white-filled) marker means the CI crosses zero -- the
direction is unresolved at n = 5 seeds; filled = resolved.

The CSV stores contrasts as explorer-cautious_explorer and
cautious_explorer-exploiter; the former is sign-flipped here (estimate and
CI bounds negated and swapped) so both series read cautious-minus-other.

Run:
    python -m utopia.visual.plot_exploration_contrasts
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, MUTED, nature_style
from utopia.visual.plot_exploration_strategies import (
    DEFAULT_ANALYSIS_DIR, DEFAULT_OUT_DIR, LABELS, PALETTE)

# Outcomes shown as forest rows: column in the hypothesis-tests CSV,
# display label, and raw-unit annotation format.
OUTCOMES = [
    ('acceptance_rate', 'Acceptance\nrate', lambda d: f'{d * 100:+.1f} pp'),
    ('ever_hit', 'High-impact\npaper prob.', lambda d: f'{d * 100:+.1f} pp'),
    ('log1p_citations', 'log(1+\ncitations)', lambda d: f'{d:+.3f}'),
    ('final_funding', 'Final\nfunding', lambda d: f'{d:+.1f} units'),
]
# (csv contrast name, flip sign so the effect reads cautious-minus-other,
#  comparison strategy, vertical dodge from the outcome's base row)
CONTRASTS = [
    ('explorer-cautious_explorer', True, 'explorer', +0.17),
    ('cautious_explorer-exploiter', False, 'exploiter', -0.17),
]


def forest_series(ax, tests: pd.DataFrame, contrast: str, flip: bool,
                  other: str, dodge: float, base_ys: np.ndarray):
    """One contrast series: cautious - <other> across the four outcomes.

    Drawn at the outcome's base row shifted by `dodge`, in the comparison
    strategy's palette color. CIs are rescaled from raw units to SMD via the
    constant smd/mean_diff ratio (the same pooled SD divides estimate and
    bounds); when `flip` is set, estimate and CI are negated (bounds swapped)
    to reverse the CSV's stored direction. A CI crossing zero is drawn as an
    open marker; the raw-unit effect is annotated past the CI's right end.
    """
    rows = tests[tests.contrast == contrast].set_index('outcome')
    color = PALETTE[other]
    for base_y, (outcome, label, fmt) in zip(base_ys, OUTCOMES):
        r = rows.loc[outcome]
        diff, smd = r.mean_diff, r.smd
        scale = smd / diff  # 1 / pooled SD, in raw-outcome units
        lo, hi = r.ci_lo * scale, r.ci_hi * scale
        if flip:
            diff, smd, lo, hi = -diff, -smd, -hi, -lo
        resolved = lo > 0 or hi < 0
        y = base_y + dodge
        ax.errorbar(smd, y, xerr=[[smd - lo], [hi - smd]], fmt='o',
                    color=color, ecolor=color, elinewidth=1.2, capsize=2,
                    capthick=1.2, markersize=4,
                    markerfacecolor=color if resolved else 'white',
                    markeredgecolor=color, markeredgewidth=0.8)
        ax.annotate(fmt(diff), xy=(hi, y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=5.5, color=MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis_dir', default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    tests = pd.read_csv(Path(args.analysis_dir) / 'primary_hypothesis_tests.csv')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Single-column Nature width (88 mm = 3.46 in), one shared axes.
    fig, ax = plt.subplots(figsize=(3.46, 2.5))
    base_ys = np.arange(len(OUTCOMES))[::-1].astype(float)
    for contrast, flip, other, dodge in CONTRASTS:
        forest_series(ax, tests, contrast, flip, other, dodge, base_ys)

    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_yticks(base_ys, [label for _, label, _ in OUTCOMES])
    ax.set_ylim(-0.7, len(OUTCOMES) - 0.3)
    # Meta-analysis convention: spell out the axis, put the direction as
    # "favors ..." labels flanking the zero line (arrows via mathtext --
    # Arial lacks the arrow glyphs).
    ax.set_xlabel('Standardized mean difference')
    ax.annotate('$\\leftarrow$ favors exploration\nor exploitation', xy=(0, 0.985),
                xycoords=('data', 'axes fraction'), xytext=(-4, 0),
                textcoords='offset points', ha='right', va='top',
                fontsize=5.5, color=MUTED)
    ax.annotate('favors cautious $\\rightarrow$', xy=(0, 0.985),
                xycoords=('data', 'axes fraction'), xytext=(4, 0),
                textcoords='offset points', ha='left', va='top',
                fontsize=5.5, color=MUTED)
    ax.set_title('Cautious-explorer advantage', pad=4)
    ax.grid(axis='x', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.margins(x=0.14)  # room for the raw-effect labels past the CI caps

    handles = [mpl.lines.Line2D([], [], color=PALETTE[s], linewidth=1.2,
                                marker='o', markersize=4,
                                markeredgecolor=PALETTE[s],
                                label=f'vs. {LABELS[s].lower()}')
               for _, _, s, _ in CONTRASTS]
    ax.legend(handles=handles, frameon=False, loc='lower right',
              handlelength=1.2, borderaxespad=0.2, fontsize=6)
    ax.annotate('open marker: 95% CI crosses zero', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=5.5, color=MUTED)
    sns.despine(fig)
    fig.tight_layout()

    for ext in {'pdf'}:
        path = out_dir / f'fig_cautious_explorer_advantage.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
