"""Paper figures for the output-doubling x budget factorial (Stage A) at 3x scale.

Experiment
    Stage A of the scale-expansion study crosses two binary factors in closed
    university-only worlds (Qwen3-32B, all researchers `balanced`, equal initial
    funding 100, 8 simulated years):
      S  papers per completed project   1 (single)  vs  2 (double; same costs)
      R  agency budget                  tracks applications  vs  fixed at
                                        20% of the initial population per year
    Pilot: 20 institutions x 5 = 100 researchers, seeds 1-3 (the inferential
    sample). Scale-up: 60 institutions x 5 = 300 researchers, seed 1, funding
    rankings in panels of 25 applications. Preregistered consistency rule: an
    effect is replicated if the scale-up contrast has the same sign and at least
    half the pilot magnitude (prereg_stage_a_scaleup_uni300_v1.json).

Figures
    fig_scale_main (1x2, 5.6 in wide; 300-researcher world)
      (a) Share of founders still active by year, one line per cell. Exits
        begin in year 5; doubling output and freezing the budget each lower
        year-8 survival, and the two effects add (0.65 -> 0.50 / 0.57 -> 0.36).
      (b) Concentration = exclusion.
        Awards per founder over 8 years as stacked shares, one bar per cell: the
        never-funded segment grows from 0.22 (single, tracks) to 0.46 (double,
        fixed) while the 1 / 2 / 3 / 4+ segments shrink together, so the rising
        all-founder Gini (0.49 -> 0.67) comes from exclusion, not from a more
        skewed split among the funded (Gini among funded 0.34-0.39).
    fig_scale_default_fixed (1x2; paper-baseline population 150 university +
        150 industry researchers, seed 42, 10 years, fixed-budget cells only,
        the tracking-budget cells being incomplete)
      (a) survival by sector: solid = university, dashed = industry. Industry
        researchers start with twice the endowment and earn pay-per-paper income,
        so their exits start ~4 years later; doubling output still lowers year-10
        survival in both sectors (university 0.45 -> 0.31, industry 0.87 -> 0.56).
      (b) awards per university founder (industry researchers do not apply to
        agencies): never funded 0.39 -> 0.45.
    fig_scale_replication  (appendix) the four preregistered contrasts on
        year-8 survival, pilot (grey, seed range) vs 300-world (blue). All pass
        the consistency rule.

Metrics
    Survival = is_active at the end-of-year checkpoint (the simulator flips
    is_active instead of deleting agents). Awards from
    funding_applications_year_<y>.jsonl (winners per applicant). Gini in [0, 1].

Data provenance (read-only inputs)
    outputs/docs/scale_expansion_analysis/per_run_endpoints.csv
        pilot, produced by utopia.analysis.scale_expansion (3 seeds x 4 cells)
    outputs/docs/scale_expansion_analysis_uni300/{per_run_endpoints,yearly_checks}.csv
        300-researcher scale-up (1 seed x 4 cells); X3_award_histogram holds the
        awards -> founders histogram used by fig_scale_main (b)
    outputs/docs/scale_expansion_analysis_default300_panel25_interim/{per_run_endpoints,yearly_checks}.csv
        paper-baseline population scale-up (seed 42, 10 years, panels of 25);
        active_researchers_{university,industry} give the per-sector curves

中文说明
    产出翻倍(S) x 预算固定(R) 的 2x2,100 人试点(3 种子)与 300 人放大版(1 种子)。
    主图 fig_scale_main 只画 300 人世界:(a) 各格逐年存活率;(b)
    每位创始人 8 年获资助次数的堆叠比例,"从未获资助"一段从 0.22 涨到 0.46,而 1/2/3/4+
    各段一起缩小:稀缺被"排除更多人"吸收,而非"头部拿更多"。
    fig_scale_replication(附录): 四个预注册对照在第 8 年存活率上的效应,全部满足
    "同号且不小于试点一半"的复现规则。

Run:
    python -m utopia.visual.plot_scale_replication
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

DEFAULT_PILOT_DIR = 'outputs/docs/scale_expansion_analysis'
DEFAULT_SCALE_DIR = 'outputs/docs/scale_expansion_analysis_uni300'
DEFAULT_OUT_DIR = 'outputs/visual/scale_expansion_analysis_uni300'
DEFAULT_POP_DIR = 'outputs/docs/scale_expansion_analysis_default300_panel25_interim'
DEFAULT_POP_OUT_DIR = 'outputs/visual/scale_expansion_analysis_default300'
FIXED_CELLS = ['S0R1', 'S1R1']

CELLS = ['S0R0', 'S0R1', 'S1R0', 'S1R1']
CELL_LABELS = {'S0R0': 'single paper, budget tracks',
               'S0R1': 'single paper, budget fixed',
               'S1R0': 'double paper, budget tracks',
               'S1R1': 'double paper, budget fixed'}
# Hue encodes output (blue = single, coral = double); lightness encodes the budget
# regime (light = tracks applications, dark = fixed). Same blue/coral family as the
# other paper figures.
PALETTE = {'S0R0': '#2a78d6', 'S0R1': '#1E466E', 'S1R0': '#E76254', 'S1R1': '#8b2c1f'}
CONTRASTS = [('S1R0', 'S0R0', '2× output\n(tracks)'),
             ('S1R1', 'S0R1', '2× output\n(fixed)'),
             ('S0R1', 'S0R0', 'fixed budget\n(1 paper)'),
             ('S1R1', 'S1R0', 'fixed budget\n(2 papers)')]
BLUE = '#2a78d6'
CELL_TICKS = {'S0R0': 'single\ntracks', 'S0R1': 'single\nfixed',
              'S1R0': 'double\ntracks', 'S1R1': 'double\nfixed'}
# Awards-per-founder bins for the stacked bars: never funded in grey, 1..4+ in blues.
AWARD_BINS = [('never funded', '#d4d4d4'), ('1 award', '#c6dbef'), ('2 awards', '#8fbbe0'),
              ('3 awards', '#4a90d9'), ('4+ awards', '#1E466E')]


# Sector -> (yearly_checks column, line style, label); None = whole population.
SECTORS = [('university', 'active_researchers_university', '-', 'university'),
           ('industry', 'active_researchers_industry', (0, (2.2, 1.4)), 'industry')]


def survival_panel(ax, scale_yearly: pd.DataFrame, cells=CELLS, by_sector=False):
    """Survival curves per cell (colour); optionally split by sector (line style)."""
    series = SECTORS if by_sector else [(None, 'active_researchers', '-', None)]
    last_year = int(scale_yearly['year'].max())
    y_min = 1.0
    for cell in cells:
        s = scale_yearly[scale_yearly['cell'] == cell].sort_values('year')
        for _, col, style, _ in series:
            share = s[col] / s[col].max()
            y_min = min(y_min, share.min())
            ax.plot(s['year'], share, color=PALETTE[cell], lw=1.6, linestyle=style,
                    marker='o', ms=3.2, markeredgecolor='white', markeredgewidth=0.5)
            ax.annotate(f'{share.iloc[-1]:.2f}', xy=(s['year'].iloc[-1], share.iloc[-1]),
                        xytext=(4, 0), textcoords='offset points', va='center',
                        fontsize=6, color=PALETTE[cell])
    ax.set_xlim(1, last_year + 0.9)
    ax.set_ylim(min(0.3, np.floor((y_min - 0.04) * 10) / 10), 1.02)  # keep the lowest label clear
    ax.set_xticks(range(2, last_year + 1, 2))
    ax.set_xlabel('Simulation year')
    ax.set_ylabel('Share of founders active')
    handles = [plt.Line2D([], [], color=PALETTE[c], lw=1.6, marker='o', ms=3,
                          markeredgecolor='white', label=CELL_LABELS[c]) for c in cells]
    if by_sector:
        handles += [plt.Line2D([], [], color=MUTED, lw=1.6, linestyle=style, label=label)
                    for _, _, style, label in SECTORS]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=5.4,
              handlelength=1.8, borderaxespad=0.2, labelspacing=0.3)


def award_histogram(scale_runs: pd.DataFrame, cell: str) -> np.ndarray:
    """Awards per founder (one entry per founder) rebuilt from X3_award_histogram."""
    hist = json.loads(scale_runs.set_index('cell').loc[cell, 'X3_award_histogram'])
    return np.repeat([int(k) for k in hist], [int(v) for v in hist.values()])


def award_share_panel(ax, scale_runs: pd.DataFrame, cells=CELLS):
    """Stacked shares of (university) founders by number of awards, one bar per cell."""
    xs = np.arange(len(cells))
    for i, cell in enumerate(cells):
        v = award_histogram(scale_runs, cell)
        shares = [np.mean(v == 0), np.mean(v == 1), np.mean(v == 2), np.mean(v == 3), np.mean(v >= 4)]
        bottom = 0.0
        for share, (label, colour) in zip(shares, AWARD_BINS):
            ax.bar(xs[i], share, bottom=bottom, width=0.72, color=colour, edgecolor='white',
                   linewidth=0.5, label=label if i == 0 else None)
            bottom += share
        ax.annotate(f'{shares[0]:.2f}', xy=(xs[i], shares[0] / 2), ha='center', va='center',
                    fontsize=6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_TICKS[c] for c in cells], fontsize=6)
    for tick, cell in zip(ax.get_xticklabels(), cells):
        tick.set_color(PALETTE[cell])
    ax.set_xlim(-0.55, len(cells) - 0.45)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('Share of founders')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=5.8, handlelength=1.0, handleheight=1.0,
              labelspacing=0.35, borderaxespad=0)


def replication_panel(ax, pilot_runs: pd.DataFrame, scale_runs: pd.DataFrame):
    """Year-8 survival effect per preregistered contrast: pilot (grey, seed range) vs 300."""
    pil = pilot_runs.pivot_table(index='seed', columns='cell', values='P1_survival_final')
    sc = scale_runs.set_index('cell')['P1_survival_final']
    xs = np.arange(len(CONTRASTS))
    for x, (treat, ctrl, _) in zip(xs, CONTRASTS):
        eff = pil[treat] - pil[ctrl]                     # one value per pilot seed
        ax.plot([x - 0.16, x - 0.16], [eff.min(), eff.max()], color=MUTED, lw=1.0)
        sns.scatterplot(x=[x - 0.16], y=[eff.mean()], ax=ax, color=MUTED, s=26,
                        edgecolor='white', linewidth=0.5, zorder=3, legend=False)
        e300 = sc[treat] - sc[ctrl]
        sns.scatterplot(x=[x + 0.16], y=[e300], ax=ax, color=BLUE, s=30,
                        edgecolor='white', linewidth=0.5, zorder=3, legend=False)
        ax.annotate(f'{e300:+.2f}', xy=(x + 0.16, e300), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=5.8, color=BLUE)
    ax.axhline(0, color=MUTED, lw=0.6, linestyle=(0, (3, 2)))
    ax.set_xticks(xs)
    ax.set_xticklabels([c[2] for c in CONTRASTS], fontsize=5.6)
    ax.set_xlim(-0.55, len(CONTRASTS) - 0.25)
    ax.set_ylim(-0.30, 0.04)
    ax.set_ylabel('Survival difference (year 8)')
    handles = [plt.Line2D([], [], color=MUTED, marker='o', ms=4, lw=0, label='pilot (mean, seed range)'),
               plt.Line2D([], [], color=BLUE, marker='o', ms=4, lw=0, label='300 researchers')]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=5.2,
              borderaxespad=0.2, labelspacing=0.25)


def save(fig, out_dir: Path, stem: str):
    for ext in ('pdf', 'png'):
        path = out_dir / f'{stem}.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot_dir', default=DEFAULT_PILOT_DIR)
    ap.add_argument('--scale_dir', default=DEFAULT_SCALE_DIR)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--default_dir', default=DEFAULT_POP_DIR,
                    help='analysis dir of the paper-baseline (default population) scale-up')
    ap.add_argument('--default_out_dir', default=DEFAULT_POP_OUT_DIR)
    args = ap.parse_args()

    pilot_runs = pd.read_csv(Path(args.pilot_dir) / 'per_run_endpoints.csv')
    pilot_runs = pilot_runs[pilot_runs['cell'].isin(CELLS)]
    scale_runs = pd.read_csv(Path(args.scale_dir) / 'per_run_endpoints.csv')
    scale_yearly = pd.read_csv(Path(args.scale_dir) / 'yearly_checks.csv')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()

    # Main figure: (a) survival curves, (b) awards per founder.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.6, 2.1), dpi=300,
                                     gridspec_kw={'width_ratios': [1.35, 1]})
    survival_panel(ax_a, scale_yearly)
    award_share_panel(ax_b, scale_runs)
    ax_a.set_title('a  Researcher survival', pad=3)
    ax_b.set_title('b  Awards per founder', pad=3)
    for ax in (ax_a, ax_b):
        ax.grid(axis='y', linewidth=0.5, color=GRID)
        ax.set_axisbelow(True)
    sns.despine(fig)
    fig.tight_layout(w_pad=1.2)
    save(fig, out_dir, 'fig_scale_main')

    # Paper-baseline population (university + industry), fixed-budget cells only.
    pop_runs = pd.read_csv(Path(args.default_dir) / 'per_run_endpoints.csv')
    pop_yearly = pd.read_csv(Path(args.default_dir) / 'yearly_checks.csv')
    pop_cells = [c for c in FIXED_CELLS if c in set(pop_runs['cell'])]
    pop_out = Path(args.default_out_dir)
    pop_out.mkdir(parents=True, exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.6, 2.1), dpi=300,
                                     gridspec_kw={'width_ratios': [1.5, 1]})
    survival_panel(ax_a, pop_yearly[pop_yearly['cell'].isin(pop_cells)], cells=pop_cells, by_sector=True)
    award_share_panel(ax_b, pop_runs, cells=pop_cells)
    ax_a.set_title('a  Researcher survival by sector', pad=3)
    ax_b.set_title('b  Awards per university founder', pad=3)
    for ax in (ax_a, ax_b):
        ax.grid(axis='y', linewidth=0.5, color=GRID)
        ax.set_axisbelow(True)
    sns.despine(fig)
    fig.tight_layout(w_pad=1.2)
    save(fig, pop_out, 'fig_scale_default_fixed')

    # Appendix: preregistered contrasts, pilot vs 300.
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=300)
    replication_panel(ax, pilot_runs, scale_runs)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    sns.despine(fig)
    fig.tight_layout()
    save(fig, out_dir, 'fig_scale_replication')


if __name__ == '__main__':
    main()
