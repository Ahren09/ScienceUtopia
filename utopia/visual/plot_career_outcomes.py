"""Paper figure: career outcomes across research strategies (fig:career_outcomes).

Experiment
    Three E4a confirmatory exploration worlds (Qwen3-32B, 1000 institutions x
    5000 researchers x 10 years, seeds 1001-1003, neutral mechanisms).
    Researchers are permanently assigned one of three strategies (explorer /
    cautious explorer / exploiter) and are removed ("attrition") when their
    funding resources fall below the survival threshold.

Panels (1x2, one shared strategy legend on top)
    (a) Forest plot of the cautious explorer's edge over each pure strategy.
        Each outcome row holds two dodged estimates: cautious minus explorer
        (coral, upper) and cautious minus exploiter (amber, lower); positive
        favors the cautious explorer. x = standardized mean difference (SMD =
        mean difference / pooled SD, so the four outcomes share one axis);
        whiskers = 95% hierarchical-bootstrap CIs (institution-block
        resampling). The signed raw-unit effect is annotated past each CI
        (percentage points for rates, log points for citations, funding
        units for funding). Open marker = CI crosses zero.
    (b) Survival: share of researchers still active by simulated year and
        strategy (mean across seeds; translucent per-seed traces). Explorers
        survive best through year 7 (82.4% vs 75.8% for exploiters), then
        fall to 41.1% by year 10 vs ~53% for the others; the three values
        quoted in the paper are annotated.

Data provenance (read-only inputs)
    outputs/docs/explore_e4a_analysis/primary_hypothesis_tests.csv
        (contrast, outcome, mean_diff, smd, ci_lo, ci_hi). Contrasts are
        stored as explorer-cautious_explorer (sign-flipped here) and
        cautious_explorer-exploiter. Calibration CSVs name the funding outcome
        final_funding; it is normalized to cumulative_funding_received.
    outputs/docs/explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed100{1-3}/
        agent_year.parquet (agent_id, year, strategy, is_active)

Run
    python -m utopia.visual.plot_career_outcomes
    (writes to the paper repo's figures/fig_career_outcomes.pdf by default;
    override with --out)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, MUTED, nature_style
from utopia.visual.plot_exploration_strategies import (
    LABELS, PALETTE, STRATEGIES, load_run_tables)

DEFAULT_ANALYSIS_DIR = 'outputs/docs/explore_e4a_analysis'
DEFAULT_RUNS_GLOB = ('outputs/docs/'
                     'explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed*')
DEFAULT_OUT = ('/nethome/yjin328/Workspace/Paper/Paper_ScienceUtopia/figures/'
               'fig_career_outcomes.pdf')

# Forest rows: hypothesis-test outcome column, display label, raw-effect format.
OUTCOMES = [
    ('acceptance_rate', 'Acceptance\nrate', lambda d: f'{d * 100:+.1f}'),
    ('ever_hit', 'Citation-hit\nprob.', lambda d: f'{d * 100:+.1f}'),
    ('log1p_citations', 'log \ncitations', lambda d: f'{d:+.3f}'),
    ('cumulative_funding_received', 'Cumulative\nfunding', lambda d: f'{d:+.1f}'),
]
# (csv contrast, flip sign so it reads cautious-minus-other, comparison
#  strategy, vertical dodge from the outcome's base row)
CONTRASTS = [
    ('explorer-cautious_explorer', True, 'explorer', +0.2),
    ('cautious_explorer-exploiter', False, 'exploiter', -0.2),
]
# Survival values quoted in the paper: (strategy, year, text offset pt, ha, va).
ANNOTATE = [('explorer', 7, (3, 2), 'left', 'bottom'),
            ('exploiter', 7, (-3, -3), 'right', 'top'),
            ('explorer', 10, (-4, 0), 'right', 'center')]


def forest_panel(ax, tests: pd.DataFrame):
    """(a) Cautious-minus-other SMDs with 95% CIs for both comparisons."""
    base_ys = np.arange(len(OUTCOMES))[::-1].astype(float)
    any_unresolved = False
    for contrast, flip, other, dodge in CONTRASTS:
        rows = tests[tests.contrast == contrast].set_index('outcome')
        color = PALETTE[other]
        for base_y, (outcome, _, fmt) in zip(base_ys, OUTCOMES):
            r = rows.loc[outcome]
            diff, smd = r.mean_diff, r.smd
            scale = smd / diff  # 1 / pooled SD: rescales raw-unit CI bounds to SMD
            lo, hi = r.ci_lo * scale, r.ci_hi * scale
            if flip:
                diff, smd, lo, hi = -diff, -smd, -hi, -lo
            resolved = lo > 0 or hi < 0
            any_unresolved |= not resolved
            y = base_y + dodge
            ax.errorbar(smd, y, xerr=[[smd - lo], [hi - smd]], fmt='o',
                        color=color, elinewidth=1.0, capsize=1.5, capthick=1.0,
                        markersize=3.2, markeredgewidth=0.8,
                        markerfacecolor=color if resolved else 'white')
            ax.annotate(fmt(diff), xy=(hi, y), xytext=(3, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=5, color=MUTED)

    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_yticks(base_ys, [label for _, label, _ in OUTCOMES])
    ax.set_ylim(-0.5, len(OUTCOMES) + 0.4)  # headroom for the "favors" labels
    ax.set_xlabel('Std. Mean difference')
    # Direction labels flank the zero line (mathtext arrows: Arial lacks them).
    for text, offset, ha in [('$\\leftarrow$ favors exploration\nor exploitation', -4, 'right'),
                             ('favors cautious$\\rightarrow$\nexploration', 4, 'left')]:
        # Semi-transparent box dims the axis and grid lines behind the text.
        ax.annotate(text, xy=(0, 0.985), xycoords=('data', 'axes fraction'),
                    xytext=(offset, 0), textcoords='offset points', ha=ha,
                    va='top', fontsize=5, color=MUTED,
                    bbox={'boxstyle': 'square,pad=0.2', 'facecolor': 'white',
                          'edgecolor': 'none', 'alpha': 0.7})
    if any_unresolved:
        ax.annotate('open marker: 95% CI crosses zero', xy=(0.02, 0.02),
                    xycoords='axes fraction', fontsize=5.5, color=MUTED)
    ax.grid(axis='x', linewidth=0.5, color=GRID)
    ax.margins(x=0.16)  # room for the raw-effect labels past the CI caps
    ax.set_title('a Advantage of\nCautious Explorers', pad=2)


def survival_panel(ax, agent_year: pd.DataFrame):
    """(b) Percent of researchers still active per year, by strategy."""
    surv = (agent_year.groupby(['strategy', 'seed', 'year'])['is_active'].mean()
            .mul(100).rename('pct').reset_index())
    for (s, _), grp in surv.groupby(['strategy', 'seed']):
        ax.plot(grp['year'], grp['pct'], color=PALETTE[s], alpha=0.18, lw=0.5)
    mean = surv.groupby(['strategy', 'year'])['pct'].mean()
    for s in STRATEGIES:
        ax.plot(mean.loc[s].index, mean.loc[s].to_numpy(), color=PALETTE[s], lw=1.3)
    for s, year, offset, ha, va in ANNOTATE:
        v = float(mean.loc[(s, year)])
        ax.plot(year, v, 'o', ms=2.5, color=PALETTE[s], zorder=3)
        ax.annotate(f'{v:.1f}%', xy=(year, v), xytext=offset,
                    textcoords='offset points', ha=ha, va=va,
                    fontsize=5, color=PALETTE[s])
        print(f'{s} year {year}: {v:.1f}% active')
    ax.set_xlabel('Simulation year')
    ax.set_ylabel('Active researchers (%)', labelpad=-1)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylim(35, 103)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_title('b  Researcher\nSurvival', pad=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis_dir', default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument('--runs_glob', default=DEFAULT_RUNS_GLOB)
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    tests = pd.read_csv(Path(args.analysis_dir) / 'primary_hypothesis_tests.csv')
    tests['outcome'] = tests['outcome'].replace(
        {'final_funding': 'cumulative_funding_received'})
    agent_year = load_run_tables(args.runs_glob, 'agent_year')

    nature_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.4, 1.95), dpi=300,
                                     gridspec_kw={'width_ratios': [1.3, 1.1]})
    forest_panel(ax_a, tests)
    survival_panel(ax_b, agent_year)
    for ax in (ax_a, ax_b):
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    # One strategy legend for both panels: in (a) a color marks the strategy
    # the cautious explorer is compared against.
    handles = [plt.Line2D([], [], color=PALETTE[s], lw=1.3, marker='o',
                          markersize=3, label=LABELS[s]) for s in STRATEGIES]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               fontsize=5.5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
               handlelength=1.4, handletextpad=0.4)
    fig.tight_layout(w_pad=-0.2, 
                     rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
