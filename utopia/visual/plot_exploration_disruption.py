"""Paper figures: citation-network disruption by topical departure and strategy.

Question
    Do papers that move further from their author's recent work also redirect
    later research (disruption), or do they reinforce existing knowledge
    (consolidation)? And does the answer depend on who makes the move?

Experiment
    The three E4a confirmatory worlds (Qwen3-32B, 1,000 institutions x 5
    researchers, 10 years, seeds 1001-1003, fixed 1/3/1 strategy mix), the same
    worlds as panel (c) of fig_topic_diversity_selection_gate. Another run
    family can be selected with --run_prefix.

Main-text panel (d) of fig_topic_diversity_selection_gate (draw(), called from
that module)
    x: topical departure of the paper, in the same five equal-width bins as the
       selection-gate figure (ticks at the bin edges). Topical departure = 1 - cosine(e_paper, c_author),
       the embedding distance between the paper's abstract and the author's
       career centroid (the novelty_score column, utopia/metrics/embedding_tracker.py).
    y: share of disruptive papers, i.e. papers with CD > 0 at paper age three.
    One line per strategy with Wilson 95% CIs over papers pooled across the
    worlds. Marker area grows linearly with the number of papers in the bin
    (size key at lower right), so the panel also shows where each strategy's
    papers sit on the departure axis.
    Every non-empty bin is drawn, however few papers it holds. The thick gray
    line pools all strategies.

Appendix figure (two panels, fig_disruption_uptake)
    Explains why the disruptive share differs: CD > 0 mostly means the paper
    was taken up at all.
    (a) CD-sign composition of all papers per departure bin (consolidating
        CD < 0, CD = 0, disruptive CD > 0). CD = 0 papers are nearly all papers
        no later paper has cited yet.
    (b) Mean CD among cited papers (at least one later paper cites the focal
        paper) per strategy and bin, with bootstrap 95% CIs (seed 0).

Metric
    CD index (Funk & Owen-Smith 2017, utopia/metrics/cd_index.py) observed
    three years after publication: (n_focal - n_both) / (n_focal + n_both +
    n_prior), where n_focal later papers cite the focal paper but none of its
    references, n_both cite it and its references, and n_prior cite only its
    references. CD is undefined when the paper has no references or too few
    later papers cite it or its references. Those papers are excluded, as are
    the few departures above the last bin edge (1.01).

Data provenance (read-only inputs)
    outputs/docs/<run_prefix>_seed<S>/paper_age.parquet
        (strategy, age, cd, cd_n_disruptive, cd_n_consolidating, novelty_score),
        written by utopia/analysis/exploration_cross_seed.py. The table holds
        accepted papers only.

Run
    python -m utopia.visual.plot_exploration_disruption
    (writes the appendix PDF under outputs/visual/exploration_disruption/ and the
    per-bin tables under outputs/docs/exploration_disruption/)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.analysis.exploration_cross_seed import discover_runs, load_table, validate_runs
from utopia.visual.common import GRID, INK, MUTED, STRATEGY_NICE, STRATEGY_PALETTE, nature_style
from utopia.visual.plot_topic_diversity_selection_gate import BIN_LABELS, BINS, EDGE_LABELS, EDGE_TICKS, RUN_PREFIX
from utopia.analysis.statistics import wilson_ci

DODGE = {'explorer': -0.1, 'cautious_explorer': 0.0, 'exploiter': 0.1}  # keeps whiskers apart
XLABEL = 'Topical departure from own recent work'
# CD-sign segments of the appendix panel (a), bottom to top: (table column, label, color)
CD_SIGNS = [('consolidating_pct', 'Consolidating', INK),
            ('zero_pct', 'CD = 0', GRID),
            ('disruptive_pct', 'Disruptive', MUTED)]


def load_papers(run_prefix):
    """Age-3 accepted papers with a defined CD index, binned by topical departure."""
    runs, _ = validate_runs(discover_runs(run_prefix))
    paper_age = load_table(runs, 'paper_age')
    papers = paper_age[(paper_age.age == 3) & paper_age.cd.notna()].copy()
    papers['shift_bin'] = pd.cut(papers.novelty_score, BINS, labels=BIN_LABELS)
    papers = papers.dropna(subset=['shift_bin'])
    papers['disruptive'] = papers.cd > 0
    papers['cited'] = papers.cd_n_disruptive + papers.cd_n_consolidating > 0
    return papers, len(runs)


def with_pooled(papers):
    """Append a copy of every paper tagged strategy == 'all'."""
    return pd.concat([papers, papers.assign(strategy='all')])


def disruption_table(papers):
    """Share of CD > 0 papers per (strategy, departure bin), pooled over worlds.

    Rows with strategy == 'all' pool every strategy. ci_low/ci_high are Wilson
    95% intervals over papers, in percent.
    """
    table = (with_pooled(papers).groupby(['strategy', 'shift_bin'], observed=True)['disruptive']
             .agg(n_papers='size', n_disruptive='sum').reset_index())
    low, high = wilson_ci(table.n_disruptive, table.n_papers)
    table['disruptive_pct'] = 100 * table.n_disruptive / table.n_papers
    table['ci_low'], table['ci_high'] = 100 * low, 100 * high
    return table


def uptake_table(papers):
    """CD-sign composition, and CD among cited papers, per (strategy, departure bin)."""
    both = with_pooled(papers).assign(consolidating=lambda d: d.cd < 0, zero=lambda d: d.cd == 0)
    keys = ['strategy', 'shift_bin']
    table = both.groupby(keys, observed=True).agg(
        n_cd_defined=('cd', 'size'), n_cited=('cited', 'sum'),
        consolidating_pct=('consolidating', 'mean'), zero_pct=('zero', 'mean'),
        disruptive_pct=('disruptive', 'mean'))
    cited = both[both.cited].groupby(keys, observed=True)
    table['disruptive_among_cited_pct'] = cited.disruptive.mean()
    table['mean_cd_cited'] = cited.cd.mean()
    pct_cols = ['consolidating_pct', 'zero_pct', 'disruptive_pct', 'disruptive_among_cited_pct']
    table[pct_cols] *= 100
    table['cited_pct'] = 100 * table.n_cited / table.n_cd_defined
    return table.reset_index()


def draw(ax, table):
    """Pooled line in the background, then one line per strategy with CIs and sized markers.

    Returns the strategy legend handles for the row legend of
    fig_topic_diversity_selection_gate.
    """
    table = table.assign(x=table.shift_bin.cat.codes + table.strategy.map(DODGE).fillna(0))
    pooled = table[table.strategy == 'all']
    ax.plot(pooled.x, pooled.disruptive_pct, color=MUTED, lw=3.2, alpha=0.25,
            solid_capstyle='round', zorder=1)

    # Marker area (pt^2) is linear in the bin's paper count, with a floor so tiny bins stay visible
    n_max = table.loc[table.strategy != 'all', 'n_papers'].max()
    def area(n):
        return 6 + 104 * n / n_max

    for strategy in DODGE:
        sub = table[table.strategy == strategy]
        ax.errorbar(sub.x, sub.disruptive_pct,
                    yerr=[sub.disruptive_pct - sub.ci_low, sub.ci_high - sub.disruptive_pct],
                    color=STRATEGY_PALETTE[strategy], lw=1.1, elinewidth=0.6, capsize=0, zorder=2)
        ax.scatter(sub.x, sub.disruptive_pct, s=area(sub.n_papers), color=STRATEGY_PALETTE[strategy],
                   edgecolor='white', linewidth=0.4, zorder=3)

    size_key = [ax.scatter([], [], s=area(n), color=MUTED, edgecolor='white', linewidth=0.4,
                           label=f'{n:,}') for n in (100, 1000)]
    ax.legend(handles=size_key, title='Papers', loc='lower right', frameon=False,
              title_fontsize=6, handletextpad=0.2, labelspacing=0.3)
    handles = [plt.Line2D([], [], color=STRATEGY_PALETTE[s], lw=1.1, marker='o', ms=3.5,
                          markeredgecolor='white', markeredgewidth=0.4, label=STRATEGY_NICE[s])
               for s in DODGE]
    handles.append(plt.Line2D([], [], color=MUTED, lw=3.2, alpha=0.25, label='All papers'))

    ax.set_xticks(EDGE_TICKS, EDGE_LABELS)
    ax.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
    ax.set_ylim(-3, 72)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel(XLABEL)
    ax.set_ylabel('Disruptive papers (%)')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    return handles


def draw_uptake(axes, papers, table):
    """(a) pooled CD-sign composition per bin, (b) mean CD among cited papers per strategy."""
    ax_sign, ax_cd = axes
    pooled = table[table.strategy == 'all']
    bottom = 0
    for column, label, color in CD_SIGNS:
        ax_sign.bar(pooled.shift_bin.cat.codes, pooled[column], bottom=bottom, width=0.7,
                    color=color, label=label)
        bottom = bottom + pooled[column].to_numpy()
    ax_sign.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    ax_sign.set_ylim(0, 100)
    ax_sign.set_ylabel('Papers (%)')
    # The bars fill the axes, so (a) keeps its legend between the title and the axes
    ax_sign.set_title('a  CD sign, all papers', loc='left', pad=15)
    ax_sign.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                   handlelength=1.0, handletextpad=0.4, columnspacing=0.9)

    cited = papers[papers.cited].assign(strategy=lambda d: d.strategy.map(STRATEGY_NICE))
    sns.pointplot(data=cited, x='shift_bin', y='cd', hue='strategy',
                  hue_order=[STRATEGY_NICE[s] for s in DODGE],
                  palette={STRATEGY_NICE[s]: STRATEGY_PALETTE[s] for s in DODGE},
                  errorbar=('ci', 95), seed=0, dodge=0.3, markersize=3.5, linewidth=1.1,
                  err_kws={'linewidth': 0.6}, ax=ax_cd)
    ax_cd.set_ylim(0, None)
    ax_cd.set_ylabel('Mean CD, cited papers')
    ax_cd.set_title('b  Disruption once cited', loc='left', pad=15)
    ax_cd.legend(loc='lower center', ncol=3, frameon=False,
                 handlelength=1.2, handletextpad=0.4, columnspacing=0.9)
    ax_cd.grid(axis='y', color=GRID, lw=0.5)
    ax_cd.set_axisbelow(True)
    for ax in axes:
        ax.set_xlabel(XLABEL)
        sns.despine(ax=ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run_prefix', default=RUN_PREFIX)
    ap.add_argument('--uptake_out', type=Path, default=Path(
        'outputs/visual/exploration_disruption/fig_disruption_uptake.pdf'))
    ap.add_argument('--summary_csv', type=Path, default=Path(
        'outputs/docs/exploration_disruption/disruption_by_shift.csv'))
    ap.add_argument('--uptake_csv', type=Path, default=Path(
        'outputs/docs/exploration_disruption/uptake_by_shift.csv'))
    args = ap.parse_args()

    papers, n_worlds = load_papers(args.run_prefix)
    table = disruption_table(papers)
    uptake = uptake_table(papers)
    print(table.pivot(index='strategy', columns='shift_bin', values='disruptive_pct')
          .round(1).to_string())
    print(table.pivot(index='strategy', columns='shift_bin', values='n_papers').to_string())
    print(uptake.round(3).to_string())
    print(f'{len(papers):,} papers with defined CD from {n_worlds} worlds')

    nature_style()
    fig_uptake, axes = plt.subplots(1, 2, figsize=(5.0, 2.1))
    draw_uptake(axes, papers, uptake)
    fig_uptake.tight_layout(w_pad=2.0)

    args.uptake_out.parent.mkdir(parents=True, exist_ok=True)
    fig_uptake.savefig(args.uptake_out, bbox_inches='tight')
    for path, frame in [(args.summary_csv, table), (args.uptake_csv, uptake)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    print(f'wrote {args.uptake_out}, {args.summary_csv}, and {args.uptake_csv}')


if __name__ == '__main__':
    main()
