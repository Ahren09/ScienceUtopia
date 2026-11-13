"""Paper figure: topic diversity (a, b), the topical-departure review gate (c), and
disruption by strategy (d) on one 1x4 row.

One shared legend above the row: colors mark the dominant strategy of each world in
(a, b) and the author's strategy in (d). Panel (c) pools all papers and uses its own
teal / violet pair so it is not read as a strategy.

Panels (a, b): ecosystem strategy composition causally controls topic diversity
    The composition-intervention many-world experiment: 10 common-seed paired
    sets (seeds 2001-2006, 2009, 2010, 2013, 2014), each simulated under three
    counterfactual within-institution strategy compositions -- explorer-dominant
    (3 explorers / 1 exploiter / 1 cautious explorer), exploiter-dominant (1/4/0), and
    cautious-dominant (1/1/3) -- with identical institutions, expertise, corpus,
    venues, funding rules, prompts, and seeds. 30 worlds of 100 institutions x
    500 researchers x 10 years; ONLY the composition differs within a set.
    (a) Normalized topic-entropy trajectories by condition: per-world traces
        (translucent) with the across-set mean line per condition. Exploiter-
        dominant worlds collapse diversity; cautious-dominant worlds sustain the
        highest entropy through year 10.
    (b) Paired comparison: each set's 10-year entropy AUC as three dots
        connected within the common seed. Annotated planned contrast
        (cautious-dominant minus the mean of the two extremes): +0.57 AUC,
        positive in 10/10 sets.
    Normalized topic entropy = Shannon entropy over chosen research directions
    / log(#directions), in [0,1]; AUC = trapezoidal area under the 10-year
    trajectory.

Panel (c): the novelty penalty operates at the review gate, not at impact
    The three E4a confirmatory worlds (Qwen3-32B, 1,000 institutions x 5
    researchers, 10 years, seeds 1001-1003, fixed 1/3/1 strategy mix) -- the
    same worlds as the individual-strategy figures. Another run family can be
    selected with --run_prefix (e.g. the five calibration worlds
    explore_calibration_qwen3_32b_neutral_i100_n500_y10).
    Novelty (as computed in utopia/metrics/embedding_tracker.py) =
    1 - cosine(e_paper, c_author), where e_paper is the all-MiniLM-L6-v2
    sentence embedding of the paper's abstract and c_author is the author's
    career centroid: the mean embedding of the author's papers from the
    preceding three simulated years (before the first paper, the mean embedding
    of the author's initial expertise directions). Novelty runs from ~0
    (continuation of the author's own recent work) to ~1 (unrelated to it).
    Papers are binned into five equal-width novelty bins (x ticks at the bin edges).
    Dual y-axes with color-matched axes, 95% CIs drawn as soft ribbons. LEFT axis
    (teal): acceptance rate per submission with Wilson 95% CIs -- falls
    monotonically 48.0% -> 21.8%. RIGHT axis (violet): mean citations by paper
    age 3 AMONG ACCEPTED papers with bootstrap 95% CIs -- rises 0.95 -> 1.98.
    The endpoint value of each series is printed next to its marker; the full
    per-bin table is printed to stdout.
    Acceptance rate is over submission attempts (resubmissions count as
    separate attempts); age-3 citations = citation-graph in-degree by the end
    of the 3rd simulated year post-publication (papers published by year 7
    only, so every paper has a complete window; the paper_age table holds
    accepted papers only).

Panel (d): share of disruptive papers (CD > 0 at age 3) per strategy in the same
    worlds and bins, drawn by utopia.visual.plot_exploration_disruption.draw
    (see that module for the CD index and the marker-size key).

Data provenance (read-only inputs)
    (a, b) outputs/docs/explore_manyworld_qwen3_32b_neutral_i100_n500_y10_seed<S>_mix<M>/
               ecosystem_year.parquet (year, topic_entropy_normalized) for
               S in {2001-2006, 2009, 2010, 2013, 2014}, M in {311, 140, 113}.
    (c, d) outputs/docs/<run_prefix>_seed<S>/
               paper.parquet      (paper_id, accepted, review_score, novelty_score)
               paper_age.parquet  (paper_id, age, citations, cd, novelty_score)

Run
    python -m utopia.visual.plot_topic_diversity_selection_gate
    (writes to the paper repo's figures/fig_topic_diversity_selection_gate.pdf
    by default; override with --out)
"""

from utopia.analysis.statistics import wilson_ci
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.analysis.exploration_cross_seed import discover_runs, validate_runs, load_table
from utopia.visual.common import nature_style, INK, MUTED, STRATEGY_PALETTE

# Panels (a, b): composition-intervention worlds
SETS = [2001, 2002, 2003, 2004, 2005, 2006, 2009, 2010, 2013, 2014]
CONDS = {'311': 'Explorer-dominant', '140': 'Exploiter-dominant', '113': 'Cautious-dominant'}
PALETTE = {'Explorer-dominant': STRATEGY_PALETTE['explorer'],
           'Exploiter-dominant': STRATEGY_PALETTE['exploiter'],
           'Cautious-dominant': STRATEGY_PALETTE['cautious_explorer']}
ORDER = ['Exploiter-dominant', 'Explorer-dominant', 'Cautious-dominant']

# Panel (c): confirmatory worlds
RUN_PREFIX = 'explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10'
BINS = [0, .2, .4, .6, .8, 1.01]
BIN_LABELS = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
# Bin i is drawn at x = i, so its edges (the tick positions) sit at i -/+ 0.5
EDGE_TICKS = [i - 0.5 for i in range(len(BINS))]
EDGE_LABELS = ['0', '0.2', '0.4', '0.6', '0.8', '1']
XLABEL = 'Topical departure\nfrom own recent work'
# Teal / violet, hues kept apart from the strategy colors of panels (a, b, d)
ACCEPT_COLOR = '#00A087'  # acceptance (left axis)
IMPACT_COLOR = '#7E4FA5'  # conditional impact (right axis)


def load_diversity():
    rows = []
    for s in SETS:
        for mix, cond in CONDS.items():
            d = (f'outputs/docs/explore_manyworld_qwen3_32b_neutral_'
                 f'i100_n500_y10_seed{s}_mix{mix}')
            eco = pd.read_parquet(f'{d}/ecosystem_year.parquet').sort_values('year')
            ent = pd.to_numeric(eco['topic_entropy_normalized'], errors='coerce')
            for y, e in zip(eco['year'], ent):
                rows.append({'seed': s, 'condition': cond, 'year': y, 'entropy': e})
    return pd.DataFrame(rows)




def boot_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values)).mean() for _ in range(n_boot)])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def draw_diversity(axes):
    """Draw panels (a) and (b) on two axes."""
    df = load_diversity()
    auc = (df.sort_values('year').groupby(['seed', 'condition'])['entropy']
           .apply(lambda s: np.trapezoid(s.dropna())).reset_index(name='auc'))

    # (a) trajectories: per-world spaghetti + mean per condition
    for (cond, seed), grp in df.groupby(['condition', 'seed']):
        axes[0].plot(grp['year'], grp['entropy'], color=PALETTE[cond], alpha=0.13, lw=0.5)
    sns.lineplot(data=df, x='year', y='entropy', hue='condition', hue_order=ORDER,
                 palette=PALETTE, errorbar=None, lw=1.4, ax=axes[0], legend=False)
    axes[0].set_title('a  Topic Diversity', pad=0)
    axes[0].set_xlabel('Simulated year')
    axes[0].set_ylabel('Normalized topic entropy')
    axes[0].set_xticks([1, 4, 7, 10])

    # (b) paired per-set AUC with within-seed connectors
    xpos = {c: i for i, c in enumerate(ORDER)}
    for s, grp in auc.groupby('seed'):
        vals = grp.set_index('condition')['auc']
        axes[1].plot([xpos[c] for c in ORDER], [vals[c] for c in ORDER],
                     color=MUTED, alpha=0.4, lw=0.5, zorder=1)
    sns.stripplot(data=auc, x='condition', y='auc', order=ORDER, hue='condition',
                  palette=PALETTE, size=3.2, jitter=False, ax=axes[1], legend=False, zorder=2)
    means = auc.groupby('condition')['auc'].mean()
    axes[1].scatter([xpos[c] for c in ORDER], [means[c] for c in ORDER],
                    marker='_', s=160, color=INK, lw=1.2, zorder=3)
    contrast = float((auc[auc.condition == 'Cautious-dominant'].set_index('seed')['auc']
                      - (auc[auc.condition == 'Explorer-dominant'].set_index('seed')['auc']
                         + auc[auc.condition == 'Exploiter-dominant'].set_index('seed')['auc']) / 2).mean())
    # lower right stays clear of the within-seed connectors
    axes[1].text(0.97, 0.04, f'planned contrast:\n+{contrast:.2f} AUC\n(10/10 sets)',
                 transform=axes[1].transAxes, fontsize=6, color=INK, ha='right', va='bottom')
    axes[1].set_title('b  Paired entropy AUC', pad=0)
    axes[1].set_xlabel('Researcher Type')
    axes[1].set_ylabel('Entropy AUC')
    # short tick names: the colors are keyed in the figure-level legend
    axes[1].set_xticks(list(xpos.values()))
    axes[1].set_xticklabels([c.split('-')[0] for c in ORDER])
    axes[1].set_xlim(-0.35, len(ORDER) - 0.65)

    for ax in axes:
        sns.despine(ax=ax)


def draw_selection_gate(ax_l, run_prefix=RUN_PREFIX):
    """Draw the dual-axis panel (c) on ax_l; returns the right axis."""
    runs, _ = validate_runs(discover_runs(run_prefix))
    paper = load_table(runs, 'paper')
    pa = load_table(runs, 'paper_age')

    paper['nov_bin'] = pd.cut(paper['novelty_score'], BINS, labels=BIN_LABELS)
    a3 = pa[pa.age == 3].copy()
    a3['nov_bin'] = pd.cut(a3['novelty_score'], BINS, labels=BIN_LABELS)

    # LEFT: acceptance rate per submission (Wilson CIs), in percent
    g = paper.groupby('nov_bin', observed=True)['accepted'].agg(['sum', 'count']).reindex(BIN_LABELS)
    rate = 100 * g['sum'].values / g['count'].values
    lo, hi = wilson_ci(g['sum'].values, g['count'].values)
    lo, hi = 100 * lo, 100 * hi

    # RIGHT: age-3 citations among accepted papers (bootstrap CIs)
    cit = [a3.loc[a3.nov_bin == b, 'citations'].values for b in BIN_LABELS]
    means = np.array([c.mean() for c in cit])
    cis = np.array([boot_ci(c) for c in cit])  # (n_bins, 2)

    print(pd.DataFrame({'n_submissions': g['count'].values, 'acceptance_pct': rate.round(1),
                        'n_age3_papers': [len(c) for c in cit],
                        'mean_age3_citations': means.round(3)}, index=BIN_LABELS).to_string())

    ax_r = ax_l.twinx()
    x = np.arange(len(BIN_LABELS))

    # 95% CIs as soft ribbons under solid lines with white-edged markers
    for ax, y, low, high, color, marker in [(ax_l, rate, lo, hi, ACCEPT_COLOR, 'o'),
                                           (ax_r, means, cis[:, 0], cis[:, 1], IMPACT_COLOR, 's')]:
        ax.fill_between(x, low, high, color=color, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, y, color=color, lw=1.4, marker=marker, ms=3.6,
                markeredgecolor='white', markeredgewidth=0.5, zorder=3)

    # Endpoint values beside their markers (the numbers quoted in the caption)
    # (first-bin labels extend right, last-bin labels extend left, so both stay inside the spines)
    lab = dict(textcoords='offset points', fontsize=6)
    ax_l.annotate(f'{rate[0]:.1f}%', (x[0], hi[0]), xytext=(-4, 3), ha='left', va='bottom', color=ACCEPT_COLOR, **lab)
    ax_l.annotate(f'{rate[-1]:.1f}%', (x[-1], lo[-1]), xytext=(4, -3), ha='right', va='top', color=ACCEPT_COLOR, **lab)
    ax_r.annotate(f'{means[0]:.2f}', (x[0], cis[0, 0]), xytext=(-4, -3), ha='left', va='top', color=IMPACT_COLOR, **lab)
    ax_r.annotate(f'{means[-1]:.2f}', (x[-1], cis[-1, 1]), xytext=(4, 3), ha='right', va='bottom', color=IMPACT_COLOR, **lab)

    # Color-matched axes so each curve maps unambiguously
    ax_l.set_ylabel('Acceptance rate (%)', color=ACCEPT_COLOR)
    ax_l.tick_params(axis='y', colors=ACCEPT_COLOR)
    ax_l.set_ylim(15, 57)
    ax_l.spines['left'].set_color(ACCEPT_COLOR)
    ax_r.set_ylabel('Mean citations by age 3', color=IMPACT_COLOR)
    ax_r.tick_params(axis='y', colors=IMPACT_COLOR)
    ax_r.set_ylim(0.6, 2.4)
    ax_r.spines['right'].set_color(IMPACT_COLOR)

    ax_l.set_xticks(EDGE_TICKS, EDGE_LABELS)
    ax_l.set_xlim(-0.5, len(BIN_LABELS) - 0.5)
    ax_l.set_xlabel(XLABEL)

    sns.despine(ax=ax_l, right=False)
    sns.despine(ax=ax_r, top=True, left=True, bottom=True, right=False)
    return ax_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/nethome/yjin328/Workspace/Paper/'
                    'Paper_ScienceUtopia/figures/fig_topic_diversity_selection_gate.pdf')
    ap.add_argument('--run_prefix', default=RUN_PREFIX)
    args = ap.parse_args()
    # Imported here because plot_exploration_disruption imports this module's bins
    from utopia.visual.plot_exploration_disruption import disruption_table, draw, load_papers

    nature_style()
    # Constrained layout sizes each gap to its own labels (c's right axis needs the widest one)
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.05), dpi=300, layout='constrained',
                             gridspec_kw={'width_ratios': [0.85, 1.0, 1.05, 1.05]})
    draw_diversity(axes[:2])
    draw_selection_gate(axes[2], args.run_prefix)
    axes[2].set_title('c  Acceptance and impact', pad=0)
    handles = draw(axes[3], disruption_table(load_papers(args.run_prefix)[0]))
    axes[3].set_xlabel(XLABEL)
    axes[3].set_title('d  Disruption by strategy', pad=0)

    # One legend for the row: strategy colors are shared by (a, b, d)
    fig.legend(handles=handles, loc='outside upper center', ncol=4, frameon=False,
               columnspacing=1.2, handlelength=1.5, handletextpad=0.4)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
