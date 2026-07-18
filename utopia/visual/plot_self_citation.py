"""Paper figure: who self-cites in the E4a large worlds (RQ9).

Experiment
    Self-citation analysis over the three frozen E4a confirmatory worlds
    (1,000 institutions x 5,000 researchers x 10 years, seeds 1001-1003,
    75,661 citation edges pooled). Every paper is solo-authored, so a
    self-citation is an edge whose citing and cited papers share the author.
    Self-citation is rare overall (308 edges, 0.41%), and removing it leaves
    the citation Gini essentially unchanged (delta <= 0.0005).

Panels (1x2, double-column halves; house dot-and-whisker style matching
fig_career_outcomes / fig_novelty_selection_gate)
    (a) Forest-style dot-and-whisker: self-citation rate by the citing
        author's strategy, pooled over the three seeds with Wilson 95% CIs
        and direct value labels. Exploiters self-cite at 0.57%, cautious
        explorers at 0.13%, and explorers never do (0 of 10,861 edges).
        All pairwise contrasts are significant under the institution-level
        hierarchical bootstrap (p_BH = 1e-4).
    (b) Dot-and-whisker over the citing author's career-funding quartile
        (within-seed quartiles, Wilson 95% CIs), connected by a thin line:
        the best-funded quartile self-cites the most (0.67%).

Metrics
    Rate = self-citation edges / all outgoing citation edges of the group.
    Error bars are Wilson score 95% intervals on the pooled proportion;
    inferential claims come from strategy_contrasts.csv (hierarchical
    bootstrap over seeds -> institutions, BH-corrected).

Data provenance (read-only inputs)
    outputs/docs/self_citation_e4a/rates_by_strategy.csv
    outputs/docs/self_citation_e4a/rates_by_status.csv
    (produced by python -m utopia.analysis.self_citation)

Run
    python -m utopia.visual.plot_self_citation
    (writes outputs/visual/self_citation_e4a/fig_self_citation.pdf by
    default; override with --out)
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import (ACCENT_COOL, INK, MUTED, STRATEGY_NICE,
                                       STRATEGY_PALETTE, nature_style)

DEFAULT_ANALYSIS_DIR = 'outputs/docs/self_citation_e4a'
DEFAULT_OUT = 'outputs/visual/self_citation_e4a/fig_self_citation.pdf'
STRATEGY_ORDER = ['explorer', 'cautious_explorer', 'exploiter']


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% interval for a binomial proportion."""
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, center - half), center + half


def _rates_ci(ks, ns):
    """Percent-scale rates with asymmetric Wilson CI half-widths (2, n)."""
    rates = np.array([100 * k / n for k, n in zip(ks, ns)])
    los, his = zip(*[wilson_ci(k, n) for k, n in zip(ks, ns)])
    err = np.array([[r - 100 * lo, 100 * hi - r]
                    for r, lo, hi in zip(rates, los, his)]).T
    return rates, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis_dir', default=DEFAULT_ANALYSIS_DIR)
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    by_strategy = pd.read_csv(os.path.join(args.analysis_dir, 'rates_by_strategy.csv'))
    pooled = (by_strategy[by_strategy['seed'] == 'pooled']
              .set_index('strategy').loc[STRATEGY_ORDER])
    by_status = pd.read_csv(os.path.join(args.analysis_dir, 'rates_by_status.csv'))
    funding = (by_status[by_status['status_measure'] == 'final_funding']
               .sort_values('quartile'))

    nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.6), dpi=300)

    # (a) forest-style dot-and-whisker: rate by strategy, pooled over 3 seeds
    rates, err = _rates_ci(pooled['n_self'].values, pooled['n_edges'].values)
    ys = np.arange(len(STRATEGY_ORDER))[::-1]  # explorer on top
    # axes[0].axvline(0, ls='--', lw=0.7, color=INK, zorder=1)
    for y, s, r, e_lo, e_hi in zip(ys, STRATEGY_ORDER, rates, err[0], err[1]):
        c = STRATEGY_PALETTE[s]
        axes[0].errorbar(r, y, xerr=[[e_lo], [e_hi]], fmt='o', ms=4.2,
                         color=c, ecolor=c, elinewidth=1.1, capsize=0, zorder=3)
        # if r > 0:
        #     axes[0].annotate(f'{r:.2f}%', xy=(r + e_hi + 0.03, y), va='center',
        #                      ha='left', fontsize=5.2, color=INK)
        # else:
        #     n = int(pooled.loc[s, 'n_edges'])
        #     axes[0].annotate(f'0/{n:,}', xy=(r + e_hi + 0.03, y), va='center',
        #                      ha='left', fontsize=5.2, color=INK)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([STRATEGY_NICE[s].replace(' ', '\n')
                             for s in STRATEGY_ORDER], fontsize=5.5)
    axes[0].set_ylim(-0.55, len(STRATEGY_ORDER) - 0.45)
    axes[0].set_xlim(-0.05, 0.7)
    axes[0].set_title('a  Self-citation by strategy', pad=0)
    axes[0].set_xlabel('Self-citation rate (%)')

    # (b) dot-and-whisker over career-funding quartiles, connected
    rates, err = _rates_ci(funding['n_self'].values, funding['n_edges'].values)
    x = np.arange(len(funding))
    axes[1].plot(x, rates, color=ACCENT_COOL, lw=1.1, zorder=2)
    axes[1].errorbar(x, rates, yerr=err, fmt='o', ms=3.6, color=ACCENT_COOL,
                     ecolor=ACCENT_COOL, elinewidth=1.0, capsize=0, zorder=3)
    axes[1].annotate(f'{rates[-1]:.2f}%', xy=(x[-1], rates[-1] + err[1][-1] + 0.04),
                     ha='center', fontsize=5.2, color=INK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'Q{int(q)}' for q in funding['quartile']])
    axes[1].set_ylim(0, 0.95)
    axes[1].set_title('b  Self-citation by funding', pad=0)
    axes[1].set_ylabel('Self-citation rate (%)')
    axes[1].set_xlabel('Career funding quartile')

    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout(w_pad=1.1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
