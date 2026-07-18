"""Paper figure for project time-to-maturity of SELECTED research directions.

Each research direction carries a nominal duration ("time-to-maturity") of
1-3 years (fast-moving applied fields = 1 yr, standard engineering = 2 yr,
deep theory = 3 yr); the value is the ``years`` field of the ResearchDirection
taxonomy (``DIRECTIONS_DICT`` in ``utopia.agents.research_direction``). This
figure asks how the maturity of the projects agents *choose* depends on two
things: their exploration strategy, and whether they narrowly won or lost an
early funding competition.

Experiment
----------
Two world families (all Qwen3-32B, neutral mechanism):
  (a) Confirmatory 10-year worlds, seeds 1001-1003 (1000 institutions x 5000
      researchers), a balanced mix of explorer / exploiter / cautious_explorer
      agents.
  (b) Matthew-effect regression-discontinuity worlds, seeds 3001-3003 (500
      agents x 6 years), which record each grant applicant's rank relative to
      the funding cutoff. The running variable x = (k - rank) + 0.5, so x > 0
      = narrowly won, x < 0 = narrowly lost.

Panels (1x2)
------------
  (a) Mean time-to-maturity of newly selected projects vs. simulation year,
      one line per exploration strategy (cross-seed mean, per-seed min-max
      band). Explorers commit to long-horizon projects, exploiters churn
      short ones (with a strong duration-cohort echo), cautious explorers sit
      in between. Dashed line = the mean maturity of all available directions
      (2.04), the neutral "no preference" reference.
  (b) Mean post-treatment (years t+1..t+3) selected-project maturity as binned
      means vs. rank distance from the funding cutoff, narrowly-lost (x < 0)
      vs. narrowly-won (x > 0). No discontinuity at the cutoff: winning early
      funding does not shift project horizon -- consistent with the Matthew-RD
      nulls on every other outcome.

Metric
------
  maturity(topic) = DIRECTIONS_DICT[topic].years  (in {1, 2, 3})
  Selected direction per agent-year is the ``direction_topic`` column of
  ``agent_year.parquet`` (non-null in an agent's selection years). Panel (a)
  is event-weighted over newly selected projects; panel (b) forward-fills each
  applicant's last selected direction across the 3-year horizon (the direction
  is recorded only in selection years, but the project is pursued in between).

Data provenance (read-only inputs)
----------------------------------
  outputs/docs/explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed*/agent_year.parquet
  outputs/docs/explore_matthew_rd_qwen3_32b_neutral_i100_n500_y6_seed{3001,3002,3003}/agent_year.parquet
  The near-cutoff RD sample (x, funded, treatment_year, applicant_id, seed) is
  built by utopia.analysis.matthew_rd_analysis.build_application_dataset -- the
  single source of truth with the Matthew-RD figure -- and binned with its
  _binned_means helper.

Run:
    python -m utopia.visual.plot_project_maturity
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utopia.agents.research_direction import DIRECTIONS_DICT
from utopia.analysis.matthew_rd_analysis import _binned_means, build_application_dataset
from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

DEFAULT_CONFIRMATORY_GLOB = ('outputs/docs/'
                             'explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed*')
DEFAULT_RD_EXP_IDS = [
    'explore_matthew_rd_qwen3_32b_neutral_i100_n500_y6_seed3001',
    'explore_matthew_rd_qwen3_32b_neutral_i100_n500_y6_seed3002',
    'explore_matthew_rd_qwen3_32b_neutral_i100_n500_y6_seed3003',
]
DEFAULT_OUT_DIR = 'outputs/visual/project_maturity'
DEFAULT_DOCS_DIR = 'outputs/docs/project_maturity'

# Time-to-maturity (years) per research direction: the single source of truth.
MATURITY = {topic: d.years for topic, d in DIRECTIONS_DICT.items()}
# Mean maturity over all available directions -- the neutral reference line.
TAXONOMY_MEAN = float(np.mean(list(MATURITY.values())))

# Strategy palette shared with plot_exploration_strategies.py.
STRATEGIES = ['explorer', 'cautious_explorer', 'exploiter']
PALETTE = {'explorer': '#d1495b', 'cautious_explorer': '#2a78d6', 'exploiter': '#b07d2b'}
STRATEGY_LABEL = {'explorer': 'Explorer', 'cautious_explorer': 'Cautious explorer',
                  'exploiter': 'Exploiter'}
# Win/lose accents for panel (b): green = funded, red = rejected.
WIN, LOSE = '#1baf7a', '#d1495b'
HORIZON = 3  # post-treatment window t+1..t+HORIZON, matching the RD outcomes


def load_strategy_maturity(runs_glob: str) -> pd.DataFrame:
    """Long-form (seed, year, strategy, maturity-of-selected-project) rows.

    One row per newly selected project (agent-year with a non-null
    direction_topic); maturity is the direction's nominal duration in years.
    """
    frames = []
    for run_dir in sorted(glob.glob(runs_glob)):
        if 'invalidated' in run_dir:
            continue
        path = Path(run_dir) / 'agent_year.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=['year', 'direction_topic', 'strategy'])
        df = df.dropna(subset=['direction_topic']).copy()
        df['maturity'] = df['direction_topic'].map(MATURITY)
        df['seed'] = int(re.search(r'seed(\d+)', run_dir).group(1))
        frames.append(df[['seed', 'year', 'strategy', 'maturity']])
    if not frames:
        raise FileNotFoundError(f'no agent_year.parquet under {runs_glob}')
    return pd.concat(frames, ignore_index=True)


def load_rd_maturity(exp_ids, outputs_dir: str) -> pd.DataFrame:
    """RD application rows augmented with post-treatment selected-project maturity.

    Reuses the canonical RD sample (build_application_dataset) so the win/lose
    contrast is computed on exactly the applications the Matthew-RD study uses.
    For each application, forward-fills the applicant's selected direction over
    years [t+1, t+HORIZON] and averages its maturity.
    """
    apps, _ = build_application_dataset(exp_ids, outputs_dir=outputs_dir, num_years=6)
    apps = apps.copy()
    apps['seed'] = apps['seed'].astype(int)

    # Per-seed agent_year, forward-filled direction so the project pursued
    # between selection years still counts.
    ay = {}
    for exp_id in exp_ids:
        seed = int(re.search(r'seed(\d+)', exp_id).group(1))
        d = pd.read_parquet(Path(outputs_dir) / 'docs' / exp_id / 'agent_year.parquet',
                            columns=['agent_id', 'year', 'direction_topic'])
        d = d.sort_values(['agent_id', 'year'])
        d['dir_ff'] = d.groupby('agent_id')['direction_topic'].ffill()
        d['mat'] = d['dir_ff'].map(MATURITY)
        ay[seed] = d

    def post_maturity(seed, aid, t):
        d = ay.get(int(seed))
        if d is None:
            return np.nan
        w = d[(d['agent_id'] == aid) & (d['year'] >= t + 1) & (d['year'] <= t + HORIZON)]
        m = w['mat'].dropna()
        return float(m.mean()) if len(m) else np.nan

    apps['post_maturity'] = [post_maturity(s, a, t) for s, a, t in
                             zip(apps['seed'], apps['applicant_id'], apps['treatment_year'])]
    return apps.dropna(subset=['post_maturity'])


def strategy_panel(ax, sm: pd.DataFrame):
    """Mean selected-project maturity by year, one line+band per strategy."""
    for strat in STRATEGIES:
        d = sm[sm['strategy'] == strat]
        stat = d.groupby(['seed', 'year'])['maturity'].mean().reset_index()
        agg = stat.groupby('year')['maturity'].agg(['mean', 'min', 'max']).reset_index()
        ax.fill_between(agg['year'], agg['min'], agg['max'],
                        color=PALETTE[strat], alpha=0.15, linewidth=0)
        sns.lineplot(data=agg, x='year', y='mean', ax=ax, color=PALETTE[strat],
                     linewidth=1.6, marker='o', markersize=3.5,
                     markeredgecolor='white', markeredgewidth=0.6,
                     label=STRATEGY_LABEL[strat], legend=False)
        # Direct end-label instead of a legend box (clearer at Nature size).
        y_end = agg.loc[agg['year'] == agg['year'].max(), 'mean'].iloc[0]
        ax.annotate(STRATEGY_LABEL[strat], xy=(agg['year'].max(), y_end),
                    xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize=6, color=PALETTE[strat])
    ax.axhline(TAXONOMY_MEAN, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.annotate('available-direction mean', xy=(1, TAXONOMY_MEAN), xytext=(0, 3),
                textcoords='offset points', ha='left', va='bottom',
                fontsize=5.5, color=MUTED)
    ax.set_xticks(range(2, 11, 2))
    ax.margins(x=0.04)
    ax.set_xlim(0.7, 12.2)  # headroom for the end labels
    ax.set_xlabel('Simulation year')
    ax.set_ylabel('Time-to-maturity of selected project (years)')
    ax.set_title('Project horizon by exploration strategy', pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def rd_maturity_panel(ax, apps: pd.DataFrame, max_x: float = 6):
    """Binned means of post-treatment maturity vs. rank distance to cutoff."""
    for side, color in ((-1, LOSE), (1, WIN)):
        sub = apps[np.sign(apps['x']) == side]
        xs, ms, ses = _binned_means(sub, 'post_maturity', max_x=max_x)
        ax.errorbar(xs, ms, yerr=1.96 * ses, fmt='o', ms=3.2, color=color,
                    ecolor=color, elinewidth=0.9, capsize=1.6, capthick=0.9,
                    markeredgecolor='white', markeredgewidth=0.4)
    ax.axvline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    # Group means near the cutoff (|x| <= 2), the preregistered RD window.
    near = apps[apps['x'].abs() <= 2]
    for side, color, ha in ((-1, LOSE, 'right'), (1, WIN, 'left')):
        m = near.loc[np.sign(near['x']) == side, 'post_maturity'].mean()
        ax.annotate(f'{m:.2f}', xy=(side * 1.5, m), xytext=(side * 6, 10),
                    textcoords='offset points', ha=ha, fontsize=6, color=color)
    ax.annotate('$\\leftarrow$ narrowly lost', xy=(0, 0.03),
                xycoords=('data', 'axes fraction'), xytext=(-4, 0),
                textcoords='offset points', ha='right', fontsize=5.5, color=MUTED)
    ax.annotate('narrowly won $\\rightarrow$', xy=(0, 0.03),
                xycoords=('data', 'axes fraction'), xytext=(4, 0),
                textcoords='offset points', ha='left', fontsize=5.5, color=MUTED)
    ax.set_xticks([])
    ax.set_xlabel('Applicant rank relative to funding cutoff')
    ax.set_ylabel('Selected-project maturity, years $t{+}1..t{+}3$')
    ax.set_title('Project horizon vs. early funding outcome', pad=4)
    ax.grid(axis='y', linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--confirmatory_glob', default=DEFAULT_CONFIRMATORY_GLOB)
    ap.add_argument('--rd_exp_ids', nargs='+', default=DEFAULT_RD_EXP_IDS)
    ap.add_argument('--outputs_dir', default='outputs')
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--docs_dir', default=DEFAULT_DOCS_DIR)
    args = ap.parse_args()

    sm = load_strategy_maturity(args.confirmatory_glob)
    apps = load_rd_maturity(args.rd_exp_ids, args.outputs_dir)

    out_dir, docs_dir = Path(args.out_dir), Path(args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Provenance tables (reproducibility).
    (sm.groupby(['seed', 'year', 'strategy'])['maturity'].mean().reset_index()
       .to_csv(docs_dir / 'strategy_maturity_by_year.csv', index=False))
    apps[['seed', 'applicant_id', 'treatment_year', 'x', 'funded', 'post_maturity']] \
        .to_csv(docs_dir / 'narrow_winlose_maturity.csv', index=False)

    nature_style()
    # Double-column Nature width (180 mm = 7.09 in), 1x2 panels.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.09, 2.7),
                                     gridspec_kw={'width_ratios': [1.25, 1]})
    strategy_panel(ax_a, sm)
    rd_maturity_panel(ax_b, apps)

    for letter, ax in zip('ab', (ax_a, ax_b)):
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', va='bottom')
    sns.despine(fig)
    fig.tight_layout(w_pad=2.0)

    n_lose = int((apps['x'].abs() <= 2).mul(np.sign(apps['x']) < 0).sum())
    n_win = int((apps['x'].abs() <= 2).mul(np.sign(apps['x']) > 0).sum())
    print(f'RD near-cutoff (|x|<=2): narrow-win n={n_win}, narrow-lose n={n_lose}')
    for ext in {'pdf'}:
        path = out_dir / f'fig_project_maturity.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
