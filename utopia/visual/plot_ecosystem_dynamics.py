"""Paper figure: ecosystem-level dynamics of the simulated research world.

Experiment
    The confirmatory exploration worlds (Qwen3-32B, 1000 institutions, 5000
    researchers, 10 years, seeds 1001/1002/1003) run the full SciUtopia loop --
    direction choice, submission, review, resubmission, citation, and grant
    allocation -- with no experimental intervention. This figure summarizes how
    the world as a whole evolves over the decade, pooled across the three seeds.

Panels (1x4, double-column)
    (a) Inequality: funding-resource Gini rising year over year, with the
        citation Gini overlaid for contrast.
    (b) Direction diversity: normalized topic entropy (occupied research
        directions, Shannon entropy / log(#directions)); ~1 = evenly spread.
    (c) Researcher attrition: number of still-active researchers, falling as
        agents run out of resources.
    (d) Review scores: yearly mean of all individual reviewer scores with a
        +-1 SD envelope, against the rubric's accept threshold (3 =
        borderline accept) and the 2.0-2.5 "borderline reject" band. The
        mean stays inside that band all decade (2.29 -> 2.40, median pinned
        at 2.0) and the envelope does NOT narrow -- scores do not converge
        (cross-review SD 0.50 -> 0.55, within-panel reviewer disagreement
        0.29 -> 0.35, all three seeds agree on the direction).

Metrics
    Gini in [0, 1] (0 = equal, 1 = maximally concentrated). Normalized topic
    entropy in [0, 1]. Active researchers counted from the per-year checkpoint.
    Review scores on the 1-5 rubric (decimals allowed); the reviewer prompt
    itself anchors scores low (">60% of papers get 2.0-2.5 or lower"), so the
    borderline-reject level is partly prompt-calibrated -- the year-over-year
    trend and dispersion are the informative part.

Data provenance (read-only inputs)
    outputs/docs/explore_e4a_analysis/ecosystem_trajectories.csv
        per-year, per-seed rows: year, funding_gini, citation_gini,
        topic_entropy_normalized, num_active_researchers, seed.
    (Produced by utopia.analysis.analyze_all_results from the confirmatory
    checkpoints; the tidy CSV is the intended plotting input.)
    outputs/docs/explore_e4a_analysis/review_score_trajectories.csv
        per-year, per-seed rows: n_reviews, mean, sd, percentiles,
        share_borderline_reject, within_event_sd. (Produced by
        utopia.analysis.exploration_cross_seed --review_scores_only from the
        final-year confirmatory checkpoints.)

中文说明(这张图在讲什么)
    三个确证世界(Qwen3-32B,1000 机构、5000 研究者、10 年,seed 1001/1002/1003)在
    无干预下跑完整个 SciUtopia 循环;本图汇总这个世界十年间整体如何演化,三个 seed 汇总。
    · 面板 a 资源不平等上升:资金资源的 Gini 逐年升高(蓝),并叠加引用 Gini(赭)作对照;
      Gini∈[0,1],0=平均、1=极度集中。注:此处用 tidy CSV,资金 Gini 约 0.04→0.36。
    · 面板 b 方向多样性基本持平:归一化主题熵(占用研究方向的香农熵/log(方向数)),≈1 表示
      均匀分布。
    · 面板 c 人口流失:仍活跃的研究者数量随资源耗尽而下降(约 5000→2531)。
    · 面板 d 评审分数:全部单条评审分的年均值 ± 1 个标准差包络,对照打分规则的接收线
      (3 = borderline accept)和 2.0-2.5 的"borderline reject"带。十年间均值一直落在该带内
      (2.29→2.40,逐年中位数恒为 2.0),包络并未收窄——分数没有逐年趋同(跨评审 SD
      0.50→0.55,同一篇论文三位评审间的分歧 0.29→0.35,三个 seed 方向一致)。注意评审
      prompt 本身要求低分锚定(">60% 论文 2.0-2.5 或更低"),分数偏低有 prompt 校准成分,
      信息量在"逐年趋势与离散度",而非绝对水平。

Run:
    python -m utopia.visual.plot_ecosystem_dynamics
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

DEFAULT_TRAJECTORIES = 'outputs/docs/explore_e4a_analysis/ecosystem_trajectories.csv'
DEFAULT_REVIEW_SCORES = 'outputs/docs/explore_e4a_analysis/review_score_trajectories.csv'
DEFAULT_OUT_DIR = 'outputs/visual/explore_e4a_analysis'

# Blue = funding/primary series, ochre = the citation-Gini reference overlay,
# green = diversity. Same validated hexes used across the paper figures.
BLUE, OCHRE, GREEN = '#2a78d6', '#b07d2b', '#1baf7a'


def seed_band(ax, df: pd.DataFrame, value: str, color: str, label: str = None):
    """Per-seed mean line with a min-max ribbon across seeds.

    df has columns year, <value>, seed. The ribbon spans the seed-wise
    min..max at each year (n = 3 seeds, too few for a bootstrap CI), and the
    line is the seed mean. Rows with NaN in `value` are dropped so a missing
    early-year metric (e.g. citation Gini at year 1) does not blank the line.

    中文:画“每个 seed 的均值线 + 跨 seed 的 min-max 阴影带”。带宽取各年 seed 间
    最小/最大值(仅 3 个 seed,不足以做自助 CI),线为 seed 均值;`value` 为 NaN 的行
    先丢弃,以免某年缺测(如第 1 年的引用 Gini)把整条线断开。
    """
    d = df[['year', value, 'seed']].dropna(subset=[value])
    stat = d.groupby('year')[value].agg(['mean', 'min', 'max']).reset_index()
    ax.fill_between(stat['year'], stat['min'], stat['max'],
                    color=color, alpha=0.16, linewidth=0)
    sns.lineplot(data=stat, x='year', y='mean', ax=ax, color=color,
                 linewidth=1.6, marker='o', markersize=3.5,
                 markeredgecolor='white', markeredgewidth=0.6, label=label,
                 legend=False)
    return stat


def end_label(ax, stat: pd.DataFrame, color: str, fmt: str, dy: int = 0):
    """Annotate the final-year mean value at the right end of a line."""
    last = stat.iloc[-1]
    ax.annotate(fmt.format(last['mean']), xy=(last['year'], last['mean']),
                xytext=(4, dy), textcoords='offset points', va='center',
                fontsize=6, color=color)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trajectories', default=DEFAULT_TRAJECTORIES)
    ap.add_argument('--review_scores', default=DEFAULT_REVIEW_SCORES)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.trajectories)
    scores = pd.read_csv(args.review_scores)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Double-column Nature width (180 mm = 7.09 in), 1x4 panels.
    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(1, 4, figsize=(7.09, 2.1))

    # (a) Inequality: funding Gini rising, citation Gini overlay.
    # (a) 资源不平等:资金 Gini 上升,叠加引用 Gini 作对照。
    f = seed_band(ax_a, df, 'funding_gini', BLUE)
    c = seed_band(ax_a, df, 'citation_gini', OCHRE)
    end_label(ax_a, f, INK, '{:.2f}', dy=-2)
    end_label(ax_a, c, MUTED, '{:.2f}', dy=4)
    ax_a.annotate('funding', xy=(f.iloc[4]['year'], f.iloc[4]['mean']),
                  xytext=(2, -8), textcoords='offset points', fontsize=6,
                  color=BLUE)
    ax_a.annotate('citations', xy=(c.iloc[4]['year'], c.iloc[4]['mean']),
                  xytext=(0, 6), textcoords='offset points', fontsize=6,
                  color=OCHRE)
    ax_a.set_ylim(0, 0.65)
    ax_a.set_ylabel('Gini coefficient')
    ax_a.set_title('Resource inequality rises', pad=4)

    # (b) Direction diversity.
    # (b) 研究方向多样性(归一化主题熵)。
    e = seed_band(ax_b, df, 'topic_entropy_normalized', GREEN)
    end_label(ax_b, e, INK, '{:.2f}')
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel('Normalized topic entropy')
    ax_b.set_title('Direction diversity holds', pad=4)

    # (c) Researcher attrition.
    # (c) 研究者流失(活跃人数下降)。
    n = seed_band(ax_c, df, 'num_active_researchers', BLUE)
    end_label(ax_c, n, INK, '{:.0f}')
    ax_c.set_ylim(0, 5300)
    ax_c.set_ylabel('Active researchers')
    ax_c.set_title('Population attrites', pad=4)

    # (d) Review scores: yearly mean of individual reviewer scores with a
    # +-1 SD envelope, vs. the accept threshold (3) and the 2.0-2.5
    # borderline-reject band. A non-narrowing envelope = no convergence.
    # (d) 评审分数:单条评审分的年均值 ± 1 SD 包络;包络不收窄 = 分数没有趋同。
    yearly = (scores.groupby('year')[['mean', 'sd']].mean().reset_index())
    ax_d.axhspan(2.0, 2.5, color=OCHRE, alpha=0.10, linewidth=0)
    ax_d.axhline(3.0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax_d.fill_between(yearly['year'], yearly['mean'] - yearly['sd'],
                      yearly['mean'] + yearly['sd'],
                      color=BLUE, alpha=0.16, linewidth=0)
    m = seed_band(ax_d, scores, 'mean', BLUE)
    end_label(ax_d, m, INK, '{:.2f}')
    ax_d.annotate('accept threshold', xy=(1, 3.0), xytext=(0, 2),
                  textcoords='offset points', ha='left', va='bottom',
                  fontsize=5.5, color=MUTED)
    ax_d.annotate('borderline-reject band', xy=(10, 2.05), xytext=(0, 0),
                  textcoords='offset points', ha='right', va='bottom',
                  fontsize=5.5, color=OCHRE)
    ax_d.annotate('mean ± SD', xy=(2, float(yearly['mean'].iloc[1]
                                            - yearly['sd'].iloc[1])),
                  xytext=(0, -7), textcoords='offset points', ha='left',
                  va='top', fontsize=5.5, color=BLUE)
    ax_d.set_ylim(1.4, 3.35)
    ax_d.set_ylabel('Review score (1–5)')
    ax_d.set_title('Scores stay borderline', pad=4)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlabel('Simulation year')
        ax.set_xticks(range(2, 11, 2))
        ax.margins(x=0.04)
        ax.grid(axis='y', linewidth=0.5, color=GRID)
        ax.set_axisbelow(True)

    for letter, ax in zip('abcd', (ax_a, ax_b, ax_c, ax_d)):
        ax.text(-0.28, 1.08, letter, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', va='bottom')
    sns.despine(fig)
    fig.tight_layout(w_pad=1.6)

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_ecosystem_dynamics.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
