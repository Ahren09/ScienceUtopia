"""Paper figure: the two-channel collaboration mechanism that sustains growth.

Experiment (Section 3.6)
    A pure repeat-tie preference collapses the collaboration graph into
    permanent dyads. SciUtopia instead forms ties through two channels each
    year -- a majority new-tie channel (attach to a not-yet-collaborator) plus
    a minority repeat-tie channel -- which keeps the network growing and
    populates the author-to-author distance strata that the network-proximity
    review-bias study (Section 3.7) needs. This figure shows that mechanism at
    work in the source world used for the bias replay (seed 7101).

Panels (1x3, double-column)
    (a) Tie-formation channel mix per year (new-tie vs repeat-tie assignments):
        the new-tie channel stays dominant, so the graph keeps acquiring
        structure instead of thickening a fixed dyad set.
    (b) Connected-component size composition per year (counts of components in
        each size band); dyads give way to larger components and the largest
        component (annotated) grows from 2 to tens of researchers.
    (c) Reviewable-proximity pairs accumulating: ordered author pairs at network
        distance 2, distance 3, and connected at distance >= 4 -- the pool of
        non-trivial proximities the bias study samples from.

Data provenance (read-only input)
    outputs/checkpoints/network_review_bias_source_seed7101/
        preferential_attachment/network_growth_log.jsonl
    One JSON object per simulated year with fields: year, channel_assigned
    {new_tie, repeat_tie}, component_size_hist {1, 2, 3-5, 6-10, >10},
    largest_component, pair_counts {d2, d3, connected_ge4, disconnected}.

中文说明(这张图在讲什么)
    纯“重复结旧交”偏好会让合作网络坍缩成永久的两两小团。SciUtopia 改用两条通道每年结边——
    以“新结交”为主(连一个还没合作过的人)+ 少量“重复结交”——从而让网络持续生长,并把作者
    间的网络距离分层填满,供第 3.7 节的“网络邻近偏倚”研究采样。本图展示该机制在偏倚复现所用
    源世界(seed 7101)中的运行情况。
    · 面板 a 双通道结边:每年新结交 vs 重复结交的分配量;新结交通道保持主导,图不会退化成
      固定小团。
    · 面板 b 连通分量合并:各年连通分量按规模分档的数量;两两小团逐渐并入更大分量,最大分量
      (已标注)从 2 长到几十人。
    · 面板 c 可评审邻近对累积:网络距离 2、距离 3、连通且距离≥4 的有序作者对——偏倚研究可采样
      的非平凡邻近对池。

Run:
    python -m utopia.visual.plot_network_mechanism
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utopia.utils.visual_utils import GRID, INK, MUTED, nature_style

DEFAULT_GROWTH_LOG = ('outputs/checkpoints/network_review_bias_source_seed7101/'
                      'preferential_attachment/network_growth_log.jsonl')
DEFAULT_OUT_DIR = 'outputs/visual/network_review_bias_source_seed7101'

# Channel palette (blue = new tie, green = repeat tie), matching the paper's
# two-category convention. Component-size bands use a light->dark blue ramp so
# larger components read as heavier ink.
# 中文:通道配色(蓝=新结交,绿=重复结交);分量规模档用由浅到深的蓝,越大的分量颜色越重。
CHANNELS = [('new_tie', 'New tie', '#2a78d6'), ('repeat_tie', 'Repeat tie', '#1baf7a')]
SIZE_BANDS = ['2', '3-5', '6-10', '>10']
BAND_COLORS = dict(zip(SIZE_BANDS, sns.light_palette('#2a78d6', n_colors=5)[1:]))
# Distance strata: distinct hues, ordered near -> far.
# 中文:距离分层,用不同色相,按“近→远”排序。
STRATA = [('d2', 'distance 2', '#2a78d6'), ('d3', 'distance 3', '#b07d2b'),
          ('connected_ge4', 'connected $\\geq$4', '#d1495b')]


def load_growth(path: str) -> pd.DataFrame:
    """One row per year from the JSONL growth log (read line by line)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


def channel_panel(ax, df: pd.DataFrame):
    # 面板 a:每年“新结交/重复结交”两通道的分配量,堆叠柱状图。
    years = df['year']
    bottom = [0.0] * len(df)
    for key, label, color in CHANNELS:
        vals = [row['channel_assigned'].get(key, 0) for row in df.to_dict('records')]
        ax.bar(years, vals, bottom=bottom, color=color, width=0.7,
               edgecolor='white', linewidth=0.4, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel('Tie proposals assigned')
    ax.set_title('Two-channel tie formation', pad=4)
    ax.legend(frameon=False, loc='upper left', handlelength=1.0,
              borderaxespad=0.2, fontsize=6)


def component_panel(ax, df: pd.DataFrame):
    # 面板 b:各年连通分量按规模分档(2 / 3-5 / 6-10 / >10 人)的数量,堆叠柱;
    # 并在末年标注最大分量的规模。
    years = df['year']
    bottom = [0.0] * len(df)
    for band in SIZE_BANDS:
        vals = [row['component_size_hist'].get(band, 0) for row in df.to_dict('records')]
        ax.bar(years, vals, bottom=bottom, color=BAND_COLORS[band], width=0.7,
               edgecolor='white', linewidth=0.4, label=f'{band} members')
        bottom = [b + v for b, v in zip(bottom, vals)]
    # Largest-component size, annotated at the final year.
    last = df.iloc[-1]
    ax.annotate(f"largest: {int(last['largest_component'])}",
                xy=(last['year'], bottom[-1]), xytext=(0, 3),
                textcoords='offset points', ha='right', fontsize=5.5, color=INK)
    ax.set_ylabel('Number of components')
    ax.set_title('Components coalesce', pad=4)
    ax.legend(frameon=False, loc='upper left', handlelength=1.0,
              borderaxespad=0.2, fontsize=5.5)


def strata_panel(ax, df: pd.DataFrame):
    # 面板 c:距离 2 / 距离 3 / 连通且距离≥4 的有序作者对数量随年份累积(折线)。
    long = pd.DataFrame({
        'year': df['year'].repeat(len(STRATA)).values,
        'pairs': [row['pair_counts'].get(k, 0)
                  for row in df.to_dict('records') for k, _, _ in STRATA],
        'stratum': [lbl for _ in range(len(df)) for _, lbl, _ in STRATA],
    })
    palette = {lbl: color for _, lbl, color in STRATA}
    sns.lineplot(data=long, x='year', y='pairs', hue='stratum',
                 hue_order=[lbl for _, lbl, _ in STRATA], palette=palette, ax=ax,
                 linewidth=1.5, marker='o', markersize=3.5,
                 markeredgecolor='white', markeredgewidth=0.6, legend=False)
    last = df.iloc[-1]['pair_counts']
    for _, lbl, color in STRATA:
        key = [k for k, l, _ in STRATA if l == lbl][0]
        ax.annotate(f'{lbl}: {int(last[key])}',
                    xy=(df.iloc[-1]['year'], last[key]), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=5.5,
                    color=color)
    ax.set_ylabel('Ordered author pairs')
    ax.set_title('Reviewable proximities accumulate', pad=4)
    ax.margins(x=0.28)  # room for the right-edge stratum labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--growth_log', default=DEFAULT_GROWTH_LOG)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = load_growth(args.growth_log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Double-column Nature width (180 mm = 7.09 in), 1x3 panels.
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.09, 2.1))
    channel_panel(ax_a, df)
    component_panel(ax_b, df)
    strata_panel(ax_c, df)

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlabel('Simulation year')
        ax.set_xticks(range(1, df['year'].max() + 1, 1))
        ax.grid(axis='y', linewidth=0.5, color=GRID)
        ax.set_axisbelow(True)

    for letter, ax in zip('abc', (ax_a, ax_b, ax_c)):
        ax.text(-0.28, 1.08, letter, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', va='bottom')
    sns.despine(fig)
    fig.tight_layout(w_pad=1.8)

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_network_mechanism.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
