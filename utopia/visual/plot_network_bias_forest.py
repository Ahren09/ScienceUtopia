"""Paper figure: two disclosure channels in LLM peer review -- proximity vs identity.

中文:主要发现(写给读者,不假定任何术语基础,读这一段即可看懂整张图)

    这项研究想回答一个问题:当我们用大语言模型(LLM)充当论文审稿人时,如果让它“知道”
    论文作者是谁、或者知道自己和作者在学术圈里关系有多近,它打的分会不会因此被影响
    (产生偏袒)?

    怎么做:准备一批条件匹配好的论文,让模型在三种不同“告知程度”下分别打分——
      · 条件 A(盲审):完全不告诉模型关于作者的任何信息。
      · 条件 B(只给身份):告诉模型作者的一个化名代号(知道“是同一个人”,但不知关系)。
      · 条件 C(再给关系):在 B 的基础上,还告诉模型它和这位作者在“合作网络”里关系有多近。
    这里“合作网络关系/距离”的意思:两位学者只要一起发过论文就算直接相连;由此定义远近——
      · “共享一个合作者”:你俩没合作过,但都和同一个人合作过(相当于“朋友的朋友”,最近);
      · “隔两步”:再往外一层;
      · “陌生人”:在合作网络里没有任何近的连接。
    通过比较不同条件下的分数,就能把两种影响单独拆开(分数是 1-5 分制,正的差值 = 该信息
    让模型打得更高、即偏袒):
      · “C 减 B”:C 比 B 只多披露了“关系”这一层,于是它衡量“知道关系近不近”是否改变打分;
      · “B 减 A”:B 比 A 只多披露了“身份”这一层,于是它衡量“露出作者身份”是否改变打分。

    发现(对应左右两张子图):
      · 图 a——告诉模型“和作者关系近”,分数几乎不变:不管作者是“朋友的朋友”还是更远,
        相对“陌生人”的分差都在 0 附近(+0.03、-0.02),且落在 ±0.12 分这个“可忽略”范围内;
        换 6 种不同统计算法结论都一样。即:LLM 审稿不会因为“作者离自己近”就手下留情。
      · 图 b——但“露出作者身份”会带来一点点加分,而且关系越近越明显:作者是“朋友的朋友”
        时约 +0.075 分(只有这一档在统计上站得住),隔两步 +0.05、陌生人 +0.04,总体平均
        +0.055 分;不过这个信号偏弱,做过严格的多重比较校正后就不再显著,只能算“提示性线索”。

    一句话要点:LLM 审稿对“关系远近”本身并不敏感;真正可能带来轻微偏袒的是“知道作者是谁”
    这件事,而且集中在关系最近的时候——这把偏袒的来源从“网络邻近”重新指向“身份”,也说明
    后续值得做一个专门针对“身份披露”的确证实验。

Experiment (Section 3.7)
    Reviewers score matched papers whose author sits at a known collaboration-
    network distance from them (distance 2 = one shared collaborator; distance 3;
    far/disconnected). Disclosure is manipulated in three conditions: A (blind),
    B (pseudonymous identity only), C (true network relationship). This figure
    separates the two disclosure channels and resolves each by author distance.

Design (three panels, shared x-axis in review-score points, 1-5 scale; the
panels read left-to-right as a decomposition, (a) + (b) = (c))
    (a) Proximity channel (C - B), distance contrasts vs the far comparator:
        does disclosing the true relationship bias scores, and does that differ
        by distance? All effects are null and flat -- the paper's headline null.
        The 6 robustness re-estimators of the primary d2 contrast collapse into
        one footnote instead of 6 near-identical rows.
    (b) Identity channel (B - A), the identity-disclosure effect by author
        distance, plus the pooled summary (diamond): a gradient that is largest
        and only resolved at distance 2.
    (c) Total disclosure (C - A) = full-information minus blind = the additive
        sum of (a) and (b), by author distance plus the pooled summary: what a
        reviewer going from blind to fully informed actually shifts by. Because
        the proximity channel is ~0, this tracks the identity channel -- largest
        and only resolved at distance 2.
    Dashed line at zero; pale cool-gray band = pre-registered
    +/-0.12-point practical-equivalence bound. Each 95% bootstrap CI is drawn
    twice over the SAME interval: a pale capsule under a thin crisp line.
    Open (white) cobalt marker = CI crosses zero; filled red marker with a bold
    red value label = CI excludes zero (the resolved identity signal).
    Colors are the vivid accent pair shared with the resubmission figure
    (VIVID_WARM / VIVID_COOL in utopia.utils.visual_utils). Rows sit at the same height in
    every panel so the same author distance lines up across (a), (b), (c).

Row meanings 每一栏含义(y 轴用大白话,括号里的 d2/d3/far 是技术标注)
    "Network distance" = number of co-authorship steps between reviewer and
    author: d2 = share one collaborator ("friend of a friend"); d3 = two steps;
    far = no close link (a stranger). 中文:网络距离 = 合作网络上审稿人与作者之间的
    步数;d2=共享一个合作者,d3=隔两步,far=没有近连接的陌生人。
    Panel (a) proximity channel C - B, each vs. a stranger (x = score change
    relative to a far/stranger author):
      * Shares a collaborator (d2): pre-registered primary contrast
        (C-B)@d2 - (C-B)@far. 中文:共享一个合作者,相对陌生人的邻近披露效应差(主对比)。
      * Two steps removed (d3): (C-B)@d3 - (C-B)@far. 中文:隔两步,相对陌生人。
      * Strangers: linked vs. not: comparator check inside "far".
        中文:陌生人内部,松散相连 vs 完全无连接的对照检验。
    Panel (b) identity channel B - A, by author distance (x = absolute score
    change from revealing the author's identity):
      * Shares a collaborator / Two steps removed / A stranger: identity-
        disclosure effect at each distance. 中文:各距离层上“揭示身份 vs 盲审”的效应
        (仅“共享合作者”一层显著)。
      * All authors (pooled): distance-pooled summary (diamond).
        中文:各距离合并的汇总(菱形)。

Data provenance (read-only input)
    outputs/docs/network_review_bias_final/results.json
        primary.primary_d2_minus_far, secondary.S1_d3_minus_far,
        secondary.S3_far_connected_vs_disconnected.disconnected,
        secondary.S4_identity_B_minus_A.{by_stratum,pooled},
        secondary.S9_total_disclosure_C_minus_A.{by_stratum,pooled}, plus the 6
        primary robustness re-estimators for the footnote range.
    (Produced by utopia.analysis.network_review_bias_analysis.)

Run:
    python -m utopia.visual.plot_network_bias_forest
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utopia.utils.visual_utils import MUTED, VIVID_COOL, VIVID_WARM, nature_style

DEFAULT_RESULTS = 'outputs/docs/network_review_bias_final/results.json'
DEFAULT_OUT_DIR = 'outputs/visual/network_review_bias_final'

# Cobalt = 95% CI crosses zero, red = CI excludes zero (resolved). The pale
# cool-gray band marks the "region of practical zero" and lets both colors pop.
NULL_COLOR, RESOLVED_COLOR = VIVID_COOL, VIVID_WARM
BAND_FILL = '#EDF2F7'
EQUIV_BOUND = 0.12  # pre-registered practical-equivalence bound (score points)

# Two-layer CI glyph: a thick pale capsule and a thin crisp line over the SAME
# interval. Butt caps so both layers stop exactly at ci_lo / ci_hi.
CAPSULE_LW, CAPSULE_TINT, CI_LW = 5.0, 0.3, 1.0
MARKER_SIZE = {'stratum': 5.2, 'summary': 5.0}  # circle vs. diamond (visually matched)


def _tint(color, amount):
    """Opaque mix of `color` with white (amount=1 -> color, 0 -> white).

    Opaque rather than alpha so the capsule never muddies over the sand band.
    """
    return tuple(1 - amount * (1 - c) for c in to_rgb(color))


def _e(d, est_key='estimate'):
    """(estimate, ci_lo, ci_hi) straight from a results.json leaf block."""
    return d[est_key], d['ci_lo'], d['ci_hi']


def build_channels(res: dict):
    """Three channels of forest rows from results.json.

    Channel A (proximity, C-B): distance contrasts vs far -- all null.
    Channel B (identity, B-A): per-stratum effect + pooled summary -- gradient.
    Channel C (total, C-A): full-disclosure effect = A + B, per-stratum + pooled;
        the identity-driven bump concentrated at 2-hop, shown as the sum readers
        would actually see going from blind (A) to fully informed (C).
    Each row: (label, estimate, ci_lo, ci_hi, kind); kind 'summary' -> diamond.
    中文:A=邻近关系通道(相对远处,均为零结果);B=身份通道(按距离的效应+合并汇总);
    C=总披露通道(盲审 A → 完全告知 C 的合并效应 = A+B,按距离的效应+菱形汇总)。
    """
    p, s = res['primary'], res['secondary']
    s4 = s['S4_identity_B_minus_A']
    s9 = s['S9_total_disclosure_C_minus_A']

    # Plain-language y labels; the technical stratum tag (d2/d3/far) is kept in
    # parentheses so the mapping to the paper is unambiguous.
    # 中文:y 轴用大白话,括号里保留 d2/d3/far 等技术标注对照。
    channel_a = [
        ('2-hop', *_e(p['primary_d2_minus_far']), 'stratum'),     # (C-B)@d2 - (C-B)@far | 共享一个合作者,相对陌生人
        ('3-hop', *_e(s['S1_d3_minus_far']), 'stratum'),              # (C-B)@d3 - (C-B)@far | 隔两步,相对陌生人
        (r'$\geq$4-hop',                                                 # far comparator check | 陌生人内部:松散相连 vs 完全无连接
         *_e(s['S3_far_connected_vs_disconnected']['disconnected']), 'stratum'),
    ]
    channel_b = [
        ('2-hop', *_e(s4['by_stratum']['d2']), 'stratum'),        # identity effect at d2 | 共享一个合作者(显著)
        ('3-hop', *_e(s4['by_stratum']['d3']), 'stratum'),            # identity effect at d3 | 隔两步
        (r'$\geq$4-hop', *_e(s4['by_stratum']['far']), 'stratum'),                # identity effect at far | 陌生人
        ('All', *_e(s4['pooled']), 'summary'),                        # distance-pooled summary | 各距离平均(菱形)
    ]
    channel_c = [
        ('2-hop', *_e(s9['by_stratum']['d2']), 'stratum'),        # total effect at d2 | 共享一个合作者(显著)
        ('3-hop', *_e(s9['by_stratum']['d3']), 'stratum'),            # total effect at d3 | 隔两步
        (r'$\geq$4-hop', *_e(s9['by_stratum']['far']), 'stratum'),                # total effect at far | 陌生人
        ('All', *_e(s9['pooled']), 'summary'),                        # distance-pooled summary | 各距离平均(菱形)
    ]
    return channel_a, channel_b, channel_c


def robustness_range(res: dict):
    """Min/max point estimate across the 6 primary d2 re-estimators (footnote)."""
    p = res['primary']
    ests = [
        p['primary_d2_minus_far']['estimate'],
        p['robust_reviewer_cluster']['estimate'],
        p['robust_twoway']['estimate'],
        p['robust_stacked_regression']['d2_vs_far']['estimate'],
        p['robust_similarity_adjusted']['d2_vs_far']['estimate'],
        p['robust_hierarchical_seed_paper'][0]['mean_diff'],
    ]
    return min(ests), max(ests)


def _ci_glyph(color, resolved, kind):
    """Artist kwargs for one CI row: (capsule, CI line, marker).

    Shared by the panels and the legend keys so both always match.
    """
    capsule = dict(color=_tint(color, CAPSULE_TINT), linewidth=CAPSULE_LW,
                   solid_capstyle='butt')
    line = dict(color=color, linewidth=CI_LW, solid_capstyle='butt')
    # Filled markers get a white halo so they pop off the capsule. Open markers
    # keep a colored ring on a white face.
    marker = dict(marker='D' if kind == 'summary' else 'o', linestyle='none',
                  markersize=MARKER_SIZE[kind],
                  markerfacecolor=color if resolved else 'white',
                  markeredgecolor='white' if resolved else color,
                  markeredgewidth=0.8 if resolved else 1.1)
    return capsule, line, marker


def forest_panel(ax, rows, title, xlabel, n_slots):
    """Draw one channel as a mini forest; accent-flag CIs that exclude zero.

    Rows fill the top `n_slots` slots so the same author distance sits at the
    same height in every panel (panel a has no pooled row).
    中文:画一条通道的森林图;置信区间不含 0 的点用实心+强调色标出。
    `xlabel` 让各面板各写各自的 x 轴含义(相对陌生人 / 揭示身份的绝对效应)。
    """
    ys = [n_slots - 1 - i for i in range(len(rows))]  # first row at the top
    ax.axvspan(-EQUIV_BOUND, EQUIV_BOUND, facecolor=BAND_FILL, linewidth=0, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)

    for y, (label, est, lo, hi, kind) in zip(ys, rows):
        resolved = lo > 0 or hi < 0  # CI excludes zero -> a resolved effect
        color = RESOLVED_COLOR if resolved else NULL_COLOR
        capsule, line, marker = _ci_glyph(color, resolved, kind)
        ax.plot([lo, hi], [y, y], zorder=2, **capsule)
        ax.plot([lo, hi], [y, y], zorder=3, **line)
        ax.plot(est, y, zorder=4, **marker)
        # Flag resolved (accent) effects with a dagger -> raw-significant only,
        # not robust to multiple-comparison correction (see the figure caption).
        label_txt = f'{est:+.3f}' + ('$^{\\dagger}$' if resolved else '')
        ax.annotate(label_txt, xy=(hi, y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=6, color=RESOLVED_COLOR if resolved else MUTED,
                    fontweight='bold' if resolved else 'normal')

    # Separator above any pooled-summary row.
    summary_ys = [y for y, r in zip(ys, rows) if r[4] == 'summary']
    if summary_ys:
        ax.axhline(max(summary_ys) + 0.5, color=MUTED, linewidth=0.5,
                   linestyle=(0, (1, 1.5)), zorder=1)

    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis='y', length=0, pad=3)  # left spine is removed
    ax.set_ylim(-0.6, n_slots - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=4)
    ax.margins(x=0.18)  # room for the effect-size labels past the CI ends


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', default=DEFAULT_RESULTS)
    ap.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    res = json.load(open(args.results))
    channel_a, channel_b, channel_c = build_channels(res)
    rlo, rhi = robustness_range(res)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nature_style()
    # Double-column Nature width (180 mm = 7.09 in), three forest panels reading
    # left-to-right as a decomposition: (a) proximity + (b) identity = (c) total
    # disclosure. Titles and x-labels are in plain language; y ticks name the
    # closeness level.
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(6.6, 1.8))
    n_slots = max(len(channel_a), len(channel_b), len(channel_c))
    forest_panel(ax_a, channel_a,
                 'Inform that the author\nis network-close',
                 'Score change vs. a stranger author', n_slots)
    forest_panel(ax_b, channel_b,
                 'Revealed author\nidentity',
                 'Score change from revealing identity', n_slots)
    forest_panel(ax_c, channel_c,
                 'Full disclosure of\nidentity and relationship',
                 'Score change vs. blind review', n_slots)

    # Shared x-range so the three channels are directly comparable.
    axes = (ax_a, ax_b, ax_c)
    lo = min(ax.get_xlim()[0] for ax in axes)
    hi = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(lo, hi)

    # Panel letters at the top-left corner, pushed left of the centered two-line
    # titles so they never touch the first title word.
    for letter, ax in zip('abc', axes):
        ax.text(-0.12, 1.13, letter, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', va='bottom', ha='right')

    # Legend keys stack the same three layers as the data rows (capsule, CI
    # line, marker) via HandlerTuple.
    handles = [
        tuple(Line2D([], [], **kw) for kw in _ci_glyph(NULL_COLOR, False, 'stratum')),
        tuple(Line2D([], [], **kw) for kw in _ci_glyph(RESOLVED_COLOR, True, 'stratum')),
        Patch(facecolor=BAND_FILL, linewidth=0),
    ]
    labels = ['CI crosses zero', 'CI excludes zero (raw)$^{\\dagger}$',
              f'±{EQUIV_BOUND:.2f}-pt equivalence']
    # Plain-language definition so the y-tick tags (d2/d3/far) are self-contained.
    # fig.text(0.5, 0.115,
    #          'Network distance d = distance on the co-authorship between reviewer and author '
    #          '(d2 = share one collaborator; d3 = two steps; far = no close link).',
    #          ha='center', va='bottom', fontsize=5.6, color=MUTED)
    fig.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
               frameon=False, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), handlelength=2.2, handleheight=0.9,
               columnspacing=2.0, handletextpad=0.7)
    # fig.text(0.5, 0.0,
    #          '$^{\\dagger}$Raw-significant only: the identity effect does not survive '
    #          'multiple-comparison (Benjamini-Hochberg) correction (pooled '
    #          '$p_\\mathrm{adj}=0.078$).',
    #          ha='center', va='bottom', fontsize=5.6, color=MUTED)
    sns.despine(fig, left=True)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.0)

    for ext in ('pdf', 'png'):
        path = out_dir / f'fig_network_bias_forest.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'✅ Saved figure to: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
