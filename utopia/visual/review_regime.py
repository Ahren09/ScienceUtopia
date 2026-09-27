"""Diagnostic figures for review regime analyses."""

import os
import numpy as np
from utopia.visual.common import save_figure
from utopia.analysis.statistics import weighted_slope


def make_figures(reviews, cells, per_paper, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    colors = {"A": "tab:red", "B": "tab:blue", "C": "tab:gray"}

    ok = reviews[reviews.success & (reviews.arm == "blind") & (reviews.panel == 1)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, g in ok.groupby("regime"):
        ax.hist(
            g.score_raw,
            bins=np.arange(0.75, 5.76, 0.5),
            alpha=0.5,
            label=f"{r} (n={len(g)})",
            color=colors.get(r),
            density=True,
        )
    ax.set_xlabel("raw review score (blind)")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title("fig1: score distributions by regime")
    save_figure(fig, os.path.join(out_dir, "fig1_score_distributions.png"), dpi=150)

    for name, frame, col, fname in (
        ("fig2: within-paper SD (paired)", per_paper["P1"], "sd", "fig2_sd_paired.png"),
        (
            "fig3: rationale diversity (paired)",
            per_paper["P2"],
            "div",
            "fig3_diversity_paired.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(frame.B, frame.A, s=12, alpha=0.5)
        lim = [np.nanmin([frame.A, frame.B]), np.nanmax([frame.A, frame.B])]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("regime B")
        ax.set_ylabel("regime A")
        ax.set_title(f"{name} (n={len(frame)} papers)")
        save_figure(fig, os.path.join(out_dir, fname), dpi=150)

    p3 = per_paper["P3"].dropna(subset=["delta_score", "novelty_score"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        p3.novelty_score,
        p3.delta_score,
        s=14,
        alpha=0.5,
        c=np.log(p3.ipw_weight),
        cmap="viridis",
    )
    b1, b0 = weighted_slope(p3.delta_score, p3.novelty_score, p3.ipw_weight)
    xs = np.linspace(p3.novelty_score.min(), p3.novelty_score.max(), 50)
    ax.plot(xs, b0 + b1 * xs, "r-", label=f"IPW slope={b1:.3f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("novelty score")
    ax.set_ylabel("delta = mean_A - mean_B (blind)")
    ax.legend()
    ax.set_title(f"fig4: paired novelty model (n={len(p3)})")
    save_figure(fig, os.path.join(out_dir, "fig4_novelty_paired.png"), dpi=150)

    p4 = per_paper["P4"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (lab, v) in enumerate(
        (
            ("d_A (hi-lo in A)", p4.d_A),
            ("d_B (hi-lo in B)", p4.d_B),
            ("DiD (d_A - d_B)", p4["diff"]),
        )
    ):
        m, s = np.nanmean(v), np.nanstd(v, ddof=1) / np.sqrt(v.notna().sum())
        ax.errorbar(m, i, xerr=1.96 * s, fmt="o", capsize=4)
        ax.text(m, i + 0.12, lab, ha="center", fontsize=9)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks([])
    ax.set_xlabel("score points")
    ax.set_title(f"fig5: synthetic institutional-prestige cue (n={len(p4)})")
    save_figure(fig, os.path.join(out_dir, "fig5_prestige_forest.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for r, g in cells[(cells.arm == "blind") & (cells.panel == 1)].groupby("regime"):
        ax.hist(
            g.mean_just_len,
            bins=30,
            alpha=0.5,
            label=r,
            color=colors.get(r),
            density=True,
        )
    ax.set_xlabel("mean justification length (chars)")
    ax.legend()
    ax.set_title("fig6: rationale length by regime")
    save_figure(fig, os.path.join(out_dir, "fig6_rationale_length.png"), dpi=150)
