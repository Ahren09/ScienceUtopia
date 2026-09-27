"""Figures for network disclosure analysis."""



def build_network_figure(result: dict):
    """Plot saved matched estimands in green without recomputing any analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if result["n_papers"] != 200:
        raise ValueError("Figure requires the original N=200 analysis")
    green = "#008000"
    style = {
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": green, "axes.labelcolor": green,
        "axes.edgecolor": green, "xtick.color": green, "ytick.color": green,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    panels = [
        (
            [.195, .30, .27, .52],
            "(a) Pooled disclosure effects",
            [
                ("identity_pooled", "Identity cue\n$I=B-A$"),
                ("proximity_pooled", "Proximity disclosure\n$R=C-B$"),
                ("total_pooled", "Full disclosure\n$T=C-A$"),
            ],
        ),
        (
            [.705, .30, .27, .52],
            "(b) Paired channel comparisons",
            [
                ("identity_minus_proximity_pooled", "Pooled difference\n$I-R$"),
                (
                    "identity_minus_proximity_d2_minus_far",
                    "Difference of gradients\n$(I_2-I_{\\mathrm{far}})-(R_2-R_{\\mathrm{far}})$",
                ),
            ],
        ),
    ]

    def signed(value):
        return "0.000" if abs(value) < .0005 else f"{value:+.3f}"

    with plt.rc_context(style):
        fig = plt.figure(figsize=(7.2, 2.75), facecolor="white")
        for index, (rect, title, rows) in enumerate(panels):
            ax = fig.add_axes(rect)
            ys = [2.0, 1.0, 0.0] if len(rows) == 3 else [1.8, .2]
            ax.axvline(0, color=green, linewidth=.75, linestyle=(0, (3, 3)))
            for y, (key, label) in zip(ys, rows):
                item = result["paper_bootstrap"][key]
                point, (lo, hi) = item["estimate"], item["ci95"]
                ax.errorbar(
                    point, y, xerr=[[point - lo], [hi - point]],
                    fmt="o", color=green, ecolor=green,
                    markerfacecolor=green, markeredgecolor=green,
                    markersize=4, elinewidth=1.1, capsize=2.7, capthick=.9,
                )
                ax.text(
                    .5, y + .25, f"{signed(point)}  [{signed(lo)}, {signed(hi)}]",
                    transform=ax.get_yaxis_transform(),
                    ha="center", va="center", color=green, fontsize=7.2,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .7},
                )
            ax.set_yticks(ys, [label for _, label in rows])
            ax.tick_params(axis="y", length=0, pad=7, labelsize=8)
            ax.tick_params(axis="x", length=3, pad=3, labelsize=8)
            ax.set_xlim(-.20, .20)
            ax.set_xticks([-.20, 0, .20], ["-0.2", "0", "+0.2"])
            ax.set_ylim(-.50, 2.62)
            ax.set_xlabel("Review-score difference (points)", fontsize=8, labelpad=4)
            for side in ("left", "right", "top"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(green)
            fig.text(
                .02 if index == 0 else .52, .92, title,
                color=green, fontsize=9, fontweight="bold", va="center",
            )
        fig.text(
            .5, .12,
            f"Original study: {result['n_papers']} papers. Equal d2/d3/far weights. "
            f"95% CIs: {result['n_boot']:,} whole-paper bootstrap draws.",
            color=green, ha="center", fontsize=7,
        )
        fig.text(
            .5, .055,
            "A: blind. B: identity + institution. C: B + relationship. "
            "Panel (b) contrasts are exploratory.",
            color=green, ha="center", fontsize=7,
        )
    return fig


def save_network_figure(result, output):
    fig = build_network_figure(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    with matplotlib.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(output, metadata={"Title": "Matched network disclosure effects"})
        fig.savefig(output.with_suffix(".png"), dpi=220)
