#!/usr/bin/env python3
"""Render the admitted A/C annual values without recomputing outcomes.

Default input: the complete reviewed annual table embedded as ADMITTED_ANNUAL_CSV.
No external data file is required. --data optionally selects an alternate CSV.
Ratios are the saved three-decimal values, not recomputed from counts.
No smoothing, fitting, resampling, confidence intervals, or new analysis.

Usage:
    python -m utopia.visual.plot_ac_resubmission
Run from the ScienceUtopia repository. Requires matplotlib >= 3.8.
Outputs editable PDF/SVG and a 300 dpi PNG in outputs/visual/ac_resubmission.
The figure uses saved counts and normalizations without recomputing metrics.
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import file_sha256

from utopia.utils.paths import project_root

import argparse
import csv
import hashlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


# Exact saved annual values, including counts that no longer have separate panels.
# A has 0 repeat submissions in every year. The fixed reviewer pool is 1202 in
# both conditions. A's annual multiplier is 1.000 in every row. Ratios retain
# their original three-decimal precision and are never recalculated here.
SOURCE_TABLE_SHA256 = "3ab0f715e5d786672818c1e7f06e98927913818a90e883dd0bdf48942ac69946"
ADMITTED_ANNUAL_CSV = """year,A_first,C_first,C_repeats,A_events,C_events,A_reviews,C_reviews,A_active,C_active,A_reviews_per_active,C_reviews_per_active,A_reviews_per_full,C_reviews_per_full,C_multiplier,A_multiplier
1,440,429,0,440,429,1320,1287,1202,1202,1.098,1.071,1.098,1.071,1.000,1.000
2,823,813,296,823,1109,2469,3327,1202,1202,2.054,2.768,2.054,2.768,1.364,1.000
3,736,744,771,736,1515,2208,4545,1202,1202,1.837,3.781,1.837,3.781,2.036,1.000
4,811,792,1051,811,1843,2433,5529,1202,1180,2.024,4.686,2.024,4.600,2.327,1.000
5,431,464,1255,431,1719,1293,5157,1202,1075,1.076,4.797,1.076,4.290,3.705,1.000
6,1104,974,994,1104,1968,3312,5904,1202,979,2.755,6.031,2.755,4.912,2.021,1.000
7,438,346,1111,438,1457,1314,4371,967,866,1.359,5.047,1.093,3.636,4.211,1.000
8,628,545,822,628,1367,1884,4101,967,782,1.948,5.244,1.567,3.412,2.508,1.000
9,619,501,824,619,1325,1857,3975,835,708,2.224,5.614,1.545,3.307,2.645,1.000
10,536,432,788,536,1220,1608,3660,835,646,1.926,5.666,1.338,3.045,2.824,1.000
"""


BLUE = "#2878B5"
CORAL = "#E64B35"
REPEAT_FILL = "#F9D1C8"
INK = "#242B33"
MUTED = "#616A75"
GRID = "#E6E9ED"


def main():
    repository = project_root(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        help="Optional CSV override. Defaults to the embedded reviewed values.")
    parser.add_argument("--output-dir", type=Path,
                        default=repository / "outputs" / "visual" / "ac_resubmission")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_bytes = args.data.read_bytes() if args.data is not None else ADMITTED_ANNUAL_CSV.encode("utf-8")
    rows = list(csv.DictReader(io.StringIO(data_bytes.decode("utf-8"))))
    assert [int(row["year"]) for row in rows] == list(range(1, 11))

    def values(column):
        return [float(row[column]) for row in rows]

    years = values("year")
    try:
        fm.findfont("Arial", fallback_to_default=False)
        font = "Arial"
    except ValueError:
        font = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": font,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.linewidth": 0.65,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.45,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    # 183 mm wide, with typography chosen for the manuscript's full-width figure.
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 118 / 25.4))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.12, top=0.875,
                        wspace=0.43, hspace=0.57)
    axes = axes.ravel()
    count_format = FuncFormatter(lambda value, pos: f"{value:,.0f}")
    titles = [
        "Submission composition",
        "Active reviewer pool",
        "Review-demand normalization",
        "Annual submission multiplier",
    ]
    labels = [
        "Submission events",
        "Year-end active reviewers",
        "All reviews / reviewer pool",
        "All events / first submissions",
    ]
    panel_manifest = []
    for index, (ax, title, ylabel) in enumerate(zip(axes, titles, labels)):
        ax.set_title(title, loc="left", pad=9)
        ax.text(-0.19, 1.073, chr(ord("a") + index), transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="baseline", color=INK)
        ax.set_ylabel(ylabel, labelpad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.55)
        ax.set_xlim(0.45, 11.8)
        ax.set_xticks(range(1, 11))
        ax.tick_params(axis="both", direction="out", pad=3)
        if index >= 2:
            ax.set_xlabel("Simulation year", labelpad=5)

    # Both total-height and first-submission bars use saved columns directly.
    # The exposed, hatched top of C is its repeat-submission component.
    ax = axes[0]
    offset = 0.19
    width = 0.32
    ax.bar([year - offset for year in years], values("A_events"), width,
           color=BLUE, edgecolor=BLUE, linewidth=0.55, zorder=3)
    ax.bar([year + offset for year in years], values("C_events"), width,
           color=REPEAT_FILL, edgecolor=CORAL, hatch="////",
           linewidth=0.55, zorder=3)
    ax.bar([year + offset for year in years], values("C_first"), width,
           color=CORAL, edgecolor=CORAL, linewidth=0.55, zorder=4)
    ax.set_ylim(0, 2200)
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.yaxis.set_major_formatter(count_format)
    panel_manifest.append({
        "panel": "a",
        "columns": ["A_events", "C_events", "C_first"],
        "meaning": "Paired bars A then C. Total bar heights are saved all-event counts. "
                   "Solid portions are first submissions, and C's hatched top is repeats. "
                   "A_first equals A_events in the admitted data.",
    })

    def pair(ax, a_column, c_column, ylim, ticks, formatter=None, decimals=None):
        a_values = values(a_column)
        c_values = values(c_column)
        ax.plot(years, a_values, color=BLUE, linewidth=1.45,
                linestyle="-", marker="s", markersize=3.8,
                markerfacecolor="white", markeredgewidth=0.85, zorder=4)
        ax.plot(years, c_values, color=CORAL, linewidth=1.55,
                marker="o", markersize=3.7, markeredgewidth=0.65,
                markeredgecolor="white", zorder=5)
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        if formatter is not None:
            ax.yaxis.set_major_formatter(formatter)
        for series in [a_values, c_values]:
            number = f"{series[-1]:,.0f}" if decimals is None else f"{series[-1]:.{decimals}f}"
            ax.annotate(number, (10, series[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7,
                        color=INK, va="center", annotation_clip=False)
        panel_manifest.append({"panel": chr(ord("a") + list(axes).index(ax)),
                               "columns": [a_column, c_column]})

    pair(axes[1], "A_active", "C_active", (0, 1300),
         [0, 400, 800, 1200], count_format)

    # Denominators share one axis. Early observations coincide exactly, so the
    # full-pool open markers are larger and drawn behind the active-pool markers.
    ax = axes[2]
    normalization_columns = []
    for denominator, style, marker_size, order in [
        ("full", (0, (3.5, 1.8)), 5.1, 3),
        ("active", "-", 3.3, 5),
    ]:
        for condition, color, marker in [("A", BLUE, "s"), ("C", CORAL, "o")]:
            column = f"{condition}_reviews_per_{denominator}"
            series = values(column)
            normalization_columns.append(column)
            ax.plot(years, series, color=color, linewidth=1.45,
                    linestyle=style, marker=marker, markersize=marker_size,
                    markerfacecolor="white" if denominator == "full" or condition == "A" else color,
                    markeredgecolor=color, markeredgewidth=0.8, zorder=order)
            ax.annotate(f"{series[-1]:.3f}", (10, series[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7,
                        color=INK, va="center", annotation_clip=False)
    ax.set_ylim(0, 7.8)
    ax.set_yticks([0, 2, 4, 6])
    ax.legend(handles=[
        Line2D([], [], color=MUTED, linewidth=1.4, linestyle="-",
               label="Active pool"),
        Line2D([], [], color=MUTED, linewidth=1.4, linestyle=(0, (3.5, 1.8)),
               label="Full pool"),
    ], loc="upper left", bbox_to_anchor=(0, 1.015), ncol=2,
       handlelength=2, columnspacing=1.1, fontsize=7, borderaxespad=0.3)
    panel_manifest.append({
        "panel": "c", "columns": normalization_columns,
        "meaning": "Saved all-review counts divided by active or full reviewer pool. "
                   "Color and marker encode condition, solid/dashed lines encode active/full pool. "
                   "Values are saved three-decimal ratios, never recalculated.",
    })
    pair(axes[3], "A_multiplier", "C_multiplier", (0, 4.6),
         [0, 1, 2, 3, 4], decimals=3)

    legend = [
        Line2D([], [], color=BLUE, linewidth=0,
               marker="s", markersize=4, markerfacecolor="white",
               markeredgewidth=0.85, label="A  No resubmission"),
        Line2D([], [], color=CORAL, linewidth=0, marker="o",
               markersize=4, markeredgecolor="white", markeredgewidth=0.65,
               label="C  Resubmission"),
        Patch(facecolor=REPEAT_FILL, edgecolor=CORAL, hatch="////",
              linewidth=0.55, label="Repeat component (a)"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.525, 0.99),
               ncol=3, handlelength=2, handletextpad=0.6, columnspacing=1.6)

    basename = args.output_dir / "fig_ac_resubmission"
    fig.savefig(basename.with_suffix(".pdf"), metadata={
        "Title": "A/C no-influx resubmission comparison",
        "Subject": "Previously admitted annual values. One ten-year world per condition.",
        "Keywords": "ScienceUtopia, resubmission, A/C, seed 42",
    })
    fig.savefig(basename.with_suffix(".svg"))
    fig.savefig(basename.with_suffix(".png"), dpi=300)
    manifest = {
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "data_source": "csv_override" if args.data is not None else "embedded",
        "source_table_sha256": SOURCE_TABLE_SHA256,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__)),
        "data_path": str(args.data.resolve()) if args.data is not None else None,
        "matplotlib_version": matplotlib.__version__,
        "font_family": font,
        "dimensions_mm": [183, 118],
        "colors": {"A": BLUE, "C": CORAL, "repeat_fill": REPEAT_FILL},
        "panels": panel_manifest,
        "rows": rows,
        "data_handling": "Saved values plotted directly. No ratio recalculation, smoothing, "
                         "interpolation beyond straight line segments, fits or intervals.",
    }
    write_json_file(manifest, basename.with_suffix('.manifest.json'), trailing_newline=True, streaming=False, indent=2)
    plt.close(fig)


if __name__ == "__main__":
    main()
