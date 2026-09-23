#!/usr/bin/env python3
"""Render two existing-result figures in green without changing source data.

Run: python -B src/render_james_existing_figures.py

Generator citations (the original functions are executed, not reimplemented):
  utopia/visual/plot_ecosystem_dynamics.py: main, seed_band, end_label
  utopia/visual/plot_influx_factorial.py: main, volume_panel,
      composition_panel, burden_panel

Original save operations are intercepted in memory. Only the two *_green.pdf
figures, their PDF-rendered PNGs, and a dedicated metadata/QA JSON are written.
The original influx multiplier figure is not generated. All trajectories,
bar geometry, seed ranges, and review-score SD bands are preserved exactly.
No simulations, model calls, network access, or GPU work occur.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import fitz
import numpy as np
import pandas as pd

GREEN = "#008000"
WHITE = "#ffffff"
PALE_GREEN = "#c4e4c4"
ARCHIVE_ROOT = "/nethome/yjin328/Workspace/PeerReview/ScienceUtopia"
OUT = ROOT / "outputs/visual/james_revision_existing"
METADATA = ROOT / "outputs/docs/james_revision_existing/existing_figures_metadata.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_record(path):
    relative = str(path.relative_to(ROOT))
    return {
        "path": relative,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "archive_path": f"{ARCHIVE_ROOT}/{relative}",
    }


def load_plot(name):
    path = ROOT / f"utopia/visual/plot_{name}.py"
    spec = importlib.util.spec_from_file_location(f"original_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_figure(module, arguments):
    """Run the untouched plot generator while suppressing all original exports."""
    captured = []

    def capture(figure, *_args, **_kwargs):
        if all(figure is not previous for previous in captured):
            captured.append(figure)

    with (
        patch.object(sys, "argv", [module.__file__, *arguments]),
        patch.object(Figure, "savefig", capture),
        patch.object(plt, "close"),
        redirect_stdout(io.StringIO()),
    ):
        if hasattr(module, "multiplier_figure"):
            with patch.object(module, "multiplier_figure", lambda *_a, **_k: None):
                module.main()
        else:
            module.main()
    if len(captured) != 1:
        raise AssertionError(f"Expected one original figure, got {len(captured)}")
    return captured[0]


def numeric_geometry(figure):
    """Digest every data line, ribbon polygon, and bar, independent of styling."""
    digest = hashlib.sha256()
    for ax in figure.axes:
        for line in ax.lines:
            digest.update(np.asarray(line.get_xydata(), dtype="<f8").tobytes())
        for collection in ax.collections:
            for path in collection.get_paths():
                digest.update(np.asarray(path.vertices, dtype="<f8").tobytes())
        for bar in ax.patches:
            digest.update(np.asarray(bar.get_path().vertices, dtype="<f8").tobytes())
            digest.update(
                np.asarray(bar.get_patch_transform().get_matrix(), dtype="<f8").tobytes()
            )
    return digest.hexdigest()


def make_green(figure):
    """Recolor every visible artist, retaining the original statistical geometry."""
    figure.patch.set_facecolor(WHITE)
    figure.patch.set_edgecolor(WHITE)
    for ax in figure.axes:
        ax.set_facecolor(WHITE)
    for artist in figure.findobj():
        if isinstance(artist, Text):
            artist.set_color(GREEN)
        elif isinstance(artist, Line2D):
            artist.set_color(GREEN)
            artist.set_markeredgecolor(GREEN)
            artist.set_markerfacecolor(WHITE)
        elif isinstance(artist, Collection):
            artist.set_facecolor(GREEN)
            artist.set_edgecolor(GREEN)
        elif isinstance(artist, Patch):
            if artist is figure.patch or any(artist is ax.patch for ax in figure.axes):
                continue
            artist.set_edgecolor(GREEN)
            if artist.get_facecolor()[3] > 0:
                artist.set_facecolor(GREEN)
    for ax in figure.axes:
        ax.tick_params(axis="both", colors=GREEN)
        for spine in ax.spines.values():
            spine.set_color(GREEN)
        for grid in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            grid.set_color(GREEN)
            grid.set_alpha(0.16)
    figure.canvas.draw()


def assert_array(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def verify_ecosystem(figure, trajectories, scores):
    df, review = pd.read_csv(trajectories), pd.read_csv(scores)
    summary = {}
    for panel, index, field, data in [
        (0, 0, "funding_gini", df),
        (0, 1, "citation_gini", df),
        (1, 0, "topic_entropy_normalized", df),
        (2, 0, "num_active_researchers", df),
        (3, 1, "mean", review),
    ]:
        grouped = data.dropna(subset=[field]).groupby("year")[field].agg(
            ["mean", "min", "max"]
        )
        line = figure.axes[panel].lines[index]
        assert_array(line.get_xdata(), grouped.index)
        assert_array(line.get_ydata(), grouped["mean"])
        summary[field] = {
            "years": grouped.index.tolist(),
            "equal_world_mean": grouped["mean"].tolist(),
            "world_min": grouped["min"].tolist(),
            "world_max": grouped["max"].tolist(),
        }
    assert_array(figure.axes[3].lines[0].get_ydata(), [3, 3])
    yearly = review.groupby("year")[["mean", "sd"]].mean()
    summary["review_sd_band"] = {
        "lower": (yearly["mean"] - yearly["sd"]).tolist(),
        "upper": (yearly["mean"] + yearly["sd"]).tolist(),
        "definition": "Equal-world mean review score +/- equal-world mean within-world SD.",
    }
    assert sorted(df.seed.unique().tolist()) == [1001, 1002, 1003]
    summary["seeds"] = [1001, 1002, 1003]
    summary["bands"] = "Original world min-max ribbons and review SD band; no confidence intervals."
    return summary


def verify_influx(figure, paths):
    stats = {cell: json.loads(path.read_text()) for cell, path in paths.items()}
    years = [str(y) for y in range(1, 11)]
    summary = {"seed": 42, "independent_worlds_per_cell": 1, "cells": {}}
    for i, (cell, data) in enumerate(stats.items()):
        volume = [data["submissions_per_year"][year] for year in years]
        burden = [data["reviews_per_active_reviewer_by_year"][year] for year in years]
        assert_array(figure.axes[0].lines[i].get_ydata(), volume)
        assert_array(figure.axes[2].lines[i].get_ydata(), burden)
        # Verify the relabeled denominator against the archived counts every year.
        assigned = [data["reviews_assigned_by_year"][year] for year in years]
        active = [data["active_researchers_by_year"][year] for year in years]
        assert_array(burden, np.asarray(assigned) / np.asarray(active))
        summary["cells"][cell] = {
            "submission_events": volume,
            "reviews_per_active_researcher": burden,
            "reviews_assigned": assigned,
            "active_researchers": active,
        }
    composition = stats["D"]["attempt_composition_by_year"]
    first = [composition[y]["first"] for y in years]
    resub = [composition[y]["resub_1st"] + composition[y]["resub_2plus"] for y in years]
    bars = figure.axes[1].patches
    assert_array([bar.get_height() for bar in bars[:10]], first)
    assert_array([bar.get_height() for bar in bars[10:]], resub)
    assert_array([bar.get_y() for bar in bars[10:]], first)
    assert_array(np.asarray(first) + resub, summary["cells"]["D"]["submission_events"])
    summary["combined_world_composition"] = {"first": first, "resubmissions": resub}
    return summary


def green_or_white(rgb):
    if rgb is None:
        return True
    r, g, b = rgb[:3]
    return (min(r, g, b) > 0.995) or (g > r + 0.01 and g > b + 0.01)


def export_and_audit(figure, name, required, forbidden, out_dir):
    pdf, png = out_dir / f"{name}.pdf", out_dir / f"{name}.png"
    figure.savefig(
        pdf, dpi=300, bbox_inches="tight", facecolor=WHITE,
        metadata={"Creator": "src/render_james_existing_figures.py; original plotting functions cited in metadata JSON"},
    )
    with fitz.open(pdf) as document:
        page = document[0]
        text = page.get_text()
        for value in required:
            if value not in text:
                raise AssertionError(f"Missing PDF text {value!r} in {pdf.name}")
        for value in forbidden:
            if value in text:
                raise AssertionError(f"Stale PDF text {value!r} in {pdf.name}")
        spans = [
            span for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", []) for span in line["spans"]
            if span["text"].strip()
        ]
        assert all(span["color"] == 0x008000 for span in spans)
        colors = []
        for drawing in page.get_drawings():
            for key in ("color", "fill"):
                color = drawing.get(key)
                assert green_or_white(color), (pdf.name, key, color)
                if color is not None:
                    colors.append(tuple(round(v, 6) for v in color))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
        pixmap.save(png)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(-1, 3).astype(int)
        # Antialiasing may differ by one channel level; no gray/blue/red marks.
        nonwhite = np.min(pixels, axis=1) < 250
        bad = nonwhite & (
            (pixels[:, 1] + 1 < pixels[:, 0]) |
            (pixels[:, 1] + 1 < pixels[:, 2]) |
            ((np.max(pixels, axis=1) - np.min(pixels, axis=1)) <= 1)
        )
        assert not np.any(bad), f"{int(bad.sum())} non-green raster pixels"
        audit = {
            "pdf": str(pdf.relative_to(ROOT)),
            "png": str(png.relative_to(ROOT)),
            "pdf_sha256": sha256(pdf),
            "png_sha256": sha256(png),
            "embedded_text": text,
            "text_colors": ["#008000"],
            "vector_colors_rgb": sorted(set(colors)),
            "non_green_raster_pixels": int(bad.sum()),
            "minimum_font_pt": min(span["size"] for span in spans),
            "page_size_pt": [page.rect.width, page.rect.height],
            "png_render_dpi": 216,
        }
    plt.close(figure)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = ROOT / "outputs/docs/explore_e4a_analysis/ecosystem_trajectories.csv"
    scores = ROOT / "outputs/docs/explore_e4a_analysis/review_score_trajectories.csv"
    ecosystem, influx = load_plot("ecosystem_dynamics"), load_plot("influx_factorial")
    influx_paths = {cell: ROOT / relative for cell, relative in influx.DEFAULT_STATS.items()}
    sources = [
        trajectories, scores, *influx_paths.values(),
        Path(ecosystem.__file__), Path(influx.__file__),
        ROOT / "utopia/utils/visual_utils.py",
    ]
    before = {str(path): sha256(path) for path in sources}
    metadata = {
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
            "citations": [
                "utopia/visual/plot_ecosystem_dynamics.py: main, seed_band, end_label",
                "utopia/visual/plot_influx_factorial.py: main, volume_panel, composition_panel, burden_panel",
            ],
        },
        "inputs_and_original_generators": [source_record(path) for path in sources],
        "scope": "CPU-only rerender of existing archived results; no new experiments.",
        "figures": {},
    }

    eco = original_figure(ecosystem, [
        "--trajectories", str(trajectories), "--review_scores", str(scores),
        "--out_dir", str(args.out_dir),
    ])
    original_geometry = numeric_geometry(eco)
    eco_values = verify_ecosystem(eco, trajectories, scores)
    label = [text for text in eco.axes[3].texts if text.get_text() == "accept threshold"]
    assert len(label) == 1
    label[0].set_text("Rubric score 3")
    resource_label = [text for text in eco.axes[0].texts if text.get_text() == "funding"]
    assert len(resource_label) == 1
    resource_label[0].set_text("resources")
    eco.axes[2].set_title("Active population declines", pad=2)
    eco.axes[3].set_title("Review score distribution", pad=2)
    make_green(eco)
    # Resource and citation Gini remain distinguishable without hue differences.
    eco.axes[0].lines[0].set(linestyle="-", marker="o")
    eco.axes[0].lines[1].set(linestyle="--", marker="s")
    assert numeric_geometry(eco) == original_geometry
    audit = export_and_audit(
        eco, "fig_ecosystem_dynamics_green",
        ["Rubric score 3", "resources", "Review score distribution",
         "0.36", "0.54", "0.85", "2531", "2.40"],
        ["accept threshold", "funding", "Population attrites", "Scores stay borderline"], args.out_dir,
    )
    metadata["figures"]["ecosystem"] = {
        "qa": audit, "verified_source_values": eco_values,
        "original_and_revised_numeric_geometry_sha256": original_geometry,
        "changes": ["accept threshold -> Rubric score 3", "funding -> resources",
                    "Descriptive population and review-score panel titles.",
                    "Green styling; resources solid/circle, citations dashed/square."],
    }

    argv = ["--out_dir", str(args.out_dir)]
    for cell, path in influx_paths.items():
        argv.extend([f"--stats_{cell}", str(path)])
    inf = original_figure(influx, argv)
    original_geometry = numeric_geometry(inf)
    inf_values = verify_influx(inf, influx_paths)
    inf.axes[2].set_ylabel("Reviews per active researcher")
    inf.axes[2].set_title("Review demand in the matched factorial", pad=2)
    make_green(inf)
    styles = [
        {"linestyle": "--", "marker": "o"},
        {"linestyle": "-", "marker": "o"},
        {"linestyle": "--", "marker": "s"},
        {"linestyle": "-", "marker": "s"},
    ]
    for ax in [inf.axes[0], inf.axes[2]]:
        for line, style in zip(ax.lines, styles):
            line.set(**style, markersize=2.6, markeredgewidth=0.65)
    for i, bar in enumerate(inf.axes[1].patches):
        bar.set_facecolor(PALE_GREEN if i < 10 else GREEN)
        bar.set_edgecolor(GREEN)
        bar.set_hatch("///" if i < 10 else None)
    # Rebuild legends so their handles reflect the new marker/hatch encodings.
    for legend in list(inf.legends):
        legend.remove()
    handles, labels = inf.axes[0].get_legend_handles_labels()
    inf.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, 1.03), labelcolor=GREEN)
    inf.axes[1].legend(fontsize=6, frameon=False, loc="upper left", labelcolor=GREEN)
    assert numeric_geometry(inf) == original_geometry
    audit = export_and_audit(
        inf, "fig_influx_factorial_green",
        ["Reviews per active researcher", "Review demand in the matched factorial",
         "193", "364", "489", "867", "2.25", "2.07", "6.64", "5.48"],
        ["Reviews per active reviewer", "Recycling, not growth, drives burden"],
        args.out_dir,
    )
    metadata["figures"]["influx"] = {
        "qa": audit, "verified_source_values": inf_values,
        "original_and_revised_numeric_geometry_sha256": original_geometry,
        "changes": [
            "Reviews per active reviewer -> Reviews per active researcher",
            "Recycling, not growth, drives burden -> Review demand in the matched factorial",
            "Green styling; dashed/solid encodes influx off/on; circle/square encodes recycling off/on.",
            "First submissions pale-green hatched; resubmissions solid green.",
        ],
    }
    assert before == {str(path): sha256(path) for path in sources}
    metadata["qa"] = {
        "all_inputs_and_original_generators_unchanged": True,
        "all_numeric_geometry_identical_to_original_plotters": True,
        "every_plotted_year_verified_against_inputs": True,
        "influx_denominator_verified_against_active_researchers_all_cells_all_years": True,
        "visual_review": "Inspect the PDF-rendered PNGs before delivery.",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Ready: {args.out_dir / 'fig_ecosystem_dynamics_green.pdf'}")
    print(f"Ready: {args.out_dir / 'fig_influx_factorial_green.pdf'}")
    print(f"Metadata and QA: {args.metadata}")


if __name__ == "__main__":
    main()
