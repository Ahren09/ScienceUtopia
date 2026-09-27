"""Read-only reanalysis of the original 200-paper network disclosure study.

Run from the code repository:
    python -m utopia.analysis.network_disclosure --manifest /path/to/corpus_manifest.json

The analysis prints JSON to stdout and never edits inputs or starts inference.
New contrasts and interval-based equivalence checks are exploratory. The
independent 120-paper identity replication is reported separately, never pooled.
"""
from __future__ import annotations

from utopia.utils.data_utils import file_sha256 as sha256

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = ("d2", "d3", "far")
CONDITIONS = ("A", "B", "C")
ORIGINAL_SOURCES = {
    "network_review_bias_source_seed7101",
    "network_review_bias_source_seed7102",
}
REVIEWS = "outputs/docs/network_review_bias_main_qwen3_32b_p300_seed42/reviews.parquet"
ARCHIVE = "outputs/docs/network_review_bias_final/results.json"
CONFIRMATORY = "outputs/docs/network_review_bias_confirmatory_final/results.json"





def complete_cells(reviews: pd.DataFrame, expected_papers: int = 200) -> pd.DataFrame:
    """Require the complete original design, with no silent deduplication."""
    required = {
        "paper_id", "reviewer_id", "stratum", "condition",
        "source_label", "success", "score_raw",
    }
    if missing := required - set(reviews.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if reviews[list(required)].isna().any().any():
        raise ValueError("Missing required cells or metadata")
    if not reviews.success.isin([True, False]).all() or not reviews.success.all():
        raise ValueError("Failed reviews require an explicit missingness analysis")
    if set(reviews.stratum) != set(STRATA) or set(reviews.condition) != set(CONDITIONS):
        raise ValueError("Expected exactly d2/d3/far and A/B/C")
    if reviews.duplicated(["paper_id", "stratum", "condition"]).any():
        raise ValueError("Duplicate successful paper/stratum/condition")
    scores = pd.to_numeric(reviews.score_raw, errors="coerce")
    if not np.isfinite(scores).all() or not scores.between(1, 5).all():
        raise ValueError("Scores must be finite and on the 1–5 scale")
    by_cell = reviews.groupby(["paper_id", "stratum"])
    if not by_cell.reviewer_id.nunique().eq(1).all():
        raise ValueError("Reviewer changed across disclosure conditions")
    if not reviews.groupby("paper_id").source_label.nunique().eq(1).all():
        raise ValueError("Paper IDs collide across source worlds")
    if not reviews.groupby("paper_id").reviewer_id.nunique().eq(3).all():
        raise ValueError("Each paper must have three distinct matched reviewers")
    n = reviews.paper_id.nunique()
    if n != expected_papers or len(reviews) != expected_papers * 9:
        raise ValueError(f"Expected {expected_papers} complete papers, got {n}")
    wide = reviews.assign(score_raw=scores).pivot(
        index=["paper_id", "stratum"], columns="condition", values="score_raw"
    )
    full_index = pd.MultiIndex.from_product(
        [sorted(reviews.paper_id.unique()), STRATA], names=["paper_id", "stratum"]
    )
    wide = wide.reindex(full_index)
    if wide[list(CONDITIONS)].isna().any().any():
        raise ValueError("Incomplete matched disclosure triplet")
    meta = by_cell[["reviewer_id", "source_label"]].first().reindex(full_index)
    cells = wide.join(meta).reset_index()
    cells["identity"] = cells.B - cells.A
    cells["proximity"] = cells.C - cells.B
    cells["total"] = cells.C - cells.A
    cells["identity_minus_proximity"] = 2 * cells.B - cells.A - cells.C
    return cells


def paper_contrasts(cells: pd.DataFrame) -> pd.DataFrame:
    """Compare channels using the same papers and the same stratum weights."""
    out = pd.DataFrame(index=sorted(cells.paper_id.unique()))
    for channel in ("identity", "proximity", "total", "identity_minus_proximity"):
        wide = cells.pivot(index="paper_id", columns="stratum", values=channel)
        out[f"{channel}_pooled"] = wide[list(STRATA)].mean(axis=1)
        out[f"{channel}_d2_minus_far"] = wide.d2 - wide.far
        for stratum in STRATA:
            out[f"{channel}_{stratum}"] = wide[stratum]
    return out


def interval_summary(estimate: float, draws: np.ndarray) -> dict:
    lo95, lo90, hi90, hi95 = np.quantile(draws, [0.025, 0.05, 0.95, 0.975])
    return {
        "estimate": float(estimate),
        "ci95": [float(lo95), float(hi95)],
        "ci90": [float(lo90), float(hi90)],
        "positive_by_ci95": bool(lo95 > 0),
        "bounds": {
            str(bound): {
                "ci95_strictly_inside": bool(lo95 > -bound and hi95 < bound),
                "ci90_strictly_inside_exploratory": bool(lo90 > -bound and hi90 < bound),
                "one_sided_95_lower_exceeds_bound": bool(lo90 > bound),
            }
            for bound in (0.10, 0.12)
        },
    }


def paired_intervals(values: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Resample whole papers, preserving dependence across strata and channels."""
    rng = np.random.default_rng(seed)
    matrix = values.to_numpy(float)
    draws = np.empty((n_boot, matrix.shape[1]))
    for start in range(0, n_boot, 500):
        count = min(500, n_boot - start)
        indices = rng.integers(len(matrix), size=(count, len(matrix)))
        draws[start:start + count] = matrix[indices].mean(axis=1)
    return {
        name: interval_summary(matrix[:, j].mean(), draws[:, j])
        for j, name in enumerate(values.columns)
    }


def reviewer_sensitivity(cells: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Reviewer and crossed paper/reviewer bootstrap of comparable contrasts.

    Fixed equal stratum weights are preserved in every bootstrap draw. A/B/C
    differences are formed before resampling, preserving the shared B score.
    The crossed bootstrap is a sensitivity, not an assertion of independent
    worlds or an exact randomization distribution.
    """
    pcode, papers = pd.factorize(cells.paper_id)
    rcode, reviewers = pd.factorize(cells.source_label + "|" + cells.reviewer_id)
    rng = np.random.default_rng(seed)
    effects = cells[["identity", "proximity"]].to_numpy(float)
    output = {}
    for mode in ("reviewer", "paper_and_reviewer"):
        retained = []
        for start in range(0, n_boot, 500):
            count = min(500, n_boot - start)
            weights = rng.multinomial(
                len(reviewers), np.full(len(reviewers), 1 / len(reviewers)), size=count
            )[:, rcode].astype(float)
            if mode == "paper_and_reviewer":
                weights *= rng.multinomial(
                    len(papers), np.full(len(papers), 1 / len(papers)), size=count
                )[:, pcode]
            stratum_means = []
            valid = np.ones(count, dtype=bool)
            for stratum in STRATA:
                mask = cells.stratum.eq(stratum).to_numpy()
                w = weights[:, mask]
                denom = w.sum(axis=1)
                valid &= denom > 0
                stratum_means.append(
                    (w @ effects[mask]) / np.maximum(denom[:, None], 1)
                )
            means = np.stack(stratum_means, axis=1)
            pooled = means.mean(axis=1)
            gradient = means[:, 0] - means[:, 2]
            retained.append(np.column_stack([
                pooled[:, 0], pooled[:, 1], pooled[:, 0] - pooled[:, 1],
                gradient[:, 0], gradient[:, 1], gradient[:, 0] - gradient[:, 1],
            ])[valid])
        draws = np.concatenate(retained)
        names = [
            "identity_pooled", "proximity_pooled", "identity_minus_proximity_pooled",
            "identity_d2_minus_far", "proximity_d2_minus_far",
            "identity_minus_proximity_d2_minus_far",
        ]
        points = paper_contrasts(cells)[names].mean()
        output[mode] = {
            "valid_draws": len(draws),
            "contrasts": {
                name: interval_summary(points[name], draws[:, j])
                for j, name in enumerate(names)
            },
        }
    return output


def archive_checks(archive: dict, contrasts: pd.DataFrame) -> dict:
    primary = archive["primary"]["primary_d2_minus_far"]
    identity = archive["secondary"]["S4_identity_B_minus_A"]["pooled"]
    actual = contrasts.mean()
    for name, archived in (
        ("proximity_d2_minus_far", primary["estimate"]),
        ("identity_pooled", identity["estimate"]),
    ):
        if not np.isclose(actual[name], archived, rtol=0, atol=1e-12):
            raise ValueError(f"Replay estimate disagrees with archive: {name}")
    if primary.get("n_papers", primary.get("n")) != 200:
        raise ValueError("Archive is not the original 200-paper study")
    return {
        "original_primary_ci95": [primary["ci_lo"], primary["ci_hi"]],
        "original_primary_ci95_inside_0.12": bool(
            primary["ci_lo"] > -0.12 and primary["ci_hi"] < 0.12
        ),
        "upper_endpoint_minus_0.12": primary["ci_hi"] - 0.12,
        "note": "0.1205 rounds to +0.121 and exceeds 0.12. A 95% interval "
                "crossing the band does not itself decide a 5% TOST using a 90% CI.",
    }




def main(argv=None):
    parser = argparse.ArgumentParser(description="Numerical network disclosure contrasts for new replay data.")
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.n_boot < 1000:
        parser.error('--n-boot must be at least 1000')
    reviews = pd.read_parquet(args.reviews)
    cells = complete_cells(reviews, expected_papers=reviews.paper_id.nunique())
    contrasts = paper_contrasts(cells)
    result = {
        'status': 'complete', 'n_papers': len(contrasts), 'n_reviews': len(reviews),
        'n_boot': args.n_boot, 'seed': args.seed,
        'reviews_sha256': sha256(args.reviews),
        'paper_bootstrap': paired_intervals(contrasts, args.n_boot, args.seed),
        'reviewer_sensitivity': reviewer_sensitivity(cells, args.n_boot, args.seed + 1),
        'interpretation': 'Sequential disclosure effects conditional on supplied source worlds.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
