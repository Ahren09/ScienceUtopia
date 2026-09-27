"""Focused validation of paired estimands and original-study safeguards."""
import unittest

import numpy as np
import pandas as pd

from utopia.analysis.network_disclosure import archive_checks, complete_cells, interval_summary, paired_intervals, paper_contrasts, reviewer_sensitivity


def fixture(n=12):
    rows = []
    for paper in range(n):
        for j, stratum in enumerate(("d2", "d3", "far")):
            a = 2.0 + paper / (2 * n)
            # Distinct channel magnitudes and gradients expose estimand mixing.
            b = a + (0.20, 0.10, 0.05)[j]
            c = b + (0.12, 0.08, 0.04)[j]
            for condition, score in zip(("A", "B", "C"), (a, b, c)):
                rows.append({
                    "paper_id": f"p{paper:02}", "reviewer_id": f"r{paper % 4}_{j}",
                    "stratum": stratum, "condition": condition,
                    "source_label": "seed1", "success": True, "score_raw": score,
                })
    return pd.DataFrame(rows)


class NetworkAnalysisTests(unittest.TestCase):
    def test_same_population_and_gradient_contrasts(self):
        values = paper_contrasts(complete_cells(fixture(), expected_papers=12))
        self.assertAlmostEqual(values.identity_pooled.mean(), 0.35 / 3)
        self.assertAlmostEqual(values.proximity_pooled.mean(), 0.08)
        self.assertAlmostEqual(values.identity_d2_minus_far.mean(), 0.15)
        self.assertAlmostEqual(values.proximity_d2_minus_far.mean(), 0.08)
        self.assertAlmostEqual(values.identity_minus_proximity_d2_minus_far.mean(), 0.07)
        self.assertAlmostEqual(values.identity_minus_proximity_pooled.mean(), 0.35 / 3 - 0.08)
        self.assertTrue(np.allclose(
            values.total_pooled, values.identity_pooled + values.proximity_pooled
        ))

    def test_preserves_shared_condition_covariance(self):
        raw = fixture()
        # A=C, with variable B: identity-proximity must be twice identity.
        for paper in raw.paper_id.unique():
            for stratum in ("d2", "d3", "far"):
                mask = raw.paper_id.eq(paper) & raw.stratum.eq(stratum)
                a = raw.loc[mask & raw.condition.eq("A"), "score_raw"].iloc[0]
                raw.loc[mask & raw.condition.eq("C"), "score_raw"] = a
                raw.loc[mask & raw.condition.eq("B"), "score_raw"] += int(paper[1:]) * .01
        values = paper_contrasts(complete_cells(raw, expected_papers=12))
        intervals = paired_intervals(values, 1000, 7)
        self.assertTrue(np.allclose(
            intervals["identity_minus_proximity_pooled"]["ci95"],
            np.array(intervals["identity_pooled"]["ci95"]) * 2,
        ))

    def test_rejects_wrong_study_size(self):
        with self.assertRaisesRegex(ValueError, "200"):
            complete_cells(fixture(120))

    def test_rejects_duplicate_or_missing_cells(self):
        raw = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            complete_cells(pd.concat([raw, raw.iloc[:1]]), 12)
        with self.assertRaises(ValueError):
            complete_cells(raw.iloc[1:], 12)

    def test_rejects_reviewer_changes_and_failed_or_invalid_scores(self):
        for field, value in [
            ("reviewer_id", "different"), ("success", False),
            ("score_raw", np.nan), ("score_raw", 6.0),
        ]:
            raw = fixture()
            raw.loc[0, field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                complete_cells(raw, 12)

    def test_preserves_reviewer_and_crossed_bootstrap_estimands(self):
        cells = complete_cells(fixture(), 12)
        res = reviewer_sensitivity(cells, 1000, 5)
        for mode in res.values():
            self.assertAlmostEqual(
                mode["contrasts"]["identity_minus_proximity_d2_minus_far"]["estimate"],
                .07,
            )
            np.testing.assert_allclose(
                mode["contrasts"]["identity_minus_proximity_d2_minus_far"]["ci95"],
                [.07, .07],
                atol=1e-14,
            )

    def test_interval_crossing_is_not_inside_equivalence_band(self):
        result = interval_summary(.0285, np.linspace(-.065, .13, 10001))
        self.assertFalse(result["bounds"]["0.12"]["ci95_strictly_inside"])
        values = pd.DataFrame({
            "identity_pooled": [.055], "proximity_d2_minus_far": [.0285],
        })
        archive = {
            "primary": {"primary_d2_minus_far": {
                "estimate": .0285, "ci_lo": -.065, "ci_hi": .1205, "n": 200,
            }},
            "secondary": {"S4_identity_B_minus_A": {"pooled": {"estimate": .055}}},
        }
        checks = archive_checks(archive, values)
        self.assertFalse(checks["original_primary_ci95_inside_0.12"])
        self.assertAlmostEqual(checks["upper_endpoint_minus_0.12"], .0005)
        values.loc[0, "identity_pooled"] = .0767
        with self.assertRaisesRegex(ValueError, "disagrees"):
            archive_checks(archive, values)

    def test_bootstrap_is_repeatable(self):
        values = paper_contrasts(complete_cells(fixture(), 12))
        self.assertEqual(paired_intervals(values, 100, 3), paired_intervals(values, 100, 3))


if __name__ == "__main__":
    unittest.main()
