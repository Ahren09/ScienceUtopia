"""Specification-derived tests for the post hoc reanalysis. No LLMs or GPUs."""
import unittest

import numpy as np
import pandas as pd

from utopia.analysis.research_strategy import career_researcher_summaries, fixed_window_yield, prepare_papers, strategy_manipulation, within_slope


class StrategyReanalysisTests(unittest.TestCase):
    def test_career_means_weight_researchers_and_survival_retains_inactive_founders(self):
        # A prolific author must not dominate the acceptance mean. A founder with
        # no submissions has an undefined acceptance fraction but remains in survival.
        ay = pd.DataFrame({
            "agent_id": ["a", "a", "b", "b", "c", "c"],
            "year": [1, 2, 1, 2, 1, 2], "strategy": ["explorer"] * 6,
            "num_papers": [10, 0, 1, 0, 0, 0],
            "num_accepted": [9, 0, 0, 0, 0, 0],
            "is_active": [True, False, True, True, True, True],
        })
        outcomes, survival = career_researcher_summaries(ay)
        self.assertAlmostEqual(outcomes.iloc[0]["mean_researcher_acceptance_rate"], 0.45)
        self.assertEqual(outcomes.iloc[0]["n_researchers_with_submissions"], 2)
        self.assertEqual(outcomes.iloc[0]["n_founders"], 3)
        yearly = survival.set_index("year")
        self.assertEqual(yearly.loc[1, "active_share_end_year"], 1.0)
        self.assertAlmostEqual(yearly.loc[2, "active_share_end_year"], 2 / 3)

    def test_first_choice_is_not_a_failed_switch_and_nonyears_are_not_choices(self):
        ay = pd.DataFrame({
            "agent_id": ["a"] * 4, "year": [1, 2, 3, 4],
            "strategy": ["explorer"] * 4, "is_active": [True] * 4,
            "direction_topic": ["x", None, "y", None],
            "topic_switched": [False, None, True, None],
            "direction_distance": [0.8, np.nan, 0.9, np.nan],
        })
        summary, _ = strategy_manipulation(ay)
        row = summary.iloc[0]
        self.assertEqual(row["stored_switch_rate_author_mean_including_first"], 0.5)
        self.assertEqual(row["repeat_choice_switch_probability_event_weighted"], 1.0)
        self.assertEqual(row["conditional_direction_distance_event_weighted"], 0.9)
        self.assertEqual(row["switches_per_active_at_year_start_year"], 0.25)

    def test_resubmission_keeps_originating_topic_and_distance_above_one(self):
        ay = pd.DataFrame({
            "agent_id": ["a", "a"], "year": [1, 2], "direction_topic": ["old", "new"],
        })
        paper = pd.DataFrame({
            "paper_id": ["p", "p"], "author_id": ["a", "a"],
            "strategy": ["explorer"] * 2, "year": [1, 2], "novelty_score": [1.1, 1.1],
        })
        result = prepare_papers(paper, ay)
        self.assertEqual(result.project_topic.tolist(), ["old", "old"])
        self.assertTrue(result.distance_bin.notna().all())

    @staticmethod
    def yield_fixture():
        # p: rejected at 1, accepted at 2, deadline 4 => publication age 2, not 3.
        # late: accepted at 5, deadline 4 => zero. never => zero. censored omitted.
        paper = pd.DataFrame({
            "paper_id": ["p", "p", "late", "late", "never", "censored"],
            "year": [1, 2, 1, 5, 1, 8],
            "accepted": [False, True, False, True, False, False],
        })
        age = pd.DataFrame({
            "paper_id": ["p", "p", "late"],
            "publication_year": [2, 2, 5], "age": [0, 2, 0],
            "observation_year": [2, 4, 5], "citations": [0, 7, 0],
        })
        return paper, age

    def test_submission_anchored_deadline_and_delayed_publication(self):
        paper, age = self.yield_fixture()
        result = fixed_window_yield(paper, age, end_year=10).set_index("paper_id")
        self.assertEqual(set(result.index), {"p", "late", "never"})
        self.assertEqual(result.loc["p", "publication_mediated_citation_yield"], 7)
        self.assertEqual(result.loc["late", "publication_mediated_citation_yield"], 0)
        self.assertEqual(result.loc["never", "publication_mediated_citation_yield"], 0)

    def test_missing_published_outcome_is_not_zero_imputed(self):
        paper, age = self.yield_fixture()
        with self.assertRaisesRegex(ValueError, "Missing citation observation"):
            fixed_window_yield(paper, age.loc[age.age.eq(0)], end_year=10)

    def test_duplicate_publication_age_is_rejected(self):
        paper, age = self.yield_fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fixed_window_yield(paper, pd.concat([age, age.iloc[[0]]]), end_year=10)

    def test_fixed_effect_slope_removes_group_mean_confounding(self):
        # Within every group y = 2*x + constant, while between groups confounded.
        frame = pd.DataFrame({
            "novelty_score": [0, 1, 10, 11],
            "outcome": [100, 102, 0, 2],
            "author_id": ["a", "a", "b", "b"],
            "paper_id": ["p", "q", "r", "s"],
        })
        result = within_slope(frame, "outcome", [["author_id"]])
        self.assertAlmostEqual(result["slope_per_0_1_distance"], 0.2)

    def test_two_fixed_effects_match_explicit_least_squares(self):
        frame = pd.DataFrame({
            "novelty_score": [0.1, 0.7, 0.9, 0.2, 0.6, 0.3],
            "outcome": [4, 2, 5, 8, 9, 3],
            "author_id": ["a", "a", "b", "b", "c", "c"],
            "context": ["u", "v", "u", "v", "u", "v"],
            "paper_id": list("abcdef"),
        })
        controls = pd.get_dummies(frame[["author_id", "context"]], dtype=float)
        matrix = np.column_stack([frame.novelty_score, controls])
        expected = np.linalg.lstsq(matrix, frame.outcome, rcond=None)[0][0]
        result = within_slope(frame, "outcome", [["author_id"], ["context"]])
        self.assertAlmostEqual(result["slope_per_0_1_distance"], 0.1 * expected)


if __name__ == "__main__":
    unittest.main()
