"""Standard-library CPU tests. No simulator import, model, network or file writes.

Run: python -B tests/experiments/test_switching_policy.py -v
The seed function and 53-topic roster are read from the actual source in this
separate worktree. Geometry is supplied explicitly: these tests do not certify
the driver's embedding coverage or separation in future evolved states.
"""

from utopia.runtime.historical import SWITCH_GATE_SEED_NAMESPACE

from utopia.utils.paths import project_root

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
import types
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import utopia.agents.switching_policy as policy

ROOT = project_root(__file__)


def native_seed_function():
    path = ROOT / "utopia/utils/seeding.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "derive_seed")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["derive_seed"]


NATIVE_SEED = native_seed_function()


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.topics = list(policy.CANONICAL_TOPICS)
        self.previous = "artificial_intelligence"
        self.eligible = [topic for topic in self.topics if topic != self.previous]
        self.distances = {topic: i / 26 for i, topic in enumerate(self.eligible)}

    def decision(self, cell="HN", **overrides):
        arguments = dict(
            cell=cell, agent_id="institution_0000_researcher_0", year=2,
            previous_topic=self.previous, canonical_topics53=self.topics,
            distance_map52=self.distances, seed=42, derive_seed_fn=NATIVE_SEED,
        )
        arguments.update(overrides)
        return policy.build_decision(**arguments)

    def test_canonical_roster_matches_actual_53_direction_objects(self):
        path = ROOT / "utopia/agents/research_direction.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        node = next(node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "AVAILABLE_DIRECTIONS"
                            for t in node.targets))
        observed = [ast.literal_eval(next(k.value for k in direction.keywords if k.arg == "topic"))
                    for direction in node.value.elts]
        self.assertEqual(len(observed), 53)
        self.assertEqual(tuple(sorted(observed)), policy.CANONICAL_TOPICS)

    def test_native_seed_namespace_year_and_agent_are_exact(self):
        calls = []
        def tracked(*parts):
            calls.append(parts)
            return NATIVE_SEED(*parts)
        decision = self.decision(derive_seed_fn=tracked)
        key = (42, SWITCH_GATE_SEED_NAMESPACE, "institution_0000_researcher_0", 2)
        self.assertEqual(calls, [key])
        self.assertEqual(decision["gate_seed"], NATIVE_SEED(*key))
        self.assertEqual(decision["u"], random.Random(NATIVE_SEED(*key)).random())
        self.assertEqual(decision["python_version"], sys.version.split()[0])

    def test_default_calls_native_utility_without_reimplementing_its_function(self):
        module = types.ModuleType("utopia.utils.seeding")
        module.derive_seed = NATIVE_SEED
        with patch.dict(sys.modules, {"utopia.utils.seeding": module}):
            actual = policy.build_decision(
                "LN", "institution_0000_researcher_0", 2, self.previous,
                self.topics, self.distances, 42)
        self.assertEqual(actual, self.decision("LN"))

    def test_u_is_common_across_cells_order_retries_and_geometry(self):
        before = random.getstate()
        first = {}
        for agent in ("institution_0000_researcher_0", "institution_0020_researcher_4"):
            for year in range(2, 11):
                rows = [self.decision(cell, agent_id=agent, year=year) for cell in policy.CELLS]
                self.assertEqual(len({row["u"] for row in rows}), 1)
                self.assertEqual(len({row["gate_seed"] for row in rows}), 1)
                self.assertEqual(rows[0]["requested_switch"], rows[1]["requested_switch"])
                self.assertEqual(rows[2]["requested_switch"], rows[3]["requested_switch"])
                self.assertLessEqual(rows[0]["requested_switch"], rows[2]["requested_switch"])
                first[agent, year] = rows[0]["u"]
        for (agent, year), value in reversed(list(first.items())):
            self.assertEqual(policy.switch_uniform(agent, year, derive_seed_fn=NATIVE_SEED), value)
            row = self.decision(
                "HF", agent_id=agent, year=year,
                canonical_topics53=list(reversed(self.topics)),
                distance_map52={key: 2 - value for key, value in reversed(list(self.distances.items()))})
            self.assertEqual(row["u"], value)
        self.assertEqual(random.getstate(), before)

    def test_gate_thresholds_are_strict(self):
        for p in (0.25, 0.75):
            self.assertTrue(policy.requested_switch(0.0, p))
            self.assertTrue(policy.requested_switch(math.nextafter(p, 0), p))
            self.assertFalse(policy.requested_switch(p, p))
            self.assertFalse(policy.requested_switch(math.nextafter(p, 1), p))
            self.assertFalse(policy.requested_switch(math.nextafter(1.0, 0), p))
        for u in (-0.01, 1.0, math.nan, math.inf, None, True, "0.1"):
            with self.subTest(u=u), self.assertRaises(policy.PolicyInputError):
                policy.requested_switch(u, 0.25)
        for p in (0, 0.5, 1, True, math.nan, "0.25"):
            with self.subTest(p=p), self.assertRaises(policy.PolicyInputError):
                policy.requested_switch(0.1, p)

    def test_exact_menu_sizes_previous_exclusion_and_canonical_presentation(self):
        for cell in policy.CELLS:
            with self.subTest(cell=cell), patch.object(policy, "_gate", return_value=(7, 0.1)):
                row = self.decision(cell, canonical_topics53=list(reversed(self.topics)),
                                    distance_map52=dict(reversed(list(self.distances.items()))))
            self.assertTrue(row["requested_switch"])
            self.assertEqual(row["near_topics"], self.eligible[:17])
            self.assertEqual(row["far_topics"], self.eligible[-17:])
            self.assertEqual(row["n_middle_unused"], 18)
            self.assertFalse(set(row["near_topics"]) & set(row["far_topics"]))
            self.assertNotIn(self.previous, row["ranked_eligible_topics"])
            self.assertNotIn(self.previous, row["candidate_topics"])
            selected = row["near_topics"] if cell.endswith("N") else row["far_topics"]
            self.assertEqual(row["candidate_topics"], selected)
            self.assertEqual(row["candidate_topics"], sorted(row["candidate_topics"]))

    def test_distance_ties_use_topic_names_not_mapping_order(self):
        ties = {topic: (i // 18) * 0.5 for i, topic in enumerate(self.eligible)}
        row = self.decision(distance_map52=dict(reversed(list(ties.items()))))
        self.assertEqual(row["ranked_eligible_topics"], self.eligible)
        self.assertEqual(row["near_topics"], self.eligible[:17])
        self.assertEqual(row["far_topics"], self.eligible[-17:])
        self.assertEqual(row["separation_gap"], 0.5)

    def test_stay_still_checks_and_logs_both_menus(self):
        with patch.object(policy, "_gate", return_value=(8, 0.99)):
            row = self.decision("LF")
        self.assertFalse(row["requested_switch"])
        self.assertEqual(row["candidate_topics"], [self.previous])
        self.assertEqual((len(row["near_topics"]), len(row["far_topics"])), (17, 17))
        self.assertGreater(row["separation_gap"], 0)

    def test_zero_gap_is_domain_failure_even_when_gate_says_stay(self):
        for u in (0.0, 0.99):
            with self.subTest(u=u), patch.object(policy, "_gate", return_value=(9, u)):
                with self.assertRaises(policy.PolicyDomainError) as caught:
                    self.decision(distance_map52=dict.fromkeys(self.eligible, 1.0))
            evidence = caught.exception.diagnostics
            self.assertEqual(evidence["status"], "out_of_domain")
            self.assertEqual(evidence["reason"], "absent_strict_separation")
            self.assertEqual(evidence["separation_gap"], 0)
            self.assertEqual((evidence["n_near"], evidence["n_far"]), (17, 17))
            self.assertEqual(evidence["u"], u)
            json.dumps(evidence, allow_nan=False)
            self.assertNotIsInstance(caught.exception, Exception)

    def test_positive_gap_is_not_rounded_or_replaced_by_tolerance(self):
        delta = math.nextafter(1.0, 2.0) - 1.0
        distances = {topic: 1.0 if i < 35 else 1.0 + delta
                     for i, topic in enumerate(self.eligible)}
        row = self.decision(distance_map52=distances)
        self.assertEqual(row["separation_gap"], delta)
        self.assertGreater(row["separation_gap"], 0)
        self.assertLess(row["separation_gap"], policy.DISTANCE_BOUNDARY_TOLERANCE)

    def test_finite_full_cosine_range_and_boundary_roundoff_preserve_values(self):
        distances = dict(self.distances)
        distances[self.eligible[0]] = -policy.DISTANCE_BOUNDARY_TOLERANCE / 2
        distances[self.eligible[-1]] = 2 + policy.DISTANCE_BOUNDARY_TOLERANCE / 2
        row = self.decision(distance_map52=distances)
        self.assertEqual(row["distance_map"], distances)
        self.assertGreater(row["far_max"], 2)
        self.assertGreater(row["far_min"], 1)
        for value in (-0.001, 2.001, math.nan, math.inf, -math.inf, None, True, "0.5", []):
            invalid = dict(distances)
            invalid[self.eligible[1]] = value
            with self.subTest(value=value), self.assertRaises(policy.PolicyInputError):
                self.decision(distance_map52=invalid)

    def test_missing_extra_previous_and_empty_distance_maps_fail(self):
        missing = dict(self.distances)
        del missing[self.eligible[0]]
        for data in ({}, None, [], missing, {**self.distances, self.previous: 0.1},
                     {**self.distances, "invented_topic": 0.5}):
            with self.subTest(data_type=type(data)), self.assertRaises(policy.PolicyInputError):
                self.decision(distance_map52=data)

    def test_topic_roster_previous_year_seed_and_cell_are_strict(self):
        bad_cases = [
            {"canonical_topics53": []}, {"canonical_topics53": self.topics[:-1]},
            {"canonical_topics53": self.topics[:-1] + [self.topics[0]]},
            {"canonical_topics53": self.topics[:-1] + ["invented"]},
            {"canonical_topics53": self.topics[:-1] + [None]},
            {"canonical_topics53": "not a roster"},
            {"previous_topic": None}, {"previous_topic": ""}, {"previous_topic": "invented"},
            {"year": 1}, {"year": 0}, {"year": 11}, {"year": True}, {"year": 2.0},
            {"seed": 43}, {"seed": True}, {"cell": "explorer"},
            {"agent_id": ""}, {"agent_id": " padded "},
        ]
        for overrides in bad_cases:
            with self.subTest(overrides=overrides), self.assertRaises(policy.PolicyInputError):
                self.decision(**overrides)

    def test_history_metadata_describes_native_inclusive_window_without_claiming_coverage(self):
        row = self.decision(year=6)
        self.assertEqual((row["history_window_start"], row["history_window_end"]), (2, 5))
        self.assertEqual(row["native_centroid_max_years"], 3)
        self.assertNotIn("reference_source", row)
        self.assertNotIn("history_coverage_valid", row)

    def test_first_candidate_fallback_respects_both_switch_and_stay(self):
        for u in (0.1, 0.99):
            for cell in policy.CELLS:
                with patch.object(policy, "_gate", return_value=(10, u)):
                    decision = self.decision(cell, distance_map52={
                        key: 2 - value for key, value in self.distances.items()})
                original = deepcopy(decision)
                for kind in (None, "parse", "validation"):
                    result = policy.finalize_choice(decision, decision["candidate_topics"][0], kind)
                    self.assertEqual(result["realized_switch"], decision["requested_switch"])
                    self.assertEqual(result["eligible_repeat_count"], 1)
                    self.assertEqual(result["new_project_choice_count"], 1)
                    self.assertEqual(result["initial_choice_count"], 0)
                    self.assertEqual(result["fallback"], kind is not None)
                    if not result["realized_switch"]:
                        self.assertIsNone(result["conditional_switch_distance"])
                    json.dumps(result, allow_nan=False)
                self.assertEqual(decision, original)

    def test_invalid_choice_and_fallback_are_not_repaired(self):
        with patch.object(policy, "_gate", return_value=(11, 0.1)):
            decision = self.decision()
        for topic, kind in ((self.previous, None), ("invented", "parse"),
                            (None, None), (decision["candidate_topics"][1], "parse"),
                            (decision["candidate_topics"][0], "unknown"),
                            (decision["candidate_topics"][0], True)):
            with self.subTest(topic=topic, kind=kind), self.assertRaises(policy.PolicyInputError):
                policy.finalize_choice(decision, topic, kind)
        with patch.object(policy, "_gate", return_value=(12, 0.99)):
            stay = self.decision()
        with self.assertRaises(policy.PolicyInputError):
            policy.finalize_choice(stay, stay["near_topics"][0])

    def test_input_mutation_global_rng_and_file_access_are_absent(self):
        original_topics, original_distances = deepcopy(self.topics), deepcopy(self.distances)
        state = random.getstate()
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")):
            row = self.decision()
            policy.finalize_choice(row, row["candidate_topics"][0])
        self.assertEqual(self.topics, original_topics)
        self.assertEqual(self.distances, original_distances)
        self.assertEqual(random.getstate(), state)


if __name__ == "__main__":
    unittest.main()
