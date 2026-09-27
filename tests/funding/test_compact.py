from copy import deepcopy
import json
import random
import unittest

from utopia.funding.compact import COMPACT_SCHEMA, compact_prompt, compact_response_format, expand_compact_result


class CompactFundingTests(unittest.TestCase):
    def test_expansion_preserves_model_order_and_exact_input_identity(self):
        apps = [{"applicant_id": "same-researcher"}, {"applicant_id": "other"},
                {"applicant_id": "same-researcher"}]
        result = {"ranked_application_ids": [2, 0, 1]}
        before = deepcopy((apps, result))
        rng = random.getstate()
        rows = expand_compact_result(result, apps)["ranked_applications"]
        self.assertEqual([row["application_id"] for row in rows], [2, 0, 1])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3])
        self.assertEqual([row["applicant_id"] for row in rows],
                         ["same-researcher", "same-researcher", "other"])
        self.assertTrue(all(row["reason"] == "" for row in rows))
        self.assertEqual((apps, result), before)
        self.assertEqual(random.getstate(), rng)

    def test_incomplete_duplicate_coerced_or_extra_field_output_is_rejected(self):
        apps = [{"applicant_id": f"r{i}"} for i in range(3)]
        invalid = [
            None, [], {"ranked_applications": []}, {"ranked_application_ids": [0, 1]},
            {"ranked_application_ids": [0, 1, 1]}, {"ranked_application_ids": [-1, 0, 1]},
            {"ranked_application_ids": [0, 1, 3]}, {"ranked_application_ids": [0, True, 2]},
            {"ranked_application_ids": [0, 1.0, 2]}, {"ranked_application_ids": [0, "1", 2]},
            {"ranked_application_ids": [0, float("nan"), 2]},
            {"ranked_application_ids": [0, 1, 2], "reason": "extra"},
            {"ranked_application_ids": (0, 1, 2)},
        ]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                expand_compact_result(result, apps)

    def test_panel_size_and_metadata_must_be_valid(self):
        for apps in ([], [{"applicant_id": f"r{i}"} for i in range(26)],
                     [{"applicant_id": 42}], [None]):
            with self.subTest(apps=apps), self.assertRaises(ValueError):
                expand_compact_result({"ranked_application_ids": [0]}, apps)

    def test_substantive_prompt_is_byte_identical_including_record_mask(self):
        prefix = (
            "Program criteria, original proposals, applicant IDs and expertise.\n"
            "## Response Format\nThis is quoted proposal text.\n"
            "Past performance is withheld in this condition.\n"
        )
        old = prefix + '\n## Response Format\n{"ranked_applications": []}\n'
        new = compact_prompt(old, 25)
        self.assertEqual(new.rpartition("\n## Response Format\n")[0], prefix)
        self.assertIn("exactly 25 integer application IDs", new)
        self.assertIn("from 0 through 24", new)
        self.assertNotIn('"ranked_applications"', new.rpartition("\n## Response Format\n")[2])

    def test_missing_response_boundary_or_invalid_panel_size_is_rejected(self):
        with self.assertRaises(ValueError):
            compact_prompt("Unrecognized original prompt", 25)
        for count in (0, 26, True, 25.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                compact_prompt('\n## Response Format\n"ranked_applications"', count)

    def test_schema_copies_are_independent_and_json_serializable(self):
        first, second = compact_response_format(), compact_response_format()
        self.assertEqual(second["json_object"]["schema"], COMPACT_SCHEMA)
        first["json_object"]["schema"]["properties"].clear()
        self.assertEqual(second["json_object"]["schema"], COMPACT_SCHEMA)
        json.dumps(second, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
