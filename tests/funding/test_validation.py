from __future__ import annotations

from utopia.utils.paths import project_root

import ast

from copy import deepcopy

import json

import logging

import os

from pathlib import Path

import random

import stat

import sys

import tempfile

from types import SimpleNamespace

import unittest

from unittest.mock import patch

from utopia.funding.validation import FundingRankingValidationError, inspect_funding_result, install_funding_validation

import utopia.funding.validation as gate

from tests.support.funding_rankings import ROOT, STOCK_PATH, stock_class, apps_for, ranking, result_pairs, metadata, PROGRAMS, GateCase




class StrictValidatorTests(GateCase):
    def test_one_of_100_stock_bug_is_rejected_without_imputation(self):
        apps = apps_for(100)
        partial = ranking(apps, [0])
        self.assertTrue(stock_class().validate_funding_result(partial, apps))
        cls, handle, path = self.install()
        before = deepcopy(partial)
        self.assertFalse(cls.validate_funding_result(partial, apps))
        self.assertEqual(partial, before)
        record = self.read_records(path)[0]
        self.assertEqual([record[key] for key in
                          ("n_expected", "n_returned", "n_valid", "n_missing", "n_rejected")],
                         [100, 1, 1, 99, 0])
        self.assertEqual(record["missing_application_ids"], list(range(1, 100)))
        self.assertEqual(handle.summary()["invalid_attempts"], 1)

    def test_repeated_applicant_ids_are_distinct_applications(self):
        apps = apps_for(7, repeat=True)
        cls, _, _ = self.install()
        self.assertTrue(cls.validate_funding_result(ranking(apps), apps))
        self.assertTrue(cls.validate_funding_result(ranking(apps)["ranked_applications"], apps))
        report = inspect_funding_result(ranking(apps), apps)
        self.assertEqual(report["n_valid"], 7)
        self.assertEqual(report["n_missing"], 0)

    def test_all_malformed_result_shapes(self):
        apps = apps_for(3)
        bad = [None, True, 1, 0.5, "ranking", object(), (), {},
               {"ranked_applications": None}, {"ranked_applications": {}},
               {"ranked_applications": "private text"},
               {"ranked_applications": (ranking(apps)["ranked_applications"],)},
               [], {"ranked_applications": []}, {"ranked_applications": [None]},
               {"ranked_applications": [[], True, "raw"]}]
        cls, _, _ = self.install()
        for value in bad:
            with self.subTest(value_type=type(value).__name__):
                self.assertFalse(cls.validate_funding_result(value, apps))

    def test_bad_ids_ranks_duplicates_and_fabricated_flags(self):
        apps = apps_for(3)
        cases = []
        for field, values in (
            ("application_id", [None, True, False, -1, 3, 1.0, 1.5, "1", [],
                                {}, float("nan"), float("inf")]),
            ("applicant_id", [None, True, 2, [], {}, "wrong", "researcher_1"]),
            ("rank", [None, True, False, 0, -1, 4, 1.0, 1.5, "1", [], {},
                      float("nan"), float("inf"), float("-inf")]),
        ):
            for value in values:
                candidate = ranking(apps, [0, 1, 2])
                candidate["ranked_applications"][0][field] = value
                cases.append((field, candidate))
            candidate = ranking(apps, [0, 1, 2])
            del candidate["ranked_applications"][0][field]
            cases.append(("missing_" + field, candidate))
        for field in ("application_id", "rank"):
            candidate = ranking(apps, [0, 1, 2])
            candidate["ranked_applications"][1][field] = candidate["ranked_applications"][0][field]
            cases.append(("duplicate_" + field, candidate))
        for flag in ("imputed_tail", "fallback_ranking"):
            for value in (True, 1, "false", None, object()):
                candidate = ranking(apps)
                candidate["ranked_applications"][0][flag] = value
                cases.append((flag, candidate))
            candidate = ranking(apps)
            candidate[flag] = True
            cases.append(("root_" + flag, candidate))
        extra = ranking(apps)
        extra["ranked_applications"].append(deepcopy(extra["ranked_applications"][0]))
        cases.append(("extra_row", extra))
        cls, _, _ = self.install()
        for case, value in cases:
            with self.subTest(case=case):
                self.assertFalse(cls.validate_funding_result(value, apps))

    def test_no_missing_id_recovery_even_for_unique_applicant(self):
        apps = apps_for(1)
        response = ranking(apps)
        del response["ranked_applications"][0]["application_id"]
        report = inspect_funding_result(response, apps)
        self.assertFalse(report["valid"])
        self.assertEqual(report["n_rejected"], 1)

    def test_duplicate_counts_are_consistent(self):
        apps = apps_for(4, repeat=True)
        response = ranking(apps, [0, 1, 2, 3])
        response["ranked_applications"][1]["application_id"] = 0
        report = inspect_funding_result(response, apps)
        self.assertEqual((report["n_valid"], report["n_missing"], report["n_rejected"]), (2, 2, 2))
        self.assertEqual(report["missing_application_ids"], [0, 1])
        self.assertEqual(report["row_error_counts"]["duplicate_application_id"], 2)

    def test_bad_expected_apps_do_not_raise_ordinary_exceptions(self):
        for apps in (None, {}, True, "private", [None], [{}], [{"applicant_id": []}],
                     [{"applicant_id": True}], [{"applicant_id": 1}]):
            with self.subTest(apps_type=type(apps).__name__):
                self.assertFalse(inspect_funding_result({}, apps)["valid"])

    def test_empty_panel_and_1001_applications_without_rng_use(self):
        state = random.getstate()
        self.assertTrue(inspect_funding_result([], [])["valid"])
        apps = apps_for(1001, repeat=True)
        self.assertTrue(inspect_funding_result(ranking(apps), apps)["valid"])
        self.assertEqual(state, random.getstate())


class StockParityTests(GateCase):
    def test_duplicate_panel_cannot_double_awards_even_with_copied_metadata(self):
        for copy_metadata in (False, True):
            with self.subTest(copy_metadata=copy_metadata):
                cls, handle, path = self.install()
                apps = apps_for(5)
                pair = (ranking(apps), [])
                meta = metadata(apps)[0]
                duplicate = deepcopy(meta) if copy_metadata else meta
                batch, metas = [pair, pair], [meta, duplicate]
                before = deepcopy([batch, metas])
                log, resources = [], {"researcher_4": 100}
                with patch.object(cls, "normalize_ranked_applications",
                                  side_effect=AssertionError("stock must not start")):
                    with self.assertRaises(FundingRankingValidationError) as caught:
                        winners = cls.process_funding_evaluation_results(
                            batch, metas, PROGRAMS, application_log=log)
                        resources["researcher_4"] += 20 * len(winners["P"])
                self.assertEqual([batch, metas], before)
                self.assertEqual(log, [])
                self.assertEqual(resources, {"researcher_4": 100})
                self.assertIn("duplicate_panel_metadata",
                              caught.exception.audit["panels"][0]["errors"])
                self.assertEqual(caught.exception.audit["panels"][0]["batch_index"], 1)
                self.assertEqual(len(self.read_records(path)), 2)
                self.assertFalse(handle.summary()["completion_allowed"])
                self.assertEqual(handle.summary()["successful_batches"], 0)

    def test_1025_applications_cap25_seed42_preserve_stock_slots_and_rng(self):
        # CPU fixture only. No population study, model call or global RNG seed.
        population = apps_for(1025)
        panels = [population[i:i + 25] for i in range(0, len(population), 25)]
        local_rng = random.Random(42)
        state = random.getstate()
        responses = []
        for apps in panels:
            order = list(range(len(apps)))
            local_rng.shuffle(order)
            responses.append(ranking(apps, order))
        batch, metas = result_pairs(*responses), metadata(*panels)
        expected_log, actual_log = [], []
        expected = stock_class().process_funding_evaluation_results(
            batch, metas, PROGRAMS, slot_override={"P": 137},
            application_log=expected_log)
        cls, handle, path = self.install()
        actual = cls.process_funding_evaluation_results(
            batch, metas, PROGRAMS, slot_override={"P": 137},
            application_log=actual_log)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_log, expected_log)
        self.assertEqual(len(actual["P"]), 137)
        self.assertEqual(len(actual_log), 1025)
        self.assertEqual(handle.summary()["final_panels"], 41)
        self.assertTrue(all(row["n_expected"] == 25 for row in self.read_records(path)))
        self.assertEqual(state, random.getstate())

    def test_valid_stock_outcomes_and_application_logs_are_exactly_preserved(self):
        for shared in (False, True):
            for penalty in (0.0, 0.7):
                for slots in (None, {"P": 0}, {"P": 3}, {"P": 30}):
                    with self.subTest(shared=shared, penalty=penalty, slots=slots):
                        panels = [apps_for(5, repeat=shared), apps_for(3, repeat=shared)]
                        panels[1][0]["applicant_id"] = "other"
                        batch = result_pairs(*(ranking(apps) for apps in panels))
                        metas = metadata(*panels)
                        penalties = {"same": 0.9, "researcher_4": 0.9, "other": 0.1}
                        stock = stock_class()
                        expected_log, actual_log = [], []
                        kwargs = {"novelty_penalties": penalties, "lambda_funding": penalty,
                                  "slot_override": slots}
                        expected = stock.process_funding_evaluation_results(
                            batch, metas, PROGRAMS, application_log=expected_log, **kwargs)
                        cls, handle, path = self.install()
                        stock_processor = cls.process_funding_evaluation_results
                        with patch.object(cls, "normalize_ranked_applications",
                                          wraps=cls.normalize_ranked_applications) as normalizer:
                            actual = cls.process_funding_evaluation_results(
                                batch, metas, PROGRAMS, application_log=actual_log, **kwargs)
                            self.assertEqual(normalizer.call_count, len(panels))
                            # functools.wraps points at the exact stock callable.
                            self.assertEqual(stock_processor.__wrapped__.__code__.co_filename,
                                             str(STOCK_PATH))
                        self.assertEqual(expected, actual)
                        self.assertEqual(expected_log, actual_log)
                        self.assertTrue(all(not row["imputed_tail"] and not row["fallback_ranking"]
                                            for row in actual_log))
                        self.assertTrue(all(row["n_missing"] == row["n_rejected"] == 0
                                            for row in self.read_records(path)))
                        self.assertEqual(handle.summary()["successful_batches"], 1)

    def test_list_result_out_of_response_order_and_missing_optional_reason(self):
        apps = apps_for(5)
        rows = ranking(apps)["ranked_applications"]
        rows = rows[2:] + rows[:2]
        for row in rows:
            del row["reason"]
            row["imputed_tail"] = row["fallback_ranking"] = False
        batch = result_pairs(rows)
        expected = stock_class().process_funding_evaluation_results(
            batch, metadata(apps), PROGRAMS)
        cls, _, _ = self.install()
        self.assertEqual(cls.process_funding_evaluation_results(
            batch, metadata(apps), PROGRAMS), expected)

    def test_valid_processor_receives_exact_objects_and_is_called_once(self):
        seen = []
        stock = stock_class()
        old_process = stock.process_funding_evaluation_results
        def spy(*args, **kwargs):
            seen.append((args, kwargs))
            return old_process(*args, **kwargs)
        stock.process_funding_evaluation_results = staticmethod(spy)
        cls, _, _ = self.install(stock)
        apps = apps_for(3)
        batch, metas, log = result_pairs(ranking(apps)), metadata(apps), []
        original_rows = deepcopy(batch[0][0])
        state = random.getstate()
        cls.process_funding_evaluation_results(batch, metas, PROGRAMS, application_log=log)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0][0], batch)
        self.assertIs(seen[0][0][1], metas)
        self.assertIs(seen[0][0][2], PROGRAMS)
        self.assertIs(seen[0][1]["application_log"], log)
        self.assertEqual(batch[0][0], original_rows)
        self.assertEqual(state, random.getstate())

    def test_no_partial_log_winner_or_resource_mutation_on_any_invalid_panel(self):
        for position in (0, 1, 2):
            for bad in (None, [], {"ranked_applications": []}):
                with self.subTest(position=position, bad=bad):
                    cls, handle, path = self.install()
                    apps = apps_for(3)
                    batch = result_pairs(ranking(apps), ranking(apps), ranking(apps))
                    batch[position] = (bad, [])
                    metas = metadata(apps, apps, apps)
                    before = deepcopy([pair[0] for pair in batch])
                    log = [{"keep": "existing application log"}]
                    resources, winners = {"same": 100}, {"prior": []}
                    with patch.object(cls, "normalize_ranked_applications",
                                      side_effect=AssertionError("stock must not start")):
                        with self.assertRaises(FundingRankingValidationError) as caught:
                            # Mimic stock broad exception paths outside phase 5.
                            try:
                                winners.update(cls.process_funding_evaluation_results(
                                    batch, metas, PROGRAMS, application_log=log))
                                resources["same"] += 20
                            except Exception:
                                resources["same"] = -1
                    self.assertEqual(resources, {"same": 100})
                    self.assertEqual(winners, {"prior": []})
                    self.assertEqual(log, [{"keep": "existing application log"}])
                    self.assertEqual([pair[0] for pair in batch], before)
                    self.assertEqual(len(self.read_records(path)), 3)
                    self.assertEqual(handle.summary()["invalid_final_panels"], 1)
                    self.assertEqual(handle.summary()["successful_batches"], 0)
                    audit = caught.exception.audit
                    self.assertEqual(audit["event"], "final_batch_rejected")
                    self.assertEqual(audit["panels"][0]["panel_index"], position)
                    self.assertNotIsInstance(caught.exception, Exception)

    def test_batch_alignment_metadata_and_program_errors_reject_before_stock(self):
        apps = apps_for(2)
        valid = result_pairs(ranking(apps))
        cases = [
            (None, metadata(apps), PROGRAMS),
            (iter(valid), metadata(apps), PROGRAMS),
            (valid, None, PROGRAMS),
            (valid, [], PROGRAMS),
            ([], metadata(apps), PROGRAMS),
            (valid + valid, metadata(apps), PROGRAMS),
            ([None], metadata(apps), PROGRAMS),
            ([(ranking(apps),)], metadata(apps), PROGRAMS),
            ([(ranking(apps), [], "extra")], metadata(apps), PROGRAMS),
            (valid, [None], PROGRAMS),
            (valid, [{"program_id": "P"}], PROGRAMS),
            (valid, [{"program_id": [], "apps": apps}], PROGRAMS),
            (valid, [{"program_id": "UNKNOWN", "apps": apps}], PROGRAMS),
            (valid, [{"program_id": "P", "panel_index": True, "apps": apps}], PROGRAMS),
            (valid, metadata(apps), None),
        ]
        for batch, metas, programs in cases:
            with self.subTest(batch_type=type(batch).__name__):
                cls, _, _ = self.install()
                log = []
                with self.assertRaises(FundingRankingValidationError):
                    cls.process_funding_evaluation_results(
                        batch, metas, programs, application_log=log)
                self.assertEqual(log, [])

    def test_missing_panel_index_when_stock_needs_it_is_preflighted(self):
        cls, _, _ = self.install()
        panels = [apps_for(2), apps_for(2)]
        metas = metadata(*panels)
        del metas[1]["panel_index"]
        log = []
        with self.assertRaises(FundingRankingValidationError):
            cls.process_funding_evaluation_results(
                result_pairs(*(ranking(apps) for apps in panels)),
                metas, PROGRAMS, application_log=log)
        self.assertEqual(log, [])
        # An absent index is compatible with stock when no application log.
        self.assertTrue(cls.process_funding_evaluation_results(
            result_pairs(ranking(panels[1])), [metas[1]], PROGRAMS))

    def test_empty_batch_preserves_stock_outcome(self):
        cls, handle, path = self.install()
        self.assertEqual(cls.process_funding_evaluation_results([], [], {}), {})
        self.assertEqual(path.read_bytes(), b"")
        self.assertEqual(handle.summary()["successful_batches"], 1)


class CompactModeTests(GateCase):
    def test_every_application_rank_mapping_and_exact_stock_winner_parity(self):
        state = random.getstate()
        for n in (1, 2, 25):
            for shift in range(n):
                with self.subTest(n=n, shift=shift):
                    apps = apps_for(n, repeat=True)
                    order = list(range(shift, n)) + list(range(shift))
                    compact_result = {"ranked_application_ids": order}
                    legacy_result = ranking(apps, order)
                    for row in legacy_result["ranked_applications"]:
                        row["reason"] = ""
                    history = [{"private": "keep original history"}]
                    batch = [(compact_result, history)]
                    before = deepcopy(batch)
                    metas = metadata(apps)
                    expected_log, actual_log = [], []
                    expected = stock_class().process_funding_evaluation_results(
                        [(legacy_result, history)], metas, PROGRAMS,
                        application_log=expected_log)
                    cls = stock_class()
                    process = cls.process_funding_evaluation_results
                    seen = []
                    def spy(*args, **kwargs):
                        seen.append(args[0])
                        return process(*args, **kwargs)
                    cls.process_funding_evaluation_results = staticmethod(spy)
                    cls, handle, path = self.install(cls, compact=True)
                    self.assertTrue(cls.validate_funding_result(compact_result, apps))
                    actual = cls.process_funding_evaluation_results(
                        batch, metas, PROGRAMS, application_log=actual_log)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual_log, expected_log)
                    self.assertEqual(batch, before)
                    self.assertIsNot(seen[0], batch)
                    self.assertIs(seen[0][0][1], history)
                    self.assertEqual(seen[0][0][0], legacy_result)
                    self.assertEqual(handle.summary()["output_representation"],
                                     "ordered_application_ids_v1")
                    self.assertTrue(handle.summary()["completion_allowed"])
                    self.assertTrue(all(row["output_representation"] == "ordered_application_ids_v1"
                                        for row in self.read_records(path)))
        self.assertEqual(state, random.getstate())

    def test_malformed_compact_and_legacy_shapes_rejected_with_diagnostics(self):
        cls, _, path = self.install(compact=True)
        apps = apps_for(3)
        bad = [
            None, [0, 1, 2], ranking(apps), ranking(apps)["ranked_applications"],
            {}, {"ranked_application_ids": [0, 1, 2], "ranked_applications": []},
            {"ranked_application_ids": (0, 1, 2)},
            {"ranked_application_ids": [0]},
            {"ranked_application_ids": [0, 0, 2]},
            {"ranked_application_ids": [0, 1, 3]},
            {"ranked_application_ids": [-1, 1, 2]},
            {"ranked_application_ids": [0, True, 2]},
            {"ranked_application_ids": [0, 1.0, 2]},
            {"ranked_application_ids": [0, "1", 2]},
            {"ranked_application_ids": [0, float("nan"), 2]},
            {"ranked_application_ids": [0, float("inf"), 2]},
            {"ranked_application_ids": [0, [], 2]},
        ]
        for result in bad:
            self.assertFalse(cls.validate_funding_result(result, apps))
        records = self.read_records(path)
        self.assertTrue(all("compact_validation_failed" in row["errors"] for row in records))
        partial = records[7]
        self.assertEqual([partial[key] for key in
                          ("n_expected", "n_returned", "n_valid", "n_missing", "n_rejected")],
                         [3, 1, 1, 2, 0])
        self.assertIn("compact_duplicate_id", records[8]["errors"])
        self.assertIn("compact_result_shape", records[2]["errors"])
        self.assertNotIn("NaN", path.read_text())

    def test_all_panels_preflight_before_conversion_for_stock_or_any_mutation(self):
        import utopia.funding.compact as compact_api
        for duplicate_panel in (False, True):
            with self.subTest(duplicate_panel=duplicate_panel):
                cls, handle, _ = self.install(compact=True)
                apps = apps_for(3)
                valid = {"ranked_application_ids": [2, 0, 1]}
                batch = [(valid, []), (valid if duplicate_panel else
                                      {"ranked_application_ids": [0]}, [])]
                metas = metadata(apps, apps)
                if duplicate_panel:
                    metas[1] = deepcopy(metas[0])
                before = deepcopy([batch, metas])
                log, resources = [], {"researcher_2": 100}
                with patch.object(compact_api, "expand_compact_result",
                                  wraps=compact_api.expand_compact_result) as expand, \
                        patch.object(cls, "normalize_ranked_applications") as normalize:
                    with self.assertRaises(FundingRankingValidationError):
                        winners = cls.process_funding_evaluation_results(
                            batch, metas, PROGRAMS, application_log=log)
                        resources["researcher_2"] += 20 * len(winners["P"])
                    # Only the individual validation probes ran. No separate
                    # expansion batch was produced for the stock processor.
                    self.assertEqual(expand.call_count, 2)
                    normalize.assert_not_called()
                self.assertEqual([batch, metas], before)
                self.assertEqual(log, [])
                self.assertEqual(resources, {"researcher_2": 100})
                self.assertFalse(handle.summary()["completion_allowed"])

    def test_prompt_tail_schema_metadata_context_and_restore(self):
        import utopia.funding.compact as compact_api
        cls = stock_class()
        panels = [apps_for(3), apps_for(2)]
        metas = metadata(*panels)
        prefixes = ["P1 substantive publication record and research proposal.",
                    "P0 Publication record withheld. Same research proposal."]
        texts = [prefix + '\n## Response Format\n{"ranked_applications": []}'
                 for prefix in prefixes]
        original_format = {"type": "legacy fixture"}
        value = (texts, original_format, metas)
        calls = []
        def builder(self, *args, **kwargs):
            calls.append((args, kwargs))
            return value
        cls.get_funding_evaluation_prompts = builder
        original_descriptors = dict(vars(cls))
        cls, handle, path = self.install(cls, compact=True)
        state = random.getstate()
        rewritten, response_format, returned_metas = cls().get_funding_evaluation_prompts(
            "original applications", panel_max_apps=25, panel_seed=42)
        self.assertEqual(calls, [(("original applications",),
                                 {"panel_max_apps": 25, "panel_seed": 42})])
        self.assertIs(returned_metas, metas)
        self.assertIsNot(rewritten, texts)
        for i, text in enumerate(rewritten):
            self.assertEqual(text.split("\n## Response Format\n")[0], prefixes[i])
            self.assertIn('"ranked_application_ids"', text)
            self.assertNotIn('"ranked_applications"', text)
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_object"]["schema"], compact_api.COMPACT_SCHEMA)
        self.assertEqual(original_format, {"type": "legacy fixture"})
        self.assertTrue(all('"ranked_applications"' in text for text in texts))
        batch = [({"ranked_application_ids": list(reversed(range(len(apps))))}, [])
                 for apps in panels]
        for pair, apps in zip(batch, panels):
            self.assertTrue(cls.validate_funding_result(pair[0], apps))
        cls.process_funding_evaluation_results(batch, metas, PROGRAMS)
        self.assertEqual([row["panel_index"] for row in self.read_records(path)], [0, 1, 0, 1])
        handle.restore()
        self.assertEqual(dict(vars(cls)), original_descriptors)
        self.assertTrue(handle.summary()["completion_allowed"])
        self.assertEqual(random.getstate(), state)

    def test_bad_prompt_boundary_is_fatal_and_empty_year_remains_valid(self):
        cls = stock_class()
        cls.get_funding_evaluation_prompts = lambda self: (
            ["missing response-format boundary"], {}, metadata(apps_for(2)))
        cls, handle, path = self.install(cls, compact=True)
        with self.assertRaises(FundingRankingValidationError):
            cls().get_funding_evaluation_prompts()
        self.assertEqual(self.read_records(path)[0]["event"], "compact_prompt_error")
        self.assertFalse(handle.summary()["completion_allowed"])
        empty, empty_handle, _ = self.install(compact=True)
        self.assertEqual(empty.process_funding_evaluation_results([], [], {}), {})
        self.assertTrue(empty_handle.summary()["completion_allowed"])

    def test_default_remains_legacy_and_does_not_load_compact_dependency(self):
        import builtins
        original_import = builtins.__import__

        def import_without_compact(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "utopia.funding" and "compact" in fromlist:
                raise AssertionError("default must not import compact module")
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", side_effect=import_without_compact):
            cls, handle, _ = self.install()
            apps = apps_for(3)
            self.assertTrue(cls.validate_funding_result(ranking(apps), apps))
        self.assertEqual(handle.summary()["output_representation"], "ranked_applications_v1")


class AuditAndLifecycleTests(GateCase):
    def test_completion_allows_recovered_attempts_and_requires_successful_finals(self):
        cls, handle, _ = self.install()
        apps = apps_for(3)
        self.assertTrue(handle.summary()["completion_allowed"])  # No panels yet.
        self.assertFalse(cls.validate_funding_result(None, apps))
        self.assertFalse(handle.summary()["completion_allowed"])
        self.assertTrue(cls.validate_funding_result(ranking(apps), apps))
        self.assertFalse(handle.summary()["completion_allowed"])  # Awaiting processing.
        cls.process_funding_evaluation_results(result_pairs(ranking(apps)), metadata(apps), PROGRAMS)
        handle.restore()
        summary = handle.summary()
        self.assertTrue(summary["completion_allowed"])
        self.assertEqual(summary["invalid_attempts"], 1)
        for key in ("invalid_final_panels", "failed_batches", "processing_failures",
                    "audit_failures", "imputed_rankings", "fallback_rankings",
                    "unprocessed_panels"):
            self.assertEqual(summary[key], 0)
        json.dumps(summary, allow_nan=False)

    def test_completion_fails_permanently_after_final_imputation_or_fallback(self):
        for flag, counter in (("imputed_tail", "imputed_rankings"),
                              ("fallback_ranking", "fallback_rankings")):
            with self.subTest(flag=flag):
                cls, handle, _ = self.install()
                apps = apps_for(2)
                result = ranking(apps)
                result["ranked_applications"][0][flag] = True
                with self.assertRaises(FundingRankingValidationError):
                    cls.process_funding_evaluation_results(
                        result_pairs(result), metadata(apps), PROGRAMS)
                self.assertEqual(handle.summary()[counter], 1)
                self.assertFalse(handle.summary()["completion_allowed"])
                # Explicitly catching the fatal and later trying valid output
                # cannot erase the failure in the completion summary.
                cls.process_funding_evaluation_results(
                    result_pairs(ranking(apps)), metadata(apps), PROGRAMS)
                handle.restore()
                self.assertFalse(handle.summary()["completion_allowed"])

    def test_restore_cannot_hide_unprocessed_registered_or_attempted_panels(self):
        cls = stock_class()
        registered, attempted = apps_for(2), apps_for(3)
        cls.get_funding_evaluation_prompts = lambda self: (["a"], {}, metadata(registered))
        cls, handle, _ = self.install(cls)
        cls().get_funding_evaluation_prompts()
        cls.validate_funding_result(ranking(attempted), attempted)
        self.assertEqual(handle.summary()["unprocessed_panels"], 2)
        handle.restore()
        self.assertEqual(handle.summary()["unprocessed_panels"], 2)
        self.assertFalse(handle.summary()["completion_allowed"])

    def test_stock_processing_exception_is_preserved_and_prohibits_completion(self):
        cls = stock_class()
        failure = RuntimeError("stock fixture failure")
        def fail(*args, **kwargs):
            raise failure
        cls.process_funding_evaluation_results = staticmethod(fail)
        cls, handle, _ = self.install(cls)
        apps = apps_for(2)
        with self.assertRaises(RuntimeError) as caught:
            cls.process_funding_evaluation_results(
                result_pairs(ranking(apps)), metadata(apps), PROGRAMS)
        self.assertIs(caught.exception, failure)
        handle.restore()
        self.assertEqual(handle.summary()["processing_failures"], 1)
        self.assertFalse(handle.summary()["completion_allowed"])

    def test_registered_panel_missing_from_both_final_lists_is_fatal(self):
        for present in (0, 1):
            with self.subTest(present=present):
                cls = stock_class()
                panels = [apps_for(3), apps_for(4)]
                metas = metadata(*panels)
                cls.get_funding_evaluation_prompts = lambda self: (["a", "b"], {}, metas)
                cls, handle, path = self.install(cls)
                cls().get_funding_evaluation_prompts()
                log = []
                with self.assertRaises(FundingRankingValidationError) as caught:
                    cls.process_funding_evaluation_results(
                        result_pairs(*(ranking(apps) for apps in panels[:present])),
                        metas[:present], PROGRAMS, application_log=log)
                self.assertEqual(log, [])
                self.assertEqual(handle.summary()["final_panels"], 2)
                self.assertEqual(handle.summary()["invalid_final_panels"], 2 - present)
                self.assertTrue(all("missing_final_panel" in record["errors"]
                                    for record in caught.exception.audit["panels"]))
                self.assertEqual(len(self.read_records(path)), 2)

    def test_registered_panel_program_cannot_be_changed_before_final_processing(self):
        cls = stock_class()
        apps = apps_for(2)
        metas = metadata(apps)
        cls.get_funding_evaluation_prompts = lambda self: (["a"], {}, metas)
        cls, _, _ = self.install(cls)
        cls().get_funding_evaluation_prompts()
        changed = [{"program_id": "Q", "panel_index": 0, "apps": apps}]
        with self.assertRaises(FundingRankingValidationError) as caught:
            cls.process_funding_evaluation_results(
                result_pairs(ranking(apps)), changed, {"Q": PROGRAMS["P"]})
        self.assertIn("panel_metadata_changed", caught.exception.audit["panels"][0]["errors"])

    def test_context_attempts_and_original_prompt_result_identity(self):
        cls = stock_class()
        apps = apps_for(3)
        value = (["unchanged prompt"], {"schema": "untouched"}, metadata(apps))
        calls = []
        def build(self, *args, **kwargs):
            calls.append((args, kwargs))
            return value
        cls.get_funding_evaluation_prompts = build
        cls, handle, path = self.install(cls)
        self.assertIs(cls().get_funding_evaluation_prompts("input", panel_max_apps=25), value)
        self.assertEqual(calls, [(("input",), {"panel_max_apps": 25})])
        for result in (None, ranking(apps, [0]), ranking(apps)):
            cls.validate_funding_result(result, apps)
        cls.process_funding_evaluation_results(result_pairs(ranking(apps)), value[2], PROGRAMS)
        records = self.read_records(path)
        self.assertEqual([r["sequence"] for r in records], [1, 2, 3, 4])
        self.assertEqual([r["attempt"] for r in records[:-1]], [1, 2, 3])
        self.assertEqual(records[-1]["attempts_seen"], 3)
        self.assertTrue(all(r["program_id"] == "P" and r["panel_index"] == 0 for r in records))
        self.assertEqual(handle.summary()["invalid_attempts"], 2)
        self.assertEqual(handle._contexts, {})

    def test_stock_ranking_retries_are_bounded_and_fatal_escapes_exception(self):
        tree = ast.parse((ROOT / "utopia/simulation.py").read_text())
        sim = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Simulation")
        rank = next(n for n in sim.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_rank_funding_applications")
        retry_code = compile(ast.fix_missing_locations(ast.Module(
            body=[rank], type_ignores=[])), str(ROOT / "utopia/simulation.py"), "exec")
        for compact, recover in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(compact=compact, recover=recover):
                cls, handle, _ = self.install(compact=compact)
                apps = apps_for(25 if compact else 100)
                calls = []
                def generate_batch(prompts, **kwargs):
                    calls.append(kwargs["seed_ctx"])
                    if compact:
                        order = (list(reversed(range(25))) if recover and len(calls) == 3
                                 else [0] if len(calls) == 1 else [0] * 25)
                        response = {"ranked_application_ids": order}
                    else:
                        response = ranking(apps) if recover and len(calls) == 3 else ranking(apps, [0])
                    return result_pairs(response)
                ns = {
                    "FundingAgency": cls,
                    "logger": logging.getLogger("stock-retry-fixture"),
                }
                exec(retry_code, ns)
                results, result_metadata = ns["_rank_funding_applications"](
                    SimpleNamespace(llm=SimpleNamespace(generate_batch=generate_batch)),
                    1, ["unchanged prompt"], metadata(apps), {},
                )
                self.assertEqual(calls, [("phase5_funding_eval", 1, i) for i in range(3)])
                if recover:
                    self.assertTrue(cls.process_funding_evaluation_results(
                        results, result_metadata, PROGRAMS))
                else:
                    with self.assertRaises(FundingRankingValidationError):
                        try:
                            cls.process_funding_evaluation_results(
                                results, result_metadata, PROGRAMS)
                        except Exception:
                            self.fail("Fatal failure was swallowed")
                self.assertEqual(handle.summary()["validation_attempts"], 3)

    def test_deterministic_safe_json_without_raw_text_or_nonfinite_values(self):
        class PrivateObject:
            def __repr__(self):
                raise AssertionError("must not stringify raw inputs")
        outputs = []
        for _ in range(2):
            cls, _, path = self.install()
            apps = apps_for(2)
            result = ranking(apps)
            result["ranked_applications"][0].update(
                reason=PrivateObject(), rank=float("nan"), private_path="/private/do-not-log")
            result["ranked_applications"][1]["reason"] = "secret\nfake-log-line"
            self.assertFalse(cls.validate_funding_result(result, apps))
            with self.assertRaises(FundingRankingValidationError):
                cls.process_funding_evaluation_results(result_pairs(result), metadata(apps), PROGRAMS)
            data = path.read_bytes()
            outputs.append(data)
            self.assertNotIn(b"secret", data)
            self.assertNotIn(b"private", data)
            self.assertNotIn(b"NaN", data)
            self.assertEqual(len(data.splitlines()), 2)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)
        self.assertEqual(*outputs)

    def test_existing_output_source_symlink_and_hardlink_are_never_modified(self):
        source = self.directory / "source.py"
        source.write_bytes(b"KEEP EXISTING BYTES")
        symlink, hardlink = self.directory / "link.jsonl", self.directory / "hard.jsonl"
        symlink.symlink_to(source)
        os.link(source, hardlink)
        for path in (source, symlink, hardlink, self.directory):
            cls = stock_class()
            original = dict(vars(cls))
            with self.subTest(path=path.name):
                with self.assertRaises(FundingRankingValidationError) as caught:
                    install_funding_validation(cls, path)
                self.assertEqual(caught.exception.audit["operation"], "open")
                self.assertEqual(dict(vars(cls)), original)
                self.assertEqual(source.read_bytes(), b"KEEP EXISTING BYTES")

    def test_missing_parent_does_not_create_directories(self):
        cls = stock_class()
        parent = self.directory / "not-created"
        with self.assertRaises(FundingRankingValidationError):
            install_funding_validation(cls, parent / "audit.jsonl")
        self.assertFalse(parent.exists())

    def test_write_or_fsync_failure_is_fatal_before_stock_mutation(self):
        for failure in ("write", "fsync"):
            for stage in ("attempt", "final"):
                with self.subTest(failure=failure, stage=stage):
                    cls, handle, _ = self.install()
                    apps, log = apps_for(2), []
                    with patch.object(gate.os, failure, side_effect=OSError("disk unavailable")):
                        with self.assertRaises(FundingRankingValidationError) as caught:
                            try:
                                if stage == "attempt":
                                    cls.validate_funding_result(ranking(apps), apps)
                                else:
                                    cls.process_funding_evaluation_results(
                                        result_pairs(ranking(apps)), metadata(apps),
                                        PROGRAMS, application_log=log)
                            except Exception:
                                self.fail("audit failure swallowed")
                    self.assertEqual(log, [])
                    self.assertEqual(caught.exception.audit["event"], "audit_error")
                    self.assertEqual(handle.summary()["successful_batches"], 0)
                    self.assertEqual(handle.summary()["audit_failures"], 1)
                    # A caller explicitly catching BaseException still cannot
                    # reuse a logger that may now contain a partial write.
                    with self.assertRaises(FundingRankingValidationError) as stopped:
                        cls.validate_funding_result(ranking(apps), apps)
                    self.assertEqual(stopped.exception.audit["event"], "audit_unusable")

    def test_short_writes_are_completed_and_zero_writes_fail(self):
        cls, _, path = self.install()
        apps = apps_for(2)
        write = os.write
        with patch.object(gate.os, "write", side_effect=lambda fd, data: write(fd, data[:7])):
            self.assertTrue(cls.validate_funding_result(ranking(apps), apps))
        self.assertEqual(len(self.read_records(path)), 1)
        with patch.object(gate.os, "write", return_value=0):
            with self.assertRaises(FundingRankingValidationError):
                cls.validate_funding_result(ranking(apps), apps)

    def test_restore_is_exact_idempotent_and_other_classes_are_isolated(self):
        first, other = stock_class(), stock_class()
        original = dict(vars(first))
        cls, handle, _ = self.install(first)
        apps = apps_for(100)
        self.assertFalse(cls.validate_funding_result(ranking(apps, [0]), apps))
        self.assertTrue(other.validate_funding_result(ranking(apps, [0]), apps))
        handle.restore()
        handle.restore()
        self.assertEqual(dict(vars(first)), original)
        self.assertFalse(handle.summary()["installed"])
        summary = handle.summary()
        summary["invalid_attempts"] = -100
        summary["restore_conflicts"].append("external mutation")
        self.assertEqual(handle.summary()["invalid_attempts"], 1)
        self.assertEqual(handle.summary()["restore_conflicts"], [])
        self.assertTrue(cls.validate_funding_result(ranking(apps, [0]), apps))

    def test_inherited_descriptors_are_deleted_on_restore(self):
        parent = stock_class()
        child = type("ChildAgency", (parent,), {})
        before = dict(vars(child))
        cls, handle, _ = self.install(child)
        apps = apps_for(100)
        self.assertFalse(cls.validate_funding_result(ranking(apps, [0]), apps))
        self.assertTrue(parent.validate_funding_result(ranking(apps, [0]), apps))
        handle.restore()
        self.assertEqual(dict(vars(child)), before)

    def test_duplicate_and_overlapping_installs_rejected_before_file_creation(self):
        parent = stock_class()
        cls, _, _ = self.install(parent)
        child = type("ChildAgency", (parent,), {})
        for target in (cls, child):
            path = self.directory / "must-not-exist.jsonl"
            with self.assertRaises(RuntimeError):
                install_funding_validation(target, path)
            self.assertFalse(path.exists())

    def test_restore_preserves_later_owner_patch_and_reports_conflict(self):
        cls, handle, _ = self.install()
        original = handle._saved["process_funding_evaluation_results"]
        later = staticmethod(lambda result, apps: "owner")
        cls.validate_funding_result = later
        with self.assertRaisesRegex(RuntimeError, "restore conflict"):
            handle.restore()
        self.assertIs(vars(cls)["validate_funding_result"], later)
        self.assertIs(vars(cls)["process_funding_evaluation_results"], original)
        self.assertEqual(handle.summary()["restore_conflicts"], ["validate_funding_result"])
        handle.restore()

    def test_stale_wrapper_cannot_silently_skip_audit_after_restore(self):
        cls, handle, _ = self.install()
        validate = cls.validate_funding_result
        handle.restore()
        with self.assertRaises(FundingRankingValidationError):
            validate([], [])

    def test_pure_module_imports_only_stdlib(self):
        tree = ast.parse(Path(gate.__file__).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            for name in names:
                self.assertIn(name.split(".")[0], sys.stdlib_module_names | {"utopia"})


if __name__ == "__main__":
    unittest.main()
