from __future__ import annotations

from utopia.utils.paths import project_root

import ast

from concurrent.futures import ThreadPoolExecutor, as_completed

from copy import deepcopy

import importlib.util

import json

import logging

from pathlib import Path

import random

import sys

import tempfile

from threading import Event, Lock

import time

from types import ModuleType, SimpleNamespace

import unittest

from unittest.mock import patch

import utopia.funding.sequential as seq

import utopia.funding.validation as gate

from tests.support.sequential_funding import (
    ROOT,
    load_file,
    audit_module,
    body_module,
    definitions,
    native_seed,
    Direction,
    TinyTokenizer,
    response,
    SDK,
    funding_class,
    model_class,
)

class SequentialCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="james-sequential-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        seed_module = ModuleType("utopia.utils.seeding")
        seed_module.derive_seed = native_seed()
        self.patch = patch.dict(sys.modules, {seed_module.__name__: seed_module})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.funding, self.model_type = funding_class(), model_class()
        self.validation = gate.install_funding_validation(
            self.funding, self.directory / "funding.jsonl", compact=True)
        self.handle = seq.install_sequential_funding(
            self.funding, self.model_type, self.directory / "sequential.jsonl")
        self.models = []
        self.addCleanup(self.cleanup_handles)

    def cleanup_handles(self):
        for model in self.models:
            if not model.request_audit._closed:
                model.request_audit.close(False)
        if self.handle.stats["installed"]:
            self.handle.restore()
        self.validation.restore()

    def model(self, script=None, cap=2):
        model = self.model_type()
        model.model_name, model.run_seed, model.enable_thinking = audit_module.MODEL, 42, True
        model.max_concurrent_requests = cap
        model.request_audit = audit_module.RequestAudit(
            self.directory / f"requests{len(self.models)}.jsonl", _tokenizer=TinyTokenizer())
        model.request_audit.bind(model.model_name)
        model.sdk = SDK(script)
        model.client = SimpleNamespace(max_retries=5, chat=SimpleNamespace(completions=model.sdk))
        model.call_stats = {k: 0 for k in (
            "n_prompts", "n_first_attempt_success", "n_retries", "n_failures",
            "elapsed_seconds", "prompt_tokens", "completion_tokens")}
        self.models.append(model)
        return model

    def register(self, n=3, *, year=1, panel_cap=25, program="P", agency_id="NSF", masked=False):
        agency = self.funding()
        agency.id = agency_id
        agency.funding_programs = {program: SimpleNamespace(
            name=program, research_directions=[Direction("algorithms")], funding_rate=.4)}
        if masked:
            from utopia.funding.feedback import record_display_prompt
            original = agency._build_program_prompt
            agency._build_program_prompt = lambda p, apps, papers: record_display_prompt(original, p, apps, papers)
        applications = [{program: {
            "submit": True, "research_proposal": f"Proposal {i}.",
            "author": SimpleNamespace(id=f"researcher_{i}", expertise=[Direction("algorithms")],
                                      university_name="Fixture University"),
            "relevant_projects": [{"status": "accept", "arxiv_id": "private"}],
        }} for i in range(n)]
        values = agency.get_funding_evaluation_prompts(
            applications, {"private": {"title": "HIDDEN_PUBLICATION_RECORD"}},
            panel_max_apps=panel_cap, panel_seed=native_seed()(42, "funding_panels", year))
        return values, agency.funding_programs

    def generate(self, model, values, *, year=1, attempt=0):
        return model.generate_batch(values[0], response_format=values[1], max_tokens=8192,
                                    seed_ctx=("phase5_funding_eval", year, attempt))

    def process(self, results, values, programs):
        self.assertTrue(all(self.funding.validate_funding_result(pair[0], meta["apps"])
                            for pair, meta in zip(results, values[2])))
        log = []
        winners = self.funding.process_funding_evaluation_results(
            results, values[2], programs, application_log=log)
        return winners, log

    def complete(self, model=None):
        if model is None:
            model = self.models[-1] if self.models else self.model()
            if not model.request_audit.summary()["n_requests"]:
                model.generate_batch(["Explicit non-funding CPU fixture"],
                                     seed_ctx=("phase1_fixture", 1, 0))
        if not model.request_audit._closed:
            model.request_audit.close(True)
        self.handle.restore()
        summary = self.handle.summary()
        self.assertTrue(seq.validate_summary(summary))
        return seq.validate_audit(self.handle.path, summary, request_audit_path=model.request_audit.path)

    def test_all_25_model_choices_include_singleton_and_native_bijection(self):
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register(25)
        model = self.model()
        results = self.generate(model, values)
        self.assertEqual(results[0][0], {"ranked_application_ids": list(reversed(range(25)))})
        self.assertEqual(len(model.sdk.calls), 25)
        self.assertEqual(model.sdk.calls[-1]["extra_body"]["structured_outputs"]["json"]
                         ["properties"]["next_application_id"]["enum"], [0])
        winners, log = self.process(results, values, programs)
        self.assertEqual([r["application_id"] for r in winners["P"]], list(reversed(range(15, 25))))
        self.assertEqual([r["position"] for r in log], list(range(1, 26)))
        self.handle.end_year(1)
        report = self.complete(model)
        self.assertEqual(report["request_audit"]["matched_sdk_calls"], 25)
        self.assertEqual(report["processed_panels"][0]["application_to_applicant"]["24"], "researcher_24")
        self.assertEqual(model.call_stats["n_prompts"], 25)
        self.assertEqual(model.call_stats["n_retries"], 0)
        self.assertEqual(self.handle.summary()["processed_panels"], self.validation.summary()["final_panels"])

    def test_three_sdk_cap_retains_successful_prefix_without_native_outer_retry(self):
        def script(payload, index):
            return response('{"next_application_id":2}' if index == 0 else '{"next_application_id":true}')
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register()
        model = self.model(script)
        with patch.object(self.funding, "process_funding_evaluation_results",
                          wraps=self.funding.process_funding_evaluation_results) as process:
            with self.assertRaises(seq.SequentialFundingError):
                self.generate(model, values)
            process.assert_not_called()
        self.assertEqual(len(model.sdk.calls), 4)  # one accepted step + exactly three failed SDK calls
        self.assertEqual(next(iter(self.handle.contexts.values()))["selected"], [2])
        self.assertEqual(self.validation.summary()["final_panels"], 0)
        self.assertFalse(self.handle.summary()["completion_allowed"])
        self.assertEqual(model.call_stats["n_failures"], 1)

    def test_recoverable_errors_do_not_restart_prefix_and_raw_reasoning_retained(self):
        def script(payload, index):
            if index == 0:
                return response('{"next_application_id":1}', finish="length", reasoning="bad-finish-reasoning")
            if index == 1:
                return RuntimeError("scripted SDK transport error")
            remaining = payload["extra_body"]["structured_outputs"]["json"]["properties"]["next_application_id"]["enum"]
            return response(json.dumps({"next_application_id": remaining[-1]}))
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register(2)
        model = self.model(script)
        results = self.generate(model, values)
        self.process(results, values, programs)
        self.handle.end_year(1)
        report = self.complete(model)
        self.assertEqual(self.handle.summary()["sdk_calls"], 4)
        self.assertEqual(self.handle.summary()["retries"], 2)
        rows = [json.loads(line) for line in self.handle.path.read_text().splitlines()]
        raw = [r for r in rows if r["event"] == "sdk_response"]
        self.assertEqual(raw[0]["raw_response"]["choices"][0]["message"]["reasoning_content"], "bad-finish-reasoning")
        self.assertEqual(raw[0]["raw_response"]["choices"][0]["message"]["reasoning"], "alternate field")
        self.assertEqual(report["request_audit"]["matched_sdk_calls"], 4)

    def test_outer_attempt_one_is_fatal_before_any_sdk_call(self):
        self.handle.begin_year(1, ["NSF"])
        values, _ = self.register()
        model = self.model()
        with self.assertRaisesRegex(seq.SequentialFundingError, "whole_panel_retry"):
            self.generate(model, values, attempt=1)
        self.assertEqual(model.sdk.calls, [])

    def test_mutated_returned_permutation_rejected_before_native_processing(self):
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register()
        model = self.model()
        results = self.generate(model, values)
        results[0][0]["ranked_application_ids"].reverse()
        with self.assertRaisesRegex(seq.SequentialFundingError, "differs_from_model"):
            self.funding.process_funding_evaluation_results(results, values[2], programs)
        self.assertEqual(self.validation.summary()["final_panels"], 0)

    def test_metadata_mutation_fails_before_first_call(self):
        self.handle.begin_year(1, ["NSF"])
        values, _ = self.register()
        values[2][0]["apps"][0]["applicant_id"] = "changed"
        model = self.model()
        with self.assertRaisesRegex(seq.SequentialFundingError, "metadata_mutated"):
            self.generate(model, values)
        self.assertFalse(model.sdk.calls)

    def test_p0_frozen_prefix_never_reintroduces_hidden_metadata(self):
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register(3, masked=True)
        frozen = values[0][0].rpartition("\n## Response Format\n")[0]
        self.assertNotIn("HIDDEN_PUBLICATION_RECORD", frozen)
        model = self.model()
        results = self.generate(model, values)
        for call in model.sdk.calls:
            prompt = call["messages"][-1]["content"].split(seq.IMPORTANT_NOTES+"\n\n", 1)[1]
            self.assertEqual(prompt.rpartition("\n## Response Format\n")[0], frozen)
            self.assertNotIn("HIDDEN_PUBLICATION_RECORD", prompt)
            self.assertNotIn('"ranked_application_ids"', prompt)
        self.process(results, values, programs)
        self.handle.end_year(1)
        self.complete(model)

    def test_processed_contexts_cleared_same_prompt_next_year_has_distinct_seed(self):
        model = self.model()
        seeds = []
        for year in (1, 2):
            self.handle.begin_year(year, ["NSF"])
            values, programs = self.register(1, year=year)
            results = self.generate(model, values, year=year)
            seeds.append(model.sdk.calls[-1]["extra_body"]["seed"])
            self.process(results, values, programs)
            self.assertEqual(self.handle.contexts, {})
            self.handle.end_year(year)
        self.assertNotEqual(*seeds)
        report = self.complete(model)
        self.assertEqual([p["year"] for p in report["processed_panels"]], [1, 2])

    def test_duplicate_identical_registered_strings_fail_closed(self):
        self.handle.begin_year(1, ["NSF"])
        agency = self.funding()
        agency.id = "NSF"
        agency.funding_programs = {"P": SimpleNamespace(name="P", research_directions=[Direction("algorithms")],
                                                     funding_rate=.4)}
        author = SimpleNamespace(id="same", expertise=[], university_name="Uni")
        apps = [{"P": {"submit": True, "author": author, "research_proposal": "identical", "relevant_projects": []}}
                for _ in range(2)]
        values = agency.get_funding_evaluation_prompts(
            apps, {}, panel_max_apps=1, panel_seed=native_seed()(42, "funding_panels", 1))
        self.assertEqual(values[0][0], values[0][1])
        model = self.model()
        with self.assertRaisesRegex(seq.SequentialFundingError, "ambiguous"):
            self.generate(model, values)
        self.assertFalse(model.sdk.calls)

    def test_seed_context_ignores_batch_position_and_global_rng(self):
        before = random.getstate()
        self.handle.begin_year(1, ["NSF", "DARPA"])
        first, p1 = self.register(1, program="P")
        second, p2 = self.register(1, program="Q", agency_id="DARPA")
        values = (second[0]+first[0], first[1], second[2]+first[2])
        model = self.model()
        results = self.generate(model, values)
        expected = {native_seed()(42, *seq.selection_seed_context(1, p, 0, 0), 0, 0) for p in ("P", "Q")}
        self.assertEqual({p["extra_body"]["seed"] for p in model.sdk.calls}, expected)
        self.assertEqual(before, random.getstate())
        self.process(results, values, p1 | p2)
        self.handle.end_year(1)
        self.complete(model)

    def test_zero_panels_need_positive_per_agency_year_coverage(self):
        self.handle.begin_year(1, ["NSF", "DARPA"])
        self.register(0)
        self.register(0, program="Q", agency_id="DARPA")
        self.handle.end_year(1)
        report = self.complete()
        self.assertEqual(report["processed_panels"], [])
        self.assertEqual(report["request_audit"]["matched_sdk_calls"], 0)
        self.assertEqual([a["native_submitted_applications"] for a in report["years"][0]["agencies"]], [0, 0])
        self.assertEqual(self.handle.summary()["empty_agency_registrations"], 2)

    def test_missing_agency_or_open_year_cannot_complete(self):
        self.handle.begin_year(1, ["NSF", "DARPA"])
        self.register(0)
        with self.assertRaises(seq.SequentialFundingError):
            self.handle.end_year(1)
        self.assertFalse(self.handle.summary()["completion_allowed"])

    def test_empty_initialization_without_scientific_year_is_valid(self):
        self.assertEqual(self.complete()["years"], [])

    def test_unrelated_native_batch_delegates_exactly(self):
        model = self.model()
        answer = model.generate_batch(["unrelated request"], temperature=.2, system_prompt="different",
                                      max_tokens=80, seed_ctx=("phase2_fixture", 1, 0))
        self.assertEqual(answer[0][0], {"next_application_id": 0})
        self.assertEqual(model.sdk.calls[0]["messages"], [
            {"role": "system", "content": "different"},
            {"role": "user", "content": seq.IMPORTANT_NOTES+"\n\nunrelated request"}])
        self.assertEqual(self.handle.summary()["sdk_calls"], 0)
        self.assertEqual(self.handle.summary()["n_clients"], 0)
        model.request_audit.close(True)
        self.complete()

    def test_cancel_other_panels_between_sdk_calls(self):
        self.handle.begin_year(1, ["NSF", "DARPA"])
        first, _ = self.register(3, program="bad")
        second, _ = self.register(3, program="peer", agency_id="DARPA")
        values = (first[0]+second[0], first[1], first[2]+second[2])
        peer_started = Event()
        def script(payload, index):
            if "for bad." in payload["messages"][-1]["content"]:
                self.assertTrue(peer_started.wait(2))
                return response('{"next_application_id":true}')
            peer_started.set()
            self.assertTrue(self.handle.cancel.wait(2))
            return response('{"next_application_id":2}')
        model = self.model(script, cap=2)
        with self.assertRaises(seq.SequentialFundingError):
            self.generate(model, values)
        self.assertEqual(len(model.sdk.calls), 4)
        self.assertEqual(sum("for peer." in p["messages"][-1]["content"] for p in model.sdk.calls), 1)
        self.assertFalse(model.request_audit.summary()["pending_requests"])

    def test_fatal_received_response_is_preserved_and_same_guard_rethrown(self):
        self.handle.begin_year(1, ["NSF"])
        values, _ = self.register(2)
        model = self.model()
        fatal = audit_module.RequestAuditFailure("scripted terminal accounting failure")
        fatal.received_response = response('{"next_application_id":1}', reasoning="terminal reasoning")
        fatal.request_ticket = {"request_id": 123}
        with patch.object(model, "_create_completion", side_effect=fatal):
            with self.assertRaises(audit_module.RequestAuditFailure) as caught:
                self.generate(model, values)
        self.assertIs(caught.exception, fatal)
        rows = [json.loads(line) for line in self.handle.path.read_text().splitlines()]
        terminal = next(r for r in rows if r["event"] == "terminal_guard")
        self.assertEqual(terminal["raw_response"]["choices"][0]["message"]["reasoning_content"], "terminal reasoning")
        self.assertEqual(terminal["request_ticket"], {"request_id": 123})
        self.assertEqual(self.handle.summary()["sdk_calls"], 1)

    def test_pre_http_fatal_records_no_invented_response(self):
        self.handle.begin_year(1, ["NSF"])
        values, _ = self.register(1)
        model = self.model()
        fatal = audit_module.RequestAuditFailure("scripted input overflow")
        with patch.object(model, "_create_completion", side_effect=fatal):
            with self.assertRaises(audit_module.RequestAuditFailure):
                self.generate(model, values)
        rows = [json.loads(line) for line in self.handle.path.read_text().splitlines()]
        terminal = next(r for r in rows if r["event"] == "terminal_guard")
        self.assertIsNone(terminal["raw_response"])
        self.assertIsNone(terminal["request_ticket"])

    def test_actual_request_guard_bad_usage_preserves_returned_reasoning_without_retry(self):
        def script(payload, index):
            returned = response('{"next_application_id":0}', reasoning="bad-usage terminal reasoning")
            returned.usage.prompt_tokens = 1  # Wrong exact input count.
            return returned
        self.handle.begin_year(1, ["NSF"])
        values, _ = self.register(1)
        model = self.model(script)
        with self.assertRaises(audit_module.RequestAuditFailure) as caught:
            self.generate(model, values)
        self.assertTrue(hasattr(caught.exception, "received_response"))
        self.assertEqual(len(model.sdk.calls), 1)
        self.assertEqual(model.request_audit.summary()["guard_failures"], 1)
        self.assertEqual(model.request_audit.summary()["pending_requests"], 0)
        rows = [json.loads(line) for line in self.handle.path.read_text().splitlines()]
        terminal = next(r for r in rows if r["event"] == "terminal_guard")
        self.assertEqual(terminal["raw_response"]["choices"][0]["message"]["reasoning_content"],
                         "bad-usage terminal reasoning")
        self.assertEqual(terminal["request_ticket"]["request_id"], 0)

    def test_audit_failure_is_fatal_before_any_sdk_request(self):
        self.handle.begin_year(1, ["NSF"])
        values, _ = self.register(1)
        model = self.model()
        with patch.object(seq.os, "write", side_effect=OSError("scripted disk full")):
            with self.assertRaises(seq.SequentialFundingError):
                self.generate(model, values)
        self.assertFalse(model.sdk.calls)
        self.assertGreater(self.handle.summary()["audit_failures"], 0)

    def test_restore_conflict_leaves_foreign_method_and_is_unclean(self):
        foreign = lambda *args, **kwargs: []
        self.model_type.generate_batch = foreign
        with self.assertRaisesRegex(seq.SequentialFundingError, "restore_conflict"):
            self.handle.restore()
        self.assertIs(self.model_type.generate_batch, foreign)
        with self.assertRaises(seq.SequentialFundingError):
            seq.validate_summary(self.handle.summary())

    def test_restoration_preserves_compact_guard_then_original_descriptors(self):
        original = self.handle.saved[(self.funding, "get_funding_evaluation_prompts")]
        self.handle.restore()
        self.assertIs(self.funding.get_funding_evaluation_prompts, original)
        self.assertTrue(self.validation.summary()["installed"])

    def test_existing_audit_or_summary_is_not_clobbered(self):
        self.handle.restore()
        before = self.handle.path.read_bytes()
        with self.assertRaises(seq.SequentialFundingError):
            seq.install_sequential_funding(self.funding, self.model_type, self.handle.path)
        self.assertEqual(self.handle.path.read_bytes(), before)

    def test_independent_audit_rejects_tampered_raw_prefix_map_order_counts_and_request_join(self):
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register(2)
        model = self.model()
        self.process(self.generate(model, values), values, programs)
        self.handle.end_year(1)
        self.complete(model)
        original = self.handle.path.read_text()
        for event, field, value in (
            ("sdk_result", "raw_content", '{"next_application_id":0}'),
            ("sdk_call", "remaining", [0]),
            ("panel_registered", "map_sha256", "0"*64),
            ("batch_processed", "rankings", [[0, 1]]),
        ):
            rows = [json.loads(line) for line in original.splitlines()]
            next(r for r in rows if r["event"] == event)[field] = value
            self.handle.path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaises(seq.SequentialFundingError):
                seq.validate_audit(self.handle.path, self.handle.summary(),
                                   request_audit_path=model.request_audit.path)
        self.handle.path.write_text(original)
        request_path = model.request_audit.path
        raw = request_path.read_text()
        rows = [json.loads(line) for line in raw.splitlines()]
        next(r for r in rows if r["event"] == "request_started")["request_seed"] += 1
        request_path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaises(seq.SequentialFundingError):
            seq.validate_audit(self.handle.path, self.handle.summary(),
                               request_audit_path=model.request_audit.path)

    def test_request_join_rejects_unknown_funding_duplicate_missing_late_terminal_and_token_count_forgery(self):
        self.handle.begin_year(1, ["NSF"])
        values, programs = self.register(1)
        model = self.model()
        self.process(self.generate(model, values), values, programs)
        self.handle.end_year(1)
        self.complete(model)
        path = model.request_audit.path
        summary_path = path.with_suffix(".summary.json")
        raw, summary_raw = path.read_text(), summary_path.read_text()
        for corruption in ("legacy", "unknown_namespace", "duplicate_terminal", "missing_terminal",
                           "post_finalization", "global_counts", "input_count", "reservation", "context_overflow"):
            with self.subTest(corruption=corruption):
                rows = [json.loads(line) for line in raw.splitlines()]
                start = next(r for r in rows if r["event"] == "request_started")
                terminal = next(r for r in rows if r["event"] == "response")
                if corruption == "legacy":
                    start["seed_ctx"] = ["phase5_funding_eval", 1, 0]
                elif corruption == "unknown_namespace":
                    start["seed_ctx"][3] = "unrecognized"
                elif corruption == "duplicate_terminal":
                    rows.insert(-1, deepcopy(terminal))
                elif corruption == "missing_terminal":
                    rows.remove(terminal)
                elif corruption == "post_finalization":
                    rows.append(deepcopy(terminal))
                elif corruption == "global_counts":
                    rows[-1]["n_requests"] += 1
                    forged = json.loads(summary_raw)
                    forged["n_requests"] += 1
                    summary_path.write_text(json.dumps(forged))
                elif corruption == "input_count":
                    start["input_tokens"] += 1
                    start["reserved_total_tokens"] += 1
                elif corruption == "reservation":
                    start["reserved_total_tokens"] += 1
                else:
                    start["input_tokens"] = 32768
                    start["reserved_total_tokens"] = 32768+8192
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
                with self.assertRaises(seq.SequentialFundingError):
                    seq.validate_audit(self.handle.path, self.handle.summary(),
                                       request_audit_path=model.request_audit.path)
                summary_path.write_text(summary_raw)
        path.write_text(raw)


    def test_zero_panels_still_join_request_audit_and_reject_legacy_funding_call(self):
        self.handle.begin_year(1, ["NSF"])
        self.register(0)
        self.handle.end_year(1)
        report = self.complete()
        model = self.models[-1]
        self.assertEqual(report["request_audit"]["path"], str(model.request_audit.path))
        self.assertEqual(report["request_audit"]["matched_sdk_calls"], 0)
        rows = [json.loads(line) for line in model.request_audit.path.read_text().splitlines()]
        next(r for r in rows if r["event"] == "request_started")["seed_ctx"] = ["phase5_funding_eval", 1, 0]
        model.request_audit.path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaises(seq.SequentialFundingError):
            seq.validate_audit(self.handle.path, self.handle.summary(),
                               request_audit_path=model.request_audit.path)
        with self.assertRaises(TypeError):
            seq.validate_audit(self.handle.path, self.handle.summary())

class PureHelpers(unittest.TestCase):
    def test_schema_and_decoder_strict_enum_key_type_and_duplicate_keys(self):
        self.assertEqual(seq.selection_schema([2, 0])["properties"]["next_application_id"]["enum"], [2, 0])
        self.assertEqual(seq.decode_selection(' {"next_application_id":2} ', [2, 0]), 2)
        for content in (
            '{"next_application_id":true}', '{"next_application_id":"2"}', '{"next_application_id":2.0}',
            '{"next_application_id":1}', '{"next_application_id":2,"extra":0}',
            '{"next_application_id":2,"next_application_id":0}', '```json\n{"next_application_id":2}\n```',
            "", "[]", '{"next_application_id":NaN}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                seq.decode_selection(content, [0, 2])
        for values in ([], [0, 0], [True], [25], [0.0]):
            with self.assertRaises(ValueError):
                seq.selection_schema(values)

    def test_native_seed_algorithm_and_disjoint_probe_namespace(self):
        before = random.getstate()
        production = seq.selection_seed_context(1, "P", 0, 0)
        probe = seq.selection_seed_context(0, "P", 0, 0, namespace="technical_probe")
        self.assertNotEqual(production, probe)
        for context in (production, probe):
            self.assertEqual(seq._derive_seed(42, *context, 0, 2), native_seed()(42, *context, 0, 2))
        self.assertEqual(before, random.getstate())
        with self.assertRaises(ValueError):
            seq.selection_seed_context(0, "P", 0, 0)

    def test_install_without_compact_guard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"untouched.jsonl"
            with self.assertRaises(seq.SequentialFundingError):
                seq.install_sequential_funding(funding_class(), model_class(), path)
            self.assertFalse(path.exists())

if __name__ == "__main__":
    unittest.main()
