from __future__ import annotations

from tests.support.simulation import initialize_actual_simulator, restore_actual_simulator, make_actual_simulation

import utopia.analysis.funding_feedback as feedback_analysis
import utopia.funding.feedback as feedback
import utopia.runtime.provenance as provenance

from utopia.utils.paths import project_root

import ast

from collections import Counter, defaultdict

from contextlib import contextmanager, redirect_stdout

from copy import deepcopy

import io

import gzip

import json

import logging

import os

from pathlib import Path

import random

import re

import sys

import tempfile

import time

from types import ModuleType, SimpleNamespace

from typing import Dict, List, Set, Union

import unittest

from unittest.mock import patch

import numpy as np

import pandas as pd

import utopia.experiments.funding_feedback as jm

import utopia.funding.sequential as sequential

from utopia.config import SIMULATION_CONFIG

from utopia.metrics.tracker import calculate_gini_coefficient

from tests.support.simulation import (
    ROOT,
    original_definitions,
    SchemaStub,
    DirectionStub,
    original_namespace,
    ORIGINAL,
    UniversityResearcher,
    FundingAgency,
    Simulation,
    FixtureLLM,
    make_simulation,
    run_phase,
    make_panel_simulation,
    panel_seed_import,
    FullYearScriptedLLM,
    CompactFundingScriptedLLM,
    SequentialScriptedLLM,
    fixture_request_scope,
    write_nonfunding_request_fixture,
    zero_sequential_fixture,
)

class TestOriginalPhase(unittest.TestCase):
    def test_panel25_control_matches_stock_requests_state_rng_and_quotas(self):
        from utopia.funding.validation import install_funding_validation
        outputs = []
        states = []
        requests = []
        logs = []
        for base in (True, False):
            with tempfile.TemporaryDirectory() as directory, panel_seed_import():
                sim, agents, agency = make_panel_simulation(directory, base=base)
                random.seed(42)
                np.random.seed(42)
                handle = install_funding_validation(
                    FundingAgency, Path(directory) / "audit.jsonl", compact=True)
                try:
                    result = run_phase(sim)
                finally:
                    handle.restore()
                summary = handle.summary()
                self.assertEqual(summary["invalid_final_panels"], 0)
                self.assertEqual(summary["final_panels"], 6)
                self.assertEqual(summary["failed_batches"], 0)
                self.assertEqual(summary["restore_conflicts"], [])
                self.assertTrue(summary["completion_allowed"])
                self.assertEqual(summary["output_representation"], "ordered_application_ids_v1")
                result.pop("funding_feedback", None)
                outputs.append(result)
                requests.append(sim.llm.calls)
                states.append((
                    [(a.id, a.resources, a.is_active, a.funding_success_history) for a in agents],
                    sim.funding_tracker.funding_allocation_per_cycle,
                    random.getstate(), np.random.get_state(),
                ))
                rows = [json.loads(line) for line in
                        Path(directory, "funding_applications_year_1.jsonl").read_text().splitlines()]
                logs.append(rows)
                self.assertEqual(len(rows), 106)
                for pid, expected_winners in (("PROGRAM_A", 11), ("PROGRAM_B", 3)):
                    program_rows = [r for r in rows if r["program_id"] == pid]
                    counts = Counter(r["panel_index"] for r in program_rows)
                    self.assertEqual(sorted(counts.values()), [17, 18, 18])
                    self.assertEqual({r["applicant_id"] for r in program_rows}, {a.id for a in agents})
                    self.assertEqual(sum(r["funded"] for r in program_rows), expected_winners)
                    self.assertFalse(any(r["imputed_tail"] or r["fallback_ranking"] for r in program_rows))
                evaluations = [call for call in sim.llm.calls if call[1][0] == "phase5_funding_eval"]
                self.assertEqual(len(evaluations), 1)
                self.assertEqual(len(evaluations[0][0]), 6)
                self.assertEqual(evaluations[0][2]["max_tokens"], 8192)
                self.assertEqual(sim.args.funding_budget_mode if hasattr(sim.args, "funding_budget_mode")
                                 else sim.funding_budget_mode, "track")
                self.assertEqual(sum(a.resources for a in agents), 5300 + 14 * 20)
                self.assertNotIn("_build_program_prompt", vars(agency))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(states[0][:3], states[1][:3])
        np.testing.assert_equal(states[0][3], states[1][3])

    def test_panel25_all_cells_keep_grouping_rates_and_factual_awards(self):
        from utopia.funding.validation import install_funding_validation
        histories = []
        assignments = []
        for cell in feedback.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, panel_seed_import():
                sim, agents, _ = make_panel_simulation(directory, cell)
                handle = install_funding_validation(
                    FundingAgency, Path(directory) / "audit.jsonl", compact=True)
                try:
                    result = run_phase(sim)
                finally:
                    handle.restore()
                feed = feedback.Mechanisms.from_cell(cell).awards_feed_resources
                events = result["funding_feedback"]["awards"]
                self.assertEqual(sum(e["earned_amount"] for e in events), 280)
                self.assertEqual(sum(e["spendable_credit"] for e in events), 280 if feed else 0)
                self.assertEqual(sum(a.resources for a in agents), 5300 + (280 if feed else 0))
                self.assertEqual(sim.funding_tracker.funding_allocation_per_cycle[1]["university"], 280)
                histories.append([a.funding_success_history for a in agents])
                rows = [json.loads(line) for line in
                        Path(directory, "funding_applications_year_1.jsonl").read_text().splitlines()]
                assignments.append([(r["program_id"], r["panel_index"], r["applicant_id"]) for r in rows])
                self.assertEqual(handle.summary()["final_panels"], 6)
                self.assertTrue(handle.summary()["completion_allowed"])
        self.assertTrue(all(rows == assignments[0] for rows in assignments))
        self.assertTrue(all(rows == histories[0] for rows in histories))

    def test_panel25_strict_final_failure_aborts_without_repaired_awards(self):
        from utopia.funding.validation import install_funding_validation, FundingRankingValidationError
        class IncompleteRanking(CompactFundingScriptedLLM):
            def generate_batch(self, prompts, seed_ctx, **kwargs):
                values = super().generate_batch(prompts, seed_ctx, **kwargs)
                if seed_ctx[0] == "phase5_funding_eval":
                    for value, _ in values:
                        value["ranked_application_ids"].pop()
                return values
        with tempfile.TemporaryDirectory() as directory, panel_seed_import():
            sim, agents, agency = make_panel_simulation(directory, "P0F0")
            sim.llm = IncompleteRanking()
            handle = install_funding_validation(
                FundingAgency, Path(directory) / "audit.jsonl", compact=True)
            try:
                with self.assertRaises(FundingRankingValidationError):
                    run_phase(sim)
            finally:
                handle.restore()
            self.assertEqual(len([c for c in sim.llm.calls if c[1][0] == "phase5_funding_eval"]), 3)
            self.assertEqual(handle.summary()["invalid_final_panels"], 6)
            self.assertEqual(handle.summary()["failed_batches"], 1)
            self.assertFalse(handle.summary()["completion_allowed"])
            self.assertTrue(all(a.resources == 100 and not a.funding_success_history for a in agents))
            self.assertEqual(sim.funding_feedback_years, {})
            self.assertNotIn("_build_program_prompt", vars(agency))
            self.assertTrue(all("update_resources" not in vars(a) for a in agents))

    def test_control_equals_original_state_prompts_rankings_and_rng(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            original, oa, _ = make_simulation(a, base=True)
            wrapped, wa, agency = make_simulation(b)
            random.seed(8401)
            before_rng = random.getstate()
            expected = run_phase(original)
            after_original = random.getstate()
            random.setstate(before_rng)
            actual = run_phase(wrapped)
            self.assertEqual(after_original, random.getstate())
            actual.pop("funding_feedback")
            self.assertEqual(expected, actual)
            self.assertEqual(original.llm.calls, wrapped.llm.calls)
            self.assertEqual([(x.resources, x.funding_success_history) for x in oa],
                             [(x.resources, x.funding_success_history) for x in wa])
            self.assertNotIn("_build_program_prompt", vars(agency))
            for agent in wa:
                self.assertNotIn("update_resources", vars(agent))
            self.assertEqual(
                Path(a, "funding_applications_year_1.jsonl").read_text(),
                Path(b, "funding_applications_year_1.jsonl").read_text())

    def test_all_four_cells_preserve_awards_and_costs_but_gate_credit(self):
        for cell in feedback.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                sim, agents, _ = make_simulation(directory, cell, cost=3)
                result = run_phase(sim)
                feed = feedback.Mechanisms.from_cell(cell).awards_feed_resources
                # Two actual awards of 20, two actual application costs of 3.
                self.assertEqual(agents[0].resources, 94 + (40 if feed else 0))
                self.assertEqual(agents[1].resources, 94)
                events = result["funding_feedback"]["awards"]
                self.assertEqual(sum(e["earned_amount"] for e in events), 40)
                self.assertEqual(sum(e["spendable_credit"] for e in events), 40 if feed else 0)
                self.assertEqual(sim.funding_tracker.funding_allocation_per_cycle[1]["university"], 40)
                self.assertEqual(len(agents[0].funding_success_history), 2)
                proposal_prompts = sim.llm.calls[0][0]
                ranking_prompts = sim.llm.calls[1][0]
                self.assertIn("1 accepted papers", proposal_prompts[0])
                self.assertEqual("Recent accepted papers:" in ranking_prompts[0], cell[1] == "1")
                self.assertEqual("Publication record withheld" in ranking_prompts[0], cell[1] == "0")
                self.assertIn("Test a new algorithm.", ranking_prompts[0])

    def test_suppression_precedes_original_attrition(self):
        for feed in (True, False):
            with self.subTest(feed=feed), tempfile.TemporaryDirectory() as directory:
                sim, agents, _ = make_simulation(directory, "P1F1" if feed else "P1F0",
                                                 balance=12, cost=1)
                run_phase(sim)
                self.assertEqual(agents[0].resources, 50 if feed else 10)
                self.assertEqual(agents[0].is_active, feed)
                self.assertFalse(agents[1].is_active)
                self.assertEqual(sum(e["amount"] for es in agents[0].funding_success_history.values()
                                     for e in es), 40)

    def test_full_funding_phase_preserves_exact_zero_discontinuity_in_all_cells(self):
        for cell in feedback.CELLS:
            # Two real application debits bring the losing applicant to the
            # target. Exact zero survives; positive 1..10 does not.
            for target in (0, 1, 10, 11):
                with self.subTest(cell=cell, target=target), tempfile.TemporaryDirectory() as d:
                    sim, agents, _ = make_simulation(d, cell, balance=target + 2, cost=1)
                    result = run_phase(sim)
                    self.assertEqual(agents[1].resources, target)
                    self.assertEqual(agents[1].is_active, target == 0 or target > 10)
                    feed = feedback.Mechanisms.from_cell(cell).awards_feed_resources
                    self.assertEqual(agents[0].resources, target + (40 if feed else 0))
                    self.assertEqual(agents[0].is_active, feed or target == 0 or target > 10)
                    audit = result["funding_feedback"]["legacy_zero_boundary"]
                    expected = (1 if feed else 2) if target == 0 else 0
                    self.assertEqual(audit["active_zero_count"], expected)
                    self.assertEqual(audit["active_zero_fraction_of_cohort"], expected / 2)

    def test_overshoot_deactivation_is_not_confused_with_active_zero(self):
        for cell in feedback.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as d:
                sim, agents, _ = make_simulation(d, cell, balance=1)
                for agent in agents:
                    agent.update_resources(-2)
                result = run_phase(sim)
                self.assertTrue(all(not a.is_active and a.resources == 0 for a in agents))
                audit = result["funding_feedback"]["legacy_zero_boundary"]
                self.assertEqual(audit["zero_resource_count"], 2)
                self.assertEqual(audit["active_zero_count"], 0)
                self.assertIsNone(audit["active_zero_fraction_of_active"])
                self.assertEqual(sim.llm.calls, [])

    def test_restore_all_instance_methods_on_llm_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, agency = make_simulation(directory, "P0F0")
            sim.llm.fail_evaluation = True
            # Also exercise restoration of an already-owned instance attribute.
            existing = agents[1].update_resources
            agents[1].update_resources = existing
            with self.assertRaisesRegex(RuntimeError, "scripted evaluation"):
                run_phase(sim)
            self.assertNotIn("_build_program_prompt", vars(agency))
            self.assertNotIn("update_resources", vars(agents[0]))
            self.assertIs(agents[1].update_resources, existing)
            self.assertEqual(sim.funding_feedback_years, {})

    def test_existing_history_untouched_and_multiple_awards_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, _ = make_simulation(directory, "P0F0")
            prior = {"year": 0, "amount": 7}
            agents[0].funding_success_history = {"PROGRAM_A": [prior.copy()]}
            result = run_phase(sim)
            self.assertEqual(agents[0].funding_success_history["PROGRAM_A"][0], prior)
            self.assertEqual(len(result["funding_feedback"]["awards"]), 2)
            self.assertEqual(agents[0].resources, 100)

    def test_no_resume_of_unlabeled_checkpoint(self):
        with self.assertRaisesRegex(RuntimeError, "fresh worlds"):
            feedback.simulation_class(Simulation).load_checkpoint(None, 3)

    def test_empty_paper_year_reuses_funding_logic_with_required_frame_schema(self):
        for cell in feedback.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                sim, agents, _ = make_simulation(directory, cell)
                getter = lambda year: pd.DataFrame()
                sim.paper_tracker.get_papers_dataframe = getter
                result = run_phase(sim)
                self.assertIs(sim.paper_tracker.get_papers_dataframe, getter)
                self.assertEqual(sum(e["earned_amount"]
                                     for e in result["funding_feedback"]["awards"]), 40)
                self.assertEqual(agents[0].resources, 140 if cell[3] == "1" else 100)

    def test_no_applicants_retains_zero_awards_without_inventing_funding(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, _ = make_simulation(directory, "P0F0")
            for agent in agents:
                agent.is_active = False
            sim.paper_tracker.get_papers_dataframe = lambda year: pd.DataFrame()
            result = run_phase(sim)
            self.assertEqual(sim.llm.calls, [])
            self.assertEqual(result["funding_feedback"]["awards"], [])
            self.assertEqual([a.resources for a in agents], [100, 100])

    def test_zero_amount_awards_remain_real_awards(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, _ = make_simulation(directory, "P1F0")
            with patch.dict(SIMULATION_CONFIG["funding"], academic_base_budget=0):
                result = run_phase(sim)
            self.assertEqual(len(result["funding_feedback"]["awards"]), 2)
            self.assertEqual(sum(e["earned_amount"]
                                 for e in result["funding_feedback"]["awards"]), 0)
            self.assertEqual(len(agents[0].funding_success_history), 2)

    def test_next_year_resources_diverge_but_award_history_remains_factual(self):
        for cell in ("P1F1", "P1F0"):
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                sim, agents, _ = make_simulation(directory, cell)
                run_phase(sim)
                sim.funding_tracker.start_cycle(2, sim.ecosystem.agent_population)
                with redirect_stdout(io.StringIO()):
                    sim._run_phase_5_update_funding(2, [], {})
                self.assertEqual(agents[0].resources, 180 if cell == "P1F1" else 100)
                history = [e for entries in agents[0].funding_success_history.values()
                           for e in entries]
                self.assertEqual(sum(e["amount"] for e in history), 80)
                self.assertEqual(Counter(e["year"] for e in history), {1: 2, 2: 2})
                next_proposal = sim.llm.calls[2][0][0]
                self.assertIn(f"Current funding: {140 if cell == 'P1F1' else 100}", next_proposal)

class TestPromptAndResourceBoundaries(unittest.TestCase):
    def test_empty_retrieval_never_calls_encoder_and_nonempty_is_unmodified(self):
        from unittest.mock import Mock
        original = Mock(return_value={"sentinel": object()})
        empty = feedback.empty_safe_retrieve(original, [], retrieval_type="submission")
        self.assertEqual(empty["topk_indices"].shape, (0, 0))
        self.assertEqual(empty["topk_indices"].dtype, np.int64)
        original.assert_not_called()
        actual = feedback.empty_safe_retrieve(original, ["query"], retrieval_type="submission")
        self.assertIs(actual, original.return_value)
        original.assert_called_once_with(["query"], retrieval_type="submission")

    def test_record_mask_preserves_inputs_proposal_identity_and_affiliation(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, agency = make_simulation(directory)
            app = {
                "applicant_id": agents[0].id, "author": agents[0],
                "research_proposal": "My accepted work inspires this project.",
                "relevant_projects": [{"arxiv_id": "paper_0", "status": "accept"}],
            }
            project_list = deepcopy(app["relevant_projects"])
            prompt = feedback.record_display_prompt(
                agency._build_program_prompt, agency.funding_programs["PROGRAM_A"],
                [app], {"paper_0": {"title": "Structured title sentinel"}})
            self.assertNotIn("Structured title sentinel", prompt)
            self.assertNotIn("Recent accepted papers:", prompt)
            self.assertNotIn("The agent has no research projects completed yet.", prompt)
            self.assertNotIn("Past performance and track record.", prompt)
            # Intentional retained path, not a falsely claimed full blind.
            self.assertIn("My accepted work inspires this project.", prompt)
            self.assertIn("Applicant ID: founder_0", prompt)
            self.assertIn("Affiliation: institution_0", prompt)
            self.assertEqual(app["relevant_projects"], project_list)
            self.assertIs(app["author"], agents[0])

    def test_upstream_prompt_drift_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "Funding prompt changed"):
            feedback.record_display_prompt(lambda *args: "new incompatible prompt", None, [{}], {})

    def test_negative_costs_and_zero_threshold_reuse_original_semantics(self):
        agent = UniversityResearcher("a", "i", funding_level=4,
                                    expertise=[DirectionStub("algorithms")])
        gate = feedback.AwardResourceGate(agent, enabled=False)
        gate(20)
        self.assertEqual(agent.resources, 4)
        gate(-4)
        self.assertEqual(agent.resources, 0)
        self.assertTrue(agent.is_active)  # Original exact-zero behavior preserved.
        gate(-1)
        self.assertFalse(agent.is_active)
        self.assertEqual(agent.resources, 0)
        gate(0)
        self.assertEqual(gate.award_calls, [20.0, 0.0])

    def test_history_rewrite_rejected(self):
        agent = SimpleNamespace(id="a", funding_success_history={"p": [{"year": 1, "amount": 20}]})
        with self.assertRaisesRegex(RuntimeError, "rewrote existing"):
            feedback.new_awards(agent, {"p": [{"year": 0, "amount": 20}]}, 1)

    def test_unrelated_positive_credit_is_detected(self):
        class ChangedUpstream:
            def _run_phase_5_update_funding(self, year, submissions, result):
                self.ecosystem.agent_population["a"].update_resources(5)
        sim = object.__new__(feedback.simulation_class(ChangedUpstream))
        sim.mechanisms = feedback.Mechanisms(False, False)
        sim.funding_feedback_years = {}
        sim.paper_tracker = SimpleNamespace(get_papers_dataframe=lambda year: pd.DataFrame())
        agent = UniversityResearcher("a", "i", expertise=[DirectionStub("algorithms")])
        sim.ecosystem = SimpleNamespace(agent_population={"a": agent})
        with self.assertRaisesRegex(RuntimeError, "do not match earned awards"):
            sim._run_phase_5_update_funding(1, [], {})
        self.assertNotIn("update_resources", vars(agent))

    def test_p0_keeps_evaluation_metadata_and_application_order(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, agency = make_simulation(directory)
            applications = [{"PROGRAM_A": {
                "submit": True, "author": agent, "research_proposal": "Proposal.",
                "relevant_projects": [{"arxiv_id": f"paper_{i}", "status": "accept"}],
            }} for i, agent in enumerate(agents)]
            papers = {f"paper_{i}": {"title": str(i)} for i in range(2)}
            visible, schema, expected = agency.get_funding_evaluation_prompts(
                applications, papers, panel_max_apps=0, panel_seed=8401)
            builder = agency._build_program_prompt
            with feedback.instance_override(agency, "_build_program_prompt",
                                      lambda *args: feedback.record_display_prompt(builder, *args)):
                hidden, actual_schema, actual = agency.get_funding_evaluation_prompts(
                    applications, papers, panel_max_apps=0, panel_seed=8401)
            self.assertNotEqual(visible, hidden)
            self.assertEqual(schema, actual_schema)
            self.assertEqual(expected, actual)
            self.assertIs(expected[0]["apps"][0], actual[0]["apps"][0])

class TestProtocolAndAnalysis(unittest.TestCase):
    def test_request_completion_gate_requires_requests_and_no_guard_failures(self):
        good = {"status": "complete", "n_clients": 1, "n_requests": 1,
                "guard_failures": 0, "pending_requests": 0}
        provenance.validate_request_audit_summary(good)
        for key, value in (("status", "failed"), ("n_clients", 0), ("n_clients", 2),
                           ("guard_failures", 1), ("pending_requests", 1),
                           ("n_requests", 0), ("n_requests", True)):
            with self.subTest(key=key, value=value), self.assertRaises(RuntimeError):
                provenance.validate_request_audit_summary({**good, key: value})



    def test_server_provenance_rejects_mismatched_runtime_or_endpoint(self):
        record = {
            "model": provenance.MODEL, "revision": provenance.MODEL_REVISION, "dtype": "bfloat16",
            "tensor_parallel_size": 2, "seed": 42, "request_seed_base": 42,
            "request_seed_policy": "derive_seed(run_seed, phase_context, item_index, retry)",
            "vllm_version": "0.12.0", "transformers_version": "4.57.3",
            "torch_version": "2.9.0+cu128", "max_model_len": 32768,
            "gpu_memory_utilization": 0.8, "reasoning_parser": "qwen3", "max_num_seqs": 128,
            "endpoint": "http://localhost:8000/v1", "server_gpus": [3, 5], "host": "test-host",
            "pid": 123, "launch_command": ["vllm", "serve"], "started_utc": "test-only",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.json"
            provenance.write_json(path, record)
            first = provenance.server_provenance(path, record["endpoint"])
            record["server_gpus"] = [6, 7]
            provenance.write_json(path, record)
            self.assertEqual(first["runtime_hash"],
                             provenance.server_provenance(path, record["endpoint"])["runtime_hash"])
            with self.assertRaisesRegex(ValueError, "endpoint mismatch"):
                provenance.server_provenance(path, "http://localhost:8001/v1")
            record["dtype"] = "float16"
            provenance.write_json(path, record)
            with self.assertRaisesRegex(ValueError, "dtype"):
                provenance.server_provenance(path, record["endpoint"])

    def test_tp1_server_requires_explicit_uniform_runtime_profile(self):
        from utopia.runtime.server_profiles import TP1_PLAN_PROFILE, TP1_PROFILE_NAME, tp1_server_spec
        record = dict(tp1_server_spec(), server_profile=TP1_PLAN_PROFILE,
                      endpoint="http://127.0.0.1:18084/v1", gpu_ids=[7],
                      host="test-host", pid=123, launch_command=["vllm", "serve"],
                      started_utc="test-only")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tp1-server.json"
            provenance.write_json(path, record)
            reference = provenance.server_provenance(
                path, record["endpoint"], runtime_profile=TP1_PROFILE_NAME)
            # Existing mechanism callers remain TP2-only without explicit opt-in.
            with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
                provenance.server_provenance(path, record["endpoint"])
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                provenance.server_provenance(path, record["endpoint"], runtime_profile="unreviewed")
            for key, bad in (
                ("tensor_parallel_size", 2), ("max_num_seqs", 128),
                ("gpu_memory_utilization", .8), ("max_model_len", 16384),
                ("dtype", "float16"), ("enforce_eager", False),
                ("enforce_eager", 1), ("max_num_batched_tokens", 4096),
                ("enable_chunked_prefill", False), ("runtime_profile", None),
                ("server_profile", None), ("gpu_ids", [6, 7]),
            ):
                with self.subTest(key=key, value=bad):
                    changed = dict(record)
                    changed[key] = bad
                    provenance.write_json(path, changed)
                    with self.assertRaises(ValueError):
                        provenance.server_provenance(
                            path, record["endpoint"], runtime_profile=TP1_PROFILE_NAME)
            record["gpu_ids"] = [3]
            provenance.write_json(path, record)
            self.assertEqual(reference["runtime_hash"], provenance.server_provenance(
                path, record["endpoint"], runtime_profile=TP1_PROFILE_NAME)["runtime_hash"])

    def test_corpus_hash_handles_original_dates_and_detects_content_and_order(self):
        documents = [
            SimpleNamespace(page_content="Abstract A.", metadata={
                "id": "a", "published": pd.Timestamp("2016-01-01")}),
            SimpleNamespace(page_content="Abstract B.", metadata={
                "id": "b", "published": np.datetime64("2016-01-02")}),
        ]
        expected = provenance.corpus_fingerprint(documents)
        self.assertEqual(expected, provenance.corpus_fingerprint(deepcopy(documents)))
        self.assertNotEqual(expected, provenance.corpus_fingerprint(list(reversed(documents))))
        documents[0].page_content = "A changed abstract."
        self.assertNotEqual(expected, provenance.corpus_fingerprint(documents))

    def test_fixed_population_blueprints_do_not_depend_on_switches(self):
        ORIGINAL["AVAILABLE_DIRECTIONS"] = [DirectionStub(str(i)) for i in range(20)]
        for seed in feedback.SEEDS:
            state = random.getstate()
            worlds = []
            for cell in feedback.CELLS:
                feedback.Mechanisms.from_cell(cell)
                world = ORIGINAL["build_population_blueprint"](
                    20, 5, seed, strategy_mix=["balanced"] * 5)
                worlds.append(world)
            self.assertEqual(random.getstate(), state)
            self.assertTrue(all(world == worlds[0] for world in worlds))
            self.assertEqual(len(worlds[0]), 100)
            self.assertEqual(len({r["researcher_name"] for r in worlds[0]}), 100)
            self.assertEqual({r["strategy"] for r in worlds[0]}, {"balanced"})


    def test_fixed_cohort_metrics_include_inactive_and_rejected_paper_citations(self):
        rows = [
            dict(active=True, cumulative_earned_funding=40, spendable_resources=100,
                 accepted_papers=1, citations_all_papers=2),
            dict(active=False, cumulative_earned_funding=0, spendable_resources=0,
                 accepted_papers=0, citations_all_papers=8),
        ]
        metrics = feedback.summarize_agents(rows)
        self.assertEqual(metrics["founder_count"], 2)
        self.assertEqual(metrics["active_fraction"], 0.5)
        self.assertEqual(metrics["mean_spendable_resources"], 50)
        self.assertEqual(metrics["cumulative_earned_funding_gini"], 0.5)
        self.assertEqual(metrics["citations_all_papers_per_founder"], 5)
        for row in rows:
            row["spendable_resources"] = 0
        self.assertIsNone(feedback.summarize_agents(rows)["spendable_resources_gini"])

    def test_contrast_signs_and_interaction(self):
        got = feedback_analysis.paired_contrasts(dict(zip(feedback.CELLS, [10, 7, 6, 5])))
        self.assertEqual(got, {
            "disable_publication_at_F1": -3, "disable_publication_at_F0": -1,
            "disable_resources_at_P1": -4, "disable_resources_at_P0": -2, "interaction": 2,
        })
        with self.assertRaises(ValueError):
            feedback_analysis.paired_contrasts({"P1F1": 1})
        undefined = dict(zip(feedback.CELLS, [10, None, 6, 5]))
        partial = feedback_analysis.paired_contrasts(undefined)
        self.assertIsNone(partial["interaction"])
        self.assertIsNone(partial["disable_publication_at_F1"])
        self.assertEqual(partial["disable_resources_at_P1"], -4)


    def test_full_paired_analysis_rejects_mismatched_world_and_missing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in feedback.SEEDS:
                for cell in feedback.CELLS:
                    path = root / feedback.experiment_id(cell, seed)
                    path.mkdir(parents=True)
                    audit = write_nonfunding_request_fixture(path)
                    provenance.write_json(path / "mechanism_manifest.json", {
                        "status": "complete", "protocol": feedback.PROTOCOL,
                        "seed": seed, "cell": cell, "source_hash": "same-source",
                        "scientific_config_hash": "same-config", "model": provenance.MODEL,
                        "effective_args_hash": "same-args",
                        "server_provenance": {"runtime_hash": "same-runtime"},
                        "initial_world_hash": f"world-{seed}", "corpus_hash": "corpus",
                        "fallback_audit": {}, "founder_ids": [f"a{i}" for i in range(1200)],
                        "funding_baseline": feedback.FUNDING_BASELINE,
                        "args": {"output_dir": str(path)},
                        "funding_selection_protocol": feedback.FUNDING_SELECTION_PROTOCOL,
                        "sequential_source_binding": feedback.sequential_source_binding(),
                        "sequential_audit_valid": True,
                        "request_audit": audit, "request_audit_valid": True,
                        "funding_validation": {"completion_allowed": True,
                                               "final_panels": 0,
                                               "output_representation": feedback.FUNDING_OUTPUT_REPRESENTATION},
                    })
                    provenance.write_json(path / "run_manifest.json", {"status": "complete", "years_completed": 6})
                    provenance.write_json(path / "funding_validation.summary.json", {
                        "completion_allowed": True, "final_panels": 0,
                        "output_representation": feedback.FUNDING_OUTPUT_REPRESENTATION})
                    (path / "funding_validation.jsonl").touch()
                    seq = zero_sequential_fixture(path)
                    manifest_path = path / "mechanism_manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["funding_sequential"] = seq
                    manifest["funding_sequential_audits"] = {
                        name: provenance.file_hash(path / name)
                        for name in (feedback.SEQUENTIAL_AUDIT_FILE, feedback.SEQUENTIAL_SUMMARY_FILE)}
                    manifest["funding_application_evidence"] = feedback.funding_ledger_evidence(path, 6)
                    manifest["funding_application_evidence_valid"] = True
                    provenance.write_json(manifest_path, manifest)
                    for year in range(1, 7):
                        rows = [
                            dict(agent_id=f"a{i}", active=(cell != "P1F1"),
                                 spendable_resources=0, cumulative_earned_funding=0,
                                 accepted_papers=0, submitted_papers=0, citations_all_papers=0)
                            for i in range(1200)
                        ]
                        provenance.write_json(path / f"mechanisms_year_{year}.json", {
                            "year": year, "cell": cell, "metrics": feedback.summarize_agents(rows),
                            "agent_rows": rows, "legacy_zero_boundary": feedback.legacy_zero_boundary(rows),
                        })
            result = feedback_analysis.analyze(root)
            contrast = result["year6_paired_contrasts"]["active_fraction"]
            self.assertEqual(contrast["disable_publication_at_F1"], 1)
            self.assertNotIn("sample_sd", json.dumps(result))
            self.assertEqual(result["independent_worlds"], 1)
            self.assertEqual(result["funding_baseline"], feedback.FUNDING_BASELINE)
            self.assertEqual(result["legacy_zero_boundary"]["P0F0"]["founder_year_denominator"], 7200)
            introduced = root / feedback.experiment_id("P0F0", 42) / "funding_applications_year_1.jsonl"
            introduced.touch()  # Absence is not interchangeable with an unreceipted empty file.
            with self.assertRaisesRegex(ValueError, "bytes/absent-year evidence"):
                feedback_analysis.analyze(root)
            introduced.unlink()
            bad_year = root / feedback.experiment_id("P0F0", 42) / "mechanisms_year_2.json"
            record = json.loads(bad_year.read_text())
            record["metrics"]["active_fraction"] = 0.5
            provenance.write_json(bad_year, record)
            with self.assertRaisesRegex(ValueError, "do not reconcile"):
                feedback_analysis.analyze(root)
            record["metrics"] = feedback.summarize_agents(record["agent_rows"])
            provenance.write_json(bad_year, record)
            changed = root / feedback.experiment_id("P0F0", 42) / "mechanism_manifest.json"
            manifest = json.loads(changed.read_text())
            manifest["initial_world_hash"] = "different-world"
            provenance.write_json(changed, manifest)
            with self.assertRaisesRegex(ValueError, "initial_world_hash mismatch"):
                feedback_analysis.analyze(root)
            changed.unlink()
            with self.assertRaises(FileNotFoundError):
                feedback_analysis.analyze(root)

class TestSequentialNativePhase(unittest.TestCase):
    def test_all_four_cells_multiple_balanced_panels_masking_and_accounting(self):
        from utopia.funding.validation import install_funding_validation
        for cell in feedback.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, panel_seed_import():
                out = Path(directory)
                sim, agents, _ = make_panel_simulation(out, cell)
                compact = install_funding_validation(
                    FundingAgency, out / "funding_validation.jsonl", compact=True)
                try:
                    with fixture_request_scope(out) as audit, feedback.sequential_funding_scope(
                            FundingAgency, SequentialScriptedLLM, out) as handle, \
                            feedback.funding_phase_coverage(Simulation, FundingAgency, handle):
                        sim.llm = SequentialScriptedLLM(request_audit=audit, include_history=True)
                        run_phase(sim)
                finally:
                    compact.restore()
                summary = feedback.validate_sequential_evidence(
                    out, compact.summary(), source_binding=feedback.sequential_source_binding(), num_years=1,
                    expected_agency_ids=("agency",),
                    program_rates={"PROGRAM_A": .23, "PROGRAM_B": .10})
                self.assertEqual(summary["processed_panels"], 6)
                self.assertEqual(summary["accepted_steps"], 106)
                self.assertEqual(summary["sdk_calls"], 106)
                ledger = [json.loads(line) for line in
                          (out / "funding_applications_year_1.jsonl").read_text().splitlines()]
                self.assertEqual(Counter(row["n_panel"] for row in ledger), {18: 72, 17: 34})
                self.assertEqual(sum(row["funded"] for row in ledger), 14)
                self.assertEqual(sum(a.resources - 100 for a in agents),
                                 280 if cell.endswith("F1") else 0)
                for request in sim.llm.selection_requests:
                    user = request["messages"][-1]["content"]
                    self.assertEqual("Accepted historical title" in user, cell.startswith("P1"))
                    self.assertEqual(request["max_tokens"], 8192)
                    self.assertEqual(request["seed_ctx"][2:4],
                                     ("sequential_remaining_ids_v1", "production"))
                # Same row count is insufficient: rank/applicant mismatch must fail.
                path = out / "funding_applications_year_1.jsonl"
                saved = path.read_text()
                ledger[0]["applicant_id"] = "NOT_THE_MODEL_SELECTED_APPLICANT"
                path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
                with self.assertRaisesRegex(ValueError, "Native funding ledger differs"):
                    feedback.validate_sequential_evidence(out, compact.summary(), num_years=1,
                                                     expected_agency_ids=("agency",),
                                                     program_rates={"PROGRAM_A": .23, "PROGRAM_B": .10})
                path.write_text(saved)
                changed = [json.loads(line) for line in saved.splitlines()]
                for row in changed:
                    row.update(funding_rate=1.0, num_winners=row["n_panel"], funded=True)
                path.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaisesRegex(ValueError, "Native funding ledger differs"):
                    feedback.validate_sequential_evidence(
                        out, compact.summary(), num_years=1, expected_agency_ids=("agency",),
                        program_rates={"PROGRAM_A": .23, "PROGRAM_B": .10})
                path.write_text(saved)

    def test_exhaustion_does_not_reset_prefix_retry_batch_or_credit_any_panel(self):
        from utopia.funding.validation import install_funding_validation
        with tempfile.TemporaryDirectory() as directory, panel_seed_import():
            out = Path(directory)
            sim, agents, _ = make_panel_simulation(out)
            compact = install_funding_validation(
                FundingAgency, out / "funding_validation.jsonl", compact=True)
            try:
                with self.assertRaises(sequential.SequentialFundingError):
                    with fixture_request_scope(out) as audit, feedback.sequential_funding_scope(
                            FundingAgency, SequentialScriptedLLM, out) as handle, \
                            feedback.funding_phase_coverage(Simulation, FundingAgency, handle):
                        sim.llm = SequentialScriptedLLM(request_audit=audit, fail_step=1)
                        run_phase(sim)
            finally:
                compact.restore()
            calls = sim.llm.selection_requests
            self.assertEqual([(r["seed_ctx"][-1], r["attempt"]) for r in calls],
                             [(0, 0), (1, 0), (1, 1), (1, 2)])
            self.assertTrue(all(a.resources == 100 for a in agents))
            self.assertTrue(all(not a.funding_success_history for a in agents))
            self.assertEqual(compact.summary()["successful_batches"], 0)
            saved = json.loads((out / feedback.SEQUENTIAL_SUMMARY_FILE).read_text())
            self.assertFalse(saved["completion_allowed"])
            self.assertFalse(saved["installed"])
            with self.assertRaises(sequential.SequentialFundingError):
                feedback.validate_sequential_evidence(out, compact.summary(), num_years=1)

    def test_zero_panels_requires_real_finalized_audit_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = zero_sequential_fixture(out)
            (out / "funding_validation.jsonl").touch()
            feedback.validate_sequential_evidence(out, {"final_panels": 0}, manifest_summary=summary)
            (out / feedback.SEQUENTIAL_AUDIT_FILE).write_text("")
            with self.assertRaises(sequential.SequentialFundingError):
                feedback.validate_sequential_evidence(out, {"final_panels": 0})

@unittest.skipUnless(os.environ.get("UTOPIA_ACTUAL_MODULE_TESTS") == "1",
                     "Enable in isolated EC2 CPU environment with exact original logger")
class TestActualSimulator(unittest.TestCase):
    """Real init/RAG/logger/round/checkpoint; compare scientific state and RNG.

    LLM transport is scripted. Independent fixture directories and wall-clock
    timestamp metadata are excluded from checkpoint equality, nothing else.
    """

    setUpClass = classmethod(initialize_actual_simulator)

    tearDownClass = classmethod(restore_actual_simulator)

    make_actual = make_actual_simulation

    def test_actual_module_full_year_p1f1_matches_original_and_rng(self):
        def scientific_state(value):
            if isinstance(value, dict):
                return {key: scientific_state(item) for key, item in value.items()
                        if key not in {"timestamp", "review_time", "submission_time", "decision_time"}}
            if isinstance(value, list):
                return [scientific_state(item) for item in value]
            return value

        checkpoints = []
        requests = []
        states = []
        for label, cell in (("baseline", None), ("control", "P1F1")):
            sim, researchers = self.make_actual(label, cell)
            self.assertFalse(sim.wandb_logger.enabled)
            result = sim.run_one_round(1)
            self.assertGreater(len(sim.paper_tracker.papers_by_id), 0)
            self.assertTrue(any(c[1][0] == "phase3_reviews" for c in sim.llm.calls))
            sim.yearly_results.append(result)
            sim._save_checkpoint(1, phase=5)
            checkpoint_path = Path(sim.output_dir) / "checkpoint_year_1.json"
            if checkpoint_path.exists():
                checkpoint = json.loads(checkpoint_path.read_text())
            else:
                with gzip.open(str(checkpoint_path) + ".gz", "rt") as stream:
                    checkpoint = json.load(stream)
            checkpoint["yearly_results"][0].pop("funding_feedback", None)
            checkpoint["paper_tracker"].pop("output_dir")  # Independent fixture directories.
            checkpoints.append(scientific_state(checkpoint))
            requests.append(sim.llm.calls)
            states.append((random.getstate(), np.random.get_state(), self.torch.get_rng_state().clone()))
            sim.wandb_logger.finish()
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(checkpoints[0], checkpoints[1])
        self.assertEqual(states[0][0], states[1][0])
        np.testing.assert_equal(states[0][1], states[1][1])
        self.assertTrue(self.torch.equal(states[0][2], states[1][2]))
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_all_inactive_full_year_all_cells(self):
        for cell in feedback.CELLS:
            with self.subTest(cell=cell):
                sim, researchers = self.make_actual(f"inactive-{cell}", cell)
                for agent in researchers:
                    agent.is_active = False
                    agent.resources = 0
                for year in (1, 2):
                    result = sim.run_one_round(year)
                    self.assertEqual(len(sim.paper_tracker.papers_by_id), 0)
                    self.assertEqual(result["funding_feedback"]["metrics"]["active_fraction"], 0)
                    self.assertEqual(result["funding_feedback"]["legacy_zero_boundary"]["active_zero_count"], 0)
                self.assertTrue(all(not prompts for prompts, _, _ in sim.llm.calls))
                sim._save_checkpoint(1, phase=5)
                sim.wandb_logger.finish()

    def test_zero_new_authors_keeps_pending_resubmission_in_original_phase(self):
        sim, researchers = self.make_actual("pending", "P0F0")
        sim.run_one_round(1)
        rejected = [p for p in sim.paper_tracker.papers_by_id.values() if p.status == "reject"]
        self.assertTrue(rejected)
        paper = rejected[0].to_dict()
        paper.update(year=2, status="pending", type="resubmission")
        sim.paper_tracker.add_or_update_paper(paper)
        sim.conference_system.reset_all_for_new_year(2)
        result = {}
        actual = sim._run_phase_2_submit_papers(2, {}, [], [paper], result)
        self.assertEqual(actual, [])
        self.assertTrue(any(p["id"] == paper["id"]
                            for c in sim.conference_system.conferences for p in c.submitted_papers))
        self.assertNotIn("batch_retrieve", vars(sim.rag))
        sim.wandb_logger.finish()

if __name__ == "__main__":
    unittest.main()
