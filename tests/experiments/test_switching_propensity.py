from __future__ import annotations

import utopia.runtime.provenance as provenance

import utopia.funding.feedback as feedback

from utopia.runtime.historical import SWITCH_GATE_SEED_NAMESPACE

from utopia.utils.paths import project_root

import ast
import pytest

from collections import Counter, defaultdict

from contextlib import contextmanager, redirect_stdout

from copy import deepcopy

from dataclasses import dataclass

import hashlib

import io

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

import unittest

from unittest.mock import patch

import numpy as np

import utopia.experiments.switching_propensity as jp
import utopia.runtime.switching_evidence as evidence

import utopia.agents.switching_policy as policy

import tests.support.simulation as legacy

import utopia.funding.sequential as sequential

from tests.support.switching import (
    ROOT, fixture_population,
    native_module_fixture,
    VectorFixture,
    native_tracker,
    ScriptedDirections,
    reduced_scientific_state,
    direction_simulation,
    prepare_year,
    cache_from_sim,
    AuditedPropensityLLM,
    tp1_manifest_runtime_fixture,
    funding_manifest_fixture,
    FullYearPropensityLLM,
)

class TestNativeChoices(unittest.TestCase):
    def setUp(self):
        self.module, self.directions = native_module_fixture()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.context = jp.one_year_context(self.module, self.directions)
        self.lookup = self.context.__enter__()
        self.small = fixture_population(3)
        self.small.__enter__()

    def tearDown(self):
        self.small.__exit__(None, None, None)
        self.context.__exit__(None, None, None)
        self.temp.cleanup()

    def producer(self):
        sim = direction_simulation(self.root / "initializer", self.module, self.lookup,
                                   cell=None, failures=("parse", "validation"))
        prepare_year(sim, 1)
        with self.assertRaises(jp.InitialChoicesComplete):
            sim._run_phase_1_research_directions(1, {})
        return sim, cache_from_sim(sim)

    def test_all_four_replay_native_memory_fallback_and_both_state_hashes(self):
        producer, cache = self.producer()
        jp.validate_choices(cache)
        for cell in jp.CELLS:
            with self.subTest(cell=cell):
                sim = direction_simulation(self.root / cell, self.module, self.lookup, cell, cache)
                prepare_year(sim, 1)
                sim._run_phase_1_research_directions(1, {})
                self.assertEqual(sim.llm.calls, [])  # Only common initializer generated first choices.
                self.assertEqual(sim.pre_choice_state_hash, producer.pre_choice_state_hash)
                self.assertEqual(sim.initial_state_hash, producer.initial_state_hash)
                self.assertEqual(sim.initial_records, producer.initial_records)
                self.assertEqual(sim._direction_fallback_stats[1],
                                 {"n_agents": 3, "n_parse_fallback": 1, "n_validation_fallback": 1})
                self.assertTrue(all(a.project_start_year == a.project_end_year == 1
                                    for a in sim.ecosystem.agent_population.values()))
                events = sim.direction_events
                self.assertEqual(sum(e["eligible_repeat_count"] for e in events), 0)
                self.assertEqual(sum(e["initial_choice_count"] for e in events), 3)
                for native, replay in zip(producer.ecosystem.agent_population.values(),
                                          sim.ecosystem.agent_population.values()):
                    self.assertEqual(jp.canonical_state(native.memory_bank),
                                     jp.canonical_state(replay.memory_bank))

    def test_cache_inputs_raw_response_and_state_tampering_fail_before_outcomes(self):
        _, cache = self.producer()
        for mutation in ("input", "raw", "pre", "post"):
            tampered = deepcopy(cache)
            if mutation == "input":
                tampered["records"][0]["input_hash"] = "0" * 64
            elif mutation == "raw":
                tampered["records"][0]["raw_response"] = {"topic": self.lookup[next(iter(self.lookup))].topic,
                                                        "detailed_focus": "changed", "reason": "changed"}
            elif mutation == "pre":
                tampered["pre_choice_state_hash"] = "0" * 64
            else:
                tampered["initial_state_hash"] = "0" * 64
            sim = direction_simulation(self.root / mutation, self.module, self.lookup, "LN", tampered)
            prepare_year(sim, 1)
            with self.assertRaises(evidence.ProtocolFailure):
                sim._run_phase_1_research_directions(1, {})
            self.assertFalse(sim.llm.calls)

    def test_cache_read_requires_immutable_complete_source_bound_records(self):
        _, cache = self.producer()
        path = self.root / "initial_choices.json"
        provenance.write_json(path, cache)
        with self.assertRaisesRegex(evidence.ProtocolFailure, "immutable"):
            jp.load_choices(path)
        path.chmod(0o444)
        loaded, checksum = jp.load_choices(path, {"fixture_only": True})
        self.assertEqual(loaded, cache)
        self.assertEqual(checksum, provenance.file_hash(path))
        with self.assertRaisesRegex(evidence.ProtocolFailure, "input_identity"):
            jp.load_choices(path, {"fixture_only": False})
        path.chmod(0o644)
        cache["records"][0]["response"]["reason"] = "modified"
        provenance.write_json(path, cache)
        path.chmod(0o444)
        with self.assertRaisesRegex(evidence.ProtocolFailure, "records_hash"):
            jp.load_choices(path)

    def test_repeat_stay_and_switch_both_native_calls_memory_duration_and_charge(self):
        sim = direction_simulation(self.root / "repeat", self.module, self.lookup)
        prepare_year(sim, 1)
        sim._run_phase_1_research_directions(1, {})
        previous = {aid: a.newest_direction["direction"].topic
                    for aid, a in sim.ecosystem.agent_population.items()}
        # Gate draw and menus are the real policy; sample both p endpoints using
        # a deterministic identity with U between .25 and .75 when available.
        for year in (2, 3):
            prepare_year(sim, year)
            sim._run_phase_1_research_directions(year, {})
            sim._charge_annual_costs(year)
            events = [e for e in sim.direction_events if e["year"] == year]
            for event in events:
                self.assertEqual(event["realized_switch"], event["u"] < event["p"])
                self.assertEqual(event["realized_switch"], event["chosen_topic"] != previous[event["agent_id"]])
                self.assertEqual(event["project_start_year"], year)
                self.assertEqual(event["project_end_year"], year)
                self.assertEqual(len(event["candidate_topics"]), 17 if event["realized_switch"] else 1)
                self.assertEqual(event["candidate_topics"], sorted(event["candidate_topics"]))
                previous[event["agent_id"]] = event["chosen_topic"]
            self.assertEqual([r["amount"] for r in sim.propensity_years[year]["annual_charges"]], [-10]*3)
        events = [e for e in sim.direction_events if e["eligible_repeat"]]
        self.assertEqual({e["realized_switch"] for e in events}, {False, True})
        self.assertTrue(all(len(a.memory_bank) == 3 for a in sim.ecosystem.agent_population.values()))
        self.assertEqual(len(sim.llm.calls), 3)

    def test_native_fallback_cannot_escape_singleton_or_switch_menu(self):
        sim = direction_simulation(self.root / "fallbacks", self.module, self.lookup,
                                   cell="HN", failures=("parse", "validation"))
        prepare_year(sim, 1)
        sim._run_phase_1_research_directions(1, {})
        prepare_year(sim, 2)
        sim._run_phase_1_research_directions(2, {})
        for event in sim.direction_events[-3:]:
            self.assertEqual(event["chosen_topic"], event["candidate_topics"][0])
            self.assertEqual(event["realized_switch"], event["u"] < event["p"])
        self.assertEqual([e["fallback_kind"] for e in sim.direction_events[-3:]], ["parse", "validation", None])

    def test_no_due_or_all_inactive_year_has_no_gate_or_llm_call(self):
        sim = direction_simulation(self.root / "empty", self.module, self.lookup)
        prepare_year(sim, 1)
        sim._run_phase_1_research_directions(1, {})
        for active in (True, False):
            for agent in sim.ecosystem.agent_population.values():
                agent.is_active = active
                agent.project_end_year = 3
            prepare_year(sim, 2)
            with patch.object(policy, "build_decision", side_effect=AssertionError("unexpected gate")):
                sim._run_phase_1_research_directions(2, {})
        self.assertEqual(len(sim.llm.calls), 1)
        self.assertEqual(sim.propensity_years[2]["phase1_eligible_ids"], [])

    def test_zero_gap_fails_as_domain_before_llm_and_restores_hooks(self):
        sim = direction_simulation(self.root / "gap", self.module, self.lookup)
        prepare_year(sim, 1)
        sim._run_phase_1_research_directions(1, {})
        prepare_year(sim, 2)
        selector = self.module.create_research_directions_batch
        with patch.object(sim.embedding_tracker, "compute_direction_distances",
                          side_effect=lambda c, ds: {d.topic: .5 for d in ds}):
            with self.assertRaises(policy.PolicyDomainError):
                sim._run_phase_1_research_directions(2, {})
        self.assertEqual(len(sim.llm.calls), 1)
        self.assertIs(self.module.create_research_directions_batch, selector)
        self.assertNotIn("generate_batch", vars(sim.llm))

    def test_exact_zero_native_charge_boundary_retained(self):
        sim = direction_simulation(self.root / "boundary", self.module, self.lookup)
        prepare_year(sim, 1)
        sim._run_phase_1_research_directions(1, {})
        agents = list(sim.ecosystem.agent_population.values())
        agents[0].resources, agents[1].resources = 10, 9
        sim._charge_annual_costs(1)
        self.assertEqual((agents[0].resources, agents[0].is_active), (0, True))
        self.assertEqual((agents[1].resources, agents[1].is_active), (0, False))

class TestHistoryAndContracts(unittest.TestCase):
    def test_one_year_copies_prompt_consistency_and_exception_restoration(self):
        module, rd = native_module_fixture()
        originals, original_builder, old_lookup = rd.AVAILABLE_DIRECTIONS, rd.build_direction_prompt, rd.DIRECTIONS_DICT
        original_years = [d.years for d in originals]
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with jp.one_year_context(module, rd) as lookup:
                self.assertTrue(all(d.years == 1 for d in lookup.values()))
                self.assertTrue(all(d is not originals[i] for i, d in enumerate(lookup.values())))
                with tempfile.TemporaryDirectory() as directory:
                    sim = direction_simulation(Path(directory)/"sim", module, lookup)
                    a = next(iter(sim.ecosystem.agent_population.values()))
                    prompt = rd.build_direction_prompt(a, 1, sim.paper_tracker, list(lookup.values()), 100)[0]
                    self.assertNotIn("1, 2, or 3", prompt)
                    self.assertNotIn("3-4 years", prompt)
                    self.assertEqual(prompt.count("Expected Project Duration: 1 year(s)"), 53)
                    self.assertIn("Only the displayed candidate directions are valid choices", prompt)
                    raise RuntimeError("fixture")
        self.assertIs(rd.AVAILABLE_DIRECTIONS, originals)
        self.assertIs(rd.DIRECTIONS_DICT, old_lookup)
        self.assertIs(rd.build_direction_prompt, original_builder)
        self.assertEqual([d.years for d in originals], original_years)

    def history(self):
        _, rd = native_module_fixture()
        tracker = native_tracker()
        history = jp.NeutralHistory(tracker, [SimpleNamespace(id="a", expertise=rd.AVAILABLE_DIRECTIONS[:3])],
                                    rd.AVAILABLE_DIRECTIONS)
        return history, tracker

    def test_all_unique_abstracts_rejected_and_accepted_first_year_not_resub_year(self):
        history, tracker = self.history()
        papers = {
            "p": {"id": "p", "author_id": "a", "abstract": "Rejected abstract.", "year": 1, "status": "reject"},
            "q": {"id": "q", "author_id": ["a"], "abstract": "Accepted abstract.", "year": 1, "status": "accept"},
        }
        history.ingest(papers, 1)
        self.assertEqual(set(tracker.embeddings), {"p", "q"})
        before = tracker.embeddings["p"].copy()
        papers["p"].update(year=3, status="pending")
        history.ingest(papers, 3)
        np.testing.assert_array_equal(before, tracker.embeddings["p"])
        self.assertEqual(tracker.paper_metadata["p"]["year"], 1)
        centroid, meta = history.reference("a", 5, papers)
        self.assertEqual(meta["history_ids"], ["p", "q"])  # inclusive t-4..t-1
        np.testing.assert_allclose(centroid, np.mean([tracker.embeddings["p"], tracker.embeddings["q"]], axis=0))
        _, meta = history.reference("a", 6, papers)
        self.assertEqual(meta["reference_source"], "initial_expertise")
        papers["p"]["abstract"] = "A conflicting abstract."
        with self.assertRaisesRegex(evidence.ProtocolFailure, "resubmission_changed"):
            history.ingest(papers, 3)

    def test_coverage_missing_zero_nonfinite_and_wrong_first_year_fail(self):
        for failure in ("missing", "zero", "nan", "firstyear"):
            history, tracker = self.history()
            papers = {"p": {"id": "p", "author_id": "a", "abstract": "An abstract.", "year": 1}}
            if failure == "firstyear":
                with self.assertRaises(evidence.ProtocolFailure):
                    history.ingest(papers, 2)
                continue
            history.ingest(papers, 1)
            if failure == "missing":
                tracker.embeddings.pop("p")
            else:
                tracker.embeddings["p"][:] = 0 if failure == "zero" else np.nan
            with self.assertRaises(evidence.ProtocolFailure):
                history.reference("a", 2, papers)

    def test_rng_neutral_bookkeeping_even_on_exception(self):
        random.seed(42)
        np.random.seed(42)
        before = random.getstate(), np.random.get_state()
        with self.assertRaises(RuntimeError):
            with jp.preserved_cpu_rng():
                random.random()
                np.random.normal()
                raise RuntimeError()
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_equal(np.random.get_state(), before[1])



    def test_cached_minilm_uses_exact_revision_and_never_refs_main(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"HF_HUB_CACHE": directory}):
            repository = Path(directory) / "models--sentence-transformers--all-MiniLM-L6-v2"
            snapshot = repository / "snapshots" / jp.EMBEDDING_REVISION
            snapshot.mkdir(parents=True)
            (snapshot / "modules.json").write_text("[]")
            (snapshot / "model.safetensors").write_bytes(b"fixture")
            with patch.object(jp, "EMBEDDING_WEIGHTS_SHA256", provenance.file_hash(snapshot/"model.safetensors")):
                path, identity = jp.resolve_embedding_snapshot()
                self.assertEqual(path, snapshot)
                self.assertEqual(identity["revision"], jp.EMBEDDING_REVISION)
                self.assertEqual(identity["device"], "cpu")
                with self.assertRaises(evidence.ProtocolFailure):
                    jp.resolve_embedding_snapshot("0"*40)
                (snapshot / "model.safetensors").write_bytes(b"mutated")
                with self.assertRaises(evidence.ProtocolFailure):
                    jp.resolve_embedding_snapshot()

class TestSequentialPropensity(unittest.TestCase):
    def test_tp1_runtime_admission_rejects_missing_or_coherently_wrong_attestations(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = tp1_manifest_runtime_fixture(directory)
            expected_source = jp.source_identity()
            for kind in ("initialization_only", "condition"):
                manifest = dict(deepcopy(runtime), kind=kind)
                evidence.validate_runtime_evidence(manifest, expected_source=expected_source)
                for key in ("server_runtime_profile", "server_runtime_hash", "runtime_amendment"):
                    altered = deepcopy(manifest)
                    altered["inputs"].pop(key)
                    with self.subTest(kind=kind, missing=key), self.assertRaises(evidence.ProtocolFailure):
                        evidence.validate_runtime_evidence(altered, expected_source=expected_source)
                missing_source = dict(expected_source)
                missing_source.pop("utopia/runtime/server_profiles.py")
                with self.assertRaises(evidence.ProtocolFailure):
                    evidence.validate_runtime_evidence(manifest, expected_source=missing_source)
                for field, wrong in (("tensor_parallel_size", 2), ("dtype", "float16"),
                                     ("max_model_len", 16384), ("enforce_eager", False),
                                     ("endpoint", ""), ("endpoint", "http://127.0.0.1:18085/v1")):
                    altered = deepcopy(manifest)
                    server = altered["server_provenance"]
                    path = Path(server["manifest_path"])
                    saved = path.read_bytes()
                    server["record"][field] = wrong
                    # Keep all stored copies and byte hashes internally consistent.
                    provenance.write_json(path, server["record"])
                    server["file_sha256"] = provenance.file_hash(path)
                    with self.subTest(kind=kind, field=field), self.assertRaises(evidence.ProtocolFailure):
                        evidence.validate_runtime_evidence(altered, expected_source=expected_source)
                    path.write_bytes(saved)
                altered = deepcopy(manifest)
                altered["args"].pop("vllm_url")
                with self.assertRaises(evidence.ProtocolFailure):
                    evidence.validate_runtime_evidence(altered, expected_source=expected_source)

    def test_native_initializer_zero_years_real_request_join_and_all_four_cache_replay(self):
        from utopia.funding.validation import install_funding_validation
        module, directions = native_module_fixture()
        with tempfile.TemporaryDirectory() as directory, fixture_population(3), \
                jp.one_year_context(module, directions) as lookup:
            root = Path(directory)
            producer = direction_simulation(root / "initial", module, lookup, cell=None)
            prepare_year(producer, 1)
            compact = install_funding_validation(
                module.FundingAgency, producer.docs / "funding_validation.jsonl", compact=True)
            try:
                with feedback.sequential_funding_scope(
                        module.FundingAgency, AuditedPropensityLLM, producer.docs) as handle, \
                        feedback.funding_phase_coverage(module.Simulation, module.FundingAgency, handle), \
                        legacy.fixture_request_scope(producer.docs) as audit:
                    producer.llm = AuditedPropensityLLM(request_audit=audit)
                    with self.assertRaises(jp.InitialChoicesComplete):
                        producer._run_phase_1_research_directions(1, {})
            finally:
                compact.restore()
            manifest = funding_manifest_fixture(producer.docs, compact, initialize_only=True)
            report = jp.validate_funding_evidence(
                producer.docs, manifest, expected_source=jp.source_identity())
            self.assertEqual(report["scientific_years"], 0)
            self.assertEqual(report["funding_sequential"]["sdk_calls"], 0)
            self.assertEqual(report["funding_sequential"]["agency_registrations"], 0)
            self.assertEqual(manifest["request_audit"]["n_requests"], 3)
            cache = cache_from_sim(producer)
            cache["inputs"] = manifest["inputs"]
            cache_path = producer.docs / "initial_choices.json"
            provenance.write_json(cache_path, cache)
            cache_path.chmod(0o444)
            manifest.update(
                initial_choices_path=str(cache_path), initial_choices_sha256=provenance.file_hash(cache_path),
                founder_ids=cache["founder_ids"], choice_count=3,
                pre_choice_state_hash=cache["pre_choice_state_hash"],
                initial_state_hash=cache["initial_state_hash"])
            provenance.write_json(producer.docs / "initialization_manifest.json", manifest)
            admitted, _, _, _ = jp.validate_initialization(cache_path, cache["inputs"])
            for cell in jp.CELLS:
                sim = direction_simulation(root / cell, module, lookup, cell, admitted)
                prepare_year(sim, 1)
                sim._run_phase_1_research_directions(1, {})
                self.assertEqual(sim.pre_choice_state_hash, producer.pre_choice_state_hash)
                self.assertEqual(sim.initial_state_hash, producer.initial_state_hash)
                self.assertEqual(sim.llm.calls, [])
            bad = deepcopy(cache)
            bad["protocol"] = "switching-propensity-v1-single-world"
            with self.assertRaises(evidence.ProtocolFailure):
                jp.validate_choices(bad, cache["inputs"])
            altered = deepcopy(manifest)
            altered["initial_choices_sha256"] = "0" * 64
            provenance.write_json(producer.docs / "initialization_manifest.json", altered)
            with self.assertRaises(evidence.ProtocolFailure):
                jp.validate_initialization(cache_path, cache["inputs"])

    def test_ten_real_funding_phases_bind_two_agencies_ledgers_and_missing_evidence(self):
        from utopia.funding.validation import install_funding_validation
        with tempfile.TemporaryDirectory() as directory, legacy.panel_seed_import(), \
                redirect_stdout(io.StringIO()):
            out = Path(directory)
            sim, _, agency = legacy.make_panel_simulation(out)
            # Actual inherited propensity phase5, with original native program rates.
            sim.__class__ = jp.simulation_class(legacy.Simulation)
            sim.args.docs_dir = str(out)
            sim.ecosystem.agent_population.pop(agency.id)
            programs = list(agency.funding_programs.values())
            for aid, pid, program in (("NSF", "NSF_THEORY", programs[0]),
                                      ("DARPA", "DARPA_AI_APPS", programs[1])):
                copy = deepcopy(agency)
                copy.id = aid
                program.program_id = program.name = pid
                copy.funding_programs = {pid: program}
                sim.ecosystem.add_agent(copy)
            compact = install_funding_validation(
                legacy.FundingAgency, out / "funding_validation.jsonl", compact=True)
            try:
                with feedback.sequential_funding_scope(
                        legacy.FundingAgency, AuditedPropensityLLM, out) as handle, \
                        feedback.funding_phase_coverage(legacy.Simulation, legacy.FundingAgency, handle), \
                        legacy.fixture_request_scope(out) as audit:
                    sim.llm = AuditedPropensityLLM(request_audit=audit)
                    for year in range(1, 11):
                        sim.funding_tracker.start_cycle(year, sim.ecosystem.agent_population)
                        sim._run_phase_5_update_funding(year, [], {})
            finally:
                compact.restore()
            manifest = funding_manifest_fixture(out, compact, initialize_only=False)
            report = jp.validate_funding_evidence(out, manifest, expected_source=jp.source_identity())
            self.assertEqual(report["funding_sequential"]["years_completed"], 10)
            self.assertEqual(report["funding_sequential"]["agency_registrations"], 20)
            self.assertEqual(report["funding_sequential"]["accepted_steps"], 1060)
            self.assertEqual(len(report["funding_application_evidence"]["present_sha256"]), 10)
            extra = out / "funding_applications_year_11.jsonl"
            extra.write_text("")
            with self.assertRaisesRegex(evidence.ProtocolFailure, "undeclared_funding_ledger_file"):
                jp.validate_funding_evidence(out, manifest, expected_source=jp.source_identity())
            extra.unlink()
            ledger = out / "funding_applications_year_1.jsonl"
            saved = ledger.read_bytes()
            ledger.write_bytes(b" " + saved)
            with self.assertRaises(evidence.ProtocolFailure):
                jp.validate_funding_evidence(out, manifest, expected_source=jp.source_identity())
            ledger.write_bytes(saved)
            path = out / feedback.SEQUENTIAL_SUMMARY_FILE
            summary = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                jp.validate_funding_evidence(out, manifest, expected_source=jp.source_identity())
            path.write_bytes(summary)
            for malformed in ('{"status":"failed","status":"complete"}', '{"counter":NaN}',
                              '{"counter":1e999}'):
                path.write_text(malformed)
                with self.assertRaises(evidence.ProtocolFailure):
                    jp.validate_funding_evidence(out, manifest, expected_source=jp.source_identity())
            path.write_bytes(summary)

@pytest.mark.integration
class TestActualNativePropensity(unittest.TestCase):
    """Imported modules, real RAG/logger/MiniLM, native phase order and checkpoints.

    Uses 12-founder/two-year CPU fixtures only; formal CLI assertions stay 1200/10.
    The existing cache subset fixture is reused, not a scientific dataset swap.
    HTTP/request guard behavior is tested by its separate owner; these calls are
    explicitly scripted and do not claim a live Qwen/schema/throughput result.
    """
    @classmethod
    def setUpClass(cls):
        cls.harness = type("ActualSimulatorFixture", (), {})
        legacy.initialize_actual_simulator(cls.harness)
        cls.actual = cls.harness.actual
        import utopia.agents.research_direction as directions_module
        cls.directions_module = directions_module
        cls.snapshot, _ = jp.resolve_embedding_snapshot()

    @classmethod
    def tearDownClass(cls):
        legacy.restore_actual_simulator(cls.harness)

    def make_actual(self, label, cell, cache, lookup, *, native_control=False):
        path = self.harness.fixture_root / label
        with patch.dict(os.environ, {"UTOPIA_DATA_CACHE_DIR": str(path / "cache")}):
            args = self.actual.parse_arguments([
                "--experiment_name", "default_propensity_cpu_fixture", "--experiment_stage", "mechanism",
                "--output-dir", str(path), "--seed", "42", "--population_mode", "university_only",
                "--num_institutions", "6", "--researchers_per_institution", "2",
                "--strategy_mix", "balanced", "--num_years", "2", "--num_conferences", "2",
                "--model", jp.MODEL, "--always_rerun", "--log_funding_applications",
                "--funding_panel_max_apps", "25", "--funding_budget_mode", "track",
                "--funding_allocation_mode", "fixed",
            ])
        args.rag_device = "cpu"
        self.actual.set_seed(42)
        llm = FullYearPropensityLLM(Path(args.docs_dir) / "fixture_audit.jsonl", self.actual.derive_seed)
        cls = feedback.simulation_class(self.actual.Simulation) if native_control else jp.simulation_class(self.actual.Simulation)
        options = {} if native_control else dict(
            cell=cell, initial_cache=cache, embedding_snapshot=self.snapshot,
            simulator_module=self.actual, direction_lookup=lookup, initialize_only=cell is None)
        sim = cls(llm=llm, args=args, num_years=2, output_dir=args.checkpoint_dir,
                  funding_allocation_mode="fixed", always_rerun=True,
                  experiment_name=args.experiment_name, **options)
        return sim

    @contextmanager
    def pristine_conferences(self):
        with patch.object(self.harness.conference_module, "CONFERENCES_BY_CATEGORY",
                          deepcopy(self.harness.pristine_conferences)):
            yield

    def setup_world(self, sim):
        sim.initialize_agents()
        agents = [a for a in sim.ecosystem.agent_population.values() if a.get_type() == "university"]
        sim.conference_system = self.actual.ConferenceSystem(
            self.actual.select_conferences_for_simulation(agents, num_conferences=2))
        for a in agents:
            sim.agent_tracker.record_agent_resources(a.id, a.resources, 0)

    @contextmanager
    def sequential_runtime(self, sim):
        from utopia.funding.validation import install_funding_validation
        handle = install_funding_validation(
            self.actual.FundingAgency, sim.docs / "funding_validation.jsonl", compact=True)
        try:
            with feedback.sequential_funding_scope(
                    self.actual.FundingAgency, AuditedPropensityLLM, sim.docs) as sequential_handle, \
                    feedback.funding_phase_coverage(
                        self.actual.Simulation, self.actual.FundingAgency, sequential_handle), \
                    legacy.fixture_request_scope(sim.docs) as audit:
                sim.llm = AuditedPropensityLLM(request_audit=audit)
                yield handle
        finally:
            handle.restore()
            sim.wandb_logger.finish()

    def test_initial_native_run_stops_before_outcomes_and_all_four_replay_full_year(self):
        with fixture_population(12), jp.one_year_context(self.actual, self.directions_module) as lookup:
            producer = self.make_actual("propensity-producer", None, None, lookup)
            with self.sequential_runtime(producer) as producer_guard:
                with self.pristine_conferences(), self.assertRaises(jp.InitialChoicesComplete):
                    producer.run()
            initial_evidence = feedback.validate_sequential_evidence(
                producer.docs, producer_guard.summary(), application_dir=producer.output_dir, num_years=0)
            self.assertEqual(initial_evidence["agency_registrations"], 0)
            self.assertEqual(initial_evidence["sdk_calls"], 0)
            self.assertEqual(len(producer.initial_records), 12)
            self.assertFalse(producer.paper_tracker.papers_by_id)
            self.assertEqual({call[1][0] for call in producer.llm.calls}, {"phase1_directions"})
            cache = cache_from_sim(producer)
            jp.validate_choices(cache)
            hashes, first_year_states, requests = [], [], []
            for cell in jp.CELLS:
                with self.subTest(cell=cell), self.pristine_conferences():
                    sim = self.make_actual(f"propensity-{cell}", cell, cache, lookup)
                    self.setup_world(sim)
                    with self.sequential_runtime(sim) as handle:
                        result = sim.run_one_round(1)
                        sim.yearly_results.append(result)
                        sim._save_checkpoint(1, phase=5)
                        self.assertTrue(handle.summary()["completion_allowed"])
                        hashes.append((sim.pre_choice_state_hash, sim.initial_state_hash))
                        first_year_states.append(sim.world_state_hash())
                        requests.append(deepcopy(sim.llm.calls))
                        self.assertFalse(sim.is_exploration_experiment)
                        self.assertFalse(sim.wandb_logger.enabled)
                        self.assertEqual(len(sim.neutral_history.registry), len(sim.paper_tracker.papers_by_id))
                        self.assertGreater(len(sim.neutral_history.registry), 0)
                        self.assertTrue(all(r["first_submission_year"] == 1
                                            for r in sim.neutral_history.registry.values()))
                        result = sim.run_one_round(2)
                        sim.yearly_results.append(result)
                        sim._save_checkpoint(2, phase=5)
                        events = sim.propensity_years[2]["direction_events"]
                        self.assertGreater(len(events), 0)
                        self.assertTrue(all(e["realized_switch"] == (e["u"] < e["p"]) for e in events))
                        self.assertTrue(all(e["history_ids"] for e in events))
                        self.assertEqual(sim.propensity_years[2]["history_coverage"]["missing_papers"], 0)
                    self.assertTrue(handle.summary()["completion_allowed"])
                    evidence = feedback.validate_sequential_evidence(
                        sim.docs, handle.summary(), application_dir=sim.output_dir, num_years=2)
                    self.assertEqual(evidence["years_completed"], 2)
                    self.assertEqual(evidence["agency_registrations"], 4)
            self.assertEqual(set(hashes), {(producer.pre_choice_state_hash, producer.initial_state_hash)})
            self.assertEqual(len(set(first_year_states)), 1)
            self.assertTrue(all(calls == requests[0] for calls in requests))
            self.assertFalse(self.harness.torch.cuda.is_initialized())

    def test_year_one_matches_stock_configured_duration_and_explicit_candidates(self):
        """At common initialization the adapter matches that same configured stock path."""
        from utopia.funding.validation import install_funding_validation
        records = []
        with fixture_population(12), jp.one_year_context(self.actual, self.directions_module) as lookup:
            for native in (True, False):
                with self.pristine_conferences():
                    sim = self.make_actual(f"propensity-parity-{native}", "LN", None, lookup,
                                           native_control=native)
                    self.setup_world(sim)
                    original_selector = self.actual.create_research_directions_batch

                    def strict_native(*args, **kwargs):
                        kwargs["candidate_map"] = {a.id: a.expertise for a in kwargs["agents"]}
                        return original_selector(*args, **kwargs)

                    handle = install_funding_validation(
                        self.actual.FundingAgency, Path(sim.args.docs_dir) / "funding_fixture.jsonl", compact=True)
                    try:
                        with (jp.module_attribute(self.actual, "create_research_directions_batch", strict_native)
                              if native else __import__("contextlib").nullcontext()):
                            result = sim.run_one_round(1)
                        state = jp.canonical_state({
                            "ecosystem": sim.ecosystem.to_dict(), "conferences": sim.conference_system.to_dict(),
                            "papers": {pid: p.to_dict() for pid, p in sim.paper_tracker.papers_by_id.items()},
                        })
                        records.append((state, sim.llm.calls, random.getstate(), np.random.get_state(),
                                        self.harness.torch.get_rng_state().clone()))
                        self.assertTrue(handle.summary()["completion_allowed"])
                    finally:
                        handle.restore()
                        sim.wandb_logger.finish()
            self.assertEqual(records[0][:3], records[1][:3])
            np.testing.assert_equal(records[0][3], records[1][3])
            self.assertTrue(self.harness.torch.equal(records[0][4], records[1][4]))

    def test_all_inactive_repeat_year_continues_native_empty_year(self):
        from utopia.funding.validation import install_funding_validation
        with fixture_population(12), jp.one_year_context(self.actual, self.directions_module) as lookup, \
             self.pristine_conferences():
            sim = self.make_actual("propensity-empty-actual", "LN", None, lookup)
            self.setup_world(sim)
            handle = install_funding_validation(
                self.actual.FundingAgency, sim.docs / "funding_fixture.jsonl", compact=True)
            try:
                sim.run_one_round(1)
                before = len(sim.llm.calls)
                for aid in sim.founder_ids:
                    sim.ecosystem.agent_population[aid].is_active = False
                    sim.ecosystem.agent_population[aid].resources = 0
                sim.run_one_round(2)
                self.assertEqual(sim.propensity_years[2]["phase1_eligible_ids"], [])
                self.assertEqual(sim.propensity_years[2]["direction_events"], [])
                self.assertEqual(sim.propensity_years[2]["history_coverage"]["missing_papers"], 0)
                self.assertTrue(all(not prompts for prompts, _, _ in sim.llm.calls[before:]))
                self.assertTrue(handle.summary()["completion_allowed"])
            finally:
                handle.restore()
                sim.wandb_logger.finish()

    def test_native_phase0_payments_remove_overshoot_author_preserve_first_year_and_exact_zero(self):
        """Two 5-unit resubmissions from 9 deactivate; one from 5 stays active."""
        from utopia.funding.validation import install_funding_validation
        with fixture_population(12), jp.one_year_context(self.actual, self.directions_module) as lookup, \
             self.pristine_conferences():
            sim = self.make_actual("propensity-phase0-actual", "LN", None, lookup)
            self.setup_world(sim)
            handle = install_funding_validation(
                self.actual.FundingAgency, sim.docs / "funding_fixture.jsonl", compact=True)
            try:
                sim.run_one_round(1)
                rejected = [p for p in sim.paper_tracker.papers_by_id.values() if p.status == "reject"]
                self.assertGreaterEqual(len(rejected), 2)
                first = rejected[0]
                other = next(p for p in rejected if p.all_author_ids[0] != first.all_author_ids[0])
                departing = sim.ecosystem.get_agent_by_id(first.all_author_ids[0])
                exact_zero = sim.ecosystem.get_agent_by_id(other.all_author_ids[0])
                # Explicit CPU state fixture: create a second rejected manuscript
                # through the original archive API, then ingest its actual abstract.
                # This is not a simulated scientific output or a corpus replacement.
                copy = first.to_dict()
                copy.update(id=first.id + "_phase0_cpu_fixture", type="submission", status="pending", year=1)
                sim.paper_tracker.add_or_update_paper(copy)
                copy.update(status="reject", final_score=6, reviews=[])
                sim.paper_tracker.add_or_update_paper(copy)
                sim._record_paper_metadata_batch(1)
                first_years = {pid: r["first_submission_year"] for pid, r in sim.neutral_history.registry.items()}
                departing.resources, exact_zero.resources = 9, 5
                self.assertTrue(departing.is_active and exact_zero.is_active)
                year_results = sim._setup_year(2)
                selected = {first.id, copy["id"], other.id}
                conference = sim.conference_system.conferences[0].conference_id

                def scripted_resubmit(prompts, *, seed_ctx, **kwargs):
                    self.assertEqual(seed_ctx[:2], ("phase0_resubmit", 2))
                    return [({"resubmitted_papers": [
                        {"arxiv_id": pid, "conference": conference}
                        for pid in re.findall(r'arXiv ID: "([^"]+)"', prompt) if pid in selected
                    ]}, []) for prompt in prompts]

                with feedback.instance_override(sim.llm, "generate_batch", scripted_resubmit):
                    resubmissions = sim._run_phase_0_resubmissions(2, year_results)
                self.assertEqual({p["id"] for p in resubmissions}, selected)
                self.assertEqual(sim.funding_tracker.consumption_per_cycle[2]["university"], 15)
                self.assertEqual((departing.resources, departing.is_active), (0, False))
                self.assertEqual((exact_zero.resources, exact_zero.is_active), (0, True))
                sim._run_phase_1_research_directions(2, year_results)
                self.assertEqual(sim.propensity_years[2]["phase0_lost_active_ids"], [departing.id])
                self.assertNotIn(departing.id, sim.propensity_years[2]["phase1_eligible_ids"])
                self.assertIn(exact_zero.id, sim.propensity_years[2]["phase1_eligible_ids"])
                self.assertFalse(any(e["agent_id"] == departing.id and e["year"] == 2
                                     for e in sim.direction_events))
                sim._record_paper_metadata_batch(2)
                self.assertEqual(first_years, {pid: r["first_submission_year"]
                                              for pid, r in sim.neutral_history.registry.items()})
                self.assertTrue(all(sim.paper_tracker.papers_by_id[pid].year == 2 for pid in selected))
                self.assertTrue(all(sim.embedding_tracker.paper_metadata[pid]["year"] == 1 for pid in selected))
                self.assertEqual(sim.neutral_history.reconcile(sim.paper_tracker.papers_by_id)["missing_papers"], 0)
                self.assertFalse(self.harness.torch.cuda.is_initialized())
            finally:
                handle.restore()
                sim.wandb_logger.finish()

if __name__ == "__main__":
    unittest.main()
