"""Pure CPU analysis fixtures; no model, simulator import, HTTP or GPU calls.

    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s src \
        -p test_switching_propensity_analysis.py -v

Synthetic events use the actual policy and AST-loaded original derive_seed.
They exercise mathematical endpoints and admission invariants, not simulation
performance or a scientific result. Temporary file-wrapper fixtures stay here.
"""

import utopia.funding.feedback as feedback
import utopia.runtime.provenance as provenance

from utopia.utils.paths import project_root

import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = project_root(__file__)
import utopia.analysis.switching_propensity as analysis
import utopia.agents.switching_policy as policy
import utopia.funding.sequential as sequential
import utopia.experiments.funding_feedback as mechanisms
import utopia.experiments.switching_propensity as driver


def native_seed():
    path = ROOT / "utopia/utils/seeding.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "derive_seed")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["derive_seed"]


NATIVE_SEED = native_seed()
SOURCE = {"utopia/experiments/switching_propensity.py": "a" * 64,
          "utopia/analysis/switching_propensity.py": "b" * 64,
          "docs/switching_propensity_protocol.md": "c" * 64,
          "utopia/funding/sequential.py": "1" * 64,
          "docs/sequential_funding_protocol.md": "2" * 64}
FOUNDERS = [f"founder_{i:04d}" for i in range(1200)]


def summary_fixture(empty=False):
    result = dict.fromkeys(sequential._COUNTERS, 0)
    result.update(
        funding_selection_protocol=analysis.SEQUENTIAL_PROTOCOL,
        output_representation="ordered_application_ids_v1",
        installed=False, status="complete", completion_allowed=True,
        source_sha256=SOURCE["utopia/funding/sequential.py"],
        source_files_sha256={name: SOURCE[name] for name in (
            "utopia/funding/sequential.py", "docs/sequential_funding_protocol.md")},
        sdk_internal_transport_retries=5, max_sdk_calls_per_step=3,
        restore_conflicts=[], unprocessed_panels=0, years_started=10, years_completed=10,
        agency_registrations=20, empty_agency_registrations=20 if empty else 0)
    if not empty:
        result.update(registered_panels=20, completed_panels=20, returned_panels=20,
                      processed_panels=20, accepted_steps=20, sdk_calls=20, n_clients=1)
    return result


def native_year_writer(cell):
    """Compile the exact driver serializer, replacing only its runtime services.

    The fixture below feeds synthetic state into this method. It produces the
    real paper-observation, first-attempt, deadline and registry-wrapper shapes;
    these are not invented analysis-side convenience fields.
    """
    path = ROOT / "utopia/experiments/switching_propensity.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PropensityMixin")
    method = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == "run_one_round")
    writes = {}

    class Base:
        def run_one_round(self, year):
            return {"year": year}

    def require(condition, reason, **details):
        if not condition:
            raise AssertionError((reason, details))

    scope = {
        "Base": Base, "YEARS": 10, "require": require,
        "provenance": SimpleNamespace(write_json=lambda path, value: writes.update({str(path): deepcopy(value)})),
    }
    cls = ast.ClassDef(name="Writer", bases=[ast.Name(id="Base", ctx=ast.Load())],
                       keywords=[], body=[method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
    writer = scope["Writer"]()
    writer.docs, writer.propensity_cell = Path(cell), cell
    writer.propensity_years, writer.funding_feedback_years, writer.direction_events = {}, {}, []
    writer.paper_tracker = SimpleNamespace(papers_by_id={})
    writer.counts = {}
    writer.citation_tracker = SimpleNamespace(get_citation_count=lambda pid: writer.counts[pid])
    writer.neutral_history = SimpleNamespace(
        registry={},
        reconcile=lambda papers: {"status": "complete", "required_papers": len(papers),
                                  "embedded_papers": len(papers), "missing_papers": 0})
    return writer, writes


@lru_cache(maxsize=2)
def _fixture(empty=False):
    manifests, records = {}, {}
    for cell in analysis.CELLS:
        manifests[cell] = {
            "status": "complete", "kind": "condition", "protocol": analysis.PROTOCOL,
            "cell": cell, "seed": 42, "founder_count": 1200, "years_completed": 10,
            "scientific_years_completed": 10, "choice_count": 1200,
            "inputs": {"source_files_sha256": deepcopy(SOURCE), "protocol": analysis.PROTOCOL, "seed": 42,
                       "funding_selection_protocol": analysis.SEQUENTIAL_PROTOCOL,
                       "sequential_source_binding": {name: SOURCE[name] for name in (
                           "utopia/funding/sequential.py", "docs/sequential_funding_protocol.md")}},
            "founder_ids": list(FOUNDERS), "initial_choices_sha256": "d" * 64,
            "initial_state_hash": "e" * 64, "pre_choice_state_hash": "f" * 64,
            "initialization_manifest_path": "/private/outputs/docs/fresh/initialization_manifest.json",
            "initialization_manifest_sha256": "9" * 64,
            "world_standard_error": None, "world_confidence_interval": None,
            "funding_baseline": deepcopy(feedback.FUNDING_BASELINE),
            "funding_selection_protocol": analysis.SEQUENTIAL_PROTOCOL,
            "sequential_source_binding": {name: SOURCE[name] for name in (
                "utopia/funding/sequential.py", "docs/sequential_funding_protocol.md")},
            "funding_sequential": summary_fixture(empty),
            "funding_validation": {
                "completion_allowed": True, "installed": False,
                "output_representation": "ordered_application_ids_v1", "restore_conflicts": [],
                "failed_batches": 0, "audit_failures": 0, "invalid_final_panels": 0,
                "processing_failures": 0, "unprocessed_panels": 0,
                "imputed_rankings": 0, "fallback_rankings": 0, "invalid_attempts": 2,
                "final_panels": 0 if empty else 20,
            },
            "request_audit": {
                "status": "complete", "n_clients": 1, "guard_failures": 0,
                "pending_requests": 0, "n_requests": 10, "n_responses": 9, "n_transport_errors": 1,
            },
        }
        manuscripts = {} if empty else {
            "published": {"paper_id": "published", "author_id": FOUNDERS[0],
                          "first_submission_year": 1, "abstract_sha256": "1" * 64},
            "late": {"paper_id": "late", "author_id": FOUNDERS[1],
                     "first_submission_year": 1, "abstract_sha256": "2" * 64},
            "never": {"paper_id": "never", "author_id": FOUNDERS[2],
                      "first_submission_year": 1, "abstract_sha256": "3" * 64},
            "year8": {"paper_id": "year8", "author_id": FOUNDERS[0],
                      "first_submission_year": 8, "abstract_sha256": "4" * 64},
        }
        writer, writes = native_year_writer(cell)
        years, mechanism_years, previous = [], [], {}
        active = list(FOUNDERS)
        balances = dict.fromkeys(FOUNDERS, 100.0)
        for year in range(1, 11):
            phase0 = [FOUNDERS[1]] if not empty and year == 6 else []
            eligible = [agent for agent in active if agent not in phase0]
            end = ([] if empty else FOUNDERS[:2]) if year == 1 else list(eligible)
            for i, agent in enumerate(eligible):
                if year == 1:
                    topic = policy.CANONICAL_TOPICS[i % 53]
                    event = {
                        "status": "complete", "cell": cell, "seed": 42, "event_type": "initial_choice",
                        "agent_id": agent, "year": year, "chosen_topic": topic,
                        "is_initial_choice": True, "eligible_repeat": False,
                        "fallback": False, "fallback_kind": None, "realized_switch": False,
                        "conditional_switch_distance": None, "candidate_topics": [topic],
                        "new_project_choice_count": 1, "initial_choice_count": 1,
                        "eligible_repeat_count": 0, "realized_switch_count": 0,
                    }
                else:
                    topics = [topic for topic in policy.CANONICAL_TOPICS if topic != previous[agent]]
                    distances = {topic: index / 26 for index, topic in enumerate(topics)}
                    decision = policy.build_decision(
                        cell, agent, year, previous[agent], policy.CANONICAL_TOPICS,
                        distances, derive_seed_fn=NATIVE_SEED)
                    kind = ("parse" if year == 3 and agent == FOUNDERS[1]
                            else "validation" if year == 4 and agent == FOUNDERS[0] else None)
                    event = policy.finalize_choice(decision, decision["candidate_topics"][0], kind)
                    history = sorted(pid for pid, paper in manuscripts.items()
                                     if paper["author_id"] == agent
                                     and year - 4 <= paper["first_submission_year"] <= year - 1)
                    event.update(
                        reference_source="paper_history" if history else "initial_expertise",
                        history_ids=history, expected_history_count=len(history), reference_norm=0.8)
                event.update(project_start_year=year, project_end_year=year,
                             detailed_focus="scripted focus", reason="scripted reason",
                             input_hash=hashlib.sha256(f"{agent}|{year}".encode()).hexdigest())
                previous[agent] = event["chosen_topic"]
                writer.direction_events.append(event)
            for pid, paper in manuscripts.items():
                if paper["first_submission_year"] == year:
                    writer.neutral_history.registry[pid] = deepcopy(paper)
                if paper["first_submission_year"] <= year:
                    accepted_year = {"published": 1, "late": 5, "never": 11, "year8": 8}[pid]
                    writer.paper_tracker.papers_by_id[pid] = SimpleNamespace(
                        status="accept" if year >= accepted_year else "reject")
                    # Deliberately nonzero raw counts on unpublished manuscripts.
                    # The driver emits zero realized publication citations for them.
                    writer.counts[pid] = {
                        "published": 7 + 3 * (year - 4) if year >= 4 else year - 1,
                        "late": 2 + year - 5 if year >= 5 else 3,
                        "never": 5, "year8": 99 if year == 10 else 0,
                    }[pid]
            awards = [] if empty else [
                {"year": year, "agent_id": FOUNDERS[0], "program_id": program,
                 "earned_amount": 20.0, "spendable_credit": 20.0}
                for program in ("NSF_THEORY", "DARPA_AUTONOMOUS")]
            rows = []
            for agent in FOUNDERS:
                papers = [pid for pid, p in manuscripts.items()
                          if p["author_id"] == agent and p["first_submission_year"] <= year]
                balance = max(0, balances[agent] - 10) if agent in eligible else balances[agent]
                if agent == FOUNDERS[0] and not empty:
                    balance += 40
                if agent not in end:
                    balance = 0
                rows.append({
                    "agent_id": agent, "active": agent in end, "spendable_resources": float(balance),
                    "cumulative_earned_funding": float(year * 40 if agent == FOUNDERS[0] and not empty else 0),
                    "submitted_papers": len(papers),
                    "accepted_papers": sum(writer.paper_tracker.papers_by_id[pid].status == "accept" for pid in papers),
                    "citations_all_papers": sum(writer.counts[pid] for pid in papers),
                })
            writer.funding_feedback_years[year] = {
                "year": year, "cell": "P1F1", "awards": awards, "agent_rows": rows, "legacy_zero_boundary": {}}
            writer.propensity_years[year] = {
                "year": year, "cell": cell, "founder_count": 1200,
                "year_start_active_ids": list(active), "phase1_eligible_ids": eligible,
                "phase0_lost_active_ids": phase0, "phase1_available_ids": list(eligible),
                "phase1_not_due_ids": [], "phase1_inactive_ids": [a for a in FOUNDERS if a not in eligible],
                "year_start_resources": [{"agent_id": a, "active": a in active, "resources": balances[a]}
                                         for a in FOUNDERS],
                "after_phase0_resources": [
                    {"agent_id": a, "active": a in eligible, "resources": 0.0 if a in phase0 else balances[a]}
                    for a in FOUNDERS],
                "annual_charges": [
                    {"agent_id": a, "amount": -10, "resources_before": balances[a],
                     "resources_after": max(0.0, balances[a] - 10), "active_before": True,
                     "active_after": True} for a in eligible],
            }
            writer.run_one_round(year)
            years.append(writes[f"{cell}/years/year_{year:02d}.json"])
            mechanism_years.append(deepcopy(writer.funding_feedback_years[year]))
            mechanism_years[-1]["agent_rows"] = years[-1]["agent_rows"]
            balances = {row["agent_id"]: row["spendable_resources"] for row in rows}
            active = list(end)
        manifests[cell]["history_coverage"] = deepcopy(years[-1]["history_coverage"])
        records[cell] = {
            "years": years, "mechanisms_years": mechanism_years,
            "direction_events": [event for row in years for event in row["direction_events"]],
            "submission_registry": writes[f"{cell}/submission_registry.json"],
            "direction_embedding_norms": dict.fromkeys(policy.CANONICAL_TOPICS, 1.0),
        }
    return manifests, records


def fixture(empty=False):
    return deepcopy(_fixture(empty))


def analyze(manifests, records):
    return analysis.analyze_records(manifests, records, expected_source=SOURCE)


def repeat(records, cell="LN", year=2):
    return next(row for row in records[cell]["direction_events"] if row["year"] == year)


def paper(records, pid="published", cell="LN"):
    return next(row for row in records[cell]["submission_registry"]["papers"] if row["paper_id"] == pid)


def observation(records, year=4, pid="published", cell="LN"):
    return next(row for row in records[cell]["years"][year - 1]["paper_observations"] if row["paper_id"] == pid)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def write_funding_evidence(directory, manifest, n_years, *, empty=False):
    """Exercise actual native funding/transport and the real closed audit helpers.

    Reuse the sequential suite's AST native-method fixture and scripted SDK.
    The tokenizer is a labelled CPU fixture, never the production Qwen tokenizer.
    """
    import tests.support.sequential_funding as native
    import utopia.funding.validation as compact

    directory.mkdir(parents=True, exist_ok=True)
    applications_dir = directory / "native"
    applications_dir.mkdir()
    manifest["args"] = dict(manifest.get("args", {}), output_dir=str(applications_dir))
    funding_type, model_type = native.funding_class(), native.model_class()
    guard = compact.install_funding_validation(
        funding_type, directory / "funding_validation.jsonl", compact=True)
    handle = sequential.install_sequential_funding(
        funding_type, model_type, directory / "funding_sequential.jsonl")
    audit = native.audit_module.RequestAudit(
        directory / "llm_request_audit.jsonl", _tokenizer=native.TinyTokenizer())
    model = model_type()
    model.model_name, model.run_seed, model.enable_thinking = native.audit_module.MODEL, 42, True
    model.max_concurrent_requests = 2
    model.request_audit = audit
    audit.bind(model.model_name)
    model.client = SimpleNamespace(max_retries=5, chat=SimpleNamespace(completions=native.SDK()))
    model.call_stats = {name: 0 for name in (
        "n_prompts", "n_first_attempt_success", "n_retries", "n_failures",
        "elapsed_seconds", "prompt_tokens", "completion_tokens")}
    seed_module = ModuleType("utopia.utils.seeding")
    seed_module.derive_seed = NATIVE_SEED
    success = False
    try:
        with patch.dict(sys.modules, {seed_module.__name__: seed_module}):
            model.generate_batch(["Labelled CPU non-funding initialization fixture"],
                                 seed_ctx=("phase1_fixture", 1, 0))
            for year in range(1, n_years + 1):
                handle.begin_year(year, ["NSF", "DARPA"])
                prompts, metadata, programs = [], [], {}
                response_format = None
                for aid in ("NSF", "DARPA"):
                    agency = funding_type()
                    agency.id = aid
                    agency.funding_programs = {
                        pid: SimpleNamespace(name=pid, research_directions=[native.Direction("algorithms")],
                                             funding_rate=rate)
                        for pid, rate in feedback.FROZEN_FUNDING_PROGRAM_RATES.items()
                        if pid.startswith(aid + "_")}
                    programs.update(agency.funding_programs)
                    pid = "NSF_THEORY" if aid == "NSF" else "DARPA_AUTONOMOUS"
                    applications = [] if empty else [{pid: {
                        "submit": True, "research_proposal": "Scripted CPU fixture proposal.",
                        "author": SimpleNamespace(id=FOUNDERS[0], expertise=[native.Direction("algorithms")],
                                                  university_name="Fixture University"),
                        "relevant_projects": [],
                    }}]
                    values = agency.get_funding_evaluation_prompts(
                        applications, {}, panel_max_apps=25,
                        panel_seed=NATIVE_SEED(42, "funding_panels", year))
                    prompts.extend(values[0])
                    response_format = values[1]
                    metadata.extend(values[2])
                if prompts:
                    results = model.generate_batch(prompts, response_format=response_format, max_tokens=8192,
                                                   seed_ctx=("phase5_funding_eval", year, 0))
                    log = []
                    funding_type.process_funding_evaluation_results(
                        results, metadata, programs, application_log=log)
                    for row in log:
                        row["year"] = year
                    (applications_dir / f"funding_applications_year_{year}.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in log))
                handle.end_year(year)
        success = True
    finally:
        audit.close(success)
        try:
            handle.restore()
        finally:
            guard.restore()
    manifest.update(funding_validation=guard.summary(), request_audit=audit.summary(),
                    funding_sequential=handle.summary(),
                    funding_output_representation="ordered_application_ids_v1",
                    funding_selection_protocol=analysis.SEQUENTIAL_PROTOCOL,
                    sequential_source_binding=feedback.sequential_source_binding(),
                    sequential_audit_valid=True, funding_application_evidence_valid=True)
    write_json(directory / "funding_validation.summary.json", manifest["funding_validation"])
    manifest["funding_sequential_audits"] = {
        name: mechanisms.file_hash(directory / name)
        for name in ("funding_sequential.jsonl", "funding_sequential.summary.json")}
    manifest["funding_application_evidence"] = feedback.funding_ledger_evidence(applications_dir, n_years)
    write_json(directory / "run_manifest.json", {
        "status": "complete" if n_years else "initialization_only_complete",
        "years_completed": n_years, "scientific_years_completed": n_years})


class EndpointTests(unittest.TestCase):
    def test_actual_serializer_emits_consumed_contract_without_convenience_outcomes(self):
        manifests, records = fixture()
        for cell in analysis.CELLS:
            self.assertNotIn("source_sha256", manifests[cell])
            self.assertEqual(manifests[cell]["inputs"]["source_files_sha256"], SOURCE)
            for row, mechanism in zip(records[cell]["years"], records[cell]["mechanisms_years"]):
                self.assertFalse({"phase0_deactivated_ids", "year_end_active_ids", "earned_funding"} & set(row))
                self.assertEqual(mechanism["cell"], "P1F1")
                self.assertEqual(row["agent_rows"], mechanism["agent_rows"])
                expected_ids = {p["paper_id"] for p in records[cell]["submission_registry"]["papers"]
                                if p["first_submission_year"] <= row["year"]}
                self.assertEqual({p["paper_id"] for p in row["paper_observations"]}, expected_ids)
            for manuscript in records[cell]["submission_registry"]["papers"]:
                self.assertFalse({"accepted_year", "citations_by_year"} & set(manuscript))
                if manuscript["first_submission_year"] <= 7:
                    self.assertEqual(manuscript["three_year_observation"], observation(
                        records, manuscript["first_submission_year"] + 3, manuscript["paper_id"], cell))
        self.assertEqual(analyze(manifests, records)["status"], "complete")

    def test_exact_entropy_denominator_all_stays_and_trapezoidal_weights(self):
        topics = policy.CANONICAL_TOPICS
        self.assertEqual(analysis.normalized_entropy({}), 0)
        self.assertEqual(analysis.normalized_entropy({topics[0]: 1200}), 0)
        self.assertAlmostEqual(analysis.normalized_entropy(dict.fromkeys(topics, 1)), 1)
        self.assertAlmostEqual(analysis.normalized_entropy({topics[0]: 1, topics[1]: 1}),
                               math.log(2) / math.log(53))
        self.assertEqual(analysis.entropy_auc([1] * 10), 9)
        self.assertEqual(analysis.entropy_auc([0] + [1] * 9), 8.5)
        self.assertEqual(analysis.entropy_auc([1] + [0] * 9), 0.5)
        for counts in ({"invented": 1}, {topics[0]: -1}, {topics[0]: True}, {topics[0]: 1.5}):
            with self.assertRaises(analysis.AnalysisError):
                analysis.normalized_entropy(counts)
        with self.assertRaises(analysis.AnalysisError):
            analysis.entropy_auc([1] * 9)

    def test_all_four_values_simple_main_effects_and_two_negative_propensity_effects(self):
        values = {"LN": 5, "LF": 4, "HN": 4, "HF": 1}
        result = analysis.factorial_contrasts(values)
        self.assertEqual(result["cell_values"], values)
        self.assertEqual(result["propensity_simple_effects"], {"near_history": -1, "far_history": -3})
        self.assertEqual(result["menu_simple_effects_near_minus_far"],
                         {"low_propensity": 1, "high_propensity": 3})
        self.assertEqual(result["averaged_main_effects"],
                         {"high_minus_low_propensity": -2, "near_minus_far_history": 2})
        self.assertEqual(result["interaction"], 2)
        self.assertTrue(result["positive_relative_interaction_in_seed42"])
        self.assertFalse(result["near_history_propensity_benefit_in_seed42"])
        self.assertFalse(result["HN_exceeds_both_one_factor_alternatives"])
        self.assertFalse(analysis.factorial_contrasts(dict.fromkeys(analysis.CELLS, 3))[
            "positive_relative_interaction_in_seed42"])

    def test_complete_four_cell_counts_participation_acceptance_and_citation_deadlines(self):
        manifests, records = fixture()
        result = analyze(manifests, records)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["world_replicates"], 1)
        self.assertEqual(set(result["primary"]["cell_values"]), set(analysis.CELLS))
        for cell, output in result["cells"].items():
            manipulation, secondary = output["manipulation"], output["secondary"]
            self.assertEqual(manipulation["initial_choices"], 1200)
            self.assertEqual(manipulation["eligible_repeat_choices"], 13)
            self.assertEqual(manipulation["year_start_active_person_years"], 1214)
            self.assertEqual(manipulation["eligible_repeat_per_year_start_active"], 13 / 1214)
            self.assertEqual(manipulation["switches"] + manipulation["stays"], 13)
            self.assertEqual(manipulation["conditional_switch_distance"]["n"], manipulation["switches"])
            self.assertEqual(sum(manipulation["reference_counts"].values()), 13)
            self.assertEqual(sum(row["parse"] + row["validation"]
                                 for row in manipulation["fallbacks"].values()), 2)
            self.assertEqual(secondary["year10_active_fraction"], 1 / 1200)
            self.assertEqual(secondary["cumulative_earned_funding_per_founder"], 400 / 1200)
            self.assertEqual(secondary["accepted_unique_manuscripts"], 3)
            self.assertEqual(secondary["first_attempt_acceptance_rate"], 0.5)
            citation = secondary["citation_yield"]
            self.assertEqual(citation["eligible_manuscripts"], 3)
            self.assertEqual(citation["published_by_deadline"], 1)
            self.assertEqual(citation["unpublished_by_deadline"], 2)
            self.assertEqual(citation["total"], 7)
            self.assertEqual(citation["per_founder"], 7 / 1200)
            self.assertEqual(citation["mean_per_eligible_manuscript"], 7 / 3)
            self.assertEqual({row["paper_id"]: row["citations"] for row in citation["manuscripts"]},
                             {"published": 7, "late": 0, "never": 0})
            annual = output["annual"]
            self.assertEqual(annual[0]["new_project_choices"], 1200)
            self.assertEqual(annual[0]["eligible_repeat_choices"], 0)
            self.assertEqual(annual[5]["phase0_lost_active"], 1)
            self.assertEqual(annual[5]["risk_sets"]["phase0_lost_active_ids"], [FOUNDERS[1]])
            expected_auc = sum((annual[i]["entropy"] + annual[i + 1]["entropy"]) / 2
                               for i in range(9))
            self.assertAlmostEqual(output["entropy_auc"], expected_auc)
            weighted = [row["new_project_choices"] / 1200 * row["entropy"] for row in annual]
            self.assertAlmostEqual(secondary["participation_weighted_entropy_auc"],
                                   sum((weighted[i] + weighted[i + 1]) / 2 for i in range(9)))
        json.dumps(result, allow_nan=False)

    def test_zero_activity_years_zero_repeat_and_zero_manuscript_denominators_are_explicit(self):
        result = analyze(*fixture(empty=True))
        for output in result["cells"].values():
            manipulation, secondary = output["manipulation"], output["secondary"]
            self.assertIsNone(manipulation["realized_switch_propensity"])
            self.assertIsNone(manipulation["conditional_switch_distance"]["mean"])
            self.assertIsNone(manipulation["conditional_switch_distance"]["median"])
            self.assertEqual(manipulation["conditional_switch_distance"]["values"], [])
            self.assertEqual(manipulation["eligible_repeat_per_year_start_active"], 0)
            self.assertIsNone(secondary["first_attempt_acceptance_rate"])
            self.assertIsNone(secondary["citation_yield"]["mean_per_eligible_manuscript"])
            for row in output["annual"][1:]:
                self.assertEqual(row["entropy"], 0)
                self.assertTrue(row["no_activity_convention"])
                self.assertIsNone(row["realized_switch_propensity"])
                self.assertIsNone(row["eligible_repeat_per_year_start_active"])
                self.assertIsNone(row["switches_per_year_start_active"])
            self.assertEqual(output["entropy_auc"], output["annual"][0]["entropy"] / 2)

    def test_no_mutation_no_global_rng_no_file_io_or_inferential_outputs(self):
        manifests, records = fixture()
        before = json.dumps([manifests, records], sort_keys=True)
        state = random.getstate()
        with patch("builtins.open", side_effect=AssertionError("pure API performed IO")):
            result = analyze(manifests, records)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(json.dumps([manifests, records], sort_keys=True), before)
        text = json.dumps(result)
        for forbidden in ('"p_value"', '"confidence_interval"', '"standard_error"', '"bootstrap"'):
            self.assertNotIn(forbidden, text)
        self.assertIsNone(result["world_standard_error"])
        self.assertIsNone(result["world_confidence_interval"])
        self.assertEqual(result["funding_selection_protocol"], "sequential_remaining_ids_v1")
        for agent in FOUNDERS[:3]:
            for year in (2, 6, 10):
                seed, uniform = analysis._expected_gate(agent, year)
                self.assertEqual(seed, NATIVE_SEED(42, policy.GATE_NAMESPACE, agent, year))
                self.assertEqual(uniform, random.Random(seed).random())


class AdmissionTests(unittest.TestCase):
    def test_old_partial_worlds_missing_sequential_coverage_and_inference_rejected(self):
        baseline, records = fixture()
        changes = [
            lambda m: m.update(protocol="switching-propensity-v1-single-world"),
            lambda m: m.update(protocol="switching-propensity-v2-single-world"),
            lambda m: m.update(scientific_years_completed=1),
            lambda m: m.update(funding_selection_protocol="ordered_application_ids_v1"),
            lambda m: m["funding_sequential"].update(agency_registrations=19),
            lambda m: m["funding_sequential"].update(years_started=9, years_completed=9),
            lambda m: m["funding_sequential"].update(installed=True),
            lambda m: m["funding_sequential"].update(processed_panels=19),
            lambda m: m["funding_sequential"].update(fatal_errors=1),
            lambda m: m["funding_sequential"]["source_files_sha256"].update(untrusted="f" * 64),
            lambda m: m["inputs"].update(sequential_source_binding={}),
            lambda m: m.update(initialization_manifest_sha256="0" * 64),
            lambda m: m.update(world_standard_error=0),
            lambda m: m.update(world_confidence_interval=[0, 0]),
        ]
        for change in changes:
            manifests = deepcopy(baseline)
            change(manifests["HF"])
            with self.subTest(change=change), self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_incomplete_extra_failed_initialization_wrong_seed_source_and_initial_state_rejected(self):
        cases = [
            lambda m, r: m.pop("HF"),
            lambda m, r: r.pop("HF"),
            lambda m, r: m.update(EXTRA=deepcopy(m["LN"])),
            lambda m, r: m["HF"].update(status="failed"),
            lambda m, r: m["HF"].update(status="out_of_domain"),
            lambda m, r: m["HF"].update(kind="initialization_only"),
            lambda m, r: m["HF"].update(seed=43),
            lambda m, r: m["HF"].update(seed=True),
            lambda m, r: m["HF"].update(founder_count=1201),
            lambda m, r: m["HF"].update(years_completed=9),
            lambda m, r: m["HF"]["inputs"].update(source_files_sha256={"stale": "a" * 64}),
            lambda m, r: m["HF"].update(initial_choices_sha256="9" * 64),
            lambda m, r: m["HF"].update(initial_state_hash="8" * 64),
            lambda m, r: m["HF"].update(pre_choice_state_hash="7" * 64),
        ]
        for change in cases:
            with self.subTest(change=change):
                manifests, records = fixture()
                change(manifests, records)
                with self.assertRaises(analysis.AnalysisError):
                    analyze(manifests, records)
        manifests, records = fixture()
        for manifest in manifests.values():
            manifest["inputs"]["source_files_sha256"] = {"mutually_consistent_but_stale": "a" * 64}
        with self.assertRaisesRegex(analysis.AnalysisError, "source_mismatch"):
            analyze(manifests, records)
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze_records(*fixture(), expected_source={})

    def test_final_funding_imputation_failures_and_request_incompleteness_never_admitted(self):
        for key in analysis._ZERO_FUNDING:
            manifests, records = fixture()
            manifests["LN"]["funding_validation"][key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(analysis.AnalysisError, "funding_not_complete"):
                analyze(manifests, records)
        for key, value in (("completion_allowed", False), ("installed", True),
                           ("output_representation", "ranked_applications_v1"),
                           ("restore_conflicts", ["method"]), ("imputed_rankings", False)):
            manifests, records = fixture()
            manifests["LN"]["funding_validation"][key] = value
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)
        for change in ({"status": "failed"}, {"pending_requests": 1}, {"n_requests": 0},
                       {"n_responses": 8}, {"n_clients": True}):
            manifests, records = fixture()
            manifests["LN"]["request_audit"].update(change)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_missing_duplicate_stale_years_and_events_fail(self):
        changes = [
            lambda r: r["LN"]["years"].pop(),
            lambda r: r["LN"]["years"].append(deepcopy(r["LN"]["years"][-1])),
            lambda r: r["LN"]["years"][1].update(year=1),
            lambda r: r["LN"]["years"][1].update(year=11),
            lambda r: r["LN"]["years"][1].update(cell="HF"),
            lambda r: r["LN"]["direction_events"].pop(),
            lambda r: r["LN"]["direction_events"].append(deepcopy(r["LN"]["direction_events"][-1])),
            lambda r: repeat(r).update(seed=43),
            lambda r: repeat(r).update(status="out_of_domain"),
        ]
        for change in changes:
            manifests, records = fixture()
            change(records)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_risk_sets_initial_roster_and_phase0_exclusions_are_not_optional(self):
        for change in (
            lambda r: r["LN"]["years"][0]["year_start_active_ids"].pop(),
            lambda r: r["LN"]["years"][1]["phase1_eligible_ids"].pop(),
            lambda r: r["LN"]["years"][5].update(phase0_lost_active_ids=[]),
            lambda r: r["LN"]["years"][6]["year_start_active_ids"].append(FOUNDERS[1]),
            lambda r: r["LN"]["years"][1]["phase1_available_ids"].pop(),
            lambda r: r["LN"]["years"][1]["phase1_inactive_ids"].pop(),
            lambda r: r["LN"]["years"][1]["annual_charges"].pop(),
            lambda r: r["LN"]["years"][1]["year_start_resources"].pop(),
            lambda r: r["LN"]["years"][1]["after_phase0_resources"].pop(),
        ):
            manifests, records = fixture()
            change(records)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_initial_stay_switch_counts_dates_fallbacks_and_choice_identity_checked(self):
        changes = [
            lambda r: r["LN"]["direction_events"][0].update(realized_switch=True),
            lambda r: r["LN"]["direction_events"][0].update(chosen_topic=policy.CANONICAL_TOPICS[-1]),
            lambda r: repeat(r).update(initial_choice_count=1),
            lambda r: repeat(r).update(eligible_repeat=False),
            lambda r: repeat(r).update(project_end_year=3),
            lambda r: r["LN"]["years"][1]["annual_charges"][0].update(amount=0),
            lambda r: repeat(r).update(previous_topic="invented"),
            lambda r: repeat(r).update(fallback_kind="unlogged repair"),
        ]
        for change in changes:
            manifests, records = fixture()
            change(records)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_gate_menu_domain_and_distance_audit_mismatches_fail(self):
        changes = [
            lambda row: row.update(u=0.5),
            lambda row: row.update(gate_seed=0),
            lambda row: row.update(p=0.75),
            lambda row: row.update(requested_switch=not row["requested_switch"]),
            lambda row: row.update(near_topics=list(reversed(row["near_topics"]))),
            lambda row: row.update(candidate_topics=[]),
            lambda row: row.update(near_min=99),
            lambda row: row.update(n_near=True),
            lambda row: row.update(requested_action="stay" if row["requested_switch"] else "switch"),
            lambda row: row.update(separation_gap=0),
            lambda row: row["distance_map"].pop(next(iter(row["distance_map"]))),
            lambda row: row["distance_map"].update({next(iter(row["distance_map"])): math.nan}),
            lambda row: row.update(distance_map=dict.fromkeys(row["distance_map"], 1.0)),
        ]
        for change in changes:
            manifests, records = fixture()
            change(repeat(records))
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_history_ingestion_missing_papers_wrong_window_and_false_cold_start_fail(self):
        changes = [
            lambda r: r["LN"]["years"][0]["history_coverage"].update(missing_papers=1),
            lambda r: r["LN"]["years"][0]["history_coverage"].update(required_papers=2),
            lambda r: repeat(r).update(history_ids=[]),
            lambda r: repeat(r).update(reference_source="initial_expertise"),
            lambda r: repeat(r).update(reference_norm=0),
            lambda r: repeat(r).update(history_window_start=1),
            lambda r: repeat(r).update(expected_history_count=True),
            lambda r: repeat(r, year=5).update(history_window_start=True),
        ]
        for change in changes:
            manifests, records = fixture()
            change(records)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_consistent_geometric_tie_has_distinct_out_of_domain_error(self):
        manifests, records = fixture()
        event = repeat(records)
        ranked = sorted(event["distance_map"])
        event.update(distance_map=dict.fromkeys(ranked, 1.0),
                     ranked_eligible_topics=ranked, near_topics=ranked[:17], far_topics=ranked[-17:],
                     near_min=1.0, near_max=1.0, far_min=1.0, far_max=1.0, separation_gap=0.0)
        with self.assertRaisesRegex(analysis.AnalysisError, "^out_of_domain"):
            analyze(manifests, records)

    def test_event_and_agent_copies_must_match_without_shared_python_references(self):
        for key in ("direction_events", "agent_rows"):
            manifests, records = fixture()
            raw_year = records["LN"]["years"][1]
            raw_year[key] = deepcopy(raw_year[key])
            if key == "direction_events":
                raw_year[key][0]["reason"] = "stale content"
            else:
                raw_year[key][0]["active"] = not raw_year[key][0]["active"]
            with self.assertRaisesRegex(analysis.AnalysisError, "sidecar_mismatch"):
                analyze(manifests, records)

    def test_published_citation_observation_missing_is_not_zero_or_later_count(self):
        for value in (None, True, -1, math.nan):
            manifests, records = fixture()
            observation(records)["citation_count"] = value
            with self.subTest(value=value), self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)
        for change in (
            lambda r: r["LN"]["years"][3]["paper_observations"].pop(0),
            lambda r: paper(r).pop("three_year_observation"),
            lambda r: paper(r)["three_year_observation"].update(year=5),
            lambda r: paper(r)["three_year_observation"].update(citation_count=99),
            lambda r: observation(r).pop("citation_count"),
            lambda r: observation(r, pid="never").pop("citation_count"),
            lambda r: observation(r, pid="never").pop("realized_publication_citations"),
            lambda r: observation(r, pid="late").update(realized_publication_citations=1),
            lambda r: observation(r, year=10).update(status="pending"),
        ):
            manifests, records = fixture()
            change(records)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_manuscript_identity_first_year_publication_status_and_acceptance_are_required(self):
        changes = [
            lambda p: p.update(paper_id="different"),
            lambda p: p.update(author_id="unknown"),
            lambda p: p.update(first_submission_year=0),
            lambda p: p.update(first_submission_year=True),
            lambda p: p.update(first_attempt_accepted=False),
            lambda p: p.pop("first_attempt_accepted"),
            lambda p: p.pop("abstract_sha256"),
        ]
        for change in changes:
            manifests, records = fixture()
            change(paper(records))
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)

    def test_ledger_counts_norms_and_duplicated_raw_records_are_checked(self):
        changes = [
            lambda m, r: r["LN"]["mechanisms_years"].pop(),
            lambda m, r: r["LN"]["mechanisms_years"][0].update(cell="LN"),
            lambda m, r: r["LN"]["mechanisms_years"][0]["awards"].pop(),
            lambda m, r: r["LN"]["mechanisms_years"][0]["awards"][0].update(spendable_credit=0),
            lambda m, r: r["LN"]["mechanisms_years"][0]["awards"][0].update(earned_amount=19),
            lambda m, r: r["LN"]["years"][0]["agent_rows"][0].update(cumulative_earned_funding=0),
            lambda m, r: r["LN"]["years"][0]["agent_rows"][0].update(citations_all_papers=99),
            lambda m, r: r["LN"]["years"][0]["agent_rows"][0].pop("cumulative_earned_funding"),
            lambda m, r: r["LN"]["years"][0]["agent_rows"].append(deepcopy(r["LN"]["years"][0]["agent_rows"][0])),
            lambda m, r: r["LN"]["submission_registry"]["papers"].append(deepcopy(paper(r))),
            lambda m, r: r["LN"]["submission_registry"].update(year=9),
            lambda m, r: r["LN"]["years"][0]["paper_observations"].append(deepcopy(observation(r, year=1))),
            lambda m, r: r["LN"]["direction_embedding_norms"].pop(policy.CANONICAL_TOPICS[0]),
            lambda m, r: r["LN"]["direction_embedding_norms"].update({policy.CANONICAL_TOPICS[0]: 0}),
            lambda m, r: m["LN"]["funding_baseline"].update(funding_budget_mode="fixed"),
            lambda m, r: m["LN"]["funding_baseline"].update(panel_max_apps=26),
        ]
        for change in changes:
            manifests, records = fixture()
            change(manifests, records)
            with self.assertRaises(analysis.AnalysisError):
                analyze(manifests, records)


class FileWrapperTests(unittest.TestCase):
    def write_fixture(self, root, *, empty=False):
        from tests.support.switching import tp1_manifest_runtime_fixture
        manifests, records = fixture(empty)
        runtime = tp1_manifest_runtime_fixture(root)
        self.source = driver.source_identity()
        binding = feedback.sequential_source_binding()
        for manifest in manifests.values():
            manifest["inputs"].update(source_files_sha256=self.source, sequential_source_binding=binding)
            manifest["inputs"].update(deepcopy(runtime["inputs"]))
            manifest["server_provenance"] = deepcopy(runtime["server_provenance"])
            manifest["args"] = deepcopy(runtime["args"])
            manifest["sequential_source_binding"] = binding
        initial = root / driver.identifier()
        initial.mkdir()
        cache_path = initial / "initial_choices.json"
        producer = deepcopy(manifests["LN"])
        producer.update(cell=None, kind="initialization_only", scientific_years_completed=0,
                        initial_choices_path=str(cache_path))
        producer.pop("years_completed")
        first_events = [row for row in records["LN"]["direction_events"] if row["year"] == 1]
        cache_records = [{
            "agent_id": row["agent_id"], "item_index": index, "year": 1,
            "candidate_topics": row["candidate_topics"], "fallback_kind": row["fallback_kind"],
            "input_hash": row["input_hash"], "seed_ctx": ["phase1_directions", 1],
            "request_attempts": [{"request_seed": NATIVE_SEED(42, "phase1_directions", 1, index, 0)}],
            "response": {"topic": row["chosen_topic"], "detailed_focus": row["detailed_focus"],
                         "reason": row["reason"]},
        } for index, row in enumerate(first_events)]
        cache = {
            "schema_version": 1, "protocol": analysis.PROTOCOL, "status": "complete", "seed": 42,
            "founder_count": 1200, "choice_count": 1200, "founder_ids": FOUNDERS,
            "inputs": deepcopy(producer["inputs"]), "pre_choice_state_hash": producer["pre_choice_state_hash"],
            "initial_state_hash": producer["initial_state_hash"],
            "records": cache_records, "records_sha256": provenance.digest(cache_records)}
        write_json(cache_path, cache)
        cache_path.chmod(0o444)
        producer["initial_choices_sha256"] = mechanisms.file_hash(cache_path)
        write_funding_evidence(initial, producer, 0)
        initial_manifest = initial / "initialization_manifest.json"
        write_json(initial_manifest, producer)
        directories = {}
        for cell in analysis.CELLS:
            directory = root / driver.identifier(cell)
            directory.mkdir()
            directories[cell] = directory
            (directory / "years").mkdir()
            manifests[cell].update(initial_choices_path=str(cache_path),
                                   initial_choices_sha256=producer["initial_choices_sha256"],
                                   initialization_manifest_path=str(initial_manifest),
                                   initialization_manifest_sha256=mechanisms.file_hash(initial_manifest))
            write_funding_evidence(directory, manifests[cell], 10, empty=empty)
            for name, value in (
                ("propensity_manifest.json", manifests[cell]),
                ("submission_registry.json", records[cell]["submission_registry"]),
                ("direction_embedding_norms.json", records[cell]["direction_embedding_norms"]),
                ("funding_validation.summary.json", manifests[cell]["funding_validation"]),
                ("llm_request_audit.summary.json", manifests[cell]["request_audit"]),
            ):
                (directory / name).write_text(json.dumps(value))
            for year in records[cell]["years"]:
                (directory / "years" / f"year_{year['year']:02d}.json").write_text(json.dumps(year))
            for year in records[cell]["mechanisms_years"]:
                (directory / f"mechanisms_year_{year['year']}.json").write_text(json.dumps(year))
            (directory / "direction_events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in records[cell]["direction_events"]))
        return manifests, records, directories

    def test_actual_file_layout_matches_pure_api_and_missing_year_or_sidecar_rejected(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            manifests, records, directories = self.write_fixture(Path(temporary))
            expected = analysis.analyze_records(manifests, records, expected_source=self.source)
            before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in Path(temporary).rglob("*") if path.is_file()}
            result = analysis.analyze_files(directories, expected_source=self.source)
            self.assertEqual(result["cells"], expected["cells"])
            self.assertEqual(result["primary"], expected["primary"])
            self.assertEqual(result["evidence_validation"],
                             "closed_raw_audits_native_ledgers_and_common_producer")
            self.assertEqual(result["initialization"]["manifest_sha256"],
                             manifests["LN"]["initialization_manifest_sha256"])
            after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in Path(temporary).rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            sidecar = directories["HF"] / "funding_validation.summary.json"
            sidecar.write_text("{}")
            with self.assertRaisesRegex(analysis.AnalysisError, "audit_sidecar_mismatch"):
                analysis.analyze_files(directories, expected_source=self.source)
            sidecar.write_text(json.dumps(manifests["HF"]["funding_validation"]))
            (directories["HF"] / "years/year_10.json").unlink()
            with self.assertRaisesRegex(analysis.AnalysisError, "incomplete_or_extra_year_files"):
                analysis.analyze_files(directories, expected_source=self.source)

    def test_duplicate_json_fields_and_nonfinite_json_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            _, _, directories = self.write_fixture(Path(temporary))
            path = directories["LN"] / "direction_events.jsonl"
            for text in ('{"year": 1, "year": 2}\n', '{"year": NaN}\n'):
                path.write_text(text)
                with self.assertRaises(analysis.AnalysisError):
                    analysis.analyze_files(directories, expected_source=self.source)

    def test_trusted_source_gate_precedes_all_raw_helpers_and_disallows_mixed_initializers(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            manifests, _, directories = self.write_fixture(Path(temporary))
            path = directories["HF"] / "propensity_manifest.json"
            stale = deepcopy(manifests["HF"])
            stale["inputs"]["source_files_sha256"]["utopia/funding/sequential.py"] = "0" * 64
            write_json(path, stale)
            with patch.object(driver, "validate_funding_evidence") as verifier:
                with self.assertRaisesRegex(analysis.AnalysisError, "source_mismatch"):
                    analysis.analyze_files(directories, expected_source=self.source)
                verifier.assert_not_called()
            write_json(path, manifests["HF"])
            manifests["HF"]["initialization_manifest_sha256"] = "0" * 64
            write_json(path, manifests["HF"])
            with self.assertRaisesRegex(analysis.AnalysisError, "producer_manifest_hash_mismatch"):
                analysis.analyze_files(directories, expected_source=self.source)

    def test_producer_worktree_alias_resolves_to_same_immutable_private_cache(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            root = Path(temporary)
            manifests, _, _ = self.write_fixture(root)
            path = Path(manifests["LN"]["initialization_manifest_path"])
            producer = analysis._read_json(path)
            alias = root / "worktree_outputs_alias"
            alias.symlink_to(path.parent, target_is_directory=True)
            producer["initial_choices_path"] = str(alias / "initial_choices.json")
            write_json(path, producer)
            for manifest in manifests.values():
                manifest["initialization_manifest_sha256"] = mechanisms.file_hash(path)
            cache, report = analysis._validate_producer(manifests, self.source)
            self.assertEqual(cache["founder_ids"], FOUNDERS)
            self.assertEqual(report["initial_choices_sha256"], producer["initial_choices_sha256"])

    def test_alias_written_four_cell_files_and_cache_pass_without_canonicalizing_raw_paths(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-alias-") as temporary:
            root = Path(temporary)
            private_outputs = root / "private" / "outputs"
            (private_outputs / "docs").mkdir(parents=True)
            worktree = root / "worktree"
            worktree.mkdir()
            (worktree / "outputs").symlink_to(private_outputs, target_is_directory=True)
            manifests, records, directories = self.write_fixture(worktree / "outputs/docs")
            before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in private_outputs.rglob("*") if path.is_file()}
            cache_path = Path(manifests["LN"]["initial_choices_path"])
            cache, _, _, _ = driver.validate_initialization(cache_path, manifests["LN"]["inputs"])
            self.assertEqual(cache["founder_ids"], FOUNDERS)
            expected = analysis.analyze_records(manifests, records, expected_source=self.source)
            result = analysis.analyze_files(directories, expected_source=self.source)
            self.assertEqual(result["primary"], expected["primary"])
            self.assertEqual(result["cells"], expected["cells"])
            canonical = {cell: path.resolve() for cell, path in directories.items()}
            self.assertTrue(all(canonical[cell] != path and canonical[cell].samefile(path)
                                for cell, path in directories.items()))
            with self.assertRaises(analysis.AnalysisError):
                analysis.analyze_files(canonical, expected_source=self.source)
            after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in private_outputs.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_exact_native_winners_reconcile_with_awards_not_just_total_funding(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            manifests, records, directories = self.write_fixture(Path(temporary))
            row = records["LN"]["mechanisms_years"][0]
            row["awards"][0]["program_id"] = "NSF_SYSTEM"
            # Amounts and recipients still total 40; only the actual program differs.
            write_json(directories["LN"] / "mechanisms_year_1.json", row)
            with self.assertRaisesRegex(analysis.AnalysisError, "native_winners_awards_mismatch"):
                analysis.analyze_files(directories, expected_source=self.source)
            row["awards"][0]["program_id"] = "NSF_THEORY"
            write_json(directories["LN"] / "mechanisms_year_1.json", row)
            # All four worlds agree with each other, but their initial content differs
            # from the single immutable producer cache.
            for cell in analysis.CELLS:
                events = records[cell]["direction_events"]
                events[0]["detailed_focus"] = "different initialization"
                (directories[cell] / "direction_events.jsonl").write_text(
                    "".join(json.dumps(event) + "\n" for event in events))
            with self.assertRaisesRegex(analysis.AnalysisError, "initial_event_cache_mismatch"):
                analysis.analyze_files(directories, expected_source=self.source)

    def test_rehashed_native_rate_rank_applicant_quota_and_funded_corruption_rejected(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            manifests, _, directories = self.write_fixture(Path(temporary))
            manifest, directory = manifests["LN"], directories["LN"]
            path = Path(manifest["args"]["output_dir"]) / "funding_applications_year_1.jsonl"
            original = list(analysis._read_jsonl(path))
            changes = [
                {"program_id": "UNKNOWN"}, {"funding_rate": .24}, {"funding_rate": .23, "program_id": "DARPA_AUTONOMOUS"},
                {"llm_rank": 2}, {"llm_rank": True}, {"position": 2}, {"applicant_id": FOUNDERS[1]},
                {"num_winners": 0}, {"n_panel": 2}, {"funded": False}, {"fallback_ranking": True},
                {"imputed_tail": True}, {"year": 2},
            ]
            for fields in changes:
                rows = deepcopy(original)
                rows[0].update(fields)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                manifest["funding_application_evidence"] = feedback.funding_ledger_evidence(path.parent, 10)
                with self.subTest(fields=fields), self.assertRaisesRegex(
                        analysis.AnalysisError, "funding_evidence_invalid"):
                    analysis._validate_funding_files(directory, manifest, self.source)
            path.write_text("".join(json.dumps(row) + "\n" for row in original * 2))
            manifest["funding_application_evidence"] = feedback.funding_ledger_evidence(path.parent, 10)
            with self.assertRaises(analysis.AnalysisError):
                analysis._validate_funding_files(directory, manifest, self.source)

    def test_raw_selection_and_independent_request_terminals_required_even_after_rehash(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            manifests, _, directories = self.write_fixture(Path(temporary))
            manifest, directory = manifests["LN"], directories["LN"]
            raw_path = directory / "funding_sequential.jsonl"
            original = raw_path.read_bytes()
            rows = list(analysis._read_jsonl(raw_path))
            returned = next(row for row in rows if row["event"] == "sdk_response")
            returned["raw_response"]["choices"][0]["message"]["content"] = '{"next_application_id":true}'
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest["funding_sequential_audits"][raw_path.name] = mechanisms.file_hash(raw_path)
            with self.assertRaisesRegex(analysis.AnalysisError, "funding_evidence_invalid"):
                analysis._validate_funding_files(directory, manifest, self.source)
            raw_path.write_bytes(original)
            manifest["funding_sequential_audits"][raw_path.name] = mechanisms.file_hash(raw_path)
            request_path = directory / "llm_request_audit.jsonl"
            rows = list(analysis._read_jsonl(request_path))
            terminal = next(row for row in rows if row["event"] == "response")
            rows.insert(-1, deepcopy(terminal))
            request_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(analysis.AnalysisError, "funding_evidence_invalid"):
                analysis._validate_funding_files(directory, manifest, self.source)

    def test_empty_years_have_positive_coverage_and_absence_is_distinct_from_empty_file(self):
        with tempfile.TemporaryDirectory(prefix="propensity-analysis-test-") as temporary:
            manifests, _, directories = self.write_fixture(Path(temporary), empty=True)
            result = analysis.analyze_files(directories, expected_source=self.source)
            for cell in analysis.CELLS:
                evidence = result["funding_evidence"][cell]
                self.assertEqual(evidence["funding_sequential"]["agency_registrations"], 20)
                self.assertEqual(evidence["funding_sequential"]["empty_agency_registrations"], 20)
                self.assertEqual(evidence["funding_application_evidence"],
                                 {"present_sha256": {}, "absent_years": list(range(1, 11))})
            self.assertEqual(result["initialization"]["funding_evidence"]["scientific_years"], 0)
            manifest, directory = manifests["LN"], directories["LN"]
            added = Path(manifest["args"]["output_dir"]) / "funding_applications_year_1.jsonl"
            added.touch()
            with self.assertRaisesRegex(analysis.AnalysisError, "funding_evidence_invalid"):
                analysis._validate_funding_files(directory, manifest, self.source)
            added.unlink()
            extra = added.with_name("funding_applications_year_11.jsonl")
            extra.touch()
            with self.assertRaisesRegex(analysis.AnalysisError, "funding_evidence_invalid"):
                analysis._validate_funding_files(directory, manifest, self.source)
            extra.unlink()
            initial = Path(manifest["initialization_manifest_path"]).parent
            # Zero funding calls still require an independent closed raw request log.
            request_path = initial / "llm_request_audit.jsonl"
            request_path.unlink()
            with self.assertRaises(analysis.AnalysisError):
                analysis._validate_producer(manifests, self.source)


if __name__ == "__main__":
    unittest.main()
