"""Strict descriptive analysis for the replacement four-cell propensity study.

API:
    analyze_records(manifests, records_by_cell, *, expected_source)
    analyze_files(cell_directories, *, expected_source)

Both mappings have exactly LN/LF/HN/HF keys. ``expected_source`` retains optional
historical source metadata; it does not control result acceptance.
No simulator/encoder/model is imported, no RNG state is changed, and no files
are written. There are no inferential intervals, p-values or extra seeds.

The input contract is the existing driver output, without schema conversion:

* manifest: status="complete", kind="condition", protocol=
  "switching-propensity-v3-single-world-sequential", cell, seed=42, founder_count=1200,
  years_completed=10, founder_ids, inputs.source_files_sha256, funding_baseline,
  initial_choices_sha256, initial_state_hash, pre_choice_state_hash,
  funding_validation and request_audit (terminal shared-helper summaries).
* records_by_cell[cell]: years (ten raw year objects), mechanisms_years (ten raw
  mechanisms objects), direction_events (the JSONL objects), submission_registry
  (the final wrapper object, including its papers list), direction_embedding_norms
  (the existing topic-to-norm object).
* Years retain phase1_* risk sets, phase0_lost_active_ids, resource snapshots,
  annual_charges, direction_events, agent_rows, history_coverage and exhaustive
  paper_observations. The native mechanisms label is P1F1 in every cell.
* Award amounts come from mechanisms_years.awards and must reconcile with each
  founder's cumulative_earned_funding. Both copies of agent_rows must agree.
* Publication history is derived from all ten years' paper_observations.
  Registry first_attempt_accepted and three_year_observation must match the
  exact observed first-submission and first-submission+3 years. Missing
  observations (published OR unpublished) are errors, never fabricated zeros.

File wrapper: each cell directory contains propensity_manifest.json,
direction_events.jsonl, submission_registry.json, years/year_01.json through
year_10.json, mechanisms_year_1.json through mechanisms_year_10.json,
direction_embedding_norms.json, funding_validation.summary.json, and
llm_request_audit.summary.json.
The terminal sidecars must equal their manifest copies. Duplicate JSON keys,
nonfinite numbers, duplicate events/years/manuscripts and extra cells/years fail.

The records API checks supplied records and terminal summaries without file IO.
Formal admission uses analyze_files, which also verifies the closed raw
sequential/request audits, native award ledgers, and the common fresh producer.
Admission compares the actual settings, founder IDs, choices and funding evidence.
The original-world descriptive companion is a separate existing-data analysis.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import median

from utopia.agents.switching_policy import CANONICAL_TOPICS, DISTANCE_BOUNDARY_TOLERANCE
from utopia.funding.sequential import (
    SEQUENTIAL_PROTOCOL,
    SequentialFundingError,
    validate_summary,
)
from utopia.runtime.historical import SWITCH_GATE_SEED_NAMESPACE, identifier_matches
from utopia.runtime.switching_evidence import scientific_inputs
from utopia.utils.data_utils import (
    DuplicateJSONKey,
    NonfiniteJSONNumber,
    decode_json,
    file_sha256,
    without_hash_metadata,
)
from utopia.utils.seeding import derive_seed

CELLS = ("LN", "LF", "HN", "HF")
PROTOCOL = "switching-propensity-v3-single-world-sequential"
POPULATION, SEED, YEARS = 1200, 42, 10
_TOPICS = frozenset(CANONICAL_TOPICS)
_ZERO_FUNDING = (
    "failed_batches", "audit_failures", "invalid_final_panels",
    "processing_failures", "unprocessed_panels", "imputed_rankings", "fallback_rankings",
)


class AnalysisError(ValueError):
    """Invalid/incomplete/out-of-domain inputs; no factorial result is returned."""

    def __init__(self, code, **context):
        self.code = code
        self.context = context
        super().__init__(code + (": " + json.dumps(context, sort_keys=True) if context else ""))


def require(condition, code, **context):
    if not condition:
        raise AnalysisError(code, **context)


def mapping(value, label):
    require(type(value) is dict, "expected_object", field=label)
    return value


def sequence(value, label):
    require(type(value) is list, "expected_array", field=label)
    return value


def integer(value, label, low=0, high=None):
    require(type(value) is int and value >= low and (high is None or value <= high),
             "invalid_integer", field=label)
    return value


def require_number(value, label, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value)
             and (low is None or value >= low) and (high is None or value <= high),
             "invalid_number", field=label)
    return value


def require_ids(value, label):
    values = sequence(value, label)
    require(all(type(item) is str and bool(item.strip()) for item in values),
             "invalid_identifier", field=label)
    require(len(set(values)) == len(values), "duplicate_identifier", field=label)
    return set(values)




def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def same_json(left, right):
    # Unlike Python equality, JSON equality here distinguishes True from 1.
    try:
        return json.dumps(without_hash_metadata(left), sort_keys=True, allow_nan=False) == json.dumps(
            without_hash_metadata(right), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AnalysisError("invalid_json_value") from error


def normalized_entropy(counts):
    """Entropy of all new-project choices, on the frozen 53-topic denominator."""
    mapping(counts, "topic_counts")
    require(set(counts) <= _TOPICS, "unknown_topic")
    for value in counts.values():
        integer(value, "topic_count")
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -math.fsum((n / total) * math.log(n / total) for n in counts.values() if n) / math.log(53)


def entropy_auc(values):
    """The specified ten-year trapezoidal sum, with neither omitted years nor /9."""
    require(type(values) in (list, tuple) and len(values) == YEARS,
             "auc_requires_ten_years")
    for value in values:
        require_number(value, "annual_entropy", 0, 1 + 1e-12)
    return math.fsum((left + right) / 2 for left, right in zip(values, values[1:]))


def factorial_contrasts(values):
    """Exact descriptive contrasts; menu effects are near MINUS far."""
    require(type(values) is dict and set(values) == set(CELLS),
             "factorial_requires_exactly_four_cells")
    for value in values.values():
        require_number(value, "cell_auc", 0, 9 + 1e-12)
    near = values["HN"] - values["LN"]
    far = values["HF"] - values["LF"]
    low = values["LN"] - values["LF"]
    high = values["HN"] - values["HF"]
    interaction = near - far
    return {
        "cell_values": {cell: values[cell] for cell in CELLS},
        "propensity_simple_effects": {"near_history": near, "far_history": far},
        "menu_simple_effects_near_minus_far": {"low_propensity": low, "high_propensity": high},
        "averaged_main_effects": {
            "high_minus_low_propensity": (near + far) / 2,
            "near_minus_far_history": (low + high) / 2,
        },
        "interaction": interaction,
        "positive_relative_interaction_in_seed42": interaction > 0,
        "near_history_propensity_benefit_in_seed42": near > 0,
        "HN_exceeds_both_one_factor_alternatives": values["HN"] > values["LN"]
        and values["HN"] > values["HF"],
    }


def distribution(values):
    ordered = sorted(values)
    return {"n": len(ordered), "mean": ratio(math.fsum(ordered), len(ordered)),
            "median": median(ordered) if ordered else None, "values": ordered}


def validate_manifest(manifest, cell, expected_source):
    """Shared producer/world metadata gate; this function performs no IO."""
    m = mapping(manifest, "manifest")
    n_years = 0 if cell is None else YEARS
    kind = "initialization_only" if cell is None else "condition"
    for key, expected in (("status", "complete"), ("kind", kind),
                          ("protocol", PROTOCOL), ("cell", cell)):
        require(identifier_matches(m.get(key), expected) if key == "protocol" else m.get(key) == expected,
                 "manifest_mismatch", cell=cell, field=key)
    for key, expected in (("seed", SEED), ("founder_count", POPULATION),
                          ("scientific_years_completed", n_years), ("choice_count", POPULATION)):
        require(type(m.get(key)) is int and m[key] == expected,
                 "manifest_mismatch", cell=cell, field=key)
    if cell is not None:
        require(type(m.get("years_completed")) is int and m["years_completed"] == YEARS,
                 "manifest_mismatch", cell=cell, field="years_completed")
        require(type(m.get("initialization_manifest_path")) is str
                 and Path(m["initialization_manifest_path"]).is_absolute(),
                 "missing_producer_path", cell=cell)
    inputs = mapping(m.get("inputs"), "manifest.inputs")
    require(identifier_matches(inputs.get('protocol'), PROTOCOL) and type(inputs.get("seed")) is int
             and inputs["seed"] == SEED, "input_identity_mismatch", cell=cell)
    baseline = mapping(m.get("funding_baseline"), "funding_baseline")
    for key, value in (("name", "stock_panel25_track"), ("panel_max_apps", 25),
                       ("funding_budget_mode", "track"),
                       ("program_funding_rates", {"NSF": 0.23, "DARPA": 0.10}),
                       ("winner_quota_rule", "max(1, int(panel_size * program.funding_rate))"),
                       ("funding_selection_protocol", SEQUENTIAL_PROTOCOL),
                       ("output_representation", "ordered_application_ids_v1")):
        require(type(baseline.get(key)) is type(value) and baseline[key] == value,
                 "funding_baseline_mismatch", field=key)
    funding = mapping(m.get("funding_validation"), "funding_validation")
    for key, expected in (("completion_allowed", True), ("installed", False),
                          ("output_representation", "ordered_application_ids_v1"),
                          ("restore_conflicts", [])):
        require(type(funding.get(key)) is type(expected) and funding[key] == expected,
                 "funding_not_complete", cell=cell, field=key)
    for key in _ZERO_FUNDING:
        require(type(funding.get(key)) is int and funding[key] == 0,
                 "funding_not_complete", cell=cell, field=key)
    request = mapping(m.get("request_audit"), "request_audit")
    require(request.get("status") == "complete", "request_audit_not_complete", cell=cell)
    for key, expected in (("n_clients", 1), ("guard_failures", 0), ("pending_requests", 0)):
        require(type(request.get(key)) is int and request[key] == expected,
                 "request_audit_not_complete", cell=cell, field=key)
    n_requests = integer(request.get("n_requests"), "n_requests", 1)
    n_responses = integer(request.get("n_responses"), "n_responses")
    n_errors = integer(request.get("n_transport_errors"), "n_transport_errors")
    require(n_requests == n_responses + n_errors, "request_audit_count_mismatch", cell=cell)
    require(m.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL,
             "funding_selection_protocol_mismatch", cell=cell)
    sequential = mapping(m.get("funding_sequential"), "funding_sequential")
    try:
        validate_summary(sequential)
    except SequentialFundingError as error:
        raise AnalysisError("sequential_not_complete", cell=cell) from error
    for key, expected in (("years_started", n_years), ("years_completed", n_years),
                          ("agency_registrations", 2 * n_years)):
        require(type(sequential.get(key)) is int and sequential[key] == expected,
                 "sequential_coverage_mismatch", cell=cell, field=key)
    require(type(funding.get("final_panels")) is int
             and funding["final_panels"] == sequential["processed_panels"],
             "sequential_compact_count_mismatch", cell=cell)
    require(inputs.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL,
             "sequential_input_binding_mismatch", cell=cell)
    if cell is None:
        require(sequential["processed_panels"] == sequential["sdk_calls"] == 0,
                 "initializer_performed_funding")
    for key in ("world_standard_error", "world_confidence_interval"):
        require(key in m and m[key] is None, "world_inference_not_identified", field=key)




def validate_coverage(raw, count, **context):
    coverage = mapping(raw, "history_coverage")
    require(coverage.get("status") == "complete", "history_ingestion_incomplete", **context)
    for key, expected in (("required_papers", count), ("embedded_papers", count),
                          ("missing_papers", 0)):
        require(type(coverage.get(key)) is int and coverage[key] == expected,
                 "history_ingestion_count_mismatch", field=key, **context)


def indexed_rows(raw, key, label):
    rows = {}
    for row in sequence(raw, label):
        mapping(row, label)
        identifier = row.get(key)
        require(type(identifier) is str and identifier.strip(), "invalid_identifier", field=label)
        require(identifier not in rows, "duplicate_identifier", field=label, identifier=identifier)
        rows[identifier] = row
    return rows


def paper_registry(records, founders, cell):
    wrapper = mapping(records, "submission_registry")
    require(wrapper.get("cell") == cell and type(wrapper.get("year")) is int
             and wrapper["year"] == YEARS, "registry_identity_mismatch")
    registry = indexed_rows(wrapper.get("papers"), "paper_id", "registry.papers")
    validate_coverage(wrapper.get("history_coverage"), len(registry))
    for pid, row in registry.items():
        require(type(row.get("author_id")) is str and row["author_id"] in founders,
                 "unknown_manuscript_author", paper_id=pid)
        integer(row.get("first_submission_year"), "first_submission_year", 1, YEARS)
        require(type(row.get("first_attempt_accepted")) is bool,
                 "missing_first_attempt_outcome", paper_id=pid)
    return registry


def paper_observations(by_year, registry):
    observations, accepted_years = {}, {}
    for year in range(1, YEARS + 1):
        rows = indexed_rows(by_year[year].get("paper_observations"), "paper_id", "paper_observations")
        expected = {pid for pid, paper in registry.items() if paper["first_submission_year"] <= year}
        require(set(rows) == expected, "paper_observation_coverage_mismatch", year=year)
        for pid, row in rows.items():
            paper = registry[pid]
            for key, value in (("author_id", paper["author_id"]),
                               ("first_submission_year", paper["first_submission_year"]), ("year", year)):
                require(type(row.get(key)) is type(value) and row[key] == value,
                         "paper_observation_identity_mismatch", paper_id=pid, field=key)
            status = row.get("status")
            require(status in ("accept", "reject", "pending"), "unknown_paper_status", paper_id=pid)
            count = integer(row.get("citation_count"), "citation_count")
            realized = integer(row.get("realized_publication_citations"), "realized_publication_citations")
            require(realized == (count if status == "accept" else 0),
                     "realized_citation_mismatch", paper_id=pid, year=year)
            if pid in accepted_years:
                require(status == "accept", "publication_status_regressed", paper_id=pid, year=year)
            elif status == "accept":
                accepted_years[pid] = year
            if year == paper["first_submission_year"]:
                require(status in ("accept", "reject")
                         and paper["first_attempt_accepted"] == (status == "accept"),
                         "first_attempt_acceptance_conflict", paper_id=pid)
            if paper["first_submission_year"] <= YEARS - 3 and year == paper["first_submission_year"] + 3:
                snapshot = mapping(paper.get("three_year_observation"), "three_year_observation")
                require(same_json(snapshot, row), "deadline_observation_mismatch", paper_id=pid, year=year)
        observations[year] = rows
    return observations, accepted_years


def expected_gate(agent_id, year):
    seed = derive_seed(42, SWITCH_GATE_SEED_NAMESPACE, agent_id, year)
    return seed, random.Random(seed).random()


def validate_event(event, cell, founders, previous, registry):
    row = mapping(event, "direction_event")
    agent, year = row.get("agent_id"), integer(row.get("year"), "event.year", 1, YEARS)
    require(type(agent) is str and agent in founders, "unknown_event_author")
    require(row.get("cell") == cell and type(row.get("seed")) is int and row["seed"] == SEED,
             "event_identity_mismatch", cell=cell, year=year)
    require(row.get("status") == "complete", "event_not_complete", cell=cell, year=year)
    initial = row.get("is_initial_choice")
    require(type(initial) is bool and initial == (year == 1), "initial_choice_flag_mismatch")
    require(type(row.get("eligible_repeat")) is bool and row["eligible_repeat"] == (not initial),
             "repeat_eligibility_flag_mismatch")
    topic = row.get("chosen_topic")
    require(type(topic) is str and topic in _TOPICS, "unknown_chosen_topic")
    fallback = row.get("fallback_kind", "MISSING")
    require(fallback in (None, "parse", "validation"), "invalid_fallback_kind")
    require(type(row.get("fallback")) is bool and row["fallback"] == (fallback is not None),
             "fallback_flag_mismatch")
    for key in ("detailed_focus", "reason"):
        require(type(row.get(key)) is str, "missing_choice_content", field=key)
    for key in ("project_start_year", "project_end_year"):
        require(type(row.get(key)) is int and row[key] == year, "project_timeline_mismatch", field=key)
    if initial:
        require(agent not in previous, "duplicate_initial_choice", agent_id=agent)
        require(row.get("event_type") == "initial_choice" and row.get("realized_switch") is False,
                 "initial_choice_counted_as_switch")
        require("conditional_switch_distance" in row and row["conditional_switch_distance"] is None,
                 "initial_distance_must_be_undefined")
        candidates = require_ids(row.get("candidate_topics"), "initial.candidate_topics")
        require(candidates <= _TOPICS and topic in candidates, "initial_candidate_mismatch")
        if fallback is not None:
            require(topic == row["candidate_topics"][0], "fallback_not_first_candidate")
        switched = False
    else:
        require(row.get("previous_topic") == previous.get(agent),
                 "previous_topic_mismatch", agent_id=agent, year=year)
        p = 0.25 if cell.startswith("L") else 0.75
        require(type(row.get("p")) in (int, float) and row["p"] == p, "propensity_mismatch")
        gate_seed, u = expected_gate(agent, year)
        require(type(row.get("u")) in (int, float) and row["u"] == u
                 and type(row.get("gate_seed")) is int and row["gate_seed"] == gate_seed,
                 "gate_draw_mismatch", agent_id=agent, year=year)
        switched = topic != row["previous_topic"]
        require(type(row.get("requested_switch")) is bool
                 and type(row.get("realized_switch")) is bool
                 and row["requested_switch"] == row["realized_switch"] == switched == (u < p),
                 "switch_gate_mismatch", agent_id=agent, year=year)
        distances = mapping(row.get("distance_map"), "distance_map")
        eligible = _TOPICS - {row["previous_topic"]}
        require(set(distances) == eligible, "distance_coverage_mismatch")
        for distance in distances.values():
            require_number(distance, "distance", -DISTANCE_BOUNDARY_TOLERANCE,
                    2 + DISTANCE_BOUNDARY_TOLERANCE)
        ranked = sorted(eligible, key=lambda name: (distances[name], name))
        near, far = sorted(ranked[:17]), sorted(ranked[-17:])
        require(row.get("near_topics") == near and row.get("far_topics") == far,
                 "menu_mismatch", agent_id=agent, year=year)
        metadata = {
            "event_type": "eligible_repeat_choice", "policy_version": "switching_propensity_distance_v1",
            "gate_namespace": SWITCH_GATE_SEED_NAMESPACE, "gate_rng": "random.Random",
            "requested_action": "switch" if switched else "stay",
            "distance_mode": "near-history" if cell.endswith("N") else "far-history",
            "distance_definition": "direction_to_historical_paper_centroid_cosine",
            "native_centroid_max_years": 3, "n_canonical_topics": 53, "n_eligible_topics": 52,
            "ranked_eligible_topics": ranked, "n_near": 17, "n_far": 17, "n_middle_unused": 18,
            "distance_boundary_tolerance": DISTANCE_BOUNDARY_TOLERANCE,
            "near_min": distances[ranked[0]], "near_max": distances[ranked[16]],
            "far_min": distances[ranked[-17]], "far_max": distances[ranked[-1]],
        }
        for key, value in metadata.items():
            require(type(row.get(key)) is type(value) and row[key] == value,
                     "policy_audit_mismatch", field=key, agent_id=agent, year=year)
        gap = distances[ranked[-17]] - distances[ranked[16]]
        require(gap > 0, "out_of_domain" if gap == 0 else "negative_menu_gap",
                 agent_id=agent, year=year)
        require(require_number(row.get("separation_gap"), "separation_gap") == gap, "separation_gap_mismatch")
        expected_candidates = (near if cell.endswith("N") else far) if switched else [previous[agent]]
        require(row.get("candidate_topics") == expected_candidates and topic in expected_candidates,
                 "candidate_membership_mismatch")
        if fallback is not None:
            require(topic == expected_candidates[0], "fallback_not_first_candidate")
        if switched:
            require(type(row.get("conditional_switch_distance")) in (int, float)
                     and row["conditional_switch_distance"] == distances[topic],
                     "chosen_distance_mismatch")
        else:
            require("conditional_switch_distance" in row
                     and row["conditional_switch_distance"] is None, "stay_distance_must_be_undefined")
        expected_history = sorted(pid for pid, paper in registry.items()
                                  if paper["author_id"] == agent
                                  and year - 4 <= paper["first_submission_year"] <= year - 1)
        require(row.get("history_ids") == expected_history
                 and type(row.get("expected_history_count")) is int
                 and row["expected_history_count"] == len(expected_history),
                 "reference_history_coverage_mismatch", agent_id=agent, year=year)
        source = "paper_history" if expected_history else "initial_expertise"
        require(row.get("reference_source") == source, "reference_fallback_mismatch")
        require(type(row.get("history_window_start")) is int and row["history_window_start"] == year - 4
                 and type(row.get("history_window_end")) is int and row["history_window_end"] == year - 1,
                 "history_window_mismatch")
        require_number(row.get("reference_norm"), "reference_norm", low=0)
        require(row["reference_norm"] > 0, "zero_reference_norm")
    for key, count in (("new_project_choice_count", 1), ("initial_choice_count", int(initial)),
                       ("eligible_repeat_count", int(not initial)),
                       ("realized_switch_count", int(switched))):
        require(type(row.get(key)) is int and row[key] == count, "event_counter_mismatch", field=key)
    previous[agent] = topic
    return initial, switched, fallback


def year_records(raw, label, cell, *, mechanisms=False):
    rows = sequence(raw, label)
    require(len(rows) == YEARS, "incomplete_years", cell=cell, field=label)
    by_year = {}
    for row in rows:
        mapping(row, label)
        year = integer(row.get("year"), label + ".year", 1, YEARS)
        require(year not in by_year, "duplicate_year", cell=cell, field=label, year=year)
        require(row.get("cell") == ("P1F1" if mechanisms else cell), "year_identity_mismatch")
        if not mechanisms:
            require(type(row.get("founder_count")) is int and row["founder_count"] == POPULATION,
                     "year_identity_mismatch")
        by_year[year] = row
    return by_year


def resource_rows(raw, founders, label):
    rows = indexed_rows(raw, "agent_id", label)
    require(set(rows) == founders, "resource_roster_mismatch", field=label)
    for row in rows.values():
        require(type(row.get("active")) is bool, "invalid_activity_flag", field=label)
        require_number(row.get("resources"), label + ".resources")
    return rows


def funding_and_agents(raw, mechanisms, founders, previous_funding, observations, year):
    require(same_json(raw.get("agent_rows"), mechanisms.get("agent_rows")),
             "agent_sidecar_mismatch", year=year)
    rows = indexed_rows(raw.get("agent_rows"), "agent_id", "agent_rows")
    require(set(rows) == founders, "agent_roster_mismatch", year=year)
    submitted, accepted, citations = Counter(), Counter(), Counter()
    for observed in observations.values():
        aid = observed["author_id"]
        submitted[aid] += 1
        accepted[aid] += int(observed["status"] == "accept")
        citations[aid] += observed["citation_count"]
    awards = sequence(mechanisms.get("awards"), "awards")
    earned = Counter()
    for award in awards:
        mapping(award, "award")
        aid = award.get("agent_id")
        require(type(aid) is str and aid in founders, "unknown_award_recipient")
        require(type(award.get("year")) is int and award["year"] == year, "award_year_mismatch")
        require(type(award.get("program_id")) is str and award["program_id"].strip(),
                 "missing_award_program")
        amount = require_number(award.get("earned_amount"), "earned_amount")
        credit = require_number(award.get("spendable_credit"), "spendable_credit")
        require(amount == credit == 20, "fixed_award_credit_mismatch", year=year)
        # Repeated applicants/awards are retained, never deduplicated.
        earned[aid] += amount
    for aid, row in rows.items():
        require(type(row.get("active")) is bool, "invalid_activity_flag")
        require_number(row.get("spendable_resources"), "spendable_resources")
        cumulative = require_number(row.get("cumulative_earned_funding"), "cumulative_earned_funding", 0)
        require(cumulative == previous_funding[aid] + earned[aid],
                 "funding_ledger_mismatch", agent_id=aid, year=year)
        for key, count in (("submitted_papers", submitted[aid]), ("accepted_papers", accepted[aid]),
                           ("citations_all_papers", citations[aid])):
            require(integer(row.get(key), key) == count, "agent_paper_count_mismatch",
                     agent_id=aid, year=year, field=key)
    return rows, math.fsum(earned.values()), len(awards)


def analyze_cell(manifest, records, cell):
    data = mapping(records, "cell_records")
    by_year = year_records(data.get("years"), "years", cell)
    mechanism_years = year_records(data.get("mechanisms_years"), "mechanisms_years", cell, mechanisms=True)
    founders = require_ids(manifest.get("founder_ids"), "manifest.founder_ids")
    require(len(founders) == POPULATION, "founder_coverage_mismatch")
    registry = paper_registry(data.get("submission_registry"), founders, cell)
    observations, accepted_years = paper_observations(by_year, registry)
    validate_coverage(manifest.get("history_coverage"), len(registry))
    norms = mapping(data.get("direction_embedding_norms"), "direction_embedding_norms")
    require(set(norms) == _TOPICS, "direction_norm_coverage_mismatch")
    for norm in norms.values():
        require(require_number(norm, "direction_norm", 0) > 0, "zero_direction_norm")
    events = sequence(data.get("direction_events"), "direction_events")
    event_years = {year: {} for year in range(1, YEARS + 1)}
    for row in events:
        mapping(row, "direction_event")
        year = integer(row.get("year"), "event.year", 1, YEARS)
        agent = row.get("agent_id")
        require(type(agent) is str, "invalid_event_author")
        require(agent not in event_years[year], "duplicate_direction_event", agent_id=agent, year=year)
        event_years[year][agent] = row
    require(set(event_years[1]) == founders, "initial_choice_coverage_mismatch")
    previous, annual, switch_distances, reference_counts = {}, [], [], Counter()
    fallback_counts = {mode: {"events": 0, "parse": 0, "validation": 0}
                       for mode in ("initial", "stay", "switch")}
    reference_distances = {"paper_history": [], "initial_expertise": []}
    initial_choices = {}
    previous_end = founders
    previous_funding = dict.fromkeys(founders, 0)  # Protocol: no pre-initialization awards.
    state_checks = []
    for year in range(1, YEARS + 1):
        raw = by_year[year]
        start = require_ids(raw.get("year_start_active_ids"), "year_start_active_ids")
        available = require_ids(raw.get("phase1_available_ids"), "phase1_available_ids")
        eligible = require_ids(raw.get("phase1_eligible_ids"), "phase1_eligible_ids")
        not_due = require_ids(raw.get("phase1_not_due_ids"), "phase1_not_due_ids")
        inactive = require_ids(raw.get("phase1_inactive_ids"), "phase1_inactive_ids")
        phase0 = require_ids(raw.get("phase0_lost_active_ids"), "phase0_lost_active_ids")
        start_rows = resource_rows(raw.get("year_start_resources"), founders, "year_start_resources")
        after0 = resource_rows(raw.get("after_phase0_resources"), founders, "after_phase0_resources")
        after0_active = {aid for aid, row in after0.items() if row["active"]}
        agent_rows, earned, n_awards = funding_and_agents(
            raw, mechanism_years[year], founders, previous_funding, observations[year], year)
        end = {aid for aid, row in agent_rows.items() if row["active"]}
        require(start == previous_end and start <= founders, "activity_continuity_mismatch", year=year)
        require(start == {aid for aid, row in start_rows.items() if row["active"]},
                 "start_resource_activity_mismatch", year=year)
        require(available <= after0_active <= start and inactive == founders - after0_active
                 and phase0 == start - available and eligible | not_due == available
                 and not eligible & not_due and end <= after0_active,
                 "risk_set_mismatch", year=year)
        require(not not_due, "one_year_project_not_due", year=year)
        require(set(event_years[year]) == eligible, "eligible_event_coverage_mismatch", year=year)
        embedded_events = indexed_rows(raw.get("direction_events"), "agent_id", "year.direction_events")
        require(same_json(embedded_events, event_years[year]), "event_sidecar_mismatch", year=year)
        required = sum(p["first_submission_year"] <= year for p in registry.values())
        validate_coverage(raw.get("history_coverage"), required, year=year)
        charges = indexed_rows(raw.get("annual_charges"), "agent_id", "annual_charges")
        require(set(charges) == available, "annual_charge_coverage_mismatch", year=year)
        for charge in charges.values():
            require(require_number(charge.get("amount"), "annual_charge.amount") == -10,
                     "annual_charge_mismatch", year=year)
            for key in ("resources_before", "resources_after"):
                require_number(charge.get(key), "annual_charge." + key)
            require(charge.get("active_before") is True and type(charge.get("active_after")) is bool,
                     "annual_charge_activity_mismatch", year=year)
        counts, initials, switches, repeats = Counter(), 0, 0, 0
        year_fallbacks = {mode: {"events": 0, "parse": 0, "validation": 0}
                          for mode in fallback_counts}
        for agent in sorted(event_years[year]):
            event = event_years[year][agent]
            initial, switched, fallback = validate_event(event, cell, founders, previous, registry)
            counts[event["chosen_topic"]] += 1
            initials += int(initial)
            repeats += int(not initial)
            switches += int(switched)
            mode = "initial" if initial else "switch" if switched else "stay"
            for table in (fallback_counts, year_fallbacks):
                table[mode]["events"] += 1
                if fallback is not None:
                    table[mode][fallback] += 1
            if initial:
                initial_choices[agent] = {key: deepcopy(event[key]) for key in (
                    "chosen_topic", "candidate_topics", "fallback_kind", "input_hash", "detailed_focus", "reason")}
            else:
                reference_counts[event["reference_source"]] += 1
                state_checks.append({key: deepcopy(event[key]) for key in (
                    "agent_id", "year", "p", "u", "gate_seed", "requested_switch", "realized_switch",
                    "chosen_topic", "near_topics", "far_topics", "n_near", "n_far",
                    "near_min", "near_max", "far_min", "far_max", "separation_gap",
                    "reference_source", "reference_norm", "history_ids")})
                if switched:
                    value = event["conditional_switch_distance"]
                    switch_distances.append(value)
                    reference_distances[event["reference_source"]].append(value)
        for paper in registry.values():
            if paper["first_submission_year"] == year:
                require(paper["author_id"] in eligible, "submission_outside_choice_risk_set", year=year)
        n = sum(counts.values())
        entropy = normalized_entropy(dict(counts))
        annual.append({
            "year": year, "year_start_active": len(start), "phase1_eligible": len(eligible),
            "phase0_lost_active": len(phase0), "phase1_exclusions": len(start - eligible),
            "risk_sets": {key: sorted(values) for key, values in (
                ("year_start_active_ids", start), ("phase1_available_ids", available),
                ("phase1_eligible_ids", eligible), ("phase1_not_due_ids", not_due),
                ("phase1_inactive_ids", inactive), ("phase0_lost_active_ids", phase0),
                ("year_end_active_ids", end))},
            "year_end_active": len(end), "new_project_choices": n,
            "initial_choices": initials, "eligible_repeat_choices": repeats,
            "switches": switches, "stays": repeats - switches,
            "realized_switch_propensity": ratio(switches, repeats),
            "eligible_repeat_per_year_start_active": ratio(repeats, len(start)),
            "switches_per_year_start_active": ratio(switches, len(start)),
            "topic_counts": {topic: counts[topic] for topic in CANONICAL_TOPICS},
            "entropy": entropy, "no_activity_convention": n == 0,
            "participation_weighted_entropy": n / POPULATION * entropy,
            "fallbacks": year_fallbacks, "earned_funding": earned, "award_count": n_awards,
            "annual_charge_count": len(charges), "annual_charges": deepcopy(raw["annual_charges"]),
            "history_coverage": deepcopy(raw["history_coverage"]),
        })
        previous_end = end
        previous_funding = {aid: row["cumulative_earned_funding"] for aid, row in agent_rows.items()}
    citation_rows = []
    for pid, paper in sorted(registry.items()):
        first = paper["first_submission_year"]
        if first > 7:
            continue
        deadline = first + 3
        observed = observations[deadline][pid]
        published = observed["status"] == "accept"
        count = observed["realized_publication_citations"]
        citation_rows.append({"paper_id": pid, "first_submission_year": first,
                              "deadline_year": deadline, "published_by_deadline": published,
                              "accepted_year": accepted_years.get(pid), "citations": count})
    total_citations = sum(row["citations"] for row in citation_rows)
    total_repeats = sum(row["eligible_repeat_choices"] for row in annual)
    total_switches = sum(row["switches"] for row in annual)
    exposure = sum(row["year_start_active"] for row in annual)
    for mode, counts in fallback_counts.items():
        counts["fallback_rate"] = ratio(counts["parse"] + counts["validation"], counts["events"])
        for row in annual:
            year_counts = row["fallbacks"][mode]
            year_counts["fallback_rate"] = ratio(
                year_counts["parse"] + year_counts["validation"], year_counts["events"])
    accepted = len(accepted_years)
    first_accepted = sum(p["first_attempt_accepted"] for p in registry.values())
    output = {
        "annual": annual,
        "entropy_auc": entropy_auc([row["entropy"] for row in annual]),
        "secondary": {
            "year10_active_founders": len(previous_end),
            "year10_active_fraction": len(previous_end) / POPULATION,
            "accepted_unique_manuscripts": accepted,
            "accepted_unique_manuscripts_per_founder": accepted / POPULATION,
            "cumulative_earned_funding": math.fsum(row["earned_funding"] for row in annual),
            "cumulative_earned_funding_per_founder":
                math.fsum(row["earned_funding"] for row in annual) / POPULATION,
            "participation_weighted_entropy_auc":
                entropy_auc([row["participation_weighted_entropy"] for row in annual]),
            "first_submitted_unique_manuscripts": len(registry),
            "first_attempt_accepted": first_accepted,
            "first_attempt_acceptance_rate": ratio(first_accepted, len(registry)),
            "citation_yield": {
                "eligible_manuscripts": len(citation_rows),
                "published_by_deadline": sum(row["published_by_deadline"] for row in citation_rows),
                "unpublished_by_deadline": sum(not row["published_by_deadline"] for row in citation_rows),
                "total": total_citations, "per_founder": total_citations / POPULATION,
                "mean_per_eligible_manuscript": ratio(total_citations, len(citation_rows)),
                "manuscripts": citation_rows,
            },
        },
        "manipulation": {
            "initial_choices": sum(row["initial_choices"] for row in annual),
            "eligible_repeat_choices": total_repeats, "switches": total_switches,
            "stays": total_repeats - total_switches,
            "realized_switch_propensity": ratio(total_switches, total_repeats),
            "year_start_active_person_years": exposure,
            "eligible_repeat_per_year_start_active": ratio(total_repeats, exposure),
            "switches_per_year_start_active": ratio(total_switches, exposure),
            "conditional_switch_distance": distribution(switch_distances),
            "reference_counts": {kind: reference_counts[kind] for kind in reference_distances},
            "conditional_switch_distance_by_reference":
                {kind: distribution(values) for kind, values in reference_distances.items()},
            "fallbacks": fallback_counts,
            "state_checks": state_checks,
            "direction_embedding_norms": deepcopy(norms),
            "funding_baseline": deepcopy(manifest["funding_baseline"]),
            "funding_validation": deepcopy(manifest["funding_validation"]),
            "funding_sequential": deepcopy(manifest["funding_sequential"]),
            "all_repeat_gates_menus_and_references_valid": True,
        },
    }
    return output, founders, initial_choices


def analyze_records(manifests, records_by_cell, *, expected_source):
    """Validate all four complete worlds, then return descriptive endpoints."""
    require(type(manifests) is dict and set(manifests) == set(CELLS)
             and type(records_by_cell) is dict and set(records_by_cell) == set(CELLS),
             "requires_exactly_four_cells")
    common = None
    for cell in CELLS:
        validate_manifest(manifests[cell], cell, expected_source)
        identity = (manifests[cell]["initialization_manifest_path"],)
        founder_ids = require_ids(manifests[cell].get("founder_ids"), "manifest.founder_ids")
        require(len(founder_ids) == POPULATION, "founder_coverage_mismatch")
        identity += (tuple(manifests[cell]["founder_ids"]),)
        require(common is None or identity == common, "initial_state_mismatch", cell=cell)
        require(same_json(scientific_inputs(manifests[cell]["inputs"]), scientific_inputs(manifests["LN"]["inputs"])),
                 "common_input_mismatch", cell=cell)
        common = identity
    outputs, common_founders, common_choices = {}, None, None
    for cell in CELLS:
        output, founders, initial_choices = analyze_cell(manifests[cell], records_by_cell[cell], cell)
        require(common_founders is None or founders == common_founders, "founder_identity_mismatch")
        require(common_choices is None or initial_choices == common_choices, "initial_choice_mismatch")
        common_founders, common_choices = founders, initial_choices
        outputs[cell] = output
    return {
        "schema": "scienceutopia.propensity.analysis.v3", "status": "complete",
        "protocol": PROTOCOL, "seed": SEED, "founder_count_per_cell": POPULATION,
        "years": YEARS, "world_replicates": 1, "inference": "descriptive_single_seed_only",
        "world_standard_error": None, "world_confidence_interval": None,
        "funding_selection_protocol": SEQUENTIAL_PROTOCOL,
        "evidence_validation": "supplied_records_and_terminal_summaries_only",
        "source_sha256": deepcopy(expected_source),
        "initial_choices_sha256": manifests["LN"].get("initial_choices_sha256"),
        "initial_state_hash": manifests["LN"].get("initial_state_hash"),
        "pre_choice_state_hash": manifests["LN"].get("pre_choice_state_hash"),
        "primary": factorial_contrasts({cell: outputs[cell]["entropy_auc"] for cell in CELLS}),
        "cells": outputs,
    }


def read_json(path):
    try:
        return decode_record(Path(path).read_text(), path)
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError("cannot_read_json", path=str(path)) from error


def file_digest(path):
    try:
        return file_sha256(path)
    except OSError as error:
        raise AnalysisError("cannot_hash_artifact", path=str(path)) from error


def read_jsonl(path):
    try:
        with Path(path).open() as stream:
            for number, line in enumerate(stream, 1):
                require(bool(line.strip()), "empty_event_line", path=str(path), line=number)
                yield decode_record(line, path)
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError("cannot_read_jsonl", path=str(path)) from error


def validate_funding_files(directory, manifest, expected_source):
    """Use the driver's shared native-ledger and raw SDK/request reconciliation."""
    from utopia.runtime.switching_evidence import (
        ProtocolFailure,
        validate_funding_evidence,
    )
    for filename, key in (("funding_validation.summary.json", "funding_validation"),
                          ("llm_request_audit.summary.json", "request_audit"),
                          ("funding_sequential.summary.json", "funding_sequential")):
        require(same_json(read_json(Path(directory) / filename), manifest.get(key)),
                 "audit_sidecar_mismatch", path=str(directory), field=key)
    try:
        return validate_funding_evidence(directory, manifest, expected_source=expected_source)
    except (SequentialFundingError, ProtocolFailure, OSError, ValueError, KeyError, TypeError) as error:
        raise AnalysisError("funding_evidence_invalid", path=str(directory)) from error


def validate_producer(manifests, expected_source):
    """Bind the new treatment-blind cache to its closed producer and all worlds."""
    from utopia.runtime.switching_evidence import (
        ProtocolFailure,
        identifier,
        load_choices,
    )
    paths = [Path(m.get("initial_choices_path", "")) for m in manifests.values()]
    require(all(p.is_absolute() for p in paths) and len({p.resolve() for p in paths}) == 1,
             "common_initial_cache_path_mismatch")
    cache_path = paths[0]
    directory = cache_path.parent
    require(cache_path.name == "initial_choices.json" and identifier_matches(directory.name, identifier()),
             "stale_initialization_namespace")
    manifest_path = directory / "initialization_manifest.json"
    expected_hash = file_digest(manifest_path)
    require(all((Path(m['initialization_manifest_path']).resolve() == manifest_path.resolve() for m in manifests.values())), "producer_manifest_path_mismatch")
    producer = read_json(manifest_path)
    validate_manifest(producer, None, expected_source)
    require(type(producer.get("initial_choices_path")) is str
             and Path(producer["initial_choices_path"]).is_absolute()
             and Path(producer["initial_choices_path"]).resolve() == cache_path.resolve(),
             "producer_cache_path_mismatch")
    common = manifests["LN"]
    require(same_json(scientific_inputs(producer["inputs"]), scientific_inputs(common["inputs"])), "producer_input_mismatch")
    for key in ("founder_ids",):
        require(same_json(producer.get(key), common.get(key)), "producer_identity_mismatch", field=key)
    try:
        cache, digest = load_choices(cache_path, common["inputs"])
    except (ProtocolFailure, OSError, ValueError, KeyError, TypeError) as error:
        raise AnalysisError("initial_cache_invalid", path=str(cache_path)) from error
    for key in ("founder_ids",):
        require(same_json(cache.get(key), common.get(key)), "cache_identity_mismatch", field=key)
    funding = validate_funding_files(directory, producer, expected_source)
    return cache, {"manifest_path": str(manifest_path), "manifest_sha256": expected_hash,
                   "initial_choices_path": str(cache_path), "initial_choices_sha256": digest,
                   "funding_evidence": funding}


def reconcile_recorded_awards(manifest, mechanisms_years):
    """Join already validated native winners to the existing resource ledger."""
    expected = Counter()
    directory = Path(manifest["args"]["output_dir"])
    for year in range(1, YEARS + 1):
        path = directory / f"funding_applications_year_{year}.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                if row["funded"]:
                    expected[year, row["program_id"], row["applicant_id"]] += 1
    observed = Counter((row["year"], award["program_id"], award["agent_id"])
                       for row in mechanisms_years for award in row["awards"])
    require(observed == expected, "native_winners_awards_mismatch")


def analyze_files(cell_directories, *, expected_source):
    """Admit only four fresh v3 worlds and their common completed initializer."""
    require(type(cell_directories) is dict and set(cell_directories) == set(CELLS),
             "requires_exactly_four_cells")
    manifests, records, funding_evidence = {}, {}, {}
    for cell in CELLS:
        directory = Path(cell_directories[cell])
        manifest = read_json(directory / "propensity_manifest.json")
        validate_manifest(manifest, cell, expected_source)
        manifests[cell] = manifest
    # Validate all source pins before invoking any run-evidence helper.
    from utopia.runtime.switching_evidence import identifier
    for cell in CELLS:
        directory = Path(cell_directories[cell])
        require(directory.is_absolute() and directory.name == identifier(cell),
                 "stale_condition_namespace", cell=cell)
        manifest = manifests[cell]
        funding_evidence[cell] = validate_funding_files(directory, manifest, expected_source)
        year_directory = directory / "years"
        expected_files = {f"year_{year:02d}.json" for year in range(1, YEARS + 1)}
        require({path.name for path in year_directory.glob("year_*.json")} == expected_files,
                 "incomplete_or_extra_year_files", cell=cell)
        expected_mechanisms = {f"mechanisms_year_{year}.json" for year in range(1, YEARS + 1)}
        require({path.name for path in directory.glob("mechanisms_year_*.json")} == expected_mechanisms,
                 "incomplete_or_extra_mechanisms_files", cell=cell)
        records[cell] = {
            "years": [read_json(year_directory / f"year_{year:02d}.json")
                      for year in range(1, YEARS + 1)],
            "mechanisms_years": [read_json(directory / f"mechanisms_year_{year}.json")
                                for year in range(1, YEARS + 1)],
            "direction_events": list(read_jsonl(directory / "direction_events.jsonl")),
            "submission_registry": read_json(directory / "submission_registry.json"),
            "direction_embedding_norms": read_json(directory / "direction_embedding_norms.json"),
        }
        reconcile_recorded_awards(manifest, records[cell]["mechanisms_years"])
    cache, initialization = validate_producer(manifests, expected_source)
    cached = indexed_rows(cache["records"], "agent_id", "initial_cache.records")
    for cell in CELLS:
        first = indexed_rows([r for r in records[cell]["direction_events"] if r["year"] == 1],
                              "agent_id", "initial_events")
        require(set(first) == set(cached), "initial_event_cache_coverage_mismatch", cell=cell)
        for aid, row in first.items():
            value = cached[aid]
            for key in ("candidate_topics", "fallback_kind", "input_hash"):
                require(same_json(row.get(key), value.get(key)),
                         "initial_event_cache_mismatch", cell=cell, agent_id=aid, field=key)
            require(same_json({key: row[key] for key in ("detailed_focus", "reason")}
                                | {"topic": row["chosen_topic"]}, value["response"]),
                     "initial_event_cache_mismatch", cell=cell, agent_id=aid, field="response")
    result = analyze_records(manifests, records, expected_source=expected_source)
    result.update(evidence_validation="closed_raw_audits_native_ledgers_and_common_producer",
                  initialization=initialization,
                  funding_evidence=funding_evidence,
                  admitted_manifest_sha256={
                      cell: file_digest(Path(cell_directories[cell]) / "propensity_manifest.json")
                      for cell in CELLS})
    return result


def decode_record(raw, path):
    try:
        return decode_json(raw, strict=True)
    except DuplicateJSONKey as error:
        raise AnalysisError("duplicate_json_key", path=str(path), key=error.key) from error
    except NonfiniteJSONNumber as error:
        raise AnalysisError("nonfinite_json_number", path=str(path), value=error.value) from error
