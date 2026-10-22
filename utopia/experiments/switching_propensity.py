"""Switching propensity experiment and reusable scientific operations."""

from __future__ import annotations

from utopia.utils.data_utils import write_json as write_json_file

from utopia.runtime.switching_evidence import CELLS, EMBEDDING_MODEL, EMBEDDING_REVISION, EMBEDDING_WEIGHTS_SHA256, FOUNDERS, FUNDING_AMENDMENT, FUNDING_GATES, FUNDING_SELECTION_PROTOCOL, MODEL, OUTPUT_REPRESENTATION, PROTOCOL, REQUEST_GATES, ROOT, RUNTIME_AMENDMENT, SEED, YEARS, canonical_state, identifier, load_choices, require, sequential_gates, source_identity, validate_choices, validate_funding_evidence, validate_initialization

from utopia.utils.data_utils import file_sha256

from utopia.runtime.commands import module_command

import utopia.funding.feedback as feedback
import utopia.runtime.provenance as provenance


import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import shlex
import sys
import tempfile
import traceback

import utopia.agents.switching_policy as policy
from utopia.runtime.server_profiles import TP1_PROFILE_NAME


PROMPT_CHANGES = {
    "projects in all directions are expected to complete in either 1, 2, or 3 years.":
        "projects in all directions last exactly one year.",
    "- Longer projects (3-4 years) require stable funding":
        "- Every displayed project lasts one year and incurs the same annual project charge",
    "First, select one research direction that best aligns with your expertise, current situation, and long-term goals.":
        "Select exactly ONE direction from the displayed candidates, using your expertise, current situation, and long-term goals as context.",
    "Propose the directions based on your expertise:":
        "Use your expertise as context. Only the displayed candidate directions are valid choices:",
}


class InitialChoicesComplete(BaseException):
    """Private control signal, caught inside the successful request-audit scope."""


def append_record(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


@contextmanager
def module_attribute(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield original
    finally:
        setattr(module, name, original)


@contextmanager
def one_year_context(simulator, directions_module):
    """Copies and restored bindings affect this process, not frozen source."""
    originals = simulator.AVAILABLE_DIRECTIONS
    require(len(originals) == 53 and len({d.topic for d in originals}) == 53,
            "direction_universe_is_not_53_unique_topics")
    require({d.topic for d in originals} == set(policy.CANONICAL_TOPICS),
            "direction_universe_differs_from_policy")
    copies = deepcopy(originals)
    for direction in copies:
        direction.years = 1
    lookup = {d.topic: d for d in copies}
    original_builder = directions_module.build_direction_prompt

    def common_prompt(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        prompt = result[0]
        for old, new in PROMPT_CHANGES.items():
            require(prompt.count(old) == 1, "native_balanced_prompt_drift", anchor=old)
            prompt = prompt.replace(old, new)
        return (prompt, *result[1:])

    with ExitStack() as stack:
        stack.enter_context(module_attribute(simulator, "AVAILABLE_DIRECTIONS", copies))
        stack.enter_context(module_attribute(directions_module, "AVAILABLE_DIRECTIONS", copies))
        stack.enter_context(module_attribute(directions_module, "DIRECTIONS_DICT", lookup))
        if hasattr(simulator, "DIRECTIONS_DICT"):
            stack.enter_context(module_attribute(simulator, "DIRECTIONS_DICT", lookup))
        stack.enter_context(module_attribute(directions_module, "build_direction_prompt", common_prompt))
        yield lookup


def resolve_embedding_snapshot(revision=EMBEDDING_REVISION):
    """Resolve the actual offline MiniLM commit BEFORE model construction."""
    hub = Path(os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
               or str(Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    repository = hub / "models--sentence-transformers--all-MiniLM-L6-v2"
    require(revision == EMBEDDING_REVISION,
            "invalid_minilm_revision")
    snapshot = repository / "snapshots" / revision
    require(snapshot.is_dir() and (snapshot / "modules.json").is_file(),
            "minilm_snapshot_incomplete", snapshot=str(snapshot))
    require((snapshot / "model.safetensors").is_file()
            and file_sha256(snapshot / "model.safetensors") == EMBEDDING_WEIGHTS_SHA256,
            "minilm_model_weights_missing_or_changed")
    files = {str(p.relative_to(snapshot)): file_sha256(p) for p in sorted(snapshot.rglob("*"))
             if p.is_file() and not {"onnx", "openvino"} & set(p.relative_to(snapshot).parts)}
    return snapshot, {"model": EMBEDDING_MODEL, "revision": revision, "files_sha256": files,
                      "device": "cpu"}


@contextmanager
def preserved_cpu_rng():
    """Neutral CPU model loading/encoding must not advance native world RNGs."""
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch = sys.modules.get("torch")
    torch_state = torch.get_rng_state() if torch is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)


class NeutralHistory:
    """Original expertise/abstract embedding methods with complete first-year provenance."""

    def __init__(self, tracker, founders, directions):
        self.tracker = tracker
        self.founders = {a.id: a for a in founders}
        self.registry = {}
        require(len(self.founders) == len(founders), "duplicate_founder_ids")
        texts = [d.topic + " " + " ".join(d.keywords or []) for d in directions]
        vectors = tracker._encode_cached(texts)
        require(len(vectors) == len(directions), "direction_embedding_count_mismatch")
        self.direction_norms = {}
        for direction, vector in zip(directions, vectors):
            self.direction_norms[direction.topic] = self.vector_norm(vector, "direction", direction.topic)
        for agent in founders:
            require(bool(agent.expertise), "empty_founder_expertise", agent_id=agent.id)
            require(all(d.topic in self.direction_norms for d in agent.expertise),
                    "unknown_expertise_topic", agent_id=agent.id)
            tracker.register_agent_expertise(
                agent.id, [d.topic + " " + " ".join(d.keywords or []) for d in agent.expertise])
            self.vector_norm(tracker.expertise_centroids.get(agent.id), "expertise", agent.id)
        require(set(tracker.expertise_centroids) == set(self.founders),
                "expertise_registration_coverage_mismatch")

    @staticmethod
    def vector_norm(vector, kind, identifier):
        import numpy as np
        array = np.asarray(vector)
        require(array.ndim == 1 and array.size > 0 and np.issubdtype(array.dtype, np.number)
                and bool(np.isfinite(array).all()), "invalid_embedding", kind=kind, identifier=identifier)
        norm = float(np.linalg.norm(array))
        require(norm > 0 and bool(np.isfinite(norm)), "zero_or_invalid_embedding_norm",
                kind=kind, identifier=identifier)
        return norm

    @staticmethod
    def paper_identity(paper):
        data = paper.to_dict() if hasattr(paper, "to_dict") else paper
        authors = data["author_id"]
        author = authors[0] if isinstance(authors, list) and authors else authors
        abstract = data.get("abstract")
        require(type(data.get("id")) is str and type(author) is str
                and type(abstract) is str and bool(abstract.strip()), "invalid_submission_provenance")
        return data, author, hashlib.sha256(abstract.encode()).hexdigest()

    def reconcile(self, papers=None):
        require(set(self.tracker.paper_metadata) == set(self.registry)
                and set(self.tracker.embeddings) == set(self.registry),
                "embedding_registry_coverage_mismatch")
        if papers is not None:
            require(set(papers) == set(self.registry), "paper_registry_coverage_mismatch")
        for pid, record in self.registry.items():
            require(self.tracker.paper_metadata[pid] == {
                "author_id": record["author_id"], "year": record["first_submission_year"]},
                "embedding_metadata_conflict", paper_id=pid)
            self.vector_norm(self.tracker.embeddings[pid], "paper", pid)
            if papers is not None:
                _, author, abstract_hash = self.paper_identity(papers[pid])
                require(author == record["author_id"] and abstract_hash == record["abstract_sha256"],
                        "resubmission_changed_author_or_abstract", paper_id=pid)
        return {"status": "complete", "required_papers": len(self.registry),
                "embedded_papers": len(self.tracker.embeddings), "missing_papers": 0}

    def ingest(self, papers, year):
        self.reconcile()
        pending = []
        for pid, paper in papers.items():
            data, author, abstract_hash = self.paper_identity(paper)
            require(pid == data["id"] and author in self.founders, "unknown_submission_identity")
            if pid in self.registry:
                record = self.registry[pid]
                require(record["author_id"] == author and record["abstract_sha256"] == abstract_hash,
                        "resubmission_changed_author_or_abstract", paper_id=pid)
                continue
            require(data["year"] == year, "missing_first_submission_ingestion", paper_id=pid)
            pending.append((pid, data["abstract"], author, year))
        self.tracker.add_paper_embeddings_batch(pending)
        for pid, abstract, author, first_year in pending:
            self.registry[pid] = {
                "paper_id": pid, "author_id": author, "first_submission_year": first_year,
                "abstract_sha256": hashlib.sha256(abstract.encode()).hexdigest(),
            }
        coverage = self.reconcile(papers)
        self.tracker.flush_cache()
        return coverage

    def reference(self, agent_id, year, papers):
        self.reconcile(papers)
        require(agent_id in self.founders, "unknown_centroid_author")
        eligible = sorted(pid for pid, record in self.registry.items()
                          if record["author_id"] == agent_id
                          and year - 4 <= record["first_submission_year"] <= year - 1)
        vector = self.tracker.compute_career_centroid(agent_id, max_years=3, current_year=year - 1)
        norm = self.vector_norm(vector, "reference", agent_id)
        return vector, {"reference_source": "paper_history" if eligible else "initial_expertise",
                        "history_ids": eligible, "expected_history_count": len(eligible),
                        "history_window_start": year - 4, "history_window_end": year - 1,
                        "reference_norm": norm}


class PropensityMixin(feedback.FundingFeedbackMixin):
    """Only direction candidates and neutral observations differ from native P1F1."""

    def __init__(self, *args, cell, initial_cache, embedding_snapshot,
                 simulator_module, direction_lookup, initialize_only=False,
                 founder_count=None, horizon=None, **kwargs):
        require(cell in CELLS or (cell is None and initialize_only), "invalid_driver_cell")
        self.propensity_cell = cell
        self.initial_cache = initial_cache
        self.embedding_snapshot = Path(embedding_snapshot)
        self.simulator_module = simulator_module
        self.direction_lookup = direction_lookup
        self.initialize_only = initialize_only
        self._propensity_founder_count = FOUNDERS if founder_count is None else founder_count
        self._propensity_horizon = YEARS if horizon is None else horizon
        self.propensity_years, self.direction_events = {}, []
        self.initial_records = []
        super().__init__(*args, mechanisms=feedback.Mechanisms(), **kwargs)

    @property
    def docs(self):
        return Path(self.args.docs_dir)

    @property
    def propensity_founder_count(self):
        return getattr(self, "_propensity_founder_count", FOUNDERS)

    @property
    def propensity_horizon(self):
        return getattr(self, "_propensity_horizon", YEARS)

    def world_state_hash(self):
        return provenance.digest(canonical_state({
            "ecosystem": self.ecosystem.to_dict(),
            "conferences": self.conference_system.to_dict(),
        }))

    def initialize_agents(self):
        result = super().initialize_agents()
        founders = [a for a in self.ecosystem.agent_population.values() if a.get_type() == "university"]
        require(len(founders) == self.propensity_founder_count and self.initial_industry_count == 0,
                "population_must_be_1200_university_founders")
        require(not self.is_exploration_experiment
                and all(a.exploration_strategy == "balanced" and a.resources == 100 for a in founders),
                "common_population_configuration_mismatch")
        self.initialize_neutral_history(founders)
        return result

    def initialize_neutral_history(self, founders):
        import utopia.metrics.embedding_tracker as embedding_module
        constructor = embedding_module.SentenceTransformer
        with preserved_cpu_rng(), module_attribute(
            embedding_module, "SentenceTransformer",
            lambda _name, **kwargs: constructor(str(self.embedding_snapshot), device="cpu", local_files_only=True)
        ):
            self.embedding_tracker = embedding_module.EmbeddingTracker(
                model_name=EMBEDDING_MODEL, cache_dir=str(Path(self.args.data_cache_dir) / "embeddings"))
            self.neutral_history = getattr(self, "history_class", NeutralHistory)(
                self.embedding_tracker, founders, list(self.direction_lookup.values()))
            self.embedding_tracker.flush_cache()
        provenance.write_json(self.docs / "direction_embedding_norms.json", self.neutral_history.direction_norms)

    def _resource_rows(self):
        return [{"agent_id": aid, "active": bool(self.ecosystem.agent_population[aid].is_active),
                 "resources": float(self.ecosystem.agent_population[aid].resources)}
                for aid in self.founder_ids]

    def _setup_year(self, year):
        result = super()._setup_year(year)
        require(len(self.founder_ids) == self.propensity_founder_count, "founder_count_changed")
        self.propensity_years[year] = {
            "year": year, "cell": self.propensity_cell, "founder_count": self.propensity_founder_count,
            "year_start_resources": self._resource_rows(), "annual_charges": [],
        }
        self.propensity_years[year]["year_start_active_ids"] = [
            r["agent_id"] for r in self.propensity_years[year]["year_start_resources"] if r["active"]]
        return result

    def _run_phase_0_resubmissions(self, year, year_results):
        result = super()._run_phase_0_resubmissions(year, year_results)
        self.propensity_years[year]["after_phase0_resources"] = self._resource_rows()
        return result

    def _request_records(self, offset, year, index):
        audit = getattr(self.llm, "request_audit", None)
        require(audit is not None, "formal_direction_calls_require_request_audit")
        with Path(audit.path).open() as stream:
            stream.seek(offset)
            records = [json.loads(line) for line in stream if line.strip()]
        attempts = [r for r in records if r.get("event") == "request_started"
                    and r.get("seed_ctx") == ["phase1_directions", year]
                    and r.get("item_index") == index]
        attempts.sort(key=lambda r: r["attempt"])
        require(bool(attempts), "direction_request_audit_missing", year=year, index=index)
        for attempt in attempts:
            require(attempt["request_seed"] == self.simulator_module.derive_seed(
                SEED, "phase1_directions", year, index, attempt["attempt"]),
                "direction_request_seed_policy_changed")
        return attempts

    def _run_phase_1_research_directions(self, year, year_results):
        require(hasattr(self.llm, "generate_batch"), "batched_native_client_required")
        authors = self.ecosystem.get_available_authors()
        due = [a for a in authors if a.project_end_year < year and a.is_active]
        record = self.propensity_years[year]
        record.update(
            phase1_available_ids=[a.id for a in authors],
            phase1_eligible_ids=[a.id for a in due],
            phase1_not_due_ids=[a.id for a in authors if a.project_end_year >= year],
            phase1_inactive_ids=[aid for aid in self.founder_ids
                                 if not self.ecosystem.agent_population[aid].is_active],
        )
        record["phase0_lost_active_ids"] = sorted(
            set(record["year_start_active_ids"]) - {a.id for a in authors})
        if year == 1:
            require(len(due) == self.propensity_founder_count and all(a.newest_direction is None for a in due),
                    "initial_choice_risk_set_mismatch")
            self.pre_choice_state_hash = self.world_state_hash()
            if self.initial_cache is not None:
                require(self.initial_cache["pre_choice_state_hash"] == self.pre_choice_state_hash
                        and self.initial_cache["founder_ids"] == [a.id for a in due],
                        "pre_choice_state_or_founder_order_mismatch")
        decisions, references, candidate_map = {}, {}, {}
        for agent in due:
            if year == 1:
                candidates = [self.direction_lookup[d.topic] for d in agent.expertise]
            else:
                require(agent.newest_direction is not None, "repeat_has_no_previous_direction")
                previous = agent.newest_direction["direction"].topic
                require(previous in self.direction_lookup, "unknown_previous_topic")
                with preserved_cpu_rng():
                    centroid, references[agent.id] = self.neutral_history.reference(
                        agent.id, year, self.paper_tracker.papers_by_id)
                    distances = self.embedding_tracker.compute_direction_distances(
                        centroid, [d for topic, d in self.direction_lookup.items() if topic != previous])
                decision = policy.build_decision(
                    self.propensity_cell, agent.id, year, previous,
                    list(self.direction_lookup), distances, seed=SEED,
                    derive_seed_fn=self.simulator_module.derive_seed)
                decisions[agent.id] = decision
                candidates = [self.direction_lookup[t] for t in decision["candidate_topics"]]
            require(bool(candidates), "empty_direction_candidates")
            candidate_map[agent.id] = candidates

        native_selector = self.simulator_module.create_research_directions_batch
        generate = self.llm.generate_batch
        capture = []

        def observed_batch(prompts, **kwargs):
            require(len(prompts) == len(due) and not capture, "direction_batch_shape_changed")
            replay = year == 1 and self.initial_cache is not None
            audit = getattr(self.llm, "request_audit", None)
            offset = Path(audit.path).stat().st_size if audit is not None else None
            inputs = [{"prompt": prompt, "kwargs": kwargs, "agent_id": agent.id,
                       "candidate_topics": [d.topic for d in candidate_map[agent.id]],
                       "item_index": index}
                      for index, (agent, prompt) in enumerate(zip(due, prompts))]
            if replay:
                results = []
                for item, cached in zip(inputs, self.initial_cache["records"]):
                    require(provenance.digest(item) == cached["input_hash"], "replayed_initial_prompt_mismatch",
                            agent_id=item["agent_id"])
                    results.append((deepcopy(cached["raw_response"]), []))
            else:
                results = generate(prompts, **kwargs)
            require(len(results) == len(due), "direction_response_count_mismatch")
            for index, (item, (raw, _messages)) in enumerate(zip(inputs, results)):
                topics = item["candidate_topics"]
                fallback = ("parse" if raw is None else "validation"
                            if not isinstance(raw, dict)
                            or not {"topic", "detailed_focus", "reason"} <= set(raw)
                            or raw["topic"] not in topics else None)
                attempts = (self.initial_cache["records"][index]["request_attempts"]
                            if replay else self._request_records(offset, year, index))
                captured = {
                    "agent_id": item["agent_id"], "year": year, "item_index": index,
                    "candidate_topics": topics, "seed_ctx": list(kwargs["seed_ctx"]),
                    "input_hash": provenance.digest(item), "input": item, "raw_response": deepcopy(raw),
                    "fallback_kind": fallback, "request_attempts": attempts,
                    "request_origin": "common_initial_cache" if replay else "production_client",
                }
                append_record(self.docs / "direction_requests.jsonl", captured)
                capture.append(captured)
            return results

        def strict_selector(*args, **kwargs):
            require(not args and kwargs["agents"] == due, "native_direction_risk_set_drift")
            require(kwargs.get("candidate_map") is None, "unexpected_native_candidate_intervention")
            kwargs["candidate_map"] = candidate_map
            return native_selector(**kwargs)

        with module_attribute(self.simulator_module, "create_research_directions_batch", strict_selector):
            with feedback.instance_override(self.llm, "generate_batch", observed_batch):
                result = super()._run_phase_1_research_directions(year, year_results)
        require(len(capture) == len(due), "direction_capture_coverage_mismatch")
        for agent, captured in zip(due, capture):
            chosen = agent.newest_direction
            response = {"topic": chosen["direction"].topic,
                        "detailed_focus": chosen["detailed_focus"], "reason": chosen["reason"]}
            require(all(type(v) is str for v in response.values())
                    and response["topic"] in captured["candidate_topics"]
                    and chosen["direction"].years == 1
                    and agent.project_start_year == year and agent.project_end_year == year,
                    "native_choice_or_one_year_timeline_mismatch", agent_id=agent.id)
            if year == 1:
                event = {"status": "complete", "event_type": "initial_choice",
                         "agent_id": agent.id, "year": 1, "seed": SEED,
                         "is_initial_choice": True, "eligible_repeat": False,
                         "initial_choice_count": 1, "eligible_repeat_count": 0,
                         "new_project_choice_count": 1, "realized_switch_count": 0,
                         "realized_switch": False, "conditional_switch_distance": None,
                         "chosen_topic": response["topic"], "candidate_topics": captured["candidate_topics"],
                         "fallback": captured["fallback_kind"] is not None,
                         "fallback_kind": captured["fallback_kind"]}
                cached = {k: deepcopy(v) for k, v in captured.items() if k not in ("input", "request_origin")}
                cached["response"] = response
                if self.initial_cache is not None:
                    require(cached == self.initial_cache["records"][len(self.initial_records)],
                            "initial_choice_replay_differs", agent_id=agent.id)
                self.initial_records.append(cached)
            else:
                event = policy.finalize_choice(
                    decisions[agent.id], response["topic"], fallback_kind=captured["fallback_kind"])
                event.update(references[agent.id])
            event.update(cell=self.propensity_cell, detailed_focus=response["detailed_focus"],
                         reason=response["reason"], input_hash=captured["input_hash"],
                         project_start_year=agent.project_start_year, project_end_year=agent.project_end_year)
            event = self.direction_observation(event)
            append_record(self.docs / "direction_events.jsonl", event)
            self.direction_events.append(event)
        if year == 1:
            self.initial_state_hash = self.world_state_hash()
            if self.initial_cache is not None:
                require(self.initial_state_hash == self.initial_cache["initial_state_hash"],
                        "post_choice_initial_state_mismatch")
            if self.initialize_only:
                raise InitialChoicesComplete()
        return result

    def direction_observation(self, event):
        return event

    def _record_paper_metadata_batch(self, year):
        # Do not turn on the exploration experiment: no keywords, novelty scores,
        # strategy bonuses or unrelated phases. Observe every unique submission.
        with preserved_cpu_rng():
            return self.neutral_history.ingest(self.paper_tracker.papers_by_id, year)

    def _charge_annual_costs(self, year):
        with ExitStack() as stack:
            for agent in self.ecosystem.get_available_authors():
                native_update = agent.update_resources

                def observed(amount, *, _agent=agent, _update=native_update):
                    before, active = float(_agent.resources), bool(_agent.is_active)
                    result = _update(amount)
                    self.propensity_years[year]["annual_charges"].append({
                        "agent_id": _agent.id, "amount": amount, "resources_before": before,
                        "resources_after": float(_agent.resources), "active_before": active,
                        "active_after": bool(_agent.is_active),
                    })
                    require(amount == -10, "native_annual_cost_changed")
                    return result
                stack.enter_context(feedback.instance_override(agent, "update_resources", observed))
            return super()._charge_annual_costs(year)

    def run_one_round(self, year):
        result = super().run_one_round(year)
        record = self.propensity_years[year]
        record["history_coverage"] = self.neutral_history.reconcile(self.paper_tracker.papers_by_id)
        record["direction_events"] = [e for e in self.direction_events if e["year"] == year]
        record["agent_rows"] = self.funding_feedback_years[year]["agent_rows"]
        record["legacy_zero_boundary"] = self.funding_feedback_years[year]["legacy_zero_boundary"]
        record["paper_observations"] = []
        for pid, submitted in self.neutral_history.registry.items():
            paper = self.paper_tracker.papers_by_id[pid]
            require(paper.status in ("accept", "reject", "pending"), "unknown_paper_status")
            observed = {
                "paper_id": pid, "author_id": submitted["author_id"],
                "first_submission_year": submitted["first_submission_year"],
                "year": year, "status": paper.status,
                "citation_count": self.citation_tracker.get_citation_count(pid),
            }
            observed["realized_publication_citations"] = (
                observed["citation_count"] if paper.status == "accept" else 0)
            record["paper_observations"].append(observed)
            if submitted["first_submission_year"] == year:
                require(paper.status in ("accept", "reject"), "first_attempt_decision_missing", paper_id=pid)
                submitted["first_attempt_accepted"] = paper.status == "accept"
            if submitted["first_submission_year"] <= self.propensity_horizon - 3 and year == submitted["first_submission_year"] + 3:
                submitted["three_year_observation"] = observed
        provenance.write_json(self.docs / "years" / f"year_{year:02d}.json", record)
        provenance.write_json(self.docs / "submission_registry.json", {
            "schema_version": 1, "cell": self.propensity_cell, "year": year,
            "history_coverage": record["history_coverage"],
            "papers": list(self.neutral_history.registry.values()),
        })
        return result


def simulation_class(base=None):
    if base is None:
        from utopia.simulation import Simulation
        base = Simulation
    return type("SwitchingPropensitySimulation", (PropensityMixin, base), {})



PROSPECTIVE_PROTOCOL = "switching-frequency-fulltext-portable-v1"
PROSPECTIVE_CELLS = ("LN", "HN")


def prospective_spec(options):
    """A new six-year question, separate from all historical ten-year runs."""
    initial = bool(getattr(options, "initialize_only", False))
    pilot = bool(getattr(options, "pilot", False))
    require(options.seed == 42 and options.cell in (("INIT",) if initial else PROSPECTIVE_CELLS),
            "invalid_prospective_switching_cell")
    require(type(options.founders) is int and options.founders > 0 and options.founders % 5 == 0,
            "prospective_founders_must_be_positive_multiple_of_five")
    require((options.founders == 5000 and options.years == 6) if not pilot else
            (options.founders <= 5000 and 1 <= options.years <= 6),
            "formal_switching_requires_5000_founders_and_six_years")
    return {
        "protocol": PROSPECTIVE_PROTOCOL, "cell": options.cell,
        "purpose": "engineering_pilot" if pilot else "prospective_scientific_world",
        "initialize_only": initial, "founders": options.founders, "years": options.years,
        "start_year": 2016, "seed": 42, "annual_influx_rate": 0.0,
        "switch_probability": None if initial else policy.CELLS[options.cell][0],
        "distance_policy": "near-history", "direction_menu_size": policy.MENU_SIZE,
        "project_years": 1, "papers_per_project": 1, "resubmissions": "native_enabled",
        "funding_cell": "P1F1", "funding_selection": "one_complete_permutation_v1",
        "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
        "embedding_weights_sha256": EMBEDDING_WEIGHTS_SHA256,
        "adjacent_distance": "cosine_previous_selected_direction_to_current_selected_direction",
        "history_distance": "native_direction_to_prior_paper_centroid_cosine",
    }



def prospective_initial_identity(options, binding, *, driver_sha256=None, policy_sha256=None):
    identity = {
        "protocol": PROSPECTIVE_PROTOCOL, "seed": options.seed,
        "founders": options.founders, "years": options.years,
        "pilot": bool(getattr(options, "pilot", False)), "start_year": 2016,
        "model": binding["model"], "model_revision": binding["model_revision"],
        "corpus_sha256": binding["corpus_sha256"],
        "context_tokens": options.context_tokens, "max_tokens": options.max_tokens,
        "config": deepcopy(binding["science_config"]["config"]),
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_weights_sha256": EMBEDDING_WEIGHTS_SHA256,
        "driver_sha256": driver_sha256 or file_sha256(Path(__file__)),
        "policy_sha256": policy_sha256 or file_sha256(Path(policy.__file__)),
    }
    return json.loads(json.dumps(identity))



def load_prospective_initial_choices(options):
    payload = Path(options.initial_choices).read_bytes()
    record = json.loads(payload)
    require(record.get("protocol") == PROSPECTIVE_PROTOCOL
            and record.get("status") in ("initial_choices_ready", "initialization_only_complete")
            and record.get("scientific_years_completed") == 0,
            "initial_choices_require_new_prospective_initializer")
    founders, rows = record.get("founder_ids", []), record.get("records", [])
    require(len(rows) == len(founders) == options.founders
            and all(type(aid) is str for aid in founders)
            and len(set(founders)) == options.founders,
            "initial_choices_founder_coverage_mismatch")
    require(record.get("records_sha256") == provenance.digest(rows),
            "initial_choices_record_hash_mismatch")
    for index, (aid, row) in enumerate(zip(founders, rows)):
        require(row.get("agent_id") == aid and row.get("item_index") == index
                and row.get("year") == 1 and row.get("fallback_kind") is None
                and isinstance(row.get("raw_response"), dict)
                and all(row["raw_response"].get(key) == row.get("response", {}).get(key)
                        for key in ("topic", "detailed_focus", "reason"))
                and row.get("response", {}).get("topic") in row.get("candidate_topics", [])
                and all(type(row.get("response", {}).get(key)) is str
                        for key in ("topic", "detailed_focus", "reason")),
                "invalid_completed_initial_choice", index=index)
    for key in ("pre_choice_state_hash", "initial_state_hash"):
        value = record.get(key)
        require(type(value) is str and len(value) == 64
                and all(c in "0123456789abcdef" for c in value), "initial_choices_state_hash_missing")
    return payload, record



def publish_prospective_initial_choices(options):
    """Publish committed local choices; safe to resume without constructing a model."""
    import base64
    import subprocess
    from urllib.parse import urlsplit
    from utopia.runtime.checkpointing import PortableStop

    spec = prospective_spec(options)
    require(spec["initialize_only"], "publication_requires_initializer")
    require(options.run_id and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                   for c in options.run_id), "invalid_prospective_run_id")
    payload, record = load_prospective_initial_choices(options)
    digest = hashlib.sha256(payload).hexdigest()
    docs = ROOT / "outputs/docs" / f"{options.run_id}_{options.cell}"
    manifest_path = docs / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    binding = manifest["binding"]
    extra = binding["science_config"]["extra"]
    require(all(binding.get(key) == value for key, value in {
        "protocol": PROSPECTIVE_PROTOCOL, "cell": "INIT", "seed": options.seed,
        "model": options.model, "model_revision": options.model_revision}.items())
        and extra.get("switching") == spec
        and record["common_identity"] == prospective_initial_identity(
            options, binding, driver_sha256=extra["switching_driver_sha256"],
            policy_sha256=extra["switching_policy_sha256"]),
        "initial_choices_publication_binding_mismatch")
    require(manifest.get("years_completed", 0) == 0
            and manifest.get("status") in ("starting", "running", "paused", "initialization_only_complete")
            and manifest.get("initial_choices_sha256", digest) == digest,
            "initial_choices_publication_receipt_mismatch")
    manifest.update(status="paused", reason="initial_choices_publication_pending",
                    years_completed=0, scientific_years_completed=0, initial_choices_sha256=digest)
    provenance.write_json(manifest_path, manifest)
    try:
        location = urlsplit(options.s3_prefix)
        if location.scheme != "s3" or not location.netloc or location.query or location.fragment:
            raise ValueError("A study s3://bucket/prefix is required")
        prefix = location.path.strip("/")
        key = "/".join(part for part in (prefix, "shared", f"initial_choices_{digest}.json") if part)
        pointer_key = "/".join(part for part in (prefix, "shared", "initial_choices.json") if part)
        uri = f"s3://{location.netloc}/{key}"
        pointer_uri = f"s3://{location.netloc}/{pointer_key}"
        metadata = {
            "schema_version": 1, "status": "published", "protocol": PROSPECTIVE_PROTOCOL,
            "run_id": options.run_id, "seed": options.seed, "founders": options.founders,
            "years": options.years, "scientific_years_completed": 0,
            "initial_choices_uri": uri, "initial_choices_sha256": digest, "bytes": len(payload),
            "records_sha256": record["records_sha256"],
            "common_identity_sha256": provenance.digest(record["common_identity"]),
        }
        environment = {**os.environ, "AWS_MAX_ATTEMPTS": "1", "AWS_PAGER": ""}
        with tempfile.TemporaryDirectory(prefix="switching-publication-") as temporary:
            def put(object_key, body, *, immutable):
                # Upload the validated bytes, even if the source path is subsequently moved.
                path = Path(temporary) / "body.json"
                path.write_bytes(body)
                checksum = base64.b64encode(hashlib.sha256(body).digest()).decode()
                command = ["aws", "s3api", "put-object", "--bucket", location.netloc,
                           "--key", object_key, "--body", str(path), "--content-type", "application/json",
                           "--checksum-sha256", checksum, "--output", "json", "--no-cli-pager"]
                if immutable:
                    command += ["--if-none-match", "*"]
                try:
                    result = subprocess.run(command, check=True, capture_output=True, text=True,
                                            timeout=600, env=environment)
                    response = json.loads(result.stdout)
                except subprocess.CalledProcessError as error:
                    if not immutable or "PreconditionFailed" not in (error.stderr or ""):
                        raise
                    result = subprocess.run(
                        ["aws", "s3api", "head-object", "--bucket", location.netloc,
                         "--key", object_key, "--checksum-mode", "ENABLED", "--output", "json",
                         "--no-cli-pager"], check=True, capture_output=True, text=True,
                        timeout=600, env=environment)
                    response = json.loads(result.stdout)
                    if response.get("ContentLength") != len(body):
                        raise ValueError("Existing immutable S3 object has different bytes")
                if response.get("ChecksumSHA256") != checksum:
                    raise ValueError("S3 did not confirm the expected object SHA256")
            put(key, payload, immutable=True)
            put(pointer_key, (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode(), immutable=False)
        provenance.write_json(docs / "initial_choices_publication.json", metadata)
        manifest.update(status="initialization_only_complete", initial_choices_publication={
            **metadata, "pointer_uri": pointer_uri})
        manifest.pop("reason", None)
        provenance.write_json(manifest_path, manifest)
    except Exception as error:
        reason = f"initial_choices_publication_failed: {type(error).__name__}: {error}"
        manifest.update(status="paused", reason=reason)
        provenance.write_json(manifest_path, manifest)
        raise PortableStop(reason) from error
    return {"status": "initialization_only_complete", "scientific_years_completed": 0,
            "initial_choices_sha256": digest, "initial_choices_uri": uri,
            "pointer_uri": pointer_uri, "founder_count": len(record["founder_ids"])}



class ProspectiveNeutralHistory(NeutralHistory):
    """Index one read-only history snapshot per phase, retaining native centroids."""

    def prepare_year(self, year):
        self.reconcile()
        self._reference_year = year
        self._papers_by_author = {aid: {} for aid in self.founders}
        for pid, metadata in self.tracker.paper_metadata.items():
            self._papers_by_author[metadata["author_id"]][pid] = metadata

    def reference(self, agent_id, year, papers):
        require(getattr(self, "_reference_year", None) == year and agent_id in self.founders,
                "history_snapshot_not_prepared")
        metadata = self._papers_by_author[agent_id]
        eligible = sorted(pid for pid, item in metadata.items() if year - 4 <= item["year"] <= year - 1)
        with module_attribute(self.tracker, "paper_metadata", metadata):
            vector = self.tracker.compute_career_centroid(agent_id, max_years=3, current_year=year - 1)
        norm = self.vector_norm(vector, "reference", agent_id)
        return vector, {"reference_source": "paper_history" if eligible else "initial_expertise",
                        "history_ids": eligible, "expected_history_count": len(eligible),
                        "history_window_start": year - 4, "history_window_end": year - 1,
                        "reference_norm": norm}



class ProspectivePropensityMixin(PropensityMixin):
    """Native policy and phases with complete, data-only portable subclass state."""

    history_class = ProspectiveNeutralHistory

    def __init__(self, *args, propensity_options, **kwargs):
        self.prospective_options = propensity_options
        self.prospective_specification = prospective_spec(propensity_options)
        self.initial_choices_path = Path(propensity_options.initial_choices)
        self._direction_request_index = None
        mechanisms = kwargs.pop("mechanisms", feedback.Mechanisms())
        require(mechanisms == feedback.Mechanisms(), "switching_requires_common_P1F1")
        import utopia.simulation as simulator
        lookup = {d.topic: d for d in simulator.AVAILABLE_DIRECTIONS}
        require(all(d.years == 1 for d in lookup.values()), "prospective_requires_one_year_context")
        snapshot, _ = resolve_embedding_snapshot()
        initial = self.prospective_specification["initialize_only"]
        initial_cache = None if initial else self.read_initial_choices()
        super().__init__(
            *args, cell=None if initial else propensity_options.cell,
            initial_cache=initial_cache, embedding_snapshot=snapshot,
            simulator_module=simulator, direction_lookup=lookup, initialize_only=initial,
            founder_count=propensity_options.founders, horizon=propensity_options.years, **kwargs)
        if initial_cache is not None:
            require(initial_cache["common_identity"] == self.common_initial_identity(),
                    "initial_choices_scientific_binding_mismatch")

    def common_initial_identity(self):
        return prospective_initial_identity(self.prospective_options, self.args.checkpoint_binding)

    def read_initial_choices(self):
        return load_prospective_initial_choices(self.prospective_options)[1]

    def write_initial_choices(self):
        require(not self.yearly_results and not self.paper_tracker.papers_by_id
                and not self.funding_feedback_years,
                "initializer_must_stop_before_scientific_outcomes")
        record = {
            "protocol": PROSPECTIVE_PROTOCOL, "status": "initial_choices_ready",
            "scientific_years_completed": 0, "common_identity": self.common_initial_identity(),
            "founder_ids": list(self.founder_ids), "records": deepcopy(self.initial_records),
            "records_sha256": provenance.digest(self.initial_records),
            "pre_choice_state_hash": self.pre_choice_state_hash,
            "initial_state_hash": self.initial_state_hash,
        }
        if self.initial_choices_path.exists():
            previous = self.read_initial_choices()
            def science(value):
                value = deepcopy(value)
                value.pop("records_sha256")
                value.pop("status")
                for item in value["records"]:
                    item.pop("request_attempts")
                return value
            require(science(previous) == science(record), "refuse_to_replace_different_initial_choices")
            return previous
        else:
            from utopia.utils.data_utils import write_json_atomic
            self.initial_choices_path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.initial_choices_path, record)
        return record

    def _run_phase_1_research_directions(self, year, year_results):
        self._direction_request_index = None
        if year > 1:
            self.neutral_history.prepare_year(year)
        try:
            return super()._run_phase_1_research_directions(year, year_results)
        except InitialChoicesComplete:
            require(self.initialize_only and year == 1, "unexpected_prospective_initialization_boundary")
            self.write_initial_choices()
            from utopia.runtime.checkpointing import PortableStop
            raise PortableStop("prospective_initial_choices_complete") from None

    def _request_records(self, offset, year, index):
        audit = getattr(self.llm, "request_audit", None)
        require(audit is not None and offset is not None, "formal_direction_calls_require_request_audit")
        if self._direction_request_index is None:
            grouped = {}
            with Path(audit.path).open() as stream:
                stream.seek(offset)
                for line in stream:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if (item.get("event") == "request_started"
                            and item.get("seed_ctx") == ["phase1_directions", year]):
                        grouped.setdefault(item["item_index"], []).append(item)
            self._direction_request_index = grouped
        attempts = sorted(self._direction_request_index.get(index, []), key=lambda r: r["attempt"])
        require(bool(attempts), "direction_request_audit_missing", year=year, index=index)
        for item in attempts:
            require(item["request_seed"] == self.simulator_module.derive_seed(
                42, "phase1_directions", year, index, item["attempt"]),
                "direction_request_seed_policy_changed")
        return attempts

    def direction_observation(self, event):
        from utopia.runtime.checkpointing import PortableCheckpointError
        if event["fallback"]:
            raise PortableCheckpointError("Prospective switching cannot replace an invalid direction response")
        event["distance_to_history"] = event["conditional_switch_distance"]
        event["adjacent_direction_distance"] = None
        if not event["is_initial_choice"]:
            previous, chosen = event["previous_topic"], event["chosen_topic"]
            if previous == chosen:
                event["adjacent_direction_distance"] = 0.0
            else:
                import numpy as np
                texts = [self.direction_lookup[t].topic + " " + " ".join(
                    self.direction_lookup[t].keywords or []) for t in (previous, chosen)]
                with preserved_cpu_rng():
                    vectors = self.embedding_tracker._encode_cached(texts)
                a, b = (np.asarray(v, dtype=np.float64) for v in vectors)
                norms = (self.neutral_history.vector_norm(a, "direction", previous),
                         self.neutral_history.vector_norm(b, "direction", chosen))
                event["adjacent_direction_distance"] = float(1.0 - np.dot(a / norms[0], b / norms[1]))
        return event

    def extra_checkpoint_state(self):
        state = super().extra_checkpoint_state()
        tracker = self.embedding_tracker
        def vectors(values):
            return {key: {"dtype": value.dtype.str, "values": value.tolist()}
                    for key, value in values.items()}
        state["prospective_switching"] = {
            "specification": deepcopy(self.prospective_specification),
            "propensity_years": deepcopy(self.propensity_years),
            "direction_events": deepcopy(self.direction_events),
            "initial_records": deepcopy(self.initial_records), "initial_cache": deepcopy(self.initial_cache),
            "pre_choice_state_hash": getattr(self, "pre_choice_state_hash", None),
            "initial_state_hash": getattr(self, "initial_state_hash", None),
            "registry": deepcopy(self.neutral_history.registry),
            "direction_norms": deepcopy(self.neutral_history.direction_norms),
            "embeddings": vectors(tracker.embeddings),
            "paper_metadata": deepcopy(tracker.paper_metadata),
            "expertise_centroids": vectors(tracker.expertise_centroids),
            "disk_cache": vectors(tracker._disk_cache),
            "near_threshold": tracker.near_threshold, "far_threshold": tracker.far_threshold,
        }
        return state

    def restore_extra_checkpoint_state(self, state):
        import numpy as np
        from utopia.runtime.checkpointing import PortableCheckpointError
        state = deepcopy(state)
        saved = state.pop("prospective_switching", None)
        if not isinstance(saved, dict) or saved.get("specification") != self.prospective_specification:
            raise PortableCheckpointError("Prospective switching checkpoint policy differs")
        super().restore_extra_checkpoint_state(state)
        founders = [a for a in self.ecosystem.agent_population.values() if a.get_type() == "university"]
        require(len(founders) == self.propensity_founder_count, "checkpoint_founder_coverage_mismatch")
        self.initialize_neutral_history(founders)
        for field in ("propensity_years", "direction_events", "initial_records", "initial_cache",
                      "pre_choice_state_hash", "initial_state_hash"):
            setattr(self, field, saved[field])
        self.neutral_history.registry = saved["registry"]
        self.neutral_history.direction_norms = saved["direction_norms"]
        tracker = self.embedding_tracker
        for target, key in (("embeddings", "embeddings"), ("expertise_centroids", "expertise_centroids"),
                            ("_disk_cache", "disk_cache")):
            setattr(tracker, target, {k: np.asarray(v["values"], dtype=np.dtype(v["dtype"]))
                                     for k, v in saved[key].items()})
        tracker.paper_metadata = saved["paper_metadata"]
        tracker.near_threshold, tracker.far_threshold = saved["near_threshold"], saved["far_threshold"]
        tracker._disk_cache_dirty = True
        tracker.flush_cache()
        self.neutral_history.reconcile(self.paper_tracker.papers_by_id)
        self._direction_request_index = None



def prospective_simulation_class(base=None):
    if base is None:
        from utopia.simulation import Simulation
        base = Simulation
    return type("ProspectiveSwitchingSimulation", (ProspectivePropensityMixin, base), {})



def configure_prospective_switching(args, options):
    """Bind the design and declare only switching-owned replayable outputs."""
    spec = prospective_spec(options)
    for name, expected in {"num_years": options.years, "start_year": 2016,
                           "papers_per_project": 1, "annual_influx_rate": 0.0,
                           "disable_resubmission": False, "initial_funding": 100}.items():
        require(getattr(args, name) == expected, "prospective_shared_setting_mismatch", field=name)
    paths = dict(getattr(args, "checkpoint_replay_paths", {}))
    docs = Path(args.docs_dir)
    # Compatibility with the first shared runner: the live manifest is not replayable.
    if paths.get("metrics") == str(docs):
        paths.pop("metrics")
    for name in ("direction_embedding_norms.json", "direction_requests.jsonl",
                 "direction_events.jsonl", "submission_registry.json"):
        paths["switching/" + name] = str(docs / name)
    for year in range(1, options.years + 1):
        name = f"years/year_{year:02d}.json"
        paths["switching/" + name] = str(docs / name)
    args.checkpoint_replay_paths = paths
    return {"switching": spec,
            "initial_choices_sha256": None if spec["initialize_only"] else file_sha256(Path(options.initial_choices)),
            "switching_driver_sha256": file_sha256(Path(__file__)),
            "switching_policy_sha256": file_sha256(Path(policy.__file__))}



def run_prospective(options):
    """Fresh initialization and both cells use the shared new portable runtime."""
    from utopia.runtime.checkpointing import PortableStop
    spec = prospective_spec(options)
    options.protocol, options.funding_cell = PROSPECTIVE_PROTOCOL, "P1F1"
    initial = spec["initialize_only"]
    choices_exist = Path(options.initial_choices).is_file()
    if initial and choices_exist:
        if not options.resume:
            raise FileExistsError("Existing initial choices require explicit --resume")
        return publish_prospective_initial_choices(options)
    if not initial and not choices_exist:
        raise FileNotFoundError("Run prospective-initialize before either LN or HN")
    from functools import partial
    import utopia.experiments.funding_feedback as runner
    import utopia.simulation as simulator
    import utopia.agents.research_direction as directions
    def configure(args):
        return configure_prospective_switching(args, options)
    def factory(base):
        return partial(prospective_simulation_class(base), propensity_options=options)
    try:
        return runner.run_prospective(
            options, simulation_factory=factory, configure=configure,
            run_context=lambda: one_year_context(simulator, directions))
    except PortableStop as error:
        if not initial or str(error) != "prospective_initial_choices_complete":
            raise
        return publish_prospective_initial_choices(options)



def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (['prospective-initialize'], ['prospective']):
        from utopia.experiments.funding_feedback import prospective_arguments
        parser = argparse.ArgumentParser(description=__doc__)
        prospective_arguments(parser)
        initial = argv[0] == 'prospective-initialize'
        parser.set_defaults(founders=5000, initialize_only=initial)
        for action in parser._actions:
            if action.dest == 'cell':
                action.choices = ('INIT',) if initial else PROSPECTIVE_CELLS
                if initial:
                    action.default, action.required = 'INIT', False
        parser.add_argument('--initial-choices', type=Path, required=True)
        parser.add_argument('--pilot', action='store_true')
        return run_prospective(parser.parse_args(argv[1:]))
    from utopia.experiments.common import main_for
    return main_for('switching_propensity', argv)

if __name__ == "__main__":
    main()
