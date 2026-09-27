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
                 simulator_module, direction_lookup, initialize_only=False, **kwargs):
        require(cell in CELLS or (cell is None and initialize_only), "invalid_driver_cell")
        self.propensity_cell = cell
        self.initial_cache = initial_cache
        self.embedding_snapshot = Path(embedding_snapshot)
        self.simulator_module = simulator_module
        self.direction_lookup = direction_lookup
        self.initialize_only = initialize_only
        self.propensity_years, self.direction_events = {}, []
        self.initial_records = []
        super().__init__(*args, mechanisms=feedback.Mechanisms(), **kwargs)

    @property
    def docs(self):
        return Path(self.args.docs_dir)

    def world_state_hash(self):
        return provenance.digest(canonical_state({
            "ecosystem": self.ecosystem.to_dict(),
            "conferences": self.conference_system.to_dict(),
        }))

    def initialize_agents(self):
        result = super().initialize_agents()
        founders = [a for a in self.ecosystem.agent_population.values() if a.get_type() == "university"]
        require(len(founders) == FOUNDERS and self.initial_industry_count == 0,
                "population_must_be_1200_university_founders")
        require(not self.is_exploration_experiment
                and all(a.exploration_strategy == "balanced" and a.resources == 100 for a in founders),
                "common_population_configuration_mismatch")
        import utopia.metrics.embedding_tracker as embedding_module
        constructor = embedding_module.SentenceTransformer
        with preserved_cpu_rng(), module_attribute(
            embedding_module, "SentenceTransformer",
            lambda _name, **_kwargs: constructor(str(self.embedding_snapshot), device="cpu", local_files_only=True)
        ):
            self.embedding_tracker = embedding_module.EmbeddingTracker(
                model_name=EMBEDDING_MODEL, cache_dir=str(Path(self.args.data_cache_dir) / "embeddings"))
            self.neutral_history = NeutralHistory(
                self.embedding_tracker, founders, list(self.direction_lookup.values()))
            self.embedding_tracker.flush_cache()
        provenance.write_json(self.docs / "direction_embedding_norms.json", self.neutral_history.direction_norms)
        return result

    def _resource_rows(self):
        return [{"agent_id": aid, "active": bool(self.ecosystem.agent_population[aid].is_active),
                 "resources": float(self.ecosystem.agent_population[aid].resources)}
                for aid in self.founder_ids]

    def _setup_year(self, year):
        result = super()._setup_year(year)
        require(len(self.founder_ids) == FOUNDERS, "founder_count_changed")
        self.propensity_years[year] = {
            "year": year, "cell": self.propensity_cell, "founder_count": FOUNDERS,
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
            require(len(due) == FOUNDERS and all(a.newest_direction is None for a in due),
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
            if submitted["first_submission_year"] <= YEARS - 3 and year == submitted["first_submission_year"] + 3:
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


def main(argv=None):
    from utopia.experiments.common import main_for
    return main_for("switching_propensity", argv)


if __name__ == "__main__":
    main()
