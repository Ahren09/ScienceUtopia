"""Native switching-propensity × near/far-history case study, seed 42.

Only phase-1 candidates, common one-year wording/directions, and neutral
embedding observations are added. Native phase ordering, parser fallbacks,
memory, costs, submissions, reviews and funding award processing remain inherited.
V3 prospectively replaces compact joint funding elicitation with the frozen
sequential remaining-ID method after the failed v2 family. All four conditions
and their shared technical initializer are fresh; no old cache or partial world
is adopted. This changes elicitation, not the direction policy or its RNG.
The initializer stops after common phase 1, before any scientific outcome.
Run only in the separate propensity worktree. This module launches no server.
"""

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


def configure(options):
    from utopia.arguments import parse_arguments
    require(options.seed == SEED and os.environ.get("PYTHONHASHSEED") == str(SEED),
            "require_seed42_and_pythonhashseed42")
    name = identifier(getattr(options, "cell", None))
    paths = {category: ROOT / "outputs" / category / name
             for category in ("docs", "logs", "checkpoints", "visual")}
    cache = ROOT / "data/cache" / name
    for path in [*paths.values(), cache]:
        require(not path.exists() and not path.is_symlink(),
                "fresh_output_and_data_cache_required", path=str(path))
    with tempfile.TemporaryDirectory(prefix="propensity-arguments-") as temporary:
        previous_cache = os.environ.get("UTOPIA_DATA_CACHE_DIR")
        os.environ["UTOPIA_DATA_CACHE_DIR"] = str(Path(temporary) / "cache")
        try:
            args = parse_arguments([
            "--experiment_name", "default_switching_propensity", "--experiment_stage", "mechanism",
            "--output-dir", temporary, "--seed", "42", "--population_mode", "university_only",
            "--num_institutions", "240", "--researchers_per_institution", "5",
            "--strategy_mix", "balanced", "--initial_funding", "100",
            "--num_years", "10", "--num_conferences", "6", "--papers_per_project", "1",
            "--funding_panel_max_apps", "25", "--funding_budget_mode", "track",
            "--funding_allocation_mode", "fixed", "--log_funding_applications",
            "--model", MODEL, "--vllm_url", options.vllm_url, "--always_rerun",
            ])
        finally:
            if previous_cache is None:
                os.environ.pop("UTOPIA_DATA_CACHE_DIR", None)
            else:
                os.environ["UTOPIA_DATA_CACHE_DIR"] = previous_cache
    args.experiment_id = name
    for attribute, category in (("docs_dir", "docs"), ("log_dir", "logs"),
                                ("checkpoint_dir", "checkpoints"), ("visual_dir", "visual")):
        setattr(args, attribute, str(paths[category]))
    args.output_dir = args.checkpoint_dir
    args.data_cache_dir, args.rag_device = str(cache), "cpu"
    for path in [*paths.values(), cache]:
        path.mkdir(parents=True, exist_ok=False)
    return args


def queue_jobs(worktree=ROOT, private_root=None):
    """Five declarative client jobs. No execution or source-dependent old run IDs."""
    worktree = Path(worktree)
    # Raw audits bind the producer's lexical worktree/outputs paths. The queue
    # verifies their canonical private-storage ownership before admitting them.
    storage = worktree
    python = str(Path(private_root) / "env/bin/python") if private_root is not None else sys.executable
    initial = storage / "outputs/docs" / identifier()
    cache = initial / "initial_choices.json"
    jobs = []
    for cell in (None, *CELLS):
        name = identifier(cell)
        docs = storage / "outputs/docs" / name
        command = module_command(python, 'utopia.experiments.switching_propensity', 'initialize' if cell is None else 'run', '--seed', '42', '--vllm-url', '__SERVER_URL__', '--server-manifest', '__SERVER_MANIFEST__')
        if cell is not None:
            command += ["--cell", cell, "--initial-choices", str(cache)]
        gates = {"/status": "complete", "/protocol": PROTOCOL, "/seed": 42,
                 "/founder_count": FOUNDERS,
                 "/funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
                 "/funding_output_representation": OUTPUT_REPRESENTATION,
                 "/sequential_source_binding": feedback.sequential_source_binding(ROOT),
                 "/inputs/source_files_sha256": source_identity(),
                 "/inputs/server_runtime_profile": TP1_PROFILE_NAME,
                 "/inputs/runtime_amendment": RUNTIME_AMENDMENT,
                 "/sequential_audit_valid": True, "/funding_application_evidence_valid": True,
                 "/funding_amendment": FUNDING_AMENDMENT}
        manifest_name = "initialization_manifest.json" if cell is None else "propensity_manifest.json"
        gates.update({"/kind": "initialization_only", "/choice_count": FOUNDERS,
                      "/scientific_years_completed": 0} if cell is None else {
                          "/kind": "condition", "/cell": cell, "/years_completed": YEARS})
        funding = {f"/{k}": v for k, v in FUNDING_GATES.items()}
        sequential = {"/" + k: v for k, v in sequential_gates(cell is None).items()}
        if cell is None:
            funding["/final_panels"] = 0
        artifacts = [
            {"path": str(docs / manifest_name), "min_bytes": 1, "equals": gates},
            {"path": str(docs / "llm_request_audit.jsonl"), "min_bytes": 1},
            {"path": str(docs / "llm_request_audit.summary.json"), "min_bytes": 1,
             "equals": {f"/{k}": v for k, v in REQUEST_GATES.items()}},
            {"path": str(docs / "funding_validation.summary.json"), "min_bytes": 1,
             "equals": funding},
            {"path": str(docs / "funding_validation.jsonl"), "min_bytes": 0},
            {"path": str(docs / feedback.SEQUENTIAL_AUDIT_FILE), "min_bytes": 1},
            {"path": str(docs / feedback.SEQUENTIAL_SUMMARY_FILE), "min_bytes": 1, "equals": sequential},
            {"path": str(docs / "run_manifest.json"), "min_bytes": 1,
             "equals": {"/status": "initialization_only_complete",
                        "/scientific_years_completed": 0, "/years_completed": 0} if cell is None
             else {"/status": "complete", "/years_completed": YEARS}},
        ]
        if cell is None:
            artifacts.append({"path": str(cache), "min_bytes": 1,
                              "equals": {"/status": "complete", "/protocol": PROTOCOL,
                                         "/seed": 42, "/founder_count": FOUNDERS,
                                         "/inputs/source_files_sha256": source_identity(),
                                         "/inputs/server_runtime_profile": TP1_PROFILE_NAME,
                                         "/inputs/runtime_amendment": RUNTIME_AMENDMENT,
                                         "/inputs/funding_selection_protocol": FUNDING_SELECTION_PROTOCOL}})
        else:
            artifacts.extend([
                {"path": str(docs / "direction_events.jsonl"), "min_bytes": 1},
                {"path": str(docs / "direction_requests.jsonl"), "min_bytes": 1},
                {"path": str(docs / "submission_registry.json"), "min_bytes": 1},
                {"path": str(docs / "direction_embedding_norms.json"), "min_bytes": 1},
            ])
            for year in range(1, YEARS + 1):
                artifacts.append({"path": str(docs / "years" / f"year_{year:02d}.json"),
                                  "min_bytes": 1,
                                  "equals": {"/year": year, "/cell": cell,
                                             "/founder_count": FOUNDERS,
                                             "/history_coverage/status": "complete"}})
                artifacts.append({"path": str(docs / f"mechanisms_year_{year}.json"),
                                  "min_bytes": 1,
                                  "equals": {"/year": year, "/cell": "P1F1",
                                             "/metrics/founder_count": FOUNDERS}})
        jobs.append({"id": name, "family": "propensity", "seed": SEED,
                     "role": "initialize" if cell is None else "condition",
                     "cell": cell, "depends_on": [] if cell is None else [identifier()],
                     "command": command, "env": {"PYTHONHASHSEED": "42", "WANDB_MODE": "disabled"},
                     "require_manifest_argument": True,
                     "output_roots": [str(storage / "outputs" / category / name)
                                      for category in ("docs", "logs", "checkpoints", "visual")],
                     "artifacts": artifacts})
    return jobs


















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
            lambda _name: constructor(str(self.embedding_snapshot), device="cpu", local_files_only=True)
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


def validate_resolved_settings(args, config):
    expected = {
        "seed": SEED, "population_mode": "university_only", "num_institutions": 240,
        "researchers_per_institution": 5, "initial_funding": 100, "strategy_mix": "balanced",
        "num_years": YEARS, "num_conferences": 6, "papers_per_project": 1,
        "funding_panel_max_apps": 25, "funding_budget_mode": "track",
        "funding_allocation_mode": "fixed", "log_funding_applications": True,
        "disable_resubmission": False, "funding_intervention": False, "citation_intervention": False,
        "reviewer_capacity": None, "reviewer_matching": "random", "reviews_per_paper": 3,
        "review_policy": "persona", "review_score_mode": "int", "acceptance_tiebreak": "stable",
        "acceptance_mode": "rate", "always_rerun": True,
    }
    for key, value in expected.items():
        require(getattr(args, key) == value, "resolved_argument_mismatch", field=key)
    require(config["conference"]["annual_cost"] == 10
            and config["conference"]["resubmission_cost"] == 5
            and config["conference"]["acceptance_rate"] == .3
            and config["conference"]["use_default_acceptance_rate"] is True
            and config["funding"]["academic_base_budget"] == 20
            and config["llm"]["max_tokens"] == 2048
            and config["llm"]["enable_thinking"] is True,
            "native_shared_configuration_changed")


def runtime_identity():
    from importlib.metadata import version
    return {"python": sys.version, "packages": {name: version(name) for name in (
        "torch", "transformers", "sentence-transformers", "openai", "numpy", "pandas", "vllm")}}


def execute(options):
    """Formal producer/condition entrypoint. All actual calls share required guards."""
    os.chdir(ROOT)
    args = configure(options)
    initialize_only = options.command == "initialize"
    cell = None if initialize_only else options.cell
    docs = Path(args.docs_dir)
    manifest_path = docs / ("initialization_manifest.json" if initialize_only else "propensity_manifest.json")
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL, "status": "running",
        "kind": "initialization_only" if initialize_only else "condition", "cell": cell,
        "seed": SEED, "founder_count": FOUNDERS, "command": shlex.join(sys.argv),
        "args": vars(args), "scientific_years_completed": 0,
        "funding_output_representation": OUTPUT_REPRESENTATION,
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "sequential_source_binding": feedback.sequential_source_binding(ROOT),
        "funding_amendment": FUNDING_AMENDMENT,
        "estimand": "Four dynamic policies conditional on one initialized ecosystem, not independent worlds.",
        "world_standard_error": None, "world_confidence_interval": None,
        "funding_baseline": deepcopy(feedback.FUNDING_BASELINE),
    }
    provenance.write_json(manifest_path, manifest)
    try:
        from utopia.config import SIMULATION_CONFIG
        validate_resolved_settings(args, SIMULATION_CONFIG)
        server = provenance.server_provenance(options.server_manifest, options.vllm_url,
                                     runtime_profile=TP1_PROFILE_NAME)
        cache_provenance = provenance.verified_cache_manifest()
        snapshot, embedding_identity = resolve_embedding_snapshot()
        import utopia.simulation as simulator
        import utopia.agents.research_direction as directions_module
        from utopia.models.request_audit import request_audit_scope
        from utopia.funding.validation import install_funding_validation, FundingRankingValidationError
        with one_year_context(simulator, directions_module) as lookup:
            inputs = json.loads(json.dumps({
                "protocol": PROTOCOL, "seed": SEED, "source_files_sha256": source_identity(),
                "resolved_config": SIMULATION_CONFIG, "effective_args": feedback.effective_args(args),
                "server_runtime_hash": server["runtime_hash"], "runtime": runtime_identity(),
                "server_runtime_profile": TP1_PROFILE_NAME,
                "runtime_amendment": RUNTIME_AMENDMENT,
                "rag_cache": cache_provenance, "embedding_model": embedding_identity,
                "one_year_directions": [d.to_dict() for d in lookup.values()],
                "common_prompt_replacements": PROMPT_CHANGES,
                "request_seed_policy": "derive_seed(run_seed, phase_context, item_index, retry)",
                "funding_output_representation": OUTPUT_REPRESENTATION,
                "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
                "sequential_source_binding": feedback.sequential_source_binding(ROOT),
                "funding_amendment": FUNDING_AMENDMENT,
            }))
            manifest.update(inputs=inputs, server_provenance=server)
            initial_cache, initial_hash = None, None
            if not initialize_only:
                expected_cache = ROOT / "outputs/docs" / identifier() / "initial_choices.json"
                require(Path(options.initial_choices).resolve() == expected_cache.resolve(),
                        "condition_requires_common_fresh_initializer_path")
                initial_cache, initial_hash, initializer_manifest, initializer_hash = validate_initialization(
                    options.initial_choices, inputs)
                manifest.update(initial_choices_path=str(Path(options.initial_choices)),
                                initial_choices_sha256=initial_hash,
                                initialization_manifest_path=str(initializer_manifest),
                                initialization_manifest_sha256=initializer_hash)
            provenance.write_json(manifest_path, manifest)
            logging.basicConfig(
                level=logging.INFO,
                handlers=[logging.FileHandler(Path(args.log_dir) / "simulation.log"), logging.StreamHandler()],
            )
            simulator.set_seed(SEED)
            funding_handle = install_funding_validation(
                simulator.FundingAgency, docs / "funding_validation.jsonl", compact=True)
            audit, sim, sequential = None, None, None
            try:
                with feedback.sequential_funding_scope(
                        simulator.FundingAgency, simulator.VLLMServerModel, docs) as sequential, \
                        feedback.funding_phase_coverage(simulator.Simulation, simulator.FundingAgency, sequential), \
                        request_audit_scope(docs / "llm_request_audit.jsonl", model_name=MODEL) as audit:
                    llm = simulator.VLLMServerModel(
                        model_name=MODEL, base_url=args.vllm_url,
                        max_concurrent_requests=args.batch_size, run_seed=SEED,
                    )
                    require(MODEL in {entry.id for entry in llm.client.models.list().data},
                            "endpoint_model_mismatch")
                    sim = simulation_class(simulator.Simulation)(
                        llm=llm, args=args, num_years=YEARS, output_dir=args.output_dir,
                        cell=cell, initial_cache=initial_cache, embedding_snapshot=snapshot,
                        simulator_module=simulator, direction_lookup=lookup, initialize_only=initialize_only,
                        **{key: getattr(args, key) for key in (
                            "verbose", "debug", "industry_funding_mode", "funding_allocation_mode",
                            "always_rerun", "use_langchain", "max_retries", "weighted_funding_assignment",
                            "experiment_name", "collaboration_mode", "collaboration_network_growth_mode",
                            "start_year")},
                    )
                    sim.cache_provenance = cache_provenance
                    stopped_at_initial = False
                    try:
                        sim.run()
                    except InitialChoicesComplete:
                        require(initialize_only, "unexpected_initialization_stop")
                        stopped_at_initial = True
                        sim.wandb_logger.finish()
                        simulator.write_run_manifest(
                            args.docs_dir, status="initialization_only_complete",
                            extra={"scientific_years_completed": 0, "years_completed": 0})
                    require(initialize_only == stopped_at_initial, "initialization_boundary_failed")
                    require(len(sim.initial_records) == FOUNDERS
                            and len(sim.founder_ids) == FOUNDERS, "initial_choice_completion_failed")
                    if initialize_only:
                        require(not sim.yearly_results and not sim.paper_tracker.papers_by_id,
                                "initializer_must_have_no_scientific_outcomes")
                    else:
                        require(set(sim.propensity_years) == set(range(1, YEARS + 1))
                                and len(sim.yearly_results) == YEARS,
                                "incomplete_ten_year_condition")
                        for submitted in sim.neutral_history.registry.values():
                            if submitted["first_submission_year"] <= YEARS - 3:
                                require("three_year_observation" in submitted,
                                        "missing_citation_deadline_observation", paper_id=submitted["paper_id"])
                        require(file_sha256(options.initial_choices) == initial_hash,
                                "initial_choice_cache_changed_during_run")
                        require(file_sha256(initializer_manifest) == initializer_hash,
                                "initializer_manifest_changed_during_run")
                    require(cache_provenance["input_files"] == provenance.cache_file_identity(),
                            "immutable_rag_cache_changed")
                    require(resolve_embedding_snapshot()[1] == embedding_identity,
                            "immutable_minilm_snapshot_changed")
                    require(source_identity() == inputs["source_files_sha256"], "source_changed_during_run")
                    if not funding_handle.summary()["completion_allowed"]:
                        raise FundingRankingValidationError(funding_handle.summary())
            finally:
                try:
                    funding_handle.restore()
                finally:
                    manifest["funding_validation"] = funding_handle.summary()
                    provenance.write_json(docs / "funding_validation.summary.json", manifest["funding_validation"])
                    if audit is not None:
                        manifest["request_audit"] = audit.summary()
                    if sequential is not None:
                        manifest["funding_sequential"] = sequential.summary()
            provenance.validate_request_audit_summary(manifest["request_audit"])
            require(all(manifest["funding_validation"].get(k) == v for k, v in FUNDING_GATES.items()),
                    "funding_completion_gate_failed")
            manifest.update(
                pre_choice_state_hash=sim.pre_choice_state_hash, initial_state_hash=sim.initial_state_hash,
                founder_ids=list(sim.founder_ids), choice_count=len(sim.initial_records),
                population_blueprint_sha256=file_sha256(Path(args.output_dir) / "population_blueprint.csv"),
                program_state=canonical_state(sim.initial_world)["ecosystem"]["agents"][-2:],
                funding_sequential_audits={
                    name: file_sha256(docs / name)
                    for name in (feedback.SEQUENTIAL_AUDIT_FILE, feedback.SEQUENTIAL_SUMMARY_FILE)},
                funding_application_evidence=feedback.funding_ledger_evidence(
                    args.output_dir, 0 if initialize_only else YEARS),
                sequential_audit_valid=True, funding_application_evidence_valid=True,
                status="complete",
            )
            validate_funding_evidence(docs, manifest, expected_source=inputs["source_files_sha256"])
            if initialize_only:
                require(manifest["funding_validation"]["final_panels"] == 0,
                        "initializer_performed_funding_evaluation")
                cache = {
                    "schema_version": 1, "status": "complete", "protocol": PROTOCOL, "seed": SEED,
                    "founder_count": FOUNDERS, "choice_count": FOUNDERS,
                    "founder_ids": list(sim.founder_ids), "inputs": inputs,
                    "pre_choice_state_hash": sim.pre_choice_state_hash,
                    "initial_state_hash": sim.initial_state_hash, "records": sim.initial_records,
                    "records_sha256": provenance.digest(sim.initial_records),
                }
                validate_choices(cache, inputs)
                path = docs / "initial_choices.json"
                require(not path.exists() and not path.is_symlink(), "initial_cache_already_exists")
                # Commit once, then make read-only before advertising completion.
                write_json_file(cache, path, mode="x", trailing_newline=True,
                                fsync=True, indent=2, allow_nan=False)
                path.chmod(0o444)
                _, cache_hash = load_choices(path, inputs)
                manifest.update(initial_choices_path=str(path), initial_choices_sha256=cache_hash)
            else:
                manifest.update(years_completed=YEARS, scientific_years_completed=YEARS,
                                corpus_hash=sim.corpus_hash,
                                history_coverage=sim.neutral_history.reconcile(sim.paper_tracker.papers_by_id))
            manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status=getattr(error, "status", "failed"), error=traceback.format_exc())
        if hasattr(error, "diagnostics"):
            manifest["failure_diagnostics"] = error.diagnostics
        provenance.write_json(manifest_path, manifest)
        provenance.mark_failed_run_manifest(docs, PROTOCOL, manifest["error"])
        raise
    provenance.write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("initialize", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--seed", type=int, choices=[SEED], required=True)
        sub.add_argument("--vllm-url", required=True)
        sub.add_argument("--server-manifest", type=Path, required=True)
        if command == "run":
            sub.add_argument("--cell", choices=CELLS, required=True)
            sub.add_argument("--initial-choices", type=Path, required=True)
    plan = commands.add_parser("queue-plan")
    plan.add_argument("--worktree", type=Path, default=ROOT)
    plan.add_argument("--private-root", type=Path)
    options = parser.parse_args(argv)
    if options.command == "queue-plan":
        print(json.dumps(queue_jobs(options.worktree, options.private_root), indent=2))
    else:
        execute(options)


if __name__ == "__main__":
    main()
