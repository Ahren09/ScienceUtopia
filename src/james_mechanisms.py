"""James's bounded 2 x 2 mechanism ablation, reusing the original simulator.

P1F1 retains the stock rules under the common panelized baseline below.
P0 hides the *structured* publication
record (counts, acceptance statuses and titles) in funding ranking prompts and
replaces the track-record criterion with an instruction about withheld data.
Original proposals, proposal generation, program choice, identities, expertise,
and affiliation remain unchanged. Proposals can reveal past work: P0 is a
record-display ablation, NOT publication-blind evaluation or removal of all
publication feedback.

F0 blocks university award credits at the original phase-5 update_resources
call, before the original attrition check. Earned awards, winners, success
histories, and FundingTracker allocation accounting remain factual. Negative
resource updates still execute. There is no compensating endowment or reset:
the estimand includes reduced resource supply and consequent attrition, rather
than an equal-budget redistribution policy. Success histories still inform
subsequent direction/program choice. Existing resource thresholds are unchanged.
In particular, the legacy phase-5 truthiness check leaves an active researcher
at exactly zero resources active, whereas positive balances at or below 10 are
removed. Negative overshoots deactivate in update_resources. Every cell retains
this discontinuity; year-end snapshots report its frequency and denominators.
All cells share an empty-year compatibility adapter: the original paper tracker
returns an empty DataFrame without columns in a year with no completed papers.
The adapter supplies the columns already required by the original funding phase.
An empty retrieval batch returns an empty result without encoding queries; all
other phase-2 work, including pending resubmissions, still executes.
An empty year also receives an empty pending-paper list for the original
acceptance phase's final invariant check. These shared schema adapters create
no papers and change no nonempty-year decision.

The protocol is 1,200 university researchers (240 x 5), all balanced, six years,
six conferences, one shared seed 42 fixed before outcomes. Other scientific
settings use the existing parser/config defaults, except that every cell sets
funding_panel_max_apps=25. Stock deterministic balanced panels replace global
program competition. The track budget mode, stock NSF/DARPA rates (0.23/0.10),
and max(1, int(panel_size * rate)) winner rule remain unchanged. Rounding now
occurs per panel and can reduce the program's total awards. This is a new
panelized baseline, not demonstrated equivalence to global ranking. Funding
application logs and strict complete-ranking validation are required; invalid
final rankings abort instead of supplying repaired or fallback awards.
V4 prospectively requests ordered application IDs, validates a full permutation,
and derives the redundant applicant IDs and ranks before stock award processing.
The substantive application text, candidates and evaluation criteria are retained.
The response instructions/schema change, so identical sampled rankings are not
claimed. V3 stopped at live preflight without starting scientific jobs.
V6 prospectively uses sequential next-best choices from remaining IDs, preserving
the condition-specific panel text (including P0 masking). Its joint elicitation
distribution and retry granularity differ from v5; no equivalence is claimed.
All v5 partial/failure evidence is retained and no condition is adopted.
Other defaults include integer reviews,
stable acceptance ties, fixed per-award amounts and resubmissions. It estimates
total six-year effects within this synthetic population, not a decomposition
of all cumulative advantage or a causal effect in academia.

Commands, from the repository root (no server launch is implemented here):
    python src/james_mechanisms.py plan --vllm-url http://localhost:8000/v1
    PYTHONHASHSEED=42 python src/james_mechanisms.py run --cell P1F1 --seed 42 \
        --vllm-url http://localhost:8000/v1 --server-manifest SERVER_MANIFEST.json
    python src/james_mechanisms.py analyze
The plan prints all four exact client commands. The main agent owns execution,
server model/precision verification and the all-eight-EC2-GPU queue as capacity
becomes available, without preempting other jobs. Only an external
Qwen/Qwen3-32B server is accepted. No heuristic simulation is provided.

Outputs: outputs/{docs,logs,checkpoints,visual}/james_mechanisms_<cell>_seed<seed>.
Run into fresh directories. Resume is deliberately unsupported, since the
original checkpoints do not encode this wrapper's intervention provenance.
Initial world, ordered loaded corpus, code and resolved-config fingerprints
are checked across paired cells. Per-request seeds remain the simulator's
existing seeds, which are index-based, NOT coupled per-agent random tapes once
worlds diverge. Original fallback counts/flags are exposed, not silently hidden.

Exploration propensity is deferred: the original choice occurs on project
completion (durations vary), with strategy-specific candidate-distance bins.
A separate future intervention should draw a seeded Bernoulli move at each
eligible decision and independently constrain distance conditional on moving.
Report moves / eligible decisions, eligible decisions / active person-years,
and conditional distance, including cold-start and empty-bin fallbacks.
This patch does not reinterpret distance as propensity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import tempfile
import traceback


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42,)
FOUNDER_COUNT = 1200
YEARS = 6
FUNDING_PANEL_MAX_APPS = 25
FUNDING_OUTPUT_REPRESENTATION = "ordered_application_ids_v1"
FUNDING_SELECTION_PROTOCOL = "sequential_remaining_ids_v1"
SEQUENTIAL_AUDIT_FILE = "funding_sequential.jsonl"
SEQUENTIAL_SUMMARY_FILE = "funding_sequential.summary.json"
SEQUENTIAL_PROTOCOL_PATH = "docs/james_sequential_funding_protocol.md"
FROZEN_FUNDING_PROGRAM_RATES = {
    "NSF_THEORY": .23, "NSF_AI_FOUNDATIONS": .23, "NSF_SYSTEM": .23,
    "DARPA_AUTONOMOUS": .10, "DARPA_SECURITY": .10, "DARPA_AI_APPS": .10,
}
FUNDING_BASELINE = {
    "name": "stock_panel25_track",
    "panel_max_apps": FUNDING_PANEL_MAX_APPS,
    "funding_budget_mode": "track",
    "program_funding_rates": {"NSF": 0.23, "DARPA": 0.10},
    "winner_quota_rule": "max(1, int(panel_size * program.funding_rate))",
    "output_representation": FUNDING_OUTPUT_REPRESENTATION,
    "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
    "elicitation_amendment": (
        "Prospective v6 sequential next-best choices from remaining application IDs "
        "replace joint ordered-ID elicitation after the v5 funding-validity failure. "
        "All n choices are model-produced, with three SDK calls per step at most "
        "and no accepted-prefix reset. Compact final validation and native award "
        "processing remain. Condition-specific application text, including P0 "
        "masking, candidates and criteria remain unchanged. Joint elicitation "
        "equivalence is not claimed. No v5 condition completed or is adopted."
    ),
    "competition": "Stock seeded balanced panels within each program.",
    "interpretation": (
        "Common panelized baseline: competition is local to each panel; per-panel "
        "rounding can reduce program-level awards relative to global ranking. "
        "No panel/global ranking equivalence is claimed."
    ),
}
MODEL = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
PROTOCOL = "james-mechanisms-v6-single-world"
CELLS = ("P1F1", "P0F1", "P1F0", "P0F0")
PRIMARY_METRICS = (
    "cumulative_earned_funding_gini", "spendable_resources_gini",
    "mean_spendable_resources", "active_fraction",
    "accepted_papers_per_founder", "citations_all_papers_per_founder",
)
CACHE_INPUTS = {
    "outputs/cache_scievo.pkl": "e39b9b68a91fe2a86a6085934bf126b20c384a7d5e6ea4537feebf6255f98396",
    "outputs/thenlper_gte-small/cache_embeddings.pt":
        "f8f9bcbe2c8d81c613de2e24fb6bec90ffc204150acc6823f70fb38bc5ede0e9",
}
CACHE_MANIFEST = "outputs/logs/ec2_preparation/cache_alignment.json"


@dataclass(frozen=True)
class Mechanisms:
    publication_record_visible: bool = True
    awards_feed_resources: bool = True

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise TypeError("Mechanism switches must be booleans")

    @property
    def cell(self):
        return f"P{int(self.publication_record_visible)}F{int(self.awards_feed_resources)}"

    @classmethod
    def from_cell(cls, cell):
        if cell not in CELLS:
            raise ValueError(f"Unknown cell: {cell}")
        return cls(cell[1] == "1", cell[3] == "1")


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_file_identity(root=ROOT):
    """The original loader uses these shared paths, not args.data_cache_dir."""
    result = {}
    for name, expected in CACHE_INPUTS.items():
        path = root / name
        if path.stat().st_mode & 0o222:
            raise ValueError(f"Shared input cache must be read-only: {path}")
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Input cache differs from the original transferred file: {name}")
        result[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return result


def audit_cache(output=ROOT / CACHE_MANIFEST):
    """Bind original ordered documents to embedding rows without rewriting caches.

    The original cache lacks embedded document identifiers. Full row bindings
    and fixed, evenly spaced re-encoding checks provide explicit provenance;
    sampled numerical agreement is not a claim of full re-encoding validation.
    """
    import pickle
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    identity = cache_file_identity()
    with (ROOT / "outputs/cache_scievo.pkl").open("rb") as stream:
        cache = pickle.load(stream)  # Exact user-authorized source, hash verified above.
    docs = cache["documents"]
    embeddings = torch.load(ROOT / "outputs/thenlper_gte-small/cache_embeddings.pt",
                            map_location="cpu", weights_only=True)
    ids = [d.metadata["id"] for d in docs]
    if (len(set(ids)) != len(ids) or embeddings.ndim != 2
            or embeddings.shape != (len(docs), 384) or not torch.isfinite(embeddings).all()):
        raise ValueError("Invalid document IDs or embedding dimensions/values")
    year_positions = Counter()
    binding = hashlib.sha256()
    for row, doc in enumerate(docs):
        year = doc.metadata["publication_year"]
        if tuple(cache["id2year_pos"][doc.metadata["id"]]) != (year, year_positions[year]):
            raise ValueError("Cached ID-to-year/position mapping does not match document order")
        year_positions[year] += 1
        binding.update(json.dumps([row, doc.metadata["id"],
                                   hashlib.sha256(doc.page_content.encode()).hexdigest(),
                                   hashlib.sha256(embeddings[row].numpy().tobytes()).hexdigest()],
                                  separators=(",", ":")).encode() + b"\n")
    indices = np.unique(np.linspace(0, len(docs) - 1, min(256, len(docs)), dtype=int))
    model = SentenceTransformer("thenlper/gte-small",
                               revision="17e1f347d17fe144873b1201da91788898c639cd",
                               device="cpu", local_files_only=True)
    recomputed = model.encode([docs[i].page_content for i in indices], convert_to_tensor=True,
                              normalize_embeddings=True, batch_size=32, device="cpu")
    cosine = torch.nn.functional.cosine_similarity(
        recomputed.float(), embeddings[indices].float(), dim=1)
    if float(cosine.min()) < 0.999:
        raise ValueError(f"Document/embedding alignment check failed: min cosine {cosine.min()}")
    record = {
        "input_files": identity, "document_count": len(docs), "embedding_shape": list(embeddings.shape),
        "corpus_hash": corpus_fingerprint(docs), "ordered_ids_hash": digest(ids),
        "all_row_binding_sha256": binding.hexdigest(),
        "embedding_model": "thenlper/gte-small",
        "embedding_revision": "17e1f347d17fe144873b1201da91788898c639cd",
        "checked_rows": [int(i) for i in indices],
        "checked_row_min_cosine": float(cosine.min()),
        "alignment_scope": (
            "Exact original cache pair copied read-only from Cheesecake; ordered ID, text and "
            "embedding-byte bindings cover every row. Fixed 256-row CPU re-encoding check "
            "tests alignment numerically; not a historical generation manifest or full re-encoding proof."
        ),
    }
    if identity != cache_file_identity():
        raise RuntimeError("Cache changed during alignment audit")
    write_json(output, record)
    return record


def verified_cache_manifest():
    record = json.loads((ROOT / CACHE_MANIFEST).read_text())
    if record["input_files"] != cache_file_identity():
        raise ValueError("Cache manifest does not match immutable input files")
    if record["checked_row_min_cosine"] < 0.999 or len(record["checked_rows"]) != 256:
        raise ValueError("Cache alignment audit is incomplete")
    return record


def record_display_prompt(original_builder, program, apps, submitted_papers):
    """Reuse the original formatter, changing only the explicit record display.

    Copies are shallow on purpose: only relevant_projects is replaced. Author
    identity, proposal, input ordering and real evaluation metadata are retained.
    Literal guards fail on incompatible upstream prompt changes.
    """
    hidden_apps = [dict(app, relevant_projects=[]) for app in apps]
    prompt = original_builder(program, hidden_apps, submitted_papers)
    empty_record = (
        "Past Performance:\nThe agent has no research projects completed yet."
    )
    criterion = (
        "3. Past performance and track record. Focus more on recent works and "
        "accepted papers. Focus less on older works and rejected papers."
    )
    if prompt.count(empty_record) != len(apps) or prompt.count(criterion) != 1:
        raise RuntimeError("Funding prompt changed or has ambiguous record markers")
    prompt = prompt.replace(
        empty_record, "Past Performance:\nPublication record withheld by the experiment."
    )
    return prompt.replace(
        criterion,
        "3. The structured publication record is withheld. Do not treat withheld "
        "data as evidence of no completed research. Evaluate the remaining information.",
    )


@contextmanager
def instance_override(instance, name, replacement):
    """Restore exact instance attribute ownership even after a phase failure."""
    absent = object()
    previous = vars(instance).get(name, absent)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if previous is absent:
            delattr(instance, name)
        else:
            setattr(instance, name, previous)


class AwardResourceGate:
    """Phase-local gate: costs pass through, nonnegative award calls are audited."""

    def __init__(self, agent, enabled):
        self.original = agent.update_resources
        self.enabled = enabled
        self.award_calls = []

    def __call__(self, delta):
        if not isinstance(delta, (int, float)) or not math.isfinite(delta):
            raise ValueError("Resource update must be finite and numeric")
        if delta < 0:
            return self.original(delta)
        self.award_calls.append(float(delta))
        if self.enabled:
            return self.original(delta)
        return None


def new_awards(agent, before, year):
    """Extract actual newly earned awards without altering success history."""
    history = agent.funding_success_history
    records = []
    if set(before) - set(history):
        raise RuntimeError("Funding phase removed existing award history")
    for program, entries in history.items():
        prefix = before.get(program, [])
        if entries[:len(prefix)] != prefix:
            raise RuntimeError("Funding phase rewrote existing award history")
        for entry in entries[len(prefix):]:
            amount = entry["amount"]
            if (entry["year"] != year or not isinstance(amount, (int, float))
                    or not math.isfinite(amount) or amount < 0):
                raise RuntimeError("Invalid newly earned award")
            records.append({
                "year": year, "agent_id": agent.id,
                "program_id": program, "earned_amount": float(amount),
            })
    return records


def funding_dataframe(original_getter, year):
    """Keep no-paper years evaluable, including when the F0 world contracts."""
    frame = original_getter(year)
    required = ("author_id", "author_type", "status", "maturity")
    if frame.empty and any(column not in frame.columns for column in required):
        return frame.reindex(columns=list(dict.fromkeys([*frame.columns, *required])))
    return frame


def empty_safe_retrieve(original, queries, *args, **kwargs):
    """No authors means no query results; never skip the rest of phase 2."""
    if len(queries) == 0:
        import numpy as np
        return {"topk_indices": np.empty((0, 0), dtype=np.int64),
                "similarity_scores": np.empty((0, 0), dtype=np.float32)}
    return original(queries, *args, **kwargs)


def legacy_zero_boundary(rows):
    """Observational year-end prevalence, not a count of resource debit events."""
    if not rows or len({r["agent_id"] for r in rows}) != len(rows):
        raise ValueError("Boundary audit requires a nonempty unique cohort")
    active = [r for r in rows if r["active"]]
    zero = [r for r in rows if r["spendable_resources"] == 0]
    active_zero = [r for r in zero if r["active"]]
    return {
        "cohort_count": len(rows), "active_count": len(active),
        "zero_resource_count": len(zero), "active_zero_count": len(active_zero),
        "active_zero_fraction_of_cohort": len(active_zero) / len(rows),
        "active_zero_fraction_of_active": len(active_zero) / len(active) if active else None,
        "active_zero_agent_ids": sorted(r["agent_id"] for r in active_zero),
        "measurement": "End of inherited phase 5; exact equality, no tolerance or mechanics change.",
    }


def source_fingerprint(root=ROOT):
    paths = [root / "run_simulation.py", root / "const.py",
             root / "src/james_mechanisms.py", root / "src/james_funding_validation.py",
             root / "src/james_funding_compact.py",
             root / "src/james_funding_sequential.py",
             root / SEQUENTIAL_PROTOCOL_PATH]
    paths += sorted((root / "utopia").rglob("*.py"))
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in paths})


def sequential_source_binding(root=ROOT):
    """The executable helper and prospective protocol must both exist at freeze."""
    return {name: file_hash(Path(root) / name) for name in
            ("src/james_funding_sequential.py", SEQUENTIAL_PROTOCOL_PATH)}


def sequential_summary_gates():
    # n_clients may be zero only when the helper proves there were zero panels.
    # validate_summary/validate_audit enforce that relationship, not this projection.
    return {
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "output_representation": FUNDING_OUTPUT_REPRESENTATION,
        "installed": False, "status": "complete", "completion_allowed": True,
        "failed_panels": 0, "cancelled_panels": 0, "unprocessed_panels": 0,
        "fatal_errors": 0, "guard_failures": 0, "processing_failures": 0,
        "audit_failures": 0, "restore_conflicts": [],
    }


@contextmanager
def sequential_funding_scope(FundingAgency, VLLMServerModel, directory):
    """Nest inside compact validation; always restore this layer first."""
    from james_funding_sequential import install_sequential_funding, validate_summary
    directory = Path(directory)
    handle = install_sequential_funding(
        FundingAgency, VLLMServerModel, directory / SEQUENTIAL_AUDIT_FILE)
    try:
        yield handle
        validate_summary(handle.summary(), require_restored=False)
    finally:
        handle.restore()  # The helper exclusively writes/finalizes its own summary.
    validate_summary(handle.summary())
    if json.loads((directory / SEQUENTIAL_SUMMARY_FILE).read_text()) != handle.summary():
        raise ValueError("Sequential helper summary differs from persisted evidence")


@contextmanager
def funding_phase_coverage(simulation_class, funding_class, handle):
    """Observe the real phase boundary, including years with zero applications."""
    name = "_run_phase_5_update_funding"
    original = getattr(simulation_class, name)

    @wraps(original)
    def phase(simulation, year, *args, **kwargs):
        agencies = sorted(agent.id for agent in simulation.ecosystem.agent_population.values()
                          if isinstance(agent, funding_class))
        handle.begin_year(year, agencies)
        result = original(simulation, year, *args, **kwargs)
        handle.end_year(year)
        return result

    with instance_override(simulation_class, name, phase):
        yield


def validate_sequential_evidence(directory, compact_summary, *, manifest_summary=None,
                                 source_binding=None, application_dir=None, num_years=YEARS,
                                 expected_agency_ids=("NSF", "DARPA"),
                                 program_rates=FROZEN_FUNDING_PROGRAM_RATES):
    """Reconstruct choices/processing from raw evidence, including true empty runs."""
    from james_funding_sequential import validate_audit, validate_summary
    directory = Path(directory)
    summary = json.loads((directory / SEQUENTIAL_SUMMARY_FILE).read_text())
    validate_summary(summary)
    report = validate_audit(directory / SEQUENTIAL_AUDIT_FILE, summary,
                            request_audit_path=directory / "llm_request_audit.jsonl")
    if ([row["year"] for row in report["years"]] != list(range(1, num_years + 1))
            or summary["years_started"] != num_years or summary["years_completed"] != num_years
            or any(set(row["expected_agency_ids"]) != set(expected_agency_ids)
                   for row in report["years"])):
        raise ValueError("Sequential coverage omits or changes a native funding year/agency")
    joined = report["request_audit"]
    if (
            Path(joined["path"]).resolve() != (directory / "llm_request_audit.jsonl").resolve()
            or joined["matched_sdk_calls"] != summary["sdk_calls"]
            or joined["sha256"] != file_hash(directory / "llm_request_audit.jsonl")):
        raise ValueError("Sequential SDK evidence is not bound to this run's request audit")
    if manifest_summary is not None and summary != manifest_summary:
        raise ValueError("Sequential funding manifest/summary mismatch")
    if summary["processed_panels"] != compact_summary.get("final_panels"):
        raise ValueError("Sequential/compact processed-panel coverage mismatch")
    if (source_binding is not None and (
            summary["source_sha256"] != source_binding["src/james_funding_sequential.py"]
            or any(summary["source_files_sha256"].get(key) != value
                   for key, value in source_binding.items()))):
        raise ValueError("Sequential funding source fingerprint mismatch")
    # The native ledger omits application_id, but retains every position and
    # llm_rank. Joining by rank preserves repeated applicants without collapsing
    # their distinct model-selected applications.
    expected, panels = {}, {}
    for panel in report["processed_panels"]:
        identity = (panel["year"], panel["program_id"], panel["panel_index"])
        order = panel["ranked_application_ids"]
        if (identity in panels or type(identity[0]) is not int
                or not 1 <= identity[0] <= num_years):
            raise ValueError("Sequential panel year or identity is invalid")
        panels[identity] = len(order)
        for rank, app_id in enumerate(order, 1):
            expected[identity + (rank,)] = panel["application_to_applicant"][str(app_id)]
    compact_path = directory / ("funding_validation.jsonl"
                                if (directory / "funding_validation.jsonl").exists()
                                else "funding_validation_audit.jsonl")
    compact_panels = Counter()
    for line in compact_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "final_panel":
            if (row.get("valid") is not True or row.get("n_expected") != row.get("n_valid")
                    or row.get("n_expected") != row.get("n_returned")
                    or any(row.get(k) != 0 for k in
                           ("n_missing", "n_rejected", "n_imputed", "n_fallback"))):
                raise ValueError("Compact final-panel evidence is invalid")
            compact_panels[row["program_id"], row["panel_index"], row["n_expected"]] += 1
    if compact_panels != Counter((program, index, n)
                                for (_, program, index), n in panels.items()):
        raise ValueError("Compact final panels differ from sequential processing trace")
    observed = {}
    application_dir = Path(application_dir) if application_dir is not None else directory
    for year in range(1, num_years + 1):
        log = application_dir / f"funding_applications_year_{year}.jsonl"
        if not log.exists():
            continue
        for line in log.read_text().splitlines():
            row = json.loads(line)
            identity = (row["year"], row["program_id"], row["panel_index"])
            rank = row["llm_rank"]
            key = identity + (rank,)
            if (identity not in panels or row["year"] != year or type(rank) is not int
                    or key in observed or key not in expected
                    or row["n_panel"] != panels[identity]
                    or row["position"] != rank or row["applicant_id"] != expected[key]
                    or row["program_id"] not in program_rates
                    or row["funding_rate"] != program_rates[row["program_id"]]
                    or row.get("fallback_ranking") is not False or row.get("imputed_tail") is not False
                    or row["num_winners"] != max(1, int(row["n_panel"] * row["funding_rate"]))
                    or row["funded"] is not (rank <= row["num_winners"])):
                raise ValueError("Native funding ledger differs from sequential model-selected ranking")
            observed[key] = row["applicant_id"]
    if observed != expected:
        raise ValueError("Native funding ledger does not cover every sequential selection")
    return summary


def funding_ledger_evidence(directory, num_years):
    """Bind observed bytes and explicit absence; absence is validated against coverage."""
    directory = Path(directory)
    present, absent = {}, []
    for year in range(1, num_years + 1):
        path = directory / f"funding_applications_year_{year}.jsonl"
        if path.exists():
            present[path.name] = file_hash(path)
        else:
            absent.append(year)
    return {"present_sha256": present, "absent_years": absent}


def mark_failed_run_manifest(directory, protocol, error):
    """Update only the current attempt's lifecycle; retain native phase/year evidence."""
    path = Path(directory) / "run_manifest.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(status="failed", wrapper_protocol=protocol, wrapper_error=error)
    write_json(path, record)


def corpus_fingerprint(documents):
    """Hash actual ordered content, string-normalizing the loader's date metadata."""
    hasher = hashlib.sha256()
    for doc in documents:
        # Original RAG stores datetime/Timestamp/np.datetime64 in "published".
        hasher.update(json.dumps(
            [doc.page_content, doc.metadata], sort_keys=True,
            separators=(",", ":"), allow_nan=False, default=str,
        ).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


class JamesMechanismsMixin:
    """Put before Simulation in the MRO. All simulation phases remain inherited."""

    def __init__(self, *args, mechanisms=Mechanisms(), **kwargs):
        self.mechanisms = mechanisms
        self.james_years = {}
        self.james_initial_world = None
        self.james_corpus_hash = None
        super().__init__(*args, **kwargs)

    def load_checkpoint(self, year):
        raise RuntimeError("James mechanism runs require fresh worlds, no checkpoint resume")

    def _setup_year(self, year):
        if year == 1:
            # to_dict is the simulator's existing state serializer, without LLMs.
            self.james_initial_world = {
                "ecosystem": self.ecosystem.to_dict(),
                "conferences": self.conference_system.to_dict(),
            }
            founders = [a for a in self.ecosystem.agent_population.values()
                        if a.get_type() == "university"]
            if hasattr(self.args, "james_mechanisms") and len(founders) != FOUNDER_COUNT:
                raise RuntimeError("Protocol requires exactly 1200 initial university founders")
            self.james_founders = tuple(a.id for a in founders)
            self.james_initial_world_hash = digest(self.james_initial_world)
            write_json(Path(self.args.docs_dir) / "initial_world.json",
                       self.james_initial_world)
        return super()._setup_year(year)

    def _run_phase_2_submit_papers(self, *args, **kwargs):
        original = self.rag.batch_retrieve
        with ExitStack() as stack:
            stack.enter_context(instance_override(
                self.rag, "batch_retrieve", lambda queries, *a, **kw: empty_safe_retrieve(
                    original, queries, *a, **kw)))
            if getattr(self, "james_cache_provenance", None):
                original_build = self.rag.build_knowledge_index

                def checked_build():
                    expected = self.james_cache_provenance
                    if (corpus_fingerprint(self.rag.documents) != expected["corpus_hash"]
                            or digest([d.metadata["id"] for d in self.rag.documents])
                            != expected["ordered_ids_hash"]):
                        raise ValueError("Loaded document order differs from audited cache")
                    result = original_build()
                    if list(self.rag.document_embeddings.shape) != expected["embedding_shape"]:
                        raise ValueError("Loaded embedding shape differs from audited cache")
                    return result

                stack.enter_context(instance_override(self.rag, "build_knowledge_index", checked_build))
            result = super()._run_phase_2_submit_papers(*args, **kwargs)
        if self.james_corpus_hash is None:
            self.james_corpus_hash = corpus_fingerprint(self.rag.documents)
        return result

    def _run_phase_5_update_funding(self, year, submissions_list, year_results):
        # Type identity is retained: do not replace agents with proxy objects.
        researchers = [a for a in self.ecosystem.agent_population.values()
                       if a.get_type() == "university"]
        agencies = [a for a in self.ecosystem.agent_population.values()
                    if a.get_type() == "funding_agency"]
        before = {
            a.id: {p: [dict(e) for e in entries]
                   for p, entries in a.funding_success_history.items()}
            for a in researchers
        }
        gates = {a.id: AwardResourceGate(a, self.mechanisms.awards_feed_resources)
                 for a in researchers}
        with ExitStack() as stack:
            original_getter = self.paper_tracker.get_papers_dataframe
            stack.enter_context(instance_override(
                self.paper_tracker, "get_papers_dataframe",
                lambda year: funding_dataframe(original_getter, year)))
            for agent in researchers:
                stack.enter_context(instance_override(
                    agent, "update_resources", gates[agent.id]))
            if not self.mechanisms.publication_record_visible:
                for agency in agencies:
                    original = agency._build_program_prompt

                    def hidden(program, apps, papers, builder=original):
                        return record_display_prompt(builder, program, apps, papers)

                    stack.enter_context(instance_override(
                        agency, "_build_program_prompt", hidden))
            result = super()._run_phase_5_update_funding(
                year, submissions_list, year_results)

        awards = []
        for agent in researchers:
            earned = new_awards(agent, before[agent.id], year)
            # Fail rather than silently suppress a newly introduced refund or
            # miss an upstream direct assignment that bypasses update_resources.
            if Counter(e["earned_amount"] for e in earned) != Counter(
                    gates[agent.id].award_calls):
                raise RuntimeError("Phase-5 resource credits do not match earned awards")
            for event in earned:
                event["spendable_credit"] = (
                    event["earned_amount"] if self.mechanisms.awards_feed_resources else 0.0
                )
            awards.extend(earned)
        record = {
            "year": year, "cell": self.mechanisms.cell, "awards": awards,
            "legacy_zero_boundary": legacy_zero_boundary([
                {"agent_id": a.id, "active": bool(a.is_active),
                 "spendable_resources": float(a.resources)} for a in researchers
            ]),
        }
        self.james_years[year] = record
        year_results["james_mechanisms"] = record
        return result

    def _run_phase_4_acceptance_decisions(self, year, year_results):
        pending = self.paper_tracker.pending_papers_by_year
        if year not in pending:
            if any(p.year == year and p.status == "pending"
                   for p in self.paper_tracker.papers_by_id.values()):
                raise RuntimeError("Pending-paper index is missing real pending papers")
            pending[year] = []
        return super()._run_phase_4_acceptance_decisions(year, year_results)

    def run_one_round(self, year):
        result = super().run_one_round(year)
        rows = []
        for aid in self.james_founders:
            agent = self.ecosystem.agent_population[aid]
            papers = [p for p in self.paper_tracker.papers_by_id.values()
                      if aid in p.all_author_ids]
            earned = sum(e["amount"] for entries in agent.funding_success_history.values()
                         for e in entries)
            rows.append({
                "agent_id": aid, "active": bool(agent.is_active),
                "spendable_resources": float(agent.resources),
                "cumulative_earned_funding": float(earned),
                "accepted_papers": sum(p.status == "accept" for p in papers),
                "submitted_papers": len(papers),
                # Full author counting, including rejected/pending papers and
                # inactive founders. No conditioning on acceptance or survival.
                "citations_all_papers": sum(self.citation_tracker.get_citation_count(p.id)
                                           for p in papers),
            })
        self.james_years[year]["agent_rows"] = rows
        self.james_years[year]["metrics"] = summarize_agents(rows)
        if self.james_years[year]["legacy_zero_boundary"] != legacy_zero_boundary(rows):
            raise RuntimeError("Founder boundary snapshot changed after funding phase")
        write_json(Path(self.args.docs_dir) / f"mechanisms_year_{year}.json",
                   self.james_years[year])
        return result


def simulation_class(base=None):
    """Lazy import keeps plan, analysis and unit tests free of GPU dependencies."""
    if base is None:
        from run_simulation import Simulation
        base = Simulation
    return type("JamesMechanismSimulation", (JamesMechanismsMixin, base), {})


def summarize_agents(rows):
    from utopia.metrics.tracker import calculate_gini_coefficient

    if not rows:
        raise ValueError("Fixed founder cohort must not be empty")
    def gini(field):
        values = [r[field] for r in rows]
        # All-zero funding is resource collapse, not evidence of equal success.
        return float(calculate_gini_coefficient(values)) if sum(values) else None

    return {
        "founder_count": len(rows),
        "cumulative_earned_funding_gini": gini("cumulative_earned_funding"),
        "spendable_resources_gini": gini("spendable_resources"),
        "mean_spendable_resources": statistics.mean(r["spendable_resources"] for r in rows),
        "active_fraction": statistics.mean(int(r["active"]) for r in rows),
        "accepted_papers_per_founder": statistics.mean(r["accepted_papers"] for r in rows),
        "citations_all_papers_per_founder": statistics.mean(r["citations_all_papers"] for r in rows),
    }


def reconcile_year(record, expected_founders=None):
    """Reject changed denominators or stored summaries inconsistent with raw rows."""
    rows = record["agent_rows"]
    ids = [r["agent_id"] for r in rows]
    if len(ids) != len(set(ids)) or (expected_founders is not None
                                   and set(ids) != set(expected_founders)):
        raise ValueError("Founder identities changed or were duplicated")
    for row in rows:
        if type(row["active"]) is not bool:
            raise ValueError("Active status must be boolean")
        for key in ("spendable_resources", "cumulative_earned_funding", "accepted_papers",
                    "submitted_papers", "citations_all_papers"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid founder outcome: {key}")
        if row["accepted_papers"] > row["submitted_papers"]:
            raise ValueError("Accepted papers exceed submitted papers")
    if record["metrics"] != summarize_agents(rows):
        raise ValueError("Stored metrics do not reconcile with founder rows")
    if record["legacy_zero_boundary"] != legacy_zero_boundary(rows):
        raise ValueError("Stored zero-boundary audit does not reconcile with founder rows")
    return ids


def effective_args(args):
    """Exclude only intervention, seed, endpoint and artifact-location fields."""
    excluded = {
        "james_mechanisms", "seed", "experiment_id", "vllm_url", "docs_dir", "log_dir",
        "checkpoint_dir", "visual_dir", "output_dir", "data_cache_dir",
    }
    return {k: v for k, v in vars(args).items() if k not in excluded}


def paired_contrasts(values):
    """Effects of disabling each mechanism, plus the difference in differences."""
    if set(values) != set(CELLS):
        raise ValueError("A paired seed must contain all four cells")
    if not all(v is None or (isinstance(v, (int, float)) and math.isfinite(v))
               for v in values.values()):
        raise ValueError("Outcomes must be finite numbers or explicitly undefined")
    coefficients = {
        "disable_publication_at_F1": {"P0F1": 1, "P1F1": -1},
        "disable_publication_at_F0": {"P0F0": 1, "P1F0": -1},
        "disable_resources_at_P1": {"P1F0": 1, "P1F1": -1},
        "disable_resources_at_P0": {"P0F0": 1, "P0F1": -1},
        "interaction": {"P0F0": 1, "P0F1": -1, "P1F0": -1, "P1F1": 1},
    }
    return {name: (sum(values[cell] * weight for cell, weight in weights.items())
                   if all(values[cell] is not None for cell in weights) else None)
            for name, weights in coefficients.items()}


def experiment_id(cell, seed):
    Mechanisms.from_cell(cell)
    if seed not in SEEDS:
        raise ValueError(f"Protocol seeds were fixed upfront: {SEEDS}")
    return f"james_mechanisms_v6_n1200_y6_{cell}_seed{seed}"


def plan_commands(url, python="python"):
    return [
        ["env", f"PYTHONHASHSEED={seed}", "WANDB_MODE=disabled", python,
         "src/james_mechanisms.py", "run", "--cell", cell, "--seed", str(seed),
         "--vllm-url", url, "--server-manifest",
         f"outputs/logs/ec2_preparation/servers/{cell}.json"]
        for seed in SEEDS for cell in CELLS
    ]


def queue_jobs(python=sys.executable):
    """Declarative queue handoff. This function starts no processes."""
    jobs = []
    for cell in CELLS:
        exp_id = experiment_id(cell, 42)
        roots = {category: str((ROOT / "outputs" / category / exp_id).resolve())
                 for category in ("docs", "logs", "checkpoints", "visual")}
        manifest = str(Path(roots["docs"]) / "mechanism_manifest.json")
        checks = [
            {"path": manifest, "json_pointer": f"/{key}", "equals": value}
            for key, value in {"status": "complete", "protocol": PROTOCOL, "seed": 42,
                               "cell": cell, "founder_count": FOUNDER_COUNT, "num_years": YEARS}.items()
        ]
        checks.extend([
            {"path": manifest, "json_pointer": "/funding_baseline", "equals": FUNDING_BASELINE},
            {"path": manifest, "json_pointer": "/funding_selection_protocol",
             "equals": FUNDING_SELECTION_PROTOCOL},
            {"path": manifest, "json_pointer": "/sequential_source_binding",
             "equals": sequential_source_binding()},
            {"path": manifest, "json_pointer": "/sequential_audit_valid", "equals": True},
            {"path": manifest, "json_pointer": "/funding_application_evidence_valid", "equals": True},
            {"path": manifest, "json_pointer": "/args/funding_panel_max_apps", "equals": 25},
            {"path": manifest, "json_pointer": "/args/funding_budget_mode", "equals": "track"},
            {"path": manifest, "json_pointer": "/args/log_funding_applications", "equals": True},
            {"path": manifest, "json_pointer": "/funding_validation/installed", "equals": False},
            {"path": manifest, "json_pointer": "/funding_validation/failed_batches", "equals": 0},
            {"path": manifest, "json_pointer": "/funding_validation/audit_failures", "equals": 0},
            {"path": manifest, "json_pointer": "/funding_validation/invalid_final_panels", "equals": 0},
            {"path": manifest, "json_pointer": "/funding_validation/restore_conflicts", "equals": []},
            {"path": manifest, "json_pointer": "/funding_validation/completion_allowed", "equals": True},
            {"path": manifest, "json_pointer": "/funding_validation/output_representation",
             "equals": FUNDING_OUTPUT_REPRESENTATION},
            {"path": manifest, "json_pointer": "/request_audit_valid", "equals": True},
        ])
        artifacts = [
            {"path": str(Path(roots["docs"]) / "llm_request_audit.jsonl"), "min_bytes": 1},
            {"path": str(Path(roots["docs"]) / "llm_request_audit.summary.json"), "min_bytes": 1,
             "equals": {"/status": "complete", "/n_clients": 1,
                        "/guard_failures": 0, "/pending_requests": 0}},
            # Zero-panel runs still require both raw files and nonempty summaries.
            {"path": str(Path(roots["docs"]) / "funding_validation.jsonl"), "min_bytes": 0},
            {"path": str(Path(roots["docs"]) / "funding_validation.summary.json"), "min_bytes": 1,
             "equals": {"/completion_allowed": True, "/installed": False,
                        "/failed_batches": 0, "/audit_failures": 0,
                        "/invalid_final_panels": 0, "/restore_conflicts": [],
                        "/output_representation": FUNDING_OUTPUT_REPRESENTATION}},
            {"path": str(Path(roots["docs"]) / SEQUENTIAL_AUDIT_FILE), "min_bytes": 1},
            {"path": str(Path(roots["docs"]) / SEQUENTIAL_SUMMARY_FILE), "min_bytes": 1,
             "equals": {**{"/" + key: value for key, value in sequential_summary_gates().items()},
                        "/years_started": YEARS, "/years_completed": YEARS,
                        "/agency_registrations": 2 * YEARS}},
        ]
        checks.extend(
            {"path": manifest, "json_pointer": f"/funding_sequential/{key}", "equals": value}
            for key, value in sequential_summary_gates().items())
        checks.extend(
            {"path": manifest, "json_pointer": f"/funding_sequential/{key}", "equals": value}
            for key, value in {"years_started": YEARS, "years_completed": YEARS,
                               "agency_registrations": 2 * YEARS}.items())
        artifacts.append({
            "path": str(Path(roots["docs"]) / "run_manifest.json"), "min_bytes": 1,
            "equals": {"/status": "complete", "/years_completed": YEARS}})
        for year in range(1, YEARS + 1):
            path = str(Path(roots["docs"]) / f"mechanisms_year_{year}.json")
            checks.extend([
                {"path": path, "json_pointer": "/year", "equals": year},
                {"path": path, "json_pointer": "/cell", "equals": cell},
                {"path": path, "json_pointer": "/metrics/founder_count", "equals": FOUNDER_COUNT},
            ])
        jobs.append({
            "id": exp_id, "cell": cell,
            "command": [python, str(ROOT / "src/james_mechanisms.py"), "run",
                        "--cell", cell, "--seed", "42", "--vllm-url", "__SERVER_URL__",
                        "--server-manifest", "__SERVER_MANIFEST__"],
            "environment": {"PYTHONHASHSEED": "42", "WANDB_MODE": "disabled"},
            "working_directory": str(ROOT), "output_roots": list(roots.values()),
            "artifact_checks": checks, "artifacts": artifacts,
        })
    return {"protocol": PROTOCOL, "jobs": jobs, "independent_worlds": 1,
            "launch_owner": "Main queue; this plan is not permission to skip preflight/capacity gates."}


def server_provenance(path, endpoint, *, runtime_profile=None):
    """Validate the queue's explicit attestation; endpoint ID alone is insufficient."""
    record = json.loads(Path(path).read_text())
    runtime_hash = validate_server_record(record, endpoint, runtime_profile=runtime_profile)
    result = {"record": record, "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              "runtime_hash": runtime_hash}
    if runtime_profile is not None:
        result["manifest_path"] = str(Path(path).resolve())
    return result


def validate_server_record(record, endpoint, *, runtime_profile=None):
    """Apply the same explicit runtime contract at launch and file admission."""
    expected = {
        "model": MODEL, "revision": MODEL_REVISION, "dtype": "bfloat16",
        "tensor_parallel_size": 2, "seed": 42, "request_seed_base": 42,
        "request_seed_policy": "derive_seed(run_seed, phase_context, item_index, retry)",
        "vllm_version": "0.12.0", "transformers_version": "4.57.3",
        "torch_version": "2.9.0+cu128",
        "max_model_len": 32768, "gpu_memory_utilization": 0.8,
        "reasoning_parser": "qwen3", "max_num_seqs": 128,
    }
    if runtime_profile is not None:
        from james_server_profiles import TP1_PLAN_PROFILE, TP1_PROFILE_NAME, tp1_server_spec
        if runtime_profile != TP1_PROFILE_NAME:
            raise ValueError("Unsupported explicit server runtime profile")
        expected = tp1_server_spec()
        if record.get("server_profile") != TP1_PLAN_PROFILE:
            raise ValueError("TP1 server provenance lacks the prospective queue profile")
    for key, value in expected.items():
        if record.get(key) != value or (
                runtime_profile is not None and type(record.get(key)) is not type(value)):
            raise ValueError(f"Server provenance mismatch: {key}, expected {value!r}")
    if record.get("endpoint", "").rstrip("/") != endpoint.rstrip("/"):
        raise ValueError("Server provenance endpoint mismatch")
    gpu_ids = record.get("gpu_ids", record.get("server_gpus", []))
    gpu_count = expected["tensor_parallel_size"]
    if (len(gpu_ids) != gpu_count or len(set(gpu_ids)) != gpu_count
            or not all(record.get(k) for k in ("host", "pid", "launch_command", "started_utc"))):
        raise ValueError("Server provenance lacks actual launch identity or the required GPU count")
    return digest(expected)


def configure_run(options):
    """Use original defaults and argument validation without editing the parser."""
    from utopia.arguments import parse_arguments

    exp_id = experiment_id(options.cell, options.seed)
    if os.environ.get("PYTHONHASHSEED") != str(options.seed):
        raise ValueError(f"Start Python with PYTHONHASHSEED={options.seed}")
    # parse_arguments creates directories. Resolve its stage defaults inside a
    # temporary output tree, then give this independent experiment canonical IDs.
    with tempfile.TemporaryDirectory(prefix="james-args-") as temporary:
        args = parse_arguments([
            "--experiment_name", "default_james_mechanisms",
            "--experiment_stage", "mechanism", "--output-dir", temporary,
            "--seed", str(options.seed), "--population_mode", "university_only",
            "--num_institutions", "240", "--researchers_per_institution", "5",
            "--strategy_mix", "balanced", "--num_years", "6", "--num_conferences", "6",
            "--model", MODEL, "--vllm_url", options.vllm_url,
            "--funding_panel_max_apps", str(FUNDING_PANEL_MAX_APPS),
            "--funding_budget_mode", "track",
            "--log_funding_applications", "--always_rerun",
        ])
    args.experiment_id = exp_id
    args.james_mechanisms = asdict(Mechanisms.from_cell(options.cell))
    for attribute, category in (
        ("docs_dir", "docs"), ("log_dir", "logs"),
        ("checkpoint_dir", "checkpoints"), ("visual_dir", "visual"),
    ):
        path = ROOT / "outputs" / category / exp_id
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Fresh run required, existing artifacts: {path}")
        setattr(args, attribute, str(path))
    args.output_dir = args.checkpoint_dir
    args.rag_device = "cpu"
    args.data_cache_dir = str(ROOT / "data" / "cache" / exp_id)
    for path in (args.docs_dir, args.log_dir, args.checkpoint_dir,
                 args.visual_dir, args.data_cache_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
    return args


def fallback_audit(simulation):
    rows = []
    missing = []
    for year in range(1, simulation.num_years + 1):
        path = Path(simulation.output_dir) / f"funding_applications_year_{year}.jsonl"
        if path.exists():
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        else:
            missing.append(year)
    return {
        "funding_ranked_applications": len(rows),
        "funding_fallback_rows": sum(bool(r.get("fallback_ranking")) for r in rows),
        "funding_imputed_rows": sum(bool(r.get("imputed_tail")) for r in rows),
        "funding_log_absent_years": missing,
        "direction_fallback_stats": {
            str(k): v for k, v in simulation._direction_fallback_stats.items()
        },
        "llm_call_stats": simulation.llm.call_stats,
        "scope": (
            "Existing ranking/direction diagnostics, not a complete audit of every retry. "
            "An absent funding log can mean no applications and must not be read as "
            "a demonstrated zero fallback count."
        ),
    }


def validate_request_audit_summary(summary):
    """A successful scope must include requests from exactly one guarded client."""
    expected = {"status": "complete", "n_clients": 1,
                "guard_failures": 0, "pending_requests": 0}
    if (any(summary.get(key) != value for key, value in expected.items())
            or type(summary.get("n_requests")) is not int or summary["n_requests"] <= 0):
        raise RuntimeError("Request audit does not establish complete guarded execution")


def run(options):
    os.chdir(ROOT)
    args = configure_run(options)
    path = Path(args.docs_dir) / "mechanism_manifest.json"
    from utopia.config import SIMULATION_CONFIG
    manifest = {
        "protocol": PROTOCOL, "cell": options.cell, "seed": options.seed,
        "status": "running", "command": shlex.join(sys.argv),
        "model": MODEL, "vllm_url": options.vllm_url,
        "source_hash": source_fingerprint(),
        "scientific_config_hash": digest(SIMULATION_CONFIG),
        "effective_args_hash": digest(effective_args(args)),
        "args": vars(args), "num_years": YEARS, "founder_count": FOUNDER_COUNT,
        "funding_baseline": FUNDING_BASELINE,
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "sequential_source_binding": sequential_source_binding(),
        "funding_compact_sha256": file_hash(ROOT / "src/james_funding_compact.py"),
        "inference_provenance": (
            "Model ID checked at endpoint. Main agent must record server revision, "
            "precision, launch command and GPU allocation separately."
        ),
    }
    write_json(path, manifest)
    try:
        manifest["server_provenance"] = server_provenance(
            options.server_manifest, options.vllm_url)
        manifest["cache_provenance"] = verified_cache_manifest()
        from run_simulation import Simulation, VLLMServerModel, FundingAgency, set_seed
        from utopia.models.request_audit import request_audit_scope
        from james_funding_validation import install_funding_validation, FundingRankingValidationError
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.FileHandler(Path(args.log_dir) / "simulation.log"),
                      logging.StreamHandler()],
        )
        set_seed(args.seed)
        funding_handle = install_funding_validation(
            FundingAgency, Path(args.docs_dir) / "funding_validation.jsonl", compact=True)
        audit = sequential = None
        try:
            with sequential_funding_scope(
                FundingAgency, VLLMServerModel, args.docs_dir
            ) as sequential, funding_phase_coverage(
                Simulation, FundingAgency, sequential
            ), request_audit_scope(
                Path(args.docs_dir) / "llm_request_audit.jsonl", model_name=MODEL
            ) as audit:
                llm = VLLMServerModel(
                    model_name=MODEL, base_url=args.vllm_url,
                    max_concurrent_requests=args.batch_size, run_seed=args.seed,
                )
                if MODEL not in {entry.id for entry in llm.client.models.list().data}:
                    raise RuntimeError(f"Endpoint does not advertise required model {MODEL}")
                sim = simulation_class(Simulation)(
                    llm=llm, args=args, mechanisms=Mechanisms.from_cell(options.cell),
                    num_years=args.num_years, output_dir=args.output_dir,
                    **{key: getattr(args, key) for key in (
                        "verbose", "debug", "industry_funding_mode", "funding_allocation_mode",
                        "always_rerun", "use_langchain", "max_retries", "weighted_funding_assignment",
                        "experiment_name", "collaboration_mode", "collaboration_network_growth_mode",
                        "start_year",
                    )},
                )
                sim.james_cache_provenance = manifest["cache_provenance"]
                sim.run()
                if (len(sim.james_founders) != FOUNDER_COUNT
                        or set(sim.james_years) != set(range(1, YEARS + 1))):
                    raise RuntimeError("Run did not satisfy the six-year, 1200-founder protocol")
                if manifest["cache_provenance"]["input_files"] != cache_file_identity():
                    raise RuntimeError("Shared input cache changed during simulation")
                if (not funding_handle.summary()["completion_allowed"]
                        or funding_handle.summary().get("output_representation")
                        != FUNDING_OUTPUT_REPRESENTATION):
                    raise FundingRankingValidationError(funding_handle.summary())
        finally:
            try:
                funding_handle.restore()
            finally:
                manifest["funding_validation"] = funding_handle.summary()
                write_json(Path(args.docs_dir) / "funding_validation.summary.json",
                           manifest["funding_validation"])
                if audit is not None:
                    manifest["request_audit"] = audit.summary()
                if sequential is not None:
                    manifest["funding_sequential"] = sequential.summary()
        validate_request_audit_summary(manifest["request_audit"])
        native = json.loads((Path(args.docs_dir) / "run_manifest.json").read_text())
        if native.get("status") != "complete" or native.get("years_completed") != YEARS:
            raise ValueError("Native simulator did not complete every funding year")
        validate_sequential_evidence(
            args.docs_dir, manifest["funding_validation"],
            manifest_summary=manifest["funding_sequential"],
            source_binding=manifest["sequential_source_binding"],
            application_dir=args.output_dir)
        manifest["funding_sequential_audits"] = {
            name: file_hash(Path(args.docs_dir) / name)
            for name in (SEQUENTIAL_AUDIT_FILE, SEQUENTIAL_SUMMARY_FILE)}
        manifest["funding_application_evidence"] = funding_ledger_evidence(args.output_dir, YEARS)
        manifest.update(
            status="complete", initial_world_hash=sim.james_initial_world_hash,
            corpus_hash=sim.james_corpus_hash, fallback_audit=fallback_audit(sim),
            founder_ids=list(sim.james_founders), request_audit_valid=True,
            sequential_audit_valid=True,
            funding_application_evidence_valid=True,
        )
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        write_json(path, manifest)
        mark_failed_run_manifest(args.docs_dir, PROTOCOL, manifest["error"])
        raise
    write_json(path, manifest)


def analyze(docs_root=ROOT / "outputs/docs"):
    """Four conditional contrasts within one world; no sampling uncertainty claim."""
    runs = {}
    boundary = {}
    for seed in SEEDS:
        for cell in CELLS:
            directory = Path(docs_root) / experiment_id(cell, seed)
            manifest = json.loads((directory / "mechanism_manifest.json").read_text())
            native = json.loads((directory / "run_manifest.json").read_text())
            if native.get("status") != "complete" or native.get("years_completed") != YEARS:
                raise ValueError("Native simulator manifest is incomplete")
            if (manifest["status"] != "complete" or manifest["protocol"] != PROTOCOL
                    or manifest["cell"] != cell or manifest["seed"] != seed):
                raise ValueError(f"Incomplete or misidentified run: {directory}")
            if manifest.get("funding_baseline") != FUNDING_BASELINE:
                raise ValueError(f"Panelized funding baseline mismatch: {directory}")
            if (manifest.get("funding_selection_protocol") != FUNDING_SELECTION_PROTOCOL
                    or manifest.get("sequential_audit_valid") is not True
                    or set(manifest.get("sequential_source_binding", {}))
                    != {"src/james_funding_sequential.py", SEQUENTIAL_PROTOCOL_PATH}):
                raise ValueError(f"Missing sequential funding protocol binding: {directory}")
            summary = json.loads((directory / "llm_request_audit.summary.json").read_text())
            funding_summary = json.loads((directory / "funding_validation.summary.json").read_text())
            validate_request_audit_summary(summary)
            if (manifest.get("request_audit") != summary
                    or manifest.get("request_audit_valid") is not True
                    or manifest.get("funding_validation", {}).get("completion_allowed") is not True
                    or funding_summary.get("output_representation") != FUNDING_OUTPUT_REPRESENTATION
                    or manifest.get("funding_validation") != funding_summary
                    or not (directory / "llm_request_audit.jsonl").stat().st_size
                    or not (directory / "funding_validation.jsonl").is_file()):
                raise ValueError(f"Required request/funding audit mismatch: {directory}")
            validate_sequential_evidence(
                directory, funding_summary, manifest_summary=manifest["funding_sequential"],
                source_binding=manifest["sequential_source_binding"],
                application_dir=manifest["args"]["output_dir"])
            if (manifest.get("funding_application_evidence_valid") is not True
                    or manifest.get("funding_application_evidence")
                    != funding_ledger_evidence(manifest["args"]["output_dir"], YEARS)):
                raise ValueError("Funding application bytes/absent-year evidence differs from manifest")
            if manifest.get("funding_sequential_audits") != {
                    name: file_hash(directory / name)
                    for name in (SEQUENTIAL_AUDIT_FILE, SEQUENTIAL_SUMMARY_FILE)}:
                raise ValueError("Sequential raw/summary evidence differs from manifest")
            years = {}
            founders = manifest["founder_ids"]
            if len(founders) != FOUNDER_COUNT or len(set(founders)) != FOUNDER_COUNT:
                raise ValueError("Manifest founder cohort is not the fixed population")
            boundary_years = {}
            affected = set()
            for year in range(1, 7):
                record = json.loads((directory / f"mechanisms_year_{year}.json").read_text())
                if (record["year"] != year or record["cell"] != cell
                        or record["metrics"]["founder_count"] != FOUNDER_COUNT):
                    raise ValueError(f"Year or founder cohort mismatch: {directory}")
                reconcile_year(record, founders)
                years[year] = record["metrics"]
                audit = record["legacy_zero_boundary"]
                boundary_years[year] = audit
                affected.update(audit["active_zero_agent_ids"])
            count = sum(a["active_zero_count"] for a in boundary_years.values())
            boundary[cell] = {
                "annual_snapshots": boundary_years,
                "active_zero_founder_year_snapshots": count,
                "founder_year_denominator": FOUNDER_COUNT * YEARS,
                "active_zero_fraction_of_founder_years": count / (FOUNDER_COUNT * YEARS),
                "unique_affected_founders": len(affected),
                "scope": "Snapshot prevalence; not the number of exact-zero debit events.",
            }
            runs[seed, cell] = (manifest, years)
    for seed in SEEDS:
        for key in ("initial_world_hash", "corpus_hash"):
            hashes = [runs[seed, cell][0].get(key) for cell in CELLS]
            if not all(hashes) or len(set(hashes)) != 1:
                raise ValueError(f"Paired {key} mismatch for seed {seed}")
    for key in ("source_hash", "scientific_config_hash", "effective_args_hash", "model",
                "sequential_source_binding"):
        if len({digest(r[0][key]) for r in runs.values()}) != 1:
            raise ValueError(f"Protocol mismatch: {key}")
    if len({tuple(m["founder_ids"]) for m, _ in runs.values()}) != 1:
        raise ValueError("Founder identities differ across cells")
    runtime_hashes = {m["server_provenance"]["runtime_hash"] for m, _ in runs.values()}
    if len(runtime_hashes) != 1:
        raise ValueError("Server runtime differs across cells")
    contrasts = {
        metric: paired_contrasts({cell: runs[42, cell][1][6][metric] for cell in CELLS})
        for metric in PRIMARY_METRICS
    }
    output = {
        "protocol": PROTOCOL, "year6_paired_contrasts": contrasts,
        "shared_seed": 42, "independent_worlds": 1,
        "world_standard_error": None, "world_confidence_interval": None,
        "funding_baseline": FUNDING_BASELINE,
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "sequential_funding_audits": {
            cell: {"summary": runs[42, cell][0]["funding_sequential"],
                   "sha256": runs[42, cell][0]["funding_sequential_audits"]}
            for cell in CELLS},
        "funding_application_evidence": {
            cell: runs[42, cell][0]["funding_application_evidence"] for cell in CELLS},
        "legacy_zero_boundary": boundary,
        "annual_metrics": {f"{s}/{c}": years for (s, c), (_, years) in runs.items()},
        "fallback_audits": {f"{s}/{c}": m["fallback_audit"]
                            for (s, c), (m, _) in runs.items()},
        "interpretation": (
            "Conditional four-cell contrasts for one initialized world (seed 42). "
            "No seed SD, confidence interval, population-level inference or equivalence claim. "
            "Researchers are not independent replicates. Record-display ablation and "
            "award-credit removal do not decompose all feedback. All contrasts are "
            "conditional on stock cap25 panel competition and per-panel quota rounding, "
            "not global program ranking. The citation measure is "
            "six-year attributed citation yield with full author counting, not age-standardized. "
            "Inspect fallback and exact-zero audits. All-zero Ginis remain undefined."
        ),
    }
    write_json(Path(docs_root) / "james_mechanisms_v6_analysis" / "paired_effects.json", output)
    return output


def main(argv=None):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--vllm-url", required=True)
        if name == "run":
            command.add_argument("--cell", choices=CELLS, default="P1F1")
            command.add_argument("--seed", type=int, choices=SEEDS, required=True)
            command.add_argument("--server-manifest", required=True)
    commands.add_parser("analyze")
    commands.add_parser("queue-plan", help="Print declarative four-job plan without executing it")
    commands.add_parser("audit-cache", help="CPU-only provenance check of immutable original caches")
    options = parser.parse_args(argv)
    if options.command == "plan":
        for command in plan_commands(options.vllm_url):
            print(shlex.join(command))
    elif options.command == "run":
        run(options)
    elif options.command == "audit-cache":
        print(json.dumps(audit_cache(), indent=2))
    elif options.command == "queue-plan":
        print(json.dumps(queue_jobs(), indent=2))
    else:
        print(json.dumps(analyze(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
