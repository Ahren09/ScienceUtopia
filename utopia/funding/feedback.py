"""Funding feedback interventions and their accounting evidence."""

from __future__ import annotations

from utopia.runtime.historical import source_binding_value, trusted_audit_sources

from utopia.utils.data_utils import file_sha256
from utopia.utils.data_utils import file_sha256 as file_hash
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
import json
import math
from pathlib import Path
import statistics
from utopia.runtime.provenance import ROOT, corpus_fingerprint, digest, write_json


SEEDS = (42,)


FOUNDER_COUNT = 1200


YEARS = 6


FUNDING_PANEL_MAX_APPS = 25


FUNDING_OUTPUT_REPRESENTATION = "ordered_application_ids_v1"


FUNDING_SELECTION_PROTOCOL = "sequential_remaining_ids_v1"


SEQUENTIAL_AUDIT_FILE = "funding_sequential.jsonl"


SEQUENTIAL_SUMMARY_FILE = "funding_sequential.summary.json"


SEQUENTIAL_PROTOCOL_PATH = "docs/sequential_funding_protocol.md"


FROZEN_FUNDING_PROGRAM_RATES = {
    "NSF_THEORY": 0.23,
    "NSF_AI_FOUNDATIONS": 0.23,
    "NSF_SYSTEM": 0.23,
    "DARPA_AUTONOMOUS": 0.10,
    "DARPA_SECURITY": 0.10,
    "DARPA_AI_APPS": 0.10,
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


PROTOCOL = "funding-feedback-v6-single-world"


CELLS = ("P1F1", "P0F1", "P1F0", "P0F0")


PRIMARY_METRICS = (
    "cumulative_earned_funding_gini",
    "spendable_resources_gini",
    "mean_spendable_resources",
    "active_fraction",
    "accepted_papers_per_founder",
    "citations_all_papers_per_founder",
)


@dataclass(frozen=True)
class Mechanisms:
    publication_record_visible: bool = True
    awards_feed_resources: bool = True

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise TypeError("Mechanism switches must be booleans")

    @property
    def cell(self):
        return (
            f"P{int(self.publication_record_visible)}F{int(self.awards_feed_resources)}"
        )

    @classmethod
    def from_cell(cls, cell):
        if cell not in CELLS:
            raise ValueError(f"Unknown cell: {cell}")
        return cls(cell[1] == "1", cell[3] == "1")


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
        empty_record,
        "Past Performance:\nPublication record withheld by the experiment.",
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
        if entries[: len(prefix)] != prefix:
            raise RuntimeError("Funding phase rewrote existing award history")
        for entry in entries[len(prefix) :]:
            amount = entry["amount"]
            if (
                entry["year"] != year
                or not isinstance(amount, (int, float))
                or not math.isfinite(amount)
                or amount < 0
            ):
                raise RuntimeError("Invalid newly earned award")
            records.append(
                {
                    "year": year,
                    "agent_id": agent.id,
                    "program_id": program,
                    "earned_amount": float(amount),
                }
            )
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

        return {
            "topk_indices": np.empty((0, 0), dtype=np.int64),
            "similarity_scores": np.empty((0, 0), dtype=np.float32),
        }
    return original(queries, *args, **kwargs)


def legacy_zero_boundary(rows):
    """Observational year-end prevalence, not a count of resource debit events."""
    if not rows or len({r["agent_id"] for r in rows}) != len(rows):
        raise ValueError("Boundary audit requires a nonempty unique cohort")
    active = [r for r in rows if r["active"]]
    zero = [r for r in rows if r["spendable_resources"] == 0]
    active_zero = [r for r in zero if r["active"]]
    return {
        "cohort_count": len(rows),
        "active_count": len(active),
        "zero_resource_count": len(zero),
        "active_zero_count": len(active_zero),
        "active_zero_fraction_of_cohort": len(active_zero) / len(rows),
        "active_zero_fraction_of_active": len(active_zero) / len(active)
        if active
        else None,
        "active_zero_agent_ids": sorted(r["agent_id"] for r in active_zero),
        "measurement": "End of inherited phase 5; exact equality, no tolerance or mechanics change.",
    }


def source_fingerprint(root=ROOT):
    paths = [
        root / "utopia/simulation.py",
        root / "utopia/constants.py",
        root / "utopia/experiments/funding_feedback.py",
        root / "utopia/funding/validation.py",
        root / "utopia/funding/compact.py",
        root / "utopia/funding/sequential.py",
        root / SEQUENTIAL_PROTOCOL_PATH,
    ]
    paths += sorted((root / "utopia").rglob("*.py"))
    return digest({str(p.relative_to(root)): file_sha256(p) for p in paths})


def sequential_source_binding(root=ROOT):
    """The executable helper and prospective protocol must both exist at freeze."""
    return {
        name: file_hash(Path(root) / name)
        for name in ("utopia/funding/sequential.py", SEQUENTIAL_PROTOCOL_PATH)
    }


def sequential_summary_gates():
    # n_clients may be zero only when the helper proves there were zero panels.
    # validate_summary/validate_audit enforce that relationship, not this projection.
    return {
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "output_representation": FUNDING_OUTPUT_REPRESENTATION,
        "installed": False,
        "status": "complete",
        "completion_allowed": True,
        "failed_panels": 0,
        "cancelled_panels": 0,
        "unprocessed_panels": 0,
        "fatal_errors": 0,
        "guard_failures": 0,
        "processing_failures": 0,
        "audit_failures": 0,
        "restore_conflicts": [],
    }


@contextmanager
def sequential_funding_scope(FundingAgency, VLLMServerModel, directory):
    """Nest inside compact validation; always restore this layer first."""
    from utopia.funding.sequential import install_sequential_funding, validate_summary

    directory = Path(directory)
    handle = install_sequential_funding(
        FundingAgency, VLLMServerModel, directory / SEQUENTIAL_AUDIT_FILE
    )
    try:
        yield handle
        validate_summary(handle.summary(), require_restored=False)
    finally:
        handle.restore()  # The helper exclusively writes/finalizes its own summary.
    validate_summary(handle.summary())
    if (
        json.loads((directory / SEQUENTIAL_SUMMARY_FILE).read_text())
        != handle.summary()
    ):
        raise ValueError("Sequential helper summary differs from persisted evidence")


@contextmanager
def funding_phase_coverage(simulation_class, funding_class, handle):
    """Observe the real phase boundary, including years with zero applications."""
    name = "_run_phase_5_update_funding"
    original = getattr(simulation_class, name)

    @wraps(original)
    def phase(simulation, year, *args, **kwargs):
        agencies = sorted(
            agent.id
            for agent in simulation.ecosystem.agent_population.values()
            if isinstance(agent, funding_class)
        )
        handle.begin_year(year, agencies)
        result = original(simulation, year, *args, **kwargs)
        handle.end_year(year)
        return result

    with instance_override(simulation_class, name, phase):
        yield


def validate_sequential_evidence(
    directory,
    compact_summary,
    *,
    manifest_summary=None,
    source_binding=None,
    application_dir=None,
    num_years=YEARS,
    expected_agency_ids=("NSF", "DARPA"),
    program_rates=FROZEN_FUNDING_PROGRAM_RATES,
    trusted_source_files=None,
):
    """Reconstruct choices/processing from raw evidence, including true empty runs."""
    from utopia.funding.sequential import validate_audit, validate_summary

    directory = Path(directory)
    summary = json.loads((directory / SEQUENTIAL_SUMMARY_FILE).read_text())
    validate_summary(summary)
    expected_sources = None
    if trusted_source_files is not None:
        expected_sources = trusted_audit_sources(
            summary["source_files_sha256"], trusted_source_files
        )
    report = validate_audit(
        directory / SEQUENTIAL_AUDIT_FILE,
        summary,
        request_audit_path=directory / "llm_request_audit.jsonl",
        expected_sources=expected_sources,
    )
    if (
        [row["year"] for row in report["years"]] != list(range(1, num_years + 1))
        or summary["years_started"] != num_years
        or summary["years_completed"] != num_years
        or any(
            set(row["expected_agency_ids"]) != set(expected_agency_ids)
            for row in report["years"]
        )
    ):
        raise ValueError(
            "Sequential coverage omits or changes a native funding year/agency"
        )
    joined = report["request_audit"]
    if (
        Path(joined["path"]).resolve()
        != (directory / "llm_request_audit.jsonl").resolve()
        or joined["matched_sdk_calls"] != summary["sdk_calls"]
        or joined["sha256"] != file_hash(directory / "llm_request_audit.jsonl")
    ):
        raise ValueError(
            "Sequential SDK evidence is not bound to this run's request audit"
        )
    if manifest_summary is not None and summary != manifest_summary:
        raise ValueError("Sequential funding manifest/summary mismatch")
    if summary["processed_panels"] != compact_summary.get("final_panels"):
        raise ValueError("Sequential/compact processed-panel coverage mismatch")
    if source_binding is not None and (
        summary["source_sha256"]
        != source_binding_value(source_binding, "utopia/funding/sequential.py")
        or any(
            summary["source_files_sha256"].get(key) != value
            for key, value in source_binding.items()
        )
    ):
        raise ValueError("Sequential funding source fingerprint mismatch")
    # The native ledger omits application_id, but retains every position and
    # llm_rank. Joining by rank preserves repeated applicants without collapsing
    # their distinct model-selected applications.
    expected, panels = {}, {}
    for panel in report["processed_panels"]:
        identity = (panel["year"], panel["program_id"], panel["panel_index"])
        order = panel["ranked_application_ids"]
        if (
            identity in panels
            or type(identity[0]) is not int
            or not 1 <= identity[0] <= num_years
        ):
            raise ValueError("Sequential panel year or identity is invalid")
        panels[identity] = len(order)
        for rank, app_id in enumerate(order, 1):
            expected[identity + (rank,)] = panel["application_to_applicant"][
                str(app_id)
            ]
    compact_path = directory / (
        "funding_validation.jsonl"
        if (directory / "funding_validation.jsonl").exists()
        else "funding_validation_audit.jsonl"
    )
    compact_panels = Counter()
    for line in compact_path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "final_panel":
            if (
                row.get("valid") is not True
                or row.get("n_expected") != row.get("n_valid")
                or row.get("n_expected") != row.get("n_returned")
                or any(
                    row.get(k) != 0
                    for k in ("n_missing", "n_rejected", "n_imputed", "n_fallback")
                )
            ):
                raise ValueError("Compact final-panel evidence is invalid")
            compact_panels[
                row["program_id"], row["panel_index"], row["n_expected"]
            ] += 1
    if compact_panels != Counter(
        (program, index, n) for (_, program, index), n in panels.items()
    ):
        raise ValueError("Compact final panels differ from sequential processing trace")
    observed = {}
    application_dir = (
        Path(application_dir) if application_dir is not None else directory
    )
    for year in range(1, num_years + 1):
        log = application_dir / f"funding_applications_year_{year}.jsonl"
        if not log.exists():
            continue
        for line in log.read_text().splitlines():
            row = json.loads(line)
            identity = (row["year"], row["program_id"], row["panel_index"])
            rank = row["llm_rank"]
            key = identity + (rank,)
            if (
                identity not in panels
                or row["year"] != year
                or type(rank) is not int
                or key in observed
                or key not in expected
                or row["n_panel"] != panels[identity]
                or row["position"] != rank
                or row["applicant_id"] != expected[key]
                or row["program_id"] not in program_rates
                or row["funding_rate"] != program_rates[row["program_id"]]
                or row.get("fallback_ranking") is not False
                or row.get("imputed_tail") is not False
                or row["num_winners"]
                != max(1, int(row["n_panel"] * row["funding_rate"]))
                or row["funded"] is not (rank <= row["num_winners"])
            ):
                raise ValueError(
                    "Native funding ledger differs from sequential model-selected ranking"
                )
            observed[key] = row["applicant_id"]
    if observed != expected:
        raise ValueError(
            "Native funding ledger does not cover every sequential selection"
        )
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


class FundingFeedbackMixin:
    """Put before Simulation in the MRO. All simulation phases remain inherited."""

    def __init__(self, *args, mechanisms=Mechanisms(), **kwargs):
        self.mechanisms = mechanisms
        self.funding_feedback_years = {}
        self.initial_world = None
        self.corpus_hash = None
        super().__init__(*args, **kwargs)

    def load_checkpoint(self, year):
        raise RuntimeError(
            "Funding feedback mechanism runs require fresh worlds, no checkpoint resume"
        )

    def _setup_year(self, year):
        if year == 1:
            # to_dict is the simulator's existing state serializer, without LLMs.
            self.initial_world = {
                "ecosystem": self.ecosystem.to_dict(),
                "conferences": self.conference_system.to_dict(),
            }
            founders = [
                a
                for a in self.ecosystem.agent_population.values()
                if a.get_type() == "university"
            ]
            if (
                hasattr(self.args, "funding_feedback")
                and len(founders) != self.args.num_institutions * self.args.researchers_per_institution
            ):
                raise RuntimeError(
                    "Founder population differs from the configured university population"
                )
            self.founder_ids = tuple(a.id for a in founders)
            self.initial_world_hash = digest(self.initial_world)
            write_json(
                Path(self.args.docs_dir) / "initial_world.json", self.initial_world
            )
        return super()._setup_year(year)

    def _run_phase_2_submit_papers(self, *args, **kwargs):
        original = self.rag.batch_retrieve
        with ExitStack() as stack:
            stack.enter_context(
                instance_override(
                    self.rag,
                    "batch_retrieve",
                    lambda queries, *a, **kw: empty_safe_retrieve(
                        original, queries, *a, **kw
                    ),
                )
            )
            if getattr(self, "cache_provenance", None):
                original_build = self.rag.build_knowledge_index

                def checked_build():
                    expected = self.cache_provenance
                    if (
                        corpus_fingerprint(self.rag.documents)
                        != expected["corpus_hash"]
                        or digest([d.metadata["id"] for d in self.rag.documents])
                        != expected["ordered_ids_hash"]
                    ):
                        raise ValueError(
                            "Loaded document order differs from audited cache"
                        )
                    result = original_build()
                    if (
                        list(self.rag.document_embeddings.shape)
                        != expected["embedding_shape"]
                    ):
                        raise ValueError(
                            "Loaded embedding shape differs from audited cache"
                        )
                    return result

                stack.enter_context(
                    instance_override(self.rag, "build_knowledge_index", checked_build)
                )
            result = super()._run_phase_2_submit_papers(*args, **kwargs)
        if self.corpus_hash is None:
            self.corpus_hash = corpus_fingerprint(self.rag.documents)
        return result

    def _run_phase_5_update_funding(self, year, submissions_list, year_results):
        # Type identity is retained: do not replace agents with proxy objects.
        researchers = [
            a
            for a in self.ecosystem.agent_population.values()
            if a.get_type() == "university"
        ]
        agencies = [
            a
            for a in self.ecosystem.agent_population.values()
            if a.get_type() == "funding_agency"
        ]
        before = {
            a.id: {
                p: [dict(e) for e in entries]
                for p, entries in a.funding_success_history.items()
            }
            for a in researchers
        }
        gates = {
            a.id: AwardResourceGate(a, self.mechanisms.awards_feed_resources)
            for a in researchers
        }
        with ExitStack() as stack:
            original_getter = self.paper_tracker.get_papers_dataframe
            stack.enter_context(
                instance_override(
                    self.paper_tracker,
                    "get_papers_dataframe",
                    lambda year: funding_dataframe(original_getter, year),
                )
            )
            for agent in researchers:
                stack.enter_context(
                    instance_override(agent, "update_resources", gates[agent.id])
                )
            if not self.mechanisms.publication_record_visible:
                for agency in agencies:
                    original = agency._build_program_prompt

                    def hidden(program, apps, papers, builder=original):
                        return record_display_prompt(builder, program, apps, papers)

                    stack.enter_context(
                        instance_override(agency, "_build_program_prompt", hidden)
                    )
            result = super()._run_phase_5_update_funding(
                year, submissions_list, year_results
            )

        awards = []
        for agent in researchers:
            earned = new_awards(agent, before[agent.id], year)
            # Fail rather than silently suppress a newly introduced refund or
            # miss an upstream direct assignment that bypasses update_resources.
            if Counter(e["earned_amount"] for e in earned) != Counter(
                gates[agent.id].award_calls
            ):
                raise RuntimeError(
                    "Phase-5 resource credits do not match earned awards"
                )
            for event in earned:
                event["spendable_credit"] = (
                    event["earned_amount"]
                    if self.mechanisms.awards_feed_resources
                    else 0.0
                )
            awards.extend(earned)
        record = {
            "year": year,
            "cell": self.mechanisms.cell,
            "awards": awards,
            "legacy_zero_boundary": legacy_zero_boundary(
                [
                    {
                        "agent_id": a.id,
                        "active": bool(a.is_active),
                        "spendable_resources": float(a.resources),
                    }
                    for a in researchers
                ]
            ),
        }
        self.funding_feedback_years[year] = record
        year_results["funding_feedback"] = record
        return result

    def _run_phase_4_acceptance_decisions(self, year, year_results):
        pending = self.paper_tracker.pending_papers_by_year
        if year not in pending:
            if any(
                p.year == year and p.status == "pending"
                for p in self.paper_tracker.papers_by_id.values()
            ):
                raise RuntimeError("Pending-paper index is missing real pending papers")
            pending[year] = []
        return super()._run_phase_4_acceptance_decisions(year, year_results)

    def run_one_round(self, year):
        result = super().run_one_round(year)
        rows = []
        for aid in self.founder_ids:
            agent = self.ecosystem.agent_population[aid]
            papers = [
                p
                for p in self.paper_tracker.papers_by_id.values()
                if aid in p.all_author_ids
            ]
            earned = sum(
                e["amount"]
                for entries in agent.funding_success_history.values()
                for e in entries
            )
            rows.append(
                {
                    "agent_id": aid,
                    "active": bool(agent.is_active),
                    "spendable_resources": float(agent.resources),
                    "cumulative_earned_funding": float(earned),
                    "accepted_papers": sum(p.status == "accept" for p in papers),
                    "submitted_papers": len(papers),
                    # Full author counting, including rejected/pending papers and
                    # inactive founders. No conditioning on acceptance or survival.
                    "citations_all_papers": sum(
                        self.citation_tracker.get_citation_count(p.id) for p in papers
                    ),
                }
            )
        self.funding_feedback_years[year]["agent_rows"] = rows
        self.funding_feedback_years[year]["metrics"] = summarize_agents(rows)
        if self.funding_feedback_years[year][
            "legacy_zero_boundary"
        ] != legacy_zero_boundary(rows):
            raise RuntimeError("Founder boundary snapshot changed after funding phase")
        write_json(
            Path(self.args.docs_dir) / f"mechanisms_year_{year}.json",
            self.funding_feedback_years[year],
        )
        return result


def simulation_class(base=None):
    """Lazy import keeps plan, analysis and unit tests free of GPU dependencies."""
    if base is None:
        from utopia.simulation import Simulation

        base = Simulation
    return type("FundingFeedbackSimulation", (FundingFeedbackMixin, base), {})


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
        "mean_spendable_resources": statistics.mean(
            r["spendable_resources"] for r in rows
        ),
        "active_fraction": statistics.mean(int(r["active"]) for r in rows),
        "accepted_papers_per_founder": statistics.mean(
            r["accepted_papers"] for r in rows
        ),
        "citations_all_papers_per_founder": statistics.mean(
            r["citations_all_papers"] for r in rows
        ),
    }


def reconcile_year(record, expected_founders=None):
    """Reject changed denominators or stored summaries inconsistent with raw rows."""
    rows = record["agent_rows"]
    ids = [r["agent_id"] for r in rows]
    if len(ids) != len(set(ids)) or (
        expected_founders is not None and set(ids) != set(expected_founders)
    ):
        raise ValueError("Founder identities changed or were duplicated")
    for row in rows:
        if type(row["active"]) is not bool:
            raise ValueError("Active status must be boolean")
        for key in (
            "spendable_resources",
            "cumulative_earned_funding",
            "accepted_papers",
            "submitted_papers",
            "citations_all_papers",
        ):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid founder outcome: {key}")
        if row["accepted_papers"] > row["submitted_papers"]:
            raise ValueError("Accepted papers exceed submitted papers")
    if record["metrics"] != summarize_agents(rows):
        raise ValueError("Stored metrics do not reconcile with founder rows")
    if record["legacy_zero_boundary"] != legacy_zero_boundary(rows):
        raise ValueError(
            "Stored zero-boundary audit does not reconcile with founder rows"
        )
    return ids


def effective_args(args):
    """Exclude only intervention, seed, endpoint and artifact-location fields."""
    excluded = {
        "funding_feedback",
        "seed",
        "experiment_id",
        "vllm_url",
        "docs_dir",
        "log_dir",
        "checkpoint_dir",
        "visual_dir",
        "output_dir",
        "data_cache_dir",
    }
    return {k: v for k, v in vars(args).items() if k not in excluded}


def experiment_id(cell, seed):
    Mechanisms.from_cell(cell)
    if seed not in SEEDS:
        raise ValueError(f"Protocol seeds were fixed upfront: {SEEDS}")
    return f"funding_feedback_v6_n1200_y6_{cell}_seed{seed}"


def fallback_audit(simulation):
    rows = []
    missing = []
    for year in range(1, simulation.num_years + 1):
        path = Path(simulation.output_dir) / f"funding_applications_year_{year}.jsonl"
        if path.exists():
            rows.extend(
                json.loads(line) for line in path.read_text().splitlines() if line
            )
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
