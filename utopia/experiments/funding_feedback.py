"""Funding feedback bounded 2 x 2 mechanism ablation, reusing the original simulator.

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
    python -m utopia.experiments.funding_feedback plan --vllm-url http://localhost:8000/v1
    PYTHONHASHSEED=42 python -m utopia.experiments.funding_feedback run --cell P1F1 --seed 42         --vllm-url http://localhost:8000/v1 --server-manifest SERVER_MANIFEST.json
    python -m utopia.experiments.funding_feedback analyze
The plan prints all four exact client commands. The main agent owns execution,
server model/precision verification and the all-eight-EC2-GPU queue as capacity
becomes available, without preempting other jobs. Only an external
Qwen/Qwen3-32B server is accepted. No heuristic simulation is provided.

Outputs: outputs/{docs,logs,checkpoints,visual}/funding_feedback_<cell>_seed<seed>.
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
This patch does not reinterpret distance as propensity."""
from __future__ import annotations

from utopia.runtime.commands import module_command
from utopia.utils.data_utils import file_sha256 as file_hash
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import tempfile
import traceback
from utopia.analysis.funding_feedback import analyze
from utopia.funding.feedback import CELLS, FOUNDER_COUNT, FUNDING_BASELINE, FUNDING_OUTPUT_REPRESENTATION, FUNDING_PANEL_MAX_APPS, FUNDING_SELECTION_PROTOCOL, Mechanisms, PROTOCOL, SEEDS, SEQUENTIAL_AUDIT_FILE, SEQUENTIAL_SUMMARY_FILE, YEARS, effective_args, experiment_id, fallback_audit, funding_ledger_evidence, funding_phase_coverage, sequential_funding_scope, sequential_source_binding, sequential_summary_gates, simulation_class, source_fingerprint, validate_sequential_evidence
from utopia.runtime.provenance import MODEL, ROOT, audit_cache, cache_file_identity, digest, mark_failed_run_manifest, server_provenance, validate_request_audit_summary, verified_cache_manifest, write_json


def plan_commands(url, python="python"):
    return [
        ["env", f"PYTHONHASHSEED={seed}", "WANDB_MODE=disabled", python,
         "-m", 'utopia.experiments.funding_feedback', "run", "--cell", cell, "--seed", str(seed),
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
            "command": module_command(python, 'utopia.experiments.funding_feedback', 'run', '--cell', cell, '--seed', '42', '--vllm-url', '__SERVER_URL__', '--server-manifest', '__SERVER_MANIFEST__'),
            "environment": {"PYTHONHASHSEED": "42", "WANDB_MODE": "disabled"},
            "working_directory": str(ROOT), "output_roots": list(roots.values()),
            "artifact_checks": checks, "artifacts": artifacts,
        })
    return {"protocol": PROTOCOL, "jobs": jobs, "independent_worlds": 1,
            "launch_owner": "Main queue; this plan is not permission to skip preflight/capacity gates."}


def configure_run(options):
    """Use original defaults and argument validation without editing the parser."""
    from utopia.arguments import parse_arguments

    exp_id = experiment_id(options.cell, options.seed)
    if os.environ.get("PYTHONHASHSEED") != str(options.seed):
        raise ValueError(f"Start Python with PYTHONHASHSEED={options.seed}")
    # parse_arguments creates directories. Resolve its stage defaults inside a
    # temporary output tree, then give this independent experiment canonical IDs.
    with tempfile.TemporaryDirectory(prefix="funding-feedback-args-") as temporary:
        args = parse_arguments([
            "--experiment_name", "default_funding_feedback",
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
    args.funding_feedback = asdict(Mechanisms.from_cell(options.cell))
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
        "funding_compact_sha256": file_hash(ROOT / "utopia/funding/compact.py"),
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
        from utopia.simulation import Simulation, VLLMServerModel, FundingAgency, set_seed
        from utopia.models.request_audit import request_audit_scope
        from utopia.funding.validation import install_funding_validation, FundingRankingValidationError
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
                sim.cache_provenance = manifest["cache_provenance"]
                sim.run()
                if (len(sim.founder_ids) != FOUNDER_COUNT
                        or set(sim.funding_feedback_years) != set(range(1, YEARS + 1))):
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
            status="complete", initial_world_hash=sim.initial_world_hash,
            corpus_hash=sim.corpus_hash, fallback_audit=fallback_audit(sim),
            founder_ids=list(sim.founder_ids), request_audit_valid=True,
            sequential_audit_valid=True,
            funding_application_evidence_valid=True,
        )
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        write_json(path, manifest)
        mark_failed_run_manifest(args.docs_dir, PROTOCOL, manifest["error"])
        raise
    write_json(path, manifest)


def main(argv=None):
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
