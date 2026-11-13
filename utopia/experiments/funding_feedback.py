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
import hashlib
import subprocess
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shlex
import shutil
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



LARGE_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
LARGE_REVISION = "ea5b4f81096f3901c91dea97f81324302495781d"


def prospective_source_fingerprint(root=ROOT):
    # The manuscript-only renderer is not part of this scientific runtime.
    excluded = root / "utopia/visual/plot_ac_resubmission.py"
    paths = sorted(path for path in (root / "utopia").rglob("*.py") if path != excluded)
    paths += [root / "requirements.txt", root / "scripts/upload_checkpoint.sh", root / "scripts/fulltext_client_env.sh"]
    return digest({str(path.relative_to(root)): file_hash(path) for path in paths})


SOURCE_AMENDMENT_ORIGIN = "4115c7848c4ed8782a756466801beaeaa7e63b6b73da4e2e4cab6f89cb599aaa"
SOURCE_AMENDMENT_FILES = {
    "utopia/data/rag.py": (
        "64760c1a2858a7780a0ab4924a9ba802aa2b636e243226ef93c9a2e93e020627",
        "a596f007674c9b1e545d52b0697dca8d3b3869a2f37b23e663e8869b0b3d095f"),
    "utopia/experiments/funding_feedback.py": (
        "c30ce27531219fc9745a65c6c2f533300210412dcb6e98fe18d09d47952620c2", None),
}
SOURCE_AMENDMENT_STUDIES = {
    "pf10000_fulltext_qwen397b_seed42_20261006": (
        ("P1F1", "P0F1", "P1F0", "P0F0"), 1791757717.881552),
    "switch5000_fulltext_qwen397b_seed42_20261006": (
        ("LN", "HN"), 1791844861.1598995),
}


# Explicit 2026-10-10 author amendment for these four existing P6-2 worlds only.
LENGTH_EXTENSION_RUN = "pf10000_fulltext_qwen397b_seed42_20261006"
LENGTH_EXTENSION_PARENT_SOURCE = "d517faa477544c3244f6587d17c26ac87148e5134455ff05bc3c5dd1ad5e8eea"
LENGTH_EXTENSION_PARENT_AMENDMENT = "62c071d5c395d9fa2fc33df79d5c06593f0a3972a7545719b6148dbcdc41d9ce"
LENGTH_EXTENSION_MODEL_SHA256 = "334f4d557c6917ed215618d91f38bdb6ecbdfe7da42c16f027c147f01ffbdc64"
LENGTH_EXTENSION_POLICY = {
    "protocol": "three-length-no-final-then-one-context-capped-attempt-v1",
    "original_attempts": 3, "original_max_tokens": 8192,
    "extra_attempt_index": 3, "extra_max_tokens": 32768,
    "context_tokens": 131072, "input_policy": "unchanged-full-input",
    "on_extra_failure": "stop", "cache_namespace": "experiment_origin",
}
LENGTH_EXTENSION_FAILED_MANIFESTS = {
    "P0F0": "66164265cc69ef983ee80b766f08f9dabc12fb6aeffe617c889e9ce5c59a3839",
    "P0F1": "7dce9400e374d203c26fa1ecb3ff3a99e82a91c5af1ec588f81cc93591f168bd",
    "P1F0": "5ba1dc0615affec63b75ccdba676a632941118dcc2e7449b0ddde1a484fd6834",
    "P1F1": "ff65d25aa23a3c24a248dbf7bca2c5001da28fbc7ab228e60cc63658f2a6b0c0",
}

ANSWER_RECOVERY_PARENT_SOURCE = "7a49201057f39a26ae7bf64bdb204a1f63af1b0be59164bbf2e6620575cce1c3"
ANSWER_RECOVERY_PARENT_AMENDMENT = "5df3f3e078e2e00b98e1ac67106a3e716670c3021fc1a102b153ff736e3b863d"
ANSWER_RECOVERY_FILE_PINS = {
    "utopia/models/models.py": "42c08d7ad628c4dceebee6a9db853d24df96d4466d6d69934d12e35b709d1597",
    "utopia/models/request_audit.py": "c4aae6e25e3f3f563d1ac25be4e2edbeafd261eec2562add9809b56c8fe1c996",
}
ANSWER_RECOVERY_POLICY = {
    "protocol": "legacy-attempts-then-one-nonthinking-answer-v1",
    "legacy_policy": LENGTH_EXTENSION_POLICY,
    "legacy_failure_handling": "preserve-replies-then-recovery-after-parse-failure",
    "recovery_attempt_index": 4, "recovery_max_tokens": 32768,
    "enable_thinking": False, "context_tokens": 131072,
    "eligibility": "three-or-four-audited-length-or-stop-replies-no-refusal-or-tools",
    "input_policy": "unchanged-full-messages-schema-temperature",
    "seed_policy": "original-derived-seed-at-index4",
    "context_accounting": "exact-nonthinking-template",
    "validation": "finish-stop-strict-json-and-existing-scientific-validators",
    "on_recovery_failure": "stop-no-sixth-attempt",
    "cache_namespace": "experiment_origin",
}
ANSWER_RECOVERY_FAILED_MANIFESTS = {
    "P1F1": "a0fad35a5037661521d86aca890c8353fbb3c9707ddc6bde9b843a4b155a7dd9",
    "P0F1": "336643bba4803b09189abc30dc59f95f21e2306f1d7bf529a8c02f53a663d5a8",
    "P1F0": "b89643524b9b632d7b7a2534884eb9df9a5943faf9ca3106c5791d7a87baf91a",
    "P0F0": "952c85192c4fb54c613c825bf0ddc8bfedac36db95e3fc53ed10e18bc9344aad",
}


def validate_answer_recovery_inventory(options, amendment, original_files, current_files):
    """Admit the reviewed P62 repair while retaining the complete prior amendment."""
    if (options.run_id != LENGTH_EXTENSION_RUN
            or options.cell not in ANSWER_RECOVERY_FAILED_MANIFESTS
            or amendment.get("generation_extension") != ANSWER_RECOVERY_POLICY
            or options.max_tokens != 8192 or options.context_tokens != 131072):
        raise ValueError("Answer recovery differs from the P62 authorization")
    parent_raw = bytes.fromhex(amendment["parent_amendment_raw_hex"])
    if hashlib.sha256(parent_raw).hexdigest() != ANSWER_RECOVERY_PARENT_AMENDMENT:
        raise ValueError("Answer recovery parent amendment bytes differ")
    parent = json.loads(parent_raw)
    parent_files = parent["execution_files"]
    if (parent["execution_source_sha256"] != ANSWER_RECOVERY_PARENT_SOURCE
            or digest(parent_files) != ANSWER_RECOVERY_PARENT_SOURCE
            or parent["original_files"] != original_files
            or parent["cells"] != amendment["cells"]
            or parent.get("generation_extension") != LENGTH_EXTENSION_POLICY
            or set(parent_files) != set(current_files)):
        raise ValueError("Answer recovery parent inventory/cells/policy differ")
    changed = {name for name in parent_files if parent_files[name] != current_files[name]}
    if (changed != set(ANSWER_RECOVERY_FILE_PINS) | {"utopia/experiments/funding_feedback.py"}
            or any(current_files[name] != pin for name, pin in ANSWER_RECOVERY_FILE_PINS.items())):
        raise ValueError("Answer recovery must change only reviewed model/audit/runner files")
    return parent


def validate_answer_recovery_transition(options, previous, parent, root):
    """The first repair activation requires each cell's own verified failed state."""
    prior_info = previous.get("source_amendment") or {}
    manifest = root / "outputs/docs" / f"{options.run_id}_{options.cell}" / "experiment.json"
    checkpoint_root = root / "outputs/checkpoints" / f"{options.run_id}_{options.cell}"
    checkpoint = checkpoint_root / "portable_checkpoint_year_0.json.gz"
    if (previous.get("source_hash") != ANSWER_RECOVERY_PARENT_SOURCE
            or prior_info.get("sha256") != ANSWER_RECOVERY_PARENT_AMENDMENT
            or prior_info.get("record") != parent
            or previous.get("status") != "failed" or previous.get("years_completed") != 0
            or previous.get("generation_extension") != LENGTH_EXTENSION_POLICY
            or file_hash(manifest) != ANSWER_RECOVERY_FAILED_MANIFESTS[options.cell]
            or set(checkpoint_root.glob("portable_checkpoint_year_*.json.gz")) != {checkpoint}
            or file_hash(checkpoint) != parent["cells"][options.cell]["origin_checkpoint_sha256"]):
        raise ValueError("Answer recovery requires the exact verified failed cell")


def validate_length_extension_inventory(options, amendment, original_files, current_files):
    """Admit the approved generation change through the exact retained parent."""
    if (options.run_id != LENGTH_EXTENSION_RUN
            or options.cell not in LENGTH_EXTENSION_FAILED_MANIFESTS
            or amendment.get("generation_extension") != LENGTH_EXTENSION_POLICY
            or options.max_tokens != 8192 or options.context_tokens != 131072):
        raise ValueError("Conditional response extension differs from the P62 authorization")
    parent_raw = bytes.fromhex(amendment["parent_amendment_raw_hex"])
    if hashlib.sha256(parent_raw).hexdigest() != LENGTH_EXTENSION_PARENT_AMENDMENT:
        raise ValueError("Conditional response extension parent amendment bytes differ")
    parent = json.loads(parent_raw)
    parent_files = parent["execution_files"]
    if (parent["execution_source_sha256"] != LENGTH_EXTENSION_PARENT_SOURCE
            or digest(parent_files) != LENGTH_EXTENSION_PARENT_SOURCE
            or parent["original_files"] != original_files
            or parent["cells"] != amendment["cells"]
            or set(parent_files) != set(current_files)):
        raise ValueError("Conditional response extension parent inventory/cells differ")
    changed = {name for name in parent_files if parent_files[name] != current_files[name]}
    if (changed != {"utopia/models/models.py", "utopia/experiments/funding_feedback.py"}
            or current_files["utopia/models/models.py"] != LENGTH_EXTENSION_MODEL_SHA256):
        raise ValueError("Conditional response extension must change only reviewed model/runner files")
    return parent


def validate_length_extension_transition(options, previous, parent, root):
    """First activation must start from the exact backed-up failed year-zero cell."""
    prior_info = previous.get("source_amendment") or {}
    manifest = root / "outputs/docs" / f"{options.run_id}_{options.cell}" / "experiment.json"
    checkpoint_root = root / "outputs/checkpoints" / f"{options.run_id}_{options.cell}"
    checkpoint = checkpoint_root / "portable_checkpoint_year_0.json.gz"
    if (previous.get("source_hash") != LENGTH_EXTENSION_PARENT_SOURCE
            or prior_info.get("sha256") != LENGTH_EXTENSION_PARENT_AMENDMENT
            or prior_info.get("record") != parent
            or previous.get("status") != "failed" or previous.get("years_completed") != 0
            or previous.get("generation_extension") is not None
            or file_hash(manifest) != LENGTH_EXTENSION_FAILED_MANIFESTS[options.cell]
            or set(checkpoint_root.glob("portable_checkpoint_year_*.json.gz")) != {checkpoint}
            or file_hash(checkpoint) != parent["cells"][options.cell]["origin_checkpoint_sha256"]):
        raise ValueError("Conditional response extension requires the exact verified failed cell")



def resolve_prospective_source(options, candidate_binding, previous, root=ROOT):
    """Admit reviewed source/generation amendments; retain the origin cache namespace.

    The candidate and fingerprint remain actual execution provenance. The returned
    origin binding preserves native contracts and response keys, not a claim that
    the original source is executing. No checkpoint or database is modified here.
    """
    execution_hash = prospective_source_fingerprint(root)
    path = getattr(options, "source_amendment", None)
    if not path:
        if previous is not None and (previous.get("source_amendment")
                                     or previous["binding"] != candidate_binding):
            raise ValueError("Scientific configuration differs; explicit --source-amendment required")
        return candidate_binding, {"source_hash": execution_hash}, None
    if not options.resume or previous is None:
        raise ValueError("Source amendment requires an existing world and explicit --resume")
    study = SOURCE_AMENDMENT_STUDIES.get(options.run_id)
    if study is None or options.cell not in study[0]:
        raise ValueError("Source amendment is restricted to the six original scientific cells")
    cells, deadline = study
    raw = Path(path).read_bytes()
    amendment = json.loads(raw)
    if amendment.get("format") == "scienceutopia-p69-authorized-continuation-v1":
        return resolve_p69_continuation(options, candidate_binding, previous, root, raw)
    if amendment.get("format") == "scienceutopia-p62-p63-continuation-v1":
        return resolve_p62_continuation(options, candidate_binding, previous, root, raw)
    if (amendment.get("format") != "scienceutopia-source-amendment-v1"
            or amendment.get("binding_source_role") != "experiment_origin"
            or amendment.get("run_id") != options.run_id
            or set(amendment.get("cells", {})) != set(cells)):
        raise ValueError("Source amendment format/study/cell coverage differs")
    original_files, execution_files = amendment["original_files"], amendment["execution_files"]
    if (amendment["origin_source_sha256"] != SOURCE_AMENDMENT_ORIGIN
            or digest(original_files) != SOURCE_AMENDMENT_ORIGIN):
        raise ValueError("Original complete source inventory differs from the frozen origin")
    excluded = root / "utopia/visual/plot_ac_resubmission.py"
    paths = sorted(p for p in (root / "utopia").rglob("*.py") if p != excluded)
    paths += [root / "requirements.txt", root / "scripts/upload_checkpoint.sh",
              root / "scripts/fulltext_client_env.sh"]
    current_files = {str(p.relative_to(root)): file_hash(p) for p in paths}
    if (execution_files != current_files or set(original_files) != set(current_files)
            or digest(current_files) != execution_hash
            or amendment["execution_source_sha256"] != execution_hash):
        raise ValueError("Current complete source inventory differs from the amendment")
    extension = amendment.get("generation_extension")
    parent = None
    answer_recovery = (extension is not None and
                       extension.get("protocol") == "legacy-attempts-then-one-nonthinking-answer-v1")
    if answer_recovery:
        parent = validate_answer_recovery_inventory(options, amendment, original_files, current_files)
    elif extension is not None:
        parent = validate_length_extension_inventory(options, amendment, original_files, current_files)
    else:
        changed = {name for name in original_files if original_files[name] != current_files[name]}
        if changed != set(SOURCE_AMENDMENT_FILES):
            raise ValueError("Source amendment must change exactly the dtype and runner files")
        for name, (old, new) in SOURCE_AMENDMENT_FILES.items():
            if original_files[name] != old or (new is not None and current_files[name] != new):
                raise ValueError(f"Source amendment file pin differs: {name}")
    cell = amendment["cells"][options.cell]
    origin = previous["binding"]
    origin_candidate = json.loads(json.dumps(candidate_binding))
    origin_candidate["science_config"]["source_hash"] = SOURCE_AMENDMENT_ORIGIN
    if (origin != origin_candidate or digest(origin) != cell["origin_binding_sha256"]
            or origin["cell"] != options.cell or origin["seed"] != 42):
        raise ValueError("Non-source scientific binding differs from the original world")
    if (options.deadline_epoch != deadline
            or previous["args"]["deadline_epoch"] != deadline
            or any(item["deadline_epoch"] != deadline for item in amendment["cells"].values())
            or previous["args"]["experiment_id"] != f"{options.run_id}_{options.cell}"):
        raise ValueError("Original study/cell/client deadline differs")
    source_info = {
        "binding_source_role": "experiment_origin",
        "source_hash": execution_hash,
        "source_amendment": {"sha256": hashlib.sha256(raw).hexdigest(), "record": amendment},
    }
    if extension is not None:
        # The origin binding remains the cache key, while this field explicitly
        # records the active generation protocol on every session and snapshot.
        source_info["generation_extension"] = extension
    if parent is not None:
        if any(previous.get(key) != value for key, value in source_info.items()):
            if answer_recovery:
                validate_answer_recovery_transition(options, previous, parent, root)
            else:
                validate_length_extension_transition(options, previous, parent, root)
    elif previous.get("source_amendment"):
        if any(previous.get(key) != value for key, value in source_info.items()):
            raise ValueError("Existing execution/source-amendment lineage differs")
    elif (previous.get("source_hash") != SOURCE_AMENDMENT_ORIGIN
          or previous.get("years_completed") != 0
          or previous.get("status") != "failed"):
        raise ValueError("First amendment requires the original failed year-zero experiment")
    else:
        checkpoint_root = root / "outputs/checkpoints" / f"{options.run_id}_{options.cell}"
        checkpoint = checkpoint_root / "portable_checkpoint_year_0.json.gz"
        if (set(checkpoint_root.glob("portable_checkpoint_year_*.json.gz")) != {checkpoint}
                or file_hash(checkpoint) != cell["origin_checkpoint_sha256"]):
            raise ValueError("First amendment requires only the exact original year-zero checkpoint")
    return origin, source_info, raw


def resolve_p62_continuation(options, candidate, previous, root, raw):
    """Resume the four preserved P62 worlds under the October 16 authorization."""
    import time
    amendment = json.loads(raw)
    if (options.run_id != LENGTH_EXTENSION_RUN
            or options.cell not in ANSWER_RECOVERY_FAILED_MANIFESTS
            or options.seed != 42 or options.founders != 10000 or options.years != 6
            or options.context_tokens != 131072 or options.max_tokens != 8192
            or options.deadline_epoch != 1792707542
            or time.time() >= options.deadline_epoch
            or amendment["client_deadline_epoch"] != options.deadline_epoch
            or amendment["generation_extension"] != ANSWER_RECOVERY_POLICY):
        raise ValueError("P62 continuation differs from the authorized worlds or deadline")
    parent_raw = bytes.fromhex(amendment["parent_amendment_raw_hex"])
    parent = json.loads(parent_raw)
    if (hashlib.sha256(parent_raw).hexdigest() !=
            "a5a71d757998e92af950ab2d5dd925cee1b8be317dfad6574b3e28aa3ea940c0"
            or parent["execution_source_sha256"] !=
            "e117153d7d9c9ced34181dc4cdb11b58be8d5310aad52dc9843b67173f90d5ea"
            or parent["generation_extension"] != ANSWER_RECOVERY_POLICY):
        raise ValueError("The preserved P62 source/generation amendment differs")
    excluded = root / "utopia/visual/plot_ac_resubmission.py"
    paths = sorted(p for p in (root / "utopia").rglob("*.py") if p != excluded)
    paths += [root / "requirements.txt", root / "scripts/upload_checkpoint.sh",
              root / "scripts/fulltext_client_env.sh"]
    current = {str(p.relative_to(root)): file_hash(p) for p in paths}
    changed = {p for p in current if current[p] != parent["execution_files"].get(p)}
    if (set(current) != set(parent["execution_files"])
            or current != amendment["execution_files"]
            or digest(current) != amendment["execution_source_sha256"]
            or changed != {"utopia/experiments/funding_feedback.py"}):
        raise ValueError("P62 continuation source differs from its declared narrow change")
    origin_candidate = json.loads(json.dumps(candidate))
    origin_candidate["science_config"]["source_hash"] = SOURCE_AMENDMENT_ORIGIN
    if (origin_candidate != previous["binding"]
            or digest(previous["binding"]) != parent["cells"][options.cell]["origin_binding_sha256"]):
        raise ValueError("P62 continuation changes the scientific binding")
    restored = amendment["restored_cells"][options.cell]
    if previous["source_hash"] == parent["execution_source_sha256"]:
        manifest = root / "outputs/docs" / f"{options.run_id}_{options.cell}" / "experiment.json"
        checkpoint = root / "outputs/checkpoints" / f"{options.run_id}_{options.cell}" / restored["checkpoint"]
        if (file_hash(manifest) != restored["manifest_sha256"]
                or file_hash(checkpoint) != restored["checkpoint_sha256"]
                or previous.get("status") != "paused" or previous.get("years_completed") != 1
                or previous["source_amendment"]["sha256"] != hashlib.sha256(parent_raw).hexdigest()
                or previous["args"]["deadline_epoch"] != 1791757717.881552):
            raise ValueError("P62 continuation requires its exact preserved stopped world")
    elif (previous["source_hash"] != digest(current)
          or previous["source_amendment"]["sha256"] != hashlib.sha256(raw).hexdigest()
          or previous["args"]["deadline_epoch"] != options.deadline_epoch
          or previous.get("status") not in {"paused", "failed"}):
        raise ValueError("P62 continuation lineage or stopped state differs")
    return previous["binding"], {
        "source_hash": digest(current),
        "binding_source_role": "experiment_origin",
        "source_amendment": {"sha256": hashlib.sha256(raw).hexdigest(), "record": amendment},
        "generation_extension": ANSWER_RECOVERY_POLICY,
        "continuation": {
            "host": "P6-3", "client_deadline_epoch": options.deadline_epoch,
            "request_timeout_seconds": 1800.0, "restored_snapshot": restored,
            "storage_policy": "server_preserved_then_s3_when_available",
        },
    }, raw


def resolve_p69_continuation(options, candidate, previous, root, raw):
    """Continue the existing four worlds under the user's renewed P69 authorization."""
    amendment = json.loads(raw)
    if (options.run_id != LENGTH_EXTENSION_RUN
            or options.cell not in ANSWER_RECOVERY_FAILED_MANIFESTS
            or options.seed != 42 or options.founders != 10000 or options.years != 6
            or options.context_tokens != 131072 or options.max_tokens != 8192
            or not options.until_complete or options.deadline_epoch is not None
            or amendment["generation_extension"] != ANSWER_RECOVERY_POLICY
            or amendment["run_id"] != options.run_id
            or set(amendment["cells"]) != set(ANSWER_RECOVERY_FAILED_MANIFESTS)):
        raise ValueError("P69 continuation differs from the authorized worlds or completion horizon")
    parent_raw = Path(amendment["parent_amendment_path"]).read_bytes()
    parent = json.loads(parent_raw)
    if parent["generation_extension"] != ANSWER_RECOVERY_POLICY:
        raise ValueError("The preserved P69 predecessor source/generation amendment differs")
    excluded = root / "utopia/visual/plot_ac_resubmission.py"
    paths = sorted(p for p in (root / "utopia").rglob("*.py") if p != excluded)
    paths += [root / "requirements.txt", root / "scripts/upload_checkpoint.sh",
              root / "scripts/fulltext_client_env.sh"]
    current = {str(p.relative_to(root)): file_hash(p) for p in paths}
    # Source inventories describe execution provenance; continuation is selected
    # by the existing world ID and non-source scientific fields.
    cell = amendment["cells"][options.cell]
    origin_candidate = json.loads(json.dumps(candidate))
    origin = json.loads(json.dumps(previous["binding"]))
    origin_candidate["science_config"].pop("source_hash", None)
    origin["science_config"].pop("source_hash", None)
    origin_candidate.pop("corpus_sha256", None)
    origin.pop("corpus_sha256", None)
    if (origin_candidate != origin
            or previous["args"]["experiment_id"] != f"{options.run_id}_{options.cell}"
            or previous.get("status") not in {"paused", "failed"}):
        raise ValueError("P69 continuation changes the scientific world or stopped state")
    if not 0 <= previous.get("years_completed", -1) <= options.years:
        raise ValueError("Preserved world year lies outside its authorized horizon")
    return previous["binding"], {
        "source_hash": digest(current),
        "binding_source_role": "experiment_origin",
        "source_amendment": {"sha256": hashlib.sha256(raw).hexdigest(), "record": amendment},
        "generation_extension": ANSWER_RECOVERY_POLICY,
        "s3_auth_pending": True,
        "continuation": {
            "host": "P6-3", "execution_host": "P6-9", "client_deadline_epoch": None,
            "completion_horizon_years": 6, "request_timeout_seconds": 1800.0,
            "restored_snapshot": cell["restored_snapshot"],
            "storage_policy": "server_preserved_then_s3_when_available",
            "authorization_sha256": amendment.get("authorization_sha256"),
        },
    }, raw


def preserve_p62_checkpoint(checkpoint, backup, docs_root, log_dir, metadata, final_manifest=None):
    """Freeze a new checkpoint and its complete reply/audit/output evidence on P63."""
    checkpoint = Path(checkpoint)
    shutil.copy2(checkpoint, backup / checkpoint.name)
    shutil.copytree(docs_root, backup / "results",
                    ignore=shutil.ignore_patterns("responses", "backup", "*.sqlite*"))
    shutil.copytree(log_dir, backup / "logs")
    shutil.copytree(checkpoint.parent, backup / "outputs",
                    ignore=shutil.ignore_patterns("portable_checkpoint_*"))
    if final_manifest is not None:
        write_json(backup / "results/experiment.json", final_manifest)
    # The immutable source package and declaration are prepared before launch.
    for name in ("source_manifest.json", "source_amendment.json", "runtime.tar.gz"):
        shutil.copy2(ROOT.parent / name, backup / name)
    files = {}
    for path in sorted(backup.rglob("*")):
        if path.is_file() and path != backup / "snapshot.json":
            files[str(path.relative_to(backup))] = {
                "bytes": path.stat().st_size, "sha256": file_hash(path)}
    metadata.update(storage_status="server_preserved_s3_pending", is_offhost=False,
                    server_directory=str(backup), preserved_files=files)
    write_json(backup / "snapshot.json", metadata)


def prospective_checkpoint_source(source_info, cell, year, checkpoint_sha256):
    """A re-uploaded origin checkpoint was not produced by the new execution."""
    continuation = source_info.get("continuation")
    if continuation:
        restored = continuation["restored_snapshot"]
        if checkpoint_sha256 == restored["checkpoint_sha256"]:
            return restored["checkpoint_source_hash"]
    amendment = source_info.get("source_amendment")
    if amendment and year == 0:
        record = amendment["record"]
        if checkpoint_sha256 != record["cells"][cell]["origin_checkpoint_sha256"]:
            raise ValueError("Refuse to relabel an altered year-zero checkpoint")
        return record["origin_source_sha256"]
    return source_info["source_hash"]


def prospective_arguments(command):
    """Shared explicit options for new full-text, resumable experiments only."""
    command.add_argument("--cell", required=True)
    command.add_argument("--seed", type=int, choices=[42], default=42)
    command.add_argument("--founders", type=int, default=10000)
    command.add_argument("--years", type=int, default=6)
    command.add_argument("--run-id", required=True)
    command.add_argument("--vllm-url", required=True)
    command.add_argument("--server-manifest", required=True)
    command.add_argument("--fulltext-index", required=True)
    deadline = command.add_mutually_exclusive_group(required=True)
    deadline.add_argument("--deadline-epoch", type=float)
    deadline.add_argument("--until-complete", action="store_true",
                          help="Resume explicitly authorized P69 worlds to their fixed six-year horizon")
    command.add_argument("--s3-prefix", required=True, help="Private per-study S3 backup prefix")
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--model", default=LARGE_MODEL)
    command.add_argument("--model-revision", default=LARGE_REVISION)
    command.add_argument("--context-tokens", type=int, default=131072)
    command.add_argument("--max-tokens", type=int, default=8192)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--source-amendment", help="Explicit reviewed origin/execution amendment JSON")


def configure_prospective(options):
    from utopia.arguments import parse_arguments
    from utopia.config import SIMULATION_CONFIG
    if os.environ.get("PYTHONHASHSEED") != str(options.seed):
        raise ValueError(f"Start Python with PYTHONHASHSEED={options.seed}")
    if options.founders <= 0 or options.founders % 5:
        raise ValueError("Founder count must be a positive multiple of five")
    if options.years <= 0 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in options.run_id + options.cell):
        raise ValueError("Invalid horizon or run/cell identifier")
    if getattr(options, "source_amendment", None) and not options.resume:
        raise ValueError("--source-amendment requires --resume")
    exp_id = f"{options.run_id}_{options.cell}"
    with tempfile.TemporaryDirectory(prefix="scienceutopia-args-") as temporary:
        args = parse_arguments([
            "--experiment_name", "default_prospective", "--experiment_stage", "mechanism",
            "--output-dir", temporary, "--seed", str(options.seed),
            "--population_mode", "university_only", "--num_institutions", str(options.founders // 5),
            "--researchers_per_institution", "5", "--strategy_mix", "balanced",
            "--num_years", str(options.years), "--num_conferences", "6",
            "--model", options.model, "--vllm_url", options.vllm_url,
            "--batch_size", str(options.batch_size), "--funding_panel_max_apps", "25",
            "--funding_budget_mode", "track", "--log_funding_applications",
        ])
    args.experiment_id = exp_id
    args.expected_founders = options.founders
    args.embedding_model_revision = "17e1f347d17fe144873b1201da91788898c639cd"
    args.portable_checkpoint = True
    args.always_rerun = False
    args.deadline_epoch = options.deadline_epoch
    funding_cell = getattr(options, "funding_cell", options.cell if options.cell in CELLS else "P1F1")
    args.funding_feedback = asdict(Mechanisms.from_cell(funding_cell))
    docs_root = ROOT / "outputs/docs" / exp_id
    args.docs_dir = str(docs_root / "metrics")
    args.log_dir = str(ROOT / "outputs/logs" / exp_id)
    args.checkpoint_dir = str(ROOT / "outputs/checkpoints" / exp_id)
    args.visual_dir = str(ROOT / "outputs/visual" / exp_id)
    args.output_dir = args.checkpoint_dir
    args.rag_device = "cpu"
    args.data_cache_dir = str(ROOT / "data/cache" / exp_id)
    args.checkpoint_replay_paths = {}  # Native simulator declares exact replay outputs.
    checkpoint_root = Path(args.checkpoint_dir)
    if not options.resume and ((docs_root.exists() and any(docs_root.iterdir()))
                               or (checkpoint_root.exists() and any(checkpoint_root.iterdir()))):
        raise FileExistsError("Existing run requires explicit --resume")
    if options.resume and not any(checkpoint_root.glob("portable_checkpoint_year_*.json.gz")):
        raise FileNotFoundError("Resume requires a portable checkpoint from this new protocol")
    for path in (args.docs_dir, args.log_dir, args.checkpoint_dir, args.visual_dir, args.data_cache_dir):
        Path(path).mkdir(parents=True, exist_ok=True)
    # Disabled W&B transport still leaves the project logger enabled. Portable
    # replay requires disabling that logger before the scientific binding is built.
    SIMULATION_CONFIG["wandb"]["enabled"] = False
    return args


def run_prospective(options, simulation_factory=None, configure=None, run_context=None):
    """One cell through the native simulator, with pinned data and portable replies."""
    import time
    from contextlib import nullcontext
    from utopia.data.fulltext import FullTextCorpus
    from utopia.config import SIMULATION_CONFIG
    from utopia.runtime.checkpointing import PortableStop
    args = configure_prospective(options)
    extra_binding = configure(args) if configure else {}
    corpus = FullTextCorpus(options.fulltext_index)
    os.environ["UTOPIA_FULLTEXT_INDEX"] = str(corpus.path)
    os.environ["UTOPIA_FULLTEXT_SHA256"] = corpus.sha256
    os.environ["UTOPIA_EMBEDDING_REVISION"] = args.embedding_model_revision
    os.environ["WANDB_MODE"] = "disabled"
    protocol = getattr(options, "protocol", "pf-fulltext-portable-v1")
    scientific_args = effective_args(args)
    for name in ("deadline_epoch", "checkpoint_replay_paths", "data_cache_dir", "rag_device", "batch_size"):
        scientific_args.pop(name, None)
    binding = {
        "protocol": protocol, "cell": options.cell, "seed": options.seed,
        "model": options.model, "model_revision": options.model_revision,
        "corpus_sha256": corpus.sha256,
        "science_config": {"args": scientific_args, "config": SIMULATION_CONFIG,
                           "source_hash": prospective_source_fingerprint(),
                           "context_tokens": options.context_tokens, "max_tokens": options.max_tokens,
                           "funding_selection": "one_complete_permutation_v1", "extra": extra_binding or {}},
    }
    binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    docs_root = Path(args.docs_dir).parent
    manifest_path = docs_root / "experiment.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    binding, source_info, amendment_bytes = resolve_prospective_source(options, binding, previous)
    args.checkpoint_binding = binding
    sessions = docs_root / "sessions"
    sessions.mkdir(exist_ok=True)
    session = sessions / str(time.time_ns())
    session.mkdir()
    if amendment_bytes is not None:
        (session / "source_amendment.json").write_bytes(amendment_bytes)
    server = json.loads(Path(options.server_manifest).read_text())
    for key, expected in {"model": options.model, "model_revision": options.model_revision,
                          "endpoint": options.vllm_url}.items():
        if server.get(key) != expected:
            raise ValueError(f"Server deployment differs from requested {key}")
    if source_info.get("continuation"):
        if (server.get("context_tokens", 0) < options.context_tokens
                or server.get("client_context_tokens") != options.context_tokens):
            raise ValueError("P62 request budget exceeds the declared P63 service capacity")
    elif server.get("context_tokens") != options.context_tokens:
        raise ValueError("Server deployment differs from requested context_tokens")
    manifest = {"binding": binding, "status": "starting", "server": server,
                "args": dict(vars(args)), **source_info,
                "corpus": corpus.reference(), "session": str(session),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sampling_scope": "One seed and one world per condition; no across-world uncertainty."}
    if previous is not None:
        if previous["binding"] != binding:
            raise ValueError("Scientific configuration differs from the saved experiment")
        if previous.get("status") == "complete" and previous.get("final_uploaded_snapshot"):
            return previous
    write_json(manifest_path, manifest)
    from utopia.simulation import Simulation, VLLMServerModel, FundingAgency, set_seed
    from utopia.models.request_audit import request_audit_scope
    from utopia.funding.validation import install_funding_validation
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(Path(args.log_dir) / "simulation.log"), logging.StreamHandler()])
    set_seed(args.seed)
    handle = install_funding_validation(FundingAgency, session / "funding_validation.jsonl", compact=True)
    llm = sim = audit = upload_checkpoint = None
    backup_failed = False
    s3_available = not source_info.get("s3_auth_pending", False)
    try:
        with request_audit_scope(session / "llm_requests.jsonl", model_name=options.model,
                                 model_revision=options.model_revision,
                                 context_tokens=options.context_tokens, sdk_retries=0, allow_empty=options.resume) as audit:
            llm = VLLMServerModel(
                model_name=options.model, base_url=options.vllm_url, run_seed=options.seed,
                max_concurrent_requests=options.batch_size,
                response_cache_dir=str(docs_root / "responses"), checkpoint_binding=binding,
                model_revision=options.model_revision, max_tokens_override=options.max_tokens,
                sdk_max_retries=0, request_timeout=(
                    1800.0 if source_info.get("continuation") else 600.0),
                length_exhaustion_max_tokens=(
                    32768 if source_info.get("generation_extension") else None),
                answer_recovery_max_tokens=(
                    32768 if source_info.get("generation_extension") == ANSWER_RECOVERY_POLICY else None))
            if options.model not in {m.id for m in llm.client.models.list().data}:
                raise RuntimeError("Endpoint does not advertise the pinned model")
            def upload_checkpoint(checkpoint_path, final=False):
                """Publish an immutable checkpoint bundle before advancing its latest pointer."""
                nonlocal backup_failed, s3_available
                try:
                    checkpoint_path = Path(checkpoint_path)
                    year = int(checkpoint_path.name.removeprefix("portable_checkpoint_year_").removesuffix(".json.gz"))
                    continuation = source_info.get("continuation")
                    if continuation and not final:
                        restored = continuation["restored_snapshot"]
                        if (llm.response_cache.committed_count() == restored["completed_responses"]
                                and file_hash(checkpoint_path) == restored["checkpoint_sha256"]):
                            logging.info("Reusing restored S3 snapshot for year %s; no new replies", year)
                            return restored
                    snapshot_id = f"snapshot_{time.time_ns()}"
                    backup = docs_root / "backup" / f"year_{year}" / snapshot_id
                    backup.mkdir(parents=True, exist_ok=False)
                    response_snapshot = backup / "responses.sqlite"
                    llm.response_cache.backup(response_snapshot)
                    cell_prefix = options.s3_prefix.rstrip("/") + f"/{options.cell}"
                    destination = cell_prefix + f"/checkpoints/year_{year}/{snapshot_id}"
                    with response_snapshot.open("rb") as stream:
                        response_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
                    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
                    metadata = {
                        "year_completed": year, "cell": options.cell, "binding": binding, **source_info,
                        "checkpoint": checkpoint_path.name,
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_source_hash": prospective_checkpoint_source(
                            source_info, options.cell, year, checkpoint_sha256),
                        "completed_responses": llm.response_cache.committed_count(),
                        "response_snapshot_sha256": response_sha256,
                        "destination": destination, "includes_final_report": final,
                        "snapshot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    final_manifest = backup / "final_experiment.json" if final and not continuation else None
                    if final_manifest is not None:
                        write_json(final_manifest, dict(manifest, final_uploaded_snapshot=destination))
                    metadata_path = backup / "snapshot.json"
                    write_json(metadata_path, metadata)
                    if continuation:
                        preserve_p62_checkpoint(checkpoint_path, backup, docs_root, args.log_dir, metadata,
                                                final_manifest=manifest if final else None)
                        if s3_available:
                            completed = subprocess.run([
                                "aws", "s3", "sync", str(backup) + "/", destination + "/",
                                "--exclude", "snapshot.json", "--only-show-errors"],
                                capture_output=True, text=True)
                            if completed.returncode == 0:
                                metadata.update(storage_status="s3_uploaded", is_offhost=True)
                                write_json(metadata_path, metadata)
                                completed = subprocess.run([
                                    "aws", "s3", "cp", str(metadata_path), destination + "/snapshot.json",
                                    "--only-show-errors"], capture_output=True, text=True)
                            if completed.returncode != 0:
                                s3_available = False
                                metadata.update(storage_status="server_preserved_s3_pending",
                                                is_offhost=False, s3_error=completed.stderr[-4000:])
                                write_json(metadata_path, metadata)
                                logging.warning("New checkpoint preserved on server; S3 pending: %s",
                                                completed.stderr[-1000:])
                        if metadata["storage_status"] == "s3_uploaded":
                            write_json(docs_root / "latest_uploaded_checkpoint.json", metadata)
                            pointer = subprocess.run([
                                "aws", "s3", "cp", str(metadata_path),
                                cell_prefix + "/latest_checkpoint.json", "--only-show-errors"],
                                capture_output=True, text=True)
                            if pointer.returncode != 0:
                                s3_available = False
                                logging.warning("Snapshot uploaded; latest S3 pointer pending: %s",
                                                pointer.stderr[-1000:])
                        write_json(docs_root / "latest_preserved_checkpoint.json", metadata)
                    else:
                        subprocess.run(["bash", str(ROOT / "scripts/upload_checkpoint.sh"),
                                        str(checkpoint_path), str(response_snapshot), str(docs_root),
                                        str(args.log_dir), str(metadata_path), destination,
                                        cell_prefix + "/latest_checkpoint.json",
                                        str(final_manifest) if final_manifest else ""], check=True)
                        write_json(docs_root / "latest_uploaded_checkpoint.json", metadata)
                except Exception as error:
                    backup_failed = True
                    raise PortableStop(f"Checkpoint preserved locally; S3 backup incomplete: {error}") from error
                logging.info("Checkpoint year %s saved: %s, %s", year,
                             metadata.get("storage_status", "s3_uploaded"), destination)
                return metadata
            args.checkpoint_callback = upload_checkpoint
            with (run_context() if run_context else nullcontext()):
                cls = (simulation_factory or simulation_class)(Simulation)
                sim = cls(llm=llm, args=args, mechanisms=Mechanisms(**args.funding_feedback),
                          num_years=args.num_years, output_dir=args.output_dir,
                          **{key: getattr(args, key) for key in (
                              "verbose", "debug", "industry_funding_mode", "funding_allocation_mode",
                              "always_rerun", "use_langchain", "max_retries", "weighted_funding_assignment",
                              "experiment_name", "collaboration_mode", "collaboration_network_growth_mode", "start_year")})
                manifest["status"] = "running"
                write_json(manifest_path, manifest)
                sim.run()
            if len(sim.founder_ids) != options.founders or set(sim.funding_feedback_years) != set(range(1, options.years + 1)):
                raise RuntimeError("The declared founder count/full horizon was not completed")
            if not handle.summary()["completion_allowed"]:
                raise RuntimeError("Funding validation did not permit completion")
        manifest.update(status="complete", years_completed=options.years,
                        initial_world_hash=sim.initial_world_hash, corpus_hash=sim.corpus_hash,
                        fallback_audit=fallback_audit(sim))
        manifest["request_audit"] = audit.summary()
        manifest["funding_validation"] = handle.summary()
        pending_upload = dict(manifest, status="completed_pending_upload")
        write_json(session / "result.json", pending_upload)
        write_json(manifest_path, pending_upload)
        # The final native report is produced after the last annual checkpoint.
        receipt = upload_checkpoint(Path(args.checkpoint_dir) / f"portable_checkpoint_year_{options.years}.json.gz", final=True)
        if source_info.get("continuation"):
            manifest["final_preserved_snapshot"] = receipt["server_directory"]
            manifest["storage_status"] = receipt["storage_status"]
            if receipt["storage_status"] == "s3_uploaded":
                manifest["final_uploaded_snapshot"] = receipt["destination"]
        else:
            manifest["final_uploaded_snapshot"] = receipt["destination"]
    except PortableStop as error:
        manifest.update(status="paused", reason=str(error),
                        years_completed=getattr(sim, "_portable_last_year", 0))
        raise
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc(),
                        years_completed=getattr(sim, "_portable_last_year", 0))
        raise
    finally:
        handle.restore()
        manifest["funding_validation"] = handle.summary()
        if audit is not None:
            manifest["request_audit"] = audit.summary()
        write_json(session / "result.json", manifest)
        write_json(manifest_path, manifest)
        if llm is not None:
            try:
                if manifest.get("status") != "complete" and upload_checkpoint is not None and not backup_failed:
                    checkpoints = list(Path(args.checkpoint_dir).glob("portable_checkpoint_year_*.json.gz"))
                    if checkpoints:
                        latest = max(checkpoints, key=lambda q: int(q.name.removeprefix("portable_checkpoint_year_").removesuffix(".json.gz")))
                        try:
                            upload_checkpoint(latest)
                        except PortableStop as error:
                            manifest["backup_error"] = str(error)
            finally:
                llm.response_cache.close()
        write_json(session / "result.json", manifest)
        write_json(manifest_path, manifest)
    return manifest


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
    prospective_arguments(commands.add_parser("run-fulltext", help="New pinned full-text portable protocol"))
    commands.add_parser("analyze")
    commands.add_parser("queue-plan", help="Print declarative four-job plan without executing it")
    commands.add_parser("audit-cache", help="CPU-only provenance check of immutable original caches")
    options = parser.parse_args(argv)
    if options.command == "run-fulltext":
        if options.cell not in CELLS:
            parser.error("P/F cell must be one of P1F1, P0F1, P1F0, P0F0")
        run_prospective(options)
    elif options.command == "plan":
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
