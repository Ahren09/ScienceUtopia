#!/usr/bin/env python3
"""Bounded, fail-closed queue. Importing this module never starts a process.

The immutable JSON plan has schema_version=1, seed=42, a unique queue_id,
absolute worktree and private_root, and exactly eight jobs (four per family).
Alternatively, propensity_initialization={job_id, artifact} declares exactly
one propensity_init job and four propensity jobs with cell LN/LF/HN/HF.
The owner's native form is also accepted: family=propensity for all five,
role=initialize/condition, cell=null for initialization, and depends_on=[]
for initialization or [initialization job_id] for each world. No other graph
or dependency is supported.
The artifact is the producer's fresh immutable initial_choices.json; every
world passes that exact path with --initial-choices. Initialization is a
treatment-blind cache operation, not a fifth world. Only its job is runnable
until its completion receipt passes the artifact/source gates. One server
is allowed until then; afterwards up to four share the existing UUID locks.
The amended propensity v3 mode keeps this five-job graph and additionally
requires sequential_remaining_ids_v1 for all five operations, including
initializer server readiness. Both propensity and sequential protocol documents
are frozen. A prior compact-probe receipt cannot authorize an amended job.
The confirmed installation is:
  worktree: /home/ubuntu/Workspace/PeerReview/ScienceUtopia-llm
  private_root: /opt/dlami/nvme/ubuntu/scienceutopia_llm_james_20260918
Python and HF_HOME default to private_root/env/bin/python and private_root/hf.
Each job contains:
  id, family ("influx" or "mechanisms"), seed (42),
  command: [<private Python>, "src/owner_script.py", ..., "--seed", "42",
            <URL flag>, "__SERVER_URL__", <manifest flag>, "__SERVER_MANIFEST__"],
  env: {}, require_manifest_argument: true,
  output_roots: [<absolute, unique, not-yet-existing output directory>],
  artifacts: [{"path": <absolute output file>, "min_bytes": 1,
               "equals": {"/status": "complete", "/seed": 42}}].
`equals` uses JSON Pointer and is mandatory on at least one artifact per job.
Other artifacts may be non-JSON files. Owners must specify their actual
completion predicates; the queue does not infer scientific completeness.
CANDIDATE infrastructure amendment, not an amendment to any frozen plan:
an influx checkpoint_year_1.json through checkpoint_year_10.json may declare
max_json_bytes (exact integer, 64–512 MiB) only as a direct child of its declared
run output root immediately below outputs, with the four native checkpoint
integer predicates. Without this field the JSON limit remains 64 MiB.
The authorized continuation kind v6_pending_six_v1 instead admits only the
six exact previously pending jobs from the pinned failed V6 parent. Four
path/hash references bind that parent plan, status, source freeze and approval.
It forbids resume, preserves A/B, and rechecks parent/source bindings before
writes, client launches and admission. Its completion is only six pending jobs.
Raw influx clients may set require_manifest_argument=false and omit the
manifest argv placeholder. JAMES_SERVER_MANIFEST is always supplied in the
environment; their receipt explicitly records parent-level attestation,
not a claim that run_simulation.py validates the server manifest.
The influx single-cell wrapper may additionally use the standalone values
__SERVER_GPUS__ for --server_gpus and __CLIENT_GPU__ for --client_gpu. They
expand to the physical pair and its first GPU. No other substitutions exist.
Optional input_hashes maps absolute, immutable input files to SHA256 strings.
Mandatory funding_probe_fixture supplies an absolute path and SHA256 for an
immutable producer-authored funding_panel25 fixture. Its 8192-token live probe
must also pass before any client starts (generic nonce probe 120s). Fresh v6
plans additionally require funding_selection_protocol=sequential_remaining_ids_v1
and eight fresh v6 jobs. The sequential panel probe makes 25 direct urllib POSTs,
including the singleton, with one HTTP attempt per step and no SDK or retries.
Each request has a socket timeout and SIGALRM deadline of at most 600 seconds.
The aggregate budget is 15000 seconds, including a private-Python CPU tokenizer
sidecar (startup at most 120s, each count at most 30s, capped by the aggregate).
The shared production helpers construct messages, schema and seed; this is
payload parity, not parity with the production SDK's internal retry stack.
Every step persists its request, prefix, remaining IDs, schema, seed, raw reply,
finish reason and exact token counts. Readiness, launch and completion recheck
these files. The protocol markdown is also part of the frozen source hashes.
Legacy plans without a selection protocol retain their single-request funding
probe and 600s bound. Sequential propensity v3 uses five fresh v3 IDs; it never
adopts an old initializer, world, or already-running server from another queue.
Optional server fields must equal SERVER_SPEC; overrides are not accepted.
The sole alternative is server_profile=propensity_tp1_v1 for the sequential
propensity family, with fresh v3tp1 IDs. Its fixed contract is shared in
james_server_profiles.py: TP1, BF16, 32768 context, 16 sequences, .94 memory,
eager execution and chunked prefill with a 2048-token batch budget. Default TP2
settings are unchanged. Every server in one plan uses the same selected profile
and must pass its own nonce and full 25-step sequential probes. One-device
allocation uses the same UUID locks, reservations and ownership checks.

--validate-only and --dry-run are offline and read-only: they parse the plan,
and do not inspect GPUs, create directories, or start a server. --run is an
explicit separate mode. Only --run --resume may recover a previously stopped
queue: the exact plan and source hashes must match, all completed artifacts
must match their receipts, and no job may have started without completing.
No simulator checkpoint resume, retries, or historical artifact adoption.

The per-GPU flock is advisory, not a CUDA reservation. Rechecks detect other
compute processes; an uncooperative process can still race the final check.
We stop only our verified process sessions, never a PID learned from nvidia-smi.
Durable intent is fsynced before a fork. Unreleased same-boot intent remains
quarantined even with a dead controller and no CUDA PID; an uncertain launch
requires explicit owner recovery, never automatic stale-lock deletion.
Artifacts reject symlinks below the sanctioned worktree/outputs alias, must
belong to their job's fresh roots, and bind launch time, inode, size, times and
content hash into receipts. All receipts, manifests and frozen source hashes
are revalidated at final success as well as resume.
Readiness additionally requires a real strict structured_outputs.json nonce
probe using the shared production builder, thinking=True, 2048 tokens and the
production seed derivation. A /models response alone can never start a client.
The server reserves .80 of device memory; the remaining memory is headroom,
not an enforced allocation guarantee for the client's GPU embedding model.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from james_server_profiles import TP1_PLAN_PROFILE, TP1_PROFILE_NAME, tp1_server_spec

MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
SERVER_SPEC = {
    "model": MODEL, "revision": REVISION, "dtype": "bfloat16",
    "tensor_parallel_size": 2, "max_model_len": 32768,
    "max_num_seqs": 128, "gpu_memory_utilization": 0.80,
    "seed": 42, "request_seed": 42, "request_seed_base": 42,
    "request_seed_policy": "derive_seed(run_seed, phase_context, item_index, retry)",
    "reasoning_parser": "qwen3",
    "vllm_version": "0.12.0", "transformers_version": "4.57.3",
    "torch_version": "2.9.0+cu128",
}
POLL_SECONDS = 30
TIMEOUT_SECONDS = 72 * 60 * 60
STARTUP_SECONDS = 30 * 60
LOW_UTILIZATION = 5
PROBE_TIMEOUT_SECONDS = 120.0
PROBE_MAX_RESPONSE_BYTES = 64 * 1024
PROBE_MAX_TOKENS = 2048
PROBE_SEED_COMPONENTS = (42, "structured_output_probe", "preflight", 0, 0)
FUNDING_PROBE_TIMEOUT_SECONDS = 600.0
FUNDING_PROBE_MAX_TOKENS = 8192
FUNDING_PROBE_MAX_RESPONSE_BYTES = 256 * 1024
FUNDING_FIXTURE_MAX_BYTES = 4 * 1024 * 1024
FUNDING_SEED_COMPONENTS = (42, "funding_panel25_preflight", 1, 0, 0)
SEQUENTIAL_PROTOCOL = "sequential_remaining_ids_v1"
SEQUENTIAL_PROBE_SECONDS = 25 * FUNDING_PROBE_TIMEOUT_SECONDS
TOKENIZER_STARTUP_SECONDS = 120.0
TOKENIZER_COUNT_SECONDS = 30.0
TOKENIZER_REPLY_BYTES = 64 * 1024
SEQUENTIAL_PROTOCOL_FILE = "docs/james_sequential_funding_protocol.md"
DEFAULT_ARTIFACT_JSON_BYTES = 64 * 1024 * 1024
MAX_CHECKPOINT_JSON_BYTES = 512 * 1024 * 1024
CHECKPOINT_JSON_YEARS = {f"checkpoint_year_{year}.json": year for year in range(1, 11)}
PENDING_KIND = "v6_pending_six_v1"
PENDING_QUEUE_ID = "james_pending_seed42_n1200_20260921_r1"
PENDING_JOB_IDS = (
    "influx_james1202_v6_C_seed42", "influx_james1202_v6_D_seed42",
    "james_mechanisms_v6_n1200_y6_P1F1_seed42", "james_mechanisms_v6_n1200_y6_P0F1_seed42",
    "james_mechanisms_v6_n1200_y6_P1F0_seed42", "james_mechanisms_v6_n1200_y6_P0F0_seed42",
)
PENDING_PARENT_SHA256 = {
    "parent_plan": "382dbb5168196c2b6adb15f2ed3ef9bb273389d7c816b65e6057ee524ed9e3e0",
    "parent_status": "5cb068350039d8dd3d2a1a581d5e5a591721c45992faefe314dc10c96ce9f4cf",
    "parent_source_freeze": "e692b66b13347faa1b6024fdd01fbed531bd00043675a602121b12a0a20bdd4b",
    "authorization": "bc82df78b9eb570b694b413d9fabafc124427634185761a157b73c767cbe762c",
}
PENDING_RECORD_PATHS = {
    "parent_plan": "outputs/logs/ec2_preparation/james_ec2_queue_plan_v6.json",
    "parent_status": "outputs/logs/james_ec2_queue/james_seed42_n1200_20260919_v6/status.json",
    "parent_source_freeze": "outputs/logs/ec2_preparation/v6_source_freeze.json",
    "authorization": "outputs/logs/ec2_preparation/pending6_execution_authorization_20260921.json",
}
PENDING_PROPOSAL_SHA256 = "cf888fa2a056257dae27916c86498de3af8227a9fb4f6444c8ef5274a78fe4ce"
PENDING_SCOPE = (
    "Only six previously unstarted V6 jobs. A/B remain excluded and unresolved. "
    "Completion means this pending subset only, never FIRST8 or all twelve conditions."
)
PROPENSITY_SEQUENTIAL_PROTOCOL = "james-propensity-v3-single-world-sequential"
PROPENSITY_SOURCE_FILES = ("src/james_funding_sequential.py",
                          SEQUENTIAL_PROTOCOL_FILE, "docs/james_propensity_protocol.md")
LIMITATION = (
    "Each four-cell family is a single seed-42 conditional simulated world. "
    "Researchers, papers, years, and cells are dependent observations, not "
    "independent world replications. Paired contrasts describe this initialized "
    "world under these interventions; shared seeds do not ensure identical "
    "random draws after trajectories or request order diverge. No between-world "
    "confidence interval, population-level causal generalization, or external "
    "validity claim is supported by this queue. Archived multi-world reanalysis "
    "is separate evidence and is not reused as a new experiment."
)
PROTECTED_ENV = {
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTHONHASHSEED", "HF_HOME",
    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
    "PYTHONPATH", "LD_PRELOAD", "VLLM_HOST_IP", "VLLM_PORT",
    "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_CACHE_ROOT", "TMPDIR",
    "HOME", "CODEX_HOME", "JAMES_SERVER_MANIFEST", "JAMES_SERVER_URL",
    "JAMES_QUEUE_PLAN_SHA256", "JAMES_QUEUE_ID", "XDG_CACHE_HOME", "TORCH_HOME",
}
PLACEHOLDERS = {"__SERVER_URL__", "__SERVER_MANIFEST__", "__SERVER_GPUS__", "__CLIENT_GPU__"}


class QueueError(RuntimeError):
    pass


class ProbeError(QueueError):
    def __init__(self, message: str, attestation: dict):
        super().__init__(message)
        self.attestation = attestation


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QueueError(message)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp,
                                  timezone.utc).isoformat()


def absolute(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value.startswith("/") and "\0" not in value,
            f"{label} must be an absolute path")
    path = Path(value)
    require(".." not in path.parts, f"{label} may not contain '..'")
    return path


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def read_json(raw: bytes) -> Any:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad_constant(value):
        raise QueueError(f"Invalid JSON numeric constant: {value}")
    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=bad_constant)
    except (ValueError, UnicodeError) as error:
        raise QueueError(f"Invalid JSON: {error}") from error


def artifact_json_limit(job: dict, artifact: dict, *, worktree: Path,
                        private_root: Path) -> int:
    """Plan-bound exception for the ten exact native influx checkpoint names."""
    if "max_json_bytes" not in artifact:
        return DEFAULT_ARTIFACT_JSON_BYTES
    limit = artifact["max_json_bytes"]
    require(type(limit) is int and
            DEFAULT_ARTIFACT_JSON_BYTES <= limit <= MAX_CHECKPOINT_JSON_BYTES,
            "max_json_bytes must be an exact integer between 64 and 512 MiB")
    path = absolute(artifact.get("path"), "artifact path")
    roots = {absolute(root, "output_root") for root in job["output_roots"]}
    require(job.get("family") == "influx" and path.name in CHECKPOINT_JSON_YEARS
            and path.parent in roots
            and path.parent.parent in {private_root / "outputs", worktree / "outputs"},
            "max_json_bytes is restricted to named influx checkpoints directly in a run output_root")
    required = {"/initial_industry_count": 2, "/initial_university_count": 1200,
                "/phase": 5, "/year": CHECKPOINT_JSON_YEARS[path.name]}
    equals = artifact.get("equals")
    require(isinstance(equals, dict) and
            all(type(equals.get(key)) is int and equals[key] == value
                for key, value in required.items()),
            "Checkpoint max_json_bytes requires all four exact native integer predicates")
    return limit


@dataclass(frozen=True)
class Plan:
    raw: bytes
    spec: dict
    sha256: str
    worktree: Path
    private_root: Path
    python: Path
    hf_home: Path

    @property
    def directory(self) -> Path:
        return self.private_root / "outputs/logs/james_ec2_queue" / self.spec["queue_id"]

    @property
    def jobs(self) -> list[dict]:
        return self.spec["jobs"]

    @property
    def initialization(self) -> dict | None:
        return self.spec.get("propensity_initialization")


def selected_server_spec(plan: Plan | dict) -> dict:
    """Only the unchanged TP2 default or the one explicit propensity TP1 profile."""
    spec = plan.spec if isinstance(plan, Plan) else plan
    if "server_profile" not in spec:
        return dict(SERVER_SPEC)
    require(spec.get("server_profile") == TP1_PLAN_PROFILE, "Unknown server_profile")
    require(isinstance(spec.get("propensity_initialization"), dict)
            and spec.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL,
            "TP1 profile is restricted to sequential propensity with initialization")
    return tp1_server_spec()


def selected_runtime_binding(plan: Plan) -> dict:
    selected = selected_server_spec(plan)
    return {"server_profile": plan.spec["server_profile"],
            "selected_server_spec": selected,
            "runtime_hash": digest_bytes(json_bytes(selected))}


def validate_selected_runtime(plan: Plan, provenance: dict, manifest: dict | None = None) -> None:
    """Bind the explicit profile through freeze, readiness, launch and completion."""
    if "server_profile" not in plan.spec:
        return
    binding = selected_runtime_binding(plan)
    require(all(provenance.get(key) == value for key, value in binding.items()),
            "Selected runtime profile differs from frozen provenance")
    if manifest is None:
        return
    selected = binding["selected_server_spec"]
    require(manifest.get("server_profile") == binding["server_profile"]
            and manifest.get("runtime_hash") == binding["runtime_hash"]
            and all(type(manifest.get(key)) is type(value) and manifest[key] == value
                    for key, value in selected.items()),
            "Server manifest differs from the selected runtime profile")
    count = selected["tensor_parallel_size"]
    ids, uuids, devices = (manifest.get(name) for name in ("gpu_ids", "gpu_uuids", "gpus"))
    require(isinstance(ids, list) and len(ids) == count
            and all(type(index) is int and 0 <= index < 8 for index in ids) and len(set(ids)) == count
            and isinstance(uuids, list) and len(uuids) == count
            and all(isinstance(value, str) and value.startswith("GPU-") for value in uuids)
            and len(set(uuids)) == count and isinstance(devices, list) and len(devices) == count
            and all(isinstance(device, dict) for device in devices)
            and [(g.get("index"), g.get("uuid")) for g in devices] == list(zip(ids, uuids)),
            "Server manifest GPU count/identity differs from selected profile")
    match = re.fullmatch(r"http://127\.0\.0\.1:(\d+)/v1", manifest.get("endpoint", ""))
    require(match is not None and manifest.get("launch_command") ==
            manifest.get("command") == server_command(plan, int(match[1])),
            "Server launch command differs from selected profile")
    require(manifest.get("server_gpus") == ",".join(map(str, ids))
            and manifest.get("client_gpu") == str(ids[0]) and manifest.get("client_gpu_uuid") == uuids[0]
            and manifest.get("environment", {}).get("CUDA_VISIBLE_DEVICES") == ",".join(map(str, ids))
            and manifest["environment"].get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "Server CUDA mask differs from selected physical GPU profile")


def validate_live_selection(plan: Plan, provenance: dict, server, manifest: dict | None = None) -> None:
    if "server_profile" not in plan.spec:
        return
    validate_selected_runtime(plan, provenance, manifest)
    require(len(server.pair) == selected_server_spec(plan)["tensor_parallel_size"]
            and len({gpu.uuid for gpu in server.pair}) == len(server.pair)
            and server.command == server_command(plan, server.port),
            "Live server device count/command differs from selected profile")
    if manifest is not None:
        require(manifest["gpu_ids"] == [gpu.index for gpu in server.pair]
                and manifest["gpu_uuids"] == [gpu.uuid for gpu in server.pair]
                and manifest["endpoint"] == server.url,
                "Live server differs from selected-profile manifest")


def is_propensity_initializer(job: dict) -> bool:
    return job["family"] == "propensity_init" or (
        job["family"] == "propensity" and job.get("role") == "initialize")


def is_propensity_world(job: dict) -> bool:
    return job["family"] == "propensity" and not is_propensity_initializer(job)


def propensity_manifest_artifact(job: dict) -> dict:
    filename = ("initialization_manifest.json" if is_propensity_initializer(job)
                else "propensity_manifest.json")
    matches = [artifact for artifact in job["artifacts"] if Path(artifact["path"]).name == filename]
    require(len(matches) == 1, f"{job['id']}: requires exactly one {filename} artifact")
    expected = {"/status": "complete", "/seed": 42, "/founder_count": 1200}
    expected.update({"/kind": "initialization_only", "/choice_count": 1200, "/scientific_years_completed": 0}
                    if is_propensity_initializer(job)
                    else {"/kind": "condition", "/cell": job["cell"], "/years_completed": 10})
    for key, value in expected.items():
        actual = matches[0].get("equals", {}).get(key)
        require(type(actual) is type(value) and actual == value,
                f"{job['id']}: completion gate must require {key}={value!r}")
    return matches[0]


def propensity_audit_artifacts(job: dict) -> dict:
    docs = Path(propensity_manifest_artifact(job)["path"]).parent
    predicates = {
        "llm_request_audit.summary.json": {
            "/status": "complete", "/n_clients": 1, "/guard_failures": 0, "/pending_requests": 0},
        "funding_validation.summary.json": {
            "/completion_allowed": True, "/installed": False,
            "/output_representation": "ordered_application_ids_v1",
            "/failed_batches": 0, "/audit_failures": 0, "/invalid_final_panels": 0,
            "/processing_failures": 0, "/unprocessed_panels": 0, "/imputed_rankings": 0,
            "/fallback_rankings": 0, "/restore_conflicts": []}}
    if is_propensity_initializer(job):
        predicates["funding_validation.summary.json"]["/final_panels"] = 0
    result = {}
    for name in (*predicates, "llm_request_audit.jsonl", "funding_validation.jsonl"):
        matches = [artifact for artifact in job["artifacts"] if Path(artifact["path"]) == docs / name]
        require(len(matches) == 1, f"{job['id']}: requires declared audit artifact {name}")
        artifact = result[name] = matches[0]
        for key, value in predicates.get(name, {}).items():
            actual = artifact.get("equals", {}).get(key)
            require(type(actual) is type(value) and actual == value,
                    f"{job['id']}: {name} must require {key}={value!r}")
        if name == "funding_validation.jsonl":
            require(artifact.get("min_bytes", 1) == 0 and not artifact.get("equals"),
                    "Funding JSONL must be declared with min_bytes=0 for valid zero-application/init runs")
    return result


def propensity_sequential_artifacts(job: dict) -> dict:
    """Additional fixed v3 gates; preserve every original propensity predicate."""
    manifest = propensity_manifest_artifact(job)
    docs = Path(manifest["path"]).parent
    gates = manifest.get("equals", {})
    for key, value in {
        "/protocol": PROPENSITY_SEQUENTIAL_PROTOCOL,
        "/funding_selection_protocol": SEQUENTIAL_PROTOCOL,
        "/sequential_audit_valid": True, "/funding_application_evidence_valid": True,
    }.items():
        require(type(gates.get(key)) is type(value) and gates[key] == value,
                f"{job['id']}: amended manifest must require {key}={value!r}")
    binding = gates.get("/sequential_source_binding")
    full_source = gates.get("/inputs/source_files_sha256")
    require(isinstance(full_source, dict) and set(PROPENSITY_SOURCE_FILES) <= set(full_source)
            and all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                    for value in full_source.values())
            and binding == {name: full_source[name] for name in PROPENSITY_SOURCE_FILES[:2]},
            "Propensity v3 requires the helper and both protocols in its exact source map")
    years = 0 if is_propensity_initializer(job) else 10
    required = {
        "/funding_selection_protocol": SEQUENTIAL_PROTOCOL,
        "/output_representation": "ordered_application_ids_v1",
        "/installed": False, "/status": "complete", "/completion_allowed": True,
        "/failed_panels": 0, "/cancelled_panels": 0, "/unprocessed_panels": 0,
        "/fatal_errors": 0, "/guard_failures": 0, "/processing_failures": 0,
        "/audit_failures": 0, "/restore_conflicts": [],
        "/years_started": years, "/years_completed": years, "/agency_registrations": 2 * years,
    }
    if is_propensity_initializer(job):
        required.update({"/" + key: 0 for key in (
            "registered_panels", "processed_panels", "accepted_steps", "sdk_calls")})
    result = {}
    for name in ("funding_sequential.jsonl", "funding_sequential.summary.json"):
        matches = [artifact for artifact in job["artifacts"] if Path(artifact["path"]) == docs / name]
        require(len(matches) == 1 and matches[0].get("min_bytes", 1) >= 1,
                f"{job['id']}: requires nonempty {name}, including initialization")
        result[name] = matches[0]
    predicates = result["funding_sequential.summary.json"].get("equals", {})
    require(all(type(predicates.get(key)) is type(value) and predicates[key] == value
                for key, value in required.items()),
            f"{job['id']}: sequential summary must require clean, role-specific year/agency gates")
    return result


def parse_plan(raw: bytes) -> Plan:
    spec = read_json(raw)
    require(isinstance(spec, dict), "Plan must be an object")
    allowed = {"schema_version", "seed", "queue_id", "worktree", "private_root",
                "python", "hf_home", "server", "input_hashes", "jobs", "funding_probe_fixture",
                "propensity_initialization", "funding_selection_protocol", "server_profile",
                "continuation"}
    require(not set(spec) - allowed, f"Unknown plan keys: {set(spec) - allowed}")
    require(spec.get("schema_version") == 1 and type(spec.get("seed")) is int
            and spec["seed"] == 42, "Plan requires schema_version=1 and seed=42")
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",
                             spec.get("queue_id", ""))), "Unsafe queue_id")
    worktree = absolute(spec.get("worktree"), "worktree")
    private = absolute(spec.get("private_root"), "private_root")
    python = absolute(spec.get("python", str(private / "env/bin/python")), "python")
    hf_home = absolute(spec.get("hf_home", str(private / "hf")), "hf_home")
    require(python == private / "env/bin/python", "Python must be private_root/env/bin/python")
    require(hf_home == private / "hf", "HF_HOME must be private_root/hf")
    require(isinstance(spec.get("server", {}), dict), "server must be an object")
    selected = selected_server_spec(spec)
    for key, value in spec.get("server", {}).items():
        require(key in selected and value == selected[key]
                and ("server_profile" not in spec or type(value) is type(selected[key])),
                f"Unsupported server setting: {key}={value!r}")
    hashes = spec.get("input_hashes", {})
    require(isinstance(hashes, dict), "input_hashes must be an object")
    for name, value in hashes.items():
        absolute(name, "input_hashes path")
        require(isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value)),
                f"Invalid SHA256 for {name}")
    fixture = spec.get("funding_probe_fixture")
    require(isinstance(fixture, dict) and set(fixture) == {"path", "sha256"},
            "Mandatory funding_probe_fixture requires immutable path and sha256")
    absolute(fixture["path"], "funding_probe_fixture path")
    require(isinstance(fixture["sha256"], str) and bool(re.fullmatch("[0-9a-f]{64}", fixture["sha256"])),
            "Invalid funding_probe_fixture SHA256")
    jobs = spec.get("jobs")
    initialization = spec.get("propensity_initialization")
    continuation = spec.get("continuation")
    require(not (continuation is not None and "propensity_initialization" in spec),
            "Pending continuation cannot include propensity initialization")
    if "propensity_initialization" in spec:
        require(isinstance(initialization, dict) and set(initialization) == {"job_id", "artifact"},
                "propensity_initialization requires exactly job_id and artifact")
        require(isinstance(initialization["job_id"], str), "Initialization job_id must be a string")
        initial_path = absolute(initialization["artifact"], "initial_choices artifact")
        require(initial_path.name == "initial_choices.json", "Initialization artifact must be initial_choices.json")
        require(isinstance(jobs, list) and len(jobs) == 5,
                "Propensity requires one initialization job and exactly four worlds")
    if continuation is not None:
        require(isinstance(continuation, dict) and set(continuation) ==
                {"schema_version", "kind", *PENDING_PARENT_SHA256},
                "Invalid pending continuation descriptor")
        require(type(continuation["schema_version"]) is int
                and continuation["schema_version"] == 1 and continuation["kind"] == PENDING_KIND,
                "Unsupported pending continuation kind")
        require(spec["queue_id"] == PENDING_QUEUE_ID, "Pending continuation queue ID differs")
        for name, checksum in PENDING_PARENT_SHA256.items():
            record = continuation[name]
            require(isinstance(record, dict) and set(record) == {"path", "sha256"}
                    and record["sha256"] == checksum, f"Unapproved pending parent binding: {name}")
            path = absolute(record["path"], name)
            require(path == private / PENDING_RECORD_PATHS[name],
                    f"Pending parent record must use its original canonical path: {name}")
        require(isinstance(jobs, list) and
                [j.get("id") for j in jobs if isinstance(j, dict)] == list(PENDING_JOB_IDS)
                and len(jobs) == 6, "Pending continuation requires the exact ordered six jobs")
    elif not initialization:
        require(isinstance(jobs, list) and len(jobs) == 8, "Exactly eight jobs are required")
    selection_protocol = spec.get("funding_selection_protocol")
    require(selection_protocol in (None, SEQUENTIAL_PROTOCOL), "Unsupported funding selection protocol")
    new_propensity = any(isinstance(job, dict) and str(job.get("id", "")).startswith(
        ("james_propensity_v3_", "james_propensity_v3tp1_", "james_propensity_v3tp1p1_")) for job in jobs)
    require(not new_propensity or (initialization and selection_protocol == SEQUENTIAL_PROTOCOL),
            "Fresh propensity v3 jobs require initialization and sequential funding")
    if initialization and selection_protocol:
        versions = ("v3tp1", "v3tp1p1") if spec.get("server_profile") == TP1_PLAN_PROFILE else ("v3",)
        expected_sets = [
            {f"james_propensity_{version}_n1200_y10_{cell}_seed42"
             for cell in ("initial", "LN", "LF", "HN", "HF")}
            for version in versions]
        require(all(isinstance(job, dict) for job in jobs)
                and {job.get("id") for job in jobs} in expected_sets,
                "Sequential propensity scope requires five fresh IDs matching its selected profile")
    if not initialization:
        require(not any(isinstance(job, dict) and "_v6_" in str(job.get("id")) for job in jobs)
                or selection_protocol == SEQUENTIAL_PROTOCOL,
                "Fresh v6 jobs require the sequential funding selection protocol")
        if selection_protocol:
            require(all(isinstance(job, dict) and "_v6_" in str(job.get("id")) for job in jobs),
                    "Sequential first-eight scope requires eight fresh v6 job IDs")
    all_roots: list[Path] = []
    ids = set()
    families = ({"propensity_init": 0, "propensity": 0} if initialization
                else {"influx": 0, "mechanisms": 0})
    cells = set()
    for job in jobs:
        require(isinstance(job, dict), "Each job must be an object")
        require(not set(job) - {"id", "family", "seed", "command", "env", "cell", "role", "depends_on",
                                "output_roots", "artifacts", "require_manifest_argument"},
                "Unknown job keys")
        name = job.get("id", "")
        require(isinstance(name, str) and
                bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name))
                and name not in ids, "Job ids must be unique safe strings")
        ids.add(name)
        require(job.get("family") in families, f"{name}: unknown family")
        normalized_family = "propensity_init" if initialization and is_propensity_initializer(job) else job["family"]
        families[normalized_family] += 1
        if initialization:
            expected_role = "initialize" if is_propensity_initializer(job) else "condition"
            require(job.get("role", expected_role) == expected_role,
                    f"{name}: role must match initialization/condition")
            expected_dependencies = [] if is_propensity_initializer(job) else [initialization["job_id"]]
            dependencies = job.get("depends_on", expected_dependencies)
            require(isinstance(dependencies, list) and dependencies == expected_dependencies,
                    f"{name}: depends_on must specify only the fixed initialization dependency")
        else:
            require("role" not in job and "depends_on" not in job, "Dependencies are only supported for propensity")
        if is_propensity_world(job):
            require(job.get("cell") in {"LN", "LF", "HN", "HF"} and job["cell"] not in cells,
                    f"{name}: propensity worlds require unique LN/LF/HN/HF cells")
            cells.add(job["cell"])
        elif initialization:
            require(job.get("cell") is None, f"{name}: initialization has no treatment cell")
        else:
            require("cell" not in job, f"{name}: only propensity worlds declare cell")
        require(type(job.get("seed")) is int and job["seed"] == 42,
                f"{name}: only seed42 is authorized")
        cmd = job.get("command")
        require(isinstance(cmd, list) and len(cmd) >= 4
                and all(isinstance(x, str) and x and "\0" not in x for x in cmd),
                f"{name}: command must be an argv array")
        require(cmd[0] == str(python) and cmd[1].endswith(".py"),
                f"{name}: invoke private Python and a .py script directly; no shell/wrapper")
        script = Path(cmd[1])
        script = script if script.is_absolute() else worktree / script
        parent_mechanism = (continuation is not None and job["family"] == "mechanisms"
                            and Path(cmd[1]).is_absolute() and script.name == "james_mechanisms.py"
                            and script.parent.name == "src")
        require(".." not in script.parts and (within(script, worktree) or parent_mechanism),
                f"{name}: script must belong to the worktree")
        manifest_argument = job.get("require_manifest_argument", True)
        require(type(manifest_argument) is bool, "require_manifest_argument must be boolean")
        require(job["family"] != "mechanisms" or manifest_argument,
                f"{name}: mechanisms require a manifest argument")
        require(not initialization or manifest_argument, f"{name}: propensity requires a manifest argument")
        if initialization:
            initial_args = [i for i, arg in enumerate(cmd) if arg.split("=", 1)[0]
                            .replace("_", "-") == "--initial-choices"]
            cell_args = [i for i, arg in enumerate(cmd) if arg.split("=", 1)[0].replace("_", "-") == "--cell"]
            if is_propensity_world(job):
                require(cmd[2] == "run" and len(cell_args) == 1 and cmd[cell_args[0]] == "--cell"
                        and cell_args[0] + 1 < len(cmd) and cmd[cell_args[0] + 1] == job["cell"],
                        f"{name}: run --cell must match the declared world cell")
                require(len(initial_args) == 1 and cmd[initial_args[0]] == "--initial-choices"
                        and initial_args[0] + 1 < len(cmd)
                        and cmd[initial_args[0] + 1] == initialization["artifact"],
                        f"{name}: --initial-choices must pass the exact declared initialization artifact")
            else:
                require(cmd[2] == "initialize" and not cell_args
                        and name == initialization["job_id"] and not initial_args,
                        f"{name}: initialization must match job_id and may not consume an existing cache")
        placeholders = ["__SERVER_URL__"]
        if manifest_argument:
            placeholders.append("__SERVER_MANIFEST__")
        else:
            require("__SERVER_MANIFEST__" not in cmd,
                    f"{name}: manifest argv contradicts require_manifest_argument=false")
        for placeholder in placeholders:
            require(cmd.count(placeholder) == 1, f"{name}: needs one standalone {placeholder}")
            index = cmd.index(placeholder)
            require(index >= 3 and cmd[index - 1].startswith("--"),
                    f"{name}: {placeholder} must be an explicit argument value")
        for placeholder, flag in (("__SERVER_GPUS__", "--server-gpus"),
                                  ("__CLIENT_GPU__", "--client-gpu")):
            if placeholder in cmd:
                require(job["family"] == "influx" and cmd.count(placeholder) == 1,
                        f"{name}: GPU placeholders are single influx argument values")
                index = cmd.index(placeholder)
                require(index >= 3 and cmd[index - 1].replace("_", "-") == flag,
                        f"{name}: {placeholder} must follow {flag}")
        require(("__SERVER_GPUS__" in cmd) == ("__CLIENT_GPU__" in cmd),
                f"{name}: supply both controlled GPU placeholders")
        require(not any(re.search(r"__[A-Z][A-Z_]*__", x) and x not in PLACEHOLDERS for x in cmd),
                f"{name}: unknown or embedded placeholder")
        if job["family"] == "influx":
            for i, arg in enumerate(cmd):
                if arg.replace("_", "-") in {"--server-gpus", "--client-gpu"}:
                    wanted = "__SERVER_GPUS__" if "server" in arg else "__CLIENT_GPU__"
                    require(i + 1 < len(cmd) and cmd[i + 1] == wanted,
                            f"{name}: GPU arguments must use the queue's controlled placeholders")
        found_seed = False
        for i, arg in enumerate(cmd[2:], 2):
            if not arg.startswith("--"):
                continue
            flag = arg.split("=", 1)[0].replace("_", "-")
            require("resume" not in flag, f"{name}: simulator resume is prohibited")
            if "seed" in flag:
                value = arg.split("=", 1)[1] if "=" in arg else (
                    cmd[i + 1] if i + 1 < len(cmd) else "")
                require(value == "42", f"{name}: all seed arguments must equal 42")
                if flag in {"--seed", "--seeds"}:
                    found_seed = True
                    require("=" in arg or i + 2 == len(cmd) or cmd[i + 2].startswith("--"),
                            f"{name}: seed list must contain only 42")
        require(found_seed, f"{name}: explicit --seed 42 or --seeds 42 required")
        env = job.get("env")
        require(isinstance(env, dict), f"{name}: env must be an object")
        for key, value in env.items():
            require(isinstance(key, str) and bool(re.fullmatch("[A-Z_][A-Z_0-9]*", key))
                    and isinstance(value, str) and "\0" not in value, f"{name}: invalid env")
            require(key not in PROTECTED_ENV or
                    (key == "PYTHONHASHSEED" and value == "42"),
                    f"{name}: env may not override {key}")
            require("SEED" not in key or value == "42", f"{name}: env seed must equal42")
        roots = job.get("output_roots")
        require(isinstance(roots, list) and roots, f"{name}: output_roots required")
        roots = [absolute(x, "output_root") for x in roots]
        for root in roots:
            valid_roots = [private / "outputs", worktree / "outputs"]
            require(any(within(root, parent) and root != parent for parent in valid_roots),
                    f"{name}: output_roots must be private output subdirectories")
            # Normalize the two supported spellings before overlap checks.
            canonical = private / root.relative_to(worktree) if within(root, worktree) else root
            queue_parent = private / "outputs/logs/james_ec2_queue"
            require(not within(canonical, queue_parent) and not within(queue_parent, canonical),
                    f"{name}: job output may not overlap queue state")
            require(not any(within(canonical, other) or within(other, canonical)
                            for other in all_roots), f"{name}: overlapping job output_roots")
            all_roots.append(canonical)
        artifacts = job.get("artifacts")
        require(isinstance(artifacts, list) and artifacts, f"{name}: artifacts required")
        seen_artifacts = set()
        has_predicate = False
        for artifact in artifacts:
            require(isinstance(artifact, dict) and
                    not set(artifact) - {"path", "min_bytes", "equals", "max_json_bytes"},
                    "Invalid artifact")
            path = absolute(artifact.get("path"), "artifact path")
            require(path not in seen_artifacts and any(within(path, root) and path != root
                                                       for root in roots),
                    f"{name}: artifact must be unique and inside an output_root")
            seen_artifacts.add(path)
            size = artifact.get("min_bytes", 1)
            compact_raw = ("funding_validation_audit.jsonl" if job["family"] == "influx"
                           else "funding_validation.jsonl")
            allow_empty = (bool(initialization) or selection_protocol == SEQUENTIAL_PROTOCOL) and path.name == compact_raw
            require(type(size) is int and (size >= 1 or (size == 0 and allow_empty)),
                    "Artifact min_bytes must be positive except the declared compact raw audit")
            equals = artifact.get("equals", {})
            require(isinstance(equals, dict) and all(isinstance(k, str) and k.startswith("/")
                                                    for k in equals), "Use JSON Pointer equals")
            artifact_json_limit(job, artifact, worktree=worktree, private_root=private)
            has_predicate |= bool(equals)
        for artifact in artifacts:
            if artifact.get("min_bytes", 1) != 0:
                continue
            summary_name = ("funding_validation_summary.json" if job["family"] == "influx"
                            else "funding_validation.summary.json")
            summary_path = str(Path(artifact["path"]).with_name(summary_name))
            summaries = [a for a in artifacts if a["path"] == summary_path]
            clean = {"/completion_allowed": True, "/installed": False,
                     "/invalid_final_panels": 0, "/failed_batches": 0, "/audit_failures": 0,
                     "/output_representation": "ordered_application_ids_v1"}
            require(not artifact.get("equals") and len(summaries) == 1
                    and summaries[0].get("min_bytes", 1) >= 1
                    and all(type(summaries[0].get("equals", {}).get(k)) is type(v)
                            and summaries[0]["equals"][k] == v for k, v in clean.items()),
                    "Empty compact audit requires its nonempty clean summary predicates")
        require(has_predicate, f"{name}: at least one explicit JSON completion predicate required")
        if initialization:
            manifest = propensity_manifest_artifact(job)
            propensity_audit_artifacts(job)
            if selection_protocol:
                propensity_sequential_artifacts(job)
            if "server_profile" in spec:
                gates = manifest.get("equals", {})
                require(gates.get("/inputs/server_runtime_profile") == TP1_PROFILE_NAME
                        and "src/james_server_profiles.py" in gates.get("/inputs/source_files_sha256", {}),
                        "TP1 propensity requires its runtime profile and shared profile source binding")
                require(all(name in Path(root).parts for root in job["output_roots"]),
                        "TP1 propensity output roots must use the fresh v3tp1 job ID")
            if is_propensity_initializer(job):
                require(Path(manifest["path"]) == Path(initialization["artifact"]).with_name(
                    "initialization_manifest.json"), "Initialization manifest must accompany its choices cache")
    if initialization:
        require(families == {"propensity_init": 1, "propensity": 4}
                and cells == {"LN", "LF", "HN", "HF"}, "Propensity requires one init and four distinct worlds")
        init_job = next(job for job in jobs if is_propensity_initializer(job))
        require(sum(artifact["path"] == initialization["artifact"] for artifact in init_job["artifacts"]) == 1,
                "Initialization choices must be a declared artifact of the initialization job")
    else:
        require(families == ({"influx": 2, "mechanisms": 4} if continuation else
                             {"influx": 4, "mechanisms": 4}), "Incorrect family membership")
    return Plan(raw, spec, digest_bytes(raw), worktree, private, python, hf_home)


def pending_file_binding(path: Path) -> tuple[bytes, dict]:
    """Read a small parent record/source through a no-symlink descriptor walk."""
    path = absolute(str(path), "pending bound file")
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
                and info.st_size <= DEFAULT_ARTIFACT_JSON_BYTES, f"Unsafe pending bound file: {path}")
        binding = {**stat_binding(info), "uid": info.st_uid, "mode": info.st_mode,
                   "nlink": info.st_nlink}
        raw = stream.read(info.st_size + 1)
        after = os.fstat(stream.fileno())
        require(len(raw) == info.st_size and
                {**stat_binding(after), "uid": after.st_uid, "mode": after.st_mode,
                 "nlink": after.st_nlink} == binding,
                f"Pending bound file changed while reading: {path}")
    current = path.lstat()
    require({**stat_binding(current), "uid": current.st_uid, "mode": current.st_mode,
             "nlink": current.st_nlink} == binding, f"Pending bound path changed: {path}")
    return raw, {**binding, "sha256": digest_bytes(raw)}


def pending_repository(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=20,
                                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).strip()
    return {"head": git("rev-parse", "HEAD"), "main": git("rev-parse", "main"),
            "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain", "--untracked-files=no")}


def pending_alias(root: Path, private: Path) -> dict:
    path = root / "outputs"
    before = path.lstat()
    require(stat.S_ISLNK(before.st_mode) and before.st_uid == os.getuid()
            and path.resolve(strict=True) == private / "outputs",
            "Pending worktree outputs alias differs")
    target = os.readlink(path)
    dest = path.stat()
    require(stat_binding(path.lstat()) == stat_binding(before) and os.readlink(path) == target,
            "Pending outputs alias changed while capturing")
    return {**stat_binding(before), "mode": before.st_mode, "uid": before.st_uid,
            "target": target, "destination_device": dest.st_dev, "destination_inode": dest.st_ino,
            "destination_uid": dest.st_uid, "destination_mode": dest.st_mode}


def validate_pending_continuation(plan: Plan, expected: dict | None = None,
                                  *, fresh_jobs: list[dict] | None = None) -> dict | None:
    """Read-only parent/source binding. Never adopt parent job state or receipts."""
    descriptor = plan.spec.get("continuation")
    if descriptor is None:
        return None
    records, bindings = {}, {}
    for name in PENDING_PARENT_SHA256:
        path = Path(descriptor[name]["path"])
        raw, binding = pending_file_binding(path)
        require(binding["sha256"] == descriptor[name]["sha256"] == PENDING_PARENT_SHA256[name],
                f"Pending parent hash changed: {name}")
        records[name] = raw
        bindings[str(path)] = binding
    parent = parse_plan(records["parent_plan"])
    require(parent.spec.get("continuation") is None and parent.worktree != plan.worktree,
            "Pending queue must use a new worktree and original eight-job parent")
    status = read_json(records["parent_status"])
    freeze = read_json(records["parent_source_freeze"])
    authority = read_json(records["authorization"])
    require(authority.get("user_message") == "continue"
            and authority.get("proposal_sha256") == PENDING_PROPOSAL_SHA256,
            "Pending execution authorization mismatch")
    require(status.get("status") == "failed" and not status.get("cleanup_errors")
            and status.get("plan_sha256") == parent.sha256
            and freeze.get("plan_sha256") == parent.sha256
            and freeze.get("worktree") == str(parent.worktree),
            "Pending parent is not the bound terminal failed queue")
    require(set(status["jobs"]) == {j["id"] for j in parent.jobs}
            and tuple(j["id"] for j in parent.jobs if status["jobs"][j["id"]] == {"status": "pending"})
            == PENDING_JOB_IDS, "Parent pending membership or no-launch state differs")
    require(all(s.get("status") == "stopped" and s.get("cleanup_errors") == []
                for s in status["servers"].values()), "Parent cleanup is incomplete")
    for record in [*status["servers"].values(), *status["jobs"].values()]:
        if record.get("identity"):
            identity = record["identity"]
            require(not same_process(proc_identity(identity["pid"]), identity),
                    "Original parent process remains alive")
            if Path("/proc/sys/kernel/random/boot_id").exists():
                require(not session_members(identity), "Original parent session remains alive")
    wanted = read_json(json_bytes(parent.spec))
    wanted["queue_id"], wanted["worktree"] = plan.spec["queue_id"], str(plan.worktree)
    wanted["continuation"] = descriptor
    wanted["jobs"] = [j for j in wanted["jobs"] if j["id"] in PENDING_JOB_IDS]
    for job in wanted["jobs"][:2]:
        require([Path(a["path"]).name for a in job["artifacts"][:10]]
                == list(CHECKPOINT_JSON_YEARS), "Parent native checkpoint order differs")
        for artifact in job["artifacts"][:10]:
            artifact["max_json_bytes"] = MAX_CHECKPOINT_JSON_BYTES
    require(json_bytes(wanted) == json_bytes(plan.spec),
            "Pending plan differs from original science/jobs beyond the authorized capacity overlay")
    aliases = {str(root): pending_alias(root, plan.private_root) for root in
               (parent.worktree, plan.worktree)}
    repository = pending_repository(parent.worktree)
    require(repository == {"head": freeze["git_head"], "main": freeze["original_main_head"],
                           "branch": freeze["branch"], "status": ""},
            "Original parent repository changed")
    sources = freeze["source_files_sha256"]
    require(len(sources) == 87, "Original 87-source freeze is incomplete")
    for name, checksum in sorted(sources.items()):
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Invalid frozen relative source path")
        roots = [parent.worktree] if name == "src/run_james_ec2_queue.py" else [parent.worktree, plan.worktree]
        for root in roots:
            _, binding = pending_file_binding(root / name)
            require(binding["sha256"] == checksum, f"Original scientific source changed: {root / name}")
            bindings[str(root / name)] = binding
    require(Path(__file__).resolve() == plan.worktree / "src/run_james_ec2_queue.py",
            "Pending controller must execute from its separately frozen worktree")
    for name in ("src/run_james_ec2_queue.py", "src/test_james_pending_continuation.py",
                 "docs/james_pending_continuation.md"):
        _, bindings[str(plan.worktree / name)] = pending_file_binding(plan.worktree / name)
    require({str(root): pending_alias(root, plan.private_root) for root in
             (parent.worktree, plan.worktree)} == aliases, "Outputs alias changed during binding")
    require(pending_repository(parent.worktree) == repository, "Parent repository changed during binding")
    for name in PENDING_PARENT_SHA256:
        path = Path(descriptor[name]["path"])
        require(pending_file_binding(path)[1] == bindings[str(path)],
                f"Pending parent changed during binding: {name}")
    result = {"kind": PENDING_KIND, "descriptor_sha256": digest_bytes(json_bytes(descriptor)),
              "parent_repository": repository, "files": bindings, "outputs_aliases": aliases,
              "pending_job_ids": list(PENDING_JOB_IDS), "scope": PENDING_SCOPE}
    if expected is not None:
        require(result == expected, "Pending continuation binding changed since startup")
    for job in fresh_jobs or []:
        require(job in plan.jobs, "Freshness check requested for an unauthorized job")
        ensure_fresh(job, plan)
    return result


def recheck_pending(plan: Plan, provenance: dict, *, fresh_jobs: list[dict] | None = None):
    if plan.spec.get("continuation") is not None:
        expected = provenance.get("continuation")
        require(isinstance(expected, dict), "Missing startup continuation binding")
        return validate_pending_continuation(plan, expected, fresh_jobs=fresh_jobs)
    return None


def queue_timeout_seconds(plan: Plan) -> int:
    """Fixed scope budget. No plan field, CLI override, resume, or extension."""
    return len(PENDING_JOB_IDS) * TIMEOUT_SECONDS if plan.spec.get("continuation") else TIMEOUT_SECONDS


def check_pending_client_deadline(plan: Plan, job_id: str, record: dict,
                                   *, checked_ns: int | None = None) -> int | None:
    """The immutable launch timestamp bounds every pending-mode client to 72h."""
    if not plan.spec.get("continuation"):
        return None
    started = record.get("started_epoch_ns")
    now = time.time_ns() if checked_ns is None else checked_ns
    require(job_id in PENDING_JOB_IDS and type(started) is int and started > 0
            and type(now) is int and now >= started, "Invalid pending client launch/deadline timestamp")
    require(now - started <= TIMEOUT_SECONDS * 1_000_000_000,
            f"Pending client {job_id} exceeded its 72-hour deadline; no retry or admission")
    return now


def job_script_root(plan: Plan, job: dict) -> tuple[Path, Path]:
    script = Path(job["command"][1])
    script = script if script.is_absolute() else plan.worktree / script
    if within(script, plan.worktree):
        return script, plan.worktree
    descriptor = plan.spec.get("continuation")
    require(descriptor is not None and job["family"] == "mechanisms",
            "External script is not a pending parent mechanism")
    raw, binding = pending_file_binding(Path(descriptor["parent_plan"]["path"]))
    require(binding["sha256"] == PENDING_PARENT_SHA256["parent_plan"],
            "Parent plan changed while resolving original mechanism command")
    parent = parse_plan(raw)
    original = next((j for j in parent.jobs if j["id"] == job["id"]), None)
    require(original is not None and job["command"] == original["command"]
            and script == parent.worktree / "src/james_mechanisms.py",
            "Pending external script must be the exact original parent mechanism command")
    return script, parent.worktree


def server_command(plan: Plan, port: int) -> list[str]:
    selected = selected_server_spec(plan)
    command = [str(plan.python), str(plan.python.parent / "vllm"), "serve", MODEL,
            "--revision", REVISION, "--tokenizer-revision", REVISION,
            "--dtype", "bfloat16", "--tensor-parallel-size", str(selected["tensor_parallel_size"]),
            "--distributed-executor-backend", "mp", "--max-model-len", "32768",
            "--max-num-seqs", str(selected["max_num_seqs"]),
            "--gpu-memory-utilization", f'{selected["gpu_memory_utilization"]:.2f}',
            "--seed", "42", "--reasoning-parser", "qwen3",
            "--host", "127.0.0.1", "--port", str(port)]
    if plan.spec.get("server_profile") == TP1_PLAN_PROFILE:
        command.extend(["--enforce-eager", "--enable-chunked-prefill",
                        "--max-num-batched-tokens", str(selected["max_num_batched_tokens"])])
    return command


def atomic_write(path: Path, value: bytes, *, exclusive: bool = False,
                 readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".queue-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            if readonly:
                os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)  # O_EXCL semantics with an already-fsynced file.
            os.unlink(temporary)
        else:
            require(not path.is_symlink(), f"Refusing symlink status file: {path}")
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Lock:
    """Locks are never unlinked: unlinking creates split-brain lock inodes."""
    def __init__(self, path: Path):
        self.path, self.fd = path, None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            require(info.st_uid == os.getuid() and stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1 and not info.st_mode & 0o022,
                    f"Unsafe lock file: {self.path}")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        return True

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def private_directory(path: Path, root: Path | None = None) -> None:
    if root is not None:
        require(within(path.resolve(), root.resolve()), f"Directory escapes private root: {path}")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid()
            and not info.st_mode & 0o022, f"Unsafe private directory: {path}")


def on_nvme(path: Path) -> bool:
    """Resolve direct partitions and Linux md/dm devices backed entirely by NVMe."""
    device = path.stat().st_dev
    node = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}")
    def check(entry: Path, seen: set[Path]) -> bool:
        resolved = entry.resolve()
        if resolved in seen or not resolved.exists():
            return False
        if re.fullmatch(r"nvme\d+n\d+(p\d+)?", resolved.name):
            return True
        children = list((resolved / "slaves").glob("*"))
        return bool(children) and all(check(x, seen | {resolved}) for x in children)
    return check(node, set())


def source_hashes(plan: Plan) -> dict:
    """Hash executable/config sources, including untracked owner helpers."""
    result = {}
    exclude = {".git", "outputs", "data", "__pycache__", ".venv", "venv", "env",
               ".pytest_cache", "node_modules", ".mypy_cache"}
    for parent, dirs, files in os.walk(plan.worktree):
        dirs[:] = sorted(x for x in dirs if x not in exclude)
        for name in sorted(files):
            path = Path(parent) / name
            if path.suffix in {".py", ".yaml", ".yml", ".toml", ".json", ".txt"}:
                require(within(path.resolve(), plan.worktree.resolve()),
                        f"Source symlink escapes worktree: {path}")
                result[str(path.relative_to(plan.worktree))] = file_hash(path)
    for name, expected in plan.spec.get("input_hashes", {}).items():
        observed = file_hash(Path(name))
        require(observed == expected, f"Input hash mismatch: {name}")
        result["input:" + name] = observed
    result["queue_controller"] = file_hash(Path(__file__))
    if plan.initialization:
        protocol = plan.worktree / "docs/james_propensity_protocol.md"
        require(protocol.is_file() and within(protocol.resolve(), plan.worktree.resolve()),
                "Propensity protocol is missing or escapes the new worktree")
        result["docs/james_propensity_protocol.md"] = file_hash(protocol)
        if plan.spec.get("funding_selection_protocol"):
            # The legacy directory exclusion skips every directory named data,
            # including executable tracker/RAG modules in utopia/data. The
            # amended driver's exact source_identity includes all utopia code.
            for path in sorted((plan.worktree / "utopia").rglob("*.py")):
                require(within(path.resolve(), plan.worktree.resolve()),
                        f"Propensity implementation source escapes worktree: {path}")
                result[str(path.relative_to(plan.worktree))] = file_hash(path)
    if plan.spec.get("funding_selection_protocol"):
        protocol = plan.worktree / SEQUENTIAL_PROTOCOL_FILE
        require(protocol.is_file() and within(protocol.resolve(), plan.worktree.resolve()),
                "Sequential funding protocol is missing or escapes the worktree")
        result[SEQUENTIAL_PROTOCOL_FILE] = file_hash(protocol)
    _, fixture_binding = load_funding_fixture(plan)
    result["funding_probe_fixture:" + fixture_binding["path"]] = fixture_binding["sha256"]
    return result


def runtime_guard(plan: Plan) -> dict:
    require(sys.platform == "linux" and hasattr(os, "pidfd_open")
            and hasattr(signal, "pidfd_send_signal"), "Linux with pidfd support is required")
    worktree = plan.worktree.resolve(strict=True)
    require(worktree == plan.worktree, "worktree must use its canonical path")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(worktree), *args],
                                       text=True, timeout=15).strip()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == worktree,
            "worktree is not a Git checkout root")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    require(branch and branch.lower() not in {"main", "master"}, "Refusing main/master/detached")
    if plan.spec.get("continuation"):
        require(branch == "codex/llm-james-pending6"
                and worktree.name == "ScienceUtopia-llm-pending6",
                "Pending continuation requires its named isolated worktree/branch")
    git_dir = Path(git("rev-parse", "--absolute-git-dir")).resolve()
    common = Path(git("rev-parse", "--git-common-dir"))
    common = (worktree / common).resolve() if not common.is_absolute() else common.resolve()
    require(git_dir != common, "An isolated linked worktree is required")
    private = plan.private_root.resolve(strict=True)
    require(private == plan.private_root, "private_root must use its canonical path")
    info = private.stat()
    require(info.st_uid == os.getuid() and not info.st_mode & 0o022,
            "private_root must be owned by this user and not group/world writable")
    require(on_nvme(private), "private_root is not verifiably backed entirely by NVMe")
    for name in ("outputs", "data"):
        target = (worktree / name).resolve(strict=True)
        require(target == private / name, f"{name} must resolve to private_root/{name}")
        require(on_nvme(target), f"{name} must be on NVMe")
        info = target.stat()
        require(info.st_uid == os.getuid() and not info.st_mode & 0o022,
                f"Unsafe {name} ownership/permissions")
    require(plan.python.is_file() and os.access(plan.python, os.X_OK),
            "Private Python executable missing")
    require((plan.python.parent / "vllm").is_file(), "Private vllm entry point missing")
    require(plan.hf_home.is_dir() and within(plan.hf_home.resolve(), private),
            "HF_HOME missing or escaping private root")
    for job in plan.jobs:
        script, script_root = job_script_root(plan, job)
        require(script.is_file() and within(script.resolve(), script_root),
                f"Missing or escaping job script: {script}")
    code = ("import importlib.metadata as m,json,sys;"
            "print(json.dumps({'python':sys.version,'executable':sys.executable,"
            "'vllm_version':m.version('vllm'),'torch_version':m.version('torch'),"
            "'transformers_version':m.version('transformers'),"
            "'packages':dict(sorted((d.metadata['Name'],d.version)"
            " for d in m.distributions() if d.metadata.get('Name')))}))")
    versions = read_json(subprocess.check_output([str(plan.python), "-I", "-c", code],
                                                timeout=30))
    for key in ("vllm_version", "torch_version", "transformers_version"):
        require(versions.get(key) == SERVER_SPEC[key], f"Runtime version mismatch: {key}")
    snapshot = plan.hf_home / "hub/models--Qwen--Qwen3-32B/snapshots" / REVISION
    require(snapshot.is_dir() and (snapshot / "config.json").is_file(),
            "Pinned Qwen snapshot/config absent from private HF cache")
    _, fixture_binding = load_funding_fixture(plan, verify_runtime_files=True)
    result = {"versions": versions, "git_head": git("rev-parse", "HEAD"), "branch": branch,
            "git_status": git("status", "--porcelain"),
            "sources": source_hashes(plan), "host": socket.gethostname(),
            "runtime_files": {
                "funding_probe_fixture": fixture_binding,
                "python_sha256": file_hash(plan.python),
                "vllm_entrypoint_sha256": file_hash(plan.python.parent / "vllm"),
                "model_config_sha256": file_hash(snapshot / "config.json"),
                "model_snapshot": str(snapshot),
            },
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    if "server_profile" in plan.spec:
        result.update(selected_runtime_binding(plan))
    return result


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    utilization: int
    driver: str
    pids: frozenset[int] = frozenset()
    pci_bus_id: str = ""

    @property
    def idle(self) -> bool:
        return not self.pids and self.used_mib < 1024 and self.utilization <= LOW_UTILIZATION


def parse_gpus(inventory: str, processes: str) -> list[GPU]:
    process_map: dict[str, set[int]] = {}
    try:
        for row in csv.reader(io.StringIO(processes)):
            if not row:
                continue
            require(len(row) == 2 and row[1].strip().isdigit(),
                    "Unparseable GPU compute PID; refusing allocation")
            process_map.setdefault(row[0].strip(), set()).add(int(row[1]))
        gpus = []
        for row in csv.reader(io.StringIO(inventory)):
            if not row:
                continue
            require(len(row) == 8, "Unparseable GPU inventory")
            index, uuid, name, total, used, util, driver, pci = (x.strip() for x in row)
            require(uuid.startswith("GPU-") and all(x.isdigit() for x in
                    (index, total, used, util)), "Unknown GPU telemetry; refusing allocation")
            require(bool(re.fullmatch(r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:"
                                     r"[0-9A-Fa-f]{2}\.[0-7]", pci)), "Unknown PCI bus ID")
            gpu = GPU(int(index), uuid, name, int(total), int(used), int(util), driver,
                      frozenset(process_map.get(uuid, set())), pci)
            require(gpu.total_mib > 0 and 0 <= gpu.utilization <= 100
                    and 0 <= gpu.used_mib <= gpu.total_mib, "Invalid GPU telemetry")
            gpus.append(gpu)
        require(len(gpus) == 8 and {x.index for x in gpus} == set(range(8))
                and len({x.uuid for x in gpus}) == 8,
                "Expected exactly the eight physical GPUs 0..7")
        require(not set(process_map) - {g.uuid for g in gpus},
                "Compute process UUID absent from inventory (MIG is unsupported)")
        # vLLM 0.12 parses mask entries as integers; influx also uses numeric IDs.
        # Make CUDA_DEVICE_ORDER=PCI_BUS_ID and NVML indices agree before launch.
        pci_order = sorted(gpus, key=lambda g: tuple(int(x, 16)
                           for x in re.split(r"[:.]", g.pci_bus_id)))
        require(len({g.pci_bus_id.upper() for g in gpus}) == 8
                and [g.index for g in pci_order] == list(range(8)),
                "NVML GPU indices differ from PCI_BUS_ID order; numeric client mask unsafe")
        return sorted(gpus, key=lambda x: x.index)
    except (ValueError, TypeError) as error:
        raise QueueError(f"Invalid nvidia-smi output: {error}") from error


def query_gpus() -> list[GPU]:
    common = ["--format=csv,noheader,nounits"]
    # Query processes after memory: any observed compute PID excludes a free pair.
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,"
         "utilization.gpu,driver_version,pci.bus_id", *common], text=True, timeout=15)
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", *common], text=True, timeout=15)
    return parse_gpus(inventory, processes)


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def reservation_path(lock_root: Path, gpu_uuid: str) -> Path:
    return lock_root / f"{gpu_uuid}.reservation.json"


def read_reservation(path: Path) -> dict | None:
    """Durable records survive flock release after a controller crash."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_uid == os.getuid() and info.st_size <= 1024 * 1024,
                f"Unsafe GPU reservation record: {path}")
        record = read_json(stream.read())
    require(isinstance(record, dict) and record.get("schema_version") == 1
            and record.get("token") and record.get("boot_id")
            and record.get("phase") in {"launch_intent", "active", "released"},
            f"Malformed GPU reservation record: {path}")
    return record


def reservation_blocks(path: Path) -> bool:
    record = read_reservation(path)
    if record is None:
        return False
    # Never infer release from an idle device or a dead/reused controller PID.
    # An intent without a child identity may already have forked a CPU-loading child.
    return record["boot_id"] == boot_id() and record["phase"] != "released"


def reserve_pair(gpus: list[GPU], lock_root: Path, query=query_gpus, *,
                 device_count: int = 2, prefer_pending_pair: bool = False):
    """Advisory locks plus a fresh telemetry check; never reserve a busy GPU."""
    require(type(device_count) is int and device_count in (1, 2), "Only one- or two-GPU reservations are supported")
    candidates = [gpu for gpu in gpus if gpu.idle]
    if prefer_pending_pair:
        candidates.sort(key=lambda gpu: ({5: 0, 6: 1}.get(gpu.index, 2), gpu.index))
    for group in combinations(candidates, device_count):
        pair = list(group)
        locks = [Lock(lock_root / f"{gpu.uuid}.lock") for gpu in pair]
        retained = False
        try:
            if not all(lock.acquire() for lock in locks):
                continue
            if any(reservation_blocks(reservation_path(lock_root, gpu.uuid)) for gpu in pair):
                continue
            current = {gpu.uuid: gpu for gpu in query()}
            if all(gpu.uuid in current and current[gpu.uuid].idle
                   and current[gpu.uuid].index == gpu.index for gpu in pair):
                retained = True
                return [current[gpu.uuid] for gpu in pair], locks
        finally:
            if not retained:
                for lock in locks:
                    lock.close()
    return None


def proc_identity(pid: int) -> dict | None:
    try:
        root = Path("/proc") / str(pid)
        raw = (root / "stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        return {"pid": pid, "start_ticks": int(tail[19]), "session": int(tail[3]),
                "state": tail[0], "uid": root.stat().st_uid,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(actual: dict | None, expected: dict) -> bool:
    return actual is not None and all(actual.get(k) == expected.get(k)
                                     for k in ("pid", "start_ticks", "uid", "boot_id", "session"))


def session_members(root: dict) -> list[dict]:
    if root["boot_id"] != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        return []
    leader = proc_identity(root["pid"])
    if leader is not None and not same_process(leader, root):
        return []  # Session ID was reused; never touch this session.
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            identity = proc_identity(int(path.name))
            if identity and identity["session"] == root["pid"] and (
                identity["uid"] == root["uid"] == os.getuid()
                and identity["start_ticks"] >= root["start_ticks"]
                and identity["state"] != "Z"
            ):
                result.append(identity)
    return result


def signal_verified(identity: dict, sig: int) -> bool:
    """pidfd closes the check-to-kill PID-reuse race. Never signal a bare PID."""
    try:
        fd = os.pidfd_open(identity["pid"])
    except ProcessLookupError:
        return False
    try:
        if not same_process(proc_identity(identity["pid"]), identity):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def stop_owned(root: dict, grace: float = 15) -> list[int]:
    deadline = time.monotonic() + grace
    sent = set()
    while True:
        members = session_members(root)
        if not members:
            return []
        for member in members:
            key = (member["pid"], member["start_ticks"])
            if key not in sent:
                signal_verified(member, signal.SIGTERM)
                sent.add(key)
        if time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    for member in session_members(root):
        signal_verified(member, signal.SIGKILL)
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        remaining = session_members(root)
        if not remaining:
            return []
        time.sleep(0.05)
    return [member["pid"] for member in session_members(root)]


def owns_listener(root: dict, port: int) -> bool:
    inodes = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(name).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].split(":")[1], 16) == port and fields[3] == "0A":
                inodes.add(fields[9])
    if not inodes:
        return False
    owned = set()
    for member in session_members(root):
        try:
            for entry in (Path("/proc") / str(member["pid"]) / "fd").iterdir():
                try:
                    target = os.readlink(entry)
                except (OSError, FileNotFoundError):
                    continue
                if target.startswith("socket:["):
                    owned.add(target[8:-1])
        except (FileNotFoundError, PermissionError):
            continue
    return inodes <= owned


def endpoint_ready(root: dict, port: int) -> bool:
    require(same_process(proc_identity(root["pid"]), root), "Server launch PID is not alive")
    if not owns_listener(root, port):
        return False
    base = f"http://127.0.0.1:{port}"
    # Do not trust proxy environment variables for the local health endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base + "/health", timeout=2) as response:
            if response.status != 200:
                return False
        with opener.open(base + "/v1/models", timeout=2) as response:
            payload = read_json(response.read(1024 * 1024))
        models = [model for model in payload.get("data", []) if model.get("id") == MODEL]
        return bool(models) and all(model.get("max_model_len", 32768) == 32768
                                    for model in models)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def probe_schema(nonce: str) -> dict:
    require(bool(re.fullmatch("[0-9a-f]{32}", nonce)), "Probe nonce must contain 128 random bits")
    return {"type": "object", "properties": {"probe_nonce": {"type": "string", "const": nonce}},
            "required": ["probe_nonce"], "additionalProperties": False}


def load_shared_probe_builder(plan: Plan, seed_components=PROBE_SEED_COMPONENTS):
    """Load the actual pure-stdlib client builder without importing utopia/Torch."""
    path = plan.worktree / "utopia/models/structured_outputs.py"
    require(path.is_file() and within(path.resolve(), plan.worktree.resolve()),
            "Required shared structured-output transport builder is missing/escaping worktree")
    checksum = file_hash(path)
    spec = importlib.util.spec_from_file_location("_james_transport_" + checksum[:16], path)
    require(spec is not None and spec.loader is not None, "Cannot load shared transport builder")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    builder = getattr(module, "build_vllm_extra_body", None)
    require(callable(builder), "Shared module lacks build_vllm_extra_body")
    # The production seed utility's module imports Torch. Execute only its
    # exact dependency-free derive_seed function, without importing that module.
    seed_path = plan.worktree / "utopia/utils/generail_utils.py"
    require(seed_path.is_file() and within(seed_path.resolve(), plan.worktree.resolve()),
            "Required production derive_seed source is missing")
    tree = ast.parse(seed_path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "derive_seed"]
    require(len(definitions) == 1, "Cannot identify the production derive_seed function")
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(seed_path), "exec"), namespace)
    sequential_seed = (
        len(seed_components) == 10
        and tuple(seed_components[:5]) == (42, "phase5_funding_eval", 0, SEQUENTIAL_PROTOCOL, "technical_probe")
        and isinstance(seed_components[5], str) and bool(seed_components[5])
        and type(seed_components[6]) is int and seed_components[6] >= 0
        and type(seed_components[7]) is int and 0 <= seed_components[7] < 25
        and tuple(seed_components[8:]) == (0, 0))
    require(tuple(seed_components) in (PROBE_SEED_COMPONENTS, FUNDING_SEED_COMPONENTS) or sequential_seed,
            "Unsupported probe seed namespace")
    request_seed = namespace["derive_seed"](*seed_components)
    def fields(schema):
        result = builder(schema, enable_thinking=True)
        result["seed"] = request_seed
        return result
    return fields, {
        "path": str(path), "sha256": checksum, "entrypoint": "build_vllm_extra_body",
        "enable_thinking": True, "request_seed_base": 42, "request_seed": request_seed,
        "seed_components": list(seed_components),
        "seed_helper": str(seed_path), "seed_helper_sha256": file_hash(seed_path)}


def probe_request(nonce: str, extra_body: dict) -> dict:
    """The nonce appears only in the JSON schema, never in either message."""
    schema = probe_schema(nonce)
    require(extra_body.get("structured_outputs") == {"json": schema}
            and extra_body.get("chat_template_kwargs") == {"enable_thinking": True}
            and "guided_json" not in extra_body
            and type(extra_body.get("seed")) is int and 0 <= extra_body["seed"] < 2 ** 31,
            "Shared client builder does not implement the required thinking/structured_outputs transport")
    body = {
        "model": MODEL, "temperature": 0.7, "max_tokens": PROBE_MAX_TOKENS,
        "stream": False,
        "messages": [
            {"role": "system", "content": "This is a transport check. Follow the user's output instruction."},
            {"role": "user", "content": "Your final answer must be exactly the plain text "
             "TRANSPORT_NOT_JSON. Do not output JSON, braces, a key/value object, or a nonce. "
             "Output TRANSPORT_NOT_JSON only."},
        ],
    }
    require(not set(body) & set(extra_body), "Shared builder overrides probe sampling/protocol fields")
    body.update(extra_body)
    require(json.dumps(body).count(nonce) == 1
            and nonce not in json.dumps(body["messages"]), "Probe nonce leaked outside the schema")
    return body


class NoProbeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@contextlib.contextmanager
def probe_deadline(seconds: float, *, maximum: float = PROBE_TIMEOUT_SECONDS):
    """A wall deadline covers slow headers/body, not merely individual socket reads."""
    require(maximum in (PROBE_TIMEOUT_SECONDS, FUNDING_PROBE_TIMEOUT_SECONDS)
            and 0 < seconds <= maximum, "Invalid constrained-output probe time budget")
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
            "Cannot install probe deadline over an existing process alarm")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise QueueError(f"Constrained-output probe exceeded {seconds:g}s wall deadline")
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def parse_probe_response(payload: dict) -> dict:
    require(isinstance(payload, dict) and payload.get("model") == MODEL,
            "Probe returned the wrong model or a malformed envelope")
    require(not any(payload.get(key) for key in ("error", "warning", "warnings")),
            "Probe endpoint reported ignored/unsupported fields or another warning/error")
    choices = payload.get("choices")
    require(isinstance(choices, list) and len(choices) == 1
            and isinstance(choices[0], dict), "Probe must return exactly one completion")
    choice = choices[0]
    require(choice.get("finish_reason") == "stop",
            f"Probe final output was truncated/unfinished: finish_reason={choice.get('finish_reason')!r}")
    message = choice.get("message")
    require(isinstance(message, dict) and message.get("role") == "assistant"
            and not message.get("tool_calls") and not message.get("function_call")
            and not message.get("refusal"), "Probe returned a non-text/refused completion")
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), "Probe has no final JSON content")
    final = read_json(content.encode("utf-8"))
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    require(reasoning is None or isinstance(reasoning, str), "Malformed probe reasoning field")
    return {"final_json": final, "finish_reason": choice["finish_reason"],
            "reasoning_present": bool(reasoning), "reasoning_characters": len(reasoning or ""),
            "reasoning_sha256": digest_bytes((reasoning or "").encode()),
            "usage": payload.get("usage")}


def verify_probe_response(payload: dict, nonce: str) -> dict:
    parsed = parse_probe_response(payload)
    require(parsed["final_json"] == {"probe_nonce": nonce},
            "Probe final JSON violates strict nonce schema/no-extra-fields")
    return parsed


def compact_funding_schema() -> dict:
    return {"type": "object", "properties": {
        "ranked_application_ids": {"type": "array",
                                   "items": {"type": "integer", "minimum": 0, "maximum": 24},
                                   "minItems": 1, "maxItems": 25}},
        "required": ["ranked_application_ids"], "additionalProperties": False}



def validate_funding_fixture(fixture: dict) -> None:
    """Validate the producer's contract; never generate or repair its prompt/schema."""
    require(isinstance(fixture, dict) and fixture.get("schema_version") == 1
            and fixture.get("kind") == "funding_panel25", "Wrong funding fixture kind/version")
    messages = fixture.get("messages")
    require(isinstance(messages, list) and len(messages) == 2
            and [m.get("role") for m in messages if isinstance(m, dict)] == ["system", "user"]
            and all(set(m) == {"role", "content"} and isinstance(m["content"], str)
                    and m["content"] for m in messages), "Funding fixture must supply exact system/user messages")
    expected = fixture.get("expected")
    require(isinstance(expected, dict), "Funding fixture expected contract is missing")
    representation = fixture.get("output_representation")
    require(representation in (None, "ordered_application_ids_v1"),
            "Unsupported funding fixture output_representation")
    compact = representation == "ordered_application_ids_v1"
    fixed = {"application_ids": list(range(25)), "ranks": list(range(1, 26)),
             "rankings_json_pointer": "/ranked_application_ids" if compact else "/ranked_applications",
             "application_id_key": "application_id", "applicant_id_key": "applicant_id",
             "rank_key": "rank", "application_count": 25, "finish_reason": "stop"}
    for key, value in fixed.items():
        require(type(expected.get(key)) is type(value) and expected[key] == value,
                f"Funding fixture contract mismatch: {key}")
    require(all(type(x) is int for x in expected["application_ids"] + expected["ranks"]),
            "Funding fixture IDs/ranks must be integers, not booleans")
    mapping = expected.get("application_to_applicant")
    require(isinstance(mapping, dict) and set(mapping) == {str(x) for x in range(25)}
            and all(isinstance(value, str) and value for value in mapping.values()),
            "Funding fixture must map every application ID to its actual applicant ID")
    schema = fixture.get("json_schema")
    if compact:
        require(json_bytes(schema) == json_bytes(compact_funding_schema()),
                "Compact funding schema differs from the exact ordered-ID production contract")
    else:
        require(isinstance(schema, dict) and schema.get("type") == "object"
                and set(schema.get("properties", {})) == {"ranked_applications"}
                and set(schema.get("required", [])) == {"ranked_applications"},
                "Funding fixture does not carry the production response schema")
        array = schema["properties"]["ranked_applications"]
        require(isinstance(array, dict) and array.get("type") == "array", "Funding schema ranking is not an array")
        item = array.get("items")
        require(isinstance(item, dict), "Funding schema item is missing")
        if "$ref" in item:
            require(isinstance(item["$ref"], str) and item["$ref"].startswith("#/"),
                    "Only local production funding schema references are supported")
            item = pointer(schema, item["$ref"][1:])
        fields = {"application_id": "integer", "applicant_id": "string", "rank": "integer", "reason": "string"}
        require(isinstance(item, dict) and item.get("type") == "object"
                and set(item.get("properties", {})) == set(fields)
                and set(item.get("required", [])) == set(fields)
                and all(item["properties"][key].get("type") == kind for key, kind in fields.items()),
                "Funding schema differs from the actual four-field ranking contract")
    tokenization = fixture.get("tokenization")
    require(isinstance(tokenization, dict), "Funding fixture tokenization metadata missing")
    settings = {"model": MODEL, "revision": REVISION, "transformers_version": "4.57.3",
                "enable_thinking": True, "add_generation_prompt": True,
                "max_model_len": 32768, "requested_max_tokens": FUNDING_PROBE_MAX_TOKENS}
    for key, value in settings.items():
        require(type(tokenization.get(key)) is type(value) and tokenization[key] == value,
                f"Funding tokenization metadata mismatch: {key}")
    tokens = tokenization.get("input_tokens")
    total = tokenization.get("total_requested_tokens")
    require(type(tokens) is int and tokens > 0 and type(total) is int
            and total == tokens + FUNDING_PROBE_MAX_TOKENS and total <= 32768,
            "Funding fixture input+8192 exceeds context or has inconsistent exact token counts")
    hashes = tokenization.get("tokenizer_files_sha256")
    require(isinstance(hashes, dict)
            and {"tokenizer.json", "tokenizer_config.json"} <= set(hashes),
            "Funding fixture lacks pinned tokenizer/config hashes")
    for name, checksum in hashes.items():
        require(isinstance(name, str) and Path(name).name == name
                and isinstance(checksum, str) and bool(re.fullmatch("[0-9a-f]{64}", checksum)),
                "Invalid funding tokenizer file/hash metadata")
    require(isinstance(tokenization.get("chat_template_sha256"), str)
            and bool(re.fullmatch("[0-9a-f]{64}", tokenization["chat_template_sha256"])),
            "Funding fixture lacks effective chat-template hash")
    selection = fixture.get("funding_selection_protocol")
    require(selection in (None, SEQUENTIAL_PROTOCOL), "Unsupported fixture selection protocol")
    if selection:
        require(compact, "Sequential selection must retain the compact final-batch representation")
        prompt = fixture.get("original_compact_prompt")
        require(isinstance(prompt, str) and bool(prompt)
                and "\n## Response Format\n" in prompt
                and messages[1]["content"].endswith(prompt),
                "Sequential fixture requires its original compact prompt without IMPORTANT_NOTES")
        require(isinstance(fixture.get("program_id"), str) and bool(fixture["program_id"])
                and type(fixture.get("panel_index")) is int and fixture["panel_index"] >= 0,
                "Sequential fixture must identify the actual program and panel")


def load_funding_fixture(plan: Plan, *, verify_runtime_files: bool = False):
    descriptor = plan.spec["funding_probe_fixture"]
    path = Path(descriptor["path"])
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and info.st_nlink == 1 and not info.st_mode & 0o222
                and 0 < info.st_size <= FUNDING_FIXTURE_MAX_BYTES,
                "Funding fixture must be a bounded, owned, regular immutable file (0444)")
        raw = stream.read(FUNDING_FIXTURE_MAX_BYTES + 1)
        require(stat_binding(info) == stat_binding(os.fstat(stream.fileno())),
                "Funding fixture changed while reading")
    require(digest_bytes(raw) == descriptor["sha256"], "Funding fixture SHA256 mismatch")
    fixture = read_json(raw)
    validate_funding_fixture(fixture)
    require(fixture.get("funding_selection_protocol") == plan.spec.get("funding_selection_protocol"),
            "Plan/fixture funding selection protocol mismatch; old compact evidence cannot authorize v6")
    tokenization = fixture["tokenization"]
    if verify_runtime_files:
        snapshot = plan.hf_home / "hub/models--Qwen--Qwen3-32B/snapshots" / REVISION
        for name, expected in tokenization["tokenizer_files_sha256"].items():
            require(file_hash(snapshot / name) == expected, f"Funding tokenizer file changed: {name}")
        template_file = snapshot / "chat_template.jinja"
        if template_file.is_file():
            template = template_file.read_text()
        else:
            template = read_json((snapshot / "tokenizer_config.json").read_bytes()).get("chat_template")
        require(isinstance(template, str) and digest_bytes(template.encode()) ==
                tokenization["chat_template_sha256"], "Funding effective chat-template hash differs")
    binding = {"path": str(path), "sha256": descriptor["sha256"],
               "kind": fixture["kind"], "tokenization": tokenization, "expected": fixture["expected"]}
    if fixture.get("output_representation"):
        binding["output_representation"] = fixture["output_representation"]
    if fixture.get("funding_selection_protocol"):
        binding.update(funding_selection_protocol=SEQUENTIAL_PROTOCOL,
                       original_compact_prompt_sha256=digest_bytes(fixture["original_compact_prompt"].encode()),
                       program_id=fixture["program_id"], panel_index=fixture["panel_index"])
    return fixture, binding


def load_sequential_api(plan: Plan):
    """Import the owner's shared pure helpers, not a copied ranking algorithm."""
    path = plan.worktree / "src/james_funding_sequential.py"
    require(path.is_file() and within(path.resolve(), plan.worktree.resolve()),
            "Sequential funding helper is missing or escapes the worktree")
    checksum = file_hash(path)
    spec = importlib.util.spec_from_file_location("_james_sequential_" + checksum[:16], path)
    require(spec is not None and spec.loader is not None, "Cannot load sequential funding helper")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
        sys.modules.pop(spec.name, None)
    require(module.SEQUENTIAL_PROTOCOL == SEQUENTIAL_PROTOCOL, "Sequential helper protocol changed")
    require(all(callable(getattr(module, name, None)) for name in (
        "selection_schema", "selection_prompt", "decode_selection",
        "build_selection_messages", "selection_seed_context")), "Incomplete sequential helper API")
    require(file_hash(path) == checksum, "Sequential helper changed while loading")
    return module, {"path": str(path), "sha256": checksum, "protocol": SEQUENTIAL_PROTOCOL}


def sequential_probe_request(plan: Plan, fixture: dict, selected: list[int]):
    api, helper = load_sequential_api(plan)
    require(len(selected) < 25 and all(type(x) is int and 0 <= x < 25 for x in selected)
            and len(set(selected)) == len(selected), "Invalid sequential probe prefix")
    remaining = [x for x in range(25) if x not in selected]
    original = fixture["original_compact_prompt"]
    require(api.build_selection_messages(original) == fixture["messages"],
            "Fixture system/IMPORTANT_NOTES layout differs from the production message builder")
    prompt = api.selection_prompt(original, selected, remaining)
    prefix, marker, _ = original.rpartition("\n## Response Format\n")
    require(bool(marker) and prompt.startswith(prefix + marker),
            "Sequential request changed substantive application/criteria prefix")
    schema = api.selection_schema(remaining)
    properties = schema.get("properties", {})
    require(schema.get("type") == "object" and schema.get("additionalProperties") is False
            and len(properties) == 1 and schema.get("required") == list(properties),
            "Sequential schema must require exactly one integer selection")
    rule = next(iter(properties.values()))
    require(rule.get("type") == "integer" and rule.get("enum") == remaining
            and all(type(x) is int for x in rule["enum"]),
            "Sequential schema does not constrain the exact remaining integer IDs")
    context = tuple(api.selection_seed_context(
        0, fixture["program_id"], fixture["panel_index"], len(selected), namespace="technical_probe"))
    require(context == ("phase5_funding_eval", 0, SEQUENTIAL_PROTOCOL, "technical_probe",
                        fixture["program_id"], fixture["panel_index"], len(selected)),
            "Sequential technical-probe seed context changed")
    builder, metadata = load_shared_probe_builder(plan, (42, *context, 0, 0))
    extra = builder(schema)
    require(extra.get("structured_outputs") == {"json": schema}
            and extra.get("chat_template_kwargs") == {"enable_thinking": True}
            and set(extra) == {"structured_outputs", "chat_template_kwargs", "seed"},
            "Sequential probe transport differs from production")
    request = {"model": MODEL, "temperature": 0.7, "max_tokens": FUNDING_PROBE_MAX_TOKENS,
               "messages": api.build_selection_messages(prompt), "stream": False, **extra}
    return request, {
        "step": len(selected), "attempt": 0, "selected": list(selected), "remaining": remaining,
        "schema": schema, "seed_context": list(context), "shared_client_builder": metadata,
        "sequential_helper": helper, "selection_prompt_sha256": digest_bytes(prompt.encode())}


def validate_token_count(record: dict, request: dict):
    require(isinstance(record, dict) and record.get("request_sha256") == digest_bytes(json_bytes(request)),
            "Tokenizer sidecar counted a different request")
    require(type(record.get("input_tokens")) is int and record["input_tokens"] > 0
            and record["input_tokens"] + request["max_tokens"] <= SERVER_SPEC["max_model_len"],
            "Sequential input plus output reservation exceeds context or has invalid count")
    require(isinstance(record.get("rendered_prompt_sha256"), str)
            and bool(re.fullmatch("[0-9a-f]{64}", record["rendered_prompt_sha256"])),
            "Tokenizer sidecar lacks rendered-prompt binding")


def validate_tokenizer_identity(identity: dict, plan: Plan, fixture: dict):
    require(isinstance(identity, dict) and identity.get("protocol") == "james-sequential-tokenizer-v1"
            and identity.get("model") == MODEL and identity.get("revision") == REVISION,
            "Tokenizer sidecar has wrong model/revision/protocol")
    tokenization = fixture["tokenization"]
    require(identity.get("chat_template_sha256") == tokenization["chat_template_sha256"]
            and all(identity.get("tokenizer_files_sha256", {}).get(name) == checksum
                    for name, checksum in tokenization["tokenizer_files_sha256"].items()),
            "Tokenizer sidecar does not match the pinned fixture files/template")
    require(identity.get("request_audit_sha256") ==
            file_hash(plan.worktree / "utopia/models/request_audit.py"),
            "Tokenizer sidecar used a different request-audit loader")


def tokenizer_sidecar_main(worktree: str, hf_home: str) -> int:
    """Private-Python CPU worker. Only local tokenizer loading and token counts."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Tokenizer sidecar must be CPU-only and offline")
    path = Path(worktree) / "utopia/models/request_audit.py"
    checksum = file_hash(path)
    with contextlib.redirect_stdout(sys.stderr):
        spec = importlib.util.spec_from_file_location("_james_probe_audit", path)
        require(spec is not None and spec.loader is not None, "Cannot load actual audit tokenizer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tokenizer = module._load_tokenizer()
    snapshot = Path(hf_home) / "hub/models--Qwen--Qwen3-32B/snapshots" / REVISION
    require(Path(tokenizer.name_or_path).resolve() == snapshot.resolve(),
            "Tokenizer loader selected another snapshot")
    names = ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "chat_template.jinja")
    identity = {
        "protocol": "james-sequential-tokenizer-v1", "model": MODEL, "revision": REVISION,
        "request_audit_sha256": checksum,
        "tokenizer_files_sha256": {name: file_hash(snapshot / name) for name in names
                                   if (snapshot / name).is_file()},
        "chat_template_sha256": digest_bytes(tokenizer.chat_template.encode())}
    print(json.dumps(identity), flush=True)
    for _ in range(25):
        raw = sys.stdin.buffer.readline(FUNDING_FIXTURE_MAX_BYTES + 1)
        if not raw:
            return 0
        require(len(raw) <= FUNDING_FIXTURE_MAX_BYTES and raw.endswith(b"\n"),
                "Oversized/incomplete tokenizer request")
        request = read_json(raw)
        require(request.get("model") == MODEL and request.get("max_tokens") == FUNDING_PROBE_MAX_TOKENS
                and request.get("chat_template_kwargs") == {"enable_thinking": True},
                "Unexpected tokenizer request settings")
        text = tokenizer.apply_chat_template(
            request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        require(file_hash(path) == checksum, "Tokenizer loader source changed")
        print(json.dumps({"request_sha256": digest_bytes(json_bytes(request)),
                          "input_tokens": len(ids), "rendered_prompt_sha256": digest_bytes(text.encode())}),
              flush=True)
    return 0


class ProbeTokenCounter:
    """Bounded pipe protocol to the private CPU environment; never a GPU client."""
    def __init__(self, plan: Plan, logpath: Path, *, cancelled=lambda: False, aggregate_deadline=None):
        self.plan, self.cancelled = plan, cancelled
        self.aggregate_deadline = aggregate_deadline
        self.process, self.owned_identity, self.log = None, None, None
        self.buffer = b""
        self.identity = None
        self.command = [str(plan.python), "-u", "-B", str(Path(__file__).resolve()),
                        "--tokenizer-sidecar", str(plan.worktree), str(plan.hf_home)]
        try:
            self.log = logpath.open("xb")
            require(not cancelled(), "Queue interrupted before tokenizer sidecar")
            env = {k: v for k, v in os.environ.items() if k not in PROTECTED_ENV
                   and not k.startswith(("CUDA_", "VLLM_", "HF_", "PYTHON"))}
            env.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(plan.hf_home),
                       HF_HUB_CACHE=str(plan.hf_home / "hub"),
                       HUGGINGFACE_HUB_CACHE=str(plan.hf_home / "hub"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="42", TOKENIZERS_PARALLELISM="false")
            self.process = subprocess.Popen(
                self.command, cwd=plan.worktree, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, start_new_session=True, close_fds=True, bufsize=0)
            captured = proc_identity(self.process.pid)
            require(captured is not None
                    and captured["session"] == self.process.pid
                    and captured["uid"] == os.getuid(), "Cannot identify owned tokenizer sidecar")
            self.owned_identity = captured
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.identity = self._read_reply(time.monotonic() + TOKENIZER_STARTUP_SECONDS)
        except BaseException:
            self.close()
            raise

    def _remaining(self, deadline):
        require(not self.cancelled(), "Queue interrupted during tokenizer accounting")
        aggregate = getattr(self, "aggregate_deadline", None)
        if aggregate is not None:
            deadline = min(deadline, aggregate)
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Tokenizer sidecar exceeded fixed time budget")
        return remaining

    def _read_reply(self, deadline):
        while b"\n" not in self.buffer:
            require(len(self.buffer) < TOKENIZER_REPLY_BYTES, "Tokenizer sidecar reply too large")
            ready, _, _ = select.select([self.process.stdout], [], [], self._remaining(deadline))
            require(bool(ready), "Tokenizer sidecar exceeded fixed time budget")
            chunk = os.read(self.process.stdout.fileno(), 4096)
            require(bool(chunk), "Tokenizer sidecar closed before its reply")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        require(len(line) <= TOKENIZER_REPLY_BYTES, "Tokenizer sidecar reply too large")
        self._remaining(deadline)
        return read_json(line)

    def count(self, request, *, timeout=TOKENIZER_COUNT_SECONDS):
        require(0 < timeout <= TOKENIZER_COUNT_SECONDS, "Invalid tokenizer count timeout")
        deadline = time.monotonic() + timeout
        raw = json.dumps(request, allow_nan=False).encode() + b"\n"
        require(len(raw) <= FUNDING_FIXTURE_MAX_BYTES, "Tokenizer request exceeds bound")
        while raw:
            _, ready, _ = select.select([], [self.process.stdin], [], self._remaining(deadline))
            require(bool(ready), "Tokenizer sidecar write exceeded fixed time budget")
            count = os.write(self.process.stdin.fileno(), raw)
            require(count > 0, "Tokenizer sidecar write failed")
            raw = raw[count:]
        result = self._read_reply(deadline)
        validate_token_count(result, request)
        return result

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                if process.stdin:
                    process.stdin.close()
                if self.owned_identity:
                    require(not stop_owned(self.owned_identity, grace=3),
                            "Owned tokenizer sidecar did not stop")
                elif process.poll() is None:
                    # This is our unreaped Popen child, never a PID from telemetry.
                    process.kill()
                process.wait(timeout=5)
                if process.stdout:
                    process.stdout.close()
        finally:
            if self.log is not None:
                self.log.close()


def verify_sequential_response(payload: dict, remaining: list[int], token_count: dict, api):
    parsed = parse_probe_response(payload)
    content = payload["choices"][0]["message"]["content"]
    chosen = api.decode_selection(content, remaining)
    require(type(chosen) is int and chosen in remaining, "Sequential response selected a nonremaining ID")
    usage = parsed["usage"]
    require(isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int
            and usage["prompt_tokens"] == token_count["input_tokens"],
            "Sequential server prompt_tokens differs from exact tokenizer accounting")
    require(type(usage.get("completion_tokens")) is int
            and 0 <= usage["completion_tokens"] <= FUNDING_PROBE_MAX_TOKENS
            and type(usage.get("total_tokens")) is int
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "Sequential response usage exceeds budget or is inconsistent")
    return {**parsed, "selected_application_id": chosen, "exact_prompt_tokens": usage["prompt_tokens"]}


def sequential_funding_probe(plan: Plan, root: dict, port: int, fixture: dict, binding: dict,
                             directory: Path, server_name: str, *, cancelled=lambda: False,
                             timeout_seconds: float = SEQUENTIAL_PROBE_SECONDS) -> dict:
    """One technical panel, 25 real selections including singleton, no probe retries."""
    record = {
        "schema_version": 1, "kind": "funding_panel25_sequential_v1", "status": "pending",
        "funding_selection_protocol": SEQUENTIAL_PROTOCOL, "fixture": binding,
        "process_identity": dict(root), "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
        "timeout_seconds": timeout_seconds, "per_request_timeout_seconds": FUNDING_PROBE_TIMEOUT_SECONDS,
        "tokenizer_startup_seconds": TOKENIZER_STARTUP_SECONDS,
        "tokenizer_count_seconds": TOKENIZER_COUNT_SECONDS,
        "transport": "stdlib_urllib_direct_http", "http_requests_per_step": 1,
        "sdk_calls": 0, "http_retries": 0,
        "attempts_per_step": 1, "steps": [], "started_utc": utc(),
        "persistence_prefix": server_name,
        "scope": "One technical 25-application panel, not a scientific world or ranking-equivalence claim."}
    started = time.monotonic()
    counter = None
    try:
        require(fixture.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL,
                "Sequential readiness requires a new sequential fixture")
        require(TOKENIZER_STARTUP_SECONDS <= timeout_seconds <= SEQUENTIAL_PROBE_SECONDS,
                "Invalid fixed sequential probe time budget")
        api, helper = load_sequential_api(plan)
        record["sequential_helper"] = helper
        counter = ProbeTokenCounter(
            plan, directory / f"{server_name}.tokenizer.log", cancelled=cancelled,
            aggregate_deadline=started + timeout_seconds)
        validate_tokenizer_identity(counter.identity, plan, fixture)
        record["tokenizer"] = counter.identity
        record["tokenizer_process"] = {"identity": counter.owned_identity, "command": counter.command}
        selected = []
        for step in range(25):
            require(not cancelled(), "Queue interrupted before sequential selection")
            remaining_seconds = timeout_seconds - (time.monotonic() - started)
            require(remaining_seconds > 0, "Sequential probe aggregate time budget exceeded")
            request, context = sequential_probe_request(plan, fixture, selected)
            require(context["sequential_helper"] == helper, "Sequential helper changed during probe")
            record["pending_step"] = {**context, "request": request}
            remaining_seconds = timeout_seconds - (time.monotonic() - started)
            count = counter.count(request, timeout=min(TOKENIZER_COUNT_SECONDS, remaining_seconds))
            record["pending_step"]["token_count"] = count
            intent = {**context, "request": request, "token_count": count,
                      "process_identity": dict(root), "endpoint": record["endpoint"], "fixture": binding}
            intent_path = directory / f"{server_name}.sequential-step-{step:02d}.request.json"
            atomic_write(intent_path, json_bytes(intent), exclusive=True, readonly=True)
            remaining_seconds = timeout_seconds - (time.monotonic() - started)
            result = perform_probe(
                root, port, lambda: request,
                lambda payload: verify_sequential_response(payload, context["remaining"], count, api),
                {**context, "kind": "funding_sequential_choice", "token_count": count,
                 "timeout_seconds": min(FUNDING_PROBE_TIMEOUT_SECONDS, remaining_seconds),
                 "response_byte_limit": FUNDING_PROBE_MAX_RESPONSE_BYTES},
                maximum=FUNDING_PROBE_TIMEOUT_SECONDS)
            result_path = directory / f"{server_name}.sequential-step-{step:02d}.json"
            atomic_write(result_path, json_bytes(result), exclusive=True, readonly=True)
            result["persistence"] = {
                "intent": {"path": str(intent_path), "sha256": file_hash(intent_path)},
                "result": {"path": str(result_path), "sha256": file_hash(result_path)}}
            record["steps"].append(result)
            record.pop("pending_step")
            selected.append(result["validation"]["selected_application_id"])
        require(sorted(selected) == list(range(25)), "Sequential probe lacks a full permutation")
        counter.close()
        counter = None
        require(time.monotonic() - started < timeout_seconds, "Sequential probe aggregate time budget exceeded")
        record.update(status="passed", http_status=200, completed_utc=utc(),
                      elapsed_seconds=time.monotonic() - started,
                      validation=sequential_final_validation(selected, fixture))
    except Exception as error:
        if isinstance(error, ProbeError):
            record["failed_step"] = error.attestation
        record.update(status="failed", error_type=type(error).__name__, error=str(error), failed_utc=utc())
        raise ProbeError(f"Sequential funding preflight failed: {error}", record) from error
    finally:
        if counter is not None:
            try:
                counter.close()
            except Exception as cleanup_error:
                record.update(status="failed", cleanup_error=str(cleanup_error))
                raise ProbeError("Sequential tokenizer cleanup failed", record) from cleanup_error
    return record


def sequential_final_validation(selected: list[int], fixture: dict):
    return {"output_representation": "ordered_application_ids_v1",
            "funding_selection_protocol": SEQUENTIAL_PROTOCOL,
            "final_json": {"ranked_application_ids": selected},
            "application_count": 25, "complete_application_permutation": True,
            "complete_rank_permutation": True, "exact_application_applicant_mapping": True,
            "model_selections": 25, "singleton_model_selection": True,
            "ranking_preference": "Any complete ordering is accepted; no preferred ranking is prescribed.",
            "normalized_application_ranks": [
                {"application_id": app_id,
                 "applicant_id": fixture["expected"]["application_to_applicant"][str(app_id)], "rank": rank}
                for rank, app_id in enumerate(selected, 1)]}


def validate_sequential_probe_attestation(record, root, port, fixture, binding, plan):
    require(plan is not None and isinstance(record, dict)
            and record.get("kind") == "funding_panel25_sequential_v1"
            and record.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL
            and record.get("status") == "passed" and record.get("http_status") == 200,
            "A legacy/failed funding probe cannot authorize sequential clients")
    require(same_process(record.get("process_identity"), root)
            and record.get("endpoint") == f"http://127.0.0.1:{port}/v1/chat/completions"
            and record.get("fixture") == binding, "Sequential probe server/fixture binding changed")
    require(record.get("attempts_per_step") == 1
            and record.get("transport") == "stdlib_urllib_direct_http"
            and record.get("http_requests_per_step") == 1
            and record.get("sdk_calls") == 0 and record.get("http_retries") == 0
            and record.get("per_request_timeout_seconds") == FUNDING_PROBE_TIMEOUT_SECONDS
            and record.get("tokenizer_startup_seconds") == TOKENIZER_STARTUP_SECONDS
            and record.get("tokenizer_count_seconds") == TOKENIZER_COUNT_SECONDS
            and type(record.get("timeout_seconds")) in (int, float)
            and TOKENIZER_STARTUP_SECONDS <= record["timeout_seconds"] <= SEQUENTIAL_PROBE_SECONDS
            and type(record.get("elapsed_seconds")) in (int, float)
            and 0 <= record["elapsed_seconds"] < record["timeout_seconds"],
            "Sequential probe bounds changed")
    api, helper = load_sequential_api(plan)
    require(record.get("sequential_helper") == helper, "Sequential helper attestation changed")
    validate_tokenizer_identity(record.get("tokenizer"), plan, fixture)
    steps = record.get("steps")
    require(isinstance(steps, list) and len(steps) == 25, "Sequential proof requires all 25 model calls")
    require(isinstance(record.get("persistence_prefix"), str)
            and bool(re.fullmatch(r"server-\d{3,}", record["persistence_prefix"])),
            "Invalid sequential proof sidecar prefix")
    selected = []
    for index, step in enumerate(steps):
        request, context = sequential_probe_request(plan, fixture, selected)
        require(all(step.get(key) == value for key, value in context.items()),
                "Sequential step prefix/remaining/schema/seed/helper binding changed")
        require(step.get("request") == request
                and step.get("request_sha256") == digest_bytes(json_bytes(request)),
                "Sequential step request changed")
        require(step.get("status") == "passed" and step.get("http_status") == 200
                and not step.get("response_warnings")
                and same_process(step.get("process_identity"), root)
                and step.get("endpoint") == record["endpoint"], "Sequential step server/result changed")
        require(type(step.get("timeout_seconds")) in (int, float)
                and 0 < step["timeout_seconds"] <= FUNDING_PROBE_TIMEOUT_SECONDS
                and step.get("response_byte_limit") == FUNDING_PROBE_MAX_RESPONSE_BYTES
                and type(step.get("elapsed_seconds")) in (int, float)
                and 0 <= step["elapsed_seconds"] < step["timeout_seconds"],
                "Sequential step exceeded its fixed bound")
        count = step.get("token_count")
        validate_token_count(count, request)
        raw = step.get("response_body")
        require(isinstance(raw, str) and len(raw.encode()) <= FUNDING_PROBE_MAX_RESPONSE_BYTES
                and step.get("response_sha256") == digest_bytes(raw.encode()),
                "Sequential step raw response changed")
        validation = verify_sequential_response(read_json(raw.encode()), context["remaining"], count, api)
        require(step.get("validation") == validation, "Sequential step validation changed")
        persisted = step.get("persistence", {})
        prefix = f"{Path(record['persistence_prefix']).name}.sequential-step-{index:02d}"
        for kind, suffix in (("intent", ".request.json"), ("result", ".json")):
            declaration = persisted.get(kind, {})
            expected_path = plan.directory / (prefix + suffix)
            require(declaration.get("path") == str(expected_path), "Sequential sidecar path binding changed")
            data = queue_record_bytes(plan, expected_path)
            require(digest_bytes(data) == declaration.get("sha256"), "Sequential sidecar content changed")
            with open_output_file(plan, expected_path) as stream:
                require(not os.fstat(stream.fileno()).st_mode & 0o222, "Sequential sidecar became writable")
            expected = ({**context, "request": request, "token_count": count,
                         "process_identity": dict(root), "endpoint": record["endpoint"], "fixture": binding}
                        if kind == "intent" else {k: v for k, v in step.items() if k != "persistence"})
            require(read_json(data) == expected, "Sequential sidecar disagrees with embedded proof")
        selected.append(validation["selected_application_id"])
    require(sorted(selected) == list(range(25))
            and record.get("validation") == sequential_final_validation(selected, fixture),
            "Sequential final permutation/count changed")


def funding_probe_request(fixture: dict, extra_body: dict) -> dict:
    validate_funding_fixture(fixture)
    require(extra_body.get("structured_outputs") == {"json": fixture["json_schema"]}
            and extra_body.get("chat_template_kwargs") == {"enable_thinking": True}
            and type(extra_body.get("seed")) is int and 0 <= extra_body["seed"] < 2 ** 31
            and set(extra_body) == {"structured_outputs", "chat_template_kwargs", "seed"},
            "Funding probe shared transport does not match production")
    return {"model": MODEL, "temperature": 0.7, "max_tokens": FUNDING_PROBE_MAX_TOKENS,
            "messages": fixture["messages"], "stream": False, **extra_body}


def verify_funding_probe_response(payload: dict, fixture: dict) -> dict:
    parsed = parse_probe_response(payload)
    final, expected = parsed["final_json"], fixture["expected"]
    compact = fixture.get("output_representation") == "ordered_application_ids_v1"
    key = "ranked_application_ids" if compact else "ranked_applications"
    require(isinstance(final, dict) and set(final) == {key},
            "Funding probe must return exactly the production ranking object")
    rows = pointer(final, expected["rankings_json_pointer"])
    require(isinstance(rows, list) and len(rows) == expected["application_count"],
            "Funding probe did not rank all 25 applications")
    if compact:
        require(all(type(app_id) is int for app_id in rows),
                "Compact funding application IDs must be integers, not booleans")
        ids, ranks = rows, list(range(1, len(rows) + 1))
    else:
        ids, ranks = [], []
        for row in rows:
            require(isinstance(row, dict) and set(row) ==
                    {"application_id", "applicant_id", "rank", "reason"},
                    "Funding probe row has missing/extra production fields")
            app_id, rank = row[expected["application_id_key"]], row[expected["rank_key"]]
            require(type(app_id) is int and type(rank) is int, "Funding application IDs/ranks must be integers")
            require(str(app_id) in expected["application_to_applicant"]
                    and row[expected["applicant_id_key"]] == expected["application_to_applicant"][str(app_id)]
                    and isinstance(row["applicant_id"], str) and isinstance(row["reason"], str),
                    "Funding probe application/applicant pairing or reason type is invalid")
            ids.append(app_id)
            ranks.append(rank)
    require(sorted(ids) == expected["application_ids"] and sorted(ranks) == expected["ranks"],
            "Funding probe application IDs and ranks must each be complete unique permutations")
    usage = payload.get("usage")
    require(isinstance(usage, dict) and type(usage.get("prompt_tokens")) is int
            and usage["prompt_tokens"] == fixture["tokenization"]["input_tokens"],
            "Funding server prompt_tokens differs from exact pinned-tokenizer fixture count")
    require(type(usage.get("completion_tokens")) is int
            and 0 <= usage["completion_tokens"] <= FUNDING_PROBE_MAX_TOKENS,
            "Funding completion_tokens must be an integer within the requested 0..8192 budget")
    require(type(usage.get("total_tokens")) is int
            and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "Funding total_tokens must equal prompt_tokens + completion_tokens")
    parsed.update(application_count=25, complete_application_permutation=True,
                  complete_rank_permutation=True, exact_application_applicant_mapping=True,
                  exact_prompt_tokens=usage["prompt_tokens"],
                  ranking_preference="Any complete ordering is accepted; no preferred ranking is prescribed.")
    if compact:
        parsed.update(
            output_representation="ordered_application_ids_v1",
            rank_source="One-based position in the returned complete ordered application-ID permutation.",
            applicant_mapping_source="Immutable fixture input; applicant strings are not generated.",
            normalized_application_ranks=[
                {"application_id": app_id, "applicant_id": expected["application_to_applicant"][str(app_id)],
                 "rank": rank} for app_id, rank in zip(ids, ranks)])
    return parsed


def probe_response_metrics(payload: Any) -> dict:
    """Keep finish/reasoning evidence even when strict final-JSON validation fails."""
    envelope = payload if isinstance(payload, dict) else {}
    choices = envelope.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    reasoning = message.get("reasoning_content")
    if reasoning is None:
        reasoning = message.get("reasoning")
    content = message.get("content")
    return {
        "finish_reason": choice.get("finish_reason"), "usage": envelope.get("usage"),
        "final_content_characters": len(content) if isinstance(content, str) else None,
        "reasoning_present": bool(reasoning),
        "reasoning_characters": len(reasoning) if isinstance(reasoning, str) else None,
        "reasoning_sha256": digest_bytes(reasoning.encode()) if isinstance(reasoning, str) else None,
    }


def validate_probe_attestation(record: dict, root: dict, port: int) -> None:
    require(isinstance(record, dict) and record.get("status") == "passed"
            and record.get("http_status") == 200, "Mandatory constrained-output probe has not passed")
    require(same_process(record.get("process_identity"), root)
            and record.get("endpoint") == f"http://127.0.0.1:{port}/v1/chat/completions",
            "Constrained-output probe is bound to another server process/endpoint")
    request = record.get("request")
    require(isinstance(request, dict) and request.get("model") == MODEL
            and request.get("max_tokens") == PROBE_MAX_TOKENS,
            "Probe request differs from pinned production settings")
    schema = request.get("structured_outputs", {}).get("json", {})
    nonce = schema.get("properties", {}).get("probe_nonce", {}).get("const", "")
    require(schema == probe_schema(nonce)
            and request == probe_request(nonce, {
                "structured_outputs": {"json": schema},
                "chat_template_kwargs": {"enable_thinking": True}, "seed": request.get("seed")}),
            "Probe attestation used another transport/thinking/schema setting")
    require(record.get("request_sha256") == digest_bytes(json_bytes(request))
            and record.get("response_sha256") == digest_bytes(record["response_body"].encode()),
            "Probe request/response hash mismatch")
    require(record.get("request_seed_base") == 42
            and record.get("request_seed_components") == list(PROBE_SEED_COMPONENTS),
            "Probe lost the production base-seed/derivation context")
    if record.get("shared_client_builder"):
        builder = record["shared_client_builder"]
        require(builder.get("request_seed_base") == 42
                and builder.get("request_seed") == request["seed"]
                and builder.get("seed_components") == list(PROBE_SEED_COMPONENTS),
                "Shared production seed helper disagrees with probe request")
    require(record.get("validation") == verify_probe_response(read_json(
        record["response_body"].encode()), nonce), "Probe result validation changed")


def perform_probe(root: dict, port: int, request_factory, response_validator,
                  record: dict, *, maximum: float) -> dict:
    """Shared owned-endpoint HTTP operation; a single request under a hard deadline."""
    started = time.monotonic()
    timeout_seconds, byte_limit = record["timeout_seconds"], record["response_byte_limit"]
    record.update({
        "schema_version": 1, "status": "pending",
        "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
        "process_identity": dict(root), "started_utc": utc(),
        "reasoning_protocol": "enable_thinking=True; qwen3 parser; constrain final content after reasoning.",
    })
    try:
        require(0 < timeout_seconds <= maximum, "Invalid constrained-output probe time budget")
        require(same_process(proc_identity(root["pid"]), root) and owns_listener(root, port),
                "Probe requires a live owned API process and socket")
        request = request_factory()
        record.update(request=request, request_sha256=digest_bytes(json_bytes(request)))
        remaining = timeout_seconds - (time.monotonic() - started)
        with probe_deadline(remaining, maximum=maximum):
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoProbeRedirect())
            req = urllib.request.Request(record["endpoint"], data=json_bytes(request),
                                         headers={"Content-Type": "application/json", "Accept": "application/json"},
                                         method="POST")
            with opener.open(req, timeout=remaining) as response:
                record["http_status"] = response.status
                require(response.status == 200, f"Probe HTTP status {response.status}, expected200")
                require(response.geturl() == record["endpoint"], "Probe endpoint redirected")
                warnings = {key: response.headers.get(key) for key in
                            ("Warning", "X-Vllm-Warning", "X-Vllm-Warnings") if response.headers.get(key)}
                record["response_warnings"] = warnings
                require(not warnings, "Probe endpoint reported ignored/unsupported fields")
                raw = response.read(byte_limit + 1)
                require(len(raw) <= byte_limit, "Probe response exceeded byte limit")
            record.update(response_body=raw.decode("utf-8"), response_sha256=digest_bytes(raw))
            payload = read_json(raw)
            record["response_metrics"] = probe_response_metrics(payload)
            record["validation"] = response_validator(payload)
            require(time.monotonic() - started < timeout_seconds, "Probe total time budget exceeded")
            require(same_process(proc_identity(root["pid"]), root) and owns_listener(root, port),
                    "Owned API process/socket changed during constrained-output probe")
        record.update(status="passed", completed_utc=utc(), elapsed_seconds=time.monotonic() - started)
        return record
    except Exception as error:
        record.update(status="failed", failed_utc=utc(), elapsed_seconds=time.monotonic() - started,
                      error_type=type(error).__name__, error=str(error))
        raise ProbeError(f"Mandatory constrained-output probe failed: {error}", record) from error


def constrained_output_probe(root: dict, port: int, extra_body_builder,
                             timeout_seconds: float = PROBE_TIMEOUT_SECONDS) -> dict:
    """One nonce request; never retry or accept /models as proof."""
    nonce = uuid.uuid4().hex
    record = perform_probe(
        root, port, lambda: probe_request(nonce, extra_body_builder(probe_schema(nonce))),
        lambda payload: verify_probe_response(payload, nonce), {
            "kind": "structured_outputs_json_nonce_v1",
            "timeout_seconds": timeout_seconds, "response_byte_limit": PROBE_MAX_RESPONSE_BYTES,
            "nonce_location": "Only in structured_outputs.json, absent from messages.",
            "request_seed_base": 42, "request_seed_components": list(PROBE_SEED_COMPONENTS),
            "scope": "Default-contract smoke with 2048 tokens; not a claim that every phase uses "
                     "this token budget or validation of every application schema/world outcome.",
        }, maximum=PROBE_TIMEOUT_SECONDS)
    validate_probe_attestation(record, root, port)
    return record


def validate_funding_probe_attestation(record: dict, root: dict, port: int,
                                       fixture: dict, binding: dict, *, plan: Plan | None = None) -> None:
    if fixture.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL:
        return validate_sequential_probe_attestation(record, root, port, fixture, binding, plan)
    require(isinstance(record, dict) and record.get("status") == "passed"
            and record.get("http_status") == 200, "Mandatory funding-panel25 probe has not passed")
    require(same_process(record.get("process_identity"), root)
            and record.get("endpoint") == f"http://127.0.0.1:{port}/v1/chat/completions",
            "Funding probe is bound to another server process/endpoint")
    require(record.get("fixture") == binding, "Funding probe immutable fixture binding changed")
    request = record.get("request")
    require(isinstance(request, dict) and request == funding_probe_request(fixture, {
        "structured_outputs": {"json": fixture["json_schema"]},
        "chat_template_kwargs": {"enable_thinking": True}, "seed": request.get("seed")}),
        "Funding probe request differs from exact fixture/production settings")
    require(record.get("request_seed_base") == 42
            and record.get("request_seed_components") == list(FUNDING_SEED_COMPONENTS),
            "Funding probe lost production base-seed/derivation context")
    builder = record.get("shared_client_builder")
    if builder:
        require(builder.get("request_seed_base") == 42
                and builder.get("request_seed") == request["seed"]
                and builder.get("seed_components") == list(FUNDING_SEED_COMPONENTS),
                "Shared production seed helper disagrees with funding probe request")
    require(record.get("request_sha256") == digest_bytes(json_bytes(request))
            and record.get("response_sha256") == digest_bytes(record["response_body"].encode()),
            "Funding probe request/response hash mismatch")
    require(record.get("validation") == verify_funding_probe_response(
        read_json(record["response_body"].encode()), fixture), "Funding probe validation changed")


def funding_panel25_probe(root: dict, port: int, extra_body_builder, fixture: dict,
                          binding: dict, timeout_seconds: float = FUNDING_PROBE_TIMEOUT_SECONDS) -> dict:
    record = perform_probe(
        root, port, lambda: funding_probe_request(fixture, extra_body_builder(fixture["json_schema"])),
        lambda payload: verify_funding_probe_response(payload, fixture), {
            "kind": "funding_panel25", "fixture": binding,
            "timeout_seconds": timeout_seconds, "response_byte_limit": FUNDING_PROBE_MAX_RESPONSE_BYTES,
            "request_seed_base": 42, "request_seed_components": list(FUNDING_SEED_COMPONENTS),
            "scope": "Producer-authored deterministic CPU proposal fixture, not a simulated observation. "
                     "One representative 25-application production prompt/schema, 8192 output tokens; "
                     "passing does not establish validity or fit of all later histories/prompts.",
        }, maximum=FUNDING_PROBE_TIMEOUT_SECONDS)
    validate_funding_probe_attestation(record, root, port, fixture, binding)
    return record


def pointer(value: Any, path: str) -> Any:
    try:
        for segment in path[1:].split("/"):
            segment = segment.replace("~1", "/").replace("~0", "~")
            value = value[int(segment)] if isinstance(value, list) else value[segment]
        return value
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise QueueError(f"Missing artifact JSON Pointer: {path}") from error


def output_path(plan: Plan, value: str | Path) -> Path:
    """Allow only the sanctioned worktree/outputs alias, never other links."""
    path = absolute(str(value), "output path")
    alias = plan.worktree / "outputs"
    if within(path, alias):
        require(alias.resolve(strict=True) == plan.private_root / "outputs",
                "Worktree outputs alias changed")
        path = plan.private_root / "outputs" / path.relative_to(alias)
    require(within(path, plan.private_root / "outputs") and path != plan.private_root / "outputs",
            f"Path escapes private outputs: {path}")
    return path


def check_output_components(plan: Plan, path: Path, *, allow_missing: bool = False) -> None:
    current = plan.private_root
    for component in path.relative_to(plan.private_root).parts:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            if allow_missing:
                return
            raise
        require(not stat.S_ISLNK(info.st_mode), f"Symlink output component prohibited: {current}")
        require(info.st_uid == os.getuid(), f"Output component is not owned: {current}")
        if current != path:
            require(stat.S_ISDIR(info.st_mode), f"Output parent is not a directory: {current}")


def open_output_file(plan: Plan, path: Path):
    """openat/O_NOFOLLOW walk also closes the parent-symlink check/open race."""
    check_output_components(plan, path)
    directory = os.open(plan.private_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        pieces = path.relative_to(plan.private_root).parts
        for name in pieces[:-1]:
            next_directory = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=directory)
            os.close(directory)
            directory = next_directory
        fd = os.open(pieces[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    return os.fdopen(fd, "rb")


def stat_binding(info) -> dict:
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def artifact_gate(job: dict, plan: Plan, *, started_ns: int | None = None) -> list[dict]:
    require(type(started_ns) is int and started_ns > 0,
            "Artifact gate requires the persisted job launch timestamp")
    roots = [output_path(plan, path) for path in job["output_roots"]]
    receipts = []
    for artifact in job["artifacts"]:
        json_limit = artifact_json_limit(job, artifact, worktree=plan.worktree,
                                         private_root=plan.private_root)
        path = output_path(plan, artifact["path"])
        require(any(within(path, root) and path != root for root in roots),
                f"Artifact is outside this job's fresh output_roots: {path}")
        with open_output_file(plan, path) as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                    and info.st_uid == os.getuid() and info.st_size >= artifact.get("min_bytes", 1),
                    f"Artifact missing, empty, linked, or not owned: {path}")
            checked_ns = time.time_ns()
            require(started_ns <= info.st_mtime_ns <= checked_ns
                    and started_ns <= info.st_ctime_ns <= checked_ns,
                    f"Artifact timestamps do not belong to this job launch: {path}")
            before = stat_binding(info)
            if artifact.get("equals"):
                require(info.st_size <= json_limit,
                        f"JSON gate exceeds byte allowance: {path}; bytes={info.st_size}, limit={json_limit}")
                # Read at most the admitted size plus one growth-detection byte.
                raw = stream.read(info.st_size + 1)
                require(len(raw) <= json_limit,
                        f"JSON gate exceeds byte allowance after read: {path}; "
                        f"bytes={len(raw)}, limit={json_limit}")
                require(len(raw) == info.st_size, f"Artifact changed while reading JSON: {path}")
                value = read_json(raw)
                del raw
                for key, expected in artifact["equals"].items():
                    actual = pointer(value, key)
                    require(type(actual) is type(expected) and actual == expected,
                            f"Artifact predicate failed: {path} {key} expected {expected!r}")
                del value
                stream.seek(0)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            require(stat_binding(os.fstat(stream.fileno())) == before,
                    f"Artifact changed while hashing: {path}")
        with open_output_file(plan, path) as check:
            require(stat_binding(os.fstat(check.fileno())) == before,
                    f"Artifact path changed while hashing: {path}")
        receipts.append({"path": artifact["path"], "canonical_path": str(path),
                         **before, "sha256": digest.hexdigest(), "job_started_epoch_ns": started_ns})
    return receipts


def ensure_fresh(job: dict, plan: Plan) -> None:
    for value in job["output_roots"]:
        root = output_path(plan, value)
        check_output_components(plan, root, allow_missing=True)
        require(not root.exists() and not root.is_symlink(),
                f"Historical or partial output exists; refusing reuse: {root}")
    for artifact in job["artifacts"]:
        require(not Path(artifact["path"]).exists(), "Historical artifact reuse prohibited")


def initial_choices_artifact(plan: Plan, artifacts: list[dict]) -> dict:
    """Require a producer-frozen cache and its exact fresh-artifact receipt binding."""
    descriptor = plan.initialization
    require(descriptor is not None, "No propensity initialization is declared")
    matches = [artifact for artifact in artifacts if artifact["path"] == descriptor["artifact"]]
    require(len(matches) == 1, "Initialization receipt lacks its unique initial_choices artifact")
    proof = matches[0]
    path = output_path(plan, descriptor["artifact"])
    require(proof["canonical_path"] == str(path), "Initial choices path binding changed")
    with open_output_file(plan, path) as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                and info.st_nlink == 1 and not info.st_mode & 0o222
                and 0 < info.st_size <= 64 * 1024 * 1024,
                "Initial choices must be an owned immutable regular JSON file (0444)")
        require(stat_binding(info) == {key: proof[key] for key in stat_binding(info)},
                "Initial choices file identity/timestamps changed")
        raw = stream.read()
        read_json(raw)
        require(digest_bytes(raw) == proof["sha256"]
                and stat_binding(os.fstat(stream.fileno())) == stat_binding(info),
                "Initial choices content changed after initialization")
    return proof


def bound_json_artifact(plan: Plan, declaration: dict, artifacts: list[dict]) -> dict:
    proof = next(artifact for artifact in artifacts if artifact["path"] == declaration["path"])
    with open_output_file(plan, output_path(plan, declaration["path"])) as stream:
        info = os.fstat(stream.fileno())
        require(info.st_size <= 64 * 1024 * 1024
                and stat_binding(info) == {key: proof[key] for key in stat_binding(info)},
                "Propensity JSON artifact file binding changed")
        raw = stream.read()
        require(digest_bytes(raw) == proof["sha256"]
                and stat_binding(os.fstat(stream.fileno())) == stat_binding(info),
                "Propensity JSON artifact content changed")
    result = read_json(raw)
    require(isinstance(result, dict), "Propensity JSON artifact must be an object")
    return result


def propensity_request_count(plan: Plan, job: dict, artifacts: list[dict]) -> int:
    declaration = propensity_audit_artifacts(job)["llm_request_audit.summary.json"]
    summary = bound_json_artifact(plan, declaration, artifacts)
    count = summary.get("n_requests")
    require(type(count) is int and count > 0, "Propensity request audit must have integer n_requests>0")
    return count


@contextlib.contextmanager
def propensity_admission_driver(plan: Plan):
    """Load only this worktree's stdlib admission code and restore import state."""
    names = ("james_mechanisms", "james_propensity_policy", "james_propensity",
             "james_funding_compact", "james_funding_validation", "james_funding_sequential")
    saved = {name: sys.modules.get(name) for name in names}
    old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(plan.worktree / "src"), str(plan.worktree)]
        for name in names:
            sys.modules.pop(name, None)
        for name in names[:3]:
            path = plan.worktree / "src" / (name + ".py")
            require(path.is_file() and within(path.resolve(), plan.worktree.resolve()),
                    "Propensity admission source escapes the frozen worktree")
            spec = importlib.util.spec_from_file_location(name, path)
            require(spec is not None and spec.loader is not None, "Cannot load propensity admission")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        require(module.PROTOCOL == PROPENSITY_SEQUENTIAL_PROTOCOL
                and module.FUNDING_SELECTION_PROTOCOL == SEQUENTIAL_PROTOCOL
                and callable(module.source_identity) and callable(module.validate_funding_evidence),
                "Propensity admission API/protocol does not match the frozen plan")
        yield module
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name in names:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]


def propensity_producer_directory(plan: Plan, job: dict) -> tuple[Path, dict]:
    """Retain the trusted producer spelling required by literal raw-audit bindings."""
    alias = plan.worktree / "outputs"

    def link_identity():
        before = alias.lstat()
        require(stat.S_ISLNK(before.st_mode) and before.st_uid == os.getuid(),
                "Propensity outputs alias must be an owned symlink")
        metadata = {**stat_binding(before), "mode": before.st_mode,
                    "uid": before.st_uid, "gid": before.st_gid}
        target = os.readlink(alias)
        after = alias.lstat()
        require(metadata == {**stat_binding(after), "mode": after.st_mode,
                             "uid": after.st_uid, "gid": after.st_gid},
                "Propensity outputs alias changed during identity capture")
        return {**metadata, "target": target}

    alias_identity = link_identity()
    declaration = propensity_manifest_artifact(job)
    canonical = output_path(plan, Path(declaration["path"]).parent)
    require(canonical == plan.private_root / "outputs/docs" / job["id"],
            "Propensity admission docs must belong to this exact job")
    producer = plan.worktree / "outputs/docs" / job["id"]
    require(output_path(plan, producer) == canonical, "Propensity producer alias changed")
    check_output_components(plan, canonical)
    directory_stat = canonical.stat()
    identity = stat_binding(directory_stat)
    require(stat.S_ISDIR(directory_stat.st_mode)
            and stat_binding(producer.stat()) == identity,
            "Propensity producer alias does not identify the owned docs directory")
    require(link_identity() == alias_identity,
            "Propensity outputs alias changed during directory identity capture")
    return producer, {"canonical_directory": identity, "outputs_alias": alias_identity}


def propensity_funding_admission(plan: Plan, job: dict, artifacts: list[dict],
                                provenance: dict, started_ns: int) -> dict:
    """Re-run the owner's exact read-only validator with trusted frozen sources."""
    declaration = propensity_manifest_artifact(job)
    directory, directory_identity = propensity_producer_directory(plan, job)
    manifest = bound_json_artifact(plan, declaration, artifacts)
    checkpoint = output_path(plan, manifest.get("args", {}).get("output_dir", ""))
    expected_checkpoint = plan.private_root / "outputs/checkpoints" / job["id"]
    require(checkpoint == expected_checkpoint
            and checkpoint in [output_path(plan, path) for path in job["output_roots"]],
            "Propensity native funding ledger must belong to this job's fresh checkpoint root")
    years = 0 if is_propensity_initializer(job) else 10
    present = [checkpoint / f"funding_applications_year_{year}.jsonl" for year in range(1, years + 1)
               if os.path.lexists(checkpoint / f"funding_applications_year_{year}.jsonl")]
    ledger_job = {**job, "artifacts": [{"path": str(path), "min_bytes": 0} for path in present]}
    ledger_files = artifact_gate(ledger_job, plan, started_ns=started_ns)
    with propensity_admission_driver(plan) as driver:
        actual = driver.source_identity()
        frozen = provenance["sources"]
        require(isinstance(actual, dict) and set(PROPENSITY_SOURCE_FILES) <= set(actual)
                and all(name in frozen and actual[name] == frozen[name] for name in actual),
                "Propensity validator source map differs from queue-frozen source hashes")
        expected = {name: frozen[name] for name in actual}
        report = driver.validate_funding_evidence(directory, manifest, expected_source=expected)
        require(driver.source_identity() == expected, "Propensity admission sources changed while validating")
    require(propensity_producer_directory(plan, job) == (directory, directory_identity),
            "Propensity producer directory changed while validating")
    require(isinstance(report, dict) and report.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL
            and type(report.get("scientific_years")) is int and report["scientific_years"] == years,
            "Propensity admission returned the wrong funding protocol or role")
    require(artifact_gate(ledger_job, plan, started_ns=started_ns) == ledger_files
            and artifact_gate(job, plan, started_ns=started_ns) == artifacts,
            "Propensity admission files changed while validating")
    observed = {Path(item["path"]).name: item["sha256"] for item in ledger_files}
    absent = [year for year in range(1, years + 1)
              if f"funding_applications_year_{year}.jsonl" not in observed]
    require(report.get("funding_application_evidence") ==
            {"present_sha256": observed, "absent_years": absent}
            and {path.name for path in checkpoint.glob("funding_applications_year_*.jsonl")} == set(observed),
            "Propensity admission ledger bytes/absence do not match the fresh owned files")
    return {"validation": report, "expected_source": expected, "native_ledger_files": ledger_files}


def initialization_state_hashes(manifest: dict) -> dict:
    hashes = {key: manifest.get(key) for key in ("pre_choice_state_hash", "initial_state_hash")}
    require(all(isinstance(value, str) and bool(re.fullmatch("[0-9a-f]{64}", value))
                for value in hashes.values()), "Initialization state hashes must be SHA256 values")
    return hashes


def initialization_binding(plan: Plan, state: dict, provenance: dict) -> dict:
    """Receipt/source/cache binding for this fixed init -> four-world dependency."""
    descriptor = plan.initialization
    require(descriptor is not None, "No propensity initialization is declared")
    init_id = descriptor["job_id"]
    record = state["jobs"][init_id]
    require(record["status"] == "complete", "World cannot start before initialization receipt is complete")
    receipt_path = plan.directory / f"job-{init_id}.receipt.json"
    raw = queue_record_bytes(plan, receipt_path)
    require(digest_bytes(raw) == record["receipt_sha256"], "Initialization receipt changed")
    with open_output_file(plan, receipt_path) as stream:
        require(not os.fstat(stream.fileno()).st_mode & 0o222, "Initialization receipt is not immutable")
    receipt = read_json(raw)
    init_job = next(job for job in plan.jobs if job["id"] == init_id)
    source_hash = digest_bytes(json_bytes(provenance["sources"]))
    require(receipt.get("plan_sha256") == plan.sha256 and receipt.get("job") == init_id
            and receipt.get("job_spec_sha256") == digest_bytes(json_bytes(init_job))
            and receipt.get("source_sha256") == source_hash
            and provenance["sources"] == state["provenance"]["sources"],
            "Initialization receipt plan/job/source binding changed")
    proof = initial_choices_artifact(plan, receipt["artifacts"])
    require(receipt.get("initial_choices") == proof and record["artifacts"] == receipt["artifacts"],
            "Initialization choices receipt binding changed")
    manifest = bound_json_artifact(plan, propensity_manifest_artifact(init_job), receipt["artifacts"])
    state_hashes = initialization_state_hashes(manifest)
    require(receipt.get("state_hashes") == state_hashes, "Initialization state-hash receipt binding changed")
    completed_ns = receipt.get("completed_epoch_ns")
    require(type(completed_ns) is int and completed_ns >= record["started_epoch_ns"],
            "Initialization receipt completion time is invalid")
    result = {"initialization_job": init_id, "initialization_receipt": str(receipt_path),
            "initialization_receipt_sha256": record["receipt_sha256"],
            "initial_choices": proof, "initialization_source_sha256": source_hash,
            "initialization_completed_epoch_ns": completed_ns, "plan_sha256": plan.sha256, **state_hashes}
    if plan.spec.get("funding_selection_protocol"):
        manifest_declaration = propensity_manifest_artifact(init_job)
        manifest_proof = next(a for a in receipt["artifacts"] if a["path"] == manifest_declaration["path"])
        result.update(initialization_manifest_path=manifest_proof["canonical_path"],
                      initialization_manifest_sha256=manifest_proof["sha256"])
    return result


def propensity_world_input_binding(plan: Plan, job: dict, artifacts: list[dict], dependency: dict) -> dict:
    declaration = propensity_manifest_artifact(job)
    manifest = bound_json_artifact(plan, declaration, artifacts)
    require(manifest.get("initial_choices_sha256") == dependency["initial_choices"]["sha256"],
            "World reported a different initial choices cache SHA256")
    require(all(manifest.get(key) == dependency[key] for key in
                ("pre_choice_state_hash", "initial_state_hash")),
            "World state hashes differ from the treatment-blind initializer")
    result = {"initial_choices_sha256": manifest["initial_choices_sha256"],
            "initial_state_hash": manifest["initial_state_hash"],
            "pre_choice_state_hash": manifest["pre_choice_state_hash"]}
    if plan.spec.get("funding_selection_protocol"):
        require(str(output_path(plan, manifest.get("initialization_manifest_path", ""))) ==
                dependency["initialization_manifest_path"]
                and manifest.get("initialization_manifest_sha256") ==
                dependency["initialization_manifest_sha256"],
                "World reported another initialization manifest path/hash")
        result.update(initialization_manifest_path=dependency["initialization_manifest_path"],
                      initialization_manifest_sha256=dependency["initialization_manifest_sha256"])
    return result


@dataclass
class Owned:
    process: Any
    identity: dict
    log: Any


@dataclass
class Server:
    name: str
    pair: list[GPU]
    locks: list[Lock]
    port: int
    owned: Owned
    command: list[str]
    started: float
    manifest_path: Path
    ready: bool = False
    job: dict | None = None
    client: Owned | None = None
    reservation_token: str | None = None
    probe: dict | None = None
    funding_probe: dict | None = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"


class Queue:
    def __init__(self, plan: Plan, provenance: dict, state: dict):
        self.plan, self.provenance, self.state = plan, provenance, state
        self.servers: list[Server] = []
        self.lock_root = Path("/tmp") / f"scienceutopia-gpu-locks-{os.getuid()}"
        self.interrupted = False
        self.probing = False

    def save(self):
        self.state["updated_utc"] = utc()
        atomic_write(self.plan.directory / "status.json", json_bytes(self.state))

    def event(self, name: str, **details):
        record = {"utc": utc(), "event": name, **details}
        with (self.plan.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        print(json.dumps(record, sort_keys=True), flush=True)

    def pending(self):
        return [job for job in self.plan.jobs if self.state["jobs"][job["id"]]["status"] == "pending"]

    def client_budget_seconds(self):
        """Also bound a blocking server probe by the earliest active client deadline."""
        remaining = TIMEOUT_SECONDS
        if self.plan.spec.get("continuation"):
            for server in self.servers:
                if server.client is not None:
                    record = self.state["jobs"][server.job["id"]]
                    now = check_pending_client_deadline(self.plan, server.job["id"], record)
                    remaining = min(remaining, TIMEOUT_SECONDS -
                                    (now - record["started_epoch_ns"]) / 1_000_000_000)
        return remaining

    def runnable(self):
        if self.plan.initialization:
            init_id = self.plan.initialization["job_id"]
            if self.state["jobs"][init_id]["status"] != "complete":
                return [job for job in self.pending() if job["id"] == init_id]
        return self.pending()

    def server_limit(self):
        if self.plan.initialization and self.state["jobs"][
                self.plan.initialization["job_id"]]["status"] != "complete":
            return 1
        return 4

    def environment(self, pair: list[GPU], job: dict | None = None) -> dict:
        # Keep credentials/network configuration, but discard unsafe inherited runtime overrides.
        env = {k: v for k, v in os.environ.items()
               if k not in (PROTECTED_ENV - {"HOME", "CODEX_HOME"})
               and not k.startswith(("VLLM_", "CUDA_", "NCCL_", "HF_", "PYTHON"))}
        if job:
            env.update(job["env"])
        env.update({
            "CUDA_VISIBLE_DEVICES": (str(pair[0].index) if job
                                     else ",".join(str(gpu.index) for gpu in pair)),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONHASHSEED": "42",
            "HF_HOME": str(self.plan.hf_home),
            "HF_HUB_CACHE": str(self.plan.hf_home / "hub"),
            "HUGGINGFACE_HUB_CACHE": str(self.plan.hf_home / "hub"),
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_CACHE_ROOT": str(self.plan.private_root / "cache/vllm"),
            "XDG_CACHE_HOME": str(self.plan.private_root / "cache"),
            "TORCH_HOME": str(self.plan.private_root / "cache/torch"),
            "TMPDIR": str(self.plan.private_root / "tmp"),
            "PYTHONDONTWRITEBYTECODE": "1", "WANDB_MODE": "disabled",
            "JAMES_QUEUE_PLAN_SHA256": self.plan.sha256,
            "JAMES_QUEUE_ID": self.plan.spec["queue_id"],
        })
        return env

    def spawn(self, cmd: list[str], env: dict, logpath: Path) -> Owned:
        stream = logpath.open("xb")
        try:
            require(not self.interrupted, "Queue interrupted before process creation")
            proc = subprocess.Popen(cmd, env=env, cwd=self.plan.worktree,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    close_fds=True)
        except BaseException:
            stream.close()
            raise
        identity = proc_identity(proc.pid)
        if identity is None:
            proc.wait(timeout=5)
            stream.close()
            raise QueueError(f"Process exited before identity capture: {cmd}")
        require(identity["session"] == proc.pid and identity["uid"] == os.getuid(),
                "Process did not start in a private owned session")
        return Owned(proc, identity, stream)

    def update_reservation(self, server: Server, phase: str):
        require(server.reservation_token is not None, "Missing durable server reservation")
        for gpu in server.pair:
            path = reservation_path(self.lock_root, gpu.uuid)
            record = read_reservation(path)
            require(record is not None and record["token"] == server.reservation_token,
                    f"GPU reservation token changed: {path}")
            record.update(phase=phase, server_identity=server.owned.identity,
                          client_identity=server.client.identity if server.client else None,
                          job=server.job["id"] if server.job else None, updated_utc=utc())
            atomic_write(path, json_bytes(record))

    def start_server(self, pair: list[GPU], locks: list[Lock]) -> None:
        try:
            recheck_pending(self.plan, self.provenance)
            if "server_profile" in self.plan.spec:
                validate_selected_runtime(self.plan, self.provenance)
                require(len(pair) == selected_server_spec(self.plan)["tensor_parallel_size"]
                        and len({gpu.uuid for gpu in pair}) == len(pair)
                        and len(locks) == len(pair),
                        "Selected profile requires exactly one reserved GPU")
            require(len(self.servers) < self.server_limit(), "Current initialization/server limit reached")
            port_lock = None
            port = None
            for candidate in range(18080, 18180):
                lock = Lock(self.lock_root / f"port-{candidate}.lock")
                if not lock.acquire():
                    continue
                try:
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", candidate))
                    port, port_lock = candidate, lock
                    break
                except OSError:
                    lock.close()
            require(port is not None, "No free queue HTTP port")
            locks.append(port_lock)
            current = {gpu.uuid: gpu for gpu in query_gpus()}
            require(all(gpu.uuid in current and current[gpu.uuid].idle
                        and current[gpu.uuid].index == gpu.index for gpu in pair),
                    "GPU became busy immediately before server launch")
            name = f"server-{len(self.state['servers']) + 1:03d}"
            command = server_command(self.plan, port)
            started = time.time()
            controller = proc_identity(os.getpid())
            require(controller is not None and controller["boot_id"] == boot_id(),
                    "Cannot attest controller identity before launch")
            token = uuid.uuid4().hex
            intent = {
                "schema_version": 1, "token": token, "phase": "launch_intent",
                "boot_id": controller["boot_id"], "controller_identity": controller,
                "server_identity": None, "client_identity": None,
                "plan_sha256": self.plan.sha256, "queue_directory": str(self.plan.directory),
                "server": name, "gpu_ids": [gpu.index for gpu in pair],
                "gpu_uuids": [gpu.uuid for gpu in pair],
                "launch_command": command, "started_utc": utc(started),
                "recovery": "Unreleased same-boot intent blocks reuse even when idle. "
                            "Never infer release from controller death or missing CUDA PIDs.",
            }
            if "server_profile" in self.plan.spec:
                intent.update(selected_runtime_binding(self.plan))
            # Both records are fsynced under both GPU flocks BEFORE Popen.
            # A crash between fork and PID capture leaves an uncertain intent;
            # another queue must quarantine the pair, not guess that it is free.
            for gpu in pair:
                path = reservation_path(self.lock_root, gpu.uuid)
                require(not reservation_blocks(path), f"Unreleased GPU reservation: {path}")
                atomic_write(path, json_bytes(intent))
            self.state["servers"][name] = {
                "status": "launch_intent", "identity": None,
                "controller_identity": controller, "reservation_token": token,
                "port": port, "gpu_ids": [gpu.index for gpu in pair], "command": command,
                "started_utc": utc(started)}
            if "server_profile" in self.plan.spec:
                self.state["servers"][name].update(
                    **selected_runtime_binding(self.plan), gpu_uuids=[gpu.uuid for gpu in pair])
            self.save()
            # Journaling can take time; recheck telemetry after the durable
            # intent and immediately before the actual process creation.
            current = {gpu.uuid: gpu for gpu in query_gpus()}
            require(all(gpu.uuid in current and current[gpu.uuid].idle
                        and current[gpu.uuid].index == gpu.index for gpu in pair),
                    "GPU became busy after reservation intent; launch canceled and intent quarantined")
            require(not self.interrupted, "Queue interrupted before server process creation")
            owned = self.spawn(command, self.environment(pair),
                               self.plan.directory / f"{name}.log")
            server = Server(name, pair, locks, port, owned, command, started,
                            self.plan.directory / f"{name}.manifest.json", reservation_token=token)
            # Register in memory before any fallible write, so finally can clean it.
            self.servers.append(server)
            self.state["servers"][name].update(status="starting", identity=owned.identity)
            self.update_reservation(server, "active")
            self.save()
            self.event("server_started", server=name, gpu_ids=[g.index for g in pair], pid=owned.process.pid)
        except BaseException:
            if not any(s.locks is locks for s in self.servers):
                for lock in locks:
                    lock.close()
            raise

    def readiness_probe(self, server: Server, kind: str, builder_record: dict, operation):
        """Persist a failed live probe and allow SIGTERM to interrupt its blocking HTTP call."""
        self.client_budget_seconds()
        self.state["servers"][server.name]["status"] = "probing_" + kind
        self.save()
        self.event(kind + "_probe_started", server=server.name)
        self.probing = True
        try:
            record = operation()
        except ProbeError as error:
            error.attestation["shared_client_builder"] = builder_record
            self.state["servers"][server.name].update(
                status=kind + "_probe_failed", probe_failure=error.attestation)
            suffix = "probe" if kind == "structured_outputs" else "funding-probe"
            atomic_write(self.plan.directory / f"{server.name}.{suffix}.failure.json",
                         json_bytes(error.attestation), exclusive=True, readonly=True)
            self.save()
            raise
        finally:
            self.probing = False
        require(not self.interrupted, "Queue interrupted during constrained-output probe")
        self.client_budget_seconds()
        record["shared_client_builder"] = builder_record
        return record

    def attest(self, server: Server):
        # Called only after live process, owned socket, /health, /models, and worker checks.
        require(not self.interrupted, "Queue interrupted before constrained-output probe")
        require(server.client is None, "Cannot probe readiness while a client is running")
        validate_live_selection(self.plan, self.provenance, server)
        server.ready, server.probe, server.funding_probe = False, None, None
        fixture, fixture_binding = load_funding_fixture(self.plan)
        builder, builder_record = load_shared_probe_builder(self.plan)
        probe = self.readiness_probe(server, "structured_outputs", builder_record,
            lambda: constrained_output_probe(
                server.owned.identity, server.port, builder,
                timeout_seconds=min(PROBE_TIMEOUT_SECONDS, self.state["deadline_epoch"] - time.time(),
                                    self.client_budget_seconds())))
        validate_probe_attestation(probe, server.owned.identity, server.port)
        self.state["servers"][server.name]["structured_output_probe"] = probe
        if self.plan.spec.get("funding_selection_protocol"):
            _, helper = load_sequential_api(self.plan)
            funding_metadata = {
                "path": str(self.plan.worktree / "utopia/models/structured_outputs.py"),
                "sha256": file_hash(self.plan.worktree / "utopia/models/structured_outputs.py"),
                "entrypoint": "build_vllm_extra_body", "sequential_helper": helper,
                "request_seed_base": 42, "namespace": "technical_probe", "attempts_per_step": 1}
            funding_probe = self.readiness_probe(server, "funding_panel25", funding_metadata,
                lambda: sequential_funding_probe(
                    self.plan, server.owned.identity, server.port, fixture, fixture_binding,
                    self.plan.directory, server.name, cancelled=lambda: self.interrupted,
                     timeout_seconds=min(SEQUENTIAL_PROBE_SECONDS,
                                         self.state["deadline_epoch"] - time.time(),
                                         self.client_budget_seconds())))
        else:
            funding_builder, funding_metadata = load_shared_probe_builder(self.plan, FUNDING_SEED_COMPONENTS)
            funding_probe = self.readiness_probe(server, "funding_panel25", funding_metadata,
                lambda: funding_panel25_probe(
                    server.owned.identity, server.port, funding_builder, fixture, fixture_binding,
                     timeout_seconds=min(FUNDING_PROBE_TIMEOUT_SECONDS,
                                         self.state["deadline_epoch"] - time.time(),
                                         self.client_budget_seconds())))
        validate_funding_probe_attestation(funding_probe, server.owned.identity, server.port,
                                           fixture, fixture_binding, plan=self.plan)
        require(load_funding_fixture(self.plan)[1] == fixture_binding,
                "Funding fixture changed during readiness probes")
        if self.plan.spec.get("funding_selection_protocol"):
            require(source_hashes(self.plan) == self.provenance["sources"],
                    "Frozen sequential source/protocol changed during readiness")
        live_inventory = query_gpus()
        self.workers_healthy(server, live_inventory, require_workers=True)
        gpu_pids = sorted(set().union(*(g.pids for g in live_inventory
                                       if g.uuid in {x.uuid for x in server.pair})))
        selected = selected_server_spec(self.plan)
        manifest = {
            **selected, "schema_version": 1, "endpoint": server.url,
            "host": self.provenance["host"], "pid": server.owned.process.pid,
            "process_identity": server.owned.identity, "worker_pids_at_ready": gpu_pids,
            "gpu_ids": [g.index for g in server.pair],
            "gpu_uuids": [g.uuid for g in server.pair],
            "gpus": [dict(index=g.index, uuid=g.uuid, name=g.name, total_mib=g.total_mib,
                          driver=g.driver, pci_bus_id=g.pci_bus_id) for g in server.pair],
            "server_gpus": ",".join(str(g.index) for g in server.pair),
            "client_gpu": str(server.pair[0].index),
            "client_gpu_uuid": server.pair[0].uuid,
            "launch_command": server.command, "command": server.command,
            "started_utc": utc(server.started), "ready_utc": utc(),
            "environment": {k: v for k, v in self.environment(server.pair).items()
                            if k in PROTECTED_ENV or k in
                            {"HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"}},
            "versions": self.provenance["versions"],
            "runtime_files": self.provenance.get("runtime_files", {}),
            "git_head": self.provenance["git_head"], "branch": self.provenance["branch"],
            "plan_sha256": self.plan.sha256,
            "source_sha256": digest_bytes(json_bytes(self.provenance["sources"])),
            "runtime_hash": digest_bytes(json_bytes(selected)),
            "reservation": "Advisory flock plus durable pre-fork intent and nvidia-smi rechecks; "
                           "unreleased same-boot records quarantine orphan/loading pairs; no foreign PID is killed.",
            "reservation_token": server.reservation_token,
            "structured_output_probe": probe,
            "funding_panel25_probe": funding_probe,
            "client_embedding_headroom": "Server fraction .80; remaining memory is not a hard reservation.",
            "health_gate": "Live owned API PID/socket, /health=200, expected /v1/models, live owned GPU workers "
                           "on both devices, mandatory real strict structured_outputs.json nonce probe "
                           "and complete funding-panel25 probe with exact input token count.",
            "interpretation_limitation": LIMITATION,
        }
        if "server_profile" in self.plan.spec:
            manifest.update(
                server_profile=self.plan.spec["server_profile"],
                client_embedding_headroom=(
                    "Server fraction .94; remaining memory is not a hard reservation."),
                health_gate=(
                    "Live owned API PID/socket, /health=200, expected /v1/models, live owned GPU "
                    "worker on the selected device, mandatory real strict structured_outputs.json "
                    "nonce probe and complete funding-panel25 probe with exact input token count."))
        validate_live_selection(self.plan, self.provenance, server, manifest)
        if self.plan.spec.get("funding_selection_protocol"):
            manifest["funding_selection_protocol"] = SEQUENTIAL_PROTOCOL
        manifest["manifest_content_sha256"] = digest_bytes(json_bytes(manifest))
        require(not self.interrupted, "Queue interrupted before readiness commit")
        atomic_write(server.manifest_path, json_bytes(manifest), exclusive=True, readonly=True)
        require(not self.interrupted, "Queue interrupted before readiness commit")
        server.probe = probe
        server.funding_probe = funding_probe
        server.ready = True
        self.state["servers"][server.name].update(status="ready",
            manifest=str(server.manifest_path), manifest_sha256=file_hash(server.manifest_path))
        self.save()

    def workers_healthy(self, server: Server, inventory: list[GPU], *, require_workers: bool):
        validate_live_selection(self.plan, self.provenance, server)
        members = {x["pid"] for x in session_members(server.owned.identity)}
        if server.client:
            members.update(x["pid"] for x in session_members(server.client.identity))
        for original in server.pair:
            gpu = next((x for x in inventory if x.uuid == original.uuid), None)
            require(gpu is not None and gpu.index == original.index, "GPU mapping changed")
            foreign = gpu.pids - members
            require(not foreign,
                    f"Foreign compute PID(s) on owned GPU {gpu.index}: {sorted(foreign)}; stop own work only")
            if require_workers:
                # Client processes cannot stand in for a dead inference worker.
                server_members = {x["pid"] for x in session_members(server.owned.identity)}
                require(bool(gpu.pids & server_members),
                        f"No live owned server compute worker on GPU {gpu.index}")

    def start_job(self, server: Server, job: dict):
        require(not self.interrupted, "Queue interrupted before client launch")
        recheck_pending(self.plan, self.provenance, fresh_jobs=[job])
        require(server.client is None and server.ready, "One job per ready server is required")
        require(any(job == candidate for candidate in self.runnable()),
                "Job is not runnable; initialization dependency is incomplete or job already started")
        require(source_hashes(self.plan) == self.provenance["sources"], "Sources changed since queue freeze")
        dependency = None
        if is_propensity_world(job):
            validate_completed(self.plan, self.state, self.provenance, require_all=False)
            dependency = initialization_binding(self.plan, self.state, self.provenance)
        ensure_fresh(job, self.plan)
        require(server.owned.process.poll() is None, "Server died before job")
        self.workers_healthy(server, query_gpus(), require_workers=True)
        require(endpoint_ready(server.owned.identity, server.port), "Server health/model gate failed before job")
        require(file_hash(server.manifest_path) ==
                self.state["servers"][server.name]["manifest_sha256"], "Server manifest changed")
        manifest = read_json(queue_record_bytes(self.plan, server.manifest_path))
        validate_live_selection(self.plan, self.provenance, server, manifest)
        probe = manifest.get("structured_output_probe")
        validate_probe_attestation(probe, server.owned.identity, server.port)
        require(probe.get("shared_client_builder"), "Probe did not attest the shared production builder")
        require(server.probe == probe, "Server probe differs from immutable readiness attestation")
        fixture, binding = load_funding_fixture(self.plan)
        funding_probe = manifest.get("funding_panel25_probe")
        validate_funding_probe_attestation(
            funding_probe, server.owned.identity, server.port, fixture, binding, plan=self.plan)
        if self.plan.spec.get("funding_selection_protocol"):
            require(manifest.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL,
                    "Server manifest lacks sequential selection authorization")
        require(funding_probe.get("shared_client_builder"),
                "Funding probe did not attest the shared production builder")
        require(server.funding_probe == funding_probe, "Server funding probe differs from immutable attestation")
        replacements = {"__SERVER_URL__": server.url,
                        "__SERVER_MANIFEST__": str(server.manifest_path),
                        "__SERVER_GPUS__": ",".join(str(g.index) for g in server.pair),
                        "__CLIENT_GPU__": str(server.pair[0].index)}
        command = [replacements.get(x, x) for x in job["command"]]
        env = self.environment(server.pair, job)
        env.update(JAMES_SERVER_URL=server.url, JAMES_SERVER_MANIFEST=str(server.manifest_path))
        record = self.state["jobs"][job["id"]]
        started_ns = time.time_ns()
        record.update(status="launching", server=server.name, command=command,
                      started_epoch_ns=started_ns, started_utc=utc(started_ns / 1e9),
                      manifest_sha256=file_hash(server.manifest_path))
        if dependency:
            record["initialization_dependency"] = dependency
        # Persist intent before spawning: an uncertain crash can never trigger a rerun.
        self.save()
        server.job = job
        if dependency:
            require(initialization_binding(self.plan, self.state, self.provenance) == dependency,
                    "Initialization dependency changed during world launch journaling")
        require(not self.interrupted, "Queue interrupted before client process creation")
        recheck_pending(self.plan, self.provenance, fresh_jobs=[job])
        require(not self.interrupted, "Queue interrupted before client process creation")
        check_pending_client_deadline(self.plan, job["id"], record)
        server.client = self.spawn(command, env, self.plan.directory / f"job-{job['id']}.log")
        record.update(status="running", identity=server.client.identity)
        if server.reservation_token:
            self.update_reservation(server, "active")
        self.save()
        self.event("job_started", job=job["id"], server=server.name, pid=server.client.process.pid)

    def finish_job(self, server: Server, code: int):
        recheck_pending(self.plan, self.provenance)
        job, client = server.job, server.client
        require(job is not None and client is not None, "Missing active job")
        record = self.state["jobs"][job["id"]]
        record["exit_code"] = code
        observed_exit_ns = check_pending_client_deadline(self.plan, job["id"], record)
        # Background descendants must be gone before artifacts can be stable.
        remaining = stop_owned(client.identity, grace=3)
        require(not remaining, f"Job left uncleaned owned descendants: {remaining}")
        client.log.close()
        require(code == 0, f"Job {job['id']} exited {code}; no retry")
        require(file_hash(server.manifest_path) == record["manifest_sha256"],
                "Server manifest changed during job")
        if "server_profile" in self.plan.spec:
            validate_live_selection(self.plan, self.provenance, server,
                                    read_json(queue_record_bytes(self.plan, server.manifest_path)))
        artifacts = artifact_gate(job, self.plan, started_ns=record["started_epoch_ns"])
        continuation = recheck_pending(self.plan, self.provenance)
        dependency = None
        world_input = None
        if self.plan.initialization:
            require(source_hashes(self.plan) == self.provenance["sources"],
                    "Sources changed during propensity initialization/world execution")
            if is_propensity_world(job):
                prior_validation = validate_completed(self.plan, self.state, self.provenance, require_all=False)
                dependency = initialization_binding(self.plan, self.state, self.provenance)
                require(dependency == record.get("initialization_dependency"),
                        "Initialization dependency changed during world execution")
                world_input = propensity_world_input_binding(self.plan, job, artifacts, dependency)
                require(prior_validation.get("initial_state_hash") in (None, world_input["initial_state_hash"]),
                        "Propensity worlds do not share the same initialized-state hash")
        receipt = {"job": job["id"], "plan_sha256": self.plan.sha256,
                   "server_manifest_sha256": record["manifest_sha256"],
                   "server_manifest": str(server.manifest_path),
                   "structured_output_probe_sha256": digest_bytes(json_bytes(server.probe)),
                   "funding_panel25_probe_sha256": digest_bytes(json_bytes(server.funding_probe)),
                   "funding_probe_fixture_sha256": self.plan.spec["funding_probe_fixture"]["sha256"],
                   "command": record["command"],
                   "command_sha256": digest_bytes(json_bytes(record["command"])),
                   "job_spec_sha256": digest_bytes(json_bytes(job)),
                   "started_epoch_ns": record["started_epoch_ns"],
                   "client_identity": client.identity,
                   "declared_environment": job["env"],
                   "client_cuda_visible_devices": str(server.pair[0].index),
                   "client_gpu_uuid": server.pair[0].uuid,
                   "server_provenance_validation": (
                       "parent validates live process/workers/HTTP and immutable manifest; "
                       "manifest argument passed to child"
                       if job.get("require_manifest_argument", True) else
                       "parent-level attestation only; raw client is not claimed to validate "
                       "the manifest; path supplied as JAMES_SERVER_MANIFEST"),
                   "request_seed_base": 42,
                   "request_seed_policy": SERVER_SPEC["request_seed_policy"],
                   "interpretation_limitation": LIMITATION,
                   "completed_utc": utc(), "artifacts": artifacts}
        if "server_profile" in self.plan.spec:
            receipt.update(selected_runtime_binding(self.plan))
        if self.plan.initialization:
            receipt.update(source_sha256=digest_bytes(json_bytes(self.provenance["sources"])),
                           completed_epoch_ns=time.time_ns(),
                           n_requests=propensity_request_count(self.plan, job, artifacts),
                           scientific_world=is_propensity_world(job))
            if self.plan.spec.get("funding_selection_protocol"):
                receipt["propensity_funding_admission"] = propensity_funding_admission(
                    self.plan, job, artifacts, self.provenance, record["started_epoch_ns"])
            if is_propensity_initializer(job):
                receipt["initial_choices"] = initial_choices_artifact(self.plan, artifacts)
                receipt["state_hashes"] = initialization_state_hashes(bound_json_artifact(
                    self.plan, propensity_manifest_artifact(job), artifacts))
                receipt["initialization_scope"] = (
                    "Treatment-blind common initial-choice cache; no scientific world or outcome.")
            else:
                receipt["cell"] = job["cell"]
                receipt["initialization_dependency"] = dependency
                receipt["world_input_binding"] = world_input
        if continuation is not None:
            receipt["continuation_binding_sha256"] = digest_bytes(json_bytes(continuation))
            receipt["execution_scope"] = PENDING_SCOPE
            receipt["client_exit_observed_epoch_ns"] = observed_exit_ns
            receipt["client_timeout_seconds"] = TIMEOUT_SECONDS
            record["client_exit_observed_epoch_ns"] = observed_exit_ns
        path = self.plan.directory / f"job-{job['id']}.receipt.json"
        atomic_write(path, json_bytes(receipt), exclusive=True, readonly=True)
        record.update(status="complete", receipt_sha256=file_hash(path),
                      artifacts=artifacts, completed_utc=receipt["completed_utc"])
        server.job, server.client = None, None
        if server.reservation_token:
            self.update_reservation(server, "active")
        self.save()
        self.event("job_complete", job=job["id"])

    def close_server(self, server: Server):
        errors = []
        for owned in (server.client, server.owned):
            if owned:
                try:
                    remaining = stop_owned(owned.identity)
                    if remaining:
                        errors.append(f"Owned PIDs still stopping: {remaining}")
                    owned.process.poll()
                except Exception as error:
                    errors.append(str(error))
                finally:
                    owned.log.close()
        if not errors and server.reservation_token:
            try:
                require(not session_members(server.owned.identity)
                        and (server.client is None or not session_members(server.client.identity)),
                        "Owned session remains; reservation must stay quarantined")
                # Mark released only while we still hold both advisory flocks.
                self.update_reservation(server, "released")
            except Exception as error:
                errors.append(str(error))
        for lock in server.locks:
            lock.close()
        self.state["servers"][server.name]["status"] = "stopped" if not errors else "cleanup_incomplete"
        self.state["servers"][server.name]["cleanup_errors"] = errors
        self.servers.remove(server)
        return errors

    def step(self) -> bool:
        require(not self.interrupted, "Queue interrupted")
        require(time.time() < self.state["deadline_epoch"],
                f"{queue_timeout_seconds(self.plan) // 3600}-hour queue deadline exceeded")
        self.client_budget_seconds()
        inventory = query_gpus()
        for server in list(self.servers):
            # Other servers may finish the queue while this model is still loading.
            if server.client is None and not self.pending():
                errors = self.close_server(server)
                require(not errors, "; ".join(errors))
                continue
            require(server.owned.process.poll() is None, f"{server.name} exited unexpectedly")
            self.workers_healthy(server, inventory, require_workers=server.ready)
            if not server.ready:
                if endpoint_ready(server.owned.identity, server.port):
                    self.workers_healthy(server, inventory, require_workers=True)
                    self.attest(server)
                else:
                    require(time.time() - server.started < STARTUP_SECONDS,
                            f"{server.name} startup timeout")
            if server.client:
                code = server.client.process.poll()
                if code is not None:
                    self.finish_job(server, code)
            if server.ready and server.client is None:
                pending = self.runnable()
                if pending:
                    self.start_job(server, pending[0])
                elif not self.pending():
                    errors = self.close_server(server)
                    require(not errors, "; ".join(errors))
        # A starting server already represents a pending job slot.
        slots = len(self.runnable()) - sum(not s.ready for s in self.servers)
        while slots > 0 and len(self.servers) < self.server_limit():
            used = {g.uuid for s in self.servers for g in s.pair}
            available = [g for g in query_gpus() if g.uuid not in used]
            recheck_pending(self.plan, self.provenance)
            reservation_options = {}
            if "server_profile" in self.plan.spec:
                reservation_options["device_count"] = selected_server_spec(self.plan)["tensor_parallel_size"]
            if self.plan.spec.get("continuation"):
                reservation_options["prefer_pending_pair"] = True
            reserved = reserve_pair(available, self.lock_root, query=query_gpus,
                                    **reservation_options)
            if reserved is None:
                break
            self.start_server(*reserved)
            slots -= 1
        if all(record["status"] == "complete" for record in self.state["jobs"].values()):
            require(not self.servers, "Completed queue still has active servers")
            self.state["final_validation"] = validate_completed(
                self.plan, self.state, self.provenance, require_all=True)
            self.state["status"] = "complete"
            self.save()
            return True
        self.state.update(status="running" if self.servers else "waiting_for_idle_pair",
                          last_gpu_snapshot=[dict(index=g.index, used_mib=g.used_mib,
                            utilization=g.utilization, compute_pids=sorted(g.pids)) for g in inventory],
                          next_poll_utc=utc(time.time() + POLL_SECONDS))
        self.state["gpu_reservations"] = [
            {"path": str(path), "record": read_reservation(path)}
            for path in sorted(self.lock_root.glob("GPU-*.reservation.json"))]
        self.save()
        return False

    def run(self) -> int:
        recheck_pending(self.plan, self.provenance, fresh_jobs=self.pending())
        private_directory(self.lock_root)
        for path in (self.plan.private_root / "tmp", self.plan.private_root / "cache/vllm"):
            private_directory(path, self.plan.private_root)
        old_handlers = {}
        def interrupt(signum, frame):
            self.interrupted = True
            if self.probing:
                raise QueueError("Queue interrupted during constrained-output probe")
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        result = 1
        try:
            while not self.step():
                # Positive fixed wait even when no GPU is available. No busy spin.
                time.sleep(POLL_SECONDS)
            self.event("queue_complete")
            result = 0
        except BaseException as error:
            self.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            for record in self.state["jobs"].values():
                if record["status"] in {"launching", "running"}:
                    record["status"] = "failed"
            self.event("queue_failed", error=self.state["error"])
        finally:
            errors = []
            for server in list(self.servers):
                errors.extend(self.close_server(server))
            if errors:
                self.state.update(status="cleanup_incomplete", cleanup_errors=errors)
                result = 1
            self.save()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return result


def queue_record_bytes(plan: Plan, path: Path) -> bytes:
    canonical = output_path(plan, path)
    require(within(canonical, plan.directory), "Queue record escapes its queue directory")
    with open_output_file(plan, canonical) as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and info.st_uid == os.getuid() and info.st_size <= 64 * 1024 * 1024,
                f"Unsafe queue record: {path}")
        raw = stream.read()
        require(stat_binding(info) == stat_binding(os.fstat(stream.fileno())),
                f"Queue record changed while reading: {path}")
    return raw


def validate_completed(plan: Plan, state: dict, provenance: dict, *, require_all: bool) -> dict:
    """One gate for resume and final success; every earlier receipt is checked again."""
    continuation = recheck_pending(plan, provenance)
    validate_selected_runtime(plan, provenance)
    validate_selected_runtime(plan, state["provenance"])
    require(state.get("plan_sha256") == plan.sha256
            and queue_record_bytes(plan, plan.directory / "plan.json") == plan.raw,
            "Immutable plan copy/hash changed")
    require(set(state["jobs"]) == {job["id"] for job in plan.jobs}, "Completed job set changed")
    fixture, binding = load_funding_fixture(plan)
    dependency = None
    if plan.initialization:
        if state["jobs"][plan.initialization["job_id"]]["status"] == "complete":
            dependency = initialization_binding(plan, state, provenance)
        else:
            require(all(state["jobs"][job["id"]]["status"] == "pending"
                        for job in plan.jobs if is_propensity_world(job)),
                    "World activity exists without completed initialization")
    count = 0
    initial_state_hashes = set()
    for job in plan.jobs:
        record = state["jobs"][job["id"]]
        if record["status"] != "complete":
            require(not require_all, f"Job is not complete: {job['id']}")
            continue
        receipt_path = plan.directory / f"job-{job['id']}.receipt.json"
        raw = queue_record_bytes(plan, receipt_path)
        require(digest_bytes(raw) == record["receipt_sha256"], "Completion receipt changed")
        receipt = read_json(raw)
        if continuation is not None:
            require(receipt.get("continuation_binding_sha256") ==
                    digest_bytes(json_bytes(continuation)) and receipt.get("execution_scope") == PENDING_SCOPE,
                    "Completed receipt lost its pending-only parent binding")
            observed_exit_ns = receipt.get("client_exit_observed_epoch_ns")
            require(type(observed_exit_ns) is int
                    and observed_exit_ns == record.get("client_exit_observed_epoch_ns")
                    and type(receipt.get("client_timeout_seconds")) is int
                    and receipt["client_timeout_seconds"] == TIMEOUT_SECONDS,
                    "Completed pending receipt lost its client deadline evidence")
            check_pending_client_deadline(plan, job["id"], record, checked_ns=observed_exit_ns)
        if "server_profile" in plan.spec:
            require(all(receipt.get(key) == value for key, value in selected_runtime_binding(plan).items()),
                    "Completion receipt selected runtime profile changed")
        require(receipt.get("plan_sha256") == plan.sha256 and receipt.get("job") == job["id"]
                and receipt.get("job_spec_sha256") == digest_bytes(json_bytes(job)),
                "Completion receipt belongs to another plan/job specification")
        started_ns = record.get("started_epoch_ns")
        require(type(started_ns) is int and started_ns > 0
                and receipt.get("started_epoch_ns") == started_ns,
                "Completion receipt launch timestamp binding changed")
        require(receipt.get("command") == record["command"]
                and receipt.get("command_sha256") == digest_bytes(json_bytes(record["command"])),
                "Completion receipt command binding changed")
        require(receipt.get("client_identity") == record.get("identity"),
                "Completion receipt client identity binding changed")
        if plan.initialization:
            require(receipt.get("source_sha256") == digest_bytes(json_bytes(provenance["sources"]))
                    and receipt.get("scientific_world") is is_propensity_world(job),
                    "Propensity receipt source/world-role binding changed")
            if is_propensity_world(job):
                require(dependency is not None and receipt.get("initialization_dependency") ==
                        record.get("initialization_dependency") == dependency
                        and receipt.get("cell") == job["cell"]
                        and started_ns >= dependency["initialization_completed_epoch_ns"],
                        "World receipt initialization/cell/chronology binding changed")
        require(artifact_gate(job, plan, started_ns=started_ns) == receipt["artifacts"]
                == record["artifacts"], "Completed artifact hash changed or file binding changed")
        if plan.initialization:
            require(receipt.get("n_requests") == propensity_request_count(plan, job, receipt["artifacts"]),
                    "Propensity request count receipt binding changed")
            if plan.spec.get("funding_selection_protocol"):
                require(receipt.get("propensity_funding_admission") == propensity_funding_admission(
                    plan, job, receipt["artifacts"], provenance, started_ns),
                    "Propensity funding admission changed since its completion receipt")
        if is_propensity_world(job):
            world_input = propensity_world_input_binding(plan, job, receipt["artifacts"], dependency)
            require(receipt.get("world_input_binding") == world_input, "World input receipt binding changed")
            initial_state_hashes.add(world_input["initial_state_hash"])
            require(len(initial_state_hashes) == 1, "Propensity worlds do not share the same initialized-state hash")
        server_name = record["server"]
        server = state["servers"][server_name]
        manifest_path = plan.directory / f"{server_name}.manifest.json"
        require(server.get("manifest") == str(manifest_path)
                and receipt.get("server_manifest") == str(manifest_path),
                "Completion receipt server path binding changed")
        checksum = digest_bytes(queue_record_bytes(plan, manifest_path))
        require(checksum == server["manifest_sha256"] == record["manifest_sha256"]
                == receipt["server_manifest_sha256"], "Completed server manifest changed")
        manifest = read_json(queue_record_bytes(plan, manifest_path))
        validate_selected_runtime(plan, provenance, manifest)
        require(receipt.get("structured_output_probe_sha256") ==
                digest_bytes(json_bytes(manifest.get("structured_output_probe")))
                and receipt.get("funding_panel25_probe_sha256") ==
                digest_bytes(json_bytes(manifest.get("funding_panel25_probe")))
                and receipt.get("funding_probe_fixture_sha256") == binding["sha256"],
                "Completion receipt readiness probe/fixture binding changed")
        endpoint = manifest.get("endpoint", "")
        match = re.fullmatch(r"http://127\.0\.0\.1:(\d+)/v1", endpoint)
        require(match is not None, "Completed server has an invalid readiness endpoint")
        port = int(match[1])
        validate_probe_attestation(manifest.get("structured_output_probe"), server["identity"], port)
        validate_funding_probe_attestation(manifest.get("funding_panel25_probe"),
                                           server["identity"], port, fixture, binding, plan=plan)
        if plan.spec.get("funding_selection_protocol"):
            require(manifest.get("funding_selection_protocol") == SEQUENTIAL_PROTOCOL,
                    "Completed server lacks sequential authorization")
        count += 1
    manifest_count = 0
    for name, record in state["servers"].items():
        if record.get("manifest"):
            path = plan.directory / f"{name}.manifest.json"
            require(record["manifest"] == str(path)
                    and digest_bytes(queue_record_bytes(plan, path)) == record["manifest_sha256"],
                    "Frozen server manifest changed")
            if "server_profile" in plan.spec:
                manifest = read_json(queue_record_bytes(plan, path))
                validate_selected_runtime(plan, provenance, manifest)
                require(all(record.get(key) == value
                            for key, value in selected_runtime_binding(plan).items())
                        and record.get("gpu_ids") == manifest["gpu_ids"]
                        and record.get("gpu_uuids") == manifest["gpu_uuids"]
                        and record.get("command") == manifest["command"],
                        "Frozen server record selected runtime/GPU binding changed")
            manifest_count += 1
    # Check last, after potentially large artifact reads, to catch source edits
    # made while later jobs were running or while final validation was underway.
    sources = source_hashes(plan)
    require(sources == provenance["sources"] == state["provenance"]["sources"],
            "Frozen sources changed before final/resume validation")
    recheck_pending(plan, provenance)
    if dependency:
        require(initialization_binding(plan, state, provenance) == dependency,
                "Initialization dependency changed during final/resume validation")
    result = {"validated_utc": utc(), "completed_jobs": count, "server_manifests": manifest_count,
              "source_sha256": digest_bytes(json_bytes(sources)), "plan_sha256": plan.sha256}
    if plan.initialization:
        result.update(initialization_dependency=dependency,
                      initial_state_hash=next(iter(initial_state_hashes), None),
                      completed_scientific_worlds=sum(state["jobs"][job["id"]]["status"] == "complete"
                                                     for job in plan.jobs if is_propensity_world(job)),
                      initialization_is_scientific_world=False)
    if continuation is not None:
        result.update(execution_scope=PENDING_SCOPE, full_first8_complete=False,
                      all12_complete=False,
                      continuation_binding_sha256=digest_bytes(json_bytes(continuation)))
    return result


def load_state(plan: Plan, provenance: dict, resume: bool) -> dict:
    require(not (resume and plan.spec.get("continuation")), "Pending continuation never permits resume")
    recheck_pending(plan, provenance, fresh_jobs=plan.jobs if not resume else None)
    validate_selected_runtime(plan, provenance)
    path = plan.directory / "status.json"
    if not resume:
        require(not path.exists() and not (plan.directory / "plan.json").exists(),
                "Queue state already exists; use a new queue_id or explicitly --resume")
        for job in plan.jobs:
            ensure_fresh(job, plan)
        state = {"schema_version": 1, "plan_sha256": plan.sha256, "status": "pending",
                  "created_utc": utc(), "deadline_epoch": time.time() + queue_timeout_seconds(plan),
                 "provenance": provenance, "interpretation_limitation": LIMITATION,
                  "jobs": {job["id"]: {"status": "pending"} for job in plan.jobs}, "servers": {}}
        if plan.spec.get("continuation"):
            state["execution_scope"] = PENDING_SCOPE
            state["full_first8_complete"] = state["all12_complete"] = False
        atomic_write(plan.directory / "plan.json", plan.raw, exclusive=True, readonly=True)
        atomic_write(path, json_bytes(state), exclusive=True)
        return state
    require(path.is_file(), "No queue state to resume")
    state = read_json(path.read_bytes())
    require(state.get("plan_sha256") == plan.sha256 and
            (plan.directory / "plan.json").read_bytes() == plan.raw,
            "Resume requires exact immutable plan hash")
    previous = state["provenance"]
    validate_selected_runtime(plan, previous)
    require(all(previous.get(key) == provenance.get(key) for key in
                ("git_head", "branch", "sources", "versions", "runtime_files", "host")),
            "Resume source/runtime/host identity differs")
    require(set(state["jobs"]) == {job["id"] for job in plan.jobs}, "Resume job set changed")
    # Refuse uncertain jobs BEFORE signaling anything. No partial job resumes.
    require(all(x["status"] in {"pending", "complete"} for x in state["jobs"].values()),
            "Started/incomplete job prohibits queue resume; use a new plan/output cohort")
    validate_completed(plan, state, provenance, require_all=state["status"] == "complete")
    for job in plan.jobs:
        if state["jobs"][job["id"]]["status"] != "complete":
            ensure_fresh(job, plan)
    # Do not adopt or automatically clean previous/uncertain sessions.
    for server in state["servers"].values():
        require(isinstance(server.get("identity"), dict),
                "Uncertain server launch prohibits resume; durable GPU intent remains quarantined")
        require(not session_members(server["identity"]),
                "Prior owned server session is still alive; explicit owner cleanup required")
    require(time.time() < state["deadline_epoch"] or state["status"] == "complete",
            "Original 72-hour deadline expired")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(not args.resume or args.run, "--resume requires --run")
        plan = parse_plan(args.plan.read_bytes())
        require(not (args.resume and plan.spec.get("continuation")),
                "Pending continuation never permits resume")
        continuation = validate_pending_continuation(plan, fresh_jobs=plan.jobs)
        if not args.run:
            result = {"valid": True, "plan_sha256": plan.sha256, "jobs": len(plan.jobs),
                      "seed": 42, "server_spec": selected_server_spec(plan),
                      "checks": "Offline schema only; runtime/worktree/NVMe/GPU/artifact and live "
                                "constrained-output probe checks still required.",
                      "interpretation_limitation": LIMITATION}
            if args.dry_run:
                result.update(server_command_template=server_command(plan, 18080),
                              jobs=plan.jobs, queue_directory=str(plan.directory),
                               poll_seconds=POLL_SECONDS, timeout_seconds=queue_timeout_seconds(plan))
            if continuation is not None:
                result.update(execution_scope=PENDING_SCOPE,
                              continuation_binding_sha256=digest_bytes(json_bytes(continuation)))
            print(json.dumps(result, indent=2))
            return 0
        provenance = runtime_guard(plan)
        if continuation is not None:
            provenance["continuation"] = continuation
            recheck_pending(plan, provenance, fresh_jobs=plan.jobs)
        private_directory(plan.directory, plan.private_root)
        queue_lock = Lock(plan.directory / "queue.lock")
        require(queue_lock.acquire(), "Another controller owns this queue")
        try:
            state = load_state(plan, provenance, args.resume)
            if state["status"] == "complete":
                print("Verified existing completed queue; no server or job launched.")
                return 0
            return Queue(plan, provenance, state).run()
        finally:
            queue_lock.close()
    except (QueueError, OSError, subprocess.SubprocessError) as error:
        print(f"Queue refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--tokenizer-sidecar":
        raise SystemExit(tokenizer_sidecar_main(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
