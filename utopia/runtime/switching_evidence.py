"""Source, initialization and funding evidence for switching experiments."""

from __future__ import annotations

from utopia.runtime.historical import (
    identifier_matches,
    source_binding_subset,
    source_binding_value,
    legacy_identifier,
    HISTORICAL_PROFILE_SHA256,
)
from utopia.utils.data_utils import file_sha256
import utopia.funding.feedback as feedback
import utopia.runtime.provenance as provenance
from utopia.utils.paths import project_root
from pathlib import Path
import re
import utopia.agents.switching_policy as policy
from utopia.runtime.server_profiles import TP1_PROFILE_NAME

ROOT = project_root(__file__)

PROTOCOL = "switching-propensity-v3-single-world-sequential"

SEED, FOUNDERS, YEARS = 42, 1200, 10

CELLS = ("LN", "LF", "HN", "HF")

MODEL, MODEL_REVISION = provenance.MODEL, provenance.MODEL_REVISION

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

EMBEDDING_WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)

OUTPUT_REPRESENTATION = "ordered_application_ids_v1"

FUNDING_SELECTION_PROTOCOL = "sequential_remaining_ids_v1"

FUNDING_AMENDMENT = {
    "predecessor_family": "switching_propensity_v2",
    "failure": "LN year1 compact ranking failed full-permutation validation",
    "collateral_stop": "LF after year1",
    "reported_failure_utc": "2026-09-19T11:40:00Z",
    "completed_conditions": 0,
    "decision_basis": "ranking validity failure; no outcome-based condition selection",
    "replacement": "fresh initializer and all four v3 conditions under common sequential funding",
    "artifact_policy": "preserve old cache and all partial/failure evidence; no adoption or resume",
    "elicitation_equivalence_claimed": False,
}

RUNTIME_AMENDMENT = {
    "reason": "User requested productive use of the lone idle GPU7",
    "predecessor_queue": "switching_propensity_seed42_n1200_20260919_v3",
    "predecessor_operations_started": 0,
    "scope": "Fresh initializer and all four conditions use the same explicit TP1 runtime",
    "runtime_profile": TP1_PROFILE_NAME,
    "model_precision_context_seed_and_scientific_policy_changed": False,
    "numerical_or_sampling_equivalence_to_tp2_claimed": False,
}

FUNDING_GATES = {
    "completion_allowed": True,
    "installed": False,
    "output_representation": OUTPUT_REPRESENTATION,
    "failed_batches": 0,
    "audit_failures": 0,
    "invalid_final_panels": 0,
    "processing_failures": 0,
    "unprocessed_panels": 0,
    "imputed_rankings": 0,
    "fallback_rankings": 0,
    "restore_conflicts": [],
}

REQUEST_GATES = {
    "status": "complete",
    "n_clients": 1,
    "guard_failures": 0,
    "pending_requests": 0,
}


class ProtocolFailure(BaseException):
    status = "invalid"

    def __init__(self, reason, **details):
        self.diagnostics = {"status": self.status, "reason": reason, **details}
        super().__init__(reason)


def require(condition, reason, **details):
    if not condition:
        raise ProtocolFailure(reason, **details)


def identifier(cell=None):
    require(cell is None or cell in CELLS, "unknown_cell")
    return f"switching_propensity_v3tp1p1_n1200_y10_{cell or 'initial'}_seed42"


def canonical_state(value):
    """Drop only known wall-clock metadata, never scientific state."""
    if isinstance(value, dict):
        return {
            k: canonical_state(v)
            for k, v in value.items()
            if k not in {"timestamp", "review_time", "submission_time", "decision_time"}
        }
    if isinstance(value, (tuple, list)):
        return [canonical_state(v) for v in value]
    return value


def source_identity():
    paths = [
        ROOT / "utopia/simulation.py",
        ROOT / "utopia/constants.py",
        ROOT / "docs/switching_propensity_protocol.md",
        ROOT / feedback.SEQUENTIAL_PROTOCOL_PATH,
    ]
    paths += sorted((ROOT / "utopia").rglob("*.py"))
    paths += [
        ROOT / name
        for name in (
            "utopia/experiments/switching_propensity.py",
            "utopia/agents/switching_policy.py",
            "utopia/experiments/funding_feedback.py",
            "utopia/funding/validation.py",
            "utopia/funding/compact.py",
            "utopia/funding/sequential.py",
            "utopia/runtime/server_profiles.py",
        )
    ]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def sequential_gates(initialize_only):
    years = 0 if initialize_only else YEARS
    gates = {
        **feedback.sequential_summary_gates(),
        "years_started": years,
        "years_completed": years,
        "agency_registrations": 2 * years,
    }
    if initialize_only:
        gates.update(
            {
                key: 0
                for key in (
                    "registered_panels",
                    "completed_panels",
                    "returned_panels",
                    "processed_panels",
                    "accepted_steps",
                    "sdk_calls",
                    "retries",
                    "n_clients",
                    "empty_agency_registrations",
                )
            }
        )
    return gates


def strict_evidence_json(text, path):
    """Reject ambiguous evidence with this protocol's original error codes."""
    from utopia.utils.data_utils import (
        decode_json,
        DuplicateJSONKey,
        NonfiniteJSONNumber,
    )

    try:
        return decode_json(text, strict=True, finite_floats=True)
    except DuplicateJSONKey as error:
        raise ProtocolFailure(
            "duplicate_evidence_json_key", path=str(path), key=error.key
        ) from error
    except NonfiniteJSONNumber as error:
        raise ProtocolFailure(
            "nonfinite_evidence_json", path=str(path), value=error.value
        ) from error


def read_evidence_json(path):
    return strict_evidence_json(Path(path).read_text(), path)


def validate_runtime_evidence(manifest, *, expected_source):
    """Require the declared TP1 runtime, including the original server attestation."""
    profile_source = "utopia/runtime/server_profiles.py"
    require(
        source_binding_value(expected_source, profile_source)
        == (
            file_sha256(ROOT / profile_source)
            if profile_source in expected_source
            else HISTORICAL_PROFILE_SHA256
        ),
        "runtime_profile_source_binding_mismatch",
    )
    inputs = manifest.get("inputs", {})
    require(
        inputs.get("server_runtime_profile") == TP1_PROFILE_NAME
        and inputs.get("runtime_amendment")
        in (
            RUNTIME_AMENDMENT,
            {
                key: legacy_identifier(value) if isinstance(value, str) else value
                for key, value in RUNTIME_AMENDMENT.items()
            },
        ),
        "runtime_profile_or_amendment_mismatch",
    )
    server = manifest.get("server_provenance", {})
    record = server.get("record", {})
    path = server.get("manifest_path")
    require(
        isinstance(path, str) and Path(path).is_absolute(),
        "runtime_server_manifest_path_missing",
    )
    require(
        read_evidence_json(path) == record
        and file_sha256(path) == server.get("file_sha256"),
        "runtime_server_manifest_binding_mismatch",
    )
    endpoint = manifest.get("args", {}).get("vllm_url")
    require(
        isinstance(endpoint, str) and bool(endpoint), "runtime_client_endpoint_missing"
    )
    try:
        runtime_hash = provenance.validate_server_record(
            record, endpoint, runtime_profile=TP1_PROFILE_NAME
        )
    except (ValueError, TypeError, KeyError) as error:
        raise ProtocolFailure(
            "runtime_attestation_mismatch", detail=str(error)
        ) from error
    require(
        inputs.get("server_runtime_hash") == server.get("runtime_hash") == runtime_hash,
        "runtime_hash_mismatch",
    )
    return {
        "runtime_profile": TP1_PROFILE_NAME,
        "runtime_hash": runtime_hash,
        "server_manifest_sha256": server["file_sha256"],
    }


def validate_funding_evidence(directory, manifest, *, expected_source):
    """Read-only admission API shared with file analysis; no replay or fabricated files."""
    directory = Path(directory)
    initialize_only = manifest.get("kind") == "initialization_only"
    years = 0 if initialize_only else YEARS
    require(
        manifest.get("kind") in ("initialization_only", "condition")
        and manifest.get("status") == "complete"
        and identifier_matches(manifest.get("protocol"), PROTOCOL)
        and manifest.get("seed") == SEED
        and manifest.get("founder_count") == FOUNDERS,
        "funding_manifest_identity_mismatch",
    )
    require(
        manifest.get("inputs", {}).get("source_files_sha256") == expected_source,
        "funding_source_identity_mismatch",
    )
    runtime = validate_runtime_evidence(manifest, expected_source=expected_source)
    binding = source_binding_subset(
        expected_source,
        ("utopia/funding/sequential.py", feedback.SEQUENTIAL_PROTOCOL_PATH),
    )
    require(
        manifest.get("sequential_source_binding") == binding
        and manifest.get("funding_selection_protocol") == FUNDING_SELECTION_PROTOCOL
        and manifest.get("funding_output_representation") == OUTPUT_REPRESENTATION
        and manifest.get("funding_baseline") == feedback.FUNDING_BASELINE,
        "funding_protocol_binding_mismatch",
    )
    compact = read_evidence_json(directory / "funding_validation.summary.json")
    request = read_evidence_json(directory / "llm_request_audit.summary.json")
    sequential_stored = read_evidence_json(directory / feedback.SEQUENTIAL_SUMMARY_FILE)
    for name in (
        "funding_validation.jsonl",
        "llm_request_audit.jsonl",
        feedback.SEQUENTIAL_AUDIT_FILE,
    ):
        with (directory / name).open() as stream:
            for line in stream:
                strict_evidence_json(line, directory / name)
    require(
        compact == manifest.get("funding_validation")
        and all(compact.get(key) == value for key, value in FUNDING_GATES.items()),
        "compact_summary_manifest_mismatch",
    )
    require(
        request == manifest.get("request_audit"), "request_summary_manifest_mismatch"
    )
    provenance.validate_request_audit_summary(request)
    sequential = feedback.validate_sequential_evidence(
        directory,
        compact,
        manifest_summary=manifest.get("funding_sequential"),
        source_binding=binding,
        application_dir=manifest["args"]["output_dir"],
        num_years=years,
        trusted_source_files=expected_source,
    )
    require(
        sequential == sequential_stored == manifest.get("funding_sequential")
        and all(
            sequential.get(key) == value
            for key, value in sequential_gates(initialize_only).items()
        ),
        "sequential_summary_manifest_mismatch",
    )
    native = read_evidence_json(directory / "run_manifest.json")
    if initialize_only:
        require(
            native.get("status") == "initialization_only_complete"
            and native.get("scientific_years_completed") == 0
            and native.get("years_completed") == 0
            and compact.get("final_panels") == 0
            and not list(
                Path(manifest["args"]["output_dir"]).glob(
                    "funding_applications_year_*.jsonl"
                )
            ),
            "initializer_has_scientific_funding_evidence",
        )
    else:
        require(
            native.get("status") == "complete"
            and native.get("years_completed") == YEARS,
            "native_funding_years_incomplete",
        )
    hashes = {
        name: file_sha256(directory / name)
        for name in (feedback.SEQUENTIAL_AUDIT_FILE, feedback.SEQUENTIAL_SUMMARY_FILE)
    }
    ledger = feedback.funding_ledger_evidence(manifest["args"]["output_dir"], years)
    ledger_paths = list(
        Path(manifest["args"]["output_dir"]).glob("funding_applications_year_*.jsonl")
    )
    require(
        {path.name for path in ledger_paths} == set(ledger["present_sha256"]),
        "undeclared_funding_ledger_file",
    )
    for path in ledger_paths:
        with path.open() as stream:
            for line in stream:
                strict_evidence_json(line, path)
    require(
        manifest.get("funding_sequential_audits") == hashes
        and manifest.get("sequential_audit_valid") is True,
        "sequential_bytes_manifest_mismatch",
    )
    require(
        manifest.get("funding_application_evidence") == ledger
        and manifest.get("funding_application_evidence_valid") is True,
        "funding_ledger_bytes_or_absence_mismatch",
    )
    return {
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "runtime_evidence": runtime,
        "scientific_years": years,
        "funding_sequential": sequential,
        "funding_sequential_audits": hashes,
        "funding_application_evidence": ledger,
    }


def validate_initialization(path, expected_inputs):
    """Only a complete, source-matched new initializer can supply the common cache."""
    cache, cache_hash = load_choices(path, expected_inputs)
    manifest_path = Path(path).parent / "initialization_manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = read_evidence_json(manifest_path)
    require(
        manifest.get("status") == "complete"
        and manifest.get("kind") == "initialization_only"
        and manifest.get("scientific_years_completed") == 0
        and manifest.get("inputs") == expected_inputs
        and manifest.get("initial_choices_sha256") == cache_hash
        and Path(manifest["initial_choices_path"]).resolve() == Path(path).resolve()
        and manifest.get("founder_ids") == cache["founder_ids"]
        and manifest.get("choice_count") == FOUNDERS
        and manifest.get("pre_choice_state_hash") == cache["pre_choice_state_hash"]
        and manifest.get("initial_state_hash") == cache["initial_state_hash"],
        "fresh_initializer_manifest_or_cache_mismatch",
    )
    validate_funding_evidence(
        Path(path).parent,
        manifest,
        expected_source=expected_inputs["source_files_sha256"],
    )
    require(
        file_sha256(manifest_path) == manifest_hash and file_sha256(path) == cache_hash,
        "initializer_evidence_changed_during_validation",
    )
    return cache, cache_hash, manifest_path, manifest_hash


def validate_choices(cache, expected_inputs=None):
    require(isinstance(cache, dict), "invalid_initial_choice_cache")
    for key, value in {
        "status": "complete",
        "protocol": PROTOCOL,
        "seed": SEED,
        "founder_count": FOUNDERS,
        "choice_count": FOUNDERS,
    }.items():
        require(
            identifier_matches(cache.get(key), value)
            if key == "protocol"
            else cache.get(key) == value,
            "initial_cache_header_mismatch",
            field=key,
        )
    for key in ("pre_choice_state_hash", "initial_state_hash"):
        require(
            isinstance(cache.get(key), str)
            and re.fullmatch(r"[0-9a-f]{64}", cache[key]),
            "invalid_state_hash",
            field=key,
        )
    records = cache.get("records")
    require(
        isinstance(records, list) and len(records) == FOUNDERS,
        "initial_cache_choice_count_mismatch",
    )
    ids = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), "invalid_initial_record")
        aid, topics, response = (
            record.get(k) for k in ("agent_id", "candidate_topics", "response")
        )
        require(type(aid) is str and bool(aid), "invalid_initial_agent")
        ids.append(aid)
        require(
            isinstance(topics, list)
            and len(topics) > 0
            and len(set(topics)) == len(topics)
            and set(topics) <= set(policy.CANONICAL_TOPICS),
            "invalid_initial_candidates",
            agent_id=aid,
        )
        require(
            isinstance(response, dict)
            and set(response) == {"topic", "detailed_focus", "reason"}
            and all(type(v) is str for v in response.values())
            and response["topic"] in topics,
            "invalid_validated_initial_response",
            agent_id=aid,
        )
        require(
            record.get("fallback_kind") in (None, "parse", "validation"),
            "invalid_initial_fallback",
        )
        require(
            record.get("item_index") == index
            and record.get("seed_ctx") == ["phase1_directions", 1]
            and isinstance(record.get("input_hash"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["input_hash"]),
            "initial_input_provenance_missing",
            agent_id=aid,
        )
        attempts = record.get("request_attempts")
        require(
            isinstance(attempts, list)
            and attempts
            and all(type(a.get("request_seed")) is int for a in attempts),
            "initial_request_seed_evidence_missing",
            agent_id=aid,
        )
    require(
        len(set(ids)) == FOUNDERS and cache.get("founder_ids") == ids,
        "initial_founder_order_mismatch",
    )
    require(
        cache.get("records_sha256") == provenance.digest(records),
        "initial_records_hash_mismatch",
    )
    if expected_inputs is not None:
        require(
            cache.get("inputs") == expected_inputs,
            "initial_cache_input_identity_mismatch",
        )
    return cache


def load_choices(path, expected_inputs=None):
    path = Path(path)
    require(
        path.is_absolute() and path.is_file() and not path.is_symlink(),
        "initial_cache_requires_absolute_regular_file",
    )
    require(path.stat().st_mode & 0o222 == 0, "initial_cache_must_be_immutable")
    before = file_sha256(path)
    cache = validate_choices(read_evidence_json(path), expected_inputs)
    require(file_sha256(path) == before, "initial_cache_changed_while_reading")
    return cache, before
