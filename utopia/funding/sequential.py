"""Sequential, model-selected funding permutations for prospectively new runs.

Install AFTER install_funding_validation(..., compact=True), restore BEFORE it.
Only registered compact phase5_funding_eval batches are intercepted. Every
choice, including a singleton, uses the existing client's _create_completion
and request audit. There are at most THREE SDK calls per step. The SDK's five
internal transport retries remain unchanged and are not extra sidecar events.
No generate_batch nesting, whole-panel retry, filling or deduplication occurs.

The compact application/criteria prefix is immutable. Only its response tail
is replaced. This changes elicitation and is not ranking-distribution parity.
Public helpers are stdlib-only and usable by the controller's technical probe.
The probe must use namespace="technical_probe" in selection_seed_context.

summary()["completion_allowed"] is the live gate. validate_summary defaults
to requiring restoration. restore writes a finalized JSONL event and exclusive
sibling .summary.json. validate_audit independently reconstructs successful
model choices, maps, seeds, returned/processed rankings and terminal counters.
Zero-panel finalized runs are valid. Failures never produce a clean summary.
"""
from __future__ import annotations

from utopia.utils.data_utils import json_sha256

from utopia.runtime.historical import source_binding_value

from utopia.utils.seeding import derive_seed as _derive_seed
from utopia.utils.data_utils import decode_json, DuplicateJSONKey, NonfiniteJSONNumber

from utopia.utils.data_utils import file_sha256

from utopia.funding import compact
from utopia.constants import IMPORTANT_NOTES

from utopia.utils.paths import project_root

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from threading import Event, Lock, RLock
import time


SEQUENTIAL_PROTOCOL = "sequential_remaining_ids_v1"
OUTPUT_REPRESENTATION = "ordered_application_ids_v1"
MAX_STEP_CALLS = 3
DEFAULT_SYSTEM_PROMPT = (
    "This is a simulation of an academic ecosystem, where researchers choose research directions "
    "and submit papers, reviewers conduct peer reviews of papers, and funding agencies allocate fundings."
)
ROOT = project_root(__file__)
_MISSING = object()
_INSTALL_LOCK = RLock()
_INSTALLATIONS = []
_COUNTERS = (
    "registered_panels", "completed_panels", "returned_panels", "processed_panels",
    "accepted_steps", "sdk_calls", "retries", "invalid_attempts", "failed_panels",
    "cancelled_panels", "fatal_errors", "guard_failures", "processing_failures", "audit_failures",
    "n_clients", "audit_records",
    "years_started", "years_completed", "agency_registrations", "empty_agency_registrations",
)


class SequentialFundingError(BaseException):
    """Fatal protocol guard, deliberately outside native Exception retry paths."""
    def __init__(self, reason, **details):
        self.diagnostics = {"reason": reason, "funding_selection_protocol": SEQUENTIAL_PROTOCOL, **details}
        super().__init__(reason)


class _Cancelled(BaseException):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return json_sha256(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)


def _text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()








def _ids(values, *, empty=False):
    if type(values) not in (list, tuple) or (not values and not empty) or len(values) > 25:
        raise ValueError("Expected up to 25 integer application indices")
    if any(type(i) is not int or not 0 <= i < 25 for i in values) or len(set(values)) != len(values):
        raise ValueError("Application indices must be unique actual integers in 0..24")
    return list(values)


def selection_schema(remaining):
    return {"type": "object", "properties": {
        "next_application_id": {"type": "integer", "enum": _ids(remaining)}},
        "required": ["next_application_id"], "additionalProperties": False}


def selection_prompt(compact_original_prompt, selected, remaining):
    selected, remaining = _ids(selected, empty=True), _ids(remaining)
    n = len(selected) + len(remaining)
    if n > 25 or sorted(selected + remaining) != list(range(n)):
        raise ValueError("Prefix and remaining indices must partition the original panel")
    if type(compact_original_prompt) is not str:
        raise ValueError("Original compact prompt must be text")
    prefix, marker, tail = compact_original_prompt.rpartition("\n## Response Format\n")
    expected = compact.compact_prompt('\n## Response Format\n"ranked_applications"', n)
    if not marker or tail != expected.rpartition(marker)[2]:
        raise ValueError("Unknown compact response-format tail")
    return prefix + marker + (
        "Select the most preferred application among the remaining application IDs below, "
        "using the unchanged program criteria and applications above.\n"
        f"Already selected IDs, in order from most preferred: {_json(selected)}\n"
        f"Remaining application IDs: {_json(remaining)}\n"
        f"This choice will receive rank {len(selected) + 1} of {n}.\n"
        'Return only a JSON object with the single key "next_application_id". '
        "Its value must be exactly one integer from the remaining IDs. "
        "Do not return the whole ranking, applicant names, rank numbers, or explanations.\n"
        "Even when only one ID remains, explicitly return that ID in the requested object.\n"
    )


def build_selection_messages(prompt, *, system_prompt=DEFAULT_SYSTEM_PROMPT):
    if type(prompt) is not str or (system_prompt is not None and type(system_prompt) is not str):
        raise ValueError("Message content must be text")
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": f"{IMPORTANT_NOTES}\n\n{prompt}"})
    return messages


def selection_seed_context(year, program_id, panel_index, step, *, namespace="production"):
    if namespace not in ("production", "technical_probe"):
        raise ValueError("Unknown selection seed namespace")
    if (type(year) is not int or year < (1 if namespace == "production" else 0)
            or type(program_id) is not str or not program_id
            or type(panel_index) is not int or panel_index < 0
            or type(step) is not int or not 0 <= step < 25):
        raise ValueError("Invalid selection seed identity")
    return ("phase5_funding_eval", year, SEQUENTIAL_PROTOCOL, namespace, program_id, panel_index, step)




def decode_selection(content, remaining):
    allowed = _ids(remaining)
    if type(content) is not str or not content.strip():
        raise ValueError("Empty selection content")

    try:
        result = decode_json(content, strict=True)
    except DuplicateJSONKey as error:
        raise ValueError("Duplicate JSON key") from error
    except NonfiniteJSONNumber as error:
        raise ValueError("Nonstandard JSON number") from error
    if type(result) is not dict or set(result) != {"next_application_id"}:
        raise ValueError("Selection must contain exactly next_application_id")
    value = result["next_application_id"]
    if type(value) is not int or value not in allowed:
        raise ValueError("Selection is not an actual remaining integer")
    return value


def _response_data(value):
    """Detach the entire returned SDK response, including vendor reasoning fields."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        return {key: _response_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_response_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _response_data(value.model_dump(mode="json"))
    return _response_data(vars(value))  # Explicit SimpleNamespace CPU fixtures.


def _source_hashes():
    paths = [Path(__file__), ROOT / "utopia/funding/compact.py",
             ROOT / "utopia/funding/validation.py", ROOT / "utopia/models/models.py",
             ROOT / "utopia/models/request_audit.py", ROOT / "utopia/agents/funding_agents.py",
             ROOT / "utopia/constants.py", ROOT / "utopia/utils/seeding.py",
             ROOT / "utopia/utils/data_utils.py", ROOT / "utopia/utils/paths.py",
             ROOT / "utopia/runtime/historical.py"]
    protocol = ROOT / "docs/sequential_funding_protocol.md"
    if protocol.is_file():
        paths.append(protocol)
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def _freeze_metadata(meta):
    if (type(meta) is not dict or type(meta.get("program_id")) is not str or not meta["program_id"]
            or type(meta.get("panel_index")) is not int or meta["panel_index"] < 0
            or type(meta.get("apps")) not in (list, tuple) or not 1 <= len(meta["apps"]) <= 25):
        raise ValueError("Invalid registered panel metadata")
    apps = []
    for app in meta["apps"]:
        if type(app) is not dict or type(app.get("applicant_id")) is not str:
            raise ValueError("Invalid registered application metadata")
        # Never serialize a live model/agent or send hidden metadata to the LLM.
        apps.append({"applicant_id": app["applicant_id"]})
    return json.loads(_json({"program_id": meta["program_id"], "panel_index": meta["panel_index"], "apps": apps}))


def validate_summary(summary, *, require_restored=True):
    """Return True or raise; live gate uses require_restored=False."""
    try:
        good = (summary["funding_selection_protocol"] == SEQUENTIAL_PROTOCOL
                and summary["output_representation"] == OUTPUT_REPRESENTATION
                and summary["completion_allowed"] is True
                and summary["sdk_internal_transport_retries"] == 5
                and summary["max_sdk_calls_per_step"] == MAX_STEP_CALLS
                and all(type(summary[k]) is int and summary[k] >= 0 for k in _COUNTERS)
                and not any(summary[k] for k in (
                    "failed_panels", "cancelled_panels", "fatal_errors", "guard_failures",
                    "processing_failures", "audit_failures", "unprocessed_panels"))
                and summary["restore_conflicts"] == []
                and summary["registered_panels"] == summary["completed_panels"]
                == summary["returned_panels"] == summary["processed_panels"]
                and summary["n_clients"] == int(summary["registered_panels"] > 0)
                and summary["accepted_steps"] <= summary["sdk_calls"] <= 3 * summary["accepted_steps"]
                and summary["retries"] == summary["sdk_calls"] - summary["accepted_steps"]
                and summary["invalid_attempts"] == summary["retries"]
                and summary["years_started"] == summary["years_completed"]
                and type(summary["source_sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", summary["source_sha256"]) is not None
                and source_binding_value(summary["source_files_sha256"], "utopia/funding/sequential.py") == summary["source_sha256"])
        if require_restored:
            good &= summary["installed"] is False and summary["status"] == "complete"
        else:
            good &= summary["status"] in ("running", "complete")
    except (KeyError, TypeError, ValueError):
        good = False
    if not good:
        raise SequentialFundingError("unclean_sequential_summary")
    return True


def _panel_report(context):
    return {
        "year": context["year"], "program_id": context["program_id"],
        "panel_index": context["panel_index"], "registration": context["registration"],
        "ranked_application_ids": list(context["selected"]),
        "application_to_applicant": {str(i): aid for i, aid in context["application_map"]},
        "map_sha256": context["map_sha256"], "prompt_sha256": context["prompt_sha256"],
    }


class _Handle:
    def __init__(self, funding_class, model_class, path, validation):
        self.funding_class, self.model_class, self.validation = funding_class, model_class, validation
        self.path, self.summary_path = Path(path), Path(path).with_suffix(".summary.json")
        self.lock, self.batch_lock, self.cancel = RLock(), Lock(), Event()
        self.contexts, self.keys, self.patches, self.saved = {}, set(), {}, {}
        self.client, self.fd = None, None
        self.current_year = None
        self.completed_years = []
        self.stats = {k: 0 for k in _COUNTERS}
        sources = _source_hashes()
        self.stats.update(
            funding_selection_protocol=SEQUENTIAL_PROTOCOL, output_representation=OUTPUT_REPRESENTATION,
            source_sha256=sources["utopia/funding/sequential.py"], source_files_sha256=sources,
            sdk_internal_transport_retries=5,
            max_sdk_calls_per_step=MAX_STEP_CALLS, installed=False, status="running",
            restore_conflicts=[], audit_path=str(self.path), summary_path=str(self.summary_path),
            sdk_retry_scope="One event per SDK call; internal transport retries are not individual events.")

    def _emit(self, event, **data):
        with self.lock:
            record = {"sequence": self.stats["audit_records"] + 1, "event": event, **data}
            try:
                remaining = memoryview((_json(record) + "\n").encode())
                while remaining:
                    written = os.write(self.fd, remaining)
                    if written <= 0:
                        raise OSError("Zero-byte audit write")
                    remaining = remaining[written:]
                os.fsync(self.fd)
            except Exception as error:
                self.stats["audit_failures"] += 1
                self.stats["fatal_errors"] += 1
                self.cancel.set()
                raise SequentialFundingError("sequential_audit_write_failed") from error
            self.stats["audit_records"] += 1
            return record

    def _fail(self, reason, **details):
        with self.lock:
            self.stats["fatal_errors"] += 1
            self.stats["guard_failures"] += 1
            self.cancel.set()
            self._emit("fatal", reason=reason, **details)
        raise SequentialFundingError(reason, **details)

    def _active(self):
        if self.cancel.is_set() or not self.stats["installed"] or not self.validation.summary()["installed"]:
            self._fail("sequential_or_compact_guard_not_installed")

    def _unchanged(self, context):
        try:
            unchanged = _sha(_freeze_metadata(context["metadata"])) == context["metadata_sha256"]
        except Exception:
            unchanged = False
        if not unchanged:
            self._fail("registered_metadata_mutated", registration=context["registration"])

    def begin_year(self, year, agency_ids):
        """Driver brackets the actual native phase5, including empty years."""
        with self.lock:
            self._active()
            if (type(year) is not int or year < 1 or self.current_year is not None
                    or year in self.completed_years or not isinstance(agency_ids, (list, tuple))
                    or not agency_ids or any(type(a) is not str or not a for a in agency_ids)
                    or len(set(agency_ids)) != len(agency_ids)):
                self._fail("invalid_funding_year_boundary")
            self.current_year = {"year": year, "expected_agency_ids": list(agency_ids), "agencies": []}
            self._emit("year_started", year=year, agency_ids=list(agency_ids))
            self.stats["years_started"] += 1

    def end_year(self, year):
        with self.lock:
            self._active()
            coverage = self.current_year
            if (coverage is None or coverage["year"] != year or self.contexts
                    or {a["agency_id"] for a in coverage["agencies"]} != set(coverage["expected_agency_ids"])):
                self._fail("incomplete_native_funding_year_coverage")
            self._emit("year_completed", coverage=coverage)
            self.completed_years.append(year)
            self.current_year = None
            self.stats["years_completed"] += 1

    def register(self, value, panel_seed, agency, applications):
        with self.lock:
            self._active()
            try:
                texts, response_format, metadata = value
                coverage = self.current_year
                if (coverage is None or agency.id not in coverage["expected_agency_ids"]
                        or agency.id in {a["agency_id"] for a in coverage["agencies"]}
                        or panel_seed != _derive_seed(42, "funding_panels", coverage["year"])):
                    raise ValueError("Missing, duplicate or mismatched native year/agency coverage")
                if (type(texts) is not list or type(metadata) is not list or len(texts) != len(metadata)
                        or response_format != compact.compact_response_format()
                        or type(panel_seed) is not int):
                    raise ValueError("Registered prompt format/count/seed mismatch")
                prepared = []
                for text, meta in zip(texts, metadata):
                    frozen = _freeze_metadata(meta)
                    remaining = list(range(len(frozen["apps"])))
                    selection_prompt(text, [], remaining)  # exact known compact tail
                    key = (panel_seed, frozen["program_id"], frozen["panel_index"])
                    if key in self.keys or key in [p["key"] for p in prepared]:
                        raise ValueError("Duplicate program/panel/seed registration")
                    prepared.append({
                        "key": key, "metadata": meta, "metadata_sha256": _sha(frozen),
                        "prompt": text, "prompt_sha256": _text_sha(text),
                        "application_map": [[i, a["applicant_id"]] for i, a in enumerate(frozen["apps"])],
                        "program_id": frozen["program_id"], "panel_index": frozen["panel_index"],
                        "panel_seed": panel_seed, "selected": [], "state": "registered",
                    })
                application_count = sum(
                    1 for application in applications for program, app in application.items()
                    if app["submit"] and program in agency.funding_programs)
                if application_count != sum(len(c["application_map"]) for c in prepared):
                    raise ValueError("Native submitted applications differ from returned panel coverage")
            except Exception as error:
                self._fail("invalid_compact_prompt_registration", error_type=type(error).__name__)
            for context in prepared:
                context["registration"] = self.stats["registered_panels"]
                context["map_sha256"] = _sha(context["application_map"])
                self._emit("panel_registered", **{k: context[k] for k in (
                    "registration", "metadata_sha256", "prompt", "prompt_sha256", "application_map",
                    "map_sha256", "program_id", "panel_index", "panel_seed")})
                self.contexts[context["registration"]] = context
                self.keys.add(context["key"])
                self.stats["registered_panels"] += 1
            observed = {"year": coverage["year"], "agency_id": agency.id, "panel_seed": panel_seed,
                        "native_submitted_applications": application_count,
                        "registrations": [c["registration"] for c in prepared],
                        "panel_count": len(prepared)}
            self._emit("agency_registration", **observed)
            coverage["agencies"].append(observed)
            self.stats["agency_registrations"] += 1
            self.stats["empty_agency_registrations"] += int(not prepared)
        return value

    def _step(self, model, context, year, options):
        selected = context["selected"]
        remaining = [i for i in range(len(context["application_map"])) if i not in selected]
        step = len(selected)
        prompt = selection_prompt(context["prompt"], selected, remaining)
        messages = build_selection_messages(prompt, system_prompt=options["system_prompt"])
        ctx = selection_seed_context(year, context["program_id"], context["panel_index"], step)
        response_format = {"type": "json_schema", "json_object": {
            "name": "NextFundingApplication", "schema": selection_schema(remaining)}}
        for attempt in range(MAX_STEP_CALLS):
            if self.cancel.is_set():
                raise _Cancelled()
            extra = model._build_extra_body(response_format)
            seed = model._request_seed(ctx, 0, attempt)
            if (seed != _derive_seed(model.run_seed, *ctx, 0, attempt)
                    or extra.get("structured_outputs", {}).get("json") != selection_schema(remaining)
                    or extra.get("chat_template_kwargs") != {"enable_thinking": True}):
                self._fail("selection_transport_or_seed_mismatch")
            extra["seed"] = seed
            details = {
                "registration": context["registration"], "year": year, "step": step, "attempt": attempt,
                "selected": list(selected), "remaining": remaining, "seed_ctx": list(ctx), "seed": seed,
                "messages_sha256": _sha(messages), "schema": selection_schema(remaining),
                "temperature": options["temperature"], "max_tokens": options["max_tokens"],
                "run_seed": model.run_seed,
                "attempt_id": f"{context['registration']}:{step}:{attempt}",
            }
            with self.lock:
                if self.cancel.is_set():
                    raise _Cancelled()
                self._emit("sdk_call", **details)
                self.stats["sdk_calls"] += 1
                self.stats["retries"] += int(attempt > 0)
            started = time.monotonic()
            content, reason, usage, error_type, chosen, n_choices = None, None, None, None, None, None
            try:
                response = model._create_completion(
                    seed_ctx=ctx, item_index=0, attempt=attempt, model=model.model_name,
                    messages=messages, temperature=options["temperature"], max_tokens=options["max_tokens"],
                    extra_body=extra)
                raw_response = _response_data(response)
                # Persist complete response/reasoning BEFORE finish or semantic
                # validation. Prefix/map references reconstruct all request bytes.
                self._emit("sdk_response", **details, raw_response=raw_response,
                           response_id=raw_response.get("id"),
                           started_monotonic=started, duration_seconds=time.monotonic() - started)
                choices = raw_response["choices"]
                n_choices = len(choices)
                choice = choices[0]
                content, reason = choice["message"].get("content"), choice.get("finish_reason")
                tokens = raw_response.get("usage") or {}
                usage = {k: tokens.get(k) for k in ("prompt_tokens", "completion_tokens")}
                if (n_choices != 1 or reason != "stop"
                        or any(type(v) is not int or v < 0 for v in usage.values())
                        or usage["completion_tokens"] > options["max_tokens"]):
                    raise ValueError("Invalid choice/finish/token usage")
                chosen = decode_selection(content, remaining)
            except Exception as error:
                error_type = type(error).__name__
            except BaseException as error:
                # The existing request guard may reject a returned response
                # before _create_completion returns. The reviewed guard exposes
                # that response/ticket solely for this terminal audit record.
                with self.lock:
                    self.cancel.set()
                    self.stats["fatal_errors"] += 1
                    self.stats["guard_failures"] += 1
                    received = getattr(error, "received_response", None)
                    try:
                        if not self.stats["audit_failures"]:
                            self._emit(
                                "terminal_guard", **details, error_type=type(error).__name__,
                                raw_response=_response_data(received) if received is not None else None,
                                request_ticket=getattr(error, "request_ticket", None),
                                started_monotonic=started, duration_seconds=time.monotonic() - started)
                    finally:
                        raise error  # Never downgrade a fatal guard into an SDK retry.
            with self.lock:
                self._emit("sdk_result", **details, raw_content=content, finish_reason=reason,
                           token_usage=usage, n_choices=n_choices,
                           accepted=chosen is not None, error_type=error_type,
                           duration_seconds=time.monotonic() - started)
                if chosen is None:
                    self.stats["invalid_attempts"] += 1
                stats = model.call_stats
                stats["n_prompts"] += int(attempt == 0)
                stats["n_retries"] += int(attempt > 0)
                stats["n_first_attempt_success"] += int(chosen is not None and attempt == 0)
                stats["n_failures"] += int(chosen is None and attempt == MAX_STEP_CALLS - 1)
                if usage is not None and all(type(v) is int and v >= 0 for v in usage.values()):
                    stats["prompt_tokens"] += usage["prompt_tokens"]
                    stats["completion_tokens"] += usage["completion_tokens"]
            if chosen is not None:
                self._unchanged(context)
                with self.lock:
                    self._emit("selection_accepted", registration=context["registration"],
                               step=step, attempt=attempt, next_application_id=chosen,
                               selected=list(selected) + [chosen])
                    selected.append(chosen)
                    self.stats["accepted_steps"] += 1
                return messages + [{"role": "assistant", "content": [{"type": "text", "text": content}]}]
        self._fail("selection_step_exhausted", registration=context["registration"],
                   step=step, selected=list(selected), remaining=remaining)

    def _panel(self, model, context, year, options):
        try:
            context["state"] = "running"
            history = []
            while len(context["selected"]) < len(context["application_map"]):
                history = self._step(model, context, year, options)
            self._unchanged(context)
            with self.lock:
                self._emit("panel_completed", registration=context["registration"],
                           ranked_application_ids=context["selected"])
                context["state"] = "completed"
                self.stats["completed_panels"] += 1
            return ({"ranked_application_ids": list(context["selected"])}, history)
        except _Cancelled:
            with self.lock:
                context["state"] = "cancelled"
                self.stats["cancelled_panels"] += 1
                self._emit("panel_cancelled", registration=context["registration"], selected=context["selected"])
            raise
        except BaseException as error:
            with self.lock:
                self.cancel.set()
                context["state"] = "failed"
                self.stats["failed_panels"] += 1
                self._emit("panel_failed", registration=context["registration"],
                           selected=context["selected"], error_type=type(error).__name__)
            raise

    def batch(self, model, options):
        if not self.batch_lock.acquire(blocking=False):
            self._fail("concurrent_funding_batches")
        started = time.monotonic()
        try:
            with self.lock:
                self._active()
                ctx = options["seed_ctx"]
                if (type(ctx) not in (tuple, list) or len(ctx) != 3 or ctx[0] != "phase5_funding_eval"
                        or type(ctx[1]) is not int or ctx[1] < 1 or type(ctx[2]) is not int or ctx[2] != 0):
                    self._fail("native_whole_panel_retry_forbidden")
                if self.current_year is None or self.current_year["year"] != ctx[1]:
                    self._fail("funding_batch_outside_declared_year")
                if (type(model.run_seed) is not int or model.run_seed != 42
                        or model.enable_thinking is not True or options["temperature"] != .7
                        or options["max_tokens"] != 8192
                        or type(model.max_concurrent_requests) is not int
                        or not 1 <= model.max_concurrent_requests <= 64
                        or getattr(model, "request_audit", None) is None
                        or model.request_audit.summary().get("n_clients") != 1
                        or getattr(model.client, "max_retries", None) != 5):
                    self._fail("production_client_configuration_mismatch")
                if self.client is None:
                    self.client = model
                    self.stats["n_clients"] = 1
                    self._emit("client_bound", run_seed=model.run_seed, max_concurrent_requests=model.max_concurrent_requests,
                               model=model.model_name, request_audit_path=str(model.request_audit.path))
                elif self.client is not model:
                    self._fail("multiple_sequential_clients")
                pending = [c for c in self.contexts.values() if c["state"] == "registered"]
                contexts = []
                for prompt in options["prompts"]:
                    matches = [c for c in pending if c["prompt"] == prompt and c not in contexts]
                    if len(matches) != 1:
                        self._fail("unregistered_or_ambiguous_compact_prompt")
                    context = matches[0]
                    self._unchanged(context)
                    if context["panel_seed"] != _derive_seed(model.run_seed, "funding_panels", ctx[1]):
                        self._fail("registered_panel_seed_year_mismatch")
                    contexts.append(context)
                if len(contexts) != len(pending) or not contexts:
                    self._fail("funding_batch_omits_registered_panels")
                self._emit("batch_started", registrations=[c["registration"] for c in contexts],
                           year=ctx[1], system_prompt=options["system_prompt"],
                           max_concurrent_requests=model.max_concurrent_requests)
                for context in contexts:
                    context["year"] = ctx[1]
            results, first_error = [None] * len(contexts), None
            with ThreadPoolExecutor(max_workers=min(model.max_concurrent_requests, len(contexts))) as executor:
                futures = {executor.submit(self._panel, model, c, ctx[1], options): i for i, c in enumerate(contexts)}
                for future in as_completed(futures):
                    try:
                        results[futures[future]] = future.result()
                    except BaseException as error:
                        self.cancel.set()
                        if first_error is None or isinstance(first_error, _Cancelled):
                            first_error = error
            if first_error is not None:
                if isinstance(first_error, (Exception, _Cancelled)):
                    self._fail("sequential_batch_aborted", error_type=type(first_error).__name__)
                raise first_error
            with self.lock:
                self._emit("batch_returned", registrations=[c["registration"] for c in contexts],
                           rankings=[pair[0]["ranked_application_ids"] for pair in results])
                self.stats["returned_panels"] += len(contexts)
                for context in contexts:
                    context["state"] = "returned"
            return results
        finally:
            with self.lock:
                if self.client is model:
                    model.call_stats["elapsed_seconds"] += time.monotonic() - started
            self.batch_lock.release()

    def process(self, original, *args, **kwargs):
        with self.lock:
            self._active()
            bound = inspect.signature(original).bind(*args, **kwargs)
            results, metadata = bound.arguments["batch_results"], bound.arguments["metadata_list"]
            contexts = []
            if len(results) != len(metadata):
                self._fail("sequential_processing_length_mismatch")
            for pair, meta in zip(results, metadata):
                matches = [c for c in self.contexts.values() if c["metadata"] is meta and c["state"] == "returned"]
                if len(matches) != 1 or matches[0] in contexts:
                    self._fail("processing_unregistered_or_reused_panel")
                context = matches[0]
                self._unchanged(context)
                if pair[0] != {"ranked_application_ids": context["selected"]}:
                    self._fail("processed_ranking_differs_from_model_choices")
                contexts.append(context)
            if len(contexts) != sum(c["state"] == "returned" for c in self.contexts.values()):
                self._fail("processing_omits_returned_panels")
            self._emit("processing_started", registrations=[c["registration"] for c in contexts])
            try:
                result = original(*args, **kwargs)  # existing whole-batch compact/permutation validation
            except BaseException:
                self.stats["processing_failures"] += 1
                self._emit("processing_failed", registrations=[c["registration"] for c in contexts])
                self.cancel.set()
                raise
            self._emit("batch_processed", registrations=[c["registration"] for c in contexts],
                       rankings=[list(c["selected"]) for c in contexts],
                       panels=[_panel_report(c) for c in contexts])
            self.stats["processed_panels"] += len(contexts)
            for context in contexts:
                del self.contexts[context["registration"]]
            return result

    def summary(self):
        with self.lock:
            result = deepcopy(self.stats)
            result["unprocessed_panels"] = result["registered_panels"] - result["processed_panels"]
            result["completion_allowed"] = (
                result["registered_panels"] == result["completed_panels"]
                == result["returned_panels"] == result["processed_panels"]
                and not any(result[k] for k in (
                    "failed_panels", "cancelled_panels", "fatal_errors", "guard_failures",
                    "processing_failures", "audit_failures", "unprocessed_panels"))
                and not result["restore_conflicts"])
            result["completion_allowed"] &= self.current_year is None
            return result

    def restore(self):
        with _INSTALL_LOCK, self.lock:
            if not self.stats["installed"]:
                return
            conflicts = []
            for (cls, name), descriptor in self.patches.items():
                if vars(cls).get(name, _MISSING) is not descriptor:
                    conflicts.append(cls.__name__ + "." + name)
                    continue
                saved = self.saved[(cls, name)]
                if saved is _MISSING:
                    delattr(cls, name)
                else:
                    setattr(cls, name, saved)
            self.stats["installed"] = False
            self.stats["restore_conflicts"] = conflicts
            if self in _INSTALLATIONS:
                _INSTALLATIONS.remove(self)
            self.stats["status"] = "complete" if self.summary()["completion_allowed"] else "failed"
            try:
                summary = self.summary()
                summary["audit_records"] += 1
                self._emit("finalized", summary=summary)
                with self.summary_path.open("x") as stream:
                    stream.write(_json(self.summary()) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as error:
                self.stats["audit_failures"] += 1
                self.stats["status"] = "failed"
                raise SequentialFundingError("sequential_summary_write_failed") from error
            finally:
                if self.fd is not None:
                    os.close(self.fd)
                    self.fd = None
            if conflicts:
                raise SequentialFundingError("sequential_restore_conflict", conflicts=conflicts)


def install_sequential_funding(funding_class, model_class, audit_path):
    """Install after the exact compact gate, return the reversible audit handle."""
    import utopia.funding.validation as validation_module
    with _INSTALL_LOCK:
        if not isinstance(funding_class, type) or not isinstance(model_class, type):
            raise TypeError("Expected funding/model classes")
        if any(issubclass(funding_class, h.funding_class) or issubclass(h.funding_class, funding_class)
               or issubclass(model_class, h.model_class) or issubclass(h.model_class, model_class)
               for h in _INSTALLATIONS):
            raise SequentialFundingError("overlapping_sequential_installation")
        validation = validation_module._INSTALLATIONS.get(funding_class)
        if (validation is None or not validation.summary()["installed"]
                or validation.summary()["output_representation"] != OUTPUT_REPRESENTATION):
            raise SequentialFundingError("compact_validation_must_be_installed_first")
        original_prompts = funding_class.get_funding_evaluation_prompts
        original_generate = model_class.generate_batch
        original_process = funding_class.process_funding_evaluation_results
        if not isinstance(inspect.getattr_static(funding_class, "process_funding_evaluation_results"), staticmethod):
            raise TypeError("Funding processor must be a staticmethod")
        prompt_signature, generate_signature = inspect.signature(original_prompts), inspect.signature(original_generate)
        if generate_signature.parameters["system_prompt"].default != DEFAULT_SYSTEM_PROMPT:
            raise SequentialFundingError("native_system_prompt_default_drift")
        handle = _Handle(funding_class, model_class, audit_path, validation)
        if (not handle.path.is_absolute() or not handle.path.parent.is_dir()
                or handle.summary_path.exists() or handle.summary_path.is_symlink()):
            raise SequentialFundingError("fresh_absolute_sequential_audit_required")

        @wraps(original_prompts)
        def prompts(self, *args, **kwargs):
            bound = prompt_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            return handle.register(original_prompts(self, *args, **kwargs), bound.arguments.get("panel_seed"),
                                   self, bound.arguments["applications"])

        @wraps(original_generate)
        def generate(self, *args, **kwargs):
            bound = generate_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            options = bound.arguments
            ctx = options.get("seed_ctx")
            funding_phase = type(ctx) in (tuple, list) and ctx and ctx[0] == "phase5_funding_eval"
            if not funding_phase:
                return original_generate(self, *args, **kwargs)
            if len(ctx) != 3 or type(ctx[2]) is not int or ctx[2] != 0:
                handle._fail("native_whole_panel_retry_forbidden")
            if options["response_format"] != compact.compact_response_format():
                if any(p == c["prompt"] for p in options["prompts"] for c in handle.contexts.values()):
                    handle._fail("registered_funding_format_changed")
                return original_generate(self, *args, **kwargs)
            return handle.batch(self, options)

        @wraps(original_process)
        def process(*args, **kwargs):
            return handle.process(original_process, *args, **kwargs)

        handle.patches = {(funding_class, "get_funding_evaluation_prompts"): prompts,
                          (funding_class, "process_funding_evaluation_results"): staticmethod(process),
                          (model_class, "generate_batch"): generate}
        handle.saved = {key: vars(key[0]).get(key[1], _MISSING) for key in handle.patches}
        try:
            handle.fd = os.open(handle.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
                                | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except OSError as error:
            raise SequentialFundingError("fresh_sequential_audit_required") from error
        try:
            handle._emit("initialized", identity=handle.summary())
            for (cls, name), descriptor in handle.patches.items():
                setattr(cls, name, descriptor)
            handle.stats["installed"] = True
            _INSTALLATIONS.append(handle)
        except BaseException:
            for (cls, name), descriptor in handle.patches.items():
                if vars(cls).get(name, _MISSING) is descriptor:
                    saved = handle.saved[(cls, name)]
                    delattr(cls, name) if saved is _MISSING else setattr(cls, name, saved)
            if handle.fd is not None:
                os.close(handle.fd)
                handle.fd = None
            raise
        return handle


def validate_audit(path, summary, *, request_audit_path, expected_sources=None):
    """Verify finalized raw trace and return normalized processed-panel mappings.

    This verifies observed model order, not winner quotas. The caller must
    crosscheck processed_panels against the native application-level award log
    and the independent compact validator's final_panels count.
    """
    validate_summary(summary)

    def check(value, reason):
        if not value:
            raise SequentialFundingError("invalid_sequential_audit", detail=reason)

    path = Path(path)
    check(path.is_absolute() and str(path) == summary["audit_path"], "audit path mismatch")
    request_audit_path = Path(request_audit_path)
    check(request_audit_path.is_absolute() and request_audit_path.parent == path.parent,
          "explicit request audit must belong to the same run output directory")
    check(summary["source_files_sha256"] == (_source_hashes() if expected_sources is None else expected_sources),
          "frozen source files differ")
    try:
        stored_summary = json.loads(Path(summary["summary_path"]).read_text())
        check(stored_summary == summary, "sibling summary differs")
        counters = {k: 0 for k in _COUNTERS}
        contexts, seen_keys, processed, years, sdk_join = {}, set(), [], [], {}
        current_year = None
        batch, processing, client, initial, terminal = None, None, None, None, False
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for raw in stream:
                checksum.update(raw)
                row = json.loads(raw)
                counters["audit_records"] += 1
                check(row["sequence"] == counters["audit_records"] and not terminal, "sequence or trailing rows")
                event = row["event"]
                if event == "initialized":
                    check(initial is None and row["sequence"] == 1, "initialization must occur first once")
                    initial = row["identity"]
                    check(initial["source_sha256"] == summary["source_sha256"]
                          and initial["source_files_sha256"] == summary["source_files_sha256"]
                          and all(initial[k] == 0 for k in _COUNTERS)
                          and initial["installed"] is False and initial["status"] == "running",
                          "initial identity/counters")
                elif event == "panel_registered":
                    check(initial is not None and current_year is not None, "missing initialization/year")
                    rid = row["registration"]
                    check(type(rid) is int and rid == counters["registered_panels"], "registration order")
                    mapping = row["application_map"]
                    check(type(mapping) is list and 1 <= len(mapping) <= 25
                          and all(type(pair) is list and len(pair) == 2 and type(pair[0]) is int
                                  and pair[0] == i and type(pair[1]) is str
                                  for i, pair in enumerate(mapping)), "application map")
                    selection_seed_context(1, row["program_id"], row["panel_index"], 0)
                    key = (row["panel_seed"], row["program_id"], row["panel_index"])
                    check(type(row["panel_seed"]) is int and key not in seen_keys, "duplicate panel key")
                    check(row["map_sha256"] == _sha(mapping)
                          and row["prompt_sha256"] == _text_sha(row["prompt"]), "registered hashes")
                    selection_prompt(row["prompt"], [], list(range(len(mapping))))
                    check(row["panel_seed"] == _derive_seed(42, "funding_panels", current_year["year"]),
                          "registered year seed mismatch")
                    # Private metadata consists ONLY of program/panel and the ordered map.
                    frozen = {"program_id": row["program_id"], "panel_index": row["panel_index"],
                              "apps": [{"applicant_id": aid} for _, aid in mapping]}
                    check(row["metadata_sha256"] == _sha(frozen), "metadata map hash")
                    contexts[rid] = {**row, "selected": [], "pending": None, "next_attempt": 0, "state": "registered"}
                    seen_keys.add(key)
                    counters["registered_panels"] += 1
                elif event == "year_started":
                    check(initial is not None and current_year is None and not contexts
                          and type(row["year"]) is int and row["year"] >= 1
                          and row["year"] not in [y["year"] for y in years]
                          and type(row["agency_ids"]) is list and bool(row["agency_ids"])
                          and all(type(a) is str and a for a in row["agency_ids"])
                          and len(set(row["agency_ids"])) == len(row["agency_ids"]), "funding year boundary")
                    current_year = {"year": row["year"], "expected_agency_ids": row["agency_ids"], "agencies": []}
                    counters["years_started"] += 1
                elif event == "agency_registration":
                    check(current_year is not None and row["year"] == current_year["year"]
                          and row["agency_id"] in current_year["expected_agency_ids"]
                          and row["agency_id"] not in [a["agency_id"] for a in current_year["agencies"]],
                          "agency coverage identity")
                    ids = row["registrations"]
                    covered = {r for agency in current_year["agencies"] for r in agency["registrations"]}
                    check(type(ids) is list and len(ids) == len(set(ids)) and not set(ids) & covered
                          and all(r in contexts for r in ids)
                          and row["panel_count"] == len(ids)
                          and row["native_submitted_applications"] == sum(
                              len(contexts[r]["application_map"]) for r in ids)
                          and row["panel_seed"] == _derive_seed(42, "funding_panels", row["year"]),
                          "empty/nonempty agency counts")
                    current_year["agencies"].append({k: row[k] for k in (
                        "year", "agency_id", "panel_seed", "native_submitted_applications", "registrations", "panel_count")})
                    counters["agency_registrations"] += 1
                    counters["empty_agency_registrations"] += int(not ids)
                elif event == "year_completed":
                    check(current_year is not None and row["coverage"] == current_year
                          and not contexts and batch is None and processing is None
                          and {a["agency_id"] for a in current_year["agencies"]} == set(
                              current_year["expected_agency_ids"]), "incomplete year coverage")
                    years.append(current_year)
                    current_year = None
                    counters["years_completed"] += 1
                elif event == "client_bound":
                    check(client is None and row["run_seed"] == 42
                          and type(row["max_concurrent_requests"]) is int
                          and 1 <= row["max_concurrent_requests"] <= 64, "bound client")
                    client = row
                    counters["n_clients"] += 1
                elif event == "batch_started":
                    ids = row["registrations"]
                    check(client is not None and batch is None and processing is None
                          and type(ids) is list and ids and len(ids) == len(set(ids))
                          and set(ids) == set(contexts), "batch omits/duplicates panels")
                    check(current_year is not None and row["year"] == current_year["year"]
                          and {r for a in current_year["agencies"] for r in a["registrations"]} == set(ids),
                          "batch lacks agency coverage")
                    check(row["max_concurrent_requests"] == client["max_concurrent_requests"], "concurrency")
                    for rid in ids:
                        c = contexts[rid]
                        check(c["state"] == "registered"
                              and c["panel_seed"] == _derive_seed(42, "funding_panels", row["year"]),
                              "panel seed/year")
                        c["year"], c["system_prompt"], c["state"] = row["year"], row["system_prompt"], "running"
                    batch = {"ids": ids, "returned": False}
                elif event == "sdk_call":
                    c = contexts[row["registration"]]
                    remaining = [i for i in range(len(c["application_map"])) if i not in c["selected"]]
                    check(batch is not None and not batch["returned"] and c["state"] == "running"
                          and c["pending"] is None and row["selected"] == c["selected"]
                          and row["remaining"] == remaining and remaining
                          and row["step"] == len(c["selected"]) and type(row["attempt"]) is int
                          and row["attempt"] == c["next_attempt"] and row["attempt"] < MAX_STEP_CALLS,
                          "step/prefix/remaining/attempt")
                    ctx = selection_seed_context(c["year"], c["program_id"], c["panel_index"], row["step"])
                    messages = build_selection_messages(
                        selection_prompt(c["prompt"], c["selected"], remaining), system_prompt=c["system_prompt"])
                    check(row["year"] == c["year"] and row["seed_ctx"] == list(ctx)
                          and row["run_seed"] == 42 and row["seed"] == _derive_seed(42, *ctx, 0, row["attempt"])
                          and row["messages_sha256"] == _sha(messages)
                          and row["schema"] == selection_schema(remaining)
                          and row["temperature"] == .7 and row["max_tokens"] == 8192, "request descriptor")
                    check(row["attempt_id"] == f"{row['registration']}:{row['step']}:{row['attempt']}",
                          "attempt ID")
                    c["pending"] = row
                    c["response"] = None
                    key = (tuple(ctx), 0, row["attempt"])
                    check(key not in sdk_join, "duplicate SDK call identity")
                    payload = {"model": client["model"], "messages": messages, "temperature": .7,
                               "max_tokens": 8192, "extra_body": {
                                   "structured_outputs": {"json": selection_schema(remaining)},
                                   "chat_template_kwargs": {"enable_thinking": True}, "seed": row["seed"]}}
                    sdk_join[key] = {
                        "seed": row["seed"], "payload_sha256": hashlib.sha256(json.dumps(
                            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                        ).encode()).hexdigest(), "usage": None, "finish_reason": None, "response": False}
                    counters["sdk_calls"] += 1
                    counters["retries"] += int(row["attempt"] > 0)
                elif event == "sdk_response":
                    c = contexts[row["registration"]]
                    pending = c["pending"]
                    check(pending is not None and pending["event"] == "sdk_call" and c["response"] is None
                          and all(row.get(k) == v for k, v in pending.items() if k not in ("event", "sequence")),
                          "raw response not paired with exact SDK call")
                    check(type(row["raw_response"]) is dict
                          and row["response_id"] == row["raw_response"].get("id")
                          and type(row["duration_seconds"]) in (float, int) and row["duration_seconds"] >= 0
                          and type(row["started_monotonic"]) in (float, int), "raw response timing/identity")
                    c["response"] = row["raw_response"]
                elif event == "sdk_result":
                    c = contexts[row["registration"]]
                    pending = c["pending"]
                    check(pending is not None and pending["event"] == "sdk_call"
                          and all(row.get(k) == v for k, v in pending.items() if k not in ("event", "sequence")),
                          "SDK result not paired to exact call")
                    usage = row["token_usage"]
                    raw_response = c["response"]
                    key = (tuple(row["seed_ctx"]), 0, row["attempt"])
                    sdk_join[key].update(usage=usage, finish_reason=row["finish_reason"],
                                         response=raw_response is not None)
                    if raw_response is None:
                        check(row["raw_content"] is None and row["finish_reason"] is None and usage is None
                              and row["n_choices"] is None and row["accepted"] is False,
                              "transport failure invented a response")
                    else:
                        choices = raw_response.get("choices", [])
                        check(type(choices) is list and bool(choices), "missing raw choice")
                        choice = choices[0]
                        raw_usage = raw_response.get("usage") or {}
                        check(row["raw_content"] == choice["message"].get("content")
                              and row["finish_reason"] == choice.get("finish_reason")
                              and row["n_choices"] == len(choices)
                              and usage == {k: raw_usage.get(k) for k in ("prompt_tokens", "completion_tokens")},
                              "selection interpretation differs from raw SDK response")
                    valid_transport = (
                        type(usage) is dict and set(usage) == {"prompt_tokens", "completion_tokens"}
                        and all(type(v) is int and v >= 0 for v in usage.values())
                        and usage["completion_tokens"] <= 8192 and row["n_choices"] == 1
                        and row["finish_reason"] == "stop")
                    chosen = None
                    if valid_transport:
                        try:
                            chosen = decode_selection(row["raw_content"], row["remaining"])
                        except ValueError:
                            pass
                    check(type(row["accepted"]) is bool and row["accepted"] == (chosen is not None),
                          "accepted flag disagrees with raw response")
                    if chosen is None:
                        check(type(row["error_type"]) is str, "rejected result lacks error type")
                        counters["invalid_attempts"] += 1
                        c["pending"] = None
                        c["next_attempt"] += 1
                    else:
                        check(row["error_type"] is None, "accepted result has error")
                        c["pending"] = {**row, "chosen": chosen}
                elif event == "selection_accepted":
                    c = contexts[row["registration"]]
                    pending = c["pending"]
                    check(pending is not None and pending["event"] == "sdk_result"
                          and pending["accepted"] is True and row["step"] == pending["step"]
                          and row["attempt"] == pending["attempt"]
                          and type(row["next_application_id"]) is int
                          and row["next_application_id"] == pending["chosen"]
                          and row["selected"] == c["selected"] + [pending["chosen"]], "accepted prefix")
                    c["selected"], c["pending"], c["next_attempt"] = row["selected"], None, 0
                    counters["accepted_steps"] += 1
                elif event == "panel_completed":
                    c = contexts[row["registration"]]
                    check(c["state"] == "running" and c["pending"] is None
                          and sorted(c["selected"]) == list(range(len(c["application_map"])))
                          and row["ranked_application_ids"] == c["selected"], "not a complete model permutation")
                    c["state"] = "completed"
                    counters["completed_panels"] += 1
                elif event == "batch_returned":
                    check(batch is not None and not batch["returned"] and row["registrations"] == batch["ids"],
                          "returned batch identity")
                    check(all(contexts[r]["state"] == "completed" for r in batch["ids"])
                          and row["rankings"] == [contexts[r]["selected"] for r in batch["ids"]], "returned orders")
                    for rid in batch["ids"]:
                        contexts[rid]["state"] = "returned"
                    counters["returned_panels"] += len(batch["ids"])
                    batch["returned"] = True
                elif event == "processing_started":
                    ids = row["registrations"]
                    check(processing is None and type(ids) is list and len(ids) == len(set(ids)), "processing IDs")
                    check((batch is not None and batch["returned"] and set(ids) == set(batch["ids"]))
                          or (batch is None and not contexts and not ids), "processing before complete batch")
                    processing = ids
                elif event == "batch_processed":
                    ids = row["registrations"]
                    check(processing is not None and ids == processing
                          and row["rankings"] == [contexts[r]["selected"] for r in ids]
                          and row["panels"] == [_panel_report(contexts[r]) for r in ids],
                          "processed order/map differs")
                    processed.extend(row["panels"])
                    counters["processed_panels"] += len(ids)
                    for rid in ids:
                        del contexts[rid]
                    processing, batch = None, None
                elif event == "finalized":
                    check(initial is not None and not contexts and batch is None and processing is None
                          and current_year is None,
                          "unfinished registered panels or requests")
                    check(row["summary"] == summary, "finalized summary mismatch")
                    terminal = True
                else:
                    check(False, "unknown/failing event in purported complete trace")
        check(terminal and counters == {k: summary[k] for k in _COUNTERS}, "missing finalization/counters")
        request_report = _validate_request_join(client, sdk_join, check, request_audit_path)
    except SequentialFundingError:
        raise
    except Exception as error:
        raise SequentialFundingError("invalid_sequential_audit", error_type=type(error).__name__) from error
    return {"funding_selection_protocol": SEQUENTIAL_PROTOCOL, "output_representation": OUTPUT_REPRESENTATION,
            "audit_sha256": checksum.hexdigest(), "source_sha256": summary["source_sha256"],
            "processed_panels": processed, "years": years, "request_audit": request_report}


def _validate_request_join(client, expected, check, path):
    """Join every admitted sequential SDK call to the independent client audit."""
    if client is None:
        check(not expected, "SDK calls without bound client")
        client = {"model": "Qwen/Qwen3-32B", "request_audit_path": str(path)}
    check(str(path) == client["request_audit_path"], "bound and supplied request audit differ")
    check(path.is_absolute(), "request audit path must be absolute")
    checksum, seen, pending = hashlib.sha256(), set(), {}
    finalized, initialized = None, None
    counts = {"n_requests": 0, "n_responses": 0, "n_transport_errors": 0, "n_clients": 0,
              "prompt_tokens": 0, "completion_tokens": 0, "max_input_tokens": 0,
              "max_reserved_total_tokens": 0, "guard_failures": 0}
    finish_reasons = {}
    with path.open("rb") as stream:
        for raw in stream:
            checksum.update(raw)
            row = json.loads(raw)
            check(finalized is None, "post-finalization request audit record")
            if row["event"] == "initialized":
                check(initialized is None and not any(counts.values()), "duplicate/late request initialization")
                initialized = row
                check(row["model"] == client["model"] and row["context_tokens"] == 32768
                      and row["sdk_retries"] == 5, "request audit identity")
            elif row["event"] == "client_bound":
                check(initialized is not None and counts["n_clients"] == 0 and not pending
                      and row["model"] == client["model"], "request client binding")
                counts["n_clients"] += 1
            elif row["event"] == "request_started":
                check(counts["n_clients"] == 1 and type(row["request_id"]) is int
                      and row["request_id"] == counts["n_requests"], "request ticket sequence")
                check(type(row["input_tokens"]) is int and row["input_tokens"] >= 0
                      and type(row["max_tokens"]) is int and row["max_tokens"] > 0
                      and row["reserved_total_tokens"] == row["input_tokens"] + row["max_tokens"]
                      and row["reserved_total_tokens"] <= 32768, "invalid exact token reservation")
                ctx = row.get("seed_ctx") or []
                key = None
                if ctx and ctx[0] == "phase5_funding_eval":
                    key = (tuple(ctx), row["item_index"], row["attempt"])
                    check(key in expected and key not in seen, "extra/duplicate sequential request")
                    spec = expected[key]
                    check(row["request_seed"] == spec["seed"] and row["payload_sha256"] == spec["payload_sha256"]
                          and row["temperature"] == .7 and row["max_tokens"] == 8192
                          and row["chat_template_kwargs"] == {"enable_thinking": True},
                          "audited HTTP request differs from sequential call")
                    seen.add(key)
                pending[row["request_id"]] = (key, row)
                counts["n_requests"] += 1
                counts["max_input_tokens"] = max(counts["max_input_tokens"], row["input_tokens"])
                counts["max_reserved_total_tokens"] = max(
                    counts["max_reserved_total_tokens"], row["reserved_total_tokens"])
            elif row["event"] in ("response", "transport_error"):
                check(row["request_id"] in pending, "duplicate/unmatched request terminal")
                key, started = pending.pop(row["request_id"])
                if row["event"] == "response":
                    check(type(row["prompt_tokens"]) is int and row["prompt_tokens"] == started["input_tokens"]
                          and type(row["completion_tokens"]) is int and 0 <= row["completion_tokens"] <= started["max_tokens"]
                          and type(row["finish_reasons"]) is list and len(row["finish_reasons"]) == 1,
                          "response contradicts exact input/output accounting")
                    if key is not None:
                        spec = expected[key]
                        check(spec["response"] and spec["usage"] == {
                            "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"]}
                            and row["finish_reasons"] == [spec["finish_reason"]], "audited response differs")
                    counts["n_responses"] += 1
                    counts["prompt_tokens"] += row["prompt_tokens"]
                    counts["completion_tokens"] += row["completion_tokens"]
                    reason = str(row["finish_reasons"][0])
                    finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
                else:
                    if key is not None:
                        check(not expected[key]["response"], "transport error claimed for raw response")
                    counts["n_transport_errors"] += 1
            elif row["event"] == "finalized":
                check(finalized is None, "multiple request audit finalizations")
                finalized = row
            else:
                check(False, "unknown/failing event in complete request audit")
    check(seen == set(expected) and not pending, "missing/pending sequential HTTP requests")
    check(finalized is not None and finalized["status"] == "complete" and finalized["n_clients"] == 1
          and finalized["guard_failures"] == 0 and finalized["pending_requests"] == 0, "unclean request audit")
    check(all(finalized.get(k) == v for k, v in counts.items())
          and finalized["finish_reasons"] == finish_reasons
          and counts["n_requests"] == counts["n_responses"] + counts["n_transport_errors"],
          "request audit global counters disagree with observed events")
    request_summary = json.loads(path.with_suffix(".summary.json").read_text())
    check(all(finalized.get(k) == v for k, v in request_summary.items()), "request summary differs")
    return {"path": str(path), "sha256": checksum.hexdigest(), "matched_sdk_calls": len(seen)}
