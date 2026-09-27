"""Opt-in request accounting for the prospectively frozen Funding feedback experiments.

No prompts, decoding parameters, request seeds, or retry policies are changed.
The exact pinned chat tokenizer checks input plus the caller's output reservation
before HTTP. Deterministic context violations, provenance mismatches and failed
audit writes escape the simulator's broad ``except Exception`` fallbacks.
"""

from utopia.utils.data_utils import json_sha256

from utopia.utils.data_utils import file_sha256

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time


MODEL = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
PINNED_MODELS = {
    MODEL: REVISION,
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
CONTEXT_TOKENS = 32768
_ACTIVE = None
_SCOPE_LOCK = threading.Lock()


class RequestAuditFailure(BaseException):
    """Abort a new experimental world instead of manufacturing fallback data."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value):
    return json_sha256(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False)


@contextmanager
def _preserve_rng():
    """Tokenizer setup runs before worker threads; preserve already loaded RNGs."""
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch_state = torch.get_rng_state() if torch is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)


def _load_tokenizer(model_name=MODEL, revision=REVISION):
    with _preserve_rng():
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        # Resolve the pinned commit entirely from the configured local HF
        # cache. Passing a Hub model ID to transformers 4.57.3 can still
        # trigger model_info in _patch_mistral_regex despite local_files_only.
        # An existing absolute snapshot directory keeps that path local too.
        snapshot = Path(snapshot_download(
            repo_id=model_name, revision=revision,
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])).resolve(strict=True)
        if (not snapshot.is_dir() or snapshot.name != revision
                or snapshot.parent.name != "snapshots"
                or snapshot.parent.parent.name != "models--" + model_name.replace("/", "--")):
            raise ValueError("Cached tokenizer snapshot does not match the pinned model/revision")
        return AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False)


class RequestAudit:
    def __init__(self, path, model_name=MODEL, *, revision=None, _tokenizer=None):
        revision = revision or PINNED_MODELS.get(model_name)
        if not revision or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise RequestAuditFailure(f"Supply an immutable --model_revision for {model_name!r}")
        self.path = Path(path)
        if not self.path.is_absolute() or not self.path.parent.is_dir():
            raise RequestAuditFailure("Request audit requires an existing absolute output directory")
        self.summary_path = self.path.with_suffix(".summary.json")
        if self.summary_path.exists():
            raise RequestAuditFailure(f"Fresh request audit required: {self.summary_path}")
        self.model_name = model_name
        self._lock = threading.RLock()
        self._token_lock = threading.Lock()
        self._fd = None
        self._closed = False
        self._stats = {
            "status": "running", "n_clients": 0, "n_requests": 0,
            "n_responses": 0, "n_transport_errors": 0,
            "guard_failures": 0, "finish_reasons": {},
            "prompt_tokens": 0, "completion_tokens": 0,
            "max_input_tokens": 0, "max_reserved_total_tokens": 0,
        }
        self._pending = set()
        try:
            self.tokenizer = _tokenizer if _tokenizer is not None else _load_tokenizer(model_name, revision)
            self._fd = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
                | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
            self.identity = {
                "protocol": "request-audit-v1", "model": model_name,
                "model_revision": revision, "context_tokens": CONTEXT_TOKENS,
                "chat_template_sha256": _sha(self.tokenizer.chat_template),
                "audit_source_sha256": file_sha256(Path(__file__)),
                "sdk_retries": 5,
                "scope": "One event per SDK call; SDK-internal transport retries are not individual events.",
            }
            self._write({"event": "initialized", **self.identity})
            os.fsync(self._fd)
        except BaseException as error:
            if self._fd is not None:
                os.close(self._fd)
            raise RequestAuditFailure(f"Cannot initialize exact request audit: {error}") from error

    def _write(self, record):
        with self._lock:
            if self._closed:
                raise RequestAuditFailure("Attempt to write a closed request audit")
            try:
                data = (_json({"time_ns": time.time_ns(), **record}) + "\n").encode()
                offset = 0
                while offset < len(data):
                    count = os.write(self._fd, data[offset:])
                    if count <= 0:
                        raise OSError("Short audit write")
                    offset += count
            except Exception as error:
                raise RequestAuditFailure(f"Cannot persist request audit: {error}") from error

    def bind(self, model_name):
        with self._lock:
            if model_name != self.model_name or self._stats["n_clients"]:
                self._fail("A request audit must bind exactly one pinned model client")
            self._stats["n_clients"] += 1
            self._write({"event": "client_bound", "model": model_name})
        return self

    def _fail(self, message, **details):
        with self._lock:
            self._stats["guard_failures"] += 1
            self._write({"event": "guard_failure", "message": message, **details})
        raise RequestAuditFailure(message)

    def before(self, payload, *, seed_ctx=None, item_index=0, attempt=0):
        """Count the exact untruncated chat request, including generation prefix."""
        try:
            if payload["model"] != self.model_name:
                self._fail("Request model differs from pinned audit model")
            output = payload["max_tokens"]
            if type(output) is not int or output <= 0:
                self._fail("Explicit positive integer max_tokens required")
            extra = payload.get("extra_body") or {}
            template_kwargs = extra.get("chat_template_kwargs") or {}
            if set(template_kwargs) - {"enable_thinking"}:
                self._fail("Unreviewed chat template arguments", keys=sorted(template_kwargs))
            if "guided_json" in extra or "truncate_prompt_tokens" in extra:
                self._fail("Legacy guided_json or prompt truncation is forbidden")
            with self._token_lock:
                # vLLM 0.12 renders text first, then tokenizes without extra
                # special tokens. Using both steps also audits the rendered text.
                text = self.tokenizer.apply_chat_template(
                    payload["messages"], tokenize=False, add_generation_prompt=True,
                    **template_kwargs)
                token_ids = self.tokenizer(
                    text, add_special_tokens=False, truncation=False)["input_ids"]
            count = len(token_ids)
            record = {
                "seed_ctx": list(seed_ctx) if seed_ctx is not None else None,
                "item_index": item_index, "attempt": attempt,
                "request_seed": extra.get("seed"), "temperature": payload["temperature"],
                "input_tokens": count, "max_tokens": output,
                "reserved_total_tokens": count + output,
                "payload_sha256": _sha(payload),
                "rendered_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "chat_template_kwargs": template_kwargs,
            }
            with self._lock:
                self._stats["max_input_tokens"] = max(self._stats["max_input_tokens"], count)
                self._stats["max_reserved_total_tokens"] = max(
                    self._stats["max_reserved_total_tokens"], count + output)
                if count + output > CONTEXT_TOKENS:
                    self._fail("Input plus requested output exceeds pinned context", **record)
                request_id = self._stats["n_requests"]
                self._stats["n_requests"] += 1
                self._pending.add(request_id)
                record["request_id"] = request_id
                self._write({"event": "request_started", **record})
            return record
        except Exception as error:
            self._fail(f"Exact input token accounting failed: {error}")

    def transport_error(self, ticket, error):
        with self._lock:
            self._pending.remove(ticket["request_id"])
            self._stats["n_transport_errors"] += 1
            self._write({"event": "transport_error", "request_id": ticket["request_id"],
                         "error_type": type(error).__name__, "error": str(error)})

    def response(self, ticket, response):
        with self._lock:
            self._pending.remove(ticket["request_id"])
            self._stats["n_responses"] += 1
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            choices = getattr(response, "choices", [])
            reasons = [getattr(choice, "finish_reason", None) for choice in choices]
            self._write({"event": "response", "request_id": ticket["request_id"],
                         "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                         "finish_reasons": reasons})
            if (type(prompt_tokens) is not int or prompt_tokens != ticket["input_tokens"]
                    or type(completion_tokens) is not int or completion_tokens < 0
                    or completion_tokens > ticket["max_tokens"]):
                self._fail("Server token usage contradicts exact request accounting",
                           request_id=ticket["request_id"], expected_prompt=ticket["input_tokens"],
                           prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            if len(choices) != 1:
                self._fail("Expected one completion choice", request_id=ticket["request_id"])
            reason = str(reasons[0])
            counts = self._stats["finish_reasons"]
            counts[reason] = counts.get(reason, 0) + 1
            self._stats["prompt_tokens"] += prompt_tokens
            self._stats["completion_tokens"] += completion_tokens

    def create(self, callback, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        ticket = self.before(payload, seed_ctx=seed_ctx, item_index=item_index, attempt=attempt)
        try:
            response = callback(**payload)
        except BaseException as error:
            self.transport_error(ticket, error)
            raise
        try:
            self.response(ticket, response)
        except RequestAuditFailure as error:
            # Preserve the received object for a caller's terminal audit even
            # when provenance validation aborts before this method returns.
            # The same fatal exception still escapes without another request.
            error.received_response = response
            error.request_ticket = dict(ticket)
            raise
        return response

    def summary(self):
        with self._lock:
            return json.loads(_json({
                **self.identity, **self._stats, "pending_requests": len(self._pending),
                "audit_path": str(self.path), "summary_path": str(self.summary_path),
            }))

    def close(self, successful):
        with self._lock:
            valid = (self._stats["n_clients"] == 1 and self._stats["n_requests"] > 0
                     and not self._pending and self._stats["guard_failures"] == 0)
            self._stats["status"] = "complete" if successful and valid else "failed"
            summary = self.summary()
            try:
                self._write({"event": "finalized", **summary})
                os.fsync(self._fd)
                data = (_json(summary) + "\n").encode()
                with self.summary_path.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as error:
                raise RequestAuditFailure(f"Cannot finalize required request audit: {error}") from error
            finally:
                os.close(self._fd)
                self._closed = True
            if successful and not valid:
                raise RequestAuditFailure("A complete run requires one client, requests and no audit violations")


def active_audit_for(model_name):
    return _ACTIVE.bind(model_name) if _ACTIVE is not None else None


@contextmanager
def request_audit_scope(path, model_name=MODEL, *, revision=None, _tokenizer=None):
    """Wrap model construction and the complete simulation in one process."""
    global _ACTIVE
    if not _SCOPE_LOCK.acquire(blocking=False):
        raise RequestAuditFailure("Nested or concurrent request audit scopes are forbidden")
    audit = None
    successful = False
    try:
        audit = RequestAudit(path, model_name, revision=revision, _tokenizer=_tokenizer)
        _ACTIVE = audit
        yield audit
        successful = True
    finally:
        _ACTIVE = None
        try:
            if audit is not None:
                audit.close(successful)
        finally:
            _SCOPE_LOCK.release()
