"""Strict, reversible funding-ranking gate; standard library only.

Install BEFORE generating funding prompts or entering the original phase 5::

    handle = install_funding_validation(FundingAgency, new_audit_jsonl_path)
    try:
        simulation.run()
    finally:
        handle.restore()
        audit_summary = handle.summary()

``audit_path`` must be a NEW file in an existing private output directory.
Creation is exclusive (0600); existing files, including symlinks, are refused.
No output directories, source files, or RNGs are changed. The default mode
preserves prompts and rankings. Opt-in ``compact=True`` uses the companion
``compact_funding`` API to replace only the response-format prompt tail
and request an ordered application-ID permutation. Valid compact output is
expanded without repair only after all final panels pass preflight; original
result lists and message histories remain untouched.
Install once per worker/run, on the actual FundingAgency class referenced by
simulation. No retries are added: the stock phase's three attempts remain
the bound. Panel cap25, seed42, population and execution-host policy belong to
the experiment owners, not this gate.

Validation attempts return a bool. Every final panel is checked and logged
before the original processor is invoked even once. A final failure, or a
failure to persist the audit, raises FundingRankingValidationError, deliberately
outside Exception so ordinary model/retry/fallback handlers cannot swallow it.
Catch that exact type only at the worker boundary to mark the run invalid;
never translate it into a ranking or resume an award update.

JSONL has deterministic per-installation sequence numbers, no timestamps or
raw model text. ``n_valid`` counts rows with valid, unique application IDs and
ranks; ``n_missing`` counts expected applications without such a row.
``n_rejected = n_returned - n_valid``. All colliding rows are rejected.
An invalid attempt may have zero rejected rows and positive missing coverage.
Final records describe preflight validation, not evidence of completed awards.
``summary()`` returns a detached snapshot and remains available after restore.
Require summary()["completion_allowed"] at the worker completion boundary. It
is false after any final, audit or stock-processing failure, any observed final
imputation/fallback, or any panel left unprocessed. Invalid attempts that recover
within the stock retry bound do not prohibit completion.
The final-batch preflight cannot undo costs or other work before phase 5 calls
the processor. Panels registered through the prompt wrapper must all reach the
final batch. Panels omitted from both final lists before installation cannot
be observed; install before prompt generation.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import wraps
import inspect
import json
import os
from threading import RLock

__all__ = [
    "FundingRankingValidationError",
    "inspect_funding_result",
    "install_funding_validation",
]

_SCHEMA = "scienceutopia.funding_validation.v1"
_MISSING = object()
_INSTALLATIONS = {}
_INSTALL_LOCK = RLock()


class FundingRankingValidationError(BaseException):
    """Fatal validity/audit failure; ``audit`` is safe, detached JSON metadata."""

    def __init__(self, audit):
        self.audit = deepcopy(audit)
        super().__init__(json.dumps(self.audit, sort_keys=True, allow_nan=False))


def inspect_funding_result(result, apps):
    """Return a pure coverage/identifier/rank-permutation report, without repair.

    Accept JSON dict/list results only. IDs and ranks require actual integers
    (not bool, floats, coercible strings, NaN or infinity). Applicant IDs must
    exactly match the string at apps[application_id]['applicant_id']; repeated
    applicant IDs are legitimate. Neither the input nor its ranking is edited.
    """
    errors = set()
    if type(apps) not in (list, tuple):
        errors.add("invalid_apps_container")
        apps = ()
    n = len(apps)
    expected = {}
    for i, app in enumerate(apps):
        if type(app) is not dict or type(app.get("applicant_id")) is not str:
            errors.add("invalid_expected_applicant_id")
        else:
            expected[i] = app["applicant_id"]

    if type(result) is list:
        rows = result
    elif type(result) is dict:
        rows = result.get("ranked_applications")
        if any(result.get(flag, False) is not False
               for flag in ("imputed_tail", "fallback_ranking")):
            errors.add("fabricated_result")
    else:
        rows = None
        errors.add("missing_result" if result is None else "invalid_result_container")
    if type(rows) is not list:
        errors.add("invalid_ranked_applications_container")
        rows = []
    if len(rows) != n:
        errors.add("wrong_number_of_rows")

    row_errors = []
    ids = Counter()
    ranks = Counter()
    n_imputed = n_fallback = 0
    for row in rows:
        reasons = set()
        row_errors.append(reasons)
        if type(row) is not dict:
            reasons.add("non_dict_row")
            continue
        app_id, rank = row.get("application_id"), row.get("rank")
        if type(app_id) is not int:
            reasons.add("application_id_not_integer")
        elif not 0 <= app_id < n:
            reasons.add("application_id_out_of_range")
        else:
            ids[app_id] += 1
            if app_id not in expected:
                reasons.add("invalid_expected_applicant_id")
            elif (type(row.get("applicant_id")) is not str
                  or row["applicant_id"] != expected[app_id]):
                reasons.add("inconsistent_ids")
        if type(rank) is not int:
            reasons.add("rank_not_integer")
        elif not 1 <= rank <= n:
            reasons.add("rank_out_of_range")
        else:
            ranks[rank] += 1
        if row.get("imputed_tail", False) is not False:
            n_imputed += 1
            reasons.add("imputed_tail")
        if row.get("fallback_ranking", False) is not False:
            n_fallback += 1
            reasons.add("fallback_ranking")

    valid_ids = set()
    reason_counts = Counter()
    for row, reasons in zip(rows, row_errors):
        if type(row) is dict:
            app_id, rank = row.get("application_id"), row.get("rank")
            if type(app_id) is int and ids[app_id] > 1:
                reasons.add("duplicate_application_id")
            if type(rank) is int and ranks[rank] > 1:
                reasons.add("duplicate_rank")
            if not reasons:
                valid_ids.add(app_id)
        reason_counts.update(reasons)
    missing = [i for i in range(n) if i not in valid_ids]
    if missing:
        errors.add("incomplete_coverage")
    if set(ranks) != set(range(1, n + 1)):
        errors.add("rank_not_permutation")
    rejected = sum(bool(reasons) for reasons in row_errors)
    return {
        "valid": not errors and not rejected,
        "n_expected": n,
        "n_returned": len(rows),
        "n_valid": len(rows) - rejected,
        "n_missing": len(missing),
        "n_rejected": rejected,
        "n_imputed": n_imputed,
        "n_fallback": n_fallback,
        "missing_application_ids": missing,
        "errors": sorted(errors),
        "row_error_counts": dict(sorted(reason_counts.items())),
    }


def _context_value(value):
    # Never stringify arbitrary inputs: their repr can expose prompts/paths or
    # raise. Nonfinite floats and booleans are not identifiers.
    return value if type(value) in (str, int) else None


class _FundingValidationHandle:
    def __init__(self, funding_class, audit_path, compact_api=None):
        self._class = funding_class
        self._compact_api = compact_api
        self._path = os.fspath(audit_path)
        if type(self._path) is not str:
            raise TypeError("audit_path must be a string or string PathLike")
        self._lock = RLock()
        self._saved = {}
        self._patches = {}
        self._contexts = {}
        self._fd = None
        self._audit_broken = False
        self._stats = {
            "schema": _SCHEMA,
            "output_representation": (
                compact_api.COMPACT_REPRESENTATION if compact_api is not None
                else "ranked_applications_v1"
            ),
            "audit_path": self._path,
            "installed": False,
            "audit_records": 0,
            "audit_failures": 0,
            "validation_attempts": 0,
            "invalid_attempts": 0,
            "final_panels": 0,
            "invalid_final_panels": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "processing_failures": 0,
            "imputed_rankings": 0,
            "fallback_rankings": 0,
            "unprocessed_panels": 0,
            "restore_conflicts": [],
        }

    def _audit_failure(self, operation, exc):
        self._audit_broken = True
        self._stats["audit_failures"] += 1
        raise FundingRankingValidationError({
            "schema": _SCHEMA, "event": "audit_error",
            "operation": operation, "audit_path": self._path,
            "error_type": type(exc).__name__,
            "persisted_records": self._stats["audit_records"],
        }) from exc

    def _emit(self, records):
        """Persist the whole preflight before permitting the stock processor."""
        with self._lock:
            start = self._stats["audit_records"]
            enriched = [
                {"schema": _SCHEMA, "sequence": start + i + 1,
                 "output_representation": self._stats["output_representation"], **record}
                for i, record in enumerate(records)
            ]
            try:
                payload = "".join(
                    json.dumps(record, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True, allow_nan=False) + "\n"
                    for record in enriched
                ).encode("utf-8")
                if payload:
                    remaining = memoryview(payload)
                    while remaining:
                        size = os.write(self._fd, remaining)
                        if size <= 0:
                            raise OSError("zero-byte audit write")
                        remaining = remaining[size:]
                    os.fsync(self._fd)
            except Exception as exc:
                self._audit_failure("write", exc)
            self._stats["audit_records"] += len(enriched)
            return enriched

    def _context(self, apps, metadata=None, *, registered=False):
        key = id(apps)
        context = self._contexts.get(key)
        if context is None or context["apps"] is not apps:
            context = {"apps": apps, "attempt": 0,
                       "program_id": None, "panel_index": None,
                       "registered": False}
            self._contexts[key] = context
        if type(metadata) is dict:
            context["program_id"] = _context_value(metadata.get("program_id"))
            context["panel_index"] = _context_value(metadata.get("panel_index"))
        context["registered"] |= registered
        return context

    def _inspect(self, result, apps):
        if self._compact_api is None:
            return inspect_funding_result(result, apps)
        try:
            expanded = self._compact_api.expand_compact_result(result, apps)
        except ValueError:
            # Diagnostic counts only: never repair or return a partial ranking.
            report = inspect_funding_result(None, apps)
            errors = {error for error in report["errors"] if error.startswith("invalid_app")
                      or error == "invalid_expected_applicant_id"}
            errors.add("compact_validation_failed")
            if type(result) is not dict or set(result) != {"ranked_application_ids"}:
                errors.add("compact_result_shape")
            ids = result.get("ranked_application_ids") if type(result) is dict else None
            if type(ids) is not list:
                errors.add("compact_ids_not_list")
                ids = []
            n = report["n_expected"]
            if len(ids) != n:
                errors.add("compact_wrong_row_count")
            counts = Counter(i for i in ids if type(i) is int and 0 <= i < n)
            valid_ids = set()
            row_errors = Counter()
            for i in ids:
                if type(i) is not int:
                    row_errors["compact_id_not_integer"] += 1
                elif not 0 <= i < n:
                    row_errors["compact_id_out_of_range"] += 1
                elif counts[i] != 1:
                    row_errors["compact_duplicate_id"] += 1
                elif type(apps[i]) is not dict or type(apps[i].get("applicant_id")) is not str:
                    row_errors["invalid_expected_applicant_id"] += 1
                else:
                    valid_ids.add(i)
            errors.update(row_errors)
            missing = [i for i in range(n) if i not in valid_ids]
            report.update(
                valid=False, n_returned=len(ids), n_valid=len(valid_ids),
                n_missing=len(missing), n_rejected=len(ids) - len(valid_ids),
                missing_application_ids=missing, errors=sorted(errors),
                row_error_counts=dict(sorted(row_errors.items())),
            )
            return report
        return inspect_funding_result(expanded, apps)

    def _validate(self, result, apps):
        with self._lock:
            self._require_active()
            report = self._inspect(result, apps)
            context = self._context(apps)
            context["attempt"] += 1
            record = {
                "event": "validation_attempt",
                "program_id": context["program_id"],
                "panel_index": context["panel_index"],
                "attempt": context["attempt"],
                **report,
            }
            self._stats["validation_attempts"] += 1
            self._stats["invalid_attempts"] += int(not report["valid"])
            self._emit([record])
            return report["valid"]

    def _preflight(self, batch_results, metadata_list, funding_programs,
                   application_log):
        """No stock calls, normalization, application-log or resource mutation."""
        batch_errors = []
        if type(batch_results) not in (list, tuple):
            batch_errors.append("invalid_batch_container")
            batch_results = ()
        if type(metadata_list) not in (list, tuple):
            batch_errors.append("invalid_metadata_container")
            metadata_list = ()
        if len(batch_results) != len(metadata_list):
            batch_errors.append("batch_metadata_length_mismatch")
        if type(funding_programs) is not dict:
            batch_errors.append("invalid_funding_programs")
        if application_log is not None and type(application_log) is not list:
            batch_errors.append("invalid_application_log")
        records = []
        seen_apps = set()
        seen_panels = set()
        seen_metadata = set()
        for i in range(max(len(batch_results), len(metadata_list))):
            extra_errors = []
            meta = metadata_list[i] if i < len(metadata_list) else None
            if type(meta) is not dict:
                extra_errors.append("invalid_panel_metadata")
                meta = {}
            # Application IDs are local to a panel. Full coverage within each
            # ranking does not prevent the same panel being processed twice.
            # Use program/panel identity, never applicant IDs or ranking values:
            # separate panels may legitimately have the same applicants/ranks.
            program_id, panel_index = meta.get("program_id"), meta.get("panel_index")
            panel_key = ((program_id, panel_index)
                         if type(program_id) is str and type(panel_index) is int
                         else None)
            if id(meta) in seen_metadata or (
                panel_key is not None and panel_key in seen_panels
            ):
                extra_errors.append("duplicate_panel_metadata")
            seen_metadata.add(id(meta))
            if panel_key is not None:
                seen_panels.add(panel_key)
            apps = meta.get("apps")
            seen_apps.add(id(apps))
            known = self._contexts.get(id(apps))
            if known is not None and known["registered"] and any(
                known[key] != _context_value(meta.get(key))
                for key in ("program_id", "panel_index")
            ):
                extra_errors.append("panel_metadata_changed")
            context = self._context(apps, meta)
            if type(meta.get("program_id")) is not str:
                extra_errors.append("invalid_program_id")
            elif (type(funding_programs) is dict
                  and meta["program_id"] not in funding_programs):
                extra_errors.append("unknown_program_id")
            # Stock only reads panel_index when application_log is enabled.
            # With no application log, absent panel_index is legitimate.
            if (application_log is not None or "panel_index" in meta) and (
                type(meta.get("panel_index")) is not int or meta["panel_index"] < 0
            ):
                extra_errors.append("invalid_panel_index")
            pair = batch_results[i] if i < len(batch_results) else None
            if type(pair) not in (list, tuple) or len(pair) != 2:
                extra_errors.append("invalid_result_pair")
                result = None
            else:
                result = pair[0]  # Message history stays private and untouched.
            report = self._inspect(result, apps)
            report["errors"] = sorted(set(report["errors"] + extra_errors))
            report["valid"] = report["valid"] and not extra_errors
            records.append({
                "event": "final_panel", "batch_index": i,
                "program_id": context["program_id"],
                "panel_index": context["panel_index"],
                "attempts_seen": context["attempt"], **report,
            })
        # Stock's retry loop enumerates the returned batch. If the transport
        # silently returns too few items it can omit a panel from BOTH final
        # lists; prompt registration lets us still fail before any awards.
        for context in self._contexts.values():
            if context["registered"] and id(context["apps"]) not in seen_apps:
                report = self._inspect(None, context["apps"])
                report["errors"] = sorted(set(report["errors"] + ["missing_final_panel"]))
                records.append({
                    "event": "final_panel", "batch_index": None,
                    "program_id": context["program_id"],
                    "panel_index": context["panel_index"],
                    "attempts_seen": context["attempt"], **report,
                })
        invalid = sum(not record["valid"] for record in records)
        self._stats["final_panels"] += len(records)
        self._stats["invalid_final_panels"] += invalid
        self._stats["imputed_rankings"] += sum(record["n_imputed"] for record in records)
        self._stats["fallback_rankings"] += sum(record["n_fallback"] for record in records)
        failed = bool(invalid or batch_errors)
        self._stats["failed_batches"] += int(failed)
        if batch_errors:
            records.append({"event": "final_batch_error", "errors": batch_errors})
        persisted = self._emit(records)
        if failed:
            raise FundingRankingValidationError({
                "schema": _SCHEMA, "event": "final_batch_rejected",
                "audit_path": self._path, "errors": batch_errors,
                "invalid_final_panels": invalid,
                "panels": [record for record in persisted
                           if record["event"] == "final_panel" and not record["valid"]],
            })

    def _require_active(self):
        if not self._stats["installed"]:
            raise FundingRankingValidationError({
                "schema": _SCHEMA, "event": "inactive_validation_handle",
                "audit_path": self._path,
            })
        if self._audit_broken:
            raise FundingRankingValidationError({
                "schema": _SCHEMA, "event": "audit_unusable",
                "audit_path": self._path,
                "persisted_records": self._stats["audit_records"],
            })

    def _pending_count(self):
        return sum(context["registered"] or context["attempt"] > 0
                   for context in self._contexts.values())

    def summary(self):
        """Return JSON counters and the explicit ``completion_allowed`` gate.

        A fresh installation with no funding panels is eligible for completion;
        a registered/attempted panel requires final processing. Restoring early
        cannot erase outstanding panels or turn a failed run into a success.
        """
        with self._lock:
            snapshot = deepcopy(self._stats)
            snapshot["unprocessed_panels"] += self._pending_count()
            snapshot["completion_allowed"] = not any(snapshot[key] for key in (
                "invalid_final_panels", "failed_batches", "processing_failures",
                "audit_failures", "imputed_rankings", "fallback_rankings",
                "unprocessed_panels",
            ))
            return snapshot

    def restore(self):
        """Restore exact owned descriptors (or inheritance); safe to repeat.

        A later replacement of a patched method is left intact and reported as
        a RuntimeError after all still-owned patches and the logger are released.
        """
        with _INSTALL_LOCK, self._lock:
            if not self._stats["installed"]:
                return
            conflicts = []
            for name, descriptor in self._patches.items():
                if vars(self._class).get(name, _MISSING) is not descriptor:
                    conflicts.append(name)
                    continue
                saved = self._saved[name]
                if saved is _MISSING:
                    delattr(self._class, name)
                else:
                    setattr(self._class, name, saved)
            self._stats["installed"] = False
            self._stats["restore_conflicts"] = conflicts
            _INSTALLATIONS.pop(self._class, None)
            self._stats["unprocessed_panels"] += self._pending_count()
            self._contexts.clear()
            fd, self._fd = self._fd, None
            try:
                if fd is not None:
                    os.close(fd)
            except Exception as exc:
                self._audit_failure("close", exc)
            if conflicts:
                raise RuntimeError("Funding validation restore conflict: " + ", ".join(conflicts))


def install_funding_validation(funding_class, audit_path, *, compact=False):
    """Install on the supplied class; return a handle with summary()/restore().

    The two static entry points are patched. If present, the original prompt
    builder is wrapped only to remember its returned program/panel metadata.
    Unrelated classes are untouched. Overlapping/nested installations on the
    same class hierarchy are refused before any file is created.
    ``compact=True`` opts into ordered application IDs; the companion module
    owns the exact schema, prompt-tail substitution and lossless expansion.
    """
    with _INSTALL_LOCK:
        if not isinstance(funding_class, type):
            raise TypeError("funding_class must be a class")
        if type(compact) is not bool:
            raise TypeError("compact must be a bool")
        for active in _INSTALLATIONS:
            if issubclass(funding_class, active) or issubclass(active, funding_class):
                raise RuntimeError("Funding validation already installed on this class hierarchy")
        for name in ("validate_funding_result", "process_funding_evaluation_results"):
            if not isinstance(inspect.getattr_static(funding_class, name), staticmethod):
                raise TypeError(name + " must be a staticmethod")
        compact_api = None
        if compact:
            from utopia.funding import compact as compact_api
        handle = _FundingValidationHandle(funding_class, audit_path, compact_api)
        original_validate = funding_class.validate_funding_result
        original_process = funding_class.process_funding_evaluation_results

        @wraps(original_validate)
        def validate(result, apps):
            return handle._validate(result, apps)

        @wraps(original_process)
        def process(batch_results, metadata_list, funding_programs,
                    novelty_penalties=None, lambda_funding=0.0,
                    application_log=None, slot_override=None):
            with handle._lock:
                handle._require_active()
                try:
                    handle._preflight(batch_results, metadata_list,
                                      funding_programs, application_log)
                    processing_results = batch_results
                    if compact_api is not None:
                        try:
                            # Entire batch has passed preflight and its audit
                            # is durable. Keep the supplied compact results and
                            # histories intact; pass a fresh expansion to stock.
                            processing_results = [
                                (compact_api.expand_compact_result(pair[0], meta["apps"]), pair[1])
                                for pair, meta in zip(batch_results, metadata_list)
                            ]
                        except ValueError as exc:
                            handle._stats["failed_batches"] += 1
                            record = {"event": "compact_expansion_error",
                                      "error_type": type(exc).__name__}
                            handle._emit([record])
                            raise FundingRankingValidationError(record) from exc
                    try:
                        outcome = original_process(
                            processing_results, metadata_list, funding_programs,
                            novelty_penalties=novelty_penalties,
                            lambda_funding=lambda_funding,
                            application_log=application_log,
                            slot_override=slot_override,
                        )
                    except BaseException:
                        handle._stats["processing_failures"] += 1
                        raise
                    handle._stats["successful_batches"] += 1
                    return outcome
                finally:
                    # Keep only in-flight context, never full-run raw panels.
                    handle._contexts.clear()

        patches = {
            "validate_funding_result": staticmethod(validate),
            "process_funding_evaluation_results": staticmethod(process),
        }
        if hasattr(funding_class, "get_funding_evaluation_prompts"):
            original_prompts = funding_class.get_funding_evaluation_prompts

            @wraps(original_prompts)
            def prompts(self, *args, **kwargs):
                with handle._lock:
                    handle._require_active()
                    value = original_prompts(self, *args, **kwargs)
                    if compact_api is not None:
                        try:
                            if type(value) not in (list, tuple) or len(value) != 3:
                                raise ValueError("invalid prompt-builder result")
                            texts, _original_format, metadata = value
                            if (type(texts) is not list or type(metadata) is not list
                                    or len(texts) != len(metadata)):
                                raise ValueError("prompt/metadata mismatch")
                            if any(type(meta) is not dict
                                   or type(meta.get("apps")) not in (list, tuple)
                                   for meta in metadata):
                                raise ValueError("invalid prompt application metadata")
                            rewritten = [
                                compact_api.compact_prompt(text, len(meta["apps"]))
                                for text, meta in zip(texts, metadata)
                            ]
                            parts = (rewritten, compact_api.compact_response_format(), metadata)
                            value = parts if type(value) is tuple else list(parts)
                        except ValueError as exc:
                            handle._stats["failed_batches"] += 1
                            record = {"event": "compact_prompt_error",
                                      "error_type": type(exc).__name__}
                            handle._emit([record])
                            raise FundingRankingValidationError(record) from exc
                    if type(value) in (list, tuple) and len(value) == 3:
                        metadata = value[2]
                        if type(metadata) in (list, tuple):
                            for meta in metadata:
                                if type(meta) is dict:
                                    handle._context(meta.get("apps"), meta, registered=True)
                    return value

            patches["get_funding_evaluation_prompts"] = prompts
        # Validate the class shape first. Exclusive creation preserves every
        # pre-existing source/output/private path, including symlinks/hardlinks.
        try:
            handle._fd = os.open(handle._path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except Exception as exc:
            handle._audit_failure("open", exc)
        handle._patches = patches
        handle._saved = {name: vars(funding_class).get(name, _MISSING) for name in patches}
        handle._stats["installed"] = True
        _INSTALLATIONS[funding_class] = handle
        try:
            for name, descriptor in patches.items():
                setattr(funding_class, name, descriptor)
        except BaseException:
            # Roll back only assignments already made, without touching the
            # original descriptor for an assignment that failed.
            handle._patches = {
                name: descriptor for name, descriptor in patches.items()
                if vars(funding_class).get(name, _MISSING) is descriptor
            }
            handle.restore()
            raise
        return handle
