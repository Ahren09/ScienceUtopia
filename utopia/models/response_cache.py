"""Cell-local durable response journal, independent of host and API endpoint.

SQLite uses DELETE journals and EXTRA synchronous commits. Each response is
committed before the caller or request audit can observe it. Copy a live cache
with backup(), never by copying the database during a transaction.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

from utopia.runtime.checkpointing import (
    PortableCheckpointError, PortableStop, canonical_json, scientific_binding, validate_binding,
)


class CompletedResponseCache:
    def __init__(self, path, *, binding, stop=None, _response_factory=None):
        self.path = Path(path)
        self.binding = validate_binding(binding)
        self.stop = stop
        self._response_factory = _response_factory
        self._mutex = threading.RLock()  # Only serialize short SQLite operations.
        self._inflight = {}  # Coalesce identical concurrent identities in one client.
        self.hits = 0
        self.misses = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        try:
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA synchronous=EXTRA")
            self._db.execute("CREATE TABLE IF NOT EXISTS metadata (binding TEXT NOT NULL)")
            self._db.execute("""CREATE TABLE IF NOT EXISTS replies (
                identity TEXT PRIMARY KEY, payload_sha TEXT NOT NULL,
                outcome TEXT NOT NULL, data TEXT NOT NULL, data_sha TEXT NOT NULL)""")
            rows = self._db.execute("SELECT binding FROM metadata").fetchall()
            identity = canonical_json(self.binding)
            if not rows:
                self._db.execute("INSERT INTO metadata VALUES (?)", (identity,))
                self._db.commit()
            elif len(rows) != 1:
                raise PortableCheckpointError("Response cache must contain one experiment binding")
            else:
                saved = json.loads(rows[0][0])
                if scientific_binding(saved) != scientific_binding(self.binding):
                    raise PortableCheckpointError("Response cache scientific binding changed")
                # Existing row IDs use this original binding. Metadata changes
                # must not create a second cache namespace or repeat requests.
                self.binding = saved
        except BaseException:
            self._db.close()
            raise

    def _decode(self, data, hit):
        import json
        if self._response_factory is None:
            from openai.types.chat import ChatCompletion
            response = ChatCompletion.model_validate(json.loads(data))
        else:
            response = self._response_factory(json.loads(data))
        # Parent request-audit can distinguish logical replay from new HTTP.
        object.__setattr__(response, "_portable_cache_hit", hit)
        return response

    def create(self, callback, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        try:
            return self._create(callback, seed_ctx=seed_ctx, item_index=item_index, attempt=attempt, **payload)
        except sqlite3.Error as error:
            failure = PortableCheckpointError(f"Response journal IO failed: {error}")
            if self.stop is not None:
                self.stop.fail(failure)
            raise failure from error
        except PortableCheckpointError as error:
            if self.stop is not None:
                self.stop.fail(error)
            raise

    def _create(self, callback, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        from concurrent.futures import Future
        extra = payload.get("extra_body") or {}
        seed = payload.get("seed", extra.get("seed"))
        if not seed_ctx or type(seed) is not int or payload.get("model") != self.binding["model"]:
            raise PortableCheckpointError("Cached calls require stable seed_ctx, seed and pinned model")
        if any(type(n) is not int or n < 0 for n in (item_index, attempt)):
            raise PortableCheckpointError("Request item index and attempt must be nonnegative integers")
        try:
            identity = canonical_json({"binding": self.binding, "seed_ctx": seed_ctx,
                                       "item_index": item_index, "attempt": attempt, "seed": seed})
            key = hashlib.sha256(identity.encode()).hexdigest()
            payload_sha = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
        except (TypeError, ValueError) as error:
            raise PortableCheckpointError(f"Invalid request identity/payload: {error}") from error
        if self.stop is not None:
            self.stop.check()
        with self._mutex:
            existing = self._db.execute("SELECT payload_sha, outcome, data, data_sha FROM replies WHERE identity=?",
                                        (key,)).fetchone()
            if existing is not None:
                return self._reuse(existing)
            pending = self._inflight.get(key)
            if pending is None:
                pending = Future()
                self._inflight[key] = pending
                owner = True
            else:
                owner = False
        if not owner:
            row = pending.result()
            with self._mutex:
                return self._reuse(row)
        try:
            if self.stop is not None:
                self.stop.check()
            try:
                response = callback(**payload)
            except Exception as error:
                # No response was received. Stop before a retry/fallback can
                # change scientific state. Resume retries this same identity.
                if self.stop is not None:
                    self.stop.request_stop()
                raise PortableStop(f"LLM transport interrupted: {type(error).__name__}: {error}") from error
            try:
                data = canonical_json(response.model_dump(mode="json"))
            except Exception as error:
                raise PortableCheckpointError(f"Cannot serialize returned response: {error}") from error
            outcome = "response"
            digest = hashlib.sha256(data.encode()).hexdigest()
            row = (payload_sha, outcome, data, digest)
            try:
                with self._mutex:
                    self._db.execute("INSERT INTO replies VALUES (?, ?, ?, ?, ?)", (key, *row))
                    self._db.commit()
                    self.misses += 1
            except Exception as error:
                raise PortableCheckpointError(f"Cannot durably store response: {error}") from error
            pending.set_result(row)
            object.__setattr__(response, "_portable_cache_hit", False)
            return response
        except BaseException as error:
            if not pending.done():
                pending.set_exception(error)
            raise
        finally:
            with self._mutex:
                self._inflight.pop(key, None)

    def _reuse(self, row):
        # The persisted request ID selects the reply. Historical content hashes
        # remain descriptive and never reject or recompute a completed request.
        saved_payload, outcome, data, digest = row
        self.hits += 1
        if outcome != "response":
            raise PortableCheckpointError(f"Unknown response outcome: {outcome}")
        try:
            response = self._decode(data, True)
            if response.model != self.binding["model"]:
                raise PortableCheckpointError("Cached response model differs from pinned model")
            return response
        except Exception as error:
            raise PortableCheckpointError(f"Cannot decode cached response: {error}") from error

    def committed_count(self):
        with self._mutex:
            return self._db.execute("SELECT count(*) FROM replies").fetchone()[0]

    def backup(self, destination):
        """Create a transactionally consistent file for parent-managed S3 upload."""
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError("Backup must use a different path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._mutex:
            target = sqlite3.connect(destination)
            try:
                self._db.backup(target)
            finally:
                target.close()
        # SQLite backup completes transactions, make the containing entry durable.
        import os
        with destination.open("rb") as stream:
            os.fsync(stream.fileno())
        fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def close(self):
        with self._mutex:
            if self._inflight:
                raise PortableCheckpointError("Cannot close cache with in-flight calls")
            self._db.close()
