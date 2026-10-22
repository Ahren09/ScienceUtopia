"""Main-thread durable response journal, independent of host and API endpoint.

SQLite uses DELETE journals and EXTRA synchronous commits. Each response is
committed before the caller or request audit can observe it. Copy a live cache
with backup(), never by copying the database during a transaction.
"""
import hashlib
from pathlib import Path
import sqlite3

from utopia.runtime.checkpointing import (
    PortableCheckpointError, PortableStop, canonical_json, validate_binding,
)


class CompletedResponseCache:
    def __init__(self, path, *, binding, stop=None, _response_factory=None):
        self.path = Path(path)
        self.binding = validate_binding(binding)
        self.stop = stop
        self._response_factory = _response_factory
        self.hits = 0
        self.misses = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
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
            elif rows != [(identity,)]:
                raise PortableCheckpointError("Response cache scientific binding changed")
        except BaseException:
            self._db.close()
            raise

    def decode_response(self, data, hit):
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
        key, payload_sha, saved = self.lookup(
            seed_ctx=seed_ctx, item_index=item_index, attempt=attempt, **payload)
        if saved is not None:
            return saved
        try:
            response = callback(**payload)
        except Exception as error:
            # Missing replies are retried only by an explicit native resume.
            if self.stop is not None:
                self.stop.request_stop()
            raise PortableStop(f"LLM transport interrupted: {type(error).__name__}: {error}") from error
        return self.store(key, payload_sha, response)

    def lookup(self, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        """Return the stable cache key, payload hash, and any completed reply."""
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
        try:
            existing = self._db.execute("SELECT payload_sha, outcome, data, data_sha FROM replies WHERE identity=?",
                                        (key,)).fetchone()
        except sqlite3.Error as error:
            raise PortableCheckpointError(f"Response journal read failed: {error}") from error
        return key, payload_sha, self.reuse_response(existing, payload_sha) if existing is not None else None

    def store(self, key, payload_sha, response):
        """Commit a received reply on the owner thread before parsing or auditing."""
        try:
            data = canonical_json(response.model_dump(mode="json"))
            digest = hashlib.sha256(data.encode()).hexdigest()
            self._db.execute("INSERT INTO replies VALUES (?, ?, ?, ?, ?)",
                             (key, payload_sha, "response", data, digest))
            self._db.commit()
        except Exception as error:
            failure = PortableCheckpointError(f"Cannot durably store response: {error}")
            if self.stop is not None:
                self.stop.fail(failure)
            raise failure from error
        self.misses += 1
        object.__setattr__(response, "_portable_cache_hit", False)
        return response

    def reuse_response(self, row, payload_sha):
        saved_payload, outcome, data, digest = row
        if saved_payload != payload_sha:
            raise PortableCheckpointError("Stable request identity replayed with a different payload")
        if hashlib.sha256(data.encode()).hexdigest() != digest:
            raise PortableCheckpointError("Corrupt cached response")
        self.hits += 1
        if outcome != "response":
            raise PortableCheckpointError(f"Unknown response outcome: {outcome}")
        try:
            return self.decode_response(data, True)
        except Exception as error:
            raise PortableCheckpointError(f"Cannot decode cached response: {error}") from error

    def committed_count(self):
        return self._db.execute("SELECT count(*) FROM replies").fetchone()[0]

    def backup(self, destination):
        """Create a transactionally consistent file for parent-managed S3 upload."""
        destination = Path(destination)
        if destination.resolve() == self.path.resolve():
            raise ValueError("Backup must use a different path")
        destination.parent.mkdir(parents=True, exist_ok=True)
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
        self._db.close()
