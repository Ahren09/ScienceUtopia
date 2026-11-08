import gzip
import hashlib
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path


class DuplicateJSONKey(ValueError):
    def __init__(self, key):
        self.key = key
        super().__init__(f"Duplicate JSON key: {key}")


class NonfiniteJSONNumber(ValueError):
    def __init__(self, value):
        self.value = value
        super().__init__(f"Invalid JSON numeric constant: {value}")


def decode_json(raw, *, strict=False, finite_floats=False):
    """Decode JSON, optionally rejecting ambiguous keys and nonfinite constants."""
    if not strict:
        return json.loads(raw)

    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DuplicateJSONKey(key)
            result[key] = value
        return result

    def finite(value):
        raise NonfiniteJSONNumber(value)

    def finite_float(value):
        import math

        parsed = float(value)
        if not math.isfinite(parsed):
            raise NonfiniteJSONNumber(value)
        return parsed

    options = {"parse_float": finite_float} if finite_floats else {}
    return json.loads(raw, object_pairs_hook=unique, parse_constant=finite, **options)


def file_sha256(path):
    """Hash file bytes with bounded memory, including large checkpoints."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value, **options):
    """Hash the caller's exact JSON representation, without changing its defaults."""
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def without_hash_metadata(value):
    """Compare recorded evidence independently of optional historical hashes."""
    if isinstance(value, dict):
        return {
            key: without_hash_metadata(item)
            for key, item in value.items()
            if key not in {"source_binding", "sequential_source_binding", "expected_source"}
            and str(key).rsplit("_", 1)[-1] not in {"sha256", "hash", "hashes", "checksum", "fingerprint"}
        }
    if isinstance(value, (list, tuple)):
        return [without_hash_metadata(item) for item in value]
    return value


@contextmanager
def atomic_open(
    path,
    mode="w",
    *,
    cleanup_on_error=False,
    create_parent=True,
):
    """Replace a file after writing succeeds; retain failed streams for recovery."""
    path = Path(path)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    published = False
    try:
        with temporary.open(mode) as stream:
            yield stream
        os.replace(temporary, path)
        published = True
    finally:
        if (published or cleanup_on_error) and temporary.exists():
            temporary.unlink()


def write_bytes_atomic(path, value, *, create_parent=True):
    """Atomically publish complete bytes, cleaning up failed publications."""
    with atomic_open(path, "wb", cleanup_on_error=True, create_parent=create_parent) as stream:
        stream.write(value)


def write_json_atomic(
    path,
    value,
    *,
    trailing_newline=True,
    streaming=False,
    create_parent=True,
    **options,
):
    """Atomically replace JSON using the caller's exact serialization options."""
    # Preserve directory creation before serialization, including failed encodes.
    if create_parent:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    if streaming:
        with atomic_open(path, create_parent=create_parent) as stream:
            json.dump(value, stream, **options)
            if trailing_newline:
                stream.write("\n")
        return
    suffix = "\n" if trailing_newline else ""
    write_bytes_atomic(
        path,
        (json.dumps(value, **options) + suffix).encode(),
        create_parent=create_parent,
    )


def write_json_gzip_atomic(path, value, *, create_parent=True, **options):
    """Stream a gzip checkpoint, preserving the original temporary-file header."""
    temporary = str(Path(path).with_suffix(Path(path).suffix + ".tmp"))
    with atomic_open(path, "wb", create_parent=create_parent) as raw:
        with gzip.GzipFile(filename=temporary, mode="wb", fileobj=raw) as compressed:
            with io.TextIOWrapper(compressed) as stream:
                json.dump(value, stream, **options)


def write_jsonl_atomic(path, rows, **options):
    """Stream a complete JSONL replacement without materializing its records."""
    with atomic_open(path) as stream:
        write_json_lines(stream, rows, options)


def write_parquet_atomic(path, frame, **options):
    """Publish a DataFrame through the same atomic stream used by JSON exporters."""
    with atomic_open(path, "wb") as stream:
        frame.to_parquet(stream, **options)


def read_jsonl(f):
    with open(f) as stream:
        return [decode_json(line) for line in stream]


def read_json(f):
    opener = gzip.open if str(f).endswith(".gz") else open
    with opener(f, "rt") as stream:
        return json.load(stream)


def write_json_lines(stream, rows, options):
    for row in rows:
        stream.write(json.dumps(row, **options) + "\n")


def write_jsonl(data: list, f, **options):
    with open(f, "w") as file:
        write_json_lines(file, data, options)


def write_json(
    data: dict,
    f,
    *,
    mode="w",
    trailing_newline=False,
    streaming=True,
    fsync=False,
    **options,
):
    """Write JSON with explicit creation and pre-encoding policies."""
    options.setdefault("indent", 2)
    if not streaming:
        text = json.dumps(data, **options) + ("\n" if trailing_newline else "")
        with open(f, mode) as file:
            file.write(text)
            if fsync:
                file.flush()
                os.fsync(file.fileno())
        return
    with open(f, mode) as file:
        json.dump(data, file, **options)
        if trailing_newline:
            file.write("\n")
        if fsync:
            file.flush()
            os.fsync(file.fileno())
