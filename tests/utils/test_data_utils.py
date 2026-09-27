import json
import gzip
import stat
from unittest.mock import patch

import pandas as pd
import pytest

from utopia.utils.data_utils import (
    write_bytes_atomic,
    write_json,
    write_json_atomic,
    write_jsonl_atomic,
    write_parquet_atomic,
    write_json_gzip_atomic,
)


def test_json_serialization_bytes_and_final_newline_are_explicit(tmp_path):
    data = {"unicode": "é", "nan": float("nan"), "rows": [2, 1]}
    for streaming in (False, True):
        for newline in (False, True):
            path = tmp_path / f"{streaming}-{newline}.json"
            write_json_atomic(
                path,
                data,
                indent=2,
                ensure_ascii=False,
                streaming=streaming,
                trailing_newline=newline,
            )
            expected = json.dumps(data, indent=2, ensure_ascii=False) + (
                "\n" if newline else ""
            )
            assert path.read_bytes() == expected.encode()


def test_atomic_stream_failure_preserves_destination_and_partial_temporary(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("original")
    with pytest.raises(TypeError):
        write_json_atomic(path, ["é", object()], streaming=True, trailing_newline=False)
    assert path.read_text() == "original"
    assert path.with_suffix(".json.tmp").read_text() == '["\\u00e9", '
    path.with_suffix(".json.tmp").unlink()
    with pytest.raises(TypeError):
        write_json_atomic(path, ["é", object()])
    assert path.read_text() == "original"
    assert not path.with_suffix(".json.tmp").exists()


def test_jsonl_atomic_writer_consumes_rows_lazily_and_keeps_failed_export(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("old")
    visited = []

    def rows():
        for i in range(3):
            visited.append(i)
            yield {"i": i}
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        write_jsonl_atomic(path, rows(), sort_keys=True, allow_nan=False)
    assert visited == [0, 1, 2]
    assert path.read_text() == "old"
    assert (
        path.with_suffix(".jsonl.tmp").read_text() == '{"i": 0}\n{"i": 1}\n{"i": 2}\n'
    )


def test_exclusive_and_private_publication_keep_permissions_and_contents(tmp_path):
    path = tmp_path / "private.json"
    write_bytes_atomic(path, b"first", private=True, exclusive=True, readonly=True)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        write_bytes_atomic(path, b"second", private=True, exclusive=True)
    assert path.read_bytes() == b"first"
    assert not list(tmp_path.glob(".atomic-*"))
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        write_bytes_atomic(link, b"second", private=True)
    assert path.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        write_json({"new": True}, path, mode="x")


def test_parquet_stream_matches_original_path_writer(tmp_path):
    frame = pd.DataFrame({"id": [1, 2], "value": ["é", None]})
    original = tmp_path / "original.parquet"
    current = tmp_path / "current.parquet"
    frame.to_parquet(original, index=False)
    write_parquet_atomic(current, frame, index=False)
    assert current.read_bytes() == original.read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(current), frame)


def test_gzip_checkpoint_preserves_payload_header_and_compression(tmp_path):
    expected = tmp_path / "before/checkpoint.json.gz.tmp"
    actual = tmp_path / "after/checkpoint.json.gz"
    expected.parent.mkdir()
    actual.parent.mkdir()
    value = {"text": "é", "values": [0.5, 2, None]}
    with patch("gzip.time.time", return_value=12345):
        with gzip.open(expected, "wt") as stream:
            json.dump(value, stream)
        write_json_gzip_atomic(actual, value, create_parent=False)
    assert actual.read_bytes() == expected.read_bytes()
    with gzip.open(actual, "rt") as stream:
        assert json.load(stream) == value


def test_preencoded_and_streaming_writes_preserve_different_failure_policies(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("previous")
    with pytest.raises(TypeError):
        write_json({"value": object()}, path, streaming=False, trailing_newline=True)
    assert path.read_text() == "previous"
    with pytest.raises(TypeError):
        write_json({"value": object()}, path)
    assert path.read_text() != "previous"
    missing = tmp_path / "missing/value.json"
    with pytest.raises(FileNotFoundError):
        write_json_atomic(missing, {}, streaming=True, create_parent=False)
    assert not missing.parent.exists()
