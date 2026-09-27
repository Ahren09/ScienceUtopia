"""Behavioral regressions for the shared implementations and historical readers."""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from utopia.analysis.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    bootstrap_mean_ci,
    hierarchical_bootstrap,
)
from utopia.runtime.commands import command_source, module_command
from utopia.runtime.historical import historical_source_path, trusted_audit_sources
from utopia.utils.data_utils import (
    DuplicateJSONKey,
    NonfiniteJSONNumber,
    decode_json,
    file_sha256,
    json_sha256,
    read_json,
    write_json_atomic,
)
from utopia.utils.paths import project_root


ROOT = project_root(__file__)
GOLDEN = read_json(ROOT / "tests/support/fixtures/refactor_golden.json")


def test_streaming_hash_and_structured_hash_have_distinct_inputs(tmp_path):
    path = tmp_path / "large.bin"
    data = b"\x00\xffpayload" * 300_000
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()
    value = {"z": "研究", "a": [1, None]}
    for options in (
        {},
        {"sort_keys": True, "ensure_ascii": False, "separators": (",", ":")},
    ):
        expected = hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()
        assert json_sha256(value, **options) == expected


def test_audit_hashes_preserve_different_unicode_encodings():
    from utopia.funding.sequential import _sha as sequential_hash
    from utopia.models.request_audit import _sha as request_hash

    value = {"message": "研究"}
    options = dict(sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert sequential_hash(value) == json_sha256(value, ensure_ascii=True, **options)
    assert request_hash(value) == json_sha256(value, ensure_ascii=False, **options)
    assert sequential_hash(value) != request_hash(value)


def test_strict_decoding_preserves_caller_error_semantics(tmp_path):
    from utopia.analysis.switching_propensity import AnalysisError, _read_json
    from utopia.funding.sequential import decode_selection
    from utopia.runtime.switching_evidence import ProtocolFailure, strict_evidence_json

    duplicate = '{"next_application_id": 0, "next_application_id": 1}'
    with pytest.raises(DuplicateJSONKey):
        decode_json(duplicate, strict=True)
    with pytest.raises(ValueError, match="^Duplicate JSON key$"):
        decode_selection(duplicate, [0, 1])
    path = tmp_path / "ambiguous.json"
    path.write_text(duplicate)
    with pytest.raises(AnalysisError, match="^duplicate_json_key"):
        _read_json(path)
    with pytest.raises(ProtocolFailure, match="^duplicate_evidence_json_key"):
        strict_evidence_json(duplicate, path)
    with pytest.raises(NonfiniteJSONNumber):
        decode_json('{"x": NaN}', strict=True)
    with pytest.raises(ProtocolFailure, match="^nonfinite_evidence_json"):
        strict_evidence_json('{"x": 1e999}', path)
    # Existing queue/analysis decoding only rejected nonstandard constants.
    assert np.isinf(decode_json('{"x": 1e999}', strict=True)["x"])


def test_json_gzip_corruption_and_atomic_replacement(tmp_path):
    value = {"year": 4, "papers": [], "text": "研究"}
    plain, compressed = tmp_path / "state.json", tmp_path / "state.json.gz"
    write_json_atomic(plain, value, indent=2, allow_nan=False)
    with gzip.open(compressed, "wt") as stream:
        json.dump(value, stream)
    assert read_json(plain) == read_json(compressed) == value
    previous = plain.read_bytes()
    with pytest.raises(ValueError):
        write_json_atomic(plain, {"x": float("nan")}, allow_nan=False)
    assert plain.read_bytes() == previous
    compressed.write_bytes(b"not gzip")
    with pytest.raises(gzip.BadGzipFile):
        read_json(compressed)
    plain.write_text("{")
    with pytest.raises(json.JSONDecodeError):
        read_json(plain)


def test_bootstraps_match_pre_migration_values_and_leave_global_rng_unchanged():
    state = np.random.get_state()
    values = [1.0, 4.0, 9.0]
    assert (
        list(bootstrap_mean_ci(values, n_boot=200, seed=42))
        == GOLDEN["bootstrap_mean_ci"]
    )
    assert list(bootstrap_ci(values, n_boot=200, seed=42)) == GOLDEN["bootstrap_ci"]
    contrasts = pd.DataFrame(
        {
            "outcome": ["x"] * 4,
            "contrast": ["A-B"] * 4,
            "seed": [1, 1, 2, 2],
            "diff": [1.0, 2.0, -1.0, 4.0],
        }
    )
    assert (
        hierarchical_bootstrap(contrasts, n_boot=200, seed=42).to_dict(orient="records")
        == GOLDEN["hierarchical_bootstrap"]
    )
    np.testing.assert_equal(state, np.random.get_state())
    assert bootstrap_mean_ci([]) == (None, None, None)
    assert bootstrap_mean_ci([3]) == (3.0, None, None)


def test_multiple_testing_ties_and_empty_input():
    np.testing.assert_allclose(
        benjamini_hochberg([0.01, 0.01, 0.04, 0.2]), [0.02, 0.02, 0.04 * 4 / 3, 0.2]
    )
    assert benjamini_hochberg([]).size == 0




def test_module_commands_bind_package_sources_and_reject_foreign_modules(tmp_path):
    command = module_command(
        "python", "utopia.experiments.influx_factorial", "--seed", 42
    )
    assert command == [
        "python",
        "-m",
        "utopia.experiments.influx_factorial",
        "--seed",
        "42",
    ]
    assert command_source(command, tmp_path) == (
        tmp_path / "utopia/experiments/influx_factorial.py",
        3,
    )
    with pytest.raises(ValueError):
        command_source(["python", "-m", "outside.runner"], tmp_path)
    with pytest.raises(ValueError):
        command_source(["python", "src/runner.py", "--run"], tmp_path)


def test_utility_and_admission_imports_do_not_load_models_or_launch_processes():
    script = """
import argparse, subprocess, sys
from unittest.mock import patch
with patch.object(subprocess, "Popen", side_effect=AssertionError("process launch")), \\
     patch.object(argparse.ArgumentParser, "parse_args", side_effect=AssertionError("argv parse")):
    import utopia.utils.data_utils
    import utopia.runtime.provenance
    import utopia.runtime.switching_evidence
    import utopia.runtime.structured_output_probe
    import utopia.funding.feedback
    import utopia.analysis.statistics
assert not {"torch", "vllm", "transformers", "openai", "langchain_core"} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, check=True)


def test_readonly_historical_audit_requires_complete_independent_source_pins(tmp_path):
    from tests.support.simulation import zero_sequential_fixture
    from utopia.funding.sequential import SequentialFundingError, validate_audit

    summary = zero_sequential_fixture(tmp_path, years=1)
    expected = GOLDEN["historical_sources"]
    source_hash = expected[historical_source_path("utopia/funding/sequential.py")]
    summary.update(source_files_sha256=expected, source_sha256=source_hash)
    audit_path = Path(summary["audit_path"])
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    rows[0]["identity"].update(source_files_sha256=expected, source_sha256=source_hash)
    assert rows[-1]["event"] == "finalized"
    rows[-1]["summary"] = summary
    audit_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary_path = Path(summary["summary_path"])
    summary_path.write_text(json.dumps(summary))
    audit_path.chmod(0o444)
    summary_path.chmod(0o444)
    before = {path: file_sha256(path) for path in (audit_path, summary_path)}
    request_path = tmp_path / "llm_request_audit.jsonl"
    with pytest.raises(SequentialFundingError, match="invalid_sequential_audit") as rejected:
        validate_audit(audit_path, summary, request_audit_path=request_path)
    assert rejected.value.diagnostics["detail"] == "frozen source files differ"
    report = validate_audit(
        audit_path,
        summary,
        request_audit_path=request_path,
        expected_sources=trusted_audit_sources(
            summary["source_files_sha256"], expected
        ),
    )
    assert len(report["years"]) == 1
    assert report["processed_panels"] == []
    assert before == {path: file_sha256(path) for path in before}
    with pytest.raises(ValueError, match="trusted source"):
        trusted_audit_sources({**expected, "const.py": "0" * 64}, expected)
    incomplete = dict(expected)
    incomplete.pop("const.py")
    with pytest.raises(ValueError, match="trusted source"):
        trusted_audit_sources(incomplete, expected)
