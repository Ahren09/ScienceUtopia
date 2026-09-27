import pytest
from utopia.runtime.historical import trusted_audit_sources
from utopia.utils.paths import project_root
import json
from types import SimpleNamespace

from utopia.runtime.provenance import write_run_manifest
from utopia.utils.data_utils import json_sha256


def test_run_manifest_preserves_start_fields_and_accumulates_resume_statistics(
    tmp_path,
):
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "command": "frozen original command",
                "timestamp_running": "original-start",
                "git_commit": "original-commit",
                "llm_call_stats": {"calls": 4, "tokens": 10},
            }
        )
    )
    args = SimpleNamespace(seed=42)
    config = {"levels": (1, 2), "strategy": "explorer"}
    write_run_manifest(
        str(tmp_path),
        "complete",
        args,
        config,
        extra={"llm_call_stats": {"calls": 2, "failures": 1}},
    )
    record = json.loads(path.read_text())
    assert record["command"] == "frozen original command"
    assert record["git_commit"] == "original-commit"
    assert record["timestamp_running"] == "original-start"
    assert record["status"] == "complete"
    assert record["args"] == {"seed": 42}
    assert record["llm_call_stats"] == {"calls": 6, "tokens": 10, "failures": 1}
    assert record["scientific_config_hash"] == json_sha256(
        config, sort_keys=True, default=str
    )
    assert not path.read_bytes().endswith(b"\n")


def test_previous_package_source_manifest_remains_strictly_bound():
    golden = json.loads(
        (project_root() / "tests/support/fixtures/ownership_golden.json").read_text()
    )
    pins = golden["previous_package_sources"]
    assert trusted_audit_sources(pins, pins) == pins
    altered = dict(pins)
    altered["utopia/utils/general_utils.py"] = "0" * 64
    with pytest.raises(ValueError):
        trusted_audit_sources(altered, pins)
    missing = dict(pins)
    missing.pop("utopia/utils/general_utils.py")
    with pytest.raises(ValueError):
        trusted_audit_sources(missing, pins)
