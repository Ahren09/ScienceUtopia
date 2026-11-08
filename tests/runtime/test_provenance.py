import json
from types import SimpleNamespace

import pytest

from utopia.runtime.historical import trusted_audit_sources
from utopia.runtime.provenance import write_run_manifest
from utopia.utils.data_utils import json_sha256, without_hash_metadata
from utopia.utils.paths import project_root


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


def test_historical_source_hashes_do_not_block_saved_results():
    golden = json.loads(
        (project_root() / "tests/support/fixtures/ownership_golden.json").read_text()
    )
    pins = golden["previous_package_sources"]
    assert trusted_audit_sources(pins, pins) == pins
    altered = dict(pins)
    altered["utopia/utils/general_utils.py"] = "0" * 64
    assert trusted_audit_sources(altered, pins) == altered
    missing = dict(pins)
    missing.pop("utopia/utils/general_utils.py")
    assert trusted_audit_sources(missing, pins) == missing
    assert trusted_audit_sources({}, pins) == {}


def test_historical_hash_metadata_is_optional_but_results_and_seed_still_match():
    evidence = {
        "request_id": 17, "score": 0.5, "seed": 42,
        "env": {"PYTHONHASHSEED": "42"},
        "steps": [{"applicant_id": "A", "request_sha256": "old"}],
        "source_binding": {"old.py": "old"},
    }
    resumed = {
        "request_id": 17, "score": 0.5, "seed": 42,
        "env": {"PYTHONHASHSEED": "42"}, "steps": [{"applicant_id": "A"}],
    }
    assert without_hash_metadata(evidence) == resumed
    assert evidence["steps"][0]["request_sha256"] == "old"
    for key, value in (("request_id", 18), ("score", 0.7), ("seed", 43)):
        assert without_hash_metadata(evidence) != {**resumed, key: value}


def test_scientific_input_comparison_keeps_settings_and_ignores_provenance():
    from utopia.runtime.switching_evidence import scientific_inputs

    original = {
        "seed": 42, "resolved_config": {"reviewers": 3},
        "source_files_sha256": {"old.py": "old"},
        "embedding_model": {"model": "encoder", "revision": "v1", "files_sha256": {"weights": "old"}},
    }
    resumed = {
        **original, "source_files_sha256": {},
        "embedding_model": {"model": "encoder", "revision": "v1"},
    }
    assert scientific_inputs(original) == scientific_inputs(resumed)
    assert scientific_inputs(original) != scientific_inputs({**resumed, "seed": 43})
    assert scientific_inputs(original) != scientific_inputs({
        **resumed, "resolved_config": {"reviewers": 2},
    })
