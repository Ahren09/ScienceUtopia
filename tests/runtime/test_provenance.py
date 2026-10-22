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


@pytest.mark.parametrize("resumed_audit", [
    {"n_requests": 0, "cache_replays": 0, "inference_responses": 0},
    {"n_requests": 2, "cache_replays": 2, "inference_responses": 0},
])
def test_portable_manifest_replaces_restored_counters(tmp_path, resumed_audit):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"command": "original-command", "git_commit": "original-commit"}))
    args = SimpleNamespace(seed=42, portable_checkpoint=True)
    cumulative = {"n_prompts": 2, "prompt_tokens": 6, "completion_tokens": 2}
    write_run_manifest(str(tmp_path), "running", args=args)
    write_run_manifest(str(tmp_path), "complete", extra={"llm_call_stats": cumulative})

    # A final-upload failure leaves completed native stats. A new process loads
    # the same authoritative checkpoint and ends again, possibly using only cache.
    write_run_manifest(str(tmp_path), "paused")
    write_run_manifest(str(tmp_path), "running", args=args)
    extra = {"llm_call_stats": dict(cumulative), "request_audit": dict(resumed_audit)}
    write_run_manifest(str(tmp_path), "complete", extra=extra)  # Native finish omits args.
    write_run_manifest(str(tmp_path), "complete", extra=extra)  # Idempotent finalization.
    record = json.loads(path.read_text())
    assert record["llm_call_stats"] == cumulative
    assert record["request_audit"] == resumed_audit
    assert record["request_audit"]["inference_responses"] == 0
    assert extra == {"llm_call_stats": cumulative, "request_audit": resumed_audit}
    assert record["command"] == "original-command"
    assert record["git_commit"] == "original-commit"


@pytest.mark.parametrize("portable", [False, True])
def test_manifest_mode_uses_persisted_args_when_final_call_omits_them(tmp_path, portable):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({
        "command": "original-command", "args": {"portable_checkpoint": portable},
        "llm_call_stats": {"n_prompts": 4, "prompt_tokens": 12, "old_only": 7},
    }))
    restored = {"n_prompts": 2, "prompt_tokens": 6}
    write_run_manifest(str(tmp_path), "complete", extra={"llm_call_stats": restored})
    expected = restored if portable else {"n_prompts": 6, "prompt_tokens": 18, "old_only": 7}
    assert json.loads(path.read_text())["llm_call_stats"] == expected
