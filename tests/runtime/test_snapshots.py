import json
from pathlib import Path
import subprocess

from utopia.runtime.snapshots import preserve_snapshot, upload_snapshot


def test_failed_upload_retains_complete_server_evidence(tmp_path, monkeypatch):
    docs, logs, outputs, source = (tmp_path / name for name in ("docs", "logs", "outputs", "source"))
    for path in (docs, logs, outputs, source / "utopia"):
        path.mkdir(parents=True)
    (source / "utopia/model.py").write_text("value = 1\n")
    (source / "requirements.txt").write_text("")
    (docs / "audit.jsonl").write_text('{"event":"response","cache_replay":false}\n')
    (logs / "simulation.log").write_text("year 2 complete\n")
    (outputs / "metrics.json").write_text('{"year":2}')
    checkpoint = outputs / "portable_checkpoint_year_2.json.gz"
    checkpoint.write_bytes(b"checkpoint fixture")

    class Cache:
        def backup(self, path):
            path.write_bytes(b"consistent response fixture")

        def committed_count(self):
            return 23

    receipt = preserve_snapshot(checkpoint, Cache(), docs, logs, source, {"cell": "P1F1"})
    backup = Path(receipt["server_directory"])
    assert receipt["completed_responses"] == 23
    assert {"responses.sqlite", checkpoint.name, "results/audit.jsonl",
            "outputs/metrics.json", "logs/simulation.log", "source.tar.gz",
            "source_manifest.json"} <= receipt["preserved_files"].keys()

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="expired credentials")

    monkeypatch.setattr(subprocess, "run", fail)
    receipt = upload_snapshot(receipt, "s3://fixture/study/P1F1")
    assert receipt["storage_status"] == "server_preserved_s3_pending"
    assert not receipt["is_offhost"] and not receipt["independent_readback_verified"]
    assert json.loads((backup / "snapshot.json").read_text()) == receipt
    assert (backup / "responses.sqlite").read_bytes() == b"consistent response fixture"


def test_upload_success_does_not_claim_independent_readback(tmp_path, monkeypatch):
    backup = tmp_path / "snapshot_1"
    backup.mkdir()
    receipt = {"server_directory": str(backup), "year_completed": 2,
               "is_offhost": False, "independent_readback_verified": False}
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    result = upload_snapshot(receipt, "s3://fixture/study/P1F1")
    assert result["storage_status"] == "s3_uploaded" and result["is_offhost"]
    assert not result["independent_readback_verified"]
    assert len(calls) == 2
