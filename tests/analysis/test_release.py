import json
from pathlib import Path

import pytest

from utopia.analysis.release import main, summarize_run


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_fixture(root):
    run = "fixture_seed42"
    docs = root / "docs" / run
    checkpoint_dir = root / "checkpoints" / run
    write(docs / "run_manifest.json", {
        "status": "complete", "years_completed": 2, "experiment_id": run,
        "args": {"num_years": 2, "seed": 42}, "model_id": "scripted-fixture",
        "dataset_identity": {"dataset": "synthetic-fixture"},
        "request_audit_paths": ["llm_request_audit.jsonl"],
    })
    write(docs / "llm_request_audit.summary.json", {
        "status": "complete", "n_requests": 1, "n_responses": 1,
        "guard_failures": 0, "n_transport_errors": 0,
    })
    (docs / "llm_request_audit.jsonl").write_text('{"event":"fixture"}\n')
    for year in (1, 2):
        write(checkpoint_dir / f"checkpoint_year_{year}.json", {
            "year": year, "phase": 5, "yearly_results": [{"year": y} for y in range(1, year + 1)],
            "ecosystem_data": {"agents": [
                {"id": "a", "type": "university", "resources": 100 - 10 * year, "is_active": True}]},
            "paper_tracker": {"papers": [
                {"id": "p", "status": "accept",
                 "review_history": [{"year": 1, "reviews": [{"score": 7}, {"score": 8}]}]}]},
            "citation_tracker": {"citations": {"p": ["q"]} if year == 2 else {}},
        })
    return run


def test_complete_numerical_report_and_evidence(tmp_path):
    root, output = tmp_path / "outputs", tmp_path / "report"
    run = completed_fixture(root)
    main(["--run", run, "--outputs-root", str(root), "--out-dir", str(output), "--require-activity"])
    report = json.loads((output / "report.json").read_text())
    record = report["runs"][0]
    assert record["total_reviews"] == 2
    assert [r["citation_edges"] for r in record["annual"]] == [0, 1]
    assert [r["total_resources"] for r in record["annual"]] == [90, 80]
    assert len(record["evidence_sha256"]) == 5
    assert {p.name for p in output.iterdir()} == {"report.json", "report.md", "yearly.csv"}


@pytest.mark.parametrize("fault", ["missing_year", "partial_phase", "failed_audit", "figure", "negative_resources"])
def test_incomplete_or_invalid_runs_cannot_pass_validation(tmp_path, fault):
    run = completed_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoints" / run / "checkpoint_year_1.json"
    if fault == "missing_year":
        checkpoint.unlink()
    elif fault == "failed_audit":
        path = tmp_path / "docs" / run / "llm_request_audit.summary.json"
        record = json.loads(path.read_text())
        record["guard_failures"] = 1
        write(path, record)
    elif fault == "figure":
        (tmp_path / "docs" / run / "chart.svg").touch()
    else:
        record = json.loads(checkpoint.read_text())
        if fault == "partial_phase":
            record["phase"] = 2
        else:
            record["ecosystem_data"]["agents"][0]["resources"] = -1
        write(checkpoint, record)
    with pytest.raises(ValueError):
        summarize_run(run, tmp_path, require_activity=True)
