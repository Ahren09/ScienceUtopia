"""CPU-only falsification tests for the campaign and independent analysis."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_project_cost as analysis
import run_project_cost_campaign as campaign
from run_scale_expansion import STAGES, build_command


def one_year():
    row = {"researcher_id": "a", "year": 1, "opening_balance": 100, "closing_balance": 90,
           "grant_income": 0, "industry_income": 0, "annual_research_cost": 10,
           "production_submission_cost": 0, "resubmission_cost": 0,
           "funding_application_cost": 0}
    event = {"event_id": "e", "year": 1, "researcher_id": "a",
             "category": "annual_research_cost", "balance_before": 100,
             "balance_after": 90, "requested_delta": -10, "delta": -10}
    return {"resource_ledger": {"years": {"1": {"a": row}}, "transactions": [event]},
            "ecosystem_data": {"agents": [{"id": "a", "type": "university", "resources": 90}]}}


def test_independent_ledger_detects_corruption_and_output_cost():
    original = one_year()
    assert len(analysis.validate_ledger(original, 1, {"a"})) == 1
    for mutation in ("silent_debit", "duplicate", "missing_founder", "output_cost"):
        checkpoint = deepcopy(original)
        if mutation == "silent_debit":
            checkpoint["ecosystem_data"]["agents"][0]["resources"] = 89
        elif mutation == "duplicate":
            checkpoint["resource_ledger"]["transactions"] *= 2
        elif mutation == "missing_founder":
            checkpoint["resource_ledger"]["years"]["1"] = {}
        else:
            checkpoint["resource_ledger"]["transactions"][0]["category"] = "resubmission_cost"
            row = checkpoint["resource_ledger"]["years"]["1"]["a"]
            row["annual_research_cost"], row["resubmission_cost"] = 0, 10
        with pytest.raises(ValueError):
            analysis.validate_ledger(checkpoint, 1, {"a"})


def test_independent_ledger_checks_year_continuity_and_inactive_agents():
    checkpoint = one_year()
    first = checkpoint["resource_ledger"]["years"]["1"]["a"]
    first["closing_active"] = False
    second = dict(first, year=2, opening_balance=90, closing_balance=90, annual_research_cost=0)
    checkpoint["resource_ledger"]["years"]["2"] = {"a": second}
    assert len(analysis.validate_ledger(checkpoint, 2, {"a"})) == 2
    second["opening_balance"] = 91
    with pytest.raises(ValueError, match="previous closing"):
        analysis.validate_ledger(checkpoint, 2, {"a"})


def test_five_seed_exact_test_and_paired_direction():
    summary = analysis.paired_summary([1, 2, 3, 4, 5])
    assert summary["mean_paired_difference"] == 3
    assert summary["exact_sign_flip_p"] == 0.0625
    opposite = analysis.paired_summary([-1, -2, -3, -4, -5])
    assert opposite["exact_sign_flip_p"] == summary["exact_sign_flip_p"]
    assert opposite["mean_paired_difference"] == -3


def test_cost_control_flags_do_not_disable_resubmission_or_add_production_debit():
    args = SimpleNamespace(population="university_only", num_institutions=60, num_years=8,
                           num_conferences=6, start_year=2016, batch_size=32,
                           funding_panel_max_apps=25, slots_frac=0.2,
                           reviewer_capacity=None, reviewer_matching="random",
                           legacy_scoring=False, always_rerun=False, vllm_url=["http://local"])
    for cell in STAGES["costcontrol"]:
        command = build_command(cell, 3, args)
        assert command[command.index("--production_cost_mode") + 1] == "per_paper"
        assert command[command.index("--resubmission_cost") + 1] == "0"
        assert "--log_resource_ledger" in command
        assert "--disable_resubmission" not in command
    legacy = build_command("S1R1", 3, args)
    assert "--production_cost_mode" not in legacy and "--resubmission_cost" not in legacy


def test_failed_pilot_never_launches_full_world(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}")
    pilot = {"stage": "costpilot", "experiment_id": "pilot"}
    full = {"stage": "costcontrol", "experiment_id": "full"}
    plan = {"directory": str(tmp_path), "git_commit": "sha", "jobs": [pilot, full]}
    runner = campaign.Campaign(plan, plan_path)
    runner.lock_root = tmp_path / "locks"
    runner.lock_root.mkdir()
    gpu = SimpleNamespace(index=1)
    runner.servers = [{"gpu": gpu, "record": {"status": "ready", "identity": {}},
                       "process": SimpleNamespace(poll=lambda: None), "client": None}]
    launched = []
    def start(server, job):
        launched.append(job["experiment_id"])
        runner.state["jobs"][job["experiment_id"]] = {"started": 0}
        server["client"] = (SimpleNamespace(poll=lambda: 2), job)
    with (patch.object(campaign, "runtime_check"), patch.object(runner, "recover"),
          patch.object(runner, "cleanup"), patch.object(runner, "save"),
          patch.object(campaign, "load_shared_probe_builder", return_value=(None, {})),
          patch.object(campaign, "git", side_effect=lambda *args: "sha" if args[0] == "rev-parse" else ""),
          patch.object(runner, "start_job", side_effect=start),
          patch.object(campaign.time, "sleep"), patch.object(campaign.time, "time", return_value=1)):
        with pytest.raises(RuntimeError, match="World exited 2"):
            runner.run()
    assert launched == ["pilot"]
