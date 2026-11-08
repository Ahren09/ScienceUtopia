"""CPU-only falsification tests for the campaign and independent analysis."""
import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import utopia.analysis.project_cost as analysis
import utopia.experiments.project_cost as campaign
from utopia.experiments.scale_expansion import STAGES, build_command


def one_year():
    row = {"researcher_id": "a", "year": 1, "opening_balance": 100, "closing_balance": 90,
           "grant_income": 0, "industry_income": 0, "annual_research_cost": 10,
           "production_submission_cost": 0, "resubmission_cost": 0,
           "funding_application_cost": 0}
    event = {"event_id": "e", "year": 1, "researcher_id": "a",
             "category": "annual_research_cost", "balance_before": 100,
             "balance_after": 90, "requested_delta": -10, "delta": -10}
    return {"resource_ledger": {"years": {"1": {"a": row}}, "transactions": [event]},
            "agent_tracker": {"resources": {"a": [{"year": 0, "resources": 100},
                                                 {"year": 1, "resources": 90}]}},
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
    checkpoint["agent_tracker"]["resources"]["a"].append({"year": 2, "resources": 90})
    assert len(analysis.validate_ledger(checkpoint, 2, {"a"})) == 2
    second["opening_balance"] = 91
    with pytest.raises(ValueError, match="resource tracker|previous closing"):
        analysis.validate_ledger(checkpoint, 2, {"a"})


def test_coherent_omitted_debit_cannot_change_opening_anchor():
    checkpoint = one_year()
    checkpoint["resource_ledger"]["transactions"] = []
    row = checkpoint["resource_ledger"]["years"]["1"]["a"]
    row.update(opening_balance=90, annual_research_cost=0)
    with pytest.raises(ValueError, match="resource tracker"):
        analysis.validate_ledger(checkpoint, 1, {"a"})


def test_wrong_cell_seed_budget_year_and_config_are_rejected():
    args = dict(papers_per_project=2, seed=3, funding_budget_mode="fixed", disable_resubmission=False)
    config = {"annual_cost": 10}
    manifest = {"git_commit": "sha", "git_dirty_files": "", "experiment_id": "planned",
                "args": args, "resolved_config": config}
    checkpoint = {"year": 8, "phase": 5, "project_cost_policy": {"papers_per_project": 2},
                  "cost_experiment_binding": {"args": deepcopy(args), "git_commit": "sha"}}
    job = {"experiment_id": "planned", "years": 8, "resolved_config": config,
           "scientific_args": deepcopy(args),
           "command": ["python", "utopia/simulation.py", "--papers_per_project", "2",
                       "--seed", "3", "--funding_budget_mode", "fixed"]}
    analysis.validate_job_binding(manifest, checkpoint, job)
    manifest.update(git_commit="newer", git_dirty_files="local edits")
    analysis.validate_job_binding(manifest, checkpoint, job)
    for field, wrong in (("papers_per_project", 1), ("seed", 4), ("funding_budget_mode", "track"),
                         ("disable_resubmission", True)):
        altered_manifest, altered_ck = deepcopy(manifest), deepcopy(checkpoint)
        altered_manifest["args"][field] = wrong
        altered_ck["cost_experiment_binding"]["args"][field] = wrong
        with pytest.raises(ValueError):
            analysis.validate_job_binding(altered_manifest, altered_ck, job)
    for field, wrong in (("year", 7), ("phase", 2)):
        altered_ck = deepcopy(checkpoint)
        altered_ck[field] = wrong
        with pytest.raises(ValueError):
            analysis.validate_job_binding(manifest, altered_ck, job)
    altered_manifest = deepcopy(manifest)
    altered_manifest["resolved_config"]["annual_cost"] = 11
    with pytest.raises(ValueError):
        analysis.validate_job_binding(altered_manifest, checkpoint, job)


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


@pytest.fixture
def cost_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    args = SimpleNamespace(
        directory=tmp_path / "plan",
        python=Path(sys.executable),
        vllm_url=["http://localhost:8032/v1", "http://localhost:8033/v1"],
    )
    campaign.make_plan(args)
    return json.loads((args.directory / "plan.json").read_text())


def test_plan_commands_keep_native_ids_settings_and_isolated_caches(cost_plan):
    from utopia.arguments import parse_arguments

    assert len(cost_plan["jobs"]) == 24
    assert len({job["experiment_id"] for job in cost_plan["jobs"]}) == 24
    assert json.loads((Path(cost_plan["directory"]) / "plan.json").read_text()) == cost_plan
    for index, job in enumerate(cost_plan["jobs"]):
        assert job["command"][1:3] == ["-m", "utopia"]
        args = parse_arguments(job["command"][3:])
        assert args.experiment_id == job["experiment_id"]
        assert analysis.scientific_arguments(vars(args)) == job["scientific_args"]
        assert args.num_institutions * 5 == job["population"]
        assert args.num_years == job["years"]
        assert args.vllm_url == f"http://localhost:{8032 + index % 2}/v1"
        assert args.funding_panel_max_apps == 25
        assert args.log_resource_ledger and not args.disable_resubmission
        assert job["resolved_config"]["conference"]["resubmission_cost"] == 0
    commands = campaign.shell_commands(cost_plan, "costpilot")
    assert len(commands) == 4
    for command, job in zip(commands, campaign.stage_jobs(cost_plan, "costpilot")):
        tokens = shlex.split(command)
        assert tokens[:3] == ["env", "CUDA_VISIBLE_DEVICES=", f"PYTHONHASHSEED={job['seed']}"]
        assert tokens[3].endswith("/data/cache/" + job["experiment_id"])
        assert tokens[4:] == job["command"]


def test_full_commands_require_all_pilot_checkpoints_to_validate(cost_plan):
    with patch.object(campaign, "validate_run", side_effect=ValueError("Incomplete world")) as validate:
        with pytest.raises(ValueError, match="Incomplete world"):
            campaign.shell_commands(cost_plan, "costcontrol")
        assert validate.call_count == 1
    with patch.object(campaign, "validate_run") as validate:
        commands = campaign.shell_commands(cost_plan, "costcontrol")
        assert len(commands) == 20
        assert [call.args[1] for call in validate.call_args_list] == campaign.stage_jobs(cost_plan, "costpilot")


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "wrong_cell", "duplicate_id"))
def test_stage_rejects_incomplete_or_duplicate_worlds(cost_plan, mutation):
    jobs = cost_plan["jobs"]
    if mutation == "missing":
        jobs.pop(0)
    elif mutation == "duplicate":
        jobs.append(jobs[0])
    elif mutation == "wrong_cell":
        jobs[0]["cell"] = "unexpected"
    else:
        jobs[0]["experiment_id"] = jobs[1]["experiment_id"]
    with pytest.raises(ValueError, match="planned cells|world IDs"):
        campaign.shell_commands(cost_plan, "costpilot")


def test_shell_commands_quote_python_and_output_paths(cost_plan):
    job = cost_plan["jobs"][0]
    job["command"][0] = "/environment with spaces/python"
    job["experiment_id"] = "world with spaces"
    tokens = shlex.split(campaign.shell_commands(cost_plan, "costpilot")[0])
    assert tokens[3].endswith("/data/cache/world with spaces")
    assert tokens[4] == "/environment with spaces/python"
