"""Plan and validate the direct-output-cost control using native simulator commands.

Servers are supplied explicitly. Run the printed pilot commands in a shell, then
request full-scale commands after the pilot checkpoints pass ledger validation.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from utopia.analysis.project_cost import scientific_arguments, validate_run
from utopia.experiments.scale_expansion import STAGES, build_command, experiment_id_for
from utopia.utils.data_utils import write_json
from utopia.utils.paths import project_root

ROOT = project_root(__file__)
PROTOCOL = {
    "question": "Does k=2 versus k=1 change allocation and review workload when direct production "
                "and resubmission fees are both zero?",
    "baseline_commit": "2bb7e6917418f2722b2217cc5453188f51734662",
    "historical_commit": "9ee31940c33f6c01a245f132ebcb489992921b47",
    "cost_audit": "Neither baseline nor historical execution debits the configured paper "
                  "production fee. Keep that effective zero, waive resubmission fees in all "
                  "new cells, and retain resubmission and review decisions.",
    "seeds": [1, 2, 3, 4, 5], "papers_per_project": [1, 2], "budget_modes": ["track", "fixed"],
    "population": 300, "years": 8, "institutions": 60, "researchers_per_institution": 5,
    "start_year": 2016, "conferences": 6, "funding_panel_max_apps": 25,
    "fixed_budget_slots_fraction": 0.2, "review_score_mode": "float",
    "acceptance_tiebreak": "seeded",
    "primary_endpoints": ["Share of original founders receiving any grant by year 8",
                          "Share of all reviews spent on resubmissions"],
    "survival_invariant": "Initial endowment 100 minus at most 8*10 research costs leaves "
                          "20, above the exit threshold. With zero direct and application fees "
                          "and nonnegative income, every founder must remain active. This is "
                          "a structural invariant, not an empirical test of no survival effect.",
    "primary_contrasts": ["k2-k1 under track", "k2-k1 under fixed",
                          "(k2-k1 under fixed) - (k2-k1 under track)"],
    "inference": "Pair all comparisons by seed. Report individual seed effects and descriptive "
                 "paired bootstrap intervals. Exact paired sign-flip diagnostics have minimum "
                 "two-sided p=0.0625 with five seeds. Holm adjustment across six primary contrasts. "
                 "Other endpoints are exploratory. Do not pool historical runs.",
    "scope": "Direct-output-cost sensitivity. Annual research costs and research duration follow "
             "the original rules. Realized total career spending is not held equal.",
    "pilot": {"seeds": [0], "population": 25, "years": 2, "cells": 4},
    "gates": "Pilot final checkpoints, founder rosters and complete resource ledgers must pass "
             "before full-scale commands are printed. Run shell commands with failure propagation "
             "and resume from the native yearly checkpoints.",
}


def make_plan(args):
    from utopia.arguments import parse_arguments
    from utopia.config import SIMULATION_CONFIG
    from utopia.funding.accounting import resolve_cost_policy

    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for stage, seeds, institutions, years in (
            ("costpilot", [0], 5, 2), ("costcontrol", PROTOCOL["seeds"], 60, 8)):
        launch_args = SimpleNamespace(
            population="university_only", num_institutions=institutions, num_years=years,
            num_conferences=6, start_year=2016, batch_size=32, funding_panel_max_apps=25,
            slots_frac=0.2, reviewer_capacity=None, reviewer_matching="random",
            legacy_scoring=False, always_rerun=False, vllm_url=args.vllm_url,
        )
        for seed in seeds:
            for cell in STAGES[stage]:
                endpoint = args.vllm_url[len(jobs) % len(args.vllm_url)]
                command = build_command(cell, seed, launch_args, vllm_url=endpoint)
                command[0] = str(args.python)
                resolved_args = parse_arguments(command[3:])
                resolved_config = deepcopy(SIMULATION_CONFIG)
                policy = resolve_cost_policy(resolved_args, resolved_config)
                resolved_config["project_cost_accounting"] = policy
                resolved_config["conference"]["resubmission_cost"] = policy["resubmission_cost"]
                jobs.append({"stage": stage, "cell": cell, "seed": seed,
                             "years": years, "population": institutions * 5,
                             "experiment_id": experiment_id_for(command), "command": command,
                             "scientific_args": scientific_arguments(vars(resolved_args)),
                             "resolved_config": resolved_config})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    plan = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "directory": str(directory), "git_commit": commit, "protocol": PROTOCOL,
            "seeds": PROTOCOL["seeds"], "jobs": jobs}
    path = directory / "plan.json"
    write_json(plan, path, mode="x", streaming=False, trailing_newline=True, allow_nan=False)
    return plan


def stage_jobs(plan, stage):
    jobs = [job for job in plan["jobs"] if job["stage"] == stage]
    seeds = PROTOCOL["pilot"]["seeds"] if stage == "costpilot" else plan["seeds"]
    expected = {(cell, seed) for cell in STAGES[stage] for seed in seeds}
    if len(jobs) != len(expected) or {(job["cell"], job["seed"]) for job in jobs} != expected:
        raise ValueError(f"Incomplete or duplicate planned cells: {stage}")
    if len({job["experiment_id"] for job in jobs}) != len(jobs):
        raise ValueError(f"Duplicate world IDs: {stage}")
    return jobs


def validate_stage(plan, stage):
    for job in stage_jobs(plan, stage):
        validate_run(ROOT, job)


def shell_commands(plan, stage):
    if stage == "costcontrol":
        validate_stage(plan, "costpilot")
    return [shlex.join(["env", "CUDA_VISIBLE_DEVICES=", f"PYTHONHASHSEED={job['seed']}",
                        f"UTOPIA_DATA_CACHE_DIR={ROOT / 'data/cache' / job['experiment_id']}",
                        *job["command"]]) for job in stage_jobs(plan, stage)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--python", type=Path, default=Path(sys.executable))
    prepare.add_argument("--vllm-url", nargs="+", required=True)
    prepare.add_argument("--directory", type=Path, required=True)
    for name in ("commands", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--stage", choices=("costpilot", "costcontrol"), required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
        print(args.directory.resolve() / "plan.json")
        return
    plan = json.loads(args.plan.read_text())
    if args.command == "commands":
        print("\n".join(shell_commands(plan, args.stage)))
    else:
        validate_stage(plan, args.stage)
        print(f"Validated {args.stage}")


if __name__ == "__main__":
    main()
