"""Validate and summarize the prospective direct-output-cost control.

Worlds, not agents, are the replication units. All contrasts pair identical
seeds. Historical runs are never pooled into this experiment.
"""
from __future__ import annotations

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import json_sha256

from utopia.utils.data_utils import read_json

from utopia.utils.paths import project_root

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics

ROOT = project_root(__file__)
COSTS = ("annual_research_cost", "production_submission_cost",
         "resubmission_cost", "funding_application_cost")
INCOMES = ("grant_income", "industry_income")
PRIMARY_METRICS = ("X3_award_coverage", "P2_resub_review_share")


def scientific_arguments(args):
    transport = {"vllm_url", "output_dir", "log_dir", "checkpoint_dir", "visual_dir",
                 "docs_dir", "data_cache_dir", "always_rerun", "verbose"}
    return {key: value for key, value in args.items() if key not in transport}


def expected_flags(command):
    result = {}
    i = 2
    while i < len(command):
        flag = command[i]
        require(flag.startswith("--"), "Malformed planned command")
        key = flag[2:].replace("-", "_")
        if i + 1 < len(command) and not command[i + 1].startswith("--"):
            result[key], i = command[i + 1], i + 2
        else:
            result[key], i = True, i + 1
    return result


def validate_job_binding(manifest, checkpoint, job, commit):
    require(manifest.get("git_commit") == commit, "Code revision mismatch")
    require(not manifest.get("git_dirty_files"), "Run started from modified source")
    require(manifest.get("experiment_id") == job["experiment_id"], "Manifest belongs to another world")
    binding = checkpoint["cost_experiment_binding"]
    require(binding["git_commit"] == commit, "Checkpoint code revision mismatch")
    require(scientific_arguments(binding["args"]) == scientific_arguments(manifest["args"]),
            "Checkpoint and manifest identify different experiments")
    require(scientific_arguments(manifest["args"]) == job["scientific_args"],
            "Parsed scientific arguments differ from the frozen plan")
    require(checkpoint["year"] == job["years"] and checkpoint["phase"] == 5,
            "Wrong checkpoint year or phase")
    for key, expected in expected_flags(job["command"]).items():
        if key == "vllm_url":
            continue
        actual = manifest["args"].get(key)
        if isinstance(actual, bool):
            require(actual is expected, f"Planned flag mismatch: {key}")
        elif isinstance(actual, (int, float)):
            require(actual == float(expected), f"Planned numeric argument mismatch: {key}")
        else:
            require(actual == expected, f"Planned argument mismatch: {key}")
    require(checkpoint["project_cost_policy"]["papers_per_project"] ==
            manifest["args"]["papers_per_project"], "Checkpoint has the wrong output treatment")
    require(manifest["resolved_config"] == job["resolved_config"], "Resolved configuration changed")
    digest = json_sha256(job["resolved_config"], sort_keys=True)
    require(manifest["scientific_config_hash"] == digest, "Scientific configuration hash mismatch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_ledger(checkpoint, years, founders, *, require_zero_output_costs=True):
    """Independent reconstruction from raw events, without production helpers."""
    ledger = checkpoint["resource_ledger"]
    require(set(ledger["years"]) == {str(y) for y in range(1, years + 1)},
            "Missing or extra accounting years")
    agents = {a["id"]: a for a in checkpoint["ecosystem_data"]["agents"]
              if a.get("type") in ("university", "industry")}
    tracker = checkpoint["agent_tracker"]["resources"]
    anchors = {}
    for aid in founders:
        records = tracker[aid]
        require(len({r["year"] for r in records}) == len(records), "Duplicate resource tracker year")
        anchors[aid] = {r["year"]: r["resources"] for r in records}
        require(set(anchors[aid]) == set(range(years + 1)), "Missing resource tracker anchor")
    events = {}
    seen = set()
    for event in ledger["transactions"]:
        require(event["event_id"] not in seen, "Duplicate transaction")
        seen.add(event["event_id"])
        require(event["category"] in COSTS + INCOMES, "Unknown accounting category")
        require(event["year"] in range(1, years + 1), "Transaction outside experiment")
        require(event["researcher_id"] in founders, "Transaction for nonfounder")
        events.setdefault((event["year"], event["researcher_id"]), []).append(event)
    rows, previous = [], {}
    for year in range(1, years + 1):
        balances = ledger["years"][str(year)]
        require(set(balances) == set(founders), "Incomplete founder accounting")
        for aid, row in sorted(balances.items()):
            require(row["closing_balance"] is not None, "Open accounting year")
            current = row["opening_balance"]
            require(math.isclose(current, anchors[aid][year - 1], abs_tol=1e-9, rel_tol=0),
                    "Opening balance disagrees with independent resource tracker")
            if year > 1:
                require(math.isclose(current, previous[aid], abs_tol=1e-9, rel_tol=0),
                        "Opening balance does not match previous closing balance")
            amounts = {key: 0.0 for key in COSTS + INCOMES}
            for event in events.get((year, aid), []):
                require(math.isclose(event["balance_before"], current, abs_tol=1e-9, rel_tol=0),
                        "Broken event balance chain")
                delta = event["delta"]
                if require_zero_output_costs and event["category"] in ("production_submission_cost", "resubmission_cost"):
                    require(event["requested_delta"] == 0, "Nonzero requested direct output fee")
                require((delta >= 0 if event["category"] in INCOMES else delta <= 0),
                        "Wrong accounting sign")
                expected = max(0, current + event["requested_delta"])
                require(math.isclose(expected, event["balance_after"], abs_tol=1e-9, rel_tol=0),
                        "Invalid clipped debit/credit")
                require(math.isclose(delta, event["balance_after"] - current,
                                     abs_tol=1e-9, rel_tol=0), "Incorrect recorded delta")
                amounts[event["category"]] += delta if event["category"] in INCOMES else -delta
                current = event["balance_after"]
            require(math.isclose(current, row["closing_balance"], abs_tol=1e-9, rel_tol=0),
                    "Unexplained closing balance")
            require(math.isclose(current, anchors[aid][year], abs_tol=1e-9, rel_tol=0),
                    "Closing balance disagrees with independent resource tracker")
            for key, value in amounts.items():
                require(math.isclose(value, row[key], abs_tol=1e-9, rel_tol=0),
                        f"Incorrect category total: {key}")
            if require_zero_output_costs:
                require(amounts["production_submission_cost"] == amounts["resubmission_cost"] == 0,
                        "Direct output cost is nonzero in a cost-control run")
            if year == years:
                require(math.isclose(current, agents[aid]["resources"], abs_tol=1e-9, rel_tol=0),
                        "Final agent resources disagree with ledger")
            previous[aid] = current
            rows.append(dict(row))
    return rows


def validate_run(root, job, commit):
    exp = job["experiment_id"]
    manifest = json.loads((root / "outputs/docs" / exp / "run_manifest.json").read_text())
    require(manifest["status"] == "complete", f"Incomplete world: {exp}")
    path = root / "outputs/checkpoints" / exp / f"checkpoint_year_{job['years']}.json"
    checkpoint = read_json(path if path.exists() else str(path) + ".gz")
    validate_job_binding(manifest, checkpoint, job, commit)
    policy = checkpoint["project_cost_policy"]
    require(policy["production_cost_mode"] == "per_paper"
            and policy["legacy_effective_per_paper_cost"] == 0
            and policy["resubmission_cost"] == 0
            and policy["log_resource_ledger"], "Unexpected cost policy")
    founders = {aid for aid, history in checkpoint["agent_tracker"]["resources"].items()
                if any(record["year"] == 0 for record in history)}
    require(len(founders) == job["population"], "Founder count mismatch")
    rows = validate_ledger(checkpoint, job["years"], founders)
    require(all(row["funding_application_cost"] == 0 for row in rows), "Unexpected application fee")
    require(all(row["opening_balance"] == 100 for row in rows if row["year"] == 1),
            "Unexpected initial endowment")
    require(all(row["closing_active"] and row["closing_balance"] >= 100 - 10 * row["year"]
                for row in rows), "Zero-fee survival bound violated")
    return checkpoint, rows


def paired_summary(values):
    values = list(values)
    n = len(values)
    require(n >= 2 and all(math.isfinite(x) for x in values), "Insufficient finite paired effects")
    mean = statistics.mean(values)
    # Exact paired sign-flip diagnostic. With five worlds, two-sided p >= 1/16.
    observed = abs(mean)
    permuted = [abs(statistics.mean(v * s for v, s in zip(values, signs)))
                for signs in itertools.product((-1, 1), repeat=n)]
    p = sum(x >= observed - 1e-12 for x in permuted) / len(permuted)
    rng = random.Random(20260924)
    boot = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(10000))
    return {"mean_paired_difference": mean, "seed_differences": values, "n_seeds": n,
            "bootstrap_95_low": boot[249], "bootstrap_95_high": boot[9749],
            "exact_sign_flip_p": p}


def analyze(root, plan, out):
    from utopia.analysis.scale_expansion import compute_run_endpoints
    runs, balances, yearly, blueprints = [], [], [], {}
    jobs = [job for job in plan["jobs"] if job["stage"] == "costcontrol"]
    require(len(jobs) == 4 * len(plan["seeds"]), "Incomplete planned factorial")
    for job in jobs:
        checkpoint, rows = validate_run(root, job, plan["git_commit"])
        blueprint = {
            a["id"]: {k: a.get(k) for k in ("type", "name", "university_name", "expertise",
                                            "exploration_strategy")}
            for a in checkpoint["ecosystem_data"]["agents"] if a.get("type") == "university"
        }
        if job["seed"] in blueprints:
            require(blueprint == blueprints[job["seed"]], "Paired founder blueprint changed")
        else:
            blueprints[job["seed"]] = blueprint
        run = compute_run_endpoints(str(root / "outputs/checkpoints" / job["experiment_id"]),
                                    job["years"])
        run.update(cell=job["cell"], seed=job["seed"])
        for key in COSTS + INCOMES:
            run[key] = sum(row[key] for row in rows) / job["population"]
        balances.extend(dict(row, experiment_id=job["experiment_id"],
                             cell=job["cell"], seed=job["seed"]) for row in rows)
        yearly.extend(dict(row, cell=job["cell"], seed=job["seed"]) for row in run["yearly_checks"])
        runs.append(run)
    by_key = {(run["cell"], run["seed"]): run for run in runs}
    require(len(by_key) == len(jobs), "Duplicate world in analysis")
    metrics = ("P1_survival_final", "P2_resub_review_share", "P2_attempts_per_paper",
               "P3_award_gini", "X3_award_coverage", "n_accepted", "total_reviews",
               *COSTS, *INCOMES)
    contrasts = []
    for metric in metrics:
        differences = {}
        for budget in (0, 1):
            values = []
            for seed in plan["seeds"]:
                treatment = by_key[(f"S1R{budget}_costcontrol", seed)][metric]
                control = by_key[(f"S0R{budget}_costcontrol", seed)][metric]
                require(treatment is not None and control is not None,
                        f"Missing endpoint in planned pair: {metric}/{seed}")
                values.append(treatment - control)
            differences[budget] = values
            contrasts.append({"metric": metric, "contrast": f"k2_minus_k1_{'fixed' if budget else 'track'}",
                              **paired_summary(values)})
        contrasts.append({"metric": metric, "contrast": "fixed_minus_track_interaction",
                          **paired_summary(a - b for a, b in zip(differences[1], differences[0]))})
    primary = [row for row in contrasts if row["metric"] in PRIMARY_METRICS]
    adjusted = 0
    for rank, row in enumerate(sorted(primary, key=lambda r: r["exact_sign_flip_p"])):
        adjusted = max(adjusted, min(1, (len(primary) - rank) * row["exact_sign_flip_p"]))
        row["holm_p_six_primary_contrasts"] = adjusted
    out.mkdir(parents=True, exist_ok=True)
    excluded = {"survival_curve", "accepted_topic_lists", "yearly_checks"}
    scalar_runs = [{k: v for k, v in r.items() if k not in excluded} for r in runs]
    for name, records in (("worlds.csv", scalar_runs), ("balances.csv", balances),
                          ("yearly.csv", yearly), ("paired_contrasts.csv", contrasts)):
        keys = sorted(set().union(*(row.keys() for row in records)))
        with (out / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    report = {"status": "complete", "git_commit": plan["git_commit"], "seeds": plan["seeds"],
              "worlds": len(jobs), "ledger_validated": True, "primary_contrasts": primary,
              "scope": "Zero direct production and resubmission fees. Annual research costs remain "
                       "endogenous to project choices and active years. Not equal realized career spending.",
              "survival_invariant": "Initial resources 100 minus at most 8*10 research costs leave "
                                    "at least 20, above the exit threshold. Survival=1 is a structural "
                                    "invariant of this control, not evidence of a causal null.",
              "inference": "Five paired seeds provide a sensitivity check. Bootstrap intervals are "
                           "descriptive, and exact two-sided sign-flip p-values cannot be below 0.0625. "
                           "Historical outcomes are excluded. Secondary endpoints are exploratory."}
    write_json_file(report, out / 'report.json', trailing_newline=True, streaming=False, indent=2, allow_nan=False)
    text = ["# Direct-output-cost control", "", report["scope"], "", report["inference"], "",
            report["survival_invariant"], "",
            "| Endpoint | Contrast | Mean difference | Bootstrap 95% interval | Exact p |",
            "| --- | --- | ---: | ---: | ---: |"]
    for row in primary:
        text.append(f"| {row['metric']} | {row['contrast']} | {row['mean_paired_difference']:.4f} | "
                    f"[{row['bootstrap_95_low']:.4f}, {row['bootstrap_95_high']:.4f}] | "
                    f"{row['exact_sign_flip_p']:.4f} |")
    (out / "report.md").write_text("\n".join(text) + "\n")
    return report




def main(argv=None):
    import sys
    from utopia.analysis.release import main as report_main
    return report_main([*(sys.argv[1:] if argv is None else argv), '--family-analysis'])


if __name__ == '__main__':
    main()
