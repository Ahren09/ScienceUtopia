"""Validate and summarize completed public runs without loading a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from utopia.utils.data_utils import file_sha256, read_json, write_json_atomic


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_json(value):
    """Undefined estimates are null, never zero or a nonstandard JSON NaN."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def checkpoint_path(directory, year):
    path = Path(directory) / f"checkpoint_year_{year}.json"
    if not path.exists():
        path = path.with_suffix(".json.gz")
    require(path.is_file(), f"Missing year {year} checkpoint in {directory}")
    return path


def summarize_run(run_id, outputs_root="outputs", *, require_activity=False):
    root = Path(outputs_root).resolve()
    require(Path(run_id).name == run_id, "Use an experiment ID, not a path")
    docs, checkpoints = root / "docs" / run_id, root / "checkpoints" / run_id
    manifest_path = docs / "run_manifest.json"
    manifest = read_json(manifest_path)
    years = manifest["args"]["num_years"]
    require(manifest.get("status") == "complete" and manifest.get("years_completed") == years,
            f"Incomplete native run: {run_id}")
    require(manifest.get("experiment_id") == run_id, "Manifest experiment ID mismatch")
    evidence = {str(manifest_path): file_sha256(manifest_path)}
    annual = []
    final = None
    for year in range(1, years + 1):
        path = checkpoint_path(checkpoints, year)
        ck = read_json(path)
        require(ck["year"] == year and ck["phase"] == 5, f"Incomplete checkpoint: {path}")
        require(len(ck["yearly_results"]) == year, f"Missing annual results: {path}")
        evidence[str(path)] = file_sha256(path)
        agents = [a for a in ck["ecosystem_data"]["agents"]
                  if a.get("type") in ("university", "industry")]
        require(len({a["id"] for a in agents}) == len(agents), "Duplicate researcher IDs")
        require(all(math.isfinite(a["resources"]) and a["resources"] >= 0 for a in agents),
                "Invalid researcher resources")
        papers = ck["paper_tracker"]["papers"]
        attempts = [h for p in papers for h in p.get("review_history", []) if h["year"] == year]
        annual.append({
            "year": year, "researchers": len(agents),
            "active_researchers": sum(bool(a["is_active"]) for a in agents),
            "total_resources": math.fsum(a["resources"] for a in agents),
            "cumulative_unique_papers": len(papers),
            "cumulative_accepted_papers": sum(p["status"] == "accept" for p in papers),
            "submission_attempts": len(attempts),
            "reviews": sum(len(h.get("reviews", [])) for h in attempts),
            "citation_edges": sum(len(v) for v in ck["citation_tracker"]["citations"].values()),
        })
        final = ck
    ledger_rows = None
    if final.get('resource_ledger'):
        from utopia.analysis.project_cost import validate_ledger
        founders = [aid for aid, rows in final['agent_tracker']['resources'].items()
                    if any(row['year'] == 0 for row in rows)]
        if len(founders) == len(agents):
            ledger_rows = len(validate_ledger(final, years, founders, require_zero_output_costs=False))
    reviews = sum(row["reviews"] for row in annual)
    if require_activity:
        require(annual[-1]["cumulative_unique_papers"] > 0 and reviews > 0,
                "The run has no meaningful paper/review activity")
    audits = []
    audit_paths = manifest.get("request_audit_paths", [])
    if not audit_paths:
        audit_paths = [p.name for p in docs.glob("llm_request_audit*.summary.json")
                       if read_json(p).get("status") == "complete"]
    for name in audit_paths:
        path = docs / Path(name).name
        if path.suffix == ".jsonl":
            path = path.with_suffix(".summary.json")
        record = read_json(path)
        require(record["status"] == "complete" and record["n_requests"] == record["n_responses"]
                and record["guard_failures"] == 0 and record["n_transport_errors"] == 0,
                f"Failed request audit: {path}")
        raw = path.with_name(path.name.replace(".summary.json", ".jsonl"))
        require(raw.is_file() and raw.stat().st_size, f"Missing raw request audit: {raw}")
        evidence[str(path)], evidence[str(raw)] = file_sha256(path), file_sha256(raw)
        audits.append(record)
    if require_activity:
        require(audits, "The live validation requires a closed request audit")
    for directory in (docs, checkpoints, root / "logs" / run_id):
        forbidden = [p for p in directory.rglob("*") if
                     p.suffix.lower() in {".png", ".svg", ".pdf", ".jpg", ".jpeg", ".html", ".ipynb"}
                     or p.name in {"visual", "figures"}]
        require(not forbidden, f"Unexpected visualization files: {forbidden}")
    experiment_path = docs / "experiment_manifest.json"
    experiment = read_json(experiment_path) if experiment_path.exists() else {}
    if experiment:
        require(experiment.get("status") == "complete" and not experiment.get("initialization_only"),
                "The experiment is incomplete or initialization-only")
        evidence[str(experiment_path)] = file_sha256(experiment_path)
    return {
        "experiment_id": run_id, "status": "complete", "family": experiment.get("family", "simulation"),
        "cell": experiment.get("cell"), "seed": manifest["args"]["seed"],
        "model": manifest["model_id"], "git_commit": manifest.get("git_commit"),
        "num_years": years, "annual": annual, "total_reviews": reviews,
        "reconciled_resource_ledger_rows": ledger_rows,
        "dataset": manifest.get("dataset_identity"), "request_audits": audits,
        "evidence_sha256": evidence,
    }, manifest, experiment, final


def family_analysis(results, manifests, experiments, checkpoints, outputs_root, out_dir):
    """Apply the retained estimators to explicit new-run inputs."""
    families = {e.get("family") for e in experiments}
    if len(families) != 1 or None in families:
        return {"scope": "Individual simulation summaries"}
    family = next(iter(families))
    require(len({r["num_years"] for r in results}) == 1, "Different observation horizons")
    require(len({r["model"] for r in results}) == 1, "Different models")
    require(len({json.dumps(e["source_files_sha256"], sort_keys=True) for e in experiments}) == 1,
            "Different simulation source versions")
    root, output = Path(outputs_root).resolve(), Path(out_dir)
    by_seed = {}
    for result, native, experiment, checkpoint in zip(results, manifests, experiments, checkpoints):
        key = result["seed"]
        require(experiment["cell"] not in by_seed.setdefault(key, {}), "Duplicate seed/cell")
        by_seed[key][experiment["cell"]] = (result, native, experiment, checkpoint)
        if family in ('funding_feedback', 'influx_factorial', 'switching_propensity'):
            from utopia.funding.feedback import validate_sequential_evidence
            docs = root / 'docs' / result['experiment_id']
            compact = read_json(docs / 'funding_validation.summary.json')
            require(compact == experiment['funding_validation'], 'Funding audit summary changed')
            validate_sequential_evidence(docs, compact,
                manifest_summary=experiment['funding_sequential'],
                application_dir=root / 'checkpoints' / result['experiment_id'],
                num_years=result['num_years'])
    for cells in by_seed.values():
        require(len({json.dumps(row[2]['config'], sort_keys=True) for row in cells.values()}) == 1,
                'Paired cells have different configurations')
        require(len({row[2].get('ordered_documents_sha256') for row in cells.values()}) == 1,
                'Paired cells have different retrieval corpora')
    report = {"family": family, "world_replicates": len(by_seed),
              "scope": "New runs only; comparisons pair cells within a seed."}
    if family in ("scale_expansion", "project_cost"):
        from utopia.analysis.scale_expansion import compute_run_endpoints, aggregate, write_report
        endpoints = [compute_run_endpoints(str(root / "checkpoints" / r["experiment_id"]),
                                          r["num_years"], min(8, r["num_years"]),
                                          metadata={
                                              "cell": e["cell"], "seed": r["seed"],
                                              "k": m["args"]["papers_per_project"],
                                              "budget": m["args"]["funding_budget_mode"],
                                              "cap": m["args"]["reviewer_capacity"],
                                              "policy": m["args"]["review_policy"],
                                              "slots": m["args"]["acceptance_mode"] == "fixed_slots",
                                              "population": m["args"]["population_mode"]})
                     for r, m, e in zip(results, manifests, experiments)]
        if family == "project_cost":
            from utopia.analysis.project_cost import validate_ledger
            for ck, r in zip(checkpoints, results):
                founders = [aid for aid, rows in ck["agent_tracker"]["resources"].items()
                            if any(row["year"] == 0 for row in rows)]
                validate_ledger(ck, r["num_years"], founders)
        tables = aggregate(endpoints)
        write_report(str(output / family), *tables, min(8, results[0]["num_years"]))
        report["numerical_tables"] = str(output / family)
        if family == 'project_cost':
            from utopia.analysis.project_cost import paired_summary, PRIMARY_METRICS
            by_key = {(row['cell'], row['seed']): row for row in endpoints}
            seeds = sorted(by_seed)
            contrasts = []
            for metric in PRIMARY_METRICS:
                differences = {}
                for budget in (0, 1):
                    values = []
                    for seed in seeds:
                        require((f'S1R{budget}_costcontrol', seed) in by_key and
                                (f'S0R{budget}_costcontrol', seed) in by_key,
                                'Project cost requires all four cells per seed')
                        treatment = by_key[f'S1R{budget}_costcontrol', seed][metric]
                        control = by_key[f'S0R{budget}_costcontrol', seed][metric]
                        require(treatment is not None and control is not None, 'Undefined paired cost endpoint')
                        values.append(treatment - control)
                    differences[budget] = values
                for label, values in [('k2_minus_k1_track', differences[0]),
                                      ('k2_minus_k1_fixed', differences[1]),
                                      ('fixed_minus_track_interaction', [a-b for a,b in zip(differences[1], differences[0])])]:
                    estimate = paired_summary(values) if len(values) > 1 else {
                        'mean_paired_difference': values[0], 'seed_differences': values, 'n_seeds': 1,
                        'bootstrap_95_low': None, 'bootstrap_95_high': None, 'exact_sign_flip_p': None}
                    contrasts.append({'metric': metric, 'contrast': label, **estimate})
            adjusted = 0
            tested = [row for row in contrasts if row['exact_sign_flip_p'] is not None]
            for rank, row in enumerate(sorted(tested, key=lambda row: row['exact_sign_flip_p'])):
                adjusted = max(adjusted, min(1, (len(tested)-rank)*row['exact_sign_flip_p']))
                row['holm_p_six_primary_contrasts'] = adjusted
            report['paired_primary_contrasts'] = contrasts
    elif family == "funding_feedback":
        from utopia.funding.feedback import CELLS, PRIMARY_METRICS, reconcile_year
        from utopia.analysis.funding_feedback import paired_contrasts
        report["paired_contrasts"] = {}
        for seed, cells in by_seed.items():
            require(set(cells) == set(CELLS), "Funding feedback requires all four cells per seed")
            outcomes, worlds = {}, []
            for cell, (r, _, _, ck) in cells.items():
                docs = root / "docs" / r["experiment_id"]
                worlds.append(read_json(docs / "initial_world.json"))
                founders = [a["id"] for a in worlds[-1]["ecosystem"]["agents"]
                            if a.get("type") == "university"]
                for year in range(1, r["num_years"] + 1):
                    record = read_json(docs / f"mechanisms_year_{year}.json")
                    reconcile_year(record, founders)
                outcomes[cell] = record["metrics"]
            require(all(world == worlds[0] for world in worlds), "Unpaired initial worlds")
            report["paired_contrasts"][seed] = {
                metric: paired_contrasts({c: values[metric] for c, values in outcomes.items()})
                for metric in PRIMARY_METRICS}
    elif family == "influx_factorial":
        from utopia.analysis.influx_factorial import cell_endpoints, decompose
        from utopia.analysis.resubmission_stats import compute_stats
        report["paired_contrasts"] = {}
        for seed, cells in by_seed.items():
            require(set(cells) == set("ABCD"), "Influx requires A/B/C/D for each seed")
            values = {cell: cell_endpoints(compute_stats(
                str(root / "checkpoints" / row[0]["experiment_id"]), row[0]["num_years"]),
                row[0]["num_years"]) for cell, row in cells.items()}
            report["paired_contrasts"][seed] = decompose(values)
    elif family == "switching_propensity":
        from utopia.analysis.switching_propensity import normalized_entropy, factorial_contrasts
        report["paired_contrasts"], report["annual_entropy"] = {}, {}
        for seed, cells in by_seed.items():
            require(set(cells) == {"LN", "LF", "HN", "HF"}, "Switching requires LN/LF/HN/HF")
            require(len({row[2]["initial_choices_sha256"] for row in cells.values()}) == 1,
                    "Switching cells have different initialization records")
            values = {}
            for cell, (r, _, _, _) in cells.items():
                docs = root / "docs" / r["experiment_id"]
                entropy = []
                for year in range(1, r["num_years"] + 1):
                    record = read_json(docs / "years" / f"year_{year:02d}.json")
                    require(record["year"] == year, "Switching annual record mismatch")
                    counts = Counter(e["chosen_topic"] for e in record["direction_events"])
                    entropy.append(normalized_entropy(counts))
                report["annual_entropy"][r["experiment_id"]] = entropy
                values[cell] = math.fsum((a + b) / 2 for a, b in zip(entropy, entropy[1:]))
            contrasts = factorial_contrasts(values)
            contrasts["positive_relative_interaction"] = contrasts.pop("positive_relative_interaction_in_seed42")
            contrasts["near_history_propensity_benefit"] = contrasts.pop("near_history_propensity_benefit_in_seed42")
            report["paired_contrasts"][seed] = contrasts
    else:
        report["scope"] = ("Individual summaries. Use the dedicated exploration, resource-size, "
                           "funding-cutoff, or replay analysis command for its estimators.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", required=True, help="Completed simulation experiment IDs.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/docs/release_report"))
    parser.add_argument("--require-activity", action="store_true", help="Require real papers, reviews and closed request audits.")
    parser.add_argument("--family-analysis", action="store_true", help="Require and compare complete paired experiment cells.")
    args = parser.parse_args(argv)
    rows = [summarize_run(run, args.outputs_root, require_activity=args.require_activity) for run in args.run]
    report = {"status": "complete", "runs": [r[0] for r in rows]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.family_analysis:
        report["analysis"] = family_analysis(*zip(*rows), args.outputs_root, args.out_dir)
    write_json_atomic(args.out_dir / "report.json", finite_json(report), indent=2, allow_nan=False)
    records = [{"experiment_id": r[0]["experiment_id"], **year} for r in rows for year in r[0]["annual"]]
    with (args.out_dir / "yearly.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# ScienceUtopia numerical report", "",
             "| Run | Years | Papers | Reviews |", "|---|---:|---:|---:|"]
    lines += [f"| {r['experiment_id']} | {r['num_years']} | "
              f"{r['annual'][-1]['cumulative_unique_papers']} | {r['total_reviews']} |"
              for r, *_ in rows]
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(args.out_dir / "report.json")}))


if __name__ == "__main__":
    main()
