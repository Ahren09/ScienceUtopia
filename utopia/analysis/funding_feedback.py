"""Paired funding-feedback analysis for a fixed simulation world."""
from __future__ import annotations

from utopia.runtime.historical import historical_source_path

from utopia.runtime.historical import identifier_matches, historical_artifact
from utopia.utils.data_utils import file_sha256 as file_hash
import json
import math
from pathlib import Path
from utopia.funding.feedback import CELLS, FOUNDER_COUNT, FUNDING_BASELINE, FUNDING_OUTPUT_REPRESENTATION, FUNDING_SELECTION_PROTOCOL, PRIMARY_METRICS, PROTOCOL, SEEDS, SEQUENTIAL_AUDIT_FILE, SEQUENTIAL_PROTOCOL_PATH, SEQUENTIAL_SUMMARY_FILE, YEARS, experiment_id, funding_ledger_evidence, reconcile_year, validate_sequential_evidence
from utopia.runtime.provenance import ROOT, digest, validate_request_audit_summary, write_json


def paired_contrasts(values):
    """Effects of disabling each mechanism, plus the difference in differences."""
    if set(values) != set(CELLS):
        raise ValueError("A paired seed must contain all four cells")
    if not all(v is None or (isinstance(v, (int, float)) and math.isfinite(v))
               for v in values.values()):
        raise ValueError("Outcomes must be finite numbers or explicitly undefined")
    coefficients = {
        "disable_publication_at_F1": {"P0F1": 1, "P1F1": -1},
        "disable_publication_at_F0": {"P0F0": 1, "P1F0": -1},
        "disable_resources_at_P1": {"P1F0": 1, "P1F1": -1},
        "disable_resources_at_P0": {"P0F0": 1, "P0F1": -1},
        "interaction": {"P0F0": 1, "P0F1": -1, "P1F0": -1, "P1F1": 1},
    }
    return {name: (sum(values[cell] * weight for cell, weight in weights.items())
                   if all(values[cell] is not None for cell in weights) else None)
            for name, weights in coefficients.items()}


def analyze(docs_root=ROOT / "outputs/docs", *, trusted_source_files=None):
    """Four conditional contrasts within one world; no sampling uncertainty claim."""
    runs = {}
    boundary = {}
    for seed in SEEDS:
        for cell in CELLS:
            directory = historical_artifact(Path(docs_root) / experiment_id(cell, seed))
            manifest = json.loads((directory / "mechanism_manifest.json").read_text())
            native = json.loads((directory / "run_manifest.json").read_text())
            if native.get("status") != "complete" or native.get("years_completed") != YEARS:
                raise ValueError("Native simulator manifest is incomplete")
            if (manifest["status"] != "complete" or not identifier_matches(manifest['protocol'], PROTOCOL)
                    or manifest["cell"] != cell or manifest["seed"] != seed):
                raise ValueError(f"Incomplete or misidentified run: {directory}")
            if manifest.get("funding_baseline") != FUNDING_BASELINE:
                raise ValueError(f"Panelized funding baseline mismatch: {directory}")
            if (manifest.get("funding_selection_protocol") != FUNDING_SELECTION_PROTOCOL
                    or manifest.get("sequential_audit_valid") is not True
                    or set(manifest.get("sequential_source_binding", {}))
                    not in ({"utopia/funding/sequential.py", SEQUENTIAL_PROTOCOL_PATH},
                            {historical_source_path("utopia/funding/sequential.py"),
                             historical_source_path(SEQUENTIAL_PROTOCOL_PATH)})):
                raise ValueError(f"Missing sequential funding protocol binding: {directory}")
            summary = json.loads((directory / "llm_request_audit.summary.json").read_text())
            funding_summary = json.loads((directory / "funding_validation.summary.json").read_text())
            validate_request_audit_summary(summary)
            if (manifest.get("request_audit") != summary
                    or manifest.get("request_audit_valid") is not True
                    or manifest.get("funding_validation", {}).get("completion_allowed") is not True
                    or funding_summary.get("output_representation") != FUNDING_OUTPUT_REPRESENTATION
                    or manifest.get("funding_validation") != funding_summary
                    or not (directory / "llm_request_audit.jsonl").stat().st_size
                    or not (directory / "funding_validation.jsonl").is_file()):
                raise ValueError(f"Required request/funding audit mismatch: {directory}")
            validate_sequential_evidence(
                directory, funding_summary, manifest_summary=manifest["funding_sequential"],
                source_binding=manifest["sequential_source_binding"],
                application_dir=manifest["args"]["output_dir"], trusted_source_files=trusted_source_files)
            if (manifest.get("funding_application_evidence_valid") is not True
                    or manifest.get("funding_application_evidence")
                    != funding_ledger_evidence(manifest["args"]["output_dir"], YEARS)):
                raise ValueError("Funding application bytes/absent-year evidence differs from manifest")
            if manifest.get("funding_sequential_audits") != {
                    name: file_hash(directory / name)
                    for name in (SEQUENTIAL_AUDIT_FILE, SEQUENTIAL_SUMMARY_FILE)}:
                raise ValueError("Sequential raw/summary evidence differs from manifest")
            years = {}
            founders = manifest["founder_ids"]
            if len(founders) != FOUNDER_COUNT or len(set(founders)) != FOUNDER_COUNT:
                raise ValueError("Manifest founder cohort is not the fixed population")
            boundary_years = {}
            affected = set()
            for year in range(1, 7):
                record = json.loads((directory / f"mechanisms_year_{year}.json").read_text())
                if (record["year"] != year or record["cell"] != cell
                        or record["metrics"]["founder_count"] != FOUNDER_COUNT):
                    raise ValueError(f"Year or founder cohort mismatch: {directory}")
                reconcile_year(record, founders)
                years[year] = record["metrics"]
                audit = record["legacy_zero_boundary"]
                boundary_years[year] = audit
                affected.update(audit["active_zero_agent_ids"])
            count = sum(a["active_zero_count"] for a in boundary_years.values())
            boundary[cell] = {
                "annual_snapshots": boundary_years,
                "active_zero_founder_year_snapshots": count,
                "founder_year_denominator": FOUNDER_COUNT * YEARS,
                "active_zero_fraction_of_founder_years": count / (FOUNDER_COUNT * YEARS),
                "unique_affected_founders": len(affected),
                "scope": "Snapshot prevalence; not the number of exact-zero debit events.",
            }
            runs[seed, cell] = (manifest, years)
    for seed in SEEDS:
        for key in ("initial_world_hash", "corpus_hash"):
            hashes = [runs[seed, cell][0].get(key) for cell in CELLS]
            if not all(hashes) or len(set(hashes)) != 1:
                raise ValueError(f"Paired {key} mismatch for seed {seed}")
    for key in ("source_hash", "scientific_config_hash", "effective_args_hash", "model",
                "sequential_source_binding"):
        if len({digest(r[0][key]) for r in runs.values()}) != 1:
            raise ValueError(f"Protocol mismatch: {key}")
    if len({tuple(m["founder_ids"]) for m, _ in runs.values()}) != 1:
        raise ValueError("Founder identities differ across cells")
    runtime_hashes = {m["server_provenance"]["runtime_hash"] for m, _ in runs.values()}
    if len(runtime_hashes) != 1:
        raise ValueError("Server runtime differs across cells")
    contrasts = {
        metric: paired_contrasts({cell: runs[42, cell][1][6][metric] for cell in CELLS})
        for metric in PRIMARY_METRICS
    }
    output = {
        "protocol": PROTOCOL, "year6_paired_contrasts": contrasts,
        "shared_seed": 42, "independent_worlds": 1,
        "world_standard_error": None, "world_confidence_interval": None,
        "funding_baseline": FUNDING_BASELINE,
        "funding_selection_protocol": FUNDING_SELECTION_PROTOCOL,
        "sequential_funding_audits": {
            cell: {"summary": runs[42, cell][0]["funding_sequential"],
                   "sha256": runs[42, cell][0]["funding_sequential_audits"]}
            for cell in CELLS},
        "funding_application_evidence": {
            cell: runs[42, cell][0]["funding_application_evidence"] for cell in CELLS},
        "legacy_zero_boundary": boundary,
        "annual_metrics": {f"{s}/{c}": years for (s, c), (_, years) in runs.items()},
        "fallback_audits": {f"{s}/{c}": m["fallback_audit"]
                            for (s, c), (m, _) in runs.items()},
        "interpretation": (
            "Conditional four-cell contrasts for one initialized world (seed 42). "
            "No seed SD, confidence interval, population-level inference or equivalence claim. "
            "Researchers are not independent replicates. Record-display ablation and "
            "award-credit removal do not decompose all feedback. All contrasts are "
            "conditional on stock cap25 panel competition and per-panel quota rounding, "
            "not global program ranking. The citation measure is "
            "six-year attributed citation yield with full author counting, not age-standardized. "
            "Inspect fallback and exact-zero audits. All-zero Ginis remain undefined."
        ),
    }
    write_json(Path(docs_root) / "funding_feedback_v6_analysis" / "paired_effects.json", output)
    return output


def main(argv=None):
    import sys
    from utopia.analysis.release import main as report_main
    return report_main([*(sys.argv[1:] if argv is None else argv), '--family-analysis'])


if __name__ == '__main__':
    main()
