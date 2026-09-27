"""Compute corpus-level dataset statistics across the simulation worlds reported in the paper.

For every world listed in a roster JSON file, this script loads the final-year
checkpoint (checkpoint_year_<max>.json or .json.gz, searched one level deep for
nested layouts) and counts the generated artifacts of the run:

  * researchers (agents, excluding funding-agency agents) and distinct
    institutions (university_name for university agents, company_name for
    industry agents)
  * simulated years
  * submitted papers (unique manuscripts in paper_tracker)
  * submission events / accept-reject decisions (one per review_history round;
    resubmissions of the same manuscript count as separate decisions)
  * individual peer reviews with written justifications
  * agent memory/thought entries (memory_bank is append-only and serialized in
    full, so the final checkpoint is cumulative), including LLM research-direction
    selections with written rationales
  * citation edges (citation_tracker references; stored as unique pairs)
  * funding applications and winners (yearly_results ecosystem_metrics),
    per-application JSONL logs where present, and per-agent award records

Counting recipe notes (verified against the checkpoint format and the
simulation source):
  * conference_system.conferences holds only the CURRENT simulated year
    (reset_all_for_new_year clears decisions/reviews at year start), so
    reviews/decisions are counted from paper_tracker review_history, which is
    cumulative and includes every venue attempt of rejected and resubmitted
    papers. The two sources hold the same review objects, so only one is used.
  * Legacy pre-fix worlds (review_history overwritten on resubmission) must
    instead union the per-year conference snapshots; list them in
    LEGACY_WORLDS. All paper-reported worlds are modern.
  * Integrity check: in solo-authored worlds the number of *_reviews_received
    memory entries equals the number of decisions; mismatches are reported.
  * Replay-style entries (no checkpoints, reviews.parquet instead) are counted
    as replayed reviews only.

Outputs (under outputs/docs/dataset_statistics/):
  * per_world_stats.csv     - one row per world
  * dataset_statistics.json - per-family and overall aggregates
  * dataset_statistics.md   - human-readable summary table

Run:
  python -m utopia.analysis.dataset_statistics \
      --roster outputs/docs/dataset_statistics/paper_world_roster.json \
      --workers 8
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import read_json

import argparse
import csv
import glob
import json
import os
import re
from multiprocessing import Pool

CHECKPOINT_RE = re.compile(r"checkpoint_year_(\d+)\.json(\.gz)?$")

# Agent types that are infrastructure rather than simulated researchers.
NON_RESEARCHER_TYPES = {"funding_agency"}

# Pre-fix worlds where paper_tracker review_history keeps only the last round
# per paper; reviews/decisions there must be unioned from per-year conference
# snapshots (each yearly snapshot is disjoint by construction).
LEGACY_WORLDS = {"90_agents_10_years", "300_researchers_10_years"}

REVIEW_MEMORY_TYPES = {
    "excellent_reviews_received", "good_reviews_received", "average_reviews_received",
    "poor_reviews_received", "harsh_reviews_received",
}


def list_checkpoints(world_dir):
    """Return {year: path} for all checkpoints in world_dir (searches one level of nesting)."""
    candidates = []
    for pattern in ("checkpoint_year_*.json", "checkpoint_year_*.json.gz",
                    "*/checkpoint_year_*.json", "*/checkpoint_year_*.json.gz"):
        candidates.extend(glob.glob(os.path.join(world_dir, pattern)))
    by_year = {}
    for path in candidates:
        m = CHECKPOINT_RE.search(os.path.basename(path))
        if m:
            by_year[int(m.group(1))] = path
    return by_year





def count_replay_dir(family, replay_dir):
    """Count replayed reviews for roster entries without checkpoints (reviews.parquet)."""
    import pandas as pd
    parquet = os.path.join(replay_dir, "reviews.parquet")
    df = pd.read_parquet(parquet, columns=["success"])
    row = {k: 0 for k in NUMERIC_KEYS}
    row.update({
        "family": family,
        "world_dir": replay_dir,
        "final_year": 0,
        "n_reviews": int(df["success"].sum()) if "success" in df else len(df),
        "report_total_submitted": None,
        "integrity_ok": True,
    })
    return row


def count_world(task):
    """Count generated artifacts in one world. task = (family, world_dir)."""
    family, world_dir = task
    try:
        if not list_checkpoints(world_dir) and os.path.exists(os.path.join(world_dir, "reviews.parquet")):
            return count_replay_dir(family, world_dir)

        checkpoints = list_checkpoints(world_dir)
        if not checkpoints:
            return {"family": family, "world_dir": world_dir, "error": "no checkpoint found"}
        final_year = max(checkpoints)
        data = read_json(checkpoints[final_year])

        agents = data["ecosystem_data"]["agents"]
        researchers = [a for a in agents if a.get("type") not in NON_RESEARCHER_TYPES]
        institutions = {a.get("university_name") for a in researchers if a.get("university_name")} \
            | {a.get("company_name") for a in researchers if a.get("company_name")}

        n_thoughts = 0
        n_direction_selections = 0
        n_review_memories = 0
        n_funding_awards = 0
        for a in researchers:
            bank = a.get("memory_bank") or []
            n_thoughts += len(bank)
            for m in bank:
                if m.get("type") == "select_research_direction":
                    n_direction_selections += 1
                elif m.get("type") in REVIEW_MEMORY_TYPES:
                    n_review_memories += 1
            history = a.get("funding_success_history") or {}
            if isinstance(history, dict):
                n_funding_awards += sum(len(v) if isinstance(v, list) else 1 for v in history.values())
            else:
                n_funding_awards += len(history)

        papers = data["paper_tracker"]["papers"]
        n_accepted = sum(1 for p in papers if p.get("status") == "accept")

        if os.path.basename(world_dir.rstrip("/")) in LEGACY_WORLDS:
            # Union of disjoint per-year conference snapshots (see module docstring).
            n_decisions = 0
            n_reviews = 0
            for year in sorted(checkpoints):
                yearly = data if year == final_year else read_json(checkpoints[year])
                for c in yearly["conference_system"]["conferences"]:
                    decisions = c.get("decisions") or {}
                    n_decisions += len(decisions.get("accept") or []) + len(decisions.get("reject") or [])
                    n_reviews += sum(len(rl) for rl in (c.get("reviews") or {}).values())
        else:
            n_decisions = 0
            n_reviews = 0
            for p in papers:
                rounds = p.get("review_history") or []
                n_decisions += len(rounds)
                n_reviews += sum(len(r.get("reviews") or []) for r in rounds)

        references = (data.get("citation_tracker") or {}).get("references") or {}
        n_citation_edges = sum(len(v) for v in references.values() if isinstance(v, list))

        n_funding_applications = 0
        n_funding_winners = 0
        for y in data.get("yearly_results") or []:
            by_type = (y.get("ecosystem_metrics") or {}).get("funding_success_by_type") or {}
            n_funding_applications += by_type.get("total_applications", 0)
            n_funding_winners += by_type.get("university", 0)

        n_funding_applications_logged = 0
        for path in glob.glob(os.path.join(world_dir, "funding_applications_year_*.jsonl")) + \
                glob.glob(os.path.join(world_dir, "*", "funding_applications_year_*.jsonl")):
            with open(path) as f:
                n_funding_applications_logged += sum(1 for line in f if line.strip())

        # Cross-check against the run's own summary report when present.
        report_submitted = None
        for report_path in glob.glob(os.path.join(world_dir, "simulation_report.json")) + \
                glob.glob(os.path.join(world_dir, "*", "simulation_report.json")):
            try:
                with open(report_path) as f:
                    report_submitted = json.load(f).get("papers", {}).get("total_submitted")
            except (json.JSONDecodeError, OSError):
                pass
            break

        return {
            "family": family,
            "world_dir": world_dir,
            "final_year": final_year,
            "n_researchers": len(researchers),
            "n_institutions": len(institutions),
            "n_active_final": sum(1 for a in researchers if a.get("is_active")),
            "n_papers": len(papers),
            "n_accepted_papers": n_accepted,
            "n_decisions": n_decisions,
            "n_reviews": n_reviews,
            "n_thoughts": n_thoughts,
            "n_direction_selections": n_direction_selections,
            "n_review_memories": n_review_memories,
            "n_citation_edges": n_citation_edges,
            "n_funding_applications": n_funding_applications,
            "n_funding_winners": n_funding_winners,
            "n_funding_awards": n_funding_awards,
            "n_funding_applications_logged": n_funding_applications_logged,
            "report_total_submitted": report_submitted,
            # Solo-authored worlds record one *_reviews_received memory per decision.
            "integrity_ok": n_review_memories == n_decisions,
        }
    except Exception as exc:  # surface per-world failures without killing the pool
        return {"family": family, "world_dir": world_dir, "error": repr(exc)}


NUMERIC_KEYS = [
    "n_researchers", "n_institutions", "n_active_final", "n_papers", "n_accepted_papers",
    "n_decisions", "n_reviews", "n_thoughts", "n_direction_selections", "n_review_memories",
    "n_citation_edges", "n_funding_applications", "n_funding_winners", "n_funding_awards",
    "n_funding_applications_logged",
]


def aggregate(rows):
    agg = {k: sum(r[k] for r in rows) for k in NUMERIC_KEYS}
    agg["n_worlds"] = sum(1 for r in rows if r["final_year"] > 0)
    agg["simulated_researcher_years"] = sum(r["n_researchers"] * r["final_year"] for r in rows)
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--roster", required=True,
                        help="JSON file: {family_name: [world_dir, ...], ...}")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/docs/dataset_statistics")
    args = parser.parse_args()

    with open(args.roster) as f:
        roster = json.load(f)

    tasks = [(family, d) for family, dirs in roster.items() for d in dirs]
    # Process the largest worlds first so the pool tail is short.
    tasks.sort(key=lambda t: -sum(os.path.getsize(p) for p in glob.glob(os.path.join(t[1], "checkpoint_year_*.json*"))
                                  + glob.glob(os.path.join(t[1], "*", "checkpoint_year_*.json*"))))
    with Pool(args.workers) as pool:
        rows = pool.map(count_world, tasks)

    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "per_world_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    families = sorted({r["family"] for r in rows})
    summary = {
        "per_family": {fam: aggregate([r for r in rows if r["family"] == fam]) for fam in families},
        "overall": aggregate(rows),
        "errors": errors,
    }
    json_path = os.path.join(args.output_dir, "dataset_statistics.json")
    write_json_file(summary, json_path, indent=2)

    md_path = os.path.join(args.output_dir, "dataset_statistics.md")
    with open(md_path, "w") as f:
        f.write("# Dataset statistics across paper-reported simulation worlds\n\n")
        header = ["family", "n_worlds"] + NUMERIC_KEYS
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for fam in families + ["overall"]:
            agg = summary["overall"] if fam == "overall" else summary["per_family"][fam]
            f.write("| " + " | ".join([fam, str(agg["n_worlds"])] + [f"{agg[k]:,}" for k in NUMERIC_KEYS]) + " |\n")
        f.write(f"\nSimulated researcher-years (overall): {summary['overall']['simulated_researcher_years']:,}\n")
        if errors:
            f.write("\n## Worlds with errors\n\n")
            for e in errors:
                f.write(f"- {e['world_dir']}: {e['error']}\n")

    report_mismatch = [r for r in rows
                       if r["report_total_submitted"] is not None
                       and abs(r["report_total_submitted"] - r["n_decisions"]) > 1]
    integrity_bad = [r for r in rows if not r["integrity_ok"]]
    print(f"Counted {len(rows)} worlds ({len(errors)} errors). Outputs: {csv_path}, {json_path}, {md_path}")
    print(f"Cross-check vs simulation_report total_submitted (tolerance 1): {len(report_mismatch)} mismatches")
    for r in report_mismatch[:10]:
        print(f"  {r['world_dir']}: decisions={r['n_decisions']} report={r['report_total_submitted']}")
    print(f"Integrity check (review memories == decisions): {len(integrity_bad)} mismatches")
    for r in integrity_bad[:10]:
        print(f"  {r['world_dir']}: decisions={r['n_decisions']} review_memories={r['n_review_memories']}")


if __name__ == "__main__":
    main()
