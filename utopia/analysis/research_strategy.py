#!/usr/bin/env python3
"""Reanalyse the frozen strategy worlds without invoking the simulator or an LLM.

Run with a Python environment containing numpy, pandas, and a Parquet engine:
    python -m utopia.analysis.research_strategy

Numerical outputs are confined to outputs/docs/research_strategy and
All outputs are numerical tables. Every estimate is
computed separately within a world. Across-world summaries contain the three
estimates and their mean/range, not paper-level or three-world confidence
intervals. These are post hoc descriptive associations, not causal effects.
"""
from __future__ import annotations

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import file_sha256

from utopia.utils.paths import project_root

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (1001, 1002, 1003)
RUN_TEMPLATE = "explore_confirmatory_qwen3_32b_neutral_i1000_n5000_y10_seed{}"
HORIZON = 3
# Retain the historical bin width, extending it to the full theoretical range.
BIN_EDGES = [i / 5 for i in range(11)]
BIN_LABELS = [
    f"{'[' if i == 0 else '('}{BIN_EDGES[i]:g},{BIN_EDGES[i + 1]:g}]"
    for i in range(10)
]
SPECS = {
    "unadjusted": [],
    "venue_year": [["conference", "year"]],
    "project_topic_year": [["project_topic", "year"]],
    "venue_project_topic_year": [["conference", "project_topic", "year"]],
    "author_and_venue_project_topic_year": [
        ["author_id"], ["conference", "project_topic", "year"]
    ],
}


def require_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} has duplicate keys {keys}")


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def strategy_manipulation(agent_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate first choices, repeat-choice propensity, and annual exposure.

    Switching is a change in the named direction from the previous logged choice.
    Conditional distance is the stored direction-to-career-centroid distance,
    conditional on such a switch. It is NOT distance between two direction labels.
    """
    require_unique(agent_year, ["agent_id", "year"], "agent_year")
    ay = agent_year.sort_values(["agent_id", "year"]).copy()
    if (ay.groupby("agent_id")["strategy"].nunique() != 1).any():
        raise ValueError("Strategy changes within researcher")
    ay["active_at_year_start"] = (
        ay.groupby("agent_id")["is_active"].shift().astype("boolean").fillna(True))
    choices = ay.loc[ay["direction_topic"].notna()].copy()
    choices["previous_topic"] = choices.groupby("agent_id")["direction_topic"].shift()
    choices["repeat_choice"] = choices["previous_topic"].notna()
    choices["switch"] = choices["repeat_choice"] & choices["direction_topic"].ne(
        choices["previous_topic"])
    stored = choices["topic_switched"].astype(bool)
    if not stored.eq(choices["switch"]).all():
        raise ValueError("Stored switches disagree with reconstructed direction history")
    rows = []
    for strategy, all_years in ay.groupby("strategy"):
        ch = choices.loc[choices["strategy"].eq(strategy)]
        repeat = ch.loc[ch["repeat_choice"]]
        switches = repeat.loc[repeat["switch"]]
        n_founders = all_years["agent_id"].nunique()
        active_years = int(all_years["active_at_year_start"].sum())
        rows.append({
            "strategy": strategy, "n_founders": n_founders,
            "n_choice_events": len(ch), "n_first_choices": int((~ch["repeat_choice"]).sum()),
            "n_repeat_choices": len(repeat), "n_switches": len(switches),
            "n_authors_with_repeat_choice": repeat["agent_id"].nunique(),
            "n_authors_with_switch": switches["agent_id"].nunique(),
            "stored_switch_rate_author_mean_including_first": float(
                ch.groupby("agent_id")["switch"].mean().mean()),
            "repeat_choice_switch_probability_event_weighted": float(repeat["switch"].mean()),
            "repeat_choice_switch_probability_author_weighted": float(
                repeat.groupby("agent_id")["switch"].mean().mean()),
            "conditional_direction_distance_event_weighted": float(
                switches["direction_distance"].mean()),
            "conditional_direction_distance_author_weighted": float(
                switches.groupby("agent_id")["direction_distance"].mean().mean()),
            "unconditional_direction_distance_event_weighted": float(ch["direction_distance"].mean()),
            "choice_events_per_founder": len(ch) / n_founders,
            "switches_per_founder": len(switches) / n_founders,
            "n_active_at_year_start_years": active_years,
            "switches_per_active_at_year_start_year": len(switches) / active_years,
            "choice_events_per_active_at_year_start_year": len(ch) / active_years,
        })
    return pd.DataFrame(rows), choices


def prepare_papers(paper: pd.DataFrame, agent_year: pd.DataFrame) -> pd.DataFrame:
    """Attach originating project topic, preserving submission-event identities."""
    require_unique(paper, ["paper_id", "year"], "paper events")
    for field in ("author_id", "strategy", "novelty_score"):
        if (paper.groupby("paper_id")[field].nunique(dropna=False) > 1).any():
            raise ValueError(f"{field} changes within a manuscript")
    if paper["novelty_score"].isna().any():
        raise ValueError("Missing stored paper centroid distances")
    if not paper["novelty_score"].between(0, 2).all():
        raise ValueError("Paper cosine distance is outside [0,2]")
    ay = agent_year.sort_values(["agent_id", "year"]).copy()
    ay["project_topic"] = ay.groupby("agent_id")["direction_topic"].ffill()
    lookup = ay[["agent_id", "year", "project_topic"]].rename(
        columns={"agent_id": "author_id"})
    first = paper.sort_values(["paper_id", "year"]).drop_duplicates("paper_id")
    origin = first[["paper_id", "author_id", "year"]].merge(
        lookup, on=["author_id", "year"], how="left", validate="many_to_one")
    if origin["project_topic"].isna().any():
        raise ValueError("Cannot reconstruct originating project topic")
    out = paper.merge(origin[["paper_id", "project_topic"]], on="paper_id",
                      how="left", validate="many_to_one")
    out["distance_bin"] = pd.cut(out["novelty_score"], BIN_EDGES, labels=BIN_LABELS,
                                include_lowest=True)
    return out


def fixed_window_yield(paper: pd.DataFrame, paper_age: pd.DataFrame,
                      end_year: int, horizon: int = HORIZON) -> pd.DataFrame:
    """Citation yield by first submission + horizon, one row per manuscript.

    Unpublished by the deadline => zero PUBLICATION-MEDIATED realized yield.
    A published manuscript without a citation observation is an error, not zero.
    Later accepted manuscripts are zero at the earlier deadline. This does not
    estimate the latent impact of rejected work or its impact if it were accepted.
    """
    require_unique(paper_age, ["paper_id", "age"], "paper-age panel")
    require_unique(paper_age, ["paper_id", "observation_year"], "paper-age observations")
    if not (paper_age["observation_year"] == paper_age["publication_year"] +
            paper_age["age"]).all():
        raise ValueError("Paper-age time origin is inconsistent")
    pubs = paper_age.loc[paper_age["age"].eq(0), ["paper_id", "publication_year"]]
    require_unique(pubs, ["paper_id"], "publication records")
    accept = paper.loc[paper["accepted"], ["paper_id", "year"]].rename(
        columns={"year": "publication_year"})
    require_unique(accept, ["paper_id"], "accepted manuscripts")
    agreement = accept.merge(pubs, on=["paper_id", "publication_year"], how="outer",
                             indicator=True, validate="one_to_one")
    if not agreement["_merge"].eq("both").all():
        raise ValueError("Accepted events and publication panel disagree")
    first = paper.sort_values(["paper_id", "year"]).drop_duplicates("paper_id").copy()
    first["deadline"] = first["year"] + horizon
    cohort = first.loc[first["deadline"].le(end_year)].merge(
        pubs, on="paper_id", how="left", validate="one_to_one")
    if cohort["publication_year"].lt(cohort["year"]).any():
        raise ValueError("Publication precedes first observed submission")
    observed = paper_age[["paper_id", "observation_year", "citations"]]
    cohort = cohort.merge(observed, left_on=["paper_id", "deadline"],
                          right_on=["paper_id", "observation_year"], how="left",
                          validate="one_to_one")
    timely = cohort["publication_year"].le(cohort["deadline"])
    if cohort.loc[timely, "citations"].isna().any():
        raise ValueError("Missing citation observation for timely publication")
    cohort["published_by_deadline"] = timely
    cohort["publication_mediated_citation_yield"] = np.where(
        timely, cohort["citations"], 0.0)
    cohort["log1p_publication_mediated_citation_yield"] = np.log1p(
        cohort["publication_mediated_citation_yield"])
    cohort["yield_status"] = np.select(
        [timely, cohort["publication_year"].gt(cohort["deadline"])],
        ["published_by_deadline", "published_after_deadline"],
        default="not_published_by_simulation_end")
    return cohort


def demean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(codes)
    sums = np.column_stack([
        np.bincount(codes, weights=values[:, k], minlength=len(counts))
        for k in range(values.shape[1])
    ])
    return values - (sums / counts[:, None])[codes]


def within_slope(frame: pd.DataFrame, outcome: str,
                 groups: list[list[str]]) -> dict:
    """Descriptive linear slope after absorbing the specified fixed effects.

    No individual-paper standard errors are reported. All rows of each researcher
    and each world remain together. Alternating projections absorb author and
    context effects without building a large dummy matrix.
    """
    needed = list(dict.fromkeys(["novelty_score", outcome, "author_id", "paper_id"] +
                               [c for group in groups for c in group]))
    usable = frame.dropna(subset=needed).copy()
    values = usable[["novelty_score", outcome]].to_numpy(dtype=float)
    if len(values) == 0:
        return {"n_observations": 0, "slope_per_0_1_distance": np.nan,
                "identified": False}
    codes = [
        pd.factorize(pd.MultiIndex.from_frame(usable[group]), sort=False)[0]
        for group in groups
    ]
    if not codes:
        values -= values.mean(axis=0)
    else:
        for iteration in range(2000):
            prior = values.copy()
            for group_codes in codes:
                values = demean(values, group_codes)
            if np.max(np.abs(prior - values)) < 1e-10:
                break
        else:
            raise ValueError("Fixed-effect demeaning did not converge")
    xx = float(values[:, 0] @ values[:, 0])
    beta = float(values[:, 0] @ values[:, 1] / xx) if xx > 1e-12 else np.nan
    return {
        "n_observations": len(usable), "n_missing_excluded": len(frame) - len(usable),
        "n_manuscripts": usable["paper_id"].nunique(),
        "n_authors": usable["author_id"].nunique(),
        "absorbed_group_counts": json.dumps([int(c.max() + 1) for c in codes]),
        "residual_distance_ss": xx, "identified": bool(xx > 1e-12),
        "n_rows_with_residual_distance": int((np.abs(values[:, 0]) > 1e-8).sum()),
        "slope_per_0_1_distance": beta * 0.1,
    }


def association_rows(paper: pd.DataFrame, paper_age: pd.DataFrame,
                     cohort: pd.DataFrame) -> pd.DataFrame:
    accepted = paper.loc[paper["accepted"]].merge(
        paper_age.loc[paper_age["age"].eq(HORIZON), ["paper_id", "citations"]],
        on="paper_id", how="inner", validate="one_to_one")
    accepted["log1p_citations_at_publication_age_3"] = np.log1p(accepted["citations"])
    first = paper.sort_values(["paper_id", "year"]).drop_duplicates("paper_id")
    analyses = [
        ("all_submission_events", paper, "accepted"),
        ("all_submission_events", paper, "review_score"),
        ("first_submission_per_manuscript", first, "accepted"),
        ("first_submission_fixed_window", cohort, "accepted"),
        ("first_submission_fixed_window", cohort, "published_by_deadline"),
        ("first_submission_fixed_window", cohort, "publication_mediated_citation_yield"),
        ("first_submission_fixed_window", cohort,
         "log1p_publication_mediated_citation_yield"),
        ("accepted_complete_publication_age_3", accepted, "citations"),
        ("accepted_complete_publication_age_3", accepted,
         "log1p_citations_at_publication_age_3"),
    ]
    rows = []
    for sample, data, outcome in analyses:
        for spec, groups in SPECS.items():
            rows.append({"sample": sample, "outcome": outcome, "specification": spec,
                         **within_slope(data, outcome, groups)})
    return pd.DataFrame(rows)


def bin_rows(paper: pd.DataFrame, paper_age: pd.DataFrame,
             cohort: pd.DataFrame) -> pd.DataFrame:
    accepted = paper.loc[paper["accepted"]].merge(
        paper_age.loc[paper_age["age"].eq(HORIZON), ["paper_id", "citations"]],
        on="paper_id", how="inner", validate="one_to_one")
    records = []
    samples = [
        ("all_submission_events", paper, ["accepted", "review_score"]),
        ("first_submission_fixed_window", cohort,
         ["accepted", "published_by_deadline", "publication_mediated_citation_yield"]),
        ("accepted_complete_publication_age_3", accepted, ["citations"]),
    ]
    for sample, data, outcomes in samples:
        for group_fields in [["distance_bin"], ["strategy"]]:
            for group, block in data.groupby(group_fields[0], observed=True):
                for outcome in outcomes:
                    records.append({
                        "sample": sample, "group_by": group_fields[0], "group": str(group),
                        "outcome": outcome, "n": len(block),
                        "mean": float(block[outcome].mean()),
                        "median": float(block[outcome].median()),
                        "mean_stored_centroid_distance": float(block["novelty_score"].mean()),
                    })
    return pd.DataFrame(records)


def audit_world(paper: pd.DataFrame, paper_age: pd.DataFrame,
                agent_year: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    first = paper.sort_values(["paper_id", "year"]).drop_duplicates("paper_id")
    return {
        "agent_year_rows": len(agent_year),
        "n_researchers": agent_year["agent_id"].nunique(),
        "submission_events": len(paper), "unique_manuscripts": len(first),
        "resubmission_events": len(paper) - len(first),
        "accepted_manuscripts": int(paper["accepted"].sum()),
        "complete_publication_age_3": int(paper_age["age"].eq(HORIZON).sum()),
        "minimum_stored_centroid_distance": float(paper["novelty_score"].min()),
        "maximum_stored_centroid_distance": float(paper["novelty_score"].max()),
        "submission_events_with_distance_above_1": int(paper["novelty_score"].gt(1).sum()),
        "unique_manuscripts_with_distance_above_1": int(first["novelty_score"].gt(1).sum()),
        "first_submission_fixed_window_n": len(cohort),
        "excluded_first_submissions_right_censored": len(first) - len(cohort),
        "yield_status_counts": {str(k): int(v) for k, v in cohort["yield_status"].value_counts().items()},
        "all_timely_publications_have_deadline_observation": True,
    }




def career_researcher_summaries(agent_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep researcher weighting and the complete founding survival denominator."""
    require_unique(agent_year, ["agent_id", "year"], "career agent-year panel")
    ay = agent_year.sort_values(["agent_id", "year"])
    if ay[["strategy", "is_active", "num_papers", "num_accepted"]].isna().any().any():
        raise ValueError("Missing career outcome inputs")
    if (ay.groupby("agent_id")["strategy"].nunique() != 1).any():
        raise ValueError("Strategy changes within researcher")
    years = sorted(ay["year"].unique())
    if not ay.groupby("agent_id")["year"].nunique().eq(len(years)).all():
        raise ValueError("Incomplete founding-cohort survival panel")
    researchers = ay.groupby("agent_id").agg(
        strategy=("strategy", "first"),
        submission_events=("num_papers", "sum"),
        accepted_manuscripts=("num_accepted", "sum"),
    )
    researchers["acceptance_fraction"] = (
        researchers["accepted_manuscripts"] /
        researchers["submission_events"].replace(0, np.nan))
    summary = researchers.groupby("strategy").agg(
        n_founders=("strategy", "size"),
        n_researchers_with_submissions=("acceptance_fraction", "count"),
        submission_events=("submission_events", "sum"),
        accepted_manuscripts=("accepted_manuscripts", "sum"),
        mean_researcher_acceptance_rate=("acceptance_fraction", "mean"),
    ).reset_index()
    survival = ay.groupby(["strategy", "year"]).agg(
        n_founders=("agent_id", "size"),
        n_active_end_year=("is_active", "sum"),
    ).reset_index()
    survival["active_share_end_year"] = (
        survival["n_active_end_year"] / survival["n_founders"])
    return summary, survival




def run(repo_root: Path, run_ids, out_dir: Path) -> None:
    manipulation, associations, bins = [], [], []
    audits, sources, run_info = {}, [], []
    for run_id in run_ids:
        folder = repo_root / "outputs/docs" / run_id
        manifest_path = folder / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        seed = manifest["run_seed"]
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete or mismatched world {folder}")
        args = manifest["args"]
        if args.get("funding_intervention") or args.get("citation_intervention"):
            raise ValueError("Expected neutral worlds")
        if manifest["resolved_config"]["exploration_experiment"].get("review_strategy_adjustment"):
            raise ValueError("Synthetic strategy review adjustment is enabled")
        paths = [folder / f"{name}.parquet" for name in ("paper", "paper_age", "agent_year")]
        paper, pa, ay = [pd.read_parquet(path) for path in paths]
        if len(ay) != ay["agent_id"].nunique() * args["num_years"]:
            raise ValueError("Incomplete founding-cohort agent-year panel")
        m, _ = strategy_manipulation(ay)
        p = prepare_papers(paper, ay)
        cohort = fixed_window_yield(p, pa, args["num_years"])
        assoc = association_rows(p, pa, cohort)
        b = bin_rows(p, pa, cohort)
        for collection, frame in [(manipulation, m), (associations, assoc), (bins, b)]:
            collection.append(frame.assign(seed=seed))
        audits[str(seed)] = audit_world(p, pa, ay, cohort)
        run_info.append({
            "seed": seed, "git_commit": manifest.get("git_commit"),
            "scientific_config_hash": manifest.get("scientific_config_hash"),
            "model": args["model"], "num_years": args["num_years"],
            "population_mode": args["population_mode"],
        })
        sources.extend(fingerprint(path) for path in paths + [manifest_path])
    if len({r["scientific_config_hash"] for r in run_info}) != 1:
        raise ValueError("Scientific configurations differ across worlds")
    m = pd.concat(manipulation, ignore_index=True)
    a = pd.concat(associations, ignore_index=True)
    b = pd.concat(bins, ignore_index=True)
    summary = a.groupby(["sample", "outcome", "specification"], sort=False).agg(
        n_worlds=("seed", "nunique"),
        mean_slope_per_0_1=("slope_per_0_1_distance", "mean"),
        minimum_slope_per_0_1=("slope_per_0_1_distance", "min"),
        maximum_slope_per_0_1=("slope_per_0_1_distance", "max"),
        n_positive=("slope_per_0_1_distance", lambda x: int((x > 0).sum())),
        n_negative=("slope_per_0_1_distance", lambda x: int((x < 0).sum())),
    ).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("strategy_manipulation", m), ("associations_by_world", a),
                        ("bin_and_strategy_outcomes", b), ("associations_world_summary", summary)]:
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    protocol = {
        "post_hoc_reanalysis": True, "run_info": run_info,
        "analysis_source": fingerprint(Path(__file__)), "inputs": sources,
        "world_audits": audits,
        "definitions": {
            "switch_propensity": "Changed named direction / repeat direction-selection events. First choices excluded.",
            "conditional_distance": "Stored direction-to-career-centroid distance among switches. Not inter-direction step distance.",
            "annual_frequency": "Switch count divided by researcher-years active at year start, separately from per-choice propensity.",
            "paper_distance": "Frozen novelty_score, not career_distance (the previous-paper metric). Not recomputed.",
            "project_topic": "Most recent logged author direction at first observed submission, carried to resubmissions. Proxy for project topic, not measured article topic.",
            "yield": "Citation count at first submission year + 3 if published by then, otherwise zero publication-mediated realized yield. Not latent quality or hypothetical impact if accepted.",
            "conditional_citations": "Accepted manuscripts with observed publication age 3. This is selected on acceptance.",
            "fixed_effects": "Descriptive within-world linear slopes with joint venue/project-topic/year and author fixed effects. Coefficients scaled to 0.1 distance.",
            "uncertainty": "No paper-independent SEs, p-values, or confidence intervals. Per-world estimates and their range only. Repeated papers/authors are never treated as independent replicates for inference.",
            "within_stratum_weights": "Every observation has equal weight within a world. A single-stratum fixed-effect slope weights strata by their residual distance sum of squares, not equally. Multiple fixed effects use alternating demeaning.",
            "distance_bins": "Equal-width 0.2 bins over [0,2], including distances above 1. Empty bins are unobserved, not zero. Curves connect occupied-bin means within each world, positioned at actual within-bin mean distance.",
        },
        "limitations": [
            "Direction switching does not establish movement to an unfamiliar topic or independent randomization of propensity and distance.",
            "Topic-specific project duration affects choice opportunities and resource costs.",
            "Stored paper-centroid distances are retained. The current code's inclusive window spans four prior calendar years despite the manuscript's three-year description.",
            "Adjusting for venue and project topic can condition on intermediates of strategy. Adjusted associations are sensitivity descriptions, not total strategy effects.",
            "Results are conditional on the supplied worlds and model. Population-level external validity is unestablished.",
            "Active-at-year-start is reconstructed from the previous year-end status. It is not an exact within-year risk-set duration.",
            "Manuscript identity is unique only within a world. All joins and regressions occur separately within each world.",
        ],
    }
    write_json_file(protocol, out_dir / 'analysis_audit.json', trailing_newline=True, streaming=False, indent=2)
    lines = [
        "Post hoc strategy analysis of the explicitly supplied worlds.",
        "Read analysis_audit.json for definitions, source hashes, and cohort accounting.",
        "No simulator, inference server, or GPU job is invoked.",
        "",
        "Repeat-choice switch propensity and conditional direction distance:",
        m[["seed", "strategy", "repeat_choice_switch_probability_event_weighted",
           "conditional_direction_distance_event_weighted",
           "switches_per_active_at_year_start_year"]].to_string(index=False),
        "",
        "Descriptive associations, equal weight per world, no inferential intervals:",
        summary.to_string(index=False),
        "",
        "The output does not identify latent rejected-paper impact or causal novelty effects.",
    ]
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=project_root(__file__))
    cli = parser.parse_args()
    run(cli.repo_root.resolve())
