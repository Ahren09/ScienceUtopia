"""Analysis for the initial-resource x institution-size one-world randomized
factorial (experiment_id
initial_resource_x_institution_size_qwen3_32b_i24_n112_y8_seed8301).

Design recap (see prereg_v1.json + mechanism_identification_audit.md):
  * PRIMARY (causal, randomized): initial-resource effect on first_project_duration.
    Estimand = within-institution mean(HIGH - LOW), aggregated over 24 institutions.
    Inference = blocked within-institution randomization inference (reassign the
    exact half-HIGH/half-LOW labels within each institution) + institution
    bootstrap / clustered CIs. Also P(choose longest available duration).
  * INSTITUTION SIZE: exploratory/descriptive heterogeneity ONLY (mechanism audit
    found no first-decision size pathway) -> compare the 24 institution-level
    HIGH-LOW contrasts across tiers (8/tier); report large-minus-small + ALL
    institution points; NO size main-effect term alongside institution FE.
  * SECONDARY family: BH-corrected within-institution HIGH-LOW contrasts.
  * Longitudinal: researcher-year outcomes clustered by researcher & institution.
  * Inequality/persistence: HIGH-LOW gaps by year, Gini trajectories, quartile
    transitions. Collaboration: UNAVAILABLE by design.

Reuses: benjamini_hochberg, hierarchical_bootstrap (institution bootstrap) from
utopia.analysis.exploration_cross_seed; calculate_gini_coefficient from
utopia.metrics.tracker; DIRECTIONS_DICT for topic->years. The ONE new estimator is
blocked_within_institution_randomization().

Run:  python -m utopia.analysis.factorial_resource_size_analysis
"""

from utopia.utils.data_utils import write_json as write_json_file
import json
import os

import numpy as np
import pandas as pd

from utopia.agents.research_direction import DIRECTIONS_DICT
from utopia.analysis.statistics import benjamini_hochberg, hierarchical_bootstrap
from utopia.metrics.tracker import calculate_gini_coefficient

EXP_ID = "initial_resource_x_institution_size_qwen3_32b_i24_n112_y8_seed8301"
DOCS_DIR = os.path.join("outputs", "docs", EXP_ID)
CKPT_DIR = os.path.join("outputs", "checkpoints", EXP_ID)
SEED = 8301
NUM_YEARS = 8
LOW, HIGH = 60, 140
MIE_YEARS = 0.25
TIER_ORDER = ["small", "medium", "large"]
CITATION_WINDOW_AGE = 2                 # principal fixed citation window
COMPLETE_FOLLOWUP_MAX_PUBYEAR = NUM_YEARS - CITATION_WINDOW_AGE   # <=6 -> full 2y window


# ---------------------------------------------------------------- data loading
def _topic_years(topic):
    d = DIRECTIONS_DICT.get(topic)
    return d.years if d is not None else np.nan


def reconstruct_agent_year(docs_dir=DOCS_DIR, ckpt_dir=CKPT_DIR, num_years=NUM_YEARS):
    """Rebuild the per-researcher-year panel from the yearly checkpoints + paper
    table. NEEDED because the stock agent_year/strategy_year exporters only emit
    rows for agents whose strategy is in exp_config['strategies']
    (explorer/exploiter/cautious) -- our all-`balanced` population yields none.
    The checkpoints carry end-of-year resources, is_active, project_end_year,
    chosen direction, and cumulative funding awards for every agent.

    Columns: agent_id, year, institution, funding (end-of-year resources),
    is_active, direction_topic, project_end_year, cumulative_awards, and
    per-year num_papers/num_accepted (from paper.parquet) + cumulative
    total_citations (from paper_age observation_year)."""
    paper = pd.read_parquet(os.path.join(docs_dir, "paper.parquet")) \
        if os.path.exists(os.path.join(docs_dir, "paper.parquet")) else pd.DataFrame()
    paper_age = pd.read_parquet(os.path.join(docs_dir, "paper_age.parquet")) \
        if os.path.exists(os.path.join(docs_dir, "paper_age.parquet")) else pd.DataFrame()

    rows = []
    for y in range(1, num_years + 1):
        for ext in (".json", ".json.gz"):
            p = os.path.join(ckpt_dir, f"checkpoint_year_{y}{ext}")
            if os.path.exists(p):
                opener = __import__("gzip").open if ext.endswith(".gz") else open
                with opener(p, "rt") as f:
                    ck = json.load(f)
                break
        else:
            continue
        for a in ck["ecosystem_data"]["agents"]:
            if a.get("type") != "university":
                continue
            nd = a.get("newest_direction")
            rows.append({
                "agent_id": a["id"], "year": y,
                "institution": a.get("university_name"),
                "funding": a.get("resources"),
                "is_active": a.get("is_active", True),
                "project_end_year": a.get("project_end_year"),
                "direction_topic": (nd or {}).get("direction") if isinstance(nd, dict) else None,
                "cumulative_awards": sum(len(v) for v in (a.get("funding_success_history") or {}).values()),
            })
    ay = pd.DataFrame(rows)
    if ay.empty:
        return ay
    # per-year paper counts
    if not paper.empty:
        pc = (paper.groupby(["author_id", "year"])
              .agg(num_papers=("paper_id", "nunique"),
                   num_accepted=("accepted", "sum")).reset_index()
              .rename(columns={"author_id": "agent_id"}))
        ay = ay.merge(pc, on=["agent_id", "year"], how="left")
    for c in ("num_papers", "num_accepted"):
        if c not in ay.columns:
            ay[c] = 0
        ay[c] = ay[c].fillna(0)
    ay["acceptance_rate"] = np.where(ay["num_papers"] > 0, ay["num_accepted"] / ay["num_papers"], np.nan)
    # cumulative total citations visible at each observation year
    if not paper_age.empty:
        tc = (paper_age.groupby(["author_id", "observation_year"])["citations"].sum()
              .reset_index().rename(columns={"author_id": "agent_id",
                                             "observation_year": "year",
                                             "citations": "total_citations"}))
        ay = ay.merge(tc, on=["agent_id", "year"], how="left")
    if "total_citations" not in ay.columns:
        ay["total_citations"] = np.nan
    ay["total_citations"] = ay["total_citations"].fillna(0)
    return ay


def load_data(docs_dir=DOCS_DIR, ckpt_dir=CKPT_DIR, num_years=NUM_YEARS):
    """Load manifests + decision log + Parquet tables; reconstruct the agent-year
    panel from checkpoints (see reconstruct_agent_year); attach treatment/tier to
    every researcher-level frame."""
    manifest = pd.read_csv(os.path.join(docs_dir, "researcher_manifest.csv"))
    manifest = manifest.rename(columns={"researcher_id": "agent_id",
                                        "institution_id": "institution",
                                        "size_tier": "tier"})
    key = manifest[["agent_id", "institution", "tier", "treatment", "initial_resources"]]

    def _read_parquet(name):
        p = os.path.join(docs_dir, name)
        return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

    agent_year = _read_parquet("agent_year.parquet")
    if agent_year.empty or "funding" not in agent_year.columns:
        agent_year = reconstruct_agent_year(docs_dir, ckpt_dir, num_years)
        if not agent_year.empty:
            agent_year.to_parquet(os.path.join(docs_dir, "agent_year_reconstructed.parquet"),
                                  index=False)
    paper = _read_parquet("paper.parquet")
    paper_age = _read_parquet("paper_age.parquet")
    ecosystem_year = _read_parquet("ecosystem_year.parquet")

    decisions_path = os.path.join(docs_dir, "project_decisions.jsonl")
    decisions = pd.DataFrame()
    if os.path.exists(decisions_path):
        with open(decisions_path) as f:
            decisions = pd.DataFrame(json.loads(l) for l in f if l.strip())

    # attach treatment/tier onto agent-level frames
    for df in (agent_year, paper, paper_age):
        if not df.empty:
            id_col = "agent_id" if "agent_id" in df.columns else "author_id"
            merged = df.merge(key.rename(columns={"agent_id": id_col}), on=id_col,
                              how="left", suffixes=("", "_m"))
            df.__dict__  # noop to keep linter calm
            for c in ("treatment", "tier"):
                df[c] = merged[c].values if c not in df.columns else df[c]
    return dict(manifest=manifest, key=key, agent_year=agent_year, paper=paper,
                paper_age=paper_age, ecosystem_year=ecosystem_year, decisions=decisions)


# ------------------------------------------------- within-institution estimator
def within_institution_contrasts(df, value_col, inst_col="institution",
                                 treat_col="treatment"):
    """Per-institution mean(HIGH) - mean(LOW). Returns a DataFrame with columns
    [institution, tier, contrast, n_high, n_low] (tier attached if present)."""
    rows = []
    tier_of = {}
    if "tier" in df.columns:
        tier_of = df.dropna(subset=["tier"]).groupby(inst_col)["tier"].first().to_dict()
    for inst, g in df.groupby(inst_col):
        hi = g.loc[g[treat_col] == "HIGH", value_col].dropna()
        lo = g.loc[g[treat_col] == "LOW", value_col].dropna()
        if len(hi) == 0 or len(lo) == 0:
            continue
        rows.append({"institution": inst, "tier": tier_of.get(inst),
                     "contrast": float(hi.mean() - lo.mean()),
                     "n_high": int(len(hi)), "n_low": int(len(lo))})
    return pd.DataFrame(rows)


def blocked_within_institution_randomization(df, value_col, n_perm=10000, seed=SEED,
                                             inst_col="institution", treat_col="treatment",
                                             stat="agg_contrast", tiers=None):
    """Blocked randomization inference: reassign the exact half-HIGH/half-LOW
    labels WITHIN each institution, preserving each institution's count, and
    rebuild the null for the aggregated within-institution contrast (or, for the
    size heterogeneity, the large-minus-small contrast of tier-mean contrasts).

    Returns dict with observed statistic and two-sided permutation p-value.
    """
    rng = np.random.default_rng(seed)
    # Pre-extract per-institution value arrays + treatment masks.
    blocks = []
    for inst, g in df.groupby(inst_col):
        g = g.dropna(subset=[value_col])
        treat = (g[treat_col].values == "HIGH")
        if treat.sum() == 0 or (~treat).sum() == 0:
            continue
        blocks.append({"inst": inst, "tier": g["tier"].iloc[0] if "tier" in g.columns else None,
                       "vals": g[value_col].values.astype(float),
                       "n_high": int(treat.sum())})

    def _statistic(assign_high_masks):
        per_inst, sizes = [], []
        tier_lists = {t: [] for t in (tiers or [])}
        for blk, mask in zip(blocks, assign_high_masks):
            c = blk["vals"][mask].mean() - blk["vals"][~mask].mean()
            per_inst.append(c)
            sizes.append(len(blk["vals"]))
            if tiers is not None and blk["tier"] in tier_lists:
                tier_lists[blk["tier"]].append(c)
        if stat == "agg_contrast":       # equal weight per institution (frozen primary)
            return float(np.mean(per_inst))
        if stat == "fe_contrast":        # researcher-weighted (institution fixed effects)
            w = np.asarray(sizes, dtype=float)
            return float(np.sum(w * np.asarray(per_inst)) / np.sum(w))
        # large - small of tier-mean contrasts (descriptive size heterogeneity)
        return float(np.mean(tier_lists["large"]) - np.mean(tier_lists["small"]))

    # Observed statistic uses the real HIGH mask per block.
    real_masks = []
    for blk in blocks:
        sub = df[df[inst_col] == blk["inst"]].dropna(subset=[value_col])
        real_masks.append(sub[treat_col].values == "HIGH")
    observed = _statistic(real_masks)

    null = np.empty(n_perm)
    for b in range(n_perm):
        perm_masks = []
        for blk in blocks:
            n = len(blk["vals"])
            m = np.zeros(n, dtype=bool)
            m[rng.choice(n, size=blk["n_high"], replace=False)] = True
            perm_masks.append(m)
        null[b] = _statistic(perm_masks)
    p = float((np.abs(null - np.mean(null)) >= abs(observed - np.mean(null))).mean())
    return {"observed": observed, "p_perm": p, "n_perm": n_perm,
            "n_blocks": len(blocks), "null_mean": float(np.mean(null))}


def _institution_bootstrap_ci(per_inst, seed=SEED):
    """Institution-clustered bootstrap CI for the aggregated contrast, via the
    reused hierarchical_bootstrap (single seed = institution resampling)."""
    if per_inst.empty:
        return {"mean_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "smd": np.nan,
                "n_institution_blocks": 0}
    contrasts = per_inst.rename(columns={"contrast": "diff"}).copy()
    contrasts["outcome"] = "x"
    contrasts["contrast"] = "HIGH-LOW"
    contrasts["seed"] = SEED
    r = hierarchical_bootstrap(contrasts[["outcome", "contrast", "seed", "diff"]],
                               n_boot=10000, seed=seed).iloc[0]
    return {"mean_diff": float(r["mean_diff"]), "ci_lo": float(r["ci_lo"]),
            "ci_hi": float(r["ci_hi"]), "smd": float(r["smd"]) if pd.notna(r["smd"]) else np.nan,
            "n_institution_blocks": int(r["n_institution_blocks"])}


def contrast_result(df, value_col, name, n_perm=10000):
    """Full within-institution HIGH-LOW result for one researcher-level endpoint."""
    per_inst = within_institution_contrasts(df, value_col)
    ci = _institution_bootstrap_ci(per_inst)
    rand = blocked_within_institution_randomization(df, value_col, n_perm=n_perm)
    return {"endpoint": name, **ci, "observed": rand["observed"],
            "p_perm": rand["p_perm"], "n_blocks": rand["n_blocks"],
            "per_institution": per_inst}


# ---------------------------------------------------------- researcher summaries
def first_decision_frame(data):
    """One row per researcher: first_project_duration + P(longest) helpers,
    from the decision log (fallback to agent_year year==1 direction_topic)."""
    dec = data["decisions"]
    key = data["key"]
    if not dec.empty:
        first = dec.sort_values("year").groupby("researcher_id").first().reset_index()
        first = first.rename(columns={"researcher_id": "agent_id"})
        first["first_duration"] = first["selected_duration"].astype(float)
        first["max_candidate_duration"] = first["candidate_durations"].apply(
            lambda xs: max(xs) if isinstance(xs, (list, tuple)) and len(xs) else np.nan)
        first["min_candidate_duration"] = first["candidate_durations"].apply(
            lambda xs: min(xs) if isinstance(xs, (list, tuple)) and len(xs) else np.nan)
        first["chose_longest"] = (first["first_duration"] >= first["max_candidate_duration"]).astype(float)
        first["chose_shortest"] = (first["first_duration"] <= first["min_candidate_duration"]).astype(float)
        out = first[["agent_id", "first_duration", "chose_longest", "chose_shortest",
                     "max_candidate_duration", "min_candidate_duration"]]
    else:
        ay = data["agent_year"]
        y1 = ay[ay["year"] == 1].copy()
        y1["first_duration"] = y1["direction_topic"].map(_topic_years).astype(float)
        y1["chose_longest"] = np.nan
        out = y1[["agent_id", "first_duration", "chose_longest"]]
    return out.merge(key, on="agent_id", how="left")


def researcher_final_frame(data):
    """One row per researcher with final/cumulative outcomes for the secondary
    family (funding, production, citations, attrition)."""
    ay = data["agent_year"].copy()
    key = data["key"]
    if ay.empty:
        return key.copy()
    ay = ay.sort_values("year")
    last = ay.groupby("agent_id").last().reset_index()
    # end-of-run values: funding + cumulative total_citations = last year
    final = last[["agent_id"]].copy()
    for col, out in [("funding", "final_resources"), ("total_citations", "total_citations"),
                     ("cumulative_awards", "cumulative_awards")]:
        if col in last.columns:
            final[out] = last[col].values
    # per-year counts -> sum over years (papers written/accepted across the run)
    summed = ay.groupby("agent_id")[[c for c in ("num_papers", "num_accepted")
                                     if c in ay.columns]].sum().reset_index()
    final = final.merge(summed.rename(columns={"num_papers": "papers_written",
                                               "num_accepted": "papers_accepted"}),
                        on="agent_id", how="left")
    final["acceptance_rate"] = np.where(final.get("papers_written", 0) > 0,
                                        final.get("papers_accepted", 0) /
                                        final.get("papers_written", np.nan), np.nan)
    # attrition: ever inactive
    if "is_active" in ay.columns:
        attr = ay.groupby("agent_id")["is_active"].min().reset_index()
        final = final.merge(attr.rename(columns={"is_active": "ever_active_min"}), on="agent_id")
        final["attrited"] = (final["ever_active_min"] == False).astype(float)  # noqa: E712
        first_inactive = (ay[ay["is_active"] == False].groupby("agent_id")["year"].min()  # noqa: E712
                          .reset_index().rename(columns={"year": "time_to_attrition"}))
        final = final.merge(first_inactive, on="agent_id", how="left")
    # years resource-constrained (resources < annual cost 10)
    if "funding" in ay.columns:
        yrs = ay.assign(low=ay["funding"] < 10).groupby("agent_id")["low"].sum().reset_index()
        final = final.merge(yrs.rename(columns={"low": "years_resource_constrained"}), on="agent_id")
    return final.merge(key, on="agent_id", how="left")


def citation_frame(data):
    """Fixed 2-year-window citations per researcher, complete-follow-up only."""
    pa = data["paper_age"]
    key = data["key"]
    if pa.empty:
        return key.assign(fixed2yr_citations=np.nan, age_std_citations=np.nan)
    win = pa[(pa["age"] == CITATION_WINDOW_AGE) &
             (pa["publication_year"] <= COMPLETE_FOLLOWUP_MAX_PUBYEAR)].copy()
    per_author = win.groupby("author_id").agg(
        fixed2yr_citations=("citations", "mean"),
        fixed2yr_citations_total=("citations", "sum"),
        n_papers_followed=("paper_id", "nunique")).reset_index()
    per_author["age_std_citations"] = np.log1p(per_author["fixed2yr_citations"])
    return per_author.rename(columns={"author_id": "agent_id"}).merge(key, on="agent_id", how="left")


# ------------------------------------------------------------------ main driver
def run_analysis(docs_dir=DOCS_DIR, n_perm=10000, *, ckpt_dir=CKPT_DIR, num_years=NUM_YEARS):
    data = load_data(docs_dir, ckpt_dir, num_years)
    os.makedirs(docs_dir, exist_ok=True)
    results = {"experiment_id": EXP_ID, "seed": SEED, "MIE_years": MIE_YEARS,
               "citation_window_age": CITATION_WINDOW_AGE}

    # ---------- PRIMARY: first_project_duration ----------
    fd = first_decision_frame(data)
    fd.to_parquet(os.path.join(docs_dir, "first_decision.parquet"), index=False)
    primary = contrast_result(fd, "first_duration", "first_project_duration", n_perm)
    per_inst = primary.pop("per_institution")
    per_inst.to_csv(os.path.join(docs_dir, "primary_first_duration_per_institution.csv"), index=False)
    # Companion researcher-weighted institution-FE estimate (prereg names both the
    # equal-weighted institution-mean-of-contrasts [frozen primary] and institution FE).
    fe = blocked_within_institution_randomization(fd, "first_duration", n_perm=n_perm,
                                                  stat="fe_contrast")
    primary["fe_researcher_weighted"] = {"observed": fe["observed"], "p_perm": fe["p_perm"]}
    # Researcher-weighted pooled raw means (context, not the estimand)
    primary["raw_pooled"] = {
        "LOW_mean": float(fd.loc[fd.treatment == "LOW", "first_duration"].mean()),
        "HIGH_mean": float(fd.loc[fd.treatment == "HIGH", "first_duration"].mean())}
    results["primary"] = primary
    if "chose_longest" in fd.columns and fd["chose_longest"].notna().any():
        results["primary_p_longest"] = contrast_result(
            fd.dropna(subset=["chose_longest"]), "chose_longest",
            "P_choose_longest_duration", n_perm)
        results["primary_p_longest"].pop("per_institution", None)

    # ---------- SIZE HETEROGENEITY (descriptive) ----------
    het = blocked_within_institution_randomization(
        fd, "first_duration", n_perm=n_perm, stat="tier_large_minus_small", tiers=TIER_ORDER)
    tier_means = per_inst.groupby("tier")["contrast"].agg(["mean", "std", "count"]).reindex(TIER_ORDER)
    results["size_heterogeneity"] = {
        "note": "DESCRIPTIVE finite-population heterogeneity; predicted null direct size effect "
                "(no first-decision size mechanism). Not a causal treatment-effect modification.",
        "large_minus_small_contrast": het["observed"], "p_perm": het["p_perm"],
        "tier_mean_contrasts": {t: (float(tier_means.loc[t, "mean"]) if t in tier_means.index
                                    and pd.notna(tier_means.loc[t, "mean"]) else None)
                                for t in TIER_ORDER}}
    per_inst.to_csv(os.path.join(docs_dir, "interaction_by_size_per_institution.csv"), index=False)

    # ---------- SECONDARY family (BH) ----------
    rf = researcher_final_frame(data)
    cf = citation_frame(data)
    rf.to_parquet(os.path.join(docs_dir, "researcher_final.parquet"), index=False)
    cf.to_parquet(os.path.join(docs_dir, "researcher_citations.parquet"), index=False)
    fam_specs = [
        (fd, "first_duration", "first_project_duration_dup"),  # anchor (also primary)
        (rf, "final_resources", "final_resources"),
        (rf, "papers_written", "papers_written"),
        (rf, "papers_accepted", "papers_accepted"),
        (rf, "acceptance_rate", "acceptance_rate"),
        (rf, "total_citations", "total_citations_raw"),
        (rf, "hit_papers", "hit_papers"),
        (rf, "attrited", "attrition"),
        (rf, "years_resource_constrained", "years_resource_constrained"),
        (cf, "fixed2yr_citations", "fixed_2yr_citations"),
        (cf, "age_std_citations", "age_standardized_citations"),
    ]
    fam = []
    for df, col, name in fam_specs:
        if not df.empty and col in df.columns and df[col].notna().any():
            r = contrast_result(df, col, name, n_perm)
            r.pop("per_institution", None)
            fam.append(r)
    if fam:
        fam_df = pd.DataFrame(fam)
        fam_df["p_bh"] = benjamini_hochberg(fam_df["p_perm"].values)
        fam_df.to_csv(os.path.join(docs_dir, "secondary_family_bh.csv"), index=False)
        results["secondary_family"] = fam_df.to_dict("records")

    # ---------- LONGITUDINAL (resources by year x treatment x tier) ----------
    ay = data["agent_year"]
    if not ay.empty and "funding" in ay.columns:
        traj = (ay.groupby(["year", "treatment", "tier"])["funding"]
                .agg(["mean", "std", "count"]).reset_index())
        traj.to_csv(os.path.join(docs_dir, "resource_trajectory_by_year.csv"), index=False)
        gap = (ay.groupby(["year", "treatment"])["funding"].mean().unstack("treatment"))
        gap["HIGH_minus_LOW"] = gap.get("HIGH") - gap.get("LOW")
        gap.reset_index().to_csv(os.path.join(docs_dir, "resource_gap_by_year.csv"), index=False)

    # ---------- INEQUALITY / persistence ----------
    ineq = {}
    if not ay.empty and "funding" in ay.columns:
        gini_rows = []
        for yr, g in ay.groupby("year"):
            row = {"year": int(yr),
                   "resource_gini_all": calculate_gini_coefficient(list(g["funding"].dropna()))}
            for t in ("LOW", "HIGH"):
                vals = list(g.loc[g["treatment"] == t, "funding"].dropna())
                row[f"resource_gini_{t}"] = calculate_gini_coefficient(vals) if vals else np.nan
            gini_rows.append(row)
        pd.DataFrame(gini_rows).to_csv(os.path.join(docs_dir, "gini_trajectory.csv"), index=False)
        ineq["final_resource_gini_all"] = gini_rows[-1]["resource_gini_all"] if gini_rows else None
    results["inequality"] = ineq

    # ---------- runtime / tokens ----------
    man_path = os.path.join(docs_dir, "run_manifest.json")
    if os.path.exists(man_path):
        with open(man_path) as f:
            man = json.load(f)
        rt = {"llm_call_stats": man.get("llm_call_stats"),
              "timestamp_running": man.get("timestamp_running"),
              "timestamp_complete": man.get("timestamp_complete"),
              "package_versions": man.get("package_versions")}
        pd.DataFrame([rt.get("llm_call_stats") or {}]).to_csv(
            os.path.join(docs_dir, "runtime_and_tokens.csv"), index=False)
        results["runtime_and_tokens"] = rt

    write_json_file(results, os.path.join(docs_dir, 'results.json'), indent=2, default=str)
    print(f"[analysis] wrote results.json + tables to {docs_dir}")
    p = results["primary"]
    print(f"[analysis] PRIMARY first_project_duration HIGH-LOW = {p['mean_diff']:+.3f} "
          f"[{p['ci_lo']:+.3f},{p['ci_hi']:+.3f}] p_perm={p['p_perm']:.4g} "
          f"(n_blocks={p['n_blocks']}, MIE={MIE_YEARS})")
    return results


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Numerical resource/size factorial analysis.")
    parser.add_argument('--docs-dir', type=Path, required=True)
    parser.add_argument('--checkpoints-dir', type=Path, required=True)
    parser.add_argument('--n-perm', type=int, default=10000)
    args = parser.parse_args(argv)
    native = json.loads((args.docs_dir / 'run_manifest.json').read_text())
    if native['status'] != 'complete':
        parser.error('The simulation must be complete')
    global EXP_ID, SEED, NUM_YEARS, COMPLETE_FOLLOWUP_MAX_PUBYEAR
    EXP_ID = native['experiment_id']
    SEED = native['args']['seed']
    NUM_YEARS = native['args']['num_years']
    COMPLETE_FOLLOWUP_MAX_PUBYEAR = NUM_YEARS - CITATION_WINDOW_AGE
    run_analysis(str(args.docs_dir), args.n_perm, ckpt_dir=str(args.checkpoints_dir), num_years=NUM_YEARS)


if __name__ == "__main__":
    main()
