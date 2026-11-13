"""Figures for research strategy analysis."""

from utopia.utils.data_utils import write_json as write_json_file
from pathlib import Path
import json
import numpy as np
import pandas as pd
from utopia.analysis.research_strategy import SEEDS, HORIZON, RUN_TEMPLATE, require_unique, career_researcher_summaries, fingerprint
GREEN = "#008000"


def green_figures(manipulation: pd.DataFrame, bins: pd.DataFrame,
                  repo_root: Path) -> list[Path]:
    """Export green-only marks/text. White is used only for the background."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    out = repo_root / "outputs/visual/research_strategy"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": GREEN, "axes.labelcolor": GREEN,
        "axes.edgecolor": GREEN, "axes.titlecolor": GREEN,
        "xtick.color": GREEN, "ytick.color": GREEN,
        "grid.color": GREEN, "patch.edgecolor": GREEN,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.edgecolor": GREEN, "legend.labelcolor": GREEN,
    })
    markers = ["o", "s", "^"]
    styles = ["-", "--", ":"]
    strategies = ["explorer", "cautious_explorer", "exploiter"]
    labels = ["Explorer", "Cautious\nexplorer", "Exploiter"]
    saved = []

    def finish(fig, name):
        for ax in fig.axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(length=3, width=0.6)
        for suffix in ("pdf", "png"):
            path = out / f"{name}.{suffix}"
            fig.savefig(path, dpi=240, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.1))
    metrics = [
        ("repeat_choice_switch_probability_event_weighted",
         "a  Switch propensity", "Switches / repeat choices", (0, 1.08)),
        ("conditional_direction_distance_event_weighted",
         "b  Conditional distance", "Direction-to-centroid distance", (0, 2)),
        ("switches_per_active_at_year_start_year",
         "c  Annual switch frequency", "Switches / active researcher-year", (0, 0.5)),
    ]
    for ax, (metric, title, ylabel, limits) in zip(axes, metrics):
        for offset, seed, marker in zip([-0.12, 0, 0.12], SEEDS, markers):
            values = manipulation.loc[manipulation.seed.eq(seed)].set_index("strategy")
            ax.plot(np.arange(3) + offset, values.loc[strategies, metric], marker=marker,
                    color=GREEN, markerfacecolor="none", markersize=4,
                    linestyle="none", markeredgewidth=0.9)
        for position, strategy in enumerate(strategies):
            mean = manipulation.loc[manipulation.strategy.eq(strategy), metric].mean()
            ax.hlines(mean, position - 0.23, position + 0.23, color=GREEN, linewidth=1.1)
        ax.set_xticks(np.arange(3), labels)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*limits)
        ax.set_title(title, fontsize=9, loc="left", pad=8)
        ax.set_ylabel(ylabel, fontsize=8)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    handles = [
        Line2D([], [], marker=marker, color=GREEN, markerfacecolor="none",
               linestyle="none", markersize=4, label=f"World {seed}")
        for seed, marker in zip(SEEDS, markers)
    ]
    handles.append(Line2D([], [], color=GREEN, lw=1.1, label="Equal-world mean"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.79, bottom=0.27, wspace=0.62)
    fig.text(0.075, 0.08,
             "First choices excluded from switch propensity. Conditional distance uses switches only.\n"
             "Active researcher-years use status at year start. Three world estimates; no confidence intervals.",
             fontsize=7, va="bottom")
    finish(fig, "fig_strategy_manipulation_green")

    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.4))
    data = bins.loc[bins["sample"].eq("first_submission_fixed_window") &
                    bins["group_by"].eq("distance_bin")]
    outcomes = [
        ("accepted", "a  First-submission acceptance", "Accepted on first submission", True),
        ("published_by_deadline", "b  Publication by deadline", "Published by submission + 3 years", True),
        ("publication_mediated_citation_yield", "c  Realized citation yield",
         "Citations per submitted manuscript", False),
    ]
    for ax, (outcome, title, ylabel, percent) in zip(axes, outcomes):
        for seed, marker, style in zip(SEEDS, markers, styles):
            block = data.loc[data.seed.eq(seed) & data.outcome.eq(outcome)].sort_values(
                "mean_stored_centroid_distance")
            ax.plot(block["mean_stored_centroid_distance"], block["mean"],
                    marker=marker, linestyle=style, color=GREEN,
                    markerfacecolor="none", markersize=3.5, linewidth=1.0)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.set_ylim(bottom=0)
        if percent:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(title, fontsize=8.5, loc="left", pad=8)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_xlabel("Stored paper-to-centroid distance", fontsize=7.5)
    handles = [
        Line2D([], [], color=GREEN, linestyle=style, marker=marker,
               markerfacecolor="none", markersize=3.5, label=f"World {seed}")
        for seed, marker, style in zip(SEEDS, markers, styles)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.30, wspace=0.64)
    counts = data.loc[data.outcome.eq("accepted")].groupby("seed")["n"].sum()
    fig.text(0.075, 0.065,
             "First submitted in years 1–7: " +
             " / ".join(f"{int(counts[s]):,}" for s in SEEDS) + " unique manuscripts.\n"
             "Deadline = first submission + 3 years. Unpublished by deadline: zero publication-mediated yield.\n"
             "Raw associations, not causal effects. Bins have width 0.2; points use within-bin mean distance.",
             fontsize=6.8, va="bottom")
    finish(fig, "fig_departure_first_submission_green")
    return saved


def career_outcomes_green(repo_root: Path) -> list[Path]:
    """Plot audited career outcomes without rerunning other statistical analyses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    docs = repo_root / "outputs/docs/research_strategy"
    visual = repo_root / "outputs/visual/research_strategy"
    bins_path = docs / "bin_and_strategy_outcomes.csv"
    aggregate_path = repo_root / "outputs/docs/explore_e4a_analysis/aggregate_results.csv"
    bins = pd.read_csv(bins_path)
    archived = pd.read_csv(aggregate_path)
    selected = bins.loc[
        bins["sample"].eq("accepted_complete_publication_age_3") &
        bins["group_by"].eq("strategy") & bins["outcome"].eq("citations"),
        ["seed", "group", "n", "mean"],
    ].rename(columns={"group": "strategy", "n": "n_accepted_papers_at_age_3",
                      "mean": "mean_citations_per_accepted_paper_at_age_3"})
    require_unique(selected, ["seed", "strategy"], "saved strategy citation means")
    require_unique(archived, ["seed", "strategy"], "archived researcher means")
    strategies = ["explorer", "cautious_explorer", "exploiter"]
    sources = [fingerprint(bins_path), fingerprint(aggregate_path)]
    summaries, survivals = [], []
    configurations = set()
    for seed in SEEDS:
        folder = repo_root / "outputs/docs" / RUN_TEMPLATE.format(seed)
        manifest_path = folder / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete" or manifest["run_seed"] != seed:
            raise ValueError(f"Incomplete or mismatched world {seed}")
        configurations.add(manifest["scientific_config_hash"])
        ay_path, pa_path = folder / "agent_year.parquet", folder / "paper_age.parquet"
        ay = pd.read_parquet(ay_path)
        if sorted(ay["year"].unique()) != list(range(1, 11)):
            raise ValueError("Career figure requires all ten simulated years")
        summary, survival = career_researcher_summaries(ay)
        if set(summary["strategy"]) != set(strategies):
            raise ValueError("Unexpected strategy cohort")
        check = summary.merge(
            archived.loc[archived.seed.eq(seed), ["strategy", "acceptance_rate"]],
            on="strategy", how="left", validate="one_to_one")
        if not np.allclose(check["mean_researcher_acceptance_rate"],
                           check["acceptance_rate"], rtol=0, atol=1e-12):
            raise ValueError("Researcher acceptance differs from archived aggregate")
        # Plot the saved age-three means, while checking their cohort against raw rows.
        pa = pd.read_parquet(pa_path)
        require_unique(pa, ["paper_id", "age"], "career paper-age panel")
        age3 = pa.loc[pa["age"].eq(HORIZON)]
        if age3["citations"].isna().any():
            raise ValueError("Missing selected-paper age-three citation count")
        raw = age3.groupby("strategy").agg(
            n=("paper_id", "size"), mean=("citations", "mean"))
        saved = selected.loc[selected.seed.eq(seed)].set_index("strategy")
        if set(saved.index) != set(strategies):
            raise ValueError("Missing saved strategy citation mean")
        if not raw.loc[strategies, "n"].eq(
                saved.loc[strategies, "n_accepted_papers_at_age_3"]).all():
            raise ValueError("Saved citation cohort sizes differ from raw age-three panel")
        if not np.allclose(raw.loc[strategies, "mean"],
                           saved.loc[strategies, "mean_citations_per_accepted_paper_at_age_3"],
                           rtol=0, atol=1e-12):
            raise ValueError("Saved citation means differ from raw age-three panel")
        summaries.append(summary.assign(seed=seed).merge(
            selected.loc[selected.seed.eq(seed)], on=["seed", "strategy"],
            how="left", validate="one_to_one"))
        survivals.append(survival.assign(seed=seed))
        sources.extend(fingerprint(p) for p in (manifest_path, ay_path, pa_path))
    if len(configurations) != 1:
        raise ValueError("Career worlds have different scientific configurations")
    outcomes = pd.concat(summaries, ignore_index=True)
    survival = pd.concat(survivals, ignore_index=True)
    means = outcomes.groupby("strategy")[
        ["mean_researcher_acceptance_rate", "mean_citations_per_accepted_paper_at_age_3"]
    ].mean()
    survival_means = survival.groupby(["strategy", "year"])["active_share_end_year"].mean()
    visual.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "text.color": GREEN, "axes.labelcolor": GREEN,
        "axes.edgecolor": GREEN, "axes.titlecolor": GREEN,
        "xtick.color": GREEN, "ytick.color": GREEN,
        "grid.color": GREEN, "patch.edgecolor": GREEN,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.edgecolor": GREEN, "legend.labelcolor": GREEN,
    })
    markers, styles = ["o", "^", "s"], ["-", "--", ":"]
    labels = ["Explorer", "Cautious explorer", "Exploiter"]
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.25),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    panels = [
        ("mean_researcher_acceptance_rate", "a  Researcher acceptance",
         "Mean researcher acceptance rate", (0, 0.5)),
        ("mean_citations_per_accepted_paper_at_age_3", "b  Accepted-paper citations",
         "Citations per accepted paper at age 3", (0, 2.5)),
    ]
    for ax, (metric, title, ylabel, limits) in zip(axes[:2], panels):
        for position, (strategy, marker) in enumerate(zip(strategies, markers)):
            values = outcomes.loc[outcomes.strategy.eq(strategy)].set_index("seed")
            ax.plot(position + np.array([-0.14, 0, 0.14]), values.loc[list(SEEDS), metric],
                    color=GREEN, marker=marker, markerfacecolor="none",
                    markeredgewidth=0.9, markersize=4, linestyle="none")
            value = float(means.loc[strategy, metric])
            ax.hlines(value, position - 0.26, position + 0.26,
                      color=GREEN, linewidth=1.1)
            text = f"{100 * value:.1f}%" if metric.startswith("mean_researcher") else f"{value:.2f}"
            ax.annotate(text, (position, float(values[metric].max())),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=GREEN)
        ax.set_xticks(range(3), ["Explorer", "Cautious\nexplorer", "Exploiter"])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(*limits)
        ax.set_title(title, fontsize=8.5, loc="left", pad=8)
        ax.set_ylabel(ylabel, fontsize=7.5)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for strategy, marker, style in zip(strategies, markers, styles):
        for seed in SEEDS:
            block = survival.loc[survival.seed.eq(seed) & survival.strategy.eq(strategy)]
            axes[2].plot(block["year"], block["active_share_end_year"],
                         color=GREEN, linestyle=style, linewidth=0.45)
        mean = survival_means.loc[strategy]
        axes[2].plot(mean.index, mean.to_numpy(), color=GREEN, linestyle=style,
                     marker=marker, markerfacecolor="white", markeredgecolor=GREEN,
                     markersize=3.2, markevery=[1, 3, 5, 7, 9], linewidth=1.35)
    axes[2].set_title("c  End-year activity", fontsize=8.5, loc="left", pad=8)
    axes[2].set_ylabel("Active share of founding researchers", fontsize=7.5)
    axes[2].set_xlabel("Simulation year", fontsize=7.5)
    axes[2].set_xlim(1, 10)
    axes[2].set_xticks([1, 2, 4, 6, 8, 10])
    axes[2].set_ylim(0, 1.04)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.6, labelsize=7)
    handles = [
        Line2D([], [], color=GREEN, marker=marker, linestyle=style,
               markerfacecolor="white", markersize=4, label=label)
        for label, marker, style in zip(labels, markers, styles)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=3, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.31, wspace=0.66)
    fig.text(0.08, 0.035,
             "Offset points / thin curves: worlds 1001–1003. Horizontal ticks / bold curves: equal-world means.\n"
             "Acceptance uses each researcher's submission attempts in years 1–10. Citations condition on publication\n"
             "and complete age-three follow-up. Activity retains all founders. No confidence intervals or causal effects.",
             fontsize=6.7, va="bottom", color=GREEN)
    saved = []
    for suffix in ("pdf", "png"):
        path = visual / f"fig_career_outcomes_green.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    outcomes.to_csv(docs / "career_outcomes_by_world.csv", index=False)
    survival.to_csv(docs / "career_survival_by_world.csv", index=False)
    metadata = {
        "post_hoc_descriptive_figure": True,
        "seeds": list(SEEDS), "scientific_config_hash": next(iter(configurations)),
        "analysis_source": fingerprint(Path(__file__)), "inputs": sources,
        "outputs": [str(p) for p in saved],
        "panel_a": {
            "outcome": "Mean researcher-level acceptance fraction over years 1–10",
            "within_world_weights": "Sum num_accepted and num_papers over all ten agent-year rows per researcher; divide each researcher's totals, then average researchers equally within strategy.",
            "attempts": "Includes repeat submission attempts; not first-attempt acceptance.",
            "zero_submission_researchers": "Undefined fractions excluded, exactly as archived researcher_level; none in these worlds.",
            "archive_check": "All nine means match explore_e4a_analysis/aggregate_results.csv at absolute tolerance 1e-12.",
            "display_units": "Percent; CSV stores fractions.",
        },
        "panel_b": {
            "outcome": "Mean citation count per accepted paper at publication age three",
            "input_rows": "Saved bin_and_strategy_outcomes.csv: sample=accepted_complete_publication_age_3, group_by=strategy, outcome=citations.",
            "cohort": "One row per accepted manuscript observed at publication age three, publication years 1–7. Later publications omitted for incomplete follow-up.",
            "within_world_weights": "Equal weight per eligible accepted paper, including zero-citation papers; prolific authors contribute more papers.",
            "raw_check": "All nine saved means and counts verified against paper_age.parquet age==3.",
            "selection": "Conditional on acceptance and complete age-three follow-up; not a yield for all submitted manuscripts and not latent impact of rejected work.",
            "display_units": "Raw citation counts per eligible accepted paper; no log transform.",
        },
        "panel_c": {
            "outcome": "Year-end active share of the founding cohort",
            "within_world_weights": "Equal weight per founding researcher within each strategy, using is_active at each year end; inactive researchers remain in the denominator.",
            "display_units": "Percent; CSV stores fractions.",
            "traces": "Thin lines are individual worlds; bold marked lines are equal-world means, without bands.",
        },
        "across_world_weights": "Equal weight 1/3 to each world within each strategy for all mean marks and curves; worlds never pooled by paper count.",
        "uncertainty": "Individual worlds displayed; no confidence intervals, p-values, or claim of precise low-n world inference.",
        "style": {"authored_ink": GREEN, "background": "white",
                  "policy_markers": dict(zip(strategies, markers)),
                  "policy_linestyles": dict(zip(strategies, styles)),
                  "point_world_order": "Within each strategy: left=1001, center=1002, right=1003"},
        "legacy_citation_warning": "agent_year.total_citations counts citations to papers in their acceptance year only; summing those annual values is not career-lifetime citation stock. The former ever_hit endpoint thresholds this sum at the within-world 90th percentile and is not probability of producing a high-impact paper. Neither legacy endpoint appears in this figure.",
        "per_world_outcomes": json.loads(outcomes.to_json(orient="records", double_precision=15)),
        "equal_world_means": json.loads(means.reset_index().to_json(orient="records", double_precision=15)),
    }
    write_json_file(metadata, docs / 'career_figure_metadata.json', trailing_newline=True, streaming=False, indent=2)
    print(means.to_string())
    print(f"Career figure: {saved[0]}")
    return saved
