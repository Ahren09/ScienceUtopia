"""Plot changes in average peer review scores over time."""

import json
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import logging
import matplotlib
matplotlib.use('tkagg')  # Use non-interactive backend to prevent popups
import matplotlib.pyplot as plt
from utopia.arguments import parse_arguments

from run_simulation import Simulation

# def load_all_reviews(output_dir: str) -> pd.DataFrame:
#     """Load reviews from all checkpoint files in output directory."""
#     all_reviews = []

#     # Find all checkpoint files
#     checkpoint_files = sorted(Path(output_dir).glob("checkpoint_year_*.json"))
    
#     if len(checkpoint_files) == 0:
#         raise ValueError("No checkpoint files found")

#     checkpoint_file = checkpoint_files[-1]

#     with open(checkpoint_file, 'r') as f:
#         checkpoint_data = json.load(f)

#     year = checkpoint_data["year"]

#     # Extract reviews from conference system
#     for conf in checkpoint_data.get("conference_system", {}).get("conferences", []):
        
#         conference_id = conf.get("conference_id")
#         conference_year = conf.get("year")
        
#         for paper_id, review_list in conf.get("reviews", {}).items():
#             for review in review_list:
#                 all_reviews.append({
#                     "year": conference_year,
#                     "reviewer_id": review.get("reviewer_id"),
#                     "conference": conference_id,
#                     "score": review.get("overall_score"),
#                     "paper_id": paper_id
#                 })

#     return pd.DataFrame(all_reviews)


SCORE_FIELD = "overall_score"
CONFERENCE_FIELD = "conference_id"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def plot_researcher_scores(df: pd.DataFrame, researcher_ids, output_dir: str):
    """
    Plot average scores over time for one or more researchers, each with a distinct color and ±1 SD shading.
    researcher_ids: list or str — one or multiple reviewer_id values.
    """
    if isinstance(researcher_ids, str):
        researcher_ids = [researcher_ids]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for multiple researchers
    palette = sns.color_palette("tab10", n_colors=len(researcher_ids))

    for i, researcher_id in enumerate(researcher_ids):
        researcher_df = df[df["reviewer_id"] == researcher_id]
        if researcher_df.empty:
            print(f"No reviews found for researcher: {researcher_id}")
            continue

        # Aggregate per year
        grouped = (
            researcher_df.groupby("year")["overall_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("year")
        )
        grouped["std"] = grouped["std"].fillna(0)

        # Plot line for this researcher
        sns.lineplot(
            data=grouped,
            x="year",
            y="mean",
            marker="o",
            linewidth=2,
            color=palette[i],
            ax=ax,
            label=f"{researcher_id} (mean ± SD)"
        )

        # Shaded band for ±1 std
        ax.fill_between(
            grouped["year"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            color=palette[i]
        )

    # Labels and title
    ax.set(
        xlabel="Year",
        ylabel="Average Overall Score",
        title="Average Scores Given by Researchers Over Time"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    output_path = Path(output_dir) / "researcher_comparison_scores.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved multi-researcher plot to {output_path}")
    plt.show()
    plt.close(fig)

def plot_conference_scores(df: pd.DataFrame, conference: str, output_dir: str):
    """
    Plot average reviewer scores for a specific conference over years,
    showing mean ± 1 std as a shaded band.
    """
    conference_df = df[df[CONFERENCE_FIELD] == conference].copy()
    if conference_df.empty:
        print(f"No reviews found for conference: {conference}")
        return

    # Aggregate by year
    summary = (conference_df
               .groupby("year")[SCORE_FIELD]
               .agg(mean="mean", std="std", n="count")
               .reset_index()
               .sort_values("year"))

    # If a year has only one review, std is NaN → treat as 0 so plotting doesn't break
    summary["std"] = summary["std"].fillna(0.0)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean line/points
    sns.lineplot(
        data=summary,
        x="year",
        y="mean",
        marker="o",
        linewidth=2,
        ax=ax,
        zorder=3,
        label="Average Score",
    )

    # Shaded band: mean ± 1 SD
    ax.fill_between(
        summary["year"].values,
        (summary["mean"] - summary["std"]).values,
        (summary["mean"] + summary["std"]).values,
        alpha=0.2,
        zorder=2,
        label="±1 SD",
    )

    # Axis / grid
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Score")
    ax.set_title(f"Average Scores for {conference} Over Time")
    ax.set_xticks(sorted(summary["year"].unique()))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"conference_{conference}_scores.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved conference score plot to: {output_path}")

def plot_funding_trends(output_dir: str):
    """Plot mean funding levels over time from simulation report."""
    report_path = Path(output_dir) / "simulation_report.json"
    if not report_path.exists():
        print(f"No simulation report found at {report_path}")
        return

    with open(report_path, 'r') as f:
        report = json.load(f)

    trends = pd.DataFrame(report['ecosystem_trends'])

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=trends, x="year", y="mean_funding", marker="o", linewidth=2, ax=ax)

    ax.set(xlabel="Year", ylabel="Average Funding Level", title="Mean Funding Levels Across Years")
    ax.set_xticks(sorted(trends["year"].unique()))
    ax.grid(True, linestyle="--", alpha=0.6)

    output_path = Path(output_dir) / "visual" / "funding_trends.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved funding trends plot to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    

    
    args = parse_arguments()
    
    args.researcher = "university_researcher_0"
    args.researcher = ["university_researcher_0","university_researcher_1", "university_researcher_3"]
    args.conference = None # "AAAI"
    

    os.makedirs(args.visual_dir, exist_ok=True)
    
    
    # Load simulation data
   
    simulation = Simulation(
        llm=None,
        num_years=args.num_years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain
    )
    
    for year in range(args.num_years, 0, -1):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_year_{year}.json")
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint at year {year}, attempting to load...")
            if simulation.load_checkpoint(year):
                print(f"Loaded checkpoint for year {year}")
                break

    # reviews_df = load_all_reviews(args.output_dir)
    reviews = []
    for author_id, paper_dict in simulation.paper_archive.papers_by_author.items():
        for paper in paper_dict:
            review_history = paper.review_history.copy()
            for i in range(len(review_history)):
                for j in range(len(review_history[i]['reviews'])):
                    review_history[i]['reviews'][j]['year'] = review_history[i]['year']
                    review_history[i]['reviews'][j]['arxiv_id'] = paper.id
                    reviews.append(review_history[i]['reviews'][j])
                    
                    

                
            
    
    reviews_df = pd.DataFrame(reviews).dropna(axis=1)

    assert not reviews_df.empty, "No reviews found in checkpoint files"

    print(f"Loaded {len(reviews_df)} reviews across {reviews_df['year'].nunique()} years")
    # Always plot funding trends
    plot_funding_trends(args.output_dir)

    if args.researcher:
        plot_researcher_scores(reviews_df, args.researcher, args.visual_dir)

    if args.conference:
        plot_conference_scores(reviews_df, args.conference, args.visual_dir)


    if not args.researcher and not args.conference:
        print("Please specify --researcher and/or --conference to plot")
