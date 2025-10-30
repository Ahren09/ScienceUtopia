"""Plot changes in average peer review scores over time."""

import json
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_checkpoints(output_dir: str) -> list:
    """Load all checkpoint files from output directory."""
    checkpoints = []
    for file in sorted(Path(output_dir).glob("checkpoint_year_*.json")):
        with open(file) as f:
            checkpoints.append(json.load(f))
    return checkpoints


def extract_reviews(checkpoints: list) -> pd.DataFrame:
    """Extract all reviews from checkpoints into a DataFrame."""
    reviews = []
    for checkpoint in checkpoints:
        year = checkpoint["year"]
        for conf in checkpoint.get("conference_system", {}).get("conferences", []):
            conference_id = conf.get("conference_id")
            for paper_id, review_list in conf.get("reviews", {}).items():
                for review in review_list:
                    reviews.append({
                        "year": year,
                        "reviewer_id": review.get("reviewer_id"),
                        "conference": conference_id,
                        "score": review.get("overall_score"),
                        "paper_id": paper_id
                    })
    return pd.DataFrame(reviews)


def plot_researcher_scores(df: pd.DataFrame, researcher_id: str, output_dir: str):
    """Plot average scores given BY a specific researcher over time."""
    researcher_df = df[df["reviewer_id"] == researcher_id]
    if researcher_df.empty:
        print(f"No reviews found for researcher: {researcher_id}")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn with errorbar for mean ± std
    sns.lineplot(data=researcher_df, x="year", y="score", marker="o",
                 errorbar="sd", err_style="band", ax=ax)
    ax.set(xlabel="Year", ylabel="Average Score",
           title=f"Average Scores Given by {researcher_id}")

    output_path = Path(output_dir) / f"researcher_{researcher_id}_scores.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved researcher plot to {output_path}")
    plt.show()
    plt.close(fig)


def plot_conference_scores(df: pd.DataFrame, conference: str, output_dir: str):
    """Plot average scores OF a specific conference over time."""
    conference_df = df[df["conference"] == conference]
    if conference_df.empty:
        print(f"No reviews found for conference: {conference}")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use seaborn with errorbar for mean ± std
    sns.lineplot(data=conference_df, x="year", y="score", marker="o",
                 errorbar="sd", err_style="band", ax=ax)
    ax.set(xlabel="Year", ylabel="Average Score",
           title=f"Average Scores for {conference} Conference")

    output_path = Path(output_dir) / f"conference_{conference}_scores.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved conference plot to {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot peer review scores from checkpoints")
    parser.add_argument("--output-dir", default="outputs", help="Directory with checkpoint files")
    parser.add_argument("--researcher", help="Researcher ID to plot (e.g., university_researcher_0)")
    parser.add_argument("--conference", help="Conference to plot (e.g., AAAI)")
    parser.add_argument("--visual-dir", default="outputs/visual", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.visual_dir, exist_ok=True)

    checkpoints = load_checkpoints(args.output_dir)
    reviews_df = extract_reviews(checkpoints)

    if reviews_df.empty:
        print("No reviews found in checkpoint files")
    else:
        print(f"Loaded {len(reviews_df)} reviews from {len(checkpoints)} checkpoints")

        if args.researcher:
            plot_researcher_scores(reviews_df, args.researcher, args.visual_dir)

        if args.conference:
            plot_conference_scores(reviews_df, args.conference, args.visual_dir)

        if not args.researcher and not args.conference:
            print("Please specify --researcher and/or --conference to plot")
