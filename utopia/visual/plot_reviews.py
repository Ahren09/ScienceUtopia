"""Plot changes in average peer review scores over time."""

import json
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

from utopia.utils.generail_utils import project_setup
from utopia.arguments import parse_arguments
from utopia.utils.visual_utils import load_visualization_data


SCORE_FIELD = "score"
CONFERENCE_FIELD = "conference"
STATUS_FIELD = "status"

logger = logging.getLogger(__name__)
matplotlib.use('Agg')  # Use non-interactive backend for headless environments


def plot_researcher_scores(df: pd.DataFrame, researcher_ids, output_dir: str):
    """
    Plot average scores over time for one or more researchers, each with a distinct color and ±1 SD shading.
    researcher_ids: list or str — one or multiple reviewer_id values.
    """
    if isinstance(researcher_ids, str):
        researcher_ids = [researcher_ids]

    # Static matplotlib plot
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
            researcher_df.groupby("year")["score"]
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
    plt.close(fig)

    # Interactive Plotly plot
    plotly_fig = go.Figure()
    # Extended color palette (tab10 + tab20 additional colors)
    plotly_palette = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
        'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
        # Additional colors from tab20
        'rgb(174, 199, 232)', 'rgb(255, 187, 120)', 'rgb(152, 223, 138)',
        'rgb(255, 152, 150)', 'rgb(197, 176, 213)', 'rgb(196, 156, 148)',
        'rgb(247, 182, 210)', 'rgb(199, 199, 199)', 'rgb(219, 219, 141)', 'rgb(158, 218, 229)',
        # Extra vibrant colors
        'rgb(255, 99, 71)', 'rgb(60, 179, 113)', 'rgb(106, 90, 205)',
        'rgb(255, 165, 0)', 'rgb(30, 144, 255)', 'rgb(220, 20, 60)',
        'rgb(0, 206, 209)', 'rgb(255, 215, 0)', 'rgb(138, 43, 226)', 'rgb(34, 139, 34)'
    ]

    for i, researcher_id in enumerate(researcher_ids):
        researcher_df = df[df["reviewer_id"] == researcher_id]
        if researcher_df.empty:
            continue

        grouped = (
            researcher_df.groupby("year")["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("year")
        )
        grouped["std"] = grouped["std"].fillna(0)

        color = plotly_palette[i % len(plotly_palette)]
        color_rgba = color.replace('rgb', 'rgba').replace(')', ', 0.2)')

        # Add upper bound (invisible)
        plotly_fig.add_trace(go.Scatter(
            x=grouped["year"],
            y=grouped["mean"] + grouped["std"],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name=f'{researcher_id}_upper',
            legendgroup=researcher_id
        ))

        # Add lower bound with fill
        plotly_fig.add_trace(go.Scatter(
            x=grouped["year"],
            y=grouped["mean"] - grouped["std"],
            mode='lines',
            line=dict(width=0),
            fillcolor=color_rgba,
            fill='tonexty',
            showlegend=False,
            name=f'{researcher_id}_lower',
            legendgroup=researcher_id,
            hovertemplate=f'{researcher_id}<br>Year: %{{x}}<br>Mean - SD: %{{y:.2f}}<extra></extra>'
        ))

        # Add mean line
        plotly_fig.add_trace(go.Scatter(
            x=grouped["year"],
            y=grouped["mean"],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            name=f'{researcher_id}',
            legendgroup=researcher_id,
            hovertemplate=f'{researcher_id}<br>Year: %{{x}}<br>Mean: %{{y:.2f}}<br>SD: {grouped["std"].values}<extra></extra>'
        ))

    plotly_fig.update_layout(
        title="Average Scores Given by Researchers Over Time",
        xaxis_title="Year",
        yaxis_title="Average Overall Score",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )

    html_output_path = Path(output_dir) / "researcher_comparison_scores.html"
    plotly_fig.write_html(html_output_path)
    print(f"✅ Saved interactive multi-researcher plot to {html_output_path}")


def plot_conference_scores(df: pd.DataFrame, conferences, top_k: int = 3, output_dir: str = "."):
    """
    Plot average reviewer scores over years for one or more conferences,
    each with its own colored mean line and ±1 SD shaded band.

    `conferences` can be a single str or a list of conference names.
    Uses CONFERENCE_FIELD and SCORE_FIELD constants from your code.
    """
    # Normalize input to list
    if isinstance(conferences, str):
        conferences = [conferences]

    # Filter to requested conferences only
    data = df[df[CONFERENCE_FIELD].isin(conferences)].copy()
    if data.empty:
        print(f"No reviews found for conferences: {conferences}")
        return

    # Aggregate: mean/std/count per conference per year
    summary = (
        data.groupby([CONFERENCE_FIELD, "year"])[SCORE_FIELD]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(["year", CONFERENCE_FIELD])
    )
    summary["std"] = summary["std"].fillna(0.0)

    # Static matplotlib plot
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette (one color per conference)
    palette = sns.color_palette("tab10", n_colors=min(len(conferences), top_k))
    years_all = sorted(summary["year"].unique())

    # Plot each conference separately so we can add its own shaded band
    for color, conf in zip(palette, conferences[:top_k]):
        sub = summary[summary[CONFERENCE_FIELD] == conf]

        # Line + markers (mean)
        sns.lineplot(
            data=sub,
            x="year",
            y="mean",
            marker="o",
            linewidth=2,
            ax=ax,
            color=color,
            label=f"{conf} (mean)"
        )

        # Shaded ±1 SD
        ax.fill_between(
            sub["year"].values,
            (sub["mean"] - sub["std"]).values,
            (sub["mean"] + sub["std"]).values,
            alpha=0.18,
            color=color,
            # label=f"{conf} ±1 SD"
        )

    # Formatting
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Score")
    title = "Average Reviewer Scores Over Time"
    if len(conferences) == 1:
        title = f"Average Scores for {conferences[0]} Over Time"
    ax.set_title(title)
    ax.set_xticks(years_all)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(conferences) == 1:
        outname = f"conference_{conferences[0]}_scores.png"
    else:
        safe_names = "-".join(str(c) for c in conferences[:3])
        if len(conferences) > 3:
            safe_names += "-etc"
        outname = f"conference_review_scores_{safe_names}.png"
    output_path = output_dir / outname
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved conference score plot to: {output_path}")
    plt.close(fig)

    # Interactive Plotly plot
    plotly_fig = go.Figure()
    # Extended color palette (tab10 + tab20 additional colors)
    plotly_palette = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
        'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
        # Additional colors from tab20
        'rgb(174, 199, 232)', 'rgb(255, 187, 120)', 'rgb(152, 223, 138)',
        'rgb(255, 152, 150)', 'rgb(197, 176, 213)', 'rgb(196, 156, 148)',
        'rgb(247, 182, 210)', 'rgb(199, 199, 199)', 'rgb(219, 219, 141)', 'rgb(158, 218, 229)',
        # Extra vibrant colors
        'rgb(255, 99, 71)', 'rgb(60, 179, 113)', 'rgb(106, 90, 205)',
        'rgb(255, 165, 0)', 'rgb(30, 144, 255)', 'rgb(220, 20, 60)',
        'rgb(0, 206, 209)', 'rgb(255, 215, 0)', 'rgb(138, 43, 226)', 'rgb(34, 139, 34)'
    ]

    for i, conf in enumerate(conferences):
        sub = summary[summary[CONFERENCE_FIELD] == conf]
        if sub.empty:
            continue

        color = plotly_palette[i % len(plotly_palette)]
        color_rgba = color.replace('rgb', 'rgba').replace(')', ', 0.18)')

        # Add upper bound (invisible)
        plotly_fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub["mean"] + sub["std"],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name=f'{conf}_upper',
            legendgroup=conf
        ))

        # Add lower bound with fill
        plotly_fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub["mean"] - sub["std"],
            mode='lines',
            line=dict(width=0),
            fillcolor=color_rgba,
            fill='tonexty',
            showlegend=False,
            name=f'{conf}_lower',
            legendgroup=conf,
            hovertemplate=f'{conf}<br>Year: %{{x}}<br>Mean - SD: %{{y:.2f}}<extra></extra>'
        ))

        # Add mean line
        plotly_fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub["mean"],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            name=f'{conf}',
            legendgroup=conf,
            hovertemplate=f'{conf}<br>Year: %{{x}}<br>Mean: %{{y:.2f}}<extra></extra>'
        ))

    title = "Average Reviewer Scores Over Time"
    if len(conferences) == 1:
        title = f"Average Scores for {conferences[0]} Over Time"

    plotly_fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Average Score",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )

    if len(conferences) == 1:
        html_outname = f"conference_{conferences[0]}_scores.html"
    else:
        safe_names = "-".join(str(c) for c in conferences[:3])
        if len(conferences) > 3:
            safe_names += "-etc"
    html_outname = f"conference_review_scores.html"
    html_output_path = output_dir / html_outname
    plotly_fig.write_html(html_output_path)
    print(f"✅ Saved interactive conference score plot to: {html_output_path}")


def plot_submission_volume(aggregated_df: pd.DataFrame, reviews_df: pd.DataFrame = None,
                           conferences=None, top_k: int = 3, output_dir: str = "."):
    """Plot submission volume over time using aggregated review metrics.

    Args:
        aggregated_df: DataFrame with columns [year, arxiv_id, ...]
        reviews_df: Optional DataFrame with [year, arxiv_id, CONFERENCE_FIELD] for conference plots
        conferences: None for aggregate, or list of conference names
        output_dir: Directory to save plots
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    if conferences is None:
        # Aggregate: count all submission events per year (including resubmissions)
        submissions = aggregated_df.groupby("year").size().reset_index()
        submissions.columns = ["year", "submissions"]

        # Static matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=submissions, x="year", y="submissions", marker="o", linewidth=2, ax=ax)
        ax.set(xlabel="Year", ylabel="Number of Submissions", title="Aggregate Submission Volume Over Time")
        ax.set_xticks(sorted(submissions["year"].unique()))
        ax.grid(True, linestyle="--", alpha=0.6)

        output_path = Path(output_dir) / "submission_volume_aggregate.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved aggregate submission volume to: {output_path}")
        plt.close(fig)

        # Interactive Plotly plot
        plotly_fig = go.Figure()
        plotly_fig.add_trace(go.Scatter(
            x=submissions["year"],
            y=submissions["submissions"],
            mode='lines+markers',
            line=dict(color='rgb(31, 119, 180)', width=2),
            marker=dict(size=8),
            name='Submissions',
            hovertemplate='Year: %{x}<br>Submissions: %{y}<extra></extra>'
        ))

        plotly_fig.update_layout(
            title="Aggregate Submission Volume Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Submissions",
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )

        html_output_path = Path(output_dir) / "submission_volume_aggregate.html"
        plotly_fig.write_html(html_output_path)
        print(f"✅ Saved interactive aggregate submission volume to: {html_output_path}")

    else:
        data = aggregated_df[aggregated_df[CONFERENCE_FIELD].isin(conferences)]
        # Count all submission events per conference per year (including resubmissions)
        submissions = data.groupby([CONFERENCE_FIELD, "year"]).size().reset_index()
        submissions.columns = [CONFERENCE_FIELD, "year", "submissions"]

        # Static matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("tab10", n_colors=min(len(conferences), top_k))

        for color, conf in zip(palette, conferences[:top_k]):
            conf_data = submissions[submissions[CONFERENCE_FIELD] == conf]
            sns.lineplot(data=conf_data, x="year", y="submissions", marker="o",
                         linewidth=2, color=color, label=conf, ax=ax)

        ax.set(xlabel="Year", ylabel="Number of Submissions",
               title="Submission Volume by Conference Over Time")
        ax.set_xticks(sorted(submissions["year"].unique()))
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        output_path = Path(output_dir) / "submission_volume_by_conference.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved conference submission volume to: {output_path}")
        plt.close(fig)

        # Interactive Plotly plot
        plotly_fig = go.Figure()
        # Extended color palette (tab10 + tab20 additional colors)
        plotly_palette = [
            'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
            'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
            # Additional colors from tab20
            'rgb(174, 199, 232)', 'rgb(255, 187, 120)', 'rgb(152, 223, 138)',
            'rgb(255, 152, 150)', 'rgb(197, 176, 213)', 'rgb(196, 156, 148)',
            'rgb(247, 182, 210)', 'rgb(199, 199, 199)', 'rgb(219, 219, 141)', 'rgb(158, 218, 229)',
            # Extra vibrant colors
            'rgb(255, 99, 71)', 'rgb(60, 179, 113)', 'rgb(106, 90, 205)',
            'rgb(255, 165, 0)', 'rgb(30, 144, 255)', 'rgb(220, 20, 60)',
            'rgb(0, 206, 209)', 'rgb(255, 215, 0)', 'rgb(138, 43, 226)', 'rgb(34, 139, 34)'
        ]

        # Add individual conference lines
        for i, conf in enumerate(conferences):
            conf_data = submissions[submissions[CONFERENCE_FIELD] == conf]
            color = plotly_palette[i % len(plotly_palette)]

            plotly_fig.add_trace(go.Scatter(
                x=conf_data["year"],
                y=conf_data["submissions"],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                name=conf,
                hovertemplate=f'{conf}<br>Year: %{{x}}<br>Submissions: %{{y}}<extra></extra>'
            ))

        # Add total submissions line (sum across all conferences)
        total_submissions = submissions.groupby("year")["submissions"].sum().reset_index()
        total_submissions.columns = ["year", "total_submissions"]

        plotly_fig.add_trace(go.Scatter(
            x=total_submissions["year"],
            y=total_submissions["total_submissions"],
            mode='lines+markers',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond'),
            name='Total (All Conferences)',
            hovertemplate='Total<br>Year: %{x}<br>Submissions: %{y}<extra></extra>'
        ))

        plotly_fig.update_layout(
            title="Submission Volume by Conference Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Submissions",
            hovermode='x unified',
            template='plotly_white',
            width=1000,
            height=600
        )

        html_output_path = Path(output_dir) / "submission_volume_by_conference.html"
        plotly_fig.write_html(html_output_path)
        print(f"✅ Saved interactive conference submission volume to: {html_output_path}")


def plot_given_up_paper_scores(aggregated_df: pd.DataFrame, paper_ids, output_dir: str = "."):
    """Plot average scores over time for papers given up by authors.

    Args:
        aggregated_df: DataFrame with columns [year, arxiv_id, mean_score, ...]
        paper_ids: List of arxiv_ids to plot
        output_dir: Directory to save plots
    """
    if isinstance(paper_ids, str):
        paper_ids = [paper_ids]

    data = aggregated_df[aggregated_df["arxiv_id"].isin(paper_ids)]
    if data.empty:
        print(f"No data found for papers: {paper_ids}")
        return

    # Static matplotlib plot
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_colors=len(paper_ids))

    for color, paper_id in zip(palette, paper_ids):
        paper_data = data[data["arxiv_id"] == paper_id]
        short_id = paper_id.split("/")[-1] if "/" in paper_id else paper_id
        sns.lineplot(data=paper_data, x="year", y="mean_score", marker="o",
                     linewidth=2, color=color, label=short_id, ax=ax)

        # Add shaded band for ±1 std
        ax.fill_between(paper_data["year"].values,
                        (paper_data["mean_score"] - paper_data["std_score"]).values,
                        (paper_data["mean_score"] + paper_data["std_score"]).values,
                        alpha=0.2, color=color)

    ax.set(xlabel="Year", ylabel="Average Review Score",
           title="Average Scores for Papers Given Up by Authors")
    ax.set_xticks(sorted(data["year"].unique()))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    output_path = Path(output_dir) / "given_up_paper_scores.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved given-up paper scores to: {output_path}")
    plt.close(fig)

    # Interactive Plotly plot
    plotly_fig = go.Figure()
    # Extended color palette (tab10 + tab20 additional colors)
    plotly_palette = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
        'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
        # Additional colors from tab20
        'rgb(174, 199, 232)', 'rgb(255, 187, 120)', 'rgb(152, 223, 138)',
        'rgb(255, 152, 150)', 'rgb(197, 176, 213)', 'rgb(196, 156, 148)',
        'rgb(247, 182, 210)', 'rgb(199, 199, 199)', 'rgb(219, 219, 141)', 'rgb(158, 218, 229)',
        # Extra vibrant colors
        'rgb(255, 99, 71)', 'rgb(60, 179, 113)', 'rgb(106, 90, 205)',
        'rgb(255, 165, 0)', 'rgb(30, 144, 255)', 'rgb(220, 20, 60)',
        'rgb(0, 206, 209)', 'rgb(255, 215, 0)', 'rgb(138, 43, 226)', 'rgb(34, 139, 34)'
    ]

    for i, paper_id in enumerate(paper_ids):
        paper_data = data[data["arxiv_id"] == paper_id]
        if paper_data.empty:
            continue

        short_id = paper_id.split("/")[-1] if "/" in paper_id else paper_id
        color = plotly_palette[i % len(plotly_palette)]
        color_rgba = color.replace('rgb', 'rgba').replace(')', ', 0.2)')

        # Add upper bound (invisible)
        plotly_fig.add_trace(go.Scatter(
            x=paper_data["year"],
            y=paper_data["mean_score"] + paper_data["std_score"],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name=f'{short_id}_upper',
            legendgroup=short_id
        ))

        # Add lower bound with fill
        plotly_fig.add_trace(go.Scatter(
            x=paper_data["year"],
            y=paper_data["mean_score"] - paper_data["std_score"],
            mode='lines',
            line=dict(width=0),
            fillcolor=color_rgba,
            fill='tonexty',
            showlegend=False,
            name=f'{short_id}_lower',
            legendgroup=short_id,
            hovertemplate=f'{short_id}<br>Year: %{{x}}<br>Mean - SD: %{{y:.2f}}<extra></extra>'
        ))

        # Add mean line
        plotly_fig.add_trace(go.Scatter(
            x=paper_data["year"],
            y=paper_data["mean_score"],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            name=short_id,
            legendgroup=short_id,
            hovertemplate=f'{short_id}<br>Year: %{{x}}<br>Mean: %{{y:.2f}}<extra></extra>'
        ))

    plotly_fig.update_layout(
        title="Average Scores for Papers Given Up by Authors",
        xaxis_title="Year",
        yaxis_title="Average Review Score",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )

    html_output_path = Path(output_dir) / "given_up_paper_scores.html"
    plotly_fig.write_html(html_output_path)
    print(f"✅ Saved interactive given-up paper scores to: {html_output_path}")


def plot_author_submission_volume(aggregated_df: pd.DataFrame, top_k: int = 10, output_dir: str = "."):
    """Plot submission volume over time for individual authors.

    Args:
        aggregated_df: DataFrame with columns [year, arxiv_id, author_id, ...]
        top_k: Number of top authors to plot in matplotlib (default: 10)
        output_dir: Directory to save plots
    """
    # Count total submission events per author (including resubmissions)
    author_submission_counts = (
        aggregated_df.groupby("author_id")
        .size()
        .sort_values(ascending=False)
    )

    if author_submission_counts.empty:
        print("No author submission data found")
        return

    print(f"Top {min(top_k, len(author_submission_counts))} authors by total submissions:")
    print(author_submission_counts.head(top_k))

    # Get submission events per author per year (including resubmissions)
    author_year_submissions = (
        aggregated_df.groupby(["author_id", "year"])
        .size()
        .reset_index()
    )
    author_year_submissions.columns = ["author_id", "year", "submissions"]

    # Top k authors for matplotlib
    top_authors = author_submission_counts.head(top_k).index.tolist()
    top_data = author_year_submissions[author_year_submissions["author_id"].isin(top_authors)]

    # Static matplotlib plot (top k authors only)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=min(len(top_authors), 10))

    for i, author_id in enumerate(top_authors[:10]):  # Limit to 10 for readability
        author_data = top_data[top_data["author_id"] == author_id]
        color = palette[i % len(palette)]
        sns.lineplot(
            data=author_data,
            x="year",
            y="submissions",
            marker="o",
            linewidth=2,
            color=color,
            label=author_id,
            ax=ax
        )

    ax.set(
        xlabel="Year",
        ylabel="Number of Submissions",
        title=f"Submission Volume for Top {min(top_k, 10)} Authors Over Time"
    )
    ax.set_xticks(sorted(author_year_submissions["year"].unique()))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_path = Path(output_dir) / "author_submission_volume.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved author submission volume plot (top {min(top_k, 10)}) to: {output_path}")
    plt.close(fig)

    # Interactive Plotly plot (ALL authors)
    all_authors = author_submission_counts.index.tolist()
    plotly_fig = go.Figure()

    # Extended color palette
    plotly_palette = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
        'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
        'rgb(174, 199, 232)', 'rgb(255, 187, 120)', 'rgb(152, 223, 138)',
        'rgb(255, 152, 150)', 'rgb(197, 176, 213)', 'rgb(196, 156, 148)',
        'rgb(247, 182, 210)', 'rgb(199, 199, 199)', 'rgb(219, 219, 141)', 'rgb(158, 218, 229)',
        'rgb(255, 99, 71)', 'rgb(60, 179, 113)', 'rgb(106, 90, 205)',
        'rgb(255, 165, 0)', 'rgb(30, 144, 255)', 'rgb(220, 20, 60)',
        'rgb(0, 206, 209)', 'rgb(255, 215, 0)', 'rgb(138, 43, 226)', 'rgb(34, 139, 34)'
    ]

    for i, author_id in enumerate(all_authors):
        author_data = author_year_submissions[author_year_submissions["author_id"] == author_id]
        if author_data.empty:
            continue

        color = plotly_palette[i % len(plotly_palette)]

        # Set visibility: top k visible by default, rest hidden
        visible = True if i < top_k else 'legendonly'

        plotly_fig.add_trace(go.Scatter(
            x=author_data["year"],
            y=author_data["submissions"],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=6),
            name=author_id,
            visible=visible,
            hovertemplate=f'{author_id}<br>Year: %{{x}}<br>Submissions: %{{y}}<extra></extra>'
        ))

    plotly_fig.update_layout(
        title=f"Submission Volume by Author Over Time (Top {top_k} shown, all {len(all_authors)} toggleable)",
        xaxis_title="Year",
        yaxis_title="Number of Submissions",
        hovermode='x unified',
        template='plotly_white',
        width=1200,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )

    html_output_path = Path(output_dir) / "submission_volume_by_author.html"
    plotly_fig.write_html(html_output_path)
    print(f"✅ Saved interactive author submission volume plot (all {len(all_authors)} authors) to: {html_output_path}")


def plot_individual_review_scores(df: pd.DataFrame, output_dir: str = "."):
    """Plot mean ± std number of reviews per reviewer over time."""
    reviews_per_reviewer_year = (
        df.groupby(["year", "reviewer_id"])
        .size()
        .reset_index(name="num_reviews")
    )

    agg_stats = (
        reviews_per_reviewer_year
        .groupby("year")["num_reviews"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Static matplotlib plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=agg_stats, x="year", y="mean", marker="o", ax=ax, color="tab:blue", label="Mean #Reviews")
    ax.fill_between(
        agg_stats["year"],
        agg_stats["mean"] - agg_stats["std"],
        agg_stats["mean"] + agg_stats["std"],
        alpha=0.3,
        color="tab:blue",
        label="±1 Stdev"
    )

    ax.set(
        title="Average Number of Reviews per Reviewer Over Years",
        xlabel="Year",
        ylabel="Number of Reviews"
    )
    ax.legend()
    plt.tight_layout()

    output_path = Path(output_dir) / "individual_review_scores.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved individual review scores plot to: {output_path}")
    plt.close(fig)

    # Interactive Plotly plot
    plotly_fig = go.Figure()

    # Add upper bound of shaded area (mean + std)
    plotly_fig.add_trace(go.Scatter(
        x=agg_stats["year"],
        y=agg_stats["mean"] + agg_stats["std"],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
        name='upper_bound'
    ))

    # Add lower bound of shaded area (mean - std) with fill
    plotly_fig.add_trace(go.Scatter(
        x=agg_stats["year"],
        y=agg_stats["mean"] - agg_stats["std"],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(31, 119, 180, 0.3)',
        fill='tonexty',
        showlegend=True,
        name='±1 Stdev',
        hovertemplate='Year: %{x}<br>Lower bound: %{y:.2f}<extra></extra>'
    ))

    # Add mean line
    plotly_fig.add_trace(go.Scatter(
        x=agg_stats["year"],
        y=agg_stats["mean"],
        mode='lines+markers',
        line=dict(color='rgb(31, 119, 180)', width=2),
        marker=dict(size=8),
        name='Mean #Reviews',
        hovertemplate='Year: %{x}<br>Mean: %{y:.2f}<extra></extra>'
    ))

    plotly_fig.update_layout(
        title="Average Number of Reviews per Reviewer Over Years",
        xaxis_title="Year",
        yaxis_title="Number of Reviews",
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )

    html_output_path = Path(output_dir) / "individual_review_scores.html"
    plotly_fig.write_html(html_output_path)
    print(f"✅ Saved interactive individual review scores plot to: {html_output_path}")

if __name__ == "__main__":
    project_setup()

    args = parse_arguments()

    args.researcher = ["Stanford_researcher_0", "MIT_researcher_1", "Google_researcher_1"]
    # args.conference = ["AAAI", "CVPR"]
    args.conference = ["ASPLOS", "AAMAS"]
    
    data = load_visualization_data(args)
    simulation, loaded_year = data["simulation"], data["loaded_year"]

    print("Collecting paper history ...")

    # Aggregated review scores for each paper
    aggregated_review_metrics = []

    # Individual review scores
    individual_review_metrics = []

    for author_id, paper_dict in simulation.paper_tracker.papers_by_author.items():
        for paper in paper_dict:
            review_history = paper.review_history.copy()
            for i in range(len(review_history)):

                scores = [review["overall_score"] for review in review_history[i]['reviews']]

                for review in review_history[i]['reviews']:
                    individual_review_metrics.append({
                        'year': review_history[i]['year'],
                        'arxiv_id': paper.id,
                        'score': review["overall_score"],
                        CONFERENCE_FIELD: review_history[i][CONFERENCE_FIELD],
                        STATUS_FIELD: paper.status,
                        'author_id': author_id,
                        'reviewer_id': review['reviewer_id']
                    })

                aggregated_review_metrics.append({
                    'year': review_history[i]['year'],
                    'arxiv_id': paper.id,
                    'scores': scores,
                    CONFERENCE_FIELD: review_history[i][CONFERENCE_FIELD],
                    STATUS_FIELD: paper.status,
                    'author_id': author_id
                })

    aggregated_review_metrics_df = pd.DataFrame(aggregated_review_metrics).sort_values(by=['arxiv_id', 'year'])

    aggregated_review_metrics_df['mean_score'] = aggregated_review_metrics_df['scores'].apply(lambda x: np.mean(x))
    aggregated_review_metrics_df['std_score'] = aggregated_review_metrics_df['scores'].apply(lambda x: np.std(x))

    reviews_df = pd.DataFrame(individual_review_metrics).sort_values(by=['reviewer_id', 'year', CONFERENCE_FIELD])

    plot_individual_review_scores(reviews_df, args.visual_dir)

    
    latest_year = reviews_df["year"].max()

    assert not reviews_df.empty, "No reviews found in checkpoint files"

    print(f"Loaded {len(reviews_df)} reviews across {reviews_df['year'].nunique()} years")

    print("Collecting papers that authors gave up on...")

    latest_reviews = (
        aggregated_review_metrics_df.sort_values("year")
        .groupby("arxiv_id", as_index=False)
        .last()  # gets the row with the highest 'year'
    )

    gave_up_df = latest_reviews[
        (latest_reviews["status"].str.lower() == "reject") & (latest_reviews["year"] != latest_year)]
    papers_given_up_by_authors = gave_up_df.groupby("arxiv_id")["arxiv_id"].nunique()

    if len(papers_given_up_by_authors) > 0:
        # Plot given-up papers (sample first 5)

        plot_given_up_paper_scores(aggregated_review_metrics_df, papers_given_up_by_authors.index.tolist()[:3],
                                   output_dir=args.visual_dir)

    # Plot submission volumes
    plot_submission_volume(aggregated_review_metrics_df, conferences=None, output_dir=args.visual_dir)  # Aggregate
    plot_submission_volume(aggregated_review_metrics_df, conferences=args.conference,
                           output_dir=args.visual_dir)  # By conference

    # Plot author submission volumes
    print("\nPlotting author submission volumes...")
    plot_author_submission_volume(aggregated_review_metrics_df, top_k=10, output_dir=args.visual_dir)

    print("Reviewers with most assignments:")
    reviewers_with_most_assignments = reviews_df.groupby('reviewer_id')['arxiv_id'].count().sort_values(ascending=False)
    print(reviewers_with_most_assignments)
    researcher_ids = reviewers_with_most_assignments.index[:6].tolist()

    plot_researcher_scores(reviews_df, researcher_ids, args.visual_dir)
        
    conferences_with_most_assignments = reviews_df.groupby('conference')['arxiv_id'].count().sort_values(ascending=False)
    print(conferences_with_most_assignments)
    conference_ids = conferences_with_most_assignments.index.tolist()

    plot_conference_scores(reviews_df, conference_ids[:3], args.visual_dir)
    plot_conference_scores(reviews_df, conference_ids[3:6], args.visual_dir)
    
    print("Done!")