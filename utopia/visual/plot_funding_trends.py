import os
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utopia.arguments import parse_arguments
from utopia.utils.data_utils import read_json
from utopia.utils.visual_utils import load_visualization_data

logger = logging.getLogger(__name__)




def plot_agent_resources(resources_df: pd.DataFrame, agent_ids, output_dir: str):
    """
    Plot resource trends over time for one or more agents, each with a distinct color.

    Args:
        resources_df: DataFrame with columns [year, agent_id, resources]
        agent_ids: str or list of agent_id values to plot
        output_dir: Directory to save the plot
    """
    if isinstance(agent_ids, str):
        agent_ids = [agent_ids]

    # Filter data for requested agents
    data = resources_df[resources_df["agent_id"].isin(agent_ids)]
    if data.empty:
        print(f"No resource data found for agents: {agent_ids}")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for multiple agents
    palette = sns.color_palette("tab10", n_colors=len(agent_ids))

    for color, agent_id in zip(palette, agent_ids):
        agent_data = data[data["agent_id"] == agent_id].sort_values("year")
        if agent_data.empty:
            print(f"No data found for agent: {agent_id}")
            continue

        # Plot line for this agent
        sns.lineplot(
            data=agent_data,
            x="year",
            y="resources",
            marker="o",
            linewidth=2,
            color=color,
            label=agent_id,
            ax=ax
        )

    # Labels and title
    ax.set(
        xlabel="Year",
        ylabel="Resources",
        title="Resource Trends by Agent Over Time"
    )
    ax.set_xticks(sorted(data["year"].unique()))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Save plot
    output_path = Path(output_dir) / "lineplot_funding_individual_trend.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved agent resources plot to: {output_path}")

    plt.close(fig)


def plot_aggregated_funding_trends(output_dir: str):
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

    output_path = Path(output_dir) / "visual" / "lineplot_funding_overall_trends.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Saved funding trends plot to: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=trends, x="year", y="gini", marker="o", linewidth=2, ax=ax)

    ax.set(xlabel="Year", ylabel="Gini Coefficient", title="Gini Coefficient of Funding")
    ax.set_xticks(sorted(trends["year"].unique()))
    ax.grid(True, linestyle="--", alpha=0.6)

    output_path = Path(output_dir) / "visual" / "lineplot_funding_gini.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Saved gini coefficient funding plot to: {output_path}")
    plt.close(fig)




if __name__ == "__main__":
    args = parse_arguments()
    
    data = load_visualization_data(args)
    simulation, loaded_year = data["simulation"], data["loaded_year"]
    
    print("Collecting funding trends...")
    resources = []  # year -> agent_id -> resources
    for year in range(loaded_year, 0, -1):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_year_{year}.json")
        if os.path.exists(checkpoint_path):
            checkpoint_data = json.load(open(checkpoint_path, 'r'))
            for agent_data in checkpoint_data['ecosystem_data']['agents']:
                if agent_data['resources'] is not None:
                    resources.append({
                        'year': year,
                        'agent_id': agent_data['id'],
                        'resources': agent_data['resources'],

                    })

    resources_df = pd.DataFrame(resources).sort_values(by=['agent_id', 'year'])

    plot_agent_resources(resources_df, 
                         agent_ids=["Google_researcher_0", "MIT_researcher_1"],
                         output_dir=args.visual_dir)
    
    
    # Plot funding trends
    plot_aggregated_funding_trends(args.output_dir)
