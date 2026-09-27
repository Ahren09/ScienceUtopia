"""Construct W&B-native charts without owning run state or logging."""

import numpy as np
from utopia.visual.embeddings import plot_paper_embeddings


def build_charts(year: int, year_results: dict, simulation, history) -> dict:
    import wandb

    plots = {}

    # Review score distribution (native wandb histogram — appears in Charts)
    all_scores = _collect_review_scores(year_results)
    if all_scores:
        plots["plots/review_score_dist"] = wandb.Histogram(all_scores)

    # Funding distribution (native wandb histogram — appears in Charts)
    funding_levels = _collect_funding_levels(simulation)
    if funding_levels:
        plots["plots/funding_distribution"] = wandb.Histogram(funding_levels)

    # Trend plots (native wandb line charts — appear in Charts)
    years_so_far = list(range(1, year + 1))
    if len(history["acceptance_rate"]) >= 2:
        plots["plots/acceptance_rate_trend"] = wandb.plot.line_series(
            xs=years_so_far,
            ys=[history["acceptance_rate"]],
            keys=["Acceptance Rate"],
            title="Acceptance Rate Trend",
            xname="Year",
        )
        plots["plots/gini_trend"] = wandb.plot.line_series(
            xs=years_so_far,
            ys=[history["gini"]],
            keys=["Gini Coefficient"],
            title="Funding Gini Trend",
            xname="Year",
        )
        plots["plots/mean_score_trend"] = wandb.plot.line_series(
            xs=years_so_far,
            ys=[history["mean_score"]],
            keys=["Mean Score"],
            title="Mean Review Score Trend",
            xname="Year",
        )

    # Embedding space PCA (exploration experiment only — uses matplotlib Image)
    if simulation.is_exploration_experiment and simulation.embedding_tracker:
        fig = plot_paper_embeddings(simulation)
        if fig:
            plots["plots/embedding_space"] = wandb.Image(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)

    return plots


def _collect_review_scores(year_results: dict) -> list:
    """Approximate individual review scores from aggregated stats."""
    score_metrics = year_results.get("peer_review", {}).get("score_metrics", {})
    mean = score_metrics.get("overall_mean", 0)
    std = score_metrics.get("overall_std", 0)
    count = score_metrics.get("total_reviews", 0)
    if count > 0 and std > 0:
        return list(np.random.normal(mean, std, count).clip(1, 10))
    elif count > 0:
        return [mean] * count
    return []


def _collect_funding_levels(simulation) -> list:
    """Collect current funding levels from all agents."""
    levels = []
    for agent in simulation.ecosystem.agent_population.values():
        if hasattr(agent, "resources") and agent.resources is not None:
            levels.append(agent.resources)
    return levels
