"""Orchestrator for generating all visualizations from a simulation object."""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationOrchestrator:
    """Orchestrator for generating all visualization plots from a simulation object."""

    def __init__(self, simulation, output_dir: str):
        """
        Initialize the visualization orchestrator.

        Args:
            simulation: Simulation object with paper_tracker, citation_tracker, etc.
            output_dir: Base output directory for the simulation
        """
        self.simulation = simulation
        self.output_dir = output_dir
        self.visual_dir = Path(output_dir) / "visual"
        self.visual_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self):
        """Generate all visualization plots."""
        logger.info("Generating visualizations...")

        # Generate review-related plots

        self._generate_review_plots()

        # Generate funding-related plots
        self._generate_funding_plots()

        logger.info(f"All visualizations saved to {self.visual_dir}")

    def _generate_review_plots(self):
        """Generate peer review visualization plots."""
        from utopia.visual.plot_reviews import (
            plot_researcher_scores,
            plot_conference_scores,
            plot_submission_volume,
            plot_given_up_paper_scores,
            plot_individual_review_scores,
            plot_author_submission_volume
        )

        # Build review dataframes (same logic as plot_reviews.py main())
        aggregated_metrics, individual_metrics = self._build_review_dataframes()

        if not aggregated_metrics:
            logger.warning("No review data found - skipping review plots")
            return

        aggregated_df = pd.DataFrame(aggregated_metrics)
        aggregated_df['mean_score'] = aggregated_df['scores'].apply(np.mean)
        aggregated_df['std_score'] = aggregated_df['scores'].apply(np.std)

        reviews_df = pd.DataFrame(individual_metrics)

        if reviews_df.empty:
            logger.warning("No reviews found - skipping review plots")
            return

        logger.info(f"Loaded {len(reviews_df)} reviews across {reviews_df['year'].nunique()} years")

        # Plot individual review scores
        plot_individual_review_scores(reviews_df, str(self.visual_dir))

        # Plot submission volumes
        plot_submission_volume(aggregated_df, conferences=None, output_dir=str(self.visual_dir))

        # Plot author submission volumes
        logger.info("Plotting author submission volumes...")
        plot_author_submission_volume(aggregated_df, top_k=10, output_dir=str(self.visual_dir))

        # Plot top reviewers
        top_reviewers = reviews_df.groupby('reviewer_id')['arxiv_id'].count().sort_values(ascending=False).index[:6].tolist()
        if top_reviewers:
            plot_researcher_scores(reviews_df, top_reviewers, str(self.visual_dir))

        # Plot top conferences
        top_conferences = reviews_df.groupby('conference')['arxiv_id'].count().sort_values(ascending=False).index.tolist()
        if top_conferences:
            plot_conference_scores(reviews_df, top_conferences, top_k=3, output_dir=str(self.visual_dir))

        # Plot submission volume by conference
        if top_conferences:
            plot_submission_volume(aggregated_df, conferences=top_conferences, top_k=3, output_dir=str(self.visual_dir))

        # Plot given-up papers
        latest_year = reviews_df["year"].max()
        latest_reviews = aggregated_df.sort_values("year").groupby("arxiv_id", as_index=False).last()
        gave_up_df = latest_reviews[(latest_reviews["status"].str.lower() == "reject") & (latest_reviews["year"] != latest_year)]

        if not gave_up_df.empty:
            papers_given_up = gave_up_df["arxiv_id"].tolist()[:3]
            plot_given_up_paper_scores(aggregated_df, papers_given_up, output_dir=str(self.visual_dir))

    def _generate_funding_plots(self):
        """Generate funding-related visualization plots."""
        from utopia.visual.plot_funding_trends import plot_aggregated_funding_trends

        # Plot aggregated funding trends (reads from simulation_report.json)
        plot_aggregated_funding_trends(self.output_dir)

    def _build_review_dataframes(self):
        """Build aggregated and individual review dataframes from simulation."""
        aggregated_metrics = []
        individual_metrics = []

        for author_id, paper_dict in self.simulation.paper_tracker.papers_by_author.items():
            for paper in paper_dict:
                review_history = paper.review_history.copy()
                if not len(review_history) == 1:
                    print()
                for entry in review_history:
                    scores = [review["overall_score"] for review in entry['reviews']]

                    for review in entry['reviews']:
                        individual_metrics.append({
                            'year': entry['year'],
                            'arxiv_id': paper.id,
                            'score': review["overall_score"],
                            'conference': entry['conference'],
                            'status': paper.status,
                            'author_id': author_id,
                            'reviewer_id': review['reviewer_id']
                        })

                    aggregated_metrics.append({
                        'year': entry['year'],
                        'arxiv_id': paper.id,
                        'scores': scores,
                        'conference': entry['conference'],
                        'status': paper.status,
                        'author_id': author_id
                    })

        return aggregated_metrics, individual_metrics
