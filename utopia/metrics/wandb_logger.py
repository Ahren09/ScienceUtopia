"""Weights & Biases logger for real-time simulation monitoring.

Logs scalar metrics per-phase within each year. Phases 0-4 accumulate with
commit=False; phase 5 commits all scalar metrics at step=year.
Controlled via SIMULATION_CONFIG['wandb']['enabled'] — all methods are no-ops when disabled.
"""

import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


class WandbLogger:
    """Logs scalar simulation metrics to wandb per phase.

    When disabled, all methods are no-ops — callers don't need if-guards.
    wandb is only imported when enabled, so it's not a required dependency.
    """

    def __init__(self, config: dict, args):
        wandb_config = config.get('wandb', {})
        self.enabled = wandb_config.get('enabled', False)

        if not self.enabled:
            return

        import wandb

        project = wandb_config.get('project', 'science-utopia')
        run_name = wandb_config.get('run_name')
        if run_name is None:
            model_name = getattr(args, 'model', 'unknown').split('/')[-1]
            experiment = getattr(args, 'experiment_name', 'default')
            run_name = f"{experiment}_{model_name}"

        wandb.init(project=project, name=run_name, config=vars(args))
        logger.info(f"wandb initialized: project={project}, run={run_name}")

        # Track historical scalar values.
        self._history = defaultdict(list)

    def log_phase(self, year: int, phase: int, year_results: dict, simulation, **extra):
        """Log metrics for a specific phase within a year.

        Phases 0-4 use commit=False to accumulate metrics.
        Phase 5 commits the year's scalar metrics.
        """
        if not self.enabled:
            return

        import wandb

        metrics = self._extract_phase_metrics(year, phase, year_results, simulation, **extra)

        # Update history in the phase where each metric is computed
        if phase == 3:
            self._history['mean_score'].append(metrics.get('review/overall_mean', 0))
        elif phase == 4:
            self._history['acceptance_rate'].append(metrics.get('papers/acceptance_rate', 0))
        elif phase == 5:
            self._history['gini'].append(metrics.get('ecosystem/funding_gini', 0))

        if phase < 5:
            wandb.log(metrics, step=year, commit=False)
            logger.debug(f"wandb: accumulated phase {phase} for year {year} ({len(metrics)} metrics)")
        else:
            wandb.log(metrics, step=year, commit=True)
            logger.info(f"wandb: committed year {year} phase {phase} ({len(metrics)} total entries)")

    def finish(self):
        """Finalize the wandb run."""
        if not self.enabled:
            return
        import wandb
        wandb.finish()
        logger.info("wandb run finished")

    # ---- Phase-specific metric extraction ----

    def _extract_phase_metrics(self, year: int, phase: int, year_results: dict, simulation, **extra) -> dict:
        metrics = {}

        if phase == 0:
            metrics['resubmissions/count'] = extra.get('num_resubmissions', 0)

        elif phase == 1:
            ra = year_results.get('research_assignment', {})
            metrics['directions/agents_assigned'] = ra.get('agents_assigned', 0)
            metrics['directions/active_authors'] = extra.get('num_active_authors', 0)
            metrics['directions/avg_funding'] = extra.get('avg_funding', 0)

        elif phase == 2:
            paper_sub = year_results.get('paper_submission', {})
            metrics['papers/submitted'] = paper_sub.get('num_papers_submitted', 0)

            # Per-conference and per-topic submissions
            topic_counts = defaultdict(int)
            for conf in simulation.conference_system.conferences:
                safe_name = conf.name.replace('/', '_').replace(' ', '_')
                metrics[f'submissions/conference/{safe_name}'] = len(conf.submitted_papers)
                for paper in conf.submitted_papers:
                    for topic in paper.get('topics', []):
                        topic_counts[topic] += 1
            for topic, count in topic_counts.items():
                safe_topic = topic.replace('/', '_').replace(' ', '_')
                metrics[f'submissions/topic/{safe_topic}'] = count

        elif phase == 3:
            score_metrics = year_results.get('peer_review', {}).get('score_metrics', {})
            metrics['review/overall_mean'] = score_metrics.get('overall_mean', 0)
            metrics['review/overall_std'] = score_metrics.get('overall_std', 0)
            metrics['review/total_reviews'] = score_metrics.get('total_reviews', 0)
            for conf_id, stats in score_metrics.get('conference_stats', {}).items():
                safe_name = conf_id.replace('/', '_').replace(' ', '_')
                metrics[f'conference/{safe_name}/mean_score'] = stats.get('mean', 0)

            # Preferential attachment: distance-score correlation + bias metrics
            if hasattr(simulation, 'network_metrics') and simulation.network_metrics:
                corr = simulation.network_metrics.correlation_by_year.get(year, {})
                if corr:
                    metrics['network/distance_score_pearson_r'] = corr.get('pearson_r', 0)
                    metrics['network/distance_score_spearman_r'] = corr.get('spearman_r', 0)
                    metrics['network/n_connected_reviews'] = corr.get('n_connected_reviews', 0)

                    # Per-distance bucket mean scores
                    for bucket in ('distance_1', 'distance_2', 'distance_3plus', 'disconnected'):
                        if f'avg_score_{bucket}' in corr:
                            metrics[f'network/avg_score_{bucket}'] = corr[f'avg_score_{bucket}']
                        if f'score_std_{bucket}' in corr:
                            metrics[f'network/score_std_{bucket}'] = corr[f'score_std_{bucket}']

                    # Author centrality bias
                    if 'author_centrality_score_pearson_r' in corr:
                        metrics['network/author_centrality_score_pearson_r'] = corr['author_centrality_score_pearson_r']
                        metrics['network/author_centrality_score_pearson_p'] = corr['author_centrality_score_pearson_p']

                    # Collaboration strength bias
                    if 'collab_strength_score_pearson_r' in corr:
                        metrics['network/collab_strength_score_pearson_r'] = corr['collab_strength_score_pearson_r']
                        metrics['network/collab_strength_score_pearson_p'] = corr['collab_strength_score_pearson_p']

        elif phase == 4:
            decisions = year_results.get('decisions', {})
            accepted = decisions.get('total_accepted', 0)
            rejected = decisions.get('total_rejected', 0)
            metrics['papers/accepted'] = accepted
            metrics['papers/rejected'] = rejected
            total = accepted + rejected
            metrics['papers/acceptance_rate'] = accepted / total if total > 0 else 0

            # Per-conference acceptance/rejection
            for conf in simulation.conference_system.conferences:
                safe_name = conf.name.replace('/', '_').replace(' ', '_')
                metrics[f'decisions/conference/{safe_name}/accepted'] = len(conf.decisions['accept'])
                metrics[f'decisions/conference/{safe_name}/rejected'] = len(conf.decisions['reject'])

        elif phase == 5:
            eco = year_results.get('ecosystem_metrics', {})
            res_dist = eco.get('resource_distribution', {})
            metrics['ecosystem/funding_mean'] = res_dist.get('mean', 0) or 0
            metrics['ecosystem/funding_std'] = res_dist.get('std', 0) or 0
            metrics['ecosystem/funding_gini'] = res_dist.get('gini', 0) or 0
            metrics['ecosystem/active_agents'] = eco.get('active_agents', 0)

            ft = simulation.funding_tracker
            consumption = ft.consumption_per_cycle.get(year, {})
            metrics['funding/consumption_university'] = consumption.get('university', 0)
            metrics['funding/consumption_industry'] = consumption.get('industry', 0)

            allocation = ft.funding_allocation_per_cycle.get(year, {})
            metrics['funding/allocation_university'] = allocation.get('university', 0)
            metrics['funding/allocation_industry'] = allocation.get('industry', 0)

            citation_counts = [len(cites) for cites in simulation.citation_tracker.citations.values()]
            metrics['citations/total_cumulative'] = sum(citation_counts)

            if simulation.experiment_name == 'exploration_vs_exploitation' and simulation.exploration_metrics:
                for strategy, year_data in simulation.exploration_metrics.strategy_metrics.items():
                    if year in year_data:
                        for metric_name, value in year_data[year].items():
                            metrics[f'exploration/{strategy}/{metric_name}'] = value

            # Preferential attachment: network structure + topology metrics
            if hasattr(simulation, 'collaboration_tracker') and simulation.collaboration_tracker:
                net_metrics = simulation.collaboration_tracker.compute_metrics()
                metrics['network/num_edges'] = net_metrics.get('num_edges', 0)
                metrics['network/num_nodes'] = net_metrics.get('num_nodes', 0)
                metrics['network/avg_degree'] = net_metrics.get('avg_degree', 0)
                metrics['network/clustering_coefficient'] = net_metrics.get('clustering_coefficient', 0)
                metrics['network/density'] = net_metrics.get('density', 0)

                # Topology metrics from network_snapshots
                if hasattr(simulation, 'network_metrics') and simulation.network_metrics:
                    snapshot = simulation.network_metrics.network_snapshots.get(year, {})
                    for key in ('degree_assortativity', 'avg_path_length', 'avg_betweenness',
                                'max_degree', 'num_connected_components'):
                        if key in snapshot:
                            metrics[f'network/{key}'] = snapshot[key]

        return metrics

    # ---- Plot generation (native wandb chart types) ----


    # ---- Helpers ----
