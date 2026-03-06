"""
Network metrics for the preferential attachment experiment.

Tracks review distance-score pairs and computes correlation between
network distance and review scores across simulation years.
"""
import logging
from typing import Dict, List
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class NetworkMetrics:
    """Track preferential attachment metrics per year."""

    def __init__(self):
        # year -> [{reviewer_id, author_id, distance, score, paper_id}]
        self.review_distances: Dict[int, List[Dict]] = defaultdict(list)
        # year -> network-level metrics snapshot
        self.network_snapshots: Dict[int, Dict] = {}
        # year -> correlation results
        self.correlation_by_year: Dict[int, Dict] = {}

    def record_review_with_distance(self, year: int, reviewer_id: str,
                                     author_id: str, distance: int,
                                     score: int, paper_id: str):
        """Record a review with network distance metadata."""
        self.review_distances[year].append({
            'reviewer_id': reviewer_id,
            'author_id': author_id,
            'distance': distance,
            'score': score,
            'paper_id': paper_id,
        })

    def compute_correlation(self, year: int) -> Dict:
        """Compute correlation between network distance and review score.

        Only considers reviews where reviewer and author are in the same
        connected component (distance > 0).
        """
        from scipy import stats

        reviews = self.review_distances.get(year, [])
        connected = [r for r in reviews if r['distance'] > 0]
        if len(connected) < 10:
            return {}

        distances = [r['distance'] for r in connected]
        scores = [r['score'] for r in connected]

        pearson_r, pearson_p = stats.pearsonr(distances, scores)
        spearman_r, spearman_p = stats.spearmanr(distances, scores)

        # Group scores by distance bucket
        score_by_distance = defaultdict(list)
        for r in reviews:
            if r['distance'] == 1:
                score_by_distance['distance_1'].append(r['score'])
            elif r['distance'] == 2:
                score_by_distance['distance_2'].append(r['score'])
            elif r['distance'] >= 3:
                score_by_distance['distance_3plus'].append(r['score'])
            else:
                score_by_distance['disconnected'].append(r['score'])

        result = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_connected_reviews': len(connected),
            'n_total_reviews': len(reviews),
        }
        for bucket, bucket_scores in score_by_distance.items():
            result[f'avg_score_{bucket}'] = (
                sum(bucket_scores) / len(bucket_scores) if bucket_scores else 0
            )
            result[f'score_std_{bucket}'] = (
                float(np.std(bucket_scores)) if len(bucket_scores) > 1 else 0.0
            )
            result[f'n_{bucket}'] = len(bucket_scores)

        self.correlation_by_year[year] = result
        return result

    def compute_author_centrality_bias(self, year: int, betweenness: Dict[str, float]) -> Dict:
        """Correlate author betweenness centrality with mean review score received.

        Detects "Matthew effect" — well-connected authors getting higher scores.
        """
        from scipy import stats

        reviews = self.review_distances.get(year, [])
        if not reviews or not betweenness:
            return {}

        # Group scores by author
        scores_by_author = defaultdict(list)
        for r in reviews:
            scores_by_author[r['author_id']].append(r['score'])

        author_centralities = []
        author_mean_scores = []
        for author_id, author_scores in scores_by_author.items():
            if author_id in betweenness:
                author_centralities.append(betweenness[author_id])
                author_mean_scores.append(np.mean(author_scores))

        if len(author_centralities) < 5:
            return {}

        r, p = stats.pearsonr(author_centralities, author_mean_scores)
        return {
            'author_centrality_score_pearson_r': r,
            'author_centrality_score_pearson_p': p,
        }

    def compute_collaboration_strength_bias(self, year: int, collaboration_tracker) -> Dict:
        """For distance-1 reviews, correlate edge weight with review score.

        Tests whether frequent collaborators give even higher scores
        than one-time collaborators.
        """
        from scipy import stats

        reviews = self.review_distances.get(year, [])
        distance_1 = [r for r in reviews if r['distance'] == 1]
        if len(distance_1) < 5:
            return {}

        weights = []
        scores = []
        graph = collaboration_tracker.graph
        for r in distance_1:
            rid, aid = r['reviewer_id'], r['author_id']
            if graph.has_edge(rid, aid):
                weights.append(graph[rid][aid]['weight'])
                scores.append(r['score'])

        if len(weights) < 5:
            return {}

        r, p = stats.pearsonr(weights, scores)
        return {
            'collab_strength_score_pearson_r': r,
            'collab_strength_score_pearson_p': p,
        }

    def compute_graph_topology_metrics(self, collaboration_tracker) -> Dict:
        """Compute degree assortativity and other topology metrics."""
        import networkx as nx

        graph = collaboration_tracker.graph
        if graph.number_of_nodes() < 10:
            return {}

        result = {}
        result['degree_assortativity'] = nx.degree_assortativity_coefficient(graph)
        return result

    def compute_all_metrics(self, year: int, collaboration_tracker, ecosystem=None) -> Dict:
        """Compute all metrics for a year. Wraps compute_correlation + new metrics.

        Args:
            year: Current simulation year
            collaboration_tracker: CollaborationTracker instance
            ecosystem: MultiAgentEcosystem (unused for now, reserved)

        Returns:
            Combined dict of all metric results for this year.
        """
        all_results = {}

        # Base correlation (distance vs score)
        correlation = self.compute_correlation(year)
        all_results.update(correlation)

        # Author centrality bias
        advanced = collaboration_tracker.compute_advanced_metrics()
        betweenness = advanced.get('betweenness_centrality', {})
        centrality_bias = self.compute_author_centrality_bias(year, betweenness)
        all_results.update(centrality_bias)

        # Collaboration strength bias
        strength_bias = self.compute_collaboration_strength_bias(year, collaboration_tracker)
        all_results.update(strength_bias)

        # Graph topology
        topology = self.compute_graph_topology_metrics(collaboration_tracker)
        all_results.update(topology)

        # Wire through advanced metrics (path length, betweenness, max degree, components)
        for key in ('avg_path_length', 'avg_betweenness', 'max_degree', 'avg_clustering'):
            if key in advanced:
                all_results[key] = advanced[key]
        basic = collaboration_tracker.compute_metrics()
        all_results['num_connected_components'] = basic.get('num_connected_components', 0)

        # Merge new metrics into correlation_by_year so wandb logger can find them
        if year in self.correlation_by_year:
            self.correlation_by_year[year].update({
                k: v for k, v in all_results.items()
                if k not in self.correlation_by_year[year] and not isinstance(v, dict)
            })

        # Store combined snapshot
        self.record_network_snapshot(year, {
            k: v for k, v in all_results.items()
            if not isinstance(v, dict)  # exclude nested dicts like betweenness_centrality
        })

        return all_results

    def record_network_snapshot(self, year: int, metrics_dict: Dict):
        """Store network-level metrics for a year."""
        self.network_snapshots[year] = metrics_dict

    def to_dict(self) -> Dict:
        """Serialize for checkpoints."""
        return {
            'review_distances': dict(self.review_distances),
            'network_snapshots': dict(self.network_snapshots),
            'correlation_by_year': dict(self.correlation_by_year),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NetworkMetrics':
        """Deserialize from checkpoint."""
        metrics = cls()
        metrics.review_distances = defaultdict(list, {
            int(y): revs
            for y, revs in data.get('review_distances', {}).items()
        })
        metrics.network_snapshots = {
            int(y): snap
            for y, snap in data.get('network_snapshots', {}).items()
        }
        metrics.correlation_by_year = {
            int(y): corr
            for y, corr in data.get('correlation_by_year', {}).items()
        }
        return metrics
