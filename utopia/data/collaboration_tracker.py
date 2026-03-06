"""
Collaboration network tracker for the preferential attachment experiment.

Tracks co-authorship / collaboration relationships between researchers
using an undirected networkx graph.
"""
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborationTracker:
    """Tracks collaboration network between researchers.

    Uses an undirected graph where:
    - Nodes = agent IDs
    - Edges = collaboration relationships
    - Edge weight = number of times agents collaborated
    - Edge attribute 'years' = list of years they collaborated
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.collaborations_by_year: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

    def add_collaboration(self, agent_a: str, agent_b: str, year: int):
        """Record a collaboration between two agents."""
        if self.graph.has_edge(agent_a, agent_b):
            self.graph[agent_a][agent_b]['weight'] += 1
            self.graph[agent_a][agent_b]['years'].append(year)
        else:
            self.graph.add_edge(agent_a, agent_b, weight=1, years=[year])
        self.collaborations_by_year[year].append((agent_a, agent_b))

    def get_network_distance(self, agent_a: str, agent_b: str) -> int:
        """Get shortest path distance between two agents.
        Returns -1 if no path exists (disconnected or unknown node).
        """
        if agent_a not in self.graph or agent_b not in self.graph:
            return -1
        try:
            return nx.shortest_path_length(self.graph, agent_a, agent_b)
        except nx.NetworkXNoPath:
            return -1

    def get_neighbors(self, agent_id: str, max_hops: int = 2) -> Dict[str, int]:
        """Get all agents within max_hops, mapped to their distance."""
        if agent_id not in self.graph:
            return {}
        lengths = dict(nx.single_source_shortest_path_length(
            self.graph, agent_id, cutoff=max_hops))
        lengths.pop(agent_id, None)
        return lengths

    def compute_metrics(self) -> Dict:
        """Compute basic network metrics."""
        if self.graph.number_of_nodes() == 0:
            return {}

        degrees = [d for _, d in self.graph.degree()]
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
            'clustering_coefficient': nx.average_clustering(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph),
            'density': nx.density(self.graph),
        }

    def compute_advanced_metrics(self) -> Dict:
        """Compute small-world and scale-free metrics."""
        metrics = {}
        if self.graph.number_of_nodes() < 10:
            return metrics

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.graph)
        metrics['betweenness_centrality'] = betweenness
        metrics['avg_betweenness'] = sum(betweenness.values()) / len(betweenness)

        # Small-world metrics on largest connected component
        largest_cc = max(nx.connected_components(self.graph), key=len)
        subgraph = self.graph.subgraph(largest_cc)
        if subgraph.number_of_nodes() > 10:
            metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
            metrics['avg_clustering'] = nx.average_clustering(subgraph)
            n = subgraph.number_of_nodes()
            m = subgraph.number_of_edges()
            if n > 1 and m > 0:
                p = 2 * m / (n * (n - 1))
                metrics['random_graph_clustering'] = p
                if n * p > 1:
                    metrics['random_graph_path_length'] = np.log(n) / np.log(n * p)

        # Degree distribution (for scale-free analysis)
        degree_sequence = sorted([d for _, d in self.graph.degree()], reverse=True)
        metrics['degree_sequence'] = degree_sequence
        metrics['max_degree'] = degree_sequence[0] if degree_sequence else 0

        return metrics

    def to_dict(self) -> Dict:
        """Serialize for checkpoints."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'a': u, 'b': v,
                'weight': data['weight'],
                'years': data['years'],
            })
        return {
            'nodes': list(self.graph.nodes()),
            'edges': edges,
            'collaborations_by_year': {
                str(y): colabs
                for y, colabs in self.collaborations_by_year.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CollaborationTracker':
        """Deserialize from checkpoint."""
        tracker = cls()
        for node in data.get('nodes', []):
            tracker.graph.add_node(node)
        for edge in data.get('edges', []):
            tracker.graph.add_edge(
                edge['a'], edge['b'],
                weight=edge['weight'],
                years=edge['years'],
            )
        tracker.collaborations_by_year = defaultdict(list, {
            int(y): colabs
            for y, colabs in data.get('collaborations_by_year', {}).items()
        })
        return tracker
