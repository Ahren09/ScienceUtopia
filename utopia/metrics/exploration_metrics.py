"""Track exploration vs exploitation metrics"""
from collections import defaultdict
from typing import Dict, Optional


class ExplorationMetrics:
    """Track topic distances, CD indices, and strategy-level metrics"""

    def __init__(self):
        self.topic_distances: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))  # author_id -> year -> [distances]
        self.cd_indices: Dict[str, Dict[int, float]] = defaultdict(dict)  # paper_id -> year -> cd_score
        self.strategy_metrics: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))  # strategy -> year -> metric -> value

    def record_topic_distance(self, author_id: str, year: int, distance: float):
        """Record topic distance for an author in a given year"""
        if distance is not None:
            self.topic_distances[author_id][year].append(distance)

    def record_cd_index(self, paper_id: str, year: int, cd_score: float):
        """Record CD index for a paper"""
        self.cd_indices[paper_id][year] = cd_score

    def record_strategy_metric(self, strategy: str, year: int, metric: str, value: float):
        """Record aggregate metric for a strategy"""
        self.strategy_metrics[strategy][year][metric] = value

    def get_author_avg_distance(self, author_id: str, year: int) -> Optional[float]:
        """Get average topic distance for author in a year"""
        distances = self.topic_distances[author_id].get(year, [])
        return sum(distances) / len(distances) if distances else None

    def to_dict(self) -> Dict:
        """Serialize for checkpoint"""
        return {
            'topic_distances': {k: dict(v) for k, v in self.topic_distances.items()},
            'cd_indices': {k: dict(v) for k, v in self.cd_indices.items()},
            'strategy_metrics': {k: {y: dict(m) for y, m in v.items()} for k, v in self.strategy_metrics.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Deserialize from checkpoint"""
        metrics = cls()
        metrics.topic_distances = defaultdict(lambda: defaultdict(list), {
            k: defaultdict(list, v) for k, v in data['topic_distances'].items()
        })
        metrics.cd_indices = defaultdict(dict, data['cd_indices'])
        metrics.strategy_metrics = defaultdict(lambda: defaultdict(dict), {
            k: defaultdict(dict, v) for k, v in data['strategy_metrics'].items()
        })
        return metrics
