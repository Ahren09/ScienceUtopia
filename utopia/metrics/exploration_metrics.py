"""Track exploration vs exploitation metrics"""
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Structured analysis tables written per year (plan 8.9). Keys = filenames.
TABLE_NAMES = ('agent_year', 'paper', 'paper_age', 'strategy_year', 'ecosystem_year')


class ExplorationMetrics:
    """Track topic distances, CD indices, and strategy-level metrics"""

    def __init__(self):
        self.topic_distances: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))  # author_id -> year -> [distances]
        self.cd_indices: Dict[str, Dict[int, float]] = defaultdict(dict)  # paper_id -> year -> cd_score
        self.strategy_metrics: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))  # strategy -> year -> metric -> value
        self.agent_stats: Dict[str, Dict[int, Dict]] = defaultdict(lambda: defaultdict(dict))  # agent_id -> year -> stats_dict
        # Flat row stores for analysis tables; one list of dicts per table
        self.rows: Dict[str, List[Dict]] = {name: [] for name in TABLE_NAMES}

    def add_row(self, table: str, row: Dict):
        """Append one structured record to a named analysis table."""
        self.rows[table].append(row)

    def export_tables(self, out_dir: str):
        """Write all analysis tables as Parquet (CSV fallback). Atomic rewrite
        each year — tables are small relative to checkpoints."""
        os.makedirs(out_dir, exist_ok=True)
        for name, rows in self.rows.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            try:
                df.to_parquet(os.path.join(out_dir, f'{name}.parquet'), index=False)
            except Exception as e:  # pyarrow missing or type issue
                logger.warning(f"Parquet export failed for {name} ({e}); writing CSV")
                df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)

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

    def record_agent_stats(self, agent_id: str, year: int, stats_dict: Dict):
        """Record per-agent yearly stats (acceptance_rate, funding, citations, etc.)"""
        self.agent_stats[agent_id][year] = stats_dict

    def get_author_avg_distance(self, author_id: str, year: int) -> Optional[float]:
        """Get average topic distance for author in a year"""
        distances = self.topic_distances[author_id].get(year, [])
        return sum(distances) / len(distances) if distances else None

    def to_dict(self) -> Dict:
        """Serialize for checkpoint"""
        return {
            'topic_distances': {k: dict(v) for k, v in self.topic_distances.items()},
            'cd_indices': {k: dict(v) for k, v in self.cd_indices.items()},
            'strategy_metrics': {k: {y: dict(m) for y, m in v.items()} for k, v in self.strategy_metrics.items()},
            'agent_stats': {k: {str(y): dict(s) for y, s in v.items()} for k, v in self.agent_stats.items()},
            'rows': self.rows,
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
        if 'agent_stats' in data:
            metrics.agent_stats = defaultdict(lambda: defaultdict(dict), {
                k: defaultdict(dict, {int(y): s for y, s in v.items()})
                for k, v in data['agent_stats'].items()
            })
        for name in TABLE_NAMES:
            metrics.rows[name] = data.get('rows', {}).get(name, [])
        return metrics
