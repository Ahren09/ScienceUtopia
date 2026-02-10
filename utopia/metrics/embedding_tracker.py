"""Track paper embeddings for exploration vs exploitation analysis"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional


class EmbeddingTracker:
    """Track and compare paper embeddings to measure topic distances"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings: Dict[str, np.ndarray] = {}  # paper_id -> embedding
        self.paper_metadata: Dict[str, Dict] = {}  # paper_id -> {author_id, year}

    def add_paper_embedding(self, paper_id: str, abstract: str, author_id: str, year: int):
        """Add embedding for a paper"""
        embedding = self.model.encode(abstract, convert_to_numpy=True)
        self.embeddings[paper_id] = embedding
        self.paper_metadata[paper_id] = {'author_id': author_id, 'year': year}

    def compute_distance_to_previous(self, paper_id: str, author_id: str) -> Optional[float]:
        """Compute cosine distance to author's most recent previous paper"""
        if paper_id not in self.embeddings:
            return None

        # Find author's previous papers
        current_year = self.paper_metadata[paper_id]['year']
        previous_papers = [
            (pid, meta['year']) for pid, meta in self.paper_metadata.items()
            if meta['author_id'] == author_id and meta['year'] < current_year
        ]

        if not previous_papers:
            return None

        # Get most recent previous paper
        prev_paper_id = max(previous_papers, key=lambda x: x[1])[0]

        # Compute cosine distance (1 - similarity)
        emb1 = self.embeddings[paper_id]
        emb2 = self.embeddings[prev_paper_id]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(1 - similarity)

    def compute_distances_batch(self, paper_author_pairs: List[tuple]) -> Dict[str, Optional[float]]:
        """Compute cosine distances for multiple (paper_id, author_id) pairs at once.

        Returns:
            Dict mapping paper_id -> distance (None if no previous paper exists)
        """
        if not paper_author_pairs:
            return {}

        # Build author -> [(paper_id, year)] index, sorted by year
        author_papers: Dict[str, List[tuple]] = {}
        for pid, meta in self.paper_metadata.items():
            author_papers.setdefault(meta['author_id'], []).append((pid, meta['year']))
        for aid in author_papers:
            author_papers[aid].sort(key=lambda x: x[1])

        # Separate pairs into those with/without a previous paper
        valid_ids, current_embs, prev_embs = [], [], []
        results: Dict[str, Optional[float]] = {}

        for paper_id, author_id in paper_author_pairs:
            if paper_id not in self.embeddings:
                results[paper_id] = None
                continue

            current_year = self.paper_metadata[paper_id]['year']
            prev = [p for p in author_papers.get(author_id, []) if p[1] < current_year]

            if not prev:
                results[paper_id] = None
                continue

            valid_ids.append(paper_id)
            current_embs.append(self.embeddings[paper_id])
            prev_embs.append(self.embeddings[prev[-1][0]])  # most recent (list is sorted)

        # Vectorized cosine distance
        if valid_ids:
            cur = np.stack(current_embs)
            prv = np.stack(prev_embs)
            sims = np.sum(cur * prv, axis=1) / (np.linalg.norm(cur, axis=1) * np.linalg.norm(prv, axis=1))
            for pid, dist in zip(valid_ids, 1 - sims):
                results[pid] = float(dist)

        return results

    def to_dict(self) -> Dict:
        """Serialize for checkpoint"""
        return {
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'paper_metadata': self.paper_metadata
        }

    @classmethod
    def from_dict(cls, data: Dict, model_name: str = 'all-MiniLM-L6-v2'):
        """Deserialize from checkpoint"""
        tracker = cls(model_name)
        tracker.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
        tracker.paper_metadata = data['paper_metadata']
        return tracker
