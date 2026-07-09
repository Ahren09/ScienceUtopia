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

    def compute_career_centroid(self, author_id: str, max_years: int = 3, current_year: int = None) -> Optional[np.ndarray]:
        """Compute the centroid embedding of an author's recent papers.

        Args:
            author_id: Author ID
            max_years: Maximum number of years to look back
            current_year: Current year (if None, uses max year in metadata)

        Returns:
            Mean embedding (centroid) or None if no papers found
        """
        if current_year is None:
            years = [m['year'] for m in self.paper_metadata.values() if m['author_id'] == author_id]
            if not years:
                return None
            current_year = max(years)

        min_year = current_year - max_years

        paper_ids = [
            pid for pid, meta in self.paper_metadata.items()
            if meta['author_id'] == author_id and min_year <= meta['year'] <= current_year
        ]

        if not paper_ids:
            return None

        embeddings = [self.embeddings[pid] for pid in paper_ids if pid in self.embeddings]
        if not embeddings:
            return None

        return np.mean(np.stack(embeddings), axis=0)

    def compute_direction_distances(self, centroid: np.ndarray, directions: List) -> Dict[str, float]:
        """Compute cosine distances from a centroid to each research direction.

        Args:
            centroid: Reference embedding (e.g. career centroid)
            directions: List of ResearchDirection objects with .topic and .keywords

        Returns:
            Dict mapping direction.topic -> cosine distance
        """
        texts = []
        topics = []
        for d in directions:
            text = d.topic + " " + " ".join(d.keywords or [])
            texts.append(text)
            topics.append(d.topic)

        if not texts:
            return {}

        dir_embeddings = self.model.encode(texts, convert_to_numpy=True)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
        dir_norms = dir_embeddings / (np.linalg.norm(dir_embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = dir_norms @ centroid_norm
        distances = 1.0 - similarities

        return {topic: float(dist) for topic, dist in zip(topics, distances)}

    def compute_novelty_score(self, paper_id: str, author_id: str, history_window_years: int = 3) -> Optional[tuple]:
        """Compute novelty score for a paper relative to the author's career centroid.

        Args:
            paper_id: Paper ID
            author_id: Author ID
            history_window_years: Years of history to consider

        Returns:
            Tuple of (novelty_score, distance_bucket) or None if insufficient data
        """
        if paper_id not in self.embeddings:
            return None

        paper_year = self.paper_metadata.get(paper_id, {}).get('year')
        if paper_year is None:
            return None

        centroid = self.compute_career_centroid(
            author_id, max_years=history_window_years, current_year=paper_year - 1
        )
        if centroid is None:
            return None

        paper_emb = self.embeddings[paper_id]
        sim = np.dot(paper_emb, centroid) / (np.linalg.norm(paper_emb) * np.linalg.norm(centroid) + 1e-10)
        novelty_score = float(1.0 - sim)

        # Bucket based on fixed thresholds (matching config quantiles)
        if novelty_score < 0.33:
            bucket = 'near'
        elif novelty_score < 0.66:
            bucket = 'mid'
        else:
            bucket = 'far'

        return (novelty_score, bucket)

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
