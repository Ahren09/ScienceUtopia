"""Track paper embeddings for exploration vs exploitation analysis"""
import hashlib
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional


class EmbeddingTracker:
    """Track and compare paper embeddings to measure topic distances"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None, model_revision: Optional[str] = None):
        self.model_name = model_name
        known = {
            'all-MiniLM-L6-v2': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
            'sentence-transformers/all-MiniLM-L6-v2': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
        }
        revision = model_revision or known.get(model_name)
        self.identity = {'model': model_name, 'revision': revision}
        self.model = SentenceTransformer(model_name, **({'revision': revision} if revision else {}))
        self.embeddings: Dict[str, np.ndarray] = {}  # paper_id -> embedding
        self.paper_metadata: Dict[str, Dict] = {}  # paper_id -> {author_id, year}
        self.expertise_centroids: Dict[str, np.ndarray] = {}  # agent_id -> initial expertise centroid
        # Empirical near/far thresholds from the direction-pair distance distribution.
        # None until compute_direction_thresholds() runs; fixed 0.33/0.66 fallback otherwise.
        self.near_threshold: Optional[float] = None
        self.far_threshold: Optional[float] = None
        # Disk cache: seed-independent, keyed by (model, sha256(text)) so repeated
        # seeds/conditions never re-encode the same abstract.
        self.cache_dir = cache_dir
        self._disk_cache: Dict[str, np.ndarray] = {}
        self._disk_cache_dirty = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_path = os.path.join(
                cache_dir, f"emb_{model_name.replace('/', '_')}_{revision or 'unversioned'}.npz")
            if os.path.exists(self._cache_path):
                with np.load(self._cache_path) as npz:
                    self._disk_cache = {k: npz[k] for k in npz.files}

    def _text_key(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _encode_cached(self, texts: List[str]) -> np.ndarray:
        """Encode texts with disk-cache lookup; one batched encode for all misses."""
        keys = [self._text_key(t) for t in texts]
        miss_idx = [i for i, k in enumerate(keys) if k not in self._disk_cache]
        if miss_idx:
            new_embs = self.model.encode([texts[i] for i in miss_idx],
                                         convert_to_numpy=True, show_progress_bar=False)
            for i, emb in zip(miss_idx, new_embs):
                self._disk_cache[keys[i]] = emb
            self._disk_cache_dirty = True
        return np.stack([self._disk_cache[k] for k in keys])

    def flush_cache(self):
        """Persist the disk cache. Called once per year, not per paper."""
        if self.cache_dir and self._disk_cache_dirty:
            np.savez(self._cache_path, **self._disk_cache)
            self._disk_cache_dirty = False

    def add_paper_embedding(self, paper_id: str, abstract: str, author_id: str, year: int):
        """Add embedding for a paper"""
        self.add_paper_embeddings_batch([(paper_id, abstract, author_id, year)])

    def add_paper_embeddings_batch(self, items: List[tuple]):
        """Add embeddings for many papers with ONE encode call.

        Args:
            items: list of (paper_id, abstract, author_id, year)
        """
        items = [it for it in items if it[0] not in self.embeddings]
        if not items:
            return
        embs = self._encode_cached([it[1] for it in items])
        for (paper_id, _, author_id, year), emb in zip(items, embs):
            self.embeddings[paper_id] = emb
            self.paper_metadata[paper_id] = {'author_id': author_id, 'year': year}

    def register_agent_expertise(self, agent_id: str, direction_texts: List[str]):
        """Store the centroid of an agent's initial expertise directions.

        Used as the career reference before the agent's first paper (plan 4.8)."""
        if not direction_texts:
            return
        embs = self._encode_cached(direction_texts)
        self.expertise_centroids[agent_id] = np.mean(embs, axis=0)

    def compute_direction_thresholds(self, directions: List) -> Dict[str, float]:
        """Compute empirical near/far thresholds from the pairwise cosine-distance
        distribution among all research directions (plan 4.7).

        Sets self.near_threshold / self.far_threshold to the 33rd / 66th
        percentiles and returns them for the resolved-config record.
        """
        texts = [d.topic + " " + " ".join(d.keywords or []) for d in directions]
        if len(texts) < 3:
            return {'near_threshold': 0.33, 'far_threshold': 0.66}
        embs = self._encode_cached(texts)
        normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        sims = normed @ normed.T
        dists = 1.0 - sims[np.triu_indices(len(texts), k=1)]
        self.near_threshold = float(np.percentile(dists, 33))
        self.far_threshold = float(np.percentile(dists, 66))
        return {'near_threshold': self.near_threshold, 'far_threshold': self.far_threshold}

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

        Before the first paper, falls back to the agent's initial expertise
        centroid registered via register_agent_expertise() (plan 4.8), so
        first-paper novelty is measured against expertise rather than 0.

        Args:
            author_id: Author ID
            max_years: Maximum number of years to look back
            current_year: Current year (if None, uses max year in metadata)

        Returns:
            Mean embedding (centroid), or None if neither papers nor expertise exist
        """
        if current_year is None:
            years = [m['year'] for m in self.paper_metadata.values() if m['author_id'] == author_id]
            if not years:
                return self.expertise_centroids.get(author_id)
            current_year = max(years)

        min_year = current_year - max_years

        paper_ids = [
            pid for pid, meta in self.paper_metadata.items()
            if meta['author_id'] == author_id and min_year <= meta['year'] <= current_year
        ]

        embeddings = [self.embeddings[pid] for pid in paper_ids if pid in self.embeddings]
        if not embeddings:
            return self.expertise_centroids.get(author_id)

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

        dir_embeddings = self._encode_cached(texts)
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

        # Empirical thresholds when available (plan 4.7); fixed fallback otherwise
        near = self.near_threshold if self.near_threshold is not None else 0.33
        far = self.far_threshold if self.far_threshold is not None else 0.66
        if novelty_score < near:
            bucket = 'near'
        elif novelty_score < far:
            bucket = 'mid'
        else:
            bucket = 'far'

        return (novelty_score, bucket)

    def to_dict(self) -> Dict:
        """Serialize for checkpoint"""
        return {
            'model_identity': self.identity,
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'paper_metadata': self.paper_metadata,
            'expertise_centroids': {k: v.tolist() for k, v in self.expertise_centroids.items()},
            'near_threshold': self.near_threshold,
            'far_threshold': self.far_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """Deserialize from checkpoint"""
        tracker = cls(model_name, cache_dir=cache_dir)
        if data.get('model_identity') and data['model_identity'] != tracker.identity:
            raise ValueError('Checkpoint metric embedding model/revision mismatch')
        tracker.embeddings = {k: np.array(v) for k, v in data['embeddings'].items()}
        tracker.paper_metadata = data['paper_metadata']
        tracker.expertise_centroids = {
            k: np.array(v) for k, v in data.get('expertise_centroids', {}).items()}
        tracker.near_threshold = data.get('near_threshold')
        tracker.far_threshold = data.get('far_threshold')
        return tracker
