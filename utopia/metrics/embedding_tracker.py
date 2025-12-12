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
