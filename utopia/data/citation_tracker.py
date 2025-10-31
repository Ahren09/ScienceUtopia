"""Citation tracking for papers."""

from typing import Dict, List, Set, Union


class CitationTracker:
    """Tracks which papers cite which papers."""

    def __init__(self):
        self.citations: Dict[str, Set[str]] = {}  # cited_paper_id -> [citing_paper_ids]

    def add_citations(self, citing_paper_id: str, cited_paper_ids: Union[str,List[str], Set[str]], year: int):
        """Add citations from a paper to its references."""
        for cited_id in cited_paper_ids:
            if cited_id not in self.citations:
                self.citations[cited_id] = set()
            if citing_paper_id not in self.citations[cited_id]:
                self.citations[cited_id].add(citing_paper_id)

    def get_citations(self, paper_id: str) -> List[str]:
        """Get papers that cite this paper."""
        return self.citations.get(paper_id, [])

    def get_citation_count(self, paper_id: str) -> int:
        """Get number of citations for a paper."""
        return len(self.citations.get(paper_id, []))

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {"citations": {k: list(v) for k, v in self.citations.items()}}

    @classmethod
    def from_dict(cls, data: Dict) -> "CitationTracker":
        """Deserialize from dict."""
        tracker = cls()
        tracker.citations = {k: set(v) for k, v in data["citations"].items()}
        return tracker
