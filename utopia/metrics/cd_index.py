"""Calculate CD (Consolidating-Disruptive) Index for papers.

Implements the Funk & Owen-Smith / Wu-Wang-Evans CD index:

    CD_t = (1 / n_t) * sum_i(-2 * f_i * b_i + f_i)

where i ranges over all later papers (up to an observation cutoff year) that
cite the focal paper AND/OR at least one of the focal paper's references
(predecessors); f_i = 1 if paper i cites the focal paper; b_i = 1 if paper i
cites at least one predecessor.

Per-term values: focal-only citer (f=1, b=0) -> +1 (disruptive);
focal+predecessor citer (f=1, b=1) -> -1 (consolidating);
predecessor-only citer (f=0, b=1) -> 0, but enlarges the denominator.
"""
from typing import Dict, Optional


class CDIndexCalculator:
    """Calculate CD index to measure disruptive vs consolidating research"""

    def __init__(self, citation_tracker, paper_tracker):
        self.citation_tracker = citation_tracker
        self.paper_tracker = paper_tracker

    def _paper_year(self, paper_id: str) -> Optional[int]:
        paper = self.paper_tracker.get_paper_by_id(paper_id)
        if paper is None:
            return None
        return paper.get('year') if isinstance(paper, dict) else getattr(paper, 'year', None)

    def calculate_cd_index(self, paper_id: str, cutoff_year: Optional[int] = None,
                           min_denominator: int = 1) -> Optional[Dict]:
        """Calculate the CD index of `paper_id` observed at `cutoff_year`.

        Args:
            paper_id: Focal paper.
            cutoff_year: Only later papers with year <= cutoff_year are counted.
                None means no cutoff (all recorded citers).
            min_denominator: Minimum index-set size n_t; returns None below it.

        Returns:
            Dict with 'cd', 'n_disruptive' (f=1,b=0), 'n_consolidating' (f=1,b=1),
            'n_predecessor_only' (f=0,b=1), 'denominator'; or None if the paper
            has no references or the denominator is below min_denominator.
        """
        focal_year = self._paper_year(paper_id)
        paper_refs = set(self.citation_tracker.get_citations_from_paper(paper_id))
        if not paper_refs:
            return None  # CD undefined without predecessors

        def _eligible(citing_id: str) -> bool:
            if citing_id == paper_id:
                return False
            if cutoff_year is None:
                return True
            year = self._paper_year(citing_id)
            if year is None:
                return True  # keep papers with unknown year rather than silently dropping
            if year > cutoff_year:
                return False
            # only papers published at/after the focal paper can be "later" papers
            return focal_year is None or year >= focal_year

        focal_citers = {c for c in self.citation_tracker.get_papers_citing(paper_id) if _eligible(c)}
        predecessor_citers = set()
        for ref in paper_refs:
            predecessor_citers.update(
                c for c in self.citation_tracker.get_papers_citing(ref) if _eligible(c)
            )

        index_set = focal_citers | predecessor_citers
        n = len(index_set)
        if n < max(min_denominator, 1):
            return None

        n_consolidating = len(focal_citers & predecessor_citers)   # f=1, b=1
        n_disruptive = len(focal_citers - predecessor_citers)      # f=1, b=0
        n_predecessor_only = len(predecessor_citers - focal_citers)  # f=0, b=1

        cd = (n_disruptive - n_consolidating) / n

        return {
            'cd': cd,
            'n_disruptive': n_disruptive,
            'n_consolidating': n_consolidating,
            'n_predecessor_only': n_predecessor_only,
            'denominator': n,
        }
