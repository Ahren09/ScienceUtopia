"""Calculate CD (Consolidating-Disruptive) Index for papers"""
from typing import Optional


class CDIndexCalculator:
    """Calculate CD index to measure disruptive vs consolidating research"""

    def __init__(self, citation_tracker, paper_tracker):
        self.citation_tracker = citation_tracker
        self.paper_tracker = paper_tracker

    def calculate_cd_index(self, paper_id: str) -> Optional[float]:
        """Calculate CD index for a paper

        CD = (n_disruptive - n_consolidating) / (n_disruptive + n_consolidating + n_mixed)

        Returns value in [-1, 1]:
        - Positive: disruptive (citations don't cite paper's references)
        - Negative: consolidating (citations also cite paper's references)
        """
        # Get papers that cite this paper
        citing_papers = self.citation_tracker.get_papers_citing(paper_id)
        if not citing_papers:
            return None

        # Get this paper's references
        paper_refs = self.citation_tracker.get_citations_from_paper(paper_id)
        if not paper_refs:
            return 0.0  # No references means neutral

        n_disruptive = 0  # Citations that don't cite any references
        n_consolidating = 0  # Citations that cite all references
        n_mixed = 0  # Citations that cite some references

        for citing_paper in citing_papers:
            citing_refs = set(self.citation_tracker.get_citations_from_paper(citing_paper))
            common_refs = citing_refs.intersection(set(paper_refs))

            if not common_refs:
                n_disruptive += 1
            elif len(common_refs) == len(paper_refs):
                n_consolidating += 1
            else:
                n_mixed += 1

        total = n_disruptive + n_consolidating + n_mixed
        if total == 0:
            return 0.0

        cd_index = (n_disruptive - n_consolidating) / total
        return cd_index
