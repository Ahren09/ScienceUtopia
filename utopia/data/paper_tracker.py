"""
Paper Tracker System for Managing Submitted Papers Across Simulation Years

This module provides functionality to persist and retrieve submitted papers
across multiple simulation years, enabling citation mechanisms and paper
references.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ArchivedPaper:
    """Represents a submitted paper in the tracker"""
    id: str
    title: str
    abstract: str  # First 500 chars or full abstract
    author_id: Union[str, List[str]]  # Single author or list for co-authored papers
    author_type: Union[str, List[str]]  # Single type or list for co-authored papers
    year: int
    conference: str
    status: str  # 'pending', 'accept', 'reject'
    topics: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    project_start_year: int = 0
    project_end_year: int = 0
    maturity: int = 1
    review_history: List[Dict] = field(default_factory=list)  # [{"conference": str, "year": int, "score": float}]

    @property
    def all_author_ids(self) -> List[str]:
        """Normalize author_id to a list (handles both single and co-authored papers)."""
        if isinstance(self.author_id, list):
            return self.author_id
        return [self.author_id]

    def to_dict(self) -> Dict:
        """Convert to dictionary (for both serialization and resubmission)"""
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'author_id': self.author_id,
            'author_type': self.author_type,
            'year': self.year,
            'conference': self.conference,
            'topics': self.topics,
            'tags': self.tags,
            'status': self.status,
            'project_start_year': self.project_start_year,
            'project_end_year': self.project_end_year,
            'maturity': self.maturity,
            'review_history': self.review_history
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchivedPaper':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            abstract=data['abstract'],
            author_id=data['author_id'],
            author_type=data['author_type'],
            year=data['year'],
            conference=data['conference'],
            topics=data['topics'],
            tags=data['tags'],
            status=data['status'],
            project_start_year=data.get('project_start_year', 0),
            project_end_year=data.get('project_end_year', 0),
            maturity=data.get('maturity', 1),
            review_history=data['review_history']
        )

    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document for RAG integration"""
        return Document(
            page_content=self.abstract,
            metadata={
                'id': self.id,
                'title': self.title,
                'author_id': self.author_id,
                'year': self.year,
                'conference': self.conference,
                'topics': list(self.topics),
                'tags': list(self.tags),
                'source': 'simulation'
            }
        )


class PaperTracker:
    """Manages accepted papers across simulation years"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.papers_by_id: Dict[str, ArchivedPaper] = {}
        self.accepted_papers_by_year: Dict[int, Dict[str, ArchivedPaper]] = {}
        self.rejected_papers_by_year: Dict[int, Dict[str, ArchivedPaper]] = {}
        self.papers_by_author: Dict[str, List[ArchivedPaper]] = {}
        self.archive_file = self.output_dir / "paper_tracker.json"
        self.pending_papers_by_year: Dict[int, Dict[str, ArchivedPaper]] = {}
        
    

    def add_or_update_paper(self, paper_dict: Dict) -> ArchivedPaper:
        """
        Add a paper to the archive (handles resubmissions)

        Args:
            paper_dict: Paper dictionary with keys: id, title, content, author_id,
                       conference, topics
            year: Simulation year
            decision: 'accept' or 'reject'

        Returns:
            ArchivedPaper object
        """
        assert "decision" not in paper_dict, "decision is a legacy attribute and must not be provided"
        # Check if paper already exists (resubmission case)
        if paper_dict['type'] == 'resubmission':

            assert paper_dict['id'] in self.papers_by_id, f"Resubmitted paper {paper_dict['id']} not found in archive"
            archived_paper = self.papers_by_id[paper_dict['id']]

            # Update the latest conference, year, and status
            original_conference = archived_paper.conference
            original_status = archived_paper.status
            archived_paper.conference = paper_dict['conference']
            archived_paper.year = paper_dict['year']

            status = paper_dict['status']

            if status == 'pending':
                # Just resubmitted - update status only
                archived_paper.status = status
                logger.info(f"[Paper Tracker]: Updated Resubmission {paper_dict['id']} ({original_conference} {original_status} -> {paper_dict['conference']} {status}) for Year {paper_dict['year']}")

            elif status in ['accept', 'reject']:
                # Decision made - append new review history entry
                archived_paper.status = status

                # Create new review history entry (same pattern as first-time submissions)
                new_review_entry = {
                    'conference': paper_dict['conference'],
                    'year': paper_dict['year'],
                    'score': paper_dict.get('final_score', 0.0),
                    'reviews': paper_dict.get('reviews', []),
                    'decision': status
                }

                # Accumulate review history (this is the fix!)
                archived_paper.review_history.append(new_review_entry)

                logger.info(f"[Paper Tracker]: Updated Resubmission {paper_dict['id']} ({original_conference} {original_status} -> {paper_dict['conference']} {status}) for Year {paper_dict['year']}. Review history entries: {len(archived_paper.review_history)}")

                # Remove from pending
                year = paper_dict['year']
                if year in self.pending_papers_by_year:
                    if paper_dict['id'] not in self.pending_papers_by_year[year]:
                        logger.error(f"[Paper Tracker]: Paper {paper_dict['id']} not found in pending papers for year {year}")
                    else:
                        del self.pending_papers_by_year[year][paper_dict['id']]

            else:
                raise ValueError(f"Invalid status for resubmission: {status}")

        elif paper_dict['type'] == 'submission':

            status = paper_dict['status']
            if status == 'pending':
                assert paper_dict[
                           'id'] not in self.papers_by_id, f"Submitted paper {paper_dict['id']} already exists in archive"

                review_history = []

                archived_paper = ArchivedPaper(
                    id=paper_dict['id'],
                    title=paper_dict['title'],
                    abstract=paper_dict['abstract'],
                    author_id=paper_dict['author_id'],
                    author_type=paper_dict['author_type'],
                    year=paper_dict['year'],  # The current year (updated time of the submission)
                    conference=paper_dict['conference'],
                    topics=paper_dict['topics'],
                    tags=paper_dict['tags'],
                    status=status,
                    project_start_year=paper_dict['project_start_year'],
                    project_end_year=paper_dict['project_end_year'],
                    maturity=paper_dict['maturity'],
                    review_history=review_history
                )

                self.papers_by_id[archived_paper.id] = archived_paper

                logger.info(f"[Paper Tracker]: Added new submission {paper_dict['id']} (Status: {status}) for Year {paper_dict['year']}")


            else:
                assert paper_dict['id'] in self.papers_by_id, f"Submitted paper {paper_dict['id']} not found in archive"
                assert self.get_paper_by_id(paper_dict[
                                                'id']).review_history == [], f"Should not be any review history for first-time submission {paper_dict['id']}"

                review_history = [{
                    'conference': paper_dict['conference'],
                    'year': paper_dict['year'],
                    'score': paper_dict.get('final_score', 0.0),
                    'reviews': paper_dict.get('reviews', []),
                    'decision': status
                }]
                original_conference = self.get_paper_by_id(paper_dict['id']).conference
                original_status = self.get_paper_by_id(paper_dict['id']).status
                archived_paper = self.update_paper(paper_dict['id'], status=status, conference=paper_dict['conference'],
                                                   year=paper_dict['year'], review_history=review_history)

                # Remove from pending if it exists there
                year = paper_dict['year']
                if year in self.pending_papers_by_year:
                    if not paper_dict['id'] in self.pending_papers_by_year[year]:
                        logger.error(f"[Paper Tracker]: Paper {paper_dict['id']} not found in pending papers for year {year}")
                    del self.pending_papers_by_year[year][paper_dict['id']]
                    
                logger.info(f"[Paper Tracker]: Updated Submission {paper_dict['id']} ({original_conference} {original_status} -> {paper_dict['conference']} {status}) for Year {paper_dict['year']} ")


        else:
            raise ValueError(f"Invalid paper type: {paper_dict['type']}. Must be 'resubmission' or 'submission'")

        # Update papers_by_author index (handles both single and co-authored papers)
        for aid in archived_paper.all_author_ids:
            if aid not in self.papers_by_author:
                self.papers_by_author[aid] = []
            if archived_paper not in self.papers_by_author[aid]:
                self.papers_by_author[aid].append(archived_paper)

        year = paper_dict['year']

        # Update accept / reject paper index
        if paper_dict['status'] == 'accept':
            if year not in self.accepted_papers_by_year:
                self.accepted_papers_by_year[year] = {}
            self.accepted_papers_by_year[year][archived_paper.id] = archived_paper

        elif paper_dict['status'] == 'reject':
            if year not in self.rejected_papers_by_year:
                self.rejected_papers_by_year[year] = {}
            self.rejected_papers_by_year[year][archived_paper.id] = archived_paper

        elif paper_dict['status'] == 'pending':
            # At the end of each round, this should be empty.
            if year not in self.pending_papers_by_year:
                self.pending_papers_by_year[year] = {}
            self.pending_papers_by_year[year][archived_paper.id] = archived_paper
        else:
            raise ValueError(f"Invalid status: {paper_dict['status']}. Must be 'accept', 'reject', or 'pending'")

        return archived_paper

    def update_paper(self, paper_id: str, status: str = None, conference: str = None,
                     year: int = None, **kwargs):
        """Update a paper's metadata in the paper tracker."""
        if status is not None or conference is not None or year is not None:

            assert kwargs.get("review_history", None), "Review history must be provided"
            assert paper_id in self.pending_papers_by_year[
                year], f"Paper {paper_id} not found in pending papers for year {year}"
            assert status is not None and conference is not None and year is not None, "All of status, conference, or year must be provided"
            original_status = self.pending_papers_by_year[year][paper_id].status

            self.papers_by_id[paper_id].status = status
            self.papers_by_id[paper_id].conference = conference
            self.papers_by_id[paper_id].year = year

            self.papers_by_id[paper_id].review_history.extend(kwargs.get("review_history", []))

            logger.info(f"[Paper Tracker]: {paper_id} {original_status} -> {status} for Year {year}")

        else:
            raise NotImplementedError("")

        return self.papers_by_id[paper_id]
    
    def get_papers_by_decision(self, author_id: str) -> List[ArchivedPaper]:
        results = self.get_papers_by_author(author_id)
        accepted_papers = []
        rejected_papers = []
        for paper in results:
            if paper.status == 'accept':
                accepted_papers.append(paper)
            elif paper.status == 'reject':
                rejected_papers.append(paper)
        return accepted_papers, rejected_papers
        

    def get_papers_dataframe(self, year: int) -> pd.DataFrame:
        """Get papers that have been accepted or rejected during a given year.

        Uses lead (first) author_id and author_type for DataFrame compatibility
        (e.g., set_index('author_id') requires hashable values).
        """
        list_of_papers = []
        for paper in self.accepted_papers_by_year.get(year, {}).values():
            aid = paper.author_id[0] if isinstance(paper.author_id, list) else paper.author_id
            atype = paper.author_type[0] if isinstance(paper.author_type, list) else paper.author_type
            list_of_papers.append({
                'id': paper.id,
                'title': paper.title,
                'author_id': aid,
                'author_type': atype,
                'year': paper.year,
                'status': paper.status,
                'project_start_year': paper.project_start_year,
                'project_end_year': paper.project_end_year,
                'maturity': paper.maturity,
            })
        for paper in self.rejected_papers_by_year.get(year, {}).values():
            aid = paper.author_id[0] if isinstance(paper.author_id, list) else paper.author_id
            atype = paper.author_type[0] if isinstance(paper.author_type, list) else paper.author_type
            list_of_papers.append({
                'id': paper.id,
                'title': paper.title,
                'author_id': aid,
                'author_type': atype,
                'year': paper.year,
                'status': paper.status,
                'project_start_year': paper.project_start_year,
                'project_end_year': paper.project_end_year,
                'maturity': paper.maturity,
            })
        return pd.DataFrame(list_of_papers)

    def get_papers_before_year(self, year: int) -> Tuple[List[ArchivedPaper], List[ArchivedPaper]]:
        """Get all papers accepted before a given year"""
        accepted_papers, rejected_papers = [], []
        for y in range(1, year):
            accepted_papers.extend(self.accepted_papers_by_year.get(y, {}).values())
            rejected_papers.extend(self.rejected_papers_by_year.get(y, {}).values())
        return accepted_papers, rejected_papers

    def get_papers_by_author(self, author_id: str) -> List[ArchivedPaper]:
        """Get all papers by a specific author"""
        return self.papers_by_author.get(author_id, [])

    def get_paper_by_id(self, paper_id: str) -> Optional[ArchivedPaper]:
        """Get a specific paper by ID"""
        return self.papers_by_id.get(paper_id)

    def get_papers_as_documents(self, year: Optional[int] = None) -> List[Document]:
        """
        Get papers as LangChain Documents for RAG integration

        Args:
            year: If provided, only return papers before this year

        Returns:
            List of LangChain Document objects
        """
        if year is not None:
            papers = self.get_papers_before_year(year)
        else:
            papers = list(self.papers_by_id.values())

        return [paper.to_langchain_document() for paper in papers]

    def get_statistics(self) -> Dict:
        """Get archive statistics"""
        total_papers = len(self.papers_by_id)
        accepted_papers_by_year = {year: len(papers) for year, papers in self.accepted_papers_by_year.items()}
        rejected_papers_by_year = {year: len(papers) for year, papers in self.rejected_papers_by_year.items()}
        total_citations = sum(paper.citation_count for paper in self.papers_by_id.values())

        return {
            'total_papers': total_papers,
            'accepted_papers_by_year': accepted_papers_by_year,
            'rejected_papers_by_year': rejected_papers_by_year,
            'total_citations': total_citations,
            'years_active': sorted(self.accepted_papers_by_year.keys()) + sorted(self.rejected_papers_by_year.keys())
        }

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'output_dir': str(self.output_dir),
            'papers': [paper.to_dict() for paper in self.papers_by_id.values()]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PaperTracker':
        """Deserialize from dictionary"""
        archive = cls(output_dir=data['output_dir'])

        # Reload papers
        for paper_dict in data.get('papers', []):
            paper = ArchivedPaper.from_dict(paper_dict)
            archive.papers_by_id[paper.id] = paper

            if paper.status == 'accept':
                if paper.year not in archive.accepted_papers_by_year:
                    archive.accepted_papers_by_year[paper.year] = {}
                archive.accepted_papers_by_year[paper.year][paper.id] = paper

            elif paper.status == 'reject':
                if paper.year not in archive.rejected_papers_by_year:
                    archive.rejected_papers_by_year[paper.year] = {}
                archive.rejected_papers_by_year[paper.year][paper.id] = paper

            elif paper.status == 'pending':
                if paper.year not in archive.pending_papers_by_year:
                    archive.pending_papers_by_year[paper.year] = {}
                archive.pending_papers_by_year[paper.year][paper.id] = paper

            for aid in paper.all_author_ids:
                if aid not in archive.papers_by_author:
                    archive.papers_by_author[aid] = []
                archive.papers_by_author[aid].append(paper)

        return archive
