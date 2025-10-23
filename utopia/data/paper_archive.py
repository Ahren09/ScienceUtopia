"""
Paper Archive System for Managing Accepted Papers Across Simulation Years

This module provides functionality to persist and retrieve accepted papers
across multiple simulation years, enabling citation mechanisms and paper
references.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from langchain_core.documents import Document
import json
from pathlib import Path


@dataclass
class ArchivedPaper:
    """Represents an accepted paper in the archive"""
    id: str
    title: str
    abstract: str  # First 500 chars or full abstract
    author_id: str
    year: int
    conference: str
    decision: str
    topics: List[str] = field(default_factory=list)
    final_score: float = 0.0
    citation_count: int = 0  # Number of times this paper has been cited
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'abstract': self.abstract,
            'author_id': self.author_id,
            'year': self.year,
            'conference': self.conference,
            'topics': self.topics,
            'decision': self.decision,
            'final_score': self.final_score,
            'citation_count': self.citation_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchivedPaper':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            abstract=data['abstract'],
            author_id=data['author_id'],
            year=data['year'],
            conference=data['conference'],
            topics=data.get('topics', []),
            final_score=data.get('final_score', 0.0),
            citation_count=data.get('citation_count', 0)
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
                'topics': ', '.join(self.topics),
                'final_score': self.final_score,
                'citation_count': self.citation_count,
                'source': 'simulation'  # Distinguish from external papers
            }
        )

    def get_preview(self, max_abstract_len: int = 100) -> str:
        """Get a preview string for display to agents"""
        abstract_preview = self.abstract[:max_abstract_len]
        if len(self.abstract) > max_abstract_len:
            abstract_preview += "..."
        return f"[{self.year}] {self.title}\n  Abstract: {abstract_preview}"


class PaperArchive:
    """Manages accepted papers across simulation years"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.papers_by_id: Dict[str, ArchivedPaper] = {}
        self.accepted_papers_by_year: Dict[int, List[ArchivedPaper]] = {}
        self.rejected_papers_by_year: Dict[int, List[ArchivedPaper]] = {}
        self.papers_by_author: Dict[str, List[ArchivedPaper]] = {}
        self.archive_file = self.output_dir / "paper_archive.json"

    def add_paper(self, paper_dict: Dict, year: int, decision: str) -> ArchivedPaper:
        """
        Add an accepted paper to the archive

        Args:
            paper_dict: Paper dictionary with keys: id, title, content, author_id,
                       conference, topics, final_score
            year: Simulation year the paper was accepted

        Returns:
            ArchivedPaper object
        """
        archived_paper = ArchivedPaper(
            id=paper_dict['id'],
            title=paper_dict['title'],
            abstract=paper_dict.get('content', ''),
            author_id=paper_dict['author_id'],
            year=year,
            conference=paper_dict['conference'],
            topics=paper_dict.get('topics', []),
            final_score=paper_dict.get('final_score', 0.0),
            decision=decision,
            citation_count=0
        )

        # Store in all indices
        self.papers_by_id[archived_paper.id] = archived_paper
        
        if decision == 'accept':

            if year not in self.accepted_papers_by_year:
                self.accepted_papers_by_year[year] = []
            self.accepted_papers_by_year[year].append(archived_paper)
        elif decision == 'reject':
            if year not in self.rejected_papers_by_year:
                self.rejected_papers_by_year[year] = []
            self.rejected_papers_by_year[year].append(archived_paper)
            
        else:
            raise ValueError(f"Invalid decision: {decision}. Must be 'accepted' or 'rejected'")

        return archived_paper

    def get_papers_before_year(self, year: int) -> List[ArchivedPaper]:
        """Get all papers accepted before a given year"""
        accepted_papers, rejected_papers = [], []
        for y in range(1, year):
            accepted_papers.extend(self.accepted_papers_by_year.get(y, []))
            rejected_papers.extend(self.rejected_papers_by_year.get(y, []))
        return accepted_papers, rejected_papers

    def get_papers_by_author(self, author_id: str) -> List[ArchivedPaper]:
        """Get all papers by a specific author"""
        return self.papers_by_author.get(author_id, [])

    def get_paper_by_id(self, paper_id: str) -> Optional[ArchivedPaper]:
        """Get a specific paper by ID"""
        return self.papers_by_id.get(paper_id)

    def increment_citation_count(self, paper_id: str):
        """Increment citation count for a paper"""
        if paper_id in self.papers_by_id:
            self.papers_by_id[paper_id].citation_count += 1

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

    def get_papers_preview(self, year: int, max_papers: int = 50) -> List[str]:
        """
        Get preview strings of papers for display to agents

        Args:
            year: Only show papers before this year
            max_papers: Maximum number of papers to return

        Returns:
            List of preview strings
        """
        papers = self.get_papers_before_year(year)
        # Sort by year (descending) and score (descending)
        papers = sorted(papers, key=lambda p: (-p.year, -p.final_score))
        papers = papers[:max_papers]
        return [paper.get_preview() for paper in papers]


    def load(self):
        """Load archive from disk"""
        if not self.archive_file.exists():
            return

        with open(self.archive_file, 'r') as f:
            data = json.load(f)

        # Clear existing data
        self.papers_by_id.clear()
        self.accepted_papers_by_year.clear()
        self.rejected_papers_by_year.clear()
        self.papers_by_author.clear()

        # Reload papers
        for paper_dict in data['papers']:
            paper = ArchivedPaper.from_dict(paper_dict)
            self.papers_by_id[paper.id] = paper
            
            if paper.decision == 'accepted':

                if paper.year not in self.accepted_papers_by_year:
                    self.accepted_papers_by_year[paper.year] = []
                self.accepted_papers_by_year[paper.year].append(paper)
                
            elif paper.decision == 'rejected':
                if paper.year not in self.rejected_papers_by_year:
                    self.rejected_papers_by_year[paper.year] = []
                self.rejected_papers_by_year[paper.year].append(paper)

            if paper.author_id not in self.papers_by_author:
                self.papers_by_author[paper.author_id] = []
            self.papers_by_author[paper.author_id].append(paper)

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
    def from_dict(cls, data: Dict) -> 'PaperArchive':
        """Deserialize from dictionary"""
        archive = cls(output_dir=data['output_dir'])

        # Reload papers
        for paper_dict in data.get('papers', []):
            paper = ArchivedPaper.from_dict(paper_dict)
            archive.papers_by_id[paper.id] = paper
            
            if paper.decision == 'accepted':
                if paper.year not in archive.accepted_papers_by_year:
                    archive.accepted_papers_by_year[paper.year] = []
                archive.accepted_papers_by_year[paper.year].append(paper)
                
            elif paper.decision == 'rejected':
                if paper.year not in archive.rejected_papers_by_year:
                    archive.rejected_papers_by_year[paper.year] = []
                archive.rejected_papers_by_year[paper.year].append(paper)

            if paper.author_id not in archive.papers_by_author:
                archive.papers_by_author[paper.author_id] = []
            archive.papers_by_author[paper.author_id].append(paper)

        return archive
