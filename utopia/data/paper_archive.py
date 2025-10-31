"""
Paper Archive System for Managing Accepted Papers Across Simulation Years

This module provides functionality to persist and retrieve accepted papers
across multiple simulation years, enabling citation mechanisms and paper
references.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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
    
    review_history: List[Dict] = field(default_factory=list)  # [{"conference": str, "year": int, "score": float}]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for both serialization and resubmission)"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.abstract,  # Use 'content' for compatibility with paper submission
            'abstract': self.abstract,
            'author_id': self.author_id,
            'year': self.year,
            'conference': self.conference,
            'topics': self.topics,
            'decision': self.decision,
            'review_history': self.review_history
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ArchivedPaper':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            title=data['title'],
            abstract=data.get('abstract', data.get('content', '')),
            author_id=data['author_id'],
            year=data['year'],
            conference=data['conference'],
            topics=data.get('topics', []),
            decision=data['decision'],
            review_history=data.get('review_history', [])
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
                'source': 'simulation'
            }
        )


class PaperArchive:
    """Manages accepted papers across simulation years"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.papers_by_id: Dict[str, ArchivedPaper] = {}
        self.accepted_papers_by_year: Dict[int, List[ArchivedPaper]] = {}
        self.rejected_papers_by_year: Dict[int, List[ArchivedPaper]] = {}
        self.papers_by_author: Dict[str, List[ArchivedPaper]] = {}
        self.archive_file = self.output_dir / "paper_archive.json"

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
        
        # Check if paper already exists (resubmission case)
        if paper_dict['type'] == 'resubmission':
            
            assert paper_dict['id'] in self.papers_by_id, f"Resubmitted paper {paper_dict['id']} not found in archive"
            archived_paper = self.papers_by_id[paper_dict['id']]
            
            # Update the latest conference, year, and decision
            archived_paper.conference = paper_dict['conference']
            archived_paper.year = paper_dict['year']
            archived_paper.decision = paper_dict['decision']
            # Append to review history
            archived_paper.review_history.append({
                'conference': paper_dict['conference'],
                'year': paper_dict['year'],
                'score': paper_dict['final_score'],
                'reviews': paper_dict['reviews'],
                'decision': paper_dict['decision'],
            })
            

        elif paper_dict['type'] == 'submission':
            
            # New paper
            assert paper_dict['id'] not in self.papers_by_id, f"Submitted paper {paper_dict['id']} already exists in archive"
            archived_paper = ArchivedPaper(
                id=paper_dict['id'],
                title=paper_dict['title'],
                abstract=paper_dict.get('content', ''),
                author_id=paper_dict['author_id'],
                year=paper_dict['year'],
                conference=paper_dict['conference'],
                topics=paper_dict.get('topics', []),
                decision=paper_dict['decision'],
                review_history=[{
                    'conference': paper_dict['conference'], 
                    'year': paper_dict['year'], 
                    'score': paper_dict['final_score'], 
                    'reviews': paper_dict['reviews'],
                    'decision': paper_dict['decision']
                }]
            )
            self.papers_by_id[archived_paper.id] = archived_paper
            
        else:
            raise ValueError(f"Invalid paper type: {paper_dict['type']}. Must be 'resubmission' or 'submission'")

        # Update papers_by_author index
        if archived_paper.author_id not in self.papers_by_author:
            self.papers_by_author[archived_paper.author_id] = []
        
        if archived_paper not in self.papers_by_author[archived_paper.author_id]:
            self.papers_by_author[archived_paper.author_id].append(archived_paper)

        year = paper_dict['year']
        # Update accept / reject paper index
        if paper_dict['decision'] == 'accept':

            if year not in self.accepted_papers_by_year:
                self.accepted_papers_by_year[year] = []
            self.accepted_papers_by_year[year].append(archived_paper)
            
        elif paper_dict['decision'] == 'reject':
            if year not in self.rejected_papers_by_year:
                self.rejected_papers_by_year[year] = []
            self.rejected_papers_by_year[year].append(archived_paper)
            
        else:
            raise ValueError(f"Invalid decision: { paper_dict['decision']}. Must be 'accepted' or 'rejected'")

        return archived_paper


    def update_paper(self, paper_id: str, citations: List[str] = None, decision: str = None, conference: str = None, year: int = None):
        
        if citations is not None:
            citation_count = len(self.papers_by_id[paper_id].citations)
            self.papers_by_id[paper_id].citations = list(set(self.papers_by_id[paper_id].citations + citations))
            logger.info(f"Updated citations for paper {paper_id} ({citation_count}) -> {len(self.papers_by_id[paper_id].citations)})")
        
        elif decision is not None or conference is not None or year is not None:
            
            assert decision is not None and conference is not None and year is not None, "All of decision, conference, or year must be provided"
            self.papers_by_id[paper_id].decision = decision
            self.papers_by_id[paper_id].conference = conference
            self.papers_by_id[paper_id].year = year
            
        else:
            raise NotImplementedError("")
        
        return self.papers_by_id[paper_id]

    def get_papers_before_year(self, year: int) -> Tuple[List[ArchivedPaper], List[ArchivedPaper]]:
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
    def from_dict(cls, data: Dict) -> 'PaperArchive':
        """Deserialize from dictionary"""
        archive = cls(output_dir=data['output_dir'])

        # Reload papers
        for paper_dict in data.get('papers', []):
            paper = ArchivedPaper.from_dict(paper_dict)
            archive.papers_by_id[paper.id] = paper
            
            if paper.decision == 'accept':
                if paper.year not in archive.accepted_papers_by_year:
                    archive.accepted_papers_by_year[paper.year] = []
                archive.accepted_papers_by_year[paper.year].append(paper)
                
            elif paper.decision == 'reject':
                if paper.year not in archive.rejected_papers_by_year:
                    archive.rejected_papers_by_year[paper.year] = []
                archive.rejected_papers_by_year[paper.year].append(paper)

            if paper.author_id not in archive.papers_by_author:
                archive.papers_by_author[paper.author_id] = []
            archive.papers_by_author[paper.author_id].append(paper)

        return archive
