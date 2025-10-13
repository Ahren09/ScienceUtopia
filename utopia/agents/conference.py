"""
Conference System for Multi-Agent Scientific Ecosystem

Supports multiple conferences targeting different research topics.
Each conference has its own submission deadlines, review processes, and acceptance criteria.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
import time
from enum import Enum


class ConferenceStatus(Enum):
    """Status of a conference in the yearly cycle"""
    ACCEPTING_SUBMISSIONS = "accepting_submissions"
    UNDER_REVIEW = "under_review"
    DECISIONS_MADE = "decisions_made"
    CONCLUDED = "concluded"


@dataclass
class Conference:
    """Represents a scientific conference with specific research focus

    Multiple conferences can run in parallel, each targeting different topics.
    This enables specialization and realistic simulation of academic ecosystem diversity.
    """

    name: str
    topics: List[str]  # Research topics this conference focuses on
    acceptance_rate: float = 0.3  # Target acceptance rate (0-1)
    review_criteria: Dict[str, float] = field(default_factory=dict)  # Criteria weights
    conference_id: str = None
    year: int = 0
    status: ConferenceStatus = ConferenceStatus.ACCEPTING_SUBMISSIONS

    # Paper tracking
    submitted_papers: List[Dict] = field(default_factory=list)
    accepted_papers: List[Dict] = field(default_factory=list)
    rejected_papers: List[Dict] = field(default_factory=list)

    # Review tracking
    reviews: Dict[str, List[Dict]] = field(default_factory=dict)  # paper_id -> [reviews]

    def __post_init__(self):
        if self.conference_id is None:
            self.conference_id = f"conf_{self.name.lower().replace(' ', '_')}_{int(time.time() * 1000) % 10000}"

        # Set default review criteria if not provided
        if not self.review_criteria:
            self.review_criteria = {
                'novelty': 0.3,
                'technical_quality': 0.3,
                'clarity': 0.2,
                'relevance': 0.2
            }

    def is_topic_match(self, paper_topics: List[str]) -> bool:
        """Check if paper topics match conference topics

        Args:
            paper_topics: List of topics associated with the paper

        Returns:
            True if there's at least one topic overlap
        """
        paper_topic_set = set(t.lower() for t in paper_topics)
        conf_topic_set = set(t.lower() for t in self.topics)
        return bool(paper_topic_set.intersection(conf_topic_set))

    def submit_paper(self, paper: Dict) -> bool:
        """Submit a paper to this conference

        Args:
            paper: Paper dictionary with metadata

        Returns:
            True if submission accepted, False if rejected (e.g., topic mismatch)
        """
        if self.status != ConferenceStatus.ACCEPTING_SUBMISSIONS:
            return False

        # Check topic match
        paper_topics = paper.get('topics', [])
        if paper_topics and not self.is_topic_match(paper_topics):
            return False

        # Add conference metadata to paper
        paper['conference_id'] = self.conference_id
        paper['conference_name'] = self.name
        paper['submission_time'] = time.time()

        self.submitted_papers.append(paper)
        return True

    def start_review_process(self):
        """Transition conference to review phase"""
        if self.status == ConferenceStatus.ACCEPTING_SUBMISSIONS:
            self.status = ConferenceStatus.UNDER_REVIEW

    def add_review(self, paper_id: str, review: Dict):
        """Add a review for a submitted paper

        Args:
            paper_id: ID of the paper being reviewed
            review: Review dictionary from reviewer
        """
        if paper_id not in self.reviews:
            self.reviews[paper_id] = []

        review['conference_id'] = self.conference_id
        review['review_time'] = time.time()
        self.reviews[paper_id].append(review)

    def make_acceptance_decisions(self) -> Dict[str, List[Dict]]:
        """Make final acceptance/rejection decisions

        Uses aggregate review scores and acceptance rate to determine which papers to accept.

        Returns:
            Dictionary with 'accepted' and 'rejected' lists
        """
        if self.status != ConferenceStatus.UNDER_REVIEW:
            return {'accepted': [], 'rejected': []}

        # Calculate aggregate scores for each paper
        paper_scores = []
        for paper in self.submitted_papers:
            paper_id = paper.get('id', paper.get('paper_id'))
            paper_reviews = self.reviews.get(paper_id, [])

            if not paper_reviews:
                # No reviews - reject by default
                avg_score = 0.0
            else:
                # Calculate weighted average score
                avg_score = self._calculate_paper_score(paper_reviews)

            paper_scores.append({
                'paper': paper,
                'score': avg_score,
                'num_reviews': len(paper_reviews)
            })

        # Sort papers by score
        paper_scores.sort(key=lambda x: x['score'], reverse=True)

        # Accept top papers according to acceptance rate
        num_to_accept = int(len(paper_scores) * self.acceptance_rate)

        for i, item in enumerate(paper_scores):
            paper = item['paper']
            paper['final_score'] = item['score']
            paper['decision_time'] = time.time()

            if i < num_to_accept:
                paper['decision'] = 'accepted'
                self.accepted_papers.append(paper)
            else:
                paper['decision'] = 'rejected'
                self.rejected_papers.append(paper)

        self.status = ConferenceStatus.DECISIONS_MADE

        return {
            'accepted': self.accepted_papers,
            'rejected': self.rejected_papers
        }

    def _calculate_paper_score(self, reviews: List[Dict]) -> float:
        """Calculate aggregate score for a paper from its reviews

        Args:
            reviews: List of review dictionaries

        Returns:
            Weighted average score (0-10)
        """
        if not reviews:
            return 0.0

        # Extract scores from reviews
        scores = []
        for review in reviews:
            overall_score = review.get('overall_score', 5.0)
            scores.append(overall_score)

        # Calculate average
        return sum(scores) / len(scores)

    def conclude_conference(self):
        """Mark conference as concluded for the year"""
        if self.status == ConferenceStatus.DECISIONS_MADE:
            self.status = ConferenceStatus.CONCLUDED

    def reset_for_next_year(self, new_year: int):
        """Reset conference for the next year's cycle

        Args:
            new_year: The year number for the next cycle
        """
        self.year = new_year
        self.status = ConferenceStatus.ACCEPTING_SUBMISSIONS
        self.submitted_papers = []
        self.accepted_papers = []
        self.rejected_papers = []
        self.reviews = {}

    def get_statistics(self) -> Dict:
        """Get conference statistics

        Returns:
            Dictionary with conference metrics
        """
        return {
            'conference_id': self.conference_id,
            'name': self.name,
            'year': self.year,
            'topics': self.topics,
            'status': self.status.value,
            'num_submissions': len(self.submitted_papers),
            'num_accepted': len(self.accepted_papers),
            'num_rejected': len(self.rejected_papers),
            'actual_acceptance_rate': len(self.accepted_papers) / len(self.submitted_papers) if self.submitted_papers else 0.0,
            'avg_reviews_per_paper': sum(len(reviews) for reviews in self.reviews.values()) / len(self.reviews) if self.reviews else 0.0
        }


class ConferenceSystem:
    """Manages multiple conferences in the ecosystem

    Allows papers to be submitted to appropriate conferences based on topic matching.
    Coordinates the review and acceptance processes across all conferences.
    """

    def __init__(self, conferences: Optional[List[Conference]] = None):
        """Initialize the conference system

        Args:
            conferences: Optional list of Conference objects to initialize with
        """
        self.conferences: List[Conference] = conferences or []
        self.conference_map: Dict[str, Conference] = {}

        for conf in self.conferences:
            self.conference_map[conf.conference_id] = conf

    def add_conference(self, conference: Conference):
        """Add a conference to the system

        Args:
            conference: Conference object to add
        """
        self.conferences.append(conference)
        self.conference_map[conference.conference_id] = conference

    def get_matching_conferences(self, paper_topics: List[str]) -> List[Conference]:
        """Find conferences that match paper topics

        Args:
            paper_topics: List of topics for the paper

        Returns:
            List of conferences that match the paper's topics
        """
        matching = []
        for conf in self.conferences:
            if conf.status == ConferenceStatus.ACCEPTING_SUBMISSIONS and conf.is_topic_match(paper_topics):
                matching.append(conf)
        return matching

    def submit_paper_to_best_match(self, paper: Dict) -> Optional[Conference]:
        """Submit paper to the best matching conference

        Finds conferences matching the paper's topics and submits to the first match.
        In future versions, could implement more sophisticated matching algorithms.

        Args:
            paper: Paper dictionary with 'topics' field

        Returns:
            Conference object where paper was submitted, or None if no match found
        """
        paper_topics = paper.get('topics', [])
        matching_conferences = self.get_matching_conferences(paper_topics)

        if not matching_conferences:
            return None

        # Submit to first matching conference (could be made more sophisticated)
        conference = matching_conferences[0]
        success = conference.submit_paper(paper)

        return conference if success else None

    def start_all_review_processes(self):
        """Start review process for all conferences"""
        for conf in self.conferences:
            conf.start_review_process()

    def make_all_decisions(self) -> Dict[str, Dict]:
        """Make acceptance decisions for all conferences

        Returns:
            Dictionary mapping conference_id to decision results
        """
        results = {}
        for conf in self.conferences:
            decisions = conf.make_acceptance_decisions()
            results[conf.conference_id] = decisions
        return results

    def conclude_all_conferences(self):
        """Mark all conferences as concluded"""
        for conf in self.conferences:
            conf.conclude_conference()

    def reset_all_for_next_year(self, new_year: int):
        """Reset all conferences for the next year

        Args:
            new_year: The year number for the next cycle
        """
        for conf in self.conferences:
            conf.reset_for_next_year(new_year)

    def get_all_statistics(self) -> List[Dict]:
        """Get statistics for all conferences

        Returns:
            List of statistics dictionaries, one per conference
        """
        return [conf.get_statistics() for conf in self.conferences]

    def get_conference_by_id(self, conference_id: str) -> Optional[Conference]:
        """Get conference by ID

        Args:
            conference_id: Conference identifier

        Returns:
            Conference object or None if not found
        """
        return self.conference_map.get(conference_id)


# Predefined conference templates for common research areas
PREDEFINED_CONFERENCES = {
    'ICML': Conference(
        name='International Conference on Machine Learning',
        topics=['machine_learning', 'deep_learning', 'neural_networks', 'reinforcement_learning', 'optimization'],
        acceptance_rate=0.25
    ),
    'NeurIPS': Conference(
        name='Neural Information Processing Systems',
        topics=['neural_networks', 'deep_learning', 'neuroscience', 'theoretical_ml', 'reinforcement_learning'],
        acceptance_rate=0.20
    ),
    'ICLR': Conference(
        name='International Conference on Learning Representations',
        topics=['representation_learning', 'deep_learning', 'neural_networks', 'generative_models'],
        acceptance_rate=0.27
    ),
    'CVPR': Conference(
        name='Computer Vision and Pattern Recognition',
        topics=['computer_vision', 'image_recognition', 'video_analysis', 'visual_perception'],
        acceptance_rate=0.29
    ),
    'NLP': Conference(
        name='Natural Language Processing Conference',
        topics=['natural_language_processing', 'language_models', 'text_mining', 'linguistics'],
        acceptance_rate=0.30
    ),
    'KDD': Conference(
        name='Knowledge Discovery and Data Mining',
        topics=['data_mining', 'knowledge_discovery', 'applied_ml', 'data_science'],
        acceptance_rate=0.15
    )
}


def create_default_conference_system() -> ConferenceSystem:
    """Create a conference system with predefined conferences

    Returns:
        ConferenceSystem with standard ML/AI conferences
    """
    conferences = list(PREDEFINED_CONFERENCES.values())
    return ConferenceSystem(conferences)
