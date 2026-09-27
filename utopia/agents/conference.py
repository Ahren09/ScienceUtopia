"""
Conference System for Multi-Agent Scientific Ecosystem

Supports multiple conferences targeting different research topics.
Each conference has its own submission deadlines, review processes, and acceptance criteria.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import random

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

    Attributes:
        name: Conference name
        topics: All arXiv categories accepted (sorted by preference, top 3 are most preferred)
        most_preferred_topics: Top 3 most preferred topics (auto-extracted from first 3 in topics)
        acceptance_rate: Target acceptance rate (0-1), uses default from config if None
        review_criteria: Criteria weights for review scoring
        conference_id: Unique identifier
        year: Current conference year
        status: Current status in the conference cycle
    """

    conference_id: str
    name: str
    topics: List[str]  # All research topics (first 3 are most preferred, sorted by preference)
    category: str = None  # General research category (e.g., "Artificial Intelligence", "Computer Networks")
    acceptance_rate: float = None  # Target acceptance rate (0-1), None = use default from config
    year: int = 1
    status: ConferenceStatus = ConferenceStatus.ACCEPTING_SUBMISSIONS

    # Paper tracking
    submitted_papers: List[Dict] = field(default_factory=list)
    decisions: Dict[str, List[Dict]] = field(default_factory=lambda: {
        'accept': [],
        'reject': [],
    })

    authors_to_accepted_papers: Dict[str, List[Dict]] = field(
        default_factory=lambda: defaultdict(list))  # author_id -> [accepted_papers]
    authors_to_rejected_papers: Dict[str, List[Dict]] = field(
        default_factory=lambda: defaultdict(list))  # author_id -> [rejected_papers]
    # Review tracking
    reviews: Dict[str, List[Dict]] = field(default_factory=dict)  # paper_id -> [reviews]
    # --acceptance_mode fixed_slots: accepted count frozen at the first decision year
    # (None = never frozen; legacy rate mode never touches it)
    fixed_slots: Optional[int] = None

    @property
    def most_preferred_topics(self) -> List[str]:
        """Get the top 3 most preferred topics for this conference"""
        return self.topics[:3] if len(self.topics) >= 3 else self.topics

    def __post_init__(self):
        from utopia.config import SIMULATION_CONFIG

        if self.conference_id is None:
            self.conference_id = f"conf_{self.name.lower().replace(' ', '_')}_{int(time.time() * 1000) % 10000}"

        # Use default acceptance rate if not specified
        if self.acceptance_rate is None or SIMULATION_CONFIG['conference']['use_default_acceptance_rate']:
            self.acceptance_rate = SIMULATION_CONFIG['conference']['acceptance_rate']

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

    def submit_paper(self, paper: Dict, year: int) -> bool:
        """Submit a paper to this conference

        Args:
            paper: Paper dictionary with metadata

        Returns:
            True if submission accepted, False if rejected (e.g., topic mismatch)
        """
        if self.status != ConferenceStatus.ACCEPTING_SUBMISSIONS:
            return False

        # # Check topic match
        # paper_topics = paper.get('topics', [])
        # if paper_topics and not self.is_topic_match(paper_topics):
        #     return False

        # Add conference metadata to paper
        if 'conference' not in paper:
            paper['conference'] = self.conference_id
        else:
            assert paper['conference'] == self.conference_id

        paper['year'] = year
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
        review['year'] = self.year
        review['review_time'] = time.time()
        self.reviews[paper_id].append(review)

    def make_acceptance_decisions(self, tiebreak_seed: Optional[int] = None,
                                  acceptance_mode: str = 'rate') -> None:
        """Make final acceptance/rejection decisions

        Uses aggregate review scores and acceptance rate to determine which papers to accept.

        Args:
            tiebreak_seed: None (default, legacy) keeps Python's stable sort, i.e. ties at the
                cutoff are resolved by submission order. An int seeds a deterministic
                per-conference random key that breaks ties (--acceptance_tiebreak seeded).
            acceptance_mode: 'rate' (default, legacy) accepts round(n x acceptance_rate);
                'fixed_slots' freezes the accepted count at its first-decision value.
        """
        assert acceptance_mode in ('rate', 'fixed_slots'), acceptance_mode
        if self.status != ConferenceStatus.UNDER_REVIEW:
            return

        # Calculate aggregate scores for each paper
        paper_scores = []
        for paper in self.submitted_papers:
            paper_id = paper.get('id', paper.get('paper_id'))
            paper_reviews = self.reviews.get(paper_id, [])

            if not paper_reviews:
                # No reviews - reject by default
                raise ValueError(f"Paper {paper_id} has no reviews")
            else:
                # Calculate weighted average score
                avg_score = self._calculate_paper_score(paper_reviews)

            paper["reviews"] = paper_reviews

            paper_scores.append({
                'paper': paper,
                'score': avg_score,
                'num_reviews': len(paper_reviews),
            })

        # Sort papers by score
        if tiebreak_seed is None:
            paper_scores.sort(key=lambda x: x['score'], reverse=True)
        else:
            from utopia.utils.seeding import derive_seed
            rng = random.Random(derive_seed(tiebreak_seed, self.conference_id))
            tiebreak_keys = [rng.random() for _ in paper_scores]
            order = sorted(range(len(paper_scores)),
                           key=lambda i: (-paper_scores[i]['score'], tiebreak_keys[i]))
            paper_scores = [paper_scores[i] for i in order]

        # Accept top papers according to acceptance rate
        num_to_accept = max(1, round(len(paper_scores) * self.acceptance_rate))
        if acceptance_mode == 'fixed_slots':
            if self.fixed_slots is None:
                self.fixed_slots = num_to_accept  # anchored at the first decision year
            num_to_accept = min(self.fixed_slots, len(paper_scores))

        for i, item in enumerate(paper_scores):
            paper = item['paper']
            paper['final_score'] = item['score']
            paper['decision_time'] = time.time()

            if i < num_to_accept:
                paper['status'] = 'accept'

                """
                paper_decision = {
                    'id': paper['id'],
                    'author_id': paper['author_id'],
                    'year': self.year,
                    'score': item['score'],
                    'conference_id': self.conference_id,
                    'status': 'accept',
                    'type': paper['type'],
                }
                
                """

                self.decisions['accept'].append(paper)
                author_ids = paper['author_id'] if isinstance(paper['author_id'], list) else [paper['author_id']]
                for aid in author_ids:
                    self.authors_to_accepted_papers[aid].append(paper)



            else:
                paper['status'] = 'reject'

                """
                paper_decision = {
                    'id': paper['id'],
                    'author_id': paper['author_id'],
                    'year': self.year,
                    'score': item['score'],
                    'conference_id': self.conference_id,
                    'status': 'reject',
                    'type': paper['type'],
                }
                """

                self.decisions['reject'].append(paper)
                author_ids = paper['author_id'] if isinstance(paper['author_id'], list) else [paper['author_id']]
                for aid in author_ids:
                    self.authors_to_rejected_papers[aid].append(paper)

        self.status = ConferenceStatus.DECISIONS_MADE

        print(
            f"Conference {self.conference_id}: Total: {len(self.submitted_papers)}. Accepted: {len(self.decisions['accept'])}. Reject: {len(self.decisions['reject'])}. ")

        self.submitted_papers.clear()

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
            scores.append(review['overall_score'])

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
        self.decisions['accept'].clear()
        self.decisions['reject'].clear()
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
            'num_accepted': len(self.decisions['accept']),
            'num_rejected': len(self.decisions['reject']),
            'actual_acceptance_rate': len(self.decisions['accept']) / len(
                self.submitted_papers) if self.submitted_papers else 0.0,
            'avg_reviews_per_paper': sum(len(reviews) for reviews in self.reviews.values()) / len(
                self.reviews) if self.reviews else 0.0
        }

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'conference_id': self.conference_id,
            'name': self.name,
            'topics': self.topics,
            'category': self.category,
            'acceptance_rate': self.acceptance_rate,
            'year': self.year,
            'decisions': self.decisions,
            'authors_to_accepted_papers': self.authors_to_accepted_papers,
            'authors_to_rejected_papers': self.authors_to_rejected_papers,
            'status': self.status.value,  # Convert enum to string
            'reviews': self.reviews,
            'fixed_slots': self.fixed_slots,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Conference':
        """Deserialize from dictionary"""
        conf = cls(
            conference_id=data['conference_id'],
            name=data['name'],
            topics=data['topics'],
            category=data.get('category'),
            acceptance_rate=data.get('acceptance_rate'),
            year=data.get('year', 0)
        )
        # Restore status
        conf.status = ConferenceStatus(data.get('status', ConferenceStatus.ACCEPTING_SUBMISSIONS.value))
        # Restore papers and reviews
        conf.submitted_papers = data.get('submitted_papers', [])
        conf.decisions = defaultdict(list, data.get('decisions', {}))
        conf.authors_to_accepted_papers = defaultdict(list, data.get('authors_to_accepted_papers', {}))
        conf.authors_to_rejected_papers = defaultdict(list, data.get('authors_to_rejected_papers', {}))
        conf.reviews = defaultdict(list, data.get('reviews', {}))
        conf.fixed_slots = data.get('fixed_slots')
        return conf


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

        self.authors_to_accepted_papers: Dict[str, List[Dict]] = defaultdict(list)  # author_id -> [accepted_papers]
        self.authors_to_rejected_papers: Dict[str, List[Dict]] = defaultdict(list)  # author_id -> [rejected_papers]

        for conf in self.conferences:
            self.conference_map[conf.conference_id] = conf

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
                
        if not matching:
            matching = random.sample(self.conferences, min(3, len(self.conferences)))
        return matching

    def start_all_review_processes(self):
        """Start review process for all conferences"""
        for conf in self.conferences:
            conf.start_review_process()

    def make_all_decisions(self, tiebreak_seed: Optional[int] = None,
                           acceptance_mode: str = 'rate') -> Dict[str, Dict]:
        """Make acceptance decisions for all conferences (see Conference.make_acceptance_decisions)
        """

        for conf in self.conferences:
            conf.make_acceptance_decisions(tiebreak_seed=tiebreak_seed, acceptance_mode=acceptance_mode)

            for paper in conf.decisions['accept']:
                author_ids = paper['author_id'] if isinstance(paper['author_id'], list) else [paper['author_id']]
                for aid in author_ids:
                    self.authors_to_accepted_papers[aid].append(paper)
            for paper in conf.decisions['reject']:
                author_ids = paper['author_id'] if isinstance(paper['author_id'], list) else [paper['author_id']]
                for aid in author_ids:
                    self.authors_to_rejected_papers[aid].append(paper)

    def conclude_all_conferences(self):
        """Mark all conferences as concluded"""
        for conf in self.conferences:
            conf.conclude_conference()

    def reset_all_for_new_year(self, new_year: int):
        """Reset all conferences for the next year

        Args:
            new_year: The year number for the next cycle
        """
        self.authors_to_accepted_papers.clear()
        self.authors_to_rejected_papers.clear()

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

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'conferences': [conf.to_dict() for conf in self.conferences]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConferenceSystem':
        """Deserialize from dictionary"""
        conferences = [Conference.from_dict(conf_data) for conf_data in data.get('conferences', [])]

        conference_system = cls(conferences=conferences)
        return conference_system


CONFERENCES = {
    # Computer Architecture / Parallel & Distributed Computing / Storage Systems
    'ASPLOS': Conference(
        name='Architectural Support for Programming Languages and Operating Systems',
        topics=['cs.AR', 'cs.OS', 'cs.PL', 'cs.DC', 'cs.PF', 'cs.SE', 'cs.LG', 'cs.NI'],
        category='Computer Architecture',
        acceptance_rate=0.18,
        conference_id='ASPLOS'
    ),
    'FAST': Conference(
        name='Conference on File and Storage Technologies',
        topics=['cs.DC', 'cs.AR', 'cs.OS', 'cs.DB', 'cs.PF', 'cs.NI', 'cs.CR', 'cs.SE'],
        category='Computer Architecture',
        acceptance_rate=0.20,
        conference_id='FAST'
    ),
    'HPCA': Conference(
        name='High-Performance Computer Architecture',
        topics=['cs.AR', 'cs.DC', 'cs.PF', 'cs.OS', 'cs.ET', 'cs.NE', 'cs.LG', 'cs.CR'],
        category='Computer Architecture',
        acceptance_rate=0.22,
        conference_id='HPCA'
    ),
    'ISCA': Conference(
        name='International Symposium on Computer Architecture',
        topics=['cs.AR', 'cs.DC', 'cs.PF', 'cs.ET', 'cs.OS', 'cs.CR', 'cs.LG', 'cs.NE'],
        category='Computer Architecture',
        acceptance_rate=0.17,
        conference_id='ISCA'
    ),
    'MICRO': Conference(
        name='IEEE/ACM International Symposium on Microarchitecture',
        topics=['cs.AR', 'cs.PF', 'cs.DC', 'cs.ET', 'cs.OS', 'cs.CR', 'cs.LG', 'cs.NE'],
        category='Computer Architecture',
        acceptance_rate=0.19,
        conference_id='MICRO'
    ),
    'PPoPP': Conference(
        name='Principles and Practice of Parallel Programming',
        topics=['cs.DC', 'cs.PL', 'cs.AR', 'cs.PF', 'cs.DS', 'cs.OS', 'cs.LG', 'cs.CR'],
        category='Computer Architecture',
        acceptance_rate=0.21,
        conference_id='PPoPP'
    ),
    'SC': Conference(
        name='International Conference for High Performance Computing, Networking, Storage and Analysis',
        topics=['cs.DC', 'cs.AR', 'cs.PF', 'cs.NI', 'cs.OS', 'cs.LG', 'cs.NA', 'cs.CE'],
        category='Computer Architecture',
        acceptance_rate=0.23,
        conference_id='SC'
    ),

    # Computer Networks
    'INFOCOM': Conference(
        name='IEEE International Conference on Computer Communications',
        topics=['cs.NI', 'cs.DC', 'cs.CR', 'cs.PF', 'cs.LG', 'cs.IT', 'cs.SY', 'cs.MA'],
        category='Computer Networks',
        acceptance_rate=0.19,
        conference_id='INFOCOM'
    ),
    'MobiCom': Conference(
        name='ACM International Conference on Mobile Computing and Networking',
        topics=['cs.NI', 'cs.DC', 'cs.HC', 'cs.CR', 'cs.LG', 'eess.SP', 'cs.IT', 'cs.SY'],
        category='Computer Networks',
        acceptance_rate=0.16,
        conference_id='MobiCom'
    ),
    'NSDI': Conference(
        name='Symposium on Networked Systems Design and Implementation',
        topics=['cs.NI', 'cs.DC', 'cs.OS', 'cs.CR', 'cs.DB', 'cs.LG', 'cs.PF', 'cs.SE'],
        category='Computer Networks',
        acceptance_rate=0.15,
        conference_id='NSDI'
    ),
    'SIGCOMM': Conference(
        name='ACM SIGCOMM Conference',
        topics=['cs.NI', 'cs.DC', 'cs.CR', 'cs.PF', 'cs.LG', 'cs.IT', 'cs.OS', 'cs.SY'],
        category='Computer Networks',
        acceptance_rate=0.14,
        conference_id='SIGCOMM'
    ),

    # Network & Information Security
    'ACM CCS': Conference(
        name='ACM Conference on Computer and Communications Security',
        topics=['cs.CR', 'cs.CY', 'cs.AI', 'cs.NI', 'cs.SE', 'cs.LG', 'cs.DB', 'cs.PL'],
        category='Network and Information Security',
        acceptance_rate=0.19,
        conference_id='ACM CCS'
    ),
    'CRYPTO': Conference(
        name='International Cryptology Conference',
        topics=['cs.CR', 'cs.IT', 'math.NT', 'cs.CC', 'quant-ph', 'cs.DS', 'math.CO', 'cs.LO'],
        category='Network and Information Security',
        acceptance_rate=0.21,
        conference_id='CRYPTO'
    ),
    'EUROCRYPT': Conference(
        name='European Cryptology Conference',
        topics=['cs.CR', 'math.NT', 'cs.IT', 'cs.CC', 'quant-ph', 'cs.DS', 'math.CO', 'cs.LO'],
        category='Network and Information Security',
        acceptance_rate=0.22,
        conference_id='EUROCRYPT'
    ),
    'S&P': Conference(
        name='IEEE Symposium on Security and Privacy',
        topics=['cs.CR', 'cs.CY', 'cs.AI', 'cs.SE', 'cs.NI', 'cs.DB', 'cs.LG', 'cs.PL'],
        category='Network and Information Security',
        acceptance_rate=0.12,
        conference_id='S&P'
    ),
    'USENIX Security': Conference(
        name='USENIX Security Symposium',
        topics=['cs.CR', 'cs.SE', 'cs.CY', 'cs.NI', 'cs.OS', 'cs.LG', 'cs.AI', 'cs.PL'],
        category='Network and Information Security',
        acceptance_rate=0.16,
        conference_id='USENIX Security'
    ),

    # Software Engineering / Programming Languages
    'ASE': Conference(
        name='Automated Software Engineering',
        topics=['cs.SE', 'cs.AI', 'cs.LG', 'cs.PL', 'cs.LO', 'cs.FL', 'cs.CY', 'cs.HC'],
        category='Software Engineering',
        acceptance_rate=0.20,
        conference_id='ASE'
    ),
    'FSE': Conference(
        name='Foundations of Software Engineering',
        topics=['cs.SE', 'cs.PL', 'cs.LO', 'cs.AI', 'cs.LG', 'cs.CY', 'cs.HC', 'cs.CR'],
        category='Software Engineering',
        acceptance_rate=0.18,
        conference_id='FSE'
    ),
    'ICSE': Conference(
        name='International Conference on Software Engineering',
        topics=['cs.SE', 'cs.AI', 'cs.HC', 'cs.PL', 'cs.LG', 'cs.CY', 'cs.CR', 'cs.LO'],
        category='Software Engineering',
        acceptance_rate=0.20,
        conference_id='ICSE'
    ),
    'OOPSLA': Conference(
        name='Object-Oriented Programming, Systems, Languages, and Applications',
        topics=['cs.PL', 'cs.SE', 'cs.LO', 'cs.FL', 'cs.DS', 'cs.AI', 'cs.LG', 'cs.CR'],
        category='Software Engineering',
        acceptance_rate=0.22,
        conference_id='OOPSLA'
    ),
    'PLDI': Conference(
        name='Programming Language Design and Implementation',
        topics=['cs.PL', 'cs.SE', 'cs.LO', 'cs.FL', 'cs.AR', 'cs.DS', 'cs.CR', 'cs.AI'],
        category='Software Engineering',
        acceptance_rate=0.18,
        conference_id='PPLDI'
    ),
    'POPL': Conference(
        name='Principles of Programming Languages',
        topics=['cs.PL', 'cs.LO', 'cs.FL', 'math.LO', 'cs.SE', 'math.CT', 'cs.DS', 'cs.AI'],
        category='Software Engineering',
        acceptance_rate=0.19,
        conference_id='POPL'
    ),

    # Databases / Data Mining / Content Retrieval
    'ICDE': Conference(
        name='IEEE International Conference on Data Engineering',
        topics=['cs.DB', 'cs.LG', 'cs.IR', 'cs.AI', 'cs.DC', 'cs.SI', 'stat.ML', 'cs.DS'],
        category='Databases and Data Mining',
        acceptance_rate=0.18,
        conference_id='ICDE'
    ),
    'KDD': Conference(
        name='ACM SIGKDD Conference on Knowledge Discovery and Data Mining',
        topics=['cs.LG', 'cs.DB', 'stat.ML', 'cs.AI', 'cs.SI', 'stat.AP', 'cs.IR', 'cs.CY'],
        category='Databases and Data Mining',
        acceptance_rate=0.15,
        conference_id='KDD'
    ),
    'SIGIR': Conference(
        name='International ACM SIGIR Conference on Research and Development in Information Retrieval',
        topics=['cs.IR', 'cs.LG', 'cs.AI', 'cs.CL', 'cs.DB', 'cs.SI', 'stat.ML', 'cs.HC'],
        category='Databases and Data Mining',
        acceptance_rate=0.20,
        conference_id='SIGIR'
    ),
    'SIGMOD': Conference(
        name='ACM SIGMOD Conference',
        topics=['cs.DB', 'cs.LG', 'cs.DS', 'cs.DC', 'cs.AI', 'cs.IR', 'stat.ML', 'cs.CR'],
        category='Databases and Data Mining',
        acceptance_rate=0.17,
        conference_id='SIGMOD'
    ),
    'VLDB': Conference(
        name='International Conference on Very Large Data Bases',
        topics=['cs.DB', 'cs.LG', 'cs.DS', 'cs.DC', 'cs.AI', 'cs.IR', 'stat.ML', 'cs.CR'],
        category='Databases and Data Mining',
        acceptance_rate=0.18,
        conference_id='VLDB'
    ),

    # Computer Graphics & Multimedia
    'ACM MM': Conference(
        name='ACM International Conference on Multimedia',
        topics=['cs.MM', 'cs.CV', 'cs.LG', 'cs.AI', 'cs.HC', 'cs.IR', 'eess.IV', 'eess.AS'],
        category='Computer Graphics and Multimedia',
        acceptance_rate=0.24,
        conference_id='ACM MM'
    ),
    'SIGGRAPH': Conference(
        name='ACM SIGGRAPH Annual Conference',
        topics=['cs.GR', 'cs.CV', 'cs.LG', 'cs.HC', 'cs.AI', 'cs.RO', 'physics.comp-ph', 'math.NA'],
        category='Computer Graphics and Multimedia',
        acceptance_rate=0.26,
        conference_id='SIGGRAPH'
    ),
    'VR': Conference(
        name='IEEE Virtual Reality',
        topics=['cs.HC', 'cs.GR', 'cs.CV', 'cs.RO', 'cs.AI', 'cs.LG', 'eess.IV', 'cs.MM'],
        category='Computer Graphics and Multimedia',
        acceptance_rate=0.23,
        conference_id='VR'
    ),

    # Artificial Intelligence
    'AAAI': Conference(
        name='AAAI Conference on Artificial Intelligence',
        topics=['cs.AI', 'cs.LG', 'cs.MA', 'cs.RO', 'cs.CV', 'cs.CL', 'cs.GT', 'cs.HC'],
        category='Artificial Intelligence',
        acceptance_rate=0.20,
        conference_id='AAAI'
    ),
    'AAMAS': Conference(
        name='International Conference on Autonomous Agents and Multiagent Systems',
        topics=['cs.MA', 'cs.AI', 'cs.GT', 'cs.LG', 'cs.RO', 'econ.TH', 'cs.CY', 'cs.SY'],
        category='Artificial Intelligence',
        acceptance_rate=0.24,
        conference_id='AAMAS'
    ),
    'IJCAI': Conference(
        name='International Joint Conference on Artificial Intelligence',
        topics=['cs.AI', 'cs.LG', 'cs.MA', 'cs.RO', 'cs.CV', 'cs.CL', 'cs.GT', 'cs.HC'],
        category='Artificial Intelligence',
        acceptance_rate=0.21,
        conference_id='IJCAI'
    ),

    # Computer Vision & Pattern Recognition
    'CVPR': Conference(
        name='IEEE/CVF Computer Vision and Pattern Recognition Conference',
        topics=['cs.CV', 'cs.LG', 'cs.AI', 'cs.RO', 'cs.GR', 'cs.MM', 'eess.IV', 'stat.ML'],
        category='Artificial Intelligence',
        acceptance_rate=0.29,
        conference_id='CVPR'
    ),
    'ECCV': Conference(
        name='European Conference on Computer Vision',
        topics=['cs.CV', 'cs.LG', 'cs.AI', 'cs.RO', 'cs.GR', 'cs.MM', 'stat.ML', 'eess.IV'],
        category='Artificial Intelligence',
        acceptance_rate=0.27,
        conference_id='ECCV'
    ),
    'ICCV': Conference(
        name='IEEE/CVF International Conference on Computer Vision',
        topics=['cs.CV', 'cs.LG', 'cs.AI', 'cs.RO', 'cs.GR', 'cs.MM', 'stat.ML', 'eess.IV'],
        category='Artificial Intelligence',
        acceptance_rate=0.25,
        conference_id='ICCV'
    ),

    # Machine Learning & Data Mining
    'COLT': Conference(
        name='Annual Conference on Learning Theory',
        topics=['cs.LG', 'stat.ML', 'math.ST', 'cs.IT', 'stat.TH', 'math.OC', 'cs.CC', 'math.PR'],
        category='Artificial Intelligence',
        acceptance_rate=0.30,
        conference_id='COLT'
    ),
    'ICML': Conference(
        name='International Conference on Machine Learning',
        topics=['cs.LG', 'stat.ML', 'cs.AI', 'cs.NE', 'stat.ME', 'math.OC', 'cs.IT', 'stat.TH'],
        category='Artificial Intelligence',
        acceptance_rate=0.25,
        conference_id='ICML'
    ),
    'ICLR': Conference(
        name='International Conference on Learning Representations',
        topics=['cs.LG', 'cs.AI', 'stat.ML', 'cs.NE', 'cs.CV', 'cs.CL', 'cs.RO', 'math.OC'],
        category='Artificial Intelligence',
        acceptance_rate=0.27,
        conference_id='ICLR'
    ),
    'NeurIPS': Conference(
        name='Conference on Neural Information Processing Systems',
        topics=['cs.LG', 'cs.AI', 'stat.ML', 'cs.NE', 'q-bio.NC', 'stat.TH', 'math.OC', 'cs.CV'],
        category='Artificial Intelligence',
        acceptance_rate=0.20,
        conference_id='NeurIPS'
    ),

    # Natural Language Processing & Computational Linguistics
    'ACL': Conference(
        name='Annual Meeting of the Association for Computational Linguistics',
        topics=['cs.CL', 'cs.AI', 'cs.LG', 'cs.IR', 'cs.HC', 'stat.ML', 'cs.CY', 'cs.SI'],
        category='Artificial Intelligence',
        acceptance_rate=0.23,
        conference_id='ACL'
    ),
    'EMNLP': Conference(
        name='Conference on Empirical Methods in Natural Language Processing',
        topics=['cs.CL', 'cs.LG', 'cs.AI', 'cs.IR', 'stat.ML', 'cs.SI', 'cs.HC', 'cs.CY'],
        category='Artificial Intelligence',
        acceptance_rate=0.26,
        conference_id='EMNLP'
    ),

    # Human-Computer Interaction & Ubiquitous Computing
    'CHI': Conference(
        name='ACM Conference on Human Factors in Computing Systems',
        topics=['cs.HC', 'cs.AI', 'cs.CY', 'cs.LG', 'cs.SI', 'cs.CV', 'cs.RO', 'stat.AP'],
        category='Human-Computer Interaction',
        acceptance_rate=0.24,
        conference_id='CHI'
    ),
    'UbiComp': Conference(
        name='ACM International Joint Conference on Pervasive and Ubiquitous Computing',
        topics=['cs.HC', 'cs.NI', 'cs.AI', 'cs.LG', 'cs.CY', 'eess.SP', 'cs.SI', 'stat.AP'],
        category='Human-Computer Interaction',
        acceptance_rate=0.21,
        conference_id='UbiComp'
    ),

    # Robotics
    'ICRA': Conference(
        name='IEEE International Conference on Robotics and Automation',
        topics=['cs.RO', 'cs.CV', 'cs.AI', 'cs.LG', 'cs.SY', 'cs.HC', 'eess.SY', 'stat.ML'],
        category='Artificial Intelligence',
        acceptance_rate=0.43,
        conference_id='ICRA'
    ),
    'IROS': Conference(
        name='IEEE/RSJ International Conference on Intelligent Robots and Systems',
        topics=['cs.RO', 'cs.CV', 'cs.AI', 'cs.LG', 'cs.SY', 'cs.HC', 'eess.SY', 'stat.ML'],
        category='Artificial Intelligence',
        acceptance_rate=0.45,
        conference_id='IROS'
    ),
    'RSS': Conference(
        name='Robotics: Science and Systems',
        topics=['cs.RO', 'cs.CV', 'cs.LG', 'cs.AI', 'cs.SY', 'cs.HC', 'stat.ML', 'eess.SY'],
        category='Artificial Intelligence',
        acceptance_rate=0.27,
        conference_id='RSS'
    ),

    # Theoretical Computer Science
    'FOCS': Conference(
        name='IEEE Symposium on Foundations of Computer Science',
        topics=['cs.CC', 'cs.DS', 'math.CO', 'cs.DM', 'quant-ph', 'cs.GT', 'math.OC', 'cs.IT'],
        category='Theoretical Computer Science',
        acceptance_rate=0.30,
        conference_id='FOCS'
    ),
    'LICS': Conference(
        name='Logic in Computer Science',
        topics=['cs.LO', 'math.LO', 'cs.PL', 'cs.FL', 'math.CT', 'cs.AI', 'cs.CC', 'cs.CR'],
        category='Theoretical Computer Science',
        acceptance_rate=0.32,
        conference_id='LICS'
    ),
    'SODA': Conference(
        name='ACM-SIAM Symposium on Discrete Algorithms',
        topics=['cs.DS', 'math.CO', 'cs.DM', 'cs.CC', 'math.OC', 'cs.CG', 'cs.IT', 'math.PR'],
        category='Theoretical Computer Science',
        acceptance_rate=0.29,
        conference_id='SODA'
    ),
    'STOC': Conference(
        name='ACM Symposium on Theory of Computing',
        topics=['cs.CC', 'cs.DS', 'math.CO', 'cs.DM', 'quant-ph', 'cs.GT', 'cs.IT', 'math.OC'],
        category='Theoretical Computer Science',
        acceptance_rate=0.31,
        conference_id='STOC'
    ),

    # Interdisciplinary / Web & Information Retrieval
    'CSCW': Conference(
        name='ACM Conference on Computer Supported Cooperative Work and Social Computing',
        topics=['cs.HC', 'cs.SI', 'cs.CY', 'cs.AI', 'cs.LG', 'stat.AP', 'cs.IR', 'econ.GN'],
        category='Interdisciplinary',
        acceptance_rate=0.25,
        conference_id='CSCW'
    ),
    'RTSS': Conference(
        name='IEEE Real-Time Systems Symposium',
        topics=['cs.OS', 'cs.DC', 'cs.AR', 'cs.PF', 'cs.SY', 'eess.SY', 'cs.NI', 'cs.SE'],
        category='Interdisciplinary',
        acceptance_rate=0.26,
        conference_id='RTSS'
    ),
    'WWW': Conference(
        name='The Web Conference',
        topics=['cs.SI', 'cs.IR', 'cs.LG', 'cs.CY', 'cs.AI', 'cs.HC', 'cs.DB', 'stat.ML'],
        category='Interdisciplinary',
        acceptance_rate=0.17,
        conference_id='WWW'
    ),
}

CONFERENCES_BY_CATEGORY = {}
for conference in CONFERENCES.values():
    if conference.category not in CONFERENCES_BY_CATEGORY:
        CONFERENCES_BY_CATEGORY[conference.category] = []
    CONFERENCES_BY_CATEGORY[conference.category].append(conference)


ARXIV_CATEGORY_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography",
    "cs.CY": "Cybersecurity",
    "cs.DB": "Databases and Data Mining",
    "cs.DC": "Distributed Computing",
    "cs.DL": "Deep Learning",
    "cs.DM": "Data Mining",
    "cs.DS": "Data Structures",
    "cs.ET": "Embedded Systems",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.NI": "Network and Information Systems",
    "cs.OH": "Other",
    "cs.OS": "Operating Systems",
    "cs.PF": "Parallel and Distributed Computing",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SE": "Software Engineering",
    "cs.SY": "Systems and Networking",
    
}

ALL_CS_TOPICS = list(ARXIV_CATEGORY_NAMES.keys())



# Mapping from research direction topics to conference categories
# This maps the topics in ResearchDirection to the conference categories
TOPIC_TO_CATEGORY_MAPPING = {
    # Artificial Intelligence category
    'artificial_intelligence': 'Artificial Intelligence',
    'planning_and_scheduling': 'Artificial Intelligence',
    'neural_symbolic_ai': 'Artificial Intelligence',
    'natural_language_processing': 'Artificial Intelligence',
    'dialogue_systems': 'Artificial Intelligence',
    'image_recognition': 'Artificial Intelligence',
    '3d_vision': 'Artificial Intelligence',
    'video_understanding': 'Artificial Intelligence',
    'deep_learning': 'Artificial Intelligence',
    'reinforcement_learning': 'Artificial Intelligence',
    'generative_models': 'Artificial Intelligence',
    'meta-learning': 'Artificial Intelligence',
    'interpretable_ml': 'Artificial Intelligence',
    'robotics': 'Artificial Intelligence',

    # Computer Architecture category
    'computer_architecture': 'Computer Architecture',
    'parallel_computing': 'Computer Architecture',
    'distributed_systems': 'Computer Architecture',

    # Computer Graphics and Multimedia category
    'computer_graphics': 'Computer Graphics and Multimedia',
    'multimedia_systems': 'Computer Graphics and Multimedia',
    'audio_computing': 'Computer Graphics and Multimedia',

    # Computer Networks category
    'computer_networks': 'Computer Networks',

    # Databases and Data Mining category
    'database_systems': 'Databases and Data Mining',
    'information_retrieval': 'Databases and Data Mining',
    'digital_libraries': 'Databases and Data Mining',

    # Human-Computer Interaction category
    'human_computer_interaction': 'Human-Computer Interaction',

    # Interdisciplinary category
    'interdisciplinary_cs': 'Interdisciplinary',
    'ai_ethics': 'Interdisciplinary',
    'computational_science': 'Interdisciplinary',
    'social_networks': 'Interdisciplinary',

    # Network and Information Security category
    'cryptography': 'Network and Information Security',
    'cybersecurity': 'Network and Information Security',

    # Software Engineering category
    'software_engineering': 'Software Engineering',
    'programming_languages': 'Software Engineering',
    'operating_systems': 'Software Engineering',
    'formal_methods': 'Software Engineering',

    # Theoretical Computer Science category
    'complexity_theory': 'Theoretical Computer Science',
    'geometric_algorithms': 'Theoretical Computer Science',
    'computational_topology': 'Theoretical Computer Science',
    'discrete_mathematics': 'Theoretical Computer Science',
    'algorithms': 'Theoretical Computer Science',
    'data_structures': 'Theoretical Computer Science',
    'logic_in_cs': 'Theoretical Computer Science',
    'algorithmic_game_theory': 'Theoretical Computer Science',
    'information_theory': 'Theoretical Computer Science',
    'numerical_analysis': 'Theoretical Computer Science',
    'symbolic_computation': 'Theoretical Computer Science',

    # Special topics that don't fit clearly - assign to most relevant
    'quantum_computing': 'Theoretical Computer Science',
    'evolutionary_computation': 'Artificial Intelligence',
    'multiagent_systems': 'Artificial Intelligence',
    'mathematical_software': 'Theoretical Computer Science',
    'performance_analysis': 'Computer Architecture',
    'control_systems': 'Interdisciplinary',
    'computing_education': 'Interdisciplinary',
}

# Mapping from research direction topics to arXiv category codes
# Derived from the arXiv category annotations in AVAILABLE_DIRECTIONS (research_direction.py).
# For directions whose primary arXiv code is not in ARXIV_CATEGORY_NAMES,
# we map to the closest available codes so that translated topic names can be produced.
DIRECTION_TO_ARXIV = {
    # cs.AI - Artificial Intelligence
    'artificial_intelligence': ['cs.AI'],
    'planning_and_scheduling': ['cs.AI'],
    'neural_symbolic_ai': ['cs.AI', 'cs.LG'],
    # cs.AR - Hardware Architecture
    'computer_architecture': ['cs.AR'],
    # cs.CC - Computational Complexity
    'complexity_theory': ['cs.CC'],
    # cs.CE - Computational Engineering
    'computational_science': ['cs.CE'],
    # cs.CG - Computational Geometry
    'geometric_algorithms': ['cs.CG'],
    'computational_topology': ['cs.CG'],
    'computer_graphics': ['cs.CG'],
    # cs.CL - Computation and Language
    'natural_language_processing': ['cs.CL', 'cs.AI'],
    'dialogue_systems': ['cs.CL', 'cs.AI'],
    'audio_computing': ['cs.CL'],
    # cs.CR - Cryptography
    'cryptography': ['cs.CR'],
    'cybersecurity': ['cs.CR', 'cs.CY'],
    # cs.CV - Computer Vision (not in ARXIV_CATEGORY_NAMES, mapped to closest)
    'image_recognition': ['cs.AI', 'cs.LG'],
    '3d_vision': ['cs.AI', 'cs.LG'],
    'video_understanding': ['cs.AI', 'cs.LG'],
    # cs.CY - Cybersecurity
    'ai_ethics': ['cs.CY', 'cs.AI'],
    # cs.DB - Databases and Data Mining
    'database_systems': ['cs.DB'],
    'information_retrieval': ['cs.DB', 'cs.AI'],
    'digital_libraries': ['cs.DB'],
    # cs.DC - Distributed Computing
    'distributed_systems': ['cs.DC'],
    'parallel_computing': ['cs.DC', 'cs.PF'],
    # cs.DM - Data Mining
    'discrete_mathematics': ['cs.DM', 'cs.DS'],
    # cs.DS - Data Structures
    'algorithms': ['cs.DS', 'cs.CC'],
    'data_structures': ['cs.DS'],
    # cs.ET - Embedded Systems / Emerging Technologies
    'quantum_computing': ['cs.ET', 'cs.CC'],
    # cs.FL - Formal Languages and Automata Theory
    'formal_methods': ['cs.FL', 'cs.LO'],
    # cs.LG - Machine Learning
    'deep_learning': ['cs.LG', 'cs.AI'],
    'reinforcement_learning': ['cs.LG', 'cs.AI'],
    'generative_models': ['cs.LG', 'cs.AI'],
    'meta-learning': ['cs.LG', 'cs.AI'],
    'interpretable_ml': ['cs.LG', 'cs.AI'],
    'evolutionary_computation': ['cs.AI', 'cs.LG'],
    # cs.LO - Logic in Computer Science
    'logic_in_cs': ['cs.LO'],
    'symbolic_computation': ['cs.LO'],
    # cs.MA - Multiagent Systems (mapped to closest)
    'multiagent_systems': ['cs.AI'],
    'algorithmic_game_theory': ['cs.AI'],
    # cs.NI - Network and Information Systems
    'computer_networks': ['cs.NI'],
    'social_networks': ['cs.NI'],
    # cs.OH - Other
    'interdisciplinary_cs': ['cs.OH'],
    'computing_education': ['cs.OH'],
    # cs.OS - Operating Systems
    'operating_systems': ['cs.OS'],
    # cs.PF - Parallel and Distributed Computing / Performance
    'performance_analysis': ['cs.PF'],
    # cs.PL - Programming Languages
    'programming_languages': ['cs.PL'],
    # cs.RO - Robotics
    'robotics': ['cs.RO', 'cs.AI'],
    # cs.SE - Software Engineering
    'software_engineering': ['cs.SE'],
    # cs.SY - Systems and Networking / Control
    'control_systems': ['cs.SY', 'cs.RO'],
    # cs.CE - Computational Engineering (math tools)
    'mathematical_software': ['cs.CE'],
    'numerical_analysis': ['cs.CE'],
    # cs.HC / cs.MM - mapped to closest available
    'human_computer_interaction': ['cs.AI'],
    'multimedia_systems': ['cs.AI'],
    # cs.IT - Information Theory
    'information_theory': ['cs.CC'],
}


def select_conferences_for_simulation(
        agents: List,
        num_conferences: int = 3,
        min_submissions_per_conf: int = 10
) -> List[Conference]:
    """Select conferences ensuring each gets sufficient submissions

    Args:
        agents: List of agents with expertise
        num_conferences: Target number of conferences
        min_submissions_per_conf: Minimum expected submissions per conference

    Returns:
        List of selected Conference objects
    """
    from collections import Counter

    category_counts = Counter()
    for agent in agents:
        if hasattr(agent, 'expertise'):
            for expertise_item in agent.expertise:
                topic = expertise_item.topic if hasattr(expertise_item, 'topic') else str(
                    expertise_item) if not isinstance(expertise_item, str) else expertise_item
                category_counts[TOPIC_TO_CATEGORY_MAPPING.get(topic, 'Interdisciplinary')] += 1

    num_agents = sum(category_counts.values())
    max_conferences = max(1, num_agents // min_submissions_per_conf)
    num_conferences = min(num_conferences, max_conferences)

    selected_conferences = []
    for category, count in category_counts.items():
        num_confs = max(1, round((count / num_agents) * num_conferences))
        if count < 3:
            print(f"  ❌ Category {category}: {count} agents -> NO conference due to low number of agents")
            continue

        print(f"  ✅ Category {category}: {count} agents -> {num_confs} conference(s)")
        selected_conferences.extend(CONFERENCES_BY_CATEGORY[category][:num_confs])

    return selected_conferences[:num_conferences]


def create_default_conference_system() -> ConferenceSystem:
    """Create a conference system with predefined conferences

    Returns:
        ConferenceSystem with standard ML/AI conferences
    """
    conferences = list(CONFERENCES.values())
    return ConferenceSystem(conferences)
