"""
Base Agent System for Multi-Agent Scientific Ecosystem

IMPORTANT: This creates NEW classes that extend existing AI-Scientist infrastructure
WITHOUT modifying any original files.
"""

import logging
import time
from typing import List, Dict, Optional, Set, Union, Tuple
from pydantic import BaseModel
import networkx as nx
from langchain_core.documents import Document
from enum import Enum
from utopia.constants import REVIEW_CRITERIA, IMPORTANT_NOTES, STANDARDIZED_REVIEW_POLICY
from utopia.agents.conference import Conference
from utopia.agents.research_direction import ResearchDirection
from utopia.config import SIMULATION_CONFIG
from utopia.data.paper_tracker import PaperTracker
from utopia.models.models import BaseLLM
from utopia.metrics.tracker import top_percentile
from utopia.constants import COI_NOTE
logger = logging.getLogger(__name__)


class RoundIntention(BaseModel):
    intention: str

class PaperReview(BaseModel):
    overall_score: float
    justification: str

class CitationResponse(BaseModel):
    citations: List[int]

class SelfCitationResponse(BaseModel):
    self_citations: List[int]

class CollaborationDecision(BaseModel):
    collaborate: bool
    collaborator_id: Optional[str] = None
    reason: str

class MultiAgentEcosystem:
    """
    Multi-agent ecosystem that manages scientific agents without inheriting from AgentManager
    to avoid complex dependency issues during Phase 1 testing
    """

    def __init__(self, agent_configs: Dict):
        # Initialize without inheriting from AgentManager for Phase 1 simplicity
        self.config = agent_configs
        self.citation_graph = nx.DiGraph()  # NetworkX graph for citations
        self.simulation_metrics = self._create_metrics_tracker()
        self.agent_population = {}  # Store all simulation agents

    def remove_agent(self, agent_id: str):
        """Remove agent from ecosystem population"""
        assert self.agent_population[agent_id].is_active, f"Agent {agent_id} is already not active"
        self.agent_population[agent_id].is_active = False

    def _create_metrics_tracker(self):
        """Initialize metrics tracking system"""
        return {
            'yearly_metrics': {},
            'agent_interactions': [],
            'citation_evolution': [],
            'review_patterns': []
        }

    def add_agent(self, agent):
        """Add agent to ecosystem population"""
        self.agent_population[agent.id] = agent

    def get_agent_by_id(self, agent_id: str):
        """Get agent by ID"""
        return self.agent_population[agent_id]

    def get_available_reviewers(self, exclude_agent_ids: List = None):
        """Get agents that can serve as reviewers"""
        exclude_agent_ids = exclude_agent_ids or []
        return [agent for agent_id, agent in self.agent_population.items()
                if hasattr(agent, 'can_review') and agent.can_review
                and agent.id not in exclude_agent_ids]

    def get_available_authors(self):
        """Get agents that can author papers"""
        return [agent for agent_id, agent in self.agent_population.items()
                if hasattr(agent, 'can_author') and agent.can_author and agent.is_active]

    def to_dict(self) -> Dict:
        """Serialize ecosystem to dictionary (without LLM objects)

        Returns:
            Dictionary with serialized state
        """
        from utopia.agents.researcher_agents import UniversityResearcher, IndustryResearcher
        from utopia.agents.funding_agents import FundingAgency

        agents_data = []
        for agent_id, agent in self.agent_population.items():
            if agent.newest_direction is not None:
                newest_direction = {
                    'direction': agent.newest_direction['direction'].topic,
                    'detailed_focus': agent.newest_direction.get('detailed_focus'),
                    'reason': agent.newest_direction.get('reason')
                }
            else:
                newest_direction = None

            agent_dict = {
                'id': agent.id,
                'type': agent.get_type(),
                'resources': getattr(agent, 'resources', None),
                'reputation': agent.reputation,
                'memory_bank': agent.memory_bank,
                'review_experiences': agent.review_experiences,
                'is_active': agent.is_active,
                'can_author': agent.can_author,
                'can_review': agent.can_review,
                'project_start_year': agent.project_start_year,
                'project_end_year': agent.project_end_year,
                'newest_direction': newest_direction,
            }
            if getattr(agent, 'production_cost_entries', None):
                agent_dict['production_cost_entries'] = agent.production_cost_entries

            if isinstance(agent, UniversityResearcher):
                agent_dict['university_name'] = agent.university_name
                agent_dict['expertise'] = [exp.topic for exp in agent.expertise]
                agent_dict['funding_success_history'] = agent.funding_success_history
                agent_dict['conflict_of_interest'] = list(agent.conflict_of_interest)
                agent_dict['generate_research_proposal'] = agent.generate_research_proposal
                agent_dict['exploration_strategy'] = agent.exploration_strategy

            elif isinstance(agent, IndustryResearcher):
                agent_dict['company_name'] = agent.company_name
                agent_dict['expertise'] = [exp.topic for exp in agent.expertise]
                agent_dict['funding_mode'] = agent.funding_mode
                agent_dict['budget'] = agent.budget
                agent_dict['conflict_of_interest'] = list(agent.conflict_of_interest)
                agent_dict['funding_success_history'] = agent.funding_success_history
                agent_dict['exploration_strategy'] = agent.exploration_strategy

            elif isinstance(agent, FundingAgency):
                agent_dict['agency_name'] = agent.id
                agent_dict['default_funding_rate'] = getattr(agent, 'default_funding_rate', None)
                # Serialize funding programs
                programs_data = {}
                for prog_id, program in agent.funding_programs.items():
                    programs_data[prog_id] = {
                        'program_id': program.program_id,
                        'name': program.name,
                        'research_directions': [rd.to_dict() if hasattr(rd, 'to_dict') else rd for rd in
                                                program.research_directions],
                        'funding_mode': program.funding_mode,
                        'budget': getattr(program, 'budget', None),
                        'budget_per_winner': getattr(program, 'budget_per_winner', None),
                        'funding_rate': getattr(program, 'funding_rate', None)
                    }
                agent_dict['funding_programs'] = programs_data
            else:
                # Unknown agent type - skip or raise error
                continue

            agents_data.append(agent_dict)

        return {
            'config': self.config,
            'agents': agents_data,
            'simulation_metrics': self.simulation_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict, llm: BaseLLM) -> 'MultiAgentEcosystem':
        """Reconstruct ecosystem from dictionary

        Args:
            data: Dictionary with serialized state
            llm: LLM instance to use for all agents

        Returns:
            Reconstructed MultiAgentEcosystem instance
        """
        from utopia.agents.researcher_agents import UniversityResearcher, IndustryResearcher
        from utopia.agents.funding_agents import FundingAgency, FundingProgram
        from utopia.agents.research_direction import ResearchDirection

        # Create new ecosystem
        ecosystem = cls(
            agent_configs=data.get('config', {}),
        )

        ecosystem.simulation_metrics = data.get('simulation_metrics', ecosystem._create_metrics_tracker())
        # Reconstruct agents
        all_agent_data = data.get('agents', [])

        from utopia.agents.research_direction import DIRECTIONS_DICT
        for agent_data in all_agent_data:
            agent_type = agent_data['type']

            newest_direction_data = agent_data['newest_direction']
            if newest_direction_data is None:
                assert agent_type == "funding_agency"
                newest_direction = None
            else:
                direction_topic = newest_direction_data['direction']
                assert direction_topic in DIRECTIONS_DICT, f"Direction {direction_topic} not found in DIRECTIONS_DICT"
                newest_direction = {
                    'direction': DIRECTIONS_DICT[direction_topic],
                    'detailed_focus': newest_direction_data.get('detailed_focus'),
                    'reason': newest_direction_data.get('reason')
                }

            if agent_type == 'university':
                # Reconstruct expertise
                expertise = []
                for direction in agent_data.get('expertise', []):
                    if isinstance(direction, dict):
                        expertise.append(ResearchDirection.from_dict(direction))
                    else:
                        assert direction in DIRECTIONS_DICT, f"Direction {direction} not found in DIRECTIONS_DICT"
                        expertise.append(DIRECTIONS_DICT[direction])

                agent = UniversityResearcher(
                    researcher_name=agent_data['id'],
                    university_name=agent_data['university_name'],
                    funding_level=agent_data['resources'],
                    expertise=expertise,
                    llm=llm,
                    generate_research_proposal=agent_data.get('generate_research_proposal', True),
                    exploration_strategy=agent_data.get('exploration_strategy', 'balanced')
                )
                agent.reputation = agent_data['reputation']
                agent.memory_bank = agent_data.get('memory_bank', [])
                agent.review_experiences = agent_data.get('review_experiences', [])
                agent.is_active = agent_data.get('is_active', True)
                agent.project_start_year = agent_data.get('project_start_year', 0)
                agent.project_end_year = agent_data.get('project_end_year', 0)
                agent.newest_direction = newest_direction
                agent.funding_success_history = agent_data.get('funding_success_history', {})
                agent.conflict_of_interest = set(agent_data.get('conflict_of_interest', []))

            elif agent_type == 'industry':
                # Reconstruct expertise
                expertise = []
                for exp_data in agent_data.get('expertise', []):
                    if isinstance(exp_data, dict):
                        expertise.append(ResearchDirection.from_dict(exp_data))
                    else:
                        assert exp_data in DIRECTIONS_DICT, f"Direction {exp_data} not found in DIRECTIONS_DICT"
                        expertise.append(DIRECTIONS_DICT[exp_data])

                agent = IndustryResearcher(
                    researcher_name=agent_data['id'],
                    company_name=agent_data['company_name'],
                    funding_level=agent_data['resources'],
                    expertise=expertise,
                    llm=llm,
                    funding_mode=agent_data.get('funding_mode', 'performance'),
                    exploration_strategy=agent_data.get('exploration_strategy', 'balanced')
                )
                agent.reputation = agent_data['reputation']
                agent.memory_bank = agent_data.get('memory_bank', [])
                agent.review_experiences = agent_data.get('review_experiences', [])
                agent.is_active = agent_data.get('is_active', True)
                agent.project_start_year = agent_data.get('project_start_year', 0)
                agent.project_end_year = agent_data.get('project_end_year', 0)
                agent.newest_direction = newest_direction
                agent.budget = agent_data.get('budget', agent.budget)
                agent.research_priorities = set(agent_data.get('research_priorities', []))
                agent.conflict_of_interest = set(agent_data.get('conflict_of_interest', []))
                agent.funding_success_history = agent_data.get('funding_success_history', {})

            elif agent_type == 'funding_agency':
                agent = FundingAgency(
                    agency_name=agent_data['agency_name'],
                    funding_level=agent_data['resources'],
                    llm=llm,
                    default_funding_rate=agent_data.get('default_funding_rate')
                )
                agent.reputation = agent_data['reputation']
                agent.memory_bank = agent_data.get('memory_bank', [])
                agent.review_experiences = agent_data.get('review_experiences', [])
                agent.is_active = agent_data.get('is_active', True)
                agent.project_start_year = agent_data.get('project_start_year', 0)
                agent.project_end_year = agent_data.get('project_end_year', 0)
                agent.newest_direction = newest_direction

                # Reconstruct funding programs
                for prog_id, prog_data in agent_data.get('funding_programs', {}).items():
                    research_directions = []
                    for rd_data in prog_data.get('research_directions', []):
                        if isinstance(rd_data, dict):
                            research_directions.append(ResearchDirection.from_dict(rd_data))
                        else:
                            research_directions.append(rd_data)

                    program = FundingProgram(
                        program_id=prog_data['program_id'],
                        name=prog_data['name'],
                        research_directions=research_directions,
                        funding_mode=prog_data['funding_mode'],
                        budget=prog_data.get('budget'),
                        budget_per_winner=prog_data.get('budget_per_winner'),
                        funding_rate=prog_data.get('funding_rate')
                    )
                    agent.funding_programs[prog_id] = program
            else:
                continue

            agent.production_cost_entries = agent_data.get('production_cost_entries', {})
            ecosystem.add_agent(agent)

        return ecosystem


class SimulationAgent:
    """Base class for simulation agents"""

    def __init__(self, reputation: int,
                 agent_id: str = None, llm: BaseLLM = None, max_retries: int = 10):
        self.id = agent_id or f"agent_{int(time.time() * 1000)}"
        self.reputation = reputation  # 1-10 scale
        self.memory_bank = []  # Store experiences and thoughts from interactions
        self.review_experiences = []  # Track experiences as both author and reviewer
        self.llm = llm
        self.is_active = True
        self.project_start_year = 0  # The year when the research project started
        self.project_end_year = 0  # The year when the research paper will be ready for submission
        self.newest_direction = None
        self.max_retries = max_retries

    def update_resources(self, delta: float, *, category=None, year=None, project=None,
                         paper=None, program=None, reason=None):
        """Native clipping/deactivation, with optional classified audit events.

        Unclassified changes fail before mutation when a ledger is attached.
        Replaying an identical event returns False and does not mutate resources.
        """
        assert isinstance(delta, (int, float)), "changes in funding level must be a valid number"
        ledger = getattr(self, 'resource_ledger', None)
        if ledger is not None:
            event = ledger.prepare(self, delta, category=category, year=year, project=project,
                                   paper=paper, program=program, reason=reason)
            if event is None:
                return False
        original_resources = self.resources
        if self.resources + delta < 0:
            logger.warning(f"[Resource] {self.id} runs out of funding - {original_resources} -> {self.resources} is negative.")
            self.resources = 0
            self.is_active = False
        else:
            self.resources += delta
            logger.debug(f"[Resource] {self.id} - {original_resources} -> {self.resources}")
        if ledger is not None:
            ledger.record(self, event)
            return True

    def add_conflict_of_interest(self, other_researcher_names: Union[Set[str], List[str], str]):
        """Add conflict of interest with other researchers"""
        if isinstance(other_researcher_names, str):
            other_researcher_names = {other_researcher_names}
        elif isinstance(other_researcher_names, list):
            other_researcher_names = set(other_researcher_names)
        self.conflict_of_interest.update(other_researcher_names)

    def get_memory_context(self) -> str:
        """Get memory context for this agent"""
        result = ""
        for memory in self.memory_bank:
            result += f"""- {memory['type']} (Year: {memory['year']}). Thought: {memory['thought']}
"""
        return result

    def get_round_intention_prompt(self, direction: Dict, year: int, paper_tracker: 'PaperTracker') -> Tuple[str, Dict]:
        """Build a prompt asking the agent to describe their research intention for this round.

        Args:
            direction: Direction dict with 'direction' (ResearchDirection) and 'detailed_focus' (str or None)
            year: Current simulation year
            paper_tracker: PaperTracker instance for retrieving past papers

        Returns:
            Tuple of (prompt string, response_format dict) for generating round intention
        """
        prompt = self._get_personality_prompt(include_exploration_strategy=True, include_funding_situation=True)

        prompt += f"""
## Your Research Direction
Topic: {direction['direction'].topic}
Keywords: {', '.join(direction['direction'].keywords)}
"""

        if direction.get('detailed_focus'):
            prompt += f"""Detailed Focus: {direction['detailed_focus']}
"""

        # Show past papers (up to 5, sorted by recency)
        past_papers = paper_tracker.get_papers_by_author(self.id)
        if past_papers:
            past_papers.sort(key=lambda x: x.year, reverse=True)
            prompt += """
## Your Past Papers
"""
            for i, paper in enumerate(past_papers[:5], 1):
                latest_score = paper.review_history[-1]['score'] if paper.review_history else 0.0
                prompt += f"{i}. \"{paper.title}\" (Year {paper.year}, {paper.conference}, {paper.status}, score {latest_score:.2f})\n"

        # Show memory context
        memory_context = self.get_memory_context()
        if memory_context:
            prompt += f"""
## Your Recent Experiences
{memory_context}
"""

        prompt += f"""
## Task
Based on your research direction, expertise, past work, and experiences, describe in 4-5 sentences what specific research you intend to pursue this round. Be concrete about the problem, method, or contribution you want to explore.

{IMPORTANT_NOTES}
"""

        prompt += """
## Response Format
Respond in JSON format. An example response is below (DO NOT directly copy the example response):
```json
{
    "intention": "YOUR 5 SENTENCE RESEARCH INTENTION."
}
```
"""

        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'round_intention',
                'strict': True,
                'schema': RoundIntention.model_json_schema()
            },
        }
        prompt += self._production_resource_statement(year)
        return prompt, response_format
    
    def get_review_prompt(self, paper: Dict, author_info: Dict = None,
                          review_policy: str = 'persona') -> Tuple[str, Dict]:
        """Build review prompt for a paper (without calling LLM)

        Args:
            paper: Paper dictionary with 'title', 'content', 'topics', etc.
            author_info: Optional dict with author identity for non-blind review.
                Keys: 'author_name', 'institution', 'network_relationship'.
            review_policy: 'persona' (default, legacy: reviewer memory shown) or
                'standardized' (memory dropped; the frozen STANDARDIZED_REVIEW_POLICY
                block is inserted before '## Review Instructions'). The 'persona'
                prompt is byte-identical to the pre-flag template.

        Returns:
            Prompt string for review generation, response format
        """
        assert review_policy in ('persona', 'standardized'), review_policy
        expertise = [e.topic if hasattr(e, 'topic') else str(e) for e in getattr(self, 'expertise', [])]
        memory_context = self.get_memory_context() if review_policy == 'persona' else ""

        if memory_context:
            memory_context = f"""## Your Recent Experiences
{memory_context}
"""
        else:
            memory_context = ""

        # Non-blind author information (preferential_attachment experiment)
        author_context = ""
        if author_info:
            author_context = f"""## Author Information (Non-Blind Review)
- Author: {author_info['author_name']}
- Institution: {author_info['institution']}
- Your relationship: {author_info['network_relationship']}

"""

        prompt = f"""You are a peer reviewer for an academic conference. Review the following paper based on your expertise and past experiences (e.g. your own past submissions, received reviews, your funding situations, your status level).

Your Expertise: {', '.join(expertise)}

## Paper

Title: {paper['title']}

Abstract:
{paper['abstract']}

Paper Topics: {', '.join(paper['topics'])}

{author_context}{memory_context}

## Review Instructions
Provide a comprehensive review with:
1. Overall Score (1-5 scale):
{REVIEW_CRITERIA}

2. A succinct, 1-sentence justification for the score.
"""

        prompt += """
## Sample Response
```json
{
    "justification": "The paper has critical flaws and needs major revisions."
    "overall_score": 2,
}
```
"""
        if review_policy == 'standardized':
            anchor = "## Review Instructions"
            assert prompt.count(anchor) == 1, "review-prompt anchor not unique"
            prompt = prompt.replace(
                anchor, f"## Your Review Approach\n{STANDARDIZED_REVIEW_POLICY}\n\n{anchor}")
        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'paper_submission',
                'strict': True,
                'schema': PaperReview.model_json_schema()
            },
        }

        return prompt, response_format

    def review_paper(self, paper: Dict, author_info: Dict = None) -> Dict:
        """Generate review for a paper using LLM

        Args:
            paper: Paper dictionary with 'title', 'content', 'topics', etc.
            author_info: Optional dict with author identity for non-blind review.

        Returns:
            Review dictionary with 'overall_score', 'strengths', 'weaknesses', 'summary'
        """
        prompt = self.get_review_prompt(paper, author_info=author_info)

        review_json, message_history = self.llm.generate(prompt=prompt)
        assert all(field in review_json for field in ['overall_score', "justification"])
        review_json['overall_score'] = int(review_json['overall_score'])
        review_json = {
            'overall_score': review_json['overall_score'],
            'justification': review_json['justification'],
            'reviewer_id': self.id,
            'conference_id': paper['conference'],
            'year': paper['year'],
            'arxiv_id': paper['id'],
            'review_time': time.time(),
        }

        return review_json

    def process_received_reviews(self, reviews: List[Dict], paper_id: str, year: int, average_scores_of_venue: dict):
        """Process reviews received for submitted paper and update memory

        Args:
            reviews: List of review dictionaries
            paper_id: ID of the paper that was reviewed
            average_scores_of_venue: Average score of all papers in the conference
        """
        if not reviews:
            return

        # Calculate average score
        avg_score = sum(r.get('overall_score', 0) for r in reviews) / len(reviews)

        percentile = top_percentile(avg_score, list(average_scores_of_venue.values()))

        status_string = f"(Top {percentile:.1f}% of all papers"  # , average score: {avg_score:.1f} on a scale of 1-5)"

        # Determine emotional response based on average score
        if percentile <= 10:  # percentile >= 4.0:

            thought = f"My paper received excellent reviews {status_string}! This validates my research direction and boosts my confidence."
            memory_type = 'excellent_reviews_received'
        elif percentile <= 30:  # percentile >= 3.0:
            thought = f"My paper received moderately good reviews {status_string}. There's room for improvement, but overall positive."
            memory_type = 'good_reviews_received'
        elif percentile <= 50:  # percentile >= 2.0:
            thought = f"My paper received average reviews {status_string}. I need to work harder to improve my research quality."
            memory_type = 'average_reviews_received'
        elif percentile <= 70:
            thought = f"My paper received poor reviews {status_string}. This is disappointing."
            memory_type = 'poor_reviews_received'

        else:
            thought = f"My paper received VERY harsh reviews {status_string}. This is too UNFAIR, harsh, and frustrating."
            memory_type = 'harsh_reviews_received'

        # Record overall experience
        self.memory_bank.append({
            'type': memory_type,
            'thought': thought,
            'timestamp': time.time(),
            'avg_score': avg_score,
            'num_reviews': len(reviews),
            'paper_id': paper_id,
            'year': year,
        })

    def get_resubmission_prompt(self, rejected_papers: Dict[str, Dict], conferences: List[Conference],
                                resubmission_cost: float = None) -> Tuple[str, Dict]:
        """Build prompt for resubmission decision WITHOUT calling LLM.

        Returns:
            Tuple of (prompt_str, response_format)
        """
        if resubmission_cost is None:
            resubmission_cost = SIMULATION_CONFIG['conference']['resubmission_cost']
        # Preserve the historical integer-formatted prompt for the default fee.
        if float(resubmission_cost).is_integer():
            resubmission_cost = int(resubmission_cost)
        papers_info = []
        for idx, (arxiv_id, p) in enumerate(rejected_papers.items()):
            paper_info = f"""### Paper {idx}
Title: \"{p['title']}\"
Abstract: {p['abstract'][:300]} ...
arXiv ID: \"{arxiv_id}\"
Review History:
"""
            for review_history in p['review_history']:
                paper_info += f"""- Score: {review_history['score']:.2f} / 5 ({review_history['conference']} conference in year {review_history['year']}))
"""

            papers_info.append(paper_info)
        prompt = f"""You are a researcher. You have submitted a few papers to previous conferences but were rejected. You are considering resubmitting some of your rejected papers. You have {self.resources} units of funding. Each resubmission costs {resubmission_cost} per paper.

## Previously Rejected Papers
"""
        prompt += '\n'.join(papers_info)

        prompt += """
## Candidate Conferences

Below are the candidate conferences you can resubmit to. For each paper, you MUST CHOOSE EXACTLY ONE of the conferences that best suits the paper. Submitting to a conference with mismatch topics has a high chance of getting rejected, although not always.

"""
        candidate_conference_ids = set([conference.conference_id for conference in conferences])
        for conference in conferences:
            prompt += f"""- \"{conference.conference_id}\": primary topics: {', '.join(conference.topics[:3])} | less preferred topics: {', '.join(conference.topics[3:])}
"""

        prompt += """
## Your Task
Please respond with a list of papers you want to RESUBMIT (the FULL URL) and its conference abbreviation (e.g. "NeurIPS" or "AAAI"). For each paper, if you decide to resubmit, you must choose exactly 1 conference that best suits the paper.

## Sample Response
If you choose to resubmit, respond with the following JSON format:
```json
{
    "resubmitted_papers": [{
        "arxiv_id": "https://arxiv.org/abs/2407.12345v5",
        "conference": "ACL"
    },
    {
        "arxiv_id": "https://arxiv.org/abs/2305.67890v1",
        "conference": "AAAI"
    }]
}
```
If you choose not to resubmit, respond with a json file that contains an empty list:
```json
{
    "resubmitted_papers": []
}
```
"""

        # Generate Conference enum dynamically from available conferences
        ConferenceEnum = Enum('Conference', {cid: cid for cid in candidate_conference_ids}, type=str)

        class ResubmissionEntry(BaseModel):
            arxiv_id: str
            conference: ConferenceEnum

        class ResubmissionResponse(BaseModel):
            resubmitted_papers: List[ResubmissionEntry]

        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'resubmission_decision',
                'strict': True,
                'schema': ResubmissionResponse.model_json_schema()
            },
        }

        return prompt, response_format

    def decide_resubmission(self, rejected_papers: Dict[str, Dict], conferences: List[Conference],
                            resubmission_cost: float = None) -> List[str]:
        """Decide which rejected papers to resubmit at the resolved fee.

        Returns:
            List of paper IDs to resubmit
        """
        if resubmission_cost is None:
            resubmission_cost = SIMULATION_CONFIG['conference']['resubmission_cost']
        if self.resources < resubmission_cost or not rejected_papers:
            return []

        prompt, response_format = self.get_resubmission_prompt(rejected_papers, conferences, resubmission_cost)
        candidate_conference_ids = set([conference.conference_id for conference in conferences])

        result, _ = self.llm.generate(prompt=prompt, response_format=response_format)
        assert 'resubmitted_papers' in result
        result = result['resubmitted_papers']
        resubmitted_paper_ids = [p['arxiv_id'] for p in result]
        assert set(resubmitted_paper_ids) <= set([p['id'] for id, p in rejected_papers.items()])
        conferences = [p['conference'] for p in result]
        assert set(conferences) <= candidate_conference_ids
        return result
    
    def _get_bio(self) -> str:
        """Get bio prompt for this agent"""
        expertise = [research_direction.topic for research_direction in self.expertise]
        prompt = f"You are a researcher specializing in {', '.join(expertise)}. \n"
        return prompt

    # Helper methods (private)

    def _production_resource_statement(self, year):
        """Only opt-in projects change prompts. Balance is already net of the fee."""
        if getattr(self, 'production_cost_mode', 'per_paper') != 'per_project':
            return ''
        from utopia.funding.accounting import project_id
        entry = getattr(self, 'production_cost_entries', {}).get(project_id(self))
        if not entry or not entry['admitted'] or entry['year'] != year:
            raise ValueError(f"Production prompt without a paid project: {self.id}/{year}")
        return (f"\nA fixed production fee of {entry['fee']:g} units has already been paid "
                f"for this completed project. Your remaining funding is {self.resources:g}. "
                "Selecting any number of outputs from this project costs no additional funding. "
                "The fee is not refunded for fewer outputs or no outputs. "
                "Do not withhold an additional output because of its production cost. "
                "Annual research costs remain unchanged.\n")

    def _record_authoring_experience(self, idea: Dict, paper_output: Dict, year: int):
        """Record paper authoring experience"""
        self.memory_bank.append({
            'type': 'paper_submission',
            'timestamp': time.time(),
            'expects_harsh': self._had_recent_negative_experience(),
            'year': year,
        })

    def _get_personality_prompt(self) -> str:
        """Get personality prompt for this agent - override in subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def _had_recent_negative_experience(self) -> bool:
        """Check if agent had recent negative review experiences"""
        recent = [e for e in self.review_experiences if time.time() - e['timestamp'] < 86400 * 7]
        return any(e.get('emotional_response') == 'negative' for e in recent)

    

    def get_paper_submission_prompt(self, candidate_papers: List[Document], year: int, conferences: List[Conference],
                                    direction: ResearchDirection, paper_tracker: PaperTracker, max_papers: int = 10,
                                    previous_papers: List[str] = None, max_submissions: int = 1,
                                    force_list: bool = False) -> str:
        """
        Get research idea prompt for this agent

        Args:
            candidate_papers: Candidate papers from RAG retrieval
            conferences: List of conferences to submit to
            max_papers: Maximum number of candidate papers to show
            previous_papers: Optional list of preview strings for previously accepted papers
            max_submissions: papers to pick in this prompt (scale experiment factor S).
                1 (default) is the legacy single-paper prompt/schema, byte-identical to the
                pre-flag template. k>1 asks for up to k DISTINCT papers and returns
                {"submissions": [{id, arxiv_id, conference, reason}, ...]}.
            force_list: use the list prompt/schema even when max_submissions == 1 (retries
                inside a k>1 batch must share the batch's response_format).
        """
        assert max_submissions >= 1, max_submissions
        use_list = max_submissions > 1 or force_list
        production_statement = self._production_resource_statement(year)
        prompt = self._get_personality_prompt(include_exploration_strategy=True, include_funding_situation=True) + f"""
        
## Your Research Direction in This Round
Detailed Focus: {direction['detailed_focus']}
Keywords: {', '.join(direction['direction'].keywords)}

"""
        if not use_list:
            prompt += """## Your Task
You plan to do a research project and submit a paper to a conference.
From the following candidate papers, please give me a paper ID that you want to submit. Then, give a conference name that you want to submit to.
"""
        else:
            prompt += f"""## Your Task
You plan to do a research project and submit up to {max_submissions} papers to conferences this round.
From the following candidate papers, please give me up to {max_submissions} DISTINCT paper IDs that you want to submit. For each paper, give a conference name that you want to submit to.
"""

        if year >= 2:
            prompt += """
## Previously Accepted Papers (for reference and citation)
The following are previous papers you wrote, either accepted or rejected. You can use them to inspire your paper.
"""
            previous_papers = paper_tracker.get_papers_by_author(self.id)
            previous_papers.sort(key=lambda x: x.year, reverse=True)
            for i, paper in enumerate(previous_papers[:10], 1):  # Show top 5
                latest_score = paper.review_history[-1]['score'] if paper.review_history else 0.0
                prompt += f"{i}. \"{paper.title}\" - Year: {paper.year}, Conference: {paper.conference} ({paper.status} with score {latest_score:.2f})\n"

        # Add previously accepted papers if available (Year 2+)
        if previous_papers and len(previous_papers) > 0:
            prompt += f"""
## Previously Accepted Papers (for reference and citation)
The following papers were accepted in previous years and can be cited in your work:

"""
            for i, paper_preview in enumerate(previous_papers[:5], 1):  # Show top 5
                prompt += f"{i}. {paper_preview}\n"

            if len(previous_papers) > 5:
                prompt += f"\n... and {len(previous_papers) - 5} more papers available.\n"

        prompt += """
## Candidate Papers
"""
        for i, paper in enumerate(candidate_papers[:max_papers]):
            prompt += f"""### Paper {i + 1}
ID: {i + 1}
arXiv ID: {paper.metadata['id']}
Title: {paper.metadata['title']}
Topics: {', '.join(paper.metadata['topics'])}
Abstract: {paper.page_content}

"""

        prompt += """## Candidate Conferences
Below are the candidate conferences you can submit to. You MUST CHOOSE EXACTLY ONE of them that best suits the paper.
"""
        candidate_conference_ids = set([conference.conference_id for conference in conferences])
        for conference in conferences:
            prompt += f"""- \"{conference.conference_id}\": primary topics: {', '.join(conference.topics[:3])} | less preferred topics: {', '.join(conference.topics[3:])}
"""
        # Generate Conference enum dynamically from available conferences
        Conference = Enum('Conference', {cid: cid for cid in candidate_conference_ids}, type=str)

        if use_list:
            prompt += f"""
## Your Task
Based on the detailed focus of your research direction, respond with a list of up to {max_submissions} DISTINCT papers you want to submit. For each, give the paper ID (the FULL URL) and its conference abbreviation name (e.g. "NeurIPS" or "AAAI"). Do not list the same arXiv ID twice.

NOTE: You can also choose to submit fewer papers or none at all, but this must be due to funding issues or competition being too fierce. Reasons such as "no paper matches my expertise" is not a valid reason.

## Sample Response

If you choose to submit, respond with the following JSON format:
```json
{{
    "submissions": [
        {{
            "id": 1,
            "arxiv_id": "https://arxiv.org/abs/2407.00005v1", <- MUST be the full URL, including the "https://", prefix
            "conference": "NeurIPS",  <- MUST be one of the candidate conferences listed above
            "reason": "..." <- Reason for submitting this particular paper to this particular conference
        }},
        {{
            "id": 3,
            "arxiv_id": "https://arxiv.org/abs/2407.00012v2",
            "conference": "AAAI",
            "reason": "..."
        }}
    ]
}}
```

If you choose not to submit anything (only due to funding issues), respond with an empty list:
```json
{{
    "submissions": []
}}
```
"""

            class PaperSubmissionEntry(BaseModel):
                id: int
                arxiv_id: str
                conference: Conference
                reason: str

            class PaperSubmissionList(BaseModel):
                submissions: List[PaperSubmissionEntry]

            response_format = {
                'type': 'json_schema',
                'json_object': {
                    'name': 'paper_submission_list',
                    'strict': True,
                    'schema': PaperSubmissionList.model_json_schema()
                },
            }
            if production_statement:
                prompt = prompt.replace("funding issues or competition being too fierce",
                                        "competition being too fierce")
                prompt = prompt.replace("only due to funding issues", "due to competition being too fierce")
                prompt += production_statement
            return prompt, response_format

        prompt += """
## Your Task
Based on the detailed focus of your research direction, respond with a paper ID you want to submit (the FULL URL) and its conference abbreviation name (e.g. "NeurIPS" or "AAAI"). 

NOTE: You can also choose to not submit any paper to any conference, but this must be due to funding issues or competition being too fierce. Reasons such as "no paper matches my expertise" is not a valid reason.

## Sample Response

If you choose to submit, respond with the following JSON format:
```json
{
    "id": 1,
    "arxiv_id": "https://arxiv.org/abs/2407.00005v1", <- MUST be the full URL, including the "https://", prefix
    "conference": "NeurIPS",  <- MUST be one of the candidate conferences listed above
    "reason": "...", <- Reason for submitting this particular paper to this particular conference
}
```

If you choose not to submit (only due to funding issues), explain using factors such as funding, competition, etc. and respond with the following JSON format:
```json
{
    "id": 1,
    "arxiv_id": "DO_NOT_SUBMIT", <- MUST be the string "DO_NOT_SUBMIT"
    "conference": "DO_NOT_SUBMIT", <- MUST be the string "DO_NOT_SUBMIT"
    "reason": "...", <- Reason for not submitting any paper to any conference. 
}
```
"""

        class PaperSubmission(BaseModel):
            id: int
            arxiv_id: str
            conference: Conference
            reason: str


        # Note: as of 2025-10-20, gpt-4o dose not support the response_format parameter. "gpt-4o-2024-08-06" does.
        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'paper_submission',
                'strict': True,
                'schema': PaperSubmission.model_json_schema()
            },
        }
        if production_statement:
            prompt = prompt.replace("funding issues or competition being too fierce",
                                    "competition being too fierce")
            prompt = prompt.replace("only due to funding issues", "due to competition being too fierce")
            prompt = prompt.replace("factors such as funding, competition, etc.", "competition")
            prompt += production_statement
        return prompt, response_format

    def get_citation_prompt(self, paper: Dict, direction: ResearchDirection, candidate_papers: List[Dict], max_papers: int = 10, citing_institution: str = None) -> Tuple[str, Dict]:
        """
        Get potential cited papers prompt for this agent
        """

        prompt = self._get_bio() + f"""
You did a research project and submitted the following paper to a conference. Please give me a list of potential cited papers that you want to cite. NOTE: It is fine if you do not cite any papers due to topic mismatch.

## Your Research Direction in This Round
Detailed Focus: {direction['detailed_focus']}
Keywords: {', '.join(direction['direction'].keywords)}

## Your Submitted Paper
ID: {paper['id']}
Title: {paper['title']}
Abstract: {paper['abstract']}
Topics: {', '.join(paper['topics'])}


## Candidate Papers

"""
        for i, cited_paper in enumerate(candidate_papers[:max_papers]):
            coi_note = ""
            if citing_institution and cited_paper['institution'] == citing_institution:
                coi_note = COI_NOTE.format(institution=citing_institution)
            prompt += f"""Paper {i}
- arXiv ID: {cited_paper['id']}
- Title: {cited_paper['title']}
- Author Affiliation: {cited_paper['institution']}{coi_note}
- Abstract: {cited_paper['abstract'][:300]}...

"""

        prompt += f"""
## Sample Response
Respond in JSON format with the numeric ids (integers) of the candidate papers you want to cite. An example response is as follows (DO NOT directly copy the exact response):
```json
{{
    "citations": [9, 13] <- must be a list of integers between 0 and {len(candidate_papers[:max_papers]) - 1}
}}
```
"""

        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'citation_response',
                'strict': True,
                'schema': CitationResponse.model_json_schema()
            },
        }
        return prompt, response_format

    def get_self_citation_prompt(self, paper: Dict, direction: ResearchDirection, own_papers: List) -> Union[Tuple[str, Dict], None]:
        """
        Get self-citation prompt for this agent

        Args:
            paper: Current paper being submitted
            direction: Current research direction
            own_papers: List of agent's own previously accepted papers

        Returns:
            Prompt string asking agent to select their own papers to cite
        """
        if not own_papers:
            return None

        prompt = self._get_bio() + f"""
You submitted a paper to a conference. Please decide if you want to cite any of your own previous work.

## Your Current Paper
Title: {paper['title']}
Abstract: {paper['abstract']}
Topics: {', '.join(paper['topics'])}

## Your Research Direction
Detailed Focus: {direction['detailed_focus']}
Keywords: {', '.join(direction['direction'].keywords)}

## Your Previous Accepted Papers
"""


        paper_idx = 0
        
        accepted_papers = [p for p in own_papers if p.status == 'accept']
        rejected_papers = [p for p in own_papers if p.status != 'accept']
        
        for i, own_paper in enumerate(accepted_papers):
            latest_score = own_paper.review_history[-1]['score'] if own_paper.review_history else 0.0
            prompt += f"""### Paper {paper_idx}
Title: {own_paper.title}
Abstract: {own_paper.abstract[:300]}...
Year: {own_paper.year}

"""
            paper_idx += 1
            
        prompt += "## Your Previous Rejected Papers\n"
        for i, own_paper in enumerate(rejected_papers):
            prompt += f"""### Paper {paper_idx}
Title: {own_paper.title}
Abstract: {own_paper.abstract[:300]}...
Year: {own_paper.year}

"""
            paper_idx += 1
        

        prompt += f"""
## Response Format
Respond in JSON format with the numeric ids (integers) of the candidate papers you want to cite. An example response is as follows (DO NOT directly copy the exact response):
```json
{{
    "citations": [2, 1] <- must be a list of integers between 0 and {len(own_papers) - 1}
}}
```
"""
        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'self_citation_response',
                'strict': True,
                'schema': SelfCitationResponse.model_json_schema()
            },
        }
        return prompt, response_format


    def _get_exploration_strategy_prompt(self) -> str:
        # Add exploration strategy guidance if applicable
        prompt = ""
        if hasattr(self, 'exploration_strategy') and self.exploration_strategy != 'balanced':
            prompt += "\n## Your Research Strategy\n"

            if self.exploration_strategy == 'explorer':
                prompt += """You are an EXPLORER. You MUST follow these rules strictly:
- You MUST choose topics DISTANT from your recent work
- Accept lower short-term success for breakthrough potential
- When in doubt, choose the riskier, more novel option
- In your response, EXPLICITLY state how far this is from your past work
- Avoid incremental improvements — seek paradigm shifts and cross-domain innovation
"""
            elif self.exploration_strategy == 'exploiter':
                prompt += """You are an EXPLOITER. You MUST follow these rules strictly:
- You MUST choose topics CLOSE to your prior accepted work
- Reuse familiar methods and framing when possible
- When in doubt, choose the safer, incremental option
- In your response, EXPLICITLY state how this builds on your past work
- Focus on deepening expertise and maximizing citation impact in known areas
"""
            elif self.exploration_strategy == 'cautious_explorer':
                prompt += """You are a CAUTIOUS EXPLORER. You MUST follow these rules strictly:
- You MUST choose topics ADJACENT to your past work (not too close, not too far)
- Combine one familiar component with one new component
- Avoid both extreme conservatism and extreme novelty
- In your response, EXPLICITLY state which part is familiar and which is new
- Seek moderate-risk directions that bridge established and emerging areas
"""

        return prompt

    def get_collaboration_decision_prompt(self, candidates: List[Dict]) -> Tuple[str, Dict]:
        """Build prompt for collaboration decision.

        Args:
            candidates: List of dicts with keys: id, institution, expertise, recent_papers, score

        Returns:
            Tuple of (prompt string, response_format dict)
        """
        expertise = [e.topic if hasattr(e, 'topic') else str(e) for e in getattr(self, 'expertise', [])]
        institution = getattr(self, 'university_name', None) or getattr(self, 'company_name', None) or 'Unknown'

        prompt = f"""You are a researcher at {institution}, specializing in {', '.join(expertise)}.

## Your Task
You are considering whether to collaborate with another researcher on your next paper. Collaborating can combine complementary expertise and improve paper quality, but requires coordination.

## Candidate Collaborators
"""
        for i, c in enumerate(candidates):
            prompt += f"""### Candidate {i}
- ID: "{c['id']}"
- Institution: {c['institution']}
- Expertise: {', '.join(c['expertise'])}
- Recent papers: {c['recent_papers']}
- Compatibility score: {c['score']:.2f}

"""

        prompt += """## Instructions
Decide whether to collaborate with one of the candidates or work solo this round. If you collaborate, specify the collaborator_id exactly as shown above.

## Response Format
```json
{
    "collaborate": true,
    "collaborator_id": "candidate_id_here",
    "reason": "Brief explanation"
}
```
Or if working solo:
```json
{
    "collaborate": false,
    "collaborator_id": null,
    "reason": "Brief explanation"
}
```
"""
        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'collaboration_decision',
                'strict': True,
                'schema': CollaborationDecision.model_json_schema()
            },
        }
        return prompt, response_format
