"""
Base Agent System for Multi-Agent Scientific Ecosystem

IMPORTANT: This creates NEW classes that extend existing AI-Scientist infrastructure
WITHOUT modifying any original files.
"""

import time
import json
import tempfile
import subprocess
import hashlib
import networkx as nx
from typing import List, Dict
from const import RESEARCH_COST, PAPER_SUBMISSION_COST, REVIEW_CRITERIA
from langchain_core.documents import Document
from utopia.models.models import LLM, GPT

from utopia.config import SIMULATION_CONFIG

from utopia.agents.conference import Conference
from utopia.utils.data_utils import top_percentile


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
                if hasattr(agent, 'can_author') and agent.can_author]

    def to_dict(self) -> Dict:
        """Serialize ecosystem to dictionary (without LLM objects)

        Returns:
            Dictionary with serialized state
        """
        from utopia.agents.researcher_agents import UniversityResearcher, IndustryResearcher
        from utopia.agents.funding_agents import FundingAgency

        agents_data = []
        for agent_id, agent in self.agent_population.items():
            agent_dict = {
                'id': agent.id,
                'resources': getattr(agent, 'resources', None),
                'reputation': agent.reputation,
                'memory_bank': agent.memory_bank,
                'review_experiences': agent.review_experiences,
                'is_active': agent.is_active,
                'can_author': agent.can_author,
                'can_review': agent.can_review,
                
            }

            if isinstance(agent, UniversityResearcher):
                agent_dict['type'] = 'UniversityResearcher'
                agent_dict['university_name'] = agent.university_name
                agent_dict['expertise'] = [exp.topic for exp in agent.expertise]
                agent_dict['funding_success_history'] = agent.funding_success_history
                agent_dict['conflict_of_interest'] = list(agent.conflict_of_interest)
                agent_dict['generate_research_proposal'] = agent.generate_research_proposal

            elif isinstance(agent, IndustryResearcher):
                agent_dict['type'] = 'IndustryResearcher'
                agent_dict['company_name'] = agent.company_name
                agent_dict['expertise'] = [exp.topic for exp in agent.expertise]
                agent_dict['funding_mode'] = agent.funding_mode
                agent_dict['budget'] = agent.budget
                agent_dict['conflict_of_interest'] = list(agent.conflict_of_interest)
                agent_dict['funding_success_history'] = agent.funding_success_history

            elif isinstance(agent, FundingAgency):
                agent_dict['type'] = 'FundingAgency'
                agent_dict['agency_name'] = agent.id
                agent_dict['default_funding_rate'] = getattr(agent, 'default_funding_rate', None)
                # Serialize funding programs
                programs_data = {}
                for prog_id, program in agent.funding_programs.items():
                    programs_data[prog_id] = {
                        'program_id': program.program_id,
                        'name': program.name,
                        'research_directions': [rd.to_dict() if hasattr(rd, 'to_dict') else rd for rd in program.research_directions],
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
    def from_dict(cls, data: Dict, llm: LLM) -> 'MultiAgentEcosystem':
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

            if agent_type == 'UniversityResearcher':
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
                    generate_research_proposal=agent_data.get('generate_research_proposal', True)
                )
                agent.reputation = agent_data['reputation']
                agent.memory_bank = agent_data.get('memory_bank', [])
                agent.review_experiences = agent_data.get('review_experiences', [])
                agent.is_active = agent_data.get('is_active', True)
                agent.funding_success_history = agent_data.get('funding_success_history', {})
                agent.conflict_of_interest = set(agent_data.get('conflict_of_interest', []))

            elif agent_type == 'IndustryResearcher':
                # Reconstruct expertise
                expertise = []
                for exp_data in agent_data.get('expertise', []):
                    if isinstance(exp_data, dict):
                        expertise.append(ResearchDirection.from_dict(exp_data))
                    else:
                        assert direction in DIRECTIONS_DICT, f"Direction {direction} not found in DIRECTIONS_DICT"
                        expertise.append(DIRECTIONS_DICT[direction])

                agent = IndustryResearcher(
                    researcher_name=agent_data['id'],
                    company_name=agent_data['company_name'],
                    funding_level=agent_data['resources'],
                    expertise=expertise,
                    llm=llm,
                    funding_mode=agent_data.get('funding_mode', 'performance')
                )
                agent.reputation = agent_data['reputation']
                agent.memory_bank = agent_data.get('memory_bank', [])
                agent.review_experiences = agent_data.get('review_experiences', [])
                agent.is_active = agent_data.get('is_active', True)
                agent.budget = agent_data.get('budget', agent.budget)
                agent.research_priorities = set(agent_data.get('research_priorities', []))
                agent.conflict_of_interest = set(agent_data.get('conflict_of_interest', []))
                agent.funding_success_history = agent_data.get('funding_success_history', {})

            elif agent_type == 'FundingAgency':
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

            ecosystem.add_agent(agent)

        return ecosystem


class SimulationAgent:
    """Base class for simulation agents"""

    def __init__(self, reputation: int,
                 agent_id: str = None, llm: LLM = None):
        self.id = agent_id or f"agent_{int(time.time() * 1000)}"
        self.reputation = reputation    # 1-10 scale
        self.memory_bank = []  # Store experiences and thoughts from interactions
        self.review_experiences = []  # Track experiences as both author and reviewer
        self.llm = llm
        self.is_active = True
        
    def generate_research_idea(self, citation_context: List, research_focus: str = None) -> Dict:
        """
        WRAPPER around existing ai_scientist/perform_ideation_temp_free.py
        Does NOT modify the original function
        """
        from utopia.perform_ideation_temp_free import generate_temp_free_idea
        
        # Check if agent has sufficient resources
        if self.resources < RESEARCH_COST:
            raise ValueError(f"Agent {self.id} has insufficient resources ({self.resources}) for research")
        
        # WRAPPER: Call existing ideation function without modification
        # Add agent-specific behavior through parameters only
        ideas = generate_temp_free_idea(
            workshop_description=research_focus if research_focus else f"Research in {', '.join(self.specializations)}",
            model=self._get_agent_model(),
            max_num_generations=1,  # Use existing parameters
            num_reflections=3       # Use existing parameters
        )
        
        # POST-PROCESS: Add agent-specific filtering without changing original
        agent_filtered_ideas = self._filter_ideas_by_specialization(ideas)
        
        # Deduct research cost for idea generation
        self.resources = max(0, self.resources - RESEARCH_COST)
        
        return agent_filtered_ideas
     
    def update_resources(self, delta: float):
        """Update researcher resources"""
        assert isinstance(delta, (int, float)), "changes in funding level must be a valid number"
        self.resources += delta
        
        
    def get_memory_context(self) -> str:
        """Get memory context for this agent"""
        result = ""
        for memory in self.memory_bank:
            result += f"""- Event: {memory['type']} (Year: {memory['year']}) - Thought: {memory['thought']}
"""
        return result
        
    def review_paper(self, paper: Dict) -> Dict:
        """Generate review for a paper using LLM

        Args:
            paper: Paper dictionary with 'title', 'content', 'topics', etc.

        Returns:
            Review dictionary with 'overall_score', 'strengths', 'weaknesses', 'summary'
        """
        expertise = [e.topic if hasattr(e, 'topic') else str(e) for e in getattr(self, 'expertise', [])]
        memory_context = self.get_memory_context()
        
        # TODO: Generate topics for the papers
        
        if memory_context:
            memory_context = f"""## Your Recent Experiences 
{memory_context}
"""
            
        else:
            memory_context = ""

        prompt = f"""You are a peer reviewer for an academic conference. Review the following paper based on your expertise and past experiences (e.g. your own past submissions, received reviews, your funding situations, your status level).

Your Expertise: {', '.join(expertise)}

## Paper

Title: {paper.get('title', 'Unknown')}

Abstract:
{paper.get('content', '')}

Paper Topics: {', '.join(paper.get('topics', []))}

{memory_context}

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
        
        status_string = f"(Top {percentile:.1f}% of all papers" #, average score: {avg_score:.1f} on a scale of 1-5)"

        # Determine emotional response based on average score
        if percentile <= 10: # percentile >= 4.0:
            
            thought = f"My paper received excellent reviews {status_string}! This validates my research direction and boosts my confidence."
            memory_type = 'excellent_reviews_received'
        elif percentile <= 30: # percentile >= 3.0:
            thought = f"My paper received moderately good reviews {status_string}. There's room for improvement, but overall positive."
            memory_type = 'good_reviews_received'
        elif percentile <= 50: # percentile >= 2.0:
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

    def decide_resubmission(self, rejected_papers: Dict[str, Dict], conferences: List[Conference], resubmission_cost: int = SIMULATION_CONFIG['conference']['resubmission_cost']) -> List[str]:
        """Decide which rejected papers to resubmit (costs 5 each)

        Returns:
            List of paper IDs to resubmit
        """
        if self.resources < 5 or not rejected_papers:
            return []

        papers_info = []
        for idx, (arxiv_id, p) in enumerate(rejected_papers.items()):
            paper_info = f"""### Paper {idx}
Title: \"{p['title']}\"
Abstract: {p['content'][:300]} ...
arXiv ID: \"{arxiv_id}\"
Review History:
"""
            for review_history in p['review_history']:
                paper_info += f"""- Score: {review_history['score']:.2f} / 5 ({review_history['conference']} conference in year {review_history['year']}))
"""

            papers_info.append(paper_info)
        prompt = f"""You are considering resubmitting some of your rejected papers. You have {self.resources} units of funding. Each resubmission costs {resubmission_cost} per paper.

## Previously Rejected Papers
"""
        prompt += '\n'.join(papers_info)

        prompt += """
## Candidate Conferences

Below are the candidate conferences you can resubmit to. For each paper, you MUST CHOOSE EXACTLY ONE of them that best suits the paper. Submitting to a conference with mismatch topics has a high chance of getting rejected, although not always.

"""
        candidate_conference_ids = set([conference.conference_id for conference in conferences])
        for conference in conferences:
            prompt += f"""- \"{conference.conference_id}\": primary topics: {', '.join(conference.topics[:3])} | less preferred topics: {', '.join(conference.topics[3:])}
"""

        prompt += """
Please respond with a list of papers you want to RESUBMIT (the FULL URL) and its conference abbreviation (e.g. "NeurIPS" or "AAAI").
        
## Sample Response
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
"""

        result, _ = self.llm.generate(prompt=prompt, response_format="json_object")
        assert 'resubmitted_papers' in result
        result = result['resubmitted_papers']
        
        resubmitted_paper_ids = [p['arxiv_id'] for p in result]
        assert set(resubmitted_paper_ids) <= set([p['id'] for id, p in rejected_papers.items()])
        conferences = [p['conference'] for p in result]
        assert set(conferences) <= candidate_conference_ids
        return result

    # Helper methods (private)
    
    def _parse_pipeline_output(self, result: subprocess.CompletedProcess, output_dir: str) -> Dict:
        """Parse output from AI-Scientist pipeline"""
        if result.returncode != 0:
            raise RuntimeError(f"Pipeline failed: {result.stderr}")
            
        # Try to read generated files
        import os
        paper_output = {
            'writeup_md': '',
            'plot_py': '',
            'experiment_py': '',
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Check if output files exist and read them
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.endswith('.md'):
                    with open(os.path.join(output_dir, file), 'r') as f:
                        paper_output['writeup_md'] = f.read()
                elif file.endswith('.py') and 'plot' in file:
                    with open(os.path.join(output_dir, file), 'r') as f:
                        paper_output['plot_py'] = f.read()
                elif file.endswith('.py') and 'experiment' in file:
                    with open(os.path.join(output_dir, file), 'r') as f:
                        paper_output['experiment_py'] = f.read()
        
        return paper_output
        
    def _add_traceable_citations(self, paper_output: Dict, references: List) -> Dict:
        """Ensure every paper has traceable citations with complete metadata"""
        citation_list = []
        
        # Extract citations from paper content and enrich with metadata
        cited_papers = self._extract_citations_from_content(paper_output.get('writeup_md', ''))
        
        for i, cited_paper in enumerate(cited_papers):
            # Create structured citation entry with traceable fields
            citation_entry = {
                'id': cited_paper.get('paper_id', f"paper_{hashlib.md5(str(cited_paper).encode()).hexdigest()[:8]}"),
                'title': cited_paper.get('title', f'Unknown Title {i+1}'),
                'authors': cited_paper.get('authors', ['Unknown Author']),
                'venue': cited_paper.get('venue', 'Unknown Venue'),
                'year': cited_paper.get('year', 'Unknown Year'),
                'abstract': cited_paper.get('abstract', ''),
                'paper_type': cited_paper.get('paper_type', 'human_written'),  # human_written or ai_generated
                'agent_author': cited_paper.get('agent_author', None),  # If AI-generated, track which agent
                'citation_context': cited_paper.get('context', ''),  # How this paper relates to the citing work
                'semantic_scholar_id': cited_paper.get('s2_id', None),  # For real papers
                'simulation_id': cited_paper.get('sim_id', None)  # For simulation-generated papers
            }
            citation_list.append(citation_entry)
        
        # Add structured citation list to paper output
        paper_output['citations'] = citation_list
        paper_output['citation_metadata'] = {
            'total_citations': len(citation_list),
            'human_papers_cited': len([c for c in citation_list if c['paper_type'] == 'human_written']),
            'ai_papers_cited': len([c for c in citation_list if c['paper_type'] == 'ai_generated']),
            'citation_diversity_score': self._calculate_citation_diversity(citation_list)
        }
        
        return paper_output
        
    def _extract_citations_from_content(self, content: str) -> List[Dict]:
        """Extract citations from paper content - simplified implementation"""
        # Simple implementation - in practice would use more sophisticated parsing
        citations = []
        # For now, return empty list - would implement actual citation extraction
        return citations
        
    def _calculate_citation_diversity(self, citation_list: List[Dict]) -> float:
        """Calculate citation diversity score"""
        if not citation_list:
            return 0.0
        
        # Simple diversity measure: ratio of unique venues
        unique_venues = len(set(c['venue'] for c in citation_list))
        total_citations = len(citation_list)
        return unique_venues / total_citations if total_citations > 0 else 0.0
        
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
        return f"You are a researcher specializing in {', '.join(self.specializations)}."
        
    
        
    def _had_recent_negative_experience(self) -> bool:
        """Check if agent had recent negative review experiences"""
        recent = [e for e in self.review_experiences if time.time() - e['timestamp'] < 86400*7]
        return any(e.get('emotional_response') == 'negative' for e in recent)
    
    def get_bio(self) -> str:
        """Get bio prompt for this agent"""
        expertise = [research_direction.topic for research_direction in self.expertise]
        prompt = f"You are a researcher specializing in {', '.join(expertise)}."
        return prompt
    
    def get_paper_submission_prompt(self, candidate_papers: List[Document], year: int, conferences: List[Conference], paper_archive, max_papers: int = 10, previous_papers: List[str] = None) -> str:
        """
        Get research idea prompt for this agent

        Args:
            candidate_papers: Candidate papers from RAG retrieval
            conferences: List of conferences to submit to
            max_papers: Maximum number of candidate papers to show
            previous_papers: Optional list of preview strings for previously accepted papers
        """
        prompt = self.get_bio() + f"""
You plan to do a project and submit a paper to a conference.
From the following candidate papers, please give me a paper ID that you want to submit. Then, give a conference name that you want to submit to.
"""

        if year >= 2:
            prompt += """
## Previously Accepted Papers (for reference and citation)
The following are previous papers you wrote, either accepted or rejected. You can use them to inspire your paper.
"""
            previous_papers = paper_archive.get_papers_by_author(self.id)
            previous_papers.sort(key=lambda x: x.year, reverse=True)
            for i, paper in enumerate(previous_papers[:10], 1):  # Show top 5
                latest_score = paper.review_history[-1]['score'] if paper.review_history else 0.0
                prompt += f"{i}. \"{paper.title}\" - Year: {paper.year}, Conference: {paper.conference} ({paper.decision} with score {latest_score:.2f})\n"


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
            prompt += f"""### Paper {i+1}
ID: {i+1}
arXiv ID: {paper.metadata['id']}
Title: {paper.metadata['title']}
Abstract: {paper.page_content}

"""

        prompt += """## Candidate Conferences
Below are the candidate conferences you can submit to. You MUST CHOOSE EXACTLY ONE of them that best suits the paper. Submitting to a conference with mismatch topics has a high chance of getting rejected, although not always.

"""
        candidate_conference_ids = set([conference.conference_id for conference in conferences])
        for conference in conferences:
            prompt += f"""- \"{conference.conference_id}\": primary topics: {', '.join(conference.topics[:3])} | less preferred topics: {', '.join(conference.topics[3:])}
"""
        prompt += """
## Your Task
Please respond with a paper ID you want to submit (the FULL URL) and its conference abbreviation name (e.g. "NeurIPS" or "AAAI").

## Sample Response
```json
{
    "id": 1,
    "arxiv_id": "https://arxiv.org/abs/2407.00005v1", <- MUST be the full URL, including the "https://" prefix
    "conference": "NeurIPS"  <- MUST be one of the candidate conferences listed above
}
```
"""

        # Note: as of 2025-10-20, gpt-4o dose not support the response_format parameter. "gpt-4o-2024-08-06" does.
        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'paper_submission',
                'strict': True,
                'schema': {
                    'type': 'object',
                    'properties': {'arxiv_id': {'type': 'string'}, 'id': {'type': 'number'}},
                    'required': ['id', 'arxiv_id'],
                }
            },
        }
        return prompt, response_format


    def get_potential_cited_papers_prompt(self, paper: Dict, candidate_papers: List[Dict], max_papers: int = 10) -> str:
        """
        Get potential cited papers prompt for this agent
        """
        prompt = f"""
Now that you have submitted the following paper, please give me a list of potential cited papers that you want to cite. NOTE: It is fine if you do not cite any papers due to topic mismatch.

## Your Submitted Paper
ID: {paper['id']}
Title: {paper['title']}
Abstract: {paper['content']}
Topics: {', '.join(paper['topics'])}


## Candidate Papers

"""
        for i, cited_paper in enumerate(candidate_papers[:max_papers]):
            prompt += f"""### Paper {i+1}
ID: {i+1}
arXiv ID: {cited_paper['id']}
Title: {cited_paper['title']}
Abstract: {cited_paper['content'][:300]}...

"""

        prompt += """
## Sample Response
```json
{
    "citations": [ FULL URL of the candidate papers you want to cite, including the "https://" prefix
        "https://arxiv.org/abs/2407.00005v1",
        "https://arxiv.org/abs/2407.00006v1",
        "https://arxiv.org/abs/2407.00007v1"
    ]
}
```
"""

        return prompt



    
