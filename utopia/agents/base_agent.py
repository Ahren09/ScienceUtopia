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
from const import RESEARCH_COST, PAPER_SUBMISSION_COST
from langchain_core.documents import Document
from utopia.models.models import LLM, GPT

from utopia.agents.conference import Conference


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
        return self.agent_population.get(agent_id)
        
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
                'resources': agent.resources,
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
            initial_paper_corpus=[]
        )

        ecosystem.simulation_metrics = data.get('simulation_metrics', ecosystem._create_metrics_tracker())
        # Reconstruct agents
        all_agent_data = data.get('agents', [])
        for agent_data in all_agent_data:
            agent_type = agent_data['type']

            if agent_type == 'UniversityResearcher':
                # Reconstruct expertise
                expertise = []
                for exp_data in agent_data.get('expertise', []):
                    if isinstance(exp_data, dict):
                        expertise.append(ResearchDirection.from_dict(exp_data))
                    else:
                        expertise.append(exp_data)

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
                        expertise.append(exp_data)

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

    def __init__(self, reputation: int, funding_level: int,
                 agent_id: str = None, llm: LLM = None):
        self.id = agent_id or f"agent_{int(time.time() * 1000)}"
        self.resources = funding_level  # Integer from 0-100 scale
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
        
    def write_paper(self, idea: Dict, references: List) -> Dict:
        """
        WRAPPER around existing launch_scientist_bfts.py pipeline
        Does NOT modify the original pipeline
        """
        # Check if agent has sufficient resources
        if self.resources < PAPER_SUBMISSION_COST:
            raise ValueError(f"Agent {self.id} has insufficient resources ({self.resources}) for paper submission")
        
        # Import existing pipeline WITHOUT modification
        import os
        
        # WRAPPER: Use existing pipeline through subprocess to avoid modification
        # Create temporary config for this agent
        agent_output_dir = self._get_agent_output_dir()
        
        # Call existing AI-Scientist pipeline as-is without modification
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_ideas:
            json.dump(idea, temp_ideas)
            temp_ideas_path = temp_ideas.name
        
        try:
            # Execute existing pipeline without modifying it
            result = subprocess.run([
                'python', 'launch_scientist_bfts.py',
                '--load_ideas', temp_ideas_path,
                '--model_writeup', self._get_writeup_model(),
                '--model_review', self._get_review_model(),
                '--output_dir', agent_output_dir
            ], capture_output=True, text=True, cwd='/nethome/yjin328/Workspace/PeerReview/AI-Scientist-v2')
            
            # POST-PROCESS: Parse output and add agent-specific enhancements
            paper_output = self._parse_pipeline_output(result, agent_output_dir)
            paper_output = self._add_traceable_citations(paper_output, references)
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_ideas_path):
                os.unlink(temp_ideas_path)
        
        # Deduct cost for paper writing and submission
        self.resources = max(0, self.resources - PAPER_SUBMISSION_COST)
        
        # Record paper writing experience
        self._record_authoring_experience(idea, paper_output)
        
        return paper_output
        
    def update_resources(self, delta: float):
        """Update researcher resources"""
        assert isinstance(delta, (int, float)), "changes in funding level must be a valid number"
        self.resources += delta
        
    def review_paper(self, paper: Dict) -> Dict:
        """Generate review for a paper using LLM

        Args:
            paper: Paper dictionary with 'title', 'content', 'topics', etc.

        Returns:
            Review dictionary with 'overall_score', 'strengths', 'weaknesses', 'summary'
        """
        expertise = [e.topic if hasattr(e, 'topic') else str(e) for e in getattr(self, 'expertise', [])]
        memory_context = self._extract_review_memory_context()
        
        # TODO: Generate topics for the papers
        
        if memory_context:
            memory_context = f"""## Memory Context
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
Provide an  comprehensive review with:
1. Overall Score (1-5 scale):
   - 5: Award-worthy (top 2.5%)
   - 4: Strong accept
   - 3: Borderline accept
   - 2: Needs major revisions
   - 1: Reject
   
2. A succinct, 1-sentence justification for the score.

## Sample Response
```json
{{
    "justification": "The paper has critical flaws and needs major revisions."
    "overall_score": 2,
}}
```
"""

        review_json = self.llm.generate(prompt=prompt)
        review = json.loads(review_json)

        # Apply agent-specific biases
        final_review = self._apply_agent_review_biases(review)
        final_review['reviewer_id'] = self.id

        return final_review
        
    def process_received_reviews(self, reviews: List[Dict], paper_id: str = None):
        """Process reviews received for submitted paper and update memory

        Args:
            reviews: List of review dictionaries
            paper_id: ID of the paper that was reviewed
        """
        if not reviews:
            return

        # Calculate average score
        avg_score = sum(r.get('overall_score', 0) for r in reviews) / len(reviews)

        # Determine emotional response based on average score
        if avg_score >= 4.0:
            
            thought = f"My paper received excellent reviews (avg: {avg_score:.2f})! This validates my research direction and boosts my confidence."
            memory_type = 'excellent_reviews_received'
        elif avg_score >= 3.0:
            thought = f"My paper received moderately good reviews (avg: {avg_score:.2f}). There's room for improvement, but overall positive."
            memory_type = 'good_reviews_received'
        elif avg_score >= 2.0:
            thought = f"My paper received average reviews (avg: {avg_score:.2f}). I need to work harder to improve my research quality."
            memory_type = 'average_reviews_received'
        else:
            thought = f"My paper received very poor reviews (avg: {avg_score:.2f}). This is disappointing and frustrating."
            memory_type = 'poor_reviews_received'

        # Record overall experience
        self.memory_bank.append({
            'type': memory_type,
            'thought': thought,
            'timestamp': time.time(),
            'avg_score': avg_score,
            'num_reviews': len(reviews),
            'paper_id': paper_id,
        })

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
        
    def _record_authoring_experience(self, idea: Dict, paper_output: Dict):
        """Record paper authoring experience"""
        self.memory_bank.append({
            'type': 'paper_submission',
            'timestamp': time.time(),
            'expects_harsh': self._had_recent_negative_experience()
        })
            
    def _extract_review_memory_context(self) -> str:
        """Extract relevant behavioral context from memory bank"""
        recent_memories = [m for m in self.memory_bank if time.time() - m['timestamp'] < 86400*30]  # Last 30 days

        context_elements = []
        for memory in recent_memories[-5:]:  # Last 5 relevant memories
            mtype = memory.get('type', '')
            if mtype == 'excellent_reviews_received':
                context_elements.append("- Your recent paper received excellent reviews, making you feel confident and generous")
            elif mtype == 'good_reviews_received':
                context_elements.append("- Your recent paper received good reviews, putting you in a positive mood")
            elif mtype == 'average_reviews_received':
                context_elements.append("- Your recent paper received mixed reviews, making you more critical in your evaluations")
            elif mtype == 'poor_reviews_received':
                context_elements.append("- Your recent paper received poor reviews, making you frustrated and potentially harsher on others")
            elif mtype == 'toxic_review_received':
                context_elements.append("- You've received harsh reviews recently, making you more critical of others")
            elif mtype == 'retaliatory_bias':
                context_elements.append("- You tend to give stricter reviews after negative experiences")
            elif mtype == 'positive_collaboration':
                context_elements.append("- Recent positive interactions make you more constructive")

        return "\n".join(context_elements) if context_elements else ""
        
    def _get_personality_prompt(self) -> str:
        """Get personality prompt for this agent - override in subclasses"""
        return f"You are a researcher specializing in {', '.join(self.specializations)}."
        
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply agent-specific biases to review - override in subclasses"""
        return review
    
        
    def _had_recent_negative_experience(self) -> bool:
        """Check if agent had recent negative review experiences"""
        recent = [e for e in self.review_experiences if time.time() - e['timestamp'] < 86400*7]
        return any(e.get('emotional_response') == 'negative' for e in recent)
    
    def get_paper_submission_prompt(self, candidate_papers: List[Document], conferences: List[Conference], max_papers: int = 10, previous_papers: List[str] = None) -> str:
        """
        Get research idea prompt for this agent

        Args:
            candidate_papers: Candidate papers from RAG retrieval
            conferences: List of conferences to submit to
            max_papers: Maximum number of candidate papers to show
            previous_papers: Optional list of preview strings for previously accepted papers
        """

        expertise = [research_direction.topic for research_direction in self.expertise]
        prompt = f"""
You are a researcher specializing in {', '.join(expertise)}.
You plan to do a project and submit a paper to a conference.
From the following candidate papers, please give me a paper ID that you want to submit. Then, give a conference name that you want to submit to.
"""

        # Add previously accepted papers if available (Year 2+)
        if previous_papers and len(previous_papers) > 0:
            raise NotImplementedError("Previous papers are not supported yet")
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
Below are the candidate conferences you can submit to. Please choose one of them that suits the paper. Submitting to a conference with mismatch topics has a high chance of getting rejected, although not always.
"""
        candidate_conference_ids = set([conference.conference_id for conference in conferences])
        for conference in conferences:
            prompt += f"""- \"{conference.conference_id}\": primary topics: {', '.join(conference.topics[:3])} | less preferred topics: {', '.join(conference.topics[3:])}
"""
        prompt += f"""
## Your Task
Please respond with a paper ID you want to submit (the full URL) and its conference abbreviated name (e.g. "NeurIPS" or "AAAI").

## Sample Response
```json
{{
    "id": 1,
    "arxiv_id": "http://arxiv.org/abs/2407.00005v1",
    "conference": "NeurIPS"
}}
```
"""

        schema = {
            "name": "city_weather_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "number",
                        "description": "ID of the paper"
                    },
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv ID of the paper"
                    },
                },
                "required": ["id", "arxiv_id"],
                "additionalProperties": False
            },
            "strict": True  # ensures model cannot output extra fields
        }

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
    
    