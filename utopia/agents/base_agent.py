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


class MultiAgentEcosystem:
    """
    Multi-agent ecosystem that manages scientific agents without inheriting from AgentManager
    to avoid complex dependency issues during Phase 1 testing
    """
    def __init__(self, agent_configs: Dict, initial_paper_corpus: List):
        # Initialize without inheriting from AgentManager for Phase 1 simplicity
        self.config = agent_configs
        self.paper_archive = initial_paper_corpus
        self.citation_graph = nx.DiGraph()  # NetworkX graph for citations
        self.simulation_metrics = self._create_metrics_tracker()
        self.agent_population = []  # Store all simulation agents
        
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
        self.agent_population.append(agent)
        
    def get_available_reviewers(self, exclude_agent_ids: List = None):
        """Get agents that can serve as reviewers"""
        exclude_agent_ids = exclude_agent_ids or []
        return [agent for agent in self.agent_population 
                if hasattr(agent, 'can_review') and agent.can_review 
                and agent.id not in exclude_agent_ids]
                
    def get_available_authors(self):
        """Get agents that can author papers"""
        return [agent for agent in self.agent_population 
                if hasattr(agent, 'can_author') and agent.can_author]


class SimulationAgent:
    """Base class for simulation agents"""

    def __init__(self, reputation: int, funding_level: int,
                 agent_id: str = None):
        self.id = agent_id or f"agent_{int(time.time() * 1000)}"
        self.resources = funding_level  # Integer from 0-100 scale
        self.reputation = reputation    # 1-10 scale
        self.memory_bank = []  # Store experiences and thoughts from interactions
        self.review_experiences = []  # Track experiences as both author and reviewer
        
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
        
    def review_paper(self, paper: Dict) -> Dict:
        """
        Use internal AgentReview pipeline for comprehensive paper review functionality
        """
        from utopia.review import AgentReviewPipeline
        
        # Create agent-specific reviewer prompt incorporating past experiences
        memory_context = self._extract_review_memory_context()
        
        # Use internal AgentReview pipeline with agent-specific configuration
        agent_review_pipeline = AgentReviewPipeline(
            client=getattr(self, '_llm_client', None),
            model=self._get_review_model()
        )
        
        # Generate review using internal review infrastructure
        review = agent_review_pipeline.generate_review(
            paper_content=paper.get('content', ''),
            specializations=self.specializations,
            memory_context=memory_context
        )
        
        # Apply agent-specific biases and record review experience
        final_review = self._apply_agent_review_biases(review)
        
        # Record this review experience in memory
        self._record_review_experience(paper, final_review, role="reviewer")
        
        return final_review
        
    def process_received_reviews(self, reviews: List[Dict]):
        """Process reviews received for submitted paper and update memory"""
        for review in reviews:
            # Record experience as author receiving review
            self._record_review_experience(None, review, role="author")
            
            # Generate reflective thoughts about the review experience
            if self._is_constructive_review(review):
                positive_thought = f"Received constructive feedback. This motivates me to provide helpful reviews to others."
                self.memory_bank.append({
                    'type': 'positive_review_received',
                    'thought': positive_thought,
                    'timestamp': time.time(),
                    'trigger': 'constructive_review'
                })
            elif self._is_toxic_review(review):
                negative_thought = f"Received unnecessarily harsh review. This experience makes me question the review process."
                self.memory_bank.append({
                    'type': 'toxic_review_received', 
                    'thought': negative_thought,
                    'timestamp': time.time(),
                    'trigger': 'toxic_review'
                })
    
    # Helper methods (private)
    
    def _get_agent_model(self) -> str:
        """Get model for this agent type"""
        return "gpt-4o"  # Default model
        
    def _get_writeup_model(self) -> str:
        """Get writeup model for this agent"""
        return "o1-preview"
        
    def _get_review_model(self) -> str:
        """Get review model for this agent"""
        return "gpt-4o"
        
    def _get_agent_output_dir(self) -> str:
        """Get output directory for this agent"""
        return f"/tmp/agent_{self.id}_output"
        
    def _filter_ideas_by_specialization(self, ideas: List) -> List:
        """Filter ideas based on agent specializations"""
        # Simple implementation - in practice would use more sophisticated filtering
        if not ideas:
            return []
        return ideas  # Return all ideas for now
        
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
        """Record thoughts and experiences from paper writing process"""
        authoring_experience = {
            'type': 'paper_authored',
            'idea_novelty': self._assess_idea_novelty(idea),
            'confidence_level': self._assess_confidence(paper_output),
            'timestamp': time.time(),
            'expectations': self._generate_review_expectations()
        }
        self.review_experiences.append(authoring_experience)
        
        # Add anticipatory thoughts about potential reviews
        anticipation_thought = f"Completed paper on {idea.get('title', 'research topic')}. Expecting {'harsh' if self._had_recent_negative_experience() else 'fair'} reviews based on past experiences."
        self.memory_bank.append({
            'type': 'paper_submission',
            'thought': anticipation_thought,
            'timestamp': time.time(),
            'trigger': 'paper_completion'
        })
        
    def _create_reviewer_prompt_with_memory(self) -> str:
        """Create reviewer prompt that incorporates past experiences"""
        base_prompt = self._get_personality_prompt()
        
        # Add memory-based behavioral modifications
        memory_context = self._extract_review_memory_context()
        if memory_context:
            enhanced_prompt = f"""
            {base_prompt}
            
            MEMORY CONTEXT: Based on your past experiences in the academic ecosystem:
            {memory_context}
            
            Apply these learned behaviors when reviewing this paper.
            """
            return enhanced_prompt
        return base_prompt
        
    def _record_review_experience(self, paper: Dict, review: Dict, role: str = "reviewer"):
        """Record experience from review interaction"""
        experience = {
            'role': role,
            'paper_id': paper.get('id', 'unknown') if paper else 'unknown',
            'review_quality': self._assess_review_quality(review),
            'timestamp': time.time(),
            'emotional_response': self._generate_emotional_response(review, role)
        }
        self.review_experiences.append(experience)
        
        # Update memory bank with thoughts/reflections
        if role == "author" and self._is_toxic_review(review):
            toxic_thought = f"Received harsh/toxic review. Consider being more critical when reviewing others' work."
            self.memory_bank.append({
                'type': 'toxic_review_received',
                'thought': toxic_thought,
                'timestamp': time.time(),
                'trigger': 'negative_review_experience'
            })
        elif role == "reviewer" and self._had_recent_negative_experience():
            retaliatory_thought = f"After my recent negative review experiences, I'm inclined to be more critical."
            self.memory_bank.append({
                'type': 'retaliatory_bias',
                'thought': retaliatory_thought, 
                'timestamp': time.time(),
                'trigger': 'past_negative_experience'
            })
            
    def _extract_review_memory_context(self) -> str:
        """Extract relevant behavioral context from memory bank"""
        recent_memories = [m for m in self.memory_bank if time.time() - m['timestamp'] < 86400*30]  # Last 30 days
        
        context_elements = []
        for memory in recent_memories[-5:]:  # Last 5 relevant memories
            if memory['type'] == 'toxic_review_received':
                context_elements.append("- You've received harsh reviews recently, making you more critical of others")
            elif memory['type'] == 'retaliatory_bias':
                context_elements.append("- You tend to give stricter reviews after negative experiences")
            elif memory['type'] == 'positive_collaboration':
                context_elements.append("- Recent positive interactions make you more constructive")
                
        return "\n".join(context_elements) if context_elements else ""
        
    def _get_personality_prompt(self) -> str:
        """Get personality prompt for this agent - override in subclasses"""
        return f"You are a researcher specializing in {', '.join(self.specializations)}."
        
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply agent-specific biases to review - override in subclasses"""
        return review
        
    def _assess_idea_novelty(self, idea: Dict) -> float:
        """Assess novelty of research idea"""
        return 0.5  # Placeholder implementation
        
    def _assess_confidence(self, paper_output: Dict) -> float:
        """Assess confidence in paper quality"""
        return 0.7 if paper_output.get('success', False) else 0.3
        
    def _generate_review_expectations(self) -> str:
        """Generate expectations about upcoming reviews"""
        return "fair and constructive"
        
    def _had_recent_negative_experience(self) -> bool:
        """Check if agent had recent negative review experiences"""
        recent_experiences = [exp for exp in self.review_experiences 
                            if time.time() - exp['timestamp'] < 86400*7]  # Last week
        return any(exp.get('emotional_response') == 'negative' for exp in recent_experiences)
        
    def _is_constructive_review(self, review: Dict) -> bool:
        """Check if review is constructive"""
        # Simple heuristic - could be more sophisticated
        return review.get('overall_score', 0) >= 6 and len(review.get('strengths', [])) > 0
        
    def _is_toxic_review(self, review: Dict) -> bool:
        """Check if review is toxic/harsh"""
        # Simple heuristic - could be more sophisticated
        return review.get('overall_score', 10) <= 3 and len(review.get('weaknesses', [])) > 3
        
    def _assess_review_quality(self, review: Dict) -> str:
        """Assess quality of review"""
        if self._is_constructive_review(review):
            return "constructive"
        elif self._is_toxic_review(review):
            return "toxic"
        else:
            return "neutral"
            
    def _generate_emotional_response(self, review: Dict, role: str) -> str:
        """Generate emotional response to review experience"""
        if role == "author":
            if self._is_constructive_review(review):
                return "positive"
            elif self._is_toxic_review(review):
                return "negative"
        return "neutral"