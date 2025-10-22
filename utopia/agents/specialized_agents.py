"""
Specialized Agent Types for Multi-Agent Scientific Ecosystem

EXTENDS base agent classes WITHOUT modifying existing AI-Scientist code
"""

import json
import random
from .base_agent import SimulationAgent
from .research_direction import ResearchDirection
from typing import List, Dict, Optional
from utopia.models.models import LLM, GPT


class UniversityResearcher(SimulationAgent):
    """University researchers serve as both authors and reviewers in the ecosystem

    Multiple UniversityResearchers can be mapped to the same University.
    The university affiliation is tracked separately from the researcher identity.
    """

    def __init__(
        self,
        researcher_name: str,
        university_name: str,
        funding_level: int = 70,
        expertise: Optional[List] = None,
        llm = None
    ):
        """Initialize a university researcher

        Args:
            researcher_name: Name of the individual researcher
            university_name: Name of the university this researcher belongs to
            funding_level: Initial funding level (0-100)
            expertise_directions: List of 3 ResearchDirection objects representing inherent expertise.
                                 If None, will be randomly sampled during initialization.
        """
        super().__init__(
            reputation=5,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )
        self.university_name = university_name
        self.can_author = True  # University researchers can write papers
        self.can_review = True  # University researchers can review papers

        # Store inherent expertise (3 research directions)
        self.expertise = expertise
        
    def _get_personality_prompt(self) -> str:
        return """You are a university researcher who both authors papers and reviews them for conferences. 
               You focus on theoretical contributions, novel methodologies, and advancing fundamental knowledge. 
               You value reproducibility, peer review rigor, and long-term scientific impact. Your reviewing 
               style is influenced by your own experiences as an author in the academic ecosystem."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply university-specific biases to reviews"""
        # Universities tend to value theoretical rigor and reproducibility
        strengths_text = ' '.join(review.get('strengths', []))
        if 'theoretical' in strengths_text.lower() or 'theoretical_contribution' in strengths_text:
            # Slight positive bias for theoretical work
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.5)

        if 'reproducibility' not in strengths_text.lower() and review.get('overall_score', 5) > 7:
            # Penalize high scores if reproducibility concerns exist
            review['overall_score'] = max(1, review.get('overall_score', 5) - 0.5)

        return review

    def prepare_funding_application(
        self,
        past_performance: int,
        current_research_direction: str
    ) -> Dict:
        """Prepare a funding application for submission to funding agencies

        Args:
            past_performance: Number of accepted papers in previous year
            current_research_direction: Description of current research direction

        Returns:
            Dictionary with application details
        """
        # University researchers create detailed research proposals
        application = {
            'applicant_id': self.id,
            'affiliation': self.university_name,
            'research_proposal': current_research_direction,
            'expertise': self.expertise if self.expertise else [],
            'past_performance': past_performance,
            'current_funding': self.resources,
            'applicant_type': 'university'
        }
        return application


class IndustryResearcher(SimulationAgent):
    """Industry researchers serve as both authors and reviewers in the ecosystem

    Multiple IndustryResearchers can be mapped to the same Company.
    The company affiliation is tracked separately from the researcher identity.
    """

    def __init__(
        self,
        researcher_name: str,
        company_name: str,
        funding_level: int = 90,
        expertise: Optional[List] = None,
        llm: LLM = None,
        funding_mode: str = "performance"  # "performance" or "consistent"
    ):
        """Initialize an industry researcher

        Args:
            researcher_name: Name of the individual researcher
            company_name: Name of the company this researcher belongs to
            funding_level: Initial funding level (0-100)
            expertise_directions: List of 3 ResearchDirection objects representing inherent expertise.
                                 If None, will be randomly sampled during initialization.
            funding_mode: Funding allocation mode - "performance" (based on papers) or "consistent" (fixed allocation)
        """

        super().__init__(
            reputation=6,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )
        self.expertise = expertise
        self.company_name = company_name
        self.can_author = True  # Industry researchers can write papers
        self.can_review = True  # Industry researchers can review papers
        self.funding_mode = funding_mode  # "performance" or "consistent"

        # Store inherent expertise (3 research directions)
        
    def _get_personality_prompt(self) -> str:
        return """You are an industry researcher who both authors papers and reviews them for conferences.
               You focus on practical applications, scalable solutions, and commercially viable innovations. 
               You prioritize efficiency, real-world impact, and competitive advantages. Your reviewing style 
               is influenced by your commercial perspective and your own publication experiences."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply industry-specific biases to reviews"""
        # Industry agents prefer practical applications and scalability
        strengths_text = ' '.join(review.get('strengths', []))
        if any(term in strengths_text.lower() for term in ['practical', 'scalable', 'real-world']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.7)

        weaknesses_text = ' '.join(review.get('weaknesses', []))
        if 'theoretical' in weaknesses_text.lower():
            # Less penalty for theoretical limitations
            review['overall_score'] = max(1, review.get('overall_score', 5) + 0.3)

        return review

    def get_funding_allocation(self, papers_accepted: int) -> int:
        """Calculate funding allocation based on funding mode

        Args:
            papers_accepted: Number of papers accepted this year

        Returns:
            Funding amount to allocate
        """
        if self.funding_mode == "performance":
            # Performance-based: funding increases with accepted papers
            if papers_accepted > 0:
                funding_gain = 30 * papers_accepted
                return funding_gain
            else:
                # Small penalty for no papers
                return -10
        elif self.funding_mode == "consistent":
            # Consistent allocation: fixed amount regardless of performance
            # Maintain funding at 80-90 range
            if self.resources < 80:
                return 20  # Boost funding back up
            elif self.resources > 90:
                return -10  # Reduce excess funding
            else:
                return 5  # Small consistent increase
        else:
            raise ValueError(f"Unknown funding mode: {self.funding_mode}")


class FreelancerAgent(SimulationAgent):
    """Freelancer agents primarily focus on authoring but may occasionally review"""

    def __init__(
        self,
        researcher_name: str,
        funding_level: int = 30,
        agent_id: str = None,
        expertise_directions: Optional[List] = None,
        llm: LLM = None
    ):

        assert agent_id is not None, "agent_id is required"
        super().__init__(
            reputation=4,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )

        self.can_author = True   # Freelancers primarily write papers
        self.can_review = True   # Freelancers can review, but less frequently invited

        # Store inherent expertise (3 research directions)
        self.expertise_directions = expertise_directions or []
        
    def _get_personality_prompt(self) -> str:
        return """You are an independent researcher who challenges conventional wisdom 
               and pursues unconventional approaches. You value radical innovation, 
               interdisciplinary thinking, and paradigm-shifting discoveries. When reviewing,
               you appreciate bold ideas and are willing to champion unconventional work."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply freelancer-specific biases to reviews"""
        # Freelancers appreciate novelty and unconventional approaches
        if any(term in str(review.get('strengths', [])).lower() 
               for term in ['novel', 'innovative', 'unconventional', 'creative']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 1.0)
            
        if 'lacks novelty' in str(review.get('weaknesses', [])).lower():
            # Strong penalty for lack of novelty
            review['overall_score'] = max(1, review.get('overall_score', 5) - 1.5)
            
        return review


class FundingProgram:
    """Represents a specific funding program offered by a funding agency"""

    def __init__(
        self,
        program_id: str,
        name: str,
        budget: int,
        focus_areas: List[ResearchDirection],
        min_score: float = 6.0,
        max_awards: int = 5
    ):
        self.program_id = program_id
        self.name = name
        self.budget = budget  # Total budget available
        self.focus_areas = focus_areas  # Research areas (ResearchDirection objects) this program targets
        self.min_score = min_score  # Minimum score to be eligible
        self.max_awards = max_awards  # Maximum number of awards per cycle
        self.awarded_this_cycle = 0
        self.remaining_budget = budget

    def reset_for_new_cycle(self):
        """Reset program state for a new funding cycle"""
        self.awarded_this_cycle = 0
        self.remaining_budget = self.budget

    def can_award(self) -> bool:
        """Check if program can still award funding"""
        return self.awarded_this_cycle < self.max_awards and self.remaining_budget > 0


class FundingAgent(SimulationAgent):
    """Funding agents primarily review and allocate resources, occasionally author policy papers"""

    def __init__(self, agency_name: str, funding_level: int = 100, llm: LLM = None):


        super().__init__(
            reputation=8,
            funding_level=funding_level,
            agent_id=agency_name,
            llm=llm
        )
        self.llm = llm
        self.funding_priorities = []
        self.risk_aversion = "moderate"
        self.can_author = False  # Funding agents typically don't write research papers
        self.can_review = False

        # Initialize funding programs
        self.funding_programs: List[FundingProgram] = []
        self._initialize_funding_programs(agency_name) 
        
    def _get_personality_prompt(self) -> str:
        return """You are a funding agency representative who primarily serves as a reviewer
               for academic conferences. You evaluate research from a strategic funding perspective,
               considering long-term impact, resource allocation efficiency, and alignment with
               funding priorities. Your reviewing style is conservative but fair."""
               
    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply funding agency-specific biases to reviews"""
        # Funding agents are conservative and strategic
        if any(term in str(review.get('strengths', [])).lower() 
               for term in ['impact', 'strategic', 'significant']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.3)
            
        if any(term in str(review.get('weaknesses', [])).lower() 
               for term in ['high-risk', 'speculative', 'unproven']):
            # Conservative penalty for high-risk work
            review['overall_score'] = max(1, review.get('overall_score', 5) - 0.8)
            
        return review
        
    def allocate_funding(self, proposals: List[Dict]) -> List[Dict]:
        """Use existing review infrastructure for funding decisions"""
        funding_decisions = []
        for proposal in proposals:
            # Use review pipeline to evaluate funding proposals
            evaluation = self.review_paper(proposal)  # Reuse review infrastructure
            funding_decision = self._make_funding_decision(evaluation, proposal)
            funding_decisions.append(funding_decision)
        return funding_decisions
        
    def _initialize_funding_programs(self, agency_name: str):
        """Initialize funding programs based on agency type

        NSF: Foundational scientific advancement - focuses on theory, fundamental research,
             longer-term projects with lower scores acceptable (patient funding)
        DARPA: Application-driven, shorter-term projects - focuses on practical systems,
               deployment-ready technology with higher score thresholds (demanding funding)
        """
        from .research_direction import DIRECTIONS_DICT

        # Different agencies have different focus areas and budgets
        if agency_name == "NSF":
            # NSF: Foundational scientific advancement
            # Focus on theory, algorithms, and fundamental understanding
            # More patient with longer-term research (lower min_score, higher budgets)
            self.funding_programs = [
                FundingProgram(
                    program_id="NSF_THEORY",
                    name="NSF Standard Grant: Theoretical Foundations",
                    budget=45,
                    focus_areas=[
                        DIRECTIONS_DICT["complexity_theory"],
                        DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["logic_in_cs"],
                        DIRECTIONS_DICT["formal_methods"],
                        DIRECTIONS_DICT["discrete_mathematics"],
                        DIRECTIONS_DICT["geometric_algorithms"]
                    ],
                    min_score=5.5,  # Lower threshold for fundamental research
                    max_awards=5
                ),
                FundingProgram(
                    program_id="NSF_AI_FOUNDATIONS",
                    name="NSF Standard Grant: AI Foundations",
                    budget=50,
                    focus_areas=[
                        DIRECTIONS_DICT["knowledge_representation"],
                        DIRECTIONS_DICT["neural_symbolic_ai"],
                        DIRECTIONS_DICT["planning_and_scheduling"],
                        DIRECTIONS_DICT["algorithmic_game_theory"],
                        DIRECTIONS_DICT["interpretable_ml"],
                        DIRECTIONS_DICT["ai_ethics"]
                    ],
                    min_score=5.5,
                    max_awards=5
                ),
                FundingProgram(
                    program_id="NSF_SYSTEMS_THEORY",
                    name="NSF Standard Grant: Systems & Architecture",
                    budget=40,
                    focus_areas=[
                        DIRECTIONS_DICT["computer_architecture"],
                        DIRECTIONS_DICT["operating_systems"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["quantum_computing"],
                        DIRECTIONS_DICT["performance_analysis"],
                        DIRECTIONS_DICT["parallel_computing"]
                    ],
                    min_score=6.0,
                    max_awards=4
                )
            ]
        elif agency_name == "DARPA":
            # DARPA: Application-driven, shorter-term projects
            # Focus on deployment-ready systems, practical applications
            # More demanding (higher min_score, smaller budgets but focused)
            self.funding_programs = [
                FundingProgram(
                    program_id="DARPA_AUTONOMOUS",
                    name="DARPA Program: Autonomous Systems Deployment",
                    budget=50,
                    focus_areas=[
                        DIRECTIONS_DICT["robotics"],
                        DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["3d_vision"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["control_systems"],
                        DIRECTIONS_DICT["multiagent_systems"]
                    ],
                    min_score=7.5,  # Higher threshold for application-readiness
                    max_awards=3
                ),
                FundingProgram(
                    program_id="DARPA_SECURITY",
                    name="DARPA Program: Cybersecurity Applications",
                    budget=45,
                    focus_areas=[
                        DIRECTIONS_DICT["cybersecurity"],
                        DIRECTIONS_DICT["cryptography"],
                        DIRECTIONS_DICT["computer_networks"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["software_engineering"]
                    ],
                    min_score=7.5,
                    max_awards=3
                ),
                FundingProgram(
                    program_id="DARPA_AI_APPS",
                    name="DARPA Program: Applied AI Systems",
                    budget=55,
                    focus_areas=[
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["natural_language_processing"],
                        DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["dialogue_systems"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["human_computer_interaction"]
                    ],
                    min_score=7.0,
                    max_awards=4
                )
            ]
        else:
            # Default programs for other agencies
            self.funding_programs = [
                FundingProgram(
                    program_id=f"{agency_name}_GENERAL",
                    name=f"{agency_name} Standard Grant: General Computing",
                    budget=40,
                    focus_areas=[
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["software_engineering"]
                    ],
                    min_score=6.5,
                    max_awards=4
                )
            ]

    def evaluate_funding_applications(
        self,
        applications: List[Dict],
        llm: LLM
    ) -> List[Dict]:
        """Evaluate funding applications from university researchers

        Args:
            applications: List of funding application dicts with keys:
                - applicant_id: Agent ID
                - research_proposal: Research description
                - expertise: List of research areas
                - past_performance: Previous year's accepted papers count

        Returns:
            List of funding decisions
        """
        
        decisions = []

        for app in applications:
            # Find matching programs based on expertise
            # Compare ResearchDirection objects (app['expertise'] is a list of ResearchDirection objects)
            matching_programs = [
                prog for prog in self.funding_programs
                if any(expert_area in prog.focus_areas for expert_area in app.get('expertise', []))
                and prog.can_award()
            ]

            if not matching_programs:
                # No matching programs available
                decisions.append({
                    'applicant_id': app['applicant_id'],
                    'decision': 'reject',
                    'funding_amount': 0,
                    'program': None,
                    'rationale': 'No matching funding programs available'
                })
                continue

            # Use LLM to evaluate the application
            # Extract topic names from ResearchDirection objects for display
            expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                              for rd in app.get('expertise', [])]

            program_descriptions = []
            for p in matching_programs:
                focus_topics = [rd.topic for rd in p.focus_areas]
                program_descriptions.append(f"- {p.name}: Focus on {', '.join(focus_topics)}, Min Score: {p.min_score}")

            prompt = f"""You are evaluating a funding application for {self.id}.

Application Details:
- Applicant: {app['applicant_id']}
- Research Proposal: {app['research_proposal']}
- Expertise Areas: {', '.join(expertise_topics)}
- Past Performance: {app.get('past_performance', 0)} accepted papers last year

Available Programs:
{chr(10).join(program_descriptions)}

Evaluate this application on a scale of 1-10 considering:
1. Alignment with program focus areas
2. Research quality and innovation
3. Past performance and track record
4. Feasibility and impact potential

Return your evaluation in JSON format with "score" (number) and "rationale" (string) fields."""

            evaluation = json.loads(llm.generate(prompt=prompt))
            score = evaluation.get('score', 5.0)

            # Find best matching program that applicant qualifies for
            selected_program = None
            for prog in sorted(matching_programs, key=lambda p: p.budget, reverse=True):
                if score >= prog.min_score:
                    selected_program = prog
                    break

            if selected_program:
                # Award funding
                funding_amount = int(selected_program.budget / selected_program.max_awards)
                selected_program.awarded_this_cycle += 1
                selected_program.remaining_budget -= funding_amount

                decisions.append({
                    'applicant_id': app['applicant_id'],
                    'decision': 'fund',
                    'funding_amount': funding_amount,
                    'program': selected_program.name,
                    'rationale': evaluation['rationale']
                })
            else:
                decisions.append({
                    'applicant_id': app['applicant_id'],
                    'decision': 'reject',
                    'funding_amount': 0,
                    'program': None,
                    'rationale': f"Score {score:.1f} below minimum threshold"
                })

        return decisions

    def reset_programs_for_new_year(self):
        """Reset all funding programs for a new funding cycle"""
        for program in self.funding_programs:
            program.reset_for_new_cycle()

    def _make_funding_decision(self, evaluation: Dict, proposal: Dict) -> Dict:
        """Make funding decision based on evaluation"""
        score = evaluation.get('overall_score', 0)

        # Conservative funding thresholds
        if score >= 8.0:
            decision = "fund"
            amount = "high"
        elif score >= 6.5:
            decision = "fund"
            amount = "medium"
        elif score >= 5.5:
            decision = "conditional_fund"
            amount = "low"
        else:
            decision = "reject"
            amount = "none"

        return {
            'proposal_id': proposal.get('id', 'unknown'),
            'decision': decision,
            'funding_amount': amount,
            'evaluation_score': score,
            'reviewer_id': self.id,
            'rationale': evaluation.get('summary', 'No rationale provided')
        }

