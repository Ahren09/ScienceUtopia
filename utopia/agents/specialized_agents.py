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
        llm = None,
        generate_research_proposal: bool = True
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
        
        # Do we use LLM to generate research proposal?
        self.generate_research_proposal = generate_research_proposal

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
        available_fundings: str
    ) -> Dict:
        """Prepare a funding application for submission to funding agencies

        Args:
            past_performance: Number of accepted papers in previous year
            current_research_direction: Description of current research direction

        Returns:
            Dictionary with application details
        """
        # University researchers create detailed research proposals
        if self.generate_research_proposal:
            prompt = f"""You are a university researcher applying for a research funding.

## About Yourself

Your expertise lies in: {self.expertise}.
You have a past performance of {past_performance} accepted papers.
You have a current funding level of {self.resources}.

## List of Available Fundings

{available_fundings}

## Your Task
Generate a short research funding proposal (around 100 words) tailored for this funding application.

## Sample Response
```json
{{
    "research_direction": "theoretical_foundations",
    "research_proposal": "We propose to study the theoretical foundations of {current_research_direction}."
}}
```
"""
            research_proposal = self.llm.generate(prompt=prompt)
            current_research_direction = research_proposal.get('research_proposal', '')
        else:
            current_research_direction = current_research_direction
        
        
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
        focus_areas: List[ResearchDirection],
        funding_mode: str = "fixed_rate",  # "fixed_budget" or "fixed_rate"
        budget: int = 100,  # Total budget for fixed_budget mode
        funding_rate: float = 0.3  # Percentage of proposals to fund for fixed_rate mode
    ):
        self.program_id = program_id
        self.name = name
        self.focus_areas = focus_areas  # Research areas this program targets
        self.funding_mode = funding_mode
        
        if funding_mode == "fixed_budget":
            assert budget is not None, "budget is required for fixed_budget mode"
            self.budget = budget  # Total budget available (for fixed_budget mode)
        elif funding_mode == "fixed_rate":
            assert funding_rate is not None, "funding_rate is required for fixed_rate mode"
            self.funding_rate = funding_rate  # Percentage to fund (for fixed_rate mode)
        else:
            raise ValueError(f"Unknown funding mode: {funding_mode}")

    def reset_for_new_cycle(self):
        """Reset program state for a new funding cycle"""
        if self.funding_mode == "fixed_budget":
            self.remaining_budget = self.budget
        elif self.funding_mode == "fixed_rate":
            pass
        else:
            raise ValueError(f"Unknown funding mode: {self.funding_mode}")


    def calculate_funding_amount(self, num_winners: int) -> float:
        """Calculate equal funding amount per winner"""
        if self.funding_mode == "fixed_budget":
            return self.budget / max(num_winners, 1) if num_winners > 0 else 0
        else:  # fixed_rate mode
            return self.budget / max(num_winners, 1) if num_winners > 0 else 0


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
        
        
    def _initialize_funding_programs(self, agency_name: str):
        """Initialize funding programs based on agency type

        NSF: Foundational scientific advancement - focuses on theory, fundamental research,
             longer-term projects with lower scores acceptable (patient funding)
        DARPA: Application-driven, shorter-term projects - focuses on practical systems,
               deployment-ready technology with higher score thresholds (demanding funding)
        """
        from .research_direction import DIRECTIONS_DICT

        # Different agencies have different focus areas and funding modes
        if agency_name == "NSF":
            # NSF: Foundational scientific advancement - uses fixed budget mode
            self.funding_programs = [
                FundingProgram(
                    program_id="NSF_THEORY",
                    name="NSF Standard Grant: Theoretical Foundations",
                    focus_areas=[
                        DIRECTIONS_DICT["complexity_theory"],
                        DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["logic_in_cs"],
                        DIRECTIONS_DICT["formal_methods"],
                        DIRECTIONS_DICT["discrete_mathematics"],
                        DIRECTIONS_DICT["geometric_algorithms"]
                    ],
                    funding_mode="fixed_budget",
                    budget=100
                ),
                FundingProgram(
                    program_id="NSF_AI_FOUNDATIONS",
                    name="NSF Standard Grant: AI Foundations",
                    focus_areas=[
                        DIRECTIONS_DICT["knowledge_representation"],
                        DIRECTIONS_DICT["neural_symbolic_ai"],
                        DIRECTIONS_DICT["planning_and_scheduling"],
                        DIRECTIONS_DICT["algorithmic_game_theory"],
                        DIRECTIONS_DICT["interpretable_ml"],
                        DIRECTIONS_DICT["ai_ethics"]
                    ],
                    funding_mode="fixed_budget",
                    budget=120
                ),
                FundingProgram(
                    program_id="NSF_SYSTEM",
                    name="NSF Standard Grant: Systems & Architecture",
                    focus_areas=[
                        DIRECTIONS_DICT["computer_architecture"],
                        DIRECTIONS_DICT["operating_systems"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["quantum_computing"],
                        DIRECTIONS_DICT["performance_analysis"],
                        DIRECTIONS_DICT["parallel_computing"]
                    ],
                    funding_mode="fixed_budget",
                    budget=80
                )
            ]
        elif agency_name == "DARPA":
            # DARPA: Application-driven - uses fixed rate mode (more selective)
            self.funding_programs = [
                FundingProgram(
                    program_id="DARPA_AUTONOMOUS",
                    name="DARPA Program: Autonomous Systems Deployment",
                    focus_areas=[
                        DIRECTIONS_DICT["robotics"],
                        DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["3d_vision"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["control_systems"],
                        DIRECTIONS_DICT["multiagent_systems"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=0.2,  # Fund top 20%
                    budget=100
                ),
                FundingProgram(
                    program_id="DARPA_SECURITY",
                    name="DARPA Program: Cybersecurity Applications",
                    focus_areas=[
                        DIRECTIONS_DICT["cybersecurity"],
                        DIRECTIONS_DICT["cryptography"],
                        DIRECTIONS_DICT["computer_networks"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["software_engineering"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=0.25,  # Fund top 25%
                    budget=90
                ),
                FundingProgram(
                    program_id="DARPA_AI_APPS",
                    name="DARPA Program: Applied AI Systems",
                    focus_areas=[
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["natural_language_processing"],
                        DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["dialogue_systems"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["human_computer_interaction"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=0.3,  # Fund top 30%
                    budget=110
                )
            ]
        else:
            # Default programs for other agencies - mixed mode
            self.funding_programs = [
                FundingProgram(
                    program_id=f"{agency_name}_GENERAL",
                    name=f"{agency_name} Standard Grant: General Computing",
                    focus_areas=[
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["software_engineering"]
                    ],
                    funding_mode="fixed_budget",
                    budget=80
                )
            ]

    def evaluate_funding_applications(
        self,
        applications: List[Dict],
        llm: LLM
    ) -> List[Dict]:
        """Evaluate funding applications using simplified funding modes

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
        
        # Group applications by matching programs
        program_applications = {}
        for app in applications:
            # Find matching programs based on expertise
            matching_programs = [
                prog for prog in self.funding_programs
                if any(expert_area in prog.focus_areas for expert_area in app.get('expertise', []))
            ]
            
            if not matching_programs:
                decisions.append({
                    'applicant_id': app['applicant_id'],
                    'decision': 'reject',
                    'funding_amount': 0,
                    'program': None,
                    'rationale': 'No matching funding programs available'
                })
                continue
                
            # Use the first matching program (simplified)
            selected_program = matching_programs[0]
            if selected_program.program_id not in program_applications:
                program_applications[selected_program.program_id] = {
                    'program': selected_program,
                    'applications': []
                }
            program_applications[selected_program.program_id]['applications'].append(app)
        
        # Process each program's applications
        for program_id, data in program_applications.items():
            program = data['program']
            apps = data['applications']
            
            # Score all applications for this program
            scored_apps = []
            for app in apps:
                score = self._score_application(app, program, llm)
                scored_apps.append((app, score))
            
            # Sort by score (highest first)
            scored_apps.sort(key=lambda x: x[1], reverse=True)
            
            # Determine winners based on funding mode
            if program.funding_mode == "fixed_rate":
                num_winners = max(1, int(len(scored_apps) * program.funding_rate))
            else:  # fixed_budget mode
                # Fund all proposals until budget runs out
                num_winners = min(len(scored_apps), len([app for app, score in scored_apps if program.remaining_budget > 0]))
            
            # Calculate equal funding amount
            if num_winners > 0:
                funding_per_winner = program.calculate_funding_amount(num_winners)
                
                # Award funding to winners
                for i, (app, score) in enumerate(scored_apps):
                    if i < num_winners:
                        decisions.append({
                            'applicant_id': app['applicant_id'],
                            'decision': 'fund',
                            'funding_amount': funding_per_winner,
                            'program': program.name,
                            'rationale': f"Score {score:.1f} - Selected for funding"
                        })
                        program.remaining_budget -= funding_per_winner
                    else:
                        decisions.append({
                            'applicant_id': app['applicant_id'],
                            'decision': 'reject',
                            'funding_amount': 0,
                            'program': program.name,
                            'rationale': f"Score {score:.1f} - Not selected for funding"
                        })
            else:
                # No winners
                for app, score in scored_apps:
                    decisions.append({
                        'applicant_id': app['applicant_id'],
                        'decision': 'reject',
                        'funding_amount': 0,
                        'program': program.name,
                        'rationale': f"Score {score:.1f} - No funding available"
                    })

        return decisions

    def _score_application(self, app: Dict, program: 'FundingProgram', llm: LLM) -> float:
        """Score a single application for a specific program"""
        # Extract topic names from ResearchDirection objects for display
        expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                          for rd in app.get('expertise', [])]
        
        focus_topics = [rd.topic for rd in program.focus_areas]
        
        prompt = f"""You are evaluating a funding application for {program.name}.

Application Details:
- Applicant: {app['applicant_id']}
- Research Proposal: {app['research_proposal']}
- Expertise Areas: {', '.join(expertise_topics)}
- Past Performance: {app.get('past_performance', 0)} accepted papers last year

Program Focus: {', '.join(focus_topics)}

Evaluate this application on a scale of 1-10 considering:
1. Alignment with program focus areas
2. Research quality and innovation
3. Past performance and track record
4. Feasibility and impact potential

Return your evaluation in JSON format with "score" (number) and "rationale" (string) fields."""

        try:
            evaluation = json.loads(llm.generate(prompt=prompt))
            return evaluation.get('score', 5.0)
        except:
            # Fallback scoring if LLM fails
            return 5.0

    def reset_programs_for_new_year(self):
        """Reset all funding programs for a new funding cycle"""
        for program in self.funding_programs:
            program.reset_for_new_cycle()


