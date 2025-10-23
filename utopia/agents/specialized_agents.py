"""
Specialized Agent Types for Multi-Agent Scientific Ecosystem

EXTENDS base agent classes WITHOUT modifying existing AI-Scientist code
"""

import json
import random
import traceback
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
        self.funding_success_history = {}
        
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

    def select_funding_programs(
        self,
        funding_programs: List['FundingProgram'],
        success_history: Optional[Dict[str, float]] = None,
        num_submitted_proposals: int = 3
    ) -> List[str]:
        """Select funding programs to apply to based on expertise overlap and success rates

        Args:
            funding_programs: List of available FundingProgram objects
            success_history: Dict mapping program_id -> historical success rate (0-1)

        Returns:
            List of selected program IDs
        """
        
        if not self.expertise:
            return []

        scores = []
        for program_id, program in funding_programs.items():
            # Calculate expertise overlap
            # focus_areas: List[str] = [area.topic for area in program.focus_areas]
            
            overlap = sum(1 for exp in self.expertise if exp in program.research_directions)
            scores.append((program.program_id, overlap))

        # Select top 2-3 programs with score > 0.3
        scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, score in scores[:num_submitted_proposals] if score >= 1]

    def prepare_funding_application(
        self,
        past_accepted_papers: List[Dict],
        past_rejected_papers: List[Dict],
        programs: List['FundingProgram'],
        selected_program_ids: List[str],
        success_history: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Prepare a funding application for submission to funding agencies

        Args:
            past_accepted_papers: accepted papers in previous years
            past_rejected_papers: rejected papers in previous years
            funding_programs: List of all available FundingProgram objects
            selected_program_ids: List of program IDs to apply to
            success_history: Historical success rates per program

        Returns:
            Dictionary with application details
        """
        if not selected_program_ids:
            return None

        # Get selected programs info
        selected_programs = [programs[program_id] for program_id in selected_program_ids]
        # for program_id, program in programs.items():
        #     if program_id in selected_program_ids:
        #         selected_programs.append(program)

        # Generate targeted proposal using LLM
        if self.generate_research_proposal:
            expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                              for rd in self.expertise]
            
            program_details = ""
            
            for program in selected_programs:
                program_details += f"""### Program ID: \"{program.program_id}\"
Name: \"{program.name}\"
Focus Areas: {', '.join([rd.topic for rd in program.research_directions])}

"""

            prompt = f"""You are a university researcher applying for research funding.

## Your Profile
- Expertise: {', '.join(expertise_topics)}
- Recent performance: {len(past_accepted_papers)} accepted papers, {len(past_rejected_papers)} rejected papers
- Current funding: {self.resources}

## Candidate Programs
{program_details}

For each program, if you think the program is a good fit for your expertise, generate a compelling 100-word proposal and mention your past works that are relevant to the program. 

The research proposal should be:
1. Aligns with program focus areas.
2. Leverages your expertise and past works, either accepted papers or preprints / rejected papers
3. Emphasizes feasibility and impact.

"""

            prompt += "## Past Accepted Papers"
            for paper in past_accepted_papers:
                prompt += f"""Title: {paper['title']}
Arxiv ID: {paper['id']}
Abstract: {paper['content']}
Status: {paper['decision']}
                
"""

            prompt += "## Past Rejected Papers"
            for paper in past_rejected_papers:
                prompt += f"""Title: {paper['title']}
Arxiv ID: \"{paper['id']}\"
Abstract: {paper['content']}
Status: {paper['decision']}

"""

## Past Accepted Papers
            prompt += """## Sample Response
```json
{
    "DARPA_AUTONOMOUS": {  // <- this is the program_id
        "research_proposal": "...",  // <- If you decide not to submit, set to your research proposal
        "submit": true,  // <- If you decide to submit, set to true
        "relevant_projects": [  //<- If you decide to submit, set to a list of your past accepted papers
            {
                "arxiv_id": "https://arxiv.org/abs/2406.12708v2", <- must be a full URL
                "status": "accepted"
            },
            {
                "arxiv_id": "https://arxiv.org/abs/2406.00001v1",
                "status": "rejected"
            },
        ], 
    },
    "DARPA_AI_APPS": {
        "research_proposal": null, <- If you decide not to submit, set to null
        "submit": false, // <- If you decide not to submit, set to false
        "relevant_projects": [] <- If you decide not to submit, set to an empty list
    }
}
```
"""
            result = self.llm.generate(prompt=prompt)
            proposal = json.loads(result)
            
            # Check if the referred papers are from the current author
            all_paper_ids_of_current_author = set(paper['id'] for paper in past_accepted_papers + past_rejected_papers)
            all_paper_ids_in_proposals = set()
            for program_id, program in proposal.items():
                if program.get('submit', True):
                    referred_papers = program['relevant_projects']
                    all_paper_ids_in_proposals.update([paper['arxiv_id'] for paper in referred_papers])
            
            assert all_paper_ids_in_proposals <= all_paper_ids_of_current_author
            
        else:
            raise NotImplementedError("This method is not supported for generate_research_proposal == False")

        return proposal


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
        self.funding_success_history = {}

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

    def select_funding_programs(
        self,
        funding_programs: List['FundingProgram'],
        success_history: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """Select funding programs based on practical application focus and success rates"""
        if not self.expertise:
            return []

        scores = []
        for program in funding_programs:
            # Industry prefers applied programs (DARPA over NSF)
            overlap = sum(1 for exp in self.expertise if exp in program.focus_areas)
            expertise_score = overlap / max(len(self.expertise), 1)

            # Boost DARPA programs
            practical_boost = 0.2 if "DARPA" in program.program_id else 0
            success_rate = success_history.get(program.program_id, 0.5) if success_history else 0.5

            final_score = 0.6 * expertise_score + 0.2 * success_rate + practical_boost
            scores.append((program.program_id, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, score in scores[:2] if score > 0.3]

    def prepare_funding_application(
        self,
        past_performance: int,
        funding_programs: List['FundingProgram'],
        selected_program_ids: List[str]
    ) -> Dict:
        """Prepare industry funding application focused on practical impact"""
        if not selected_program_ids:
            return None

        selected_programs = [p for p in funding_programs if p.program_id in selected_program_ids]
        focus_topics = list(set([rd.topic for p in selected_programs for rd in p.focus_areas]))

        expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                          for rd in self.expertise]

        prompt = f"""You are an industry researcher applying for research funding.

## Your Profile
- Company: {self.company_name}
- Expertise: {', '.join(expertise_topics)}
- Recent performance: {past_performance} papers
- Focus: Practical applications, scalability, commercial impact

## Target Focus Areas
{', '.join(focus_topics)}

Generate a 100-word proposal emphasizing:
1. Real-world applications and deployment
2. Scalability and efficiency
3. Commercial viability

Return JSON with "research_proposal" field only."""

        result = self.llm.generate(prompt=prompt)
        proposal = result.get('research_proposal', 'Industry research proposal.')

        return {
            'applicant_id': self.id,
            'affiliation': self.company_name,
            'research_proposal': proposal,
            'expertise': self.expertise if self.expertise else [],
            'past_performance': past_performance,
            'current_funding': self.resources,
            'applicant_type': 'industry',
            'target_programs': selected_program_ids
        }

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
        research_directions: List[ResearchDirection],
        funding_mode: str = "fixed_rate",  # "fixed_budget" or "fixed_rate"
        budget: int = 100,  # Total budget for fixed_budget mode
        funding_rate: float = 0.3  # Percentage of proposals to fund for fixed_rate mode
    ):
        self.program_id = program_id
        self.name = name
        self.research_directions = research_directions  # Research areas this program targets
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
        self.funding_priorities = []
        self.risk_aversion = "moderate"
        self.can_author = False  # Funding agents typically don't write research papers
        self.can_review = False

        # Initialize funding programs
        self.funding_programs: Dict[FundingProgram] = {}
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
            self.funding_programs = {
                "NSF_THEORY": FundingProgram(
                    program_id="NSF_THEORY",
                    name="NSF Standard Grant: Theoretical Foundations",
                    research_directions=[
                        # Core theoretical CS (sorted by relevance)
                        DIRECTIONS_DICT["complexity_theory"],
                        DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["logic_in_cs"],
                        DIRECTIONS_DICT["formal_methods"],
                        DIRECTIONS_DICT["discrete_mathematics"],
                        DIRECTIONS_DICT["geometric_algorithms"],
                        DIRECTIONS_DICT["computational_topology"],
                        DIRECTIONS_DICT["data_structures"],
                        DIRECTIONS_DICT["symbolic_computation"],
                        DIRECTIONS_DICT["information_theory"]
                    ],
                    funding_mode="fixed_rate"
                ),
                "NSF_AI_FOUNDATIONS": FundingProgram(
                    program_id="NSF_AI_FOUNDATIONS",
                    name="NSF Standard Grant: AI Foundations",
                    research_directions=[
                        # AI theory and foundations (sorted by relevance)
                        DIRECTIONS_DICT["knowledge_representation"],
                        DIRECTIONS_DICT["neural_symbolic_ai"],
                        DIRECTIONS_DICT["planning_and_scheduling"],
                        DIRECTIONS_DICT["interpretable_ml"],
                        DIRECTIONS_DICT["algorithmic_game_theory"],
                        DIRECTIONS_DICT["multiagent_systems"],
                        DIRECTIONS_DICT["meta-learning"],
                        DIRECTIONS_DICT["ai_ethics"],
                        DIRECTIONS_DICT["evolutionary_computation"],
                        DIRECTIONS_DICT["mathematical_software"]
                    ],
                    funding_mode="fixed_rate",
                ),
                "NSF_SYSTEM": FundingProgram(
                    program_id="NSF_SYSTEM",
                    name="NSF Standard Grant: Systems & Architecture",
                    research_directions=[
                        # Systems and architecture (sorted by relevance)
                        DIRECTIONS_DICT["computer_architecture"],
                        DIRECTIONS_DICT["operating_systems"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["parallel_computing"],
                        DIRECTIONS_DICT["quantum_computing"],
                        DIRECTIONS_DICT["performance_analysis"],
                        DIRECTIONS_DICT["database_systems"],
                        DIRECTIONS_DICT["computer_networks"],
                        DIRECTIONS_DICT["programming_languages"],
                        DIRECTIONS_DICT["software_engineering"]
                    ],
                    funding_mode="fixed_rate",
                )
            }
        elif agency_name == "DARPA":
            # DARPA: Application-driven - uses fixed rate mode (more selective)
            self.funding_programs = {
                "DARPA_AUTONOMOUS": FundingProgram(
                    program_id="DARPA_AUTONOMOUS",
                    name="DARPA Program: Autonomous Systems Deployment",
                    research_directions=[
                        # Autonomous systems and robotics (sorted by relevance)
                        DIRECTIONS_DICT["robotics"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["control_systems"],
                        DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["3d_vision"],
                        DIRECTIONS_DICT["video_understanding"],
                        DIRECTIONS_DICT["multiagent_systems"],
                        DIRECTIONS_DICT["planning_and_scheduling"],
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["performance_analysis"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=0.2,  # Fund top 20%
                ),
                "DARPA_SECURITY": FundingProgram(
                    program_id="DARPA_SECURITY",
                    name="DARPA Program: Cybersecurity Applications",
                    research_directions=[
                        # Cybersecurity and secure systems (sorted by relevance)
                        DIRECTIONS_DICT["cybersecurity"],
                        DIRECTIONS_DICT["cryptography"],
                        DIRECTIONS_DICT["computer_networks"],
                        DIRECTIONS_DICT["formal_methods"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["software_engineering"],
                        DIRECTIONS_DICT["operating_systems"],
                        DIRECTIONS_DICT["database_systems"],
                        DIRECTIONS_DICT["programming_languages"],
                        DIRECTIONS_DICT["performance_analysis"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=0.25,  # Fund top 25%
                ),
                "DARPA_AI_APPS": FundingProgram(
                    program_id="DARPA_AI_APPS",
                    name="DARPA Program: Applied AI Systems",
                    research_directions=[
                        # Applied AI and deployment (sorted by relevance)
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["natural_language_processing"],
                        DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["dialogue_systems"],
                        DIRECTIONS_DICT["human_computer_interaction"],
                        DIRECTIONS_DICT["generative_models"],
                        DIRECTIONS_DICT["interpretable_ml"],
                        DIRECTIONS_DICT["multiagent_systems"],
                        DIRECTIONS_DICT["computer_graphics"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=0.3,  # Fund top 30%
                )
            }
        else:
            # Default programs for other agencies - mixed mode
            self.funding_programs = {
                FundingProgram(
                    program_id=f"{agency_name}_GENERAL",
                    name=f"{agency_name} Standard Grant: General Computing",
                    research_directions=[
                        # Broad coverage of CS areas (sorted by popularity)
                        DIRECTIONS_DICT["deep_learning"],
                        DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["software_engineering"],
                        DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["natural_language_processing"],
                        DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["computer_networks"],
                        DIRECTIONS_DICT["database_systems"],
                        DIRECTIONS_DICT["human_computer_interaction"],
                        DIRECTIONS_DICT["image_recognition"]
                    ],
                    funding_mode="fixed_rate",
                )
            }

    def evaluate_funding_applications(
        self,
        applications: List[Dict],
        submitted_papers_dict: Dict[str, Dict]
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
        
        # Group applications by program
        program_to_applications = {}
        
        # Populate program_to_applications from the applications format
        for author_applications in applications:
            author = author_applications['author']
            
            # Each author_applications dict contains program_id -> application_data
            for program_id, application_data in author_applications.items():
                if program_id == 'author' or not application_data['submit']:
                    continue
                    
                if program_id not in program_to_applications:
                    program_to_applications[program_id] = []
                
                # Add author info to the application
                application_data['author'] = author
                application_data['applicant_id'] = author.id
                program_to_applications[program_id].append(application_data)
        
        # Process each program's applications using LLM to decide winners
        for program_id, apps in program_to_applications.items():
            # Find the corresponding program object
            program = self.funding_programs[program_id]
            
            # Calculate number of winners based on funding mode
            if program.funding_mode == "fixed_rate":
                num_winners = min(1, round(len(apps) * program.funding_rate))
            
            else:  # fixed_budget mode
                raise NotImplementedError("Fixed budget mode is not supported for evaluation")
                # For fixed budget, we'll let LLM decide but cap at reasonable number
                num_winners = min(len(apps), max(1, len(apps) // 2))
            
            # Prepare applications for LLM evaluation
            applications_text = ""
            for i, app in enumerate(apps):
                author = app.get('author')
                expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                                  for rd in (author.expertise if author and hasattr(author, 'expertise') else [])]
                
                accepted_papers = [paper for paper in app['relevant_projects'] if paper['status'] == 'accepted']
                rejected_papers = [paper for paper in app['relevant_projects'] if paper['status'] == 'rejected']
                past_performance = f"""{len(accepted_papers)} accepted papers, {len(rejected_papers)} rejected papers.
"""
                
                if len(accepted_papers) > 0:
                    past_performance += "- Recent accepted papers:\n"
                    for paper in accepted_papers:
                        past_performance += f" - {submitted_papers_dict[paper['arxiv_id']]['title']}\n"
                
                if len(rejected_papers) > 0:
                    past_performance += "- Recent rejected papers:\n"
                    for paper in rejected_papers:
                        past_performance += f" - {submitted_papers_dict[paper['arxiv_id']]['title']}\n"
                applications_text += f"""
### Application {i+1}

Applicant ID: {app['applicant_id']}
Research Proposal: {app.get('research_proposal', 'No proposal provided')}
Expertise: {', '.join(expertise_topics)}
Affiliation: {getattr(author, 'university_name', getattr(author, 'company_name', 'Unknown')) if author else 'Unknown'}
Past Performance:
{past_performance}

"""
            
            # Create LLM prompt for funding decisions
            focus_topics = [rd.topic for rd in program.research_directions]
            
            sample_response = """
```json
{
    "winners": [{
        "id": "University_Researcher_888", <- Must be EXACTLY the same as the applicant_id in the application
        "reason": "Reason for selection" <- a 1-sentence brief explanation for why this applicant is selected
    },
    {
        "id": "University_Researcher_404",
        "reason": "Reason for selection"
    },
    ...
    ]
}
```
"""
            prompt = f"""You are a funding agency evaluating applications for {program.name}.

## Program Details
Program Focus Areas: {', '.join(focus_topics)}
You MUST select EXACTLY {num_winners} winner(s) from the {len(apps)} applications.


## Selection Criteria
Evaluate all applications and select the top {num_winners} winners, from the most preferred to the least preferred. 
1. Alignment with program focus areas
2. Research quality and innovation potential
3. Past performance and track record. Focus more on recent works and accepted papers. Focus less on older works and rejected papers.
4. Feasibility and impact potential

## Applications to Evaluate
{applications_text}

## Sample Response
{sample_response}
"""

            try:
                llm_response = self.llm.generate(prompt=prompt)
                result = json.loads(llm_response)
                winners = result.get('winners', [])
                assert len(winners) == num_winners
                # Ensure we don't exceed the number of applications
                winners = winners[:len(apps)]
                
            except Exception as e:
                traceback.print_exc()
                # Fallback: select first N applications
        
        return winners

    def _score_application(self, app: Dict, program: 'FundingProgram', llm: LLM) -> float:
        """Score a single application for a specific program"""
        # Extract topic names from ResearchDirection objects for display
        author = app.get('author')
        expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                          for rd in (author.expertise if author and hasattr(author, 'expertise') else [])]
        
        focus_topics = [rd.topic for rd in program.research_directions]
        
        # Get research proposal from the application data
        research_proposal = app.get('research_proposal', 'No proposal provided')
        
        prompt = f"""You are evaluating a funding application for {program.name}.

Application Details:
- Applicant: {app['applicant_id']}
- Research Proposal: {research_proposal}
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
        for program_id, program in self.funding_programs.items():
            program.reset_for_new_cycle()


