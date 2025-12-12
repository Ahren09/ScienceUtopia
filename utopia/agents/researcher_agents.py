"""
Specialized Agent Types for Multi-Agent Scientific Ecosystem

EXTENDS base agent classes WITHOUT modifying existing AI-Scientist code
"""

from typing import List, Dict, Optional, Set, Tuple

from pydantic import BaseModel, RootModel

from utopia.config import SIMULATION_CONFIG
from utopia.models.models import BaseLLM
from .base_agent import SimulationAgent
from .funding_agents import FundingProgram
from .research_direction import ResearchDirection


class RelevantProject(BaseModel):
    arxiv_id: str
    status: str


class ProgramApplication(BaseModel):
    research_proposal: Optional[str] = None
    submit: bool
    relevant_projects: List[RelevantProject]


class FundingApplicationResponse(RootModel[Dict[str, ProgramApplication]]):
    pass


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
            llm: BaseLLM = None,
            generate_research_proposal: bool = True,
            max_retries: int = 10,
            exploration_strategy: str = "balanced"  # NEW: "explorer", "exploiter", "cautious_explorer", "balanced"
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
            agent_id=researcher_name,
            llm=llm,
            max_retries=max_retries
        )
        self.university_name = university_name
        self.can_author = True  # University researchers can write papers
        self.can_review = True  # University researchers can review papers
        self.resources = funding_level
        # Do we use LLM to generate research proposal?
        self.generate_research_proposal = generate_research_proposal
        self.newest_direction = None
        # NEW: Exploration strategy for exploration vs exploitation experiment
        self.exploration_strategy = exploration_strategy

        assert isinstance(expertise, List) and all(isinstance(rd, ResearchDirection) for rd in
                                                   expertise), "expertise must be a list of ResearchDirection objects"
        assert len(expertise) > 0, "expertise must be a non-empty list"
        # Inherent expertise (3 research directions)
        self.expertise = expertise

        self.funding_success_history = {}  # program_id -> successes
        self.conflict_of_interest = set()  # set of authors they have collaborated with or belong to the same institution

    def get_type(self) -> str:
        return "university"

    def _get_personality_prompt(self, include_exploration_strategy: bool = False, include_funding_situation: bool = False) -> str:
        prompt = f"""You are a university researcher at {self.university_name}. 
You author papers and review them for conferences. 
Compared with industry researchers, you focus more on theoretical contributions, novel methodologies, and advancing fundamental knowledge. 
"""
        if include_exploration_strategy:
            prompt += self._get_exploration_strategy_prompt()
                        
        if include_funding_situation:
            prompt += f"""
Your current funding level is {self.resources}.
"""

        return prompt
    
               
    

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

            prompt = f"""You are a university researcher from {self.university_name} applying for research funding based on your expertise and past papers.

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
Abstract: {paper['abstract']}
Status: {paper['status']}

"""

            prompt += "## Past Rejected Papers"
            for paper in past_rejected_papers:
                prompt += f"""Title: {paper['title']}
Arxiv ID: \"{paper['id']}\"
Abstract: {paper['abstract']}
Status: {paper['status']}

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
                "status": "accept"
            },
            {
                "arxiv_id": "https://arxiv.org/abs/2406.00001v1",
                "status": "reject"
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
            proposal, message_history = self.llm.generate(prompt=prompt)

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

    def get_funding_application_prompt(
            self,
            past_accepted_papers: List[Dict],
            past_rejected_papers: List[Dict],
            programs: List['FundingProgram'],
            selected_program_ids: List[str],
            success_history: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[str], Optional[Dict], Optional[Dict]]:
        """Get the funding application prompt without making an LLM call

        Returns:
            Tuple of (prompt, response_format, metadata) where metadata contains info for post-processing
        """
        if not selected_program_ids:
            return None, None, None

        # Get selected programs info
        selected_programs = [programs[program_id] for program_id in selected_program_ids]

        if not self.generate_research_proposal:
            raise NotImplementedError("This method is not supported for generate_research_proposal == False")

        expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                            for rd in self.expertise]

        program_details = ""
        for program in selected_programs:
            program_details += f"""### Program ID: \"{program.program_id}\"
Name: \"{program.name}\"
Focus Areas: {', '.join([rd.topic for rd in program.research_directions])}

"""

        prompt = f"""You are a university researcher from {self.university_name} applying for research funding based on your expertise and past papers.

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
Abstract: {paper['abstract']}
Status: {paper['status']}

"""

        prompt += "## Past Rejected Papers"
        for paper in past_rejected_papers:
            prompt += f"""Title: {paper['title']}
Arxiv ID: \"{paper['id']}\"
Abstract: {paper['abstract']}
Status: {paper['status']}

"""

        prompt += """## Sample Response
```json
{
    "DARPA_AUTONOMOUS": {  // <- this is the program_id
        "research_proposal": "...",  // <- If you decide not to submit, set to your research proposal
        "submit": true,  // <- If you decide to submit, set to true
        "relevant_projects": [  //<- If you decide to submit, set to a list of your past accepted papers
            {
                "arxiv_id": "https://arxiv.org/abs/2406.12708v2", <- must be a full URL
                "status": "accept"
            },
            {
                "arxiv_id": "https://arxiv.org/abs/2406.00001v1",
                "status": "reject"
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
        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'funding_application',
                'strict': True,
                'schema': FundingApplicationResponse.model_json_schema()
            },
        }

        # Metadata for post-processing validation
        all_paper_ids = set(paper['id'] for paper in past_accepted_papers + past_rejected_papers)
        metadata = {
            'agent': self,
            'all_paper_ids': all_paper_ids,
            'selected_program_ids': selected_program_ids
        }

        return prompt, response_format, metadata


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
            llm: BaseLLM = None,
            funding_mode: str = "performance",  # "performance" or "consistent"
            max_retries: int = 10
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
            agent_id=researcher_name,
            llm=llm,
            max_retries=max_retries
        )
        self.resources = funding_level
        assert isinstance(expertise, List) and all(isinstance(rd, ResearchDirection) for rd in
                                                   expertise), "expertise must be a list of ResearchDirection objects"
        assert len(expertise) > 0, "expertise must be a non-empty list"
        self.expertise = expertise
        self.company_name = company_name
        self.can_author = True
        self.can_review = True
        self.funding_mode = funding_mode
        self.funding_success_history = {}
        self.budget = SIMULATION_CONFIG['funding']['industry_base_budget']
        self.research_priorities: Set[str] = set()

        self.newest_direction = None
        if expertise:
            self.research_priorities = {rd.topic if hasattr(rd, 'topic') else str(rd) for rd in expertise}

        self.conflict_of_interest = set()  # set of authors they have collaborated with or belong to the same institution

    def get_type(self) -> str:
        return "industry"

    def _get_personality_prompt(self, include_exploration_strategy: bool = False, include_funding_situation: bool = False) -> str:
        prompt = f"""You are an industry researcher at {self.company_name}. 
You author papers and review them for conferences. 
Compared with university researchers, you focus more on practical applications, scalable solutions, and commercially viable innovations. 
"""
        if include_exploration_strategy:
            prompt += self._get_exploration_strategy_prompt()
            
        if include_funding_situation:
            prompt += f"""
Your current funding level is {self.resources}.
"""
        return prompt

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

        result, message_history = self.llm.generate(prompt=prompt)
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

    def update_research_direction(self, direction: 'ResearchDirection'):
        """Update company research priorities via LLM"""
        prompt = f"""You are {self.company_name}. Based on market trends and company strategy,
propose a research direction for next year. Current priorities: {self.research_priorities}

Return JSON with format:
{{"topic": "topic_name", "method": "method_name", "application": "application_area"}}"""

        result, message_history = self.llm.generate(prompt=prompt) if self.llm else {}
        new_dir = ResearchDirection(
            topic=result.get('topic', 'AI'),
            method=result.get('method', 'deep learning'),
            application=result.get('application', 'general')
        )
        self.research_priorities.add(new_dir.topic)
        return new_dir

    def get_funding_allocation(self, papers_accepted: int) -> int:
        """Legacy method for backward compatibility"""
        if self.funding_mode == "performance":
            return 30 * papers_accepted if papers_accepted > 0 else -10
        elif self.funding_mode == "consistent":
            if self.resources < 80:
                return 20
            elif self.resources > 90:
                return -10
            return 5
        raise ValueError(f"Unknown funding mode: {self.funding_mode}")


# Archived class. Not used in the simulation.
class FreelancerAgent(SimulationAgent):
    """Freelancer agents primarily focus on authoring but may occasionally review"""

    def __init__(
            self,
            researcher_name: str,
            funding_level: int = 30,
            agent_id: str = None,
            expertise_directions: Optional[List] = None,
            llm: BaseLLM = None
    ):

        assert agent_id is not None, "agent_id is required"
        super().__init__(
            reputation=4,
            funding_level=funding_level,
            agent_id=researcher_name,
            llm=llm
        )

        self.can_author = True  # Freelancers primarily write papers
        self.can_review = True  # Freelancers can review, but less frequently invited

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
