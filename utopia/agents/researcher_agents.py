"""
Specialized Agent Types for Multi-Agent Scientific Ecosystem

EXTENDS base agent classes WITHOUT modifying existing AI-Scientist code
"""

from typing import List, Dict, Optional, Set, Tuple

from pydantic import BaseModel

from utopia.config import SIMULATION_CONFIG
from utopia.models.models import BaseLLM
from .base_agent import SimulationAgent
from .funding_agents import FundingProgram
from .research_direction import ResearchDirection


class ProgramApplication(BaseModel):
    research_proposal: str
    submit: bool
    relevant_projects: List[int]



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
            funding_programs: Dict[str, 'FundingProgram'],
            past_accepted_papers: List[Dict],
            past_rejected_papers: List[Dict],
    ) -> List[str]:
        """Select funding programs based on paper topic overlap and expertise match.

        Returns:
            List of program_ids the agent should apply to.
        """
        # Collect topics from past papers
        paper_topics_lower = set()
        for paper in past_accepted_papers + past_rejected_papers:
            paper_topics_lower.update(t.lower() for t in paper.get('topics', []))

        expertise_topics = {rd.topic for rd in self.expertise}

        scores = []
        for program_id, program in funding_programs.items():
            # Paper topic overlap score
            if program.topics and paper_topics_lower:
                overlap = sum(1 for t in program.topics if t.lower() in paper_topics_lower)
                topic_score = overlap / len(program.topics)
            else:
                topic_score = 0

            # Expertise overlap score (research direction matching)
            program_dir_topics = {rd.topic for rd in program.research_directions}
            expertise_overlap = len(expertise_topics & program_dir_topics) / max(len(program_dir_topics), 1)

            # Success history bonus
            success_rate = 0.5  # neutral default
            if self.funding_success_history and program_id in self.funding_success_history:
                success_rate = min(1.0, len(self.funding_success_history[program_id]) * 0.2)

            final_score = 0.4 * topic_score + 0.4 * expertise_overlap + 0.2 * success_rate
            scores.append((program_id, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        # Return programs above threshold, at least 1
        selected = [pid for pid, score in scores if score > 0.1]
        return selected if selected else [scores[0][0]] if scores else []

    def funding_application_prompt(
            self,
            program: 'FundingProgram',
            past_accepted_papers: List[Dict],
            past_rejected_papers: List[Dict],
    ) -> Tuple[Optional[str], Optional[Dict], Optional[Dict]]:
        """Get a funding application prompt for a single program.

        Returns:
            Tuple of (prompt, response_format, metadata) where metadata contains info for post-processing
        """
        if not self.generate_research_proposal:
            raise NotImplementedError("This method is not supported for generate_research_proposal == False")

        expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                            for rd in self.expertise]

        prompt = f"""You are a university researcher from {self.university_name} applying for research funding based on your expertise and past papers.

## Your Profile
- Expertise: {', '.join(expertise_topics)}
- Recent performance: {len(past_accepted_papers)} accepted papers, {len(past_rejected_papers)} rejected papers
- Current funding: {self.resources}

## Funding Program
Program ID: "{program.program_id}"
Name: "{program.name}"
Focus Areas: {', '.join([rd.topic for rd in program.research_directions])}

If you think this program is a good fit for your expertise, generate a compelling 100-word proposal and mention your past works that are relevant to the program.

The research proposal should:
1. Align with program focus areas.
2. Leverage your expertise and past works, either accepted papers or preprints / rejected papers
3. Emphasize feasibility and impact.

## Important Notes
- Applying to a funding is time-consuming. You should only apply to programs that are a good match to your papers and that you are confident about. 

"""

        # Build ordered list of all papers: accepted first, then rejected
        all_papers = []
        paper_idx = 0

        if past_accepted_papers:
            prompt += "## Past Accepted Papers\n"
            for paper in past_accepted_papers:
                prompt += f"""### Paper {paper_idx}
Title: {paper['title']}
Abstract: {paper['abstract']}

"""
                all_papers.append({'id': paper['id'], 'status': paper['status']})
                paper_idx += 1

        if past_rejected_papers:
            prompt += "## Past Rejected Papers\n"
            for paper in past_rejected_papers:
                prompt += f"""### Paper {paper_idx}
Title: {paper['title']}
Abstract: {paper['abstract']}

"""
                all_papers.append({'id': paper['id'], 'status': paper['status']})
                paper_idx += 1

        num_papers = len(all_papers)
        citation_prompt = f"A ranked list of paper indices (most relevant first). Each value must be an integer between 0 and {num_papers - 1}."

        prompt += f"""## Response Format
Respond in JSON format. Set "submit" to true with a proposal and relevant projects to apply, or false with null proposal and empty list to skip.

If you choose to submit, An example response (DO NOT directly copy the exact response):
```json
{{
    "submit": true,
    "research_proposal": "Your 100-word proposal here.",
    "relevant_projects": [0, 2, 1] <- {citation_prompt}
}}
```
If you choose not to submit, respond with the following JSON format:
```json
{{
    "submit": false,
    "research_proposal": "",
    "relevant_projects": []
}}

```

"""

        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'funding_application',
                'strict': True,
                'schema': ProgramApplication.model_json_schema()
            },
        }

        metadata = {
            'agent': self,
            'program_id': program.program_id,
            'all_papers': all_papers,
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
            max_retries: int = 10,
            exploration_strategy: str = "balanced"
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
        self.exploration_strategy = exploration_strategy

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
        for program_id, program in funding_programs.items():
            # Industry prefers applied programs (DARPA over NSF)
            overlap = sum(1 for exp in self.expertise if exp in program.research_directions)
            expertise_score = overlap / max(len(self.expertise), 1)

            # Boost DARPA programs
            practical_boost = 0.2 if "DARPA" in program.program_id else 0
            success_rate = success_history[program.program_id] if success_history and program.program_id in success_history else 0.5

            final_score = 0.6 * expertise_score + 0.2 * success_rate + practical_boost
            scores.append((program.program_id, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, score in scores[:2] if score > 0.3]

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
            agent_id=researcher_name,
            llm=llm
        )
        self.resources = funding_level

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
