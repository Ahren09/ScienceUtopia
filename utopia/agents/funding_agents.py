"""Funding agents and systems for academic and industry funding"""
import logging
import random
import traceback
from collections import Counter
from typing import Dict, Set, List

from utopia.config import SIMULATION_CONFIG
from utopia.models.models import LLM
from .base_agent import SimulationAgent
from .research_direction import ResearchDirection

logger = logging.getLogger(__name__)


class IndustryFundingSystem:
    """Manages industry self-funding through performance and topic alignment"""

    def __init__(self, base_budget: int = 20, acceptance_bonus: int = 20, rejection_penalty: int = 10,
                 hot_window_years: int = 2):
        self.base_budget = base_budget
        self.acceptance_bonus = acceptance_bonus
        self.rejection_penalty = rejection_penalty
        self.hot_window_years = hot_window_years
        self.hot_topics: Dict[str, float] = {}

    def update_budget(self, funding_per_unit: int, accepted: int, maturity: int, weighted_funding_assignment: bool = False) -> float:
        """Update industry lab budget based on performance"""
        
        assert maturity > 0, f"Maturity must be greater than 0. Got {maturity}"
        # new_budget = self.base_budget + self.acceptance_bonus * accepted - self.rejection_penalty * rejected
        if weighted_funding_assignment:
            new_budget = funding_per_unit * accepted * maturity  # We assume that industry researchers are awarded only if they have accepted papers
        else:
            new_budget = funding_per_unit * accepted

        return new_budget

    def evaluate_proposal(self, keywords: Set[str], priorities: Set[str], budget: float, amount: float) -> bool:
        """Decide whether to fund a proposal (heuristic-based)"""
        if budget < amount:
            return False

        overlap = len(keywords & priorities)
        alignment = overlap / len(priorities) if priorities else 0
        hot_score = sum(self.hot_topics.get(kw, 0) for kw in keywords) / len(keywords) if keywords else 0
        prob = self.base_prob + alignment * self.align_bonus_max + hot_score * self.hot_bonus_max
        return random.random() < prob

    def update_hot_topics(self, papers: List[Dict], current_year: int):
        """Track trending keywords from recent top-tier papers"""
        counts = Counter()
        for p in papers:
            if p.get('tier') == 'top' and (current_year - p.get('year', 0)) <= self.hot_window_years:
                counts.update(p.get('keywords', []))

        if counts:
            max_count = max(counts.values())
            self.hot_topics = {kw: cnt / max_count for kw, cnt in counts.items()}

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'base_budget': self.base_budget,
            'acceptance_bonus': self.acceptance_bonus,
            'rejection_penalty': self.rejection_penalty,
            'hot_window_years': self.hot_window_years,
            'hot_topics': self.hot_topics
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'IndustryFundingSystem':
        """Deserialize from dictionary"""
        system = cls(
            base_budget=data.get('base_budget', 20),
            acceptance_bonus=data.get('acceptance_bonus', 20),
            rejection_penalty=data.get('rejection_penalty', 10),
            hot_window_years=data.get('hot_window_years', 2)
        )
        system.hot_topics = data.get('hot_topics', {})
        return system


class FundingProgram:
    """Represents a specific funding program offered by a funding agency"""

    def __init__(
            self,
            program_id: str,
            name: str,
            research_directions: List[ResearchDirection],
            funding_mode: str = "fixed_rate",
            budget: int = None,
            budget_per_winner: int = None,
            funding_rate: float = None
    ):
        self.program_id = program_id
        self.name = name
        self.research_directions = research_directions
        self.funding_mode = funding_mode

        if funding_mode == "fixed_budget":
            self.budget = budget if budget is not None else SIMULATION_CONFIG['funding']['default_budget']
        elif funding_mode == "fixed_rate":
            self.funding_rate = funding_rate if funding_rate is not None else SIMULATION_CONFIG['funding'][
                'default_funding_rate']
            self.budget_per_winner = budget_per_winner if budget_per_winner is not None else \
            SIMULATION_CONFIG['funding']['academic_base_budget']
        else:
            raise ValueError(f"Unknown funding mode: {funding_mode}")

    def reset_for_new_cycle(self):
        """Reset program state for a new funding cycle"""
        if self.funding_mode == "fixed_budget":
            self.remaining_budget = self.budget

    def calculate_funding_amount(self, num_winners: int) -> float:
        """Calculate equal funding amount per winner"""
        return self.budget / max(num_winners, 1) if num_winners > 0 else 0


class FundingAgency(SimulationAgent):
    """Funding agents primarily review and allocate resources"""

    def __init__(self, agency_name: str, funding_level: int = 100, llm: LLM = None, default_funding_rate: float = None, max_retries: int = 10):
        super().__init__(
            reputation=8,
            agent_id=agency_name,
            llm=llm,
            max_retries=max_retries
        )
        self.funding_priorities = []
        self.risk_aversion = "moderate"
        self.can_author = False
        self.can_review = False
        self.default_funding_rate = default_funding_rate if default_funding_rate is not None else \
        SIMULATION_CONFIG['funding']['default_funding_rate']
        self.funding_programs: Dict[str, FundingProgram] = {}
        self._initialize_funding_programs(agency_name)

    def get_type(self) -> str:
        return "funding_agency"

    def _get_personality_prompt(self) -> str:
        return """You are a funding agency representative who evaluates research from a strategic funding perspective,
               considering long-term impact, resource allocation efficiency, and alignment with funding priorities."""

    def _apply_agent_review_biases(self, review: Dict) -> Dict:
        """Apply funding agency-specific biases to reviews"""
        if any(term in str(review.get('strengths', [])).lower() for term in ['impact', 'strategic', 'significant']):
            review['overall_score'] = min(10, review.get('overall_score', 5) + 0.3)

        if any(term in str(review.get('weaknesses', [])).lower() for term in ['high-risk', 'speculative', 'unproven']):
            review['overall_score'] = max(1, review.get('overall_score', 5) - 0.8)

        return review

    def _initialize_funding_programs(self, agency_name: str):
        """Initialize funding programs based on agency type"""
        from .research_direction import DIRECTIONS_DICT

        if agency_name == "NSF":
            self.funding_programs = {
                "NSF_THEORY": FundingProgram(
                    program_id="NSF_THEORY",
                    name="NSF Standard Grant: Theoretical Foundations",
                    research_directions=[
                        DIRECTIONS_DICT["complexity_theory"], DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["logic_in_cs"], DIRECTIONS_DICT["formal_methods"],
                        DIRECTIONS_DICT["discrete_mathematics"], DIRECTIONS_DICT["geometric_algorithms"],
                        DIRECTIONS_DICT["computational_topology"], DIRECTIONS_DICT["data_structures"],
                        DIRECTIONS_DICT["symbolic_computation"], DIRECTIONS_DICT["information_theory"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
                ),
                "NSF_AI_FOUNDATIONS": FundingProgram(
                    program_id="NSF_AI_FOUNDATIONS",
                    name="NSF Standard Grant: AI Foundations",
                    research_directions=[
                        DIRECTIONS_DICT["artificial_intelligence"], DIRECTIONS_DICT["neural_symbolic_ai"],
                        DIRECTIONS_DICT["planning_and_scheduling"], DIRECTIONS_DICT["interpretable_ml"],
                        DIRECTIONS_DICT["algorithmic_game_theory"], DIRECTIONS_DICT["multiagent_systems"],
                        DIRECTIONS_DICT["meta-learning"], DIRECTIONS_DICT["ai_ethics"],
                        DIRECTIONS_DICT["evolutionary_computation"], DIRECTIONS_DICT["mathematical_software"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
                ),
                "NSF_SYSTEM": FundingProgram(
                    program_id="NSF_SYSTEM",
                    name="NSF Standard Grant: Systems & Architecture",
                    research_directions=[
                        DIRECTIONS_DICT["computer_architecture"], DIRECTIONS_DICT["operating_systems"],
                        DIRECTIONS_DICT["distributed_systems"], DIRECTIONS_DICT["parallel_computing"],
                        DIRECTIONS_DICT["quantum_computing"], DIRECTIONS_DICT["performance_analysis"],
                        DIRECTIONS_DICT["database_systems"], DIRECTIONS_DICT["computer_networks"],
                        DIRECTIONS_DICT["programming_languages"], DIRECTIONS_DICT["software_engineering"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
                )
            }
        elif agency_name == "DARPA":
            self.funding_programs = {
                "DARPA_AUTONOMOUS": FundingProgram(
                    program_id="DARPA_AUTONOMOUS",
                    name="DARPA Program: Autonomous Systems Deployment",
                    research_directions=[
                        DIRECTIONS_DICT["robotics"], DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["control_systems"], DIRECTIONS_DICT["image_recognition"],
                        DIRECTIONS_DICT["3d_vision"], DIRECTIONS_DICT["video_understanding"],
                        DIRECTIONS_DICT["multiagent_systems"], DIRECTIONS_DICT["planning_and_scheduling"],
                        DIRECTIONS_DICT["deep_learning"], DIRECTIONS_DICT["performance_analysis"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
                ),
                "DARPA_SECURITY": FundingProgram(
                    program_id="DARPA_SECURITY",
                    name="DARPA Program: Cybersecurity Applications",
                    research_directions=[
                        DIRECTIONS_DICT["cybersecurity"], DIRECTIONS_DICT["cryptography"],
                        DIRECTIONS_DICT["computer_networks"], DIRECTIONS_DICT["formal_methods"],
                        DIRECTIONS_DICT["distributed_systems"], DIRECTIONS_DICT["software_engineering"],
                        DIRECTIONS_DICT["operating_systems"], DIRECTIONS_DICT["database_systems"],
                        DIRECTIONS_DICT["programming_languages"], DIRECTIONS_DICT["performance_analysis"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
                ),
                "DARPA_AI_APPS": FundingProgram(
                    program_id="DARPA_AI_APPS",
                    name="DARPA Program: Applied AI Systems",
                    research_directions=[
                        DIRECTIONS_DICT["deep_learning"], DIRECTIONS_DICT["natural_language_processing"],
                        DIRECTIONS_DICT["image_recognition"], DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["dialogue_systems"], DIRECTIONS_DICT["human_computer_interaction"],
                        DIRECTIONS_DICT["generative_models"], DIRECTIONS_DICT["interpretable_ml"],
                        DIRECTIONS_DICT["multiagent_systems"], DIRECTIONS_DICT["computer_graphics"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
                )
            }
        else:
            self.funding_programs = {
                f"{agency_name}_GENERAL": FundingProgram(
                    program_id=f"{agency_name}_GENERAL",
                    name=f"{agency_name} Standard Grant: General Computing",
                    research_directions=[
                        DIRECTIONS_DICT["deep_learning"], DIRECTIONS_DICT["algorithms"],
                        DIRECTIONS_DICT["software_engineering"], DIRECTIONS_DICT["reinforcement_learning"],
                        DIRECTIONS_DICT["natural_language_processing"], DIRECTIONS_DICT["distributed_systems"],
                        DIRECTIONS_DICT["computer_networks"], DIRECTIONS_DICT["database_systems"],
                        DIRECTIONS_DICT["human_computer_interaction"], DIRECTIONS_DICT["image_recognition"]
                    ],
                    funding_mode="fixed_rate",
                    funding_rate=self.default_funding_rate
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

        # Group applications by program
        program_to_applications = {}

        # Populate program_to_applications from the applications format
        for apps in applications:

            # Each author_applications dict contains program_id -> application_data
            for program_id, application_data in apps.items():

                if not application_data['submit']:
                    continue

                author = application_data['author']

                if program_id not in program_to_applications:
                    program_to_applications[program_id] = []

                # Add author info to the application
                application_data['applicant_id'] = author.id
                program_to_applications[program_id].append(application_data)

        winners_all_programs = {}
        # Process each program's applications using LLM to decide winners
        for program_id, apps in program_to_applications.items():
            if not program_id in self.funding_programs:
                continue
            # Find the corresponding program object
            program = self.funding_programs[program_id]
            logger.info(f"Eval funding apps for {program_id}")

            # Prepare applications for LLM evaluation
            applications_text = ""
            for i, app in enumerate(apps):
                author = app.get('author')
                expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                                    for rd in (author.expertise if author and hasattr(author, 'expertise') else [])]

                accepted_papers = [paper for paper in app['relevant_projects'] if paper['status'] == 'accept']
                rejected_papers = [paper for paper in app['relevant_projects'] if paper['status'] == 'reject']
                
                if len(accepted_papers) == 0 and len(rejected_papers) == 0:
                    past_performance = "The agent has no research projects completed yet."
                past_performance = f"""The agent has completed {len(accepted_papers) + len(rejected_papers)} research projects, with {len(accepted_papers)} accepted papers and {len(rejected_papers)} rejected papers.
                
"""

                if len(accepted_papers) > 0:
                    past_performance += "### Recent accepted papers:\n"
                    for paper in accepted_papers:
                        past_performance += f" - {submitted_papers_dict[paper['arxiv_id']]['title']}\n"

                if len(rejected_papers) > 0:
                    past_performance += "### Recent rejected papers:\n"
                    for paper in rejected_papers:
                        past_performance += f" - {submitted_papers_dict[paper['arxiv_id']]['title']}\n"
                applications_text += f"""### Application {i + 1}
Applicant ID: {app['applicant_id']}
Research Proposal: {app.get('research_proposal', 'No proposal provided')}
Expertise: {', '.join(expertise_topics)}
Affiliation: {getattr(author, 'university_name', getattr(author, 'company_name', 'Unknown')) if author else 'Unknown'}
Past Performance:
{past_performance}

"""

            # Create LLM prompt for funding decisions
            focus_topics = [rd.topic for rd in program.research_directions]

            # Extract all valid applicant IDs
            valid_ids = [app['applicant_id'] for app in apps]
            valid_ids_text = '\n'.join([f"  - {aid}" for aid in valid_ids])

            sample_response = """
```json
{
    "ranked_applications": [{  <- Must be a ranked list according to preferences.
        "id": "University_Researcher_888", <- Must be EXACTLY the same as the applicant_id in the application
        "rank": 1, <- Must be 1.
        "reason": "Reason for this ranking" <- a 1-sentence brief explanation for the ranking
    },
    {
        "id": "University_Researcher_404",
        "rank": 2, <- Must be 2
        "reason": "Reason for this ranking"
    },
    ...
    {
        "id": "Industry_Researcher_325",
        "rank": N, <- Must be N
        "reason": "Reason for this ranking"
    },
    ]
}
```
"""
            prompt = f"""You are a funding agency evaluating applications for {program.name}.

## Program Details
Program Focus Areas: {', '.join(focus_topics)}

## Selection Criteria
You MUST rank ALL {len(apps)} applications from most preferred (rank 1) to least preferred (rank {len(apps)}).

Based on the following criteria, provide a complete ranking of all applications:
1. Alignment with program focus areas
2. Research quality and innovation potential
3. Past performance and track record. Focus more on recent works and accepted papers. Focus less on older works and rejected papers.
4. Feasibility and impact potential

## Applications to Evaluate
{applications_text}

## Sample Response
{sample_response}
"""
            ranked_apps = None
            max_retries = self.max_retries
            for retry in range(self.max_retries):
                try:
                    result, message_history = self.llm.generate(prompt=prompt)

                    ranked_apps = result.get('ranked_applications', [])

                    # Validate count
                    if len(ranked_apps) < len(apps) // 2:
                        if retry < self.max_retries - 1:
                            logger.error(f"Expected {len(apps)} ranked applications, got {len(ranked_apps)}. Retrying... (attempt {retry + 1}/{max_retries})")
                            continue
                        else:
                            raise AssertionError(f"Expected {len(apps)} ranked applications, got {len(ranked_apps)}")

                    # Validate all application IDs are present
                    ranked_ids = {app['id'] for app in ranked_apps}
                    expected_ids = {app['applicant_id'] for app in apps}
                    
                    

                    if ranked_ids != expected_ids:
                        
                        logger.debug(f"Ranked IDs (length: {len(ranked_ids)}): {ranked_ids}")
                        logger.debug(f"Expected IDs (length: {len(expected_ids)}): {expected_ids}")
                        ranked_apps = [app for app in ranked_apps if app['id'] in expected_ids & ranked_ids]
                        
                        for id in expected_ids - ranked_ids:
                            ranked_apps.append({
                                'id': id,
                                'rank': len(ranked_apps) + 1,
                                'reason': ""
                            })
                    # Sort by rank to ensure proper ordering (rank 1 is most =
                    ranked_apps_sorted = sorted(ranked_apps, key=lambda x: x.get('rank', float('inf')))
                    num_winners = max(1, int(len(ranked_apps_sorted) * program.funding_rate))

                    ranked_apps_sorted = ranked_apps_sorted[:num_winners]
                    winners_all_programs[program_id] = ranked_apps_sorted
                    break  # Success, exit retry loop

                except Exception as e:
                    if retry == max_retries - 1:
                        logger.error(f"Error evaluating funding applications for {program_id} after {max_retries} attempts: {e}")
                        traceback.print_exc()
                        # Fallback: select first N applications
                        winners_all_programs[program_id] = []
                    else:
                        logger.warning(f"Error on attempt {retry + 1}/{max_retries}: {e}")
                        continue
            
        return winners_all_programs
