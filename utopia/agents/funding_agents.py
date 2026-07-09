"""Funding agents and systems for academic and industry funding"""
import logging
import random
from collections import Counter
from typing import Dict, Optional, Set, List, Tuple

from pydantic import BaseModel

from utopia.config import SIMULATION_CONFIG
from utopia.models.models import BaseLLM
from .base_agent import SimulationAgent
from .conference import DIRECTION_TO_ARXIV, ARXIV_CATEGORY_NAMES
from .research_direction import ResearchDirection


class RankedApplication(BaseModel):
    application_id: int
    applicant_id: str
    rank: int
    reason: str


class FundingEvaluationResponse(BaseModel):
    ranked_applications: List[RankedApplication]

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

        # Derive translated arXiv topic names from research directions
        arxiv_codes = set()
        for rd in self.research_directions:
            arxiv_codes.update(DIRECTION_TO_ARXIV.get(rd.topic, []))
        self.topics = sorted(set(
            ARXIV_CATEGORY_NAMES[code] for code in arxiv_codes if code in ARXIV_CATEGORY_NAMES
        ))

        if funding_mode == "fixed_budget":
            self.budget = budget if budget is not None else SIMULATION_CONFIG['funding']['default_budget']
        elif funding_mode == "fixed_rate":
            self.funding_rate = funding_rate if funding_rate is not None else SIMULATION_CONFIG['funding'][
                'default_funding_rate']
            self.budget_per_winner = budget_per_winner if budget_per_winner is not None else \
            SIMULATION_CONFIG['funding']['academic_base_budget']
        else:
            raise ValueError(f"Unknown funding mode: {funding_mode}")

    def is_topic_match(self, paper_topics: List[str]) -> bool:
        """Check if paper topics match program topics"""
        paper_topic_set = set(t.lower() for t in paper_topics)
        program_topic_set = set(t.lower() for t in self.topics)
        return bool(paper_topic_set.intersection(program_topic_set))

    def reset_for_new_cycle(self):
        """Reset program state for a new funding cycle"""
        if self.funding_mode == "fixed_budget":
            self.remaining_budget = self.budget

    def calculate_funding_amount(self, num_winners: int) -> float:
        """Calculate equal funding amount per winner"""
        return self.budget / max(num_winners, 1) if num_winners > 0 else 0


class FundingAgency(SimulationAgent):
    """Funding agents primarily review and allocate resources"""

    def __init__(self, agency_name: str, funding_level: int = 100, llm: BaseLLM = None, default_funding_rate: float = None, max_retries: int = 10):
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
        self.conflict_of_interest = set()
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

    def get_funding_evaluation_prompts(
            self,
            applications: List[Dict],
            submitted_papers_dict: Dict[str, Dict],
            panel_max_apps: Optional[int] = None,
            panel_seed: Optional[int] = None,
    ) -> Tuple[List[str], Dict, List[Dict]]:
        """Build evaluation prompts for all programs belonging to this agency.

        When a program receives more than `panel_max_apps` applications, they are
        split into deterministic seeded panels of at most that size (one prompt
        per panel); the funding rate is applied per panel. This is the documented
        scale control for large populations — its fidelity vs global ranking is
        validated by the panelization fidelity experiment before confirmatory use.

        Returns:
            Tuple of (prompts, response_format, metadata_list) where metadata_list
            is parallel to prompts: [{'program_id': ..., 'apps': ..., 'panel_index': ...}, ...]
        """

        # Group applications by program
        program_to_applications = {}

        for apps in applications:
            for program_id, application_data in apps.items():

                if not application_data['submit']:
                    continue

                author = application_data['author']

                if program_id not in program_to_applications:
                    program_to_applications[program_id] = []

                application_data['applicant_id'] = author.id
                program_to_applications[program_id].append(application_data)

        # Deterministic panelization for programs exceeding the panel cap
        program_panels = {}
        for program_id, apps in program_to_applications.items():
            if panel_max_apps and len(apps) > panel_max_apps:
                import math as _math
                from utopia.utils.generail_utils import derive_seed
                rng = random.Random(derive_seed(panel_seed if panel_seed is not None else 0,
                                                'funding_panels', program_id))
                shuffled = list(apps)
                rng.shuffle(shuffled)
                n_panels = _math.ceil(len(shuffled) / panel_max_apps)
                program_panels[program_id] = [shuffled[i::n_panels] for i in range(n_panels)]
            else:
                program_panels[program_id] = [apps]

        batch_prompts = []
        metadata_list = []

        for program_id, panels in program_panels.items():
            if program_id not in self.funding_programs:
                continue

            program = self.funding_programs[program_id]
            logger.info(f"Preparing funding eval prompt(s) for {program_id} ({len(panels)} panel(s))")
            for panel_index, apps in enumerate(panels):
                batch_prompts.append(
                    self._build_program_prompt(program, apps, submitted_papers_dict))
                metadata_list.append({'program_id': program_id, 'apps': apps,
                                      'panel_index': panel_index})

        response_format = {
            'type': 'json_schema',
            'json_object': {
                'name': 'FundingEvaluationResponse',
                'strict': True,
                'schema': FundingEvaluationResponse.model_json_schema()
            }
        }

        return batch_prompts, response_format, metadata_list

    @staticmethod
    def _build_program_prompt(program: 'FundingProgram', apps: List[Dict],
                              submitted_papers_dict: Dict[str, Dict]) -> str:
        """Build one ranking prompt for a program (or one panel of it)."""
        applications_text = ""
        for i, app in enumerate(apps):
            author = app.get('author')
            expertise_topics = [rd.topic if isinstance(rd, ResearchDirection) else str(rd)
                                for rd in (author.expertise if author and hasattr(author, 'expertise') else [])]

            accepted_papers = [paper for paper in app['relevant_projects'] if paper['status'] == 'accept']
            rejected_papers = [paper for paper in app['relevant_projects'] if paper['status'] == 'reject']

            if len(accepted_papers) == 0 and len(rejected_papers) == 0:
                past_performance = "The agent has no research projects completed yet."
            else:
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

            applications_text += f"""### Application {i}
Applicant ID: {app['applicant_id']}
Research Proposal: {app.get('research_proposal', 'No proposal provided')}
Expertise: {', '.join(expertise_topics)}
Affiliation: {getattr(author, 'university_name', getattr(author, 'company_name', 'Unknown')) if author else 'Unknown'}
Past Performance:
{past_performance}

"""

        focus_topics = [rd.topic for rd in program.research_directions]

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
"""
        instruction = f"MUST BE an integer between 0 and {len(apps) - 1}."
        prompt += f"""
## Response Format
Respond in JSON format with all applications ranked from most preferred (rank 1) to least preferred. An example response is shown below (DO NOT directly copy the exact response):
```json
{{
    "ranked_applications": [
        {{
            "application_id": 21, <- {instruction}
            "applicant_id": "Harvard_researcher_4", <- MUST BE a string that is the same as the "Applicant ID" in the applications
            "rank": 1, <- MUST BE 1 because this is the most preferred application
            "reason": "BRIEF REASON"
        }},
        {{
            "application_id": 13, <- {instruction}
            "applicant_id": "Gatech_researcher_2", <- MUST BE a string that is the same as the "Applicant ID" in the applications
            "rank": 2, <- MUST BE 2 because this is the second most preferred application
            "reason": "BRIEF REASON"
        }},
        ...
    ]
}}
```
"""
        return prompt

    @staticmethod
    def process_funding_evaluation_results(
            batch_results: List[Tuple[Dict, List[Dict]]],
            metadata_list: List[Dict],
            funding_programs: Dict[str, 'FundingProgram'],
            novelty_penalties: Optional[Dict[str, float]] = None,
            lambda_funding: float = 0.0,
    ) -> Dict[str, List[Dict]]:
        """Process batch LLM results into winners per program (or per panel).

        Funding-conservatism intervention (plan 4.4/6.2): when lambda_funding > 0,
        each application's LLM rank is converted to a normalized raw score
        (1 = best, 0 = worst), penalized by lambda_funding * novelty_percentile,
        and winners are selected on the ADJUSTED score BEFORE the funding-rate
        cut. Both scores are stored on each application. lambda_funding == 0
        (neutral condition) reproduces the original rank-order selection exactly.

        Args:
            batch_results: List of (result_dict, message_history) from generate_batch / generate.
            metadata_list: Parallel list of {'program_id', 'apps', 'panel_index'} dicts.
            funding_programs: Combined dict of all funding programs across agencies.
            novelty_penalties: applicant_id -> recent-novelty percentile in [0, 1].
            lambda_funding: penalty coefficient (0 = off).

        Returns:
            Dict[program_id, List[Dict]] — winner lists per program (panels merged).
        """
        winners_all_programs = {}
        for idx, (result, _message_history) in enumerate(batch_results):
            program_id = metadata_list[idx]['program_id']
            apps = metadata_list[idx]['apps']
            program = funding_programs[program_id]

            if result is None:
                logger.warning(f"No response from LLM for program {program_id}")
                continue
            elif isinstance(result, list):
                ranked_apps = result
            else:
                ranked_apps = result.get('ranked_applications', [])

            # Validate all applicant IDs are present
            ranked_applicant_ids = {app['applicant_id'] for app in ranked_apps}
            expected_applicant_ids = {app['applicant_id'] for app in apps}

            if ranked_applicant_ids != expected_applicant_ids:
                logger.debug(f"Ranked applicant IDs (length: {len(ranked_applicant_ids)}): {ranked_applicant_ids}")
                logger.debug(f"Expected applicant IDs (length: {len(expected_applicant_ids)}): {expected_applicant_ids}")
                # Keep only ranked apps whose applicant_id is valid
                ranked_apps = [app for app in ranked_apps if app['applicant_id'] in expected_applicant_ids]

                # Append missing applicant IDs at the end with fallback ranking
                matched_ids = {app['applicant_id'] for app in ranked_apps}
                for missing_id in expected_applicant_ids - matched_ids:
                    ranked_apps.append({
                        'application_id': -1,
                        'applicant_id': missing_id,
                        'rank': len(ranked_apps) + 1,
                        'reason': ""
                    })

            # Sort by rank (rank 1 = most preferred)
            ranked_apps_sorted = sorted(ranked_apps, key=lambda x: x.get('rank', float('inf')))

            # Normalized raw score from rank: 1 (best) .. 0 (worst)
            n = len(ranked_apps_sorted)
            for pos, app in enumerate(ranked_apps_sorted):
                raw = 1.0 - (pos / (n - 1)) if n > 1 else 1.0
                penalty = (novelty_penalties or {}).get(app['applicant_id'], 0.0)
                app['normalized_raw_score'] = raw
                app['novelty_penalty'] = lambda_funding * penalty
                app['adjusted_score'] = raw - lambda_funding * penalty

            if lambda_funding > 0:
                # Stable sort keeps LLM rank order among equal adjusted scores
                ranked_apps_sorted = sorted(
                    ranked_apps_sorted, key=lambda x: -x['adjusted_score'])

            num_winners = max(1, int(n * program.funding_rate))
            winners_all_programs.setdefault(program_id, []).extend(
                ranked_apps_sorted[:num_winners])

        return winners_all_programs

    @staticmethod
    def validate_funding_result(result, apps):
        """Check if a single funding evaluation result is valid."""
        if result is None:
            return False
        ranked_apps = result if isinstance(result, list) else result.get('ranked_applications', [])
        if not ranked_apps:
            return False

        # Check all expected applicant_ids are present
        ranked_applicant_ids = {app.get('applicant_id') for app in ranked_apps}
        expected_applicant_ids = {app['applicant_id'] for app in apps}
        if ranked_applicant_ids > expected_applicant_ids:
            return False

        # Check application_id <-> applicant_id correspondence
        # In the prompt, Application i has applicant_id = apps[i]['applicant_id']
        id_to_applicant = {i: app['applicant_id'] for i, app in enumerate(apps)}
        for app in ranked_apps:
            app_id = app.get('application_id')
            if app_id not in id_to_applicant or id_to_applicant[app_id] != app.get('applicant_id'):
                return False

        return True
