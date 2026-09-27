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
                from utopia.utils.seeding import derive_seed
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
    def normalize_ranked_applications(ranked_apps, apps):
        """Normalize a possibly malformed LLM ranking into a complete,
        deterministic per-application ranking.

        The unit is the APPLICATION: `application_id` == index into `apps`,
        with expected `applicant_id = apps[application_id]['applicant_id']`.
        Multiple applications from the same applicant are legitimate and
        preserved — deduplication is by application_id, never applicant_id.

        Per returned entry:
        - non-dict entries are rejected;
        - a valid in-range int application_id with missing applicant_id
          recovers the applicant from the input application;
        - a missing application_id with a present applicant_id is inferred
          only when that applicant has exactly one application in this panel
          (otherwise ambiguous -> rejected);
        - inconsistent id pairs and out-of-range application_ids are rejected;
        - duplicates keep the best valid rank, then earliest response position.

        Every expected application absent after the above is appended to a
        fallback tail using its REAL input application_id, flagged
        `imputed_tail: True`, with ranks after all valid entries.

        Returns (normalized, rejects): `normalized` has exactly len(apps)
        entries whose application_ids are a permutation of 0..len(apps)-1;
        `rejects` records each dropped entry with its reason. Never raises on
        malformed input.
        """
        n = len(apps)
        expected = {i: apps[i]['applicant_id'] for i in range(n)}
        indices_by_applicant = {}
        for i in range(n):
            indices_by_applicant.setdefault(expected[i], []).append(i)

        rejects = []
        kept = {}  # application_id -> (sort_key, response_pos, entry)
        for pos, entry in enumerate(ranked_apps or []):
            if not isinstance(entry, dict):
                rejects.append({'response_pos': pos, 'reason': 'non_dict_entry'})
                continue
            app_id = entry.get('application_id')
            applicant_id = entry.get('applicant_id')
            if isinstance(app_id, bool) or not isinstance(app_id, int):
                app_id = None
            if app_id is not None and not (0 <= app_id < n):
                rejects.append({'response_pos': pos, 'reason': 'application_id_out_of_range',
                                'application_id': entry.get('application_id')})
                continue
            if app_id is None:
                candidates = indices_by_applicant.get(applicant_id, [])
                if len(candidates) == 1:
                    app_id = candidates[0]
                else:
                    reason = ('ambiguous_applicant_multiple_applications' if candidates
                              else 'no_valid_identifier')
                    rejects.append({'response_pos': pos, 'reason': reason,
                                    'applicant_id': applicant_id})
                    continue
            elif applicant_id is not None and applicant_id != expected[app_id]:
                rejects.append({'response_pos': pos, 'reason': 'inconsistent_ids',
                                'application_id': app_id, 'applicant_id': applicant_id})
                continue

            norm = dict(entry)
            norm['application_id'] = app_id
            norm['applicant_id'] = expected[app_id]
            rank = norm.get('rank')
            valid_rank = isinstance(rank, (int, float)) and not isinstance(rank, bool)
            sort_key = (rank if valid_rank else float('inf'), pos)
            prev = kept.get(app_id)
            if prev is None or sort_key < prev[0]:
                if prev is not None:
                    rejects.append({'response_pos': prev[1],
                                    'reason': 'duplicate_application_id',
                                    'application_id': app_id})
                kept[app_id] = (sort_key, pos, norm)
            else:
                rejects.append({'response_pos': pos,
                                'reason': 'duplicate_application_id',
                                'application_id': app_id})

        # Valid entries in original response order (downstream sorts by rank;
        # the stable sort makes this order the deterministic tie-break).
        normalized = [rec[2] for rec in sorted(kept.values(), key=lambda rec: rec[1])]
        valid_ranks = [rec[2]['rank'] for rec in kept.values()
                       if isinstance(rec[2].get('rank'), (int, float))
                       and not isinstance(rec[2].get('rank'), bool)]
        tail_rank = max(valid_ranks, default=0)
        for app_id in range(n):
            if app_id not in kept:
                tail_rank += 1
                normalized.append({
                    'application_id': app_id,
                    'applicant_id': expected[app_id],
                    'rank': tail_rank,
                    'reason': '',
                    'imputed_tail': True,
                })
        return normalized, rejects

    @staticmethod
    def allocate_slots_largest_remainder(total: int, weights: List[int]) -> List[int]:
        """Split `total` integer slots across groups in proportion to `weights`
        (Hamilton / largest-remainder apportionment). Deterministic: ties in the
        fractional part go to the earlier group. Groups with weight 0 get 0.
        sum(result) == min(total, sum(weights)) so no group exceeds its demand.
        """
        assert total >= 0 and all(w >= 0 for w in weights)
        demand = sum(weights)
        if demand == 0 or total == 0:
            return [0] * len(weights)
        total = min(total, demand)
        quotas = [total * w / demand for w in weights]
        base = [int(q) for q in quotas]
        remainder = total - sum(base)
        order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - base[i]), i))
        for i in order[:remainder]:
            base[i] += 1
        # never exceed demand of a group
        assert all(b <= w for b, w in zip(base, weights))
        return base

    @staticmethod
    def process_funding_evaluation_results(
            batch_results: List[Tuple[Dict, List[Dict]]],
            metadata_list: List[Dict],
            funding_programs: Dict[str, 'FundingProgram'],
            novelty_penalties: Optional[Dict[str, float]] = None,
            lambda_funding: float = 0.0,
            application_log: Optional[List[Dict]] = None,
            slot_override: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict]]:
        """Process batch LLM results into winners per program (or per panel).

        slot_override (--funding_budget_mode fixed): program_id -> total winner slots
        for that program this year. When given, replaces the legacy
        max(1, int(n * funding_rate)) rule; a program's slots are split across its
        panels by largest remainder over panel sizes, and a program may get 0 winners.
        None (default) reproduces the legacy rule exactly.

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
        # Fixed-budget mode: per-panel slot table computed up front from panel sizes
        panel_slots = {}
        if slot_override is not None:
            panels_by_program = {}
            for idx, meta in enumerate(metadata_list):
                panels_by_program.setdefault(meta['program_id'], []).append(idx)
            for program_id, idxs in panels_by_program.items():
                sizes = [len(metadata_list[i]['apps']) for i in idxs]
                shares = FundingAgency.allocate_slots_largest_remainder(
                    int(slot_override.get(program_id, 0)), sizes)
                for i, share in zip(idxs, shares):
                    panel_slots[i] = share
        for idx, (result, _message_history) in enumerate(batch_results):
            program_id = metadata_list[idx]['program_id']
            apps = metadata_list[idx]['apps']
            program = funding_programs[program_id]

            if result is None:
                # Deterministic strategy-blind safety net: rank in panel order
                # (already a seeded shuffle) so persistent parse failures never
                # zero out a program-year. Flagged for the provenance audit.
                logger.warning(f"No response from LLM for program {program_id}; "
                               f"using flagged panel-order fallback ranking")
                ranked_apps = [{'application_id': i, 'applicant_id': app['applicant_id'],
                                'rank': i + 1, 'reason': 'fallback_ranking',
                                'fallback_ranking': True}
                               for i, app in enumerate(apps)]
            elif isinstance(result, list):
                ranked_apps = result
            else:
                ranked_apps = result.get('ranked_applications', [])

            # Normalize the LLM ranking into a complete per-application
            # ranking (recover/reject malformed entries, impute a fallback
            # tail); tolerates arbitrary malformed structured output.
            ranked_apps, rejects = FundingAgency.normalize_ranked_applications(
                ranked_apps, apps)
            if rejects:
                logger.warning(
                    f"Program {program_id}: rejected {len(rejects)} malformed "
                    f"ranking entries: "
                    f"{sorted({r['reason'] for r in rejects})}")

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

            if slot_override is not None:
                num_winners = min(panel_slots.get(idx, 0), n)
            else:
                num_winners = max(1, int(n * program.funding_rate))
            winners_all_programs.setdefault(program_id, []).extend(
                ranked_apps_sorted[:num_winners])

            # Matthew-RD audit log: persist the FULL pre-decision ranking
            # (winners AND losers) so the funding cutoff is reconstructable.
            # `position` is the final selection order actually used for the
            # funding-rate cut; funded == (position <= num_winners) exactly.
            if application_log is not None:
                for pos, app in enumerate(ranked_apps_sorted):
                    application_log.append({
                        'program_id': program_id,
                        'panel_index': metadata_list[idx]['panel_index'],
                        'applicant_id': app['applicant_id'],
                        'llm_rank': app.get('rank'),
                        'position': pos + 1,
                        'normalized_raw_score': app['normalized_raw_score'],
                        'adjusted_score': app['adjusted_score'],
                        'novelty_penalty': app['novelty_penalty'],
                        'fallback_ranking': bool(app.get('fallback_ranking', False)),
                        'imputed_tail': bool(app.get('imputed_tail', False)),
                        'n_panel': n,
                        'funding_rate': program.funding_rate,
                        'num_winners': num_winners,
                        'funded': pos < num_winners,
                    })

        return winners_all_programs

    @staticmethod
    def validate_funding_result(result, apps):
        """Check if a single funding evaluation result is valid."""
        if result is None:
            return False
        ranked_apps = result if isinstance(result, list) else result.get('ranked_applications', [])
        if not ranked_apps:
            return False
        if not all(isinstance(app, dict) for app in ranked_apps):
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
