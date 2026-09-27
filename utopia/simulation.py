#!/usr/bin/env python3
"""
Main simulation orchestration script for the Academic Society Ecosystem

This script runs the full multi-agent simulation with:
- Multiple UniversityResearchers and IndustryResearchers
- Multiple conferences targeting different topics
- Configurable research direction assignment (batch or individual mode)
- Yearly cycles with funding management
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import write_json_atomic, write_json_gzip_atomic

# CRITICAL: Set environment variables for vLLM multiprocessing BEFORE any imports
# This tells vLLM to use 'spawn' instead of 'fork' for CUDA compatibility
import os
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
# Disable tokenizers parallelism to avoid fork issues
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Set multiprocessing start method before any CUDA imports
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import json
import logging
import random
import time
import traceback
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.conference import ConferenceSystem, select_conferences_for_simulation
from utopia.agents.funding_agents import FundingAgency, IndustryFundingSystem
from utopia.agents.research_direction import assign_directions_individual, create_research_directions_batch, get_strategy_filtered_candidate_directions, AVAILABLE_DIRECTIONS
from utopia.agents.researcher_agents import UniversityResearcher, IndustryResearcher
from utopia.metrics.wandb_logger import WandbLogger
from utopia.analysis.analyze_all_results import SimulationAnalyzer
from utopia.arguments import parse_arguments
from utopia.config import SIMULATION_CONFIG
from utopia.data.paper_tracker import PaperTracker
from utopia.data.rag import RAG, RAGConfig
from utopia.data.tracker import CitationTracker, AgentTracker, FundingTracker
from utopia.metrics.tracker import calculate_gini_coefficient, calculate_conference_score_metrics
from utopia.models.models import GPT, BaseLLM, VLLMModel, VLLMServerModel, Gemini
from utopia.runtime.setup import project_setup
from utopia.utils.seeding import set_seed, derive_seed
from utopia.runtime.provenance import write_run_manifest


logger = logging.getLogger(__name__)


def check_citation_integrity(paper_tracker, citation_tracker) -> List[str]:
    """Core citation-graph integrity rules (plan 11.4); returns failure messages.

    Module-level so unit tests can exercise it on hand-built trackers.
    """
    failures = []
    papers = paper_tracker.papers_by_id

    def _first_year(paper):
        # ArchivedPaper.year moves forward on resubmission; the temporal rule
        # must use the FIRST submission year (a paper is citable from then on).
        years = [r.get('year') for r in getattr(paper, 'review_history', []) or []
                 if r.get('year') is not None]
        return min(years + [paper.year])

    for cited_id, citers in citation_tracker.citations.items():
        if cited_id not in papers:
            failures.append(f"citation target {cited_id} not in paper tracker")
            continue
        cited_year = _first_year(papers[cited_id])
        for citer in citers:
            if citer == cited_id:
                failures.append(f"self-loop citation on {cited_id}")
            if citer in papers and papers[citer].year < cited_year:
                failures.append(f"future citation: {citer} (y{papers[citer].year}) cites "
                                f"{cited_id} (first submitted y{cited_year})")
    for pid in papers:
        count = citation_tracker.get_citation_count(pid)
        in_degree = len(citation_tracker.get_papers_citing(pid))
        if count != in_degree or not isinstance(count, int):
            failures.append(f"citation count mismatch for {pid}: count={count}, in_degree={in_degree}")
    return failures


# Default-population institution rosters (single source of truth for
# initialize_agents() and build_influx_cohort(); the [:10]/[:20] slices below
# select the active ecosystem: 10 rich + 20 normal of each kind, 5 researchers
# each = 300 researchers).
RICH_UNIVERSITIES = [
    "MIT", "Stanford", "Berkeley", "CMU", "GeorgiaTech",
    "Princeton", "Cornell", "UCLA", "UIUC", "UW",
]
NORMAL_UNIVERSITIES = [
    # Top US Universities
    "Harvard", "Yale", "Caltech", "UCSD", "UT Austin",
    "UMich", "Columbia", "NYU", "USC", "Northwestern",
    "UPenn", "Duke", "Johns Hopkins", "Brown", "Dartmouth",
    "UChicago", "Rice", "Vanderbilt", "Notre Dame", "Emory",
    "UNC Chapel Hill", "UVA", "UCSB", "UCI", "UC Davis",
    "Purdue", "Penn State", "OSU", "Wisconsin", "Maryland",
    # European Universities
    "Oxford", "Cambridge", "Imperial College", "UCL", "Edinburgh",
    "ETH Zurich", "EPFL", "TU Munich", "KU Leuven", "Heidelberg",
    # Asian Universities
    "Tsinghua", "Peking University", "NUS", "NTU", "HKUST",
    "Tokyo", "Kyoto", "KAIST", "SNU", "IIT Bombay",
    # Others
    "Toronto", "UBC", "McGill", "Melbourne", "ANU"
]
RICH_COMPANIES = ["Google", "Meta", "Microsoft", "Amazon", "Apple",
                  "OpenAI", "DeepMind", "Anthropic", "Waymo", "Netflix",
                  ]
NORMAL_COMPANIES = [
    # Major Tech Giants
    "IBM", "Oracle", "Salesforce", "SAP", "Cisco",
    # AI Companies & Labs
    "Cohere", "Stability AI",
    "Hugging Face", "Mistral AI", "Inflection AI", "Adept AI", "Character.AI",
    # Hardware & Semiconductors
    "Nvidia", "Intel", "AMD", "Qualcomm", "Broadcom",
    "TSMC", "Samsung Electronics", "SK Hynix", "Micron", "ARM",
    # Software & Cloud
    "Adobe", "Autodesk", "ServiceNow", "Workday", "Snowflake",
    "Databricks", "MongoDB", "Atlassian", "VMware", "Red Hat",
    # Research Labs & Institutes
    "Allen Institute", "FAIR", "Google Research", "MSR", "Sony AI",
    "NVIDIA Research", "IBM Research", "Bosch Research", "Toyota Research", "Samsung Research",
    # Chinese Tech Companies
    "Tencent", "Alibaba", "Baidu", "ByteDance", "Huawei",
    "JD.com", "Xiaomi", "Meituan", "DiDi", "SenseTime",
    # Others
    "Tesla", "Uber", "Lyft", "Spotify", "Netflix",
    "Shopify", "Square", "Stripe", "Airbnb", "DoorDash"
]
# Initial funding by institution tier — entrant funding (influx2x2) must equal
# founder funding, so both code paths read these constants.
RICH_UNIVERSITY_FUNDING = 160
NORMAL_UNIVERSITY_FUNDING = 80
RICH_COMPANY_FUNDING = 320
NORMAL_COMPANY_FUNDING = 160


def build_population_blueprint(num_institutions: int, researchers_per_institution: int,
                               seed: int, num_expertise: int = 3,
                               strategy_mix: List[str] = None) -> List[Dict]:
    """Build a deterministic balanced university-only population blueprint.

    Within every institution the strategy mix is exactly the configured list
    (default 1 explorer / 3 exploiters / 1 cautious_explorer), shuffled with a
    seeded RNG so assignment is random but reproducible (plan 4.2). Expertise
    directions are sampled with the same RNG. Pure function — unit-testable
    without any Simulation or LLM.

    Returns:
        List of rows: {institution, researcher_name, strategy, expertise_topics,
        expertise_indices, initial_funding_placeholder}
    """
    exp_config = SIMULATION_CONFIG['exploration_experiment']
    # E4b composition intervention passes an explicit mix; the seeded RNG stream
    # is identical regardless of mix, so paired seeds share the same blueprint
    # base (institutions, expertise) and differ ONLY in strategy labels.
    strategy_mix = list(strategy_mix or exp_config['institution_strategy_mix'])
    assert len(strategy_mix) == researchers_per_institution, (
        f"institution_strategy_mix has {len(strategy_mix)} entries but "
        f"researchers_per_institution={researchers_per_institution}")

    rng = random.Random(derive_seed(seed, 'population'))
    blueprint = []
    for inst_idx in range(num_institutions):
        institution = f"institution_{inst_idx:04d}"
        strategies = list(strategy_mix)
        rng.shuffle(strategies)
        for r_idx, strategy in enumerate(strategies):
            expertise_indices = rng.sample(
                range(len(AVAILABLE_DIRECTIONS)), min(num_expertise, len(AVAILABLE_DIRECTIONS)))
            blueprint.append({
                'institution': institution,
                'researcher_name': f"{institution}_researcher_{r_idx}",
                'strategy': strategy,
                'expertise_indices': expertise_indices,
                'expertise_topics': [AVAILABLE_DIRECTIONS[i].topic for i in expertise_indices],
            })
    return blueprint


def build_influx_cohort(year: int, seed: int) -> List[Dict]:
    """Deterministic yearly junior-researcher cohort for the DEFAULT population
    (influx2x2 experiment).

    Staggered exact-10% schedule: the 60 selected institutions (10 rich + 20
    normal universities, 10 rich + 20 normal companies) are split into two
    stratified halves by index parity WITHIN each tier group, so each half is
    5 rich-uni + 10 normal-uni + 5 rich-co + 10 normal-co and matches the
    ecosystem composition. H1 (even indices) hires 1 junior per institution in
    ODD sim years (1,3,5,...); H2 (odd indices) in EVEN years -> exactly
    +30/year = 10% of the initial 300, N(t) = 300 + 30t (linear growth).

    Entrant IDs continue per-institution numbering in closed form (founders are
    indices 0-4): researcher index = 5 + (# scheduled hire-years for that
    institution strictly before `year`), i.e. 5,6,7,... over the run —
    collision-free and independent of runtime population state (robust to
    attrition culls and checkpoint resume).

    Expertise: 3 draws from a dedicated random.Random seeded with
    derive_seed(seed, 'influx', year, institution) — the GLOBAL random stream
    is never touched, so influx-off runs stay byte-identical to legacy runs.

    Pure function — unit-testable without any Simulation or LLM.

    Returns:
        List of rows: {institution, kind ('university'|'industry'),
        funding_level, researcher_name, expertise_indices}
    """
    groups = [
        ('university', RICH_UNIVERSITIES[:10], RICH_UNIVERSITY_FUNDING),
        ('university', NORMAL_UNIVERSITIES[:20], NORMAL_UNIVERSITY_FUNDING),
        ('industry', RICH_COMPANIES[:10], RICH_COMPANY_FUNDING),
        ('industry', NORMAL_COMPANIES[:20], NORMAL_COMPANY_FUNDING),
    ]
    cohort = []
    for kind, names, funding_level in groups:
        for idx, institution in enumerate(names):
            in_h1 = (idx % 2 == 0)
            hires_now = (year % 2 == 1) if in_h1 else (year % 2 == 0)
            if not hires_now:
                continue
            prior_hires = sum(1 for h in range(1, year) if ((h % 2 == 1) == in_h1))
            rng = random.Random(derive_seed(seed, 'influx', year, institution))
            expertise_indices = rng.sample(range(len(AVAILABLE_DIRECTIONS)), 3)
            cohort.append({
                'institution': institution,
                'kind': kind,
                'funding_level': funding_level,
                'researcher_name': f"{institution}_researcher_{5 + prior_hires}",
                'expertise_indices': expertise_indices,
            })
    return cohort


def charge_application_costs(university_applications: List[Dict], app_cost: int,
                             year: int) -> List[Dict]:
    """Matthew-RD cost intervention: charge the frozen GRANT-APPLICATION cost
    at submission time, BEFORE the funding result is known, to winners and
    losers alike, once per application (incl. multiple same-year apps).

    Feasibility rule (mirrors the paper-writing minimum-funding gate): an agent
    that cannot pay withdraws the application — no charge, no debt, no negative
    balances. app_cost <= 0 (the default) returns immediately with the
    applications unmodified, reproducing pre-intervention behavior exactly.
    Paper resubmission mechanics are untouched.

    Mutates `university_applications` in place (withdrawn apps removed) and
    returns per-application financial event records.
    """
    from utopia.funding.accounting import apply_resource_change, stable_id
    events = []
    if app_cost <= 0:
        return events
    for apps_dict in university_applications:
        for program_id in list(apps_dict.keys()):
            app_data = apps_dict[program_id]
            if not app_data.get('submit'):
                continue
            agent = app_data['author']
            pre_balance = agent.resources
            ledger = getattr(agent, 'resource_ledger', None)
            prior = (ledger.event(stable_id(year, agent.id, 'funding_application_cost',
                                            None, None, program_id, None))
                     if ledger is not None else None)
            if prior is not None:
                pre_balance = prior['balance_before']
            if prior is None and pre_balance < app_cost:
                if getattr(agent, 'resource_ledger', None) is not None:
                    apply_resource_change(
                        agent, 0, category='funding_application_cost', year=year,
                        program=program_id, reason='withdrawn_unaffordable')
                del apps_dict[program_id]
                events.append({
                    'year': year, 'applicant_id': agent.id,
                    'program_id': program_id, 'pre_balance': pre_balance,
                    'cost_charged': 0, 'post_cost_balance': pre_balance,
                    'withdrawn_unaffordable': True,
                })
                continue
            apply_resource_change(agent, -app_cost, category='funding_application_cost',
                                  year=year, program=program_id)
            events.append({
                'year': year, 'applicant_id': agent.id,
                'program_id': program_id, 'pre_balance': pre_balance,
                'cost_charged': app_cost,
                'post_cost_balance': prior['balance_after'] if prior is not None else agent.resources,
                'withdrawn_unaffordable': False,
            })
    return events


def argparse_namespace_placeholder():
    """Empty namespace used when Simulation is built without CLI args (tests)."""
    import argparse
    return argparse.Namespace()


class Simulation:
    """Main simulation orchestrator for the academic ecosystem"""

    def __init__(
            self,
            llm,
            num_years: int = 5,
            output_dir: str = "outputs/",
            args=None,
            **kwargs
    ):

        """Initialize the simulation

        Args:
            num_years: Number of years to simulate
            output_dir: Directory to save simulation results
            debug: Enable debug mode
            industry_funding_mode: Funding mode for industry researchers - "performance" or "consistent"
            always_rerun: If True, ignore all checkpoints and start from scratch
            args: Command-line arguments object (optional)
        """
        self.llm: BaseLLM = llm
        self.num_years = num_years
        self.output_dir = output_dir
        self.current_year = 1
        self.debug = kwargs.get('debug', False)
        self.verbose = kwargs.get('verbose', False)
        self.citation_tracker = CitationTracker()
        self.agent_tracker = AgentTracker()
        
        assert kwargs.get('funding_allocation_mode') in ['fixed', 'consumption_based'], f"Invalid funding allocation mode: {kwargs.get('funding_allocation_mode', 'consumption_based')}"
        self.funding_tracker = FundingTracker(funding_allocation_mode=kwargs.get('funding_allocation_mode'))
        self.industry_funding_mode = kwargs.get('industry_funding_mode', 'performance')
        self.always_rerun = kwargs.get('always_rerun', False)
        self.use_langchain = kwargs.get('use_langchain', False)
        self.max_retries = kwargs.get('max_retries', 5)
        self.weighted_funding_assignment = kwargs.get('weighted_funding_assignment', False)
        self.funding_allocation_mode = kwargs.get('funding_allocation_mode', 'consumption_based')
        self.experiment_name = kwargs.get('experiment_name', 'default')
        self.collaboration_mode = kwargs.get('collaboration_mode',
                                              SIMULATION_CONFIG['preferential_attachment_experiment']['collaboration_mode'])
        self.collaboration_network_growth_mode = kwargs.get('collaboration_network_growth_mode', 'legacy')

        assert self.experiment_name.startswith(('default', 'acl-acceptance-rates', 'acl_schedule_smoke', 'influx', 'scale')) or self.experiment_name in ['exploration_vs_exploitation', 'preferential_attachment'], f"Invalid experiment name: {self.experiment_name}"

        # ---- Scale-expansion design factors (all default = legacy behaviour) ----
        _a = args if args is not None else argparse_namespace_placeholder()
        self.papers_per_project = int(getattr(_a, 'papers_per_project', 1) or 1)
        from utopia.funding.accounting import resolve_cost_policy, ResourceLedger
        self.cost_policy = resolve_cost_policy(_a, SIMULATION_CONFIG)
        self.resource_ledger = ResourceLedger() if self.cost_policy['log_resource_ledger'] else None
        self.funding_budget_mode = getattr(_a, 'funding_budget_mode', 'track') or 'track'
        self.funding_budget_slots_frac = getattr(_a, 'funding_budget_slots_frac', None)
        self.reviewer_capacity = getattr(_a, 'reviewer_capacity', None)
        self.reviews_per_paper = int(getattr(_a, 'reviews_per_paper', 3) or 3)
        self.reviewer_matching = getattr(_a, 'reviewer_matching', 'random') or 'random'
        self.review_policy = getattr(_a, 'review_policy', 'persona') or 'persona'
        self.review_score_mode = getattr(_a, 'review_score_mode', 'int') or 'int'
        self.acceptance_tiebreak = getattr(_a, 'acceptance_tiebreak', 'stable') or 'stable'
        self.acceptance_mode = getattr(_a, 'acceptance_mode', 'rate') or 'rate'
        # Capacity-aware reviewer assignment replaces the legacy random.sample only when
        # one of these deviates from its legacy value (keeps the global RNG stream intact).
        self._capacity_mode_on = (self.reviewer_capacity is not None
                                  or self.reviews_per_paper != 3
                                  or self.reviewer_matching != 'random')
        # N0: initial university / industry populations, anchors for --funding_budget_mode fixed
        self.initial_university_count = None
        self.initial_industry_count = None

        # Store args for backward compatibility with existing code
        if args is None:
            # Create a simple namespace with experiment_name if args not provided
            import argparse
            self.args = argparse.Namespace(experiment_name=self.experiment_name)
        else:
            self.args = args

        # Create output directory

        # Initialize components
        self.ecosystem = MultiAgentEcosystem(
            agent_configs={},
        )
        self.conference_system: ConferenceSystem = None
        self.industry_funding_system = IndustryFundingSystem(
            base_budget=SIMULATION_CONFIG['funding']['industry_base_budget'])
        self.yearly_results: List[Dict] = []
        self.paper_tracker = PaperTracker(output_dir=output_dir)  # Track all submitted papers
        self.submitted_paper_ids: Set[str] = set()  # Track all submitted papers across years

        t0 = time.time()
        rag_device = getattr(self.args, 'rag_device', None)
        rag_config = RAGConfig(
            use_langchain=self.use_langchain,
            cache_dir=os.path.join(getattr(self.args, 'data_cache_dir', 'data/cache'), 'rag'),
            **({'dataset_revision': self.args.dataset_revision} if getattr(self.args, 'dataset_revision', None) else {}),
            **({'device': rag_device} if rag_device else {}))
        self.start_year = kwargs.get('start_year', 2016)
        self.rag: RAG = RAG(config=rag_config, debug=self.debug, model=None, start_year=self.start_year)
        logger.info(f"RAG initialized in {time.time() - t0:.1f}s (device={rag_config.device})")

        logging_config = {**SIMULATION_CONFIG, 'wandb': {
            **SIMULATION_CONFIG['wandb'],
            'enabled': bool(getattr(self.args, 'wandb', False))}}
        self.wandb_logger = WandbLogger(logging_config, self.args)

        # Exploration experiment trackers (lazily initialized)
        self.embedding_tracker = None
        self.cd_calculator = None
        self.exploration_metrics = None
        self.keyword_extractor = None

        # Exploration experiment: paper novelty scores cache
        self._paper_novelty_scores = {}  # paper_id -> (novelty_score, bucket)
        # Latent citation potential; metadata only — NEVER multiplied into
        # observed citation counts (plan 4.5). Used solely for candidate
        # reranking in the citation intervention.
        self._citation_potential_multipliers = {}  # paper_id -> multiplier
        self._pending_paper_metadata = []  # (paper_dict, author_id, year) queue for batch processing
        self._direction_records = {}  # (agent_id, year) -> {'topic', 'distance', 'switched'}
        self._direction_fallback_stats = {}  # year -> fallback counter dict

        # Preferential attachment experiment trackers (lazily initialized)
        self.collaboration_tracker = None
        self.network_metrics = None
        self._collaboration_pairs = {}  # {year: {lead_id: co_author_id}}

    @property
    def is_exploration_experiment(self) -> bool:
        return self.experiment_name == 'exploration_vs_exploitation'

    def initialize_agents(self):
        """Initialize the agent population

        Creates multiple researchers mapped to universities and companies.
        """
        logger.info("Initializing agent population...")

        if getattr(self.args, 'population_mode', 'default') == 'university_only':
            return self._initialize_university_only_population()

        # Rosters hoisted to module constants (shared with build_influx_cohort)
        rich_universities = RICH_UNIVERSITIES
        normal_universities = NORMAL_UNIVERSITIES
        if self.debug:
            universities = rich_universities[:3] + normal_universities[:3]
            num_researchers = 5
        else:
            universities = rich_universities[:10] + normal_universities[:20]
            num_researchers = 5

        logger.info(f"Selecting {len(universities)} universities")

        counter = 0
        for university in universities:
            # Create 2-3 researchers per university]
            researcher_names = []

            for i in range(num_researchers):
                if self.is_exploration_experiment:
                    expertise = random.sample(AVAILABLE_DIRECTIONS, min(3, len(AVAILABLE_DIRECTIONS)))
                elif self.debug:
                    expertise = [AVAILABLE_DIRECTIONS[0]]
                else:
                    expertise = random.sample(AVAILABLE_DIRECTIONS, 3)
                # researcher_name = f"university_researcher_{counter}"
                researcher_name = f"{university}_researcher_{i}"

                funding_level = (RICH_UNIVERSITY_FUNDING if university in rich_universities
                                 else NORMAL_UNIVERSITY_FUNDING)

                # Assign exploration strategy if experiment is enabled
                if self.is_exploration_experiment:
                    bucket = counter % 10
                    if bucket < 2:        # 0,1 -> 20%
                        exploration_strategy = "explorer"
                    elif bucket < 8:      # 2-7 -> 60%
                        exploration_strategy = "exploiter"
                    else:                 # 8,9 -> 20%
                        exploration_strategy = "cautious_explorer"
                else:
                    exploration_strategy = "balanced"

                researcher = UniversityResearcher(
                    researcher_name=researcher_name,
                    university_name=university,
                    funding_level=funding_level,
                    expertise=expertise,
                    llm=self.llm,
                    generate_research_proposal=True,
                    exploration_strategy=exploration_strategy,
                )
                self.ecosystem.add_agent(researcher)
                counter += 1
                logger.info(f"Added {researcher.id} from {university}")
                researcher_names.append(researcher_name)

            researcher_names = set(researcher_names)
            for researcher_name in researcher_names:
                researcher = self.ecosystem.get_agent_by_id(researcher_name)
                researcher.add_conflict_of_interest(researcher_names - {researcher_name})

                if self.verbose:
                    logger.debug(f"Added {len(researcher.conflict_of_interest)} CoI for {researcher_name}")

        rich_companies = RICH_COMPANIES
        normal_companies = NORMAL_COMPANIES

        if self.debug:
            companies = rich_companies[:3] + normal_companies[:3]
            print(f"TODO: select {len(companies)} companies ONLY")
            num_researchers = 2

        else:
            companies = rich_companies[:10] + normal_companies[:20]
            num_researchers = 5

        counter = 0
        for company in companies:
            # Create 2-3 researchers per company
            researcher_names = []

            funding_level = (RICH_COMPANY_FUNDING if company in rich_companies
                             else NORMAL_COMPANY_FUNDING)

            for i in range(num_researchers):
                if self.is_exploration_experiment:
                    expertise = random.sample(AVAILABLE_DIRECTIONS, min(3, len(AVAILABLE_DIRECTIONS)))
                elif self.debug:
                    expertise = [AVAILABLE_DIRECTIONS[0]]
                else:
                    expertise = random.sample(AVAILABLE_DIRECTIONS, 3)
                researcher_name = f"{company}_researcher_{i}"

                # Assign exploration strategy for industry researchers too
                if self.is_exploration_experiment:
                    bucket = counter % 10
                    if bucket < 2:
                        exploration_strategy = "explorer"
                    elif bucket < 8:
                        exploration_strategy = "exploiter"
                    else:
                        exploration_strategy = "cautious_explorer"
                else:
                    exploration_strategy = "balanced"

                researcher = IndustryResearcher(
                    researcher_name=researcher_name,
                    company_name=company,
                    funding_level=funding_level,
                    expertise=expertise,
                    llm=self.llm,
                    funding_mode=self.industry_funding_mode,
                    exploration_strategy=exploration_strategy,
                )
                self.ecosystem.add_agent(researcher)
                logger.info(f"Added {researcher.id} from {company} (funding mode: {self.industry_funding_mode})")
                counter += 1
                researcher_names.append(researcher_name)

            researchers_of_company = set(researcher_names)

            for researcher_name in researchers_of_company:
                researcher = self.ecosystem.get_agent_by_id(researcher_name)
                researcher.add_conflict_of_interest(researchers_of_company - {researcher_name})
                logger.debug(f"Added {len(researcher.conflict_of_interest)} CoI for {researcher_name}")

        """
        # Add some freelance researchers
        for i in range(3):
            freelancer = FreelancerAgent(
                researcher_name=f"freelancer_{counter}",
                funding_level=5,
                expertise=random.sample(AVAILABLE_DIRECTIONS, 3),
                llm=self.llm
            )
            self.ecosystem.add_agent(freelancer)
            logger.info(f"Added {freelancer.id}")
        """

        self._record_initial_university_count()
        self._add_funding_agencies()

        logger.info(f"Total agents initialized: {len(self.ecosystem.agent_population)}")

    def _record_initial_university_count(self):
        """N0 per sector at initialization: anchors for --funding_budget_mode fixed
        (agency slots = round(f x N0_university); industry pool = round(f x N0_industry) x base budget)."""
        self.initial_university_count = sum(
            1 for a in self.ecosystem.agent_population.values() if isinstance(a, UniversityResearcher))
        self.initial_industry_count = sum(
            1 for a in self.ecosystem.agent_population.values() if isinstance(a, IndustryResearcher))

    def _fixed_industry_pool_per_unit(self, denominator: int) -> Tuple[int, Dict]:
        """--funding_budget_mode fixed for the industry sector (symmetric with the agency slots).

        Annual industry pool = round(f x N0_industry) x industry_base_budget, paid out per
        accepted-paper maturity unit: per_unit = max(pool // denominator, 1) where denominator
        is this year's total maturity of accepted industry papers (the legacy rule pays
        base_budget per unit, so the pool then grows with output). Returns (per_unit, info).
        """
        assert self.initial_industry_count is not None, "N0_industry unknown: population not initialized"
        base = SIMULATION_CONFIG['funding']['industry_base_budget']
        pool = int(round(self.funding_budget_slots_frac * self.initial_industry_count)) * base
        per_unit = max(pool // denominator, 1) if denominator > 0 else 0
        info = {'mode': 'fixed', 'slots_frac': self.funding_budget_slots_frac,
                'initial_industry_count': self.initial_industry_count, 'pool': pool,
                'maturity_units_accepted': int(denominator), 'per_unit': per_unit,
                'paid': int(per_unit * denominator)}
        return per_unit, info

    def _add_funding_agencies(self):
        """Add the funding agencies shared by all population modes.

        NSF: foundational scientific advancement (patient, lower thresholds).
        DARPA: application-driven projects (demanding, higher thresholds).
        """
        self.ecosystem.add_agent(FundingAgency(
            agency_name="NSF",
            default_funding_rate=0.23,  # NSF has a funding rate between [0.2, 0.25]
            llm=self.llm
        ))

        self.ecosystem.add_agent(FundingAgency(
            agency_name="DARPA",
            default_funding_rate=0.1,  # DARPA has a funding rate between [0.05, 0.15]
            llm=self.llm
        ))

    def _initialize_university_only_population(self):
        """Balanced university-only population for the exploration experiment (plan 4.2).

        Equal initial funding, synthetic institutions, and exactly the configured
        within-institution strategy mix, all derived deterministically from the
        run seed via build_population_blueprint(). The blueprint is written to
        the checkpoint directory as population_blueprint.csv for audit/balance checks.
        """
        num_institutions = getattr(self.args, 'num_institutions', None) or \
            SIMULATION_CONFIG['exploration_experiment']['num_institutions']
        per_institution = getattr(self.args, 'researchers_per_institution', None) or \
            SIMULATION_CONFIG['exploration_experiment']['researchers_per_institution']
        initial_funding = getattr(self.args, 'initial_funding', None) or \
            SIMULATION_CONFIG['exploration_experiment']['initial_funding']
        run_seed = getattr(self.args, 'seed', 42)

        blueprint = build_population_blueprint(
            num_institutions, per_institution, run_seed,
            strategy_mix=getattr(self.args, 'strategy_mix_list', None))

        for row in blueprint:
            researcher = UniversityResearcher(
                researcher_name=row['researcher_name'],
                university_name=row['institution'],
                funding_level=initial_funding,
                expertise=[AVAILABLE_DIRECTIONS[i] for i in row['expertise_indices']],
                llm=self.llm,
                generate_research_proposal=True,
                exploration_strategy=row['strategy'],
            )
            self.ecosystem.add_agent(researcher)

        # Within-institution conflicts of interest
        by_institution = defaultdict(set)
        for row in blueprint:
            by_institution[row['institution']].add(row['researcher_name'])
        for institution, names in by_institution.items():
            for name in names:
                self.ecosystem.get_agent_by_id(name).add_conflict_of_interest(names - {name})

        # Persist blueprint for audit and balance checks
        blueprint_path = os.path.join(self.output_dir, 'population_blueprint.csv')
        pd.DataFrame(blueprint).to_csv(blueprint_path, index=False)
        logger.info(f"University-only population: {len(blueprint)} researchers at "
                    f"{num_institutions} institutions (equal funding {initial_funding}); "
                    f"blueprint saved to {blueprint_path}")

        self._record_initial_university_count()
        self._add_funding_agencies()

        logger.info(f"Total agents initialized: {len(self.ecosystem.agent_population)}")

    def calculate_average_funding_level(self):
        """Calculate the average funding level of all agents"""
        funding_levels = [agent.resources for agent_id, agent in self.ecosystem.agent_population.items() if
                          getattr(agent, "resources", None) is not None]
        return np.mean(funding_levels)

    def _initialize_experiment_trackers(self):
        """Lazy initialization of exploration experiment trackers"""
        if not self.is_exploration_experiment:
            return

        cache_root = getattr(self.args, 'data_cache_dir', os.path.join('data', 'cache'))

        if self.embedding_tracker is None:
            from utopia.metrics.embedding_tracker import EmbeddingTracker
            exp_config = SIMULATION_CONFIG['exploration_experiment']
            self.embedding_tracker = EmbeddingTracker(
                model_name=exp_config['embedding_model'],
                cache_dir=os.path.join(cache_root, 'embeddings'))
            logger.info("Initialized EmbeddingTracker")

        # Empirical near/far thresholds from the direction-pair distance
        # distribution (plan 4.7); computed once, then frozen in checkpoints.
        if self.embedding_tracker.near_threshold is None:
            thresholds = self.embedding_tracker.compute_direction_thresholds(AVAILABLE_DIRECTIONS)
            logger.info(f"Empirical novelty thresholds: {thresholds}")

        # Expertise centroids as pre-first-paper career reference (plan 4.8)
        for agent in self.ecosystem.agent_population.values():
            if (isinstance(agent, (UniversityResearcher, IndustryResearcher))
                    and agent.id not in self.embedding_tracker.expertise_centroids
                    and getattr(agent, 'expertise', None)):
                texts = [d.topic + " " + " ".join(d.keywords or []) for d in agent.expertise]
                self.embedding_tracker.register_agent_expertise(agent.id, texts)

        if self.cd_calculator is None:
            from utopia.metrics.cd_index import CDIndexCalculator
            self.cd_calculator = CDIndexCalculator(self.citation_tracker, self.paper_tracker)
            logger.info("Initialized CDIndexCalculator")

        if self.exploration_metrics is None:
            from utopia.metrics.exploration_metrics import ExplorationMetrics
            self.exploration_metrics = ExplorationMetrics()
            logger.info("Initialized ExplorationMetrics")

        if self.keyword_extractor is None:
            from utopia.data.keyword_extractor import KeywordExtractor
            self.keyword_extractor = KeywordExtractor(
                cache_dir=os.path.join(cache_root, 'keywords'),
                model_id=getattr(self.llm, 'model_name', 'unknown'))
            logger.info("Initialized KeywordExtractor")

    def _parse_intention(self, result_json, direction_dict: Dict) -> str:
        """Parse intention from LLM response, falling back to old-style query on failure."""
        if result_json and isinstance(result_json, dict) and result_json.get('intention'):
            return result_json['intention']
        return self._build_fallback_query(direction_dict)

    def _build_fallback_query(self, direction_dict: Dict) -> str:
        """Build old-style RAG query when intention generation fails."""
        if direction_dict.get('detailed_focus'):
            return direction_dict['detailed_focus']
        return " ".join(direction_dict['direction'].keywords)

    def _record_paper_metadata(self, paper_dict: Dict, author_id: str, year: int):
        """Enqueue a paper for batched metadata processing (plan 4.6).

        The actual keyword extraction and embedding happen once per year in
        _record_paper_metadata_batch() — never one LLM/encode call per paper.
        """
        if not self.is_exploration_experiment:
            return
        abstract = paper_dict.get('abstract', '')
        if not abstract or not abstract.strip():
            return
        self._pending_paper_metadata.append((paper_dict, author_id, year))

    def _record_paper_metadata_batch(self, year: int):
        """Process all enqueued papers with batched keywords + embeddings + novelty."""
        if not self.is_exploration_experiment or not self._pending_paper_metadata:
            return

        self._initialize_experiment_trackers()
        exp_config = SIMULATION_CONFIG['exploration_experiment']
        pending = self._pending_paper_metadata
        self._pending_paper_metadata = []

        # Batched keyword extraction (LLM, cached by paper_id)
        kw_results = self.keyword_extractor.extract_keywords_batch(
            self.llm,
            [(p['id'], p.get('abstract', '')) for p, _, _ in pending],
            n_keywords=exp_config['num_keywords'],
        )
        n_fallback = sum(1 for r in kw_results.values() if r.get('provenance') == 'fallback')
        if n_fallback:
            logger.warning(f"Keyword extraction fell back for {n_fallback}/{len(pending)} papers")

        # Batched embedding (one encode call, disk-cached)
        self.embedding_tracker.add_paper_embeddings_batch(
            [(p['id'], p.get('abstract', ''), author_id, yr) for p, author_id, yr in pending])
        self.embedding_tracker.flush_cache()

        for paper_dict, author_id, yr in pending:
            kw = kw_results.get(paper_dict['id'], {})
            paper_dict['keywords'] = kw.get('keywords', [])
            paper_dict['keywords_in_abstract'] = kw.get('keywords_in_abstract', [])
            paper_dict['keywords_not_in_abstract'] = kw.get('keywords_not_in_abstract', [])
            paper_dict['keyword_provenance'] = kw.get('provenance', 'missing')

            novelty = self.embedding_tracker.compute_novelty_score(
                paper_dict['id'], author_id,
                history_window_years=exp_config.get('history_window_years', 3)
            )
            if novelty is not None:
                paper_dict['novelty_score'] = novelty[0]
                paper_dict['career_distance_bucket'] = novelty[1]
                self._paper_novelty_scores[paper_dict['id']] = novelty
            else:
                paper_dict['novelty_score'] = 0.0
                paper_dict['career_distance_bucket'] = 'unknown'

            agent = self.ecosystem.get_agent_by_id(author_id)
            paper_dict['exploration_strategy'] = getattr(agent, 'exploration_strategy', 'balanced')

            # Papers already archived before metadata was computed need an update
            archived = self.paper_tracker.papers_by_id.get(paper_dict['id'])
            if archived is not None:
                for field in ('keywords', 'novelty_score', 'career_distance_bucket',
                              'exploration_strategy', 'keyword_provenance'):
                    if hasattr(archived, field):
                        setattr(archived, field, paper_dict.get(field))

        logger.info(f"Year {year}: batched metadata for {len(pending)} papers "
                    f"({n_fallback} keyword fallbacks)")

    def _collect_exploration_metrics(self, year: int):
        """Collect exploration vs exploitation metrics at end of year

        Args:
            year: Current year number
        """
        if not self.is_exploration_experiment:
            return

        self._initialize_experiment_trackers()

        exp_config = SIMULATION_CONFIG['exploration_experiment']

        # Compute topic distances for papers submitted this year (batch)
        papers_this_year = [(p.id, p.all_author_ids[0]) for p in self.paper_tracker.papers_by_id.values()
                            if p.year == year]
        paper_ids_this_year = [pid for pid, _ in papers_this_year]

        if not paper_ids_this_year:
            logger.info(f"Year {year}: No new papers; still collecting paper-age panel")
            self._collect_paper_age_panel(year)
            self.exploration_metrics.export_tables(getattr(self.args, 'docs_dir', self.output_dir))
            return

        distances = self.embedding_tracker.compute_distances_batch(papers_this_year)
        for paper_id, author_id in papers_this_year:
            self.exploration_metrics.record_topic_distance(author_id, year, distances[paper_id])

        # Compute CD index for papers with sufficient citations
        min_citations = exp_config['min_citations_for_cd']

        for paper_id in paper_ids_this_year:
            citations = len(self.citation_tracker.get_citations_to_paper(paper_id))

            if citations >= min_citations:
                cd_score = self.cd_calculator.calculate_cd_index(paper_id)

                self.exploration_metrics.record_cd_index(
                    paper_id=paper_id,
                    year=year,
                    cd_score=cd_score
                )

        # Build lookup structures for per-strategy metrics
        accepted_paper_ids = set()
        for conf in self.conference_system.conferences:
            for paper in conf.decisions.get('accept', []):
                accepted_paper_ids.add(paper['id'])

        # Collect all citation counts for accepted papers this year.
        # Observed counts are ALWAYS the citation-graph in-degree — the latent
        # citation-potential multiplier is metadata and is never applied here
        # (plan 4.5).
        all_citation_counts = []
        paper_citation_map = {}
        for pid in paper_ids_this_year:
            if pid in accepted_paper_ids:
                count = len(self.citation_tracker.get_citations_to_paper(pid))
                all_citation_counts.append(count)
                paper_citation_map[pid] = count

        # Compute hit paper threshold
        hit_percentile = exp_config.get('hit_paper_percentile', 0.95)
        hit_threshold = np.percentile(all_citation_counts, hit_percentile * 100) if all_citation_counts else float('inf')

        # Top-10% citation threshold
        top10_threshold = np.percentile(all_citation_counts, 90) if all_citation_counts else float('inf')

        # Aggregate metrics by strategy
        for strategy in exp_config['strategies']:
            agents = [a for a in self.ecosystem.agent_population.values()
                     if (isinstance(a, (UniversityResearcher, IndustryResearcher))
                         and hasattr(a, 'exploration_strategy')
                         and a.exploration_strategy == strategy)]

            if not agents:
                continue

            # Topic distances
            topic_distances = []
            for a in agents:
                avg_dist = self.exploration_metrics.get_author_avg_distance(a.id, year)
                if avg_dist is not None:
                    topic_distances.append(avg_dist)

            if topic_distances:
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='avg_topic_distance', value=float(np.mean(topic_distances))
                )

            # Per-strategy acceptance rate
            strategy_agent_ids = {a.id for a in agents}
            strategy_papers = [pid for pid, aid in papers_this_year if aid in strategy_agent_ids]
            strategy_accepted = [pid for pid in strategy_papers if pid in accepted_paper_ids]
            if strategy_papers:
                acc_rate = len(strategy_accepted) / len(strategy_papers)
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='avg_acceptance_rate', value=acc_rate
                )

            # Per-strategy funding
            strategy_funding = [a.resources for a in agents if hasattr(a, 'resources')]
            if strategy_funding:
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='avg_funding', value=float(np.mean(strategy_funding))
                )

            # Per-strategy citations
            strategy_citations = [paper_citation_map[pid] for pid in strategy_accepted if pid in paper_citation_map]
            if strategy_citations:
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='avg_citations', value=float(np.mean(strategy_citations))
                )
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='median_citations', value=float(np.median(strategy_citations))
                )

                # Citation Gini
                sorted_cites = sorted(strategy_citations)
                n = len(sorted_cites)
                if n > 0 and sum(sorted_cites) > 0:
                    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_cites)) / (n * np.sum(sorted_cites))) - (n + 1) / n
                    self.exploration_metrics.record_strategy_metric(
                        strategy=strategy, year=year,
                        metric='citation_gini', value=float(gini)
                    )

            # Hit paper rate (citations > 95th percentile)
            hit_count = sum(1 for c in strategy_citations if c >= hit_threshold) if strategy_citations else 0
            if strategy_accepted:
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='hit_paper_rate', value=hit_count / len(strategy_accepted)
                )

            # Top-10% citation share
            top10_count = sum(1 for c in strategy_citations if c >= top10_threshold) if strategy_citations else 0
            total_top10 = sum(1 for c in all_citation_counts if c >= top10_threshold) if all_citation_counts else 0
            if total_top10 > 0:
                self.exploration_metrics.record_strategy_metric(
                    strategy=strategy, year=year,
                    metric='top10_citation_share', value=top10_count / total_top10
                )

            # Record per-agent stats + structured agent_year rows
            for a in agents:
                agent_papers_yr = [pid for pid, aid in papers_this_year if aid == a.id]
                agent_accepted_yr = [pid for pid in agent_papers_yr if pid in accepted_paper_ids]
                agent_citations = [paper_citation_map.get(pid, 0) for pid in agent_accepted_yr]
                dir_record = self._direction_records.get((a.id, year), {})

                stats = {
                    'strategy': strategy,
                    'num_papers': len(agent_papers_yr),
                    'num_accepted': len(agent_accepted_yr),
                    'acceptance_rate': len(agent_accepted_yr) / len(agent_papers_yr) if agent_papers_yr else 0,
                    'funding': getattr(a, 'resources', 0),
                    'total_citations': sum(agent_citations),
                    'hit_papers': sum(1 for c in agent_citations if c >= hit_threshold),
                }
                self.exploration_metrics.record_agent_stats(a.id, year, stats)
                self.exploration_metrics.add_row('agent_year', {
                    'agent_id': a.id, 'year': year,
                    'institution': getattr(a, 'university_name', None) or getattr(a, 'company_name', None),
                    'is_active': getattr(a, 'is_active', True),
                    'direction_topic': dir_record.get('topic'),
                    'direction_distance': dir_record.get('distance'),
                    'topic_switched': dir_record.get('switched'),
                    'avg_career_distance': self.exploration_metrics.get_author_avg_distance(a.id, year),
                    **stats,
                })

        # Structured paper rows for this year's papers
        for pid, aid in papers_this_year:
            paper = self.paper_tracker.papers_by_id[pid]
            novelty = self._paper_novelty_scores.get(pid)
            agent = self.ecosystem.get_agent_by_id(aid)
            last_review = paper.review_history[-1] if paper.review_history else {}
            self.exploration_metrics.add_row('paper', {
                'paper_id': pid, 'author_id': aid, 'year': year,
                'strategy': getattr(agent, 'exploration_strategy', 'balanced') if agent else None,
                'institution': (getattr(agent, 'university_name', None)
                                or getattr(agent, 'company_name', None)) if agent else None,
                'accepted': pid in accepted_paper_ids,
                'review_score': last_review.get('score'),
                'conference': getattr(paper, 'conference', None),
                'novelty_score': novelty[0] if novelty else None,
                'distance_bucket': novelty[1] if novelty else None,
                'career_distance': distances.get(pid),
                'keyword_provenance': getattr(paper, 'keyword_provenance', None),
                'citation_potential_multiplier': self._citation_potential_multipliers.get(pid),
            })

        # Ecosystem-year row (diversity, inequality, population)
        chosen_topics = [r['topic'] for (aid, y), r in self._direction_records.items() if y == year]
        topic_counts = Counter(chosen_topics)
        if topic_counts:
            probs = np.array(list(topic_counts.values()), dtype=float)
            probs /= probs.sum()
            entropy = float(-(probs * np.log(probs)).sum())
            norm_entropy = entropy / np.log(len(AVAILABLE_DIRECTIONS))
        else:
            entropy, norm_entropy = None, None
        active_researchers = [a for a in self.ecosystem.agent_population.values()
                              if isinstance(a, (UniversityResearcher, IndustryResearcher))
                              and getattr(a, 'is_active', True)]
        all_funding = [a.resources for a in active_researchers if getattr(a, 'resources', None) is not None]
        cumulative_citation_counts = [len(v) for v in self.citation_tracker.citations.values()]
        self.exploration_metrics.add_row('ecosystem_year', {
            'year': year,
            'topic_entropy': entropy,
            'topic_entropy_normalized': norm_entropy,
            'occupied_directions': len(topic_counts),
            'num_active_researchers': len(active_researchers),
            'num_submissions': len(paper_ids_this_year),
            'num_accepted': len([p for p in paper_ids_this_year if p in accepted_paper_ids]),
            'funding_gini': calculate_gini_coefficient(all_funding) if all_funding else None,
            'citation_gini': (calculate_gini_coefficient(cumulative_citation_counts)
                              if cumulative_citation_counts else None),
            'direction_fallback_stats': json.dumps(self._direction_fallback_stats.get(year, {})),
        })

        # Strategy-year rows from the aggregate metric dict
        for strategy in exp_config['strategies']:
            metrics_dict = dict(self.exploration_metrics.strategy_metrics[strategy].get(year, {}))
            if metrics_dict:
                self.exploration_metrics.add_row('strategy_year', {
                    'strategy': strategy, 'year': year, **metrics_dict})

        # Paper-age panel + annual CD recomputation for ALL prior accepted papers
        self._collect_paper_age_panel(year)

        # Persist structured tables (small; atomic rewrite per year)
        self.exploration_metrics.export_tables(getattr(self.args, 'docs_dir', self.output_dir))

        logger.info(f"Year {year}: Collected exploration metrics for {len(paper_ids_this_year)} papers")

    def _collect_paper_age_panel(self, year: int):
        """Append paper-age outcomes for every accepted paper each year (plan 4.10).

        Records cumulative citations (graph in-degree), time to first citation,
        and the corrected CD index (plan 4.9) at the paper's current age, so the
        analysis can compare papers at fixed ages and CD windows.
        """
        exp_config = SIMULATION_CONFIG['exploration_experiment']
        cd_min_denominator = exp_config['cd_min_denominator']
        age_windows = set(exp_config['citation_age_windows'])

        for pid, paper in self.paper_tracker.papers_by_id.items():
            if paper.status != 'accept' or paper.year > year:
                continue
            age = year - paper.year
            if age not in age_windows:
                continue
            citing = self.citation_tracker.get_papers_citing(pid)
            citing_years = [self.paper_tracker.papers_by_id[c].year
                            for c in citing if c in self.paper_tracker.papers_by_id]
            time_to_first = (min(citing_years) - paper.year) if citing_years else None
            cd_result = self.cd_calculator.calculate_cd_index(
                pid, cutoff_year=year, min_denominator=cd_min_denominator)
            aid = paper.all_author_ids[0]
            agent = self.ecosystem.get_agent_by_id(aid)
            if cd_result is not None:
                self.exploration_metrics.record_cd_index(pid, year, cd_result['cd'])
            self.exploration_metrics.add_row('paper_age', {
                'paper_id': pid, 'author_id': aid,
                'strategy': getattr(agent, 'exploration_strategy', 'balanced') if agent else None,
                'institution': (getattr(agent, 'university_name', None)
                                or getattr(agent, 'company_name', None)) if agent else None,
                'publication_year': paper.year,
                'observation_year': year,
                'age': age,
                'citations': len(citing),
                'time_to_first_citation': time_to_first,
                'cd': cd_result['cd'] if cd_result else None,
                'cd_denominator': cd_result['denominator'] if cd_result else None,
                'cd_n_disruptive': cd_result['n_disruptive'] if cd_result else None,
                'cd_n_consolidating': cd_result['n_consolidating'] if cd_result else None,
                'novelty_score': (self._paper_novelty_scores.get(pid) or (None,))[0],
            })

    def _run_integrity_checks(self, year: int) -> Dict:
        """Graph and temporal integrity checks, written per year (plan 11.4).

        Any failure is recorded (and loudly logged) rather than silently ignored.
        """
        checks = {'year': year, 'passed': True, 'failures': []}
        failures = check_citation_integrity(self.paper_tracker, self.citation_tracker)

        # No duplicate paper submissions across conferences in the same year
        seen = set()
        for conf in self.conference_system.conferences:
            for paper in conf.submitted_papers:
                key = (paper['id'], year)
                if key in seen:
                    failures.append(f"paper {paper['id']} submitted to multiple conferences in year {year}")
                seen.add(key)

        if failures:
            checks['passed'] = False
            checks['failures'] = failures
            for msg in failures:
                logger.error(f"[Integrity] {msg}")

        checks['n_papers'] = len(self.paper_tracker.papers_by_id)
        checks['n_citation_edges'] = sum(len(v) for v in self.citation_tracker.citations.values())

        docs_dir = getattr(self.args, 'docs_dir', self.output_dir)
        os.makedirs(docs_dir, exist_ok=True)
        path = os.path.join(docs_dir, 'integrity_checks.json')
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        existing = [c for c in existing if c.get('year') != year] + [checks]
        write_json_file(existing, path, indent=2)
        return checks

    # ==================== Preferential Attachment Experiment ====================

    def _initialize_preferential_attachment_trackers(self):
        """Lazily initialize trackers for the preferential attachment experiment."""
        if self.experiment_name != 'preferential_attachment':
            return
        if self.collaboration_tracker is None:
            from utopia.data.collaboration_tracker import CollaborationTracker
            self.collaboration_tracker = CollaborationTracker()
        if self.network_metrics is None:
            from utopia.metrics.network_metrics import NetworkMetrics
            self.network_metrics = NetworkMetrics()

    def _run_phase_1_5_collaboration_formation(self, year: int, directions: Dict,
                                                active_authors: List, year_results: Dict):
        """Phase 1.5: Form collaboration pairs for co-authored papers.

        Hybrid approach: heuristic scoring filters candidates, then LLM decides.
        Only runs for the preferential_attachment experiment.
        """
        if self.experiment_name != 'preferential_attachment':
            return

        self._initialize_preferential_attachment_trackers()

        pa_config = SIMULATION_CONFIG['preferential_attachment_experiment']
        top_k = pa_config['top_k_candidates']
        collaboration_mode = self.collaboration_mode

        # Build topic vectors for all active authors
        agent_topics = {}
        for agent in active_authors:
            direction = directions.get(agent.id)
            keywords = set()
            if direction and direction.get('direction'):
                d = direction['direction']
                keywords = set(getattr(d, 'keywords', []))
            expertise_topics = {e.topic if hasattr(e, 'topic') else str(e)
                                for e in getattr(agent, 'expertise', [])}
            agent_topics[agent.id] = keywords | expertise_topics

        def _get_institution(agent):
            return getattr(agent, 'university_name', None) or getattr(agent, 'company_name', None)

        # --- Step 1: Heuristic scoring for each agent ---
        agent_candidates = {}  # agent_id -> [(candidate_id, score, candidate_info_dict)]

        active_author_ids = set(a.id for a in active_authors)
        # growth_ctx is non-None only in the default-off 'expanding' mode
        # (prereg_v1_1_source_fix); 'legacy' keeps the original path untouched.
        growth_ctx = None
        if self.collaboration_network_growth_mode == 'expanding':
            agent_candidates, growth_ctx = self._expanding_candidate_selection(
                year, active_authors, agent_topics, _get_institution,
                collaboration_mode, pa_config)
        else:
            for agent in active_authors:
                topics_a = agent_topics.get(agent.id, set())
                if not topics_a:
                    continue

                inst_a = _get_institution(agent)
                recent_papers_a = self.paper_tracker.get_papers_by_author(agent.id)
                recent_topics_a = set()
                for p in sorted(recent_papers_a, key=lambda x: x.year, reverse=True)[:3]:
                    recent_topics_a.update(p.topics)

                scored = []
                for other in active_authors:
                    if other.id == agent.id:
                        continue
                    inst_b = _get_institution(other)

                    # Filter by collaboration mode
                    if collaboration_mode == 'intra_institute' and inst_a != inst_b:
                        continue
                    if collaboration_mode == 'cross_institute' and inst_a == inst_b:
                        continue

                    topics_b = agent_topics.get(other.id, set())
                    union = topics_a | topics_b
                    if not union:
                        continue

                    # Topic overlap (Jaccard): weight 0.4
                    topic_score = len(topics_a & topics_b) / len(union)

                    # Recent publication overlap: weight 0.4
                    recent_papers_b = self.paper_tracker.get_papers_by_author(other.id)
                    recent_topics_b = set()
                    for p in sorted(recent_papers_b, key=lambda x: x.year, reverse=True)[:3]:
                        recent_topics_b.update(p.topics)
                    pub_union = recent_topics_a | recent_topics_b
                    pub_score = len(recent_topics_a & recent_topics_b) / len(pub_union) if pub_union else 0.0

                    # Network proximity bonus: weight 0.2
                    net_score = 0.0
                    if self.collaboration_tracker.graph.has_node(agent.id) and self.collaboration_tracker.graph.has_node(other.id):
                        dist = self.collaboration_tracker.get_network_distance(agent.id, other.id)
                        if dist == 1:
                            net_score = 1.0
                        elif dist == 2:
                            net_score = 0.5

                    final_score = 0.4 * topic_score + 0.4 * pub_score + 0.2 * net_score
                    if final_score > pa_config['min_similarity_threshold']:
                        expertise_b = [e.topic if hasattr(e, 'topic') else str(e) for e in getattr(other, 'expertise', [])]
                        recent_paper_titles = [p.title for p in sorted(recent_papers_b, key=lambda x: x.year, reverse=True)[:3]]
                        scored.append((other.id, final_score, {
                            'id': other.id,
                            'institution': inst_b or 'Unknown',
                            'expertise': expertise_b,
                            'recent_papers': '; '.join(recent_paper_titles) if recent_paper_titles else 'None',
                            'score': final_score,
                        }))

                scored.sort(key=lambda x: x[1], reverse=True)
                agent_candidates[agent.id] = scored[:top_k]

        # --- Step 2: LLM collaboration decision ---
        supports_batching = hasattr(self.llm, 'generate_batch')
        collab_decisions = {}  # agent_id -> (collaborate, collaborator_id)

        if supports_batching:
            batch_prompts = []
            batch_metadata = []
            collab_response_format = None

            for agent in active_authors:
                candidates = agent_candidates.get(agent.id, [])
                if not candidates:
                    continue
                candidate_infos = [c[2] for c in candidates]
                prompt, response_format = agent.get_collaboration_decision_prompt(candidate_infos)
                batch_prompts.append(prompt)
                batch_metadata.append({'agent': agent, 'candidate_ids': [c[0] for c in candidates]})
                if collab_response_format is None:
                    collab_response_format = response_format

            if batch_prompts:
                batch_results = self.llm.generate_batch(
                    batch_prompts, response_format=collab_response_format,
                    desc=f"{len(batch_prompts)} collaboration decisions",
                    seed_ctx=('phase15_collab', year),
                )
                for i, (result, _) in enumerate(batch_results):
                    meta = batch_metadata[i]
                    agent_id = meta['agent'].id
                    if result and result.get('collaborate') and result.get('collaborator_id'):
                        cid = result['collaborator_id']
                        if cid in meta['candidate_ids']:
                            collab_decisions[agent_id] = (True, cid)
        else:
            for agent in active_authors:
                candidates = agent_candidates.get(agent.id, [])
                if not candidates:
                    continue
                candidate_infos = [c[2] for c in candidates]
                prompt, response_format = agent.get_collaboration_decision_prompt(candidate_infos)
                result, _ = self.llm.generate(prompt=prompt, response_format=response_format)
                if result and result.get('collaborate') and result.get('collaborator_id'):
                    cid = result['collaborator_id']
                    candidate_ids = [c[0] for c in candidates]
                    if cid in candidate_ids:
                        collab_decisions[agent.id] = (True, cid)

        # --- Step 3: Form pairs (mutual or one-sided matching) ---
        growth_stats = None
        if growth_ctx is not None:
            collaboration_pairs, growth_stats = self._expanding_form_pairs(
                year, collab_decisions, agent_candidates, growth_ctx)
        else:
            claimed = set()
            collaboration_pairs = {}  # lead_id -> co_author_id

            for agent_id, (_, chosen_id) in collab_decisions.items():
                if agent_id in claimed or chosen_id in claimed:
                    continue

                # Check if mutual (both chose each other)
                other_decision = collab_decisions.get(chosen_id)
                is_mutual = other_decision and other_decision[1] == agent_id

                # Check if one-sided (chosen_id didn't choose anyone else)
                chosen_available = chosen_id not in collab_decisions or is_mutual

                if is_mutual or chosen_available:
                    collaboration_pairs[agent_id] = chosen_id
                    claimed.add(agent_id)
                    claimed.add(chosen_id)

                    # Update collaboration tracker and COI
                    self.collaboration_tracker.add_collaboration(agent_id, chosen_id, year)
                    agent_a = self.ecosystem.get_agent_by_id(agent_id)
                    agent_b = self.ecosystem.get_agent_by_id(chosen_id)
                    agent_a.add_conflict_of_interest(chosen_id)
                    agent_b.add_conflict_of_interest(agent_id)

        self._collaboration_pairs[year] = collaboration_pairs

        # Record network snapshot
        basic_metrics = self.collaboration_tracker.compute_metrics()
        self.network_metrics.record_network_snapshot(year, basic_metrics)

        year_results['collaboration_formation'] = {
            'new_pairs': len(collaboration_pairs),
            'total_edges': self.collaboration_tracker.graph.number_of_edges(),
            'total_nodes': self.collaboration_tracker.graph.number_of_nodes(),
            **basic_metrics,
        }
        if growth_stats is not None:
            year_results['collaboration_formation'].update(growth_stats)

        logger.info(f"Year {year} Phase 1.5: Formed {len(collaboration_pairs)} collaboration pairs "
                     f"(total: {self.collaboration_tracker.graph.number_of_edges()} edges, "
                     f"{self.collaboration_tracker.graph.number_of_nodes()} nodes)")

    def _expanding_candidate_selection(self, year, active_authors, agent_topics,
                                       _get_institution, collaboration_mode, pa_config):
        """Expanding growth mode (prereg_v1_1_source_fix): deterministic 70/30
        new-tie/repeat-tie proposal channels, scored against the pre-formation
        graph snapshot. Returns (agent_candidates, growth_ctx).
        """
        import math
        import networkx as nx
        from utopia.utils.seeding import derive_seed

        run_seed = getattr(self.args, 'seed', 42)
        # All eligibility, distances, and degree terms use the graph BEFORE any
        # current-year collaboration is added.
        pre_graph = self.collaboration_tracker.graph.copy()
        max_log1p_deg = max((math.log1p(d) for _, d in pre_graph.degree()), default=0.0)

        def _recent_topics(agent_id):
            topics = set()
            papers = self.paper_tracker.get_papers_by_author(agent_id)
            for p in sorted(papers, key=lambda x: x.year, reverse=True)[:3]:
                topics.update(p.topics)
            return topics

        agent_candidates = {}
        channels = {}
        distances = {}       # agent_id -> {other_id: pre-formation distance}
        components_map = {}  # (agent_id, candidate_id) -> score components

        for agent in active_authors:
            topics_a = agent_topics.get(agent.id, set())
            if not topics_a:
                continue
            channel = ('new_tie'
                       if derive_seed(run_seed, 'collab_growth_channel', year, agent.id) % 100 < 70
                       else 'repeat_tie')
            channels[agent.id] = channel
            inst_a = _get_institution(agent)
            dist_map = (dict(nx.single_source_shortest_path_length(pre_graph, agent.id))
                        if pre_graph.has_node(agent.id) else {})
            distances[agent.id] = dist_map
            recent_topics_a = _recent_topics(agent.id)
            coi_a = set(getattr(agent, 'conflict_of_interest', set()) or set())

            scored = []
            for other in active_authors:
                if other.id == agent.id:
                    continue
                inst_b = _get_institution(other)
                if collaboration_mode == 'intra_institute' and inst_a != inst_b:
                    continue
                if collaboration_mode == 'cross_institute' and inst_a == inst_b:
                    continue

                dist = dist_map.get(other.id, -1)
                if channel == 'new_tie':
                    # New ties only: existing d1 partners and COI (either
                    # direction) are ineligible; d2 and disconnected allowed.
                    if dist == 1:
                        continue
                    if other.id in coi_a or agent.id in set(
                            getattr(other, 'conflict_of_interest', set()) or set()):
                        continue
                else:
                    # Repeat ties only: must already be a d1 collaborator.
                    if dist != 1:
                        continue

                topics_b = agent_topics.get(other.id, set())
                union = topics_a | topics_b
                if not union:
                    continue
                topic_score = len(topics_a & topics_b) / len(union)
                recent_topics_b = _recent_topics(other.id)
                pub_union = recent_topics_a | recent_topics_b
                pub_score = (len(recent_topics_a & recent_topics_b) / len(pub_union)
                             if pub_union else 0.0)

                if channel == 'new_tie':
                    deg = pre_graph.degree(other.id) if pre_graph.has_node(other.id) else 0
                    degree_score = math.log1p(deg) / max_log1p_deg if max_log1p_deg > 0 else 0.0
                    d2_indicator = 1.0 if dist == 2 else 0.0
                    final_score = (0.40 * topic_score + 0.30 * pub_score
                                   + 0.20 * degree_score + 0.10 * d2_indicator)
                    score_parts = {'topic': topic_score, 'pub': pub_score,
                                   'degree': degree_score, 'd2_indicator': d2_indicator}
                else:
                    # Existing compatibility logic; net term is constant (1.0)
                    # across d1 candidates so ranking is topical/publication.
                    final_score = 0.4 * topic_score + 0.4 * pub_score + 0.2 * 1.0
                    score_parts = {'topic': topic_score, 'pub': pub_score, 'net': 1.0}

                if final_score > pa_config['min_similarity_threshold']:
                    recent_papers_b = self.paper_tracker.get_papers_by_author(other.id)
                    expertise_b = [e.topic if hasattr(e, 'topic') else str(e)
                                   for e in getattr(other, 'expertise', [])]
                    recent_paper_titles = [p.title for p in sorted(
                        recent_papers_b, key=lambda x: x.year, reverse=True)[:3]]
                    scored.append((other.id, final_score, {
                        'id': other.id,
                        'institution': inst_b or 'Unknown',
                        'expertise': expertise_b,
                        'recent_papers': '; '.join(recent_paper_titles) if recent_paper_titles else 'None',
                        'score': final_score,
                    }))
                    components_map[(agent.id, other.id)] = score_parts

            # Deterministic: score desc, then candidate ID
            scored.sort(key=lambda x: (-x[1], x[0]))
            agent_candidates[agent.id] = scored[:pa_config['top_k_candidates']]

        growth_ctx = {'pre_graph': pre_graph, 'channels': channels,
                      'distances': distances, 'components': components_map,
                      'run_seed': run_seed}
        return agent_candidates, growth_ctx

    def _expanding_form_pairs(self, year, collab_decisions, agent_candidates, growth_ctx):
        """Pair formation for the expanding mode: same mutual/one-sided rules
        as legacy, but iterated in a seeded deterministic order, with every
        proposal and realized event logged and classified against the
        pre-formation snapshot. Returns (collaboration_pairs, growth_stats).
        """
        import random
        from collections import Counter
        import networkx as nx
        from utopia.utils.seeding import derive_seed

        pre_graph = growth_ctx['pre_graph']
        distances = growth_ctx['distances']
        channels = growth_ctx['channels']

        order = sorted(collab_decisions.keys())
        random.Random(derive_seed(growth_ctx['run_seed'], 'collab_growth_order', year)).shuffle(order)

        claimed = set()
        collaboration_pairs = {}
        growth_log = []
        realized_counts = Counter()
        merge_counts = Counter()
        # Working copy for per-edge component classification (state BEFORE
        # each insertion), updated as edges land this year.
        work_graph = pre_graph.copy()

        for res_idx, agent_id in enumerate(order):
            _, chosen_id = collab_decisions[agent_id]
            pre_dist = distances.get(agent_id, {}).get(chosen_id, -1)
            other_decision = collab_decisions.get(chosen_id)
            is_mutual = bool(other_decision and other_decision[1] == agent_id)
            rec = {
                'agent': agent_id,
                'candidate': chosen_id,
                'channel': channels.get(agent_id),
                'pre_d1': pre_dist == 1,
                'pre_distance': pre_dist,
                'mutual': is_mutual,
                'score_components': growth_ctx['components'].get((agent_id, chosen_id)),
                'resolution_order': res_idx,
            }
            if collaboration_pairs.get(chosen_id) == agent_id:
                # Mirror side of an already-formed mutual pair: the same
                # collaboration event, already logged — not a rejection.
                continue
            chosen_available = chosen_id not in collab_decisions or is_mutual
            if agent_id in claimed or chosen_id in claimed or not (is_mutual or chosen_available):
                rec['formed'] = False
                rec['realized'] = 'rejected'
                realized_counts['rejected'] += 1
                growth_log.append(rec)
                continue

            rec['formed'] = True
            if pre_dist == 1:
                rec['realized'] = 'repeat_edge'
                realized_counts['repeat_edge'] += 1
            else:
                rec['realized'] = 'new_edge'
                realized_counts['new_edge'] += 1
                a_conn = work_graph.has_node(agent_id) and work_graph.degree(agent_id) > 0
                b_conn = work_graph.has_node(chosen_id) and work_graph.degree(chosen_id) > 0
                if a_conn and b_conn:
                    rec['merge_class'] = ('within_component'
                                          if nx.has_path(work_graph, agent_id, chosen_id)
                                          else 'merging_components')
                else:
                    rec['merge_class'] = 'attaching_new_node'
                merge_counts[rec['merge_class']] += 1
            work_graph.add_edge(agent_id, chosen_id)

            collaboration_pairs[agent_id] = chosen_id
            claimed.add(agent_id)
            claimed.add(chosen_id)
            self.collaboration_tracker.add_collaboration(agent_id, chosen_id, year)
            agent_a = self.ecosystem.get_agent_by_id(agent_id)
            agent_b = self.ecosystem.get_agent_by_id(chosen_id)
            agent_a.add_conflict_of_interest(chosen_id)
            agent_b.add_conflict_of_interest(agent_id)
            growth_log.append(rec)

        # Agents that had a candidate menu but made no valid proposal
        for agent_id, candidates in sorted(agent_candidates.items()):
            if candidates and agent_id not in collab_decisions:
                realized_counts['unmatched'] += 1
                growth_log.append({
                    'agent': agent_id, 'candidate': None,
                    'channel': channels.get(agent_id),
                    'formed': False, 'realized': 'unmatched',
                })

        post_graph = self.collaboration_tracker.graph
        comp_sizes = sorted((len(c) for c in nx.connected_components(post_graph)), reverse=True)
        pre_comp_count = nx.number_connected_components(pre_graph)
        size_hist = {'1': 0, '2': 0, '3-5': 0, '6-10': 0, '>10': 0}
        for s in comp_sizes:
            if s == 1:
                size_hist['1'] += 1
            elif s == 2:
                size_hist['2'] += 1
            elif s <= 5:
                size_hist['3-5'] += 1
            elif s <= 10:
                size_hist['6-10'] += 1
            else:
                size_hist['>10'] += 1

        pair_counts = {'d2': 0, 'd3': 0, 'connected_ge4': 0, 'disconnected': 0}
        nodes = list(post_graph.nodes)
        for a in nodes:
            lengths = nx.single_source_shortest_path_length(post_graph, a)
            for b in nodes:
                if b == a:
                    continue
                d = lengths.get(b)
                if d is None:
                    pair_counts['disconnected'] += 1
                elif d == 2:
                    pair_counts['d2'] += 1
                elif d == 3:
                    pair_counts['d3'] += 1
                elif d >= 4:
                    pair_counts['connected_ge4'] += 1

        growth_stats = {
            'growth_mode': 'expanding',
            'channel_assigned': dict(Counter(channels.values())),
            'proposals': len(collab_decisions),
            'realized': dict(realized_counts),
            'new_edge_classes': dict(merge_counts),
            'components_before': pre_comp_count,
            'components_after': len(comp_sizes),
            'largest_component': comp_sizes[0] if comp_sizes else 0,
            'component_size_hist': size_hist,
            'degree_hist': dict(Counter(d for _, d in post_graph.degree())),
            'pair_counts': pair_counts,
            'growth_log': growth_log,
        }

        # Persist per-year growth diagnostics next to the checkpoints
        # (year_results is not fully exported to simulation_report.json).
        out_dir = getattr(self, 'output_dir', None)
        if out_dir:
            with open(os.path.join(out_dir, 'network_growth_log.jsonl'), 'a') as f:
                f.write(json.dumps({'year': year, **growth_stats}) + '\n')

        return collaboration_pairs, growth_stats

    def _apply_coauthorship(self, paper_dict: Dict, agent, year: int):
        """If agent has a collaboration partner this round, update paper_dict with co-author info."""
        collab_pairs = self._collaboration_pairs.get(year, {})
        co_author_id = collab_pairs.get(agent.id)
        if co_author_id:
            co_author = self.ecosystem.get_agent_by_id(co_author_id)
            paper_dict['author_id'] = [agent.id, co_author_id]
            paper_dict['author_type'] = [agent.get_type(), co_author.get_type()]

    @staticmethod
    def _get_lead_author_id(paper: Dict) -> str:
        """Get the lead (first) author ID from a paper dict, handling both single and co-authored papers."""
        aid = paper['author_id']
        return aid[0] if isinstance(aid, list) else aid

    @staticmethod
    def _get_all_author_ids(paper: Dict) -> List[str]:
        """Get all author IDs from a paper dict as a list."""
        aid = paper['author_id']
        return aid if isinstance(aid, list) else [aid]

    def _describe_network_relationship(self, distance: int) -> str:
        """Describe the collaboration network relationship for non-blind review prompts."""
        if distance == 1:
            return "You have directly collaborated with this author before."
        elif distance == 2:
            return "You share a mutual collaborator with this author (2-hop connection)."
        elif distance == 3:
            return "You have a distant connection through the collaboration network (3 hops)."
        elif distance > 3:
            return f"You are distantly connected ({distance} hops away in the collaboration network)."
        else:
            return "You have no collaboration history with this author."

    def _build_author_info_for_review(self, reviewer, paper: Dict) -> Dict:
        """Build author info dict for non-blind review in the preferential attachment experiment.

        Returns None if the experiment is not active.
        """
        if self.experiment_name != 'preferential_attachment' or not self.collaboration_tracker:
            return None

        # network_review_bias source runs: keep source reviews blind (no author
        # identity or network relationship in review prompts) while collaboration
        # tracking stays on. Default (env unset) is byte-identical to before.
        if os.environ.get('UTOPIA_PA_BLIND_REVIEW') == '1':
            return None

        lead_id = self._get_lead_author_id(paper)
        author = self.ecosystem.get_agent_by_id(lead_id)
        institution = getattr(author, 'university_name', None) or getattr(author, 'company_name', None) or 'Unknown'
        distance = self.collaboration_tracker.get_network_distance(reviewer.id, lead_id)

        return {
            'author_name': lead_id,
            'institution': institution,
            'network_relationship': self._describe_network_relationship(distance),
            'network_distance': distance,
        }

    def _collect_preferential_attachment_metrics(self, year: int):
        """Collect preferential attachment metrics at end of year."""
        if self.experiment_name != 'preferential_attachment':
            return
        self._initialize_preferential_attachment_trackers()

        all_metrics = self.network_metrics.compute_all_metrics(
            year, self.collaboration_tracker, self.ecosystem
        )
        if all_metrics.get('pearson_r') is not None:
            logger.info(f"Year {year}: Distance-score correlation: "
                        f"Pearson r={all_metrics['pearson_r']:.3f} (p={all_metrics['pearson_p']:.4f}), "
                        f"Spearman r={all_metrics['spearman_r']:.3f}")

    def _apply_annual_influx(self, year: int) -> List[str]:
        """Influx2x2: inject this year's junior cohort at the top of _setup_year.

        OFF (annual_influx_rate <= 0, the default): returns [] immediately —
        no RNG use, no side effects, byte-identical to the pre-influx code path.
        ON: constructs entrants via the stock researcher constructors
        (reputation defaults 5/6, zero publications via empty papers_by_author,
        no collaboration edges), wires conflict-of-interest bidirectionally with
        ALL current same-institution members (active or culled — culled agents
        still review), and adds them to the ecosystem. Idempotent under
        checkpoint resume: an id already in agent_population is skipped.
        """
        rate = getattr(self.args, 'annual_influx_rate', 0.0) or 0.0
        if rate <= 0:
            return []
        assert abs(rate - 0.1) < 1e-12, \
            "only the staggered 10%/yr influx schedule is implemented"
        assert getattr(self.args, 'population_mode', 'default') == 'default' and not self.debug, \
            "annual influx is defined only for the non-debug default population"

        entrant_ids = []
        for spec in build_influx_cohort(year, getattr(self.args, 'seed', 42)):
            name = spec['researcher_name']
            if name in self.ecosystem.agent_population:  # resume idempotence guard
                continue
            expertise = [AVAILABLE_DIRECTIONS[i] for i in spec['expertise_indices']]
            if spec['kind'] == 'university':
                agent = UniversityResearcher(
                    researcher_name=name,
                    university_name=spec['institution'],
                    funding_level=spec['funding_level'],
                    expertise=expertise,
                    llm=self.llm,
                    generate_research_proposal=True,
                    exploration_strategy="balanced",
                )
            else:
                agent = IndustryResearcher(
                    researcher_name=name,
                    company_name=spec['institution'],
                    funding_level=spec['funding_level'],
                    expertise=expertise,
                    llm=self.llm,
                    funding_mode=self.industry_funding_mode,
                    exploration_strategy="balanced",
                )
            peers = [a for a in self.ecosystem.agent_population.values()
                     if getattr(a, 'university_name', None) == spec['institution']
                     or getattr(a, 'company_name', None) == spec['institution']]
            agent.add_conflict_of_interest({p.id for p in peers})
            for p in peers:
                p.add_conflict_of_interest(agent.id)
            self.ecosystem.add_agent(agent)
            entrant_ids.append(name)
            logger.info(f"Added {agent.id} from {spec['institution']} (influx junior, year {year})")

        logger.info(f"[Influx] Year {year}: +{len(entrant_ids)} junior researchers "
                    f"(population={len(self.ecosystem.agent_population)})")
        return entrant_ids

    def _account_resources(self, agent, delta, *, category, year, project=None,
                           paper=None, program=None, reason=None):
        from utopia.funding.accounting import apply_resource_change
        return apply_resource_change(
            agent, delta, category=category, year=year, project=project,
            paper=paper, program=program, reason=reason)

    def _resubmission_fee(self):
        override = getattr(self.args, 'resubmission_cost', None)
        return SIMULATION_CONFIG['conference']['resubmission_cost'] if override is None else override

    def _prepare_project_production(self, year, active_authors):
        """Completed lead projects pay before any selection, including failed selection."""
        from utopia.funding.accounting import enter_project_production
        policy = getattr(self, 'cost_policy', None)
        if policy is None or policy['production_cost_mode'] == 'per_paper':
            return active_authors
        admitted = []
        for agent in active_authors:
            agent.production_cost_mode = policy['production_cost_mode']
            if enter_project_production(agent, year, policy, self.funding_tracker):
                admitted.append(agent)
        return admitted

    def _record_output_cost(self, agent, paper_dict, year):
        """The native baseline debits no per-paper fee. Link each output to its project."""
        from utopia.funding.accounting import project_id, cost_policy_enabled
        policy = getattr(self, 'cost_policy', None)
        if policy is None or not cost_policy_enabled(policy):
            return
        pid = project_id(agent)
        paper_dict['project_id'] = pid
        if getattr(agent, 'resource_ledger', None) is not None:
            self._account_resources(agent, 0, category='production_submission_cost',
                                    year=year, project=pid, paper=paper_dict['id'],
                                    reason='output_included')

    def _charge_resubmission(self, agent, paper_dict, year):
        from utopia.funding.accounting import stable_id
        pid = paper_dict.get('project_id')
        if pid is None and all(k in paper_dict for k in ('project_start_year', 'project_end_year')):
            pid = stable_id('project', self._get_lead_author_id(paper_dict),
                            paper_dict['project_start_year'], paper_dict['project_end_year'])
        fee = self._resubmission_fee()
        if self._account_resources(agent, -fee, category='resubmission_cost', year=year,
                                   project=pid, paper=paper_dict['id']):
            self.funding_tracker.record_funding_consumption(year, fee, agent.get_type())

    def _setup_year(self, year: int) -> Dict:
        """Setup year: (influx2x2) inject junior cohort, log start, reset funding
        tracker, reset conferences

        Args:
            year: Current year number

        Returns:
            year_results: Dict initialized with year number
        """

        logger.info(f"*** YEAR {year} ***")

        # Influx2x2: entrants join BEFORE funding_tracker.start_cycle so their
        # resources count in the year's opening total and they participate in
        # every phase of their entry year (no-op when the flag is off).
        influx_entrants = self._apply_annual_influx(year)
        if getattr(self, 'resource_ledger', None) is not None:
            self.resource_ledger.start_year(year, self.ecosystem.agent_population)

        # Reset all fundings to 0 at start of cycle
        total_funding = self.funding_tracker.start_cycle(year, self.ecosystem.agent_population)
        logger.info(f"Total funding at start of year {year}: {total_funding:.2f}")

        # Reset conferences for the new year of simulation
        self.conference_system.reset_all_for_new_year(year)

        # Initialize year results dict
        year_results = {'year': year}
        if influx_entrants:
            year_results['influx_entrants'] = influx_entrants
        return year_results

    def _run_phase_0_resubmissions(self, year: int, year_results: Dict) -> List[Dict]:
        """Resubmit rejected papers when the experiment and current year allow it."""
        if getattr(self.args, 'disable_resubmission', False):
            logger.info(f"Year {year} Phase 0: resubmissions disabled (--disable_resubmission); skipping")
            return []
        if year <= 1:
            return []
        logger.info(f"Year {year} Preparation: Processing resubmissions from year {year - 1}...")
        if hasattr(self.llm, 'generate_batch'):
            return self._resubmit_papers_batch(year)
        return self._resubmit_papers_sequential(year)

    def _record_resubmission(self, agent, rejected_papers, metadata, year, resubmissions, submitted_ids):
        """Restore a rejected paper to pending once, preserving reviews and charging its fee."""
        arxiv_id = metadata['arxiv_id']
        if arxiv_id in submitted_ids:
            logger.error(f"[Resubmission] Paper {arxiv_id} already resubmitted for year {year}")
            return
        paper = rejected_papers[arxiv_id].copy()
        paper['conference'] = metadata['conference']
        paper['year'] = year
        paper['type'] = 'resubmission'
        review_history = paper.get('review_history', [])
        del paper['status']
        paper.pop('review_history', None)
        paper['status'] = 'pending'
        paper['review_history'] = review_history
        resubmissions.append(paper)
        self.paper_tracker.add_or_update_paper(paper)
        self._charge_resubmission(agent, paper, year)
        submitted_ids.add(arxiv_id)

    def _process_resubmission_response(self, result, request, conference_ids, year, resubmissions, submitted_ids):
        """Record valid choices and return unresolved rejected papers for a retry."""
        agent = request['agent']
        rejected_papers = request['rejected_papers']
        if result is None or 'resubmitted_papers' not in result:
            logger.warning(f"Failed to parse resubmission decision for agent {agent.id}, adding to retry")
            return request
        has_invalid = False
        for metadata in result['resubmitted_papers']:
            if metadata.get('arxiv_id', '') not in rejected_papers or metadata.get('conference', '') not in conference_ids:
                logger.warning(f"Invalid resubmission entry for agent {agent.id}: {metadata}, skipping")
                has_invalid = True
                continue
            self._record_resubmission(agent, rejected_papers, metadata, year, resubmissions, submitted_ids)
        if not has_invalid:
            return None
        remaining = {pid: paper for pid, paper in rejected_papers.items() if pid not in submitted_ids}
        return {'agent': agent, 'rejected_papers': remaining} if remaining else None

    def _resubmit_papers_batch(self, year):
        """Batch resubmission decisions and retry unresolved rejected papers."""
        resubmissions_list = []
        resubmissions_paper_ids = set()
        # Step 1: Collect prompts and metadata
        batch_prompts = []
        batch_metadata = []
        resubmission_response_format = None

        for author_id, papers in self.paper_tracker.papers_by_author.items():
            agent = self.ecosystem.get_agent_by_id(author_id)
            if not agent:
                continue

            rejected_papers = {paper.id: paper.to_dict() for paper in papers if paper.status == 'reject'}
            if agent.resources < self._resubmission_fee() or not rejected_papers:
                continue

            prompt, response_format = agent.get_resubmission_prompt(
                rejected_papers, self.conference_system.conferences, self._resubmission_fee())
            batch_prompts.append(prompt)
            batch_metadata.append({
                'agent': agent,
                'rejected_papers': rejected_papers,
            })
            if resubmission_response_format is None:
                resubmission_response_format = response_format

        # Step 2: Batch call + process with retries
        MAX_RESUBMISSION_RETRIES = 4
        agents_needing_retry = []

        for attempt in range(MAX_RESUBMISSION_RETRIES):
            if not batch_prompts:
                break

            batch_results = self.llm.generate_batch(batch_prompts, response_format=resubmission_response_format, desc=f"{len(batch_prompts)} resubmission decisions (attempt {attempt+1}/{MAX_RESUBMISSION_RETRIES})", seed_ctx=('phase0_resubmit', year, attempt))

            agents_needing_retry = []

            # Step 3: Process results
            candidate_conference_ids = set([c.conference_id for c in self.conference_system.conferences])
            for i, (result, _) in enumerate(batch_results):
                retry = self._process_resubmission_response(
                    result, batch_metadata[i], candidate_conference_ids,
                    year, resubmissions_list, resubmissions_paper_ids)
                if retry is not None:
                    agents_needing_retry.append(retry)

            # Rebuild batch for next attempt
            if not agents_needing_retry:
                break

            batch_prompts = []
            batch_metadata = []

            for retry_info in agents_needing_retry:
                agent = retry_info['agent']
                remaining_rejected = retry_info['rejected_papers']

                if agent.resources < self._resubmission_fee() or not remaining_rejected:
                    continue

                prompt, response_format = agent.get_resubmission_prompt(
                    remaining_rejected, self.conference_system.conferences, self._resubmission_fee())
                batch_prompts.append(prompt)
                batch_metadata.append({
                    'agent': agent,
                    'rejected_papers': remaining_rejected,
                })
                if resubmission_response_format is None:
                    resubmission_response_format = response_format

        # Log remaining failures after all retries (no random fallback)
        if agents_needing_retry:
            logger.info(f"After {MAX_RESUBMISSION_RETRIES} attempts, {len(agents_needing_retry)} agents could not resubmit")

        return resubmissions_list

    def _resubmit_papers_sequential(self, year):
        """Use agent resubmission decisions with the shared recording logic."""
        resubmissions_list = []
        resubmissions_paper_ids = set()
        for author_id, papers in self.paper_tracker.papers_by_author.items():
            agent = self.ecosystem.get_agent_by_id(author_id)

            if not agent:
                continue

            rejected_papers = {paper.id: paper.to_dict() for paper in papers if paper.status == 'reject'}
            if not rejected_papers:
                continue

            resubmissions_of_author = agent.decide_resubmission(
                rejected_papers, self.conference_system.conferences, self._resubmission_fee())

            for metadata in resubmissions_of_author:
                self._record_resubmission(
                    agent, rejected_papers, metadata, year, resubmissions_list, resubmissions_paper_ids)

        return resubmissions_list

    def _run_phase_1_research_directions(self, year: int, year_results: Dict):
        """Phase 1: Assign research directions to agents

        Args:
            year: Current year number
            year_results: Year results dict to update

        Returns:
            Tuple of (directions, active_authors, average_funding_level)
        """
        logger.info(f"Year {year} Phase 1: Assigning research directions...")

        # Get funding agencies and calculate average funding level
        funding_agencies = [
            agent for agent_id, agent in self.ecosystem.agent_population.items()
            if isinstance(agent, FundingAgency)
        ]
        average_funding_level = self.calculate_average_funding_level()

        # Reset funding programs for new year
        all_funding_programs_dict = {}
        for funding_agent in funding_agencies:
            for program_id, program in funding_agent.funding_programs.items():
                program.reset_for_new_cycle()
                all_funding_programs_dict[program_id] = program

        # Get authors who need new directions
        authors = self.ecosystem.get_available_authors()
        authors_with_directions_pending = [a for a in authors if a.project_end_year < year and a.is_active]
        logger.info(f"[Direction] {len(authors_with_directions_pending)} Authors")

        # Check if LLM supports batching
        supports_batching = hasattr(self.llm, 'generate_batch')

        # For exploration experiment: pre-filter candidate directions per agent.
        # Candidates are passed EXPLICITLY so the prompt and validation use the
        # same set (plan 4.1) — no hidden agent attribute.
        candidate_map = None
        if self.is_exploration_experiment and authors_with_directions_pending:
            self._initialize_experiment_trackers()
            exp_config = SIMULATION_CONFIG['exploration_experiment']
            candidate_map = {}
            for agent in authors_with_directions_pending:
                candidate_map[agent.id] = get_strategy_filtered_candidate_directions(
                    agent, AVAILABLE_DIRECTIONS, self.embedding_tracker, year, exp_config
                )

        if supports_batching and authors_with_directions_pending:

            fallback_counter = {}
            directions = create_research_directions_batch(
                agents=authors_with_directions_pending,
                year=year,
                paper_tracker=self.paper_tracker,
                llm=self.llm,
                average_funding_level=average_funding_level,
                candidate_map=candidate_map,
                fallback_counter=fallback_counter,
            )
            self._direction_fallback_stats[year] = fallback_counter
            # Update agent project timelines
            for agent in authors_with_directions_pending:
                direction = directions[agent.id]
                agent.project_start_year = year
                agent.project_end_year = (year - 1) + direction['direction'].years
                agent.newest_direction = direction

            # Record chosen-direction distance and topic-switch indicator (plan 5.4)
            if self.is_exploration_experiment:
                exp_config = SIMULATION_CONFIG['exploration_experiment']
                for agent in authors_with_directions_pending:
                    chosen = directions[agent.id]['direction']
                    centroid = self.embedding_tracker.compute_career_centroid(
                        agent.id, max_years=exp_config['history_window_years'],
                        current_year=year - 1)
                    distance = None
                    if centroid is not None:
                        dist_map = self.embedding_tracker.compute_direction_distances(
                            centroid, [chosen])
                        distance = dist_map.get(chosen.topic)
                    prev = next((self._direction_records[k] for k in
                                 [(agent.id, y) for y in range(year - 1, 0, -1)]
                                 if k in self._direction_records), None)
                    self._direction_records[(agent.id, year)] = {
                        'topic': chosen.topic,
                        'distance': distance,
                        'switched': bool(prev and prev['topic'] != chosen.topic),
                    }
        else:
            # ==================== SEQUENTIAL PATH (GPT/non-batch) ====================
            directions = {}
            for agent in tqdm(authors_with_directions_pending, desc="Research Directions"):
                direction = assign_directions_individual(
                    agent,
                    year=year,
                    paper_tracker=self.paper_tracker,
                    average_funding_level=average_funding_level
                )
                directions[agent.id] = direction
                agent.project_start_year = year
                agent.project_end_year = (year - 1) + direction['direction'].years
                agent.newest_direction = direction

        # For authors not assigned this year, use their existing direction
        for author in authors:
            if author.id not in directions:
                assert author.newest_direction is not None, f"Author {author.id} has no newest direction"
                directions[author.id] = author.newest_direction

        # Get active authors (those ready to submit papers)
        active_authors = [a for a in authors if a.project_end_year <= year and a.is_active]
        logger.info(f"[Select Direction] {len(active_authors)} Active Authors")

        # Update year results
        year_results['research_assignment'] = {
            'num_agents_assigned': len(directions),
            "directions": {
                agent_id: {"direction": direction['direction'].topic, "detailed_focus": direction['detailed_focus'],
                           "reason": direction['reason']} for agent_id, direction in directions.items()}
        }

        return directions, active_authors, average_funding_level

    def _build_candidate_papers(self, rag_documents, indices):
        """Build enriched candidate paper dicts with author_id and institution."""
        candidate_papers = []
        for idx in indices:
            paper = rag_documents[idx]
            paper_id = paper.metadata['id']
            archived = self.paper_tracker.papers_by_id[paper_id]
            lead_id = archived.all_author_ids[0]
            cp_agent = self.ecosystem.get_agent_by_id(lead_id)
            institution = getattr(cp_agent, 'university_name', None) or getattr(cp_agent, 'company_name', None)
            assert institution is not None, f"Paper {paper_id} has no valid institution."
            candidate_papers.append({
                'id': paper_id,
                'title': paper.metadata['title'],
                'abstract': paper.page_content,
                'author_id': lead_id,
                'institution': institution,
            })
        return candidate_papers

    def _run_phase_2_submit_papers(self, year: int, directions: Dict, active_authors: List,
                                    resubmissions_list: List[Dict], year_results: Dict) -> List[Dict]:
        """Prepare authors, submit new papers, select citations, and register venues."""
        logger.info(f"Year {year} Phase 2: Submitting papers...")
        co_author_ids = set(self._collaboration_pairs.get(year, {}).values())
        if co_author_ids:
            original_count = len(active_authors)
            active_authors = [a for a in active_authors if a.id not in co_author_ids]
            logger.info(f"Phase 2: {original_count - len(active_authors)} agents are co-authors this round, "
                        f"{len(active_authors)} agents will submit as lead authors")
        active_authors = self._prepare_project_production(year, active_authors)

        self._prepare_paper_index(year)
        intentions = self._generate_round_intentions(year, directions, active_authors)
        year_results['round_intentions'] = intentions
        queries = [
            f"{intentions[agent.id]}\nKeywords: {', '.join(directions[agent.id]['direction'].keywords)}"
            for agent in active_authors
        ]
        logger.info(f"Retrieving {len(queries)} paper submissions ...")
        topk_indices = self.rag.batch_retrieve(queries, retrieval_type="submission")['topk_indices']
        assert topk_indices.shape[0] == len(active_authors)

        if hasattr(self.llm, 'generate_batch'):
            submissions = self._submit_papers_batch(year, directions, active_authors, topk_indices)
        else:
            submissions = self._submit_papers_sequential(year, directions, active_authors, topk_indices)

        if year >= 2:
            self._record_submission_citations(year, directions, submissions, resubmissions_list)

        submitted_by_author = {}
        for paper in submissions + resubmissions_list:
            conference = self.conference_system.conference_map[paper['conference']]
            lead_id = self._get_lead_author_id(paper)
            submitted_by_author.setdefault(lead_id, [])
            if not conference.submit_paper(paper, year=year):
                raise ValueError(f"Paper {paper['id']} submission to conference {paper['conference']} failed")
            submitted_by_author[lead_id].append(paper)
            self.submitted_paper_ids.add(paper['id'])
        year_results['paper_submission'] = {
            'num_papers_submitted': sum(len(papers) for papers in submitted_by_author.values()),
        }
        return submissions

    def _prepare_paper_index(self, year: int):
        """Load the paper index and preserve submission statuses when resuming."""
        if year == 1:
            self.rag.load_documents(num_years=self.num_years)
            self.rag.build_knowledge_index()
        elif self.rag.documents == []:
            # Loading documents resets statuses, so restore the checkpoint state.
            saved_document_status = self.rag.document_status.copy()
            self.rag.load_documents(num_years=self.num_years)
            self.rag.build_knowledge_index()
            self.rag.document_status = saved_document_status
            logger.info(f"Resumed from checkpoint with {len(saved_document_status)} document statuses preserved")

        # Set the current simulation year for year-based data partitioning
        self.rag.set_current_year(year)

    def _generate_round_intentions(self, year, directions, active_authors):
        """Generate intentions in author order using the existing model transport."""
        intention_prompts = []
        intention_response_format = None
        for agent in active_authors:
            prompt, response_format = agent.get_round_intention_prompt(directions[agent.id], year, self.paper_tracker)
            intention_prompts.append(prompt)
            if intention_response_format is None:
                intention_response_format = response_format

        supports_batching = hasattr(self.llm, 'generate_batch')
        round_intentions = {}

        if supports_batching:
            intention_results = self.llm.generate_batch(
                intention_prompts, max_tokens=SIMULATION_CONFIG['llm']['max_tokens'],
                response_format=intention_response_format,
                desc=f"Generating {len(intention_prompts)} intentions",
                seed_ctx=('phase2_intentions', year),
            )
            for agent, (result_json, _) in zip(active_authors, intention_results):
                intention = self._parse_intention(result_json, directions[agent.id])
                round_intentions[agent.id] = intention
        else:
            for agent, prompt in zip(active_authors, intention_prompts):
                result_json, _ = self.llm.generate(prompt=prompt, response_format=intention_response_format)
                intention = self._parse_intention(result_json, directions[agent.id])
                round_intentions[agent.id] = intention

        logger.info(f"Generated {len(round_intentions)} intentions")
        return round_intentions

    def _record_new_submission(self, agent, arxiv_id, conference, year, direction, submissions):
        """Record one paper, including costs, coauthors, metadata, and retrieval status."""
        paper_doc = self.rag.id2docs[arxiv_id]
        assert agent.project_start_year <= year, (
            f"Agent {agent.id} MUST started the project before the current year: "
            f"{agent.project_start_year} <= {year}")
        assert agent.project_end_year == year, f"Agent {agent.id} MUST completed the project in the current year"
        paper = {
            'id': arxiv_id,
            'author_id': agent.id,
            'author_type': agent.get_type(),
            'title': paper_doc.metadata['title'],
            'abstract': paper_doc.page_content,
            'tags': paper_doc.metadata['tags'],
            'topics': paper_doc.metadata['topics'],
            'conference': conference,
            'type': 'submission',
            'year': year,
            'project_start_year': agent.project_start_year,
            'project_end_year': agent.project_end_year,
            'maturity': direction['direction'].years,
            'status': 'pending',
        }
        self._record_output_cost(agent, paper, year)
        self._apply_coauthorship(paper, agent, year)
        submissions.append(paper)
        self.paper_tracker.add_or_update_paper(paper)
        self._record_paper_metadata(paper, agent.id, year)
        self.rag.document_status.loc[self.rag.document_status['id'] == arxiv_id, 'status'] = 'submitted'

    def _validate_submission_entry(self, entry, candidate_papers):
        """Validate one model-selected paper and venue, including the opt-out sentinel."""
        arxiv_id = entry['arxiv_id']
        assert "reason" in entry, f"Reason is not in the submission metadata: {entry}"
        assert arxiv_id in [p.metadata['id'] for p in candidate_papers] or arxiv_id == "DO_NOT_SUBMIT"
        assert entry['conference'] in self.conference_system.conference_map or entry['conference'] == "DO_NOT_SUBMIT", (
            f"Generated invalid conference name: {entry['conference']}")
        return arxiv_id

    def _process_submission_response(self, response, metadata, year, directions, submissions):
        """Record valid entries and return the remaining work when a retry is needed."""
        agent = metadata['agent']
        candidate_papers = metadata['candidate_papers']
        n_needed = metadata.get('n_needed', self.papers_per_project)
        fallback = {
            'agent': agent, 'candidate_papers': candidate_papers,
            'conference': None, 'n_needed': n_needed,
        }
        if response is None:
            logger.warning(f"Failed to parse response for {agent.id}, adding to fallback")
            return fallback
        if self.papers_per_project == 1:
            entries = [response]
        else:
            entries = response.get('submissions') if isinstance(response, dict) else None
            if not isinstance(entries, list):
                logger.warning(f"Invalid submission list from {agent.id}, adding to fallback")
                return fallback
            entries = entries[:n_needed]
            if not entries:
                logger.info(f"Agent {agent.id} did not submit any paper due to funding issues")
                return None

        seen_ids = set()
        n_submitted = 0
        needs_fallback = False
        for entry in entries:
            try:
                arxiv_id = self._validate_submission_entry(entry, candidate_papers)
                assert arxiv_id not in seen_ids, f"Duplicate arxiv_id in one response: {arxiv_id}"
            except (KeyError, AssertionError) as error:
                traceback.print_exc()
                logger.warning(f"Invalid submission from {agent.id}: {error}")
                needs_fallback = True
                break
            if arxiv_id == "DO_NOT_SUBMIT":
                logger.info(f"Agent {agent.id} did not submit any paper due to funding issues")
                break
            seen_ids.add(arxiv_id)
            if arxiv_id in self.paper_tracker.papers_by_id:
                logger.warning(f"Paper {arxiv_id} already taken, {agent.id} needs fallback selection")
                fallback['conference'] = entry['conference']
                needs_fallback = True
                break
            logger.info(f"{agent.id} - {entry['conference']} - submitted '{self.rag.id2docs[arxiv_id].metadata['title']}'")
            self._record_new_submission(agent, arxiv_id, entry['conference'], year, directions[agent.id], submissions)
            n_submitted += 1

        remaining = n_needed - n_submitted
        if needs_fallback and remaining > 0:
            fallback['n_needed'] = remaining
            return fallback
        return None

    def _submit_papers_batch(self, year, directions, active_authors, topk_indices):
        """Submit in batches, retry invalid selections, then fill unresolved requests."""
        submissions = []
        batch_prompts = []
        batch_metadata = []
        response_format = None
        for agent_idx, agent in enumerate(active_authors):
            if agent.project_end_year > year:
                if self.verbose:
                    logger.info(f"[Submission] Agent {agent.id} is not ready to submit papers yet. "
                                f"Maturity year: {agent.project_end_year}, Current year: {year}")
                continue
            candidates = [
                self.rag.documents[idx] for idx in topk_indices[agent_idx]
                if self.rag.document_status.loc[idx, 'status'] == 'unsubmitted'
            ]
            if not candidates:
                logger.warning(f"No candidate papers found for agent {agent.id}, skipping")
                continue
            prompt, schema = agent.get_paper_submission_prompt(
                candidate_papers=candidates, year=year, direction=directions[agent.id],
                conferences=self.conference_system.conferences, paper_tracker=self.paper_tracker,
                max_submissions=self.papers_per_project,
            )
            batch_prompts.append(prompt)
            batch_metadata.append({'agent': agent, 'candidate_papers': candidates, 'n_needed': self.papers_per_project})
            if response_format is None:
                response_format = schema

        max_attempts = 4
        unresolved = []
        for attempt in range(max_attempts):
            if not batch_prompts:
                break
            results = self.llm.generate_batch(
                batch_prompts, max_tokens=SIMULATION_CONFIG['llm']['max_tokens'],
                temperature=0.7, response_format=response_format,
                desc=f"{len(batch_prompts)} paper submissions (attempt {attempt+1}/{max_attempts})",
                seed_ctx=('phase2_submissions', year, attempt),
            )
            unresolved = []
            for index, (response, _) in enumerate(results):
                retry = self._process_submission_response(response, batch_metadata[index], year, directions, submissions)
                if retry is not None:
                    unresolved.append(retry)
            if not unresolved:
                break

            batch_prompts = []
            batch_metadata = []
            for retry in unresolved:
                agent = retry['agent']
                candidates = [p for p in retry['candidate_papers'] if p.metadata['id'] not in self.paper_tracker.papers_by_id]
                if not candidates:
                    continue
                prompt, _ = agent.get_paper_submission_prompt(
                    candidate_papers=candidates, year=year, direction=directions[agent.id],
                    conferences=self.conference_system.conferences, paper_tracker=self.paper_tracker,
                    max_submissions=retry['n_needed'], force_list=self.papers_per_project > 1,
                )
                batch_prompts.append(prompt)
                batch_metadata.append({'agent': agent, 'candidate_papers': candidates, 'n_needed': retry['n_needed']})

        if unresolved:
            self._assign_fallback_submissions(unresolved, year, directions, submissions)
        return submissions

    def _assign_fallback_submissions(self, unresolved, year, directions, submissions):
        """Draw remaining papers before the venue to preserve the legacy RNG stream."""
        logger.info(f"After retries, {len(unresolved)} agents need random assignment")
        for retry in unresolved:
            agent = retry['agent']
            pool = [p for p in retry['candidate_papers'] if p.metadata['id'] not in self.paper_tracker.papers_by_id]
            if not pool:
                logger.warning(f"{agent.id} final fallback failed - no available papers remaining")
                continue
            chosen_papers = []
            for _ in range(min(retry['n_needed'], len(pool))):
                paper = random.choice(pool)
                pool.remove(paper)
                chosen_papers.append(paper)
            conference = retry['conference'] or random.choice(list(self.conference_system.conference_map.keys()))
            for paper in chosen_papers:
                logger.info(f"{agent.id} random fallback - {conference} - submitted '{paper.metadata['title']}'")
                self._record_new_submission(
                    agent, paper.metadata['id'], conference, year, directions[agent.id], submissions)

    def _submit_papers_sequential(self, year, directions, active_authors, topk_indices):
        """Keep the single-paper transport and its existing retry behavior."""
        submissions = []
        assert self.papers_per_project == 1, "papers_per_project>1 requires the batched path"
        for agent_idx, agent in enumerate(tqdm(active_authors, desc="Submission (active researchers)")):

            if agent.project_end_year > year:
                if self.verbose:
                    logger.info(
                        f"[Submission] Agent {agent.id} is not ready to submit papers yet. Maturity year: {agent.project_end_year}, Current year: {year}")
                continue

            candidate_paper_indices = topk_indices[agent_idx]
            # Use submission_documents since we retrieved from submission index
            candidate_papers = [self.rag.documents[idx] for idx in candidate_paper_indices if
                                self.rag.document_status.loc[idx, 'status'] == 'unsubmitted']

            assert len(candidate_papers) > 0, f"No candidate papers found for agent {agent.id}"

            prompt, response_format = agent.get_paper_submission_prompt(
                candidate_papers=candidate_papers,
                year=year,
                direction=directions[agent.id],
                conferences=self.conference_system.conferences,
                paper_tracker=self.paper_tracker
            )

            num_retries = 0

            while True:

                try:
                    submission_metadata, message_history = self.llm.generate(prompt=prompt,
                                                                             response_format=response_format)
                    arxiv_id = self._validate_submission_entry(submission_metadata, candidate_papers)

                    if arxiv_id != "DO_NOT_SUBMIT":
                        logger.info(
                            f"{agent.id} - {submission_metadata['conference']} - submitted \'{self.rag.id2docs[arxiv_id].metadata['title']}\'")
                    break

                except AssertionError as e:
                    if "Generated invalid conference name" in str(e):
                        logger.warning(f"Invalid conference name generated for {agent.id}: {e}")
                    else:
                        traceback.print_exc()

                    num_retries += 1
                    if num_retries >= self.max_retries:
                        raise Exception("Failed to submit paper after 3 retries")

                except Exception as e:
                    logger.warning(f"Error submitting paper: {e}")
                    traceback.print_exc()
                    num_retries += 1
                    if num_retries >= self.max_retries:
                        raise Exception("Failed to submit paper after 3 retries")

            if arxiv_id == "DO_NOT_SUBMIT":
                logger.info(f"Agent {agent.id} did not submit any paper due to funding issues")
                continue

            self._record_new_submission(
                agent, arxiv_id, submission_metadata['conference'], year, directions[agent.id], submissions)

        return submissions

    def _record_submission_citations(self, year, directions, submissions_list, resubmissions_list):
        """Retrieve citation candidates and record citations for new submissions."""
        retrieval_prompts = []

        for submitted_paper in resubmissions_list + submissions_list:
            retrieval_prompts.append(f"{submitted_paper['title']} \n{submitted_paper['abstract']}")

        # First we try to cite accepted papers. If no proper ones, we add rejected papers.
        topk_indices_citation_accepted = self.rag.batch_retrieve(
            retrieval_prompts,
            retrieval_type="accepted_papers"
        )['topk_indices']

        topk_indices_citation_both = self.rag.batch_retrieve(
            retrieval_prompts,
            retrieval_type="accepted_or_rejected_papers"
        )['topk_indices']

        accepted_papers_arxiv_ids = set(
            self.rag.document_status[self.rag.document_status['status'] == "accept"].id.values.tolist())

        if hasattr(self.llm, 'generate_batch'):
            self._record_batch_citations(
                year, directions, submissions_list, topk_indices_citation_accepted,
                topk_indices_citation_both, accepted_papers_arxiv_ids)
        else:
            self._record_sequential_citations(
                year, directions, submissions_list, topk_indices_citation_accepted,
                topk_indices_citation_both, accepted_papers_arxiv_ids)

    def _select_citations_batch(self, year, directions, submissions_list, paper_indices,
                                candidate_indices, citation_kind, citation_response_format=None, accepted_ids=None):
        """Run a regular or extended citation pass, retrying only invalid responses."""
        max_attempts = 4
        batch_prompts = []
        batch_metadata = []

        for i in paper_indices:
            submitted_paper = submissions_list[i]
            agent = self.ecosystem.get_agent_by_id(self._get_lead_author_id(submitted_paper))
            candidate_papers = self._build_candidate_papers(
                self.rag.documents,
                self._rerank_citation_candidates(
                    candidate_indices[i],
                    SIMULATION_CONFIG['citation']['num_candidate_papers'])
            )

            citing_institution = getattr(agent, 'university_name', None) or getattr(agent, 'company_name', None)
            if accepted_ids is not None:
                assert {cp['id'] for cp in candidate_papers} <= accepted_ids, (
                    f"Submitted paper {submitted_paper['id']} is trying to cite non-accepted papers")
                assert citing_institution is not None, f"Agent {agent.id} has no valid institution."
            prompt, response_format = agent.get_citation_prompt(
                paper=submitted_paper, direction=directions[agent.id],
                candidate_papers=candidate_papers, max_papers=30,
                citing_institution=citing_institution
            )
            batch_prompts.append(prompt)
            batch_metadata.append({
                'paper_idx': i, 'paper': submitted_paper, 'agent': agent,
                'candidate_papers': candidate_papers,
            })
            if citation_response_format is None:
                citation_response_format = response_format

        citations_by_paper = {}  # paper_idx -> citation list

        for attempt in range(max_attempts):
            if not batch_prompts:
                break

            batch_results = self.llm.generate_batch(
                batch_prompts, response_format=citation_response_format,
                desc=f"{len(batch_prompts)} {citation_kind} citations (attempt {attempt+1}/{max_attempts})",
                seed_ctx=(f'phase2_citations_{citation_kind}', year, attempt),
            )

            items_needing_retry = []
            for j, (result, _) in enumerate(batch_results):
                meta = batch_metadata[j]
                idx = meta['paper_idx']

                if result is None or result.get('citations') is None:
                    items_needing_retry.append(meta)
                    continue

                citations = result['citations']
                num_candidates = len(meta['candidate_papers'])
                if not all(isinstance(c, int) and 0 <= c < num_candidates for c in citations):
                    logger.warning(f"Invalid citations for paper {meta['paper']['id']}, retrying")
                    items_needing_retry.append(meta)
                    continue

                citations_by_paper[idx] = [meta['candidate_papers'][c]['id'] for c in citations]

            if not items_needing_retry:
                break

            # Rebuild batch for retry
            batch_prompts = []
            batch_metadata = []
            for meta in items_needing_retry:
                citing_inst = getattr(meta['agent'], 'university_name', None) or getattr(meta['agent'], 'company_name', None)
                prompt, _ = meta['agent'].get_citation_prompt(
                    paper=meta['paper'], direction=directions[meta['agent'].id],
                    candidate_papers=meta['candidate_papers'], max_papers=30,
                    citing_institution=citing_inst
                )
                batch_prompts.append(prompt)
                batch_metadata.append(meta)

        return citations_by_paper, citation_response_format

    def _select_self_citations_batch(self, year, directions, submissions_list):
        """Select each author's earlier papers, retrying invalid citation indices."""
        max_attempts = 4
        self_citations_dict = {}  # paper_idx -> self-citation list
        batch_prompts = []
        batch_metadata = []
        self_citation_response_format = None

        for i in range(len(submissions_list)):
            submitted_paper = submissions_list[i]
            agent = self.ecosystem.get_agent_by_id(self._get_lead_author_id(submitted_paper))
            own_papers = self.paper_tracker.get_papers_by_author(agent.id)
            own_papers = [p for p in own_papers if p.project_end_year < year]
            if not own_papers:
                continue

            result = agent.get_self_citation_prompt(
                paper=submitted_paper, direction=directions[agent.id], own_papers=own_papers
            )
            if result is None:
                continue
            prompt, response_format = result

            batch_prompts.append(prompt)
            batch_metadata.append({'paper_idx': i, 'paper': submitted_paper, 'agent': agent, 'own_papers': own_papers})
            if self_citation_response_format is None:
                self_citation_response_format = response_format

        for attempt in range(max_attempts):
            if not batch_prompts:
                break

            batch_results = self.llm.generate_batch(
                batch_prompts, response_format=self_citation_response_format,
                desc=f"{len(batch_prompts)} self-citations (attempt {attempt+1}/{max_attempts})",
                seed_ctx=('phase2_citations_self', year, attempt),
            )

            items_needing_retry = []
            for j, (result, _) in enumerate(batch_results):
                meta = batch_metadata[j]
                idx = meta['paper_idx']

                if result is None or result.get('self_citations') is None:
                    items_needing_retry.append(meta)
                    continue

                self_cites = result['self_citations']
                num_own = len(meta['own_papers'])
                if not all(isinstance(c, int) and 0 <= c < num_own for c in self_cites):
                    items_needing_retry.append(meta)
                    continue

                self_citations_dict[idx] = [meta['own_papers'][c].id for c in self_cites]

            if not items_needing_retry:
                break

            batch_prompts = []
            batch_metadata = []
            for meta in items_needing_retry:
                result = meta['agent'].get_self_citation_prompt(
                    paper=meta['paper'], direction=directions[meta['agent'].id], own_papers=meta['own_papers']
                )
                if result is None:
                    continue
                prompt, _ = result
                batch_prompts.append(prompt)
                batch_metadata.append(meta)

        return self_citations_dict

    def _record_batch_citations(self, year, directions, submissions_list, accepted_indices, all_indices, accepted_ids):
        """Select accepted-paper citations, extend sparse lists, then add self-citations."""
        regular, response_format = self._select_citations_batch(
            year, directions, submissions_list, range(len(submissions_list)),
            accepted_indices, 'regular', accepted_ids=accepted_ids)
        for index, paper in enumerate(submissions_list):
            if index not in regular:
                logger.warning(f"Paper {paper['id']} failed all citation retries, using empty citations")
                regular[index] = []

        sparse_indices = [
            index for index in range(len(submissions_list))
            if len(regular[index]) <= SIMULATION_CONFIG['citation']['expected_citation_per_round']
        ]
        extended, _ = self._select_citations_batch(
            year, directions, submissions_list, sparse_indices, all_indices, 'extended', response_format)
        self_citations = self._select_self_citations_batch(year, directions, submissions_list)
        for index, paper in enumerate(submissions_list):
            citations = extended.get(index, regular.get(index, []))
            if index in self_citations:
                citations = list(set(citations + self_citations[index]))
            self.citation_tracker.add_citations(paper['id'], citations, year)

    def _select_citations_sequential(self, agent, paper, direction, candidates, institution):
        """Select candidate IDs, preserving the single-request failure policy."""
        prompt, response_format = agent.get_citation_prompt(
            paper=paper, direction=direction, candidate_papers=candidates,
            max_papers=30, citing_institution=institution)
        num_retries = 0
        while True:
            try:
                response, _ = self.llm.generate(prompt=prompt, response_format=response_format)
                assert response.get('citations') is not None
                indices = response['citations']
                assert all(isinstance(index, int) and 0 <= index < len(candidates) for index in indices)
                return [candidates[index]['id'] for index in indices]
            except Exception as error:
                logger.warning(f"Error getting potential cited papers: {error}")
                traceback.print_exc()
                num_retries += 1
                if num_retries >= self.max_retries:
                    raise Exception("Failed to get potential cited papers after 3 retries")

    def _select_self_citations_sequential(self, agent, paper, direction, year):
        """Return selected self-citations, or None when unavailable or retries fail."""
        logger.info(f"[Self-Citation] Getting self-citations for {agent.id} in year {year}.")
        own_papers = [p for p in self.paper_tracker.get_papers_by_author(agent.id) if p.project_end_year < year]
        if not own_papers:
            return None
        request = agent.get_self_citation_prompt(paper=paper, direction=direction, own_papers=own_papers)
        if not request:
            return None
        prompt, response_format = request
        num_retries = 0
        while True:
            try:
                response, _ = self.llm.generate(prompt=prompt, response_format=response_format)
                assert response.get('self_citations') is not None
                indices = response['self_citations']
                assert all(isinstance(index, int) and 0 <= index < len(own_papers) for index in indices)
                return [own_papers[index].id for index in indices]
            except Exception as error:
                logger.warning(f"Error getting self-citations: {error}")
                traceback.print_exc()
                num_retries += 1
                if num_retries >= self.max_retries:
                    logger.warning("Failed to get self-citations after retries, skipping self-citation")
                    return None

    def _record_sequential_citations(self, year, directions, submissions_list,
                                     topk_indices_citation_accepted, topk_indices_citation_both, accepted_papers_arxiv_ids):
        """Select accepted-paper citations, extend sparse lists, and merge self-citations."""
        for index, paper in enumerate(submissions_list):
            agent = self.ecosystem.get_agent_by_id(self._get_lead_author_id(paper))
            candidates = self._build_candidate_papers(
                self.rag.documents,
                self._rerank_citation_candidates(
                    topk_indices_citation_accepted[index], SIMULATION_CONFIG['citation']['num_candidate_papers']))
            assert {candidate['id'] for candidate in candidates} <= accepted_papers_arxiv_ids, (
                f"Submitted paper {paper['id']} is trying to cite non-accepted papers")
            institution = getattr(agent, 'university_name', None) or getattr(agent, 'company_name', None)
            citations = self._select_citations_sequential(agent, paper, directions[agent.id], candidates, institution)
            if len(citations) <= SIMULATION_CONFIG['citation']['expected_citation_per_round']:
                candidates = self._build_candidate_papers(
                    self.rag.documents,
                    self._rerank_citation_candidates(
                        topk_indices_citation_both[index], SIMULATION_CONFIG['citation']['num_candidate_papers']))
                citations = self._select_citations_sequential(agent, paper, directions[agent.id], candidates, institution)
            self_citations = self._select_self_citations_sequential(agent, paper, directions[agent.id], year)
            if self_citations is not None:
                citations = list(set(citations + self_citations))
            self.citation_tracker.add_citations(paper['id'], citations, year)

    def _charge_annual_costs(self, year: int):
        """Charge annual costs for all active researchers working on projects

        Args:
            year: Current year number
        """
        for agent in self.ecosystem.get_available_authors():
            if not agent.is_active or agent.project_end_year < year:
                continue
            annual_cost = SIMULATION_CONFIG['conference']['annual_cost']
            from utopia.funding.accounting import project_id
            if self._account_resources(agent, -annual_cost, category='annual_research_cost',
                                       year=year, project=project_id(agent)):
                self.funding_tracker.record_funding_consumption(year, annual_cost, agent.get_type())

    def _run_phase_3_peer_review(self, year: int, year_results: Dict) -> Dict[str, Dict[str, float]]:
        """Phase 3: Conduct peer reviews

        Args:
            year: Current year number
            year_results: Year results dict to update

        Returns:
            average_scores: Dict mapping conf_id -> paper_id -> average score
        """
        logger.info(f"Year {year} Phase 3: Conducting peer reviews...")

        self.conference_system.start_all_review_processes()

        # Map conference ids to dict of papers
        average_scores = {}
        reviews_by_conference = {}  # {conf_id: {paper_id: [reviews]}}
        reviews_by_reviewer = defaultdict(list)  # {reviewer_id: [scores]}

        # Initialize structures for all conferences
        for conference in self.conference_system.conferences:
            average_scores[conference.conference_id] = {}
            reviews_by_conference[conference.conference_id] = {}

        # Check if LLM supports batching (vLLM)
        supports_batching = hasattr(self.llm, 'generate_batch')

        if supports_batching:
            # ==================== BATCHED PATH (vLLM) ====================
            # Step 1: Collect ALL review tasks (NO LLM calls yet)
            batch_prompts = []
            batch_metadata = []  # Store (reviewer, paper, conference) per prompt
            peer_review_response_format = None
            # Capacity-aware assignment (factor K) is computed up front from a dedicated
            # RNG; the legacy path below is untouched when the flags are off.
            capacity_assignment = (self._assign_reviewers_with_capacity(year, year_results)
                                   if self._capacity_mode_on else None)
            for conference in self.conference_system.conferences:
                for paper in conference.submitted_papers:
                    if capacity_assignment is not None:
                        selected_reviewers = capacity_assignment[paper['id']]
                    else:
                        # Assign reviewers (excluding all paper authors + their COIs)
                        author_ids = self._get_all_author_ids(paper)
                        all_coi = set(author_ids)
                        for aid in author_ids:
                            author = self.ecosystem.get_agent_by_id(aid)
                            all_coi.update(author.conflict_of_interest)
                        reviewers = self.ecosystem.get_available_reviewers(
                            exclude_agent_ids=list(all_coi)
                        )

                        # Select 3 random reviewers
                        selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                    for reviewer in selected_reviewers:
                        # reviewer: SimulationAgent
                        author_info = self._build_author_info_for_review(reviewer, paper)
                        prompt, response_format = reviewer.get_review_prompt(
                            paper, author_info=author_info, review_policy=self.review_policy)

                        if peer_review_response_format is None:
                            peer_review_response_format = response_format

                        batch_prompts.append(prompt)

                        batch_metadata.append({
                            'reviewer': reviewer,
                            'paper': paper,
                            'conference': conference,
                            'author_info': author_info,
                        })

            # Step 2: Single batch call for ALL prompts
            if batch_prompts:
                batch_results = self.llm.generate_batch(batch_prompts, temperature=0.7, response_format=peer_review_response_format, desc=f"{len(batch_prompts)} peer reviews", seed_ctx=('phase3_reviews', year))

                # Step 3: Process results
                for i, (review_json, message_history) in enumerate(batch_results):
                    meta = batch_metadata[i]
                    reviewer = meta['reviewer']
                    paper = meta['paper']
                    conference = meta['conference']

                    # Handle failed JSON parse from batch
                    if review_json is None:
                        logger.warning(f"Failed to parse review response for paper {paper['id']} by {reviewer.id}, skipping")
                        continue

                    # Validate response
                    try:
                        assert all(field in review_json for field in ['overall_score', "justification"])
                        review_json['overall_score'] = self._cast_review_score(review_json['overall_score'])
                        review = {
                            'overall_score': review_json['overall_score'],
                            'justification': review_json['justification'],
                            'reviewer_id': reviewer.id,
                            'conference_id': paper['conference'],
                            'year': paper['year'],
                            'arxiv_id': paper['id'],
                            'review_time': time.time(),
                        }
                        conference.add_review(paper['id'], review)
                        reviews_by_reviewer[reviewer.id].append(review['overall_score'])

                        # Record distance-score pair for preferential attachment analysis
                        if self.experiment_name == 'preferential_attachment' and self.network_metrics:
                            ai = meta.get('author_info')
                            self.network_metrics.record_review_with_distance(
                                year=year,
                                reviewer_id=reviewer.id,
                                author_id=self._get_lead_author_id(paper),
                                distance=ai['network_distance'] if ai else -1,
                                score=review['overall_score'],
                                paper_id=paper['id'],
                            )
                    except (KeyError, AssertionError, ValueError) as e:
                        raise ValueError(f"Invalid review from {reviewer.id} for paper {paper['id']}: {e}")
                        

            # Step 4: Calculate average scores for each paper
            for conference in self.conference_system.conferences:
                for paper in conference.submitted_papers:
                    paper_reviews = conference.reviews.get(paper['id'], [])
                    if paper_reviews:
                        reviews_by_conference[conference.conference_id][paper['id']] = paper_reviews
                        average_scores[conference.conference_id][paper['id']] = sum(
                            r['overall_score'] for r in paper_reviews) / len(paper_reviews)
                        logger.debug(f"[Review] \"{paper['title']}\": Average score {average_scores[conference.conference_id][paper['id']]:.2f}")
                    else:
                        logger.warning(f"No valid reviews for paper {paper['id']}")
                        # Assign a default score to avoid missing papers
                        average_scores[conference.conference_id][paper['id']] = 2.5

        else:
            # ==================== SEQUENTIAL PATH (OpenAI API) ====================
            assert not self._capacity_mode_on and self.review_policy == 'persona', \
                "reviewer capacity/matching and standardized review require the batched path"
            for conference in tqdm(self.conference_system.conferences, desc="Peer reviews"):
                for paper in tqdm(conference.submitted_papers, desc=f"{conference.conference_id}"):
                    # Assign reviewers (excluding all paper authors + their COIs)
                    author_ids = self._get_all_author_ids(paper)
                    all_coi = set(author_ids)
                    for aid in author_ids:
                        author = self.ecosystem.get_agent_by_id(aid)
                        all_coi.update(author.conflict_of_interest)
                    reviewers = self.ecosystem.get_available_reviewers(
                        exclude_agent_ids=list(all_coi)
                    )

                    # Select 3 random reviewers
                    selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                    for reviewer in selected_reviewers:
                        author_info = self._build_author_info_for_review(reviewer, paper)
                        num_retries = 0
                        review = None
                        while True:
                            try:
                                review = reviewer.review_paper(paper, author_info=author_info)
                                conference.add_review(paper['id'], review)
                                reviews_by_reviewer[reviewer.id].append(review['overall_score'])

                                # Record distance-score pair for preferential attachment analysis
                                if self.experiment_name == 'preferential_attachment' and self.network_metrics:
                                    self.network_metrics.record_review_with_distance(
                                        year=year,
                                        reviewer_id=reviewer.id,
                                        author_id=self._get_lead_author_id(paper),
                                        distance=author_info['network_distance'] if author_info else -1,
                                        score=review['overall_score'],
                                        paper_id=paper['id'],
                                    )

                                break

                            except json.decoder.JSONDecodeError as e:
                                logger.warning(f"Error reviewing paper: {e}")

                                if not "Unterminated string" in str(e):
                                    traceback.print_exc()

                                num_retries += 1
                                if num_retries >= self.max_retries:
                                    raise Exception("Failed to review paper after 3 retries")
                                time.sleep(1)

                            except Exception as e:
                                logger.warning(f"Error reviewing paper: {e}")
                                traceback.print_exc()
                                num_retries += 1
                                if num_retries >= self.max_retries:
                                    raise Exception("Failed to review paper after 3 retries")
                                time.sleep(1)

                    paper_reviews = conference.reviews[paper['id']]
                    reviews_by_conference[conference.conference_id][paper['id']] = paper_reviews
                    average_scores[conference.conference_id][paper['id']] = sum(
                        r['overall_score'] for r in paper_reviews) / len(paper_reviews)
                    logger.debug(f"[Review] \"{paper['title']}\": Average score {average_scores[conference.conference_id][paper['id']]:.2f}")

        # Calculate score metrics
        conference_score_metrics = calculate_conference_score_metrics(reviews_by_conference, dict(reviews_by_reviewer))

        year_results['peer_review'] = {
            'reviews_conducted': sum(len(c.reviews) for c in self.conference_system.conferences),
            'score_metrics': conference_score_metrics,
        }

        # Process received reviews for authors
        logger.info(f"Year {year} Phase 3: Processing author experiences from received reviews...")
        for conference in self.conference_system.conferences:
            for paper in conference.submitted_papers:
                paper_reviews = conference.reviews.get(paper['id'], [])
                if paper_reviews:
                    # Send reviews to ALL co-authors
                    for aid in self._get_all_author_ids(paper):
                        author = self.ecosystem.get_agent_by_id(aid)
                        author.process_received_reviews(paper_reviews, paper_id=paper['id'], year=year,
                                                        average_scores_of_venue=average_scores[conference.conference_id])

        return average_scores

    def _cast_review_score(self, value):
        """--review_score_mode: 'int' (legacy truncation) or 'float' (keep decimals)."""
        return float(value) if self.review_score_mode == 'float' else int(value)

    def _assign_reviewers_with_capacity(self, year: int, year_results: Dict) -> Dict[str, List]:
        """Capacity-aware reviewer assignment (scale experiment factor K).

        reviews per paper r = min(reviews_per_paper, floor(c * N_reviewers / N_submissions))
        with a hard floor of 1 when a per-reviewer annual cap c is set (r = reviews_per_paper
        when c is None). Papers are visited in a seeded random order; each takes up to r
        eligible reviewers (no author, no conflict of interest, remaining capacity). With
        --reviewer_matching topic, reviewers whose expertise category overlaps the paper's
        arXiv tags are drawn first. Fails fast (RuntimeError) if any paper would receive
        zero reviews, so the preregistered capacity range must guarantee >= 1 review.

        Uses a dedicated random.Random so the global RNG stream is never consumed.
        Returns paper_id -> [reviewer agents]; assignment stats go to year_results.
        """
        papers = [paper for conf in self.conference_system.conferences for paper in conf.submitted_papers]
        reviewers_all = self.ecosystem.get_available_reviewers()
        n_sub, n_rev = len(papers), len(reviewers_all)
        c = self.reviewer_capacity
        r_target = self.reviews_per_paper
        if n_sub == 0:
            year_results['review_assignment'] = {'n_submissions': 0, 'n_reviewers': n_rev}
            return {}
        if c is not None:
            if c * n_rev < n_sub:
                raise RuntimeError(
                    f"Year {year}: reviewer capacity {c} x {n_rev} reviewers < {n_sub} submissions; "
                    f"cannot give every paper one review (outside the preregistered range)")
            r = max(1, min(r_target, (c * n_rev) // n_sub))
        else:
            r = r_target

        rng = random.Random(derive_seed(getattr(self.args, 'seed', 42), 'phase3_assign', year))
        order = list(range(n_sub))
        rng.shuffle(order)

        # Topic matching via coarse categories: reviewer expertise topic -> category
        # (TOPIC_TO_CATEGORY_MAPPING); paper arXiv tag -> categories of conferences that
        # list the tag (derived from the conference roster).
        tag_to_categories = None
        reviewer_categories = {}
        if self.reviewer_matching == 'topic':
            from utopia.agents.conference import CONFERENCES, TOPIC_TO_CATEGORY_MAPPING
            tag_to_categories = defaultdict(set)
            for conf in CONFERENCES.values():
                for tag in conf.topics:
                    tag_to_categories[tag.lower()].add(conf.category)
            for a in reviewers_all:
                reviewer_categories[a.id] = {
                    TOPIC_TO_CATEGORY_MAPPING.get(d.topic, 'Interdisciplinary')
                    for d in getattr(a, 'expertise', [])}

        load = defaultdict(int)
        assignment = {}
        for idx in order:
            paper = papers[idx]
            author_ids = self._get_all_author_ids(paper)
            all_coi = set(author_ids)
            for aid in author_ids:
                author = self.ecosystem.get_agent_by_id(aid)
                if author is not None:
                    all_coi.update(author.conflict_of_interest)
            eligible = [a for a in reviewers_all
                        if a.id not in all_coi and (c is None or load[a.id] < c)]
            if tag_to_categories is not None:
                paper_cats = set()
                for tag in paper.get('tags', []):
                    paper_cats |= tag_to_categories.get(str(tag).lower(), set())
                matched = [a for a in eligible if reviewer_categories.get(a.id, set()) & paper_cats]
                unmatched = [a for a in eligible if a not in matched]
                chosen = rng.sample(matched, min(r, len(matched)))
                if len(chosen) < r and unmatched:
                    chosen += rng.sample(unmatched, min(r - len(chosen), len(unmatched)))
            else:
                chosen = rng.sample(eligible, min(r, len(eligible)))
            if not chosen:
                raise RuntimeError(f"Year {year}: paper {paper['id']} has no eligible reviewer "
                                   f"with remaining capacity (c={c}, n_rev={n_rev}, n_sub={n_sub})")
            for a in chosen:
                load[a.id] += 1
            assignment[paper['id']] = chosen

        loads = [load[a.id] for a in reviewers_all]
        year_results['review_assignment'] = {
            'n_submissions': n_sub, 'n_reviewers': n_rev, 'reviewer_capacity': c,
            'reviews_per_paper_target': r, 'reviewer_matching': self.reviewer_matching,
            'reviews_assigned': int(sum(loads)), 'max_load': int(max(loads)),
            'mean_load': float(np.mean(loads)),
            'papers_below_target': int(sum(1 for v in assignment.values() if len(v) < r)),
        }
        logger.info(f"Year {year} reviewer assignment: {n_sub} papers, {n_rev} reviewers, "
                    f"r={r}, cap={c}, max load {max(loads)}")
        return assignment

    def _fixed_budget_slots(self, eval_metadata: List[Dict]) -> Tuple[Dict[str, int], Dict]:
        """--funding_budget_mode fixed: annual agency slots anchored to N0.

        total_slots = round(f * N0); split across programs by largest remainder over this
        year's application counts (so relative program sizes are preserved and the budget,
        not the application count, bounds the number of winners).
        """
        assert self.initial_university_count, "N0 unknown: population not initialized"
        apps_by_program = {}
        for meta in eval_metadata:
            apps_by_program[meta['program_id']] = apps_by_program.get(meta['program_id'], 0) + len(meta['apps'])
        program_ids = list(apps_by_program.keys())
        total_slots = int(round(self.funding_budget_slots_frac * self.initial_university_count))
        shares = FundingAgency.allocate_slots_largest_remainder(
            total_slots, [apps_by_program[p] for p in program_ids])
        slot_override = dict(zip(program_ids, shares))
        info = {
            'mode': 'fixed', 'slots_frac': self.funding_budget_slots_frac,
            'initial_university_count': self.initial_university_count,
            'total_slots': total_slots, 'total_applications': int(sum(apps_by_program.values())),
            'slots_by_program': slot_override, 'apps_by_program': apps_by_program,
            'effective_rate_by_program': {p: (slot_override[p] / apps_by_program[p] if apps_by_program[p] else None)
                                          for p in program_ids},
        }
        return slot_override, info

    def _run_phase_4_acceptance_decisions(self, year: int, year_results: Dict):
        """Phase 4: Make acceptance decisions

        Args:
            year: Current year number
            year_results: Year results dict to update
        """
        logger.info(f"Year {year} Phase 4: Making acceptance decisions...")

        schedule = getattr(self.args, 'acceptance_rate_schedule', None)
        if schedule:
            # Per-year override applied to every conference before decisions; the year-N
            # checkpoint (written after Phase 4) therefore records the year-N rate.
            rate = schedule[year - 1]
            for conf in self.conference_system.conferences:
                conf.acceptance_rate = rate
            logger.info(f"Year {year}: acceptance rate set to {rate:.3f} for all conferences "
                        f"(schedule override)")

        tiebreak_seed = (derive_seed(getattr(self.args, 'seed', 42), 'phase4_tiebreak', year)
                         if self.acceptance_tiebreak == 'seeded' else None)
        self.conference_system.make_all_decisions(tiebreak_seed=tiebreak_seed,
                                                  acceptance_mode=self.acceptance_mode)

        total_accepted = sum(len(conf.decisions['accept']) for conf in self.conference_system.conferences)
        total_rejected = sum(len(conf.decisions['reject']) for conf in self.conference_system.conferences)

        # Update paper decisions in tracker
        accepted_paper_ids, rejected_paper_ids = [], []
        for conf in self.conference_system.conferences:
            for paper in conf.decisions['accept']:
                accepted_paper_ids.append(paper['id'])
                self.paper_tracker.add_or_update_paper(paper)
            for paper in conf.decisions['reject']:
                rejected_paper_ids.append(paper['id'])
                self.paper_tracker.add_or_update_paper(paper)

        assert len(
            self.paper_tracker.pending_papers_by_year[year]) == 0, f"There should be NO pending papers for year {year}"

        self.rag.mark_papers_as_submitted(accepted_paper_ids, rejected_paper_ids)

        year_results['decisions'] = {
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
        }

        logger.info(f"  Accepted: {total_accepted}, Rejected: {total_rejected}")

    def _validate_single_funding_proposal(self, proposal, metadata):
        """Validate a single-program funding proposal from batch generation.

        Returns:
            (is_valid, cleaned_proposal_or_None, reason)
        """
        
        if proposal is None:
            return False, None, "proposal is None"
        if not isinstance(proposal, dict):
            return False, None, f"proposal is not a dict (got {type(proposal).__name__})"

        all_papers = metadata['all_papers']  # ordered list of {'id': ..., 'status': ...}
        num_papers = len(all_papers)

        # Check required fields
        if 'submit' not in proposal:
            return False, None, "missing 'submit' field"
        if 'research_proposal' not in proposal:
            return False, None, "missing 'research_proposal' field"
        if 'relevant_projects' not in proposal:
            return False, None, "missing 'relevant_projects' field"
        if not isinstance(proposal.get('relevant_projects'), list):
            return False, None, "relevant_projects is not a list"

        # Validate integer indices and convert to {arxiv_id, status} dicts
        if proposal.get('submit', True):
            indices = proposal.get('relevant_projects', [])
            valid_papers = []
            invalid_count = 0
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < num_papers:
                    p = all_papers[idx]
                    valid_papers.append({"arxiv_id": p['id'], "status": p['status']})
                else:
                    invalid_count += 1
            if indices and invalid_count > len(indices) * 0.5:
                return False, None, f">50% invalid paper indices ({invalid_count}/{len(indices)})"
            if invalid_count > 0:
                logger.warning(f"Agent {metadata['agent'].id} program {metadata['program_id']}: filtered {invalid_count} invalid paper indices")
            proposal['relevant_projects'] = valid_papers

        return True, proposal, "ok"

    def _run_phase_5_update_funding(self, year: int, submissions_list: List[Dict], year_results: Dict):
        """Allocate industry income, collect and evaluate grants, then close the funding cycle."""
        logger.info(f"Year {year} Phase 5: Updating funding...")
        consumed = self.funding_tracker.get_funding_consumption(year)
        logger.info(f"Funding consumed this cycle: University: {consumed['university']:.2f}, "
                    f"Industry: {consumed['industry']:.2f}")
        active_papers = self.paper_tracker.get_papers_dataframe(year)
        funding_agencies = [
            agent for agent in self.ecosystem.agent_population.values()
            if isinstance(agent, FundingAgency)
        ]
        self._allocate_industry_funding(year, active_papers, year_results)
        applications = self._collect_funding_applications(year, funding_agencies)
        cost_events = charge_application_costs(
            applications, getattr(self.args, 'funding_application_cost', 0) or 0, year)
        winners = self._evaluate_funding_applications(year, applications, funding_agencies, year_results)
        funding_per_unit = self._allocate_university_funding(year, winners, active_papers)
        self._record_funding_costs(year, cost_events, winners, funding_per_unit)
        self._finish_funding_cycle(year, year_results, winners, applications)

    def _allocate_industry_funding(self, year, active_papers_dataframe, year_results):
        """Credit accepted industry papers using the configured annual funding rule."""
        logger.info(f"[Funding Allocation] Update fundings for industry researchers in year {year}")
        # Calculate the number of recipients of industry funding
        denominator = active_papers_dataframe.loc[
                (active_papers_dataframe['author_type'] == 'industry')
                & (active_papers_dataframe['status'] == 'accept'),
                'maturity'
            ].sum()

        if denominator == 0:
            industry_funding_per_unit = 0
        elif self.funding_budget_mode == 'fixed':
            # factor R for the industry sector: fixed annual pool instead of pay-per-paper
            industry_funding_per_unit, industry_budget_info = self._fixed_industry_pool_per_unit(int(denominator))
            year_results['funding_budget_industry'] = industry_budget_info
            logger.info(f"Year {year} fixed industry pool: {industry_budget_info['pool']} for "
                        f"{industry_budget_info['maturity_units_accepted']} maturity units")
        elif self.funding_allocation_mode == 'fixed':
            industry_funding_per_unit = SIMULATION_CONFIG['funding']['industry_base_budget']
        else:
            industry_funding_per_unit = max(self.funding_tracker.get_funding_allocation_amount(year, 'industry') // denominator, 1)

        for agent in tqdm(self.ecosystem.get_available_authors(), desc="Updating funding"):
            if not isinstance(agent, (IndustryResearcher, UniversityResearcher)):
                raise ValueError(f"Unknown agent type: {type(agent)}")
            if not isinstance(agent, IndustryResearcher):
                continue
            for paper in self.conference_system.authors_to_accepted_papers.get(agent.id, []):
                new_budget = self.industry_funding_system.update_budget(
                    funding_per_unit=industry_funding_per_unit,
                    accepted=1,
                    maturity=paper['maturity'],
                    weighted_funding_assignment=True,
                )
                if new_budget <= 0:
                    continue
                if self._account_resources(agent, int(new_budget), category='industry_income',
                                           year=year, paper=paper['id']):
                    self.funding_tracker.record_funding_allocation(year, new_budget, 'industry')

    def _collect_funding_applications(self, year, funding_agencies):
        """Prepare each selected program and group successful proposals by applicant."""
        # Batch process university researcher funding applications
        all_programs = {pid: program for funding_agent in funding_agencies for pid, program in funding_agent.funding_programs.items()}
        university_researchers = [
            agent for agent in self.ecosystem.get_available_authors()
            if isinstance(agent, UniversityResearcher)
        ]

        # Collect prompts and metadata for batch processing — one prompt per (agent, program)
        batch_prompts = []
        batch_metadata = []
        funding_response_format = None

        for agent in university_researchers:
            past_accepted = self.conference_system.authors_to_accepted_papers.get(agent.id, [])
            past_rejected = self.conference_system.authors_to_rejected_papers.get(agent.id, [])

            selected_program_ids = agent.select_funding_programs(
                all_programs, past_accepted, past_rejected
            )

            for program_id in selected_program_ids[:3]:
                prompt, response_format, metadata = agent.funding_application_prompt(
                    program=all_programs[program_id],
                    past_accepted_papers=past_accepted,
                    past_rejected_papers=past_rejected,
                )
                if not prompt:
                    continue
                batch_prompts.append(prompt)
                batch_metadata.append(metadata)
                if funding_response_format is None:
                    funding_response_format = response_format

        if not batch_prompts:
            return []
        if hasattr(self.llm, 'generate_batch'):
            successful_items = self._generate_funding_applications_batch(
                year, all_programs, batch_prompts, batch_metadata, funding_response_format)
        else:
            successful_items = self._generate_funding_applications_sequential(batch_prompts, batch_metadata)

        agent_applications = {}
        for agent_id, program_id, app_data, agent in successful_items:
            app_data['author'] = agent
            agent_applications.setdefault(agent_id, {})[program_id] = app_data
        return list(agent_applications.values())

    def _generate_funding_applications_batch(self, year, all_programs, batch_prompts, batch_metadata, funding_response_format):
        """Retry failed proposal requests while retaining successful proposals in arrival order."""
        MAX_FUNDING_RETRIES = 5
        # Collect successful (agent_id, program_id, app_data) tuples
        successful_items = []

        for attempt in range(MAX_FUNDING_RETRIES):
            if not batch_prompts:
                break

            try:
                batch_results = self.llm.generate_batch(
                    batch_prompts, temperature=0.7,
                    response_format=funding_response_format,
                    desc=f"{len(batch_prompts)} funding applications (attempt {attempt+1}/{MAX_FUNDING_RETRIES})",
                    seed_ctx=('phase5_funding_apps', year, attempt),
                )
            except Exception as e:
                logger.warning(f"Batch funding generation failed on attempt {attempt+1}: {e}")
                if attempt < MAX_FUNDING_RETRIES - 1:
                    time.sleep(1)
                continue

            items_needing_retry = []
            for j, (proposal, _) in enumerate(batch_results):
                meta = batch_metadata[j]
                agent = meta['agent']
                program_id = meta['program_id']

                is_valid, cleaned, reason = self._validate_single_funding_proposal(proposal, meta)
                if not is_valid:
                    logger.warning(f"Funding proposal invalid for agent {agent.id} program {program_id} (attempt {attempt+1}): {reason}")
                    items_needing_retry.append(meta)
                    continue

                successful_items.append((agent.id, program_id, cleaned, agent))

            if not items_needing_retry:
                batch_prompts = []
                batch_metadata = []
                break

            # Rebuild batch with only failed items
            batch_prompts = []
            batch_metadata = []
            for meta in items_needing_retry:
                agent = meta['agent']
                prompt, _, new_meta = agent.funding_application_prompt(
                    program=all_programs[meta['program_id']],
                    past_accepted_papers=self.conference_system.authors_to_accepted_papers.get(agent.id, []),
                    past_rejected_papers=self.conference_system.authors_to_rejected_papers.get(agent.id, []),
                )
                assert prompt
                batch_prompts.append(prompt)
                batch_metadata.append(new_meta)

        # Log items that failed all retries
        if batch_prompts:
            for meta in batch_metadata:
                logger.warning(f"Agent {meta['agent'].id} program {meta['program_id']} funding application failed all {MAX_FUNDING_RETRIES} retries, skipping")

        return successful_items

    def _generate_funding_applications_sequential(self, batch_prompts, batch_metadata):
        """Generate proposals with the existing single-request retry behavior."""
        sequential_items = []
        for i, prompt in enumerate(tqdm(batch_prompts, desc="Generating funding applications")):
            metadata = batch_metadata[i]
            agent = metadata['agent']
            program_id = metadata['program_id']
            num_retries = 0
            while True:
                try:
                    proposal, _ = self.llm.generate(prompt=prompt)

                    is_valid, cleaned, reason = self._validate_single_funding_proposal(proposal, metadata)
                    if not is_valid:
                        raise ValueError(f"Invalid proposal: {reason}")

                    sequential_items.append((agent.id, program_id, cleaned, agent))
                    break

                except Exception as e:
                    logger.warning(f"Error preparing funding application: {e}")
                    traceback.print_exc()
                    num_retries += 1
                    if num_retries >= self.max_retries:
                        raise Exception("Failed to prepare funding application after retries")
                    time.sleep(1)

        return sequential_items

    def _rank_funding_applications(self, year, all_funding_proposal_eval_prompts, all_eval_metadata, eval_response_format):
        """Retry invalid rankings and preserve final failures for the configured fallback."""
        MAX_FUNDING_EVAL_RETRIES = 3
        final_eval_results = []
        final_eval_metadata = []

        for attempt in range(MAX_FUNDING_EVAL_RETRIES):
            if not all_funding_proposal_eval_prompts:
                break

            if hasattr(self.llm, 'generate_batch'):
                # Ranking ALL panel applications produces long JSON after thinking
                # tokens; the 2048 default truncated ~13% of program-years in E3.
                eval_results = self.llm.generate_batch(
                    all_funding_proposal_eval_prompts, response_format=eval_response_format,
                    max_tokens=8192,
                    desc=f"{len(all_funding_proposal_eval_prompts)} funding evaluations (attempt {attempt+1}/{MAX_FUNDING_EVAL_RETRIES})",
                    seed_ctx=('phase5_funding_eval', year, attempt),
                )
            else:
                eval_results = []
                for prompt in all_funding_proposal_eval_prompts:
                    result, msg_hist = self.llm.generate(prompt=prompt, response_format=eval_response_format)
                    eval_results.append((result, msg_hist))

            # Validate each result, separate valid from invalid
            retry_prompts = []
            retry_metadata = []

            for i, (result, msg_hist) in enumerate(eval_results):
                if FundingAgency.validate_funding_result(result, all_eval_metadata[i]['apps']):
                    final_eval_results.append((result, msg_hist))
                    final_eval_metadata.append(all_eval_metadata[i])
                    continue
                program_id = all_eval_metadata[i]['program_id']
                logger.warning(f"Funding eval for {program_id} invalid (attempt {attempt+1}/{MAX_FUNDING_EVAL_RETRIES})")
                if attempt < MAX_FUNDING_EVAL_RETRIES - 1:
                    retry_prompts.append(all_funding_proposal_eval_prompts[i])
                    retry_metadata.append(all_eval_metadata[i])
                    continue
                # Preserve final failures for the funding evaluator's fallback.
                final_eval_results.append((result, msg_hist))
                final_eval_metadata.append(all_eval_metadata[i])

            if not retry_prompts:
                break

            all_funding_proposal_eval_prompts = retry_prompts
            all_eval_metadata = retry_metadata

        return final_eval_results, final_eval_metadata

    def _evaluate_funding_applications(self, year, university_applications, funding_agencies, year_results):
        """Build funding panels and apply the configured ranking, novelty, and budget rules."""
        # Evaluate applications across all funding agencies
        all_funding_winners_dict, all_funding_programs_dict = {}, {}

        logger.debug(f"Calculating weights for funding applications in year {year}...")

        submitted_papers_dict = {pid: paper.to_dict() for pid, paper in self.paper_tracker.papers_by_id.items()}

        # Step 1: Collect prompts from ALL funding agencies into one batch
        all_funding_proposal_eval_prompts = []
        all_eval_metadata = []
        eval_response_format = None

        # Deterministic panelization keeps funding-ranking prompts bounded at scale;
        # fidelity vs global ranking is validated before confirmatory use.
        # CLI override supports the fidelity experiment (0 = disable panels).
        panel_max_apps = (SIMULATION_CONFIG['exploration_experiment']['funding_panel_max_apps']
                          if self.is_exploration_experiment else None)
        cli_panel = getattr(self.args, 'funding_panel_max_apps', None)
        if cli_panel is not None:
            panel_max_apps = cli_panel if cli_panel > 0 else None
        for funding_agent in funding_agencies:
            all_funding_programs_dict.update(funding_agent.funding_programs)
            funding_proposal_eval_prompts, resp_fmt, eval_metadata = funding_agent.get_funding_evaluation_prompts(
                university_applications, submitted_papers_dict=submitted_papers_dict,
                panel_max_apps=panel_max_apps,
                panel_seed=derive_seed(getattr(self.args, 'seed', 42), 'funding_panels', year),
            )
            all_funding_proposal_eval_prompts.extend(funding_proposal_eval_prompts)
            all_eval_metadata.extend(eval_metadata)
            if eval_response_format is None:
                eval_response_format = resp_fmt

        if not all_funding_proposal_eval_prompts:
            return {}
        final_eval_results, final_eval_metadata = self._rank_funding_applications(
            year, all_funding_proposal_eval_prompts, all_eval_metadata, eval_response_format)
        # Step 3: Process all results (valid + any final failures with fallback).
        # Funding-conservatism intervention: novelty penalties are applied to
        # scores BEFORE winner selection (plan 4.4); neutral mode passes none.
        novelty_penalties = None
        lambda_funding = 0.0
        if self.is_exploration_experiment and getattr(self.args, 'funding_intervention', False):
            applicant_ids = list({app['applicant_id']
                                  for meta in final_eval_metadata for app in meta['apps']})
            novelty_penalties = self._compute_recent_novelty_percentiles(applicant_ids)
            lambda_funding = SIMULATION_CONFIG['exploration_experiment']['novelty_penalty_in_funding']

        # Matthew-RD experiment: persist full pre-decision rankings so the
        # funding cutoff is reconstructable for winners AND losers.
        # Default-off flag; no behavior change for any other experiment.
        application_log = [] if getattr(self.args, 'log_funding_applications', False) else None
        # Fixed agency budget (factor R): slots anchored to N0 replace the
        # per-program int(n x rate) rule; 'track' (default) passes None.
        slot_override = None
        if self.funding_budget_mode == 'fixed':
            slot_override, budget_info = self._fixed_budget_slots(final_eval_metadata)
            year_results['funding_budget'] = budget_info
            logger.info(f"Year {year} fixed agency budget: {budget_info['total_slots']} slots "
                        f"for {budget_info['total_applications']} applications")
        all_funding_winners_dict = FundingAgency.process_funding_evaluation_results(
            final_eval_results, final_eval_metadata, all_funding_programs_dict,
            novelty_penalties=novelty_penalties, lambda_funding=lambda_funding,
            application_log=application_log, slot_override=slot_override,
        )
        if application_log is not None:
            app_log_path = os.path.join(self.output_dir,
                                        f'funding_applications_year_{year}.jsonl')
            with open(app_log_path, 'w') as f:
                for rec in application_log:
                    rec['year'] = year
                    f.write(json.dumps(rec) + '\n')
            logger.info(f"Funding application log: {len(application_log)} records "
                        f"-> {app_log_path}")

        return all_funding_winners_dict

    def _allocate_university_funding(self, year, all_funding_winners_dict, active_papers_dataframe):
        """Credit each awarded program and update funding history once per applied credit."""
        winners_list = [winner['applicant_id'] for winners in all_funding_winners_dict.values() for winner in winners]
        active_papers_dataframe = active_papers_dataframe.set_index('author_id')

        # Note: One agent can win multiple fundings

        if self.weighted_funding_assignment:
            valid_winners = [w for w in winners_list if w in active_papers_dataframe.index]
            denominator = active_papers_dataframe.loc[valid_winners, "maturity"].sum() if valid_winners else 0

        else:
            denominator = len(winners_list)

        if self.funding_allocation_mode == 'fixed':
            academic_funding_per_unit = SIMULATION_CONFIG['funding']['academic_base_budget']
        elif denominator == 0:
            academic_funding_per_unit = 0
        else:
            academic_funding_per_unit = int(
                self.funding_tracker.get_funding_allocation_amount(year, 'university') / denominator)

        # Apply funding decisions to university researchers and update success history
        for program_id, winners_for_program in all_funding_winners_dict.items():
            for winner in winners_for_program:
                agent = self.ecosystem.get_agent_by_id(winner['applicant_id'])

                funding_to_researcher = academic_funding_per_unit

                applied = self._account_resources(agent, funding_to_researcher, category='grant_income',
                                                  year=year, program=program_id)
                if not applied:
                    continue
                self.funding_tracker.record_funding_allocation(year, funding_to_researcher, 'university')
                agent.funding_success_history.setdefault(program_id, []).append({
                    'year': year,
                    'amount': funding_to_researcher
                })

        return academic_funding_per_unit

    def _record_funding_costs(self, year, funding_cost_events, all_funding_winners_dict, academic_funding_per_unit):
        """Write application costs after awards so records include final balances."""
        if not funding_cost_events:
            return
        won = {(w['applicant_id'], pid)
               for pid, winners in all_funding_winners_dict.items() for w in winners}
        for ev in funding_cost_events:
            funded = (ev['applicant_id'], ev['program_id']) in won \
                and not ev['withdrawn_unaffordable']
            ev['funded'] = funded
            ev['award_amount'] = academic_funding_per_unit if funded else 0
            agent = self.ecosystem.get_agent_by_id(ev['applicant_id'])
            ev['post_award_balance'] = getattr(agent, 'resources', None)
        cost_log_path = os.path.join(self.output_dir,
                                     f'funding_costs_year_{year}.jsonl')
        with open(cost_log_path, 'w') as f:
            for ev in funding_cost_events:
                f.write(json.dumps(ev) + '\n')
        logger.info(f"Funding cost log: {len(funding_cost_events)} records "
                    f"-> {cost_log_path}")

    def _finish_funding_cycle(self, year, year_results, all_funding_winners_dict, university_applications):
        """Apply the legacy attrition rule and record resources and ecosystem statistics."""
        for agent_id, agent in self.ecosystem.agent_population.items():
            if getattr(agent, 'resources', None) and agent.resources <= 10 and hasattr(agent,
                                                                                       'can_author') and agent.can_author and agent.is_active:
                print(f"Removing agent {agent.id} due to insufficient funding")
                self.ecosystem.remove_agent(agent.id)

        # Calculate agent behavior and ecosystem health metrics
        logger.debug(f"[Resource] Recording resources for agents in year {year}...")
        for agent_id, agent in self.ecosystem.agent_population.items():
            if getattr(agent, 'resources', None) is not None:
                self.agent_tracker.record_agent_resources(agent_id, agent.resources, year)

        funding_levels = {agent_id: agent.resources for agent_id, agent in self.ecosystem.agent_population.items() if
                          getattr(agent, 'resources', None) is not None}
        year_results['ecosystem_metrics'] = {
            'resource_distribution': {
                'mean': float(np.mean(list(funding_levels.values()))) if funding_levels else None,
                'std': float(np.std(list(funding_levels.values()))) if funding_levels else None,
                'gini': calculate_gini_coefficient(funding_levels.values())
            },
            'active_agents': len(
                [a for a in self.ecosystem.agent_population.values() if hasattr(a, 'can_author') and a.can_author]),
            'funding_success_by_type': {
                'university': len([w for p_id, winners in all_funding_winners_dict.items() for w in winners]),
                'total_applications': len(university_applications)
            }
        }

        # Get conference statistics
        year_results['conference_statistics'] = self.conference_system.get_all_statistics()

    def _apply_strategy_quality_adjustment(self, year: int):
        """SYNTHETIC STRESS TEST ONLY — never part of the main (neutral) results.

        Adds strategy-dependent bias/variance noise directly to review scores.
        Disabled by default (config review_strategy_adjustment=False) because it
        manufactures the expected outcome (plan 4.3/6.4).
        """
        if not self.is_exploration_experiment:
            return

        exp_config = SIMULATION_CONFIG['exploration_experiment']
        if not exp_config.get('review_strategy_adjustment', False):
            return
        bias_map = {
            'explorer': (exp_config['explorer_success_bias'], exp_config['explorer_variance']),
            'exploiter': (exp_config['exploiter_success_bias'], exp_config['exploiter_variance']),
            'cautious_explorer': (exp_config['cautious_success_bias'], exp_config['cautious_variance']),
        }

        for conference in self.conference_system.conferences:
            for paper in conference.submitted_papers:
                lead_id = self._get_lead_author_id(paper)
                agent = self.ecosystem.get_agent_by_id(lead_id)
                strategy = getattr(agent, 'exploration_strategy', 'balanced')

                if strategy not in bias_map:
                    continue

                bias, variance = bias_map[strategy]
                reviews = conference.reviews.get(paper['id'], [])
                for review in reviews:
                    adjustment = np.random.normal(bias, variance)
                    review['overall_score'] = max(1, min(10, review['overall_score'] + adjustment))

    def _assign_citation_potential(self, year: int):
        """Assign a deterministic latent citation-potential multiplier to each
        newly accepted paper (plan 6.3).

        Stored as metadata ONLY — observed citation counts always equal graph
        in-degree. The multiplier affects outcomes solely by reranking citation
        candidates when the citation intervention is enabled; in neutral runs
        (lambda_citation=0) it is inert bookkeeping.
        """
        if not self.is_exploration_experiment:
            return

        exp_config = SIMULATION_CONFIG['exploration_experiment']
        clip_lo, clip_hi = exp_config['citation_potential_clip']
        run_seed = getattr(self.args, 'seed', 42)

        for conf in self.conference_system.conferences:
            for paper in conf.decisions.get('accept', []):
                if paper['id'] in self._citation_potential_multipliers:
                    continue
                novelty_data = self._paper_novelty_scores.get(paper['id'])
                q = novelty_data[0] if novelty_data else 0.0
                # Deterministic seeded lognormal: mean and variance both grow
                # modestly with novelty (high novelty = high risk AND high potential)
                rng = random.Random(derive_seed(run_seed, 'citepot', paper['id']))
                multiplier = float(np.clip(
                    np.exp(rng.gauss(0.15 * q, 0.10 + 0.40 * q)), clip_lo, clip_hi))
                self._citation_potential_multipliers[paper['id']] = multiplier

    def _compute_recent_novelty_percentiles(self, applicant_ids: List[str]) -> Dict[str, float]:
        """Recent-novelty percentile per applicant for the funding-conservatism
        intervention (plan 6.2). Percentiles are computed among the given
        applicants; applicants with no scored papers get 0.0."""
        if self.embedding_tracker is None:
            return {}
        by_author = defaultdict(list)
        for pid, meta in self.embedding_tracker.paper_metadata.items():
            if pid in self._paper_novelty_scores:
                by_author[meta['author_id']].append(self._paper_novelty_scores[pid][0])
        avg_novelty = {aid: float(np.mean(by_author[aid]))
                       for aid in applicant_ids if by_author.get(aid)}
        if not avg_novelty:
            return {aid: 0.0 for aid in applicant_ids}
        values = np.array(sorted(avg_novelty.values()))
        return {aid: float(np.searchsorted(values, avg_novelty[aid], side='right') / len(values))
                if aid in avg_novelty else 0.0
                for aid in applicant_ids}

    def _rerank_citation_candidates(self, indices, k: int):
        """Select k citation candidates from semantically ranked indices.

        Neutral mode: top-k by semantic rank (unchanged legacy behavior).
        Citation intervention: rerank the top-2k by
        semantic_rank_score + lambda_citation * log(citation_potential_multiplier),
        keeping semantic relevance dominant; the LLM still makes the final choice.
        """
        indices = list(indices)
        if (not self.is_exploration_experiment
                or not getattr(self.args, 'citation_intervention', False)):
            return indices[:k]
        exp_config = SIMULATION_CONFIG['exploration_experiment']
        lam = exp_config.get('lambda_citation', 0.0)
        if lam <= 0:
            return indices[:k]
        pool = indices[:2 * k]
        scored = []
        for rank, idx in enumerate(pool):
            sem_score = 1.0 - rank / max(len(pool) - 1, 1)
            paper_id = self.rag.documents[idx].metadata['id']
            mult = self._citation_potential_multipliers.get(paper_id, 1.0)
            scored.append((sem_score + lam * np.log(mult), idx))
        scored.sort(key=lambda x: -x[0])
        return [idx for _, idx in scored[:k]]

    def run_one_round(self, year: int) -> Dict:
        """Run simulation for one year - orchestrates all phases

        Args:
            year: Current year number

        Returns:
            Dictionary with year results
        """
        # Year setup
        year_results = self._setup_year(year)

        # Phase 0: Resubmissions (year 2+)
        resubmissions_list = self._run_phase_0_resubmissions(year, year_results)
        self.wandb_logger.log_phase(year, 0, year_results, self, num_resubmissions=len(resubmissions_list))

        # Phase 1: Assign research directions
        directions, active_authors, avg_funding = self._run_phase_1_research_directions(year, year_results)
        self.wandb_logger.log_phase(year, 1, year_results, self, num_active_authors=len(active_authors), avg_funding=avg_funding)

        # Phase 1.5: Collaboration formation (preferential_attachment experiment only)
        self._run_phase_1_5_collaboration_formation(year, directions, active_authors, year_results)

        # Phase 2: Generate and submit papers
        submissions_list = self._run_phase_2_submit_papers(
            year, directions, active_authors, resubmissions_list, year_results
        )
        # Batched metadata (keywords + embeddings + novelty) for all new papers this year
        self._record_paper_metadata_batch(year)
        self.wandb_logger.log_phase(year, 2, year_results, self)

        # Charge annual costs
        self._charge_annual_costs(year)

        # Phase 3: Peer review
        average_scores = self._run_phase_3_peer_review(year, year_results)
        self.wandb_logger.log_phase(year, 3, year_results, self)

        # Apply strategy-dependent quality adjustment (between review and acceptance)
        self._apply_strategy_quality_adjustment(year)

        # Phase 4: Make acceptance decisions
        self._run_phase_4_acceptance_decisions(year, year_results)
        self.wandb_logger.log_phase(year, 4, year_results, self)

        # Assign latent citation-potential metadata to newly accepted papers
        self._assign_citation_potential(year)

        # Phase 5: Update funding
        self._run_phase_5_update_funding(year, submissions_list, year_results)

        # Collect exploration metrics (if enabled)
        self._collect_exploration_metrics(year)

        # Graph/temporal integrity checks, written per year (exploration experiment)
        if self.is_exploration_experiment:
            self._run_integrity_checks(year)

        # Collect preferential attachment metrics (if enabled)
        self._collect_preferential_attachment_metrics(year)

        # Get conference statistics
        year_results['conference_statistics'] = self.conference_system.get_all_statistics()
        if getattr(self, 'resource_ledger', None) is not None:
            year_results['resource_balances'] = self.resource_ledger.close_year(
                year, self.ecosystem.agent_population)
        self.wandb_logger.log_phase(year, 5, year_results, self)

        return year_results

    def run(self):
        """Run the complete simulation"""
        logger.info("Starting Academic Ecosystem Simulation")
        if not hasattr(self.llm, 'generate_batch') and (
                self.papers_per_project > 1 or self._capacity_mode_on
                or self.review_policy != 'persona'):
            raise NotImplementedError(
                "papers_per_project>1, reviewer capacity/matching and review_policy=standardized "
                "are implemented for the batched (vLLM) path only")
        logger.info(f"Configuration: {self.num_years} years")

        docs_dir = getattr(self.args, 'docs_dir', self.output_dir)
        from copy import deepcopy
        resolved_config = deepcopy(SIMULATION_CONFIG)
        resolved_config['project_cost_accounting'] = self.cost_policy
        resolved_config['conference']['resubmission_cost'] = self.cost_policy['resubmission_cost']
        write_run_manifest(
            docs_dir, status='running', args=self.args,
            resolved_config=resolved_config,
            extra={'experiment_id': getattr(self.args, 'experiment_id', self.experiment_name),
                   'model_id': getattr(self.llm, 'model_name', 'unknown'),
                   'run_seed': getattr(self.args, 'seed', None)})

        start_year = 1
        checkpoint_loaded = False
        loaded_year = None

        # Try to load from checkpoint unless always_rerun is specified
        if not self.always_rerun:
            # Try to find the most recent checkpoint
            for year in range(self.num_years, 0, -1):
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")
                if os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path + '.gz'):
                    logger.info(f"Found checkpoint at year {year}, attempting to load...")
                    if self.load_checkpoint(year):
                        start_year = year + 1
                        checkpoint_loaded = True
                        logger.info(f"Resuming simulation from year {start_year}")
                        break
        else:
            logger.info("--always-rerun specified, ignoring all checkpoints and starting from scratch")

        if not checkpoint_loaded:
            # Initialize agents
            t0 = time.time()
            self.initialize_agents()
            logger.info(f"Agents initialized in {time.time() - t0:.1f}s")

            # Select conferences based on agent expertise distribution
            t0 = time.time()
            
            logger.info("*** CONFERENCE SELECTION ***")


            # Get all agents with research capability
            research_agents = [
                agent for agent_id, agent in self.ecosystem.agent_population.items() if
                isinstance(agent, (UniversityResearcher, IndustryResearcher)) and agent.expertise
            ]

            # Select conferences proportionally to agent expertise
            num_conferences = getattr(self.args, 'num_conferences', None) or (1 if self.debug else 10)
            selected_conferences = select_conferences_for_simulation(
                agents=research_agents,
                num_conferences=num_conferences
            )

            # Create conference system with selected conferences
            self.conference_system = ConferenceSystem(conferences=selected_conferences)

            # Log conference selection results
            logger.info(f"\nSelected {len(selected_conferences)} conferences:")

            category_distribution = Counter(conf.category for conf in selected_conferences)

            for category, count in sorted(category_distribution.items()):
                logger.info(f"  {category}: {count} conferences")
                conferences_in_category = [
                    conf.name for conf in selected_conferences if conf.category == category
                ]
                for conf_name in conferences_in_category:
                    logger.info(f"    - {conf_name}")
            logger.debug(f"Conference selection completed in {time.time() - t0:.1f}s")

        # Initialize RAG (always reconstruct fresh from data source)
        logger.info("\nInitializing RAG system...")
        if checkpoint_loaded:
            logger.info(f"RAG will filter out {len(self.submitted_paper_ids)} already submitted papers")

        assert start_year >= 1, "Start year must be >= 1"
        if start_year == 1:
            for agent_id, agent in self.ecosystem.agent_population.items():
                if getattr(agent, 'resources', None) is not None:
                    self.agent_tracker.record_agent_resources(agent_id, agent.resources, 0)

        # Run simulation for each year
        for year in range(start_year, self.num_years + 1):
            year_results = self.run_one_round(year)
            self.yearly_results.append(year_results)
            self._save_checkpoint(year, phase=5)
        # Generate final report
        self._generate_final_report()
        self.wandb_logger.finish()

        write_run_manifest(
            docs_dir, status='complete',
            extra={'llm_call_stats': getattr(self.llm, 'call_stats', None),
                   'direction_fallback_stats': {str(k): v for k, v in
                                                self._direction_fallback_stats.items()},
                   'years_completed': self.num_years,
                   'dataset_identity': getattr(self.rag, 'dataset_identity', None),
                   'ordered_documents_sha256': getattr(self.rag, 'document_identity', None),
                   'retrieval_embedding': {'model': self.rag.config.embedding_model,
                                           'revision': self.rag.config.embedding_revision} if hasattr(self.rag, 'config') else None,
                   'metric_embedding': getattr(self.embedding_tracker, 'identity', None)})

        logger.info("\nSimulation completed!")
        logger.info(f"Results saved to: {self.output_dir}")

    def _save_checkpoint(self, year: int, phase: int = 5):
        """Save complete simulation state as checkpoint

        Args:
            year: Year number
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")
        if getattr(self, 'resource_ledger', None) is not None:
            self.resource_ledger.reconcile(year, self.ecosystem.agent_population, require_closed=True)

        # Serialize everything to JSON-compatible format
        checkpoint_data = {
            'year': year,
            'phase': phase,
            'ecosystem_data': self.ecosystem.to_dict(),
            'conference_system': self.conference_system.to_dict(),
            'citation_tracker': self.citation_tracker.to_dict(),
            'agent_tracker': self.agent_tracker.to_dict(),
            'funding_tracker': self.funding_tracker.to_dict(),
            'rag': self.rag.to_dict(),
            'paper_tracker': self.paper_tracker.to_dict(),
            'yearly_results': self.yearly_results,
            'industry_funding_system': self.industry_funding_system.to_dict(),
            'submitted_paper_ids': list(self.submitted_paper_ids),  # Convert set to list
            'initial_university_count': self.initial_university_count,
            'initial_industry_count': self.initial_industry_count,
        }
        if hasattr(self, 'cost_policy'):
            checkpoint_data['project_cost_policy'] = self.cost_policy
        if getattr(self, 'resource_ledger', None) is not None:
            checkpoint_data['resource_ledger'] = self.resource_ledger.to_dict()
            # Bind the checkpoint itself to its cell, independently of its path
            # or a neighboring manifest. Transport endpoints may change on resume.
            from utopia.runtime.provenance import _git
            checkpoint_data['cost_experiment_binding'] = {
                'args': dict(vars(self.args)),
                'git_commit': _git(['git', 'rev-parse', 'HEAD']),
            }

        # Save exploration experiment trackers if enabled
        if self.is_exploration_experiment:
            checkpoint_data['exploration_experiment'] = {
                'enabled': True,
                'embedding_tracker': self.embedding_tracker.to_dict() if self.embedding_tracker else None,
                'exploration_metrics': self.exploration_metrics.to_dict() if self.exploration_metrics else None,
                'paper_novelty_scores': {k: list(v) for k, v in self._paper_novelty_scores.items()},
                'citation_potential_multipliers': self._citation_potential_multipliers,
                'direction_records': {f"{aid}|{y}": rec for (aid, y), rec in
                                      self._direction_records.items()},
                'direction_fallback_stats': {str(k): v for k, v in
                                             self._direction_fallback_stats.items()},
            }

        # Save preferential attachment experiment trackers if enabled
        if self.experiment_name == 'preferential_attachment':
            checkpoint_data['preferential_attachment'] = {
                'enabled': True,
                'collaboration_tracker': self.collaboration_tracker.to_dict() if self.collaboration_tracker else None,
                'network_metrics': self.network_metrics.to_dict() if self.network_metrics else None,
                'collaboration_pairs': {str(k): v for k, v in self._collaboration_pairs.items()},
            }

        # Large-stage runs write gzipped checkpoints (~10x smaller; plan scale rule)
        if getattr(self.args, 'experiment_stage', None) in ('calibration', 'confirmatory', 'manyworld', 'mechanism', 'funding_cutoff', 'funding_cutoff_cost'):
            checkpoint_path += '.gz'
            write_json_gzip_atomic(checkpoint_path, checkpoint_data, create_parent=False)
        else:
            write_json_atomic(checkpoint_path, checkpoint_data, indent=2,
                              trailing_newline=False, streaming=True, create_parent=False)
        if getattr(self, 'resource_ledger', None) is not None:
            self.resource_ledger.write(self.output_dir)

        logger.info(f"Checkpoint saved for year {year}: {checkpoint_path}")

    def load_checkpoint(self, year: int):
        """Load complete simulation state from checkpoint

        Args:
            year: Year number to load from

        Returns:
            bool: True if checkpoint loaded successfully, False otherwise
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")

        from utopia.utils.data_utils import read_json
        if not os.path.exists(checkpoint_path):
            if not os.path.exists(checkpoint_path + '.gz'):
                logger.warning(f"Checkpoint not found: {checkpoint_path}(.gz)")
                return False
            checkpoint_path += '.gz'
        checkpoint_data = read_json(checkpoint_path)

        from utopia.funding.accounting import resolve_cost_policy, cost_policy_enabled, ResourceLedger
        current_policy = getattr(self, 'cost_policy', resolve_cost_policy(self.args, SIMULATION_CONFIG))
        saved_policy = checkpoint_data.get('project_cost_policy')
        if saved_policy is not None and saved_policy != current_policy:
            raise ValueError("Checkpoint cost policy differs from requested policy")
        if saved_policy is None and cost_policy_enabled(current_policy):
            raise ValueError("Cannot resume a legacy checkpoint under a new cost/audit policy")
        if cost_policy_enabled(current_policy) and checkpoint_data.get('phase', 5) != 5:
            raise ValueError("Cost accounting supports the native end-of-year checkpoints only")
        if current_policy['log_resource_ledger'] and 'resource_ledger' not in checkpoint_data:
            raise ValueError("Audit checkpoint is missing its resource ledger")
        if current_policy['log_resource_ledger']:
            binding = checkpoint_data.get('cost_experiment_binding')
            if not binding:
                raise ValueError("Audit checkpoint is missing its experiment binding")
            from utopia.analysis.project_cost import scientific_arguments
            if scientific_arguments(binding['args']) != scientific_arguments(vars(self.args)):
                raise ValueError("Checkpoint scientific arguments differ from requested experiment")
        self.cost_policy = current_policy
        self.resource_ledger = (ResourceLedger(checkpoint_data['resource_ledger'])
                                if current_policy['log_resource_ledger'] else None)

        self.current_year = checkpoint_data['year']

        # Reconstruct ecosystem with LLM from serialized data
        self.ecosystem = MultiAgentEcosystem.from_dict(
            checkpoint_data['ecosystem_data'],
            llm=self.llm
        )
        if self.resource_ledger is not None:
            self.resource_ledger.attach(self.ecosystem.agent_population)
            self.resource_ledger.reconcile(year, self.ecosystem.agent_population, require_closed=True)
            # A crash may leave newer exports than the last committed checkpoint.
            self.resource_ledger.write(self.output_dir)

        # Reconstruct conference system
        self.conference_system = ConferenceSystem.from_dict(checkpoint_data['conference_system'])

        # Reset conferences for the next year (checkpoint was saved before reset)
        self.conference_system.reset_all_for_new_year(year)

        # Reconstruct citation tracker
        self.citation_tracker = CitationTracker.from_dict(checkpoint_data.get('citation_tracker', {'citations': {}}))
        self.agent_tracker = AgentTracker.from_dict(checkpoint_data.get('agent_tracker', {'resources': {}}))
        self.funding_tracker = FundingTracker.from_dict(checkpoint_data.get('funding_tracker', {}))
        self.funding_tracker.funding_allocation_mode = self.funding_allocation_mode
        # Reconstruct paper tracker
        self.paper_tracker = PaperTracker.from_dict(
            checkpoint_data.get('paper_tracker', checkpoint_data.get('paper_tracker', {})))

        self.yearly_results = checkpoint_data['yearly_results']

        # Reconstruct industry funding system
        self.industry_funding_system = IndustryFundingSystem.from_dict(checkpoint_data['industry_funding_system'])

        # Convert submitted_paper_ids from list back to set
        self.submitted_paper_ids = set(checkpoint_data['submitted_paper_ids'])
        # N0 anchor (older checkpoints lack it: fall back to the current university count)
        self.initial_university_count = checkpoint_data.get('initial_university_count')
        self.initial_industry_count = checkpoint_data.get('initial_industry_count')
        if self.initial_university_count is None or self.initial_industry_count is None:
            self._record_initial_university_count()
            if self.funding_budget_mode == 'fixed':
                logger.warning("Checkpoint has no initial_university_count; using current count as N0")

        # Reconstruct RAG with document_status from checkpoint
        if 'rag' in checkpoint_data:
            rag_device = getattr(self.args, 'rag_device', None)
            rag_config = RAGConfig(use_langchain=self.use_langchain,
                cache_dir=os.path.join(getattr(self.args, 'data_cache_dir', 'data/cache'), 'rag'),
                **({'dataset_revision': self.args.dataset_revision} if getattr(self.args, 'dataset_revision', None) else {}),
                **({'device': rag_device} if rag_device else {}))
            self.rag = RAG.from_dict(checkpoint_data['rag'], config=rag_config, model=None,
                                     debug=self.debug, start_year=self.start_year)
            logger.info(f"  Loaded RAG with {len(self.rag.document_status)} document statuses")

        # Reconstruct exploration experiment trackers if enabled
        if 'exploration_experiment' in checkpoint_data and checkpoint_data['exploration_experiment'].get('enabled', False):
            self.experiment_name = 'exploration_vs_exploitation'
            exp_data = checkpoint_data['exploration_experiment']

            if exp_data.get('embedding_tracker'):
                from utopia.metrics.embedding_tracker import EmbeddingTracker
                self.embedding_tracker = EmbeddingTracker.from_dict(
                    exp_data['embedding_tracker'],
                    model_name=SIMULATION_CONFIG['exploration_experiment']['embedding_model'],
                    cache_dir=os.path.join(
                        getattr(self.args, 'data_cache_dir', os.path.join('data', 'cache')),
                        'embeddings'))
                logger.info(f"  Loaded EmbeddingTracker with {len(self.embedding_tracker.embeddings)} embeddings")

            if exp_data.get('exploration_metrics'):
                from utopia.metrics.exploration_metrics import ExplorationMetrics
                self.exploration_metrics = ExplorationMetrics.from_dict(exp_data['exploration_metrics'])
                logger.info(f"  Loaded ExplorationMetrics")

            # Restore paper novelty scores and citation potential metadata
            if exp_data.get('paper_novelty_scores'):
                self._paper_novelty_scores = {k: tuple(v) for k, v in exp_data['paper_novelty_scores'].items()}
                logger.info(f"  Loaded {len(self._paper_novelty_scores)} paper novelty scores")
            # 'breakthrough_papers' is the pre-rename key from old checkpoints
            self._citation_potential_multipliers = exp_data.get(
                'citation_potential_multipliers', exp_data.get('breakthrough_papers', {}))
            self._direction_records = {
                (k.rsplit('|', 1)[0], int(k.rsplit('|', 1)[1])): rec
                for k, rec in exp_data.get('direction_records', {}).items()}
            self._direction_fallback_stats = {
                int(k): v for k, v in exp_data.get('direction_fallback_stats', {}).items()}

            # CD calculator will be lazily initialized when needed
            self.cd_calculator = None
            self.keyword_extractor = None

        # Reconstruct preferential attachment experiment trackers if enabled
        if 'preferential_attachment' in checkpoint_data and checkpoint_data['preferential_attachment'].get('enabled', False):
            self.experiment_name = 'preferential_attachment'
            pa_data = checkpoint_data['preferential_attachment']

            if pa_data.get('collaboration_tracker'):
                from utopia.data.collaboration_tracker import CollaborationTracker
                self.collaboration_tracker = CollaborationTracker.from_dict(pa_data['collaboration_tracker'])
                logger.info(f"  Loaded CollaborationTracker with {self.collaboration_tracker.graph.number_of_edges()} edges")

            if pa_data.get('network_metrics'):
                from utopia.metrics.network_metrics import NetworkMetrics
                self.network_metrics = NetworkMetrics.from_dict(pa_data['network_metrics'])
                logger.info(f"  Loaded NetworkMetrics")

            if pa_data.get('collaboration_pairs'):
                self._collaboration_pairs = {int(k): v for k, v in pa_data['collaboration_pairs'].items()}
                logger.info(f"  Loaded {len(self._collaboration_pairs)} collaboration pair records")

        logger.info(f"Checkpoint loaded successfully from year {checkpoint_data['year']}")
        logger.info(f"  Loaded {len(self.ecosystem.agent_population)} agents")
        logger.info(f"  Loaded {len(self.submitted_paper_ids)} submitted paper IDs")
        return True

    def _generate_final_report(self):
        """Generate final simulation report using analyzer module"""
        report_file = os.path.join(self.output_dir, "simulation_report.json")

        # Use SimulationAnalyzer to generate comprehensive report
        analyzer = SimulationAnalyzer(self.output_dir)
        analyzer.load_from_simulation(self)
        analyzer.generate_report(output_file=report_file)



def main():
    project_setup()
    args = parse_arguments()
    from contextlib import nullcontext
    from utopia.models.request_audit import request_audit_scope
    from utopia.runtime.provenance import mark_failed_run_manifest
    from pathlib import Path
    # A completed invocation is a validated no-op and needs no new model client.
    existing_manifest = Path(args.docs_dir) / 'run_manifest.json'
    if not args.always_rerun and existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous.get('status') == 'complete' and previous.get('years_completed') == args.num_years:
            from utopia.analysis.release import summarize_run
            summarize_run(args.experiment_id, Path(args.docs_dir).resolve().parent.parent)
            logger.info('Simulation is already complete: %s', args.experiment_id)
            return
    audit_path = Path(args.docs_dir).resolve() / 'llm_request_audit.jsonl'
    if audit_path.exists() or audit_path.with_suffix('.summary.json').exists():
        audit_path = audit_path.with_name(f'llm_request_audit_resume_{time.time_ns()}.jsonl')
    scope = (request_audit_scope(audit_path, model_name=args.model,
                                revision=args.model_revision) if args.vllm_url else nullcontext())
    try:
        with scope as audit:
            run_from_args(args)
        if audit is not None:
            manifest = json.loads((Path(args.docs_dir) / 'run_manifest.json').read_text())
            paths = manifest.get('request_audit_paths', []) + [audit_path.name]
            write_run_manifest(args.docs_dir, status='complete', extra={
                'request_audit_paths': paths, 'requested_model_revision': audit.identity['model_revision']})
    except BaseException:
        mark_failed_run_manifest(args.docs_dir, 'simulation', traceback.format_exc())
        raise


def run_from_args(args):
    # Single run seed: Python/NumPy/torch state plus all derived per-request seeds
    set_seed(args.seed)
    logger.info(f"Run seed: {args.seed}")

    # Auto-select LLM based on model name
    t0 = time.time()
    if "gpt" in args.model:
        logger.info(f"Using OpenAI API for model: {args.model}")
        llm = GPT(model_name=args.model)

    elif "gemini" in args.model:
        logger.info(f"Using Gemini API for model: {args.model}")
        llm = Gemini(model_name=args.model)
    elif args.vllm_url:
        logger.info(f"Using external vLLM server at {args.vllm_url} for model: {args.model}")
        llm = VLLMServerModel(
            model_name=args.model,
            base_url=args.vllm_url,
            max_concurrent_requests=args.batch_size,
            run_seed=args.seed,
        )
    else:
        # Any non-OpenAI model uses vLLM (e.g., "Qwen/Qwen3-8B", "meta-llama/Llama-3-8B")
        logger.info(f"Using vLLM for model: {args.model}")
        llm = VLLMModel(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    logger.info(f"LLM initialized in {time.time() - t0:.1f}s")

    # Create and run simulation
    simulation = Simulation(
        llm=llm,
        verbose=args.verbose,
        num_years=args.num_years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode,
        funding_allocation_mode=args.funding_allocation_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain,
        max_retries=args.max_retries,
        weighted_funding_assignment=args.weighted_funding_assignment,
        experiment_name=args.experiment_name,
        collaboration_mode=args.collaboration_mode,
        collaboration_network_growth_mode=args.collaboration_network_growth_mode,
        start_year=args.start_year,
        args=args
    )

    simulation.run()


if __name__ == '__main__':
    main()
