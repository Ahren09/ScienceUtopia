#!/usr/bin/env python3
"""
Main simulation orchestration script for the Academic Society Ecosystem

This script runs the full multi-agent simulation with:
- Multiple UniversityResearchers and IndustryResearchers
- Multiple conferences targeting different topics
- Configurable research direction assignment (batch or individual mode)
- Yearly cycles with funding management
"""

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
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.conference import (
    ConferenceSystem,
    Conference,
    select_conferences_for_simulation
)
from utopia.agents.funding_agents import (
    FundingAgency,
    IndustryFundingSystem

)
from utopia.agents.research_direction import (
    assign_directions_individual,
    create_research_directions_batch,
    AVAILABLE_DIRECTIONS
)
from utopia.agents.researcher_agents import (
    UniversityResearcher,
    IndustryResearcher
)
from utopia.metrics.wandb_logger import WandbLogger
from utopia.analysis.analyze_all_results import SimulationAnalyzer
from utopia.arguments import parse_arguments
from utopia.config import SIMULATION_CONFIG
from utopia.data.paper_tracker import PaperTracker
from utopia.data.rag import RAG, RAGConfig
from utopia.data.tracker import CitationTracker, AgentTracker, FundingTracker
from utopia.metrics.tracker import calculate_gini_coefficient, calculate_conference_score_metrics
from utopia.models.models import GPT, BaseLLM, VLLMModel, VLLMServerModel, Gemini
from utopia.utils.generail_utils import project_setup


logger = logging.getLogger(__name__)


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
        self.funding_tracker = FundingTracker()
        self.funding_tracker.funding_allocation_mode = kwargs.get('funding_allocation_mode', 'consumption_based')
        self.industry_funding_mode = kwargs.get('industry_funding_mode', 'performance')
        self.always_rerun = kwargs.get('always_rerun', False)
        self.use_langchain = kwargs.get('use_langchain', False)
        self.max_retries = kwargs.get('max_retries', 5)
        self.weighted_funding_assignment = kwargs.get('weighted_funding_assignment', False)
        self.funding_allocation_mode = kwargs.get('funding_allocation_mode', 'consumption_based')
        self.experiment_name = kwargs.get('experiment_name', 'default')
        
        assert self.experiment_name in ['default', 'exploration_vs_exploitation'], f"Invalid experiment name: {self.experiment_name}"

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
        rag_config = RAGConfig(use_langchain=self.use_langchain, **({'device': rag_device} if rag_device else {}))
        self.start_year = kwargs.get('start_year', 2016)
        self.rag: RAG = RAG(config=rag_config, debug=self.debug, model=None, start_year=self.start_year)
        logger.info(f"RAG initialized in {time.time() - t0:.1f}s (device={rag_config.device})")

        self.wandb_logger = WandbLogger(SIMULATION_CONFIG, self.args)

        # NEW: Initialize exploration experiment trackers (opt-in via flag)
        if self.experiment_name == 'exploration_vs_exploitation':
            logger.info("Initializing Exploration vs Exploitation experiment...")
            # Will be lazily imported to avoid dependencies when experiment is disabled
            self.embedding_tracker = None  # Will initialize after first use
            self.cd_calculator = None
            self.exploration_metrics = None
            self.keyword_extractor = None
        else:
            self.embedding_tracker = None
            self.cd_calculator = None
            self.exploration_metrics = None
            self.keyword_extractor = None

    def initialize_agents(self):
        """Initialize the agent population

        Creates multiple researchers mapped to universities and companies.
        """
        logger.info("Initializing agent population...")

        rich_universities = [
            "MIT", "Stanford", "Berkeley", "CMU", "GeorgiaTech",
            "Princeton", "Cornell", "UCLA", "UIUC", "UW",
        ]

        # Create multiple university researchers mapped to same universities
        normal_universities = [
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
                expertise = [AVAILABLE_DIRECTIONS[0]] if self.debug else random.sample(AVAILABLE_DIRECTIONS, 3)
                # researcher_name = f"university_researcher_{counter}"
                researcher_name = f"{university}_researcher_{i}"

                funding_level = 160 if university in rich_universities else 80

                # Assign exploration strategy if experiment is enabled
                if self.experiment_name == 'exploration_vs_exploitation':
                    # 50% explorer, 50% exploiter
                    exploration_strategy = "explorer" if counter % 2 == 0 else "exploiter"
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

        rich_companies = ["Google", "Meta", "Microsoft", "Amazon", "Apple",
                          "OpenAI", "DeepMind", "Anthropic", "Waymo", "Netflix",
                          ]
        # Create multiple industry researchers mapped to same companies
        normal_companies = [
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

            funding_level = 320 if company in rich_companies else 160

            for i in range(num_researchers):
                expertise = random.sample(AVAILABLE_DIRECTIONS, 3) if self.debug else [AVAILABLE_DIRECTIONS[0]]
                researcher_name = f"{company}_researcher_{i}"
                researcher = IndustryResearcher(
                    researcher_name=researcher_name,
                    company_name=company,
                    funding_level=funding_level,
                    expertise=expertise,
                    llm=self.llm,
                    funding_mode=self.industry_funding_mode
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

        # Add funding agents
        # NSF: Foundational scientific advancement (patient, lower thresholds)
        # DARPA: Application-driven projects (demanding, higher thresholds)

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

        logger.info(f"Total agents initialized: {len(self.ecosystem.agent_population)}")

    def calculate_average_funding_level(self):
        """Calculate the average funding level of all agents"""
        funding_levels = [agent.resources for agent_id, agent in self.ecosystem.agent_population.items() if
                          getattr(agent, "resources", None) is not None]
        return np.mean(funding_levels)

    def _initialize_experiment_trackers(self):
        """Lazy initialization of exploration experiment trackers"""
        if self.experiment_name != 'exploration_vs_exploitation':
            return

        if self.embedding_tracker is None:
            from utopia.metrics.embedding_tracker import EmbeddingTracker
            exp_config = SIMULATION_CONFIG['exploration_experiment']
            self.embedding_tracker = EmbeddingTracker(model_name=exp_config['embedding_model'])
            logger.info("Initialized EmbeddingTracker")

        if self.cd_calculator is None:
            from utopia.metrics.cd_index import CDIndexCalculator
            self.cd_calculator = CDIndexCalculator(self.citation_tracker, self.paper_tracker)
            logger.info("Initialized CDIndexCalculator")

        if self.exploration_metrics is None:
            from utopia.metrics.exploration_metrics import ExplorationMetrics
            self.exploration_metrics = ExplorationMetrics()
            logger.info("Initialized ExplorationMetrics")

        if self.keyword_extractor is None:
            from utopia.metrics.keyword_extractor import KeywordExtractor
            self.keyword_extractor = KeywordExtractor()
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
        """Record keywords and embeddings for experiment tracking

        Args:
            paper_dict: Paper dictionary with 'id', 'abstract', etc.
            author_id: Author ID
            year: Current year
        """
        if self.experiment_name != 'exploration_vs_exploitation':
            return

        # Lazy initialize trackers on first use
        self._initialize_experiment_trackers()

        exp_config = SIMULATION_CONFIG['exploration_experiment']

        # Extract keywords
        if self.keyword_extractor is not None:
            keywords = self.keyword_extractor.extract_keywords_llm(
                abstract=paper_dict['abstract'],
                llm=self.llm,
                n_keywords=exp_config['num_keywords']
            )
            paper_dict['keywords'] = keywords

        # Add embedding
        if self.embedding_tracker is not None:
            self.embedding_tracker.add_paper_embedding(
                paper_id=paper_dict['id'],
                abstract=paper_dict['abstract'],
                author_id=author_id,
                year=year
            )
            logger.debug(f"Added embedding for {paper_dict['id']}")

    def _collect_exploration_metrics(self, year: int):
        """Collect exploration vs exploitation metrics at end of year

        Args:
            year: Current year number
        """
        if self.experiment_name != 'exploration_vs_exploitation':
            return

        self._initialize_experiment_trackers()

        exp_config = SIMULATION_CONFIG['exploration_experiment']

        # Compute topic distances for papers submitted this year
        paper_ids_this_year = [p.id for p in self.paper_tracker.papers_by_id.values()
                              if p.year == year]

        for paper_id in paper_ids_this_year:
            paper = self.paper_tracker.papers_by_id[paper_id]
            author_id = paper.author_id

            # Calculate distance to author's previous work
            distance = self.embedding_tracker.compute_distance_to_previous(
                paper_id=paper_id,
                author_id=author_id
            )

            self.exploration_metrics.record_topic_distance(
                author_id=author_id,
                year=year,
                distance=distance
            )

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

        # Aggregate metrics by strategy
        for strategy in exp_config['strategies']:
            agents = [a for a in self.ecosystem.get_agents_by_type('university')
                     if hasattr(a, 'exploration_strategy') and a.exploration_strategy == strategy]

            if agents:
                distances = []
                for a in agents:
                    avg_dist = self.exploration_metrics.get_author_avg_distance(a.id, year)
                    if avg_dist is not None:
                        distances.append(avg_dist)

                if distances:
                    avg_distance = np.mean(distances)

                    self.exploration_metrics.record_strategy_metric(
                        strategy=strategy,
                        year=year,
                        metric='avg_topic_distance',
                        value=avg_distance
                    )

        logger.info(f"Year {year}: Collected exploration metrics for {len(paper_ids_this_year)} papers")

    def _setup_year(self, year: int) -> Dict:
        """Setup year: log start, reset funding tracker, reset conferences

        Args:
            year: Current year number

        Returns:
            year_results: Dict initialized with year number
        """
        
        logger.info(f"*** YEAR {year} ***")
        

        # Reset all fundings to 0 at start of cycle
        total_funding = self.funding_tracker.start_cycle(year, self.ecosystem.agent_population)
        logger.info(f"Total funding at start of year {year}: {total_funding:.2f}")

        # Reset conferences for the new year of simulation
        self.conference_system.reset_all_for_new_year(year)

        # Initialize year results dict
        return {'year': year}

    def _run_phase_0_resubmissions(self, year: int, year_results: Dict) -> List[Dict]:
        """Phase 0: Process resubmissions from previous year (year 2+)

        Args:
            year: Current year number
            year_results: Year results dict to update

        Returns:
            resubmissions_list: List of resubmitted papers
        """
        resubmissions_list = []
        resubmissions_paper_ids = set()

        if year <= 1:
            return resubmissions_list

        logger.info(f"Year {year} Preparation: Processing resubmissions from year {year - 1}...")

        supports_batching = hasattr(self.llm, 'generate_batch')

        if supports_batching:
            # ==================== BATCHED PATH (vLLM) ====================
            # Step 1: Collect prompts and metadata
            batch_prompts = []
            batch_metadata = []
            resubmission_response_format = None

            for author_id, papers in self.paper_tracker.papers_by_author.items():
                agent = self.ecosystem.get_agent_by_id(author_id)
                if not agent:
                    continue

                rejected_papers = {paper.id: paper.to_dict() for paper in papers if paper.status == 'reject'}
                if agent.resources < 5 or not rejected_papers:
                    continue

                prompt, response_format = agent.get_resubmission_prompt(rejected_papers, self.conference_system.conferences)
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

                batch_results = self.llm.generate_batch(batch_prompts, response_format=resubmission_response_format, desc=f"{len(batch_prompts)} resubmission decisions (attempt {attempt+1}/{MAX_RESUBMISSION_RETRIES})")

                agents_needing_retry = []

                # Step 3: Process results
                candidate_conference_ids = set([c.conference_id for c in self.conference_system.conferences])
                for i, (result, message_history) in enumerate(batch_results):
                    meta = batch_metadata[i]
                    agent = meta['agent']
                    rejected_papers = meta['rejected_papers']

                    # Parse failure -> add to retry
                    if result is None or 'resubmitted_papers' not in result:
                        logger.warning(f"Failed to parse resubmission decision for agent {agent.id}, adding to retry")
                        agents_needing_retry.append({'agent': agent, 'rejected_papers': rejected_papers})
                        continue

                    resubmissions_of_author = result['resubmitted_papers']
                    has_invalid = False

                    for metadata in resubmissions_of_author:
                        arxiv_id = metadata.get('arxiv_id', '')
                        conf = metadata.get('conference', '')
                        if arxiv_id not in rejected_papers or conf not in candidate_conference_ids:
                            logger.warning(f"Invalid resubmission entry for agent {agent.id}: {metadata}, skipping")
                            has_invalid = True
                            continue

                        if arxiv_id not in resubmissions_paper_ids:
                            paper_dict = rejected_papers[arxiv_id].copy()
                            paper_dict['conference'] = conf
                            paper_dict['year'] = year
                            paper_dict['type'] = 'resubmission'
                            review_history_backup = paper_dict.get('review_history', [])
                            del paper_dict['status']
                            if 'review_history' in paper_dict:
                                del paper_dict['review_history']
                            paper_dict['status'] = 'pending'
                            paper_dict['review_history'] = review_history_backup

                            resubmissions_list.append(paper_dict)
                            self.paper_tracker.add_or_update_paper(paper_dict)
                            resubmission_cost = SIMULATION_CONFIG['conference']['resubmission_cost']
                            agent.update_resources(-resubmission_cost)
                            self.funding_tracker.record_funding_consumption(year, resubmission_cost, agent.get_type())
                            resubmissions_paper_ids.add(arxiv_id)
                        else:
                            logger.error(f"[Resubmission] Paper {arxiv_id} already resubmitted for year {year}")

                    # If agent had invalid entries, add to retry with remaining papers
                    if has_invalid:
                        remaining_rejected = {pid: p for pid, p in rejected_papers.items()
                                              if pid not in resubmissions_paper_ids}
                        if remaining_rejected:
                            agents_needing_retry.append({'agent': agent, 'rejected_papers': remaining_rejected})

                # Rebuild batch for next attempt
                if not agents_needing_retry:
                    break

                batch_prompts = []
                batch_metadata = []

                for retry_info in agents_needing_retry:
                    agent = retry_info['agent']
                    remaining_rejected = retry_info['rejected_papers']

                    if agent.resources < 5 or not remaining_rejected:
                        continue

                    prompt, response_format = agent.get_resubmission_prompt(remaining_rejected, self.conference_system.conferences)
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

        else:
            # ==================== SEQUENTIAL PATH (GPT/OpenAI) ====================
            for author_id, papers in self.paper_tracker.papers_by_author.items():
                agent = self.ecosystem.get_agent_by_id(author_id)

                if not agent:
                    continue

                rejected_papers = {paper.id: paper.to_dict() for paper in papers if paper.status == 'reject'}
                if not rejected_papers:
                    continue

                resubmissions_of_author = agent.decide_resubmission(rejected_papers, self.conference_system.conferences)

                for metadata in resubmissions_of_author:
                    if metadata['arxiv_id'] not in resubmissions_paper_ids:
                        paper_dict = rejected_papers[metadata['arxiv_id']].copy()
                        paper_dict['conference'] = metadata['conference']
                        paper_dict['year'] = year
                        paper_dict['type'] = 'resubmission'
                        review_history_backup = paper_dict.get('review_history', [])
                        del paper_dict['status']
                        if 'review_history' in paper_dict:
                            del paper_dict['review_history']
                        paper_dict['status'] = 'pending'
                        paper_dict['review_history'] = review_history_backup

                        resubmissions_list.append(paper_dict)
                        self.paper_tracker.add_or_update_paper(paper_dict)
                        resubmission_cost = SIMULATION_CONFIG['conference']['resubmission_cost']
                        agent.update_resources(-resubmission_cost)
                        self.funding_tracker.record_funding_consumption(year, resubmission_cost, agent.get_type())
                        resubmissions_paper_ids.add(metadata['arxiv_id'])
                    else:
                        logger.error(f"[Resubmission] Paper {metadata['arxiv_id']} already resubmitted for year {year}")

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

        if supports_batching and authors_with_directions_pending:
            
            directions = create_research_directions_batch(
                agents=authors_with_directions_pending,
                year=year,
                paper_tracker=self.paper_tracker,
                llm=self.llm,
                average_funding_level=average_funding_level
            )
            # Update agent project timelines
            for agent in authors_with_directions_pending:
                direction = directions[agent.id]
                agent.project_start_year = year
                agent.project_end_year = (year - 1) + direction['direction'].years
                agent.newest_direction = direction
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

    def _run_phase_2_submit_papers(self, year: int, directions: Dict, active_authors: List,
                                    resubmissions_list: List[Dict], year_results: Dict) -> List[Dict]:
        """Phase 2: Generate and submit papers

        Args:
            year: Current year number
            directions: Research directions for each agent
            active_authors: List of authors ready to submit
            resubmissions_list: List of papers being resubmitted
            year_results: Year results dict to update

        Returns:
            submissions_list: List of newly submitted papers (not resubmissions)
        """
        logger.info(f"Year {year} Phase 2: Submitting papers...")

        # ---- Load RAG index if needed (before intention generation) ----
        if year == 1:
            self.rag.load_documents(num_years=self.num_years)
            self.rag.build_knowledge_index()
        else:
            # Get submitted papers from paper archive
            accepted_papers, rejected_papers = self.paper_tracker.get_papers_before_year(year=year)

            if self.rag.documents == []:
                # Resuming from checkpoint - preserve document_status
                saved_document_status = self.rag.document_status.copy()
                self.rag.load_documents(num_years=self.num_years)
                self.rag.build_knowledge_index()
                # Restore document_status from checkpoint
                self.rag.document_status = saved_document_status
                logger.info(f"Resumed from checkpoint with {len(saved_document_status)} document statuses preserved")

        # Set the current simulation year for year-based data partitioning
        self.rag.set_current_year(year)

        # ---- Generate round intentions (what the agent want to write in their papers) for all active authors ----
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
                intention_prompts, max_tokens=512,
                response_format=intention_response_format,
                desc=f"{len(intention_prompts)} round intentions"
            )
            for agent, (result_json, _) in zip(active_authors, intention_results):
                intention = self._parse_intention(result_json, directions[agent.id])
                round_intentions[agent.id] = intention
        else:
            for agent, prompt in zip(active_authors, intention_prompts):
                result_json, _ = self.llm.generate(prompt=prompt, response_format=intention_response_format)
                intention = self._parse_intention(result_json, directions[agent.id])
                round_intentions[agent.id] = intention

        logger.info(f"Generated {len(round_intentions)} round intentions")
        year_results['round_intentions'] = round_intentions

        # ---- Build RAG queries from intentions + keywords ----
        paper_submission_prompts: List[str] = []
        for agent in active_authors:
            keywords = ", ".join(directions[agent.id]['direction'].keywords)
            paper_submission_prompts.append(
                f"{round_intentions[agent.id]}\nKeywords: {keywords}"
            )

        # Batch retrieve relevant documents for all research ideas
        # Use SUBMISSION index to get papers available for submission
        logger.info(f"Retrieving {len(paper_submission_prompts)} paper submissions ...")
        retrieval_results = self.rag.batch_retrieve(
            paper_submission_prompts,
            retrieval_type="submission"  # Only retrieve papers not yet submitted
        )

        topk_indices = retrieval_results['topk_indices']

        assert topk_indices.shape[0] == len(active_authors)

        submitted_papers = {}
        submissions_list = []
        potential_cited_papers_list = []

        message_history_dict = {}

        submitted_paper_indices_current_year = set()

        # Paper submission (supports_batching already set above)
        paper_submission_response_format = None
        if supports_batching:
            # ==================== BATCHED PATH (vLLM) ====================
            # Step 1: Collect ALL prompts and metadata (NO LLM calls yet)
            batch_prompts = []
            batch_metadata = []  # Store (agent, candidate_papers, response_format) per prompt

            for agent_idx, agent in enumerate(active_authors):
                if agent.project_end_year > year:
                    if self.verbose:
                        logger.info(
                            f"[Submission] Agent {agent.id} is not ready to submit papers yet. Maturity year: {agent.project_end_year}, Current year: {year}")
                    continue

                candidate_paper_indices = topk_indices[agent_idx]
                candidate_papers = [self.rag.documents[idx] for idx in candidate_paper_indices if
                                    self.rag.document_status.loc[idx, 'status'] == 'unsubmitted']

                if len(candidate_papers) == 0:
                    logger.warning(f"No candidate papers found for agent {agent.id}, skipping")
                    continue

                prompt, response_format = agent.get_paper_submission_prompt(
                    candidate_papers=candidate_papers,
                    year=year,
                    direction=directions[agent.id],
                    conferences=self.conference_system.conferences,
                    paper_tracker=self.paper_tracker
                )

                batch_prompts.append(prompt)
                batch_metadata.append({
                    'agent': agent,
                    'agent_idx': agent_idx,
                    'candidate_papers': candidate_papers,
                })
                if paper_submission_response_format is None:
                    paper_submission_response_format = response_format

            # Step 2: Batch call + process with retries (reuses same logic each iteration)
            MAX_SUBMISSION_RETRIES = 4  # 1 initial + 3 retries
            agents_needing_fallback = []

            for attempt in range(MAX_SUBMISSION_RETRIES):
                if not batch_prompts:
                    break

                batch_results = self.llm.generate_batch(batch_prompts, temperature=0.7, response_format=paper_submission_response_format, desc=f"{len(batch_prompts)} paper submissions (attempt {attempt+1}/{MAX_SUBMISSION_RETRIES})")

                agents_needing_fallback = []

                # Step 3: Process results
                for i, (submission_metadata, message_history) in enumerate(batch_results):
                    meta = batch_metadata[i]
                    agent = meta['agent']
                    candidate_papers = meta['candidate_papers']

                    # Handle failed JSON parse from batch
                    if submission_metadata is None:
                        logger.warning(f"Failed to parse response for {agent.id}, adding to fallback")
                        agents_needing_fallback.append({
                            'agent': agent,
                            'candidate_papers': candidate_papers,
                            'conference': None,
                            'message_history': message_history
                        })
                        continue

                    # Validate response
                    try:
                        arxiv_id = submission_metadata['arxiv_id']
                        assert "reason" in submission_metadata, f"Reason is not in the submission metadata: {submission_metadata}"
                        assert arxiv_id in [p.metadata['id'] for p in candidate_papers] or arxiv_id == "DO_NOT_SUBMIT"
                        assert submission_metadata['conference'] in self.conference_system.conference_map or \
                               submission_metadata['conference'] == "DO_NOT_SUBMIT", f"Generated invalid conference name: {submission_metadata['conference']}"
                    except (KeyError, AssertionError) as e:
                        traceback.print_exc()
                        logger.warning(f"Invalid submission from {agent.id}: {e}")
                        agents_needing_fallback.append({
                            'agent': agent,
                            'candidate_papers': candidate_papers,
                            'conference': None,
                            'message_history': message_history
                        })
                        continue

                    if arxiv_id == "DO_NOT_SUBMIT":
                        logger.info(f"Agent {agent.id} did not submit any paper due to funding issues")
                        continue

                    # Skip if paper was already submitted by another agent in this batch
                    if arxiv_id in self.paper_tracker.papers_by_id:
                        logger.warning(f"Paper {arxiv_id} already taken, {agent.id} needs fallback selection")
                        agents_needing_fallback.append({
                            'agent': agent,
                            'candidate_papers': candidate_papers,
                            'conference': submission_metadata['conference'],
                            'message_history': message_history
                        })
                        continue

                    logger.info(
                        f"{agent.id} - {submission_metadata['conference']} - submitted \'{self.rag.id2docs[arxiv_id].metadata['title']}\'")

                    # Process successful submission
                    paper_doc = self.rag.id2docs[arxiv_id]

                    assert agent.project_start_year <= year, f"Agent {agent.id} MUST started the project before the current year: {agent.project_start_year} <= {year}"
                    assert agent.project_end_year == year, f"Agent {agent.id} MUST completed the project in the current year"

                    paper_dict = {
                        'id': arxiv_id,
                        'author_id': agent.id,
                        'author_type': agent.get_type(),
                        'title': paper_doc.metadata['title'],
                        'abstract': paper_doc.page_content,
                        'topics': paper_doc.metadata.get('categories', '').split(),
                        'conference': submission_metadata['conference'],
                        'type': 'submission',
                        'year': year,
                        'project_start_year': agent.project_start_year,
                        'project_end_year': agent.project_end_year,
                        'maturity': directions[agent.id]['direction'].years,
                        'status': 'pending'
                    }

                    message_history_dict[agent.id] = message_history

                    submissions_list.append(paper_dict)
                    self.paper_tracker.add_or_update_paper(paper_dict)
                    self._record_paper_metadata(paper_dict, agent.id, year)
                    self.rag.document_status.loc[self.rag.document_status['id'] == arxiv_id, 'status'] = 'submitted'

                # Rebuild batch_prompts/batch_metadata from fallback agents for next attempt
                if not agents_needing_fallback:
                    break

                batch_prompts = []
                batch_metadata = []

                for fallback_info in agents_needing_fallback:
                    agent = fallback_info['agent']
                    available_papers = [p for p in fallback_info['candidate_papers']
                                      if p.metadata['id'] not in self.paper_tracker.papers_by_id]

                    if not available_papers:
                        continue

                    prompt, _ = agent.get_paper_submission_prompt(
                        candidate_papers=available_papers,
                        year=year,
                        direction=directions[agent.id],
                        conferences=self.conference_system.conferences,
                        paper_tracker=self.paper_tracker
                    )

                    batch_prompts.append(prompt)
                    batch_metadata.append({
                        'agent': agent,
                        'candidate_papers': available_papers,
                    })

            # Step 4: Final random assignment for agents still remaining after all retries
            if agents_needing_fallback:
                logger.info(f"After retries, {len(agents_needing_fallback)} agents need random assignment")

                for fallback_info in agents_needing_fallback:
                    agent = fallback_info['agent']
                    available_papers = [p for p in fallback_info['candidate_papers']
                                      if p.metadata['id'] not in self.paper_tracker.papers_by_id]

                    if not available_papers:
                        logger.warning(f"{agent.id} final fallback failed - no available papers remaining")
                        continue

                    selected_paper = random.choice(available_papers)
                    arxiv_id = selected_paper.metadata['id']
                    conference = fallback_info['conference'] or random.choice(list(self.conference_system.conference_map.keys()))

                    logger.info(f"{agent.id} random fallback - {conference} - submitted '{selected_paper.metadata['title']}'")

                    paper_doc = self.rag.id2docs[arxiv_id]

                    assert agent.project_start_year <= year, f"Agent {agent.id} MUST started the project before the current year: {agent.project_start_year} <= {year}"
                    assert agent.project_end_year == year, f"Agent {agent.id} MUST completed the project in the current year"

                    paper_dict = {
                        'id': arxiv_id,
                        'author_id': agent.id,
                        'author_type': agent.get_type(),
                        'title': paper_doc.metadata['title'],
                        'abstract': paper_doc.page_content,
                        'topics': paper_doc.metadata.get('categories', '').split(),
                        'conference': conference,
                        'type': 'submission',
                        'year': year,
                        'project_start_year': agent.project_start_year,
                        'project_end_year': agent.project_end_year,
                        'maturity': directions[agent.id]['direction'].years,
                        'status': 'pending'
                    }

                    message_history_dict[agent.id] = fallback_info['message_history']
                    submissions_list.append(paper_dict)
                    self.paper_tracker.add_or_update_paper(paper_dict)
                    self._record_paper_metadata(paper_dict, agent.id, year)
                    self.rag.document_status.loc[self.rag.document_status['id'] == arxiv_id, 'status'] = 'submitted'

        else:
            # ==================== [START] SEQUENTIAL PATH (GPT/Gemini) ====================
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
                        arxiv_id = submission_metadata['arxiv_id']
                        assert "reason" in submission_metadata, f"Reason is not in the submission metadata: {submission_metadata}"
                        assert arxiv_id in [p.metadata['id'] for p in candidate_papers] or arxiv_id == "DO_NOT_SUBMIT"
                        assert submission_metadata[
                                   'conference'] in self.conference_system.conference_map or submission_metadata['conference'] == "DO_NOT_SUBMIT", f"Generated invalid conference name: {submission_metadata['conference']}"

                        if arxiv_id == "DO_NOT_SUBMIT":
                            break

                        else:
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


                else:
                    paper_doc = self.rag.id2docs[arxiv_id]

                    assert agent.project_start_year <= year, f"Agent {agent.id} MUST started the project before the current year: {agent.project_start_year} <= {year}"
                    assert agent.project_end_year == year, f"Agent {agent.id} MUST completed the project in the current year"

                    paper_dict = {
                        'id': arxiv_id,
                        'author_id': agent.id,
                        'author_type': agent.get_type(),
                        'title': paper_doc.metadata['title'],
                        'abstract': paper_doc.page_content,
                        'topics': paper_doc.metadata.get('categories', '').split(),
                        'conference': submission_metadata['conference'],
                        'type': 'submission',
                        'year': year,  # Current year for submission
                        'project_start_year': agent.project_start_year,  # The year when the agent started the project.
                        'project_end_year': agent.project_end_year,
                        # The year when the agent's latest research paper will be ready for submission
                        'maturity': directions[agent.id]['direction'].years,
                        'status': 'pending'
                    }

                    message_history_dict[agent.id] = message_history

                submissions_list.append(paper_dict)
                self.paper_tracker.add_or_update_paper(paper_dict)
                # Record metadata for exploration experiment (if enabled)
                self._record_paper_metadata(paper_dict, agent.id, year)

                self.rag.document_status.loc[self.rag.document_status['id'] == arxiv_id, 'status'] = 'submitted'
            # ==================== [END] SEQUENTIAL PATH (GPT/Gemini) ====================
        del retrieval_results

        if year >= 2:

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

            supports_batching = hasattr(self.llm, 'generate_batch')

            if supports_batching:
                # ==================== BATCHED PATH (vLLM) ====================
                MAX_CITATION_RETRIES = 4

                # --- Batch 1: Regular Citations (accepted papers only) ---
                batch_prompts = []
                batch_metadata = []
                citation_response_format = None

                for i in range(len(submissions_list)):
                    submitted_paper = submissions_list[i]
                    agent = self.ecosystem.get_agent_by_id(submitted_paper['author_id'])
                    candidate_papers = [self.rag.documents[idx] for idx in topk_indices_citation_accepted[i][
                                                                           :SIMULATION_CONFIG['citation'][
                                                                               'num_candidate_papers']]]
                    candidate_papers = [
                        {'id': paper.metadata['id'], 'title': paper.metadata['title'], 'abstract': paper.page_content}
                        for paper in candidate_papers]

                    assert set([cp['id'] for cp in candidate_papers]) <= accepted_papers_arxiv_ids, \
                        f"Submitted paper {submitted_paper['id']} is trying to cite non-accepted papers"

                    prompt, response_format = agent.get_potential_cited_papers_prompt(
                        paper=submitted_paper, direction=directions[agent.id],
                        candidate_papers=candidate_papers, max_papers=30
                    )
                    batch_prompts.append(prompt)
                    batch_metadata.append({
                        'paper_idx': i, 'paper': submitted_paper, 'agent': agent,
                        'candidate_papers': candidate_papers,
                        'candidate_paper_ids': set(cp['id'] for cp in candidate_papers),
                    })
                    if citation_response_format is None:
                        citation_response_format = response_format

                regular_citations = {}  # paper_idx -> citation list

                for attempt in range(MAX_CITATION_RETRIES):
                    if not batch_prompts:
                        break

                    batch_results = self.llm.generate_batch(
                        batch_prompts, response_format=citation_response_format,
                        desc=f"{len(batch_prompts)} regular citations (attempt {attempt+1}/{MAX_CITATION_RETRIES})"
                    )

                    items_needing_retry = []
                    for j, (result, _) in enumerate(batch_results):
                        meta = batch_metadata[j]
                        idx = meta['paper_idx']

                        if result is None or result.get('citations') is None:
                            items_needing_retry.append(meta)
                            continue

                        citations = result['citations']
                        if not set(citations) <= meta['candidate_paper_ids']:
                            logger.warning(f"Invalid citations for paper {meta['paper']['id']}, retrying")
                            items_needing_retry.append(meta)
                            continue

                        regular_citations[idx] = citations

                    if not items_needing_retry:
                        break

                    # Rebuild batch for retry
                    batch_prompts = []
                    batch_metadata = []
                    for meta in items_needing_retry:
                        prompt, _ = meta['agent'].get_potential_cited_papers_prompt(
                            paper=meta['paper'], direction=directions[meta['agent'].id],
                            candidate_papers=meta['candidate_papers'], max_papers=30
                        )
                        batch_prompts.append(prompt)
                        batch_metadata.append(meta)

                # Papers that still failed get empty citations
                for i in range(len(submissions_list)):
                    if i not in regular_citations:
                        logger.warning(f"Paper {submissions_list[i]['id']} failed all citation retries, using empty citations")
                        regular_citations[i] = []

                # --- Batch 2: Extended Citations (papers with too few regular citations) ---
                extended_citations = {}  # paper_idx -> citation list
                batch_prompts = []
                batch_metadata = []

                for i in range(len(submissions_list)):
                    if len(regular_citations[i]) <= SIMULATION_CONFIG['citation']['expected_citation_per_round']:
                        submitted_paper = submissions_list[i]
                        agent = self.ecosystem.get_agent_by_id(submitted_paper['author_id'])
                        candidate_papers = [self.rag.documents[idx] for idx in topk_indices_citation_both[i][
                                                                               :SIMULATION_CONFIG['citation'][
                                                                                   'num_candidate_papers']]]
                        candidate_papers = [
                            {'id': paper.metadata['id'], 'title': paper.metadata['title'], 'abstract': paper.page_content}
                            for paper in candidate_papers]

                        prompt, response_format = agent.get_potential_cited_papers_prompt(
                            paper=submitted_paper, direction=directions[agent.id],
                            candidate_papers=candidate_papers, max_papers=30
                        )
                        batch_prompts.append(prompt)
                        batch_metadata.append({
                            'paper_idx': i, 'paper': submitted_paper, 'agent': agent,
                            'candidate_papers': candidate_papers,
                            'candidate_paper_ids': set(cp['id'] for cp in candidate_papers),
                        })
                        if citation_response_format is None:
                            citation_response_format = response_format

                for attempt in range(MAX_CITATION_RETRIES):
                    if not batch_prompts:
                        break

                    batch_results = self.llm.generate_batch(
                        batch_prompts, response_format=citation_response_format,
                        desc=f"{len(batch_prompts)} extended citations (attempt {attempt+1}/{MAX_CITATION_RETRIES})"
                    )

                    items_needing_retry = []
                    for j, (result, _) in enumerate(batch_results):
                        meta = batch_metadata[j]
                        idx = meta['paper_idx']

                        if result is None or result.get('citations') is None:
                            items_needing_retry.append(meta)
                            continue

                        citations = result['citations']
                        if not set(citations) <= meta['candidate_paper_ids']:
                            logger.warning(f"Invalid extended citations for paper {meta['paper']['id']}, retrying")
                            items_needing_retry.append(meta)
                            continue

                        extended_citations[idx] = citations

                    if not items_needing_retry:
                        break

                    batch_prompts = []
                    batch_metadata = []
                    for meta in items_needing_retry:
                        prompt, _ = meta['agent'].get_potential_cited_papers_prompt(
                            paper=meta['paper'], direction=directions[meta['agent'].id],
                            candidate_papers=meta['candidate_papers'], max_papers=30
                        )
                        batch_prompts.append(prompt)
                        batch_metadata.append(meta)

                # --- Batch 3: Self-Citations ---
                self_citations_dict = {}  # paper_idx -> self-citation list
                batch_prompts = []
                batch_metadata = []
                self_citation_response_format = None

                for i in range(len(submissions_list)):
                    submitted_paper = submissions_list[i]
                    agent = self.ecosystem.get_agent_by_id(submitted_paper['author_id'])
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

                for attempt in range(MAX_CITATION_RETRIES):
                    if not batch_prompts:
                        break

                    batch_results = self.llm.generate_batch(
                        batch_prompts, response_format=self_citation_response_format,
                        desc=f"{len(batch_prompts)} self-citations (attempt {attempt+1}/{MAX_CITATION_RETRIES})"
                    )

                    items_needing_retry = []
                    for j, (result, _) in enumerate(batch_results):
                        meta = batch_metadata[j]
                        idx = meta['paper_idx']

                        if result is None or result.get('self_citations') is None:
                            items_needing_retry.append(meta)
                            continue

                        self_citations_dict[idx] = result['self_citations']

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

                # --- Combine and record citations ---
                for i in range(len(submissions_list)):
                    submitted_paper = submissions_list[i]
                    # Use extended citations if available, else regular
                    citations_of_paper = extended_citations.get(i, regular_citations.get(i, []))
                    # Add self-citations (deduplicated)
                    if i in self_citations_dict:
                        citations_of_paper = list(set(citations_of_paper + self_citations_dict[i]))
                    self.citation_tracker.add_citations(submitted_paper['id'], citations_of_paper, year)

            else:
                # ==================== SEQUENTIAL PATH (GPT/OpenAI) ====================
                for i in range(len(submissions_list)):
                    submitted_paper = submissions_list[i]
                    agent = self.ecosystem.get_agent_by_id(submitted_paper['author_id'])
                    candidate_papers = [self.rag.documents[idx] for idx in topk_indices_citation_accepted[i][
                                                                           :SIMULATION_CONFIG['citation'][
                                                                               'num_candidate_papers']]]

                    candidate_papers = [
                        {'id': paper.metadata['id'], 'title': paper.metadata['title'], 'abstract': paper.page_content} for
                        paper in candidate_papers]

                    assert set([candidate_paper['id'] for candidate_paper in
                                candidate_papers]) <= accepted_papers_arxiv_ids, f"Submitted paper {submitted_paper['id']} is trying to cite non-accepted papers"

                    prompt, response_format = agent.get_potential_cited_papers_prompt(
                        paper=submitted_paper,
                        direction=directions[agent.id],
                        candidate_papers=candidate_papers,
                        max_papers=30
                    )

                    num_retries = 0

                    while True:
                        try:
                            llm_response, message_history = self.llm.generate(prompt=prompt, response_format=response_format)

                            assert llm_response.get('citations') is not None
                            citations_of_paper = llm_response.get('citations')

                            assert set(citations_of_paper) <= set(
                                [candidate_paper['id'] for candidate_paper in candidate_papers])

                            break

                        except Exception as e:
                            logger.warning(f"Error getting potential cited papers: {e}")
                            traceback.print_exc()
                            num_retries += 1
                            if num_retries >= self.max_retries:
                                raise Exception("Failed to get potential cited papers after 3 retries")

                    # Too few citations, we try to cite rejected papers.
                    if len(citations_of_paper) <= SIMULATION_CONFIG['citation']['expected_citation_per_round']:

                        # Get candidates from all papers, including rejected papers.
                        candidate_papers = [self.rag.documents[idx] for idx in topk_indices_citation_both[i][
                                                                               :SIMULATION_CONFIG['citation'][
                                                                                   'num_candidate_papers']]]

                        candidate_papers = [
                            {'id': paper.metadata['id'], 'title': paper.metadata['title'], 'abstract': paper.page_content}
                            for paper in candidate_papers]

                        prompt, response_format = agent.get_potential_cited_papers_prompt(
                            paper=submitted_paper,
                            direction=directions[agent.id],
                            candidate_papers=candidate_papers,
                            max_papers=30
                        )

                        num_retries = 0

                        while True:
                            try:

                                llm_response, message_history = self.llm.generate(prompt=prompt, response_format=response_format)
                                assert llm_response.get('citations') is not None
                                citations_of_paper = llm_response.get('citations')
                                assert set(citations_of_paper) <= set(
                                    [candidate_paper['id'] for candidate_paper in candidate_papers])
                                break

                            except Exception as e:
                                logger.warning(f"Error getting potential cited papers: {e}")
                                traceback.print_exc()
                                num_retries += 1
                                if num_retries >= self.max_retries:
                                    raise Exception("Failed to get potential cited papers after 3 retries")

                    # Get self-citations
                    logger.info(f"[Self-Citation] Getting self-citations for {agent.id} in year {year}.")
                    own_papers = self.paper_tracker.get_papers_by_author(agent.id)
                    own_papers = [p for p in own_papers if p.project_end_year < year]
                    if own_papers:
                        self_citation_result = agent.get_self_citation_prompt(
                            paper=submitted_paper,
                            direction=directions[agent.id],
                            own_papers=own_papers
                        )

                        if self_citation_result:
                            self_citation_prompt, self_citation_response_format = self_citation_result
                            num_retries = 0
                            while True:
                                try:
                                    llm_response, _ = self.llm.generate(prompt=self_citation_prompt, response_format=self_citation_response_format)
                                    assert llm_response.get('self_citations') is not None
                                    self_citations = llm_response.get('self_citations')
                                    # Combine regular citations with self-citations
                                    citations_of_paper = list(set(citations_of_paper + self_citations))
                                    break
                                except Exception as e:
                                    logger.warning(f"Error getting self-citations: {e}")
                                    traceback.print_exc()
                                    num_retries += 1
                                    if num_retries >= self.max_retries:
                                        logger.warning("Failed to get self-citations after retries, skipping self-citation")
                                        break

                    self.citation_tracker.add_citations(submitted_paper['id'], citations_of_paper, year)

        del message_history_dict
        submitted_papers_dict = {paper['id']: paper for paper in submissions_list + resubmissions_list}

        for paper_dict in submissions_list + resubmissions_list:

            conf: Conference = self.conference_system.conference_map[paper_dict['conference']]
            if paper_dict['author_id'] not in submitted_papers:
                submitted_papers[paper_dict['author_id']] = []

            if conf.submit_paper(paper_dict, year=year):
                submitted_papers[paper_dict['author_id']].append(paper_dict)
                self.submitted_paper_ids.add(paper_dict['id'])

            else:
                raise ValueError(f"Paper {arxiv_id} submission to conference {paper_dict['conference']} failed")

        year_results['paper_submission'] = {
            'num_papers_submitted': sum(len(papers) for author_id, papers in submitted_papers.items()),
        }

        return submissions_list

    def _charge_annual_costs(self, year: int):
        """Charge annual costs for all active researchers working on projects

        Args:
            year: Current year number
        """
        for agent in self.ecosystem.get_available_authors():
            if agent.is_active:

                if agent.project_end_year >= year:
                    # Agent is working on a project
                    annual_cost = SIMULATION_CONFIG['conference']['annual_cost']
                    agent.update_resources(-annual_cost)
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
            for conference in self.conference_system.conferences:
                for paper in conference.submitted_papers:
                    # Assign reviewers (excluding paper author)
                    author = self.ecosystem.get_agent_by_id(paper['author_id'])
                    reviewers = self.ecosystem.get_available_reviewers(
                        exclude_agent_ids=[paper['author_id']] + list(author.conflict_of_interest)
                    )

                    # Select 3 random reviewers
                    selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                    for reviewer in selected_reviewers:
                        # reviewer: SimulationAgent
                        prompt, response_format = reviewer.get_review_prompt(paper)
                        
                        if peer_review_response_format is None:
                            peer_review_response_format = response_format
                        
                        batch_prompts.append(prompt)
                        
                        batch_metadata.append({
                            'reviewer': reviewer,
                            'paper': paper,  
                            'conference': conference,
                            
                        })

            # Step 2: Single batch call for ALL prompts
            if batch_prompts:
                batch_results = self.llm.generate_batch(batch_prompts, temperature=0.7, response_format=peer_review_response_format, desc=f"{len(batch_prompts)} peer reviews")

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
                        review_json['overall_score'] = int(review_json['overall_score'])
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
            for conference in tqdm(self.conference_system.conferences, desc="Peer reviews"):
                for paper in tqdm(conference.submitted_papers, desc=f"{conference.conference_id}"):
                    # Assign reviewers (excluding paper author)
                    author = self.ecosystem.get_agent_by_id(paper['author_id'])
                    reviewers = self.ecosystem.get_available_reviewers(
                        exclude_agent_ids=[paper['author_id']] + list(author.conflict_of_interest)
                    )

                    # Select 3 random reviewers
                    selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                    for reviewer in selected_reviewers:
                        num_retries = 0
                        review = None
                        while True:
                            try:
                                review = reviewer.review_paper(paper)
                                conference.add_review(paper['id'], review)
                                reviews_by_reviewer[reviewer.id].append(review['overall_score'])

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
                    # Find the author agent
                    author = self.ecosystem.get_agent_by_id(paper['author_id'])

                    author.process_received_reviews(paper_reviews, paper_id=paper['id'], year=year,
                                                    average_scores_of_venue=average_scores[conference.conference_id])

        return average_scores

    def _run_phase_4_acceptance_decisions(self, year: int, year_results: Dict):
        """Phase 4: Make acceptance decisions

        Args:
            year: Current year number
            year_results: Year results dict to update
        """
        logger.info(f"Year {year} Phase 4: Making acceptance decisions...")

        self.conference_system.make_all_decisions()

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

    def _run_phase_5_update_funding(self, year: int, submissions_list: List[Dict], year_results: Dict):
        """Phase 5: Update funding for all agents

        Args:
            year: Current year number
            submissions_list: List of submitted papers from Phase 2
            year_results: Year results dict to update
        """
        logger.info(f"Year {year} Phase 5: Updating funding...")

        # Calculate replenishment needed to maintain constant total funding
        funding_consumed = self.funding_tracker.get_funding_consumption(year)
        logger.info(
            f"Funding consumed this cycle: University: {funding_consumed['university']:.2f}, Industry: {funding_consumed['industry']:.2f}")

        # Process university researchers - apply for funding
        university_applications = []

        # all papers (accepted, rejected, or long-term projects in preparation)
        active_papers_dataframe = self.paper_tracker.get_papers_dataframe(year)

        print(active_papers_dataframe)

        logger.info(f"[Funding Allocation] Update fundings for industry researchers in year {year}")
        # Calculate the number of recipients of industry funding
        denominator = active_papers_dataframe.loc[
                (active_papers_dataframe['author_type'] == 'industry')
                & (active_papers_dataframe['status'] == 'accept'),
                'maturity'
            ].sum()

        if denominator == 0:
            industry_funding_per_unit = 0
        elif self.funding_allocation_mode == 'fixed':
            industry_funding_per_unit = SIMULATION_CONFIG['funding']['industry_base_budget']
        else:
            industry_funding_per_unit = max(self.funding_tracker.get_funding_allocation_amount(year, 'industry') // denominator, 1)

        # Get funding agencies
        funding_agencies = [
            agent for agent_id, agent in self.ecosystem.agent_population.items()
            if isinstance(agent, FundingAgency)
        ]

        # Update funding for industry researchers through heuristics. Ask university researchers to apply for funding.
        for agent in tqdm(self.ecosystem.get_available_authors(), desc="Updating funding"):
            if isinstance(agent, IndustryResearcher):
                # Update industry budget based on performance
                accepted = self.conference_system.authors_to_accepted_papers.get(agent.id, [])

                for paper in accepted:
                    new_budget = self.industry_funding_system.update_budget(
                        funding_per_unit=industry_funding_per_unit,
                        accepted=1,
                        maturity=paper['maturity'],
                        weighted_funding_assignment=True
                    )
                    if new_budget > 0:
                        agent.update_resources(delta=int(new_budget))
                        self.funding_tracker.record_funding_allocation(year, new_budget, 'industry')


                # Count citations for each accepted paper
                citation_counts = []
                for paper in self.conference_system.authors_to_accepted_papers.get(agent.id, []):
                    citation_counts.append(self.citation_tracker.get_citation_count(paper['id']))

            elif isinstance(agent, UniversityResearcher):
                pass  # Will be processed in batch below

            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")

        # Batch process university researcher funding applications
        all_programs = {pid: program for funding_agent in funding_agencies for pid, program in funding_agent.funding_programs.items()}
        university_researchers = [
            agent for agent in self.ecosystem.get_available_authors()
            if isinstance(agent, UniversityResearcher)
        ]

        # Collect prompts and metadata for batch processing
        batch_prompts = []
        batch_metadata = []
        funding_response_format = None

        for agent in university_researchers:
            success_history = agent.funding_success_history
            selected_program_ids = agent.select_funding_programs(
                all_programs,
                success_history,
                num_submitted_proposals=3
            )
            if selected_program_ids:
                prompt, response_format, metadata = agent.get_funding_application_prompt(
                    past_accepted_papers=self.conference_system.authors_to_accepted_papers.get(agent.id, []),
                    past_rejected_papers=self.conference_system.authors_to_rejected_papers.get(agent.id, []),
                    programs=all_programs,
                    selected_program_ids=selected_program_ids,
                    success_history=success_history
                )
                if prompt:
                    batch_prompts.append(prompt)
                    batch_metadata.append(metadata)
                    if funding_response_format is None:
                        funding_response_format = response_format

        # Generate all funding applications in one batch call (if vLLM) or sequentially
        if batch_prompts:
            if hasattr(self.llm, 'generate_batch'):
                batch_results = self.llm.generate_batch(batch_prompts, temperature=0.7, response_format=funding_response_format, desc=f"{len(batch_prompts)} funding applications")

                # Process results with validation
                for i, (proposal, batch_message_history) in enumerate(batch_results):
                    if proposal is None:
                        logger.warning(f"Failed to parse funding application for agent {batch_metadata[i]['agent'].id}")
                        continue

                    metadata = batch_metadata[i]
                    agent = metadata['agent']
                    all_paper_ids = metadata['all_paper_ids']

                    # Filter out invalid referred papers (instead of discarding entire proposal)
                    for program_id, program_data in proposal.items():
                        if program_data.get('submit', True):
                            referred_papers = program_data.get('relevant_projects', [])
                            valid_papers = [p for p in referred_papers if p.get('arxiv_id') in all_paper_ids]
                            if len(valid_papers) < len(referred_papers):
                                logger.warning(f"Agent {agent.id} program {program_id}: filtered {len(referred_papers) - len(valid_papers)} invalid paper references")
                            program_data['relevant_projects'] = valid_papers

                    # Add author to application
                    for k in proposal:
                        proposal[k]['author'] = agent
                    university_applications.append(proposal)
            else:
                # Sequential fallback for non-vLLM models
                for i, prompt in enumerate(tqdm(batch_prompts, desc="Generating funding applications")):
                    metadata = batch_metadata[i]
                    agent = metadata['agent']
                    num_retries = 0
                    while True:
                        try:
                            proposal, _ = self.llm.generate(prompt=prompt)

                            # Filter out invalid referred papers
                            all_paper_ids = metadata['all_paper_ids']
                            for program_id, program_data in proposal.items():
                                if program_data.get('submit', True):
                                    referred_papers = program_data.get('relevant_projects', [])
                                    valid_papers = [p for p in referred_papers if p.get('arxiv_id') in all_paper_ids]
                                    if len(valid_papers) < len(referred_papers):
                                        logger.warning(f"Agent {agent.id} program {program_id}: filtered {len(referred_papers) - len(valid_papers)} invalid paper references")
                                    program_data['relevant_projects'] = valid_papers

                            for k in proposal:
                                proposal[k]['author'] = agent
                            university_applications.append(proposal)
                            break

                        except Exception as e:
                            logger.warning(f"Error preparing funding application: {e}")
                            traceback.print_exc()
                            num_retries += 1
                            if num_retries >= self.max_retries:
                                raise Exception("Failed to prepare funding application after retries")
                            time.sleep(1)

        # Evaluate applications across all funding agencies
        all_funding_winners_dict, all_funding_programs_dict = {}, {}

        logger.debug(f"Calculating weights for funding applications in year {year}...")

        total_weights = []

        winners_list = []
        submitted_papers_dict = {pid: paper.to_dict() for pid, paper in self.paper_tracker.papers_by_id.items()}
        for funding_agent in funding_agencies:
            winners_for_program = funding_agent.evaluate_funding_applications(
                university_applications,
                submitted_papers_dict=submitted_papers_dict
            )
            all_funding_winners_dict.update(winners_for_program)
            all_funding_programs_dict.update(funding_agent.funding_programs)

            for program_id, winners in winners_for_program.items():
                winners_list.extend([agent['id'] for agent in winners])

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
                program = all_funding_programs_dict[program_id]
                agent = self.ecosystem.get_agent_by_id(winner['id'])

                funding_to_researcher = academic_funding_per_unit

                agent.update_resources(delta=funding_to_researcher)
                self.funding_tracker.record_funding_allocation(year, funding_to_researcher, 'university')
                if not program_id in agent.funding_success_history:
                    agent.funding_success_history[program_id] = []

                agent.funding_success_history[program_id].append({
                    'year': year,
                    'amount': funding_to_researcher
                })

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

        # Phase 2: Generate and submit papers
        submissions_list = self._run_phase_2_submit_papers(
            year, directions, active_authors, resubmissions_list, year_results
        )
        self.wandb_logger.log_phase(year, 2, year_results, self)

        # Charge annual costs
        self._charge_annual_costs(year)

        # Phase 3: Peer review
        average_scores = self._run_phase_3_peer_review(year, year_results)
        self.wandb_logger.log_phase(year, 3, year_results, self)

        # Phase 4: Make acceptance decisions
        self._run_phase_4_acceptance_decisions(year, year_results)
        self.wandb_logger.log_phase(year, 4, year_results, self)

        # Phase 5: Update funding
        self._run_phase_5_update_funding(year, submissions_list, year_results)

        # Collect exploration metrics (if enabled)
        self._collect_exploration_metrics(year)

        # Get conference statistics
        year_results['conference_statistics'] = self.conference_system.get_all_statistics()
        self.wandb_logger.log_phase(year, 5, year_results, self)

        return year_results

    def run(self):
        """Run the complete simulation"""
        logger.info("Starting Academic Ecosystem Simulation")
        logger.info(f"Configuration: {self.num_years} years")

        start_year = 1
        checkpoint_loaded = False
        loaded_year = None

        # Try to load from checkpoint unless always_rerun is specified
        if not self.always_rerun:
            # Try to find the most recent checkpoint
            for year in range(self.num_years, 0, -1):
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")
                if os.path.exists(checkpoint_path):
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
            selected_conferences = select_conferences_for_simulation(
                agents=research_agents,
                num_conferences=1 if self.debug else 10
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

        logger.info("\nSimulation completed!")
        logger.info(f"Results saved to: {self.output_dir}")

    def _save_checkpoint(self, year: int, phase: int = 5):
        """Save complete simulation state as checkpoint

        Args:
            year: Year number
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")

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
            'submitted_paper_ids': list(self.submitted_paper_ids)  # Convert set to list
        }

        # Save exploration experiment trackers if enabled
        if self.experiment_name == 'exploration_vs_exploitation':
            checkpoint_data['exploration_experiment'] = {
                'enabled': True,
                'embedding_tracker': self.embedding_tracker.to_dict() if self.embedding_tracker else None,
                'exploration_metrics': self.exploration_metrics.to_dict() if self.exploration_metrics else None,
            }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved for year {year}: {checkpoint_path}")

    def load_checkpoint(self, year: int):
        """Load complete simulation state from checkpoint

        Args:
            year: Year number to load from

        Returns:
            bool: True if checkpoint loaded successfully, False otherwise
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")

        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False

        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        self.current_year = checkpoint_data['year']

        # Reconstruct ecosystem with LLM from serialized data
        self.ecosystem = MultiAgentEcosystem.from_dict(
            checkpoint_data['ecosystem_data'],
            llm=self.llm
        )

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

        # Reconstruct RAG with document_status from checkpoint
        if 'rag' in checkpoint_data:
            rag_device = getattr(self.args, 'rag_device', None)
            rag_config = RAGConfig(use_langchain=self.use_langchain, **({'device': rag_device} if rag_device else {}))
            self.rag = RAG.from_dict(checkpoint_data['rag'], config=rag_config, model=None,
                                     debug=self.debug, start_year=self.start_year)
            logger.info(f"  Loaded RAG with {len(self.rag.document_status)} document statuses")

        # Reconstruct exploration experiment trackers if enabled
        if 'exploration_experiment' in checkpoint_data and checkpoint_data['exploration_experiment'].get('enabled', False):
            self.experiment_name = 'exploration_vs_exploitation'
            exp_data = checkpoint_data['exploration_experiment']

            if exp_data.get('embedding_tracker'):
                from utopia.metrics.embedding_tracker import EmbeddingTracker
                self.embedding_tracker = EmbeddingTracker.from_dict(exp_data['embedding_tracker'])
                logger.info(f"  Loaded EmbeddingTracker with {len(self.embedding_tracker.embeddings)} embeddings")

            if exp_data.get('exploration_metrics'):
                from utopia.metrics.exploration_metrics import ExplorationMetrics
                self.exploration_metrics = ExplorationMetrics.from_dict(exp_data['exploration_metrics'])
                logger.info(f"  Loaded ExplorationMetrics")

            # CD calculator will be lazily initialized when needed
            self.cd_calculator = None
            self.keyword_extractor = None

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
    """Main entry point"""

    project_setup()

    args = parse_arguments()

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
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain,
        max_retries=args.max_retries,
        weighted_funding_assignment=args.weighted_funding_assignment,
        funding_allocation_mode=args.funding_allocation_mode,
        experiment_name=args.experiment_name,
        start_year=args.start_year,
        args=args
    )

    simulation.run()


if __name__ == '__main__':
    main()
