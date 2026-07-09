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
    get_strategy_filtered_candidate_directions,
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

        assert self.experiment_name.startswith('default') or self.experiment_name in ['exploration_vs_exploitation', 'preferential_attachment'], f"Invalid experiment name: {self.experiment_name}"

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

        # Exploration experiment trackers (lazily initialized)
        self.embedding_tracker = None
        self.cd_calculator = None
        self.exploration_metrics = None
        self.keyword_extractor = None

        # Exploration experiment: paper novelty scores cache
        self._paper_novelty_scores = {}  # paper_id -> (novelty_score, bucket)
        self._breakthrough_papers = {}  # paper_id -> citation_potential_multiplier

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
                if self.is_exploration_experiment:
                    expertise = random.sample(AVAILABLE_DIRECTIONS, min(3, len(AVAILABLE_DIRECTIONS)))
                elif self.debug:
                    expertise = [AVAILABLE_DIRECTIONS[0]]
                else:
                    expertise = random.sample(AVAILABLE_DIRECTIONS, 3)
                # researcher_name = f"university_researcher_{counter}"
                researcher_name = f"{university}_researcher_{i}"

                funding_level = 160 if university in rich_universities else 80

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
        if not self.is_exploration_experiment:
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
        if not self.is_exploration_experiment:
            return

        # Guard against empty abstract
        abstract = paper_dict.get('abstract', '')
        if not abstract or not abstract.strip():
            return

        # Lazy initialize trackers on first use
        self._initialize_experiment_trackers()

        exp_config = SIMULATION_CONFIG['exploration_experiment']

        # Extract keywords (now returns dict)
        if self.keyword_extractor is not None:
            kw_result = self.keyword_extractor.extract_keywords_llm(
                abstract=abstract,
                llm=self.llm,
                n_keywords=exp_config['num_keywords']
            )
            paper_dict['keywords'] = kw_result.get('keywords', [])
            paper_dict['keywords_in_abstract'] = kw_result.get('keywords_in_abstract', [])
            paper_dict['keywords_not_in_abstract'] = kw_result.get('keywords_not_in_abstract', [])

        # Add embedding
        if self.embedding_tracker is not None:
            self.embedding_tracker.add_paper_embedding(
                paper_id=paper_dict['id'],
                abstract=abstract,
                author_id=author_id,
                year=year
            )
            logger.debug(f"Added embedding for {paper_dict['id']}")

            # Compute and store novelty score
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

            # Store exploration strategy on paper
            agent = self.ecosystem.get_agent_by_id(author_id)
            paper_dict['exploration_strategy'] = getattr(agent, 'exploration_strategy', 'balanced')

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
            logger.info(f"Year {year}: No papers to collect exploration metrics for")
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

        # Collect all citation counts for accepted papers this year
        all_citation_counts = []
        paper_citation_map = {}
        for pid in paper_ids_this_year:
            if pid in accepted_paper_ids:
                count = len(self.citation_tracker.get_citations_to_paper(pid))
                # Apply breakthrough multiplier for effective citations
                multiplier = self._breakthrough_papers.get(pid, 1.0)
                effective_count = count * multiplier
                all_citation_counts.append(effective_count)
                paper_citation_map[pid] = effective_count

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

            # Record per-agent stats
            for a in agents:
                agent_papers_yr = [pid for pid, aid in papers_this_year if aid == a.id]
                agent_accepted_yr = [pid for pid in agent_papers_yr if pid in accepted_paper_ids]
                agent_citations = [paper_citation_map.get(pid, 0) for pid in agent_accepted_yr]

                self.exploration_metrics.record_agent_stats(a.id, year, {
                    'strategy': strategy,
                    'num_papers': len(agent_papers_yr),
                    'num_accepted': len(agent_accepted_yr),
                    'acceptance_rate': len(agent_accepted_yr) / len(agent_papers_yr) if agent_papers_yr else 0,
                    'funding': getattr(a, 'resources', 0),
                    'total_citations': sum(agent_citations),
                    'hit_papers': sum(1 for c in agent_citations if c >= hit_threshold),
                })

        logger.info(f"Year {year}: Collected exploration metrics for {len(paper_ids_this_year)} papers")

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
                    desc=f"{len(batch_prompts)} collaboration decisions"
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

        logger.info(f"Year {year} Phase 1.5: Formed {len(collaboration_pairs)} collaboration pairs "
                     f"(total: {self.collaboration_tracker.graph.number_of_edges()} edges, "
                     f"{self.collaboration_tracker.graph.number_of_nodes()} nodes)")

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

        # For exploration experiment: pre-filter candidate directions per agent
        if self.is_exploration_experiment and authors_with_directions_pending:
            self._initialize_experiment_trackers()
            exp_config = SIMULATION_CONFIG['exploration_experiment']
            for agent in authors_with_directions_pending:
                filtered = get_strategy_filtered_candidate_directions(
                    agent, AVAILABLE_DIRECTIONS, self.embedding_tracker, year, exp_config
                )
                agent._filtered_candidates = filtered

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

        # Identify co-authors claimed by collaboration pairs (they don't submit their own paper)
        collab_pairs = self._collaboration_pairs.get(year, {})
        co_author_ids = set(collab_pairs.values())
        if co_author_ids:
            original_count = len(active_authors)
            active_authors = [a for a in active_authors if a.id not in co_author_ids]
            logger.info(f"Phase 2: {original_count - len(active_authors)} agents are co-authors this round, "
                        f"{len(active_authors)} agents will submit as lead authors")

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
                intention_prompts, max_tokens=SIMULATION_CONFIG['llm']['max_tokens'],
                response_format=intention_response_format,
                desc=f"Generating {len(intention_prompts)} intentions"
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

                batch_results = self.llm.generate_batch(batch_prompts, max_tokens=SIMULATION_CONFIG['llm']['max_tokens'], temperature=0.7, response_format=paper_submission_response_format, desc=f"{len(batch_prompts)} paper submissions (attempt {attempt+1}/{MAX_SUBMISSION_RETRIES})")

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
                        'tags': paper_doc.metadata['tags'],
                        'topics': paper_doc.metadata['topics'],
                        'conference': submission_metadata['conference'],
                        'type': 'submission',
                        'year': year,
                        'project_start_year': agent.project_start_year,
                        'project_end_year': agent.project_end_year,
                        'maturity': directions[agent.id]['direction'].years,
                        'status': 'pending'
                    }
                    self._apply_coauthorship(paper_dict, agent, year)

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
                        'tags': paper_doc.metadata['tags'],
                        'topics': paper_doc.metadata['topics'],
                        'conference': conference,
                        'type': 'submission',
                        'year': year,
                        'project_start_year': agent.project_start_year,
                        'project_end_year': agent.project_end_year,
                        'maturity': directions[agent.id]['direction'].years,
                        'status': 'pending'
                    }
                    self._apply_coauthorship(paper_dict, agent, year)

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
                        'tags': paper_doc.metadata['tags'],
                        'topics': paper_doc.metadata['topics'],
                        'conference': submission_metadata['conference'],
                        'type': 'submission',
                        'year': year,  # Current year for submission
                        'project_start_year': agent.project_start_year,  # The year when the agent started the project.
                        'project_end_year': agent.project_end_year,
                        # The year when the agent's latest research paper will be ready for submission
                        'maturity': directions[agent.id]['direction'].years,
                        'status': 'pending'
                    }
                    self._apply_coauthorship(paper_dict, agent, year)

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
                    agent = self.ecosystem.get_agent_by_id(self._get_lead_author_id(submitted_paper))
                    candidate_papers = self._build_candidate_papers(
                        self.rag.documents,
                        topk_indices_citation_accepted[i][:SIMULATION_CONFIG['citation']['num_candidate_papers']]
                    )

                    assert set([cp['id'] for cp in candidate_papers]) <= accepted_papers_arxiv_ids, \
                        f"Submitted paper {submitted_paper['id']} is trying to cite non-accepted papers"
                    citing_institution = getattr(agent, 'university_name', None) or getattr(agent, 'company_name', None)
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
                        num_candidates = len(meta['candidate_papers'])
                        if not all(isinstance(c, int) and 0 <= c < num_candidates for c in citations):
                            logger.warning(f"Invalid citations for paper {meta['paper']['id']}, retrying")
                            items_needing_retry.append(meta)
                            continue

                        regular_citations[idx] = [meta['candidate_papers'][c]['id'] for c in citations]

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
                        agent = self.ecosystem.get_agent_by_id(self._get_lead_author_id(submitted_paper))
                        candidate_papers = self._build_candidate_papers(
                            self.rag.documents,
                            topk_indices_citation_both[i][:SIMULATION_CONFIG['citation']['num_candidate_papers']]
                        )

                        citing_institution = getattr(agent, 'university_name', None) or getattr(agent, 'company_name', None)
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
                        num_candidates = len(meta['candidate_papers'])
                        if not all(isinstance(c, int) and 0 <= c < num_candidates for c in citations):
                            logger.warning(f"Invalid citations for paper {meta['paper']['id']}, retrying")
                            items_needing_retry.append(meta)
                            continue

                        extended_citations[idx] = [meta['candidate_papers'][c]['id'] for c in citations]

                    if not items_needing_retry:
                        break

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

                # --- Batch 3: Self-Citations ---
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
                    agent = self.ecosystem.get_agent_by_id(self._get_lead_author_id(submitted_paper))
                    candidate_papers = self._build_candidate_papers(
                        self.rag.documents,
                        topk_indices_citation_accepted[i][:SIMULATION_CONFIG['citation']['num_candidate_papers']]
                    )

                    assert set([candidate_paper['id'] for candidate_paper in
                                candidate_papers]) <= accepted_papers_arxiv_ids, f"Submitted paper {submitted_paper['id']} is trying to cite non-accepted papers"

                    citing_institution = getattr(agent, 'university_name', None) or getattr(agent, 'company_name', None)
                    prompt, response_format = agent.get_citation_prompt(
                        paper=submitted_paper,
                        direction=directions[agent.id],
                        candidate_papers=candidate_papers,
                        max_papers=30,
                        citing_institution=citing_institution
                    )

                    num_retries = 0

                    while True:
                        try:
                            llm_response, message_history = self.llm.generate(prompt=prompt, response_format=response_format)

                            assert llm_response.get('citations') is not None
                            citations_of_paper = llm_response.get('citations')

                            num_candidates = len(candidate_papers)
                            assert all(isinstance(c, int) and 0 <= c < num_candidates for c in citations_of_paper)
                            citations_of_paper = [candidate_papers[c]['id'] for c in citations_of_paper]

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
                        candidate_papers = self._build_candidate_papers(
                            self.rag.documents,
                            topk_indices_citation_both[i][:SIMULATION_CONFIG['citation']['num_candidate_papers']]
                        )

                        prompt, response_format = agent.get_citation_prompt(
                            paper=submitted_paper,
                            direction=directions[agent.id],
                            candidate_papers=candidate_papers,
                            max_papers=30,
                            citing_institution=citing_institution
                        )

                        num_retries = 0

                        while True:
                            try:

                                llm_response, message_history = self.llm.generate(prompt=prompt, response_format=response_format)
                                assert llm_response.get('citations') is not None
                                citations_of_paper = llm_response.get('citations')
                                num_candidates = len(candidate_papers)
                                assert all(isinstance(c, int) and 0 <= c < num_candidates for c in citations_of_paper)
                                citations_of_paper = [candidate_papers[c]['id'] for c in citations_of_paper]
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
                                    assert all(isinstance(c, int) and 0 <= c < len(own_papers) for c in self_citations)
                                    self_citations = [own_papers[c].id for c in self_citations]
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
            lead_id = self._get_lead_author_id(paper_dict)
            if lead_id not in submitted_papers:
                submitted_papers[lead_id] = []

            if conf.submit_paper(paper_dict, year=year):
                submitted_papers[lead_id].append(paper_dict)
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
                        prompt, response_format = reviewer.get_review_prompt(paper, author_info=author_info)

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

        # Collect prompts and metadata for batch processing — one prompt per (agent, program)
        batch_prompts = []
        batch_metadata = []
        funding_response_format = None

        for agent in university_researchers:
            success_history = agent.funding_success_history

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
                if prompt:
                    batch_prompts.append(prompt)
                    batch_metadata.append(metadata)
                    if funding_response_format is None:
                        funding_response_format = response_format

        # Generate all funding applications in one batch call (if vLLM) or sequentially
        if batch_prompts:
            if hasattr(self.llm, 'generate_batch'):
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
                            desc=f"{len(batch_prompts)} funding applications (attempt {attempt+1}/{MAX_FUNDING_RETRIES})"
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

                # Reassemble into per-agent dicts: {program_id: app_data}
                agent_applications = {}
                for agent_id, program_id, app_data, agent in successful_items:
                    app_data['author'] = agent
                    agent_applications.setdefault(agent_id, {})[program_id] = app_data
                university_applications.extend(agent_applications.values())
            else:
                # Sequential fallback for non-vLLM models
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

                # Reassemble into per-agent dicts
                agent_applications = {}
                for agent_id, program_id, app_data, agent in sequential_items:
                    app_data['author'] = agent
                    agent_applications.setdefault(agent_id, {})[program_id] = app_data
                university_applications.extend(agent_applications.values())

        # Evaluate applications across all funding agencies
        all_funding_winners_dict, all_funding_programs_dict = {}, {}

        logger.debug(f"Calculating weights for funding applications in year {year}...")

        total_weights = []

        winners_list = []
        submitted_papers_dict = {pid: paper.to_dict() for pid, paper in self.paper_tracker.papers_by_id.items()}

        # Step 1: Collect prompts from ALL funding agencies into one batch
        all_funding_proposal_eval_prompts = []
        all_eval_metadata = []
        eval_response_format = None

        for funding_agent in funding_agencies:
            all_funding_programs_dict.update(funding_agent.funding_programs)
            funding_proposal_eval_prompts, resp_fmt, eval_metadata = funding_agent.get_funding_evaluation_prompts(
                university_applications, submitted_papers_dict=submitted_papers_dict
            )
            all_funding_proposal_eval_prompts.extend(funding_proposal_eval_prompts)
            all_eval_metadata.extend(eval_metadata)
            if eval_response_format is None:
                eval_response_format = resp_fmt

        # Step 2: LLM call with retry for invalid results
        if all_funding_proposal_eval_prompts:
            MAX_FUNDING_EVAL_RETRIES = 3
            final_eval_results = []
            final_eval_metadata = []

            for attempt in range(MAX_FUNDING_EVAL_RETRIES):
                if not all_funding_proposal_eval_prompts:
                    break

                if hasattr(self.llm, 'generate_batch'):
                    eval_results = self.llm.generate_batch(
                        all_funding_proposal_eval_prompts, response_format=eval_response_format,
                        desc=f"{len(all_funding_proposal_eval_prompts)} funding evaluations (attempt {attempt+1}/{MAX_FUNDING_EVAL_RETRIES})"
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
                    else:
                        program_id = all_eval_metadata[i]['program_id']
                        logger.warning(f"Funding eval for {program_id} invalid (attempt {attempt+1}/{MAX_FUNDING_EVAL_RETRIES})")
                        if attempt < MAX_FUNDING_EVAL_RETRIES - 1:
                            retry_prompts.append(all_funding_proposal_eval_prompts[i])
                            retry_metadata.append(all_eval_metadata[i])
                        else:
                            # Last attempt — pass through for fallback handling
                            final_eval_results.append((result, msg_hist))
                            final_eval_metadata.append(all_eval_metadata[i])

                if not retry_prompts:
                    break

                all_funding_proposal_eval_prompts = retry_prompts
                all_eval_metadata = retry_metadata

            # Step 3: Process all results (valid + any final failures with fallback)
            all_funding_winners_dict = FundingAgency.process_funding_evaluation_results(
                final_eval_results, final_eval_metadata, all_funding_programs_dict
            )

            # Apply funding conservatism for exploration experiment
            self._apply_funding_conservatism(all_funding_winners_dict, year)

            for program_id, winners in all_funding_winners_dict.items():
                winners_list.extend([agent['applicant_id'] for agent in winners])

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
                agent = self.ecosystem.get_agent_by_id(winner['applicant_id'])

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

    def _apply_strategy_quality_adjustment(self, year: int):
        """Apply strategy-dependent quality adjustment to review scores.

        Adds noise based on exploration strategy: explorers have higher variance
        but slight negative bias, exploiters have low variance and slight positive bias.
        Only active for exploration experiment.
        """
        if not self.is_exploration_experiment:
            return

        exp_config = SIMULATION_CONFIG['exploration_experiment']
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

    def _apply_citation_breakthrough_bonus(self, year: int):
        """Mark high-novelty accepted papers as potential breakthroughs.

        Stores multiplier in self._breakthrough_papers for use in metrics.
        Only active for exploration experiment.
        """
        if not self.is_exploration_experiment:
            return

        exp_config = SIMULATION_CONFIG['exploration_experiment']
        threshold = exp_config['breakthrough_probability_threshold']
        multiplier_mean = exp_config['breakthrough_multiplier_mean']

        for conf in self.conference_system.conferences:
            for paper in conf.decisions.get('accept', []):
                novelty_data = self._paper_novelty_scores.get(paper['id'])
                if novelty_data is None:
                    continue

                novelty_score = novelty_data[0]
                if novelty_score > threshold:
                    # Probability of breakthrough proportional to novelty
                    if random.random() < novelty_score:
                        self._breakthrough_papers[paper['id']] = multiplier_mean
                        logger.info(f"Breakthrough paper: {paper['id']} (novelty={novelty_score:.3f}, multiplier={multiplier_mean})")

    def _apply_funding_conservatism(self, all_funding_winners_dict: Dict, year: int):
        """Re-rank funding winners to penalize high-novelty researchers.

        Reduces effective scores for researchers with high average novelty,
        modeling real-world conservatism in funding decisions.
        Only active for exploration experiment.
        """
        if not self.is_exploration_experiment:
            return

        exp_config = SIMULATION_CONFIG['exploration_experiment']
        penalty_weight = exp_config['novelty_penalty_in_funding']

        for program_id, winners in all_funding_winners_dict.items():
            for winner in winners:
                agent_id = winner.get('applicant_id')
                if not agent_id:
                    continue

                # Get average novelty from recent papers
                agent_papers = [
                    pid for pid, meta in self.embedding_tracker.paper_metadata.items()
                    if meta['author_id'] == agent_id
                ] if self.embedding_tracker else []

                if not agent_papers:
                    continue

                novelty_scores = [
                    self._paper_novelty_scores[pid][0]
                    for pid in agent_papers
                    if pid in self._paper_novelty_scores
                ]

                if novelty_scores:
                    avg_novelty = np.mean(novelty_scores)
                    # Reduce effective score (stored as penalty for downstream use)
                    winner['novelty_penalty'] = penalty_weight * avg_novelty

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

        # Apply citation breakthrough bonus for high-novelty accepted papers
        self._apply_citation_breakthrough_bonus(year)

        # Phase 5: Update funding
        self._run_phase_5_update_funding(year, submissions_list, year_results)

        # Collect exploration metrics (if enabled)
        self._collect_exploration_metrics(year)

        # Collect preferential attachment metrics (if enabled)
        self._collect_preferential_attachment_metrics(year)

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
        if self.is_exploration_experiment:
            checkpoint_data['exploration_experiment'] = {
                'enabled': True,
                'embedding_tracker': self.embedding_tracker.to_dict() if self.embedding_tracker else None,
                'exploration_metrics': self.exploration_metrics.to_dict() if self.exploration_metrics else None,
                'paper_novelty_scores': {k: list(v) for k, v in self._paper_novelty_scores.items()},
                'breakthrough_papers': self._breakthrough_papers,
            }

        # Save preferential attachment experiment trackers if enabled
        if self.experiment_name == 'preferential_attachment':
            checkpoint_data['preferential_attachment'] = {
                'enabled': True,
                'collaboration_tracker': self.collaboration_tracker.to_dict() if self.collaboration_tracker else None,
                'network_metrics': self.network_metrics.to_dict() if self.network_metrics else None,
                'collaboration_pairs': {str(k): v for k, v in self._collaboration_pairs.items()},
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

            # Restore paper novelty scores and breakthrough papers
            if exp_data.get('paper_novelty_scores'):
                self._paper_novelty_scores = {k: tuple(v) for k, v in exp_data['paper_novelty_scores'].items()}
                logger.info(f"  Loaded {len(self._paper_novelty_scores)} paper novelty scores")
            if exp_data.get('breakthrough_papers'):
                self._breakthrough_papers = exp_data['breakthrough_papers']
                logger.info(f"  Loaded {len(self._breakthrough_papers)} breakthrough papers")

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
        funding_allocation_mode=args.funding_allocation_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain,
        max_retries=args.max_retries,
        weighted_funding_assignment=args.weighted_funding_assignment,
        experiment_name=args.experiment_name,
        collaboration_mode=args.collaboration_mode,
        start_year=args.start_year,
        args=args
    )

    simulation.run()


if __name__ == '__main__':
    main()
