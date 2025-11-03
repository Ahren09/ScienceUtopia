#!/usr/bin/env python3
"""
Main simulation orchestration script for the Academic Society Ecosystem

This script runs the full multi-agent simulation with:
- Multiple UniversityResearchers and IndustryResearchers
- Multiple conferences targeting different topics
- Configurable research direction assignment (batch or individual mode)
- Yearly cycles with funding management
"""

import json
import logging
import os
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
    assign_directions_batch,
    assign_directions_individual,
    initialize_agent_expertise,
    AVAILABLE_DIRECTIONS
)
from utopia.agents.researcher_agents import (
    UniversityResearcher,
    IndustryResearcher
)
from utopia.analysis.analyze_all_results import SimulationAnalyzer
from utopia.arguments import parse_arguments
from utopia.config import SIMULATION_CONFIG
from utopia.data.paper_tracker import PaperTracker
from utopia.data.rag import RAG, RAGConfig
from utopia.data.tracker import CitationTracker, AgentTracker, FundingTracker
from utopia.metrics.tracker import calculate_gini_coefficient, calculate_conference_score_metrics
from utopia.models.models import GPT, LLM
from utopia.utils.generail_utils import project_setup


logger = logging.getLogger(__name__)


class Simulation:
    """Main simulation orchestrator for the academic ecosystem"""

    def __init__(
            self,
            llm,
            num_years: int = 5,
            output_dir: str = "outputs/",
            **kwargs
    ):

        """Initialize the simulation

        Args:
            num_years: Number of years to simulate
            output_dir: Directory to save simulation results
            debug: Enable debug mode
            industry_funding_mode: Funding mode for industry researchers - "performance" or "consistent"
            always_rerun: If True, ignore all checkpoints and start from scratch
        """
        self.llm: LLM = llm
        self.num_years = num_years
        self.output_dir = output_dir
        self.current_year = 1
        self.debug = kwargs.get('debug', False)
        self.verbose = kwargs.get('verbose', False)
        self.citation_tracker = CitationTracker()
        self.agent_tracker = AgentTracker()
        self.funding_tracker = FundingTracker()
        self.industry_funding_mode = kwargs.get('industry_funding_mode', 'performance')
        self.always_rerun = kwargs.get('always_rerun', False)
        self.use_langchain = kwargs.get('use_langchain', False)
        self.max_retries = kwargs.get('max_retries', 5)
        self.weighted_funding_assignment = kwargs.get('weighted_funding_assignment', False)
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

        rag_config = RAGConfig(use_langchain=self.use_langchain)

        sentence_transformer_model = None
        if not self.use_langchain:
            from sentence_transformers import SentenceTransformer
            sentence_transformer_model = SentenceTransformer(rag_config.embedding_model, device=rag_config.device)
        self.rag: RAG = RAG(config=rag_config, debug=self.debug, model=sentence_transformer_model)

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
            universities = rich_universities[:1] + normal_universities[:1]
            num_researchers = 2
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

                researcher = UniversityResearcher(
                    researcher_name=researcher_name,
                    university_name=university,
                    funding_level=funding_level,
                    expertise=expertise,
                    llm=self.llm,
                    generate_research_proposal=True,
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
            companies = rich_companies[:1] + normal_companies[:1]
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

        # Initialize expertise for all agents (3 research directions per agent)
        all_agents = [agent for agent in self.ecosystem.agent_population if hasattr(agent, 'expertise_directions')]
        initialize_agent_expertise(all_agents, num_expertise=3)
        logger.info(f"Initialized expertise for {len(all_agents)} agents")

    def calculate_average_funding_level(self):
        """Calculate the average funding level of all agents"""
        funding_levels = [agent.resources for agent_id, agent in self.ecosystem.agent_population.items() if
                          getattr(agent, "resources", None) is not None]
        return np.mean(funding_levels)

    def run_one_round(self, year: int) -> Dict:
        """Run simulation for one year

        Args:
            year: Current year number

        Returns:
            Dictionary with year results
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"YEAR {year}")
        logger.info(f"{'=' * 60}\n")

        # Reset all fundings to 0 at start of cycle
        total_funding = self.funding_tracker.start_cycle(year, self.ecosystem.agent_population)
        logger.info(f"Total funding at start of year {year}: {total_funding:.2f}")

        # Reset conferences for the new year of simulation
        self.conference_system.reset_all_for_new_year(year)


        # Determine assignment mode based on year
        research_direction_selection_mode = "batch" if year == 1 else "individual"

        year_results = {
            'year': year
        }
        # Phase 0: Resubmissions (year 2+)
        resubmissions_list = []
        resubmissions_paper_ids = set()
        if year > 1:
            logger.info(f"Year {year} Phase 0: Processing resubmissions from year {year - 1}...")
            for author_id, papers in self.paper_tracker.papers_by_author.items():
                agent = self.ecosystem.get_agent_by_id(author_id)

                if not agent:
                    continue

                rejected_papers = {paper.id: paper.to_dict() for paper in papers if paper.status == 'reject'}
                if not rejected_papers:
                    continue

                resubmissions_of_author = agent.decide_resubmission(rejected_papers, self.conference_system.conferences)

                for metadata in resubmissions_of_author:
                    if not metadata['arxiv_id'] in resubmissions_paper_ids:
                        paper_dict = rejected_papers[metadata['arxiv_id']].copy()
                        paper_dict['conference'] = metadata['conference']
                        paper_dict['year'] = year
                        paper_dict['type'] = 'resubmission'
                        # Preserve review_history for accumulation across submissions
                        review_history_backup = paper_dict.get('review_history', [])
                        del paper_dict['status']
                        if 'review_history' in paper_dict:
                            del paper_dict['review_history']
                        paper_dict['status'] = 'pending'
                        paper_dict['review_history'] = review_history_backup  # Will be restored by paper_tracker

                        resubmissions_list.append(paper_dict)
                        self.paper_tracker.add_or_update_paper(paper_dict)
                        resubmission_cost = SIMULATION_CONFIG['conference']['resubmission_cost']
                        agent.update_resources(-resubmission_cost)

                        self.funding_tracker.record_funding_consumption(year, resubmission_cost, agent.get_type())
                        
                        resubmissions_paper_ids.add(metadata['arxiv_id'])
                        
                    else:
                        logger.error(f"[Resubmission] Paper {metadata['arxiv_id']} already resubmitted for year {year}")

        # Get academic funding agencies (For now, only NSF and DARPA)
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

        # Phase 1: Assign research directions
        logger.info(f"Year {year} Phase 1: Assigning research directions...")
        authors = self.ecosystem.get_available_authors()

        authors_with_directions_pending = [a for a in authors if a.project_end_year < year and a.is_active]  # Filter by minimum funding and maturity year

        logger.info(f"[Direction] {len(authors_with_directions_pending)} Authors")

        if year == 1:
            directions = assign_directions_batch(authors_with_directions_pending)

            for agent_id, result_dict in directions.items():
                agent = self.ecosystem.get_agent_by_id(agent_id)
                agent.project_start_year = year  # The year when the agent's latest research paper will be ready for submission
                agent.project_end_year = (year - 1) + result_dict[
                    'direction'].years  # The year when the agent's latest research paper will be ready for submission

                if self.verbose:
                    logger.info(
                        f"[Direction Assignment] Agent {agent.id}, direction {result_dict['direction'].topic} (mature in {result_dict['direction'].years} years (Year {year} -> {agent.project_end_year})")

                agent.newest_direction = result_dict  # Contains all fields like direction, detailed_focus, reason
                # Charge annual cost for the first year of the project (subsequent years charged at start of each year)
                
                # NOTE: Charging of submission in preparation has been moved downwards
                # agent.update_resources(-SIMULATION_CONFIG['conference']['annual_cost'])
                # self.funding_tracker.record_funding_consumption(year, SIMULATION_CONFIG['conference']['annual_cost'], agent.get_type())

        else:
            # Get past papers from paper archive for LLM-based direction selection

            directions = {}
            for agent in tqdm(authors_with_directions_pending, desc="Research Directions"):
                # Use LLM-based individual assignment with past research context
                direction = assign_directions_individual(
                    agent,
                    year=year,
                    paper_tracker=self.paper_tracker,
                    average_funding_level=average_funding_level
                )
                directions[agent.id] = direction
                agent.project_start_year = year
                agent.project_end_year = (year - 1) + direction[
                    'direction'].years  # The year when the research paper will be ready for submission
                agent.newest_direction = direction  # Contains all fields like direction, detailed_focus, reason
                # Charge annual cost for the first year of the project (subsequent years charged at start of each year)
                # agent.update_resources(-SIMULATION_CONFIG['conference']['annual_cost'])
                # self.funding_tracker.record_funding_consumption(year, SIMULATION_CONFIG['conference']['annual_cost'], agent.get_type())

        for author in authors:
            if author.id not in directions:
                assert author.newest_direction is not None, f"Author {author.id} has no newest direction"
                directions[author.id] = author.newest_direction

        active_authors = [a for a in authors if
                          a.project_end_year <= year and a.is_active]  # Filter by maturity year

        # assert len(active_authors) == len(directions), f"Number of active authors {len(active_authors)} does not match number of directions {len(directions)}"

        logger.info(f"[Select Direction] {len(active_authors)} Active Authors")

        year_results['research_assignment'] = {
            'mode': research_direction_selection_mode,
            'num_agents_assigned': len(directions),
            "directions": {
                agent_id: {"direction": direction['direction'].topic, "detailed_focus": direction['detailed_focus'],
                           "reason": direction['reason']} for agent_id, direction in directions.items()}
        }

        # Phase 2: Generate and submit papers
        logger.info(f"Year {year} Phase 2: Submitting papers...")

        paper_submission_prompts: List[str] = []

        # If RAG does not contain documents, load
        if year == 1:
            for agent in active_authors:
                prompt = " ".join(directions[agent.id]['direction'].keywords)
                paper_submission_prompts.append(prompt)

            self.rag.load_documents()
            self.rag.build_knowledge_index()

        else:
            for agent in active_authors:
                if directions[agent.id]['detailed_focus'] is None:
                    assert agent.project_start_year == 1
                    prompt = " ".join(directions[agent.id]['direction'].keywords)

                else:
                    prompt = directions[agent.id]['detailed_focus'] + " \nKeywords: " + " ".join(
                        directions[agent.id]['direction'].keywords)

                paper_submission_prompts.append(prompt)

            # Get submitted papers from paper archive
            accepted_papers, rejected_papers = self.paper_tracker.get_papers_before_year(year=year)

            # NOT year 1
            if self.rag.documents == []:
                # Resuming from checkpoint - preserve document_status
                saved_document_status = self.rag.document_status.copy()
                self.rag.load_documents()
                self.rag.build_knowledge_index()
                # Restore document_status from checkpoint
                self.rag.document_status = saved_document_status
                logger.info(f"Resumed from checkpoint with {len(saved_document_status)} document statuses preserved")

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
                        pass
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

            self.rag.document_status.loc[self.rag.document_status['id'] == arxiv_id, 'status'] = 'submitted'

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

                prompt = agent.get_potential_cited_papers_prompt(
                    paper=submitted_paper,
                    direction=directions[agent.id],
                    candidate_papers=candidate_papers,
                    max_papers=30
                )

                num_retries = 0

                while True:
                    try:
                        llm_response, message_history = self.llm.generate(prompt=prompt,
                            # message_history=message_history_dict[agent.id]
                        )

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

                    prompt = agent.get_potential_cited_papers_prompt(
                        paper=submitted_paper,
                        direction=directions[agent.id],
                        candidate_papers=candidate_papers,
                        max_papers=30
                    )

                    num_retries = 0

                    while True:
                        try:

                            llm_response, message_history = self.llm.generate(prompt=prompt,
                                # message_history=message_history_dict[agent.id]
                            )
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
                    self_citation_prompt = agent.get_self_citation_prompt(
                        paper=submitted_paper,
                        direction=directions[agent.id],
                        own_papers=own_papers
                    )

                    if self_citation_prompt:
                        num_retries = 0
                        while True:
                            try:
                                llm_response, _ = self.llm.generate(prompt=self_citation_prompt)
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
        
        
        # Charge annual cost for all active researchers working on projects
        for agent in self.ecosystem.get_available_authors():
            if agent.is_active:

                if agent.project_end_year >= year:
                    # Agent is working on a project
                    annual_cost = SIMULATION_CONFIG['conference']['annual_cost']
                    agent.update_resources(-annual_cost)
                    self.funding_tracker.record_funding_consumption(year, annual_cost, agent.get_type())

        # Phase 3: Peer review
        logger.info(f"Year {year} Phase 3: Conducting peer reviews...")
        self.conference_system.start_all_review_processes()
        # Map conference ids to dict of papers
        average_scores = {}
        reviews_by_conference = {}  # {conf_id: {paper_id: [reviews]}}
        reviews_by_reviewer = defaultdict(list)  # {reviewer_id: [scores]}

        for conference in tqdm(self.conference_system.conferences, desc="Peer reviews"):
            average_scores[conference.conference_id] = {}
            reviews_by_conference[conference.conference_id] = {}

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
            # 'average_scores': average_scores
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

        # Phase 4: Make acceptance decisions
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
        # self.paper_tracker.update_paper(submitted_paper['id'], citations=citations_of_paper)

        self.rag.mark_papers_as_submitted(accepted_paper_ids, rejected_paper_ids)
        # Save paper archive to disk

        year_results['decisions'] = {
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,

        }

        logger.info(f"  Accepted: {total_accepted}, Rejected: {total_rejected}")

        # Phase 5: Update funding
        logger.info(f"Year {year} Phase 5: Updating funding...")

        # Calculate replenishment needed to maintain constant total funding
        funding_consumed = self.funding_tracker.get_funding_consumption(year)
        logger.info(
            f"Funding consumed this cycle: University: {funding_consumed['university']:.2f}, Industry: {funding_consumed['industry']:.2f}")

        # Update hot topics from recent papers
        # self.industry_funding_system.update_hot_topics(self.all_papers_history, year)

        # Process university researchers - apply for funding
        university_applications = []

        # all papers (accepted, rejected, or long-term projects in preparation)
        active_papers_dataframe = self.paper_tracker.get_papers_dataframe(year)

        print(active_papers_dataframe)

        logger.info(f"[Funding Allocation] Update fundings for industry researchers in year {year}")

        denominator = active_papers_dataframe.loc[
                (active_papers_dataframe['author_type'] == 'industry')
                & (active_papers_dataframe['status'] == 'accept'),
                'maturity'
            ].sum()

        if denominator == 0:
            industry_funding_per_unit = 0
        else:
            industry_funding_per_unit = max(self.funding_tracker.get_funding_allocation_amount(year, 'industry') // denominator, 1)

        # Update funding for industry researchers. Ask university researchers to apply for funding.
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
                success_history = agent.funding_success_history
                selected_program_ids = agent.select_funding_programs(
                    all_funding_programs_dict,
                    success_history,
                    num_submitted_proposals=3  # at most 3 programs to apply to
                )
                if selected_program_ids:
                    application = None
                    num_retries = 0
                    while True:
                        try:

                            application = agent.prepare_funding_application(
                                past_accepted_papers=self.conference_system.authors_to_accepted_papers.get(agent.id,
                                                                                                           []),
                                past_rejected_papers=self.conference_system.authors_to_rejected_papers.get(agent.id,
                                                                                                           []),
                                programs=all_funding_programs_dict,
                                selected_program_ids=selected_program_ids,
                                success_history=success_history
                            )

                            break


                        except Exception as e:
                            logger.warning(f"Error preparing funding application: {e}")
                            traceback.print_exc()
                            num_retries += 1
                            if num_retries >= self.max_retries:
                                raise Exception("Failed to prepare funding application after 3 retries")
                            time.sleep(1)

                    if application:
                        for k in application:
                            application[k]['author'] = agent
                        university_applications.append(application)

            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")

        # Evaluate applications across all funding agencies
        all_funding_winners_dict, all_funding_programs_dict = {}, {}

        logger.debug(f"Calculating weights for funding applications in year {year}...")

        total_weights = []

        winners_list = []
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
            denominator = active_papers_dataframe.loc[winners_list, "maturity"].sum()

        else:
            denominator = len(winners_list)

        academic_funding_per_unit = int(
            self.funding_tracker.get_funding_allocation_amount(year, 'university') / denominator)
        # active_papers_dataframe = active_papers_dataframe.reset_index()

        # Apply funding decisions to university researchers and update success history
        for program_id, winners_for_program in all_funding_winners_dict.items():
            for winner in winners_for_program:
                program = all_funding_programs_dict[program_id]
                agent = self.ecosystem.get_agent_by_id(winner['id'])

                # value = active_papers_dataframe.loc[agent.id, 'maturity']
                # if isinstance(value, pd.Series):
                #     value = value.iloc[0]
                # maturity = int(value)
                
                # if self.weighted_funding_assignment:
                #     funding_to_researcher = academic_funding_per_unit * maturity
                # else:
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
            self.initialize_agents()

            # Select conferences based on agent expertise distribution
            logger.info("\n" + "=" * 60)
            logger.info("CONFERENCE SELECTION")
            logger.info("=" * 60)

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
            'funding_tracker': self.funding_tracker.to_dict(),
            'rag': self.rag.to_dict(),
            'paper_tracker': self.paper_tracker.to_dict(),
            'yearly_results': self.yearly_results,
            'industry_funding_system': self.industry_funding_system.to_dict(),
            'submitted_paper_ids': list(self.submitted_paper_ids)  # Convert set to list
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
        self.agent_tracker = AgentTracker.from_dict(checkpoint_data.get('agent_tracker', {'metrics': {}}))
        self.funding_tracker = FundingTracker.from_dict(checkpoint_data.get('funding_tracker', {}))
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
            rag_config = RAGConfig(use_langchain=self.use_langchain)
            sentence_transformer_model = None
            if not self.use_langchain:
                from sentence_transformers import SentenceTransformer
                sentence_transformer_model = SentenceTransformer(rag_config.embedding_model, device=rag_config.device)
            self.rag = RAG.from_dict(checkpoint_data['rag'], config=rag_config, model=sentence_transformer_model,
                                     debug=self.debug)
            logger.info(f"  Loaded RAG with {len(self.rag.document_status)} document statuses")

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

    llm = GPT(model_name=args.model)

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
        weighted_funding_assignment=args.weighted_funding_assignment

    )

    simulation.run()


if __name__ == '__main__':
    main()
