#!/usr/bin/env python3
"""
Main simulation orchestration script for the Academic Society Ecosystem

This script runs the full multi-agent simulation with:
- Multiple UniversityResearchers and IndustryResearchers
- Multiple conferences targeting different topics
- Configurable research direction assignment (batch or individual mode)
- Yearly cycles with funding management
"""

import argparse
import json
import logging
import os
import random
from typing import List, Dict, Set, Union
from tqdm import tqdm
from collections import Counter

from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.conference import (
    ConferenceSystem,
    create_default_conference_system,
    Conference,
    select_conferences_for_simulation
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
from utopia.agents.funding_agents import (
    FundingAgency,
    FundingProgram,
    IndustryFundingSystem
    
)
from utopia.agents.citation import Citation
from utopia.data.rag import RAG, RAGConfig
from utopia.data.paper_archive import PaperArchive
from utopia.models.models import GPT, LLM
from utopia.utils.citation_extractor import extract_citations_from_content

from utopia.utils.generail_utils import project_setup
from utopia.config import SIMULATION_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.ERROR)

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
        self.current_year = 0
        self.debug = kwargs.get('debug', False)
        self.citations: Dict[str, Set[Citation]] = {} # arxiv_id of paper A -> list of arxiv_ids of papers citing paper A
        self.industry_funding_mode = kwargs.get('industry_funding_mode', 'performance')
        self.always_rerun = kwargs.get('always_rerun', False)
        self.use_langchain = kwargs.get('use_langchain', False)
        # Create output directory
        

        # Initialize components
        self.ecosystem = MultiAgentEcosystem(
            agent_configs={},
        )
        self.conference_system: ConferenceSystem = None
        self.industry_funding_system = IndustryFundingSystem(base_budget=SIMULATION_CONFIG['funding']['industry_base_budget'])
        self.yearly_results: List[Dict] = []
        self.all_papers_history: List[Dict] = []  # Track all papers for hot topic analysis
        self.paper_archive = PaperArchive(output_dir=output_dir)  # Archive for accepted papers
        self.submitted_paper_ids: Set[str] = set()  # Track all submitted papers across years
        
        rag_config = RAGConfig(use_langchain=self.use_langchain)
        
        sentence_transformer_model = None
        if not self.use_langchain:
            from sentence_transformers import SentenceTransformer
            sentence_transformer_model = SentenceTransformer(rag_config.embedding_model, device=rag_config.device)
        self.rag: RAG = RAG(config=rag_config, debug=self.debug, model=sentence_transformer_model)
        
        

    def get_citations_for_paper(self, arxiv_id: str) -> Set[Citation]:
        """Get citations for a paper"""
        return self.citations.get(arxiv_id, set())
    
    def update_citations_for_paper(self, arxiv_id: str, citations: Union[Set[Citation], List[Citation], Citation]):
        """Update citations for a paper"""
        if isinstance(citations, Citation):
            citations = {citations}
        elif isinstance(citations, list):
            citations = set(citations)
        self.citations[arxiv_id].update(citations)
        

    def initialize_agents(self):
        """Initialize the agent population

        Creates multiple researchers mapped to universities and companies.
        """
        logger.info("Initializing agent population...")

        # Create multiple university researchers mapped to same universities
        universities = ["MIT", "Stanford", "Berkeley", "CMU", "GeorgiaTech"]
        universities = universities[:2]
        print(f"TODO: select {len(universities)} universities ONLY")

        counter = 0
        for university in universities:
            # Create 2-3 researchers per university]
            researcher_names = []
            for i in range(2):
                
                expertise = [AVAILABLE_DIRECTIONS[0]] if self.debug else random.sample(AVAILABLE_DIRECTIONS, 3)
                researcher_name = f"university_researcher_{counter}"
                researcher = UniversityResearcher(
                    researcher_name=researcher_name,
                    university_name=university,
                    funding_level=70,
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
                
                
        

        # Create multiple industry researchers mapped to same companies
        companies = ["Google", "Meta", "OpenAI", "Microsoft", "Amazon", "IBM", "Apple", "Tesla"]
        companies = companies[:2]
        print(f"TODO: select {len(companies)} companies ONLY")

        counter = 0
        for company in companies:
            # Create 2-3 researchers per company
            researcher_names = []
            for i in range(2):
                expertise = random.sample(AVAILABLE_DIRECTIONS, 3) if self.debug else [AVAILABLE_DIRECTIONS[0]]
                researcher = IndustryResearcher(
                    researcher_name=f"industry_researcher_{counter}",
                    company_name=company,
                    funding_level=90,
                    expertise=expertise,
                    llm=self.llm,
                    funding_mode=self.industry_funding_mode
                )
                self.ecosystem.add_agent(researcher)
                logger.info(f"Added {researcher.id} from {company} (funding mode: {self.industry_funding_mode})")
                counter += 1
                researcher_names.append(researcher_name)
            researcher_names = set(researcher_names)
            
            for researcher_name in researcher_names:
                researcher = self.ecosystem.get_agent_by_id(researcher_name)
                researcher.add_conflict_of_interest(researcher_names - {researcher_name})

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
            default_funding_rate=0.23, # NSF has a funding rate between [0.2, 0.25]
            llm=self.llm
        ))
        
        self.ecosystem.add_agent(FundingAgency(
            agency_name="DARPA", 
            default_funding_rate=0.1, # DARPA has a funding rate between [0.05, 0.15]
            llm=self.llm
        ))
        
        logger.info(f"Total agents initialized: {len(self.ecosystem.agent_population)}")

        # Initialize expertise for all agents (3 research directions per agent)
        all_agents = [agent for agent in self.ecosystem.agent_population if hasattr(agent, 'expertise_directions')]
        initialize_agent_expertise(all_agents, num_expertise=3)
        logger.info(f"Initialized expertise for {len(all_agents)} agents")

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

        # Determine assignment mode based on year
        research_direction_selection_mode = "batch" if year == 1 else "individual"

        year_results = {
            'year': year,
            'phases': {}
        }
        
        # Get academic funding agencies (For now, only NSF and DARPA)
        funding_agencies = [
            agent for agent_id, agent in self.ecosystem.agent_population.items()
            if isinstance(agent, FundingAgency)
        ]

        # Reset funding programs for new year
        all_funding_programs_dict = {}
        for funding_agent in funding_agencies:
            for program_id, program in funding_agent.funding_programs.items():
                program.reset_for_new_cycle()
                all_funding_programs_dict[program_id] = program

        # Phase 1: Assign research directions
        logger.info(f"Year {year} Phase 1: Assigning research directions...")
        authors = self.ecosystem.get_available_authors()
        active_authors = [a for a in authors if a.resources >= 20]  # Filter by minimum funding

        if year == 1:
            directions = assign_directions_batch(active_authors)
        else:
            directions = assign_directions_individual(active_authors)

        year_results['phases']['research_assignment'] = {
            'mode': research_direction_selection_mode,
            'num_agents_assigned': len(directions)
        }

        # Phase 2: Generate and submit papers
        logger.info(f"Year {year} Phase 2: Submitting papers...")
        

        research_ideas: List[str] = []
        for agent in active_authors:
            idea = " ".join(directions[agent.id].keywords)
            research_ideas.append(idea)
            
        # If RAG does not contain documents, load
        if year == 1:
            self.rag.load_documents()
            self.rag.build_knowledge_index()
            
        else:
            submitted_docs = self.paper_archive.get_papers_before_year(year=year)
            # NOT year 1
            if self.rag.documents == []:
                # Resuming from checkpoint
                self.rag.load_documents()
            
            else:
                # Year 2+: Add previously accepted papers to RAG corpus
                logger.info(f"Adding {len(self.paper_archive.papers_by_id)} previously accepted papers to RAG corpus...")
                self.rag.add_documents_to_corpus(submitted_docs)

        # Batch retrieve relevant documents for all research ideas
        # Use SUBMISSION index to get papers available for submission
        logger.info(f"Batch retrieving relevant documents for {len(research_ideas)} research ideas from SUBMISSION index...")
        retrieval_results = self.rag.batch_retrieve(
            research_ideas,
            retrieval_type="submission"  # Only retrieve papers not yet submitted
        )

        topk_indices = retrieval_results['topk_indices']

        assert topk_indices.shape[0] == len(active_authors)
        
        submitted_papers = {}
        submitted_papers_list = []
        
        # Get previous papers preview for Year 2+
        previous_papers_preview = None
        if year >= 2:
            previous_papers_preview = self.paper_archive.get_papers_preview(year=year, max_papers=20)

        for agent_idx, agent in enumerate(tqdm(active_authors, desc="Submitting papers")):
            candidate_paper_indices = topk_indices[agent_idx]
            # Use submission_documents since we retrieved from submission index
            candidate_papers = [self.rag.documents[idx] for idx in candidate_paper_indices]
            prompt, response_format = agent.get_paper_submission_prompt(
                candidate_papers=candidate_papers,
                conferences=self.conference_system.conferences,
                previous_papers=previous_papers_preview
            )
            submission_metadata = json.loads(self.llm.generate(prompt=prompt, response_format=response_format))
            arxiv_id = submission_metadata['arxiv_id']
            assert arxiv_id in [p.metadata['id'] for p in candidate_papers]
            assert submission_metadata['conference'] in self.conference_system.conference_map
            paper_doc = self.rag.id2docs[arxiv_id]
            paper_dict = {
                'id': arxiv_id,
                'author_id': agent.id,
                'title': paper_doc.metadata['title'],
                'content': paper_doc.page_content,
                'topics': paper_doc.metadata.get('categories', '').split(),
                'conference': submission_metadata['conference']
            }
            submitted_papers_list.append(paper_dict)


        submitted_papers_dict = {paper['id']: paper for paper in submitted_papers_list}

        for paper_dict in submitted_papers_list:

            conf: Conference = self.conference_system.conference_map[paper_dict['conference']]
            if conf.submit_paper(paper_dict):
                submitted_papers[paper_dict['author_id']] = paper_dict
                self.submitted_paper_ids.add(paper_dict['id'])
            else:
                raise ValueError(f"Paper {arxiv_id} submission to conference {paper_dict['conference']} failed")

        # Mark papers as submitted - removes them from submission index but keeps in citation index
        submitted_paper_ids_list = list(self.submitted_paper_ids)

        year_results['phases']['paper_submission'] = {
            'num_papers_submitted': len(submitted_papers),
            'papers': submitted_papers
        }

        # Phase 3: Peer review



        logger.info(f"Year {year} Phase 3: Conducting peer reviews...")
        self.conference_system.start_all_review_processes()
        cache_path = os.path.join(self.output_dir, f"year_{year}_conference_system.json")
        if self.debug and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                conference_data = json.load(f)
                self.conference_system = ConferenceSystem.from_dict(conference_data)

        else:
            for conference in tqdm(self.conference_system.conferences, desc="Peer reviews"):
                for paper in conference.submitted_papers:
                    # Assign reviewers (excluding paper author)
                    author = self.ecosystem.get_agent_by_id(paper['author_id'])
                    reviewers = self.ecosystem.get_available_reviewers(
                        exclude_agent_ids=[paper['author_id']] + list(author.conflict_of_interest)
                    )

                    # Select 3 random reviewers
                    selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                    for reviewer in selected_reviewers:
                        review = reviewer.review_paper(paper)
                        conference.add_review(paper['id'], review)

            with open(cache_path, "w") as f:
                json.dump(self.conference_system.to_dict(), f, indent=2)

        del cache_path
        year_results['phases']['peer_review'] = {
            'reviews_conducted': sum(len(c.reviews) for c in self.conference_system.conferences)
        }

        # Process received reviews for authors
        logger.info(f"Year {year} Phase 3: Processing author experiences from received reviews...")
        for conference in self.conference_system.conferences:
            for paper in conference.submitted_papers:
                paper_reviews = conference.reviews.get(paper['id'], [])
                if paper_reviews:
                    # Find the author agent
                    author = self.ecosystem.get_agent_by_id(paper['author_id'])
                    
                    author.process_received_reviews(paper_reviews, paper_id=paper['id'])

        

        # Phase 4: Make acceptance decisions
        logger.info(f"Year {year} Phase 4: Making acceptance decisions...")
        decisions = {}
        authors_to_accepted_papers = {}
        authors_to_rejected_papers = {}
        for conf in self.conference_system.conferences:
            decision_one_conf = conf.make_acceptance_decisions()
            decisions[conf.conference_id] = decision_one_conf
            for paper in decision_one_conf['accept']:
                if paper['author_id'] not in authors_to_accepted_papers:
                    authors_to_accepted_papers[paper['author_id']] = []
                authors_to_accepted_papers[paper['author_id']].append(paper)
            for paper in decision_one_conf['reject']:
                if paper['author_id'] not in authors_to_rejected_papers:
                    authors_to_rejected_papers[paper['author_id']] = []
                authors_to_rejected_papers[paper['author_id']].append(paper)

        total_accepted = sum(len(d['accept']) for d in decisions.values())
        total_rejected = sum(len(d['reject']) for d in decisions.values())

        # Track accepted papers for hot topic analysis and archive them
        accepted_paper_ids, rejected_paper_ids = [], []
        for conf_decisions in decisions.values():
            for decision in ['accept', 'reject']:
                for paper in conf_decisions[decision]:
                    self.all_papers_history.append({
                        'id': paper['id'],
                        'year': year,
                        'keywords': paper.get('topics', [])
                    })
                    
                    if decision == 'accept':
                        accepted_paper_ids.append(paper['id'])
                    else:
                        rejected_paper_ids.append(paper['id'])

                    # Archive accepted paper for future reference
                    self.paper_archive.add_paper(paper, year, decision)
                    
        self.rag.mark_papers_as_submitted(accepted_paper_ids, rejected_paper_ids)
        # Extract citations from accepted papers (Year 2+)
        if year >= 2:
            raise NotImplementedError("Citations extraction not implemented yet")
            logger.info(f"Extracting citations from {total_accepted} accepted papers...")
            available_papers = [
                {
                    'id': p.id,
                    'title': p.title,
                    'abstract': p.abstract
                }
                for p in self.paper_archive.get_papers_before_year(year)
            ]

            for conf_decisions in decisions.values():
                for paper in conf_decisions['accepted']:
                    # Extract citations using LLM
                    citations = extract_citations_from_content(
                        paper_id=paper['id'],
                        paper_content=paper['content'],
                        available_papers=available_papers,
                        llm=self.llm,
                        model=self.llm.model_name
                    )

                    # Update citation tracking
                    if paper['id'] not in self.citations:
                        self.citations[paper['id']] = set()
                    self.citations[paper['id']].update(citations)

                    # Increment citation counts in archive
                    for citation in citations:
                        self.paper_archive.increment_citation_count(citation.cited_paper_id)

            logger.info(f"  Extracted {sum(len(cites) for cites in self.citations.values())} total citations")

        # Save paper archive to disk

        year_results['phases']['decisions'] = {
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'conference_decisions': decisions
        }

        logger.info(f"  Accepted: {total_accepted}, Rejected: {total_rejected}")

        # Phase 5: Update funding
        logger.info(f"Year {year} Phase 5: Updating funding...")
        funding_updates = {}

        # Update hot topics from recent papers
        self.industry_funding_system.update_hot_topics(self.all_papers_history, year)

        

        # Process university researchers - apply for funding
        university_applications = []
        
        # Update funding for industry researchers. Ask university researchers to apply for funding.
        for agent in self.ecosystem.get_available_authors():
            if isinstance(agent, IndustryResearcher):
                # Update industry budget based on performance
                accepted = len(authors_to_accepted_papers.get(agent.id, []))
                rejected = len(authors_to_rejected_papers.get(agent.id, []))
                
                print(f"TODO: Track real citations for {agent.id}.")
                citations = self.citations[agent.id] = [0] * accepted  # TODO: track real citations

                new_budget = self.industry_funding_system.update_budget(
                    accepted, rejected, citations
                )
                agent.update_resources(delta=new_budget)

            elif isinstance(agent, UniversityResearcher):
                success_history = agent.funding_success_history
                selected_program_ids = agent.select_funding_programs(
                    all_funding_programs_dict,
                    success_history,
                    num_submitted_proposals=3 # at most 3 programs to apply to
                )
                if selected_program_ids:
                    application = agent.prepare_funding_application(
                        past_accepted_papers=authors_to_accepted_papers.get(agent.id, []),
                        past_rejected_papers=authors_to_rejected_papers.get(agent.id, []),
                        programs=all_funding_programs_dict,
                        selected_program_ids=selected_program_ids,
                        success_history=success_history
                    )
                    if application:
                        application['author'] = agent
                        university_applications.append(application)
                        
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")

            

        # Evaluate applications across all funding agencies
        all_funding_winners_dict, all_funding_programs_dict = {}, {}
        for funding_agent in funding_agencies:
            winners_for_program = funding_agent.evaluate_funding_applications(
                university_applications,
                submitted_papers_dict=submitted_papers_dict
            )
            all_funding_winners_dict.update(winners_for_program)
            all_funding_programs_dict.update(funding_agent.funding_programs)

        # Apply funding decisions to university researchers and update success history
        for program_id, winners_for_program in all_funding_winners_dict.items():
            for winner in winners_for_program:
                program = all_funding_programs_dict[program_id]
                agent = self.ecosystem.get_agent_by_id(winner['id'])
                agent.update_resources(delta=program.budget_per_winner)
                if not program_id in agent.funding_success_history:
                    agent.funding_success_history[program_id] = []
                agent.funding_success_history[program_id].append({
                    'year': year,
                })
                
        for agent_id, agent in self.ecosystem.agent_population.items():
            if agent.resources <= 10 and hasattr(agent, 'can_author') and agent.can_author:
                print(f"Removing agent {agent.id} due to insufficient funding")
                self.ecosystem.remove_agent(agent.id)

        # Get conference statistics
        year_results['conference_statistics'] = self.conference_system.get_all_statistics()

        # Save year results
        self._save_year_results(year, year_results)

        # Save checkpoint after each year in debug mode
        if self.debug:
            self._save_checkpoint(year)

        # Reset conferences for next year
        self.conference_system.reset_all_for_next_year(year + 1)

        return year_results

    def run(self):
        """Run the complete simulation"""
        logger.info("Starting Academic Ecosystem Simulation")
        logger.info(f"Configuration: {self.num_years} years")

        start_year = 1
        checkpoint_loaded = False
        

        # Try to load from checkpoint unless always_rerun is specified
        if not self.always_rerun:
            # Try to find the most recent checkpoint
            for year in range(self.num_years, 0, -1):
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")
                if os.path.exists(checkpoint_path):
                    logger.info(f"Found checkpoint at year {year}, attempting to load...")
                    if self._load_checkpoint(year):
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
                agent for agent_id, agent in self.ecosystem.agent_population.items() if isinstance(agent, (UniversityResearcher, IndustryResearcher)) and agent.expertise
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

            logger.info("")

        # Initialize RAG (always reconstruct fresh from data source)
        logger.info("\nInitializing RAG system...")
        if checkpoint_loaded:
            logger.info(f"RAG will filter out {len(self.submitted_paper_ids)} already submitted papers")

        # Run simulation for each year
        for year in range(start_year, self.num_years + 1):
            year_results = self.run_one_round(year)
            self.yearly_results.append(year_results)

        # Generate final report
        self._generate_final_report()

        logger.info("\nSimulation completed!")
        logger.info(f"Results saved to: {self.output_dir}")

    def _save_checkpoint(self, year: int):
        """Save complete simulation state as checkpoint

        Args:
            year: Year number
        """
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_year_{year}.json")

        # Convert citations from Dict[str, Set[Citation]] to JSON-serializable format

        # Serialize everything to JSON-compatible format
        checkpoint_data = {
            'year': year,
            'ecosystem_data': self.ecosystem.to_dict(),
            'conference_system': self.conference_system.to_dict(),
            'citations': self.citations,
            'rag': self.rag.to_dict(),
            'all_papers_history': self.all_papers_history,
            'paper_archive': self.paper_archive.to_dict(),
            'yearly_results': self.yearly_results,
            'industry_funding_system': self.industry_funding_system.to_dict(),
            'submitted_paper_ids': list(self.submitted_paper_ids)  # Convert set to list
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Checkpoint saved for year {year}: {checkpoint_path}")

    def _load_checkpoint(self, year: int):
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

        try:
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

            # Reconstruct citations from JSON format
            self.citations = checkpoint_data['citations']

            self.all_papers_history = checkpoint_data['all_papers_history']

            # Reconstruct paper archive
            self.paper_archive = PaperArchive.from_dict(checkpoint_data['paper_archive'])

            self.yearly_results = checkpoint_data['yearly_results']

            # Reconstruct industry funding system
            self.industry_funding_system = IndustryFundingSystem.from_dict(checkpoint_data['industry_funding_system'])

            # Convert submitted_paper_ids from list back to set
            self.submitted_paper_ids = set(checkpoint_data['submitted_paper_ids'])

            # RAG will be reconstructed in run() method
            logger.info(f"Checkpoint loaded successfully from year {checkpoint_data['year']}")
            logger.info(f"  Loaded {len(self.ecosystem.agent_population)} agents")
            logger.info(f"  Loaded {len(self.submitted_paper_ids)} submitted paper IDs")
            return True
        
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _save_year_results(self, year: int, results: Dict):
        """Save results for a specific year

        Args:
            year: Year number
            results: Results dictionary
        """
        output_file = os.path.join(self.output_dir, f"year_{year}_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def _generate_final_report(self):
        """Generate final simulation report"""
        report_file = os.path.join(self.output_dir, "simulation_report.json")

        # Get paper archive statistics
        archive_stats = self.paper_archive.get_statistics()

        # Calculate citation statistics
        total_citations = sum(len(cites) for cites in self.citations.values())
        papers_with_citations = sum(1 for cites in self.citations.values() if len(cites) > 0)

        report = {
            'configuration': {
                'num_years': self.num_years,
                'num_agents': len(self.ecosystem.agent_population)
            },
            'yearly_results': self.yearly_results,
            'summary': {
                'total_papers_submitted': sum(
                    r['phases']['paper_submission']['num_papers_submitted']
                    for r in self.yearly_results
                ),
                'total_papers_accepted': sum(
                    r['phases']['decisions']['total_accepted']
                    for r in self.yearly_results
                ),
                'total_citations': total_citations,
                'papers_with_citations': papers_with_citations,
                'archive_statistics': archive_stats
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Final report saved to {report_file}")
        logger.info(f"Paper archive saved with {archive_stats['total_papers']} papers")
        logger.info(f"Total citations tracked: {total_citations}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Academic Society Ecosystem Simulation"
    )
    parser.add_argument(
        '--years',
        type=int,
        default=5,
        help='Number of years to simulate (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for results (default: ./simulation_output)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='Model to use for LLM (default: gpt-4o)'
    )

    parser.add_argument(
        '--industry_funding_mode',
        type=str,
        default='performance',
        choices=['performance', 'consistent'],
        help='Funding mode for industry researchers: "performance" (based on papers) or "consistent" (fixed allocation) (default: performance)'
    )

    parser.add_argument(
        '--always_rerun',
        action='store_true',
        default=False,
        help='Ignore all checkpoints and start simulation from scratch (default: resume from latest checkpoint if available)'
    )
    parser.add_argument(
        '--use_langchain',
        action='store_true',
        default=False,
        help='Use LangChain for RAG (by default, we use SentenceTransformer)'
    )

    args = parser.parse_args()

    project_setup()
    
    os.makedirs(args.output_dir, exist_ok=True)

    llm = GPT(model_name=args.model)

    # Create and run simulation
    simulation = Simulation(
        llm=llm,
        num_years=args.years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain
    )

    simulation.run()


if __name__ == '__main__':
    main()
