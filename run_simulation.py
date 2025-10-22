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
from typing import List, Dict
import pickle
from tqdm import tqdm


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
from utopia.agents.specialized_agents import (
    UniversityResearcher,
    IndustryResearcher,
    FundingAgent,
    FundingProgram
)
from utopia.data.rag import RAG, RAGConfig
from utopia.models.models import GPT, LLM

from utopia.utils.generail_utils import project_setup

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
            output_dir: str = "./simulation_output",
            debug: bool = False,
            industry_funding_mode: str = "performance"
    ):
        """Initialize the simulation

        Args:
            num_years: Number of years to simulate
            output_dir: Directory to save simulation results
            debug: Enable debug mode
            industry_funding_mode: Funding mode for industry researchers - "performance" or "consistent"
        """
        self.llm: LLM = llm
        self.num_years = num_years
        self.output_dir = output_dir
        self.current_year = 0
        self.debug = debug
        self.industry_funding_mode = industry_funding_mode
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.ecosystem = MultiAgentEcosystem(
            agent_configs={},
            initial_paper_corpus=[]
        )
        self.conference_system = None  # Will be initialized after agents are created
        # Storage for simulation data
        self.yearly_results: List[Dict] = []
        
        self.conference_system: ConferenceSystem = None

    def initialize_agents(self):
        """Initialize the agent population

        Creates multiple researchers mapped to universities and companies.
        """
        logger.info("Initializing agent population...")

        # Create multiple university researchers mapped to same universities
        universities = ["MIT", "Stanford", "Berkeley", "CMU", "GeorgiaTech"]

        counter = 0
        for university in universities:
            # Create 2-3 researchers per university
            for i in range(2):
                researcher = UniversityResearcher(
                    researcher_name=f"university_researcher_{counter}",
                    university_name=university,
                    funding_level=70,
                    expertise=random.sample(AVAILABLE_DIRECTIONS, 3),
                    llm=self.llm
                )
                self.ecosystem.add_agent(researcher)
                counter += 1
                logger.info(f"Added {researcher.id} from {university}")

        # Create multiple industry researchers mapped to same companies
        companies = ["Google", "Meta", "OpenAI", "Microsoft", "Amazon", "IBM", "Apple", "Tesla"]

        counter = 0
        for company in companies:
            # Create 2-3 researchers per company
            for i in range(2):
                researcher = IndustryResearcher(
                    researcher_name=f"industry_researcher_{counter}",
                    company_name=company,
                    funding_level=90,
                    expertise=random.sample(AVAILABLE_DIRECTIONS, 3),
                    llm=self.llm,
                    funding_mode=self.industry_funding_mode
                )
                self.ecosystem.add_agent(researcher)
                logger.info(f"Added {researcher.id} from {company} (funding mode: {self.industry_funding_mode})")
                counter += 1

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
        funding_agencies = ["NSF", "DARPA"]
        for agency in funding_agencies:
            funding_agent = FundingAgent(
                agency_name=agency,
                funding_level=100,

            )
            self.ecosystem.add_agent(funding_agent)
            logger.info(f"Added funding agent: {agency}")

        logger.info(f"Total agents initialized: {len(self.ecosystem.agent_population)}")

        # Initialize expertise for all agents (3 research directions per agent)
        all_agents = [agent for agent in self.ecosystem.agent_population if hasattr(agent, 'expertise_directions')]
        initialize_agent_expertise(all_agents, num_expertise=3)
        logger.info(f"Initialized expertise for {len(all_agents)} agents")

        # Keep track of the submitted papers so that they are not submitted again
        self.submitted_papers = set()

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
        submitted_papers = {}

        research_ideas: List[str] = []
        for agent in active_authors:
            idea = " ".join(directions[agent.id].keywords)
            research_ideas.append(idea)

        # If RAG does not contain documents, load 
        if year == 1:
            self.rag.load_documents()
        else:
            raise NotImplementedError("Update documents")

        self.rag.build_knowledge_index()

        # Batch retrieve relevant documents for all research ideas
        logger.info(f"Batch retrieving relevant documents for {len(research_ideas)} research ideas...")
        retrieval_results = self.rag.batch_retrieve(research_ideas, use_reranker=self.rag.config.use_reranker)

        topk_indices = retrieval_results['topk_indices']

        assert topk_indices.shape[0] == len(active_authors)
        
        cache_path = os.path.join(self.output_dir, f"year_{year}_submitted_papers.pkl")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        submitted_papers, submitted_paper_arxiv_ids = {}, set()
        submitted_papers_list = []
        if self.debug and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                submitted_papers_list = pickle.load(f)
        else:
            
            
            for agent_idx, agent in enumerate(tqdm(active_authors, desc="Submitting papers")):
                candidate_paper_indices = topk_indices[agent_idx]
                candidate_papers = [self.rag.documents[idx] for idx in candidate_paper_indices if self.rag.documents[idx].metadata['id'] not in submitted_paper_arxiv_ids]
                prompt, response_format = agent.get_paper_submission_prompt(candidate_papers, self.conference_system.conferences)
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
                

            with open(cache_path, "wb") as f:
                pickle.dump(submitted_papers_list, f)
            
        for paper_dict in submitted_papers_list:
            
            conf: Conference = self.conference_system.conference_map[paper_dict['conference']]
            if conf.submit_paper(paper_dict):
                submitted_papers[paper_dict['author_id']] = paper_dict
                submitted_paper_arxiv_ids.add(paper_dict['id'])
            else:
                raise ValueError(f"Paper {arxiv_id} submission to conference {paper_dict['conference']} failed")
            
        year_results['phases']['paper_submission'] = {
            'num_papers_submitted': len(submitted_papers),
            'papers': submitted_papers
        }

        # Phase 3: Peer review



        logger.info(f"Year {year} Phase 3: Conducting peer reviews...")
        self.conference_system.start_all_review_processes()
        
        if self.debug:
            with open(os.path.join(self.output_dir, f"conference_system.pkl"), "rb") as f:
                self.conference_system = pickle.load(f)
            

        else:
            for conference in tqdm(self.conference_system.conferences, desc="Peer reviews"):
                for paper in conference.submitted_papers:
                    # Assign reviewers (excluding paper author)
                    reviewers = self.ecosystem.get_available_reviewers(
                        exclude_agent_ids=[paper['author_id']]
                    )

                    # Select 3 random reviewers
                    selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                    for reviewer in selected_reviewers:
                        review = reviewer.review_paper(paper)
                        conference.add_review(paper['id'], review)
            
            with open(os.path.join(self.output_dir, f"conference_system.pkl"), "wb") as f:
                pickle.dump(self.conference_system, f)

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
                    author = next((a for a in self.ecosystem.agent_population if a.id == paper['author_id']), None)
                    if author and hasattr(author, 'process_received_reviews'):
                        author.process_received_reviews(paper_reviews, paper_id=paper['id'])

        

        # Phase 4: Make acceptance decisions
        logger.info(f"Year {year} Phase 4: Making acceptance decisions...")
        decisions = {}
        for conf in self.conference_system.conferences:
            decision_one_conf = conf.make_acceptance_decisions()
            decisions[conf.conference_id] = decision_one_conf

        total_accepted = sum(len(d['accepted']) for d in decisions.values())
        total_rejected = sum(len(d['rejected']) for d in decisions.values())

        year_results['phases']['decisions'] = {
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'conference_decisions': decisions
        }

        logger.info(f"  Accepted: {total_accepted}, Rejected: {total_rejected}")

        # Phase 5: Update funding
        logger.info(f"Year {year} Phase 5: Updating agent funding...")
        funding_updates = {}

        # Get funding agents
        funding_agents = [
            agent for agent in self.ecosystem.agent_population
            if isinstance(agent, FundingAgent)
        ]

        # Reset funding programs for new year
        for funding_agent in funding_agents:
            funding_agent.reset_programs_for_new_year()

        # Process university researchers - apply for funding
        university_applications = []
        for agent in self.ecosystem.get_available_authors():
            if isinstance(agent, UniversityResearcher):
                # Count accepted papers
                agent_accepted = sum(
                    1 for conf_decisions in decisions.values()
                    for paper in conf_decisions['accepted']
                    if paper['author_id'] == agent.id
                )

                # Prepare funding application
                research_direction = directions.get(agent.id)
                if research_direction:
                    proposal = f"Research in {', '.join(research_direction.keywords)}"
                else:
                    proposal = f"Research in {', '.join(agent.expertise if agent.expertise else ['general AI'])}"

                application = agent.prepare_funding_application(
                    past_performance=agent_accepted,
                    current_research_direction=proposal
                )
                university_applications.append(application)

        # Evaluate applications across all funding agencies
        all_funding_decisions = []
        for funding_agent in funding_agents:
            decisions_from_agent = funding_agent.evaluate_funding_applications(
                university_applications,
                self.llm
            )
            all_funding_decisions.extend(decisions_from_agent)

        # Apply funding decisions to university researchers
        for decision in all_funding_decisions:
            if decision['decision'] == 'fund':
                agent = next(
                    (a for a in self.ecosystem.agent_population if a.id == decision['applicant_id']),
                    None
                )
                if agent:
                    funding_amount = decision['funding_amount']
                    agent.resources = min(100, agent.resources + funding_amount)
                    funding_updates[agent.id] = {
                        'type': 'grant',
                        'amount': f"+{funding_amount}",
                        'source': decision.get('program', 'Unknown')
                    }

        # Process industry researchers - apply funding mode
        for agent in self.ecosystem.get_available_authors():
            if isinstance(agent, IndustryResearcher):
                # Count accepted papers
                agent_accepted = sum(
                    1 for conf_decisions in decisions.values()
                    for paper in conf_decisions['accepted']
                    if paper['author_id'] == agent.id
                )

                # Get funding allocation based on mode
                funding_change = agent.get_funding_allocation(agent_accepted)
                agent.resources = max(0, min(100, agent.resources + funding_change))

                funding_updates[agent.id] = {
                    'type': agent.funding_mode,
                    'amount': f"+{funding_change}" if funding_change > 0 else str(funding_change),
                    'papers': agent_accepted
                }

        year_results['phases']['funding_update'] = funding_updates
        year_results['phases']['funding_decisions'] = all_funding_decisions

        # Eliminate agents with zero funding
        eliminated_agents = [a.id for a in self.ecosystem.agent_population if
                             a.resources <= 0 and hasattr(a, 'can_author') and a.can_author]
        if eliminated_agents:
            logger.info(f"  Eliminated {len(eliminated_agents)} agents due to funding depletion")
            year_results['eliminated_agents'] = eliminated_agents

        # Get conference statistics
        year_results['conference_statistics'] = self.conference_system.get_all_statistics()

        # Save year results
        self._save_year_results(year, year_results)

        # Reset conferences for next year
        self.conference_system.reset_all_for_next_year(year + 1)

        return year_results

    def run(self):
        """Run the complete simulation"""
        logger.info("Starting Academic Ecosystem Simulation")
        logger.info(f"Configuration: {self.num_years} years")

        # Initialize agents
        self.initialize_agents()

        self.rag = RAG(config=RAGConfig(), debug=self.debug)

        # Select conferences based on agent expertise distribution
        logger.info("\n" + "=" * 60)
        logger.info("CONFERENCE SELECTION")
        logger.info("=" * 60)

        # Get all agents with research capability
        research_agents = [
            agent for agent in self.ecosystem.agent_population
            if hasattr(agent, 'expertise') and agent.expertise
        ]

        # Select conferences proportionally to agent expertise
        selected_conferences = select_conferences_for_simulation(
            agents=research_agents,
            num_conferences=5
        )
        
       

        # Create conference system with selected conferences
        self.conference_system = ConferenceSystem(conferences=selected_conferences)

        # Log conference selection results
        logger.info(f"\nSelected {len(selected_conferences)} conferences:")

        from collections import Counter
        category_distribution = Counter(conf.category for conf in selected_conferences)

        for category, count in sorted(category_distribution.items()):
            logger.info(f"  {category}: {count} conferences")
            conferences_in_category = [
                conf.name for conf in selected_conferences if conf.category == category
            ]
            for conf_name in conferences_in_category:
                logger.info(f"    - {conf_name}")

        logger.info("")

        # Run simulation for each year
        for year in range(1, self.num_years + 1):
            year_results = self.run_one_round(year)
            self.yearly_results.append(year_results)

        # Generate final report
        self._generate_final_report()

        logger.info("\nSimulation completed!")
        logger.info(f"Results saved to: {self.output_dir}")

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
                )
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)


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
        default='./simulation_output',
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
        '--industry-funding-mode',
        type=str,
        default='performance',
        choices=['performance', 'consistent'],
        help='Funding mode for industry researchers: "performance" (based on papers) or "consistent" (fixed allocation) (default: performance)'
    )

    args = parser.parse_args()

    project_setup()

    llm = GPT(model_name=args.model)

    # Create and run simulation
    simulation = Simulation(
        llm=llm,
        num_years=args.years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode
    )

    simulation.run()


if __name__ == '__main__':
    main()
