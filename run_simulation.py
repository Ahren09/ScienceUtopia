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
import os
from typing import List, Dict
import logging

from utopia.agents.specialized_agents import (
    UniversityResearcher,
    IndustryResearcher,
    FreelancerAgent,
    FundingAgent
)
from utopia.agents.conference import (
    ConferenceSystem,
    create_default_conference_system,
    Conference
)
from utopia.agents.research_direction import (
    ResearchDirectionAssigner,
    create_batch_assigner,
    create_individual_assigner
)
from utopia.agents.base_agent import MultiAgentEcosystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Simulation:
    """Main simulation orchestrator for the academic ecosystem"""

    def __init__(
        self,
        num_years: int = 5,
        assignment_mode: str = "batch",
        output_dir: str = "./simulation_output"
    ):
        """Initialize the simulation

        Args:
            num_years: Number of years to simulate
            assignment_mode: Mode for research direction assignment - "batch" or "individual"
            output_dir: Directory to save simulation results
        """
        if assignment_mode not in ["batch", "individual"]:
            raise ValueError(f"assignment_mode must be 'batch' or 'individual', got: {assignment_mode}")

        self.num_years = num_years
        self.assignment_mode = assignment_mode
        self.output_dir = output_dir
        self.current_year = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.ecosystem = MultiAgentEcosystem(
            agent_configs={},
            initial_paper_corpus=[]
        )
        self.conference_system = create_default_conference_system()
        
        research_directions

        # Initialize research direction assigner based on mode
        if assignment_mode == "batch":
            self.direction_assigner = create_batch_assigner(research_directions)
        else:
            self.direction_assigner = create_individual_assigner(research_directions)

        # Storage for simulation data
        self.yearly_results: List[Dict] = []

    def initialize_agents(self):
        """Initialize the agent population

        Creates multiple researchers mapped to universities and companies.
        """
        logger.info("Initializing agent population...")

        # Create multiple university researchers mapped to same universities
        universities = ["MIT", "Stanford", "Berkeley", "CMU", "Oxford"]
        
        counter = 0
        for university in universities:
            # Create 2-3 researchers per university
            for i in range(2):
                researcher = UniversityResearcher(
                    researcher_name=f"university_researcher_{counter}",
                    university_name=university,
                    funding_level=70
                )
                self.ecosystem.add_agent(researcher)
                counter += 1
                logger.info(f"Added {researcher.researcher_name} from {university}")

        # Create multiple industry researchers mapped to same companies
        companies = ["Google", "Meta", "OpenAI", "Microsoft", "DeepMind"]
        
        counter = 0
        for company in companies:
            # Create 2-3 researchers per company
            for i in range(2):
                researcher = IndustryResearcher(
                    researcher_name=f"industry_researcher_{counter}",
                    company_name=company,
                    funding_level=90
                )
                self.ecosystem.add_agent(researcher)
                logger.info(f"Added {researcher.researcher_name} from {company}")
                counter += 1

        """
        # Add some freelance researchers
        for i in range(3):
            freelancer = FreelancerAgent(
                researcher_name=f"freelancer_{counter}",
                funding_level=5
            )
            self.ecosystem.add_agent(freelancer)
            logger.info(f"Added {freelancer.researcher_name}")
        """

        # Add funding agents
        funding_agencies = ["NSF", "NIH", "DARPA"]
        for agency in funding_agencies:
            funding_agent = FundingAgent(
                agency_name=agency,
                funding_level=100
            )
            self.ecosystem.add_agent(funding_agent)
            logger.info(f"Added funding agent: {agency}")

        logger.info(f"Total agents initialized: {len(self.ecosystem.agent_population)}")

    def run_year(self, year: int) -> Dict:
        """Run simulation for one year

        Args:
            year: Current year number

        Returns:
            Dictionary with year results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"YEAR {year}")
        logger.info(f"{'='*60}\n")

        year_results = {
            'year': year,
            'phases': {}
        }

        # Phase 1: Assign research directions
        logger.info("Phase 1: Assigning research directions...")
        authors = self.ecosystem.get_available_authors()
        active_authors = [a for a in authors if a.resources >= 20]  # Filter by minimum funding
        
        research_directions = self.direction_assigner.available_directions

        if year == 1:
            self.direction_assigner = create_batch_assigner(research_directions)
        else:
            self.direction_assigner = create_individual_assigner(research_directions)
            
        assignments = self.direction_assigner.assign_directions(active_authors, year)
        year_results['phases']['research_assignment'] = {
            'mode': self.assignment_mode,
            'num_agents_assigned': len(assignments),
            'assignment_stats': self.direction_assigner.get_assignment_statistics()
        }

        # Phase 2: Generate and submit papers
        logger.info("Phase 2: Generating and submitting papers...")
        submitted_papers = []

        for agent in active_authors:
            if agent.id in assignments:
                direction = assignments[agent.id]

                try:
                    # Generate paper based on assigned direction
                    logger.info(f"  {agent.id} researching {direction.topic}...")

                    # Create paper metadata (simplified for demonstration)
                    paper = {
                        'id': f"paper_{year}_{agent.id}_{int(__import__('time').time()*1000) % 10000}",
                        'title': f"Research on {direction.topic}",
                        'topics': [direction.topic] + direction.focus_areas,
                        'author_id': agent.id,
                        'author_name': getattr(agent, 'researcher_name', agent.id),
                        'affiliation': getattr(agent, 'university_name', None) or getattr(agent, 'company_name', 'Independent'),
                        'year': year,
                        'direction': direction
                    }

                    # Submit to appropriate conference
                    conference = self.conference_system.submit_paper_to_best_match(paper)

                    if conference:
                        submitted_papers.append(paper)
                        logger.info(f"    -> Submitted to {conference.name}")
                    else:
                        logger.info(f"    -> No matching conference found")

                except Exception as e:
                    logger.error(f"  Error with {agent.id}: {e}")

        year_results['phases']['paper_submission'] = {
            'num_papers_submitted': len(submitted_papers),
            'papers': submitted_papers
        }

        # Phase 3: Peer review
        logger.info("Phase 3: Conducting peer reviews...")
        self.conference_system.start_all_review_processes()

        for conference in self.conference_system.conferences:
            for paper in conference.submitted_papers:
                # Assign reviewers (excluding paper author)
                reviewers = self.ecosystem.get_available_reviewers(
                    exclude_agent_ids=[paper['author_id']]
                )

                # Select 3 random reviewers
                import random
                selected_reviewers = random.sample(reviewers, min(3, len(reviewers)))

                for reviewer in selected_reviewers:
                    try:
                        # Generate review (simplified for demonstration)
                        review = {
                            'reviewer_id': reviewer.id,
                            'paper_id': paper['id'],
                            'overall_score': random.uniform(4.0, 9.0),  # Placeholder
                            'strengths': ['Interesting approach', 'Well-motivated'],
                            'weaknesses': ['Limited evaluation', 'Could improve clarity']
                        }

                        conference.add_review(paper['id'], review)

                    except Exception as e:
                        logger.error(f"  Review error: {e}")

        year_results['phases']['peer_review'] = {
            'reviews_conducted': sum(len(c.reviews) for c in self.conference_system.conferences)
        }

        # Phase 4: Make acceptance decisions
        logger.info("Phase 4: Making acceptance decisions...")
        decisions = self.conference_system.make_all_decisions()

        total_accepted = sum(len(d['accepted']) for d in decisions.values())
        total_rejected = sum(len(d['rejected']) for d in decisions.values())

        year_results['phases']['decisions'] = {
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'conference_decisions': decisions
        }

        logger.info(f"  Accepted: {total_accepted}, Rejected: {total_rejected}")

        # Phase 5: Update funding
        logger.info("Phase 5: Updating agent funding...")
        funding_updates = {}

        for agent in self.ecosystem.get_available_authors():
            # Check if agent had papers accepted
            agent_accepted = sum(
                1 for conf_decisions in decisions.values()
                for paper in conf_decisions['accepted']
                if paper['author_id'] == agent.id
            )

            # Simple funding model: gain funding for accepted papers, lose funding otherwise
            if agent_accepted > 0:
                funding_gain = 30 * agent_accepted
                agent.resources = min(100, agent.resources + funding_gain)
                funding_updates[agent.id] = f"+{funding_gain}"
            else:
                funding_loss = 10
                agent.resources = max(0, agent.resources - funding_loss)
                funding_updates[agent.id] = f"-{funding_loss}"

        year_results['phases']['funding_update'] = funding_updates

        # Eliminate agents with zero funding
        eliminated_agents = [a.id for a in self.ecosystem.agent_population if a.resources <= 0 and hasattr(a, 'can_author') and a.can_author]
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
        logger.info(f"Configuration: {self.num_years} years, {self.assignment_mode} mode")

        # Initialize agents
        self.initialize_agents()

        # Run simulation for each year
        for year in range(1, self.num_years + 1):
            year_results = self.run_year(year)
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
                'assignment_mode': self.assignment_mode,
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
                'assignment_mode': self.assignment_mode
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
        '--mode',
        type=str,
        choices=['batch', 'individual'],
        default='batch',
        help='Research direction assignment mode (default: batch)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./simulation_output',
        help='Output directory for results (default: ./simulation_output)'
    )

    args = parser.parse_args()

    # Create and run simulation
    simulation = Simulation(
        num_years=args.years,
        assignment_mode=args.mode,
        output_dir=args.output_dir
    )

    simulation.run()


if __name__ == '__main__':
    main()
