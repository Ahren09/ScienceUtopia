"""Analysis functions for simulation results"""

from utopia.utils.data_utils import write_json as write_json_file

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from utopia.metrics.tracker import calculate_citation_metrics

logger = logging.getLogger(__name__)


def analyze_citations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze citation metrics"""
    papers_by_year = defaultdict(list)
    paper_authors = {}
    paper_institutions = {}

    # Build lookups
    agent_lookup = {agent['id']: agent for agent in data['ecosystem_data']['agents']}
    papers_list = data['paper_tracker']['papers']

    for paper_data in papers_list:
        paper_id = paper_data['id']
        papers_by_year[paper_data['year']].append(paper_id)
        paper_authors[paper_id] = paper_data['author_id']

        author_ids = paper_data['author_id'] if isinstance(paper_data['author_id'], list) else [paper_data['author_id']]
        for aid in author_ids:
            agent = agent_lookup.get(aid)
            if agent is None:
                continue
            if 'university_name' in agent:
                paper_institutions[aid] = agent['university_name']
            elif 'company_name' in agent:
                paper_institutions[aid] = agent['company_name']

    return calculate_citation_metrics(data['citations'], dict(papers_by_year), paper_authors, paper_institutions)


def analyze_papers(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze paper statistics"""
    yearly_results = data['yearly_results']
    papers_list = data['paper_tracker']['papers']

    return {
        'total_submitted': sum(r['paper_submission']['num_papers_submitted'] for r in yearly_results),
        'total_accepted': sum(r['decisions']['total_accepted'] for r in yearly_results),
        'total_papers': len(papers_list),
        'acceptance_rate': sum(1 for p in papers_list if p['status'] == 'accept') / len(papers_list) if papers_list else 0
    }


def analyze_scores(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Analyze score trends across years"""
    return [
        {
            'year': r['year'],
            'mean': r['peer_review']['score_metrics']['overall_mean'],
            'std': r['peer_review']['score_metrics']['overall_std']
        }
        for r in data['yearly_results']
        if 'peer_review' in r and 'score_metrics' in r['peer_review']
    ]


def analyze_ecosystem(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Analyze ecosystem health trends"""
    return [
        {
            'year': r['year'],
            'gini': r['ecosystem_metrics']['resource_distribution']['gini'],
            'active_agents': r['ecosystem_metrics']['active_agents'],
            'mean_funding': r['ecosystem_metrics']['resource_distribution']['mean']
        }
        for r in data['yearly_results'] if 'ecosystem_metrics' in r
    ]


def generate_report(data: Dict[str, Any], output_file: Optional[str] = None) -> Dict[str, Any]:
    """Generate full report (calls all analysis functions)"""
    report = {
        'configuration': {
            'num_years': len(data['yearly_results']),
            'num_agents': len(data['ecosystem_data']['agents'])
        },
        'papers': analyze_papers(data),
        'citations': analyze_citations(data),
        'score_trends': analyze_scores(data),
        'ecosystem_trends': analyze_ecosystem(data)
    }

    if output_file:
        write_json_file(report, output_file, indent=2, default=str)
        logger.info(f"Report saved to {output_file}")

    return report


class SimulationAnalyzer:
    """Wrapper for use in simulation runtime"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.data = None

    def load_from_simulation(self, sim) -> None:
        """Load from simulation object"""
        self.data = {
            'yearly_results': sim.yearly_results,
            'paper_tracker': sim.paper_tracker.to_dict(),
            'citations': sim.citation_tracker.citations,
            'ecosystem_data': sim.ecosystem.to_dict(),
            'industry_funding_system': {'funding_mode': sim.industry_funding_mode}
        }

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate report from loaded data"""
        return generate_report(self.data, output_file)


def main():
    """CLI entry point"""
    from utopia.simulation import Simulation
    from utopia.arguments import parse_arguments
    from utopia.runtime.setup import project_setup

    args = parse_arguments()
    project_setup()
    simulation = Simulation(
        llm=None,
        num_years=args.num_years,
        output_dir=args.output_dir,
        debug=args.debug,
        industry_funding_mode=args.industry_funding_mode,
        funding_allocation_mode=args.funding_allocation_mode,
        always_rerun=args.always_rerun,
        use_langchain=args.use_langchain
    )


    success = simulation.load_checkpoint(year=args.num_years)
    if success:
        # Generate analysis report
        analyzer = SimulationAnalyzer(output_dir=args.output_dir)
        analyzer.load_from_simulation(simulation)
        analyzer.generate_report(str(Path(args.output_dir) / "simulation_report.json"))
        logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
