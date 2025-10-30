"""Analysis functions for simulation results"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from utopia.metrics.tracker import calculate_citation_metrics

logger = logging.getLogger(__name__)


def load_checkpoint(output_dir: str, year: Optional[int] = None) -> Dict[str, Any]:
    """Load simulation checkpoint data"""
    path = Path(output_dir)
    if year is None:
        checkpoints = list(path.glob("checkpoint_year_*.json"))
        year = max(int(cp.stem.split('_')[-1]) for cp in checkpoints) if checkpoints else None

    if year is None:
        logger.error("No checkpoints found")
        return {}

    with open(path / f"checkpoint_year_{year}.json", 'r') as f:
        return json.load(f)


def analyze_citations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze citation metrics"""
    papers_by_year = defaultdict(list)
    paper_authors = {}
    paper_institutions = {}

    # Build lookups
    agent_lookup = {agent['id']: agent for agent in data['ecosystem_data']['agents']}
    papers_list = data['paper_archive']['papers']

    for paper_data in papers_list:
        paper_id = paper_data['id']
        papers_by_year[paper_data['year']].append(paper_id)
        paper_authors[paper_id] = paper_data['author_id']

        agent = agent_lookup[paper_data['author_id']]
        if 'university_name' in agent:
            paper_institutions[paper_data['author_id']] = agent['university_name']
        elif 'company_name' in agent:
            paper_institutions[paper_data['author_id']] = agent['company_name']

    return calculate_citation_metrics(data['citations'], dict(papers_by_year), paper_authors, paper_institutions)


def analyze_papers(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze paper statistics"""
    yearly_results = data['yearly_results']
    papers_list = data['paper_archive']['papers']

    return {
        'total_submitted': sum(r['phases']['paper_submission']['num_papers_submitted'] for r in yearly_results),
        'total_accepted': sum(r['phases']['decisions']['total_accepted'] for r in yearly_results),
        'total_papers': len(papers_list),
        'acceptance_rate': sum(1 for p in papers_list if p['decision'] == 'accept') / len(papers_list) if papers_list else 0
    }


def analyze_scores(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Analyze score trends across years"""
    return [
        {
            'year': r['year'],
            'mean': r['phases']['peer_review']['score_metrics']['overall_mean'],
            'std': r['phases']['peer_review']['score_metrics']['overall_std']
        }
        for r in data['yearly_results']
        if 'peer_review' in r['phases'] and 'score_metrics' in r['phases']['peer_review']
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
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
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
            'paper_archive': sim.paper_archive.to_dict(),
            'citations': sim.citations,
            'ecosystem_data': sim.ecosystem.to_dict(),
            'industry_funding_system': {'funding_mode': sim.industry_funding_mode}
        }

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate report from loaded data"""
        return generate_report(self.data, output_file)


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--year', type=int, default=None, help='Year to analyze')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    data = load_checkpoint(args.output_dir, args.year)
    if data:
        generate_report(data, str(Path(args.output_dir) / "simulation_report.json"))


if __name__ == '__main__':
    main()
