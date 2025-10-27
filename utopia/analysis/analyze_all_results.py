"""Analysis and reporting for simulation results"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from utopia.metrics.tracker import calculate_citation_metrics

logger = logging.getLogger(__name__)


class SimulationAnalyzer:
    """Analyzes simulation results and generates reports"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.data = None

    def load_from_checkpoint(self, year: int) -> bool:
        """Load from checkpoint file"""
        checkpoint_path = self.output_dir / f"checkpoint_year_{year}.json"
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False

        with open(checkpoint_path, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded checkpoint from year {year}")
        return True

    def load_from_simulation(self, sim) -> None:
        """Load from simulation object"""
        self.data = {
            'yearly_results': sim.yearly_results,
            'paper_archive': sim.paper_archive.to_dict(),
            'citations': sim.citations,
            'ecosystem_data': sim.ecosystem.to_dict(),
            'industry_funding_system': {'funding_mode': sim.industry_funding_mode}
        }

    def _get_citation_metrics(self) -> Dict[str, Any]:
        """Calculate citation metrics"""
        papers_by_year = defaultdict(list)
        paper_authors = {}
        paper_institutions = {}

        for paper_id, paper_data in self.data['paper_archive']['papers_by_id'].items():
            papers_by_year[paper_data['year']].append(paper_id)
            paper_authors[paper_id] = paper_data['author_id']

            agent = self.data['ecosystem_data']['agent_population'].get(paper_data['author_id'])
            if agent:
                paper_institutions[paper_data['author_id']] = agent.get('university_name') or agent.get('company_name')

        return calculate_citation_metrics(
            self.data['citations'],
            dict(papers_by_year),
            paper_authors,
            paper_institutions
        )

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive report"""
        logger.info("Generating simulation report...")

        yearly_results = self.data['yearly_results']
        papers_by_id = self.data['paper_archive']['papers_by_id']

        # Build report
        report = {
            'configuration': {
                'num_years': len(yearly_results),
                'num_agents': len(self.data['ecosystem_data']['agent_population']),
                'industry_funding_mode': self.data.get('industry_funding_system', {}).get('funding_mode', 'performance')
            },
            'yearly_results': yearly_results,
            'summary': {
                'total_papers_submitted': sum(r['phases']['paper_submission']['num_papers_submitted'] for r in yearly_results),
                'total_papers_accepted': sum(r['phases']['decisions']['total_accepted'] for r in yearly_results),
                'total_papers': len(papers_by_id),
                'acceptance_rate': sum(1 for p in papers_by_id.values() if p['decision'] == 'accept') / len(papers_by_id) if papers_by_id else 0
            },
            'citation_metrics': self._get_citation_metrics(),
            'trends': {
                'score_trends': [
                    {
                        'year': r['year'],
                        'mean': r['phases']['peer_review']['score_metrics']['overall_mean'],
                        'std': r['phases']['peer_review']['score_metrics']['overall_std']
                    }
                    for r in yearly_results if 'score_metrics' in r['phases'].get('peer_review', {})
                ],
                'ecosystem_trends': [
                    {
                        'year': r['year'],
                        'gini': r['ecosystem_metrics']['resource_distribution']['gini'],
                        'active_agents': r['ecosystem_metrics']['active_agents'],
                        'mean_funding': r['ecosystem_metrics']['resource_distribution']['mean']
                    }
                    for r in yearly_results if 'ecosystem_metrics' in r
                ]
            }
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")

        # Log summary
        cm = report['citation_metrics']
        logger.info(f"Papers: {report['summary']['total_papers']}, Citations: {cm['total_citations']}, Gini: {cm['citation_gini']:.3f}")

        if report['trends']['ecosystem_trends']:
            et = report['trends']['ecosystem_trends'][-1]
            logger.info(f"Final state - Active agents: {et['active_agents']}, Funding Gini: {et['gini']:.3f}")

        return report


def analyze_simulation(output_dir: str, year: Optional[int] = None) -> Dict[str, Any]:
    """Analyze simulation from output directory"""
    analyzer = SimulationAnalyzer(output_dir)

    if year is None:
        checkpoints = list(Path(output_dir).glob("checkpoint_year_*.json"))
        if checkpoints:
            year = max(int(cp.stem.split('_')[-1]) for cp in checkpoints)
        else:
            logger.error("No checkpoints found")
            return {}

    if not analyzer.load_from_checkpoint(year):
        return {}

    return analyzer.generate_report(output_file=str(Path(output_dir) / "simulation_report.json"))


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--year', type=int, default=None, help='Year to analyze (default: latest)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    analyze_simulation(args.output_dir, args.year)


if __name__ == '__main__':
    main()
