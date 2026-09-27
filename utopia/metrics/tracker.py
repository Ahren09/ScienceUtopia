"""Metrics calculation utilities for simulation analysis"""

import numpy as np
from typing import Dict, List, Set, Any, Union


def calculate_gini_coefficient(values: Union[List[float], Dict[str, float]]) -> float:
    """Calculate Gini coefficient for inequality measurement"""
    if isinstance(values, Dict):
        values = list(values.values())
    if not values or len(values) == 0:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_values) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]) if cumsum[-1] > 0 else 0.0


def calculate_conference_score_metrics(reviews_by_conference: Dict[str, Dict[str, List[Dict]]],
                            reviews_by_reviewer: Dict[str, List[float]]) -> Dict[str, Any]:
    """Calculate score fluctuation metrics

    Args:
        reviews_by_conference: {conference_id: {paper_id: [reviews]}}
        reviews_by_reviewer: {reviewer_id: [scores]}

    Returns:
        Dict with per-year, per-conference, and per-reviewer statistics
    """
    all_scores = []
    conference_stats = {}

    # Per-conference statistics
    for conf_id, papers in reviews_by_conference.items():
        conf_scores = []
        for paper_id, reviews in papers.items():
            scores = [r['overall_score'] for r in reviews]
            conf_scores.extend(scores)
            all_scores.extend(scores)

        if conf_scores:
            conference_stats[conf_id] = {
                'mean': float(np.mean(conf_scores)),
                'std': float(np.std(conf_scores)),
                'count': len(conf_scores)
            }

    # Per-reviewer consistency
    reviewer_consistency = {}
    for reviewer_id, scores in reviews_by_reviewer.items():
        if len(scores) > 1:
            reviewer_consistency[reviewer_id] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            }

    return {
        'overall_mean': float(np.mean(all_scores)) if all_scores else 0.0,
        'overall_std': float(np.std(all_scores)) if all_scores else 0.0,
        'conference_stats': conference_stats,
        'reviewer_consistency': reviewer_consistency,
        'total_reviews': len(all_scores)
    }


def calculate_citation_metrics(citations: Dict[str, Set], papers_by_year: Dict[int, List[str]],
                               paper_authors: Dict[str, str], paper_institutions: Dict[str, str]) -> Dict[str, Any]:
    """Calculate citation dynamics metrics

    Args:
        citations: {paper_id: set of citing paper_ids}
        papers_by_year: {year: [paper_ids]}
        paper_authors: {paper_id: author_id}
        paper_institutions: {author_id: institution}

    Returns:
        Dict with citation growth, density, inequality, and self-citation metrics
    """
    citation_counts = [len(cites) for cites in citations.values()]
    total_citations = sum(citation_counts)

    # Citation inequality (Gini coefficient)
    gini = calculate_gini_coefficient(citation_counts) if citation_counts else 0.0

    # Citation network density
    num_papers = len(citations)
    max_possible_citations = num_papers * (num_papers - 1)
    density = total_citations / max_possible_citations if max_possible_citations > 0 else 0.0

    # Self-citation analysis
    self_citations = 0
    institutional_citations = 0

    def _get_author_set(author_id):
        """Normalize author_id to a set of author IDs."""
        if isinstance(author_id, list):
            return set(author_id)
        return {author_id} if author_id else set()

    for cited_paper_id, citing_paper_ids in citations.items():
        cited_authors = _get_author_set(paper_authors.get(cited_paper_id))
        cited_institutions = {paper_institutions.get(a) for a in cited_authors} - {None}

        for citing_paper_id in citing_paper_ids:
            citing_authors = _get_author_set(paper_authors.get(citing_paper_id))
            citing_institutions = {paper_institutions.get(a) for a in citing_authors} - {None}

            if cited_authors & citing_authors:
                self_citations += 1
            elif cited_institutions & citing_institutions:
                institutional_citations += 1

    return {
        'total_citations': total_citations,
        'papers_with_citations': sum(1 for c in citation_counts if c > 0),
        'citation_gini': gini,
        'citation_density': density,
        'self_citation_rate': self_citations / total_citations if total_citations > 0 else 0.0,
        'institutional_citation_rate': institutional_citations / total_citations if total_citations > 0 else 0.0,
        'mean_citations_per_paper': float(np.mean(citation_counts)) if citation_counts else 0.0,
        'max_citations': max(citation_counts) if citation_counts else 0
    }


def top_percentile(value, data):
    data = np.array(data)
    percentile = np.sum(data > value) / len(data) * 100
    return max(percentile, 1)
