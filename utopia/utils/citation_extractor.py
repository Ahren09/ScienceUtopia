"""
Citation Extraction Utility

This module provides LLM-based citation extraction from paper content.
When agents write papers, they can reference previously accepted papers,
and this utility extracts those citations.
"""

import json
from typing import List, Set
from utopia.agents.citation import Citation
from utopia.models.models import GPT


CITATION_EXTRACTION_PROMPT = """You are a citation extraction expert. Given a paper abstract/content and a list of potentially cited papers, identify which papers are actually referenced or cited in the content.

Paper Content:
{content}

Available Papers to Cite (ID | Title):
{available_papers}

Task:
1. Read the paper content carefully
2. Identify which of the available papers are likely referenced or related to this work
3. Return ONLY the paper IDs (arxiv IDs) that are cited or should be cited

Return your answer as a comma-separated list of paper IDs. If no citations are found, return "NONE".

Example output format:
arxiv_2301.00001, arxiv_2302.00123, arxiv_2303.00456

Paper IDs (comma-separated):"""


def extract_citations_from_content(
    paper_id: str,
    paper_content: str,
    available_papers: List[dict],
    llm=None,
    model: str = "gpt-4o",
    max_papers_to_show: int = 20
) -> Set[Citation]:
    """
    Extract citations from paper content using LLM

    Args:
        paper_id: ID of the current paper
        paper_content: The paper's abstract or full content
        available_papers: List of papers that could be cited, each dict with keys:
                         'id', 'title', 'abstract' (optional)
        llm: LLM instance (if None, will create one)
        model: LLM model to use
        max_papers_to_show: Maximum number of papers to show to LLM

    Returns:
        Set of Citation objects
    """
    if not available_papers:
        return set()

    # Create LLM if not provided
    if llm is None:
        llm = GPT(model_name=model)

    # Format available papers for prompt (limit to most relevant/recent)
    papers_list = []
    for i, paper in enumerate(available_papers[:max_papers_to_show]):
        paper_info = f"{paper['id']} | {paper['title']}"
        papers_list.append(paper_info)

    available_papers_str = "\n".join(papers_list)

    # Build prompt
    prompt = CITATION_EXTRACTION_PROMPT.format(
        content=paper_content[:2000],  # Limit content length
        available_papers=available_papers_str
    )

    # Get LLM response
    try:
        response = llm.generate(
            prompt=prompt,
            system_prompt="You are a helpful citation extraction assistant.",
            temperature=0.3  # Lower temperature for more deterministic extraction
        )

        # Parse response (handle both JSON and plain text)
        response = response.strip()

        # Try to parse as JSON first
        try:
            response_json = json.loads(response)
            if isinstance(response_json, dict) and 'citations' in response_json:
                response = response_json['citations']
            elif isinstance(response_json, dict) and 'paper_ids' in response_json:
                response = response_json['paper_ids']
        except json.JSONDecodeError:
            pass  # Not JSON, treat as plain text

        if isinstance(response, str):
            response = response.strip()
            if response.upper() == "NONE" or not response:
                return set()

            # Extract paper IDs from comma-separated string
            cited_ids = [pid.strip() for pid in response.split(',')]
            cited_ids = [pid for pid in cited_ids if pid]  # Remove empty strings
        elif isinstance(response, list):
            cited_ids = response
        else:
            return set()

        # Create Citation objects
        citations = set()
        available_ids = {p['id'] for p in available_papers}
        for cited_id in cited_ids:
            if cited_id in available_ids:
                citations.add(Citation(
                    current_paper_id=paper_id,
                    cited_paper_id=cited_id
                ))

        return citations

    except Exception as e:
        print(f"Error extracting citations: {e}")
        return set()


def extract_citations_simple(
    paper_id: str,
    paper_content: str,
    available_paper_ids: List[str]
) -> Set[Citation]:
    """
    Simple citation extraction by keyword matching (fallback method)

    Args:
        paper_id: ID of the current paper
        paper_content: The paper's content
        available_paper_ids: List of paper IDs that could be cited

    Returns:
        Set of Citation objects
    """
    citations = set()
    content_lower = paper_content.lower()

    for cited_id in available_paper_ids:
        # Simple heuristic: check if paper ID appears in content
        if cited_id.lower() in content_lower:
            citations.add(Citation(
                current_paper_id=paper_id,
                cited_paper_id=cited_id
            ))

    return citations
