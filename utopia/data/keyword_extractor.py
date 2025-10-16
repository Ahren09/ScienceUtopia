"""
Keyword extraction from research paper abstracts using LLMs.

This module provides functionality to extract keywords from academic abstracts
using various LLM models (GPT).
"""

import os
import sys
from typing import List, Optional

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import GPT


class KeywordExtractor:
    """Extract keywords from research paper abstracts using LLMs."""

    def __init__(self, llm):
        """
        Initialize the keyword extractor with specified model.

        Args:
            model_type: Type of model to use ('gpt' or 'gemini')
            model_name: Specific model name (optional, uses defaults if not provided)
        """
        self.model = llm
        
    def extract_keywords(self, abstract: str, num_keywords: int = 10) -> List[str]:
        """
        Extract keywords from an abstract.

        Args:
            abstract: The research paper abstract text
            num_keywords: Number of keywords to extract (default: 10)

        Returns:
            List of extracted keywords
        """
        prompt = self._create_prompt(abstract, num_keywords)

        response = self.model.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt="You are an expert at analyzing academic papers and extracting relevant keywords."
        )

        if response is None:
            raise ValueError("Failed to generate keywords from the model")

        # Parse the response to extract keywords
        keywords = self._parse_keywords(response)

        return keywords[:num_keywords]  # Ensure we return exactly the requested number

    def _create_prompt(self, abstract: str, num_keywords: int) -> str:
        """
        Create the prompt for keyword extraction.

        Args:
            abstract: The research paper abstract
            num_keywords: Number of keywords to extract

        Returns:
            Formatted prompt string
        """
        prompt = f"""Given the following research paper abstract, extract exactly {num_keywords} keywords that best represent the main topics, methods, and concepts in the paper.

Abstract:
{abstract}

Instructions:
- Extract exactly {num_keywords} keywords or short keyphrases (1-3 words each)
- Focus on technical terms, methods, domains, and key concepts
- Keywords should be relevant to the research field and methodology
- Return ONLY the keywords separated by commas, nothing else
- Do not include explanations, numbering, or any other text

Keywords:"""

        return prompt

    def _parse_keywords(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract clean keywords.

        Args:
            response: Raw response from the LLM

        Returns:
            List of cleaned keywords
        """
        # Remove common prefixes/suffixes
        response = response.strip()

        # Split by comma and clean each keyword
        keywords = [kw.strip() for kw in response.split(',')]

        # Remove empty strings and filter out any non-keyword text
        keywords = [kw for kw in keywords if kw and len(kw) > 0]

        return keywords


def extract_keywords_from_abstract(
    abstract: str,
    llm,
    num_keywords: int = 10
) -> List[str]:
    """
    Convenience function to extract keywords from an abstract.

    Args:
        abstract: The research paper abstract text
        model_type: Type of model to use ('gpt' or 'gemini')
        model_name: Specific model name (optional)
        num_keywords: Number of keywords to extract (default: 10)

    Returns:
        List of extracted keywords

    Example:
        >>> abstract = "This paper presents a novel approach to..."
        >>> keywords = extract_keywords_from_abstract(abstract, model_type="gpt")
        >>> print(keywords)
        ['machine learning', 'neural networks', 'deep learning', ...]
    """
    extractor = KeywordExtractor(llm)
    return extractor.extract_keywords(abstract, num_keywords=num_keywords)


if __name__ == "__main__":
    # Example usage
    sample_abstract = """
    This paper presents a novel deep learning approach for sentiment analysis
    of social media text. We propose a transformer-based architecture that
    combines convolutional neural networks with attention mechanisms to capture
    both local and global contextual information. Our model achieves state-of-the-art
    performance on three benchmark datasets, outperforming previous methods by 5-7%
    in accuracy. We also introduce a new dataset of Twitter posts with fine-grained
    emotion labels. Extensive experiments demonstrate the effectiveness of our
    approach across multiple languages and domains.
    """

    print("Extracting keywords from sample abstract...\n")
    print(f"Abstract: {sample_abstract.strip()}\n")

    llm = GPT(model_name="gpt-4o")
    print("Using GPT model:")
    keywords = extract_keywords_from_abstract(
        abstract=sample_abstract,
        llm=llm,
        num_keywords=10
    )
    print(f"Keywords: {', '.join(keywords)}\n")