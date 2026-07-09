"""Extract keywords from paper abstracts using LLM"""
from typing import List, Dict, Union


class KeywordExtractor:
    """Extract keywords from abstracts using LLM"""

    def __init__(self):
        pass

    def extract_keywords_llm(self, abstract: str, llm, n_keywords: int = 10) -> Dict:
        """Extract keywords from abstract using LLM

        Args:
            abstract: Paper abstract text
            llm: LLM instance
            n_keywords: Number of keywords to extract

        Returns:
            Dict with 'keywords', 'keywords_in_abstract', 'keywords_not_in_abstract'
        """
        if not abstract or not abstract.strip():
            return {'keywords': [], 'keywords_in_abstract': [], 'keywords_not_in_abstract': []}

        prompt = f"""Extract {n_keywords} technical keywords from the following research abstract.
Return only the keywords as a JSON array of strings.

Abstract: {abstract}

Keywords (JSON array):"""

        try:
            response, _ = llm.generate(prompt=prompt, response_format="json")

            # Handle different response formats
            if isinstance(response, dict) and 'keywords' in response:
                keywords = response['keywords']
            elif isinstance(response, list):
                keywords = response
            else:
                keywords = []

            keywords = keywords[:n_keywords] if keywords else []

        except Exception as e:
            # Fallback: simple extraction from abstract
            words = abstract.lower().split()
            keywords = [w.strip('.,;:()[]') for w in words if len(w) > 6][:n_keywords]

        # Split into keywords found in abstract vs not
        abstract_lower = abstract.lower()
        keywords_in = [k for k in keywords if k.lower() in abstract_lower]
        keywords_not_in = [k for k in keywords if k.lower() not in abstract_lower]

        return {
            'keywords': keywords,
            'keywords_in_abstract': keywords_in,
            'keywords_not_in_abstract': keywords_not_in,
        }
