"""Extract keywords from paper abstracts using LLM"""
from typing import List


class KeywordExtractor:
    """Extract keywords from abstracts using LLM"""

    def __init__(self):
        pass

    def extract_keywords_llm(self, abstract: str, llm, n_keywords: int = 10) -> List[str]:
        """Extract keywords from abstract using LLM

        Args:
            abstract: Paper abstract text
            llm: LLM instance
            n_keywords: Number of keywords to extract

        Returns:
            List of extracted keywords
        """
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

            return keywords[:n_keywords] if keywords else []

        except Exception as e:
            # Fallback: simple extraction from abstract
            words = abstract.lower().split()
            keywords = [w.strip('.,;:()[]') for w in words if len(w) > 6][:n_keywords]
            return keywords
