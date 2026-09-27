"""Extract keywords from paper abstracts using LLM"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import write_json_atomic
import json
import os
from typing import List, Dict, Optional

KEYWORD_PROMPT_VERSION = 'v1'

KEYWORD_RESPONSE_FORMAT = {
    'type': 'json_schema',
    'json_object': {
        'name': 'keyword_extraction',
        'strict': True,
        'schema': {
            'type': 'object',
            'properties': {
                'keywords': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['keywords'],
        },
    },
}


class KeywordExtractor:
    """Extract keywords from abstracts using LLM"""

    def __init__(self, cache_dir: Optional[str] = None, model_id: str = 'unknown'):
        # JSON cache keyed by (paper_id, model_id, prompt_version) so repeated
        # seeds and conditions never re-extract keywords for the same paper.
        self.cache_dir = cache_dir
        self.model_id = model_id
        self._cache: Dict[str, Dict] = {}
        self._cache_dirty = False
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_path = os.path.join(
                cache_dir, f"kw_{model_id.replace('/', '_')}_{KEYWORD_PROMPT_VERSION}.json")
            if os.path.exists(self._cache_path):
                with open(self._cache_path) as f:
                    self._cache = json.load(f)

    def flush_cache(self):
        if self.cache_dir and self._cache_dirty:
            write_json_atomic(self._cache_path, self._cache,
                              trailing_newline=False, streaming=True)
            self._cache_dirty = False

    def _build_prompt(self, abstract: str, n_keywords: int) -> str:
        return f"""Extract {n_keywords} technical keywords from the following research abstract.
Return a JSON object of the form {{"keywords": ["keyword1", "keyword2", ...]}}.

Abstract: {abstract}

JSON:"""

    @staticmethod
    def _parse_keywords(response, n_keywords: int) -> Optional[List[str]]:
        if isinstance(response, dict) and 'keywords' in response:
            keywords = response['keywords']
        elif isinstance(response, list):
            keywords = response
        else:
            return None
        if not isinstance(keywords, list):
            return None
        # unique, order-preserving, string-only, capped at n_keywords
        seen, out = set(), []
        for k in keywords:
            k = str(k).strip()
            if k and k.lower() not in seen:
                seen.add(k.lower())
                out.append(k)
        return out[:n_keywords] if out else None

    @staticmethod
    def _fallback_keywords(abstract: str, n_keywords: int) -> List[str]:
        words = abstract.lower().split()
        seen, out = set(), []
        for w in words:
            w = w.strip('.,;:()[]')
            if len(w) > 6 and w not in seen:
                seen.add(w)
                out.append(w)
        return out[:n_keywords]

    @staticmethod
    def _split_in_abstract(keywords: List[str], abstract: str) -> Dict:
        abstract_lower = abstract.lower()
        return {
            'keywords_in_abstract': [k for k in keywords if k.lower() in abstract_lower],
            'keywords_not_in_abstract': [k for k in keywords if k.lower() not in abstract_lower],
        }

    def extract_keywords_batch(self, llm, items: List[tuple], n_keywords: int = 10,
                               max_retries: int = 2) -> Dict[str, Dict]:
        """Extract keywords for many papers with batched LLM calls.

        Args:
            llm: model exposing generate_batch(prompts, response_format=...)
            items: list of (paper_id, abstract)
            n_keywords: max keywords per paper
            max_retries: batch retries for failed members only

        Returns:
            Dict paper_id -> {'keywords', 'keywords_in_abstract',
            'keywords_not_in_abstract', 'provenance': 'llm'|'fallback'|'cache'|'empty'}
        """
        results: Dict[str, Dict] = {}
        pending = []  # (paper_id, abstract)
        for paper_id, abstract in items:
            if not abstract or not abstract.strip():
                results[paper_id] = {'keywords': [], 'keywords_in_abstract': [],
                                     'keywords_not_in_abstract': [], 'provenance': 'empty'}
            elif paper_id in self._cache:
                results[paper_id] = {**self._cache[paper_id], 'provenance': 'cache'}
            else:
                pending.append((paper_id, abstract))

        attempt = 0
        while pending and attempt <= max_retries and hasattr(llm, 'generate_batch'):
            prompts = [self._build_prompt(abstract, n_keywords) for _, abstract in pending]
            batch_results = llm.generate_batch(
                prompts, response_format=KEYWORD_RESPONSE_FORMAT, temperature=0.3,
                desc=f"keywords batch ({len(prompts)} papers, attempt {attempt + 1})",
                seed_ctx=('keywords', attempt),
            )
            still_pending = []
            for (paper_id, abstract), (response, _) in zip(pending, batch_results):
                keywords = self._parse_keywords(response, n_keywords)
                if keywords is None:
                    still_pending.append((paper_id, abstract))
                    continue
                entry = {'keywords': keywords, **self._split_in_abstract(keywords, abstract)}
                self._cache[paper_id] = entry
                self._cache_dirty = True
                results[paper_id] = {**entry, 'provenance': 'llm'}
            pending = still_pending
            attempt += 1

        # Deterministic fallback for members that failed all retries; flagged, never cached
        for paper_id, abstract in pending:
            keywords = self._fallback_keywords(abstract, n_keywords)
            results[paper_id] = {'keywords': keywords,
                                 **self._split_in_abstract(keywords, abstract),
                                 'provenance': 'fallback'}

        self.flush_cache()
        return results


class CorpusKeywordExtractor:
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
            num_keywords: Number of keywords to extract

        Returns:
            List of extracted keywords
        """

        sample_response = """
```json
{
    "keywords_in_abstract": ["Deep Learning", "Sentiment Analysis", "Convolutional Neural Networks", ...],  <- A list of keywords that are contained in the abstract, sorted by relevance to the abstract
    "keywords_not_in_abstract": ["Natural Language Processing", "Large Language Models", "Keyword Extraction", ...] <- A list of keywords that are NOT contained in the abstract, sorted by relevance to the abstract
}
```
"""

        prompt = f"""Given an abstract of a research paper, extract exactly {num_keywords} terminologies that best represent the technical terms, methods, and concepts in the paper.

Instructions:
- Extract both terminologies that are contained in the abstract (`keywords_in_abstract`) and keywords that are NOT contained in the abstract (`keywords_not_in_abstract`).
- Write the keywords in title case (capitalize the first letter of each word),
- Extract at most {num_keywords} keywords or short keyphrases (1-5 words each), You can extract fewer than {num_keywords} keywords if the abstract is short.
- Keywords should be sorted by relevance (from most relevant to least relevant) to the abstract.
- Do not include explanations, numbering, or any other text
- Keywords like "NP-hard Problems" is allowed, but "Problems" alone is not allowed.

## Sample Input
This paper presents a novel deep learning approach for sentiment analysis of social media text. We propose a transformer-based architecture that combines convolutional neural networks with attention mechanisms to capture both local and global contextual information. Our model achieves state-of-the-art performance on three benchmark datasets, outperforming previous methods by 5-7% in accuracy. We also introduce a new dataset of Twitter posts with fine-grained emotion labels. Extensive experiments demonstrate the effectiveness of our approach across multiple languages and domains.

## Sample Response
{sample_response}

## Abstract to extract keywords from

{abstract}
"""

        response, _ = self.model.generate(
            prompt=prompt,
            temperature=0.3,
            system_prompt="You are an expert at analyzing academic papers and extracting relevant keywords.",
            response_format="json_object"
        )

        assert 'keywords_in_abstract' in response and 'keywords_not_in_abstract' in response, "Failed to generate keywords from the model"

        if response is None:
            raise ValueError("Failed to generate keywords from the model")

        return response


