"""Extract keywords from paper abstracts using LLM"""
import json
import os
from typing import List, Dict, Optional, Union

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
            tmp = self._cache_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self._cache, f)
            os.replace(tmp, self._cache_path)
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
