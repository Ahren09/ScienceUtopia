"""
Keyword extraction from research paper abstracts using LLMs.

This module provides functionality to extract keywords from academic abstracts
using various LLM models (GPT).
"""

import json
import os
import sys
import time
import traceback
from typing import List

from datasets import load_dataset
from tqdm import trange

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


if __name__ == "__main__":

    llm = GPT(model_name="gpt-4o")

    extractor = KeywordExtractor(llm)

    dataset = load_dataset("Ahren09/SciEvo", "arxiv", split="train")

    all_results = {}

    output_file = "outputs/keywords/keywords_scievo.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        all_results = json.load(open(output_file))

    num_generated = 0
    for i in trange(len(dataset) - 1, 0, -1, desc="Extracting keywords"):
        item = dataset[i]
        paper_id = item["id"]
        if paper_id in all_results:
            continue
        else:
            abstract = item["summary"]

            num_retries = 0

            for _ in range(5):
                try:
                    keywords = extractor.extract_keywords(abstract, num_keywords=10)
                    all_results[paper_id] = {
                        "id": paper_id,
                        "keywords_in_abstract": keywords["keywords_in_abstract"],
                        "keywords_not_in_abstract": keywords["keywords_not_in_abstract"],
                    }
                    num_generated += 1
                    break

                except Exception as e:
                    traceback.print_exc()
                    num_retries += 1
                    if num_retries >= 5:
                        raise Exception("Failed to extract keywords after 5 retries")

                    time.sleep(1)

            if num_generated % 100 == 0:
                json.dump(all_results, open(output_file, "w"), indent=2)
