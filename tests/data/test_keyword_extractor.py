import json

from tests.support.keyword_extraction import keyword_contract
from utopia.data.keyword_extractor import CorpusKeywordExtractor, KeywordExtractor
from utopia.utils.paths import project_root


def test_keyword_protocols_match_pre_consolidation_requests_and_cache(tmp_path):
    golden = json.loads(
        (project_root() / "tests/support/fixtures/ownership_golden.json").read_text()
    )
    assert (
        keyword_contract(KeywordExtractor, CorpusKeywordExtractor, tmp_path)
        == golden["keywords"]
    )
    assert not hasattr(KeywordExtractor, "extract_keywords_llm")
