"""Deterministic keyword transports shared by contract tests and golden capture."""

import hashlib
import json


def _request_summary(arguments):
    result = dict(arguments)
    for key in ("prompt", "prompts"):
        if key not in result:
            continue
        values = result.pop(key)
        if isinstance(values, str):
            result[key + "_sha256"] = hashlib.sha256(values.encode()).hexdigest()
        else:
            result[key + "_sha256"] = [
                hashlib.sha256(value.encode()).hexdigest() for value in values
            ]
    return json.loads(json.dumps(result))


class KeywordTransport:
    def __init__(self):
        self.requests = []

    def generate_batch(self, prompts, **options):
        self.requests.append(_request_summary(dict(prompts=prompts, **options)))
        replies = [
            [
                {"keywords": ["Neural", "Language", "Theory"]},
                None,
                {"keywords": "invalid"},
            ],
            [{"keywords": ["Graph", "Search", "Optimization"]}, None],
            [None],
        ][len(self.requests) - 1]
        assert len(replies) == len(prompts)
        return [(reply, None) for reply in replies]


class CorpusKeywordTransport:
    def __init__(self):
        self.requests = []

    def generate(self, **options):
        self.requests.append(_request_summary(options))
        return {
            "keywords_in_abstract": ["Graph Search"],
            "keywords_not_in_abstract": ["Optimization"],
        }, None


def keyword_contract(extractor_type, corpus_type, directory):
    transport = KeywordTransport()
    extractor = extractor_type(cache_dir=str(directory), model_id="fake/model")
    extractor._cache["cached"] = {
        "keywords": ["Cached"],
        "keywords_in_abstract": ["Cached"],
        "keywords_not_in_abstract": [],
    }
    results = extractor.extract_keywords_batch(
        transport,
        [
            ("empty", "  "),
            ("cached", "Cached abstract"),
            ("ok", "Neural Language Modeling"),
            ("retry", "Graph Search Methods"),
            ("fail", "Complicated Unknown Abstract"),
        ],
    )
    cache_path = directory / "kw_fake_model_v1.json"
    replay = extractor_type(cache_dir=str(directory), model_id="fake/model")
    cached = replay.extract_keywords_batch(
        transport, [("ok", "Neural Language Modeling")]
    )
    corpus_transport = CorpusKeywordTransport()
    corpus = corpus_type(corpus_transport).extract_keywords("Graph Search Methods")
    return {
        "batch_requests": transport.requests,
        "results": results,
        "cached_result": cached,
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "corpus_requests": corpus_transport.requests,
        "corpus_result": corpus,
    }
