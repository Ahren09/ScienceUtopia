import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from utopia.data.fulltext import (
    EXTRACTION,
    FORMAT,
    FullTextCorpus,
    decode_source_record,
    source_archive,
    canonical_arxiv_id,
    corpus_from_env,
    extract_unarxive_text,
    prepare_corpus,
)


def source_paper(paper_id="1601.00001", body=None):
    return {
        "paper_id": paper_id,
        "_source_hash": "source-fixture",
        "metadata": {"title": "Source title", "versions": [{"version": "v3"}]},
        "abstract": {"text": "SOURCE ABSTRACT"},
        "body_text": body if body is not None else [
            {"section": "Methods", "sec_number": "2",
             "text": "FIRST paragraph {{formula:eq1}} and citation {{cite:ref1}}."},
            {"section": "Appendix", "sec_number": "A",
             "text": "APPENDIX END " + "body material " * 1000 + "LAST SENTENCE"},
        ],
        "ref_entries": {
            "eq1": {"type": "formula", "latex": "E=mc^2"},
            "unused": {"type": "formula", "latex": "x_{end}=0.125"},
            "fig1": {"type": "figure", "caption": "FIGURE CAPTION END"},
            "table1": {"type": "table", "caption": "TABLE CAPTION END"},
        },
        "bib_entries": {"ref1": {"bib_entry_raw": "FINAL REFERENCE 2021"}},
    }


def write_corpus(root, text=None):
    root.mkdir(parents=True, exist_ok=True)
    text = text or extract_unarxive_text(source_paper())
    raw = text.encode()
    (root / "texts.utf8").write_bytes(raw)
    record = {
        "page_content": "ORIGINAL SCIEVO ABSTRACT",
        "metadata": {
            "id": "https://arxiv.org/abs/1601.00001v1", "title": "SciEvo title",
            "publication_year": 2016, "published": "2016-01-01",
            "tags": ["cs.AI"], "topics": ["Artificial Intelligence"],
        },
    }
    abstracts = (json.dumps(record) + "\n").encode()
    (root / "abstracts.jsonl").write_bytes(abstracts)
    index = {
        "format": FORMAT, "extraction": EXTRACTION, "years": [2016],
        "max_text_tokens": 110000,
        "tokenizer": {"model": "test-tokenizer", "revision": "frozen"},
        "text_file": "texts.utf8", "abstracts_file": "abstracts.jsonl",
        "abstracts_sha256": hashlib.sha256(abstracts).hexdigest(),
        "papers": {
            "1601.00001": {
                "offset": 0, "nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "token_count": 3000, "publication_year": 2016,
            }
        },
    }
    path = root / "index.json"
    path.write_text(json.dumps(index, sort_keys=True))
    return path


@pytest.mark.parametrize("identifier", [
    "1601.00001", "1601.00001v4", "arXiv:1601.00001v2",
    "https://arxiv.org/abs/1601.00001v2",
    "http://arxiv.org/pdf/1601.00001v3.pdf?download=1",
])
def test_canonical_aliases(identifier):
    assert canonical_arxiv_id(identifier) == "1601.00001"
    assert canonical_arxiv_id("https://arxiv.org/abs/cs/9901001v2") == "cs/9901001"


def test_non_arxiv_identifier_is_rejected():
    with pytest.raises(ValueError):
        canonical_arxiv_id("https://example.com/abs/1601.00001")


def test_extraction_keeps_all_sections_equations_captions_and_references():
    text = extract_unarxive_text(source_paper())
    for content in (
        "SOURCE ABSTRACT", "FIRST paragraph", "$E=mc^2$", "$x_{end}=0.125$",
        "APPENDIX END", "LAST SENTENCE", "FIGURE CAPTION END",
        "TABLE CAPTION END", "FINAL REFERENCE 2021",
    ):
        assert content in text
    assert text.index("FIRST paragraph") < text.index("APPENDIX END")
    assert extract_unarxive_text(source_paper(body=[])) == ""


def test_2024_sections_preserve_source_text_and_end_matter():
    paper = source_paper()
    body = paper.pop("body_text")
    paper["sections"] = {
        "Methods": {"text": body[0]["text"], "cite_spans": [], "ref_spans": []},
        "Appendix A": {"text": body[1]["text"] + "\n\n第二段 ∑ tail", "cite_spans": []},
    }
    text = extract_unarxive_text(paper)
    for content in (
        "FIRST paragraph $E=mc^2$", "APPENDIX END", "LAST SENTENCE",
        "\n\n第二段 ∑ tail", "TABLE CAPTION END", "FIGURE CAPTION END",
        "$x_{end}=0.125$", "FINAL REFERENCE 2021",
    ):
        assert content in text
    assert text.index("## Methods") < text.index("## Appendix A") < text.index("## References")
    paper["sections"] = {}
    assert extract_unarxive_text(paper) == ""
    paper["sections"] = {"broken": {"text": ["not a string"]}}
    with pytest.raises(ValueError, match="text string"):
        extract_unarxive_text(paper)


def test_exact_duplicated_eof_fragment_preserves_complete_paper_and_evidence():
    paper = {
        "paper_id": "2105.04669",
        "sections": {"Appendix": {"text": "COMPLETE END SECTION"}},
        "spans": [{"text": "{{figure:source-uuid}}", "ref_id": "source-uuid"}],
    }
    raw = (json.dumps(paper) + "\n").encode()
    fragment = raw[raw.index(b"{figure:"):]
    artifacts = []
    assert decode_source_record(raw, None, "source.jsonl", 92, False, artifacts) == paper
    assert decode_source_record(fragment, raw, "source.jsonl", 93, True, artifacts) is None
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert bytes.fromhex(artifact["raw_hex"]) == fragment
    assert artifact["sha256"] == hashlib.sha256(fragment).hexdigest()
    assert artifact["previous_record_sha256"] == hashlib.sha256(raw).hexdigest()
    assert artifact["previous_paper_id"] == "2105.04669"
    # A valid second paper remains a record, even when its bytes repeat exactly.
    assert decode_source_record(raw, raw, "source.jsonl", 94, True, artifacts) == paper
    assert len(artifacts) == 1
    for bad, previous, at_eof in [
        (fragment, raw, False),             # Same bytes inside a member are not accepted.
        (b"{novel-fragment}\n", raw, True),  # Novel bytes cannot be discarded.
        (fragment, b"broken" + fragment, True),  # A malformed predecessor proves nothing.
        (b"{}\n", raw, True),               # A valid non-paper JSON is not an artifact.
    ]:
        with pytest.raises(ValueError, match="source JSON|not a paper record"):
            decode_source_record(bad, previous, "source.jsonl", 93, at_eof, [])


def test_parallel_gzip_preserves_member_order_and_every_source_byte(tmp_path):
    pytest.importorskip("rapidgzip")
    path = tmp_path / "source.tar.gz"
    expected = [
        ("z.jsonl", (json.dumps(source_paper("1601.00002")) + "\n").encode()),
        ("a.jsonl", (json.dumps(source_paper("1601.00001")) + "\n").encode()),
    ]
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in expected:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    for threads in (0, 2):
        with source_archive(path, threads) as archive:
            assert [(member.name, archive.extractfile(member).read()) for member in archive] == expected


def test_corpus_aliases_missing_text_and_content_integrity(tmp_path):
    corpus = FullTextCorpus(write_corpus(tmp_path))
    assert "LAST SENTENCE" in corpus.text("https://arxiv.org/pdf/1601.00001v9.pdf")
    with pytest.raises(KeyError, match="fallback is disabled"):
        corpus.text("1601.99999")
    (tmp_path / "texts.utf8").write_text("corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        corpus.text("1601.00001")


def test_index_pin_and_size_eligibility(tmp_path):
    path = write_corpus(tmp_path)
    with pytest.raises(ValueError, match="pinned corpus"):
        FullTextCorpus(path, expected_sha256="0" * 64)
    index = json.loads(path.read_text())
    index["papers"]["1601.00001"]["token_count"] = 110001
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="token count"):
        FullTextCorpus(path)


def test_abstract_and_year_membership_fail_closed(tmp_path):
    corpus = FullTextCorpus(write_corpus(tmp_path))
    assert corpus.abstract_records(2016, 1)[0]["page_content"] == "ORIGINAL SCIEVO ABSTRACT"
    with pytest.raises(ValueError, match="outside"):
        corpus.abstract_records(2016, 2)
    (tmp_path / "abstracts.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="abstract records"):
        corpus.abstract_records(2016, 1)


def test_rag_retrieval_stays_abstract_and_checkpoint_pin_is_portable(tmp_path, monkeypatch):
    from utopia.data.rag import RAG, RAGConfig

    path = write_corpus(tmp_path / "machine_a")
    monkeypatch.setenv("UTOPIA_FULLTEXT_INDEX", str(path))
    monkeypatch.delenv("UTOPIA_FULLTEXT_SHA256", raising=False)
    rag = RAG(RAGConfig(device="cuda:0"), start_year=2016)
    rag.load_documents(num_years=1)
    assert rag.device == "cpu"
    assert len(rag.documents) == 1
    assert rag.documents[0].page_content == "ORIGINAL SCIEVO ABSTRACT"
    assert "LAST SENTENCE" not in rag.documents[0].page_content
    assert rag.id2year_pos["https://arxiv.org/abs/1601.00001v1"] == (2016, 0)
    state = rag.to_dict()
    assert "LAST SENTENCE" not in json.dumps(state)
    second_path = write_corpus(tmp_path / "machine_b")
    monkeypatch.setenv("UTOPIA_FULLTEXT_INDEX", str(second_path))
    restored = RAG.from_dict(state, RAGConfig(device="cpu"))
    assert restored.fulltext_corpus.reference() == state["fulltext"]
    monkeypatch.delenv("UTOPIA_FULLTEXT_INDEX")
    with pytest.raises(ValueError, match="mode differs"):
        RAG.from_dict(state, RAGConfig(device="cpu"))


def test_only_actual_review_receives_whole_paper(tmp_path, monkeypatch):
    from utopia.agents.base_agent import SimulationAgent

    monkeypatch.setenv("UTOPIA_FULLTEXT_INDEX", str(write_corpus(tmp_path)))
    monkeypatch.delenv("UTOPIA_FULLTEXT_SHA256", raising=False)
    agent = SimulationAgent.__new__(SimulationAgent)
    agent.expertise = []
    agent.get_memory_context = lambda: ""
    paper = {
        "id": "https://arxiv.org/abs/1601.00001v1", "title": "Title",
        "abstract": "ORIGINAL SCIEVO ABSTRACT", "topics": ["Artificial Intelligence"],
    }
    prompt, _ = agent.get_review_prompt(paper)
    for end_section in ("LAST SENTENCE", "TABLE CAPTION END", "FINAL REFERENCE 2021"):
        assert end_section in prompt
    assert "not a visual PDF-page review" in prompt
    with pytest.raises(KeyError):
        agent.get_review_prompt({**paper, "id": "1601.99999"})
    monkeypatch.delenv("UTOPIA_FULLTEXT_INDEX")
    legacy_prompt, _ = agent.get_review_prompt(paper)
    assert "ORIGINAL SCIEVO ABSTRACT" in legacy_prompt
    assert "Full Paper" not in legacy_prompt
    assert "LAST SENTENCE" not in legacy_prompt
    assert corpus_from_env() is None


@pytest.mark.parametrize("source_format", ["2022", "2024"])
def test_preparation_freezes_intersection_and_exclusions_before_worlds(
    tmp_path, monkeypatch, source_format
):
    from datasets import Dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    import datasets
    import huggingface_hub

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(tokenizer_path))
    rows = [{
        "id": f"https://arxiv.org/abs/1601.0000{i}", "published": "2016-01-01",
        "tags": ["cs.AI"], "summary": f"ABSTRACT {i}", "title": f"Title {i}",
    } for i in range(1, 5)]
    # A non-CS paper must not enter the intersected scientific corpus.
    rows.append({**rows[0], "id": "https://arxiv.org/abs/1601.00005", "tags": ["math.PR"]})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: Dataset.from_list(rows))
    archive = tmp_path / ("fixture.tar.xz" if source_format == "2022" else "unarXive_2024.tar.gz")
    records = [
        source_paper("1601.00001", body=[{"text": "SHORT COMPLETE BODY", "section": "Appendix"}]),
        source_paper("1601.00002", body=[]),
        source_paper("1601.00003", body=[{"text": "word " * 2000}]),
        source_paper("1601.00005", body=[{"text": "MATH BODY"}]),
    ]
    if source_format == "2024":
        for paper in records:
            body = paper.pop("body_text")
            paper["sections"] = {f"Section {i}": {"text": p["text"]} for i, p in enumerate(body)}
        records[0]["paper_id"] = records[0].pop("paper_id")
        # A later usable duplicate cannot replace an empty first source record.
        duplicate = source_paper("1601.00002", body=[{"text": "LATER COMPLETE TEXT"}])
        duplicate["sections"] = {"End": {"text": duplicate.pop("body_text")[0]["text"]}}
        records.append(duplicate)
        # Filtering must happen before extraction/tokenization, even for a malformed
        # out-of-year record. The requested SciEvo years remain unchanged.
        records.append({"paper_id": "2201.00001", "sections": {"broken": ["invalid"]}})
    payload = "".join(json.dumps(row) + "\n" for row in records).encode()
    if source_format == "2024":
        # Published full2024 contains an exact duplicated suffix after a complete
        # non-CS record at member EOF. All logical paper records must survive.
        fragment = (json.dumps(records[-1]) + "\n").encode()[-10:]
        payload += fragment
    with tarfile.open(archive, "w:xz" if source_format == "2022" else "w:gz") as stream:
        member = tarfile.TarInfo("arXiv_src_1601_001.jsonl")
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    metadata = tmp_path / "source.json"
    source_metadata = {
        "id": 7752615, "metadata": {"title": "Test source"},
        "files": [{"key": archive.name, "size": archive.stat().st_size,
                   "checksum": "md5:" + hashlib.md5(archive.read_bytes()).hexdigest()}],
    }
    if source_format == "2024":
        source_metadata = {
            "repo_id": "test/unarxive_2024", "revision": "fixed-source-revision",
            "primary_zenodo_record": 17431594, "files": [],
            "total_bytes": archive.stat().st_size,
            "primary_archive_md5": hashlib.md5(archive.read_bytes()).hexdigest(),
        }
    metadata.write_text(json.dumps(source_metadata))
    options = SimpleNamespace(
        output_dir=str(tmp_path / "corpus"), source_metadata=str(metadata), archive=str(archive),
        scievo_revision="fixed-test-revision", tokenizer_model="test", tokenizer_revision="fixed",
        start_year=2016, num_years=1, max_text_tokens=100,
        tokenizer_threads=2, duplicate_source_policy="first" if source_format == "2024" else "error",
    )
    prepare_corpus(options)
    corpus = FullTextCorpus(Path(options.output_dir) / "index.json")
    assert set(corpus.papers) == {"1601.00001"}
    assert corpus.index["counts"]["excluded_by_reason"] == {
        "empty_text": 1, "missing_fulltext": 1, "over_token_limit": 1,
    }
    assert corpus.abstract_records(2016, 1)[0]["page_content"] == "ABSTRACT 1"
    assert "FINAL REFERENCE 2021" in corpus.text("1601.00001")
    assert corpus.index["counts"]["excluded_by_year"] == {"2016": {
        "empty_text": 1, "missing_fulltext": 1, "over_token_limit": 1,
    }}
    assert corpus.papers["1601.00001"]["source_member"] == "arXiv_src_1601_001.jsonl"
    if source_format == "2024":
        assert corpus.index["source"]["revision"] == "fixed-source-revision"
        assert corpus.index["counts"]["duplicate_source_rows"] == 1
        assert corpus.index["counts"]["source_rows_scanned"] == 6
        assert corpus.index["counts"]["source_nonempty_physical_lines"] == 7
        assert corpus.index["counts"]["duplicated_trailing_fragments"] == 1
        assert bytes.fromhex(corpus.index["source_artifacts"][0]["raw_hex"]) == fragment
        assert corpus.index["counts"]["source_rows_parsed"] == 4
        assert corpus.index["duplicate_source_ids"] == {"1601.00002": 1}
        # Independent preparations from the same archive produce identical packs.
        options.output_dir = str(tmp_path / "second_corpus")
        prepare_corpus(options)
        second = FullTextCorpus(Path(options.output_dir) / "index.json")
        assert second.sha256 == corpus.sha256
        assert (second.root / "texts.utf8").read_bytes() == (corpus.root / "texts.utf8").read_bytes()
        options.output_dir = str(tmp_path / "duplicate_error")
        options.duplicate_source_policy = "error"
        with pytest.raises(ValueError, match="Duplicate source"):
            prepare_corpus(options)
        options.output_dir = str(corpus.root)
    with pytest.raises(FileExistsError):
        prepare_corpus(options)


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    import sys
    import torch

    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

        def encode(self, texts, **kwargs):
            return torch.tensor([[1.0, 2.0, 3.0]]).repeat(len(texts), 1)

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer,
    ))
    return calls


def test_fulltext_embedding_revision_constructors_cache_and_checkpoint(
    tmp_path, monkeypatch, fake_sentence_transformer
):
    from dataclasses import asdict
    from utopia.data.rag import RAG, RAGConfig

    revision = "17e1f347d17fe144873b1201da91788898c639cd"
    path = write_corpus(tmp_path / "corpus")
    monkeypatch.setenv("UTOPIA_FULLTEXT_INDEX", str(path))
    monkeypatch.delenv("UTOPIA_FULLTEXT_SHA256", raising=False)
    monkeypatch.setenv("UTOPIA_EMBEDDING_REVISION", revision)
    rag = RAG(RAGConfig(device="cpu"), start_year=2016)
    rag.load_documents(num_years=1)
    rag.build_knowledge_index()
    rag.set_current_year(1)
    rag.sentence_transformer_model = None
    assert rag.batch_retrieve(["abstract query"])["topk_indices"].tolist() == [[0]]
    assert len(fake_sentence_transformer) == 2
    assert all(kwargs["revision"] == revision for _, kwargs in fake_sentence_transformer)
    assert asdict(rag.config)["embedding_revision"] == revision
    state = rag.to_dict()
    assert state["embedding_config"] == {
        "embedding_model": "thenlper/gte-small",
        "embedding_revision": revision,
        "normalize_embeddings": True,
    }

    monkeypatch.delenv("UTOPIA_EMBEDDING_REVISION")
    restored = RAG.from_dict(state, RAGConfig(device="cpu"))
    assert restored.config.embedding_revision == revision
    assert restored.to_dict()["embedding_config"] == state["embedding_config"]
    monkeypatch.setenv("UTOPIA_EMBEDDING_REVISION", "b" * 40)
    with pytest.raises(ValueError, match="embedding configuration differs"):
        RAG.from_dict(state, RAGConfig(device="cpu"))
    with pytest.raises(ValueError, match="Conflicting full-text embedding revisions"):
        RAG(RAGConfig(device="cpu", embedding_revision=revision))

    alternate = RAG(RAGConfig(device="cpu"), start_year=2016)
    alternate.load_documents(num_years=1)
    alternate.build_knowledge_index()
    assert len(list((path.parent / "embeddings").glob("*.pt"))) == 2


def test_legacy_ignores_fulltext_embedding_revision_env(
    tmp_path, monkeypatch, fake_sentence_transformer
):
    from langchain_core.documents import Document
    from utopia.data.rag import RAG, RAGConfig

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UTOPIA_FULLTEXT_INDEX", raising=False)
    monkeypatch.setenv("UTOPIA_EMBEDDING_REVISION", "a" * 40)
    rag = RAG(RAGConfig(device="cpu", embedding_revision="b" * 40))
    rag.documents = [Document(page_content="abstract", metadata={"id": "1601.00001"})]
    rag.build_knowledge_index()
    assert all(kwargs["revision"] == "b" * 40 for _, kwargs in fake_sentence_transformer)
    assert len(list((tmp_path / "data/cache/rag").glob("embeddings-*.pt"))) == 1
    assert "embedding_config" not in rag.to_dict()


def test_four_fulltext_clients_publish_complete_cache_atomically(
    tmp_path, monkeypatch, fake_sentence_transformer
):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import torch
    from utopia.data.rag import RAG, RAGConfig

    path = write_corpus(tmp_path / "corpus")
    monkeypatch.setenv("UTOPIA_FULLTEXT_INDEX", str(path))
    monkeypatch.delenv("UTOPIA_FULLTEXT_SHA256", raising=False)
    monkeypatch.setenv("UTOPIA_EMBEDDING_REVISION", "a" * 40)
    clients = [RAG(RAGConfig(device="cpu"), start_year=2016) for _ in range(4)]
    for client in clients:
        client.load_documents(num_years=1)

    original_save = torch.save
    halfway = threading.Barrier(5)
    resume = threading.Event()

    def save_in_two_halves(value, output):
        buffer = io.BytesIO()
        original_save(value, buffer)
        data = buffer.getvalue()
        output.write(data[:len(data) // 2])
        output.flush()
        halfway.wait(timeout=15)
        assert resume.wait(timeout=15)
        output.write(data[len(data) // 2:])

    monkeypatch.setattr(torch, "save", save_in_two_halves)
    cache_root = path.parent / "embeddings"
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(client.build_knowledge_index) for client in clients]
        try:
            halfway.wait(timeout=15)
            assert not list(cache_root.glob("*.pt"))
            assert len(list(cache_root.glob("*.tmp"))) == 4
        finally:
            resume.set()
        for future in futures:
            future.result(timeout=15)

    files = list(cache_root.glob("*.pt"))
    assert len(files) == 1
    assert torch.equal(torch.load(files[0], weights_only=True), clients[0].document_embeddings)
    assert not list(cache_root.glob("*.tmp"))
    committed_bytes = files[0].read_bytes()

    def fail_after_partial_write(value, output):
        output.write(b"incomplete serialization")
        raise OSError("simulated interrupted cache writer")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    monkeypatch.setenv("UTOPIA_EMBEDDING_REVISION", "b" * 40)
    interrupted = RAG(RAGConfig(device="cpu"), start_year=2016)
    interrupted.load_documents(num_years=1)
    with pytest.raises(OSError, match="interrupted cache writer"):
        interrupted.build_knowledge_index()
    assert list(cache_root.glob("*.pt")) == files
    assert files[0].read_bytes() == committed_bytes
    assert not list(cache_root.glob("*.tmp"))
