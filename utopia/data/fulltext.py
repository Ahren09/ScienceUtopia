"""Frozen extracted paper text for opt-in peer review.

Retrieval and author selection retain the original SciEvo abstracts. Text comes
from unarXive's structured source extraction, not visual PDF-page inspection.
Run ``python -m utopia.data.fulltext --help`` to prepare one shared corpus.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen


FORMAT = "scienceutopia-fulltext-v1"
EXTRACTION = "unarxive-body-or-sections-equations-captions-references-v2"
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?")
_MARKER = re.compile(r"\{\{(formula|cite|figure|table|ref):([^{}]+)\}\}")
_LEADING_PAPER_ID = re.compile(rb'^\s*\{\s*"paper_id"\s*:\s*"([^"\\]+)"')


@contextmanager
def source_archive(path, gzip_threads=0):
    """Read the same tar members in archive order, optionally using parallel gzip."""
    if gzip_threads:
        if gzip_threads < 0 or not str(path).endswith(".gz"):
            raise ValueError("Parallel gzip requires a .gz archive and positive threads")
        import rapidgzip

        with rapidgzip.open(str(path), parallelization=gzip_threads) as stream:
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                yield archive
    else:
        with tarfile.open(path, "r|*") as archive:
            yield archive


def decode_source_record(raw, previous_raw, member, line_number, at_eof, artifacts):
    """Retain complete JSON papers and document an exact duplicated trailing suffix.

    One published member ends with a second copy of 103 bytes from its preceding,
    already complete JSON paper. This is not a multiline paper or another record.
    Only this provable redundancy is accepted. Every other parse error remains fatal.
    """
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        previous = None
        if at_eof and previous_raw and len(raw) < len(previous_raw) and previous_raw.endswith(raw):
            try:
                previous = json.loads(previous_raw)
                canonical_arxiv_id(previous["paper_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                previous = None
        if previous is not None:
            artifacts.append({
                "kind": "duplicated_trailing_fragment",
                "source_member": member, "physical_line": line_number,
                "previous_paper_id": previous["paper_id"],
                "previous_record_sha256": hashlib.sha256(previous_raw).hexdigest(),
                "nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "raw_hex": raw.hex(),
            })
            return None
        raise ValueError(f"Invalid source JSON at {member}:{line_number}: {error}") from error
    if not isinstance(record, dict) or "paper_id" not in record:
        raise ValueError(f"Source JSON is not a paper record at {member}:{line_number}")
    return record


def canonical_arxiv_id(value):
    """Normalize bare, versioned, abs/PDF URL and legacy arXiv identifiers."""
    value = unquote(str(value)).strip()
    if value.startswith(("https://", "http://")):
        parsed = urlsplit(value)
        if parsed.hostname not in ("arxiv.org", "www.arxiv.org", "export.arxiv.org"):
            raise ValueError(f"Not an arXiv identifier: {value}")
        value = re.sub(r"^/(?:abs|pdf)/", "", parsed.path)
    else:
        value = re.sub(r"^arxiv:\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value)
    if not _ARXIV_ID.fullmatch(value):
        raise ValueError(f"Not an arXiv identifier: {value}")
    return re.sub(r"v\d+$", "", value)


def extract_unarxive_text(paper):
    """Keep every body block and all available equations, captions and references.

    Only source markers are expanded. Body order and textual content are not
    shortened, summarized, or regenerated. Unresolved source markers stay visible.
    """
    body = paper.get("body_text")
    if body is None:
        sections = paper.get("sections") or {}
        if not isinstance(sections, dict):
            raise ValueError("unarXive sections must be keyed objects")
        body = []
        for title, section in sections.items():
            if not isinstance(section, dict) or not isinstance(section.get("text", ""), str):
                raise ValueError("unarXive section must contain a text string")
            body.append({**section, "section": title})
    if not isinstance(body, list):
        raise ValueError("unarXive body must be a list of blocks")
    if not any(isinstance(p, dict) and str(p.get("text") or "").strip() for p in body):
        return ""
    refs = paper.get("ref_entries") or {}
    bibliography = paper.get("bib_entries") or {}
    if not isinstance(refs, dict) or not isinstance(bibliography, dict):
        raise ValueError("unarXive reference entries must be keyed objects")
    citation_labels = {key: str(i) for i, key in enumerate(bibliography, 1)}
    ref_labels = {key: str(i) for i, key in enumerate(refs, 1)}
    expanded_formulas = set()

    def expand(text):
        def replace(match):
            kind, key = match.groups()
            entry = refs.get(key, {})
            if kind == "formula" and entry.get("latex") is not None:
                expanded_formulas.add(key)
                return "$" + str(entry["latex"]) + "$"
            if kind == "cite" and key in citation_labels:
                return "[" + citation_labels[key] + "]"
            if key in ref_labels:
                return f"[{entry.get('type', kind)} {ref_labels[key]}]"
            return match.group()
        return _MARKER.sub(replace, str(text or ""))

    metadata = paper.get("metadata") or {}
    abstract = paper.get("abstract") or metadata.get("abstract") or ""
    if isinstance(abstract, dict):
        abstract = abstract.get("text", "")
    parts = ["# " + str(metadata.get("title") or paper["paper_id"])]
    if abstract:
        parts.extend(["## Abstract", expand(abstract)])
    previous_section = None
    for paragraph in body:
        if not isinstance(paragraph, dict):
            raise ValueError("unarXive body block is not an object")
        section = (paragraph.get("sec_number"), paragraph.get("section"))
        if section != previous_section and any(section):
            parts.append("## " + " ".join(str(x) for x in section if x))
        previous_section = section
        parts.append(expand(paragraph.get("text", "")))
    # Formula markers were expanded in place. Include even unreferenced formula
    # entries, and every figure/table caption available in the extraction.
    for key, entry in refs.items():
        fields = []
        if entry.get("caption") is not None:
            fields.append(expand(entry["caption"]))
        if entry.get("text") is not None:
            fields.append(expand(entry["text"]))
        if entry.get("latex") is not None and key not in expanded_formulas:
            fields.append("$" + str(entry["latex"]) + "$")
        if fields:
            parts.append(f"## {entry.get('type', 'Reference')} {ref_labels[key]}")
            parts.extend(fields)
    if bibliography:
        parts.append("## References")
        for key, entry in bibliography.items():
            parts.append(f"[{citation_labels[key]}] " + str(entry.get("bib_entry_raw") or ""))
    return "\n\n".join(parts) + "\n"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FullTextCorpus:
    """Read one immutable text pack. Checkpoints store its index hash only."""

    def __init__(self, index_path, expected_sha256=None):
        self.path = Path(index_path).resolve()
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("Full-text index differs from the pinned corpus")
        self.index = json.loads(raw)
        if self.index.get("format") != FORMAT:
            raise ValueError("Unsupported full-text index format")
        self.root = self.path.parent
        self.papers = self.index["papers"]
        if not self.papers:
            raise ValueError("Full-text corpus has no eligible papers")
        for key, entry in self.papers.items():
            if key != canonical_arxiv_id(key):
                raise ValueError(f"Noncanonical full-text key: {key}")
            if not 0 < entry["token_count"] <= self.index["max_text_tokens"]:
                raise ValueError(f"Ineligible full-text token count: {key}")

    def data_path(self, name):
        result = (self.root / name).resolve()
        if not result.is_relative_to(self.root):
            raise ValueError("Full-text data path escapes the corpus directory")
        return result

    def text(self, arxiv_id):
        key = canonical_arxiv_id(arxiv_id)
        if key not in self.papers:
            raise KeyError(f"No eligible full text for {key}; abstract fallback is disabled")
        entry = self.papers[key]
        with self.data_path(self.index["text_file"]).open("rb") as stream:
            stream.seek(entry["offset"])
            raw = stream.read(entry["nbytes"])
        if len(raw) != entry["nbytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Full-text content hash mismatch: {key}")
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Empty full text: {key}")
        return text

    def abstract_records(self, start_year, num_years):
        """Read frozen eligible SciEvo records without changing their abstracts."""
        wanted = set(range(start_year, start_year + num_years))
        if not wanted.issubset(self.index["years"]):
            raise ValueError("Requested simulation years are outside the frozen full-text corpus")
        path = self.data_path(self.index["abstracts_file"])
        if file_sha256(path) != self.index["abstracts_sha256"]:
            raise ValueError("Frozen SciEvo abstract records have changed")
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        seen = set()
        for record in records:
            key = canonical_arxiv_id(record["metadata"]["id"])
            if key not in self.papers or key in seen:
                raise ValueError("Frozen abstract/full-text membership differs")
            seen.add(key)
        if seen != set(self.papers):
            raise ValueError("Frozen abstract/full-text membership differs")
        return [r for r in records if r["metadata"]["publication_year"] in wanted]

    def reference(self):
        return {
            "format": FORMAT,
            "index_sha256": self.sha256,
            "extraction": self.index["extraction"],
            "tokenizer": self.index["tokenizer"],
            "max_text_tokens": self.index["max_text_tokens"],
            "paper_count": len(self.papers),
        }


@lru_cache(maxsize=4)
def load_corpus(path):
    return FullTextCorpus(path)


def corpus_from_env():
    """Opt in explicitly, with an optional independently pinned index hash."""
    path = os.environ.get("UTOPIA_FULLTEXT_INDEX")
    if not path:
        return None
    corpus = load_corpus(str(Path(path).resolve()))
    expected = os.environ.get("UTOPIA_FULLTEXT_SHA256")
    if expected and corpus.sha256 != expected:
        raise ValueError("Full-text index differs from UTOPIA_FULLTEXT_SHA256")
    return corpus


def prepare_corpus(options):
    """Intersect SciEvo CS/year records with complete nonempty size-eligible texts."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from utopia.agents.conference import ALL_CS_TOPICS, ARXIV_CATEGORY_NAMES

    root = Path(options.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "index.json").exists():
        raise FileExistsError("Corpus is already frozen; use its index, do not overwrite it")
    source = json.loads(Path(options.source_metadata).read_text())
    archive = Path(options.archive)
    if source.get("repo_id"):
        if archive.name != "unarXive_2024.tar.gz":
            raise ValueError("Expected the concatenated unarXive 2024 archive")
        source_file = {
            "size": source["total_bytes"],
            "checksum": "md5:" + source["primary_archive_md5"],
        }
        provenance = {
            "repo_id": source["repo_id"], "revision": source["revision"],
            "record": str(source["primary_zenodo_record"]),
            "title": "unarXive 2024",
            "parts": source["files"],
        }
    else:
        source_file = next(f for f in source["files"] if f["key"] == archive.name)
        provenance = {"record": str(source["id"]), "title": source["metadata"]["title"]}
    algorithm, checksum = source_file["checksum"].split(":", 1)
    if archive.stat().st_size != source_file["size"]:
        raise ValueError("Source archive download is incomplete")
    print("Verifying source archive", flush=True)
    digest, archive_digest = hashlib.new(algorithm), hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
            archive_digest.update(chunk)
    actual_checksum = digest.hexdigest()
    if actual_checksum != checksum:
        raise ValueError("Source archive checksum mismatch")
    archive_sha = archive_digest.hexdigest()
    revision = options.scievo_revision
    if not revision:
        with urlopen("https://huggingface.co/api/datasets/Ahren09/SciEvo", timeout=30) as response:
            revision = json.load(response)["sha"]
    threads = getattr(options, "tokenizer_threads", 16)
    if threads < 1:
        raise ValueError("tokenizer_threads must be positive")
    # encode_batch uses Rayon. Set the bound before loading the tokenizer.
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer_path = hf_hub_download(
        options.tokenizer_model, "tokenizer.json", revision=options.tokenizer_revision
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    print(f"Loading SciEvo revision {revision}", flush=True)
    dataset = load_dataset("Ahren09/SciEvo", "arxiv", split="train", revision=revision)
    candidates = {}
    years = list(range(options.start_year, options.start_year + options.num_years))
    duplicate_abstract_ids = 0
    cs_topics = set(ALL_CS_TOPICS)
    columns = ["id", "published", "tags", "summary", "title"]
    rows = (
        dict(zip(columns, values))
        for batch in dataset.select_columns(columns).iter(batch_size=8192)
        for values in zip(*(batch[column] for column in columns))
    )
    for row in rows:
        published = str(row.get("published") or "")
        if not re.match(r"^\d{4}", published):
            continue
        year = int(published[:4])
        tags = list(row.get("tags") or [])
        if year not in years or not set(tags) & cs_topics:
            continue
        original_id = row["id"].replace("http://", "https://")
        key = canonical_arxiv_id(original_id)
        record = {
            "page_content": row.get("summary") or "",
            "metadata": {
                "id": original_id, "title": row.get("title") or "",
                "published": published, "publication_year": year, "tags": tags,
                "topics": [ARXIV_CATEGORY_NAMES[tag] for tag in tags if tag in ARXIV_CATEGORY_NAMES],
            },
        }
        if key in candidates:
            duplicate_abstract_ids += 1
            # Stable across file order. Prefer the largest explicit version.
            def version(doc):
                match = re.search(r"v(\d+)$", doc["metadata"]["id"])
                return (int(match[1]) if match else 0, json.dumps(doc, sort_keys=True))
            record = max(candidates[key], record, key=version)
        candidates[key] = record
    candidate_year_counts = Counter(r["metadata"]["publication_year"] for r in candidates.values())
    print(json.dumps({"scievo_candidates": len(candidates),
                      "candidates_by_year": dict(sorted(candidate_year_counts.items())),
                      "tokenizer_threads": threads}), flush=True)
    papers, seen = {}, set()
    rejected = {}
    pending = []
    text_partial = root / "texts.utf8.partial"
    duplicate_policy = getattr(options, "duplicate_source_policy", "error")
    if duplicate_policy not in ("error", "first"):
        raise ValueError("Unsupported duplicate source policy")
    duplicate_source_ids = Counter()
    source_rows = 0
    source_rows_parsed = 0
    source_artifacts = []
    next_progress = 2000

    def flush(batch, output):
        encodings = tokenizer.encode_batch([item[2] for item in batch], add_special_tokens=False)
        for (key, source_paper, text, member_name, record_sha), encoding in zip(batch, encodings):
            count = len(encoding.ids)
            if count == 0 or count > options.max_text_tokens:
                rejected[key] = {"reason": "over_token_limit" if count else "empty_text", "token_count": count}
                continue
            raw = text.encode("utf-8")
            offset = output.tell()
            output.write(raw)
            metadata = source_paper.get("metadata") or {}
            papers[key] = {
                "offset": offset, "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "token_count": count,
                "publication_year": candidates[key]["metadata"]["publication_year"],
                "source_hash": source_paper.get("_source_hash"),
                "source_versions": metadata.get("versions", []),
                "source_member": member_name, "source_record_sha256": record_sha,
            }

    gzip_threads = getattr(options, "gzip_threads", 0)
    with text_partial.open("xb") as output, source_archive(archive, gzip_threads) as source_tar:
        for member in source_tar:
            if not member.isfile() or not member.name.endswith(".jsonl"):
                continue
            source_file_stream = source_tar.extractfile(member)
            previous_line = None
            for line_number, line in enumerate(source_file_stream, 1):
                prior_line, previous_line = previous_line, line
                if not line.strip():
                    continue
                source_rows += 1
                # Published JSONL starts with paper_id. Avoid decoding full text
                # for unrelated papers. Other key orders use the normal JSON path.
                leading_id = _LEADING_PAPER_ID.match(line)
                if leading_id and canonical_arxiv_id(leading_id[1].decode("utf-8")) not in candidates:
                    continue
                source_paper = decode_source_record(
                    line, prior_line, member.name, line_number,
                    source_file_stream.tell() == member.size, source_artifacts,
                )
                if source_paper is None:
                    print(json.dumps({"source_artifact": {
                        key: value for key, value in source_artifacts[-1].items() if key != "raw_hex"
                    }}), flush=True)
                    continue
                source_rows_parsed += 1
                key = canonical_arxiv_id(source_paper["paper_id"])
                if key not in candidates:
                    continue
                if key in seen:
                    duplicate_source_ids[key] += 1
                    if duplicate_policy == "error":
                        raise ValueError(f"Duplicate source full text requires explicit resolution: {key}")
                    # The first record in the pinned archive wins before content
                    # or size eligibility. Never select a better-looking duplicate.
                    continue
                seen.add(key)
                text = extract_unarxive_text(source_paper)
                if not text.strip():
                    rejected[key] = {"reason": "empty_text"}
                    continue
                pending.append((key, source_paper, text, member.name, hashlib.sha256(line).hexdigest()))
                if len(pending) == 32:
                    flush(pending, output)
                    pending.clear()
                if len(seen) >= next_progress:
                    print(json.dumps({
                        "source_rows": source_rows - len(source_artifacts),
                        "nonpaper_fragments": len(source_artifacts),
                        "matched": len(seen), "eligible": len(papers),
                        "eligible_by_year": dict(sorted(Counter(
                            item["publication_year"] for item in papers.values()).items())),
                    }), flush=True)
                    next_progress = len(seen) + 2000
        if pending:
            flush(pending, output)
        output.flush()
        os.fsync(output.fileno())
    for key in sorted(set(candidates) - seen):
        rejected[key] = {"reason": "missing_fulltext"}
    ordered_ids = sorted(papers, key=lambda key: (papers[key]["publication_year"], key))
    if not ordered_ids:
        raise ValueError("No full-text eligible corpus")
    year_counts = Counter(papers[key]["publication_year"] for key in papers)
    if set(year_counts) != set(years):
        raise ValueError(f"No eligible full texts in some years: {dict(year_counts)}")
    abstracts_partial = root / "abstracts.jsonl.partial"
    with abstracts_partial.open("x", encoding="utf-8") as output:
        for key in ordered_ids:
            output.write(json.dumps(candidates[key], ensure_ascii=False, sort_keys=True) + "\n")
    index = {
        "format": FORMAT, "extraction": EXTRACTION, "years": years,
        "source": {
            **provenance,
            "archive": archive.name, "archive_sha256": archive_sha,
            "archive_checksum": source_file["checksum"],
            "metadata_sha256": file_sha256(options.source_metadata),
            "duplicate_policy": duplicate_policy,
            "archive_reader": {"backend": "rapidgzip" if gzip_threads else "tarfile",
                               "gzip_threads": gzip_threads},
            "nonpaper_fragment_policy": "Only an invalid member-final fragment identical to a "
                                        "suffix of the preceding complete paper JSON is recorded "
                                        "as redundant bytes. Raw bytes and hashes are retained "
                                        "in source_artifacts. No paper is removed by this rule.",
            "note": "Extracted full text, not visual PDF-page review. Source metadata versions "
                    "are retained, but do not by themselves identify the exact extracted version. "
                    "Selected records are pinned by source member, source hash and record SHA256.",
        },
        "scievo_revision": revision,
        "tokenizer": {
            "model": options.tokenizer_model, "revision": options.tokenizer_revision,
            "tokenizer_json_sha256": file_sha256(tokenizer_path),
        },
        "max_text_tokens": options.max_text_tokens,
        "text_file": "texts.utf8", "text_sha256": file_sha256(text_partial),
        "abstracts_file": "abstracts.jsonl", "abstracts_sha256": file_sha256(abstracts_partial),
        "papers": {key: papers[key] for key in ordered_ids},
        "counts": {
            "scievo_cs_year_candidates": len(candidates),
            "candidates_by_year": dict(sorted(candidate_year_counts.items())),
            "duplicate_scievo_aliases": duplicate_abstract_ids,
            "source_rows_scanned": source_rows - len(source_artifacts),
            "source_nonempty_physical_lines": source_rows,
            "duplicated_trailing_fragments": len(source_artifacts),
            "source_rows_parsed": source_rows_parsed,
            "duplicate_source_rows": sum(duplicate_source_ids.values()),
            "eligible": len(papers),
            "eligible_by_year": dict(sorted(year_counts.items())),
            "excluded_by_reason": dict(sorted(Counter(v["reason"] for v in rejected.values()).items())),
            "excluded_by_year": {
                year: dict(sorted(Counter(
                    value["reason"] for key, value in rejected.items()
                    if candidates[key]["metadata"]["publication_year"] == year
                ).items())) for year in years
            },
        },
        "duplicate_source_ids": dict(sorted(duplicate_source_ids.items())),
        "source_artifacts": source_artifacts,
        "excluded": rejected,
    }
    os.replace(text_partial, root / index["text_file"])
    os.replace(abstracts_partial, root / index["abstracts_file"])
    index_partial = root / "index.json.partial"
    index_partial.write_text(json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(index_partial, root / "index.json")
    print(json.dumps({"index_sha256": file_sha256(root / "index.json"), **index["counts"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scievo-revision")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-text-tokens", type=int, default=110000)
    parser.add_argument("--tokenizer-threads", type=int, default=16)
    parser.add_argument("--gzip-threads", type=int, default=0,
                        help="Optional rapidgzip CPU reader. Zero uses the original tar reader.")
    parser.add_argument("--duplicate-source-policy", choices=("error", "first"), default="error",
                        help="For first, keep the first record in the pinned archive before eligibility.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--num-years", type=int, default=6)
    prepare_corpus(parser.parse_args())


if __name__ == "__main__":
    main()
