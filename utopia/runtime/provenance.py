"""Cache, server, request and content provenance shared by experiment drivers."""

from __future__ import annotations

import os
from datetime import datetime

from utopia.utils.data_utils import file_sha256
from functools import partial
from utopia.utils.data_utils import write_json_atomic
from utopia.utils.data_utils import file_sha256 as file_hash
from utopia.utils.paths import project_root
from collections import Counter
import hashlib
import json
from pathlib import Path


write_json = partial(write_json_atomic, indent=2, allow_nan=False)


ROOT = project_root(__file__)


MODEL = "Qwen/Qwen3-32B"


MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"


CACHE_INPUTS = {
    "outputs/cache_scievo.pkl": "e39b9b68a91fe2a86a6085934bf126b20c384a7d5e6ea4537feebf6255f98396",
    "outputs/thenlper_gte-small/cache_embeddings.pt": "f8f9bcbe2c8d81c613de2e24fb6bec90ffc204150acc6823f70fb38bc5ede0e9",
}


CACHE_MANIFEST = "outputs/logs/ec2_preparation/cache_alignment.json"


def digest(value):
    from utopia.utils.data_utils import json_sha256

    return json_sha256(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cache_file_identity(root=ROOT):
    """The original loader uses these shared paths, not args.data_cache_dir."""
    result = {}
    for name, expected in CACHE_INPUTS.items():
        path = root / name
        if path.stat().st_mode & 0o222:
            raise ValueError(f"Shared input cache must be read-only: {path}")
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(
                f"Input cache differs from the original transferred file: {name}"
            )
        result[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return result


def audit_cache(output=ROOT / CACHE_MANIFEST):
    """Bind original ordered documents to embedding rows without rewriting caches.

    The original cache lacks embedded document identifiers. Full row bindings
    and fixed, evenly spaced re-encoding checks provide explicit provenance;
    sampled numerical agreement is not a claim of full re-encoding validation.
    """
    import pickle
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    identity = cache_file_identity()
    with (ROOT / "outputs/cache_scievo.pkl").open("rb") as stream:
        cache = pickle.load(
            stream
        )  # Exact user-authorized source, hash verified above.
    docs = cache["documents"]
    embeddings = torch.load(
        ROOT / "outputs/thenlper_gte-small/cache_embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )
    ids = [d.metadata["id"] for d in docs]
    if (
        len(set(ids)) != len(ids)
        or embeddings.ndim != 2
        or embeddings.shape != (len(docs), 384)
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid document IDs or embedding dimensions/values")
    year_positions = Counter()
    binding = hashlib.sha256()
    for row, doc in enumerate(docs):
        year = doc.metadata["publication_year"]
        if tuple(cache["id2year_pos"][doc.metadata["id"]]) != (
            year,
            year_positions[year],
        ):
            raise ValueError(
                "Cached ID-to-year/position mapping does not match document order"
            )
        year_positions[year] += 1
        binding.update(
            json.dumps(
                [
                    row,
                    doc.metadata["id"],
                    hashlib.sha256(doc.page_content.encode()).hexdigest(),
                    hashlib.sha256(embeddings[row].numpy().tobytes()).hexdigest(),
                ],
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
    indices = np.unique(np.linspace(0, len(docs) - 1, min(256, len(docs)), dtype=int))
    model = SentenceTransformer(
        "thenlper/gte-small",
        revision="17e1f347d17fe144873b1201da91788898c639cd",
        device="cpu",
        local_files_only=True,
    )
    recomputed = model.encode(
        [docs[i].page_content for i in indices],
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=32,
        device="cpu",
    )
    cosine = torch.nn.functional.cosine_similarity(
        recomputed.float(), embeddings[indices].float(), dim=1
    )
    if float(cosine.min()) < 0.999:
        raise ValueError(
            f"Document/embedding alignment check failed: min cosine {cosine.min()}"
        )
    record = {
        "input_files": identity,
        "document_count": len(docs),
        "embedding_shape": list(embeddings.shape),
        "corpus_hash": corpus_fingerprint(docs),
        "ordered_ids_hash": digest(ids),
        "all_row_binding_sha256": binding.hexdigest(),
        "embedding_model": "thenlper/gte-small",
        "embedding_revision": "17e1f347d17fe144873b1201da91788898c639cd",
        "checked_rows": [int(i) for i in indices],
        "checked_row_min_cosine": float(cosine.min()),
        "alignment_scope": (
            "Exact original cache pair copied read-only from Cheesecake; ordered ID, text and "
            "embedding-byte bindings cover every row. Fixed 256-row CPU re-encoding check "
            "tests alignment numerically; not a historical generation manifest or full re-encoding proof."
        ),
    }
    if identity != cache_file_identity():
        raise RuntimeError("Cache changed during alignment audit")
    write_json(output, record)
    return record


def verified_cache_manifest():
    record = json.loads((ROOT / CACHE_MANIFEST).read_text())
    if record["input_files"] != cache_file_identity():
        raise ValueError("Cache manifest does not match immutable input files")
    if record["checked_row_min_cosine"] < 0.999 or len(record["checked_rows"]) != 256:
        raise ValueError("Cache alignment audit is incomplete")
    return record


def mark_failed_run_manifest(directory, protocol, error):
    """Update only the current attempt's lifecycle; retain native phase/year evidence."""
    path = Path(directory) / "run_manifest.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(status="failed", wrapper_protocol=protocol, wrapper_error=error)
    write_json(path, record)


def corpus_fingerprint(documents):
    """Hash actual ordered content, string-normalizing the loader's date metadata."""
    hasher = hashlib.sha256()
    for doc in documents:
        # Original RAG stores datetime/Timestamp/np.datetime64 in "published".
        hasher.update(
            json.dumps(
                [doc.page_content, doc.metadata],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                default=str,
            ).encode()
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def server_provenance(path, endpoint, *, runtime_profile=None):
    """Validate the queue's explicit attestation; endpoint ID alone is insufficient."""
    record = json.loads(Path(path).read_text())
    runtime_hash = validate_server_record(
        record, endpoint, runtime_profile=runtime_profile
    )
    result = {
        "record": record,
        "file_sha256": file_sha256(Path(path)),
        "runtime_hash": runtime_hash,
    }
    if runtime_profile is not None:
        result["manifest_path"] = str(Path(path).resolve())
    return result


def validate_server_record(record, endpoint, *, runtime_profile=None):
    """Apply the same explicit runtime contract at launch and file admission."""
    expected = {
        "model": MODEL,
        "revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 2,
        "seed": 42,
        "request_seed_base": 42,
        "request_seed_policy": "derive_seed(run_seed, phase_context, item_index, retry)",
        "vllm_version": "0.12.0",
        "transformers_version": "4.57.3",
        "torch_version": "2.9.0+cu128",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.8,
        "reasoning_parser": "qwen3",
        "max_num_seqs": 128,
    }
    if runtime_profile is not None:
        from utopia.runtime.server_profiles import (
            TP1_PLAN_PROFILE,
            TP1_PROFILE_NAME,
            tp1_server_spec,
        )

        if runtime_profile != TP1_PROFILE_NAME:
            raise ValueError("Unsupported explicit server runtime profile")
        expected = tp1_server_spec()
        if record.get("server_profile") != TP1_PLAN_PROFILE:
            raise ValueError(
                "TP1 server provenance lacks the prospective queue profile"
            )
    for key, value in expected.items():
        if record.get(key) != value or (
            runtime_profile is not None and type(record.get(key)) is not type(value)
        ):
            raise ValueError(f"Server provenance mismatch: {key}, expected {value!r}")
    if record.get("endpoint", "").rstrip("/") != endpoint.rstrip("/"):
        raise ValueError("Server provenance endpoint mismatch")
    gpu_ids = record.get("gpu_ids", record.get("server_gpus", []))
    gpu_count = expected["tensor_parallel_size"]
    if (
        len(gpu_ids) != gpu_count
        or len(set(gpu_ids)) != gpu_count
        or not all(
            record.get(k) for k in ("host", "pid", "launch_command", "started_utc")
        )
    ):
        raise ValueError(
            "Server provenance lacks actual launch identity or the required GPU count"
        )
    return digest(expected)


def validate_request_audit_summary(summary):
    """A successful scope must include requests from exactly one guarded client."""
    expected = {
        "status": "complete",
        "n_clients": 1,
        "guard_failures": 0,
        "pending_requests": 0,
    }
    if (
        any(summary.get(key) != value for key, value in expected.items())
        or type(summary.get("n_requests")) is not int
        or summary["n_requests"] <= 0
    ):
        raise RuntimeError(
            "Request audit does not establish complete guarded execution"
        )


def config_hash(config: dict) -> str:
    """SHA-256 hex digest of a config dict (sorted-key JSON, tuples->lists)."""
    from utopia.utils.data_utils import json_sha256
    return json_sha256(config, sort_keys=True, default=str)


def write_run_manifest(docs_dir: str, status: str, args=None, resolved_config: dict = None,
                       extra: dict = None) -> str:
    """Write/update outputs/docs/<experiment_id>/run_manifest.json.

    Called at run start (status='running') and end (status='complete'/'failed').
    Merges into an existing manifest so start-time fields survive the final write.
    """
    import numpy as np
    import pandas as pd
    import torch
    import subprocess
    import sys

    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "run_manifest.json")
    manifest = {}
    if os.path.exists(path):
        with open(path) as f:
            manifest = json.load(f)

    def _git(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    manifest.update({
        "status": status,
        f"timestamp_{status}": datetime.now().isoformat(),
    })
    if "command" not in manifest:
        manifest.update({
            "command": " ".join(sys.argv),
            "git_commit": _git(["git", "rev-parse", "HEAD"]),
            "git_dirty_files": _git(["git", "status", "--short"]),
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        try:
            import torch, vllm, transformers, sentence_transformers
            manifest["package_versions"] = {
                "torch": torch.__version__, "vllm": vllm.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
            }
        except ImportError:
            pass
    if args is not None:
        manifest["args"] = {k: v for k, v in sorted(vars(args).items())}
    if resolved_config is not None:
        manifest["resolved_config"] = resolved_config
        manifest["scientific_config_hash"] = config_hash(resolved_config)
    if extra:
        # Portable checkpoints restore cumulative counters, so a later final
        # write replaces them (including a resume with no new inference).
        # Legacy callers still supply session counters that must be added.
        saved_args = manifest.get("args")
        portable = isinstance(saved_args, dict) and saved_args.get("portable_checkpoint") is True
        new_stats = extra.get("llm_call_stats")
        old_stats = manifest.get("llm_call_stats")
        if not portable and isinstance(new_stats, dict) and isinstance(old_stats, dict):
            extra = dict(extra)
            extra["llm_call_stats"] = {
                k: (old_stats.get(k, 0) or 0) + (new_stats.get(k, 0) or 0)
                for k in set(old_stats) | set(new_stats)}
        manifest.update(extra)

    write_json_atomic(path, manifest, indent=2, default=str,
                      trailing_newline=False, streaming=True)
    return path
