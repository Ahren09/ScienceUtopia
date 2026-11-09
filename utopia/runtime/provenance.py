"""Cache, server, request and content provenance shared by experiment drivers."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path

from utopia.utils.data_utils import file_sha256, write_json_atomic
from utopia.utils.paths import project_root

write_json = partial(write_json_atomic, indent=2, allow_nan=False)


ROOT = project_root(__file__)

def latest_native_checkpoint(directory):
    """Prefer a committed stage cursor; retain annual-checkpoint compatibility."""
    from utopia.utils.data_utils import read_json
    directory = Path(directory)
    latest = directory / 'latest_checkpoint.json'
    if latest.exists():
        path = directory / read_json(latest)['path']
        if not path.is_file():
            raise ValueError(f'Committed checkpoint is missing: {path}')
        return str(path)
    candidates = list(directory.glob('checkpoint_year_*.json*'))
    candidates = [p for p in candidates if '_stage_' not in p.name and not p.name.endswith('.tmp')]
    return str(max(candidates, key=lambda p: int(p.name.split('_')[2].split('.')[0]))) if candidates else None


def s3_location(uri):
    from urllib.parse import urlsplit
    parsed = urlsplit(uri)
    if parsed.scheme != 's3' or not parsed.netloc:
        raise ValueError(f'Expected an s3://bucket/prefix location: {uri}')
    return parsed.netloc, parsed.path.strip('/')


def upload_verified(client, path, bucket, key):
    """Publish and independently read back bytes; no content-hash admission gate."""
    from boto3.s3.transfer import TransferConfig
    path = Path(path)
    client.upload_file(str(path), bucket, key, Config=TransferConfig(max_concurrency=4))
    remote = client.get_object(Bucket=bucket, Key=key)['Body']
    try:
        with path.open('rb') as local:
            while True:
                chunk = local.read(8 * 1024 * 1024)
                if remote.read(len(chunk) or 1) != chunk:
                    raise IOError(f'S3 readback differs: s3://{bucket}/{key}')
                if not chunk:
                    break
    finally:
        remote.close()
    return dict(key=key, bytes=path.stat().st_size)


def publish_native_snapshot(simulation):
    """Upload immutable objects before publishing the checkpoint/reply manifest."""
    import boto3
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor
    import tarfile
    import tempfile
    import time
    from utopia.utils.data_utils import atomic_open, read_json
    args = simulation.args
    bucket, prefix = s3_location(args.checkpoint_s3_prefix)
    client = boto3.client('s3', config=Config(max_pool_connections=64))
    output = Path(simulation.output_dir)
    inventory_path = output / 's3_inventory.json'
    inventory = read_json(inventory_path) if inventory_path.exists() else {'files': {}, 'replies': [], 'offset': 0}
    if inventory.get('destination') != [bucket, prefix]:
        inventory = {'files': {}, 'replies': [], 'offset': 0}
    generation = str(time.time_ns())
    with tempfile.TemporaryDirectory(dir=output, prefix='snapshot-') as temporary:
        temporary = Path(temporary)
        # The archive makes the code used to write a checkpoint available on a new host.
        code = output / 'restore_code.tar.gz'
        source_files = list((ROOT / 'utopia').rglob('*.py')) + [ROOT / 'requirements.txt', ROOT / 'CLAUDE.md']
        if not code.exists() or max(p.stat().st_mtime_ns for p in source_files) > code.stat().st_mtime_ns:
            with atomic_open(code, 'wb') as stream:
                with tarfile.open(fileobj=stream, mode='w:gz') as archive:
                    for path in sorted(source_files):
                        archive.add(path, arcname=str(path.relative_to(ROOT)))
        files = {'code/restore_code.tar.gz': code,
                 'checkpoint/' + Path(simulation.latest_checkpoint_path).name: Path(simulation.latest_checkpoint_path)}
        for path in output.iterdir():
            if (path.is_file() and path.suffix in ('.json', '.csv', '.jsonl')
                    and not path.name.startswith(('checkpoint_', 'latest_', 's3_', 'model_replies'))):
                files['results/' + path.name] = path
        docs = Path(getattr(args, 'docs_dir', output))
        if docs != output and docs.exists():
            for path in docs.iterdir():
                if path.is_file() and path.suffix in ('.json', '.csv'):
                    files['docs/' + path.name] = path
        cache = Path(args.data_cache_dir)
        if cache.exists():
            for path in cache.rglob('*'):
                if path.is_file() and not path.name.endswith(('.tmp', '.lock')):
                    files['cache/' + str(path.relative_to(cache))] = path
        if args.fulltext_index:
            index = Path(args.fulltext_index)
            stamp = [index.stat().st_size, index.stat().st_mtime_ns]
            previous = inventory['files'].get('fulltext/index.json', {})
            if previous.get('stamp') == stamp and inventory.get('fulltext_body'):
                body = Path(inventory['fulltext_body'])
                files['fulltext/index.json'] = index
            else:
                data = read_json(index)
                body = index.parent / data['text_file']
                # A portable index always resolves its text file relative to itself.
                data['text_file'] = 'texts.utf8'
                normalized = temporary / 'index.json'
                write_json(normalized, data)
                files['fulltext/index.json'] = normalized
            inventory['fulltext_body'] = str(body)
            files['fulltext/texts.utf8'] = body
        def upload_snapshot_file(item):
            logical, path = item
            stat = path.stat()
            old = inventory['files'].get(logical)
            stamp = [stat.st_size, stat.st_mtime_ns]
            if logical == 'fulltext/index.json':
                source_stat = Path(args.fulltext_index).stat()
                stamp = [source_stat.st_size, source_stat.st_mtime_ns]
            if old and old.get('stamp') == stamp:
                entry = old
            else:
                entry = upload_verified(client, path, bucket, f'{prefix}/objects/{generation}/{logical}')
                entry['stamp'] = stamp
            return logical, entry

        # Finish every file's readback before publishing the recovery manifest.
        with ThreadPoolExecutor(max_workers=16) as executor:
            entries = dict(executor.map(upload_snapshot_file, files.items()))
        replies = Path(args.log_dir) / 'model_replies.jsonl'
        offset = inventory['offset']
        segments = list(inventory['replies'])
        if replies.exists():
            chunk_path = temporary / 'replies.jsonl'
            # Serialize against the existing reply writer; never upload a half record.
            with simulation.llm.response_write_lock, replies.open('rb') as source:
                source.seek(offset)
                with chunk_path.open('wb') as target:
                    for line in source:
                        if not line.endswith(b'\n'):
                            break
                        json.loads(line)
                        target.write(line)
            length = chunk_path.stat().st_size
            if length:
                entry = upload_verified(client, chunk_path, bucket,
                                        f'{prefix}/replies/{generation}/{offset:020d}-{offset + length:020d}.jsonl')
                segments.append(dict(entry, start=offset, end=offset + length))
                offset += length
        manifest = dict(version=1, bucket=bucket, files=entries, replies=segments,
                        checkpoint='checkpoint/' + Path(simulation.latest_checkpoint_path).name,
                        experiment_id=getattr(args, 'experiment_id', simulation.experiment_name))
        manifest_path = temporary / 'manifest.json'
        write_json(manifest_path, manifest)
        entry = upload_verified(client, manifest_path, bucket, f'{prefix}/snapshots/{generation}.json')
        # A reader sees either the previous complete snapshot or this complete one.
        latest = temporary / 'latest.json'
        write_json(latest, dict(manifest=entry['key']))
        upload_verified(client, latest, bucket, f'{prefix}/latest.json')
        write_json(inventory_path, dict(destination=[bucket, prefix], files=entries,
                                        replies=segments, offset=offset,
                                        fulltext_body=inventory.get('fulltext_body')))


def restore_native_snapshot(uri, destination):
    """Restore files and ordered complete reply segments before model construction."""
    import boto3
    from utopia.utils.data_utils import atomic_open
    bucket, key = s3_location(uri)
    client = boto3.client('s3')
    key = key if key.endswith('.json') else key + '/latest.json'
    payload = json.loads(client.get_object(Bucket=bucket, Key=key)['Body'].read())
    if 'manifest' in payload:
        payload = json.loads(client.get_object(Bucket=bucket, Key=payload['manifest'])['Body'].read())
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for logical, entry in payload['files'].items():
        target = (destination / logical).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f'Invalid snapshot path: {logical}')
        with atomic_open(target, 'wb') as stream:
            client.download_fileobj(bucket, entry['key'], stream)
            stream.flush()
            if os.fstat(stream.fileno()).st_size != entry['bytes']:
                raise IOError(f'Incomplete snapshot object: {logical}')
    offset = 0
    with atomic_open(destination / 'model_replies.jsonl', 'wb') as stream:
        for entry in payload['replies']:
            if entry['start'] != offset:
                raise ValueError('Reply snapshot has a gap or duplicate segment')
            body = client.get_object(Bucket=bucket, Key=entry['key'])['Body']
            try:
                while chunk := body.read(8 * 1024 * 1024):
                    stream.write(chunk)
            finally:
                body.close()
            offset = entry['end']
            if stream.tell() != offset:
                raise IOError('Incomplete reply segment')
    write_json(destination / 'restore_manifest.json', payload)
    return str(destination / payload['checkpoint'])


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
    for name in CACHE_INPUTS:
        path = root / name
        result[name] = {"bytes": path.stat().st_size}
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
        cache = pickle.load(stream)
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
    write_json(output, record)
    return record


def verified_cache_manifest():
    record = json.loads((ROOT / CACHE_MANIFEST).read_text())
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
    import subprocess
    import sys

    import numpy as np
    import pandas as pd
    import torch

    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "run_manifest.json")
    manifest = {}
    if os.path.exists(path):
        with open(path) as f:
            manifest = json.load(f)

    def run_git(cmd):
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
            "git_commit": run_git(["git", "rev-parse", "HEAD"]),
            "git_dirty_files": run_git(["git", "status", "--short"]),
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        try:
            import sentence_transformers
            import torch
            import transformers
            import vllm
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
        # Resume-safe: llm_call_stats accumulate across restarts instead of
        # being overwritten by a short resume run's counters.
        new_stats = extra.get("llm_call_stats")
        old_stats = manifest.get("llm_call_stats")
        if isinstance(new_stats, dict) and isinstance(old_stats, dict):
            extra = dict(extra)
            extra["llm_call_stats"] = {
                k: (old_stats.get(k, 0) or 0) + (new_stats.get(k, 0) or 0)
                for k in set(old_stats) | set(new_stats)}
        manifest.update(extra)

    write_json_atomic(path, manifest, indent=2, default=str,
                      trailing_newline=False, streaming=True)
    return path
