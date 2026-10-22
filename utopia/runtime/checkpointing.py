"""Portable year-boundary snapshots for explicitly opted-in, new runs.

Only data is serialized. Models, tokenizers, transports and live clients are never
pickled. A caller owns each output/cache directory and starts one writer per cell.
"""
from contextlib import contextmanager
import base64
import gzip
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

from utopia.utils.data_utils import atomic_open

FORMAT = "scienceutopia-portable-checkpoint-v1"


class PortableCheckpointError(BaseException):
    """A persistence/provenance error must escape scientific fallback handlers."""


class PortableStop(BaseException):
    """Graceful interruption, leaving the last completed year authoritative."""


class RunStop:
    """Cooperative wallclock/signals, checked between phases and HTTP requests."""
    def __init__(self, max_seconds=None):
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.deadline = None if max_seconds is None else time.monotonic() + max_seconds
        self.reason = None
        self.failure = None

    def fail(self, error):
        if self.failure is None:
            self.failure = error

    def request_stop(self, signum=None, frame=None):
        self.reason = "stop requested" if signum is None else f"signal {signum}"

    def check(self):
        if self.failure is not None:
            raise self.failure
        if self.reason is not None:
            raise PortableStop(self.reason)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise PortableStop("wallclock budget reached")

    @contextmanager
    def signals(self):
        previous = {s: signal.signal(s, self.request_stop)
                    for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield self
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def validate_binding(binding):
    required = {"protocol", "cell", "model", "model_revision", "corpus_sha256",
                "science_config", "seed"}
    if not isinstance(binding, dict) or required - binding.keys():
        raise PortableCheckpointError(f"Portable binding requires {sorted(required)}")
    if any(not binding[k] for k in required - {"seed"}) or type(binding["seed"]) is not int:
        raise PortableCheckpointError("Portable binding contains an empty identity")
    return json.loads(canonical_json(binding))


def pack_state(value, _memo=None):
    """Data-only JSON, preserving shared mutable histories across native serializers."""
    memo = {} if _memo is None else _memo
    reference = {}
    if isinstance(value, (dict, list, set)):
        previous = memo.get(id(value))
        if previous is not None:
            return {"type": "ref", "id": previous[1]}
        ref = len(memo)
        # Keep a strong reference: domain to_dict() creates temporary dictionaries.
        memo[id(value)] = (value, ref)
        reference = {"id": ref}
    if isinstance(value, dict):
        return {"type": "dict", **reference,
                "items": [[pack_state(k, memo), pack_state(v, memo)] for k, v in value.items()]}
    if isinstance(value, (tuple, set, list)):
        return {"type": type(value).__name__, **reference,
                "items": [pack_state(v, memo) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    domain = (type(value).__module__, type(value).__name__)
    if domain == ("utopia.agents.research_direction", "ResearchDirection"):
        return {"type": "ResearchDirection", "state": pack_state(value.to_dict(), memo)}
    if domain == ("utopia.agents.funding_agents", "FundingProgram"):
        return {"type": "FundingProgram", "state": pack_state(vars(value), memo)}
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(value, numpy.generic):
        return pack_state(value.item(), memo)
    raise PortableCheckpointError(f"Unsupported checkpoint value: {type(value).__name__}")


def unpack_state(value, _memo=None):
    if not isinstance(value, dict):
        return value
    memo = {} if _memo is None else _memo
    kind = value["type"]
    if kind == "ref":
        return memo[value["id"]]
    if kind in ("dict", "list", "set"):
        result = {"dict": dict, "list": list, "set": set}[kind]()
        if "id" in value:
            memo[value["id"]] = result
        if kind == "dict":
            result.update((unpack_state(k, memo), unpack_state(v, memo)) for k, v in value["items"])
        elif kind == "list":
            result.extend(unpack_state(v, memo) for v in value["items"])
        else:
            result.update(unpack_state(v, memo) for v in value["items"])
        return result
    if kind == "tuple":
        return tuple(unpack_state(v, memo) for v in value["items"])
    if kind == "ResearchDirection":
        from utopia.agents.research_direction import ResearchDirection
        return ResearchDirection.from_dict(unpack_state(value["state"], memo))
    if kind == "FundingProgram":
        from utopia.agents.funding_agents import FundingProgram
        program = FundingProgram.__new__(FundingProgram)
        program.__dict__.update(unpack_state(value["state"], memo))
        return program
    raise PortableCheckpointError(f"Unknown checkpoint value type: {kind}")


def capture_rng():
    state = {"python": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        name, keys, pos, gaussian, cached = numpy.random.get_state()
        state["numpy"] = (name, keys.tolist(), pos, gaussian, cached)
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state().tolist()
        # Do not initialize CUDA merely to save a client-side checkpoint.
        if torch.cuda.is_initialized():
            state["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    if "numpy" in state:
        import numpy as np
        name, keys, pos, gaussian, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    if "torch_cpu" in state:
        import torch
        torch.set_rng_state(torch.tensor(state["torch_cpu"], dtype=torch.uint8))
        if "torch_cuda" in state:
            if torch.cuda.device_count() != len(state["torch_cuda"]):
                raise PortableCheckpointError("CUDA RNG device count changed")
            torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state["torch_cuda"]])
    random.setstate(state["python"])


def write_portable_checkpoint(path, data):
    try:
        with atomic_open(path, "wb", private=True, cleanup_on_error=True) as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                # Closing TextIOWrapper would close raw before atomic_open fsync.
                encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                for chunk in encoder.iterencode({"format": FORMAT, "state": pack_state(data)}):
                    compressed.write(chunk.encode())
    except PortableCheckpointError:
        raise
    except Exception as error:
        raise PortableCheckpointError(f"Cannot persist checkpoint: {error}") from error


def read_portable_checkpoint(path):
    try:
        with gzip.open(path, "rt") as stream:
            data = json.load(stream)
        if data.get("format") != FORMAT:
            raise PortableCheckpointError("Not a portable checkpoint")
        return unpack_state(data["state"])
    except PortableCheckpointError:
        raise
    except Exception as error:
        raise PortableCheckpointError(f"Cannot read checkpoint: {error}") from error


def simulation_replay_paths(output_dir, docs_dir, num_years, *, table_names=()):
    """Exact replayable files for Simulation and FundingFeedbackMixin.

    Enumerate the entire horizon, including absent future files, so replay also
    removes incomplete-year output. Checkpoints, response caches, run manifests,
    physical request/audit journals, runtime logs and immutable corpus/embedding
    caches are deliberately excluded. Subclasses add only their own metric files.
    """
    output, docs = Path(output_dir), Path(docs_dir)
    paths = {"paper_tracker": output / "paper_tracker.json"}
    for filename in ("population_blueprint.csv", "network_growth_log.jsonl",
                     "resource_ledger.jsonl", "resource_balances.jsonl",
                     "simulation_report.json"):
        paths["output/" + filename] = output / filename
    for filename in ("initial_world.json", "integrity_checks.json"):
        paths["docs/" + filename] = docs / filename
    for year in range(1, num_years + 1):
        for stem in ("funding_applications", "funding_costs"):
            filename = f"{stem}_year_{year}.jsonl"
            paths["output/" + filename] = output / filename
        filename = f"mechanisms_year_{year}.json"
        paths["docs/" + filename] = docs / filename
    for table in table_names:
        for suffix in ("parquet", "csv"):
            filename = f"{table}.{suffix}"
            paths["docs/" + filename] = docs / filename
    return paths


def capture_outputs(paths):
    """Capture only caller-declared, cell-owned outputs. Never include caches."""
    result = {}
    for name, path in paths.items():
        root = Path(path)
        if root.is_dir():
            files = {}
            for child in sorted(root.rglob("*")):
                if child.is_symlink():
                    raise PortableCheckpointError(f"Symlink in replay output: {child}")
                if child.is_file():
                    files[str(child.relative_to(root))] = base64.b64encode(child.read_bytes()).decode()
            result[name] = {"kind": "directory", "files": files}
        else:
            result[name] = {"kind": "file", "content":
                            base64.b64encode(root.read_bytes()).decode() if root.exists() else None}
    return result


def restore_outputs(paths, state):
    """Roll back uncommitted metrics, including append-only files and rewrites."""
    from utopia.utils.data_utils import write_bytes_atomic
    if set(paths) != set(state):
        raise PortableCheckpointError("Replay output names changed")
    for name, record in state.items():
        root = Path(paths[name])
        if record["kind"] == "directory":
            root.mkdir(parents=True, exist_ok=True)
            files = record["files"]
            for relative in files:
                part = Path(relative)
                if part.is_absolute() or ".." in part.parts:
                    raise PortableCheckpointError("Invalid replay output relative path")
            for child in root.rglob("*"):
                if child.is_symlink():
                    raise PortableCheckpointError(f"Symlink in replay output: {child}")
                if child.is_file() and str(child.relative_to(root)) not in files:
                    child.unlink()
            for relative, content in files.items():
                write_bytes_atomic(root / relative, base64.b64decode(content), private=True)
        elif record["content"] is None:
            root.unlink(missing_ok=True)
        else:
            write_bytes_atomic(root, base64.b64decode(record["content"]), private=True)
