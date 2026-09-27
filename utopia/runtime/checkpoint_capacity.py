#!/usr/bin/env python3
"""CPU-only capacity check for one exact 512 MiB synthetic checkpoint.

This measures a synthetic file shape, not future scientific horizon capacity.
It sends no inference and never constructs a queue controller.
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import file_sha256 as digest
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

sys.dont_write_bytecode = True
import utopia.runtime.experiment_queue as queue


def main():
    if not __debug__:
        raise RuntimeError("Capacity checks require non-optimized Python")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise RuntimeError("Refusing to overwrite a capacity check report")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.nice(15)
    # Bound this process's virtual address space. This does not reserve CPU,
    # physical RAM, or disk bandwidth.
    address_limit = 8 * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
    sources = [Path(__file__), Path(queue.__file__)]
    before = {str(p): digest(p) for p in sources}
    result = {
        "schema": "scienceutopia.checkpoint_capacity_resource.v1",
        "scope": "Synthetic CPU-only resource/boundary check. No scientific admission.",
        "python": sys.version,
        "source_before": before,
        "address_space_limit_bytes": address_limit,
        "file_bytes": 512 * 1024 * 1024,
        "checks": [],
    }
    started = time.monotonic()
    exit_code = 1
    temporary_path = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="capacity-512-", dir=args.report.parent
        ) as directory:
            private = Path(directory)
            temporary_path = private
            worktree = private / "worktree"
            worktree.mkdir()
            run = private / "outputs" / "synthetic_checkpoint"
            run.mkdir(parents=True)
            path = run / "checkpoint_year_1.json"
            launch_ns = time.time_ns()
            prefix = (
                b'{"initial_industry_count":2,"initial_university_count":1200,'
                b'"phase":5,"year":1,"synthetic_padding":"'
            )
            suffix = b'"}'
            remaining = result["file_bytes"] - len(prefix) - len(suffix)
            with path.open("xb") as stream:
                stream.write(prefix)
                block = b"x" * (1024 * 1024)
                while remaining:
                    n = min(remaining, len(block))
                    stream.write(block[:n])
                    remaining -= n
                stream.write(suffix)
                stream.flush()
                os.fsync(stream.fileno())
            assert path.stat().st_size == result["file_bytes"]
            artifact = {
                "path": str(path),
                "min_bytes": 1,
                "max_json_bytes": result["file_bytes"],
                "equals": {
                    "/initial_industry_count": 2,
                    "/initial_university_count": 1200,
                    "/phase": 5,
                    "/year": 1,
                },
            }
            job = {
                "family": "influx",
                "output_roots": [str(run)],
                "artifacts": [artifact],
            }
            # Direct artifact-gate check with a synthetic filesystem only.
            # Plan parsing and pending-job admission have their own contract tests.
            plan = queue.Plan(
                b"{}",
                {},
                digest(__file__),
                worktree,
                private,
                private / "env/bin/python",
                private / "hf",
            )
            gate_start = time.monotonic()
            evidence = queue.artifact_gate(job, plan, started_ns=launch_ns)
            assert len(evidence) == 1 and evidence[0]["size"] == result["file_bytes"]
            assert evidence[0]["sha256"] == digest(path)
            result["checks"].append(
                {
                    "name": "complete_parse_and_predicates_at_exact_512_MiB",
                    "passed": True,
                    "seconds": time.monotonic() - gate_start,
                    "artifact_binding": evidence[0],
                }
            )
            with path.open("ab") as stream:
                stream.write(b" ")
            rejected = False
            rejection = None
            try:
                queue.artifact_gate(job, plan, started_ns=launch_ns)
            except queue.QueueError as exc:
                rejection = str(exc)
                rejected = "JSON gate exceeds byte allowance:" in rejection
            assert rejected, (
                "An artifact one byte above the allowance was not rejected by the size gate"
            )
            result["checks"].append(
                {
                    "name": "reject_512_MiB_plus_one_before_parsing",
                    "passed": True,
                    "error": rejection,
                }
            )
        result["source_after"] = {str(p): digest(p) for p in sources}
        assert before == result["source_after"]
        result["status"] = "passed"
        exit_code = 0
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["peak_rss_native_units"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        result["peak_rss_units"] = "KiB on Linux, bytes on macOS"
        result["temporary_artifacts_removed"] = (
            temporary_path is None or not temporary_path.exists()
        )
        write_json_file(result, args.report, mode="x", trailing_newline=True, indent=2)
        print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
