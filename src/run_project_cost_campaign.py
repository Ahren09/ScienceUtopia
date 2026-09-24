"""Frozen, non-preemptive eight-GPU queue for the direct-output-cost control.

Reuses the scale-expansion command builder and existing GPU ownership helpers.
Four small pilot worlds must pass independent accounting validation before any
300-researcher world starts. Only this controller's processes may be stopped.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from analyze_project_cost import analyze, scientific_arguments, validate_run
from copy import deepcopy
from james_server_profiles import tp1_server_spec
from run_james_ec2_queue import (
    Lock, atomic_write, boot_id, constrained_output_probe, endpoint_ready,
    file_hash, json_bytes, load_shared_probe_builder, proc_identity, query_gpus,
    read_reservation, reservation_path, reserve_pair, stop_owned, utc,
)
from run_scale_expansion import STAGES, build_command, experiment_id_for

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "question": "Does k=2 versus k=1 change allocation and review workload when direct production "
                "and resubmission fees are both zero?",
    "baseline_commit": "2bb7e6917418f2722b2217cc5453188f51734662",
    "historical_commit": "9ee31940c33f6c01a245f132ebcb489992921b47",
    "cost_audit": "Neither baseline nor historical execution debits the configured paper "
                  "production fee. Keep that effective zero, waive resubmission fees in all "
                  "new cells, and retain resubmission and review decisions.",
    "seeds": [1, 2, 3, 4, 5], "papers_per_project": [1, 2], "budget_modes": ["track", "fixed"],
    "population": 300, "years": 8, "institutions": 60, "researchers_per_institution": 5,
    "start_year": 2016, "conferences": 6, "funding_panel_max_apps": 25,
    "fixed_budget_slots_fraction": 0.2, "review_score_mode": "float",
    "acceptance_tiebreak": "seeded",
    "primary_endpoints": ["Share of original founders receiving any grant by year 8",
                          "Share of all reviews spent on resubmissions"],
    "survival_invariant": "Initial endowment 100 minus at most 8*10 research costs leaves "
                          "20, above the exit threshold. With zero direct and application fees "
                          "and nonnegative income, every founder must remain active. This is "
                          "a structural invariant, not an empirical test of no survival effect.",
    "primary_contrasts": ["k2-k1 under track", "k2-k1 under fixed",
                          "(k2-k1 under fixed) - (k2-k1 under track)"],
    "inference": "Pair all comparisons by seed. Report individual seed effects and descriptive "
                 "paired bootstrap intervals. Exact paired sign-flip diagnostics have minimum "
                 "two-sided p=0.0625 with five seeds. Holm adjustment across six primary contrasts. "
                 "Other endpoints are exploratory. Do not pool historical runs.",
    "scope": "Direct-output-cost sensitivity. Annual research costs and research duration follow "
             "the original rules. Realized total career spending is not held equal.",
    "pilot": {"seeds": [0], "population": 25, "years": 2, "cells": 4},
    "gates": "All pilot exit codes, final checkpoints, founder rosters, code hashes and complete "
             "resource ledgers must pass before full scale. Any job failure stops the campaign "
             "with logs and checkpoints preserved. Resume only from complete yearly checkpoints.",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_identity():
    require(not git("status", "--porcelain", "--untracked-files=normal"), "Repository is not clean")
    files = git("ls-files").splitlines()
    return git("rev-parse", "HEAD"), {
        path: file_hash(ROOT / path) for path in files if (ROOT / path).is_file()
    }


def make_plan(args):
    require(args.python.absolute() == Path(sys.executable).absolute(),
            "Create and run the plan using its selected Python environment")
    commit, sources = source_identity()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for stage, seeds, institutions, years in (
            ("costpilot", [0], 5, 2), ("costcontrol", PROTOCOL["seeds"], 60, 8)):
        launch_args = SimpleNamespace(
            population="university_only", num_institutions=institutions, num_years=years,
            num_conferences=6, start_year=2016, batch_size=32, funding_panel_max_apps=25,
            slots_frac=0.2, reviewer_capacity=None, reviewer_matching="random",
            legacy_scoring=False, always_rerun=False, vllm_url=["http://127.0.0.1:1/v1"],
        )
        for seed in seeds:
            for cell in STAGES[stage]:
                command = build_command(cell, seed, launch_args)
                command[0] = str(args.python.absolute())
                from utopia.arguments import parse_arguments
                from utopia.config import SIMULATION_CONFIG
                from project_cost_accounting import resolve_cost_policy
                resolved_args = parse_arguments(command[2:])
                resolved_config = deepcopy(SIMULATION_CONFIG)
                policy = resolve_cost_policy(resolved_args, resolved_config)
                resolved_config["project_cost_accounting"] = policy
                resolved_config["conference"]["resubmission_cost"] = policy["resubmission_cost"]
                jobs.append({"stage": stage, "cell": cell, "seed": seed,
                             "years": years, "population": institutions * 5,
                             "experiment_id": experiment_id_for(command), "command": command,
                             "scientific_args": scientific_arguments(vars(resolved_args)),
                             "resolved_config": resolved_config})
    inputs = ["outputs/cache_scievo.pkl", "outputs/thenlper_gte-small/cache_embeddings.pt"]
    plan = {
        "schema_version": 1, "created_utc": utc(), "directory": str(directory),
        "git_commit": commit, "source_sha256": sources, "protocol": PROTOCOL,
        "seeds": PROTOCOL["seeds"], "jobs": jobs, "python": str(args.python.absolute()),
        "hf_home": str(args.hf_home.resolve()), "server_spec": tp1_server_spec(),
        "input_sha256": {path: file_hash(ROOT / path) for path in inputs},
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                             ("vllm", "torch", "transformers", "datasets", "sentence-transformers")},
        "gpu_policy": "Use all eight physical GPUs as they become idle. Never preempt another job.",
        "client_device": "CPU for cached retrieval and embeddings", "max_parallel": 8,
    }
    path = directory / "plan.json"
    atomic_write(path, json_bytes(plan), exclusive=True)
    print(path, flush=True)
    return plan


def runtime_check(plan):
    require(Path(plan["python"]).absolute() == Path(sys.executable).absolute(),
            "Controller interpreter differs from selected child interpreter")
    commit, sources = source_identity()
    require(commit == plan["git_commit"] and sources == plan["source_sha256"],
            "Frozen source state changed")
    for path, digest in plan["input_sha256"].items():
        require(file_hash(ROOT / path) == digest, f"Frozen input changed: {path}")
    for name, version in plan["runtime_versions"].items():
        require(importlib.metadata.version(name) == version, f"Runtime changed: {name}")
    for name in ("vllm", "torch", "transformers"):
        require(plan["runtime_versions"][name] == plan["server_spec"][name + "_version"],
                f"Unsupported server runtime: {name}")
    snapshot = Path(plan["hf_home"]) / "hub/models--Qwen--Qwen3-32B/snapshots"
    require((snapshot / plan["server_spec"]["revision"]).is_dir(), "Pinned model is not cached")


def server_command(plan, port):
    spec = plan["server_spec"]
    python = Path(plan["python"])
    return [
        str(python), str(python.parent / "vllm"), "serve", spec["model"],
        "--revision", spec["revision"], "--tokenizer-revision", spec["revision"],
        "--dtype", spec["dtype"], "--tensor-parallel-size", "1",
        "--distributed-executor-backend", "mp", "--max-model-len", str(spec["max_model_len"]),
        "--max-num-seqs", str(spec["max_num_seqs"]),
        "--gpu-memory-utilization", str(spec["gpu_memory_utilization"]),
        "--seed", str(spec["seed"]), "--reasoning-parser", spec["reasoning_parser"],
        "--enforce-eager", "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(spec["max_num_batched_tokens"]),
        "--host", "127.0.0.1", "--port", str(port),
    ]


def pending_training_gpus(proc_root=Path("/proc")):
    """Respect EC2's existing DLM jobs during CPU preparation between GPU phases.

    Those controllers explicitly carry ``--gpu N`` before allocating CUDA.
    Zero memory at that moment does not mean the device is available to us.
    """
    claimed = set()
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            parts = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if b"src.unified.run_plan" not in parts:
            continue
        try:
            device = int(parts[parts.index(b"--gpu") + 1])
            require(0 <= device < 8, "Unknown training GPU assignment")
            claimed.add(device)
        except (ValueError, IndexError, RuntimeError):
            # An active recognized controller with an unknown assignment may
            # claim any GPU. Do not guess from its current CUDA memory usage.
            claimed.update(range(8))
    return claimed


def available_gpus():
    claimed = pending_training_gpus()
    return [gpu for gpu in query_gpus() if gpu.index not in claimed]


class Campaign:
    def __init__(self, plan, plan_path):
        self.plan, self.path = plan, plan_path
        self.directory = Path(plan["directory"])
        self.lock_root = Path("/tmp") / f"scienceutopia-gpu-locks-{os.getuid()}"
        self.lock_root.mkdir(mode=0o700, exist_ok=True)
        self.servers = []
        self.stopping = False
        self.state_path = self.directory / "state.json"
        self.state = {"status": "waiting_for_idle_gpu", "plan_sha256": file_hash(plan_path),
                      "controller": proc_identity(os.getpid()), "started_utc": utc(),
                      "jobs": {}, "servers": [], "used_gpu_ids": []}

    def save(self):
        self.state["updated_utc"] = utc()
        atomic_write(self.state_path, json_bytes(self.state))

    def environment(self, devices):
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=devices, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   HF_HOME=self.plan["hf_home"], HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   WANDB_MODE="disabled")
        # Existing wheel binaries locate libpython in the environment library dir.
        lib = str(Path(self.plan["python"]).parent.parent / "lib")
        env["LD_LIBRARY_PATH"] = lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return env

    def spawn(self, command, env, log):
        stream = log.open("a")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            stream.close()
        identity = proc_identity(process.pid)
        require(identity is not None and identity["session"] == process.pid,
                "Child did not start in its own process session")
        return process, identity

    def reservation(self, server, phase):
        path = reservation_path(self.lock_root, server["gpu"].uuid)
        record = read_reservation(path)
        require(record and record["token"] == server["record"]["token"], "Reservation owner changed")
        record.update(phase=phase, server_identity=server["record"].get("identity"),
                      client_identity=server["record"].get("client_identity"), updated_utc=utc())
        atomic_write(path, json_bytes(record))

    def start_server(self, reserved):
        gpus, locks = reserved
        gpu = gpus[0]
        record = {"gpu": asdict(gpu), "status": "launch_intent", "token": uuid.uuid4().hex,
                  "identity": None, "client_identity": None, "started": time.time()}
        record["gpu"]["pids"] = sorted(gpu.pids)
        server = {"gpu": gpu, "locks": locks, "record": record, "process": None, "client": None}
        self.servers.append(server)
        self.state["servers"].append(record)
        if gpu.index not in self.state["used_gpu_ids"]:
            self.state["used_gpu_ids"].append(gpu.index)
        for port in range(19200, 19300):
            lock = Lock(self.lock_root / f"port-{port}.lock")
            if not lock.acquire():
                continue
            try:
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", port))
                locks.append(lock)
                break
            except OSError:
                lock.close()
        else:
            raise RuntimeError("No free local inference port")
        record["port"] = port
        command = server_command(self.plan, port)
        intent = {"schema_version": 1, "token": record["token"], "phase": "launch_intent",
                  "boot_id": boot_id(), "controller_identity": self.state["controller"],
                  "server_identity": None, "client_identity": None, "queue_directory": str(self.directory),
                  "plan_sha256": self.state["plan_sha256"], "launch_command": command}
        atomic_write(reservation_path(self.lock_root, gpu.uuid), json_bytes(intent))
        self.save()
        fresh = {item.uuid: item for item in available_gpus()}
        require(gpu.uuid in fresh and fresh[gpu.uuid].idle, "GPU became busy or claimed before launch")
        process, identity = self.spawn(command, self.environment(str(gpu.index)),
                                       self.directory / f"server-gpu{gpu.index}-{port}.log")
        server["process"] = process
        record.update(identity=identity, status="starting")
        self.reservation(server, "active")
        self.save()
        print(f"Started Qwen3-32B on GPU {gpu.index}, port {port}", flush=True)

    def start_job(self, server, job):
        record = server["record"]
        command = list(job["command"])
        command[command.index("--vllm_url") + 1] = f"http://127.0.0.1:{record['port']}/v1"
        log_dir = ROOT / "outputs/logs" / job["experiment_id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        env = self.environment("")
        env["PYTHONHASHSEED"] = str(job["seed"])
        env["UTOPIA_DATA_CACHE_DIR"] = str(ROOT / "data/cache" / job["experiment_id"])
        receipt_path = log_dir / "campaign_command.json"
        if receipt_path.exists():
            prior = json.loads(receipt_path.read_text())
            require(prior["git_commit"] == self.plan["git_commit"]
                    and prior["plan_sha256"] == self.state["plan_sha256"],
                    "Existing artifacts belong to another frozen campaign")
        else:
            for directory in ("docs", "checkpoints"):
                existing = ROOT / "outputs" / directory / job["experiment_id"]
                require(not existing.exists() or not any(existing.iterdir()),
                        "Existing world has no matching campaign receipt")
            atomic_write(receipt_path, json_bytes(
                {"command": command, "git_commit": self.plan["git_commit"],
                 "plan_sha256": self.state["plan_sha256"], "gpu": server["gpu"].index,
                 "runtime": self.plan["server_spec"]}), exclusive=True)
        # State precedes the fork. A crash in this small window requires an
        # explicit restart inspection rather than guessing whether it launched.
        entry = {**job, "status": "launch_intent", "identity": None, "started": time.time(),
                 "gpu": server["gpu"].index}
        self.state["jobs"][job["experiment_id"]] = entry
        self.save()
        process, identity = self.spawn(command, env, log_dir / "simulation.log")
        server["client"] = (process, job)
        entry.update(status="running", identity=identity)
        record["client_identity"] = identity
        self.reservation(server, "active")
        self.save()
        print(f"Started {job['experiment_id']} on GPU {server['gpu'].index}", flush=True)

    def recover(self):
        if not self.state_path.exists():
            return
        prior = json.loads(self.state_path.read_text())
        require(prior["plan_sha256"] == self.state["plan_sha256"], "State belongs to another plan")
        for job in prior["jobs"].values():
            if job["status"] == "launch_intent" and job["identity"] is None:
                raise RuntimeError("Ambiguous prior client launch. Inspect before resuming.")
            if job.get("identity"):
                require(not stop_owned(job["identity"]), "Prior owned client could not stop")
            if job["status"] == "complete":
                original = next(j for j in self.plan["jobs"] if j["experiment_id"] == job["experiment_id"])
                validate_run(ROOT, original, self.plan["git_commit"])
                self.state["jobs"][job["experiment_id"]] = job
        for server in prior["servers"]:
            if server["status"] == "released":
                continue
            require(server.get("identity") is not None, "Ambiguous prior server launch")
            require(not stop_owned(server["identity"]), "Prior owned server could not stop")
            path = reservation_path(self.lock_root, server["gpu"]["uuid"])
            reservation = read_reservation(path)
            require(reservation and reservation["token"] == server["token"], "Prior reservation changed")
            reservation.update(phase="released", updated_utc=utc())
            atomic_write(path, json_bytes(reservation))
        self.state["prior_attempt"] = prior
        self.state["used_gpu_ids"] = prior.get("used_gpu_ids", [])

    def cleanup(self):
        errors = []
        for server in self.servers:
            record = server["record"]
            stopped = True
            try:
                for identity in (record.get("client_identity"), record.get("identity")):
                    if not identity:
                        continue
                    try:
                        require(not stop_owned(identity), "Owned child could not stop")
                    except Exception as error:
                        stopped = False
                        errors.append(str(error))
                if record.get("identity") and stopped:
                    try:
                        self.reservation(server, "released")
                        record["status"] = "released"
                    except Exception as error:
                        errors.append(str(error))
                # Uncertain or live owned sessions retain their durable record.
            finally:
                for lock in server["locks"]:
                    try:
                        lock.close()
                    except Exception as error:
                        errors.append(str(error))
        if errors:
            self.state.update(status="cleanup_failed", cleanup_errors=errors)
        self.save()
        require(not errors, "Cleanup incomplete: " + "; ".join(errors))

    def run(self):
        lock = Lock(self.directory / "controller.lock")
        require(lock.acquire(), "Campaign controller is already running")
        initialized = False
        try:
            runtime_check(self.plan)
            self.recover()
            self.save()
            initialized = True
            builder, binding = load_shared_probe_builder(SimpleNamespace(worktree=ROOT))
            for stage in ("costpilot", "costcontrol"):
                queue = [j for j in self.plan["jobs"] if j["stage"] == stage
                         and self.state["jobs"].get(j["experiment_id"], {}).get("status") != "complete"]
                self.state["stage"] = stage
                self.save()
                while queue or any(s["client"] for s in self.servers):
                    require(not self.stopping, "Controller interrupted")
                    # Source check is inexpensive, and prevents later jobs from
                    # silently using code synced by another task during the run.
                    require(git("rev-parse", "HEAD") == self.plan["git_commit"]
                            and not git("status", "--porcelain", "--untracked-files=normal"),
                            "Frozen checkout changed during campaign")
                    for server in self.servers:
                        record, process = server["record"], server["process"]
                        require(process.poll() is None, f"Inference server exited: GPU {server['gpu'].index}")
                        if record["status"] == "starting":
                            require(time.time() - record["started"] < 1800, "Server startup timed out")
                            if endpoint_ready(record["identity"], record["port"]):
                                probe = constrained_output_probe(record["identity"], record["port"], builder)
                                probe["shared_client_builder"] = binding
                                atomic_write(self.directory / f"probe-{record['port']}.json", json_bytes(probe))
                                record["status"] = "ready"
                        if server["client"]:
                            child, job = server["client"]
                            rc = child.poll()
                            require(time.time() - self.state["jobs"][job["experiment_id"]]["started"]
                                    < 96 * 3600, f"World exceeded 96 hours: {job['experiment_id']}")
                            if rc is not None:
                                entry = self.state["jobs"][job["experiment_id"]]
                                entry.update(returncode=rc, completed_utc=utc())
                                require(rc == 0, f"World exited {rc}: {job['experiment_id']}")
                                validate_run(ROOT, job, self.plan["git_commit"])
                                entry["status"] = "complete"
                                server["client"] = None
                                record["client_identity"] = None
                                self.reservation(server, "active")
                                self.save()
                                print(f"Validated {job['experiment_id']}", flush=True)
                        if queue and record["status"] == "ready" and server["client"] is None:
                            self.start_job(server, queue.pop(0))
                    desired = min(8, len(queue) + sum(s["client"] is not None for s in self.servers))
                    if len(self.servers) < desired:
                        reserved = reserve_pair(available_gpus(), self.lock_root,
                                                query=available_gpus, device_count=1)
                        if reserved:
                            self.start_server(reserved)
                    self.state["status"] = "running" if any(s["client"] for s in self.servers) else "waiting_for_idle_gpu_or_server"
                    self.state["pending_jobs"] = len(queue)
                    self.save()
                    time.sleep(15)
                self.state[stage + "_validated"] = True
                self.save()
            self.state["status"] = "analyzing"
            self.save()
            result = analyze(ROOT, self.plan, ROOT / "outputs/docs/project_cost_control/results")
            self.state.update(status="complete", report=result, completed_utc=utc())
            self.save()
        except BaseException as error:
            # Preserve earlier ownership records if a changed checkout/runtime
            # prevents safe recovery before the new controller initializes.
            if initialized:
                self.state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_utc=utc())
                self.save()
            raise
        finally:
            try:
                if initialized:
                    self.cleanup()
            finally:
                lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--python", type=Path, default=Path(sys.executable))
    prepare.add_argument("--hf-home", type=Path, required=True)
    prepare.add_argument("--directory", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
        return
    campaign = Campaign(json.loads(args.plan.read_text()), args.plan)
    def stop(signum, frame):
        campaign.stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    campaign.run()


if __name__ == "__main__":
    main()
