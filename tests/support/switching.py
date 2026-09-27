"""CPU regressions for native direction selection, cache replay and observations.

    PYTHONHASHSEED=42 python -B -m unittest discover -s src -p test_switching_propensity.py -v

Default tests compile unmodified native definitions when local Torch is absent.
They use explicitly scripted transport/embeddings; they are not science runs.
UTOPIA_ACTUAL_MODULE_TESTS=1 additionally exercises the imported simulator, real
logger/RAG, frozen CPU MiniLM, full rounds and checkpoints in the private env.
"""

from __future__ import annotations

import utopia.runtime.provenance as provenance

import utopia.funding.feedback as feedback

from utopia.utils.paths import project_root

import ast

from collections import Counter, defaultdict

from contextlib import contextmanager, redirect_stdout

from copy import deepcopy

from dataclasses import dataclass

import hashlib

import io

import json

import logging

import os

from pathlib import Path

import random

import re

import sys

import tempfile

import time

from types import ModuleType, SimpleNamespace

import unittest

from unittest.mock import patch

import numpy as np

import utopia.experiments.switching_propensity as jp

import utopia.agents.switching_policy as policy

import tests.support.simulation as legacy

import utopia.funding.sequential as sequential

ROOT = project_root(__file__)


def native_module_fixture():
    """Exact native classes/functions; replace only unavailable service imports."""
    directions = ModuleType("propensity_test_native_directions")
    sys.modules[directions.__name__] = directions
    directions.__dict__.update(
        dataclass=dataclass,
        BaseModel=legacy.SchemaStub,
        time=time,
        random=random,
        logger=logging.getLogger("propensity-test"),
        traceback=__import__("traceback"),
    )
    tree = ast.parse((ROOT / "utopia/agents/research_direction.py").read_text())
    names = {
        "ResearchDirection",
        "ResearchDirectionSelection",
        "build_direction_prompt",
        "create_research_directions_batch",
    }
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    body += [
        n
        for n in tree.body
        if (isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names)
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id in ("AVAILABLE_DIRECTIONS", "DIRECTIONS_DICT")
                for t in n.targets
            )
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(ROOT / "utopia/agents/research_direction.py"),
            "exec",
        ),
        directions.__dict__,
    )
    sim_module = ModuleType("propensity_test_native_simulator")
    sim_module.__dict__.update(legacy.original_namespace())
    sim_module.ResearchDirection = directions.ResearchDirection
    sim_module.AVAILABLE_DIRECTIONS = directions.AVAILABLE_DIRECTIONS
    sim_module.create_research_directions_batch = (
        directions.create_research_directions_batch
    )
    for path, definitions in (
        ("utopia/agents/base_agent.py", ["SimulationAgent", "MultiAgentEcosystem"]),
        (
            "utopia/agents/researcher_agents.py",
            ["ProgramApplication", "UniversityResearcher", "IndustryResearcher"],
        ),
        ("utopia/simulation.py", ["Simulation"]),
    ):
        legacy.original_definitions(path, definitions, sim_module.__dict__)
    return sim_module, directions


class VectorFixture:
    """Deterministic nonzero scripted vectors; never a substitute for MiniLM."""

    def encode(self, texts, **kwargs):
        return np.stack(
            [
                np.frombuffer(
                    hashlib.sha256(t.encode()).digest(), dtype=np.uint8
                ).astype(float)
                + 1
                for t in texts
            ]
        )


def native_tracker():
    ns = {
        "np": np,
        "hashlib": hashlib,
        "os": os,
        "SentenceTransformer": lambda name, **kwargs: VectorFixture(),
    }
    legacy.original_definitions(
        "utopia/metrics/embedding_tracker.py", ["EmbeddingTracker"], ns
    )
    return ns["EmbeddingTracker"]()


class ScriptedDirections:
    """Writes explicitly scripted request evidence to exercise audit ingestion."""

    def __init__(self, path, derive_seed, failures=()):
        self.request_audit = SimpleNamespace(path=Path(path))
        self.request_audit.path.touch()
        self.derive_seed = derive_seed
        self.failures = failures
        self.calls = []

    def generate_batch(self, prompts, **kwargs):
        self.calls.append((deepcopy(prompts), deepcopy(kwargs)))
        result = []
        for index, prompt in enumerate(prompts):
            topic = re.findall(r"### Direction \d+: ([^\n]+)", prompt)[0]
            mode = self.failures[index] if index < len(self.failures) else None
            response = (
                None
                if mode == "parse"
                else {"topic": "not_in_the_menu"}
                if mode == "validation"
                else {
                    "topic": topic,
                    "detailed_focus": "Scripted project detail.",
                    "reason": "Scripted reason.",
                }
            )
            jp.append_record(
                self.request_audit.path,
                {
                    "event": "request_started",
                    "seed_ctx": list(kwargs["seed_ctx"]),
                    "item_index": index,
                    "attempt": 0,
                    "request_seed": self.derive_seed(42, *kwargs["seed_ctx"], index, 0),
                    "fixture_only": True,
                },
            )
            result.append((response, []))
        return result


def reduced_scientific_state(sim):
    """Fixture-only state view; actual-module tests use full native serializers."""
    return provenance.digest(
        jp.canonical_state(
            [
                {
                    "id": a.id,
                    "resources": a.resources,
                    "active": a.is_active,
                    "memory_bank": a.memory_bank,
                    "project_start_year": a.project_start_year,
                    "project_end_year": a.project_end_year,
                    "direction": a.newest_direction["direction"].topic
                    if a.newest_direction
                    else None,
                }
                for a in sim.ecosystem.agent_population.values()
            ]
        )
    )


def direction_simulation(
    directory, module, lookup, cell="LN", initial_cache=None, failures=()
):
    cls = jp.simulation_class(module.Simulation)
    sim = object.__new__(cls)
    sim.args = SimpleNamespace(docs_dir=str(directory), seed=42)
    Path(directory).mkdir(parents=True)
    sim.simulator_module, sim.direction_lookup = module, lookup
    sim.propensity_cell, sim.initialize_only, sim.initial_cache = (
        cell,
        cell is None,
        initial_cache,
    )
    sim.initial_records, sim.direction_events, sim.propensity_years = [], [], {}
    sim.experiment_name = "default_propensity_cpu_fixture"
    sim._direction_fallback_stats = {}
    sim.llm = ScriptedDirections(
        Path(directory) / "fixture_audit.jsonl", module.derive_seed, failures
    )
    sim.ecosystem = module.MultiAgentEcosystem({})
    sim.paper_tracker = SimpleNamespace(
        papers_by_id={}, get_papers_by_decision=lambda aid: ([], [])
    )
    for i in range(3):
        a = module.UniversityResearcher(
            f"founder_{i}",
            f"institution_{i}",
            funding_level=100,
            expertise=list(lookup.values())[i : i + 3],
            llm=sim.llm,
            exploration_strategy="balanced",
        )
        sim.ecosystem.add_agent(a)
    sim.founder_ids = tuple(sim.ecosystem.agent_population)
    sim.funding_tracker = module.FundingTracker(funding_allocation_mode="fixed")
    sim.funding_tracker.start_cycle(1, sim.ecosystem.agent_population)
    sim.embedding_tracker = native_tracker()
    sim.neutral_history = jp.NeutralHistory(
        sim.embedding_tracker,
        list(sim.ecosystem.agent_population.values()),
        list(lookup.values()),
    )
    sim.world_state_hash = lambda: reduced_scientific_state(sim)
    return sim


def prepare_year(sim, year):
    rows = sim._resource_rows()
    sim.propensity_years[year] = {
        "year": year,
        "cell": sim.propensity_cell,
        "founder_count": len(rows),
        "year_start_resources": rows,
        "after_phase0_resources": rows,
        "year_start_active_ids": [r["agent_id"] for r in rows if r["active"]],
        "annual_charges": [],
    }


def cache_from_sim(sim):
    return {
        "schema_version": 1,
        "protocol": jp.PROTOCOL,
        "status": "complete",
        "seed": 42,
        "founder_count": len(sim.founder_ids),
        "choice_count": len(sim.initial_records),
        "founder_ids": list(sim.founder_ids),
        "pre_choice_state_hash": sim.pre_choice_state_hash,
        "initial_state_hash": sim.initial_state_hash,
        "records": sim.initial_records,
        "records_sha256": provenance.digest(sim.initial_records),
        "inputs": {"fixture_only": True},
    }


class AuditedPropensityLLM(legacy.SequentialScriptedLLM):
    """Scripted SDK, actual RequestAudit and frozen sequential transport; no HTTP/GPU."""

    def generate_batch(
        self,
        prompts,
        seed_ctx=None,
        response_format=None,
        system_prompt=sequential.DEFAULT_SYSTEM_PROMPT,
        temperature=0.7,
        max_tokens=2048,
        **kwargs,
    ):
        if seed_ctx[0] == "phase5_funding_eval":
            raise AssertionError(
                "Sequential funding must intercept the native compact batch"
            )
        if seed_ctx[0] == "phase0_resubmit":
            self.calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
            answers = [({"resubmitted_papers": []}, []) for _ in prompts]
        else:
            answers = super().generate_batch(
                prompts,
                seed_ctx=seed_ctx,
                response_format=response_format,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        for index, (prompt, (answer, _)) in enumerate(zip(prompts, answers)):
            response = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=json.dumps(answer)),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
            )
            self.request_audit.create(
                lambda response=response, **unused: response,
                seed_ctx=seed_ctx,
                item_index=index,
                attempt=0,
                model=jp.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "seed": self._request_seed(seed_ctx, index, 0),
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            )
        return answers


def tp1_manifest_runtime_fixture(directory):
    """A labelled file attestation fixture; no server is started."""
    from utopia.runtime.server_profiles import TP1_PLAN_PROFILE, tp1_server_spec

    path = Path(directory) / "server.runtime.fixture.json"
    record = dict(
        tp1_server_spec(),
        server_profile=TP1_PLAN_PROFILE,
        endpoint="http://127.0.0.1:18084/v1",
        gpu_ids=[7],
        host="test-only",
        pid=123,
        started_utc="fixture",
        launch_command=["labelled-file-fixture"],
    )
    provenance.write_json(path, record)
    server = provenance.server_provenance(
        path, record["endpoint"], runtime_profile=jp.TP1_PROFILE_NAME
    )
    return {
        "server_provenance": server,
        "args": {"vllm_url": record["endpoint"]},
        "inputs": {
            "server_runtime_profile": jp.TP1_PROFILE_NAME,
            "runtime_amendment": deepcopy(jp.RUNTIME_AMENDMENT),
            "server_runtime_hash": server["runtime_hash"],
        },
    }


def funding_manifest_fixture(out, compact, *, initialize_only):
    """Bind real fixture sidecars; only native lifecycle records below are test stubs."""
    out = Path(out)
    years = 0 if initialize_only else jp.YEARS
    native = (
        {
            "status": "initialization_only_complete",
            "years_completed": 0,
            "scientific_years_completed": 0,
        }
        if initialize_only
        else {"status": "complete", "years_completed": years}
    )
    provenance.write_json(out / "run_manifest.json", native)
    provenance.write_json(out / "funding_validation.summary.json", compact.summary())
    runtime = tp1_manifest_runtime_fixture(out)
    return {
        "status": "complete",
        "kind": "initialization_only" if initialize_only else "condition",
        "protocol": jp.PROTOCOL,
        "seed": 42,
        "founder_count": jp.FOUNDERS,
        "scientific_years_completed": years,
        "args": dict(runtime["args"], output_dir=str(out)),
        "inputs": dict(runtime["inputs"], source_files_sha256=jp.source_identity()),
        "server_provenance": runtime["server_provenance"],
        "funding_selection_protocol": jp.FUNDING_SELECTION_PROTOCOL,
        "funding_output_representation": jp.OUTPUT_REPRESENTATION,
        "funding_baseline": deepcopy(feedback.FUNDING_BASELINE),
        "sequential_source_binding": feedback.sequential_source_binding(ROOT),
        "funding_validation": compact.summary(),
        "request_audit": json.loads(
            (out / "llm_request_audit.summary.json").read_text()
        ),
        "funding_sequential": json.loads(
            (out / feedback.SEQUENTIAL_SUMMARY_FILE).read_text()
        ),
        "funding_sequential_audits": {
            name: provenance.file_hash(out / name)
            for name in (
                feedback.SEQUENTIAL_AUDIT_FILE,
                feedback.SEQUENTIAL_SUMMARY_FILE,
            )
        },
        "funding_application_evidence": feedback.funding_ledger_evidence(out, years),
        "sequential_audit_valid": True,
        "funding_application_evidence_valid": True,
    }


class FullYearPropensityLLM(legacy.FullYearScriptedLLM):
    """Scripted native phase inputs, compact rankings and audit-ingestion records."""

    def __init__(self, audit_path, derive_seed):
        super().__init__()
        self.request_audit = SimpleNamespace(path=Path(audit_path))
        self.request_audit.path.touch()
        self.derive_seed = derive_seed

    def generate_batch(self, prompts, seed_ctx, **kwargs):
        if seed_ctx[0] == "phase5_funding_eval":
            self.calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
            results = [
                (
                    {
                        "ranked_application_ids": [
                            int(index)
                            for index in re.findall(r"### Application (\d+)\n", prompt)
                        ]
                    },
                    [],
                )
                for prompt in prompts
            ]
        elif seed_ctx[0] == "phase0_resubmit":
            self.calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
            results = [({"resubmitted_papers": []}, []) for _ in prompts]
        else:
            results = super().generate_batch(prompts, seed_ctx, **kwargs)
        for index in range(len(prompts)):
            jp.append_record(
                self.request_audit.path,
                {
                    "event": "request_started",
                    "seed_ctx": list(seed_ctx),
                    "item_index": index,
                    "attempt": 0,
                    "request_seed": self.derive_seed(42, *seed_ctx, index, 0),
                    "fixture_only": True,
                },
            )
        return results


@contextmanager
def fixture_population(count):
    """Keep producer and admission contracts aligned for small CPU fixtures."""
    from utopia.runtime import switching_evidence

    with (
        patch.object(jp, "FOUNDERS", count),
        patch.object(switching_evidence, "FOUNDERS", count),
    ):
        yield
