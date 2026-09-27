"""CPU invariant tests, including the ORIGINAL simulator funding phase.

Run from the repository root:
    python -m unittest tests.funding.test_feedback -v

The supplied review checkout is partial and the local environment lacks Torch
and Pydantic. The harness compiles selected, unmodified AST definitions from
the original source, with postponed type annotations. Only unavailable schema,
graph and LLM services are stubs. Funding proposal construction, program choice,
ranking validation/selection, costs, resource updates, history and attrition
execute original code. This is unit/integration evidence, NOT a Qwen run.
"""

from __future__ import annotations

import utopia.funding.feedback as feedback
import utopia.runtime.provenance as provenance

from utopia.utils.paths import project_root
from utopia.utils.data_utils import write_json_atomic, write_json_gzip_atomic

import ast

from collections import Counter, defaultdict

from contextlib import contextmanager, redirect_stdout

from copy import deepcopy

import io

import gzip

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

from typing import Dict, List, Set, Union

import unittest

from unittest.mock import patch

import numpy as np

import pandas as pd

ROOT = project_root(__file__)

import utopia.experiments.funding_feedback as jm

import utopia.funding.sequential as sequential

from utopia.config import SIMULATION_CONFIG

from utopia.metrics.tracker import calculate_gini_coefficient


def original_definitions(relative_path, names, namespace):
    """Execute original definitions, not copies/reimplementations of their bodies."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise AssertionError(f"Requested original definitions missing from {path}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)


class SchemaStub:
    @classmethod
    def model_json_schema(cls):
        return {}  # Schema transport is outside this CPU test's scope.


class DirectionStub:
    def __init__(self, topic):
        self.topic = topic


def original_namespace():
    ns = {
        "__name__": __name__,
        "logger": logging.getLogger("original-funding-test"),
        "SIMULATION_CONFIG": SIMULATION_CONFIG,
        "BaseModel": SchemaStub,
        "ResearchDirection": DirectionStub,
        "random": random,
        "time": time,
        "np": np,
        "pd": pd,
        "os": __import__("os"),
        "json": json,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "List": List,
        "Dict": Dict,
        "Set": Set,
        "Union": Union,
        "nx": SimpleNamespace(DiGraph=dict),
        "tqdm": lambda iterable, **kwargs: iterable,
        "calculate_gini_coefficient": calculate_gini_coefficient,
        "write_json_atomic": write_json_atomic,
        "write_json_gzip_atomic": write_json_gzip_atomic,
    }
    original_definitions("utopia/utils/seeding.py", ["derive_seed"], ns)
    original_definitions(
        "utopia/agents/base_agent.py", ["SimulationAgent", "MultiAgentEcosystem"], ns
    )
    original_definitions(
        "utopia/agents/researcher_agents.py",
        ["ProgramApplication", "UniversityResearcher", "IndustryResearcher"],
        ns,
    )
    original_definitions(
        "utopia/agents/funding_agents.py",
        ["FundingEvaluationResponse", "FundingAgency"],
        ns,
    )
    original_definitions("utopia/data/tracker.py", ["FundingTracker"], ns)
    original_definitions(
        "utopia/simulation.py",
        ["charge_application_costs", "build_population_blueprint", "Simulation"],
        ns,
    )
    return ns


ORIGINAL = original_namespace()

UniversityResearcher = ORIGINAL["UniversityResearcher"]

FundingAgency = ORIGINAL["FundingAgency"]

Simulation = ORIGINAL["Simulation"]


class FixtureLLM:
    """Explicit scripted transport fixture. It never represents Qwen results."""

    def __init__(self):
        self.calls = []
        self.fail_evaluation = False

    def generate_batch(self, prompts, seed_ctx, **kwargs):
        self.calls.append((deepcopy(prompts), seed_ctx))
        if seed_ctx[0] == "phase5_funding_apps":
            return [
                (
                    {
                        "submit": True,
                        "research_proposal": "Test a new algorithm.",
                        "relevant_projects": [0],
                    },
                    [],
                )
                for _ in prompts
            ]
        if self.fail_evaluation:
            raise RuntimeError("scripted evaluation failure")
        # There are two applicants per program. Rank the same applicant first
        # across cells so monetary mechanics can be checked independently of P.
        return [
            (
                {
                    "ranked_applications": [
                        {
                            "application_id": 0,
                            "applicant_id": "founder_0",
                            "rank": 1,
                            "reason": "fixture",
                        },
                        {
                            "application_id": 1,
                            "applicant_id": "founder_1",
                            "rank": 2,
                            "reason": "fixture",
                        },
                    ]
                },
                [],
            )
            for _ in prompts
        ]


def make_simulation(
    directory, cell="P1F1", *, base=False, balance=100, cost=0, num_authors=2
):
    cls = Simulation if base else feedback.simulation_class(Simulation)
    sim = object.__new__(cls)  # Skip RAG/server construction, use real phase below.
    sim.mechanisms = feedback.Mechanisms.from_cell(cell)
    sim.funding_feedback_years = {}
    sim.args = SimpleNamespace(
        seed=8401,
        funding_application_cost=cost,
        funding_panel_max_apps=25,
        log_funding_applications=True,
    )
    sim.ecosystem = ORIGINAL["MultiAgentEcosystem"]({})
    agents = [
        UniversityResearcher(
            f"founder_{i}",
            f"institution_{i}",
            funding_level=balance,
            expertise=[DirectionStub("algorithms")],
        )
        for i in range(num_authors)
    ]
    for agent in agents:
        sim.ecosystem.add_agent(agent)
    agency = object.__new__(FundingAgency)
    ORIGINAL["SimulationAgent"].__init__(agency, reputation=8, agent_id="agency")
    agency.can_author = agency.can_review = False
    agency.funding_programs = {
        name: SimpleNamespace(
            program_id=name,
            name=name,
            topics=["algorithms"],
            research_directions=[DirectionStub("algorithms")],
            funding_rate=0.5,
        )
        for name in ("PROGRAM_A", "PROGRAM_B")
    }
    sim.ecosystem.add_agent(agency)
    papers = {
        f"paper_{i}": {
            "id": f"paper_{i}",
            "author_id": a.id,
            "author_type": "university",
            "title": f"Accepted historical title {i}",
            "abstract": "Historical abstract.",
            "topics": ["algorithms"],
            "maturity": 1,
            "status": "accept",
        }
        for i, a in enumerate(agents)
    }
    sim.paper_tracker = SimpleNamespace(
        get_papers_dataframe=lambda year: pd.DataFrame(papers.values()),
        papers_by_id={
            key: SimpleNamespace(to_dict=lambda paper=p: dict(paper))
            for key, p in papers.items()
        },
    )
    sim.conference_system = SimpleNamespace(
        authors_to_accepted_papers={
            a.id: [papers[f"paper_{i}"]] for i, a in enumerate(agents)
        },
        authors_to_rejected_papers={},
        get_all_statistics=lambda: {},
    )
    sim.funding_tracker = ORIGINAL["FundingTracker"]("fixed")
    sim.funding_tracker.start_cycle(1, sim.ecosystem.agent_population)
    sim.agent_tracker = SimpleNamespace(record_agent_resources=lambda *args: None)
    sim.output_dir = str(directory)
    sim.funding_budget_mode = "track"
    sim.funding_allocation_mode = "fixed"
    sim.weighted_funding_assignment = False
    sim.experiment_name = "default_funding_feedback"
    sim.llm = FixtureLLM()
    return sim, agents, agency


def run_phase(sim):
    result = {}
    with redirect_stdout(io.StringIO()):
        sim._run_phase_5_update_funding(1, [], result)
    return result


def make_panel_simulation(directory, cell="P1F1", *, base=False):
    """53 applicants/program forces balanced panels of 18, 18 and 17."""
    sim, agents, agency = make_simulation(directory, cell, base=base, num_authors=53)
    sim.args.seed = 42
    # Read the rates passed by the stock simulator, rather than config defaults.
    tree = ast.parse((ROOT / "utopia/simulation.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_add_funding_agencies"
    )
    rates = {}
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FundingAgency"
        ):
            keywords = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in node.keywords
                if kw.arg in ("agency_name", "default_funding_rate")
            }
            rates[keywords["agency_name"]] = keywords["default_funding_rate"]
    assert rates == {"NSF": 0.23, "DARPA": 0.1}
    for program, agency_name in zip(agency.funding_programs.values(), ("NSF", "DARPA")):
        program.funding_rate = rates[agency_name]
    sim.llm = CompactFundingScriptedLLM()
    return sim, agents, agency


def panel_seed_import():
    """Load the exact stock pure seed function without importing Torch locally."""
    module = ModuleType("utopia.utils.seeding")
    module.derive_seed = ORIGINAL["derive_seed"]
    return patch.dict(sys.modules, {module.__name__: module})


class FullYearScriptedLLM:
    """Deterministic transport fixture only; all simulator modules remain real."""

    model_name = "SCRIPTED_TEST_FIXTURE_NOT_QWEN"

    def __init__(self):
        self.calls = []
        self.call_stats = {}
        self.chosen = set()

    def generate_batch(self, prompts, seed_ctx, **kwargs):
        self.calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
        answers = []
        for prompt in prompts:
            phase = seed_ctx[0]
            if phase == "phase1_directions":
                choices = re.findall(
                    r"### Direction \d+: ([^\n]+)\n.*?- Expected Project Duration: (\d+)",
                    prompt,
                    re.S,
                )
                topic, _ = min(choices, key=lambda pair: int(pair[1]))
                answer = dict(
                    topic=topic,
                    detailed_focus="Scripted CPU regression project.",
                    reason="Select an explicitly permitted candidate.",
                )
            elif phase == "phase2_intentions":
                answer = {
                    "intention": "Investigate algorithms and evaluate reproducibility."
                }
            elif phase == "phase2_submissions":
                choices = re.findall(r"arXiv ID: ([^\n]+)", prompt)
                chosen = next(p for p in choices if p not in self.chosen)
                self.chosen.add(chosen)
                conference = re.search(r'^- "([^"]+)": primary topics:', prompt, re.M)[
                    1
                ]
                answer = dict(
                    id=choices.index(chosen) + 1,
                    arxiv_id=chosen,
                    conference=conference,
                    reason="Scripted valid selection.",
                )
            elif phase == "phase3_reviews":
                answer = {
                    "overall_score": 6,
                    "justification": "Scripted review fixture.",
                }
            elif phase == "phase5_funding_apps":
                answer = dict(
                    submit=True,
                    research_proposal="Scripted research proposal.",
                    relevant_projects=[],
                )
            elif phase == "phase5_funding_eval":
                entries = re.findall(
                    r"### Application (\d+)\nApplicant ID: ([^\n]+)", prompt
                )
                answer = {
                    "ranked_applications": [
                        dict(
                            application_id=int(index),
                            applicant_id=aid,
                            rank=rank + 1,
                            reason="Scripted application order.",
                        )
                        for rank, (index, aid) in enumerate(entries)
                    ]
                }
            elif phase.startswith("phase2_citations_"):
                answer = (
                    {"self_citations": []}
                    if phase.endswith("_self")
                    else {"citations": []}
                )
            else:
                raise AssertionError(f"Unexpected scripted-test request phase: {phase}")
            answers.append((answer, []))
        return answers


class CompactFundingScriptedLLM(FullYearScriptedLLM):
    """Same explicit scripted preference order, emitted in the new wire format.

    This fixture tests deterministic decoding/award equivalence for a supplied
    order. It does not claim Qwen samples identical orders across representations.
    """

    def generate_batch(self, prompts, seed_ctx, **kwargs):
        answers = super().generate_batch(prompts, seed_ctx, **kwargs)
        if seed_ctx[0] == "phase5_funding_eval":
            return [
                (
                    {
                        "ranked_application_ids": [
                            row["application_id"]
                            for row in answer["ranked_applications"]
                        ]
                    },
                    history,
                )
                for answer, history in answers
            ]
        return answers


class SequentialScriptedLLM(FullYearScriptedLLM):
    """Exercise actual sequential transport/guard with explicit scripted SDK responses."""

    def __init__(self, request_audit=None, fail_step=None, include_history=False):
        super().__init__()
        self.run_seed = 42
        self.call_stats = dict(
            n_prompts=0,
            n_first_attempt_success=0,
            n_retries=0,
            n_failures=0,
            elapsed_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )
        self.enable_thinking = True
        self.max_concurrent_requests = 1
        self.client = SimpleNamespace(max_retries=5)
        self.request_audit = request_audit or SimpleNamespace(
            path=Path("CPU_FIXTURE_ONLY"), summary=lambda: {"n_clients": 1}
        )
        if request_audit is not None:
            self.model_name = (
                provenance.MODEL
            )  # Explicit scripted SDK under the real audit, not Qwen output.
        self.fail_step = fail_step
        self.include_history = include_history
        self.selection_requests = []

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
        answers = super().generate_batch(
            prompts,
            seed_ctx,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        if seed_ctx[0] == "phase5_funding_apps" and self.include_history:
            for answer, _ in answers:
                answer["relevant_projects"] = [0]
        return answers

    def _request_seed(self, seed_ctx, item_index, attempt):
        return ORIGINAL["derive_seed"](42, *seed_ctx, item_index, attempt)

    def _build_extra_body(self, response_format):
        return {
            "structured_outputs": {"json": response_format["json_object"]["schema"]},
            "chat_template_kwargs": {"enable_thinking": True},
        }

    def _create_completion(self, *, seed_ctx, item_index, attempt, **kwargs):
        self.selection_requests.append(
            deepcopy(
                {
                    "seed_ctx": seed_ctx,
                    "item_index": item_index,
                    "attempt": attempt,
                    **kwargs,
                }
            )
        )
        self.calls.append((deepcopy(kwargs["messages"]), seed_ctx, {}))
        allowed = kwargs["extra_body"]["structured_outputs"]["json"]["properties"][
            "next_application_id"
        ]["enum"]
        # Native scripted preference = original application order. Invalid test
        # answers exercise failure behavior; they are never repaired here.
        chosen = -1 if seed_ctx[-1] == self.fail_step else min(allowed)
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"next_application_id": chosen})
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )
        if hasattr(self.request_audit, "create"):
            return self.request_audit.create(
                lambda **unused: response,
                seed_ctx=seed_ctx,
                item_index=item_index,
                attempt=attempt,
                **kwargs,
            )
        return response


@contextmanager
def fixture_request_scope(directory):
    """Real RequestAudit with an explicit fixed-token CPU fixture."""
    from utopia.models.request_audit import request_audit_scope

    class Tokenizer:
        chat_template = "CPU_FIXTURE_ONLY"

        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(20))}

    with request_audit_scope(
        Path(directory) / "llm_request_audit.jsonl",
        model_name=provenance.MODEL,
        _tokenizer=Tokenizer(),
    ) as audit:
        yield audit.bind(provenance.MODEL)


def write_nonfunding_request_fixture(directory):
    """A true closed audit with a scripted non-funding request, for empty-panel tests."""
    directory = Path(directory)
    if (directory / "llm_request_audit.summary.json").exists():
        return json.loads((directory / "llm_request_audit.summary.json").read_text())
    with fixture_request_scope(directory) as audit:
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )
        audit.create(
            lambda **kwargs: response,
            model=provenance.MODEL,
            seed_ctx=("CPU_NONFUNDING_FIXTURE",),
            temperature=0.7,
            max_tokens=8192,
            messages=[{"role": "user", "content": "Explicit CPU fixture only."}],
        )
    return audit.summary()


def zero_sequential_fixture(directory, years=feedback.YEARS):
    """Real installation/finalization proves zero panels; never synthesize a clean summary."""
    from utopia.funding.validation import install_funding_validation

    write_nonfunding_request_fixture(directory)
    with tempfile.TemporaryDirectory() as guard_dir:
        compact = install_funding_validation(
            FundingAgency, Path(guard_dir) / "compact.jsonl", compact=True
        )
        try:
            with feedback.sequential_funding_scope(
                FundingAgency, SequentialScriptedLLM, directory
            ) as handle:
                for year in range(1, years + 1):
                    handle.begin_year(year, ["NSF", "DARPA"])
                    for aid in ("NSF", "DARPA"):
                        agency = object.__new__(FundingAgency)
                        ORIGINAL["SimulationAgent"].__init__(
                            agency, reputation=8, agent_id=aid
                        )
                        agency.agency_name = aid
                        agency.funding_programs = {}
                        agency.get_funding_evaluation_prompts(
                            [],
                            submitted_papers_dict={},
                            panel_max_apps=25,
                            panel_seed=ORIGINAL["derive_seed"](
                                42, "funding_panels", year
                            ),
                        )
                    handle.end_year(year)
        finally:
            compact.restore()
    return handle.summary()


def initialize_actual_simulator(cls):
    import hashlib
    import torch
    import utopia.simulation as actual
    import utopia.agents.conference as conference_module
    from langchain_core.documents import Document
    from utopia.data.rag import RAG

    cls.actual = actual
    cls.conference_module = conference_module
    cls.pristine_conferences = deepcopy(conference_module.CONFERENCES_BY_CATEGORY)
    cls.torch = torch
    cls.temporary = tempfile.TemporaryDirectory(prefix="utopia-actual-module-")
    cls.fixture_root = Path(cls.temporary.name)
    cls.original_cwd = Path.cwd()
    cls.output = cls.fixture_root / "outputs"
    cls.output.mkdir()

    class FixtureEncoder:
        def encode(self, texts, **kwargs):
            # Hashing each string gives stable nonzero vectors without RNG or downloads.
            values = [list(hashlib.sha256(text.encode()).digest()) * 12 for text in texts]
            vectors = torch.tensor(values, dtype=torch.float32)
            return torch.nn.functional.normalize(vectors, dim=1)

    def load_documents(rag, data_path=None, num_years=2):
        rag.documents = [Document(page_content=f"Synthetic study {i} in algorithms and machine learning.",
            metadata={"id": f"fixture-{year}-{i}", "title": f"Fixture paper {i}",
                      "topics": ["algorithms", "machine learning"], "tags": ["cs.AI"],
                      "publication_year": year, "published": f"{year}-01-01"})
            for year in range(rag.start_year, rag.start_year + num_years) for i in range(64)]
        rag.id2docs = {doc.metadata["id"]: doc for doc in rag.documents}
        rag.id2year_pos = {doc.metadata["id"]: (doc.metadata["publication_year"], i % 64)
                           for i, doc in enumerate(rag.documents)}
        rag.document_status = pd.DataFrame([{"id": doc.metadata["id"], "status": "unsubmitted",
            "publication_year": doc.metadata["publication_year"]} for doc in rag.documents])
        rag.dataset_identity = {"dataset": "synthetic-test-fixture", "num_years": num_years}
        rag.document_identity = provenance.corpus_fingerprint(rag.documents)
        rag.sentence_transformer_model = FixtureEncoder()
        return rag.documents

    cls.document_patch = patch.object(RAG, "load_documents", load_documents)
    cls.document_patch.start()
    os.chdir(cls.fixture_root)


def restore_actual_simulator(cls):
    cls.document_patch.stop()
    os.chdir(cls.original_cwd)
    cls.temporary.cleanup()


def make_actual_simulation(self, name, cell=None):
    path = self.fixture_root / name
    args = self.actual.parse_arguments(
        [
            "--experiment_name",
            "default_james_cpu_fixture",
            "--experiment_stage",
            "mechanism",
            "--output-dir",
            str(path),
            "--seed",
            "42",
            "--population_mode",
            "university_only",
            "--num_institutions",
            "6",
            "--researchers_per_institution",
            "2",
            "--strategy_mix",
            "balanced",
            "--num_years",
            "2",
            "--num_conferences",
            "2",
            "--model",
            provenance.MODEL,
            "--always_rerun",
            "--log_funding_applications",
            "--funding_panel_max_apps",
            "25",
            "--funding_budget_mode",
            "track",
        ]
    )
    args.rag_device = "cpu"
    self.actual.set_seed(42)
    cls = (
        self.actual.Simulation
        if cell is None
        else feedback.simulation_class(self.actual.Simulation)
    )
    options = (
        {} if cell is None else {"mechanisms": feedback.Mechanisms.from_cell(cell)}
    )
    sim = cls(
        llm=FullYearScriptedLLM(),
        args=args,
        num_years=2,
        output_dir=args.checkpoint_dir,
        funding_allocation_mode="fixed",
        always_rerun=True,
        experiment_name=args.experiment_name,
        **options,
    )
    sim.initialize_agents()
    researchers = [
        a
        for a in sim.ecosystem.agent_population.values()
        if a.get_type() == "university"
    ]
    with patch.object(
        self.conference_module,
        "CONFERENCES_BY_CATEGORY",
        deepcopy(self.pristine_conferences),
    ):
        sim.conference_system = self.actual.ConferenceSystem(
            self.actual.select_conferences_for_simulation(
                researchers, num_conferences=2
            )
        )
    for agent in sim.ecosystem.agent_population.values():
        if getattr(agent, "resources", None) is not None:
            sim.agent_tracker.record_agent_resources(agent.id, agent.resources, 0)
    return sim, researchers
