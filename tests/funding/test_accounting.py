"""Local accounting tests, with explicitly mocked LLM/RAG/schema services.

Run: python -B -m pytest tests/funding/test_accounting.py

The native agent, Phase 0/2/5, annual debit, trackers and checkpoint methods are
compiled from their source definitions using the existing CPU fixture loader.
No simulator logic is copied. This tests bookkeeping, NOT model quality.
"""
from __future__ import annotations


from utopia.utils.paths import project_root

import ast
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import gzip
import hashlib
from functools import wraps
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = project_root(__file__)
import tests.support.simulation as native
from utopia.funding.accounting import CATEGORIES, ResourceLedger, apply_resource_change, cost_policy_tag, enter_project_production, project_id, resolve_cost_policy, stable_id
from utopia.arguments import parse_arguments
from utopia.config import SIMULATION_CONFIG
from utopia.constants import IMPORTANT_NOTES


def policy(**overrides):
    return resolve_cost_policy(SimpleNamespace(**overrides), SIMULATION_CONFIG)


def namespace():
    ns = native.original_namespace()
    ns["__file__"] = str(ROOT / "utopia/simulation.py")
    ns.update(Enum=Enum, IMPORTANT_NOTES=IMPORTANT_NOTES, traceback=traceback,
              dataclass=dataclass, field=field, Path=Path)
    native.original_definitions("utopia/agents/base_agent.py", ["RoundIntention"], ns)
    native.original_definitions("utopia/data/tracker.py", ["AgentTracker", "CitationTracker"], ns)
    native.original_definitions("utopia/data/paper_tracker.py", ["ArchivedPaper", "PaperTracker"], ns)
    native.original_definitions("utopia/agents/funding_agents.py", ["IndustryFundingSystem"], ns)
    ns.update(ConferenceSystem=ConferenceFixture, RAG=RAGFixture,
              RAGConfig=lambda **kw: SimpleNamespace(**kw))
    return ns


def reference_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        default=lambda v: sorted(v) if isinstance(v, set) else str(v)).encode()).hexdigest()


REFERENCE = json.loads((Path(__file__).parent / "fixtures/accounting_reference.json").read_text())


def fixed_hash_seed(function):
    """Golden prompts include native set iteration; verify in a fixed-hash process."""
    @wraps(function)
    def check(self):
        if os.environ.get('PYTHONHASHSEED') == '0':
            return function(self)
        selector = f'{Path(__file__).resolve()}::{type(self).__name__}::{function.__name__}'
        result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', selector],
            cwd=ROOT, env={**os.environ, 'PYTHONHASHSEED': '0'}, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return check


class ConferenceFixture:
    """Only venue admission/storage is mocked, not monetary behavior."""
    def __init__(self):
        conf = SimpleNamespace(conference_id="C", topics=["algorithms"], submitted_papers=[])
        conf.submit_paper = lambda paper, **kw: conf.submitted_papers.append(dict(paper)) or True
        self.conferences = [conf]
        self.conference_map = {"C": conf}

    def reset_all_for_new_year(self, year):
        pass

    def to_dict(self):
        return {}

    @classmethod
    def from_dict(cls, data):
        return cls()


class RAGFixture:
    """Deterministic retrieval fixture, never embeddings or production data."""
    def __init__(self, count=3):
        self.documents = [
            SimpleNamespace(page_content=f"Abstract {i}", metadata={
                "id": f"p{i}", "title": f"Paper {i}", "topics": ["algorithms"], "tags": ["cs.AI"],
            }) for i in range(count)]
        self.id2docs = {d.metadata["id"]: d for d in self.documents}
        self.document_status = pd.DataFrame(
            [{"id": d.metadata["id"], "status": "unsubmitted"} for d in self.documents],
            columns=["id", "status"])

    def load_documents(self, **kwargs):
        pass

    def build_knowledge_index(self):
        pass

    def set_current_year(self, year):
        pass

    def batch_retrieve(self, queries, **kwargs):
        return {"topk_indices": np.tile(np.arange(len(self.documents)), (len(queries), 1))}

    def to_dict(self):
        return {"count": len(self.documents), "status": self.document_status.to_dict("records")}

    @classmethod
    def from_dict(cls, data, **kwargs):
        result = cls(data["count"])
        result.document_status = pd.DataFrame(data["status"])
        return result


def entry(index):
    return {"id": index + 1, "arxiv_id": f"p{index}", "conference": "C", "reason": "fixture"}


class BatchLLMFixture:
    """Scripted responses for native control-flow tests, no real generation."""
    def __init__(self, results):
        self.results = results
        self.prompts = []

    def generate_batch(self, prompts, seed_ctx, **kwargs):
        self.prompts.extend(prompts)
        if seed_ctx[0] == "phase2_intentions":
            result = {"intention": "A scripted research intention."}
        elif seed_ctx[0] == "phase0_resubmit":
            result = {"resubmitted_papers": [{"arxiv_id": "p0", "conference": "C"},
                                            {"arxiv_id": "p1", "conference": "C"}]}
        else:
            result = self.results[min(seed_ctx[-1], len(self.results) - 1)]
        return [(deepcopy(result), []) for _ in prompts]


class SequentialLLMFixture:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def generate(self, prompt, response_format, **kwargs):
        self.prompts.append(prompt)
        schema = response_format["json_object"]["name"]
        result = {"intention": "A scripted research intention."} if schema == "round_intention" else self.result
        return deepcopy(result), []


def make_sim(ns, directory, *, k=1, mode="per_paper", fee=None, resubmission=0,
             audit=True, balance=100, results=None, sequential=False, count=3):
    sim = object.__new__(ns["Simulation"])
    sim.args = SimpleNamespace(
        production_cost_mode=mode, project_production_cost=fee,
        resubmission_cost=resubmission, log_resource_ledger=audit, papers_per_project=k,
    )
    sim.cost_policy = resolve_cost_policy(sim.args, SIMULATION_CONFIG)
    sim.resource_ledger = ResourceLedger() if audit else None
    sim.papers_per_project = k
    sim._collaboration_pairs = {}
    sim.ecosystem = ns["MultiAgentEcosystem"]({})
    direction = ns["ResearchDirection"]("algorithms")
    direction.keywords, direction.years = ["algorithms"], 1
    agent = ns["UniversityResearcher"]("founder", "institution", funding_level=balance,
                                     expertise=[direction], exploration_strategy="balanced")
    agent.project_start_year = agent.project_end_year = 1
    agent.newest_direction = {"direction": direction, "detailed_focus": "Focus", "reason": "fixture"}
    sim.ecosystem.add_agent(agent)
    sim.llm = (SequentialLLMFixture(results[0] if results else entry(0)) if sequential
               else BatchLLMFixture(results or ([entry(0)] if k == 1 else
                                                [{"submissions": [entry(i) for i in range(k)]}])))
    agent.llm = sim.llm
    sim.paper_tracker = ns["PaperTracker"](directory)
    sim.funding_tracker = ns["FundingTracker"]("fixed")
    sim.funding_tracker.start_cycle(1, sim.ecosystem.agent_population)
    sim.agent_tracker = ns["AgentTracker"]()
    sim.agent_tracker.record_agent_resources(agent.id, balance, 0)
    sim.citation_tracker = ns["CitationTracker"]()
    sim.industry_funding_system = ns["IndustryFundingSystem"]()
    sim.conference_system = ConferenceFixture()
    sim.rag = RAGFixture(count)
    sim.submitted_paper_ids = set()
    sim.experiment_name = "default_cost_fixture"
    sim.output_dir = str(directory)
    sim.yearly_results = []
    sim.initial_university_count, sim.initial_industry_count = 1, 0
    sim.num_years, sim.start_year = 2, 2016
    sim.funding_allocation_mode, sim.funding_budget_mode = "fixed", "track"
    sim.debug = sim.verbose = sim.use_langchain = False
    sim.max_retries = 2
    if audit:
        sim.resource_ledger.start_year(1, sim.ecosystem.agent_population)
    return sim, agent


def submit(sim, agent):
    with redirect_stderr(io.StringIO()):
        return sim._run_phase_2_submit_papers(1, {agent.id: agent.newest_direction}, [agent], [], {})


@contextmanager
def serializer_imports(ns, direction):
    """Unavailable module imports are replaced, native serializers still run."""
    definitions = {
        "utopia.agents.researcher_agents": {
            key: ns[key] for key in ("UniversityResearcher", "IndustryResearcher")},
        "utopia.agents.funding_agents": {
            "FundingAgency": ns["FundingAgency"], "FundingProgram": SimpleNamespace},
        "utopia.agents.research_direction": {
            "ResearchDirection": ns["ResearchDirection"], "DIRECTIONS_DICT": {direction.topic: direction}},
    }
    modules = {}
    for name, values in definitions.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module
    with patch.dict(sys.modules, modules):
        yield


class TestPrimaryAndLegacy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()

    def test_primary_k1_k2_zero_production_zero_resubmission_same_annual_cost(self):
        for k in (1, 2):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(self.ns, directory, k=k)
                papers = submit(sim, agent)
                self.assertEqual(len(papers), k)
                self.assertEqual(agent.resources, 100)
                sim._charge_annual_costs(1)
                row = sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)[0]
                self.assertEqual((row["opening_balance"], row["closing_balance"]), (100, 90))
                self.assertEqual(row["production_submission_cost"], 0)
                self.assertEqual(row["annual_research_cost"], 10)
                self.assertEqual(row["residual"], 0)
                output_events = sim.resource_ledger.transactions
                self.assertEqual(len([e for e in output_events if e["paper_id"]]), k)
                self.assertEqual({e["project_id"] for e in output_events}, {project_id(agent)})

    def test_primary_no_income_eight_year_survival_bound_and_ledger_continuity(self):
        """Native money/exit paths, mocked papers and no-income funding decisions.

        Every founder works all eight years, attaining the maximum annual debit:
        100 - 8 * 10 = 20 > the native Phase 5 exit threshold of 10. Any
        nonnegative funding or fewer research years can only raise that bound.
        This is a structural invariant test, not a model/survival-effect estimate.
        """
        class NoIncomeLLM:
            def generate_batch(self, prompts, seed_ctx, **kwargs):
                if seed_ctx[0] != "phase5_funding_apps":
                    raise AssertionError("No-income fixture must never evaluate awards")
                return [({"submit": False, "research_proposal": "No-income bound fixture.",
                          "relevant_projects": []}, []) for _ in prompts]

        self.assertEqual(SIMULATION_CONFIG["conference"]["annual_cost"], 10)
        final_balances = []
        for k in (1, 2):
            with self.subTest(k=k), tempfile.TemporaryDirectory() as directory:
                sim, founders, _ = native.make_simulation(
                    directory, base=True, balance=100, cost=0, num_authors=300)
                sim.args.production_cost_mode = "per_paper"
                sim.args.resubmission_cost = 0
                sim.args.log_resource_ledger = True
                sim.args.papers_per_project = sim.papers_per_project = k
                sim.cost_policy = resolve_cost_policy(sim.args, SIMULATION_CONFIG)
                self.assertEqual(sim.cost_policy["legacy_effective_per_paper_cost"], 0)
                self.assertEqual(sim.cost_policy["resubmission_cost"], 0)
                self.assertEqual(sim.args.funding_application_cost, 0)
                sim.resource_ledger = ResourceLedger()
                sim.llm = NoIncomeLLM()
                sim.conference_system.reset_all_for_new_year = lambda year: None
                population = sim.ecosystem.agent_population
                previous_papers = []

                for year in range(1, 9):
                    year_results = sim._setup_year(year)
                    # Native resubmission debit boundary, using last year's papers.
                    for agent, paper in previous_papers:
                        sim._charge_resubmission(agent, paper, year)
                    for agent in founders:
                        agent.project_start_year = agent.project_end_year = year
                    self.assertEqual(sim._prepare_project_production(year, founders), founders)
                    previous_papers = []
                    for agent in founders:
                        for output in range(k):
                            paper = {
                                "id": f"{agent.id}/year{year}/output{output}", "author_id": agent.id,
                                "project_start_year": year, "project_end_year": year,
                            }
                            sim._record_output_cost(agent, paper, year)
                            previous_papers.append((agent, paper))
                    sim._charge_annual_costs(year)
                    # Executes native zero application cost, no-award processing,
                    # and the actual resource-based researcher removal condition.
                    with redirect_stdout(io.StringIO()):
                        sim._run_phase_5_update_funding(year, [], year_results)
                    rows = sim.resource_ledger.close_year(year, population)
                    self.assertEqual(len(rows), 300)
                    for row in rows:
                        self.assertEqual(row["opening_balance"], 100 - 10 * (year - 1))
                        self.assertEqual(row["closing_balance"], 100 - 10 * year)
                        self.assertEqual(row["annual_research_cost"], 10)
                        self.assertTrue(row["founder"] and row["closing_active"])
                        for category in set(CATEGORIES) - {"annual_research_cost"}:
                            self.assertEqual(row[category], 0)
                        self.assertEqual(row["residual"], 0)
                        self.assertEqual(row["uncollected_cost"], 0)
                    # Opening the same ledger year again must not erase its history
                    # or permit a second annual charge. No full-year replay assumed.
                    before = sim.resource_ledger.to_dict()
                    sim.resource_ledger.start_year(year, population)
                    sim._charge_annual_costs(year)
                    self.assertEqual(sim.resource_ledger.to_dict(), before)
                    if year == 4:
                        sim.resource_ledger = ResourceLedger(json.loads(json.dumps(before)))
                        sim.resource_ledger.attach(population)
                        self.assertEqual(sim.resource_ledger.to_dict(), before)

                self.assertEqual(len(sim.resource_ledger.years), 8)
                transactions = sim.resource_ledger.transactions
                self.assertEqual(len({e["event_id"] for e in transactions}), len(transactions))
                debits = [e for e in transactions if e["delta"] < 0]
                self.assertEqual(len(debits), 300 * 8)
                self.assertEqual({(e["category"], e["delta"]) for e in debits},
                                 {("annual_research_cost", -10)})
                self.assertFalse(any(e["delta"] > 0 for e in transactions))
                balances = [a.resources for a in founders]
                self.assertEqual(min(balances), 20)
                self.assertTrue(all(a.is_active and a.resources > 10 for a in founders))
                final_balances.append(balances)
        self.assertEqual(final_balances[0], final_balances[1])

    @fixed_hash_seed
    def test_legacy_submission_and_logging_match_base_state_prompts_and_rng(self):
        for k, sequential, failed in ((1, False, False), (2, False, False),
                                     (1, True, False), (2, False, True)):
            outcomes = []
            for ns, audit in ((self.ns, False), (self.ns, True)):
                with tempfile.TemporaryDirectory() as directory:
                    sim, agent = make_sim(ns, directory, k=k, sequential=sequential,
                                          audit=audit, resubmission=None,
                                          results=[None] if failed else None)
                    random.seed(4242)
                    np.random.seed(4242)
                    papers = submit(sim, agent)
                    sim._charge_annual_costs(1)
                    outcomes.append((
                        [p["id"] for p in papers], agent.resources, agent.is_active,
                        sim.llm.prompts, random.getstate(), repr(np.random.get_state()),
                        sim.funding_tracker.to_dict(),
                    ))
            with self.subTest(k=k, sequential=sequential, failed=failed):
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertEqual(reference_digest(outcomes[0]),
                                 REFERENCE["legacy"][f"{k}:{sequential}:{failed}"])
                self.assertEqual(outcomes[0][1], 90)  # Configured 15 was never debited.

    def test_resubmission_override_shared_by_batch_and_sequential_gates_prompts_charges(self):
        for sequential in (False, True):
            for fee, opening, expected_count, closing in ((0, 0, 2, 0), (2, 4, 2, 0),
                                                          (5, 4, 0, 4), (None, 7, 2, 0)):
                with self.subTest(sequential=sequential, fee=fee), tempfile.TemporaryDirectory() as directory:
                    sim, agent = make_sim(self.ns, directory, balance=opening,
                                          resubmission=fee, sequential=sequential)
                    for i in range(2):
                        paper = {"id": f"p{i}", "author_id": agent.id, "author_type": "university",
                                 "title": "Title", "abstract": "Abstract", "conference": "C",
                                 "topics": ["algorithms"], "tags": ["cs.AI"],
                                 "status": "pending", "type": "submission", "year": 1,
                                 "project_start_year": 1, "project_end_year": 1, "maturity": 1}
                        sim.paper_tracker.add_or_update_paper(paper)
                        sim.paper_tracker.add_or_update_paper(dict(paper, status="reject"))
                    sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)
                    sim.resource_ledger.start_year(2, sim.ecosystem.agent_population)
                    sim.funding_tracker.start_cycle(2, sim.ecosystem.agent_population)
                    if sequential:
                        sim.llm.result = {"resubmitted_papers": [
                            {"arxiv_id": f"p{i}", "conference": "C"} for i in range(2)]}
                    papers = sim._run_phase_0_resubmissions(2, {})
                    self.assertEqual(len(papers), expected_count)
                    self.assertEqual(agent.resources, closing)
                    effective = 5 if fee is None else fee
                    if papers:
                        self.assertIn(f"Each resubmission costs {effective} per paper", sim.llm.prompts[0])
                    row = sim.resource_ledger.close_year(2, sim.ecosystem.agent_population)[0]
                    self.assertEqual(row["resubmission_cost"], opening - closing)
                    self.assertEqual(row["uncollected_cost"], 3 if fee is None else 0)
                    self.assertEqual(row["residual"], 0)
                    self.assertEqual(len(sim._run_phase_0_resubmissions(2, {})), 0)


class TestOptionalProjectPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()

    def test_same_fee_full_partial_failed_empty_and_fallback_outputs(self):
        cases = [
            (1, [entry(0)], 3, 1), (2, [{"submissions": [entry(0), entry(1)]}], 3, 2),
            (2, [{"submissions": [entry(0)]}], 3, 1),
            (2, [{"submissions": []}], 3, 0), (2, [None], 3, 2),
            (2, [{"submissions": [entry(0), {}]}, None], 3, 2),
            (2, [None], 1, 1), (2, [None], 0, 0),
        ]
        for k, responses, count, expected in cases:
            with self.subTest(k=k, responses=responses, count=count), tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(self.ns, directory, k=k, mode="per_project",
                                      results=responses, count=count)
                papers = submit(sim, agent)
                self.assertEqual(len(papers), expected)
                self.assertEqual(agent.resources, 85)
                row = sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)[0]
                self.assertEqual(row["production_submission_cost"], 15)
                self.assertEqual(len([e for e in sim.resource_ledger.transactions if e["delta"]]), 1)

    def test_affordability_exact_fee_allows_two_outputs_no_second_gate(self):
        for balance, expected in ((14, 0), (15, 2), (19, 2)):
            with self.subTest(balance=balance), tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(self.ns, directory, k=2, mode="per_project", balance=balance)
                self.assertEqual(len(submit(sim, agent)), expected)
                self.assertEqual(agent.resources, balance - (15 if expected else 0))
                if expected:
                    self.assertIn("Selecting any number of outputs", sim.llm.prompts[-1])
                    self.assertNotIn("funding issues", sim.llm.prompts[-1])
                    self.assertIn(f"Your remaining funding is {balance - 15}", sim.llm.prompts[-1])

    def test_sequential_success_and_failure_pay_at_same_boundary(self):
        for result, failed in ((entry(0), False), (None, True)):
            with tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(self.ns, directory, mode="per_project",
                                      sequential=True, results=[result])
                if failed:
                    with self.assertRaisesRegex(Exception, "Failed to submit"):
                        submit(sim, agent)
                else:
                    self.assertEqual(len(submit(sim, agent)), 1)
                    self.assertIn("already been paid", sim.llm.prompts[-1])
                self.assertEqual(agent.resources, 85)
                self.assertTrue(sim._prepare_project_production(1, [agent]))
                self.assertEqual(agent.resources, 85)

    def test_zero_override_and_repeated_entry_without_ledger(self):
        for fee in (0, 15):
            with tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(self.ns, directory, k=2, mode="per_project",
                                      fee=fee, audit=False)
                submit(sim, agent)
                for _ in range(3):
                    sim._prepare_project_production(1, [agent])
                self.assertEqual(agent.resources, 100 - fee)
                self.assertEqual(len(agent.production_cost_entries), 1)


class TestLedgerReconciliation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()

    def test_every_category_clipping_inactive_founders_and_two_year_continuity(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agent = make_sim(self.ns, directory, balance=50)
            inactive = self.ns["UniversityResearcher"](
                "inactive", "I", funding_level=3, expertise=agent.expertise)
            inactive.is_active = False
            sim.ecosystem.add_agent(inactive)
            sim.resource_ledger = ResourceLedger()
            sim.resource_ledger.start_year(1, sim.ecosystem.agent_population)
            changes = [
                ("grant_income", 20), ("industry_income", 7), ("annual_research_cost", -10),
                ("production_submission_cost", -15), ("resubmission_cost", -5),
                ("funding_application_cost", -2),
            ]
            rng = random.getstate(), repr(np.random.get_state())
            for category, delta in changes:
                apply_resource_change(agent, delta, category=category, year=1)
            rows = sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)
            self.assertEqual(rows[0]["closing_balance"], 45)
            self.assertEqual(rows[1]["closing_balance"], 3)
            self.assertFalse(rows[1]["closing_active"])
            self.assertTrue(all(row["founder"] and row["residual"] == 0 for row in rows))
            self.assertEqual(rng, (random.getstate(), repr(np.random.get_state())))
            restored = ResourceLedger(json.loads(json.dumps(sim.resource_ledger.to_dict())))
            restored.start_year(2, sim.ecosystem.agent_population)
            apply_resource_change(agent, -60, category="annual_research_cost", year=2)
            rows = restored.close_year(2, sim.ecosystem.agent_population)
            self.assertEqual(rows[0]["opening_balance"], 45)
            self.assertEqual(rows[0]["annual_research_cost"], 45)
            self.assertEqual(rows[0]["uncollected_cost"], 15)
            self.assertEqual(rows[0]["closing_balance"], 0)
            self.assertFalse(agent.is_active)
            restored.write(directory)
            self.assertEqual(len((Path(directory) / "resource_balances.jsonl").read_text().splitlines()), 4)
            restored.write(directory)
            self.assertEqual(len((Path(directory) / "resource_ledger.jsonl").read_text().splitlines()), 7)

    def test_uncategorized_direct_assignment_wrong_sign_and_changed_replay_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agent = make_sim(self.ns, directory)
            with self.assertRaisesRegex(ValueError, "Unclassified"):
                agent.update_resources(-1)
            self.assertEqual(agent.resources, 100)
            with self.assertRaisesRegex(ValueError, "sign"):
                apply_resource_change(agent, -1, category="grant_income", year=1)
            apply_resource_change(agent, -10, category="annual_research_cost", year=1)
            self.assertFalse(apply_resource_change(agent, -10, category="annual_research_cost", year=1))
            with self.assertRaisesRegex(ValueError, "replay changed"):
                apply_resource_change(agent, -11, category="annual_research_cost", year=1)
            agent.resources += 1
            with self.assertRaisesRegex(ValueError, "residual"):
                sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)
            apply_resource_change(agent, 5, category="grant_income", year=1)
            with self.assertRaisesRegex(ValueError, "Unexplained resource change"):
                sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)

    def test_native_funding_phase_records_application_debits_and_both_income_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agents, agency = native.make_simulation(directory, base=True, cost=3)
            industry = native.ORIGINAL["IndustryResearcher"](
                "industry", "Company", funding_level=20, expertise=agents[0].expertise)
            sim.ecosystem.add_agent(industry)
            sim.conference_system.authors_to_accepted_papers[industry.id] = [
                {"id": "industry_paper", "maturity": 2}]
            original_frame = sim.paper_tracker.get_papers_dataframe(1)
            sim.paper_tracker.get_papers_dataframe = lambda year: pd.concat([
                original_frame, pd.DataFrame([{
                    "id": "industry_paper", "author_id": industry.id, "author_type": "industry",
                    "title": "Industry paper", "abstract": "Fixture", "topics": ["algorithms"],
                    "maturity": 2, "status": "accept",
                }])], ignore_index=True)
            sim.industry_funding_system = self.ns["IndustryFundingSystem"]()
            sim.citation_tracker = self.ns["CitationTracker"]()
            sim.resource_ledger = ResourceLedger()
            sim.resource_ledger.start_year(1, sim.ecosystem.agent_population)
            with redirect_stdout(io.StringIO()):
                sim._run_phase_5_update_funding(1, [], {})
            rows = {r["researcher_id"]: r for r in
                    sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)}
            self.assertEqual(rows["founder_0"]["funding_application_cost"], 6)
            self.assertEqual(rows["founder_1"]["funding_application_cost"], 6)
            self.assertEqual(rows["founder_0"]["grant_income"], 40)
            self.assertEqual(rows["founder_1"]["grant_income"], 0)
            self.assertEqual(rows["industry"]["industry_income"], 40)
            self.assertTrue(all(r["residual"] == 0 for r in rows.values()))

    def test_application_replay_at_zero_balance_is_not_withdrawn_or_charged_again(self):
        with tempfile.TemporaryDirectory() as directory:
            sim, agent = make_sim(self.ns, directory, balance=3)
            applications = [{"P": {"author": agent, "submit": True}}]
            first = self.ns["charge_application_costs"](applications, 3, 1)
            again = self.ns["charge_application_costs"](applications, 3, 1)
            self.assertEqual(first, again)
            self.assertEqual(agent.resources, 0)
            self.assertIn("P", applications[0])
            self.assertEqual(len(sim.resource_ledger.transactions), 1)

    def test_native_checkpoint_roundtrip_and_exports_rollback(self):
        for mode in ("per_paper", "per_project"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(self.ns, directory, k=2, mode=mode)
                submit(sim, agent)
                sim._charge_annual_costs(1)
                sim.resource_ledger.close_year(1, sim.ecosystem.agent_population)
                with serializer_imports(self.ns, agent.expertise[0]):
                    sim._save_checkpoint(1)
                    state = sim.resource_ledger.to_dict()
                    balance = agent.resources
                    # Simulate exports written beyond the last committed checkpoint.
                    Path(directory, "resource_ledger.jsonl").write_text("stale\nstale\n")
                    self.assertTrue(sim.load_checkpoint(1))
                    restored = sim.ecosystem.get_agent_by_id(agent.id)
                    self.assertEqual(restored.resources, balance)
                    self.assertEqual(sim.resource_ledger.to_dict(), state)
                    sim._prepare_project_production(1, [restored])
                    sim._charge_annual_costs(1)
                    self.assertEqual(restored.resources, balance)
                    self.assertEqual(sim.resource_ledger.to_dict(), state)
                    self.assertEqual(len(Path(directory, "resource_ledger.jsonl").read_text().splitlines()),
                                     len(state["transactions"]))
                    sim.resource_ledger.start_year(2, sim.ecosystem.agent_population)
                    rows = sim.resource_ledger.close_year(2, sim.ecosystem.agent_population)
                    self.assertEqual(rows[0]["opening_balance"], balance)


@unittest.skipUnless(os.environ.get("UTOPIA_ACTUAL_MODULE_TESTS") == "1",
                     "Requires the isolated full-dependency CPU environment and real data cache")
class TestActualPrimaryLedger(unittest.TestCase):
    """Real modules, RAG, rounds and checkpoints. Only LLM transport is scripted."""
    setUpClass = classmethod(native.initialize_actual_simulator)
    tearDownClass = classmethod(native.restore_actual_simulator)
    make_actual = native.make_actual_simulation

    def test_two_year_primary_ledger_checkpoint_and_independent_audit(self):
        from utopia.analysis.project_cost import validate_ledger

        class PrimaryScriptedLLM(native.FullYearScriptedLLM):
            def generate_batch(self, prompts, seed_ctx, **kwargs):
                if seed_ctx[0] != "phase0_resubmit":
                    return super().generate_batch(prompts, seed_ctx, **kwargs)
                self.calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
                return [({"resubmitted_papers": [
                    {"arxiv_id": pid,
                     "conference": re.search(r'^- "([^"]+)": primary topics:', prompt, re.M)[1]}
                    for pid in re.findall(r'^arXiv ID: "([^"]+)"', prompt, re.M)
                ]}, []) for prompt in prompts]

        sim, researchers = self.make_actual("primary-ledger")
        try:
            sim.args.production_cost_mode = "per_paper"
            sim.args.resubmission_cost = sim.args.funding_application_cost = 0
            sim.args.log_resource_ledger = True
            sim.cost_policy = resolve_cost_policy(sim.args, SIMULATION_CONFIG)
            sim.resource_ledger = ResourceLedger()
            sim.llm = PrimaryScriptedLLM()
            for agent in sim.ecosystem.agent_population.values():
                agent.llm = sim.llm
            founders = {a.id for a in researchers}
            self.assertEqual(len(founders), 12)
            for year in (1, 2):
                sim.yearly_results.append(sim.run_one_round(year))
                sim._save_checkpoint(year, phase=5)
                path = Path(sim.output_dir) / f"checkpoint_year_{year}.json"
                opener = path.open if path.exists() else lambda mode: gzip.open(str(path) + ".gz", mode)
                with opener("rt") as stream:
                    checkpoint = json.load(stream)
                self.assertEqual(len(validate_ledger(checkpoint, year, founders)), 12 * year)
                self.assertEqual(checkpoint["project_cost_policy"], sim.cost_policy)
                self.assertEqual(checkpoint["cost_experiment_binding"]["args"], vars(sim.args))
                self.assertTrue(sim.load_checkpoint(year))
                self.assertEqual(sim.resource_ledger.to_dict(), checkpoint["resource_ledger"])
            resubmissions = [e for e in sim.resource_ledger.transactions
                            if e["category"] == "resubmission_cost"]
            self.assertTrue(resubmissions)
            self.assertTrue(all(e["requested_delta"] == e["delta"] == 0 for e in resubmissions))
            self.assertFalse(self.torch.cuda.is_initialized())
        finally:
            sim.wandb_logger.finish()


class TestCostConfiguration(unittest.TestCase):
    def test_distinct_ids_and_hash_inputs_in_both_layouts_default_paths_preserved(self):
        for stage in ([], ["--experiment_stage", "scale"]):
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                    "UTOPIA_DATA_CACHE_DIR": str(Path(directory) / "cache")}):
                common = ["--output-dir", directory] + stage
                default = parse_arguments(common)
                primary1 = parse_arguments(common + ["--resubmission_cost", "0", "--log_resource_ledger"])
                primary2 = parse_arguments(common + ["--resubmission_cost", "0", "--log_resource_ledger",
                                                     "--papers_per_project", "2"])
                sensitivity = parse_arguments(common + ["--resubmission_cost", "5", "--log_resource_ledger"])
                fixed = parse_arguments(common + ["--production_cost_mode", "per_project"])
                free = parse_arguments(common + ["--production_cost_mode", "per_project",
                                                "--project_production_cost", "0"])
                args = (default, primary1, primary2, sensitivity, fixed, free)
                self.assertEqual(len({a.experiment_id for a in args}), len(args))
                self.assertEqual(len({a.output_dir for a in args}), len(args))
                self.assertNotIn("_costv", default.experiment_id)
                policies = [resolve_cost_policy(a, SIMULATION_CONFIG) for a in args]
                self.assertEqual(len({json.dumps(p, sort_keys=True) for p in policies}), len(args))
                self.assertEqual(policies[1]["legacy_effective_per_paper_cost"], 0)
                self.assertEqual(policies[1]["resubmission_cell"], "zero_fee")
                self.assertEqual(policies[3]["resubmission_cell"], "positive_fee_sensitivity")

    def test_negative_and_nonfinite_costs_rejected_before_output_creation(self):
        for field in ("--resubmission_cost", "--project_production_cost"):
            for value in ("-1", "nan", "inf"):
                with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ValueError):
                        parse_arguments(["--output-dir", str(Path(directory) / "absent"), field, value])
                    self.assertFalse(Path(directory, "absent").exists())

    def test_wrong_or_unlabeled_checkpoint_rejected(self):
        ns = namespace()
        with tempfile.TemporaryDirectory() as directory:
            sim, agent = make_sim(ns, directory)
            path = Path(directory, "checkpoint_year_1.json")
            path.write_text(json.dumps({"year": 1}))
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                sim.load_checkpoint(1)
            path.write_text(json.dumps({"project_cost_policy": policy(), "year": 1}))
            with self.assertRaisesRegex(ValueError, "policy differs"):
                sim.load_checkpoint(1)


class TestSubmissionRefactorCompatibility(unittest.TestCase):
    """Compare observable Phase 0/2 behavior with the version before extraction."""

    @classmethod
    def setUpClass(cls):
        from utopia.constants import COI_NOTE

        cls.namespaces = [namespace()]
        for ns in cls.namespaces:
            ns.update(COI_NOTE=COI_NOTE, CitationResponse=native.SchemaStub,
                      SelfCitationResponse=native.SchemaStub)

    def citation_scenario(self, ns, scenario, sequential=False, no_authors=False):
        with tempfile.TemporaryDirectory() as directory:
            sim, agent = make_sim(ns, directory, k=1 if sequential else 2, count=4, audit=False)
            for index, status in ((0, "accept"), (1, "reject")):
                document = sim.rag.documents[index]
                paper = {
                    "id": f"p{index}", "author_id": agent.id, "author_type": "university",
                    "title": document.metadata["title"], "abstract": document.page_content,
                    "conference": "C", "topics": ["algorithms"], "tags": ["cs.AI"],
                    "status": "pending", "type": "submission", "year": 1, "project_start_year": 1,
                    "project_end_year": 1, "maturity": 1,
                }
                sim.paper_tracker.add_or_update_paper(paper)
                sim.paper_tracker.add_or_update_paper(dict(paper, status=status))
                sim.rag.document_status.loc[index, "status"] = status

            calls, retrievals, attempts = [], [], {}

            def retrieve(queries, retrieval_type):
                retrievals.append((list(queries), retrieval_type))
                indices = {"submission": [2, 3], "accepted_papers": [0],
                           "accepted_or_rejected_papers": [0, 1]}[retrieval_type]
                return {"topk_indices": np.tile(indices, (len(queries), 1))}

            def response(kind, attempt, position=0):
                if kind in ("phase2_intentions", "round_intention"):
                    return {"intention": "Investigate the prior results."}
                if kind == "phase2_submissions" or kind not in (
                    "phase2_citations_regular", "phase2_citations_extended",
                    "phase2_citations_self", "citation_response", "self_citation_response",
                ):
                    return entry(2) if sequential else {"submissions": [entry(2), entry(3)]}
                if kind in ("phase2_citations_self", "self_citation_response"):
                    if scenario == "self_exhausted":
                        return {"self_citations": [-1]}
                    return {"self_citations": [0, 0]}
                if scenario == "regular_exhausted" and kind == "phase2_citations_regular":
                    return None
                if scenario == "extended_exhausted" and kind == "phase2_citations_extended":
                    return {"citations": [100]}
                if scenario == "empty":
                    return {"citations": []}
                if scenario == "retry" and attempt == 0 and position == 0:
                    return {"citations": [-1]}
                return {"citations": [0]}

            def generate_batch(prompts, seed_ctx, **kwargs):
                calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
                attempt = seed_ctx[-1] if len(seed_ctx) == 3 else 0
                return [(response(seed_ctx[0], attempt, index), []) for index in range(len(prompts))]

            def generate(prompt, response_format, **kwargs):
                kind = response_format["json_object"]["name"]
                attempt = attempts.get(kind, 0)
                attempts[kind] = attempt + 1
                calls.append((prompt, deepcopy(response_format), deepcopy(kwargs)))
                return response(kind, attempt), []

            sim.rag.batch_retrieve = retrieve
            sim.llm = SimpleNamespace(generate=generate) if sequential else SimpleNamespace(generate_batch=generate_batch)
            agent.llm = sim.llm
            agent.project_start_year = agent.project_end_year = 2
            sim.funding_tracker.start_cycle(2, sim.ecosystem.agent_population)
            resubmission = sim.paper_tracker.papers_by_id["p1"].to_dict()
            resubmission.update(year=2, status="pending", type="resubmission")
            active = [] if no_authors else [agent]
            year_results = {}
            random.seed(821)
            np.random.seed(821)
            with patch.dict(SIMULATION_CONFIG["citation"], expected_citation_per_round=1), \
                    redirect_stderr(io.StringIO()):
                papers = sim._run_phase_2_submit_papers(
                    2, {agent.id: agent.newest_direction}, active, [resubmission], year_results)
            return {
                "papers": papers, "calls": calls, "retrievals": retrievals,
                "citations": sim.citation_tracker.to_dict(), "results": year_results,
                "archive": {key: paper.to_dict() for key, paper in sim.paper_tracker.papers_by_id.items()},
                "statuses": sim.rag.document_status.to_dict("records"),
                "venues": sim.conference_system.conferences[0].submitted_papers,
                "submitted_ids": sim.submitted_paper_ids, "resources": agent.resources,
                "funding": sim.funding_tracker.to_dict(), "rng": random.getstate(),
                "numpy_rng": repr(np.random.get_state()),
            }

    @fixed_hash_seed
    def test_batch_citation_retries_fallbacks_and_resubmission_order_match(self):
        for scenario in ("valid", "retry", "regular_exhausted", "extended_exhausted", "self_exhausted", "empty"):
            with self.subTest(scenario=scenario):
                after = self.citation_scenario(self.namespaces[0], scenario)
                self.assertEqual(reference_digest(after), REFERENCE["citations"][f"{scenario}:False:False"])
                self.assertEqual([paper["id"] for paper in after["papers"]], ["p2", "p3"])
                self.assertEqual(after["results"]["paper_submission"]["num_papers_submitted"], 3)

    @fixed_hash_seed
    def test_sequential_citation_retries_and_optional_self_citations_match(self):
        for scenario in ("valid", "retry", "self_exhausted", "empty"):
            with self.subTest(scenario=scenario):
                after = self.citation_scenario(self.namespaces[0], scenario, sequential=True)
                self.assertEqual(reference_digest(after), REFERENCE["citations"][f"{scenario}:True:False"])

    @fixed_hash_seed
    def test_year_with_only_resubmissions_preserves_requests_and_venue_admission(self):
        for sequential in (False, True):
            with self.subTest(sequential=sequential):
                after = self.citation_scenario(self.namespaces[0], "valid", sequential=sequential, no_authors=True)
                self.assertEqual(reference_digest(after), REFERENCE["citations"][f"valid:{sequential}:True"])
                self.assertEqual(after["papers"], [])
                self.assertEqual(after["results"]["paper_submission"]["num_papers_submitted"], 1)

    @fixed_hash_seed
    def test_partial_and_duplicate_resubmissions_preserve_fees_and_reviews(self):
        outcomes = []
        for ns in self.namespaces:
            with tempfile.TemporaryDirectory() as directory:
                sim, agent = make_sim(ns, directory, k=2, resubmission=3, audit=False)
                papers = submit(sim, agent)
                for paper in papers:
                    sim.paper_tracker.add_or_update_paper(dict(paper, status="reject"))
                calls = []

                def decisions(prompts, seed_ctx, **kwargs):
                    calls.append((deepcopy(prompts), seed_ctx, deepcopy(kwargs)))
                    entries = [{"arxiv_id": "p0", "conference": "C"},
                               {"arxiv_id": "p0", "conference": "C"},
                               {"arxiv_id": "p1", "conference": "missing"}]
                    if seed_ctx[-1]:
                        entries = [{"arxiv_id": "p1", "conference": "C"}]
                    return [({"resubmitted_papers": entries}, []) for _ in prompts]

                sim.llm.generate_batch = decisions
                sim.funding_tracker.start_cycle(2, sim.ecosystem.agent_population)
                result = sim._run_phase_0_resubmissions(2, {})
                outcomes.append((result, agent.resources, calls, sim.funding_tracker.to_dict()))
        self.assertEqual(reference_digest(outcomes[0]), REFERENCE["resubmissions"])
        self.assertEqual([paper["id"] for paper in outcomes[0][0]], ["p0", "p1"])
        self.assertEqual(outcomes[0][1], 94)
        self.assertEqual(len(outcomes[0][2]), 2)


if __name__ == "__main__":
    unittest.main()
