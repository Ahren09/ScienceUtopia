"""Real Simulation + FundingFeedbackMixin phases; fake only corpus/retrieval/HTTP.

No AST extraction, model downloads, CUDA allocation or scientific inference.
This small single-seed fixture tests replay, not the experimental treatment.
"""
import json
import os
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from langchain_core.documents import Document
from openai.types.chat import ChatCompletion

from utopia.agents.funding_agents import FundingAgency, FundingProgram
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS
from utopia.agents.researcher_agents import UniversityResearcher
from utopia.config import SIMULATION_CONFIG
from utopia.data.rag import RAG
from utopia.funding.feedback import FundingFeedbackMixin
from utopia.models.models import VLLMServerModel
from utopia.models.response_cache import CompletedResponseCache
from utopia.runtime.checkpointing import PortableStop, read_portable_checkpoint
from utopia.simulation import Simulation, check_citation_integrity

DIRECTION = next(d for d in AVAILABLE_DIRECTIONS if d.years == 1)
BINDING = dict(protocol="real-phases-cpu-fixture-v1", cell="P1F1", seed=42,
               model="fake-model", model_revision="fake-revision",
               corpus_sha256="fixed-test-documents-v1", science_config={"years": 6, "n": 4})


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def fixed_corpus(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("PYTHONHASHSEED") == "42"
    monkeypatch.setitem(SIMULATION_CONFIG["wandb"], "enabled", False)
    # No environment-selected production corpus can enter this fake test.
    monkeypatch.setattr("utopia.data.rag.corpus_from_env", lambda: None)

    def load(rag, data_path=None, num_years=6):
        rag.num_years = num_years
        rag.documents = [Document(page_content=f"Fixture abstract {i}", metadata={
            "id": f"fixture-{i:03d}", "title": f"Fixture paper {i}",
            "topics": ["Artificial Intelligence"], "tags": ["cs.AI"],
            "publication_year": 2016 + i // 8}) for i in range(48)]
        rag.id2docs = {d.metadata["id"]: d for d in rag.documents}
        rag.document_status = pd.DataFrame([
            dict(id=d.metadata["id"], status="unsubmitted", publication_year=d.metadata["publication_year"])
            for d in rag.documents])
        return rag.documents

    def build(rag):
        rag.document_embeddings = torch.zeros((len(rag.documents), 384))
        rag.original_doc_count = len(rag.documents)

    def retrieve(rag, queries, retrieval_type="submission", **kwargs):
        df = rag.document_status
        if retrieval_type == "submission":
            mask = (df.status == "unsubmitted") & (df.publication_year == 2016 + rag.current_sim_year - 1)
        elif retrieval_type == "accepted_papers":
            mask = df.status == "accept"
        elif retrieval_type == "accepted_or_rejected_papers":
            mask = df.status.isin(["accept", "reject"])
        else:
            raise AssertionError(retrieval_type)
        candidates = df.index[mask].to_numpy(dtype=int)
        indices = np.tile(candidates, (len(queries), 1))
        return dict(topk_indices=indices, similarity_scores=np.zeros(indices.shape))

    monkeypatch.setattr(RAG, "load_documents", load)
    monkeypatch.setattr(RAG, "build_knowledge_index", build)
    monkeypatch.setattr(RAG, "batch_retrieve", retrieve)
    seed()


class FakeHTTPModel(VLLMServerModel):
    def __init__(self, root, requests, *, interrupt=False):
        self.model_name, self.model_revision = BINDING["model"], BINDING["model_revision"]
        self.run_seed, self.enable_thinking, self.max_tokens_override = 42, True, 8192
        self.max_concurrent_requests = 1
        self.request_audit = None
        self.call_stats = dict(n_prompts=0, n_first_attempt_success=0, n_retries=0,
                               n_failures=0, elapsed_seconds=0., prompt_tokens=0, completion_tokens=0)
        self.response_cache = CompletedResponseCache(root / "cache" / "responses.sqlite", binding=BINDING)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.reply)))
        self.requests, self.interrupt = requests, interrupt
        self.reused = 0
        self.contexts = {}

    def _create_completion(self, *, seed_ctx=None, item_index=0, attempt=0, **payload):
        self.context = (seed_ctx, item_index, attempt)
        response = super()._create_completion(seed_ctx=seed_ctx, item_index=item_index, attempt=attempt, **payload)
        self.reused += bool(getattr(response, "_portable_cache_hit", False))
        return response

    def batch_completions(self, executor, requests, *, seed_ctx, attempt):
        for idx, payload in requests:
            self.contexts[payload['extra_body']['seed']] = (seed_ctx, idx, attempt)
        replies = super().batch_completions(executor, requests, seed_ctx=seed_ctx, attempt=attempt)
        self.reused += sum(bool(getattr(reply, '_portable_cache_hit', False)) for reply in replies.values())
        return replies

    def reply(self, **payload):
        self.context = self.contexts.get(payload["extra_body"]["seed"], getattr(self, "context", None))
        context, item, attempt = self.context
        phase, year = context[:2]
        if self.interrupt and phase == "phase3_reviews" and year == 3 and item == 1:
            raise TimeoutError("injected disconnect after first review in a batch")
        prompt = payload["messages"][-1]["content"]
        conference = self.sim.conference_system.conferences[0].conference_id
        if phase == "phase1_directions":
            data = dict(topic=DIRECTION.topic, detailed_focus="Fixture direction", reason="Fixture reason")
        elif phase == "phase2_intentions":
            data = dict(intention="Fixture intention")
        elif phase == "phase2_submissions":
            ids = re.findall(r'^arXiv ID: ([^\n]+)', prompt, re.M)
            assert ids, prompt
            idx = item % len(ids)
            data = dict(id=idx + 1, arxiv_id=ids[idx].strip(), conference=conference, reason="Fixture selection")
        elif phase == "phase0_resubmit":
            ids = re.findall(r'^arXiv ID: "([^"\n]+)"', prompt, re.M)
            data = dict(resubmitted_papers=[dict(arxiv_id=ids[0], conference=conference)] if ids else [])
        elif phase == "phase2_citations_self":
            data = dict(self_citations=[0])
        elif phase.startswith("phase2_citations_"):
            data = dict(citations=[0] if re.search(r'(?m)^(?:- arXiv ID:|Paper 0)', prompt) else [])
        elif phase == "phase3_reviews":
            data = dict(overall_score=4, justification="Fixture review")
        elif phase == "phase5_funding_apps":
            data = dict(submit=True, research_proposal="Fixture proposal", relevant_projects=[])
        elif phase == "phase5_funding_eval":
            apps = re.findall(r'### Application (\d+)\s+Applicant ID: ([^\n]+)', prompt)
            assert apps, prompt
            data = dict(ranked_applications=[dict(application_id=int(idx), applicant_id=aid.strip(),
                                                rank=rank, reason="Fixture ranking")
                                            for rank, (idx, aid) in enumerate(apps, 1)])
        else:
            raise AssertionError(phase)
        self.requests.append((self.context, payload))
        return ChatCompletion.model_validate(dict(
            id=f"fake-{year}-{item}", object="chat.completion", created=0, model=self.model_name,
            choices=[dict(index=0, message=dict(role="assistant", content=json.dumps(data)), finish_reason="stop")],
            usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))


class SmallRealSimulation(FundingFeedbackMixin, Simulation):
    def __init__(self, root, model, *, after_effects=False, ledger=False):
        self.root, self.after_effects = root, after_effects
        output, docs = root / "checkpoints", root / "docs" / "metrics"
        output.mkdir(parents=True, exist_ok=True)
        docs.mkdir(parents=True, exist_ok=True)
        self.uploaded_years = []
        args = SimpleNamespace(seed=42, num_conferences=1, docs_dir=str(docs), rag_device="cpu",
                               checkpoint_callback=self.record_upload,
                               portable_checkpoint=True, checkpoint_binding=BINDING,
                               funding_feedback=True, expected_founders=4,
                               log_resource_ledger=ledger, log_funding_applications=True,
                               funding_application_cost=1.)
        super().__init__(model, num_years=6, output_dir=str(output), args=args,
                         funding_allocation_mode="fixed", debug=True)
        model.sim = self

    def record_upload(self, path):
        assert isinstance(path, Path)
        snapshot = read_portable_checkpoint(path)
        assert snapshot["year"] == self._portable_last_year
        assert snapshot["phase"] == (0 if snapshot["year"] == 0 else 5)
        self.uploaded_years.append(snapshot["year"])

    def initialize_agents(self):
        for i in range(4):
            agent = UniversityResearcher(f"researcher-{i}", f"university-{i}", funding_level=1000,
                                         expertise=[DIRECTION], llm=self.llm)
            self.ecosystem.add_agent(agent)
        agency = FundingAgency("NSF", llm=self.llm)
        agency.funding_programs = {"NSF_FIXTURE": FundingProgram(
            "NSF_FIXTURE", "Fixture funding", [DIRECTION], funding_rate=.5, budget_per_winner=20)}
        self.ecosystem.add_agent(agency)
        self._record_initial_university_count()

    def run_one_round(self, year):
        result = super().run_one_round(year)
        if self.after_effects and year == 3:
            raise PortableStop("injected after mechanisms/awards before year commit")
        return result


def science_snapshot(sim):
    snapshot = read_portable_checkpoint(Path(sim.output_dir) / "portable_checkpoint_year_6.json.gz")
    # Wall timestamps and diagnostic durations are operational metadata. They
    # are not regenerated claims of new inference and are not science outcomes.
    ignored = {"timestamp", "submission_time", "review_time", "decision_time", "elapsed_seconds"}
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items() if k not in ignored}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, set):
            return sorted(normalize(v) for v in value)
        if hasattr(value, "to_dict"):
            return normalize(value.to_dict())
        if isinstance(value, FundingProgram):
            return normalize(vars(value))
        return value
    # Compare every scientific serializer and subclass hook, not file gzip bytes.
    snapshot["portable"].pop("outputs")
    snapshot["paper_tracker"].pop("output_dir")  # Expected relocation, not an outcome.
    snapshot.pop("cost_experiment_binding", None)  # Legacy provenance includes host-specific args.
    return normalize(snapshot)


def assert_same_state(actual, expected, path="state"):
    assert type(actual) is type(expected), path
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            assert_same_state(actual[key], expected[key], f"{path}/{key}")
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_same_state(left, right, f"{path}/{index}")
    else:
        assert actual == expected, path


@pytest.mark.parametrize("failure", ["midbatch", "after-effects"])
@pytest.mark.parametrize("ledger", [False, True])
def test_real_six_year_phases_resume_without_duplicate_effects(tmp_path, fixed_corpus, failure, ledger):
    if ledger:
        import inspect
        from utopia.funding.feedback import AwardResourceGate
        if not any(p.kind == p.VAR_KEYWORD for p in inspect.signature(AwardResourceGate.__call__).parameters.values()):
            pytest.xfail("Parent integration: AwardResourceGate must forward resource accounting keywords")
    reference_requests = []
    reference = SmallRealSimulation(tmp_path / "reference", FakeHTTPModel(tmp_path / "reference", reference_requests), ledger=ledger)
    reference.run()
    expected = science_snapshot(reference)
    assert len(reference.yearly_results) == 6
    assert reference.uploaded_years == list(range(7))
    manifest = json.loads((Path(reference.args.docs_dir) / "run_manifest.json").read_text())
    assert "checkpoint_callback" not in manifest["args"]
    assert reference.paper_tracker.papers_by_id
    assert reference.citation_tracker.citations
    assert not check_citation_integrity(reference.paper_tracker, reference.citation_tracker)
    assert reference.funding_feedback_years
    reference.llm.response_cache.close()

    seed()
    old = tmp_path / "old-host"
    requests = []
    interrupted = SmallRealSimulation(old, FakeHTTPModel(old, requests, interrupt=failure == "midbatch"),
                                     after_effects=failure == "after-effects", ledger=ledger)
    with pytest.raises(PortableStop):
        interrupted.run()
    assert (old / "checkpoints" / "portable_checkpoint_year_2.json.gz").exists()
    assert not (old / "checkpoints" / "portable_checkpoint_year_3.json.gz").exists()
    interrupted.llm.response_cache.close()
    relocated = tmp_path / "new-host"
    shutil.copytree(old, relocated)
    resumed = SmallRealSimulation(relocated, FakeHTTPModel(relocated, requests), ledger=ledger)
    resumed.run()
    assert resumed.llm.reused > 0
    assert interrupted.uploaded_years == [0, 1, 2]
    assert resumed.uploaded_years == [2, 3, 4, 5, 6]
    assert len(requests) == len(reference_requests), "Completed replies must not be sent again"
    assert_same_state(science_snapshot(resumed), expected)
    for name, path in resumed._portable_replay_paths.items():
        other = reference._portable_replay_paths[name]
        assert path.exists() == other.exists(), name
        if path.exists() and path.suffix in (".json", ".jsonl"):
            assert path.read_bytes() == other.read_bytes(), name
    assert not torch.cuda.is_initialized()
    resumed.llm.response_cache.close()
