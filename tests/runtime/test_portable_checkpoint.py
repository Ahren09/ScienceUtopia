"""Replay production checkpoint/model methods with deterministic CPU responses."""
import json
import os
from pathlib import Path
import random
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from openai.types.chat import ChatCompletion

from utopia.models.models import VLLMServerModel
from utopia.models.response_cache import CompletedResponseCache
from utopia.runtime.checkpointing import (
    PortableCheckpointError, PortableStop, RunStop, capture_rng, restore_rng,
    read_portable_checkpoint, pack_state, unpack_state,
)
# Compile unchanged production class definitions, following the existing CPU
# transport fixture pattern. This avoids importing optional FAISS/LangChain
# loaders and runtime provenance probes. Real domain serializers are exercised.
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
import pandas as pd
from utopia.config import SIMULATION_CONFIG
from utopia.agents.base_agent import MultiAgentEcosystem
from utopia.agents.conference import ConferenceSystem, select_conferences_for_simulation
from utopia.agents.funding_agents import FundingAgency, IndustryFundingSystem
from utopia.agents.research_direction import AVAILABLE_DIRECTIONS
from utopia.agents.researcher_agents import UniversityResearcher, IndustryResearcher
from utopia.data.paper_tracker import PaperTracker
from utopia.data.fulltext import corpus_from_env
from utopia.data.tracker import AgentTracker, CitationTracker, FundingTracker
from utopia.metrics.wandb_logger import WandbLogger
from utopia.utils.paths import project_root
from utopia.utils.data_utils import json_sha256
from utopia.runtime.provenance import _git
from utopia.utils.data_utils import write_json_atomic, write_json_gzip_atomic


def production_classes():
    namespace = dict(globals(), logger=logging.getLogger("portable-test"),
                     write_run_manifest=lambda *args, **kwargs: None)
    for file, names in (("utopia/data/rag.py", {"RAG", "RAGConfig"}),
                        ("utopia/simulation.py", {"Simulation", "argparse_namespace_placeholder"})):
        path = project_root() / file
        tree = ast.parse(path.read_text())
        def node_name(node):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                return node.name
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                return node.targets[0].id
        selected = [node for node in tree.body if node_name(node) in names]
        assert {node_name(node) for node in selected} == names
        module = ast.fix_missing_locations(ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *selected], type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["Simulation"]


Simulation = production_classes()


def binding():
    return dict(protocol="portable-test-v1", cell="P1F1", model="fake-model", model_revision="revision-a",
                corpus_sha256="corpus-a", science_config={"years": 2, "n": 2}, seed=42)


def fake_model(root, requests, interrupt_after=None):
    model = VLLMServerModel.__new__(VLLMServerModel)
    model.model_name = "fake-model"
    model.model_revision = "revision-a"
    model.run_seed = 42
    model.enable_thinking = True
    model.max_tokens_override = 8192
    model.max_concurrent_requests = 1
    model.request_audit = None
    model.call_stats = dict(n_prompts=0, n_first_attempt_success=0, n_retries=0,
                            n_failures=0, elapsed_seconds=0., prompt_tokens=0, completion_tokens=0)
    model.response_cache = CompletedResponseCache(root / "cache" / "responses.sqlite", binding=binding())
    def complete(**payload):
        if interrupt_after is not None and len(requests) == interrupt_after:
            model.response_cache.stop.request_stop()
            raise TimeoutError("injected midbatch disconnect")
        assert payload["max_tokens"] == 8192
        requests.append(payload)
        return ChatCompletion.model_validate(dict(
            id="fake", object="chat.completion", created=0, model="fake-model",
            choices=[dict(index=0, message=dict(role="assistant", content=json.dumps(
                {"value": payload["extra_body"]["seed"] % 13})), finish_reason="stop")],
            usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    return model


class ToySimulation(Simulation):
    """Real run/save/load and domain serializers, only scientific work is fake."""
    def __init__(self, root, model, *, crash_after_effects=False, checkpoint_callback=None, ledger=False):
        self.root = root
        self.awards = []
        self.crash_after_effects = crash_after_effects
        self.finalized = 0
        output = root / "checkpoints"
        output.mkdir(parents=True, exist_ok=True)
        (root / "logs").mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(seed=42, num_conferences=1, docs_dir=str(root / "docs"), rag_device="cpu",
                               checkpoint_callback=checkpoint_callback, log_resource_ledger=ledger)
        super().__init__(model, num_years=2, output_dir=str(output), args=args,
                         funding_allocation_mode="fixed", debug=True)
        self.enable_portable_checkpoints(binding(), replay_paths={"metrics": root / "logs" / "metrics.jsonl"})

    def initialize_agents(self):
        for i in range(2):
            agent = UniversityResearcher(f"researcher-{i}", "university", expertise=[AVAILABLE_DIRECTIONS[0]], llm=self.llm)
            agent.can_review = bool(i)
            agent.max_retries = 7
            self.ecosystem.add_agent(agent)
        self._record_initial_university_count()

    def run_one_round(self, year):
        draws = [random.random(), float(np.random.random()), float(torch.rand(()))]
        results = self.llm.generate_batch([json.dumps([year, i, draws]) for i in range(3)],
                                          seed_ctx=("toy-year", year), max_tokens=1)
        values = [r[0]["value"] for r in results]
        for agent in self.ecosystem.agent_population.values():
            agent.resources += sum(values)
        self.awards.append((year, sum(values)))
        with (self.root / "logs" / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"year": year, "values": values}) + "\n")
        if self.crash_after_effects:
            raise PortableStop("injected after funding/metrics but before year commit")
        return dict(year=year, draws=draws, values=values)

    def extra_checkpoint_state(self):
        return {"awards": self.awards}

    def restore_extra_checkpoint_state(self, state):
        self.awards = state["awards"]
        random.random()  # Rebuilding a hook must not perturb the restored stream.

    def _generate_final_report(self):
        self.finalized += 1


@pytest.fixture(autouse=True)
def deterministic_process():
    with patch.dict(os.environ, {"PYTHONHASHSEED": "42"}), \
         patch.dict(SIMULATION_CONFIG["wandb"], {"enabled": False}):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        yield


def science_state(sim):
    stats = {k: v for k, v in sim.llm.call_stats.items() if k != "elapsed_seconds"}
    return dict(years=sim.yearly_results, awards=sim.awards, stats=stats,
                agents={aid: (a.resources, a.can_review, a.newest_direction, a.max_retries)
                        for aid, a in sim.ecosystem.agent_population.items()},
                metrics=(sim.root / "logs" / "metrics.jsonl").read_text(),
                next_draws=[random.random(), float(np.random.random()), float(torch.rand(()))])


@pytest.mark.parametrize("failure", ["midbatch", "after-effects", "after-year"])
def test_uninterrupted_equals_crash_resume_after_relocation(tmp_path, failure):
    reference = ToySimulation(tmp_path / "reference", fake_model(tmp_path / "reference", []))
    reference.run()
    expected = science_state(reference)
    reference.llm.response_cache.close()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    old = tmp_path / "old-machine"
    requests = []
    model = fake_model(old, requests, interrupt_after=1 if failure == "midbatch" else None)
    interrupted = ToySimulation(old, model, crash_after_effects=failure == "after-effects")
    if failure == "after-year":
        original_save = interrupted._save_checkpoint
        def save_and_stop(year, phase=5):
            original_save(year, phase)
            if year == 1:
                interrupted.portable_stop.request_stop()
        interrupted._save_checkpoint = save_and_stop
    with pytest.raises(PortableStop):
        interrupted.run()
    assert not interrupted.finalized
    last = 1 if failure == "after-year" else 0
    state = read_portable_checkpoint(old / "checkpoints" / f"portable_checkpoint_year_{last}.json.gz")
    assert state["year"] == last
    model.response_cache.close()
    new = tmp_path / "new-machine"
    shutil.copytree(old, new)
    resumed_model = fake_model(new, requests)
    resumed = ToySimulation(new, resumed_model)
    configured_rag = resumed.rag
    resumed.run()
    assert resumed.rag is configured_rag
    assert resumed.paper_tracker.output_dir == new / "checkpoints"
    assert all(a.llm is resumed_model for a in resumed.ecosystem.agent_population.values())
    assert science_state(resumed) == expected
    assert len(requests) == 6, "Every completed request is sent exactly once across both sessions"
    resumed.llm.response_cache.close()


def test_binding_mismatch_legacy_rejection_and_rng_roundtrip(tmp_path):
    sim = ToySimulation(tmp_path, fake_model(tmp_path, []))
    sim.initialize_agents()
    from utopia.agents.conference import ConferenceSystem
    sim.conference_system = ConferenceSystem([])
    sim._save_checkpoint(0, phase=0)
    sim.num_years = 3
    with pytest.raises(PortableCheckpointError, match="binding changed"):
        sim.load_checkpoint(0)
    sim.num_years = 2
    (tmp_path / "checkpoints" / "checkpoint_year_1.json").write_text("{}")
    with pytest.raises(PortableCheckpointError, match="legacy"):
        sim.load_checkpoint(1)
    state = unpack_state(pack_state(capture_rng()))
    expected = (random.random(), np.random.random(), torch.rand(()))
    restore_rng(state)
    assert (random.random(), np.random.random(), torch.rand(())) == expected
    assert not torch.cuda.is_initialized()
    sim.llm.response_cache.close()


def test_expired_budget_and_signal_safe_point():
    stop = RunStop(max_seconds=10)
    stop.deadline = time.monotonic() - 1
    with pytest.raises(PortableStop):
        stop.check()
    stop = RunStop()
    with stop.signals():
        stop.request_stop(15)
        with pytest.raises(PortableStop, match="signal 15"):
            stop.check()


def test_cache_older_than_snapshot_is_rejected(tmp_path):
    sim = ToySimulation(tmp_path, fake_model(tmp_path, []))
    sim.run()
    db = sim.llm.response_cache._db
    db.execute("DELETE FROM replies WHERE identity = (SELECT identity FROM replies LIMIT 1)")
    db.commit()
    with pytest.raises(PortableCheckpointError, match="older than"):
        sim.load_checkpoint(2)
    sim.llm.response_cache.close()


def test_exact_replay_files_restore_and_remove_future_outputs(tmp_path):
    from utopia.runtime.checkpointing import (
        simulation_replay_paths, capture_outputs, restore_outputs,
    )
    output, docs = tmp_path / "outputs", tmp_path / "docs" / "metrics"
    output.mkdir(); docs.mkdir(parents=True)
    paths = simulation_replay_paths(output, docs, 6, table_names=("paper",))
    for key, path in paths.items():
        if "year_3" not in key:
            path.write_bytes(b"committed\n")
    saved = capture_outputs(paths)
    for path in paths.values():
        path.write_bytes(b"uncommitted\n")
    protected = [output / "portable_checkpoint_year_2.json.gz",
                 output / "responses.sqlite", output / "runtime.log",
                 docs / "run_manifest.json", docs.parent / "audit" / "requests.jsonl"]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"physical/provenance: preserve\n")
    restore_outputs(paths, saved)
    for key, path in paths.items():
        if "year_3" in key:
            assert not path.exists()
        else:
            assert path.read_bytes() == b"committed\n"
    assert all(path.read_bytes() == b"physical/provenance: preserve\n" for path in protected)


def test_portable_preserves_shared_paper_and_review_containers(tmp_path):
    from utopia.runtime.checkpointing import write_portable_checkpoint
    reviews = [{"year": 1}]
    paper = {"id": "paper", "review_history": reviews}
    state = {"tracker": {"paper": paper}, "conference": [paper], "history": reviews}
    path = tmp_path / "portable.json.gz"
    write_portable_checkpoint(path, state)
    restored = read_portable_checkpoint(path)
    assert restored["conference"][0] is restored["tracker"]["paper"]
    assert restored["history"] is restored["tracker"]["paper"]["review_history"]
    restored["history"].append({"year": 2})
    assert len(restored["conference"][0]["review_history"]) == 2


def test_replay_rejects_live_manifest_directory(tmp_path):
    sim = ToySimulation(tmp_path, fake_model(tmp_path, []))
    with pytest.raises(PortableCheckpointError, match="live run manifest"):
        sim.enable_portable_checkpoints(binding(), replay_paths={"metrics": tmp_path / "docs"})
    sim.llm.response_cache.close()


@pytest.mark.parametrize("failed_year", [0, 1, 2])
def test_upload_callback_pause_retains_checkpoint_and_retries_on_resume(tmp_path, failed_year):
    requests, callbacks = [], []
    sim = ToySimulation(tmp_path, fake_model(tmp_path, requests))
    def upload(path):
        assert isinstance(path, Path)
        saved = read_portable_checkpoint(path)
        year = saved["year"]
        callbacks.append(year)
        assert sim._portable_last_year == year
        assert saved["portable"]["response_cache_count"] == sim.llm.response_cache.committed_count()
        if year == failed_year:
            raise OSError("injected upload failure")
    sim.checkpoint_callback = upload
    with pytest.raises(PortableStop, match="retained locally; upload callback failed"):
        sim.run()
    checkpoint = tmp_path / "checkpoints" / f"portable_checkpoint_year_{failed_year}.json.gz"
    committed = checkpoint.read_bytes()
    assert callbacks == list(range(failed_year + 1))
    assert sim.finalized == 0
    sim.llm.response_cache.close()

    resumed = ToySimulation(tmp_path, fake_model(tmp_path, requests))
    uploaded = []
    resumed.checkpoint_callback = lambda path: uploaded.append(read_portable_checkpoint(path)["year"])
    resumed.run()
    assert uploaded == list(range(failed_year, 3))
    assert checkpoint.read_bytes() == committed
    assert resumed.finalized == 1
    assert len(requests) == 6
    resumed.llm.response_cache.close()


def test_args_upload_callback_is_runtime_only_with_cost_accounting(tmp_path):
    calls = []
    def callback(path):
        calls.append(path)
        assert read_portable_checkpoint(path)["year"] == 0
    sim = ToySimulation(tmp_path, fake_model(tmp_path, []), checkpoint_callback=callback, ledger=True)
    assert sim.args.checkpoint_callback is callback
    sim.initialize_agents()
    sim.conference_system = ConferenceSystem([])
    sim._save_checkpoint(0, phase=0)
    assert len(calls) == 1
    saved = read_portable_checkpoint(calls[0])
    assert "checkpoint_callback" not in saved["cost_experiment_binding"]["args"]
    sim.llm.response_cache.close()
