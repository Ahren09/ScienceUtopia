"""Real prospective constructor regression with no worlds, inference or uploads."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utopia.config import SIMULATION_CONFIG
from utopia.experiments import funding_feedback as runner
from utopia.funding.feedback import FundingFeedbackMixin
from utopia.models import request_audit
from utopia.models.response_cache import CompletedResponseCache
from utopia.runtime.checkpointing import PortableCheckpointError, PortableStop
from utopia.simulation import Simulation


@pytest.fixture
def constructor_only(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("PYTHONHASHSEED") == "42"
    monkeypatch.setenv("WANDB_MODE", "disabled")
    for name in ("UTOPIA_FULLTEXT_INDEX", "UTOPIA_FULLTEXT_SHA256", "UTOPIA_EMBEDDING_REVISION"):
        monkeypatch.setenv(name, "")
    monkeypatch.setitem(SIMULATION_CONFIG, "wandb", dict(SIMULATION_CONFIG["wandb"], enabled=True))
    wandb = SimpleNamespace(init=Mock(), finish=Mock())
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    # Native RAG construction is lazy. Do not load a production corpus or model.
    monkeypatch.setattr("utopia.data.rag.corpus_from_env", lambda: None)
    monkeypatch.setattr(request_audit, "_load_tokenizer", lambda *args: SimpleNamespace(chat_template="fixture"))
    return wandb


@pytest.mark.parametrize("cell", ["P1F1", "INIT"])
@pytest.mark.parametrize("thinking", [None, True, False])
def test_shared_prospective_runner_disables_wandb_before_binding_and_constructor(
        tmp_path, monkeypatch, constructor_only, cell, thinking):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "prospective_source_fingerprint", lambda root=None: "fixture-source")
    corpus = SimpleNamespace(path=tmp_path / "fixture-index.json", sha256="0" * 64,
                             reference=lambda: {"fixture": True})
    monkeypatch.setattr("utopia.data.fulltext.FullTextCorpus", lambda path: corpus)
    options = SimpleNamespace(seed=42, founders=5, years=6, run_id="constructor_fixture", cell=cell,
                              model="fixture/model", model_revision="a" * 40,
                              vllm_url="http://127.0.0.1:1/v1", batch_size=1,
                              deadline_epoch=4102444800., context_tokens=131072, max_tokens=8192,
                              fulltext_index=str(corpus.path), resume=False,
                              server_manifest=str(tmp_path / "server.json"),
                              s3_prefix="s3://fixture-never-uploaded/study")
    if thinking is not None:
        options.thinking = thinking
    Path(options.server_manifest).write_text(json.dumps(dict(
        model=options.model, model_revision=options.model_revision,
        endpoint=options.vllm_url, context_tokens=options.context_tokens)))
    inference = Mock(side_effect=AssertionError("Constructor must not request inference"))
    uploaded = Mock(side_effect=AssertionError("Constructor must not upload a checkpoint"))
    monkeypatch.setattr(runner.subprocess, "run", uploaded)
    created = []

    def model(**kwargs):
        assert kwargs["thinking_override"] is thinking
        science = kwargs["checkpoint_binding"]["science_config"]
        if thinking is None:
            assert "thinking_override" not in science
        else:
            assert science["thinking_override"] is thinking
        cache = CompletedResponseCache(Path(kwargs["response_cache_dir"]) / "responses.sqlite",
                                       binding=kwargs["checkpoint_binding"])
        return SimpleNamespace(
            model_name=kwargs["model_name"], model_revision=kwargs["model_revision"],
            run_seed=kwargs["run_seed"], response_cache=cache,
            request_audit=request_audit.active_audit_for(kwargs["model_name"]),
            client=SimpleNamespace(
                models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=options.model)])),
                chat=SimpleNamespace(completions=SimpleNamespace(create=inference))))

    monkeypatch.setattr("utopia.simulation.VLLMServerModel", model)

    class ConstructorOnly(FundingFeedbackMixin, Simulation):
        def run(self):
            assert not self.ecosystem.agent_population
            assert self.llm.response_cache.committed_count() == 0
            created.append(self)
            raise PortableStop("constructor regression reached pre-world boundary")

    with pytest.raises(PortableStop, match="constructor regression reached pre-world boundary"):
        runner.run_prospective(options, simulation_factory=lambda base: ConstructorOnly)
    sim, = created
    assert sim.wandb_logger.enabled is False
    assert sim._portable_binding["science_config"]["config"]["wandb"]["enabled"] is False
    manifest = json.loads((Path(sim.args.docs_dir).parent / "experiment.json").read_text())
    assert manifest["binding"] == sim._portable_binding == sim.args.checkpoint_binding
    assert manifest["request_audit"]["n_requests"] == 0
    assert manifest["request_audit"]["n_responses"] == 0
    assert not list(Path(sim.output_dir).iterdir())
    constructor_only.init.assert_not_called()
    inference.assert_not_called()
    uploaded.assert_not_called()


@pytest.mark.parametrize("portable", [False, True])
def test_explicit_wandb_opt_in_and_portable_guard(tmp_path, constructor_only, portable):
    binding = dict(protocol="constructor-fixture-v1", cell="P1F1", seed=42,
                   model="fixture/model", model_revision="a" * 40,
                   corpus_sha256="0" * 64, science_config={"fixture": True})
    cache = CompletedResponseCache(tmp_path / "responses" / "responses.sqlite", binding=binding)
    model = SimpleNamespace(model_name=binding["model"], model_revision=binding["model_revision"],
                            run_seed=42, response_cache=cache)
    args = SimpleNamespace(seed=42, rag_device="cpu", docs_dir=str(tmp_path / "docs"),
                           portable_checkpoint=portable, checkpoint_binding=binding, wandb=True)
    try:
        if portable:
            with pytest.raises(PortableCheckpointError, match="Portable replay requires wandb disabled"):
                Simulation(model, args=args, output_dir=str(tmp_path / "checkpoints"),
                           funding_allocation_mode="fixed")
        else:
            sim = Simulation(model, args=args, output_dir=str(tmp_path / "checkpoints"),
                             funding_allocation_mode="fixed")
            assert sim.wandb_logger.enabled is True
        assert SIMULATION_CONFIG["wandb"]["enabled"] is True
        constructor_only.init.assert_called_once()
    finally:
        cache.close()
