"""Explicit prospective thinking choices reach wire, audit and durable binding."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests.support import model_transport as transport
from utopia.experiments.funding_feedback import prospective_arguments
from utopia.models import models
from utopia.models.request_audit import request_audit_scope, active_audit_for, RequestAuditFailure
from utopia.models.response_cache import CompletedResponseCache
from utopia.runtime.checkpointing import PortableCheckpointError
from tests.models.test_response_cache import BINDING, payload, response


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("configured", [None, False, True])
@pytest.mark.parametrize("override", [None, False, True])
def test_explicit_choice_preserves_schema_seed_and_wire(batched, configured, override):
    case = transport.TransportFixture()
    case.setUp()
    try:
        client, wire = case.make_client(thinking=configured, thinking_override=override)
        fmt = next(iter(transport.FORMATS.values()))
        options = dict(response_format=fmt, seed_ctx=("review", 1), max_tokens=32768)
        if batched:
            client.generate_batch(["complete paper fixture"], **options)
        else:
            client.generate(prompt="complete paper fixture", **options)
        body, = wire.payloads
        effective = configured if override is None else override
        if override is None and not configured:
            assert "chat_template_kwargs" not in body
        else:
            assert body["chat_template_kwargs"] == {"enable_thinking": effective}
        assert body["structured_outputs"] == {"json": fmt["json_object"]["schema"]}
        assert body["max_tokens"] == 32768
        assert body["seed"] == transport.SEED_MODULE.derive_seed(42, "review", 1, 0, 0)
        assert client.enable_thinking is effective
    finally:
        case.doCleanups()


@pytest.mark.parametrize("override", [0, 1, "false", [], {}])
def test_invalid_override_rejected_before_http(monkeypatch, override):
    monkeypatch.setattr(models, "OpenAI", lambda **_: pytest.fail("No HTTP client allowed"))
    with pytest.raises(ValueError, match="thinking_override"):
        models.VLLMServerModel("fake-model", thinking_override=override)


@pytest.mark.parametrize("bound,override", [(None, False), (False, None), (True, False), (0, False)])
def test_binding_mismatch_rejected_before_cache_or_http(tmp_path, monkeypatch, bound, override):
    binding = deepcopy(BINDING)
    if bound is not None:
        binding["science_config"]["thinking_override"] = bound
    monkeypatch.setattr(models, "OpenAI", lambda **_: pytest.fail("No HTTP client allowed"))
    with pytest.raises(PortableCheckpointError, match="Thinking override"):
        models.VLLMServerModel("fake-model", model_revision="revision-a", run_seed=42,
                               sdk_max_retries=0, response_cache_dir=tmp_path,
                               checkpoint_binding=binding, thinking_override=override)
    assert not list(tmp_path.iterdir())


def test_changed_thinking_cannot_reuse_durable_response(tmp_path):
    binding = deepcopy(BINDING)
    binding["science_config"].update(thinking_override=False, max_tokens=32768, context_tokens=196608)
    db = tmp_path / "responses.sqlite"
    original = CompletedResponseCache(db, binding=binding)
    request = payload()
    request["max_tokens"] = 32768
    request["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
    original.create(lambda **_: response(), seed_ctx=("review", 1), **request)
    assert original.create(lambda **_: pytest.fail("Must replay"), seed_ctx=("review", 1), **request)._portable_cache_hit
    changed = deepcopy(request)
    changed["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
    with pytest.raises(PortableCheckpointError, match="different payload"):
        original.create(lambda **_: pytest.fail("Must refuse"), seed_ctx=("review", 1), **changed)
    original.close()
    for field, value in [("thinking_override", True), ("max_tokens", 8192), ("context_tokens", 131072)]:
        changed = deepcopy(binding)
        changed["science_config"][field] = value
        with pytest.raises(PortableCheckpointError):
            CompletedResponseCache(db, binding=changed)


@pytest.mark.parametrize("excess", [0, 1])
def test_full_input_output_reservation_uses_explicit_template(tmp_path, excess):
    class Tokenizer:
        chat_template = "fixture"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["tokenize"] is False
            return "untruncated fixture"
        def __call__(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False, truncation=False)
            return {"input_ids": [1] * (196608 - 32768 + excess)}
    model = "fake-model"
    request = payload()
    request["max_tokens"] = 32768
    request["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
    sent = []
    def complete(**kwargs):
        sent.append(kwargs)
        result = response()
        result.usage.prompt_tokens = 196608 - 32768
        result.usage.total_tokens = result.usage.prompt_tokens + result.usage.completion_tokens
        return result
    def run():
        with request_audit_scope(tmp_path / "audit.jsonl", model_name=model,
                                 model_revision="a" * 40, context_tokens=196608, sdk_retries=0,
                                 _tokenizer=Tokenizer()) as audit:
            active_audit_for(model)
            audit.create(complete, seed_ctx=("review", 1), **request)
    if excess:
        with pytest.raises(RequestAuditFailure, match="exceeds pinned context"):
            run()
        assert not sent
    else:
        run()
        assert len(sent) == 1


def test_prospective_cli_is_tristate():
    parser = argparse.ArgumentParser()
    prospective_arguments(parser)
    required = ["--cell", "P1F1", "--run-id", "fixture", "--vllm-url", "http://local/v1",
                "--server-manifest", "server.json", "--fulltext-index", "index.json",
                "--deadline-epoch", "4102444800", "--s3-prefix", "s3://fixture"]
    assert parser.parse_args(required).thinking is None
    assert parser.parse_args(required + ["--thinking"]).thinking is True
    assert parser.parse_args(required + ["--no-thinking"]).thinking is False
