"""CPU only: completed HTTP results are durable before scientific use."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from utopia.models.response_cache import CompletedResponseCache
from utopia.runtime.checkpointing import PortableCheckpointError, PortableStop, RunStop

BINDING = dict(protocol="portable-test-v1", cell="P1F1", model="fake-model",
               model_revision="revision-a", corpus_sha256="corpus-a",
               science_config={"years": 2, "fulltext": True}, seed=42)


def response(text='{"value":7}'):
    return ChatCompletion.model_validate(dict(
        id="fake-completion", object="chat.completion", created=0, model="fake-model",
        choices=[dict(index=0, message=dict(role="assistant", content=text), finish_reason="stop")],
        usage=dict(prompt_tokens=5, completion_tokens=3, total_tokens=8)))


def payload():
    return dict(model="fake-model", messages=[dict(role="user", content="paper")],
                temperature=0.7, max_tokens=8192, extra_body={"seed": 42})


def test_durable_before_exposure_and_relocated_backup(tmp_path):
    original = CompletedResponseCache(tmp_path / "old" / "responses.sqlite", binding=BINDING)
    count = []
    def callback(**kwargs):
        count.append(kwargs)
        return response()
    first = original.create(callback, seed_ctx=("reviews", 1), **payload())
    assert not first._portable_cache_hit
    # Another connection can see the committed response before the caller acts.
    with sqlite3.connect(original.path) as db:
        assert db.execute("SELECT count(*) FROM replies").fetchone()[0] == 1
    destination = tmp_path / "new-machine" / "responses.sqlite"
    original.backup(destination)
    relocated = CompletedResponseCache(destination, binding=BINDING)
    second = relocated.create(callback, seed_ctx=("reviews", 1), **payload())
    assert second._portable_cache_hit
    assert first.model_dump() == second.model_dump()
    assert len(count) == 1
    original.close()
    relocated.close()


def test_cell_config_prompt_and_attempt_isolation(tmp_path):
    path = tmp_path / "responses.sqlite"
    cache = CompletedResponseCache(path, binding=BINDING)
    cache.create(lambda **_: response(), seed_ctx=("review", 1), **payload())
    for field, value in (("cell", "P0F1"), ("model_revision", "revision-b"),
                         ("corpus_sha256", "corpus-b"), ("science_config", {"years": 3})):
        with pytest.raises(PortableCheckpointError):
            CompletedResponseCache(path, binding={**BINDING, field: value})
    with pytest.raises(PortableCheckpointError, match="different payload"):
        cache.create(lambda **_: pytest.fail("must not send"), seed_ctx=("review", 1),
                     **{**payload(), "max_tokens": 8191})
    assert not cache.create(lambda **_: response(), seed_ctx=("review", 1), attempt=1,
                            **payload())._portable_cache_hit
    with pytest.raises(PortableCheckpointError):
        cache.create(lambda **_: response(), **payload())
    cache.close()


def test_owner_thread_duplicates_send_once_and_bad_json_is_reused(tmp_path):
    cache = CompletedResponseCache(tmp_path / "responses.sqlite", binding=BINDING)
    calls = []
    def callback(**kwargs):
        calls.append(kwargs)
        time.sleep(0.02)
        return response("incomplete json")
    results = [cache.create(callback, seed_ctx=("one", 0), **payload()) for _ in range(4)]
    assert len(calls) == 1
    assert all(r.choices[0].message.content == "incomplete json" for r in results)
    assert sum(r._portable_cache_hit for r in results) == 3
    cache.close()


def test_transport_failure_stops_without_poisoning_retry(tmp_path):
    stop = RunStop()
    cache = CompletedResponseCache(tmp_path / "responses.sqlite", binding=BINDING, stop=stop)
    def broken(**kwargs):
        raise TimeoutError("disconnected before receiving a reply")
    with pytest.raises(PortableStop):
        cache.create(broken, seed_ctx=("one", 0), **payload())
    cache.stop = RunStop()
    assert not cache.create(lambda **_: response(), seed_ctx=("one", 0), **payload())._portable_cache_hit
    cache.close()


def test_io_failure_never_becomes_model_fallback(tmp_path):
    cache = CompletedResponseCache(tmp_path / "responses.sqlite", binding=BINDING)
    cache._db.execute("PRAGMA query_only=ON")
    with pytest.raises(PortableCheckpointError):
        cache.create(lambda **_: response(), seed_ctx=("one", 0), **payload())
    cache.close()


def test_server_constructor_relocation_and_strict_exhausted_responses(tmp_path, monkeypatch):
    import utopia.models.models as models
    sent = []
    options = []
    def factory(**kwargs):
        options.append(kwargs)
        def complete(**request):
            sent.append(request)
            return response("this is invalid JSON")
        return SimpleNamespace(models=SimpleNamespace(list=lambda: []),
                               chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    monkeypatch.setattr(models, "OpenAI", factory)
    def new_client(endpoint):
        return models.VLLMServerModel(
            "fake-model", endpoint, run_seed=42, max_concurrent_requests=1,
            response_cache_dir=tmp_path / "cache", checkpoint_binding=BINDING,
            model_revision="revision-a", max_tokens_override=8192,
            sdk_max_retries=0, request_timeout=120.0)
    first = new_client("http://old-host.invalid/v1")
    with pytest.raises(PortableCheckpointError, match="exhausted 3"):
        first.generate_batch(["paper"], response_format={"type": "json_object"}, seed_ctx=("year", 1))
    assert len(sent) == 3
    assert all(p["max_tokens"] == 8192 for p in sent)
    first.response_cache.close()
    relocated = new_client("http://new-host.invalid/v1")
    with pytest.raises(PortableCheckpointError, match="exhausted 3"):
        relocated.generate_batch(["paper"], response_format={"type": "json_object"}, seed_ctx=("year", 1))
    assert len(sent) == 3, "Invalid completed replies must not be re-sampled after resume"
    assert all(o["max_retries"] == 0 and o["timeout"] == 120 for o in options)
    relocated.response_cache.close()


def test_foreign_thread_cannot_write_the_owner_connection(tmp_path):
    cache = CompletedResponseCache(tmp_path / 'responses.sqlite', binding=BINDING)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(cache.create, lambda **kw: pytest.fail('must not send'),
                                 seed_ctx=('one', 0), **payload())
        with pytest.raises(PortableCheckpointError, match='same thread'):
            future.result()
    assert cache.committed_count() == 0
    cache.close()
