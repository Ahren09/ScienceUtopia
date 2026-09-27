"""CPU coverage of deterministic request rejection and unchanged wire behavior."""

from utopia.utils.paths import project_root

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = project_root(__file__)
from utopia.models import request_audit as module
from utopia.models.request_audit import CONTEXT_TOKENS, MODEL, REVISION, RequestAudit, RequestAuditFailure, active_audit_for, request_audit_scope


class FakeTokenizer:
    chat_template = "fixture template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False and add_generation_prompt is True
        return json.dumps(messages) + str(kwargs.get("enable_thinking", True))

    def __call__(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return {"input_ids": [1] * len(text)}


def payload(max_tokens=2048):
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": "system"},
                     {"role": "user", "content": "example"}],
        "temperature": 0.7, "max_tokens": max_tokens,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "seed": 42,
                       "structured_outputs": {"json": {"type": "object"}}},
    }


def response_for(body, reason="stop", completion_tokens=7):
    text = FakeTokenizer().apply_chat_template(
        body["messages"], tokenize=False, add_generation_prompt=True,
        **body["extra_body"].get("chat_template_kwargs", {}))
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=reason)],
        usage=SimpleNamespace(prompt_tokens=len(text), completion_tokens=completion_tokens))


class TestRequestAudit(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "requests.jsonl"

    def scope(self):
        return request_audit_scope(self.path, _tokenizer=FakeTokenizer())

    def test_default_no_active_guard(self):
        self.assertIsNone(active_audit_for("any historical model"))

    def test_valid_call_has_unchanged_payload_and_complete_accounting(self):
        body = payload(8192)
        original = deepcopy(body)
        seen = []
        with self.scope() as audit:
            self.assertIs(active_audit_for(MODEL), audit)
            audit.create(lambda **values: seen.append(values) or response_for(values),
                         seed_ctx=("funding", 3), item_index=2, attempt=1, **body)
        self.assertEqual(body, original)
        self.assertEqual(seen, [original])
        summary = json.loads(self.path.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["n_clients"], 1)
        self.assertEqual(summary["guard_failures"], 0)
        self.assertEqual(summary["n_responses"], 1)
        self.assertEqual(summary["model_revision"], REVISION)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        event = next(r for r in rows if r["event"] == "request_started")
        self.assertEqual(event["max_tokens"], 8192)
        self.assertEqual(event["seed_ctx"], ["funding", 3])
        self.assertEqual(event["request_seed"], 42)
        self.assertEqual(event["attempt"], 1)
        self.assertIsNone(active_audit_for(MODEL))

    def test_exact_context_boundary_and_overflow_rejected_before_http(self):
        body = payload()
        count = response_for(body).usage.prompt_tokens
        with self.scope() as audit:
            active_audit_for(MODEL)
            body["max_tokens"] = CONTEXT_TOKENS - count
            audit.create(response_for_kwargs, **body)
        other = self.path.with_name("overflow.jsonl")
        called = []
        with self.assertRaises(RequestAuditFailure):
            with request_audit_scope(other, _tokenizer=FakeTokenizer()):
                guard = active_audit_for(MODEL)
                body["max_tokens"] += 1
                # Mirrors stock retry handlers: this failure must escape them.
                for attempt in range(3):
                    try:
                        guard.create(lambda **kw: called.append(kw), **body)
                    except Exception:
                        continue
        self.assertEqual(called, [])
        failed = json.loads(other.with_suffix(".summary.json").read_text())
        self.assertEqual(failed["n_requests"], 0)
        self.assertEqual(failed["guard_failures"], 1)
        self.assertEqual(failed["status"], "failed")

    def test_short_or_missing_server_token_provenance_is_fatal(self):
        for usage in (None, SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                      SimpleNamespace(prompt_tokens=True, completion_tokens=1)):
            with self.subTest(usage=usage):
                path = self.path.with_name(f"usage-{id(usage)}.jsonl")
                with self.assertRaises(RequestAuditFailure):
                    with request_audit_scope(path, _tokenizer=FakeTokenizer()):
                        guard = active_audit_for(MODEL)
                        guard.create(lambda **kw: SimpleNamespace(
                            choices=[SimpleNamespace(finish_reason="stop")], usage=usage), **payload())
                self.assertEqual(json.loads(path.with_suffix(".summary.json").read_text())
                                 ["guard_failures"], 1)

    def test_observed_truncation_is_logged_without_changing_stock_retry_policy(self):
        with self.scope() as audit:
            active_audit_for(MODEL)
            result = audit.create(lambda **body: response_for(body, "length", 2048), **payload())
            self.assertEqual(result.choices[0].finish_reason, "length")
        self.assertEqual(audit.summary()["finish_reasons"], {"length": 1})

    def test_received_response_survives_fatal_provenance_rejection_without_retry(self):
        for violation in ("usage", "choices"):
            with self.subTest(violation=violation):
                path = self.path.with_name(f"received-{violation}.jsonl")
                body = payload(8192)
                received = response_for(body)
                received.choices[0].message = SimpleNamespace(
                    content='{"next_application_id": 2}',
                    reasoning_content="fixture reasoning")
                if violation == "usage":
                    received.usage.prompt_tokens += 1
                else:
                    received.choices.append(deepcopy(received.choices[0]))
                callback = Mock(return_value=received)
                with self.assertRaises(RequestAuditFailure) as caught:
                    with request_audit_scope(path, _tokenizer=FakeTokenizer()):
                        guard = active_audit_for(MODEL)
                        guard.create(callback, seed_ctx=("funding", 3),
                                     item_index=0, attempt=2, **body)
                callback.assert_called_once_with(**body)
                error = caught.exception
                self.assertIs(error.received_response, received)
                self.assertEqual(error.request_ticket["request_id"], 0)
                self.assertEqual(error.request_ticket["seed_ctx"], ["funding", 3])
                self.assertEqual(error.request_ticket["item_index"], 0)
                self.assertEqual(error.request_ticket["attempt"], 2)
                summary = json.loads(path.with_suffix(".summary.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["n_requests"], 1)
                self.assertEqual(summary["n_responses"], 1)
                self.assertEqual(summary["n_transport_errors"], 0)
                self.assertEqual(summary["pending_requests"], 0)
                self.assertEqual(summary["guard_failures"], 1)

    def test_transport_error_is_logged_and_original_error_propagates(self):
        with self.scope() as audit:
            active_audit_for(MODEL)
            def fail(**kwargs):
                raise ValueError("fixture transport failure")
            with self.assertRaisesRegex(ValueError, "fixture"):
                audit.create(fail, **payload())
            audit.create(response_for_kwargs, attempt=1, **payload())
        self.assertEqual(audit.summary()["n_transport_errors"], 1)
        self.assertEqual(audit.summary()["n_requests"], 2)

    def test_concurrent_calls_do_not_corrupt_audit_or_change_rng(self):
        python_before = random.getstate()
        with self.scope() as audit:
            active_audit_for(MODEL)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(audit.create, response_for_kwargs,
                                           item_index=i, **payload()) for i in range(40)]
                for future in futures:
                    future.result()
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(audit.summary()["n_responses"], 40)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        ids = [r["request_id"] for r in rows if r["event"] == "request_started"]
        self.assertEqual(sorted(ids), list(range(40)))

    def test_tokenizer_setup_restores_loaded_rngs(self):
        import numpy as np
        python_before, numpy_before = random.getstate(), np.random.get_state()
        with module._preserve_rng():
            random.random()
            np.random.random()
        self.assertEqual(random.getstate(), python_before)
        self.assertTrue((np.random.get_state()[1] == numpy_before[1]).all())
        self.assertEqual(np.random.get_state()[2:], numpy_before[2:])

    def test_tokenizer_uses_only_verified_pinned_local_snapshot_and_preserves_rng(self):
        snapshot = Path(self.directory.name) / "models--Qwen--Qwen3-32B" / "snapshots" / REVISION
        snapshot.mkdir(parents=True)
        tokenizer = FakeTokenizer()
        def resolve(**kwargs):
            random.random()
            return str(snapshot)
        def load(*args, **kwargs):
            random.random()
            return tokenizer
        download, pretrained = Mock(side_effect=resolve), Mock(side_effect=load)
        before = random.getstate()
        with patch.dict(sys.modules, {
            "huggingface_hub": SimpleNamespace(snapshot_download=download),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=pretrained)),
        }):
            self.assertIs(module._load_tokenizer(), tokenizer)
        download.assert_called_once_with(repo_id=MODEL, revision=REVISION,
                                             allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
        pretrained.assert_called_once_with(
            str(snapshot.resolve()), local_files_only=True, trust_remote_code=False)
        self.assertEqual(random.getstate(), before)

    def test_missing_or_wrong_snapshot_fails_before_tokenizer_load_without_fallback(self):
        wrong = Path(self.directory.name) / "models--Qwen--Qwen3-32B" / "snapshots" / ("0" * 40)
        wrong.mkdir(parents=True)
        missing = wrong.with_name(REVISION)
        for resolver in (Mock(side_effect=OSError("pinned snapshot is not cached")),
                         Mock(return_value=str(missing)), Mock(return_value=str(wrong))):
            with self.subTest(resolver=resolver):
                pretrained = Mock(side_effect=AssertionError("must not fall back to a Hub model ID"))
                before = random.getstate()
                with patch.dict(sys.modules, {
                    "huggingface_hub": SimpleNamespace(snapshot_download=resolver),
                    "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(
                        from_pretrained=pretrained)),
                }):
                    with self.assertRaises(RequestAuditFailure):
                        RequestAudit(self.path)
                pretrained.assert_not_called()
                self.assertEqual(random.getstate(), before)
                self.assertFalse(self.path.exists())

    def test_wrong_model_second_client_and_empty_scope_fail_closed(self):
        with self.assertRaises(RequestAuditFailure):
            with self.scope():
                active_audit_for("unreviewed-model")
        with self.assertRaises(RequestAuditFailure):
            with request_audit_scope(self.path.with_name("second.jsonl"),
                                     _tokenizer=FakeTokenizer()):
                active_audit_for(MODEL)
                active_audit_for(MODEL)
        with self.assertRaises(RequestAuditFailure):
            with request_audit_scope(self.path.with_name("empty.jsonl"),
                                     _tokenizer=FakeTokenizer()):
                pass

    def test_existing_artifacts_are_never_adopted_or_truncated(self):
        self.path.write_text("old evidence")
        with self.assertRaises(RequestAuditFailure):
            with self.scope():
                pass
        self.assertEqual(self.path.read_text(), "old evidence")

    def test_scope_body_failure_restores_hook_and_marks_failed(self):
        with self.assertRaisesRegex(RuntimeError, "simulation"):
            with self.scope() as audit:
                active_audit_for(MODEL)
                audit.create(response_for_kwargs, **payload())
                raise RuntimeError("simulation failed")
        self.assertIsNone(active_audit_for(MODEL))
        self.assertEqual(audit.summary()["status"], "failed")

    def test_nested_scope_cannot_replace_active_audit(self):
        with self.scope() as audit:
            with self.assertRaises(RequestAuditFailure):
                with request_audit_scope(self.path.with_name("nested.jsonl"),
                                         _tokenizer=FakeTokenizer()):
                    pass
            self.assertIs(active_audit_for(MODEL), audit)
            audit.create(response_for_kwargs, **payload())

    def test_invalid_overrides_and_legacy_transport_fail_before_http(self):
        variants = []
        for value in (True, None, 0, -1, 8192.0):
            body = payload()
            body["max_tokens"] = value
            variants.append(body)
        for key in ("guided_json", "truncate_prompt_tokens"):
            body = payload()
            body["extra_body"][key] = 1
            variants.append(body)
        body = payload()
        body["extra_body"]["chat_template_kwargs"]["unreviewed"] = True
        variants.append(body)
        for i, body in enumerate(variants):
            with self.subTest(i=i):
                with self.assertRaises(RequestAuditFailure):
                    with request_audit_scope(self.path.with_name(f"invalid{i}.jsonl"),
                                             _tokenizer=FakeTokenizer()) as audit:
                        active_audit_for(MODEL)
                        audit.create(lambda **kw: self.fail("HTTP must not occur"), **body)

    def test_failed_audit_write_is_not_swallowed_as_model_failure(self):
        with self.assertRaises(RequestAuditFailure):
            with self.scope() as audit:
                active_audit_for(MODEL)
                original = os.write
                count = 0
                def fail_once(fd, data):
                    nonlocal count
                    count += 1
                    if count == 1:
                        raise OSError("fixture disk failure")
                    return original(fd, data)
                with patch.object(os, "write", fail_once):
                    audit.create(lambda **kw: self.fail("HTTP must not occur"), **payload())
        self.assertEqual(audit.summary()["status"], "failed")


def response_for_kwargs(**body):
    return response_for(body)


class TestActualClientAudit(unittest.TestCase):
    def test_both_actual_sdk_paths_keep_request_overrides_and_audit_usage(self):
        import tests.support.model_transport as transport
        for batched in (False, True):
            with self.subTest(batched=batched), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "requests.jsonl"
                class MatchingWire(transport.WireRecorder):
                    def handle(self, request):
                        result = super().handle(request)
                        if request.method == "POST":
                            data = result.json()
                            body = json.loads(request.content)
                            text = FakeTokenizer().apply_chat_template(
                                body["messages"], tokenize=False, add_generation_prompt=True,
                                **body.get("chat_template_kwargs", {}))
                            data["usage"]["prompt_tokens"] = len(text)
                            data["usage"]["total_tokens"] = len(text) + 7
                            return transport.httpx.Response(200, json=data)
                        return result
                case = transport.TransportFixture()
                case.setUp()
                try:
                    with request_audit_scope(path, _tokenizer=FakeTokenizer()) as audit:
                        client, wire = case.make_client(MatchingWire())
                        kwargs = {"seed_ctx": ("funding", 1), "max_tokens": 8192,
                                  "response_format": next(iter(transport.FORMATS.values()))}
                        if batched:
                            client.generate_batch(["fixture"], **kwargs)
                        else:
                            client.generate(prompt="fixture", **kwargs)
                    self.assertEqual(wire.payloads[0]["max_tokens"], 8192)
                    self.assertEqual(audit.summary()["status"], "complete")
                    self.assertEqual(audit.summary()["n_responses"], 1)
                finally:
                    case.doCleanups()

    @unittest.skipUnless(os.environ.get("UTOPIA_ACTUAL_TOKENIZER_TESTS") == "1",
                         "Opt-in pinned local tokenizer CPU test")
    def test_pinned_tokenizer_two_step_count_matches_direct_template_ids(self):
        # Run in the actual private runtime with HF_HUB_OFFLINE=1 and the
        # production HF_HOME. A metadata call is forbidden even if Hub offline
        # mode would stop it before a network connection.
        import huggingface_hub
        import socket
        with patch.object(huggingface_hub, "model_info",
                          side_effect=AssertionError("offline loader must not query model_info")), \
                patch.object(socket, "create_connection",
                             side_effect=AssertionError("tokenizer setup must stay offline")), \
                patch.object(socket.socket, "connect",
                             side_effect=AssertionError("tokenizer setup must stay offline")), \
                tempfile.TemporaryDirectory() as directory:
            import numpy as np
            import torch
            python_before, numpy_before = random.getstate(), np.random.get_state()
            torch_before = torch.get_rng_state().clone()
            tokenizer = module._load_tokenizer()
            self.assertTrue(Path(tokenizer.name_or_path).is_absolute())
            self.assertEqual(Path(tokenizer.name_or_path).name, REVISION)
            with request_audit_scope(Path(directory) / "actual.jsonl") as audit:
                active_audit_for(MODEL)
                body = payload(8192)
                direct = tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True,
                    enable_thinking=True)
                def respond(**unused):
                    return SimpleNamespace(
                        choices=[SimpleNamespace(finish_reason="stop")],
                        usage=SimpleNamespace(prompt_tokens=len(direct), completion_tokens=7))
                audit.create(respond, **body)
            self.assertEqual(audit.summary()["prompt_tokens"], len(direct))
            self.assertEqual(random.getstate(), python_before)
            self.assertTrue((np.random.get_state()[1] == numpy_before[1]).all())
            self.assertEqual(np.random.get_state()[2:], numpy_before[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))


if __name__ == "__main__":
    unittest.main()
