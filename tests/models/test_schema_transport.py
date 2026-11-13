from __future__ import annotations

from tests.support.model_transport import TransportFixture

import utopia.funding.feedback as feedback

from utopia.utils.paths import project_root

import ast

from concurrent.futures import ThreadPoolExecutor, as_completed

from copy import deepcopy

from enum import Enum

import importlib.util

import json

import logging

import os

from pathlib import Path

import random

import re

import subprocess

import sys

import threading

import time

from types import ModuleType

from typing import Dict, List, Optional, Tuple, Union

import unittest

from unittest.mock import patch

import httpx

import jsonschema

from openai import OpenAI

from pydantic import BaseModel

from utopia.constants import IMPORTANT_NOTES

from utopia.config import SIMULATION_CONFIG

from tests.support.model_transport import (
    ROOT,
    load_shared_helper,
    BUILD_EXTRA_BODY,
    compile_nodes,
    client_definitions,
    CLIENT_NS,
    Client,
    actual_seed_module,
    SEED_MODULE,
    actual_response_formats,
    FORMATS,
    WireRecorder,
)

class TestVLLMSchemaTransport(TransportFixture):

    def test_thinking_budget_preserves_output_room_and_structured_answer(self):
        for batched in (False, True):
            client, wire = self.make_client()
            client.thinking_token_budget = 8192
            schema = next(iter(FORMATS.values()))
            if batched:
                client.generate_batch(['budget'], max_tokens=16384,
                                      response_format=schema, seed_ctx=('budget',))
            else:
                client.generate('budget', max_tokens=16384,
                                response_format=schema, seed_ctx=('budget',))
            payload = wire.payloads[0]
            self.assertEqual(payload['thinking_token_budget'], 8192)
            self.assertEqual(payload['max_tokens'], 16384)
            self.assertEqual(payload['structured_outputs']['json'],
                             schema['json_object']['schema'])
            self.assertEqual(payload['chat_template_kwargs'], {'enable_thinking': True})

    def test_configured_output_budget_applies_to_both_generation_paths(self):
        for batched in (False, True):
            client, wire = self.make_client()
            client.completion_token_floor = 8192
            if batched:
                client.generate_batch(['budget'], max_tokens=2048)
            else:
                client.generate('budget', max_tokens=2048)
            self.assertEqual(wire.payloads[0]['max_tokens'], 8192)

    def test_valid_json_at_output_limit_is_not_a_complete_response(self):
        client, wire = self.make_client()
        create_completion = client.create_completion

        def truncated_completion(**kwargs):
            response = create_completion(**kwargs)
            response.choices[0].finish_reason = 'length'
            return response

        captured = []
        with patch.object(client, 'create_completion', truncated_completion):
            result = client.generate_batch(
                ['truncated JSON'], response_format=next(iter(FORMATS.values())),
                on_result=lambda index, value, metadata: captured.append(metadata))
        self.assertIsNone(result[0][0])
        self.assertEqual(len(wire.payloads), 3)
        self.assertEqual(captured[0]['finish_reason'], 'length')
        self.assertFalse(captured[0]['success'])

    def test_saved_request_ids_resume_without_new_http_calls(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            client, wire = self.make_client()
            client.response_cache_path = Path(directory) / 'responses.jsonl'
            client.response_write_lock = threading.Lock()
            client.saved_responses = {}
            first = client.generate_batch(
                ['first', 'second'], item_ids=['a', 'b'], seed_ctx=('world', 1))
            captured = []
            resumed = client.generate_batch(
                ['second'], item_ids=['b'], seed_ctx=('world', 1),
                on_result=lambda index, value, metadata: captured.append(metadata))
            self.assertEqual(len(wire.payloads), 2)
            self.assertEqual(first[1][0], resumed[0][0])
            rows = [json.loads(line) for line in client.response_cache_path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row['cache_replay'] is False for row in rows))
            self.assertTrue(captured[0]['cache_replay'])
            self.assertEqual(len({row['request_id'] for row in rows}), 2)

    def test_item_id_seeds_survive_reordering_and_callbacks_capture_new_replies(self):
        first, first_wire = self.make_client()
        captured = []
        first.generate_batch(
            ['first', 'second'], item_ids=['paper-a', 'paper-b'], seed_ctx=('replay',),
            on_result=lambda index, result, metadata: captured.append((index, result, metadata)),
        )
        second, second_wire = self.make_client()
        second.generate_batch(['second'], item_ids=['paper-b'], seed_ctx=('replay',))
        before = next(p for p in first_wire.payloads if p['messages'][-1]['content'].endswith('second'))
        self.assertEqual(before['seed'], second_wire.payloads[0]['seed'])
        self.assertEqual(len(captured), 2)
        self.assertTrue(all(row[2]['cache_replay'] is False for row in captured))
        self.assertTrue(all(row[2]['time_ns'] > 0 for row in captured))

    def test_retry_index_collision_preserves_both_inputs_and_budget_changes_reuse(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            client, wire = self.make_client()
            client.response_cache_path = Path(directory) / 'responses.jsonl'
            client.generate_batch(['panel A'], seed_ctx=('funding', 1, 1))
            client.generate_batch(['panel B'], seed_ctx=('funding', 1, 1))
            self.assertEqual(len(wire.payloads), 2)
            client.completion_token_floor = 16384
            client.thinking_token_budget = 8192
            for prompt in ('panel A', 'panel B'):
                captured = []
                client.generate_batch(
                    [prompt], seed_ctx=('funding', 1, 1),
                    on_result=lambda index, result, metadata: captured.append(metadata))
                self.assertTrue(captured[0]['cache_replay'])
            self.assertEqual(len(wire.payloads), 2)
            rows = [json.loads(line) for line in client.response_cache_path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['request_id'], rows[1]['request_id'])
            self.assertNotEqual(rows[0]['payload']['messages'], rows[1]['payload']['messages'])

    def test_batch_keeps_each_panel_schema_attached_to_its_prompt(self):
        client, wire = self.make_client()
        formats = [{
            'type': 'json_schema',
            'json_object': {'schema': {
                'type': 'object', 'properties': {'panel': {'const': name}},
                'required': ['panel'], 'additionalProperties': False,
            }},
        } for name in ('A', 'B')]
        client.generate_batch(['panel A', 'panel B'], response_formats=formats)
        for payload in wire.payloads:
            name = payload['messages'][-1]['content'][-1]
            self.assertEqual(payload['structured_outputs']['json']['properties']['panel'],
                             {'const': name})


    def test_every_current_agent_schema_reaches_both_wire_paths_unchanged(self):
        # Catch accidental omission of an emitter from this regression suite.
        self.assertEqual(len(FORMATS), 11)
        for name, fmt in FORMATS.items():
            for batched in (False, True):
                with self.subTest(schema=name, batched=batched):
                    client, wire = self.make_client()
                    original = deepcopy(fmt)
                    schema = fmt["json_object"]["schema"]
                    jsonschema.Draft202012Validator.check_schema(schema)
                    prompt = f"Schema fixture: {name}"
                    kwargs = dict(response_format=fmt, seed_ctx=("transport", 3),
                                  temperature=0.35, max_tokens=137, system_prompt="System fixture.")
                    if batched:
                        client.generate_batch([prompt], **kwargs)
                    else:
                        client.generate(prompt=prompt, **kwargs)
                    self.assertEqual(fmt, original)
                    self.assertEqual(len(wire.payloads), 1)
                    body = wire.payloads[0]
                    self.assertEqual(body["structured_outputs"], {"json": schema})
                    self.assertNotIn("guided_json", body)
                    self.assertNotIn("response_format", body)  # One supported constraint transport.
                    self.assertEqual(body["messages"], [
                        {"role": "system", "content": "System fixture."},
                        {"role": "user", "content": f"{IMPORTANT_NOTES}\n\n{prompt}" if batched else prompt},
                    ])
                    self.assertEqual(body["seed"], SEED_MODULE.derive_seed(42, "transport", 3, 0, 0))
                    self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
                    self.assertEqual(body["temperature"], 0.35)
                    self.assertEqual(body["max_tokens"], 137)
                    self.assertEqual(body["model"], "Qwen/Qwen3-32B")
                    self.assertEqual(set(body), {"model", "messages", "temperature", "max_tokens",
                                                 "structured_outputs", "seed", "chat_template_kwargs"})

    def test_shared_helper_loads_by_file_without_package_or_runtime_imports(self):
        program = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("standalone_schema_helper", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ("utopia", "torch", "vllm", "transformers", "openai", "pydantic"):
    assert name not in sys.modules, name
schema = {"type": "object", "properties": {"nonce": {"const": "fixture-nonce"}},
          "required": ["nonce"], "additionalProperties": False}
value = module.build_vllm_extra_body(schema)
assert value["structured_outputs"]["json"] is schema
assert value["chat_template_kwargs"] == {"enable_thinking": True}
print(json.dumps(value))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", program,
             str(ROOT / "utopia/models/structured_outputs.py")],
            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["structured_outputs"]["json"]["required"],
                         ["nonce"])

    def test_shared_helper_preserves_optional_schema_and_thinking_semantics(self):
        for schema in (None, {}):
            self.assertEqual(BUILD_EXTRA_BODY(schema),
                             {"chat_template_kwargs": {"enable_thinking": True}})
            for thinking in (False, None):
                self.assertEqual(BUILD_EXTRA_BODY(schema, enable_thinking=thinking), {})
        schema = {"type": "object", "$defs": {"value": {"type": "integer"}}}
        original = deepcopy(schema)
        value = BUILD_EXTRA_BODY(schema, enable_thinking=False)
        self.assertEqual(value, {"structured_outputs": {"json": schema}})
        self.assertIs(value["structured_outputs"]["json"], schema)
        self.assertEqual(schema, original)

    def test_schema_object_is_not_rewritten_and_no_schema_behavior_is_unchanged(self):
        client, wire = self.make_client(thinking=False, seed=None)
        fmt = next(iter(FORMATS.values()))
        self.assertIs(client.build_extra_body(fmt)["structured_outputs"]["json"],
                      fmt["json_object"]["schema"])
        for absent in (None, {}, {"type": "json_object"}, {"type": "json_schema", "json_object": {}}):
            self.assertEqual(client.build_extra_body(absent), {})
        client.generate(prompt="unconstrained fixture", response_format=None,
                        system_prompt=None, temperature=0.6, max_tokens=23)
        self.assertEqual(wire.payloads, [{
            "model": "Qwen/Qwen3-32B", "messages": [{"role": "user", "content": "unconstrained fixture"}],
            "temperature": 0.6, "max_tokens": 23,
        }])

    def test_history_prompt_is_preserved_without_mutating_the_callers_history(self):
        client, wire = self.make_client()
        history = [{"role": "system", "content": "history system"},
                   {"role": "assistant", "content": "history answer"}]
        before = deepcopy(history)
        client.generate(prompt="follow-up", message_history=history,
                        response_format=next(iter(FORMATS.values())), seed_ctx=("history", 2))
        self.assertEqual(history, before)
        self.assertEqual(wire.payloads[0]["messages"], before + [
            {"role": "user", "content": "follow-up"}])

    def test_thinking_toggle_leaves_schema_and_other_request_fields_unchanged(self):
        bodies = []
        for thinking in (True, False):
            client, wire = self.make_client(thinking=thinking)
            client.generate_batch(["thinking fixture"], response_format=next(iter(FORMATS.values())),
                                  seed_ctx=("thinking", 2), temperature=0.4, max_tokens=91)
            body = wire.payloads[0]
            if thinking:
                self.assertEqual(body.pop("chat_template_kwargs"), {"enable_thinking": True})
            else:
                self.assertNotIn("chat_template_kwargs", body)
            bodies.append(body)
        self.assertEqual(bodies[0], bodies[1])

    def test_production_defaults_match_the_mandatory_probe_options(self):
        self.assertIs(SIMULATION_CONFIG["llm"]["enable_thinking"], True)
        self.assertEqual(SIMULATION_CONFIG["llm"]["max_tokens"], 2048)
        for batched in (False, True):
            with self.subTest(batched=batched):
                client, wire = self.make_client()
                kwargs = dict(response_format=next(iter(FORMATS.values())),
                              seed_ctx=("production_defaults", 0))
                if batched:
                    client.generate_batch(["default options fixture"], **kwargs)
                else:
                    client.generate(prompt="default options fixture", **kwargs)
                body = wire.payloads[0]
                self.assertEqual(body["max_tokens"], 2048)
                self.assertEqual(body["temperature"], 0.7)
                self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
                self.assertEqual(body["seed"],
                                 SEED_MODULE.derive_seed(42, "production_defaults", 0, 0, 0))

    def test_batch_retries_preserve_constraints_and_derive_attempt_seed(self):
        wire = WireRecorder(lambda body, index: "invalid JSON" if index == 0 else '{"ok": true}')
        client, wire = self.make_client(wire)
        fmt = next(iter(FORMATS.values()))
        result = client.generate_batch(["retry fixture"], response_format=fmt,
                                       seed_ctx=("phase1_directions", 1))
        self.assertEqual(result[0][0], {"ok": True})
        self.assertEqual([b["seed"] for b in wire.payloads],
                         [SEED_MODULE.derive_seed(42, "phase1_directions", 1, 0, i) for i in (0, 1)])
        without_seed = [{k: v for k, v in b.items() if k != "seed"} for b in wire.payloads]
        self.assertEqual(without_seed[0], without_seed[1])
        self.assertEqual(client.call_stats["n_retries"], 1)
        self.assertEqual(client.call_stats["n_first_attempt_success"], 0)

    def test_empty_batch_sends_no_http_request(self):
        client, wire = self.make_client()
        self.assertEqual(client.generate_batch([], response_format=next(iter(FORMATS.values()))), [])
        self.assertEqual(wire.payloads, [])
        self.assertEqual(client.call_stats["n_prompts"], 0)

    def test_concurrent_batch_preserves_result_order_and_item_seeds(self):
        def respond(body, index):
            return json.dumps({"prompt": body["messages"][-1]["content"]})
        client, wire = self.make_client(WireRecorder(respond))
        fmt = next(iter(FORMATS.values()))
        state = random.getstate()
        result = client.generate_batch(["first", "second"], response_format=fmt,
                                       seed_ctx=("phase3_reviews", 4))
        self.assertEqual(random.getstate(), state)
        self.assertEqual([r[0]["prompt"] for r in result],
                         [f"{IMPORTANT_NOTES}\n\n{p}" for p in ("first", "second")])
        self.assertEqual({b["seed"] for b in wire.payloads},
                         {SEED_MODULE.derive_seed(42, "phase3_reviews", 4, i, 0) for i in (0, 1)})

    def test_exhausted_parse_retries_keep_original_failure_contract(self):
        client, wire = self.make_client(WireRecorder(lambda body, index: "<think>unfinished"))
        result = client.generate_batch(["failure"], response_format=next(iter(FORMATS.values())),
                                       seed_ctx=("failure", 1))
        self.assertIsNone(result[0][0])
        self.assertEqual(len(wire.payloads), 3)
        self.assertEqual(client.call_stats["n_failures"], 1)
        self.assertEqual(client.call_stats["n_retries"], 3)

    def test_transport_does_not_replace_existing_semantic_validation(self):
        # The mock deliberately violates the supplied schema. The compatibility
        # fix must not silently add parsing/validation/fallback policy changes.
        client, wire = self.make_client()
        result = client.generate_batch(["semantic fixture"], response_format=next(iter(FORMATS.values())))
        self.assertEqual(result[0][0], {"sentinel": True})
        self.assertEqual(client.call_stats["n_first_attempt_success"], 1)
        self.assertEqual(len(wire.payloads), 1)

    def test_mechanism_cells_keep_factual_awards_and_send_the_same_schema(self):
        import tests.support.simulation as harness
        import utopia.experiments.funding_feedback as jm
        import tempfile
        states = {}
        payloads = {}
        for cell in feedback.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                sim, agents, _ = harness.make_simulation(directory, cell)
                def respond(body, index):
                    prompt = body["messages"][-1]["content"]
                    if "### Application " in prompt:
                        entries = re.findall(r"### Application (\d+)\nApplicant ID: ([^\n]+)", prompt)
                        return json.dumps({"ranked_applications": [
                            {"application_id": int(i), "applicant_id": aid, "rank": rank + 1,
                             "reason": "transport fixture"}
                            for rank, (i, aid) in enumerate(entries)
                        ]})
                    return json.dumps({"submit": True, "research_proposal": "Transport fixture.",
                                       "relevant_projects": [0]})
                sim.llm, wire = self.make_client(WireRecorder(respond))
                # The funding harness normally stubs schema generation only.
                # Supply the real Pydantic schema emitter while retaining the
                # original funding methods, costs, awards and attrition.
                schema_methods = {}
                for cls_name in ("ProgramApplication", "FundingEvaluationResponse"):
                    fmt = next(v for v in FORMATS.values()
                               if v["json_object"]["schema"]["title"] == cls_name)
                    schema_methods[cls_name] = fmt["json_object"]["schema"]
                with patch.object(harness.ORIGINAL["ProgramApplication"], "model_json_schema",
                                  return_value=schema_methods["ProgramApplication"]):
                    with patch.object(harness.ORIGINAL["FundingEvaluationResponse"], "model_json_schema",
                                      return_value=schema_methods["FundingEvaluationResponse"]):
                        result = harness.run_phase(sim)
                self.assertEqual(sum(e["earned_amount"] for e in result["funding_feedback"]["awards"]), 40)
                self.assertEqual(agents[0].resources, 140 if cell[3] == "1" else 100)
                self.assertEqual(agents[1].resources, 100)
                self.assertTrue(all("guided_json" not in b and "structured_outputs" in b for b in wire.payloads))
                states[cell] = [a.funding_success_history for a in agents]
                payloads[cell] = sorted(wire.payloads, key=lambda body: body["seed"])
        self.assertTrue(all(value == states["P1F1"] for value in states.values()))
        self.assertEqual(payloads["P1F1"], payloads["P1F0"])
        self.assertEqual(payloads["P0F1"], payloads["P0F0"])
        for visible, hidden in zip(payloads["P1F1"], payloads["P0F1"]):
            self.assertEqual({k: v for k, v in visible.items() if k != "messages"},
                             {k: v for k, v in hidden.items() if k != "messages"})


class TestBedrockBatchTransport(TransportFixture):
    def make_bedrock(self, recorder=None, concurrency=2, cache=None):
        recorder = recorder or WireRecorder()
        http = httpx.Client(transport=httpx.MockTransport(recorder.handle))
        self.addCleanup(http.close)
        with patch.dict(CLIENT_NS, {'OpenAI': lambda **kw: OpenAI(http_client=http, **kw)}):
            with patch.dict(os.environ, {'AWS_BEARER_TOKEN_BEDROCK': 'fixture-key'}):
                client = CLIENT_NS['BedrockModel'](
                    'openai.gpt-5.6-luna', base_url='http://schema-fixture.invalid/openai/v1',
                    max_concurrent_requests=concurrency, response_cache_path=cache,
                    completion_token_floor=32768)
        return client, recorder

    def test_entire_body_budget_order_concurrency_and_id_resume(self):
        import tempfile
        active = peak = 0
        lock = threading.Lock()
        completion_order = []
        def respond(body, index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            text = body['messages'][-1]['content']
            # Force replies to complete out of input order.
            time.sleep(0.06 if text.endswith('TAIL-0') else 0.005)
            with lock:
                active -= 1
                completion_order.append(int(text[-1]))
            return json.dumps({'item': int(text[-1])})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'replies.jsonl'
            client, wire = self.make_bedrock(WireRecorder(respond), cache=cache)
            prompts = ['body token ' * 7000 + f'TAIL-{i}' for i in range(4)]
            result = client.generate_batch(prompts, item_ids=list('abcd'), seed_ctx=('fulltext',))
            self.assertEqual([r[0]['item'] for r in result], list(range(4)))
            self.assertEqual(peak, 2)
            self.assertNotEqual(completion_order, list(range(4)))
            for body in wire.payloads:
                i = int(body['messages'][-1]['content'][-1])
                self.assertEqual(body['messages'][-1]['content'], IMPORTANT_NOTES + '\n\n' + prompts[i])
                self.assertEqual(body['max_completion_tokens'], 32768)
                self.assertEqual(body['reasoning_effort'], 'none')
            resumed, resume_wire = self.make_bedrock(cache=cache)
            result = resumed.generate_batch([prompts[3], prompts[0]], item_ids=['d','a'], seed_ctx=('fulltext',))
            self.assertEqual([r[0]['item'] for r in result], [3,0])
            self.assertFalse(resume_wire.payloads)

    def test_each_funding_schema_retains_its_identifiers(self):
        tree = ast.parse((ROOT / 'utopia/agents/funding_agents.py').read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'mapped_evaluation_format')
        method.decorator_list = []
        namespace = {}
        compile_nodes([method], namespace, 'actual_funding_schema')
        panels = [[{'applicant_id': 'alice'}, {'applicant_id': 'bob'}],
                  [{'applicant_id': 'carol'}, {'applicant_id': 'dan'}, {'applicant_id': 'eve'}]]
        formats = [namespace['mapped_evaluation_format'](apps) for apps in panels]
        originals = deepcopy(formats)
        def respond(body, index):
            schema = body['response_format']['json_schema']['schema']
            choices = schema['properties']['ranked_applications']['items']['anyOf']
            return json.dumps({'ranked_applications': [dict(
                application_id=choice['properties']['application_id']['enum'][0],
                applicant_id=choice['properties']['applicant_id']['enum'][0],
                rank=rank+1, reason='fixture') for rank, choice in enumerate(choices)]})
        client, wire = self.make_bedrock(WireRecorder(respond))
        result = client.generate_batch(['panel A','panel B'], response_formats=formats)
        for index, (ranking, _) in enumerate(result):
            jsonschema.validate(ranking, formats[index]['json_object']['schema'])
            self.assertEqual([r['applicant_id'] for r in ranking['ranked_applications']],
                             [app['applicant_id'] for app in panels[index]])
        self.assertEqual(formats, originals)
        for body in wire.payloads:
            self.assertTrue(body['response_format']['json_schema']['strict'])
            self.assertEqual(body['response_format']['json_schema']['schema']['additionalProperties'], False)

    def test_review_escape_pattern_preserves_inputs_and_legacy_cached_replies(self):
        import tempfile
        review_format = {'type': 'json_schema', 'json_object': {'schema': {
            'type': 'object', 'properties': {
                'overall_score': {'type': 'number'},
                'justification': {'type': 'string'},
            }, 'required': ['overall_score', 'justification'],
            'additionalProperties': False,
        }}}
        original = deepcopy(review_format)
        reply = {'overall_score': 2.5, 'justification': 'Clear argument. Full text considered.'}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'replies.jsonl'
            client, wire = self.make_bedrock(
                WireRecorder(lambda body, index: json.dumps(reply)), cache=cache)
            prompt = 'Full body ' * 7000 + 'FINAL PAGE'
            client.generate_batch([prompt], response_format=review_format,
                                  seed_ctx=('phase3_reviews', 2))
            payload = wire.payloads[0]
            schema = payload['response_format']['json_schema']['schema']
            self.assertEqual(payload['messages'][-1]['content'], IMPORTANT_NOTES + '\n\n' + prompt)
            self.assertEqual(payload['max_completion_tokens'], 32768)
            self.assertEqual(payload['reasoning_effort'], 'none')
            self.assertEqual(review_format, original)
            jsonschema.validate(reply, schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**reply, 'justification': '\x1b[1m' * 5}, schema)
            for codepoint in list(range(32)) + list(range(127, 160)):
                with self.subTest(control_character=codepoint):
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate({**reply, 'justification': chr(codepoint) * 5}, schema)
            jsonschema.validate({**reply, 'justification': 'First sentence. Second sentence. Café α.'}, schema)
            resumed, resumed_wire = self.make_bedrock(cache=cache)
            resumed.generate_batch([prompt], response_format=review_format,
                                   seed_ctx=('phase3_reviews', 2))
            self.assertFalse(resumed_wire.payloads)

            # An older complete reply may contain ESC: preserve it without a new call.
            row = json.loads(cache.read_text())
            row['payload']['response_format']['json_schema']['schema']['properties'][
                'justification'].pop('pattern')
            row['response']['choices'][0]['message']['content'] = json.dumps(
                {**reply, 'justification': '\x1b[1mEarlier complete review\x1b[0m'})
            cache.write_text(json.dumps(row) + '\n')
            legacy, legacy_wire = self.make_bedrock(cache=cache)
            result = legacy.generate_batch([prompt], response_format=review_format,
                                           seed_ctx=('phase3_reviews', 2))
            self.assertIn('Earlier complete review', result[0][0]['justification'])
            self.assertFalse(legacy_wire.payloads)

            # The prior ESC-only schema allowed DEL in a complete saved reply.
            row['payload']['response_format']['json_schema']['schema']['properties'][
                'justification']['pattern'] = r'^[^\u001b]*$'
            row['response']['choices'][0]['message']['content'] = json.dumps(
                {**reply, 'justification': 'Earlier complete review\x7f'})
            cache.write_text(json.dumps(row) + '\n')
            prior, prior_wire = self.make_bedrock(cache=cache)
            result = prior.generate_batch([prompt], response_format=review_format,
                                          seed_ctx=('phase3_reviews', 2))
            self.assertEqual(result[0][0]['justification'], 'Earlier complete review\x7f')
            self.assertFalse(prior_wire.payloads)

            # The prior ESC/DEL schema allowed other controls in complete replies.
            row['payload']['response_format']['json_schema']['schema']['properties'][
                'justification']['pattern'] = r'^[^\u001b\u007f]*$'
            row['response']['choices'][0]['message']['content'] = json.dumps(
                {**reply, 'justification': 'Earlier complete review\x1a'})
            cache.write_text(json.dumps(row) + '\n')
            prior, prior_wire = self.make_bedrock(cache=cache)
            result = prior.generate_batch([prompt], response_format=review_format,
                                          seed_ctx=('phase3_reviews', 2))
            self.assertEqual(result[0][0]['justification'], 'Earlier complete review\x1a')
            self.assertFalse(prior_wire.payloads)

            # Complete replies with earlier permitted newlines remain reusable.
            row['payload']['response_format']['json_schema']['schema']['properties'][
                'justification']['pattern'] = r'^[^\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]*$'
            row['response']['choices'][0]['message']['content'] = json.dumps(
                {**reply, 'justification': 'Earlier complete review.\nSecond paragraph.'})
            cache.write_text(json.dumps(row) + '\n')
            prior, prior_wire = self.make_bedrock(cache=cache)
            result = prior.generate_batch([prompt], response_format=review_format,
                                          seed_ctx=('phase3_reviews', 2))
            self.assertEqual(result[0][0]['justification'],
                             'Earlier complete review.\nSecond paragraph.')
            self.assertFalse(prior_wire.payloads)

    def test_keyword_escape_pattern_preserves_year_inputs_and_legacy_cache(self):
        import tempfile
        keyword_format = {'type': 'json_schema', 'json_object': {'schema': {
            'type': 'object', 'properties': {
                'keywords': {'type': 'array', 'items': {'type': 'string'}},
            }, 'required': ['keywords'], 'additionalProperties': False,
        }}}
        original = deepcopy(keyword_format)
        reply = {'keywords': ['graph learning', 'spectral methods']}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'replies.jsonl'
            client, wire = self.make_bedrock(
                WireRecorder(lambda body, index: json.dumps(reply)), cache=cache)
            prompt = 'Complete paper abstract from the first year.'
            client.generate_batch([prompt], response_format=keyword_format,
                                  seed_ctx=('keywords', 0), temperature=0.3)
            payload = wire.payloads[0]
            schema = payload['response_format']['json_schema']['schema']
            self.assertEqual(payload['messages'][-1]['content'], IMPORTANT_NOTES + '\n\n' + prompt)
            self.assertEqual(payload['max_completion_tokens'], 32768)
            self.assertEqual(payload['reasoning_effort'], 'none')
            self.assertEqual(payload['temperature'], 0.3)
            self.assertEqual(keyword_format, original)
            jsonschema.validate(reply, schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({'keywords': ['\x1b[1m' * 5]}, schema)

            # Preserve an older complete reply, including its unconstrained schema.
            row = json.loads(cache.read_text())
            row['payload']['response_format']['json_schema']['schema']['properties'][
                'keywords']['items'].pop('pattern')
            row['response']['choices'][0]['message']['content'] = json.dumps(
                {'keywords': ['\x1b[1mEarlier keyword\x1b[0m']})
            cache.write_text(json.dumps(row) + '\n')
            resumed, resumed_wire = self.make_bedrock(
                WireRecorder(lambda body, index: json.dumps(reply)), cache=cache)
            result = resumed.generate_batch([prompt], response_format=keyword_format,
                                            seed_ctx=('keywords', 0), temperature=0.3)
            self.assertIn('Earlier keyword', result[0][0]['keywords'][0])
            self.assertFalse(resumed_wire.payloads)

            # The next year reuses the positional ID, but has a different paper input.
            next_prompt = 'Complete paper abstract from the next year.'
            resumed.generate_batch([next_prompt], response_format=keyword_format,
                                   seed_ctx=('keywords', 0), temperature=0.3)
            self.assertEqual(len(resumed_wire.payloads), 1)
            next_payload = resumed_wire.payloads[0]
            self.assertEqual(next_payload['messages'][-1]['content'],
                             IMPORTANT_NOTES + '\n\n' + next_prompt)
            jsonschema.validate(reply, next_payload['response_format']['json_schema']['schema'])
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({'keywords': ['\x1b[1m']},
                                    next_payload['response_format']['json_schema']['schema'])

    def test_success_after_retry_resumes_without_rebilling_failed_attempt(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            client, wire = self.make_bedrock(WireRecorder(
                lambda body,index: 'invalid JSON' if index==0 else '{"ok":true}'),
                cache=Path(directory)/'replies.jsonl')
            result = client.generate_batch(['retry'], item_ids=['r'], seed_ctx=('retry',))
            self.assertTrue(result[0][0]['ok'])
            resumed = client.generate_batch(['retry'], item_ids=['r'], seed_ctx=('retry',))
            self.assertEqual(resumed[0][0],result[0][0])
            self.assertEqual(len(wire.payloads),2)

    def test_rate_limit_drains_without_hidden_sdk_retries(self):
        client, wire = self.make_bedrock()
        import openai
        barrier = threading.Barrier(2)
        def rate_limited(**kwargs):
            barrier.wait(timeout=5)
            raise openai.RateLimitError(
                'rate limited', response=httpx.Response(429,request=httpx.Request('POST','http://fixture')),
                body=None)
        with patch.object(client.client.chat.completions, 'create', side_effect=rate_limited) as create:
            with self.assertRaises(CLIENT_NS['ModelRequestError']):
                client.generate_batch(['a','b'])
        self.assertEqual(create.call_count,2)
        self.assertEqual(client.client.max_retries,0)

    def test_tokenizer_regex_limit_preserves_full_text_and_id_resume(self):
        import tempfile
        import tiktoken
        tokenizer = tiktoken.get_encoding('o200k_base')
        prompt = 'Complete text: ' + ' ' * 3108206 + 'TAIL café'
        with self.assertRaisesRegex(ValueError, 'Regex error while tokenizing:'):
            tokenizer.encode(json.dumps(prompt, ensure_ascii=False), disallowed_special=())
        with tempfile.TemporaryDirectory() as directory:
            client, wire = self.make_bedrock(
                concurrency=1, cache=Path(directory) / 'replies.jsonl')
            client.input_tokenizer = tokenizer
            client.token_rates = [300000, 0]
            client.token_balances = [0, 0]
            client.quota_time = 0
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
            with patch.object(CLIENT_NS['time'], 'monotonic', lambda: clock[0]), \
                    patch.object(CLIENT_NS['time'], 'sleep', sleep):
                result = client.generate_batch([prompt], seed_ctx=('fulltext', 8))
                waited = clock[0]
                replay = client.generate_batch([prompt], seed_ctx=('fulltext', 8))
                self.assertEqual(clock[0], waited)
            self.assertGreater(waited, 0)
            self.assertLess(waited, 10)
            self.assertEqual(result[0][0], replay[0][0])
            self.assertEqual(len(wire.payloads), 1)
            self.assertEqual(wire.payloads[0]['messages'][-1]['content'],
                             IMPORTANT_NOTES + '\n\n' + prompt)
            self.assertEqual(wire.payloads[0]['max_completion_tokens'], 32768)
            self.assertEqual(wire.payloads[0]['reasoning_effort'], 'none')

    def test_unrelated_tokenizer_errors_propagate_before_dispatch(self):
        from types import SimpleNamespace
        client, wire = self.make_bedrock()
        client.token_rates = [300000, 0]
        def fail(*args, **kwargs):
            raise ValueError('unrelated tokenizer failure')
        client.input_tokenizer = SimpleNamespace(encode=fail)
        with self.assertRaisesRegex(ValueError, 'unrelated tokenizer failure'):
            client.wait_for_quota({'messages': []}, 'reviews')
        self.assertFalse(wire.payloads)

    def test_input_quota_paces_new_calls_and_cache_replay_bypasses_it(self):
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            client, wire = self.make_bedrock(cache=Path(directory)/'replies.jsonl')
            client.generate_batch(['saved'], seed_ctx=('quota',))
            client.token_rates = [120, 0]
            client.token_balances = [120, 0]
            client.input_tokenizer = SimpleNamespace(encode=lambda *a, **kw: [0] * 56)
            client.quota_time = 0
            clock = [0.0]
            delays = []
            def sleep(seconds):
                delays.append(seconds)
                clock[0] += seconds
            with patch.object(CLIENT_NS['time'], 'monotonic', lambda: clock[0]), \
                    patch.object(CLIENT_NS['time'], 'sleep', sleep):
                payload = {'messages': [{'role': 'user', 'content': 'full body'}]}
                client.wait_for_quota(payload, 'reviews')
                client.wait_for_quota(payload, 'reviews')
                self.assertEqual(sum(delays), 60.0)
                client.generate_batch(['saved'], seed_ctx=('quota',))
                self.assertEqual(sum(delays), 60.0)
            self.assertEqual(len(wire.payloads), 1)

    def test_output_quota_uses_observed_phase_lengths_and_stops_waiters(self):
        client, _ = self.make_bedrock()
        client.token_rates = [0, 120]
        client.token_balances = [0, 120]
        client.output_estimates['ranking'] = 120
        client.quota_time = 0
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        with patch.object(CLIENT_NS['time'], 'monotonic', lambda: clock[0]), \
                patch.object(CLIENT_NS['time'], 'sleep', sleep):
            payload = {'messages': []}
            self.assertEqual(client.wait_for_quota(payload, 'ranking')[1], 120)
            client.wait_for_quota(payload, 'ranking')
            self.assertEqual(clock[0], 60)
            client.batch_cancelled.set()
            with self.assertRaises(CLIENT_NS['ModelRequestError']):
                client.wait_for_quota(payload, 'ranking')

    def test_quota_failure_drains_and_preserves_other_issued_reply(self):
        import tempfile
        import openai
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'replies.jsonl'
            client, wire = self.make_bedrock(cache=cache)
            create = client.client.chat.completions.create
            barrier = threading.Barrier(2)
            def respond(**payload):
                barrier.wait(timeout=5)
                if payload['messages'][-1]['content'].endswith('failure'):
                    raise openai.RateLimitError(
                        'output quota', response=httpx.Response(
                            429, request=httpx.Request('POST', 'http://fixture')), body=None)
                time.sleep(0.02)
                return create(**payload)
            with patch.object(client.client.chat.completions, 'create', side_effect=respond):
                with self.assertRaises(CLIENT_NS['ModelRequestError']):
                    client.generate_batch(['failure', 'success'], item_ids=['drop', 'keep'],
                                          seed_ctx=('drain',))
            rows = [json.loads(line) for line in cache.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]['complete'])
            resumed, resumed_wire = self.make_bedrock(cache=cache)
            resumed.generate_batch(['success'], item_ids=['keep'], seed_ctx=('drain',))
            self.assertEqual(len(wire.payloads), 1)
            self.assertFalse(resumed_wire.payloads)

    def test_spending_cap_includes_saved_usage_but_allows_free_id_replay(self):
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'replies.jsonl'
            client, _ = self.make_bedrock(cache=cache)
            client.generate_batch(['saved'], seed_ctx=('budget',))
            self.assertGreater(client.cost_usd, 0)
            resumed, wire = self.make_bedrock(cache=cache)
            self.assertEqual(resumed.cost_usd, client.cost_usd)
            resumed.api_budget_usd = client.cost_usd
            resumed.input_tokenizer = SimpleNamespace(encode=lambda *a, **kw: [0] * 56)
            resumed.generate_batch(['saved'], seed_ctx=('budget',))
            with self.assertRaisesRegex(CLIENT_NS['ModelRequestError'], 'API budget'):
                resumed.generate_batch(['new'], seed_ctx=('budget',))
            self.assertFalse(wire.payloads)


if __name__ == "__main__":
    unittest.main()
