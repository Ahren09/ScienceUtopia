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
        self.assertIs(client._build_extra_body(fmt)["structured_outputs"]["json"],
                      fmt["json_object"]["schema"])
        for absent in (None, {}, {"type": "json_object"}, {"type": "json_schema", "json_object": {}}):
            self.assertEqual(client._build_extra_body(absent), {})
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

if __name__ == "__main__":
    unittest.main()
