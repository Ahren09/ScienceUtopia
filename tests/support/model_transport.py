"""CPU transport regressions for the shared vLLM 0.12 schema adapter.

Run from the scratch repository root:
  /opt/anaconda3/bin/python3 -B -m unittest discover -s src \
      -p test_vllm_schema_transport.py -v

The real OpenAI SDK serializes requests into an HTTPX MockTransport: no network
or inference is performed. Because importing the whole models module requires
Torch and unrelated providers, its exact BaseLLM/VLLMServerModel/parser source
definitions are compiled unchanged. Original Pydantic schema definitions and
response-format expressions are likewise executed, with fixture conference
enums for dynamic schemas. This tests outgoing wire payloads, not a running
vLLM grammar backend or Qwen's ability to finish within its token budget.
"""

from __future__ import annotations

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

ROOT = project_root(__file__)


from utopia.constants import IMPORTANT_NOTES

from utopia.config import SIMULATION_CONFIG


def load_shared_helper():
    path = ROOT / "utopia/models/structured_outputs.py"
    spec = importlib.util.spec_from_file_location("schema_probe_shared_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_vllm_extra_body


BUILD_EXTRA_BODY = load_shared_helper()


def compile_nodes(nodes, namespace, filename):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *deepcopy(nodes)], type_ignores=[])
    )
    exec(compile(module, str(filename), "exec"), namespace)


def client_definitions():
    path = ROOT / "utopia/models/models.py"
    tree = ast.parse(path.read_text())
    names = {"BaseLLM", "VLLMServerModel", "parse_json_response"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == names
    ns = {
        "__name__": __name__,
        "SIMULATION_CONFIG": SIMULATION_CONFIG,
        "IMPORTANT_NOTES": IMPORTANT_NOTES,
        "OpenAI": OpenAI,
        "build_vllm_extra_body": BUILD_EXTRA_BODY,
        "os": os,
        "json": json,
        "re": re,
        "time": time,
        "logger": logging.getLogger("schema-transport-test"),
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "as_completed": as_completed,
        "tqdm": lambda iterable, **kwargs: iterable,
    }
    compile_nodes(nodes, ns, path)
    return ns


CLIENT_NS = client_definitions()

Client = CLIENT_NS["VLLMServerModel"]


def actual_seed_module():
    path = ROOT / "utopia/utils/seeding.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "derive_seed"
    )
    module = ModuleType("utopia.utils.seeding")
    compile_nodes([node], vars(module), path)
    return module


SEED_MODULE = actual_seed_module()


def actual_response_formats():
    """Discover every current agent json_schema emitter; preserve its schema."""
    formats = {}
    for path in sorted((ROOT / "utopia/agents").glob("*.py")):
        tree = ast.parse(path.read_text())
        models = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)
            and any(isinstance(b, ast.Name) and b.id == "BaseModel" for b in n.bases)
        ]
        if not models:
            continue
        ns = {
            "__name__": __name__,
            "BaseModel": BaseModel,
            "List": List,
            "Dict": Dict,
            "Optional": Optional,
            "Tuple": Tuple,
            "Union": Union,
            "Conference": Enum(
                "Conference", {"NeurIPS": "NeurIPS", "ICLR": "ICLR"}, type=str
            ),
            "ConferenceEnum": Enum(
                "ConferenceEnum", {"NeurIPS": "NeurIPS", "ICLR": "ICLR"}, type=str
            ),
        }
        compile_nodes(models, ns, path)
        for model in models:
            ns[model.name].model_rebuild(_types_namespace=ns)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            literal = node.value
            fields = {
                k.value: v
                for k, v in zip(literal.keys, literal.values)
                if isinstance(k, ast.Constant)
            }
            if not (
                isinstance(fields.get("type"), ast.Constant)
                and fields["type"].value == "json_schema"
                and "json_object" in fields
            ):
                continue
            expression = ast.fix_missing_locations(ast.Expression(deepcopy(literal)))
            value = eval(compile(expression, str(path), "eval"), ns)
            name = value["json_object"]["name"]
            # Single- and multi-paper submission deliberately share an API
            # name, so source location identifies each distinct emitter.
            formats[f"{path.name}:{node.lineno}:{name}"] = value
    return formats


FORMATS = actual_response_formats()


class WireRecorder:
    """Real SDK serialization, in-memory HTTP response; socket access is unnecessary."""

    def __init__(self, respond=None):
        self.payloads = []
        self.respond = respond or (lambda body, index: '{"sentinel": true}')
        self.lock = threading.Lock()

    def handle(self, request):
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "Qwen/Qwen3-32B",
                            "object": "model",
                            "created": 1,
                            "owned_by": "fixture",
                        }
                    ],
                },
            )
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            raise AssertionError(
                f"Unexpected transport request: {request.method} {request.url}"
            )
        body = json.loads(request.content)
        with self.lock:
            index = len(self.payloads)
            self.payloads.append(deepcopy(body))
        content = self.respond(body, index)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-fixture-{index}",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )


class TransportFixture(unittest.TestCase):
    def setUp(self):
        replacement = patch.dict(
            sys.modules, {"utopia.utils.seeding": SEED_MODULE}
        )
        replacement.start()
        self.addCleanup(replacement.stop)

    def make_client(self, recorder=None, *, thinking=True, seed=42):
        recorder = recorder or WireRecorder()
        http = httpx.Client(transport=httpx.MockTransport(recorder.handle))
        self.addCleanup(http.close)

        def factory(**kwargs):
            # Exercise the original client's initialization and /models request
            # while fixing the SDK transport to an in-memory handler.
            return OpenAI(http_client=http, **kwargs)

        with patch.dict(CLIENT_NS, {"OpenAI": factory}):
            with patch.dict(os.environ, {"VLLM_API_KEY": "fixture-key"}):
                client = Client(
                    "Qwen/Qwen3-32B",
                    "http://schema-fixture.invalid/v1",
                    max_concurrent_requests=2,
                    run_seed=seed,
                )
        client.enable_thinking = thinking
        return client, recorder
