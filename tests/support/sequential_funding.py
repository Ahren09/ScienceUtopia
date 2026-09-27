"""Stdlib-only CPU tests. All SDK responses are scripted, never network/GPU calls.

    PYTHONPATH=src python -B -m unittest test_james_funding_sequential -v

Original funding prompt/panel/processor, VLLM transport methods and request audit
are executed from their actual files. Heavy optional imports alone are bypassed.
"""

from __future__ import annotations

from utopia.utils.paths import project_root

import ast

from concurrent.futures import ThreadPoolExecutor, as_completed

from copy import deepcopy

import importlib.util

import json

import logging

from pathlib import Path

import random

import sys

import tempfile

from threading import Event, Lock

import time

from types import ModuleType, SimpleNamespace

import unittest

from unittest.mock import patch

import utopia.funding.sequential as seq

import utopia.funding.validation as gate

ROOT = project_root(__file__)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_module = load_file(
    "sequential_test_request_audit", ROOT / "utopia/models/request_audit.py"
)

body_module = load_file(
    "sequential_test_structured", ROOT / "utopia/models/structured_outputs.py"
)


def definitions(path, class_name, methods, namespace):
    tree = ast.parse(path.read_text())
    original = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name in methods
        ],
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[class_name]


def native_seed():
    path = ROOT / "utopia/utils/seeding.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "derive_seed"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["derive_seed"]


class Direction:
    def __init__(self, topic):
        self.topic = topic


class TinyTokenizer:
    chat_template = "SCRIPTED CPU TOKENIZER, NOT QWEN"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def response(
    content,
    *,
    finish="stop",
    reasoning="scripted private reasoning",
    completion_tokens=7,
):
    return SimpleNamespace(
        id="scripted-response",
        object="chat.completion",
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=finish,
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                    reasoning="alternate field",
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=completion_tokens),
    )


class SDK:
    def __init__(self, script=None):
        self.calls, self.script, self.lock = [], script, Lock()

    def create(self, **payload):
        with self.lock:
            index = len(self.calls)
            self.calls.append(deepcopy(payload))
        if self.script is None:
            remaining = (
                payload["extra_body"]
                .get("structured_outputs", {})
                .get("json", {})
                .get("properties", {})
                .get("next_application_id", {})
                .get("enum", [0])
            )
            result = response(json.dumps({"next_application_id": remaining[-1]}))
        else:
            result = self.script(payload, index)
        if isinstance(result, BaseException):
            raise result
        if result.usage.prompt_tokens == 0:
            result.usage.prompt_tokens = len(
                TinyTokenizer().apply_chat_template(payload["messages"])
            )
        return result


def funding_class():
    ns = {
        "random": random,
        "logger": logging.getLogger("sequential-funding-test"),
        "ResearchDirection": Direction,
        "FundingEvaluationResponse": SimpleNamespace(model_json_schema=lambda: {}),
    }
    return definitions(
        ROOT / "utopia/agents/funding_agents.py",
        "FundingAgency",
        {
            "get_funding_evaluation_prompts",
            "_build_program_prompt",
            "validate_funding_result",
            "process_funding_evaluation_results",
            "normalize_ranked_applications",
            "allocate_slots_largest_remainder",
        },
        ns,
    )


def model_class():
    ns = {
        "build_vllm_extra_body": body_module.build_vllm_extra_body,
        "SIMULATION_CONFIG": {"llm": {"max_tokens": 2048}},
        "logger": logging.getLogger("sequential-model-test"),
        "IMPORTANT_NOTES": seq.IMPORTANT_NOTES,
        "parse_json_response": json.loads,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "as_completed": as_completed,
        "tqdm": lambda values, **kwargs: values,
        "time": time,
    }
    return definitions(
        ROOT / "utopia/models/models.py",
        "VLLMServerModel",
        {"_request_seed", "_build_extra_body", "_create_completion", "generate_batch"},
        ns,
    )
