"""Pure stdlib tests; no imports of Torch, Pydantic, NumPy, or the simulator.

Run from the scratch code root (no GPU/HTTP calls):
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s src \
        -p test_james_funding_validation.py -v

The parity tests compile exact stock method ASTs, not rewritten equivalents.
Temporary audit files live under the scratch root and are removed by unittest.
"""

from __future__ import annotations

from utopia.utils.paths import project_root

import ast

from copy import deepcopy

import json

import logging

import os

from pathlib import Path

import random

import stat

import sys

import tempfile

from types import SimpleNamespace

import unittest

from unittest.mock import patch

from utopia.funding.validation import (
    FundingRankingValidationError,
    inspect_funding_result,
    install_funding_validation,
)

import utopia.funding.validation as gate

ROOT = project_root(__file__)

STOCK_PATH = ROOT / "utopia/agents/funding_agents.py"


def stock_class():
    """Fresh class with unmodified stock methods and no heavy imports."""
    names = {
        "validate_funding_result",
        "normalize_ranked_applications",
        "allocate_slots_largest_remainder",
        "process_funding_evaluation_results",
    }
    tree = ast.parse(STOCK_PATH.read_text(), filename=str(STOCK_PATH))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FundingAgency"
    )
    methods = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in methods} != names:
        raise AssertionError("Stock funding method missing")
    cls = ast.ClassDef(
        name="FundingAgency", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                cls,
            ],
            type_ignores=[],
        )
    )
    ns = {"logger": logging.getLogger("stock-funding-validity-test")}
    exec(compile(module, str(STOCK_PATH), "exec"), ns)
    return ns["FundingAgency"]


def apps_for(n, repeat=False):
    return [
        {
            "applicant_id": "same" if repeat else f"researcher_{i}",
            # Deliberately unrelated: expected application ID is the index.
            "application_id": 9999 - i,
        }
        for i in range(n)
    ]


def ranking(apps, order=None):
    order = list(reversed(range(len(apps)))) if order is None else order
    return {
        "ranked_applications": [
            {
                "application_id": i,
                "applicant_id": apps[i]["applicant_id"],
                "rank": rank,
                "reason": "fixture",
            }
            for rank, i in enumerate(order, 1)
        ]
    }


def result_pairs(*results):
    return [(result, [{"private": object()}]) for result in results]


def metadata(*panels):
    return [
        {"program_id": "P", "panel_index": i, "apps": apps}
        for i, apps in enumerate(panels)
    ]


PROGRAMS = {"P": SimpleNamespace(funding_rate=0.4)}


class GateCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            prefix=".funding-validation-test-", dir=ROOT
        )
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self._serial = 0

    def install(self, cls=None, path=None, *, compact=False):
        cls = stock_class() if cls is None else cls
        self._serial += 1
        path = self.directory / f"audit-{self._serial}.jsonl" if path is None else path
        handle = install_funding_validation(cls, path, compact=compact)
        self.addCleanup(handle.restore)
        return cls, handle, path

    def read_records(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]
