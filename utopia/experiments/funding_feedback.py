"""Funding feedback experiment and reusable scientific operations."""
from __future__ import annotations

from utopia.runtime.commands import module_command
from utopia.utils.data_utils import file_sha256 as file_hash
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import tempfile
import traceback
from utopia.analysis.funding_feedback import analyze
from utopia.funding.feedback import CELLS, FOUNDER_COUNT, FUNDING_BASELINE, FUNDING_OUTPUT_REPRESENTATION, FUNDING_PANEL_MAX_APPS, FUNDING_SELECTION_PROTOCOL, Mechanisms, PROTOCOL, SEEDS, SEQUENTIAL_AUDIT_FILE, SEQUENTIAL_SUMMARY_FILE, YEARS, effective_args, experiment_id, fallback_audit, funding_ledger_evidence, funding_phase_coverage, sequential_funding_scope, sequential_source_binding, sequential_summary_gates, simulation_class, source_fingerprint, validate_sequential_evidence
from utopia.runtime.provenance import MODEL, ROOT, digest, mark_failed_run_manifest, server_provenance, validate_request_audit_summary, write_json


def main(argv=None):
    from utopia.experiments.common import main_for
    return main_for("funding_feedback", argv)


if __name__ == "__main__":
    main()
