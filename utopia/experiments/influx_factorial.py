"""Influx factorial experiment and reusable scientific operations."""
#!/usr/bin/env python

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import write_json_atomic

from utopia.runtime.historical import INFLUX_POPULATION_SEED_NAMESPACE

from utopia.runtime.commands import module_command

from utopia.utils.data_utils import file_sha256 as sha256

from utopia.utils.paths import project_root
from utopia.runtime.provenance import digest as object_hash
import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import urllib.parse
import urllib.request

REPO_ROOT = project_root(__file__)
SEED = 42
PROTOCOL = 'influx1202_v6'
POPULATION_RNG_NAMESPACE = INFLUX_POPULATION_SEED_NAMESPACE
STRUCTURED_OUTPUT_TRANSPORT = 'structured_outputs.json'
UNIVERSITIES = 240
RESEARCHERS_PER_UNIVERSITY = 5
INDUSTRY_RESEARCHERS = 2
FOUNDER_RESEARCHERS = UNIVERSITIES * RESEARCHERS_PER_UNIVERSITY + INDUSTRY_RESEARCHERS
FUNDING_AGENCIES = 2
MODEL = 'Qwen/Qwen3-32B'
REVISION = '9216db5781bf21249d130ec9da846c4624c16137'
NUM_YEARS = 10
FUNDING_PANEL_MAX_APPS = 25
FUNDING_VALIDATION_POLICY = 'complete_permutation_no_fallback_v1'
FUNDING_OUTPUT_REPRESENTATION = 'ordered_application_ids_v1'
FUNDING_AUDIT_FILE = 'funding_validation_audit.jsonl'
FUNDING_SUMMARY_FILE = 'funding_validation_summary.json'
FUNDING_SELECTION_PROTOCOL = 'sequential_remaining_ids_v1'
SEQUENTIAL_AUDIT_FILE = 'funding_sequential.jsonl'
SEQUENTIAL_SUMMARY_FILE = 'funding_sequential.summary.json'
SEQUENTIAL_PROTOCOL_PATH = 'docs/sequential_funding_protocol.md'
CELLS = {
    'A': {'tag': 'noinflux_noresub', 'flags': ['--disable_resubmission']},
    'B': {'tag': 'influx_noresub',
          'flags': ['--annual_influx_rate', '0.1', '--disable_resubmission']},
    'C': {'tag': 'noinflux_resub', 'flags': []},
    'D': {'tag': 'influx_resub', 'flags': ['--annual_influx_rate', '0.1']},
}


def funding_spec():
    """Common scientific condition; panelization is not a performance-only change."""
    return {
        'baseline': 'panelized', 'panel_max_apps': FUNDING_PANEL_MAX_APPS,
        'budget_mode': 'track', 'program_rates': {'NSF': 0.23, 'DARPA': 0.10},
        'quota_rule': 'max(1, floor(panel_size * program_rate))',
        'log_funding_applications': True,
        'validation_policy': FUNDING_VALIDATION_POLICY,
        'output_representation': FUNDING_OUTPUT_REPRESENTATION,
        'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
        'global_ranking_equivalent': False,
    }


def sequential_spec():
    return {
        'funding_selection_protocol': FUNDING_SELECTION_PROTOCOL,
        'audit_file': SEQUENTIAL_AUDIT_FILE, 'summary_file': SEQUENTIAL_SUMMARY_FILE,
        'protocol_document': SEQUENTIAL_PROTOCOL_PATH,
        'max_sdk_calls_per_step': 3, 'accepted_prefix_reset': False,
        'all_choices_from_remaining_ids': True,
        'seed_context': ['phase5_funding_eval', 'year', FUNDING_SELECTION_PROTOCOL, 'production',
                         'program_id', 'panel_index', 'step'],
        'step_index_base': 0, 'sdk_attempts': [0, 1, 2],
    }


def request_audit_spec():
    return {'required': True, 'model': MODEL, 'revision': REVISION,
            'max_model_len': 32768, 'budget_rule': 'exact_input_tokens + actual_max_tokens <= 32768',
            'n_clients': 1, 'guard_failures': 0}


def population_spec():
    return {
        'founder_researchers': FOUNDER_RESEARCHERS,
        'university_researchers': UNIVERSITIES * RESEARCHERS_PER_UNIVERSITY,
        'universities': UNIVERSITIES, 'researchers_per_university': RESEARCHERS_PER_UNIVERSITY,
        'industry_researchers': INDUSTRY_RESEARCHERS, 'company': 'Google',
        'funding_agencies': FUNDING_AGENCIES, 'total_initial_agents': FOUNDER_RESEARCHERS + 2,
        'university_funding': {'rich_institutions': UNIVERSITIES // 3, 'rich': 160, 'normal': 80},
        'industry_funding': 320, 'strategy': 'balanced',
        'growth_denominator': '1202 founder researchers; excludes funding agencies',
        'growth_schedule': '120 university entrants/year; 1 Google entrant in years 5 and 10',
        'cumulative_entrants': 'floor(1202 * year / 10)',
        'population_comparability': 'New population AND composition; not paired with historical n302',
    }


def build_founders(simulator, seed=SEED):
    """Pure blueprint, using the existing seeded university blueprint generator."""
    if seed != SEED:
        raise ValueError('only seed42 is allowed')
    rows = simulator.build_population_blueprint(
        UNIVERSITIES, RESEARCHERS_PER_UNIVERSITY, seed,
        strategy_mix=['balanced'] * RESEARCHERS_PER_UNIVERSITY)
    founders = []
    for index, row in enumerate(rows):
        founders.append({**row, 'kind': 'university',
                         'funding_level': (simulator.RICH_UNIVERSITY_FUNDING
                                           if index // RESEARCHERS_PER_UNIVERSITY < UNIVERSITIES // 3
                                           else simulator.NORMAL_UNIVERSITY_FUNDING)})
    if simulator.RICH_COMPANIES[0] != 'Google':
        raise ValueError('original rich company roster changed')
    for index in range(INDUSTRY_RESEARCHERS):
        rng = random.Random(simulator.derive_seed(seed, POPULATION_RNG_NAMESPACE, 'industry_founder', index))
        expertise = rng.sample(range(len(simulator.AVAILABLE_DIRECTIONS)), 3)
        founders.append({
            'institution': 'Google', 'researcher_name': f'Google_researcher_{index}',
            'kind': 'industry', 'strategy': 'balanced',
            'funding_level': simulator.RICH_COMPANY_FUNDING,
            'expertise_indices': expertise,
            'expertise_topics': [simulator.AVAILABLE_DIRECTIONS[i].topic for i in expertise],
        })
    if (len(founders) != FOUNDER_RESEARCHERS
            or {r['funding_level'] for r in founders if r['kind'] == 'university'} != {80, 160}
            or any(r['funding_level'] != 320 for r in founders if r['kind'] == 'industry')):
        raise ValueError('founder population or original funding constants changed')
    return founders


def build_fixed_cohort(simulator, year, seed=SEED):
    """Staggered, tier-balanced growth anchored to researchers, with no global RNG use."""
    if seed != SEED or not 1 <= year <= NUM_YEARS:
        raise ValueError('cohort requires seed42 and years 1..10')
    cohort = []
    for index in range(UNIVERSITIES):
        if index % 2 != (year - 1) % 2:
            continue
        cohort.append({
            'institution': f'institution_{index:04d}', 'kind': 'university',
            'funding_level': (simulator.RICH_UNIVERSITY_FUNDING if index < UNIVERSITIES // 3
                              else simulator.NORMAL_UNIVERSITY_FUNDING),
            'researcher_name': f'institution_{index:04d}_researcher_{5 + (year - 1) // 2}',
        })
    if year % 5 == 0:
        cohort.append({
            'institution': 'Google', 'kind': 'industry',
            'funding_level': simulator.RICH_COMPANY_FUNDING,
            'researcher_name': f'Google_researcher_{1 + year // 5}',
        })
    for row in cohort:
        rng = random.Random(simulator.derive_seed(seed, POPULATION_RNG_NAMESPACE, 'influx', year, row['institution']))
        row['expertise_indices'] = rng.sample(range(len(simulator.AVAILABLE_DIRECTIONS)), 3)
    return cohort


def initialize_population(simulation, simulator, founders):
    """Stock researcher constructors, stock funding agencies and stock N0 anchors."""
    if (getattr(simulation.args, 'seed', None) != SEED or simulation.debug
            or getattr(simulation.args, 'population_mode', 'default') != 'default'):
        raise ValueError(f'{PROTOCOL} requires seed42, non-debug default treatment mechanics')
    by_institution = {}
    for row in founders:
        kwargs = {
            'researcher_name': row['researcher_name'], 'funding_level': row['funding_level'],
            'expertise': [simulator.AVAILABLE_DIRECTIONS[i] for i in row['expertise_indices']],
            'llm': simulation.llm, 'exploration_strategy': 'balanced',
        }
        if row['kind'] == 'university':
            agent = simulator.UniversityResearcher(
                **kwargs, university_name=row['institution'], generate_research_proposal=True)
        else:
            agent = simulator.IndustryResearcher(
                **kwargs, company_name=row['institution'], funding_mode=simulation.industry_funding_mode)
        simulation.ecosystem.add_agent(agent)
        by_institution.setdefault(row['institution'], set()).add(agent.id)
    for names in by_institution.values():
        for name in names:
            simulation.ecosystem.get_agent_by_id(name).add_conflict_of_interest(names - {name})
    simulation._record_initial_university_count()
    simulation._add_funding_agencies()
    if (simulation.initial_university_count != 1200 or simulation.initial_industry_count != 2
            or len(simulation.ecosystem.agent_population) != 1204):
        raise ValueError('initialized population differs from frozen 1202-researcher protocol')


def main(argv=None):
    from utopia.experiments.common import main_for
    return main_for("influx_factorial", argv)


if __name__ == "__main__":
    main()


def expected_researcher_ids(cell, year):
    ids = {f'institution_{index:04d}_researcher_{r}'
           for index in range(UNIVERSITIES) for r in range(RESEARCHERS_PER_UNIVERSITY)}
    ids.update(f'Google_researcher_{r}' for r in range(INDUSTRY_RESEARCHERS))
    if cell in ('B', 'D'):
        for entry_year in range(1, year + 1):
            ids.update(f'institution_{index:04d}_researcher_{5 + (entry_year - 1) // 2}'
                       for index in range(UNIVERSITIES) if index % 2 == (entry_year - 1) % 2)
            if entry_year % 5 == 0:
                ids.add(f'Google_researcher_{1 + entry_year // 5}')
    return ids


@contextlib.contextmanager
def funding_validation_scope(simulator, job, root=REPO_ROOT):
    """Audit native compact sequential funding and restore hooks on every exit."""
    from utopia.funding.validation import install_funding_validation, FundingRankingValidationError
    from utopia.funding.feedback import sequential_funding_scope, funding_phase_coverage
    out = Path(job['output_dir']).resolve()
    handle = install_funding_validation(simulator.FundingAgency, out / FUNDING_AUDIT_FILE, compact=True)
    succeeded = False
    try:
        with sequential_funding_scope(simulator.FundingAgency, simulator.VLLMServerModel, out) as sequential, \
                funding_phase_coverage(simulator.Simulation, simulator.FundingAgency, sequential):
            yield handle
            if not handle.summary()['completion_allowed']:
                raise FundingRankingValidationError(handle.summary())
        succeeded = True
    finally:
        handle.restore()
        write_json_atomic(out / FUNDING_SUMMARY_FILE, {
            **handle.summary(), 'status': 'complete' if succeeded else 'failed',
            'cell': job['cell'], 'seed': SEED,
        }, indent=2)
