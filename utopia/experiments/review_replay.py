"""Reviewer-monoculture review-replay pilot driver.

Replays stored paper submissions through controlled reviewer regimes:
  A - one standardized reviewer policy (monoculture)
  B - heterogeneous reviewer-priority profiles
  C - original stored reviews (ecological reference; replayed only on the smoke set)

Preregistration: configs/review_replay.json (frozen texts,
endpoints, sampling rules). This driver is the single new entry point for the pilot;
it reuses the frozen simulator unmodified: SimulationAgent.get_review_prompt,
MultiAgentEcosystem.from_dict, VLLMServerModel.generate_batch, derive_seed,
write_run_manifest. It never calls project_setup() (check_cwd asserts cwd name).

Subcommands:
  extract   build eligibility table, sample disjoint smoke/pilot/stability sets,
            emit task files with final prompts + IPW weights (data/review_replay/)
  generate  run one stage's reviews against the local vLLM server (resume-safe)
  analyze   delegate to utopia.analysis.review_regime_analysis
  status    report chunk completion for a stage

All writes are restricted to data/review_replay/ and outputs/{logs,docs,visual}/
review_replay_* (assert_write_allowed). E3.5/E4 artifacts are read-only inputs.
"""

from utopia.utils.data_utils import write_json_atomic, write_parquet_atomic

from utopia.utils.data_utils import file_sha256 as sha256_file

from utopia.utils.data_utils import read_json

from utopia.utils.paths import project_root

from pathlib import Path
import argparse
import re
import glob
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = str(project_root(__file__))

from utopia.utils.seeding import derive_seed, set_seed
from utopia.runtime.provenance import write_run_manifest

logger = logging.getLogger('review_replay')

GLOBAL_SEED = 42
CORPUS_VERSION = 'v1'
START_SIM_YEAR, END_SIM_YEAR = 1, 10
MIN_CELL_ELIGIBLE = 8
CHUNK_SIZE = 512
PREREG_V1 = os.path.join(REPO_ROOT, 'configs/review_replay.json')
CORPUS_DIR = 'data/review_replay'
MODEL_NAME = 'Qwen/Qwen3-32B'
MODEL_REVISION = '9216db5781bf21249d130ec9da846c4624c16137'

SOURCE_IDS = [f'explore_calibration_qwen3_32b_neutral_i100_n500_y10_seed{s}'
              for s in (501, 502, 503, 504, 505)]
SOURCE_SEEDS = [501, 502, 503, 504, 505]
SMOKE_QUOTAS = {501: 5, 502: 5, 503: 5, 504: 5, 505: 4}   # 24 total
PILOT_QUOTA_PER_SEED = 60                                  # 300 total
STABILITY_N = 60

STAGES = {
    'smoke': 'review_replay_smoke_qwen3_32b_p24_seed42',
    'pilot': 'review_replay_pilot_qwen3_32b_p300_seed42',
    'stability': 'review_replay_stability_qwen3_32b_p60_seed42',
}

ALLOWED_WRITE_PREFIXES = (
    'data/review_replay',
    'outputs/logs/review_replay_',
    'outputs/docs/review_replay_',
)

TERCILES = ['t1', 't2', 't3']
BUCKETS = ['near', 'mid', 'far']
POLICY_ANCHOR = '## Review Instructions'
PAPER_HEADING = '## Paper'
MEMORY_HEADING = '## Your Recent Experiences'

# --------------------------------------------------------------------------
# Study registry (--study). 'reviewer_monoculture' holds the exact original
# literals above and is the default, so legacy behavior is unchanged.
# 'network_proximity' = professional-network proximity bias study
# (prereg configs/network_review_bias.json).
# --------------------------------------------------------------------------

NRB_STRATA = ('d2', 'd3', 'far')
NRB_CONDITIONS = ('A', 'B', 'C')
POLICY_HEADING = '## Your Review Approach'

STUDIES = {
    'reviewer_monoculture': {
        'prereg': PREREG_V1, 'corpus_dir': CORPUS_DIR, 'stages': dict(STAGES),
        'write_prefixes': ALLOWED_WRITE_PREFIXES,
        'source_ids': list(SOURCE_IDS), 'source_seeds': list(SOURCE_SEEDS),
        'seed_ctx_root': 'review_replay', 'task_sort_column': 'task_uid',
        'serving_note': 'Model and endpoint are supplied for each public run',
    },
    'network_proximity': {
        'prereg': os.path.join(REPO_ROOT, 'configs/network_review_bias.json'),
        'corpus_dir': 'data/network_review_bias',
        'stages': {'smoke': 'network_review_bias_smoke_qwen3_32b_p24_seed42',
                   'main': 'network_review_bias_main_qwen3_32b_p300_seed42',
                   'stability': 'network_review_bias_stability_qwen3_32b_p60_seed42'},
        'write_prefixes': ('data/network_review_bias',
                           'outputs/logs/network_review_bias_',
                           'outputs/docs/network_review_bias_',
),
        'source_ids': [], 'source_seeds': [7001, 7002],
        'seed_ctx_root': 'nrb', 'task_sort_column': 'exec_key',
        'serving_note': 'Model and endpoint are supplied for each public run',
    },

}
ACTIVE_STUDY = 'reviewer_monoculture'


def _activate_study(study: str):
    """Point the module globals at one study's frozen configuration. The
    monoculture entry holds the original literals, so the default is a no-op."""
    global ACTIVE_STUDY, PREREG_V1, CORPUS_DIR, STAGES, ALLOWED_WRITE_PREFIXES
    global SOURCE_IDS, SOURCE_SEEDS
    cfg = STUDIES[study]
    ACTIVE_STUDY = study
    PREREG_V1 = cfg['prereg']
    CORPUS_DIR = cfg['corpus_dir']
    STAGES = cfg['stages']
    ALLOWED_WRITE_PREFIXES = cfg['write_prefixes']
    SOURCE_IDS = cfg['source_ids']
    SOURCE_SEEDS = cfg['source_seeds']

# The exact strings generate_batch prepends (models.py): scanned for leakage too.
DEFAULT_SYSTEM_PROMPT = ("This is a simulation of an academic ecosystem, where researchers choose "
                         "research directions and submit papers, reviewers conduct peer reviews of "
                         "papers, and funding agencies allocate fundings.")


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def assert_write_allowed(path: str) -> str:
    """Hard path guard: every pilot write must live under an allowed prefix."""
    rel = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    if not any(rel == p.rstrip('/') or rel.startswith(p) for p in ALLOWED_WRITE_PREFIXES):
        raise PermissionError(f"write to '{rel}' is outside the reviewer-replay allowlist "
                              f"{ALLOWED_WRITE_PREFIXES}")
    return path


def assert_safe_vllm_url(url: str):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('Supply an HTTP(S) model endpoint')


# --------------------------------------------------------------------------
# Prompt construction (pure adapters over the unmodified simulator template)
# --------------------------------------------------------------------------

def load_prereg() -> dict:
    with open(PREREG_V1) as f:
        return json.load(f)


def insert_policy_block(prompt: str, policy_text: str) -> str:
    """Insert the frozen reviewer-policy block immediately before the unique
    '## Review Instructions' anchor. Pure string operation; the base template
    stays byte-identical otherwise."""
    assert prompt.count(POLICY_ANCHOR) == 1, "review-prompt anchor not unique"
    block = f"## Your Review Approach\n{policy_text}\n\n"
    return prompt.replace(POLICY_ANCHOR, block + POLICY_ANCHOR)


def build_controlled_prompt(paper: dict, slot_expertise: list, policy_text: str,
                            author_info: dict = None):
    """Regime A/B prompt: unmodified get_review_prompt on a bare SimulationAgent
    carrying the slot's paper-matched expertise and an empty memory bank."""
    from utopia.agents.base_agent import SimulationAgent
    agent = SimulationAgent(reputation=5, agent_id='replay_reviewer', llm=None)
    agent.expertise = list(slot_expertise)
    agent.memory_bank = []
    prompt, response_format = agent.get_review_prompt(paper, author_info=author_info)
    return insert_policy_block(prompt, policy_text), response_format


def review_response_format():
    from utopia.agents.base_agent import PaperReview
    return {
        'type': 'json_schema',
        'json_object': {
            'name': 'paper_submission',
            'strict': True,
            'schema': PaperReview.model_json_schema(),
        },
    }


def filter_memory_at_review_time(memory_bank: list, review_year: int) -> list:
    """Reviewer memory as it stood when year-`review_year` reviews were written:
    all prior-year entries plus this year's pre-review direction selections
    (verified append sites: research_direction.py phase 1; *_reviews_received
    is appended only after phase-3 reviews)."""
    kept = []
    for m in memory_bank:
        assert 'year' in m and 'thought' in m and 'type' in m, f"malformed memory entry: {list(m)}"
        if m['year'] < review_year or (m['year'] == review_year
                                       and m['type'] == 'select_research_direction'):
            kept.append(m)
    return kept


def assign_profiles(paper_id: str, profile_keys: list) -> list:
    """3 distinct regime-B profiles per paper, frozen per (paper, slot)."""
    rng = np.random.default_rng(derive_seed('profile_assign', paper_id))
    return list(rng.choice(sorted(profile_keys), size=3, replace=False))


def prestige_author_info(prereg: dict, arm: str) -> dict:
    p = prereg['prestige_intervention']
    return {
        'author_name': p['author_identifier_both_arms'],
        'institution': p['institution_high'] if arm == 'prestige_high' else p['institution_low'],
        'network_relationship': p['relationship_both_arms'],
    }


# --------------------------------------------------------------------------
# Section-aware blind-leakage scanner
# --------------------------------------------------------------------------

def split_sections(assembled: str) -> dict:
    """Split the assembled request into '## '-delimited sections; text before
    the first heading is 'preamble' (system prompt + notes + role line)."""
    sections, name, buf = {}, 'preamble', []
    for line in assembled.splitlines(keepends=True):
        if line.startswith('## '):
            sections[name] = sections.get(name, '') + ''.join(buf)
            name, buf = line.strip(), []
        else:
            buf.append(line)
    sections[name] = sections.get(name, '') + ''.join(buf)
    return sections


def scan_leakage(prompt: str, needles: dict, check_title_in_memory: bool = False):
    """Return (ok, reason, section). The '## Paper' section may legitimately
    contain the title/abstract/topics; every other section must contain none of
    the identity needles (author id, submitting institution, arxiv id), and for
    C replay the current title must not appear inside reviewer memory."""
    assembled = f"{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"
    sections = split_sections(assembled)
    for sec_name, text in sections.items():
        if sec_name.startswith(PAPER_HEADING):
            continue
        for kind, needle in needles.items():
            if kind == 'title':
                continue
            if needle and needle in text:
                return False, f'{kind} present outside paper section', sec_name
    if check_title_in_memory:
        for sec_name, text in sections.items():
            if sec_name.startswith(MEMORY_HEADING) and needles.get('title') \
                    and needles['title'] in text:
                return False, 'current paper title inside reviewer memory', sec_name
    return True, '', ''


# --------------------------------------------------------------------------
# Sampling: cells, deterministic collapse, largest-remainder quotas, IPW
# --------------------------------------------------------------------------

def year_tercile(sim_year: int) -> str:
    return 't1' if sim_year <= 3 else ('t2' if sim_year <= 6 else 't3')


def collapse_cells(counts: dict, min_n: int = MIN_CELL_ELIGIBLE) -> dict:
    """Deterministic collapse for ONE source seed.

    counts: {(tercile, bucket, accepted): eligible_count} over the full grid.
    Returns {(tercile, bucket, accepted): final_cell_label}.

    Prereg v1 algorithm: (1) merge adjacent year terciles (t1+t2 first, then +t3)
    within the same bucket x accepted class; (2) if still sparse, merge distance
    buckets near->mid->far within the same accepted class (tercile partitions
    aligned to the coarsest of the merged buckets); never merge accepted with
    rejected; never merge across source seeds.
    """
    PARTITIONS = {0: [('t1',), ('t2',), ('t3',)], 1: [('t1', 't2'), ('t3',)], 2: [('t1', 't2', 't3')]}

    def part_ok(level, bucket_group, acc):
        return all(sum(counts.get((t, b, acc), 0) for t in group for b in bucket_group) >= min_n
                   for group in PARTITIONS[level])

    def finest_partition(bucket_group, acc):
        for level in (0, 1, 2):
            if part_ok(level, bucket_group, acc):
                return level
        return 2

    mapping = {}
    for acc in (False, True):
        # step 1: per-bucket tercile partitions
        levels = {b: finest_partition((b,), acc) for b in BUCKETS}
        sparse = {b for b in BUCKETS if not part_ok(levels[b], (b,), acc)}
        groups = [((b,), levels[b]) for b in BUCKETS]
        # step 2: bucket merging, near->mid then ->far
        if sparse & {'near', 'mid'}:
            g = ('near', 'mid')
            lvl = finest_partition(g, acc)
            groups = [(g, lvl), (('far',), levels['far'])]
            if (not part_ok(lvl, g, acc)) or 'far' in sparse:
                g = ('near', 'mid', 'far')
                groups = [(g, finest_partition(g, acc))]
        elif 'far' in sparse:
            g = ('mid', 'far')
            lvl = finest_partition(g, acc)
            groups = [(('near',), levels['near']), (g, lvl)]
            if not part_ok(lvl, g, acc):
                g = ('near', 'mid', 'far')
                groups = [(g, finest_partition(g, acc))]
        for bucket_group, level in groups:
            for terc_group in PARTITIONS[level]:
                label = f"{'+'.join(terc_group)}|{'+'.join(bucket_group)}|{'acc' if acc else 'rej'}"
                for t in terc_group:
                    for b in bucket_group:
                        mapping[(t, b, acc)] = label
    return mapping


def largest_remainder(quota: int, counts: dict) -> dict:
    """Allocate `quota` across cells proportionally to eligible counts with the
    deterministic largest-remainder rule (ties broken by cell label)."""
    total = sum(counts.values())
    if total == 0:
        return {c: 0 for c in counts}
    labels = sorted(counts)
    shares = {c: quota * counts[c] / total for c in labels}
    alloc = {c: min(int(np.floor(shares[c])), counts[c]) for c in labels}
    while sum(alloc.values()) < min(quota, total):
        candidates = [c for c in labels if alloc[c] < counts[c]]
        c = max(candidates, key=lambda c: (shares[c] - np.floor(shares[c]), c))
        alloc[c] += 1
        shares[c] -= 1  # so repeated top-ups rotate deterministically
    return alloc


def sample_stage(df_seed: pd.DataFrame, quota: int, rng_label: str, source_seed: int) -> pd.DataFrame:
    """Sample `quota` papers within one source seed using the frozen final-cell
    partition already present in df_seed['final_cell']. Adds inclusion stats."""
    counts = df_seed.groupby('final_cell').size().to_dict()
    alloc = largest_remainder(quota, counts)
    picks = []
    for cell, n in sorted(alloc.items()):
        if n == 0:
            continue
        pool = sorted(df_seed.loc[df_seed.final_cell == cell, 'paper_id'])
        rng = np.random.default_rng(derive_seed(GLOBAL_SEED, rng_label, source_seed, cell))
        chosen = list(np.array(pool)[rng.permutation(len(pool))[:n]])
        sub = df_seed[df_seed.paper_id.isin(chosen)].copy()
        sub['cell_eligible'] = counts[cell]
        sub['cell_selected'] = n
        sub['inclusion_prob'] = n / counts[cell]
        sub['ipw_weight'] = counts[cell] / n
        picks.append(sub)
    return pd.concat(picks, ignore_index=True)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def load_checkpoint(source_id: str, year: int, outputs_root: str = 'outputs') -> dict:
    base = os.path.join(outputs_root, 'checkpoints', source_id, f'checkpoint_year_{year}.json')
    return read_json(base + '.gz' if os.path.exists(base + '.gz') else base)


def build_eligibility(outputs_root: str = 'outputs') -> pd.DataFrame:
    """One pass over all 50 source checkpoints: per submission-year ledger row,
    resolve the matching review round + reviewer agents and capture the frozen
    paper content and slot metadata."""
    rows = []
    for source_id, source_seed in zip(SOURCE_IDS, SOURCE_SEEDS):
        ledger = pd.read_parquet(os.path.join(outputs_root, 'docs', source_id, 'paper.parquet'))
        integ = json.load(open(os.path.join(outputs_root, 'docs', source_id, 'integrity_checks.json')))
        assert all(y.get('passed', True) for y in integ.values()) if isinstance(integ, dict) else True, \
            f'integrity failures in {source_id}'
        for year, ydf in ledger.groupby('year'):
            ck = load_checkpoint(source_id, int(year), outputs_root)
            papers = {p['id']: p for p in ck['paper_tracker']['papers']}
            agents = {a['id']: a for a in ck['ecosystem_data']['agents']}
            for _, r in ydf.iterrows():
                rec = {
                    'paper_id': r.paper_id, 'source_seed': source_seed, 'source_id': source_id,
                    'year': int(r.year), 'conference': r.conference, 'accepted': bool(r.accepted),
                    'strategy': r.strategy, 'institution': r.institution,
                    'novelty_score': float(r.novelty_score), 'distance_bucket': r.distance_bucket,
                    'review_score_orig': float(r.review_score), 'author_id': r.author_id,
                    'eligible': False, 'ineligible_reason': '',
                }
                p = papers.get(r.paper_id)
                if p is None:
                    rec['ineligible_reason'] = 'paper missing from submission-year checkpoint'
                    rows.append(rec); continue
                rounds = [rh for rh in p.get('review_history', [])
                          if rh.get('year') == int(r.year) and rh.get('conference') == r.conference]
                if len(rounds) > 1:
                    raise RuntimeError(f'ambiguous review round for {r.paper_id} y{year} {r.conference}')
                if not rounds:
                    rec['ineligible_reason'] = 'no matching review round'
                    rows.append(rec); continue
                reviews = rounds[0].get('reviews', [])
                if len(reviews) != 3:
                    rec['ineligible_reason'] = f'{len(reviews)} reviews (need 3)'
                    rows.append(rec); continue
                slot_agents = [agents.get(rv['reviewer_id']) for rv in reviews]
                if any(a is None or not a.get('expertise') for a in slot_agents):
                    rec['ineligible_reason'] = 'unresolvable reviewer'
                    rows.append(rec); continue
                if not (p.get('abstract') or '').strip():
                    rec['ineligible_reason'] = 'empty abstract'
                    rows.append(rec); continue
                author = agents.get(r.author_id, {})
                rec.update({
                    'eligible': True,
                    'title': p['title'], 'abstract': p['abstract'],
                    'topics': json.dumps(list(p.get('topics') or [])),
                    'author_university': author.get('university_name') or author.get('company_name') or '',
                    'orig_reviewer_ids': json.dumps([rv['reviewer_id'] for rv in reviews]),
                    'orig_scores': json.dumps([rv['overall_score'] for rv in reviews]),
                    'orig_justifications': json.dumps([rv['justification'] for rv in reviews]),
                    'slot_expertise': json.dumps([
                        [e if isinstance(e, str) else e.get('topic', str(e)) for e in a['expertise']]
                        for a in slot_agents]),
                })
                rows.append(rec)
            del ck, papers, agents
            logger.info(f'eligibility: {source_id} year {year} done ({len(rows)} rows so far)')
    return pd.DataFrame(rows)


def make_tasks_for_paper(row: pd.Series, prereg: dict, stage: str, regimes: tuple,
                         arms: tuple, panel: int) -> list:
    paper = {'title': row.title, 'abstract': row.abstract, 'topics': json.loads(row.topics)}
    slot_expertise = json.loads(row.slot_expertise)
    profiles = assign_profiles(row.paper_id, list(prereg['prompt_construction']['policy_texts_B']))
    arxiv = row['arxiv_id'] if 'arxiv_id' in row.index else row.paper_id
    needles = {'author_id': row.author_id, 'institution': row.author_university,
               'arxiv_id': arxiv, 'title': row.title}
    tasks = []
    for regime in regimes:
        for arm in arms:
            author_info = None if arm == 'blind' else prestige_author_info(prereg, arm)
            for slot in range(3):
                policy_key = 'standard' if regime == 'A' else profiles[slot]
                policy_text = (prereg['prompt_construction']['policy_text_A'] if regime == 'A'
                               else prereg['prompt_construction']['policy_texts_B'][policy_key])
                prompt, _ = build_controlled_prompt(paper, slot_expertise[slot],
                                                    policy_text, author_info)
                ok, reason, section = scan_leakage(prompt, needles)
                if not ok:
                    raise RuntimeError(f'leakage in controlled prompt {row.paper_id} '
                                       f'{regime}/{arm}/{slot}: {reason} @ {section}')
                tasks.append({
                    'task_uid': f'{row.paper_id}|{regime}|{arm}|{panel}|{slot}',
                    'stage': stage, 'paper_id': row.paper_id, 'source_seed': row.source_seed,
                    'year': row.year, 'conference': row.conference, 'regime': regime,
                    'arm': arm, 'panel': panel, 'slot': slot, 'policy_key': policy_key,
                    'orig_reviewer_id': json.loads(row.orig_reviewer_ids)[slot],
                    'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                })
    return tasks


def make_c_replay_tasks(smoke_df: pd.DataFrame, outputs_root: str = 'outputs') -> tuple:
    """Regime-C sanity replay for the smoke set only: reconstruct the original
    reviewer agents (year-filtered memory) via the unmodified from_dict +
    get_review_prompt path. Returns (tasks, leakage_exclusions)."""
    from utopia.agents.base_agent import MultiAgentEcosystem
    tasks, exclusions = [], []
    for (source_id, year), grp in smoke_df.groupby(['source_id', 'year']):
        ck = load_checkpoint(source_id, int(year), outputs_root)
        eco = MultiAgentEcosystem.from_dict(ck['ecosystem_data'], llm=None)
        agents = dict(eco.agent_population)
        for _, row in grp.iterrows():
            paper = {'title': row.title, 'abstract': row.abstract, 'topics': json.loads(row.topics)}
            arxiv = row['arxiv_id'] if 'arxiv_id' in row.index else row.paper_id
            needles = {'author_id': row.author_id, 'institution': row.author_university,
                       'arxiv_id': arxiv, 'title': row.title}
            for slot, reviewer_id in enumerate(json.loads(row.orig_reviewer_ids)):
                agent = agents[reviewer_id]
                agent.memory_bank = filter_memory_at_review_time(agent.memory_bank, int(row.year))
                prompt, _ = agent.get_review_prompt(paper, author_info=None)
                ok, reason, section = scan_leakage(prompt, needles, check_title_in_memory=True)
                if not ok:
                    exclusions.append({'paper_id': row.paper_id, 'regime': 'C', 'slot': slot,
                                       'reviewer_id': reviewer_id, 'reason': reason,
                                       'section': section})
                    continue
                tasks.append({
                    'task_uid': f'{row.paper_id}|C|blind|1|{slot}',
                    'stage': 'smoke', 'paper_id': row.paper_id, 'source_seed': row.source_seed,
                    'year': row.year, 'conference': row.conference, 'regime': 'C',
                    'arm': 'blind', 'panel': 1, 'slot': slot, 'policy_key': 'original_agent',
                    'orig_reviewer_id': reviewer_id, 'prompt': prompt,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                })
        del ck, eco, agents
    return tasks, exclusions


# --------------------------------------------------------------------------
# Network-proximity study (network_review_bias): prompt adapter, pre-review
# graph snapshots, eligibility, frozen triplet matching, extraction.
# All frozen rules live in the study prereg (prereg_v1.json); code implements
# them mechanically.
# --------------------------------------------------------------------------

def nrb_synthetic_id(focal_author_id: str) -> str:
    return f"Researcher {hashlib.sha256(focal_author_id.encode()).hexdigest()[:8]}"


def nrb_author_block(synthetic_id: str, institution: str, relationship_value: str) -> str:
    return ("## Author Information (Non-Blind Review)\n"
            f"- Author: {synthetic_id}\n"
            f"- Institution: {institution}\n"
            f"- Professional-network relationship: {relationship_value}\n\n")


def nrb_relationship_value(prereg: dict, condition: str, stratum: str, far_kind: str) -> str:
    """Frozen relationship-field content. B: the 'undisclosed' placeholder;
    C: the accurate sentence for the pair's true stratum/far_kind."""
    if condition == 'B':
        return prereg['conditions']['B_identity_only']['relationship_value']
    key = {'d2': 'd2', 'd3': 'd3'}.get(stratum) or \
        ('disconnected' if far_kind == 'disconnected' else 'far_connected')
    return prereg['conditions']['C_relationship'][key]


def nrb_build_prompt(paper: dict, expertise: list, policy_text: str, author_block: str):
    """A/B/C prompt: blind template + policy block, then (B/C only) the study
    author block inserted immediately above the policy heading — the position
    the simulator template itself uses for its author block."""
    prompt, response_format = build_controlled_prompt(paper, expertise, policy_text, None)
    if author_block:
        assert prompt.count(POLICY_HEADING) == 1, 'policy heading not unique'
        prompt = prompt.replace(POLICY_HEADING, author_block + POLICY_HEADING)
    return prompt, response_format


def nrb_condition_order(paper_id: str, reviewer_id: str) -> list:
    """Deterministic near-balanced A/B/C execution order per triplet."""
    import itertools
    perms = list(itertools.permutations(NRB_CONDITIONS))
    return list(perms[derive_seed(GLOBAL_SEED, 'nrb', 'order', paper_id, reviewer_id) % 6])


def nrb_graph_snapshot(edges: list, upto_year: int):
    """Graph of edges first formed strictly before `upto_year`+1, i.e. usable
    for reviews in year upto_year+1. Asserts every kept edge pre-dates it."""
    import networkx as nx
    g = nx.Graph()
    for e in edges:
        years = e.get('years') or []
        if years and min(years) <= upto_year:
            g.add_edge(e['a'], e['b'])
    return g


def nrb_future_edge_violations(edges: list, review_year: int, snapshot_year: int) -> int:
    """Count edges in the snapshot that would violate the timing rule for a
    review in `review_year` (snapshot must be <= review_year - 1)."""
    if snapshot_year > review_year - 1:
        return sum(1 for e in edges if e.get('years') and min(e['years']) <= snapshot_year
                   and min(e['years']) >= review_year)
    return 0


def nrb_distance(G, focal: str, other: str):
    """(distance:int|None, far_kind) — None distance means disconnected/absent."""
    import networkx as nx
    if not (G.has_node(focal) and G.has_node(other)):
        return None, 'disconnected'
    try:
        d = nx.shortest_path_length(G, focal, other)
    except nx.NetworkXNoPath:
        return None, 'disconnected'
    return d, ('connected_ge4' if d >= 4 else '')


def nrb_stratum_of(distance, far_kind: str):
    if distance == 2:
        return 'd2'
    if distance == 3:
        return 'd3'
    if distance is None or distance >= 4:
        return 'far'
    return None  # distance 0/1: never a stratum


def nrb_candidates_for_paper(G, paper_row: dict, agents: dict, coi: dict,
                             connected_far_only: bool) -> tuple:
    """Per-stratum candidate reviewer pools after ALL exclusions (screened
    against every author): authorship, distance-1 to any author, recorded COI
    with any author (either direction), same institution as any author.
    Returns ({'d2': [rid...], 'd3': [...], 'far': [...]}, per-candidate meta).
    Single-source BFS per author keeps this O(authors * |E|) per paper."""
    import networkx as nx
    authors = paper_row['author_list']
    focal = paper_row['focal_author_id']
    author_insts = {agents[a]['institution'] for a in authors if a in agents}
    author_coi = set()
    for a in authors:
        author_coi |= coi.get(a, set())
    dists = {a: (nx.single_source_shortest_path_length(G, a) if G.has_node(a) else {})
             for a in authors}
    d1_of_any_author = {n for a in authors for n, d in dists[a].items() if d == 1}
    pools = {'d2': [], 'd3': [], 'far': []}
    meta = {}
    for rid, ag in agents.items():
        if rid in authors or not ag['can_review'] or not ag['expertise']:
            continue
        if rid in d1_of_any_author or rid in author_coi or (coi.get(rid, set()) & set(authors)):
            continue
        if ag['institution'] in author_insts:
            continue
        d = dists[focal].get(rid)
        if d is None:
            far_kind = 'disconnected'
        else:
            far_kind = 'connected_ge4' if d >= 4 else ''
        stratum = nrb_stratum_of(d, far_kind)
        if stratum is None:
            continue
        if stratum == 'far' and connected_far_only and far_kind == 'disconnected':
            continue
        pools[stratum].append(rid)
        min_d_any = min((dists[a].get(rid, 10 ** 6) for a in authors), default=10 ** 6)
        meta[rid] = {'distance': -1 if d is None else d, 'far_kind': far_kind,
                     'min_distance_any_author': min_d_any,
                     'expertise': list(ag['expertise']),
                     'institution': ag['institution']}
    return pools, meta


def nrb_best_triplet(pools: dict, sims: dict, use_counts: dict, reuse_cap: int,
                     tol: float):
    """Frozen triplet objective: among feasible (d2,d3,far) triplets (similarity
    gap <= tol, every reviewer under the reuse cap) minimize, in order:
    (1) max pairwise |similarity gap|; (2) sum of current use counts;
    (3) max use count; (4) lexicographic reviewer tuple.
    Implemented as a sliding window over the similarity-sorted candidate list —
    the max gap of a triplet depends only on its (min,max) similarity, and the
    remaining keys are separable per stratum within a window; all minimal-width
    windows are compared under the full key (brute-force-equivalence unit-tested)."""
    avail = {s: [r for r in pools[s] if use_counts.get(r, 0) < reuse_cap] for s in NRB_STRATA}
    if not all(avail[s] for s in NRB_STRATA):
        return None
    items = sorted((sims[r], s, r) for s in NRB_STRATA for r in avail[s])
    best = None
    for i in range(len(items)):
        by_stratum = {s: [] for s in NRB_STRATA}
        for sim, s, r in items[i:]:
            gap = sim - items[i][0]
            if gap > tol:
                break
            by_stratum[s].append((sim, r))
            if all(by_stratum[t] for t in NRB_STRATA):
                # candidate window [items[i].sim, sim]: pick per-stratum best
                pick = {}
                for t in NRB_STRATA:
                    pick[t] = min(by_stratum[t],
                                  key=lambda p: (use_counts.get(p[1], 0), p[1]))[1]
                trip = (pick['d2'], pick['d3'], pick['far'])
                g = max(sims[x] for x in trip) - min(sims[x] for x in trip)
                key = (round(g, 6),
                       sum(use_counts.get(x, 0) for x in trip),
                       max(use_counts.get(x, 0) for x in trip),
                       trip)
                if best is None or key < best[0]:
                    best = (key, trip, g)
    if best is None:
        return None
    (_, trip, gap) = best
    return {'d2': trip[0], 'd3': trip[1], 'far': trip[2], 'max_gap': gap}


def nrb_match_corpus(paper_rows: list, pools_by_paper: dict, sims_by_paper: dict,
                     reuse_cap: int, tol: float) -> dict:
    """Frozen deterministic bounded greedy: papers sorted by increasing number
    of feasible triplets (product of pool sizes as the frozen proxy recorded in
    the prereg algorithm; ties by paper ID), one pass + exactly one second pass
    over unmatched papers; assignments never revisited; cap never exceeded."""
    def feasibility(pid):
        p = pools_by_paper[pid]
        return (min(len(p[s]) for s in NRB_STRATA), len(p['d2']) * len(p['d3']) * len(p['far']))

    order = sorted((pid for pid in pools_by_paper
                    if all(pools_by_paper[pid][s] for s in NRB_STRATA)),
                   key=lambda pid: (feasibility(pid), pid))
    use_counts, matched = {}, {}
    for _pass in (1, 2):
        for pid in order:
            if pid in matched:
                continue
            trip = nrb_best_triplet(pools_by_paper[pid], sims_by_paper[pid],
                                    use_counts, reuse_cap, tol)
            if trip is None:
                continue
            matched[pid] = trip
            for s in NRB_STRATA:
                use_counts[trip[s]] = use_counts.get(trip[s], 0) + 1
    return {'matched': matched, 'use_counts': use_counts}


def nrb_load_source(source_dir: str, through_year: int = None) -> dict:
    """Read one legacy preferential-attachment source run: final checkpoint only
    (papers/agents/collaboration are cumulative). Read-only."""
    paths = sorted(glob.glob(os.path.join(source_dir, 'checkpoint_year_*.json')) +
                   glob.glob(os.path.join(source_dir, 'checkpoint_year_*.json.gz')))
    years = sorted({int(os.path.basename(p).split('_')[2].split('.')[0]) for p in paths})
    assert years, f'no checkpoints under {source_dir}'
    year = max(y for y in years if through_year is None or y <= through_year)
    path = os.path.join(source_dir, f'checkpoint_year_{year}.json')
    ck = read_json(path if os.path.exists(path) else path + '.gz')
    pa = ck.get('preferential_attachment') or {}
    collab = pa.get('collaboration_tracker') or {}
    agents = {}
    for a in ck['ecosystem_data']['agents']:
        agents[a['id']] = {
            'institution': a.get('university_name') or a.get('company_name') or '',
            'can_review': bool(a.get('can_review', True)),
            'expertise': [e if isinstance(e, str) else e.get('topic', str(e))
                          for e in (a.get('expertise') or [])],
            'coi': set(a.get('conflict_of_interest') or []),
            'start': a.get('project_start_year'), 'end': a.get('project_end_year'),
        }
    return {'checkpoint_year': year, 'papers': ck['paper_tracker']['papers'],
            'agents': agents, 'edges': collab.get('edges') or [],
            'source_dir': source_dir}


def nrb_paper_records(src: dict) -> list:
    """Per-paper record with focal author, full author list, review year."""
    recs = []
    for p in src['papers']:
        rhs = p.get('review_history') or []
        if not rhs or not (p.get('abstract') or '').strip():
            continue
        review_year = min(rh['year'] for rh in rhs)
        first_round = min(rhs, key=lambda rh: rh['year'])
        aid = p['author_id']
        authors = aid if isinstance(aid, list) else [aid]
        recs.append({
            'paper_id': p['id'], 'title': p['title'], 'abstract': p['abstract'],
            'topics': list(p.get('topics') or []), 'review_year': int(review_year),
            'focal_author_id': authors[0], 'author_list': list(authors),
            'accepted': bool(first_round.get('decision') in (True, 'accept', 'accepted')
                             or p.get('status') in ('accepted', 'published')),
            'n_authors': len(authors),
        })
    return recs


def cmd_extract_nrb(args):
    """Extraction for --study network_proximity. Builds pre-review graph
    snapshots, eligibility funnel, expertise-matched reviewer triplets under the
    frozen algorithm, the structural audit/gate inputs, and (unless
    --structural_audit_only) the frozen smoke/main/stability task files."""
    import networkx as nx
    set_seed(GLOBAL_SEED, use_torch=False)
    prereg = load_prereg()
    os.makedirs(assert_write_allowed(CORPUS_DIR), exist_ok=True)
    tol = 0.10
    reuse_cap = args.reuse_cap
    assert reuse_cap in (10, 15), 'reuse cap frozen to 10 (preregistered fallback 15)'

    from utopia.metrics.embedding_tracker import EmbeddingTracker
    tracker = EmbeddingTracker(cache_dir=os.path.join(CORPUS_DIR, 'embedding_cache'))

    all_rows, funnel = [], {}
    pools_by_paper, sims_by_paper, meta_by_paper = {}, {}, {}
    pools_conn_by_paper = {}
    node_metrics, topology = {}, {}
    for source_dir in args.source_dirs:
        src = nrb_load_source(source_dir, args.through_year)
        seed_label = os.path.basename(os.path.dirname(source_dir.rstrip('/'))) or source_dir
        recs = nrb_paper_records(src)
        funnel[f'{seed_label}:papers_with_reviews_and_abstract'] = len(recs)
        years = sorted({r['review_year'] for r in recs})
        snapshots = {y: nrb_graph_snapshot(src['edges'], y - 1) for y in years}
        # exact betweenness / degree / components per snapshot year
        for y in years:
            G = snapshots[y]
            bt = nx.betweenness_centrality(G) if len(G) > 2 else {n: 0.0 for n in G}
            comp = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
            csize = {n: sum(1 for m in comp if comp[m] == comp[n]) for n in G}
            node_metrics[(seed_label, y)] = {
                n: {'betweenness': bt.get(n, 0.0), 'degree': G.degree(n),
                    'component': comp.get(n, -1), 'component_size': csize.get(n, 0)}
                for n in G}
            topology[f'{seed_label}:y{y}'] = {
                'nodes': G.number_of_nodes(), 'edges': G.number_of_edges(),
                'components': nx.number_connected_components(G) if len(G) else 0,
            }
        coi = {rid: ag['coi'] for rid, ag in src['agents'].items()}
        # embeddings: papers + all reviewer expertise texts, batched via cache
        paper_texts = {r['paper_id']: f"{r['title']}\n{r['abstract']}\n{', '.join(r['topics'])}"
                       for r in recs}
        rev_texts = {rid: ', '.join(sorted(ag['expertise']))
                     for rid, ag in src['agents'].items() if ag['expertise']}
        ids = list(paper_texts) + list(rev_texts)
        embs = tracker._encode_cached([paper_texts.get(i) or rev_texts[i] for i in ids])
        embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
        emb = dict(zip(ids, embs))
        tracker.flush_cache()

        for r in recs:
            y = r['review_year']
            G = snapshots[y]
            pid = f"{seed_label}:{r['paper_id']}"
            focal = r['focal_author_id']
            if not (G.has_node(focal) and G.degree(focal) >= 1):
                funnel[f'{seed_label}:excluded_focal_not_in_pre_review_graph'] = \
                    funnel.get(f'{seed_label}:excluded_focal_not_in_pre_review_graph', 0) + 1
                continue
            pools, meta = nrb_candidates_for_paper(G, r, src['agents'], coi,
                                                   connected_far_only=False)
            pools_c, _ = nrb_candidates_for_paper(G, r, src['agents'], coi,
                                                  connected_far_only=True)
            if not all(pools[s] for s in NRB_STRATA):
                missing = [s for s in NRB_STRATA if not pools[s]]
                funnel[f'{seed_label}:excluded_missing_stratum_{"+".join(missing)}'] = \
                    funnel.get(f'{seed_label}:excluded_missing_stratum_{"+".join(missing)}', 0) + 1
                continue
            sims = {rid: float(emb[r['paper_id']] @ emb[rid])
                    for s in NRB_STRATA for rid in pools[s] if rid in emb}
            pools = {s: [rid for rid in pools[s] if rid in sims] for s in NRB_STRATA}
            pools_c = {s: [rid for rid in pools_c[s] if rid in sims] for s in NRB_STRATA}
            if not all(pools[s] for s in NRB_STRATA):
                funnel[f'{seed_label}:excluded_no_embeddable_candidates'] = \
                    funnel.get(f'{seed_label}:excluded_no_embeddable_candidates', 0) + 1
                continue
            nm = node_metrics[(seed_label, y)]
            row = dict(r)
            row.update({
                'paper_id': pid, 'arxiv_id': r['paper_id'], 'source_label': seed_label,
                'source_dir': source_dir,
                'author_institution': src['agents'].get(focal, {}).get('institution', ''),
                'focal_degree': nm.get(focal, {}).get('degree', 0),
                'focal_betweenness': nm.get(focal, {}).get('betweenness', 0.0),
                'focal_component_size': nm.get(focal, {}).get('component_size', 0),
            })
            all_rows.append(row)
            pools_by_paper[pid] = pools
            pools_conn_by_paper[pid] = pools_c
            sims_by_paper[pid] = sims
            meta_by_paper[pid] = meta
        logger.info(f'{seed_label}: {len(recs)} papers scanned, '
                    f'{sum(1 for x in all_rows if x["source_label"] == seed_label)} candidates')

    papers = {r['paper_id']: r for r in all_rows}
    funnel['candidate_papers_all_strata_present'] = len(papers)

    # matching under both far definitions (structural, pre-generation)
    match_pooled = nrb_match_corpus(list(papers), pools_by_paper, sims_by_paper,
                                    reuse_cap, tol)
    conn_pools = {pid: p for pid, p in pools_conn_by_paper.items()
                  if all(p[s] for s in NRB_STRATA)}
    match_conn = nrb_match_corpus(list(conn_pools), conn_pools, sims_by_paper,
                                  reuse_cap, tol)
    m_pool, m_conn = len(match_pooled['matched']), len(match_conn['matched'])
    _cfg = STUDIES[ACTIVE_STUDY]
    n_smoke = _cfg.get('n_smoke', 24)
    stability_n = _cfg.get('stability_n', STABILITY_N)
    ladder = _cfg.get('target_ladder', (300, 240, 200))
    target = next((t for t in ladder if m_pool - n_smoke >= t), None)
    comparator = 'connected_ge4' if (target and m_conn - n_smoke >= target) else 'far_or_disconnected'
    chosen = match_conn if comparator == 'connected_ge4' else match_pooled

    audit = {
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'prereg_v1_sha256': sha256_file(PREREG_V1),
        'source_dirs': list(args.source_dirs), 'through_year': args.through_year,
        'reuse_cap': reuse_cap, 'similarity_tolerance': tol,
        'funnel': funnel,
        'matched_pooled_far': m_pool, 'matched_connected_far': m_conn,
        'target_ladder_decision': target, 'far_comparator_decision': comparator,
        'topology': topology,
        'reviewer_use_distribution': sorted(chosen['use_counts'].values(), reverse=True)[:50],
        'max_reviewer_reuse': max(chosen['use_counts'].values(), default=0),
        'effective_n_reviewers': (
            (sum(chosen['use_counts'].values()) ** 2 /
             sum(v ** 2 for v in chosen['use_counts'].values()))
            if chosen['use_counts'] else 0),
        'future_edge_violations': 0,  # excluded at snapshot construction; asserted in tests
    }
    _artifact_value, _artifact_path = audit, f'{CORPUS_DIR}/structural_audit.json'
    assert_write_allowed(_artifact_path)
    write_json_atomic(_artifact_path, _artifact_value, indent=2, default=str, trailing_newline=False, streaming=True)
    logger.info(f'structural audit: pooled={m_pool} connected={m_conn} '
                f'target={target} comparator={comparator}')
    if args.structural_audit_only:
        return
    assert target is not None, (
        f'insufficient structural support: fewer than {ladder[-1]} matched papers '
        f'after smoke removal (pooled={m_pool}, smoke={n_smoke}) — stopping per prereg')

    # assemble the matched corpus frame
    rows = []
    for pid, trip in chosen['matched'].items():
        r = papers[pid]
        meta = meta_by_paper[pid]
        sims = sims_by_paper[pid]
        base = {k: v for k, v in r.items() if k not in ('author_list', 'topics')}
        base['author_list'] = json.dumps(r['author_list'])
        base['topics'] = json.dumps(r['topics'])
        base['max_similarity_gap'] = trip['max_gap']
        for s in NRB_STRATA:
            rid = trip[s]
            base[f'reviewer_{s}'] = rid
            base[f'similarity_{s}'] = sims[rid]
            base[f'distance_{s}'] = meta[rid]['distance']
            base[f'far_kind_{s}'] = meta[rid]['far_kind']
            base[f'min_distance_any_author_{s}'] = meta[rid]['min_distance_any_author']
            base[f'reviewer_expertise_{s}'] = json.dumps(meta[rid]['expertise'])
            base[f'reviewer_institution_{s}'] = meta[rid]['institution']
            nm = node_metrics[(r['source_label'], r['review_year'])]
            base[f'reviewer_degree_{s}'] = nm.get(rid, {}).get('degree', 0)
            base[f'reviewer_betweenness_{s}'] = nm.get(rid, {}).get('betweenness', 0.0)
        rows.append(base)
    corpus = pd.DataFrame(rows).sort_values('paper_id').reset_index(drop=True)

    # frozen stratified sampling: smoke 24 first (disjoint), then main target,
    # then the 60-paper stability subset from the SELECTED main sample
    corpus['degree_bucket'] = pd.qcut(corpus.focal_degree.rank(method='first'),
                                      2, labels=['dlo', 'dhi']).astype(str)
    corpus['cell'] = (corpus.source_label + '|y' + corpus.review_year.astype(str) +
                      '|' + corpus.accepted.map({True: 'acc', False: 'rej'}) +
                      '|' + corpus.degree_bucket)

    def stratified_pick(frame, n, label):
        alloc = largest_remainder(n, frame.groupby('cell').size().to_dict())
        ids = []
        for cell, k in sorted(alloc.items()):
            pool = sorted(frame.loc[frame.cell == cell, 'paper_id'])
            rng = np.random.default_rng(derive_seed(GLOBAL_SEED, 'nrb', label, cell))
            ids += list(np.array(pool)[rng.permutation(len(pool))[:k]])
        return ids

    smoke_ids = stratified_pick(corpus, n_smoke, 'smoke_select')
    rest = corpus[~corpus.paper_id.isin(smoke_ids)].copy()
    main_ids = stratified_pick(rest, target, 'main_select')
    main_df = rest[rest.paper_id.isin(main_ids)].copy()
    counts = main_df.groupby('cell').size()
    elig_counts = rest.groupby('cell').size()
    main_df['inclusion_prob'] = main_df.cell.map(counts / elig_counts).astype(float)
    stab_ids = stratified_pick(main_df, stability_n, 'stability_select')
    main_df['in_stability_subset'] = main_df.paper_id.isin(stab_ids)
    smoke_df = corpus[corpus.paper_id.isin(smoke_ids)].copy()
    assert len(smoke_df) == n_smoke and len(main_df) == target
    assert main_df.in_stability_subset.sum() == stability_n
    assert not set(smoke_df.paper_id) & set(main_df.paper_id)

    smoke_tasks = nrb_make_tasks(smoke_df, prereg, 'smoke', panel='primary')
    main_tasks = nrb_make_tasks(main_df, prereg, 'main', panel='primary')
    stab_tasks = nrb_make_tasks(main_df[main_df.in_stability_subset], prereg,
                                'stability', panel='stability')
    assert len(smoke_tasks) == n_smoke * 9 and len(main_tasks) == target * 9
    assert len(stab_tasks) == stability_n * 9
    uids = [t['task_uid'] for t in smoke_tasks + main_tasks + stab_tasks]
    assert len(uids) == len(set(uids)), 'task UID collision'

    for name, frame in (('matched_corpus', corpus), ('smoke_corpus', smoke_df),
                        ('main_corpus', main_df)):
        _artifact_value, _artifact_path = frame, f'{CORPUS_DIR}/{name}.parquet'
        assert_write_allowed(_artifact_path)
        write_parquet_atomic(_artifact_path, _artifact_value, index=False)
    for name, tasks in (('smoke', smoke_tasks), ('main', main_tasks),
                        ('stability', stab_tasks)):
        _artifact_value, _artifact_path = pd.DataFrame(sorted(tasks, key=lambda t: t['exec_key'])), f'{CORPUS_DIR}/replay_tasks_{name}.parquet'
        assert_write_allowed(_artifact_path)
        write_parquet_atomic(_artifact_path, _artifact_value, index=False)

    manifest = {
        'study': 'network_proximity', 'corpus_version': CORPUS_VERSION,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'global_seed': GLOBAL_SEED,
        'prereg_v1_sha256': sha256_file(PREREG_V1),
        'source_dirs': list(args.source_dirs), 'through_year': args.through_year,
        'chunk_size': CHUNK_SIZE, 'reuse_cap': reuse_cap,
        'far_comparator': comparator, 'main_target': target,
        'n_smoke': int(len(smoke_df)), 'n_main': int(len(main_df)),
        'n_stability': int(STABILITY_N),
        'task_counts': {'smoke': len(smoke_tasks), 'main': len(main_tasks),
                        'stability': len(stab_tasks)},
        'file_sha256': {os.path.basename(p): sha256_file(p)
                        for p in sorted(glob.glob(f'{CORPUS_DIR}/*.parquet'))},
    }
    _artifact_value, _artifact_path = manifest, f'{CORPUS_DIR}/corpus_manifest.json'
    assert_write_allowed(_artifact_path)
    write_json_atomic(_artifact_path, _artifact_value, indent=2, default=str, trailing_newline=False, streaming=True)
    logger.info(f"nrb extract complete: {manifest['task_counts']}, "
                f"comparator={comparator}, target={target}")


def nrb_make_tasks(df: pd.DataFrame, prereg: dict, stage: str, panel: str) -> list:
    """9 tasks per paper (3 strata x 3 conditions) with block-randomized A/B/C
    execution order, full provenance, and per-task leakage/manipulation checks."""
    policy_text = prereg['reviewer_policy_and_memory']['policy_text']
    tasks = []
    for _, row in df.iterrows():
        paper = {'title': row.title, 'abstract': row.abstract,
                 'topics': json.loads(row.topics)}
        syn = nrb_synthetic_id(row.focal_author_id)
        inst = row.author_institution
        c_sentences = list(prereg['conditions']['C_relationship'].values())
        for s in NRB_STRATA:
            rid = row[f'reviewer_{s}']
            order = nrb_condition_order(row.paper_id, rid)
            prompts = {}
            for cond in NRB_CONDITIONS:
                if cond == 'A':
                    block = ''
                else:
                    rel = nrb_relationship_value(prereg, cond, s, row[f'far_kind_{s}'])
                    block = nrb_author_block(syn, inst, rel)
                prompt, _ = nrb_build_prompt(paper, json.loads(row[f'reviewer_expertise_{s}']),
                                             policy_text, block)
                prompts[cond] = prompt
            # manipulation + leakage checks (frozen)
            b_rel = prereg['conditions']['B_identity_only']['relationship_value']
            c_rel = nrb_relationship_value(prereg, 'C', s, row[f'far_kind_{s}'])
            assert prompts['B'].replace(b_rel, '@REL@') == prompts['C'].replace(c_rel, '@REL@'), \
                f'B/C differ outside relationship field: {row.paper_id}/{s}'
            for needle in [syn, inst, b_rel] + c_sentences:
                assert needle not in prompts['A'], \
                    f'condition-A leakage ({needle[:30]}...): {row.paper_id}/{s}'
            for needle in c_sentences:
                assert needle not in prompts['B'], f'relationship leaked into B: {row.paper_id}/{s}'
            assert row.focal_author_id not in prompts['B'] + prompts['C'], \
                f'raw author id leaked: {row.paper_id}'
            for cond in NRB_CONDITIONS:
                tasks.append({
                    'task_uid': f'{row.paper_id}|{rid}|{cond}|{s}|{panel}',
                    'exec_key': f'{row.paper_id}|{s}|{order.index(cond)}|{cond}',
                    'stage': stage, 'panel': panel, 'paper_id': row.paper_id,
                    'reviewer_id': rid, 'focal_author_id': row.focal_author_id,
                    'condition': cond, 'stratum': s, 'far_kind': row[f'far_kind_{s}'],
                    'source_label': row.source_label, 'review_year': int(row.review_year),
                    'condition_order': ''.join(order),
                    'distance': int(row[f'distance_{s}']),
                    'similarity': float(row[f'similarity_{s}']),
                    'reviewer_betweenness': float(row[f'reviewer_betweenness_{s}']),
                    'focal_betweenness': float(row.focal_betweenness),
                    'prompt': prompts[cond],
                    'prompt_sha256': hashlib.sha256(prompts[cond].encode()).hexdigest(),
                })
    return tasks


def cmd_extract(args):
    set_seed(GLOBAL_SEED, use_torch=False)
    prereg = load_prereg()
    os.makedirs(assert_write_allowed(CORPUS_DIR), exist_ok=True)
    log_dir = assert_write_allowed('outputs/logs/review_replay_corpus_v1')
    os.makedirs(log_dir, exist_ok=True)

    elig = build_eligibility(args.source_outputs_root)
    funnel = elig.groupby(['eligible', 'ineligible_reason']).size().reset_index(name='n')
    logger.info(f'eligibility funnel:\n{funnel.to_string(index=False)}')

    df = elig[elig.eligible].copy()
    # Sampling unit = one submission: keep the EARLIEST review round per
    # (source_seed, arxiv paper) — resubmissions create duplicate arxiv ids across
    # years, and the same corpus paper can appear in several source-seed worlds.
    n_before = len(df)
    df = df.sort_values(['source_seed', 'paper_id', 'year']) \
           .drop_duplicates(['source_seed', 'paper_id'], keep='first').copy()
    logger.info(f'dedup to earliest round per (seed, paper): {n_before} -> {len(df)} units')
    df['arxiv_id'] = df.paper_id
    df['paper_id'] = df.source_seed.astype(str) + ':' + df.arxiv_id  # unique unit key
    df['tercile'] = df.year.map(year_tercile)
    # frozen final-cell partition per source seed, computed on the FULL eligible pool
    df['final_cell'] = ''
    for seed in SOURCE_SEEDS:
        m = df.source_seed == seed
        counts = df[m].groupby(['tercile', 'distance_bucket', 'accepted']).size().to_dict()
        full = {(t, b, a): counts.get((t, b, a), 0)
                for t in TERCILES for b in BUCKETS for a in (False, True)}
        mapping = collapse_cells(full)
        df.loc[m, 'final_cell'] = [mapping[(t, b, a)] for t, b, a in
                                   zip(df.loc[m, 'tercile'], df.loc[m, 'distance_bucket'],
                                       df.loc[m, 'accepted'])]
    df['orig_cell'] = df.tercile + '|' + df.distance_bucket + '|' + \
        df.accepted.map({True: 'acc', False: 'rej'})

    # smoke first (disjoint holdout), then pilot from the remaining pool
    smoke_parts, pilot_parts = [], []
    for seed in SOURCE_SEEDS:
        pool = df[df.source_seed == seed]
        smoke = sample_stage(pool, SMOKE_QUOTAS[seed], 'smoke_sample', seed)
        remaining = pool[~pool.paper_id.isin(smoke.paper_id)]
        pilot = sample_stage(remaining, PILOT_QUOTA_PER_SEED, 'corpus_sample', seed)
        smoke_parts.append(smoke)
        pilot_parts.append(pilot)
    smoke_df = pd.concat(smoke_parts, ignore_index=True)
    pilot_df = pd.concat(pilot_parts, ignore_index=True)
    assert len(smoke_df) == 24 and len(pilot_df) == 300, (len(smoke_df), len(pilot_df))
    assert not set(smoke_df.paper_id) & set(pilot_df.paper_id), 'smoke/pilot overlap'

    # stability subset: strategy x pilot-novelty-tercile x accepted, largest remainder
    q = pilot_df.novelty_score.quantile([1 / 3, 2 / 3]).values
    pilot_df['novelty_tercile'] = np.select(
        [pilot_df.novelty_score <= q[0], pilot_df.novelty_score <= q[1]], ['n1', 'n2'], 'n3')
    pilot_df['stab_cell'] = pilot_df.strategy + '|' + pilot_df.novelty_tercile + '|' + \
        pilot_df.accepted.map({True: 'acc', False: 'rej'})
    stab_counts = pilot_df.groupby('stab_cell').size().to_dict()
    stab_alloc = largest_remainder(STABILITY_N, stab_counts)
    stab_ids = []
    for cell, n in sorted(stab_alloc.items()):
        pool = sorted(pilot_df.loc[pilot_df.stab_cell == cell, 'paper_id'])
        rng = np.random.default_rng(derive_seed(GLOBAL_SEED, 'stability_subset', cell))
        stab_ids += list(np.array(pool)[rng.permutation(len(pool))[:n]])
    pilot_df['in_stability_subset'] = pilot_df.paper_id.isin(stab_ids)
    assert pilot_df.in_stability_subset.sum() == STABILITY_N

    # task files
    regimes, arms = ('A', 'B'), ('blind', 'prestige_high', 'prestige_low')
    smoke_tasks = [t for _, r in smoke_df.iterrows()
                   for t in make_tasks_for_paper(r, prereg, 'smoke', regimes, arms, panel=1)]
    c_tasks, c_exclusions = make_c_replay_tasks(smoke_df, args.source_outputs_root)
    smoke_tasks += c_tasks
    pilot_tasks = [t for _, r in pilot_df.iterrows()
                   for t in make_tasks_for_paper(r, prereg, 'pilot', regimes, arms, panel=1)]
    stab_rows = pilot_df[pilot_df.in_stability_subset]
    stability_tasks = [t for _, r in stab_rows.iterrows()
                       for t in make_tasks_for_paper(r, prereg, 'stability', regimes,
                                                     ('blind',), panel=2)]
    assert len(pilot_tasks) == 5400 and len(stability_tasks) == 360, \
        (len(pilot_tasks), len(stability_tasks))
    assert len([t for t in smoke_tasks if t['regime'] != 'C']) == 432

    # prestige manipulation check: arm prompts differ only in the institution string
    diff_fail = 0
    hi = prereg['prestige_intervention']['institution_high']
    lo = prereg['prestige_intervention']['institution_low']
    by_key = {}
    for t in pilot_tasks + smoke_tasks:
        if t['regime'] in ('A', 'B') and t['arm'] != 'blind':
            by_key.setdefault((t['paper_id'], t['regime'], t['slot']), {})[t['arm']] = t['prompt']
    for key, pair in by_key.items():
        if pair['prestige_high'].replace(hi, '@INST@') != pair['prestige_low'].replace(lo, '@INST@'):
            diff_fail += 1
    assert diff_fail == 0, f'{diff_fail} prestige pairs differ outside the frozen block'

    # persist corpus
    for name, frame in (('eligibility', elig.drop(columns=['abstract', 'orig_justifications',
                                                           'slot_expertise'], errors='ignore')),
                        ('smoke_corpus', smoke_df), ('pilot_corpus', pilot_df)):
        _artifact_value, _artifact_path = frame, f'{CORPUS_DIR}/{name}.parquet'
        assert_write_allowed(_artifact_path)
        write_parquet_atomic(_artifact_path, _artifact_value, index=False)
    for name, tasks in (('smoke', smoke_tasks), ('pilot', pilot_tasks),
                        ('stability', stability_tasks)):
        _artifact_value, _artifact_path = pd.DataFrame(sorted(tasks, key=lambda t: t['task_uid'])), f'{CORPUS_DIR}/replay_tasks_{name}.parquet'
        assert_write_allowed(_artifact_path)
        write_parquet_atomic(_artifact_path, _artifact_value, index=False)
    pd.DataFrame(c_exclusions or [{'paper_id': None, 'regime': None, 'slot': None,
                                   'reviewer_id': None, 'reason': 'none', 'section': None}]) \
        .to_csv(assert_write_allowed(f'{CORPUS_DIR}/leakage_exclusions.csv'), index=False)
    bal = pilot_df.groupby(['source_seed', 'final_cell']).agg(
        n=('paper_id', 'size'), eligible=('cell_eligible', 'first'),
        inclusion_prob=('inclusion_prob', 'first')).reset_index()
    bal.to_csv(assert_write_allowed(f'{CORPUS_DIR}/stratification_balance.csv'), index=False)

    w = pilot_df.ipw_weight
    manifest = {
        'corpus_version': CORPUS_VERSION,
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'global_seed': GLOBAL_SEED,
        'prereg_v1_sha256': sha256_file(PREREG_V1),
        'source_experiment_ids': SOURCE_IDS,
        'eligibility_funnel': funnel.to_dict('records'),
        'n_eligible': int(elig.eligible.sum()),
        'n_smoke': len(smoke_df), 'n_pilot': len(pilot_df), 'n_stability': int(STABILITY_N),
        'task_counts': {'smoke': len(smoke_tasks), 'smoke_c_replay': len(c_tasks),
                        'pilot': len(pilot_tasks), 'stability': len(stability_tasks)},
        'c_replay_leakage_exclusions': len(c_exclusions),
        'pilot_effective_sample_size': float((w.sum() ** 2) / (w ** 2).sum()),
        'prestige_pair_diff_failures': diff_fail,
        'file_sha256': {os.path.basename(p): sha256_file(p)
                        for p in sorted(glob.glob(f'{CORPUS_DIR}/*.parquet'))},
    }
    _artifact_value, _artifact_path = manifest, f'{CORPUS_DIR}/corpus_manifest.json'
    assert_write_allowed(_artifact_path)
    write_json_atomic(_artifact_path, _artifact_value, indent=2, default=str, trailing_newline=False, streaming=True)
    logger.info(f"extract complete: {manifest['task_counts']}, "
                f"ESS={manifest['pilot_effective_sample_size']:.1f}")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def freeze_environment_report(vllm_url: str) -> dict:
    """Record actual public inputs without claiming historical server identity."""
    import importlib.metadata
    return {'model': MODEL_NAME, 'endpoint': vllm_url,
            'client_versions': {name: importlib.metadata.version(name)
                                for name in ('openai', 'transformers')},
            'study_config_sha256': sha256_file(PREREG_V1),
            'server_precision_and_revision': 'Record the actual server launch command separately.'}


def cmd_generate(args):
    assert_safe_vllm_url(args.vllm_url)
    experiment_id = STAGES[args.stage]
    docs_dir = assert_write_allowed(f'outputs/docs/{experiment_id}')
    log_dir = assert_write_allowed(f'outputs/logs/{experiment_id}')
    chunk_dir = os.path.join(docs_dir, 'review_chunks')
    for d in (docs_dir, log_dir, chunk_dir):
        os.makedirs(d, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, 'generate.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(fh)
    with open(os.path.join(log_dir, 'command.txt'), 'a') as f:
        f.write(' '.join(sys.argv) + '\n')

    corpus_manifest = json.load(open(f'{CORPUS_DIR}/corpus_manifest.json'))
    if corpus_manifest['prereg_v1_sha256'] != sha256_file(PREREG_V1):
        raise ValueError('Study configuration differs from extracted corpus')
    tasks_path = f'{CORPUS_DIR}/replay_tasks_{args.stage}.parquet'
    recorded = corpus_manifest['file_sha256'].get(os.path.basename(tasks_path))
    actual = sha256_file(tasks_path)
    assert recorded == actual, f'task file hash mismatch: {actual} != {recorded}'
    if 'chunk_size' in corpus_manifest and corpus_manifest['chunk_size'] != CHUNK_SIZE:
        sys.exit(f"[guard] REFUSED: chunk_size {CHUNK_SIZE} differs from the frozen "
                 f"manifest value {corpus_manifest['chunk_size']} — changing chunking "
                 "after preregistration would silently change request seeds.")
    sort_col = STUDIES[ACTIVE_STUDY]['task_sort_column']
    tasks = pd.read_parquet(tasks_path).sort_values(sort_col).reset_index(drop=True)

    env_report = freeze_environment_report(args.vllm_url)
    resolved_config = {
        'experiment_id': experiment_id, 'stage': args.stage, 'model': MODEL_NAME,
        'revision': MODEL_REVISION, 'vllm_url': args.vllm_url, 'global_seed': GLOBAL_SEED,
        'corpus_version': CORPUS_VERSION, 'temperature': 0.7,
        'max_tokens': 'SIMULATION_CONFIG llm.max_tokens default (matches live reviews)',
        'thinking_mode': True, 'chunk_size': CHUNK_SIZE, 'n_tasks': len(tasks),
        'batch_size': args.batch_size,
        'corpus_manifest_sha256': sha256_file(f'{CORPUS_DIR}/corpus_manifest.json'),
        'prereg_v1_sha256': corpus_manifest['prereg_v1_sha256'],
        'source_experiment_ids': corpus_manifest.get('source_experiment_ids', []),
        'freeze_environment_report': env_report,
    }
    write_run_manifest(docs_dir, 'running', args=args, resolved_config=resolved_config)

    from utopia.models.models import VLLMServerModel
    llm = VLLMServerModel(model_name=MODEL_NAME, base_url=args.vllm_url,
                          max_concurrent_requests=args.batch_size, run_seed=GLOBAL_SEED)
    response_format = review_response_format()

    # monoculture keeps its original seed context byte-identically; the network
    # study derives per-stage contexts so stability gets fresh request seeds
    if ACTIVE_STUDY == 'reviewer_monoculture':
        chunk_ctx = lambda k: ('review_replay', CORPUS_VERSION, k)          # noqa: E731
        sweep_ctx = ('review_replay', CORPUS_VERSION, 'sweep', 0)
        meta_columns = None
    else:
        chunk_ctx = lambda k: ('nrb', CORPUS_VERSION, args.stage, k)        # noqa: E731
        sweep_ctx = ('nrb', CORPUS_VERSION, args.stage, 'sweep', 0)
        meta_columns = [c for c in tasks.columns if c != 'prompt']

    n_chunks = int(np.ceil(len(tasks) / CHUNK_SIZE))
    t_start = time.time()
    for k in range(n_chunks):
        chunk_path = os.path.join(chunk_dir, f'chunk_{k:04d}.parquet')
        chunk = tasks.iloc[k * CHUNK_SIZE:(k + 1) * CHUNK_SIZE].reset_index(drop=True)
        if os.path.exists(chunk_path):
            done = pd.read_parquet(chunk_path)
            if len(done) == len(chunk):
                logger.info(f'chunk {k}: already complete, skipping')
                continue
        rows = run_chunk(llm, chunk, response_format, seed_ctx=chunk_ctx(k),
                         meta_columns=meta_columns, chunk_id=k)
        _artifact_value, _artifact_path = pd.DataFrame(rows), chunk_path
        assert_write_allowed(_artifact_path)
        write_parquet_atomic(_artifact_path, _artifact_value, index=False)
        logger.info(f'chunk {k + 1}/{n_chunks} done '
                    f'({sum(r["success"] for r in rows)}/{len(rows)} ok, '
                    f'{time.time() - t_start:.0f}s elapsed)')

    # tail sweep: one re-attempt for residual failures under a dedicated seed_ctx
    all_rows = pd.concat([pd.read_parquet(p) for p in
                          sorted(glob.glob(os.path.join(chunk_dir, 'chunk_*.parquet')))],
                         ignore_index=True)
    failed = all_rows[~all_rows.success]
    if len(failed):
        logger.info(f'tail sweep: retrying {len(failed)} failed tasks')
        sweep_tasks = tasks[tasks.task_uid.isin(failed.task_uid)].reset_index(drop=True)
        sweep_rows = run_chunk(llm, sweep_tasks, response_format, seed_ctx=sweep_ctx,
                               meta_columns=meta_columns, chunk_id=-1)
        sweep_df = pd.DataFrame(sweep_rows)
        _artifact_value, _artifact_path = sweep_df, os.path.join(chunk_dir, 'sweep_0000.parquet')
        assert_write_allowed(_artifact_path)
        write_parquet_atomic(_artifact_path, _artifact_value, index=False)
        fixed = sweep_df[sweep_df.success].set_index('task_uid')
        all_rows = all_rows.set_index('task_uid')
        all_rows.loc[fixed.index, fixed.columns] = fixed
        all_rows = all_rows.reset_index()

    out_path = os.path.join(docs_dir, 'reviews.parquet')
    _artifact_value, _artifact_path = all_rows, out_path
    assert_write_allowed(_artifact_path)
    write_parquet_atomic(_artifact_path, _artifact_value, index=False)
    trunc_rate = float('nan')
    n_success = int(all_rows.success.sum())
    extra = {
        'llm_call_stats': llm.call_stats,
        'n_tasks': len(tasks), 'n_success': n_success,
        'n_failed_final': int(len(all_rows) - n_success),
        'raw_score_noninteger_fraction': float(
            (all_rows.loc[all_rows.success, 'score_raw'] % 1 != 0).mean()) if n_success else None,
        'wall_clock_seconds': time.time() - t_start,
        'reviews_parquet': out_path,
    }
    write_run_manifest(docs_dir, 'complete', args=args, resolved_config=resolved_config,
                       extra=extra)
    logger.info(f'{experiment_id}: {n_success}/{len(tasks)} reviews complete '
                f'({extra["wall_clock_seconds"]:.0f}s); stats={llm.call_stats}')


def run_chunk(llm, chunk: pd.DataFrame, response_format: dict, seed_ctx: tuple,
              meta_columns: list = None, chunk_id: int = None) -> list:
    t0 = time.time()
    results = llm.generate_batch(list(chunk.prompt), response_format=response_format,
                                 temperature=0.7,
                                 desc=f'{len(chunk)} replay reviews {seed_ctx}',
                                 seed_ctx=seed_ctx)
    elapsed = time.time() - t0
    rows = []
    for idx, ((resp, _hist), (_, t)) in enumerate(zip(results, chunk.iterrows())):
        ok = isinstance(resp, dict) and 'overall_score' in resp and 'justification' in resp
        score_raw = float(resp['overall_score']) if ok else np.nan
        if meta_columns is None:   # original monoculture row schema, unchanged
            row = {
                'task_uid': t.task_uid, 'stage': t.stage, 'paper_id': t.paper_id,
                'source_seed': t.source_seed, 'year': t.year, 'conference': t.conference,
                'regime': t.regime, 'arm': t.arm, 'panel': t.panel, 'slot': t.slot,
                'policy_key': t.policy_key, 'orig_reviewer_id': t.orig_reviewer_id,
            }
        else:
            row = {c: t[c] for c in meta_columns}
            row['chunk_id'] = chunk_id
        row.update({
            'success': bool(ok),
            'score_raw': score_raw,
            'score_int': int(score_raw) if ok else -1,   # mirrors utopia/simulation.py int-cast
            'justification': (resp.get('justification') or '') if ok else '',
            'justification_len': len(resp.get('justification') or '') if ok else 0,
            'request_seed_attempt0': derive_seed(GLOBAL_SEED, *seed_ctx, idx, 0),
            'chunk_elapsed_s': elapsed,
        })
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Status / analyze
# --------------------------------------------------------------------------

def cmd_status(args):
    for stage, experiment_id in STAGES.items():
        chunk_dir = f'outputs/docs/{experiment_id}/review_chunks'
        tasks_path = f'{CORPUS_DIR}/replay_tasks_{stage}.parquet'
        n_tasks = len(pd.read_parquet(tasks_path, columns=['task_uid'])) \
            if os.path.exists(tasks_path) else 0
        done = 0
        for p in glob.glob(os.path.join(chunk_dir, 'chunk_*.parquet')):
            done += len(pd.read_parquet(p, columns=['task_uid']))
        print(f'{stage:10s} {experiment_id}: {done}/{n_tasks} task-results on disk')


def cmd_analyze(args):
    if ACTIVE_STUDY in ('network_proximity', 'network_proximity_confirmatory'):
        from utopia.analysis.network_review_bias_analysis import run_analysis
        out_name = ('network_review_bias_confirmatory_final'
                    if ACTIVE_STUDY == 'network_proximity_confirmatory'
                    else 'network_review_bias_final')
        run_analysis(stage=args.stage, corpus_dir=CORPUS_DIR, stages=STAGES,
                     prereg_path=PREREG_V1, n_boot=args.n_boot, out_name=out_name)
    else:
        from utopia.analysis.review_regime_analysis import run_analysis
        run_analysis(stage=args.stage, corpus_dir=CORPUS_DIR, stages=STAGES,
                     prereg_path=PREREG_V1, n_boot=args.n_boot)


def dispatch_extract(args):
    if ACTIVE_STUDY in ('network_proximity', 'network_proximity_confirmatory'):
        cmd_extract_nrb(args)
    else:
        cmd_extract(args)


def main():
    global MODEL_NAME, MODEL_REVISION, GLOBAL_SEED, PREREG_V1, CORPUS_DIR, SOURCE_IDS, SOURCE_SEEDS, SMOKE_QUOTAS, PILOT_QUOTA_PER_SEED, STABILITY_N, STAGES, ALLOWED_WRITE_PREFIXES
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', choices=list(STUDIES), default='reviewer_monoculture',
                    help='which frozen study configuration to activate')
    ap.add_argument('--config', type=Path)
    ap.add_argument('--model', default=MODEL_NAME)
    ap.add_argument('--model-revision', help='Immutable model commit; known Qwen3 revisions are selected automatically.')
    ap.add_argument('--seed', type=int, default=GLOBAL_SEED)
    ap.add_argument('--run-name', default='public')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_ex = sub.add_parser('extract')
    p_ex.add_argument('--source-run', nargs='+', help='Fresh source simulation experiment IDs.')
    p_ex.add_argument('--smoke-per-source', type=int, default=5)
    p_ex.add_argument('--pilot-per-source', type=int, default=60)
    p_ex.add_argument('--stability-n', type=int, default=60)
    p_ex.add_argument('--source_outputs_root', default='outputs',
                      help='read-only root holding the E3.5 artifacts (monoculture)')
    p_ex.add_argument('--source_dirs', nargs='+', default=[],
                      help='network_proximity: legacy source-run dirs holding '
                           'checkpoint_year_*.json with a preferential_attachment block')
    p_ex.add_argument('--through_year', type=int, default=None,
                      help='network_proximity: use checkpoints up to this source year')
    p_ex.add_argument('--structural_audit_only', action='store_true',
                      help='network_proximity: write structural_audit.json and stop')
    p_ex.add_argument('--reuse_cap', type=int, default=10,
                      help='network_proximity: frozen reviewer-reuse cap (10; fallback 15)')
    p_ex.set_defaults(func=dispatch_extract)

    p_gen = sub.add_parser('generate')
    p_gen.add_argument('--stage', required=True)
    p_gen.add_argument('--vllm_url', default='http://localhost:8000/v1')
    p_gen.add_argument('--batch_size', type=int, default=32)
    p_gen.set_defaults(func=cmd_generate)

    p_an = sub.add_parser('analyze')
    p_an.add_argument('--stage', required=True)
    p_an.add_argument('--n_boot', type=int, default=10000)
    p_an.set_defaults(func=cmd_analyze)

    p_st = sub.add_parser('status')
    p_st.set_defaults(func=cmd_status)

    args = ap.parse_args()
    _activate_study(args.study)
    from utopia.models.request_audit import PINNED_MODELS
    MODEL_REVISION = args.model_revision or PINNED_MODELS.get(args.model)
    if not MODEL_REVISION:
        ap.error('Unknown model: provide --model-revision')
    MODEL_NAME, GLOBAL_SEED = args.model, args.seed
    if args.config:
        PREREG_V1 = str(args.config.resolve())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        ap.error('--run-name must contain only letters, digits, underscores and hyphens')
    CORPUS_DIR = os.path.join('data', f'{args.study}_{args.run_name}')
    STAGES = {stage: f'review_replay_{args.study}_{args.run_name}_{stage}' for stage in STAGES}
    ALLOWED_WRITE_PREFIXES = (CORPUS_DIR, 'outputs/docs/review_replay_', 'outputs/logs/review_replay_')
    if args.cmd == 'extract' and args.study == 'reviewer_monoculture':
        if not args.source_run:
            ap.error('extract requires --source-run with fresh simulation IDs')
        SOURCE_IDS = args.source_run
        SOURCE_SEEDS = []
        for source_id in SOURCE_IDS:
            manifest = json.load(open(Path(args.source_outputs_root) / 'docs' / source_id / 'run_manifest.json'))
            if manifest.get('status') != 'complete':
                raise ValueError(f'Incomplete source simulation: {source_id}')
            SOURCE_SEEDS.append(manifest['run_seed'])
        if len(set(SOURCE_SEEDS)) != len(SOURCE_SEEDS):
            raise ValueError('Each source run must have a different seed')
        SMOKE_QUOTAS = dict.fromkeys(SOURCE_SEEDS, args.smoke_per_source)
        PILOT_QUOTA_PER_SEED, STABILITY_N = args.pilot_per_source, args.stability_n
    if args.cmd == 'generate':
        assert args.stage in STAGES, f'unknown stage {args.stage} for study {args.study}'
    if args.cmd == 'analyze':
        assert args.stage in list(STAGES) + ['final'], f'unknown stage {args.stage}'
    if args.cmd != 'generate':
        args.func(args)
        return
    from utopia.models.request_audit import request_audit_scope
    from utopia.runtime.provenance import mark_failed_run_manifest
    directory = Path('outputs/docs') / STAGES[args.stage]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        settings = previous.get('resolved_config', {})
        if (settings.get('model') != MODEL_NAME or settings.get('revision') != MODEL_REVISION
                or settings.get('global_seed') != GLOBAL_SEED):
            raise ValueError('Use a new --run-name when changing model, revision, or seed')
        if previous.get('status') == 'complete':
            logger.info('Replay is already complete: %s', directory)
            return
    audit_path = directory.resolve() / f'llm_request_audit_{time.time_ns()}.jsonl'
    try:
        with request_audit_scope(audit_path, model_name=MODEL_NAME, revision=MODEL_REVISION):
            args.func(args)
        write_run_manifest(directory, 'complete', extra={'request_audit_path': audit_path.name})
    except BaseException as error:
        mark_failed_run_manifest(directory, 'review_replay', repr(error))
        raise


if __name__ == '__main__':
    main()
