"""Analysis for the scale-expansion experiment (design v1,
outputs/docs/scale_expansion_design/design_v1.md).

Reads the FINAL yearly checkpoint of every scale_* run (plus the
funding_applications_year_<y>.jsonl logs written by --log_funding_applications)
and computes, per run (= one simulated world, the unit of replication):

Primary endpoints
  P1_survival_h        share of founders still ACTIVE at year h (default 3); the full
                       survival curve is stored. Exit = is_active False in the year-t
                       checkpoint (the simulator never deletes agents: "removal" flips
                       is_active, after which the agent stops authoring, applying and
                       paying costs). Falls back to resources > 10 when a year checkpoint
                       is missing.
  P2_resub_review_share share of all review slots spent on resubmission attempts
                       (attempt index >= 1 in a paper's review_history); plus mean
                       attempts per paper (cascade depth + 1).
  P3_award_gini        Gini of cumulative agency awards over all UNIVERSITY founders (0 for
                       the never-funded, including culled agents), from the application
                       logs; plus top-decile award share. Industry researchers never apply
                       to agencies (pay-per-paper income), so they are excluded here and
                       reported through the per-sector survival companions.
  Caveat on P4 levels: citations are generated only by first submissions and spread over
                       the accepted pool, so per-paper citation LEVELS are comparable only
                       between cells with similar submission and pool sizes (e.g. within
                       one k and one resubmission regime); cutting resubmission halves the
                       pool and mechanically doubles per-paper citations.
  P4_followup_share    share of accepted papers (accepted at year <= N - h so the
                       window is fully observed) that receive >= 1 citation from a
                       paper with a disjoint author set within h years; plus the
                       rarefied topic entropy of accepted papers at a common n.

Manipulation checks (settings, not findings): submissions, accepted, active
researchers, university awards per founder (comparable across budget modes),
reviews per paper, per year.

Aggregation: cell mean with seed-bootstrap CI; preregistered pairwise contrasts
(difference of cell means, bootstrap CI, exact-style permutation p over seed
labels), Benjamini-Hochberg over the four primaries within each contrast.

Outputs (--out_dir, default outputs/docs/scale_expansion_analysis/):
  per_run_endpoints.csv, yearly_checks.csv, cell_summary.csv, contrasts.csv,
  report.md, analysis.json

Run:
  python -m utopia.analysis.scale_expansion --num_years 8
  python -m utopia.analysis.scale_expansion --glob "outputs/checkpoints/scale_*" --horizon 3
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.analysis.statistics import bootstrap_mean_ci

from utopia.utils.data_utils import read_json
import argparse
import glob
import gzip
import json
import os
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utopia.analysis.statistics import benjamini_hochberg
from utopia.metrics.tracker import calculate_gini_coefficient

PRIMARY = ['P1_survival_h', 'P2_resub_review_share', 'P3_award_gini', 'P4_followup_share']
SECONDARY = ['P1_survival_final', 'P1_survival_final_university', 'P1_survival_final_industry',
             'P2_attempts_per_paper', 'P3_top_decile_award_share',
             'P4_topic_entropy_rarefied', 'P4_nonself_citations_per_paper',
             # exploratory companions (not preregistered primaries)
             'X1_precarious_share', 'X1_resources_p10', 'X3_award_coverage', 'X3_gini_among_funded',
             # validity of the LLM funding rankings (share of winners from fallback / imputed rankings)
             'V_fallback_winner_share', 'V_imputed_winner_share']
# (treatment, control, label) — fixed before results are read (design v1 branch readings)
CONTRASTS = [
    ('S1R0', 'S0R0', 'S: 2 papers vs 1 under tracking budget'),
    ('S0R1', 'S0R0', 'R: fixed vs tracking budget under 1 paper'),
    ('S1R1', 'S1R0', 'R: fixed vs tracking budget under 2 papers'),
    ('S1R1', 'S0R1', 'S: 2 papers vs 1 under fixed budget'),
    ('S1R1_K1', 'S1R1', 'K: review capacity cap under pressure'),
    ('S1R1_E1', 'S1R1', 'E: standardized review under pressure'),
    ('S1R1_K1E1', 'S1R1', 'K+E under pressure'),
    ('S0R0_E1', 'S0R0', 'E: standardized review without pressure'),
    ('S1R0_slots', 'S1R0', 'sensitivity: fixed acceptance slots'),
    ('S1R1_noresub', 'S1R1', 'counterfactual: resubmission channel cut in the pressure world'),
]
CELL_ORDER = ['S0R0', 'S1R0', 'S0R1', 'S1R1', 'S1R1_K1', 'S1R1_E1', 'S1R1_K1E1', 'S0R0_E1', 'S1R0_slots',
              'S1R1_noresub']


# ---------------------------------------------------------------- loading

def load_active_by_year(run_dir: str, num_years: int) -> Dict[int, Dict[str, bool]]:
    """year -> {agent_id: is_active} from every available yearly checkpoint (agents only)."""
    out = {}
    for year in range(1, num_years + 1):
        base = os.path.join(run_dir, f'checkpoint_year_{year}.json')
        for path, opener in ((base, open), (base + '.gz', gzip.open)):
            if os.path.exists(path):
                with opener(path, 'rt') as f:
                    ck = json.load(f)
                agents = {a['id']: bool(a.get('is_active', True)) for a in ck['ecosystem_data']['agents']
                          if a.get('type') in ('university', 'industry')}
                if agents:  # a checkpoint without researcher records falls back to the resource rule
                    out[year] = agents
                break
    return out


def load_checkpoint(run_dir: str, num_years: int) -> Tuple[dict, int]:
    """Final checkpoint (year num_years, else the latest available); json or .gz."""
    for year in range(num_years, 0, -1):
        base = os.path.join(run_dir, f'checkpoint_year_{year}.json')
        for path, opener in ((base, open), (base + '.gz', gzip.open)):
            if os.path.exists(path):
                return read_json(path), year
    raise FileNotFoundError(f'no checkpoint in {run_dir}')


def parse_experiment_id(exp_id: str) -> dict:
    """scale_<model>_k<k>_budget<mode>_cap<c|none>_<policy>[_float][_tieseed][_slots]...
    _i<I>_n<N>_y<Y>_seed<S>_mixbal  ->  factor levels + canonical cell name."""
    m = re.search(r'_k(\d+)_budget(track|fixed)_cap(none|\d+)_(persona|standardized)', exp_id)
    if not m:
        raise ValueError(f'not a scale experiment id: {exp_id}')
    k, budget, cap, policy = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    slots = '_slots' in exp_id
    noresub = '_noresub' in exp_id
    seed = int(re.search(r'_seed(\d+)', exp_id).group(1))
    name = f"S{1 if k > 1 else 0}R{1 if budget == 'fixed' else 0}"
    suffix = ('K1' if cap != 'none' else '') + ('E1' if policy == 'standardized' else '')
    if suffix:
        name += '_' + suffix
    if slots:
        name += '_slots'
    if noresub:
        name += '_noresub'
    return {'cell': name, 'k': k, 'budget': budget, 'cap': None if cap == 'none' else int(cap),
            'policy': policy, 'slots': slots, 'seed': seed,
            'population': 'default' if '_popdefault_' in exp_id else 'university_only'}


# ---------------------------------------------------------------- per-run endpoints

def _first_year(paper: dict) -> int:
    return paper['review_history'][0]['year'] if paper.get('review_history') else paper['year']


def _accept_year(paper: dict) -> Optional[int]:
    if paper.get('status') != 'accept':
        return None
    for entry in reversed(paper.get('review_history', [])):
        if entry.get('decision') == 'accept':
            return entry['year']
    return paper['year']


def _authors(paper: dict) -> set:
    a = paper['author_id']
    return set(a) if isinstance(a, list) else {a}


def load_award_counts(run_dir: str) -> Optional[Dict[str, int]]:
    """applicant_id -> number of funded applications, from funding_applications_year_*.jsonl."""
    files = sorted(glob.glob(os.path.join(run_dir, 'funding_applications_year_*.jsonl')))
    if not files:
        return None
    counts = Counter()
    for path in files:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('funded'):
                    counts[rec['applicant_id']] += 1
    return dict(counts)


def funding_ranking_validity(run_dir: str) -> Dict[str, Optional[float]]:
    """Share of funding WINNERS decided by a fallback (panel-order) ranking or by an imputed
    tail, from the application logs. A fallback ranking is used when every LLM attempt for a
    program-year fails (e.g. HTTP 400: prompt + 8192 output tokens exceed the 32k context once
    a program receives ~100+ applications); winners from such panels are effectively random.
    Rule of thumb: > 5% fallback winners invalidates the funding endpoints of that run."""
    files = sorted(glob.glob(os.path.join(run_dir, 'funding_applications_year_*.jsonl')))
    if not files:
        return {'V_fallback_winner_share': None, 'V_imputed_winner_share': None, 'V_fallback_years': None}
    winners = fallback = imputed = 0
    years = set()
    for path in files:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get('fallback_ranking'):
                    years.add(rec.get('year'))
                if rec.get('funded'):
                    winners += 1
                    fallback += bool(rec.get('fallback_ranking'))
                    imputed += bool(rec.get('imputed_tail'))
    return {'V_fallback_winner_share': fallback / winners if winners else None,
            'V_imputed_winner_share': imputed / winners if winners else None,
            'V_fallback_years': ','.join(str(y) for y in sorted(y for y in years if y is not None))}


def rarefied_topic_entropy(topic_lists: List[List[str]], n: int, n_draws: int = 200,
                           seed: int = 0) -> Optional[float]:
    """Mean normalized entropy of the (multi-label) topic distribution over n_draws
    random subsamples of n papers. Normalized by log(#distinct topics in the full set)
    so values are comparable across cells at the same n."""
    if n <= 0 or len(topic_lists) < n:
        return None
    all_topics = sorted({t for ts in topic_lists for t in ts})
    if len(all_topics) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = np.arange(len(topic_lists))
    vals = []
    for _ in range(n_draws):
        pick = rng.choice(idx, size=n, replace=False)
        counts = Counter(t for i in pick for t in topic_lists[i])
        p = np.array(list(counts.values()), dtype=float)
        p /= p.sum()
        vals.append(float(-(p * np.log(p)).sum() / np.log(len(all_topics))))
    return float(np.mean(vals))


def compute_run_endpoints(run_dir: str, num_years: int, horizon: int = 3, *, metadata=None) -> dict:
    ck, last_year = load_checkpoint(run_dir, num_years)
    papers = ck['paper_tracker']['papers']
    tracker = ck['agent_tracker']['resources']          # agent -> [{year, resources}]
    yearly = ck.get('yearly_results', [])
    citations = ck.get('citation_tracker', {}).get('citations', {})   # cited -> [citing]
    exp_id = os.path.basename(os.path.normpath(run_dir))
    out = {'experiment_id': exp_id, 'run_dir': run_dir, 'years_completed': last_year}
    out.update(metadata if metadata is not None else parse_experiment_id(exp_id))

    # ---- P1 survival (founders = recorded at year 0; alive = is_active at year t) ----
    founders = [aid for aid, recs in tracker.items() if any(r['year'] == 0 for r in recs)]
    n0 = len(founders)
    # sector of every founder (agents are never deleted, so the final checkpoint lists all)
    sector = {a['id']: a.get('type') for a in ck['ecosystem_data']['agents']}
    founders_by_sector = {sec: [aid for aid in founders if sector.get(aid) == sec]
                          for sec in ('university', 'industry')}
    n0_university = len(founders_by_sector['university']) or n0
    active_by_year = load_active_by_year(run_dir, last_year)
    survival, active_count = {}, {}
    active_count_sector = {sec: {} for sec in founders_by_sector}
    for t in range(1, last_year + 1):
        if t in active_by_year:
            def is_alive(aid, t=t):
                return active_by_year[t].get(aid, False)
        else:  # fallback: above the cull line at year t
            def is_alive(aid, t=t):
                return any(r['year'] == t and r['resources'] > 10 for r in tracker[aid])
        alive = sum(1 for aid in founders if is_alive(aid))
        survival[t] = alive / n0 if n0 else None
        active_count[t] = alive
        for sec, ids in founders_by_sector.items():
            active_count_sector[sec][t] = sum(1 for aid in ids if is_alive(aid))
    # per-sector final survival (industry only exists in the default population)
    final_active = active_by_year.get(last_year, {})
    for sec, ids in founders_by_sector.items():
        out[f'P1_survival_final_{sec}'] = (sum(1 for aid in ids if final_active.get(aid, False)) / len(ids)
                                           if ids and final_active else None)
    out['n_founders_university'] = len(founders_by_sector['university'])
    out['n_founders_industry'] = len(founders_by_sector['industry'])
    out['n_founders'] = n0
    out['survival_curve'] = survival
    out['P1_survival_h'] = survival.get(min(horizon, last_year))
    out['P1_survival_final'] = survival.get(last_year)
    # Exploratory companion: with initial funding 100 and annual cost 10 the cull line (10)
    # is rarely reached within 8 years, so also report the precarious share (final-year
    # resources < 30, i.e. < 2 years of runway) and the 10th percentile of resources.
    final_res = [recs[-1]['resources'] for aid, recs in tracker.items() if aid in founders and recs]
    out['X1_precarious_share'] = float(np.mean([r < 30 for r in final_res])) if final_res else None
    out['X1_resources_p10'] = float(np.percentile(final_res, 10)) if final_res else None

    # ---- P2 recycling burden ----
    total_reviews = resub_reviews = 0
    attempts = []
    reviews_by_year, attempts_by_year = Counter(), Counter()
    for p in papers:
        hist = p.get('review_history', [])
        if not hist:
            continue
        attempts.append(len(hist))
        for j, entry in enumerate(hist):
            n_rev = len(entry.get('reviews', []) or [])
            total_reviews += n_rev
            reviews_by_year[entry['year']] += n_rev
            attempts_by_year[entry['year']] += 1
            if j >= 1:
                resub_reviews += n_rev
    out['P2_resub_review_share'] = resub_reviews / total_reviews if total_reviews else None
    out['P2_attempts_per_paper'] = float(np.mean(attempts)) if attempts else None
    out['total_reviews'] = total_reviews

    # ---- P3 concentration of agency awards over all founders ----
    awards = load_award_counts(run_dir)
    out.update(funding_ranking_validity(run_dir))
    if awards is None:  # fallback: survivors' funding_success_history (culled agents missing)
        awards = {}
        for a in ck['ecosystem_data']['agents']:
            hist = a.get('funding_success_history') or {}
            awards[a['id']] = sum(len(v) for v in hist.values())
        out['P3_award_source'] = 'survivor_history_only'
    else:
        out['P3_award_source'] = 'application_log'
    per_founder = np.array([awards.get(aid, 0) for aid in founders_by_sector['university'] or founders], dtype=float)
    out['P3_award_gini'] = float(calculate_gini_coefficient(list(per_founder))) if per_founder.sum() > 0 else 0.0
    if per_founder.sum() > 0:
        top = int(np.ceil(0.1 * len(per_founder)))
        out['P3_top_decile_award_share'] = float(np.sort(per_founder)[::-1][:top].sum() / per_founder.sum())
    else:
        out['P3_top_decile_award_share'] = None
    out['P3_total_awards'] = float(per_founder.sum())
    # Companions: coverage (share of founders ever funded) and Gini among the funded only,
    # because a fixed budget lowers the award count and raises the all-founder Gini mechanically.
    out['X3_award_coverage'] = float((per_founder > 0).mean()) if len(per_founder) else None
    funded = per_founder[per_founder > 0]
    out['X3_gini_among_funded'] = float(calculate_gini_coefficient(list(funded))) if len(funded) > 1 else None
    # Full award-count distribution (awards -> number of founders) so figures can draw the
    # never-funded share and Lorenz curves without re-reading the application logs.
    out['X3_award_histogram'] = json.dumps({int(k): int(v) for k, v in sorted(Counter(per_founder.astype(int)).items())})

    # ---- P4 continuation & diversity ----
    by_id = {p['id']: p for p in papers}
    accepted = [p for p in papers if p.get('status') == 'accept']
    observed = [p for p in accepted if _accept_year(p) is not None and _accept_year(p) <= last_year - horizon]
    followed, nonself_counts = 0, []
    for p in observed:
        ay = _accept_year(p)
        n_nonself = 0
        for citing_id in citations.get(p['id'], []):
            citing = by_id.get(citing_id)
            if citing is None:
                continue
            if _first_year(citing) <= ay + horizon and not (_authors(citing) & _authors(p)):
                n_nonself += 1
        followed += 1 if n_nonself > 0 else 0
        nonself_counts.append(n_nonself)
    out['P4_followup_share'] = followed / len(observed) if observed else None
    out['P4_nonself_citations_per_paper'] = float(np.mean(nonself_counts)) if nonself_counts else None
    out['P4_n_accepted_observed'] = len(observed)
    out['n_accepted'] = len(accepted)
    out['accepted_topic_lists'] = [list(p.get('topics', [])) for p in accepted]

    # ---- manipulation checks per year ----
    checks = []
    for yr in yearly:
        y = yr.get('year')
        eco = yr.get('ecosystem_metrics', {})
        fs = eco.get('funding_success_by_type', {})
        fb = yr.get('funding_budget')
        # Comparable across budget modes: actual university awards per founder (f-scale).
        # (funding_success_by_type.total_applications counts applicant AGENTS, funding_budget
        # .total_applications counts program-level applications; neither is used as denominator.)
        awards_per_founder = fs.get('university', 0) / n0_university if n0_university else None
        slots_per_founder = fb['total_slots'] / n0_university if fb and n0_university else None
        fbi = yr.get('funding_budget_industry')
        dec = yr.get('decisions', {})
        n_dec = dec.get('total_accepted', 0) + dec.get('total_rejected', 0)
        # realised reviews per submission event this year (from review_history; the
        # peer_review.reviews_conducted counter counts reviewed papers, not reviews)
        rpp = reviews_by_year[y] / attempts_by_year[y] if attempts_by_year[y] else None
        checks.append({'year': y,
                       'submissions': yr.get('paper_submission', {}).get('num_papers_submitted'),
                       'accepted': dec.get('total_accepted'),
                       'active_researchers': active_count.get(y),  # is_active founders (eco.active_agents counts can_author, always N0)
                       'active_researchers_university': active_count_sector['university'].get(y),
                       'active_researchers_industry': active_count_sector['industry'].get(y),
                       'awards': fs.get('university'),
                       'awards_per_founder': awards_per_founder,
                       'budget_slots_per_founder': slots_per_founder,
                       'industry_pool_paid': fbi.get('paid') if fbi else None,
                       'reviews_per_paper': rpp,
                       'funding_gini_active': (eco.get('resource_distribution') or {}).get('gini')})
    out['yearly_checks'] = checks
    return out


# ---------------------------------------------------------------- aggregation



def permutation_pvalue(a, b, n_perm=5000, seed=0) -> Optional[float]:
    """Two-sided permutation test of the mean difference over world (seed) labels.
    Exhaustive when the number of splits is small, else Monte Carlo."""
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.array(a + b, dtype=float)
    obs = abs(np.mean(a) - np.mean(b))
    n_a, n = len(a), len(pooled)
    idx = range(n)
    all_splits = [c for c in combinations(idx, n_a)]
    if len(all_splits) <= n_perm:
        splits = all_splits
    else:
        rng = np.random.default_rng(seed)
        splits = [tuple(rng.choice(n, size=n_a, replace=False)) for _ in range(n_perm)]
    count = 0
    for s in splits:
        mask = np.zeros(n, dtype=bool); mask[list(s)] = True
        if abs(pooled[mask].mean() - pooled[~mask].mean()) >= obs - 1e-12:
            count += 1
    return count / len(splits)


def aggregate(runs: List[dict], rarefaction_min_n: int = 20):
    # common n for rarefied entropy
    counts = [r['n_accepted'] for r in runs]
    n_common = min(counts) if counts else 0
    n_common = n_common if n_common >= rarefaction_min_n else 0
    for r in runs:
        r['P4_topic_entropy_rarefied'] = (rarefied_topic_entropy(r['accepted_topic_lists'], n_common, seed=r['seed'])
                                          if n_common else None)
        r['rarefaction_n'] = n_common
    metrics = PRIMARY + SECONDARY
    per_run = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ('survival_curve', 'yearly_checks', 'accepted_topic_lists')} for r in runs])
    yearly = pd.DataFrame([{'cell': r['cell'], 'seed': r['seed'], **c} for r in runs for c in r['yearly_checks']])

    rows = []
    for cell, g in per_run.groupby('cell'):
        row = {'cell': cell, 'n_seeds': len(g), 'seeds': sorted(g['seed'].tolist())}
        for m in metrics:
            mean, lo, hi = bootstrap_mean_ci(g[m].tolist())
            row[f'{m}_mean'], row[f'{m}_ci_lo'], row[f'{m}_ci_hi'] = mean, lo, hi
        rows.append(row)
    cell_summary = pd.DataFrame(rows)
    cell_summary['order'] = cell_summary['cell'].map({c: i for i, c in enumerate(CELL_ORDER)})
    cell_summary = cell_summary.sort_values('order').drop(columns='order')

    crows = []
    by_cell = {c: g for c, g in per_run.groupby('cell')}
    for treat, ctrl, label in CONTRASTS:
        if treat not in by_cell or ctrl not in by_cell:
            continue
        pvals = {}
        for m in metrics:
            a, b = by_cell[treat][m].tolist(), by_cell[ctrl][m].tolist()
            a_v = [x for x in a if x is not None]; b_v = [x for x in b if x is not None]
            if not a_v or not b_v:
                continue
            diff = float(np.mean(a_v) - np.mean(b_v))
            rng = np.random.default_rng(1)
            boots = [np.mean(rng.choice(a_v, len(a_v))) - np.mean(rng.choice(b_v, len(b_v)))
                     for _ in range(2000)] if len(a_v) > 1 and len(b_v) > 1 else []
            p = permutation_pvalue(a_v, b_v)
            crow = {'treatment': treat, 'control': ctrl, 'label': label, 'metric': m,
                    'primary': m in PRIMARY, 'diff': diff,
                    'ci_lo': float(np.percentile(boots, 2.5)) if boots else None,
                    'ci_hi': float(np.percentile(boots, 97.5)) if boots else None,
                    'p_perm': p, 'n_treat': len(a_v), 'n_ctrl': len(b_v)}
            crows.append(crow)
            if m in PRIMARY and p is not None:
                pvals[m] = p
        if pvals:
            adj = benjamini_hochberg(list(pvals.values()))
            adj_map = dict(zip(pvals.keys(), adj))
            for crow in crows:
                if crow['treatment'] == treat and crow['control'] == ctrl and crow['metric'] in adj_map:
                    crow['p_bh'] = adj_map[crow['metric']]
    contrasts = pd.DataFrame(crows)
    return per_run, yearly, cell_summary, contrasts


def write_report(out_dir: str, per_run, yearly, cell_summary, contrasts, horizon: int):
    os.makedirs(out_dir, exist_ok=True)
    per_run.to_csv(os.path.join(out_dir, 'per_run_endpoints.csv'), index=False)
    yearly.to_csv(os.path.join(out_dir, 'yearly_checks.csv'), index=False)
    cell_summary.to_csv(os.path.join(out_dir, 'cell_summary.csv'), index=False)
    contrasts.to_csv(os.path.join(out_dir, 'contrasts.csv'), index=False)
    lines = ['# Scale-expansion analysis', '',
             f'Runs: {len(per_run)}  |  cells: {per_run["cell"].nunique()}  |  horizon h = {horizon} years', '',
             '## Cell means (seed-bootstrap 95% CI)', '',
             '| cell | n | ' + ' | '.join(PRIMARY) + ' |', '|---|---|' + '---|' * len(PRIMARY)]
    for _, r in cell_summary.iterrows():
        cells = []
        for m in PRIMARY:
            mean, lo, hi = r[f'{m}_mean'], r[f'{m}_ci_lo'], r[f'{m}_ci_hi']
            cells.append('n/a' if mean is None or pd.isna(mean) else
                         (f'{mean:.3f}' if lo is None or pd.isna(lo) else f'{mean:.3f} [{lo:.3f}, {hi:.3f}]'))
        lines.append(f'| {r["cell"]} | {r["n_seeds"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Preregistered contrasts (primary endpoints; BH within contrast)', '',
              '| contrast | metric | diff | 95% CI | p_perm | p_BH |', '|---|---|---|---|---|---|']
    if len(contrasts):
        for _, r in contrasts[contrasts['primary']].iterrows():
            ci = 'n/a' if r['ci_lo'] is None or pd.isna(r['ci_lo']) else f'[{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]'
            p = 'n/a' if r['p_perm'] is None or pd.isna(r['p_perm']) else f'{r["p_perm"]:.3f}'
            pbh = 'n/a' if 'p_bh' not in r or r.get('p_bh') is None or pd.isna(r.get('p_bh')) else f'{r["p_bh"]:.3f}'
            lines.append(f'| {r["treatment"]} vs {r["control"]} | {r["metric"]} | {r["diff"]:+.3f} | {ci} | {p} | {pbh} |')
    bad = per_run[per_run['V_fallback_winner_share'].fillna(0) > 0.05] if 'V_fallback_winner_share' in per_run else per_run.iloc[0:0]
    if len(bad):
        lines += ['', '## VALIDITY WARNING: funding winners decided by fallback rankings', '',
                  '| run | fallback winner share | years |', '|---|---|---|']
        for _, r in bad.iterrows():
            lines.append(f"| {r['experiment_id']} | {r['V_fallback_winner_share']:.0%} | {r.get('V_fallback_years', '')} |")
    lines += ['', '## Notes', '',
              '- Unit of replication is the simulated world (seed). With 5 seeds per cell the smallest '
              'attainable two-sided permutation p is 2/252 ~ 0.008.',
              '- P3 uses the application logs when present (`P3_award_source`); the survivor-only fallback '
              'understates concentration because culled agents are missing.',
              '- P4 counts only papers whose h-year window is fully observed; rarefied entropy uses the '
              'common n reported in `rarefaction_n` (0 = too few accepted papers).',
              '- Effective funding rate, reviews per paper, submissions and active counts are SETTINGS '
              '(manipulation checks), not findings.']
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    write_json_file({'cell_summary': cell_summary.to_dict(orient='records'), 'contrasts': contrasts.to_dict(orient='records')}, os.path.join(out_dir, 'analysis.json'), indent=2, default=str)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--glob', default=os.path.join('outputs', 'checkpoints', 'scale_*'))
    parser.add_argument('--num_years', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--out_dir', default=os.path.join('outputs', 'docs', 'scale_expansion_analysis'))
    parser.add_argument('--complete_only', action='store_true',
                        help='use only runs that reached --num_years (interim reports while others run)')
    args = parser.parse_args(argv)

    run_dirs = sorted(d for d in glob.glob(args.glob) if os.path.isdir(d))
    runs = []
    for d in run_dirs:
        try:
            runs.append(compute_run_endpoints(d, args.num_years, args.horizon))
            print(f'[scale] {os.path.basename(d)}: years={runs[-1]["years_completed"]} cell={runs[-1]["cell"]}')
        except (FileNotFoundError, ValueError) as e:
            print(f'[scale] skip {d}: {e}')
    if args.complete_only:
        skipped = [r['experiment_id'] for r in runs if r['years_completed'] < args.num_years]
        runs = [r for r in runs if r['years_completed'] >= args.num_years]
        print(f'[scale] complete_only: using {len(runs)} runs, skipped {len(skipped)} incomplete')
    if not runs:
        print('[scale] no runs found'); return
    per_run, yearly, cell_summary, contrasts = aggregate(runs)
    write_report(args.out_dir, per_run, yearly, cell_summary, contrasts, args.horizon)
    print(open(os.path.join(args.out_dir, 'report.md')).read())


if __name__ == '__main__':
    main()
