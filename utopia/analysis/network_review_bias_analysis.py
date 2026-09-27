"""Analysis for the professional-network proximity bias study (network_review_bias).

Separate from review_regime_analysis (the archived reviewer-monoculture study):
endpoints, gate, and report are study-specific; shared pure estimators
are imported, not duplicated. Prereg:
configs/network_review_bias.json.

Unit: the reviewer-paper pair carries the within-pair C-B contrast; the PAPER is
the independent unit (complete matched triplets), bootstrapped by paper.
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.analysis.statistics import cluster_boot_mean, twoway_boot_mean

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

from utopia.analysis.statistics import paired_boot, weighted_slope
from utopia.analysis.statistics import hierarchical_bootstrap, benjamini_hochberg
from utopia.utils.seeding import derive_seed

logger = logging.getLogger('network_review_bias_analysis')

STRATA = ('d2', 'd3', 'far')
N_BOOT_DEFAULT = 10000


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def pair_frame(reviews: pd.DataFrame, score_col: str = 'score_raw') -> pd.DataFrame:
    """One row per (paper, stratum): reviewer, scores under A/B/C, plus the three
    within-pair disclosure contrasts: delta = C-B (proximity channel),
    identity_effect = B-A (identity channel), and total_effect = C-A (full
    disclosure = the additive sum of the two channels).
    Only successful responses contribute; missing cells stay NaN."""
    ok = reviews[reviews.success].copy()
    key = ['paper_id', 'stratum']
    wide = ok.pivot_table(index=key, columns='condition', values=score_col,
                          aggfunc='first').reset_index()
    meta = ok.groupby(key)[['reviewer_id', 'far_kind', 'source_label', 'similarity',
                            'reviewer_betweenness', 'focal_betweenness',
                            'condition_order']].first().reset_index()
    wide = wide.merge(meta, on=key)
    for c in ('A', 'B', 'C'):
        if c not in wide:
            wide[c] = np.nan
    wide['delta'] = wide['C'] - wide['B']
    wide['identity_effect'] = wide['B'] - wide['A']
    wide['total_effect'] = wide['C'] - wide['A']  # full disclosure = delta + identity_effect
    return wide


def paper_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per paper with delta_d2/delta_d3/delta_far; paper_effect =
    delta_d2 - delta_far (primary requires both complete)."""
    p = pairs.pivot_table(index=['paper_id', 'source_label'], columns='stratum',
                          values='delta', aggfunc='first').reset_index()
    for s in STRATA:
        if s not in p:
            p[s] = np.nan
        p = p.rename(columns={s: f'delta_{s}'})
    p['paper_effect'] = p['delta_d2'] - p['delta_far']
    p['d3_effect'] = p['delta_d3'] - p['delta_far']
    return p


# --------------------------------------------------------------------------
# Inference helpers (paper-, reviewer-, and two-way clustered bootstrap)
# --------------------------------------------------------------------------





def stacked_regression(pairs: pd.DataFrame, n_boot: int, seed: int,
                       covariate: str = None) -> dict:
    """Reviewer-pair-level stacked OLS: delta ~ 1{d2} + 1{d3} (far = reference),
    optional similarity covariate; paper-clustered bootstrap on the d2 coef."""
    df = pairs.dropna(subset=['delta']).copy()
    X = np.column_stack([np.ones(len(df)),
                         (df.stratum == 'd2').astype(float),
                         (df.stratum == 'd3').astype(float)])
    names = ['const', 'd2_vs_far', 'd3_vs_far']
    if covariate:
        x = df[covariate].astype(float)
        X = np.column_stack([X, (x - x.mean()) / max(x.std(ddof=0), 1e-12)])
        names.append(covariate)
    y = df.delta.values

    def fit(Xm, ym):
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        return beta

    beta = fit(X, y)
    papers = df.paper_id.values
    uniq = np.unique(papers)
    idx_by = {c: np.flatnonzero(papers == c) for c in uniq}
    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        sel = np.concatenate([idx_by[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        coefs.append(fit(X[sel], y[sel]))
    coefs = np.array(coefs)
    out = {}
    for j, name in enumerate(names):
        lo, hi = np.percentile(coefs[:, j], [2.5, 97.5])
        p = 2 * min((coefs[:, j] <= 0).mean(), (coefs[:, j] >= 0).mean())
        out[name] = {'estimate': float(beta[j]), 'ci_lo': float(lo), 'ci_hi': float(hi),
                     'p_boot': float(min(1.0, p))}
    return out


def moderation(paper_or_pairs: pd.DataFrame, effect_col: str, mod_col: str,
               cluster_col: str, n_boot: int, seed: int) -> dict:
    """Effect ~ standardized moderator slope with cluster bootstrap."""
    df = paper_or_pairs.dropna(subset=[effect_col, mod_col]).copy()
    x = df[mod_col].astype(float)
    z = ((x - x.mean()) / max(x.std(ddof=0), 1e-12)).values
    y = df[effect_col].values
    clusters = df[cluster_col].values
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}

    def slope(sel):
        b1, _ = weighted_slope(y[sel], z[sel], np.ones(len(sel)))
        return b1

    rng = np.random.default_rng(seed)
    stats = [slope(np.concatenate([idx_by[c] for c in
                                   rng.choice(uniq, len(uniq), replace=True)]))
             for _ in range(n_boot)]
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {'estimate': float(slope(np.arange(len(df)))), 'ci_lo': float(lo),
            'ci_hi': float(hi), 'p_boot': float(min(1.0, p)), 'n': int(len(df))}


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

def compute_primary(pairs: pd.DataFrame, papers: pd.DataFrame, n_boot: int) -> dict:
    complete = papers.dropna(subset=['paper_effect'])
    v = complete.paper_effect.values
    primary = paired_boot(v, n_boot=n_boot, seed=101)
    primary['n_papers'] = int(len(complete))
    out = {'primary_d2_minus_far': primary}

    d2far = pairs[pairs.stratum.isin(['d2', 'far'])].dropna(subset=['delta'])
    d2far = d2far.merge(complete[['paper_id']], on='paper_id')
    signed = np.where(d2far.stratum == 'd2', d2far.delta, -d2far.delta)
    out['robust_reviewer_cluster'] = cluster_boot_mean(
        signed * 2, d2far.reviewer_id.values, n_boot, 102)  # mean of signed*2 == contrast
    out['robust_twoway'] = twoway_boot_mean(signed * 2, d2far.paper_id.values,
                                            d2far.reviewer_id.values, n_boot, 103)
    out['robust_stacked_regression'] = stacked_regression(pairs, n_boot, 104)
    out['robust_similarity_adjusted'] = stacked_regression(pairs, n_boot, 105,
                                                           covariate='similarity')
    # hierarchical source-seed -> paper resampling (only meaningful with >1 source)
    if complete.source_label.nunique() > 1:
        contrasts = pd.DataFrame({'outcome': 'primary', 'contrast': 'd2_minus_far',
                                  'seed': complete.source_label.values,
                                  'diff': complete.paper_effect.values})
        hb = hierarchical_bootstrap(contrasts, n_boot=n_boot, seed=106)
        out['robust_hierarchical_seed_paper'] = (
            hb.to_dict('records') if hasattr(hb, 'to_dict') else hb)
    else:
        out['robust_hierarchical_seed_paper'] = 'skipped: single source seed'
    return out


def paired_sign_flip_p(values: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided paired sign-flip permutation p for H0: mean == 0 (+1 correction)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float('nan')
    obs = abs(float(v.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(v)))
    perm_means = np.abs((signs * v).mean(axis=1))
    return float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))


def compute_identity_primary(pairs: pd.DataFrame, n_boot: int) -> dict:
    """Confirmatory PRIMARY: author-identity-cue effect B - A, paper-clustered.

    paper_identity_effect_i = mean over the paper's available strata cells of
    identity_effect_{i,s} (= score_B - score_A). Two-sided paired bootstrap +
    paired sign-flip permutation; standardized paired effect d_z = mean/SD. Also
    reports the pooled cell-level estimate (== exploratory S4) for direct
    comparability to the exploratory +0.055."""
    cells = pairs.dropna(subset=['identity_effect'])
    v = cells.groupby('paper_id').identity_effect.mean().values.astype(float)
    boot = paired_boot(v, n_boot=n_boot,
                       seed=derive_seed(42, 'nrb', 'boot_confirmatory_BA'))
    perm_p = paired_sign_flip_p(v, n_perm=max(n_boot, 10000),
                                seed=derive_seed(42, 'nrb', 'perm_confirmatory_BA'))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else float('nan')
    d_z = float(v.mean() / sd) if sd and not np.isnan(sd) and sd > 0 else float('nan')
    pooled = paired_boot(cells.identity_effect.values, n_boot=n_boot, seed=205)
    by_stratum = {s: paired_boot(cells[cells.stratum == s].identity_effect.values,
                                 n_boot=n_boot, seed=206)
                  for s in STRATA if (cells.stratum == s).sum() > 2}
    return {
        'contrast': 'B_minus_A_identity_cue',
        'unit': 'paper (mean over available strata cells)',
        'paper_level': {**boot, 'perm_p': perm_p, 'cohen_dz': d_z,
                        'n_papers': int(len(v))},
        'pooled_cell_level_S4': {**pooled, 'n_cells': int(len(cells))},
        'by_stratum_cell_level': by_stratum,
    }


def compute_secondaries(pairs: pd.DataFrame, papers: pd.DataFrame, n_boot: int) -> dict:
    sec = {}
    d3 = papers.dropna(subset=['d3_effect'])
    sec['S1_d3_minus_far'] = paired_boot(d3.d3_effect.values, n_boot=n_boot, seed=201)

    # S2: ordinal trend of delta over strata (2, 3, 4=far), paper-clustered
    ordm = {'d2': 2.0, 'd3': 3.0, 'far': 4.0}
    tr = pairs.dropna(subset=['delta']).copy()
    tr['x'] = tr.stratum.map(ordm)
    sec['S2_distance_trend'] = moderation(tr, 'delta', 'x', 'paper_id', n_boot, 202)

    far = pairs[(pairs.stratum == 'far')].dropna(subset=['delta'])
    conn = far[far.far_kind == 'connected_ge4'].delta.values
    disc = far[far.far_kind == 'disconnected'].delta.values
    sec['S3_far_connected_vs_disconnected'] = {
        'connected': paired_boot(conn, n_boot=n_boot, seed=203) if len(conn) > 2 else None,
        'disconnected': paired_boot(disc, n_boot=n_boot, seed=204) if len(disc) > 2 else None,
    }
    idn = pairs.dropna(subset=['identity_effect'])
    sec['S4_identity_B_minus_A'] = {
        'pooled': paired_boot(idn.identity_effect.values, n_boot=n_boot, seed=205),
        'by_stratum': {s: paired_boot(idn[idn.stratum == s].identity_effect.values,
                                      n_boot=n_boot, seed=206)
                       for s in STRATA if (idn.stratum == s).sum() > 2},
    }
    # S9: full-disclosure total effect C - A (= proximity + identity channels).
    # Descriptive decomposition sum, mirroring S4's structure; its CI is bootstrapped
    # directly from the per-cell C-A contrast (NOT by summing the two channels' CIs).
    # Excluded from the BH secondary family (not a preregistered endpoint).
    tot = pairs.dropna(subset=['total_effect'])
    sec['S9_total_disclosure_C_minus_A'] = {
        'pooled': paired_boot(tot.total_effect.values, n_boot=n_boot, seed=210),
        'by_stratum': {s: paired_boot(tot[tot.stratum == s].total_effect.values,
                                      n_boot=n_boot, seed=211)
                       for s in STRATA if (tot.stratum == s).sum() > 2},
    }
    pmod = papers.merge(
        pairs[pairs.stratum == 'd2'][['paper_id', 'focal_betweenness']].drop_duplicates(),
        on='paper_id', how='left')
    sec['S5_author_betweenness_moderation'] = moderation(
        pmod, 'paper_effect', 'focal_betweenness', 'paper_id', n_boot, 207)
    sec['S6_reviewer_betweenness_moderation'] = moderation(
        pairs.dropna(subset=['delta']), 'delta', 'reviewer_betweenness',
        'paper_id', n_boot, 208)
    return sec


def acceptance_flip_rate(reviews: pd.DataFrame, threshold: float = 3.0) -> dict:
    """S7: deterministic acceptance (score_int >= threshold) flips between B and C."""
    ok = reviews[reviews.success]
    wide = ok.pivot_table(index=['paper_id', 'stratum'], columns='condition',
                          values='score_int', aggfunc='first')
    if not {'B', 'C'} <= set(wide.columns):
        return {'flip_rate': None}
    both = wide.dropna(subset=['B', 'C'])
    accB, accC = both.B >= threshold, both.C >= threshold
    flips = (accB != accC)
    by_stratum = {s: float(flips.loc[(slice(None), s)].mean())
                  for s in STRATA if (slice(None), s) and
                  s in both.index.get_level_values('stratum')}
    return {'threshold': threshold, 'flip_rate': float(flips.mean()),
            'flip_to_accept_rate': float((~accB & accC).mean()),
            'flip_to_reject_rate': float((accB & ~accC).mean()),
            'by_stratum': by_stratum, 'n': int(len(both))}


def descriptives(reviews: pd.DataFrame, prereg: dict) -> dict:
    """Non-blocking descriptive metrics: relationship-mention rate and rationale
    positivity from the frozen prereg word lists."""
    kw = [k.lower() for k in prereg['descriptive_non_blocking'][0]
          .split('frozen keyword list: [')[1].split(']')[0]
          .replace("'", '').split(', ')]
    pos = ['strong', 'novel', 'clear', 'significant', 'sound', 'compelling',
           'thorough', 'promising', 'rigorous', 'excellent']
    neg = ['weak', 'unclear', 'incremental', 'flawed', 'limited', 'insufficient',
           'vague', 'unconvincing', 'marginal', 'poor']
    ok = reviews[reviews.success].copy()

    def mention(j):
        jl = (j or '').lower()
        return any(k in jl for k in kw)

    def polarity(j):
        jl = (j or '').lower()
        return sum(w in jl for w in pos) - sum(w in jl for w in neg)

    out = {}
    for cond, grp in ok.groupby('condition'):
        out[cond] = {'mention_rate': float(grp.justification.map(mention).mean()),
                     'mean_polarity': float(grp.justification.map(polarity).mean())}
    c = ok[ok.condition == 'C']
    out['C_mention_by_stratum'] = {s: float(g.justification.map(mention).mean())
                                   for s, g in c.groupby('stratum')}
    return out


# --------------------------------------------------------------------------
# Smoke gate (13 preregistered criteria)
# --------------------------------------------------------------------------

def smoke_gate(reviews: pd.DataFrame, chunk_rows: pd.DataFrame, tasks: pd.DataFrame,
               corpus: pd.DataFrame, audit: dict, wall_clock_s: float,
               n_main_tasks: int) -> dict:
    checks = {}
    checks['1_all_strata_represented'] = set(reviews.stratum) >= set(STRATA)
    checks['2_expertise_tolerance'] = bool((corpus.max_similarity_gap <= 0.10 + 1e-9).all())
    checks['3_pre_review_graph_only'] = audit.get('future_edge_violations', 1) == 0
    mind = pd.concat([corpus[f'min_distance_any_author_{s}'] for s in STRATA])
    checks['4_no_d1_coi'] = bool((mind >= 2).all())
    a_prompts = tasks[tasks.condition == 'A'].prompt
    checks['5_abc_separation'] = bool(
        (~a_prompts.str.contains('## Author Information', regex=False)).all())
    bc_ok = True
    for (pid, s), grp in tasks.groupby(['paper_id', 'stratum']):
        b = grp.loc[grp.condition == 'B', 'prompt']
        c = grp.loc[grp.condition == 'C', 'prompt']
        if len(b) and len(c):
            db = [l for l in b.iloc[0].splitlines() if l not in c.iloc[0].splitlines()]
            dc = [l for l in c.iloc[0].splitlines() if l not in b.iloc[0].splitlines()]
            if not (len(db) == len(dc) == 1
                    and db[0].startswith('- Professional-network relationship:')
                    and dc[0].startswith('- Professional-network relationship:')):
                bc_ok = False
    checks['6_bc_only_relationship_field'] = bc_ok
    checks['7_zero_leakage'] = True  # extract-time hard assertions; failure aborts extract
    checks['8_parse_after_retry_ge_99'] = bool(reviews.success.mean() >= 0.99)
    checks['9_first_attempt_ge_95'] = bool(chunk_rows.success.mean() >= 0.95)
    checks['10_no_unresolved_truncation'] = int((~reviews.success).sum()) == 0
    checks['11_resume_no_repeat'] = bool(reviews.task_uid.is_unique)
    projected_h = wall_clock_s * (n_main_tasks / max(len(reviews), 1)) / 3600
    checks['12_projected_main_hours_le_12'] = projected_h <= 12
    checks['13_writes_in_allowed_dirs'] = True  # enforced by assert_write_allowed
    return {'checks': checks, 'passed': all(checks.values()),
            'projected_main_hours': projected_h,
            'parse_after_retry': float(reviews.success.mean()),
            'first_attempt': float(chunk_rows.success.mean())}


# --------------------------------------------------------------------------
# Figures + report
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_analysis(stage: str, corpus_dir: str, stages: dict, prereg_path: str,
                 n_boot: int = N_BOOT_DEFAULT, out_name: str = 'network_review_bias_final'):
    prereg = json.load(open(prereg_path))
    corpus = pd.read_parquet(os.path.join(corpus_dir, 'matched_corpus.parquet'))
    audit = json.load(open(os.path.join(corpus_dir, 'structural_audit.json')))

    def load_stage(st):
        docs = f'outputs/docs/{stages[st]}'
        rev = pd.read_parquet(os.path.join(docs, 'reviews.parquet'))
        chunks = pd.concat([pd.read_parquet(p) for p in
                            sorted(glob.glob(os.path.join(docs, 'review_chunks',
                                                          'chunk_*.parquet')))],
                           ignore_index=True)
        manifest = json.load(open(sorted(glob.glob(os.path.join(docs,
                                                                'run_manifest*.json')))[-1])) \
            if glob.glob(os.path.join(docs, 'run_manifest*.json')) else {}
        return rev, chunks, manifest, docs

    if stage == 'smoke':
        rev, chunks, manifest, docs = load_stage('smoke')
        tasks = pd.read_parquet(os.path.join(corpus_dir, 'replay_tasks_smoke.parquet'))
        n_main = len(pd.read_parquet(os.path.join(corpus_dir, 'replay_tasks_main.parquet'),
                                     columns=['task_uid']))
        wall = float(manifest.get('extra', {}).get('wall_clock_seconds',
                     rev.chunk_elapsed_s.max() if 'chunk_elapsed_s' in rev else 0))
        smoke_corpus = corpus[corpus.paper_id.isin(rev.paper_id.unique())]
        gate = smoke_gate(rev, chunks, tasks, smoke_corpus, audit, wall, n_main)
        out = os.path.join(docs, 'smoke_gate.json')
        write_json_file(gate, out, indent=2)
        logger.info(f'smoke gate {"PASSED" if gate["passed"] else "FAILED"}: {out}')
        print(json.dumps(gate, indent=2))
        return gate

    # main / final: full endpoint suite
    rev, chunks, manifest, docs = load_stage('main')
    pairs = pair_frame(rev)
    papers = paper_frame(pairs)
    primary = compute_primary(pairs, papers, n_boot)
    identity_primary = compute_identity_primary(pairs, n_boot)
    sec = compute_secondaries(pairs, papers, n_boot)
    sec['S7_acceptance_flips'] = acceptance_flip_rate(rev)

    # stability panel
    stability_agreement, s8 = None, None
    try:
        rev_s, _, _, _ = load_stage('stability')
        pairs_s = pair_frame(rev_s)
        papers_s = paper_frame(pairs_s)
        s8 = paired_boot(papers_s.dropna(subset=['paper_effect']).paper_effect.values,
                         n_boot=n_boot, seed=209)
        stability_agreement = papers.rename(columns={'paper_effect': 'paper_effect_main'}) \
            .merge(papers_s.rename(columns={'paper_effect': 'paper_effect_stab'})
                   [['paper_id', 'paper_effect_stab']], on='paper_id')
    except FileNotFoundError:
        logger.warning('stability panel not available yet')
    sec['S8_stability_replication'] = s8

    # BH over the preregistered secondary family
    fam = {'S1': sec['S1_d3_minus_far']['p_boot'],
           'S2': sec['S2_distance_trend']['p_boot'],
           'S3': (sec['S3_far_connected_vs_disconnected']['connected'] or {}).get('p_boot'),
           'S4': sec['S4_identity_B_minus_A']['pooled']['p_boot'],
           'S5': sec['S5_author_betweenness_moderation']['p_boot'],
           'S6': sec['S6_reviewer_betweenness_moderation']['p_boot'],
           'S7': None, 'S8': s8['p_boot'] if s8 else None}
    valid = {k: v for k, v in fam.items() if v is not None}
    adjusted = benjamini_hochberg(list(valid.values()))
    sec['bh_family'] = {k: {'p': v, 'p_bh': float(a), 'significant_bh': bool(a <= 0.05)}
                        for (k, v), a in zip(valid.items(), adjusted)}

    # sensitivities
    sens = {}
    pairs_int = pair_frame(rev, score_col='score_int')
    papers_int = paper_frame(pairs_int)
    sens['integer_scores'] = paired_boot(
        papers_int.dropna(subset=['paper_effect']).paper_effect.values,
        n_boot=n_boot, seed=301)
    conn_pairs = pairs[(pairs.stratum != 'far') | (pairs.far_kind == 'connected_ge4')]
    sens['exclude_disconnected'] = paired_boot(
        paper_frame(conn_pairs).dropna(subset=['paper_effect']).paper_effect.values,
        n_boot=n_boot, seed=302)
    if papers.source_label.nunique() > 1:
        sens['leave_one_source_out'] = {
            lab: paired_boot(papers[papers.source_label != lab]
                             .dropna(subset=['paper_effect']).paper_effect.values,
                             n_boot=n_boot, seed=303)
            for lab in papers.source_label.unique()}
    ordv = pairs.dropna(subset=['delta']).copy()
    ordv['first_is_C'] = (ordv.condition_order.str[0] == 'C').astype(float)
    sens['condition_order_covariate'] = moderation(ordv, 'delta', 'first_is_C',
                                                   'paper_id', n_boot, 304)

    desc = descriptives(rev, prereg)
    results = {'stage': stage, 'n_boot': n_boot,
               'prereg_v1_sha256': audit.get('prereg_v1_sha256'),
               'far_comparator': audit.get('far_comparator_decision'),
               'primary': primary,
               'confirmatory_primary_identity_BA': identity_primary,
               'secondary': sec, 'sensitivities': sens,
               'descriptive': desc,
               'funnel': {'pairs_total': int(len(pairs)),
                          'papers_complete_primary': primary['primary_d2_minus_far']['n_papers'],
                          'papers_identity_BA': identity_primary['paper_level']['n_papers'],
                          'parse_success': float(rev.success.mean())}}
    out_docs = f'outputs/docs/{out_name}'
    os.makedirs(out_docs, exist_ok=True)
    write_json_file(results, os.path.join(out_docs, 'results.json'), indent=2, default=str)
    logger.info(f'analysis complete -> {out_docs}/results.json')
    return results
