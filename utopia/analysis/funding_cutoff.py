"""Matthew-effect regression discontinuity at the funding cutoff.

Local rank-cutoff RD (discrete running variable) around the per-panel funding
cut num_winners = max(1, int(n_panel * funding_rate)). Estimates the causal
effect of narrowly winning vs narrowly missing an early funding award.

Study configuration: configs/funding_cutoff.json.
Data: funding_applications_year_<y>.jsonl written by utopia/simulation.py with
--log_funding_applications, plus yearly checkpoints and parquet extracts.

Driver: utopia/experiments/funding_cutoff.py. Unit tests: tests/experiments/test_funding_cutoff.py.
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.utils.data_utils import read_json

from utopia.analysis.statistics import benjamini_hochberg

from utopia.utils.paths import project_root

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = str(project_root(__file__))

# Prereg-frozen constants (prereg_v1.json)
TREATMENT_MAX_FIRST_APP_YEAR = 3
HORIZON = 3
CITATION_AGE_WINDOW = 2
CANDIDATE_BANDWIDTHS = [2, 3, 4, 5, 6]
SENSITIVITY_BANDWIDTHS = [2, 3, 4, 6, 8]
LOCAL_RAND_WINDOWS = [0.5, 1.5]
PLACEBO_CUTOFFS = [-2.0, 2.0]
MIN_NEAR_CUTOFF_N = 150          # |X| <= 2 pooled, credibility gate
MIE_PRIMARY = 10.0               # half of one award (academic_base_budget = 20)
N_PERMUTATIONS = 5000

SECONDARY_ENDPOINTS = [
    'pubs_written_3y', 'pubs_accepted_3y', 'citations_age2_matched',
    'any_application_3y', 'time_to_next_application', 'any_award_3y',
    'attrition_by_3y', 'time_to_attrition', 'exploration_distance_3y',
]

BALANCE_COVARIATES = [
    'prior_awards', 'prior_funding_amount', 'prior_papers', 'prior_accepted',
    'prior_citations', 'n_concurrent_apps', 'is_explorer', 'is_exploiter',
    'expertise_size',
]

_ALLOWED_WRITE_PREFIXES = (
    'data/funding_cutoff/',
    'data/funding_cutoff_cost/',
    'outputs/docs/funding_cutoff_',
    'outputs/logs/funding_cutoff_',
    # Simulator-side run directories for the funding_cutoff stage (experiment_id is
    # derived by utopia/arguments.py as explore_funding_cutoff_*):
    'outputs/checkpoints/explore_funding_cutoff_',
    'outputs/logs/explore_funding_cutoff_',
    'outputs/docs/explore_funding_cutoff_',
)


def guard_write_path(path: str) -> str:
    """Reject any write outside the Matthew-RD sandbox. Returns realpath-safe path."""
    rel = os.path.relpath(os.path.realpath(os.path.abspath(path)),
                          os.path.realpath(REPO_ROOT))
    # outputs/ is a symlink; also accept the resolved target by re-checking the
    # repo-relative UNresolved form.
    rel_unresolved = os.path.relpath(os.path.abspath(path), REPO_ROOT)
    for cand in (rel, rel_unresolved):
        if any(cand.startswith(p) for p in _ALLOWED_WRITE_PREFIXES):
            return path
    raise PermissionError(f"funding_cutoff path guard: refusing write outside sandbox: {path} (rel={rel})")


def guarded_makedirs(path: str) -> str:
    guard_write_path(os.path.join(path, '_'))
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def load_application_log(run_dir: str) -> pd.DataFrame:
    """Load all funding_applications_year_*.jsonl for one run."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, 'funding_applications_year_*.jsonl'))):
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise FileNotFoundError(f"no funding_applications_year_*.jsonl in {run_dir}")
    return pd.DataFrame(rows)


def load_final_checkpoint(run_dir: str) -> Dict:
    paths = sorted(glob.glob(os.path.join(run_dir, 'checkpoint_year_*.json*')),
                   key=lambda p: int(p.split('checkpoint_year_')[1].split('.')[0]))
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    path = paths[-1]
    return read_json(path)


def _award_years(funding_success_history: Dict) -> List[Tuple[int, float]]:
    out = []
    for _pid, awards in (funding_success_history or {}).items():
        for a in awards:
            out.append((int(a['year']), float(a['amount'])))
    return out


def build_application_dataset(experiment_ids: List[str], outputs_dir: str = 'outputs',
                              num_years: int = 6) -> Tuple[pd.DataFrame, Dict]:
    """Build the application-level RD dataset across runs (seeds).

    Returns (df, exclusion_accounting). One row per included application:
    running variable, treatment, competition id, pre-treatment covariates,
    primary and secondary outcomes per prereg_v1.
    """
    all_rows = []
    excl = {'panels_total': 0, 'panels_fallback': 0, 'panels_imputed_tail': 0,
            'panels_winner_mismatch': 0, 'panels_no_loser_side': 0,
            'apps_total_logged': 0, 'apps_first_year_gt_max': 0,
            'apps_untrackable': 0, 'apps_in_excluded_panels': 0}

    for exp_id in experiment_ids:
        run_dir = os.path.join(outputs_dir, 'checkpoints', exp_id)
        docs_dir = os.path.join(outputs_dir, 'docs', exp_id)
        seed = exp_id.split('seed')[-1].split('_')[0]

        apps = load_application_log(run_dir)
        apps['seed'] = seed
        apps['competition'] = (apps['seed'] + '|' + apps['year'].astype(str) + '|'
                               + apps['program_id'] + '|' + apps['panel_index'].astype(str))
        excl['apps_total_logged'] += len(apps)

        ckpt = load_final_checkpoint(run_dir)
        agents = {a['id']: a for a in ckpt['ecosystem_data']['agents']}
        agent_year = pd.read_parquet(os.path.join(docs_dir, 'agent_year.parquet'))
        paper = pd.read_parquet(os.path.join(docs_dir, 'paper.parquet'))
        paper_year = dict(zip(paper['paper_id'], paper['year']))
        papers_by_author = paper.groupby('author_id')
        citations = ckpt['citation_tracker']['citations']  # cited_id -> [citer ids]

        # awards by (agent, year)
        awards = {aid: _award_years(a.get('funding_success_history'))
                  for aid, a in agents.items()}

        # Cutoff reconstruction: for every (program, year), the union of logged
        # funded applicants across panels must EXACTLY equal the winners
        # recorded in checkpoint funding_success_history.
        def _recorded_winners(pid: str, yr: int) -> set:
            return {aid for aid, a in agents.items()
                    if any(int(e['year']) == yr
                           for e in (a.get('funding_success_history') or {}).get(pid, []))}

        program_year_ok = {}
        for (pid, yr), g in apps.groupby(['program_id', 'year']):
            logged = set(g.loc[g['funded'], 'applicant_id'])
            program_year_ok[(pid, int(yr))] = (logged == _recorded_winners(pid, int(yr)))

        # panel-level validity
        panel_valid = {}
        for comp, g in apps.groupby('competition'):
            excl['panels_total'] += 1
            ok = True
            if g['fallback_ranking'].any():
                excl['panels_fallback'] += 1
                ok = False
            if ok and g['imputed_tail'].any():
                excl['panels_imputed_tail'] += 1
                ok = False
            if ok and (g['funded'].all() or (~g['funded']).all()):
                excl['panels_no_loser_side'] += 1
                ok = False
            if ok and not program_year_ok[(g['program_id'].iloc[0], int(g['year'].iloc[0]))]:
                excl['panels_winner_mismatch'] += 1
                ok = False
            panel_valid[comp] = ok

        # first application year per applicant (from the full log, valid or not)
        first_year = apps.groupby('applicant_id')['year'].min()

        # per-applicant application years (for participation outcomes)
        app_years = apps.groupby('applicant_id')['year'].apply(lambda s: sorted(set(s)))

        # active years per agent for attrition
        ay_active = agent_year[agent_year['is_active'] == True]  # noqa: E712
        last_active = ay_active.groupby('agent_id')['year'].max()

        for _, r in apps.iterrows():
            aid = r['applicant_id']
            t = int(r['year'])
            if int(first_year[aid]) != t:
                continue  # not a first-year application
            if t > TREATMENT_MAX_FIRST_APP_YEAR:
                excl['apps_first_year_gt_max'] += 1
                continue
            if not panel_valid[r['competition']]:
                excl['apps_in_excluded_panels'] += 1
                continue
            if aid not in agents:
                excl['apps_untrackable'] += 1
                continue

            k = int(r['num_winners'])
            x = (k - int(r['position'])) + 0.5
            ag = agents[aid]
            ag_years = agent_year[agent_year['agent_id'] == aid].set_index('year')

            # ---- pre-treatment covariates (papers of year t are written
            # before the funding phase of year t, hence pre-treatment) ----
            try:
                my_papers = papers_by_author.get_group(aid)
            except KeyError:
                my_papers = paper.iloc[0:0]
            pre_papers = my_papers[my_papers['year'] <= t]
            prior_awards_list = [(y, amt) for y, amt in awards.get(aid, []) if y < t]
            pre_citers = 0
            for pid_ in pre_papers['paper_id']:
                pre_citers += sum(1 for c in citations.get(pid_, [])
                                  if paper_year.get(c, 10**9) <= t)

            # ---- outcomes over [t+1, t+HORIZON] ----
            w0, w1 = t + 1, t + HORIZON
            fund_3y = sum(amt for y, amt in awards.get(aid, []) if w0 <= y <= w1)
            post_papers = my_papers[(my_papers['year'] >= w0) & (my_papers['year'] <= w1)]
            # age-matched citations: papers in [t+1, min(t+3, num_years - AGE)]
            cit_last_pub_year = min(w1, num_years - CITATION_AGE_WINDOW)
            cite_papers = my_papers[(my_papers['year'] >= w0)
                                    & (my_papers['year'] <= cit_last_pub_year)]
            cit_age2 = 0
            for pid_, py in zip(cite_papers['paper_id'], cite_papers['year']):
                cit_age2 += sum(1 for c in citations.get(pid_, [])
                                if paper_year.get(c, 10**9) <= py + CITATION_AGE_WINDOW)

            later_apps = [y for y in app_years.get(aid, []) if y > t]
            any_app = int(any(w0 <= y <= w1 for y in later_apps))
            any_award = int(any(w0 <= y <= w1 for y, _ in awards.get(aid, [])))

            la = int(last_active.get(aid, t))
            exited = la < num_years            # absent/inactive before end of run
            exit_year = la + 1 if exited else None
            attr_3y = int(exited and exit_year <= w1)

            # discrete survival times (censored at num_years)
            if later_apps:
                t_next_app, next_app_event = later_apps[0] - t, 1
            else:
                t_next_app, next_app_event = (min(la, num_years) - t), 0
            if exited:
                t_attr, attr_event = exit_year - t, 1
            else:
                t_attr, attr_event = num_years - t, 0

            surv = ag_years[(ag_years.index >= w0) & (ag_years.index <= w1)]
            expl_dist = float(surv['direction_distance'].dropna().mean()) \
                if len(surv) and surv['direction_distance'].notna().any() else np.nan
            # Resource trajectory over the outcome window ('funding' column is
            # the year-end resource stock). Exited agents have no rows -> nan
            # mean but below-threshold=1 by construction (they hit <=10).
            res_w = surv['funding'].dropna() if len(surv) else pd.Series(dtype=float)
            mean_res = float(res_w.mean()) if len(res_w) else np.nan
            min_res = float(res_w.min()) if len(res_w) else np.nan
            below_paper_thr = int((res_w < 20).any()) if len(res_w) else 1

            strategy = ag.get('exploration_strategy') or ''
            all_rows.append({
                'seed': r['seed'], 'competition': r['competition'],
                'applicant_id': aid, 'program_id': r['program_id'],
                'panel_index': r['panel_index'], 'treatment_year': t,
                'n_panel': int(r['n_panel']), 'num_winners': k,
                'position': int(r['position']), 'x': x,
                'funded': int(bool(r['funded'])),
                # covariates
                'prior_awards': len(prior_awards_list),
                'prior_funding_amount': sum(a for _, a in prior_awards_list),
                'prior_papers': len(pre_papers),
                'prior_accepted': int((pre_papers['accepted'] == True).sum()),  # noqa: E712
                'prior_citations': pre_citers,
                'n_concurrent_apps': 0,  # filled below
                'strategy': strategy,
                'is_explorer': int(strategy == 'explorer'),
                'is_exploiter': int(strategy == 'exploiter'),
                'expertise_size': len(ag.get('expertise') or []),
                'institution': ag.get('university_name'),
                # outcomes
                'cumulative_subsequent_funding_3y': fund_3y,
                'pubs_written_3y': len(post_papers),
                'pubs_accepted_3y': int((post_papers['accepted'] == True).sum()),  # noqa: E712
                'citations_age2_matched': cit_age2,
                'any_application_3y': any_app,
                'time_to_next_application': t_next_app,
                'next_application_event': next_app_event,
                'any_award_3y': any_award,
                'attrition_by_3y': attr_3y,
                'time_to_attrition': t_attr,
                'attrition_event': attr_event,
                'exploration_distance_3y': expl_dist,
                'mean_resources_3y': mean_res,
                'min_resources_3y': min_res,
                'below_paper_threshold_3y': below_paper_thr,
            })

    df = pd.DataFrame(all_rows)
    if len(df):
        conc = df.groupby(['seed', 'applicant_id'])['competition'].transform('count')
        df['n_concurrent_apps'] = conc
    return df, excl


# --------------------------------------------------------------------------
# Estimation core
# --------------------------------------------------------------------------

def _triangular_weights(x: np.ndarray, h: float) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(x) / h)





def _weighted_demean(M: np.ndarray, w: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Weighted within-group demeaning (competition fixed effects)."""
    out = M.astype(float).copy()
    dfm = pd.DataFrame(M)
    dfm['_w'] = w
    dfm['_g'] = groups
    for j in range(M.shape[1]):
        wm = dfm.groupby('_g').apply(
            lambda g, j=j: np.average(g[j], weights=g['_w']), include_groups=False)
        out[:, j] = M[:, j] - dfm['_g'].map(wm).to_numpy()
    return out


def _cluster_vcov(Xw: np.ndarray, resid_w: np.ndarray, XtX_inv: np.ndarray,
                  clusters: np.ndarray) -> np.ndarray:
    """CR1 cluster-robust vcov for WLS expressed in weighted form."""
    n, kdim = Xw.shape
    u = Xw * resid_w[:, None]
    dfu = pd.DataFrame(u)
    dfu['_c'] = clusters
    S = dfu.groupby('_c').sum().to_numpy()
    meat = S.T @ S
    G = len(np.unique(clusters))
    dof = (G / max(G - 1, 1)) * ((n - 1) / max(n - kdim, 1))
    return dof * XtX_inv @ meat @ XtX_inv


def local_linear_rd(df: pd.DataFrame, y: str, h: float,
                    x_col: str = 'x', d_col: str = 'funded',
                    cutoff: float = 0.0, fe: bool = True) -> Dict:
    """Local-linear RD: Y ~ D + (X-c) + D*(X-c), triangular kernel, competition FE.

    Returns tau, SEs clustered by applicant, by competition, and two-way (CGM).
    """
    sub = df[df[y].notna()].copy()
    xc = sub[x_col].to_numpy(dtype=float) - cutoff
    keep = np.abs(xc) <= h
    sub, xc = sub[keep], xc[keep]
    if len(sub) < 10 or sub[d_col].nunique() < 2:
        return {'tau': np.nan, 'se': np.nan, 'n': int(len(sub)), 'h': h}
    w = _triangular_weights(xc, h)
    D = sub[d_col].to_numpy(dtype=float)
    M = np.column_stack([D, xc, D * xc])
    Y = sub[y].to_numpy(dtype=float)

    if fe:
        comp = sub['competition'].to_numpy()
        MY = _weighted_demean(np.column_stack([M, Y]), w, comp)
        Md, Yd = MY[:, :3], MY[:, 3]
    else:
        Md = np.column_stack([np.ones(len(sub)), M])
        Yd = Y

    sw = np.sqrt(w)
    Xw = Md * sw[:, None]
    Yw = Yd * sw
    XtX = Xw.T @ Xw
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xw.T @ Yw)
    resid_w = Yw - Xw @ beta
    j = 0 if fe else 1   # index of D coefficient

    cl_a = sub['applicant_id'].to_numpy(dtype=str)
    cl_c = sub['competition'].to_numpy(dtype=str)
    V_a = _cluster_vcov(Xw, resid_w, XtX_inv, cl_a)
    V_c = _cluster_vcov(Xw, resid_w, XtX_inv, cl_c)
    cl_ac = np.char.add(np.char.add(cl_a, '#'), cl_c)
    V_ac = _cluster_vcov(Xw, resid_w, XtX_inv, cl_ac)
    V_2w = V_a + V_c - V_ac
    se_2w = float(np.sqrt(max(V_2w[j, j], 0)))
    tau = float(beta[j])

    from scipy import stats
    G_min = min(len(np.unique(cl_a)), len(np.unique(cl_c)))
    p = 2 * stats.t.sf(abs(tau / se_2w), df=max(G_min - 1, 1)) if se_2w > 0 else np.nan
    tcrit = stats.t.ppf(0.975, df=max(G_min - 1, 1))
    return {
        'tau': tau, 'se': se_2w,
        'se_applicant': float(np.sqrt(max(V_a[j, j], 0))),
        'se_competition': float(np.sqrt(max(V_c[j, j], 0))),
        'ci_lo': tau - tcrit * se_2w, 'ci_hi': tau + tcrit * se_2w,
        'p': float(p), 'n': int(len(sub)), 'h': h,
        'n_competitions': int(sub['competition'].nunique()),
        'n_applicants': int(sub['applicant_id'].nunique()),
    }


def local_quadratic_rd(df: pd.DataFrame, y: str, h: float) -> Dict:
    """Sensitivity: adds X^2 and D*X^2 terms within the same window."""
    sub = df[df[y].notna()].copy()
    xc = sub['x'].to_numpy(dtype=float)
    keep = np.abs(xc) <= h
    sub, xc = sub[keep], xc[keep]
    if len(sub) < 14 or sub['funded'].nunique() < 2:
        return {'tau': np.nan, 'se': np.nan, 'n': int(len(sub)), 'h': h}
    w = _triangular_weights(xc, h)
    D = sub['funded'].to_numpy(dtype=float)
    M = np.column_stack([D, xc, D * xc, xc ** 2, D * xc ** 2])
    Y = sub[y].to_numpy(dtype=float)
    MY = _weighted_demean(np.column_stack([M, Y]), w, sub['competition'].to_numpy())
    Md, Yd = MY[:, :5], MY[:, 5]
    sw = np.sqrt(w)
    Xw, Yw = Md * sw[:, None], Yd * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ Yw)
    resid_w = Yw - Xw @ beta
    V = _cluster_vcov(Xw, resid_w, XtX_inv, sub['applicant_id'].astype(str).to_numpy())
    return {'tau': float(beta[0]), 'se': float(np.sqrt(max(V[0, 0], 0))),
            'n': int(len(sub)), 'h': h, 'spec': 'local_quadratic'}


def local_randomization(df: pd.DataFrame, y: str, window: float,
                        rng: np.random.Generator) -> Dict:
    """Difference in means within |X| <= window + Fisher permutation p-value,
    permuting funded status within competition."""
    sub = df[(df[y].notna()) & (df['x'].abs() <= window)].copy()
    if sub['funded'].nunique() < 2:
        return {'tau': np.nan, 'n': int(len(sub)), 'window': window}
    y1 = sub.loc[sub['funded'] == 1, y].mean()
    y0 = sub.loc[sub['funded'] == 0, y].mean()
    tau = float(y1 - y0)

    Yv = sub[y].to_numpy(dtype=float)
    Dv = sub['funded'].to_numpy()
    comp_codes = pd.factorize(sub['competition'])[0]
    n1_by_comp = pd.Series(Dv).groupby(comp_codes).sum().to_numpy().astype(int)
    idx_by_comp = [np.where(comp_codes == c)[0] for c in range(comp_codes.max() + 1)]
    null = np.empty(N_PERMUTATIONS)
    for b in range(N_PERMUTATIONS):
        Db = np.zeros_like(Dv)
        for c, idx in enumerate(idx_by_comp):
            pick = rng.choice(idx, size=n1_by_comp[c], replace=False)
            Db[pick] = 1
        m1, m0 = Db.sum(), (1 - Db).sum()
        null[b] = (Yv[Db == 1].mean() if m1 else np.nan) - (Yv[Db == 0].mean() if m0 else np.nan)
    p_fisher = float(np.mean(np.abs(null) >= abs(tau)))

    # FE diff-in-means with uniform weights, applicant-clustered SE
    w = np.ones(len(sub))
    MY = _weighted_demean(np.column_stack([Dv.astype(float), Yv]), w, sub['competition'].to_numpy())
    Dd, Yd = MY[:, 0:1], MY[:, 1]
    XtX_inv = np.linalg.pinv(Dd.T @ Dd)
    beta = XtX_inv @ (Dd.T @ Yd)
    resid = Yd - (Dd @ beta)
    V = _cluster_vcov(Dd, resid, XtX_inv, sub['applicant_id'].astype(str).to_numpy())
    return {'tau': tau, 'tau_fe': float(beta[0]), 'p_fisher': p_fisher,
            'se_fe_dim': float(np.sqrt(max(V[0, 0], 0))),
            'n': int(len(sub)), 'window': window,
            'n_funded': int(Dv.sum()), 'n_unfunded': int((1 - Dv).sum())}


def cv_bandwidth(df: pd.DataFrame, covariate: str = 'prior_papers',
                 candidates: List[int] = None) -> Tuple[int, Dict]:
    """Leave-one-competition-out CV MSE of the local-linear fit on a
    PRE-TREATMENT covariate (prereg: outcome-blind bandwidth selection)."""
    candidates = candidates or CANDIDATE_BANDWIDTHS
    losses = {}
    comps = df['competition'].unique()
    for h in candidates:
        sq_errs = []
        for g in comps:
            train = df[df['competition'] != g]
            test = df[(df['competition'] == g) & (df['x'].abs() <= h)]
            if not len(test):
                continue
            sub = train[train[covariate].notna() & (train['x'].abs() <= h)]
            if len(sub) < 10 or sub['funded'].nunique() < 2:
                continue
            xc = sub['x'].to_numpy(dtype=float)
            w = _triangular_weights(xc, h)
            D = sub['funded'].to_numpy(dtype=float)
            M = np.column_stack([np.ones(len(sub)), D, xc, D * xc])
            Y = sub[covariate].to_numpy(dtype=float)
            sw = np.sqrt(w)
            beta = np.linalg.pinv((M * sw[:, None]).T @ (M * sw[:, None])) @ \
                ((M * sw[:, None]).T @ (Y * sw))
            xt = test['x'].to_numpy(dtype=float)
            Dt = test['funded'].to_numpy(dtype=float)
            Mt = np.column_stack([np.ones(len(test)), Dt, xt, Dt * xt])
            pred = Mt @ beta
            sq_errs.extend((test[covariate].to_numpy(dtype=float) - pred) ** 2)
        losses[h] = float(np.mean(sq_errs)) if sq_errs else np.inf
    best = min(losses, key=losses.get)
    return best, losses


def placebo_cutoff(df: pd.DataFrame, y: str, c0: float, h: float) -> Dict:
    """Pseudo-jump at c0 estimated within ONE side of the true cutoff."""
    side = df[df['x'] < 0] if c0 < 0 else df[df['x'] > 0]
    side = side.copy()
    side['funded'] = (side['x'] > c0).astype(int)
    res = local_linear_rd(side, y, h=h, cutoff=c0)
    res['placebo_cutoff'] = c0
    return res


def interaction_rd(df_base: pd.DataFrame, df_cost: pd.DataFrame, y: str,
                   h: float) -> Dict:
    """Difference-in-discontinuities across environments (funding_cutoff_cost).

    Stacked local-linear model within |X| <= h, triangular kernel:
        Y ~ D + X + D:X + E:D + E:X + E:D:X  (+ competition FE)
    where E = costly environment. Competition FE absorb the E main effect
    (every competition lies wholly in one environment). The coefficient on
    E:D is tau_cost - tau_baseline. SEs cluster by (seed, applicant_id)
    POOLED across environments: the same applicant id under the same seed is
    the paired parallel-world realization, so the pair forms one cluster.
    """
    from scipy import stats
    df_base = df_base.assign(_env=0.0)
    df_cost = df_cost.assign(_env=1.0)
    sub = pd.concat([df_base, df_cost], ignore_index=True)
    sub = sub[sub[y].notna()]
    xc = sub['x'].to_numpy(dtype=float)
    keep = np.abs(xc) <= h
    sub, xc = sub[keep], xc[keep]
    if len(sub) < 20 or sub['funded'].nunique() < 2 or sub['_env'].nunique() < 2:
        return {'tau_diff': np.nan, 'n': int(len(sub)), 'h': h}
    w = _triangular_weights(xc, h)
    D = sub['funded'].to_numpy(dtype=float)
    E = sub['_env'].to_numpy(dtype=float)
    # competition ids are unique per environment (seed|year|program|panel with
    # distinct experiment stages) but guard against collisions explicitly:
    comp = (sub['_env'].astype(int).astype(str) + '~' + sub['competition'].astype(str)).to_numpy()
    M = np.column_stack([D, xc, D * xc, E * D, E * xc, E * D * xc])
    Y = sub[y].to_numpy(dtype=float)
    MY = _weighted_demean(np.column_stack([M, Y]), w, comp)
    Md, Yd = MY[:, :6], MY[:, 6]
    sw = np.sqrt(w)
    Xw, Yw = Md * sw[:, None], Yd * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ Yw)
    resid_w = Yw - Xw @ beta
    j = 3  # E:D coefficient = tau_cost - tau_baseline

    cl_pair = (sub['seed'].astype(str) + '|' + sub['applicant_id'].astype(str)).to_numpy()
    cl_comp = comp
    V_p = _cluster_vcov(Xw, resid_w, XtX_inv, cl_pair)
    V_c = _cluster_vcov(Xw, resid_w, XtX_inv, cl_comp)
    cl_int = np.char.add(np.char.add(cl_pair.astype(str), '#'), cl_comp.astype(str))
    V_i = _cluster_vcov(Xw, resid_w, XtX_inv, cl_int)
    V = V_p + V_c - V_i
    se = float(np.sqrt(max(V[j, j], 0)))
    tau_diff = float(beta[j])
    G_min = min(len(np.unique(cl_pair)), len(np.unique(cl_comp)))
    p = 2 * stats.t.sf(abs(tau_diff / se), df=max(G_min - 1, 1)) if se > 0 else np.nan
    tcrit = stats.t.ppf(0.975, df=max(G_min - 1, 1))
    return {'tau_diff': tau_diff, 'se': se, 'ci_lo': tau_diff - tcrit * se,
            'ci_hi': tau_diff + tcrit * se, 'p': float(p), 'n': int(len(sub)),
            'tau_base_within': float(beta[0]),
            'tau_cost_within': float(beta[0] + beta[3]), 'h': h}


def paired_seed_differences(df_base: pd.DataFrame, df_cost: pd.DataFrame,
                            y: str, h: float) -> Dict:
    """Per-seed RD estimates in each environment and their paired differences."""
    out = {}
    for s in sorted(set(df_base['seed']) & set(df_cost['seed'])):
        rb = local_linear_rd(df_base[df_base['seed'] == s], y, h=h)
        rc = local_linear_rd(df_cost[df_cost['seed'] == s], y, h=h)
        out[s] = {'tau_base': rb['tau'], 'tau_cost': rc['tau'],
                  'diff': (rc['tau'] - rb['tau'])
                  if np.isfinite(rb['tau']) and np.isfinite(rc['tau']) else np.nan,
                  'n_base': rb['n'], 'n_cost': rc['n']}
    diffs = [v['diff'] for v in out.values() if np.isfinite(v['diff'])]
    out['summary'] = {'mean_diff': float(np.mean(diffs)) if diffs else np.nan,
                      'min_diff': float(np.min(diffs)) if diffs else np.nan,
                      'max_diff': float(np.max(diffs)) if diffs else np.nan,
                      'n_seeds': len(diffs)}
    return out


def km_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete-time Kaplan-Meier survival curve."""
    ts = np.arange(1, int(times.max()) + 1)
    surv, s = [], 1.0
    for t in ts:
        at_risk = np.sum(times >= t)
        d = np.sum((times == t) & (events == 1))
        if at_risk > 0:
            s *= (1 - d / at_risk)
        surv.append(s)
    return ts, np.array(surv)


# --------------------------------------------------------------------------
# Synthetic data (unit tests / effect-recovery validation)
# --------------------------------------------------------------------------

def make_synthetic_rd(n_competitions: int = 120, panel_n: int = 20, k: int = 5,
                      tau: float = 12.0, slope: float = 1.5,
                      noise: float = 4.0, seed: int = 0) -> pd.DataFrame:
    """Panels with known treatment effect tau at the cutoff.

    Y = 50 + slope * X + tau * funded + applicant noise; X = (k - pos) + 0.5.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_competitions):
        for pos in range(1, panel_n + 1):
            x = (k - pos) + 0.5
            funded = int(pos <= k)
            y = 50 + slope * x + tau * funded + rng.normal(0, noise)
            rows.append({'competition': f'c{g}', 'applicant_id': f'a{g}_{pos}',
                         'x': x, 'funded': funded, 'y': y,
                         'position': pos, 'num_winners': k, 'n_panel': panel_n,
                         'seed': 's0'})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Full analysis pipeline
# --------------------------------------------------------------------------

def sample_size_gate(df: pd.DataFrame) -> Dict:
    near = {f'abs_x_le_{h}': int((df['x'].abs() <= h).sum())
            for h in sorted(set(CANDIDATE_BANDWIDTHS + SENSITIVITY_BANDWIDTHS))}
    w = _triangular_weights(df.loc[df['x'].abs() <= 2, 'x'].to_numpy(dtype=float), 2.0)
    kish = float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0
    reps = df.groupby(['seed', 'applicant_id']).size()
    gate = {
        'n_competitions': int(df['competition'].nunique()),
        'n_applications': int(len(df)),
        'n_funded': int(df['funded'].sum()),
        'n_unique_applicants': int(df.groupby(['seed', 'applicant_id']).ngroups),
        'repeat_applications_mean': float(reps.mean()),
        'obs_within_bandwidth': near,
        'effective_n_kish_h2': kish,
        'passes_credibility_threshold': bool(near['abs_x_le_2'] >= MIN_NEAR_CUTOFF_N),
        'threshold': MIN_NEAR_CUTOFF_N,
    }
    return gate


def run_full_analysis(df: pd.DataFrame, out_docs: str,
                      exclusions: Optional[Dict] = None,
                      rng_seed: int = 20260712) -> Dict:
    """Prereg_v1 analysis: primary + secondaries + diagnostics + robustness."""
    guarded_makedirs(out_docs)
    rng = np.random.default_rng(rng_seed)
    results: Dict = {'prereg': 'funding_cutoff_v1', 'exclusions': exclusions or {}}

    results['sample_size_gate'] = sample_size_gate(df)

    h_main, cv_losses = cv_bandwidth(df)
    results['bandwidth'] = {'h_main': h_main, 'cv_losses': cv_losses,
                            'procedure': 'LOCO CV on prior_papers (outcome-blind)'}

    primary = 'cumulative_subsequent_funding_3y'
    results['primary'] = local_linear_rd(df, primary, h=h_main)
    results['primary']['mie'] = MIE_PRIMARY
    results['primary']['endpoint'] = primary

    # Secondary family + BH correction
    sec = {}
    for ep in SECONDARY_ENDPOINTS:
        sec[ep] = local_linear_rd(df, ep, h=h_main)
    pvals = [(ep, sec[ep]['p']) for ep in SECONDARY_ENDPOINTS
             if np.isfinite(sec[ep].get('p', np.nan))]
    if pvals:
        eps, ps = zip(*pvals)
        for ep, q in zip(eps, benjamini_hochberg(np.array(ps))):
            sec[ep]['p_bh'] = float(q)
    results['secondary'] = sec

    # Diagnostics
    diag = {}
    diag['treatment_jump'] = local_linear_rd(df, 'funded', h=h_main)
    diag['balance'] = {c: local_linear_rd(df, c, h=h_main) for c in BALANCE_COVARIATES}
    counts = df.groupby('x').size()
    diag['running_variable_counts'] = {str(k): int(v) for k, v in counts.items()}
    results['diagnostics'] = diag

    # Robustness
    rob = {'bandwidth_sensitivity': {}, 'polynomial': {}, 'leave_one_out': {},
           'placebo_cutoffs': {}, 'local_randomization': {}}
    for h in SENSITIVITY_BANDWIDTHS:
        rob['bandwidth_sensitivity'][h] = local_linear_rd(df, primary, h=h)
    rob['polynomial'][h_main] = local_quadratic_rd(df, primary, h=h_main)
    for prog in df['program_id'].unique():
        rob['leave_one_out'][f'drop_program_{prog}'] = local_linear_rd(
            df[df['program_id'] != prog], primary, h=h_main)
    for s in df['seed'].unique():
        rob['leave_one_out'][f'drop_seed_{s}'] = local_linear_rd(
            df[df['seed'] != s], primary, h=h_main)
    for c0 in PLACEBO_CUTOFFS:
        rob['placebo_cutoffs'][c0] = placebo_cutoff(df, primary, c0=c0, h=h_main)
    for wdw in LOCAL_RAND_WINDOWS:
        rob['local_randomization'][wdw] = local_randomization(df, primary, wdw, rng)
    results['robustness'] = rob

    # Exploratory heterogeneity
    het = {}
    for strat in ['explorer', 'exploiter', 'cautious_explorer']:
        het[f'strategy_{strat}'] = local_linear_rd(
            df[df['strategy'] == strat], primary, h=h_main)
    med = df['n_concurrent_apps'].median()
    het['concurrent_le_median'] = local_linear_rd(
        df[df['n_concurrent_apps'] <= med], primary, h=h_main)
    het['concurrent_gt_median'] = local_linear_rd(
        df[df['n_concurrent_apps'] > med], primary, h=h_main)
    results['heterogeneity_exploratory'] = het


    out_path = guard_write_path(os.path.join(out_docs, 'rd_results.json'))
    write_json_file(results, out_path, indent=1, default=_json_safe)
    return results


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _binned_means(df: pd.DataFrame, y: str, max_x: float = 8):
    g = df[df[y].notna() & (df['x'].abs() <= max_x)].groupby('x')[y]
    m = g.mean()
    se = g.std() / np.sqrt(g.size().clip(lower=1))
    return m.index.to_numpy(), m.to_numpy(), se.to_numpy()


def local_linear_fit_line(df: pd.DataFrame, y: str, h: float, side: int):
    """Triangular-kernel weighted local-linear fit on one side of the cutoff.

    Returns (grid, yhat) for plotting over [0, h] (side=+1) or [-h, 0]
    (side=-1), or None when fewer than 5 observations fall in the window.
    """
    sub = df[(df[y].notna()) & (np.sign(df['x']) == side) & (df['x'].abs() <= h)]
    if len(sub) < 5:
        return None
    xc = sub['x'].to_numpy(dtype=float)
    w = _triangular_weights(xc, h)
    M = np.column_stack([np.ones(len(sub)), xc])
    Y = sub[y].to_numpy(dtype=float)
    sw = np.sqrt(w)
    beta = np.linalg.pinv((M * sw[:, None]).T @ (M * sw[:, None])) @ ((M * sw[:, None]).T @ (Y * sw))
    grid = np.linspace(0 if side > 0 else -h, h if side > 0 else 0, 20)
    return grid, beta[0] + beta[1] * grid


def local_linear_fit_band(df: pd.DataFrame, y: str, h: float, side: int,
                          n_grid: int = 25):
    """One-sided triangular-kernel local-linear fit with a pointwise 95% band.

    Same window, kernel and sample as ``local_linear_rd`` restricted to one
    side of the cutoff, but WITHOUT competition fixed effects (the level of a
    one-sided fit is not identified under FE). The band uses the two-way
    (applicant x competition) CR1 cluster vcov with a t critical value at
    df = G_min - 1, mirroring the estimator's inference. The gap between the
    two sides at x = 0 equals ``local_linear_rd(..., fe=False)['tau']``.

    Returns (grid, yhat, lo, hi) over [-h, 0] (side=-1) or [0, h] (side=+1),
    or None when fewer than 5 observations fall in the window.
    """
    sub = df[(df[y].notna()) & (np.sign(df['x']) == side) & (df['x'].abs() <= h)]
    if len(sub) < 5:
        return None
    xc = sub['x'].to_numpy(dtype=float)
    sw = np.sqrt(_triangular_weights(xc, h))
    Xw = np.column_stack([np.ones(len(sub)), xc]) * sw[:, None]
    Yw = sub[y].to_numpy(dtype=float) * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ Yw)
    resid_w = Yw - Xw @ beta
    cl_a = sub['applicant_id'].to_numpy(dtype=str)
    cl_c = sub['competition'].to_numpy(dtype=str)
    cl_ac = np.char.add(np.char.add(cl_a, '#'), cl_c)
    V = (_cluster_vcov(Xw, resid_w, XtX_inv, cl_a)
         + _cluster_vcov(Xw, resid_w, XtX_inv, cl_c)
         - _cluster_vcov(Xw, resid_w, XtX_inv, cl_ac))
    from scipy import stats
    G_min = min(len(np.unique(cl_a)), len(np.unique(cl_c)))
    tcrit = stats.t.ppf(0.975, df=max(G_min - 1, 1))
    grid = np.linspace(-h if side < 0 else 0, 0 if side < 0 else h, n_grid)
    Xg = np.column_stack([np.ones_like(grid), grid])
    yhat = Xg @ beta
    # Two-way CR1 vcov is not guaranteed PSD: guard the quadratic form.
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', Xg, V, Xg), 0.0))
    return grid, yhat, yhat - tcrit * se, yhat + tcrit * se


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Extract and analyze funding cutoffs from completed new runs.")
    parser.add_argument('--run', nargs='+', required=True)
    parser.add_argument('--outputs-root', default='outputs')
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/docs/funding_cutoff_public'))
    args = parser.parse_args(argv)
    from utopia.analysis.release import summarize_run, finite_json
    summaries = [summarize_run(run, args.outputs_root)[0] for run in args.run]
    horizons = {row['num_years'] for row in summaries}
    if len(horizons) != 1 or min(horizons) < 6:
        parser.error('Funding cutoff analysis requires matched complete runs of at least six years')
    frame, exclusions = build_application_dataset(args.run, args.outputs_root, next(iter(horizons)))
    # This explicit output location applies only to this invocation.
    global _ALLOWED_WRITE_PREFIXES
    previous = _ALLOWED_WRITE_PREFIXES
    _ALLOWED_WRITE_PREFIXES = (os.path.relpath(args.out_dir.resolve(), REPO_ROOT) + '/',)
    try:
        guarded_makedirs(str(args.out_dir))
        frame.to_parquet(args.out_dir / 'applications.parquet', index=False)
        if frame.empty:
            write_json_file({'status': 'not_estimable', 'reason': 'No eligible applications',
                             'exclusions': exclusions}, args.out_dir / 'rd_results.json', indent=2)
        else:
            result = run_full_analysis(frame, str(args.out_dir), exclusions=exclusions)
            write_json_file(finite_json(result), args.out_dir / 'rd_results.json', indent=2, allow_nan=False)
    finally:
        _ALLOWED_WRITE_PREFIXES = previous


if __name__ == '__main__':
    main()
