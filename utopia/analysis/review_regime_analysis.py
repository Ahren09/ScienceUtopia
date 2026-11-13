"""Analysis for the reviewer-monoculture review-replay pilot.

Implements the preregistered endpoints (outputs/docs/review_replay_prereg/prereg_v1.json):
  P1  paired within-paper score-dispersion difference (A - B, blind)
  P2  paired rationale embedding-diversity difference (A - B, blind)
  P3  paired paper-level novelty model: delta_i = meanA_i - meanB_i = b0 + b1*novelty_i (IPW)
  P4  synthetic institutional-prestige cue difference-in-differences (d_A - d_B)

Reuses utopia.analysis.statistics.hierarchical_bootstrap / benjamini_hochberg and
EmbeddingTracker's batched cached encoder. Raw float scores are primary; int-cast
scores are the preregistered sensitivity and the convention for acceptance decisions.
Secondary analyses (ICC(1), Krippendorff interval alpha, distinct-2, acceptance rules,
panel stability, C-reference comparison, manipulation diagnostics) are non-blocking.
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.analysis.statistics import paired_boot, boot_weighted_slope

import json
import logging
import os
from itertools import combinations

import numpy as np
import pandas as pd

from utopia.analysis.statistics import hierarchical_bootstrap, benjamini_hochberg

logger = logging.getLogger('review_regime_analysis')

MIN_SLOTS = 2  # a (paper, regime, arm) cell is valid with >= 2 parsed slots


def analyze_review_sensitivity(corpus_dir, output_dir, stage, n_boot=10000, seed=42):
    """Paper-level paired estimates for a complete review or recognition stage."""
    from difflib import SequenceMatcher
    import re
    import unicodedata
    from utopia.analysis.statistics import paired_spearman_ci

    if stage not in ('main', 'recognition'):
        raise ValueError('Analyze main reviews or independent recognition')
    expected_count = 10800 if stage == 'main' else 1200
    tasks = pd.read_parquet(corpus_dir / f'replay_tasks_{stage}.parquet')
    responses = {}
    for line in (output_dir / 'responses.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('success'):
            responses[row['task_uid']] = row
    if set(responses) != set(tasks.task_uid) or len(responses) != expected_count:
        raise ValueError(f'Need all {expected_count} task IDs; found {len(responses)}')
    common = {
        'n_papers': 600, 'n_responses': expected_count,
        'model': next(iter(responses.values()))['model'], 'sampling_unit': 'paper',
    }
    if stage == 'main':
        scores = pd.DataFrame([{
            'task_uid': row['task_uid'], 'score': float(row['response']['overall_score']),
        } for row in responses.values()])
        scores = tasks.drop(columns='prompt').merge(scores, on='task_uid', validate='one_to_one')
        counts = scores.groupby(['paper_id', 'condition']).size()
        if len(counts) != 3600 or not counts.eq(3).all():
            raise ValueError('Need three profiles for every paper and all six conditions')
        paper = scores.pivot_table(index='paper_id', columns='condition', values='score')
        result = {
            **common,
            'means': {condition: float(paper[condition].mean()) for condition in paper},
            'paired_score_difference_from_original': {
                condition: paired_boot((paper[condition] - paper.original).to_numpy(),
                                       n_boot=n_boot, seed=seed)
                for condition in paper if condition != 'original'},
            'rank_agreement_with_original': {
                condition: paired_spearman_ci(
                    paper.original, paper.original, paper[condition], n_boot=n_boot,
                    seed=seed)['history']
                for condition in paper if condition != 'original'},
        }
    else:
        references = {
            row['paper_id']: row['title']
            for row in json.loads((corpus_dir / 'papers.json').read_text())
        }
        def normalize_title(text):
            return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))
        values = []
        for row in responses.values():
            target = normalize_title(references[row['paper_id']])
            guess = normalize_title(row['response']['title'])
            values.append({
                'paper_id': row['paper_id'], 'condition': row['condition'],
                'exact': float(guess == target),
                'similar': float(SequenceMatcher(None, guess, target).ratio() >= 0.90),
                'declared': float(row['response']['recognized']),
            })
        frame = pd.DataFrame(values)
        paper = frame.pivot(index='paper_id', columns='condition', values='exact')
        result = {
            **common,
            'exact_identification_rate': paper.mean().to_dict(),
            'similar_title_rate': frame.groupby('condition').similar.mean().to_dict(),
            'declared_recognition_rate': frame.groupby('condition').declared.mean().to_dict(),
            'paired_exact_difference': paired_boot(
                (paper.paraphrased_abstract - paper.original_abstract).to_numpy(),
                n_boot=n_boot, seed=seed),
            'interpretation': 'Successful source identification from the supplied abstract',
        }
        frame.to_csv(output_dir / 'recognition_details.csv', index=False)
    if len(paper) != 600:
        raise ValueError('Need all 600 paper-level rows')
    paper.to_csv(output_dir / 'paper_scores.csv')
    write_json_file(result, output_dir / 'analysis.json')
    return result


def analyze_historical_directions(corpus_dir, output_dir, n_boot=10000, seed=42):
    """Paired author-level accuracy on the complete fixed cohort."""
    tasks = pd.read_parquet(corpus_dir / 'replay_tasks_main.parquet')
    responses = {}
    with (output_dir / 'responses.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('success'):
                responses[row['task_uid']] = row
    if set(responses) != set(tasks.task_uid) or len(responses) != 3000:
        raise ValueError(f'Need all 3000 valid responses; found {len(responses)}')
    frame = tasks[['task_uid', 'author_id', 'condition']].merge(pd.DataFrame([{
        'task_uid': row['task_uid'], 'prediction': row['response']['topic'],
    } for row in responses.values()]), on='task_uid', validate='one_to_one')
    paper = frame.pivot(index='author_id', columns='condition', values='prediction')
    gold = pd.read_parquet(corpus_dir / 'hidden_targets.parquet').set_index('author_id')
    paper = paper.join(gold, how='inner', validate='one_to_one')
    if len(paper) != 1000:
        raise ValueError('Need all 1000 author IDs')
    conditions = ['history', 'expertise_only', 'shuffled_history', 'last_topic']
    for condition in conditions:
        paper[condition + '_correct'] = (paper[condition] == paper.target_topic).astype(float)
        paper[condition + '_any_topic'] = [
            float(prediction in json.loads(topics))
            for prediction, topics in zip(paper[condition], paper.target_topics)
        ]
        paper[condition + '_switch'] = (paper[condition] != paper.last_topic).astype(float)
    result = {
        'n_authors': 1000, 'n_model_responses': 3000,
        'n_unique_target_papers': int(paper.target_paper_id.nunique()),
        'model': next(iter(responses.values()))['model'],
        'sampling_unit': 'author; all conditions retained in each paired resample',
        'target': 'First 2021 CS paper; primary CS category is the primary endpoint',
        'accuracy': {
            condition: paired_boot(paper[condition + '_correct'].to_numpy(), n_boot=n_boot, seed=seed)
            for condition in conditions
        },
        'any_topic_accuracy': {
            condition: float(paper[condition + '_any_topic'].mean()) for condition in conditions
        },
        'history_minus_baseline_accuracy': {
            condition: paired_boot(
                (paper.history_correct - paper[condition + '_correct']).to_numpy(),
                n_boot=n_boot, seed=seed)
            for condition in conditions if condition != 'history'
        },
        'predicted_topic_switch_rate': {
            condition: float(paper[condition + '_switch'].mean()) for condition in conditions
        },
        'observed_topic_switch_rate': float((paper.target_topic != paper.last_topic).mean()),
    }
    paper.to_csv(output_dir / 'author_predictions.csv')
    write_json_file(result, output_dir / 'analysis.json')
    logger.info('Historical direction analysis: %s', result['accuracy'])
    return result


def analyze_human_reviews(corpus_dir, output_dir, n_boot=10000, seed=42):
    """Require all planned paper/profile/condition IDs before benchmark analysis."""
    from utopia.analysis.statistics import paired_spearman_ci

    if (corpus_dir / 'evaluation_gold.jsonl').exists():
        return analyze_review_decisions(corpus_dir, output_dir, n_boot=n_boot, seed=seed)
    tasks = pd.read_parquet(corpus_dir / 'replay_tasks_main.parquet')
    responses = {}
    with (output_dir / 'responses.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('success'):
                responses[row['task_uid']] = row
    expected = set(tasks.task_uid)
    if set(responses) != expected or len(expected) != 1200:
        raise ValueError(f"Need all 1200 valid task IDs, found {len(responses)}")
    frame = pd.DataFrame([{
        'task_uid': row['task_uid'],
        'score': float(row['response']['overall_score']),
    } for row in responses.values()])
    frame = tasks.drop(columns='prompt').merge(frame, on='task_uid', validate='one_to_one')
    counts = frame.groupby(['paper_id', 'condition']).size()
    if not counts.eq(3).all() or len(counts) != 400:
        raise ValueError('Require three profiles for each paper and condition')
    paper = frame.pivot_table(index='paper_id', columns='condition', values='score')
    gold = pd.read_parquet(corpus_dir / 'human_gold.parquet')
    paper = paper.reset_index().merge(gold, on='paper_id', validate='one_to_one')
    if len(paper) != 200:
        raise ValueError('Require all 200 paper-level human references')
    correlations = paired_spearman_ci(
        paper.human_mean, paper.expertise_only, paper.history, n_boot=n_boot, seed=seed,
    )
    differences = paired_boot(
        (paper.history - paper.expertise_only).to_numpy(), n_boot=n_boot, seed=seed,
    )
    result = {
        'n_papers': 200, 'n_generated_reviews': 1200,
        'n_human_reviews': int(paper.n_human_reviews.sum()),
        'model': 'Qwen/Qwen3.5-397B-A17B-FP8',
        'sampling_unit': 'paper, with both conditions and all profiles kept together',
        'spearman': correlations, 'paired_score_difference': differences,
        'mean_scores': {
            name: float(paper[name].mean())
            for name in ('history', 'expertise_only', 'human_mean')
        },
        'model_score_scale': [1, 5],
        'human_score_scale': 'original benchmark rating scale',
        'profile_semantics': 'retrieved experts, not the anonymous human reviewers',
    }
    paper.to_csv(output_dir / 'paper_scores.csv', index=False)
    write_json_file(result, output_dir / 'analysis.json')
    logger.info('Human review analysis complete: %s', correlations)
    return result


def fit_review_calibration(calibration_scores, calibration_labels):
    """Fit an unregularized two-parameter logistic mapping on calibration data."""
    from scipy.optimize import minimize
    from scipy.special import expit

    scores = np.asarray(calibration_scores, dtype=float)
    labels = np.asarray(calibration_labels, dtype=float)
    if scores.shape != labels.shape or set(labels) != {0, 1}:
        raise ValueError('Calibration needs aligned scores and both decision classes')
    design = np.column_stack([np.ones(len(scores)), scores])

    def objective(parameters):
        logits = design @ parameters
        return float(np.mean(np.logaddexp(0, logits) - labels * logits))

    def gradient(parameters):
        return design.T @ (expit(design @ parameters) - labels) / len(labels)

    rate = float(labels.mean())
    fitted = minimize(objective, [np.log(rate / (1 - rate)), 0.0],
                      jac=gradient, method='BFGS', options={'gtol': 1e-8})
    if not fitted.success and np.max(np.abs(gradient(fitted.x))) > 1e-6:
        raise ValueError(f'Calibration did not converge: {fitted.message}')
    return {'intercept': float(fitted.x[0]), 'slope': float(fitted.x[1]),
            'base_rate': rate, 'n_calibration': len(labels),
            'objective': 'unregularized binary cross entropy'}


def analyze_review_decisions(corpus_dir, output_dir, n_boot=10000, seed=42):
    """Evaluate a complete external review corpus with a separate calibration set."""
    from scipy.special import expit
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
    from utopia.agents.conference import Conference, ConferenceStatus

    manifest = json.loads((corpus_dir / 'corpus_manifest.json').read_text())
    tasks = pd.read_parquet(corpus_dir / 'replay_tasks_main.parquet')
    responses = {}
    for line in (output_dir / 'responses.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('success'):
            responses[row['task_uid']] = row
    expected = set(tasks.task_uid)
    if (len(tasks) != len(expected) or len(expected) != manifest['n_tasks']
            or set(responses) != expected):
        raise ValueError(f"Need all {manifest['n_tasks']} prescribed valid review IDs")
    if {row['model'] for row in responses.values()} != {manifest['model']}:
        raise ValueError('Responses must use the prescribed model')
    values = pd.DataFrame([{
        'task_uid': uid, 'score': float(row['response']['overall_score']),
    } for uid, row in responses.items()])
    if not np.isfinite(values.score).all() or not values.score.between(1, 5).all():
        raise ValueError('Review scores must be finite and within the native 1-5 scale')
    frame = tasks.drop(columns='prompt').merge(values, on='task_uid', validate='one_to_one')
    counts = frame.groupby('paper_id').size()
    if (len(counts) != manifest['n_papers']
            or not counts.eq(manifest['profiles_per_paper']).all()
            or set(frame.condition) != {manifest['condition']}):
        raise ValueError('Require every prescribed profile and one condition per paper')
    paper = frame.groupby('paper_id').score.agg(list).rename('model_scores').reset_index()
    paper['model_mean'] = paper.model_scores.map(np.mean)
    gold = pd.read_json(corpus_dir / 'evaluation_gold.jsonl', lines=True).rename(
        columns={'id': 'paper_id', 'mean_rating': 'human_mean'})
    if gold.paper_id.duplicated().any() or set(gold.paper_id) != set(paper.paper_id):
        raise ValueError('Human references and generated papers must have identical IDs')
    paper = paper.merge(gold[['paper_id', 'split', 'accepted', 'human_mean', 'ratings',
                              'status', 'primary_area']], on='paper_id', validate='one_to_one')
    if set(paper.split) != {'calibration', 'test'}:
        raise ValueError('Require separate calibration and test papers')
    calibration = paper[paper.split == 'calibration'].copy()
    test = paper[paper.split == 'test'].copy().sort_values('paper_id').reset_index(drop=True)
    fitted = fit_review_calibration(calibration.model_mean, calibration.accepted)
    paper['accept_probability'] = expit(fitted['intercept'] + fitted['slope'] * paper.model_mean)
    test['accept_probability'] = expit(fitted['intercept'] + fitted['slope'] * test.model_mean)
    conference = Conference(
        conference_id='external_review_calibration', name='External review replay',
        topics=[], acceptance_rate=fitted['base_rate'], use_configured_rate=True,
        status=ConferenceStatus.UNDER_REVIEW,
    )
    conference.submitted_papers = [
        {'id': row.paper_id, 'author_id': row.paper_id} for row in test.itertuples()
    ]
    conference.reviews = {
        row.paper_id: [{'overall_score': score} for score in row.model_scores]
        for row in test.itertuples()
    }
    conference.make_acceptance_decisions(tiebreak_seed=seed, acceptance_mode='rate')
    accepted_ids = {row['id'] for row in conference.decisions['accept']}
    test['quota_prediction'] = test.paper_id.isin(accepted_ids).astype(int)
    vectors = test[['model_mean', 'human_mean', 'accepted',
                    'accept_probability', 'quota_prediction']].to_numpy(dtype=float)

    def metrics(sample):
        model, human, labels, probability, prediction = vectors[sample].T
        left, right = np.triu_indices(len(sample), 1)
        human_order = np.sign(human[left] - human[right])
        model_order = np.sign(model[left] - model[right])
        eligible = human_order != 0
        pairwise = np.mean(np.where(model_order[eligible] == 0, 0.5,
                                   model_order[eligible] == human_order[eligible]))
        both_classes = len(np.unique(labels)) == 2
        return {
            'spearman': float(spearmanr(model, human).statistic),
            'acceptance_auroc': float(roc_auc_score(labels, model)) if both_classes else np.nan,
            'human_mean_acceptance_auroc': float(roc_auc_score(labels, human)) if both_classes else np.nan,
            'pairwise_ordering': float(pairwise),
            'quota_accuracy': float(np.mean(prediction == labels)),
            'quota_balanced_accuracy': float(balanced_accuracy_score(labels, prediction)) if both_classes else np.nan,
            'calibrated_brier': float(np.mean((probability - labels) ** 2)),
            'base_rate_brier': float(np.mean((fitted['base_rate'] - labels) ** 2)),
            'brier_difference': float(np.mean((probability - labels) ** 2
                                             - (fitted['base_rate'] - labels) ** 2)),
        }

    point = metrics(np.arange(len(test)))
    rng = np.random.default_rng(seed)
    draws = pd.DataFrame([metrics(rng.integers(0, len(test), len(test))) for _ in range(n_boot)])
    estimates = {}
    for name, value in point.items():
        valid = draws[name][np.isfinite(draws[name])]
        estimates[name] = {
            'estimate': value, 'ci_lo': float(valid.quantile(0.025)),
            'ci_hi': float(valid.quantile(0.975)), 'valid_bootstrap_draws': len(valid),
        }
    reliability = []
    bins = np.minimum((test.accept_probability.to_numpy() * 5).astype(int), 4)
    for index in range(5):
        subset = test.loc[bins == index]
        reliability.append({
            'lower': index / 5, 'upper': (index + 1) / 5, 'n_papers': len(subset),
            'mean_probability': float(subset.accept_probability.mean()) if len(subset) else None,
            'accept_rate': float(subset.accepted.mean()) if len(subset) else None,
        })
    matrix = confusion_matrix(test.accepted, test.quota_prediction, labels=[0, 1])
    human = test.human_mean.to_numpy()
    left, right = np.triu_indices(len(test), 1)
    threshold = min(row['final_score'] for row in conference.decisions['accept'])
    result = {
        'n_papers': len(paper), 'n_generated_reviews': len(responses),
        'n_human_reviews': int(paper.ratings.map(len).sum()),
        'n_calibration': len(calibration), 'n_test': len(test),
        'test_accepted': int(test.accepted.sum()), 'test_rejected': int((1 - test.accepted).sum()),
        'model': manifest['model'], 'condition': manifest['condition'],
        'input_scope': 'pre-review archived title and abstract; retrieved expertise profiles',
        'model_score_scale': [1, 5], 'human_score_categories': [0, 2, 4, 6, 8, 10],
        'metrics': estimates, 'calibration': fitted, 'reliability_equal_width_bins': reliability,
        'confusion_matrix_rows_true_columns_predicted_reject_accept': matrix.tolist(),
        'quota_accepted': len(accepted_ids), 'quota_threshold': float(threshold),
        'quota_threshold_tied_papers': int(np.isclose(test.model_mean, threshold, rtol=0, atol=1e-12).sum()),
        'pairwise_eligible_pairs': int(np.count_nonzero(human[left] != human[right])),
        'test_score_means_by_decision': test.groupby('accepted')[['model_mean', 'human_mean']].mean().to_dict(),
        'bootstrap': 'Test papers sampled together; fitted calibration and quota decisions held fixed',
        'n_boot': n_boot, 'seed': seed, 'conference_id_for_ties': conference.conference_id,
    }
    paper = paper.merge(test[['paper_id', 'quota_prediction']], on='paper_id', how='left')
    paper.to_csv(output_dir / 'paper_scores.csv', index=False)
    write_json_file(result, output_dir / 'analysis.json')
    logger.info('External review decision analysis complete: %s', estimates)
    return result


# --------------------------------------------------------------------------
# Pure estimators (unit-tested)
# --------------------------------------------------------------------------

def within_paper_sd(scores) -> float:
    s = np.asarray(scores, dtype=float)
    return float(np.std(s, ddof=1)) if len(s) >= 2 else np.nan

def mean_pairwise_cosine_distance(embs: np.ndarray) -> float:
    """Mean pairwise cosine distance among row vectors (>=2 rows)."""
    if embs.shape[0] < 2:
        return np.nan
    x = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    dists = [1.0 - float(x[i] @ x[j]) for i, j in combinations(range(x.shape[0]), 2)]
    return float(np.mean(dists))

def icc1(matrix: np.ndarray) -> float:
    """ICC(1), one-way random effects, single rater. matrix: papers x k raters."""
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    grand = m.mean()
    msb = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msw = ((m - m.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    denom = msb + (k - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else np.nan

def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    """Krippendorff's alpha, interval metric, complete data (papers x raters).

    alpha = 1 - D_o / D_e with squared-difference distance; D_o averages within-unit
    pair differences, D_e averages all cross-value pair differences.
    """
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    vals = m.ravel()
    d_o = np.mean([np.mean([(a - b) ** 2 for a, b in combinations(row, 2)]) for row in m])
    diffs = (vals[:, None] - vals[None, :]) ** 2
    d_e = diffs[np.triu_indices(len(vals), 1)].mean()
    return float(1.0 - d_o / d_e) if d_e > 0 else np.nan

def distinct2(texts) -> float:
    """Distinct-2: unique bigrams / total bigrams over pooled texts."""
    bigrams, total = set(), 0
    for t in texts:
        toks = str(t).lower().split()
        for i in range(len(toks) - 1):
            bigrams.add((toks[i], toks[i + 1]))
            total += 1
    return len(bigrams) / total if total else np.nan





# --------------------------------------------------------------------------
# Cell construction
# --------------------------------------------------------------------------

def cell_table(reviews: pd.DataFrame, score_col='score_raw') -> pd.DataFrame:
    """One row per (paper, regime, arm, panel) cell with mean score, SD, slot count,
    justifications. Cells with < MIN_SLOTS parsed slots are dropped (missingness
    is reported separately)."""
    ok = reviews[reviews.success].copy()
    rows = []
    for key, g in ok.groupby(['paper_id', 'regime', 'arm', 'panel']):
        if len(g) < MIN_SLOTS:
            continue
        rows.append({
            'paper_id': key[0], 'regime': key[1], 'arm': key[2], 'panel': key[3],
            'source_seed': g.source_seed.iloc[0],
            'mean_score': float(g[score_col].mean()),
            'mean_score_int': float(g.score_int.mean()),
            'sd_score': within_paper_sd(g[score_col]),
            'n_slots': len(g),
            'justifications': json.dumps(list(g.sort_values('slot').justification)),
            'mean_just_len': float(g.justification_len.mean()),
        })
    return pd.DataFrame(rows)


def paired_frame(cells: pd.DataFrame, arm='blind', panel=1, col='sd_score') -> pd.DataFrame:
    """Wide A/B frame for one arm/panel: one row per paper with col_A, col_B."""
    sub = cells[(cells.arm == arm) & (cells.panel == panel)]
    wide = sub.pivot_table(index=['paper_id', 'source_seed'], columns='regime',
                           values=col, aggfunc='first').reset_index()
    wide = wide.dropna(subset=[c for c in ('A', 'B') if c in wide.columns])
    return wide


def rationale_diversity(cells: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    """Add per-cell mean pairwise cosine distance of justification embeddings.
    One batched encode over all unique justifications (EmbeddingTracker cache)."""
    from utopia.metrics.embedding_tracker import EmbeddingTracker
    texts = []
    for js in cells.justifications:
        texts.extend(json.loads(js))
    uniq = sorted(set(t for t in texts if t))
    tracker = EmbeddingTracker(cache_dir=cache_dir)
    embs = tracker._encode_cached(uniq)
    tracker.flush_cache()
    lookup = {t: e for t, e in zip(uniq, embs)}
    div = []
    for js in cells.justifications:
        vecs = [lookup[t] for t in json.loads(js) if t]
        div.append(mean_pairwise_cosine_distance(np.stack(vecs)) if len(vecs) >= 2 else np.nan)
    out = cells.copy()
    out['rationale_diversity'] = div
    return out


# --------------------------------------------------------------------------
# Acceptance rules
# --------------------------------------------------------------------------

def original_thresholds(eligibility: pd.DataFrame) -> dict:
    """Frozen absolute acceptance threshold per (source_seed, conference, year):
    midpoint between the lowest accepted and highest rejected original mean score
    in the FULL original cohort. Falls back to min accepted (or +inf) when one
    side is empty. Also returns the tie fraction diagnostics."""
    thresholds = {}
    for key, g in eligibility.groupby(['source_seed', 'conference', 'year']):
        acc = g.loc[g.accepted, 'review_score_orig']
        rej = g.loc[~g.accepted, 'review_score_orig']
        if len(acc) and len(rej):
            thr = (acc.min() + rej.max()) / 2.0
        elif len(acc):
            thr = float(acc.min())
        else:
            thr = np.inf
        thresholds[key] = float(thr)
    return thresholds


def apply_absolute_rule(cells: pd.DataFrame, corpus: pd.DataFrame, thresholds: dict,
                        arm='blind', panel=1) -> pd.DataFrame:
    meta = corpus.set_index('paper_id')[['conference', 'year']]
    sub = cells[(cells.arm == arm) & (cells.panel == panel)].copy()
    sub = sub.join(meta, on='paper_id', rsuffix='_c')
    sub['threshold'] = [thresholds.get((s, c, y), np.inf) for s, c, y in
                        zip(sub.source_seed, sub.conference, sub.year)]
    sub['accept_replay'] = sub.mean_score_int >= sub.threshold
    return sub


def apply_capacity_rule(cells_arm: pd.DataFrame, rate=0.30) -> pd.DataFrame:
    """Capacity-constrained sensitivity: within (regime, source_seed, conference, year),
    accept the top `rate` fraction by int-cast panel mean (>=1 accept per non-empty cell,
    mirroring Conference.make_acceptance_decisions)."""
    out = []
    for key, g in cells_arm.groupby(['regime', 'source_seed', 'conference', 'year']):
        g = g.sort_values('mean_score_int', ascending=False).copy()
        n_accept = max(1, int(round(len(g) * rate)))
        g['accept_capacity'] = [i < n_accept for i in range(len(g))]
        out.append(g)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Manipulation diagnostics
# --------------------------------------------------------------------------

PROFILE_KEYWORDS = {
    'rigor': ['rigor', 'sound', 'valid', 'evidence', 'method'],
    'novelty': ['novel', 'original', 'new', 'incremental'],
    'feasibility': ['feasib', 'practical', 'reproduc', 'implement', 'realistic'],
    'impact': ['impact', 'significan', 'field', 'application'],
    'clarity': ['clarity', 'clear', 'presentation', 'writing', 'communicat'],
    'interdisciplinarity': ['interdisciplin', 'bridge', 'domain', 'cross-'],
}
PRESTIGE_KEYWORDS = ['institution', 'university', 'prestig', 'leading', 'standing',
                     'northlake', 'westbrook', 'reputation', 'affiliat']


def profile_adherence(reviews: pd.DataFrame) -> pd.DataFrame:
    """Blinded keyword summary: for each B profile, rate of profile-keyword mentions in
    B rationales with that profile vs the same keywords' rate in A rationales."""
    ok = reviews[reviews.success & (reviews.arm == 'blind') & (reviews.panel == 1)]
    rows = []
    for prof, kws in PROFILE_KEYWORDS.items():
        b_texts = ok[(ok.regime == 'B') & (ok.policy_key == prof)].justification.str.lower()
        a_texts = ok[ok.regime == 'A'].justification.str.lower()
        hit = lambda ts: float(np.mean([any(k in t for k in kws) for t in ts])) if len(ts) else np.nan
        rows.append({'profile': prof, 'n_B': len(b_texts),
                     'B_keyword_rate': hit(b_texts), 'A_baseline_rate': hit(a_texts)})
    return pd.DataFrame(rows)


def prestige_mention_rate(reviews: pd.DataFrame) -> dict:
    ok = reviews[reviews.success]
    out = {}
    for arm in ('blind', 'prestige_high', 'prestige_low'):
        ts = ok[ok.arm == arm].justification.str.lower()
        out[arm] = float(np.mean([any(k in t for k in PRESTIGE_KEYWORDS) for t in ts])) \
            if len(ts) else np.nan
    return out


# --------------------------------------------------------------------------
# Primary endpoint computation
# --------------------------------------------------------------------------

def compute_primaries(reviews: pd.DataFrame, corpus: pd.DataFrame, cache_dir: str,
                      n_boot=10000) -> tuple:
    """Returns (results dict, per-paper frame, cells frame with diversity)."""
    cells = cell_table(reviews)
    cells = rationale_diversity(cells, cache_dir)
    w = corpus.set_index('paper_id')[['ipw_weight', 'novelty_score']]

    res = {}
    # P1: dispersion
    p1 = paired_frame(cells, col='sd_score')
    p1['diff'] = p1.A - p1.B
    res['P1_dispersion'] = paired_boot(p1['diff'].values, n_boot, seed=101)
    # P2: rationale diversity (+ length-adjusted sensitivity)
    p2 = paired_frame(cells, col='rationale_diversity')
    p2['diff'] = p2.A - p2.B
    res['P2_rationale_diversity'] = paired_boot(p2['diff'].values, n_boot, seed=102)
    lenw = paired_frame(cells, col='mean_just_len')
    merged = p2.merge(lenw, on=['paper_id', 'source_seed'], suffixes=('', '_len'))
    if len(merged) > 3:
        ld = merged.dropna(subset=['diff'])
        len_diff = (ld.A_len - ld.B_len).values
        beta = np.polyfit(len_diff, ld['diff'].values, 1)[0]
        resid = ld['diff'].values - beta * (len_diff - len_diff.mean())
        res['P2_length_adjusted_sensitivity'] = paired_boot(resid, n_boot, seed=112)
    # P3: paired paper-level novelty model
    p3 = paired_frame(cells, col='mean_score')
    p3['delta_score'] = p3.A - p3.B
    p3 = p3.join(w, on='paper_id')
    res['P3_novelty_interaction'] = boot_weighted_slope(
        p3.rename(columns={'delta_score': 'y', 'novelty_score': 'x', 'ipw_weight': 'w'}),
        'y', 'x', 'w', n_boot, seed=103)
    # P4: prestige DiD
    hi = paired_frame(cells, arm='prestige_high', col='mean_score').rename(
        columns={'A': 'A_hi', 'B': 'B_hi'})
    lo = paired_frame(cells, arm='prestige_low', col='mean_score').rename(
        columns={'A': 'A_lo', 'B': 'B_lo'})
    p4 = hi.merge(lo, on=['paper_id', 'source_seed'])
    p4['d_A'] = p4.A_hi - p4.A_lo
    p4['d_B'] = p4.B_hi - p4.B_lo
    p4['diff'] = p4.d_A - p4.d_B
    res['P4_prestige_did'] = paired_boot(p4['diff'].values, n_boot, seed=104)
    res['P4_within_regime'] = {
        'd_A': paired_boot(p4.d_A.values, n_boot, seed=114),
        'd_B': paired_boot(p4.d_B.values, n_boot, seed=124),
    }
    # BH across the four primaries
    keys = ['P1_dispersion', 'P2_rationale_diversity', 'P3_novelty_interaction', 'P4_prestige_did']
    adj = benjamini_hochberg([res[k]['p_boot'] for k in keys])
    for k, a in zip(keys, adj):
        res[k]['p_bh'] = float(a)

    # seed-level robustness via the reused two-level bootstrap
    contrasts = []
    for name, frame in (('P1', p1), ('P2', p2), ('P4', p4)):
        f = frame.dropna(subset=['diff'])
        contrasts.append(pd.DataFrame({'outcome': name, 'contrast': 'A-B',
                                       'seed': f.source_seed, 'institution': f.paper_id,
                                       'diff': f['diff']}))
    res['seed_level_robustness'] = hierarchical_bootstrap(
        pd.concat(contrasts, ignore_index=True), n_boot=min(n_boot, 2000), seed=7
    ).to_dict('records')

    per_paper = {'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4}
    return res, per_paper, cells


def compute_secondaries(reviews, cells, corpus, eligibility, stability_reviews, n_boot=2000):
    sec = {}
    blind = paired_frame(cells, col='mean_score')  # ensures common paper set
    for regime in ('A', 'B'):
        ok = reviews[reviews.success & (reviews.arm == 'blind') & (reviews.panel == 1)
                     & (reviews.regime == regime)]
        full = ok.groupby('paper_id').filter(lambda g: len(g) == 3)
        if len(full):
            mat = full.pivot_table(index='paper_id', columns='slot', values='score_raw').values
            sec[f'icc1_{regime}'] = icc1(mat)
            sec[f'kripp_alpha_{regime}'] = krippendorff_alpha_interval(mat)
            sec[f'distinct2_{regime}'] = distinct2(full.justification)
    thresholds = original_thresholds(eligibility)
    dec = apply_absolute_rule(cells, corpus, thresholds)
    sec['absolute_rule'] = {}
    orig = corpus.set_index('paper_id')['accepted']
    for regime in ('A', 'B'):
        d = dec[dec.regime == regime].join(orig, on='paper_id', rsuffix='_orig')
        sec['absolute_rule'][regime] = {
            'accept_rate_unweighted_sample': float(d.accept_replay.mean()),
            'flip_rate_vs_original': float((d.accept_replay != d.accepted).mean()),
            'n': len(d),
        }
    cap = apply_capacity_rule(dec)
    sec['capacity_rule_flip_vs_absolute'] = {
        r: float((g.accept_capacity != g.accept_replay).mean())
        for r, g in cap.groupby('regime')}
    # panel stability (60-paper subset, panel 2)
    if stability_reviews is not None and len(stability_reviews):
        cells2 = cell_table(pd.concat([reviews, stability_reviews], ignore_index=True))
        dec_all = apply_absolute_rule(cells2, corpus, thresholds, panel=1)
        dec2 = apply_absolute_rule(cells2, corpus, thresholds, panel=2)
        sec['panel_stability'] = {}
        for regime in ('A', 'B'):
            a = dec_all[dec_all.regime == regime].set_index('paper_id').accept_replay
            b = dec2[dec2.regime == regime].set_index('paper_id').accept_replay
            common = a.index.intersection(b.index)
            if len(common):
                fa, fb = a.loc[common], b.loc[common]
                inter = (fa & fb).sum()
                union = (fa | fb).sum()
                sec['panel_stability'][regime] = {
                    'n_papers': int(len(common)),
                    'decision_flip_rate': float((fa != fb).mean()),
                    'accept_jaccard': float(inter / union) if union else np.nan,
                }
    # C reference: replay blind means vs original recorded means (descriptive)
    from scipy import stats
    for regime in ('A', 'B'):
        d = blind.join(corpus.set_index('paper_id')['review_score_orig'], on='paper_id')
        x = d[regime].values
        y = d.review_score_orig.values
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() > 3:
            sec[f'orig_comparison_{regime}'] = {
                'pearson_r': float(stats.pearsonr(x[m], y[m])[0]),
                'spearman_r': float(stats.spearmanr(x[m], y[m])[0]),
                'mean_bias': float(np.mean(x[m] - y[m])), 'n': int(m.sum()),
            }
    sec['profile_adherence'] = profile_adherence(reviews).to_dict('records')
    sec['prestige_mention_rate'] = prestige_mention_rate(reviews)
    return sec


def fidelity_c_replay(smoke_reviews: pd.DataFrame, smoke_corpus: pd.DataFrame) -> dict:
    """Smoke-only regime-C behavioral fidelity vs the original stored reviews."""
    from scipy import stats
    c = smoke_reviews[(smoke_reviews.regime == 'C') & smoke_reviews.success]
    if not len(c):
        return {'available': False}
    rep = c.groupby('paper_id').score_raw.mean()
    orig = smoke_corpus.set_index('paper_id')['review_score_orig']
    common = rep.index.intersection(orig.index)
    x, y = rep.loc[common].values, orig.loc[common].values
    orig_scores = [s for js in smoke_corpus.orig_scores for s in json.loads(js)]
    hist_r, _ = np.histogram(c.score_int, bins=np.arange(0.5, 6.5), density=True)
    hist_o, _ = np.histogram(orig_scores, bins=np.arange(0.5, 6.5), density=True)
    slot_bias = {}
    for slot, g in c.groupby('slot'):
        oslot = smoke_corpus.set_index('paper_id').orig_scores.map(
            lambda js: json.loads(js)[slot] if slot < len(json.loads(js)) else np.nan)
        merged = g.set_index('paper_id').score_raw.to_frame('rep').join(oslot.rename('orig'))
        slot_bias[int(slot)] = float((merged.rep - merged.orig).mean())
    orig_lens = [len(j) for js in smoke_corpus.orig_justifications for j in json.loads(js)]
    return {
        'available': True, 'n_papers': int(len(common)),
        'pearson_r': float(stats.pearsonr(x, y)[0]) if len(common) > 3 else np.nan,
        'spearman_r': float(stats.spearmanr(x, y)[0]) if len(common) > 3 else np.nan,
        'mean_bias': float(np.mean(x - y)),
        'score_hist_tv_distance': float(0.5 * np.abs(hist_r - hist_o).sum()),
        'score_wasserstein': float(stats.wasserstein_distance(c.score_int, orig_scores)),
        'slot_mean_bias': slot_bias,
        'mean_justification_len_replay': float(c.justification_len.mean()),
        'mean_justification_len_original': float(np.mean(orig_lens)),
    }


# --------------------------------------------------------------------------
# Figures + report
# --------------------------------------------------------------------------



def write_report(results, out_docs, stage):
    lines = [f'# Reviewer-Monoculture Replay — {stage} analysis report', '']
    lines.append('All outcomes are simulated reviews of frozen E3.5 submissions. '
                 'Primary contrast: regime A (standardized policy) − regime B '
                 '(heterogeneous priority profiles). Raw float scores; blind arm; '
                 'paper-level paired bootstrap CIs; BH across the 4 primary endpoints. '
                 'Recommendation criteria are validity-based, never significance-based.')
    lines.append('')
    for k in ('P1_dispersion', 'P2_rationale_diversity', 'P3_novelty_interaction',
              'P4_prestige_did'):
        r = results['primaries'].get(k)
        if r:
            lines.append(f"- **{k}**: est={r['estimate']:.4f} "
                         f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}], p_boot={r['p_boot']:.4f}"
                         + (f", p_BH={r['p_bh']:.4f}" if 'p_bh' in r else '')
                         + f", n={r['n']}")
    lines.append('')
    lines.append('```json')
    lines.append(json.dumps({k: v for k, v in results.items() if k != 'primaries'},
                            indent=2, default=str)[:8000])
    lines.append('```')
    path = os.path.join(out_docs, 'report.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


# --------------------------------------------------------------------------
# Entry point (called by utopia/experiments/review_replay.py analyze)
# --------------------------------------------------------------------------

def run_analysis(stage: str, corpus_dir: str, stages: dict, prereg_path: str, n_boot=10000):
    from utopia.visual.review_regime import make_figures
    logging.basicConfig(level=logging.INFO)
    cache_dir = os.path.join(corpus_dir, 'embedding_cache')

    def load_reviews(st):
        p = f'outputs/docs/{stages[st]}/reviews.parquet'
        return pd.read_parquet(p) if os.path.exists(p) else None

    eligibility = pd.read_parquet(f'{corpus_dir}/eligibility.parquet')

    if stage == 'smoke':
        reviews = load_reviews('smoke')
        corpus = pd.read_parquet(f'{corpus_dir}/smoke_corpus.parquet')
        out_docs = f'outputs/docs/{stages["smoke"]}'
        manifest = json.load(open(os.path.join(out_docs, 'run_manifest.json')))
        stats_ = manifest.get('llm_call_stats', {})
        n_prompts = stats_.get('n_prompts', 0) or 1
        ab = reviews[reviews.regime != 'C']
        primaries, per_paper, cells = compute_primaries(
            ab, corpus, cache_dir, n_boot=min(n_boot, 2000))
        fidelity = fidelity_c_replay(reviews, corpus)
        wall = manifest.get('wall_clock_seconds', np.nan)
        gate = {
            'valid_json_rate': 1 - stats_.get('n_failures', 0) / n_prompts,
            'first_attempt_rate': stats_.get('n_first_attempt_success', 0) / n_prompts,
            'parse_fail_proxy_for_truncation': stats_.get('n_failures', 0) / n_prompts,
            'n_leakage_excluded_generated': 0,
            'projected_pilot_hours': float(wall) / len(reviews) * 5400 / 3600
            if wall == wall else None,
            'smoke_se_scaled_to_pilot': {
                k: primaries[k]['se_boot'] * np.sqrt(primaries[k]['n'] / 300)
                for k in ('P1_dispersion', 'P2_rationale_diversity', 'P4_prestige_did')
                if k in primaries},
            'c_replay_fidelity': fidelity,
        }
        gate['pass'] = bool(gate['valid_json_rate'] >= 0.99
                            and gate['first_attempt_rate'] >= 0.95
                            and (gate['projected_pilot_hours'] or 0) <= 12)
        results = {'primaries': primaries, 'gate': gate}
        write_json_file(gate, os.path.join(out_docs, 'smoke_gate.json'), indent=2, default=str)
        write_report(results, out_docs, stage)
        logger.info(f'smoke gate: {json.dumps(gate, indent=2, default=str)}')
        return results

    # pilot / final
    reviews = load_reviews('pilot')
    stability = load_reviews('stability')
    corpus = pd.read_parquet(f'{corpus_dir}/pilot_corpus.parquet')
    out_docs = f'outputs/docs/{stages["pilot"]}'
    out_vis = f'outputs/visual/{stages["pilot"]}'
    primaries, per_paper, cells = compute_primaries(reviews, corpus, cache_dir, n_boot)
    secondaries = compute_secondaries(reviews, cells, corpus, eligibility, stability,
                                      n_boot=min(n_boot, 2000))
    w = corpus.ipw_weight
    missing = {
        'n_tasks_failed': int((~reviews.success).sum()),
        'papers_missing_P1': int(300 - primaries['P1_dispersion']['n']),
        'papers_missing_P4': int(300 - primaries['P4_prestige_did']['n']),
        'effective_sample_size_ipw': float((w.sum() ** 2) / (w ** 2).sum()),
        'raw_noninteger_fraction': float(
            (reviews.loc[reviews.success, 'score_raw'] % 1 != 0).mean()),
    }
    results = {'primaries': primaries, 'secondaries': secondaries, 'missingness': missing}
    make_figures(reviews, cells, per_paper, out_vis)
    pd.DataFrame([{**{'endpoint': k}, **{kk: vv for kk, vv in v.items()
                                         if not isinstance(vv, dict)}}
                  for k, v in primaries.items() if isinstance(v, dict) and 'estimate' in v]
                 ).to_csv(os.path.join(out_docs, 'primary_endpoints.csv'), index=False)
    write_json_file(results, os.path.join(out_docs, 'results.json'), indent=2, default=str)
    write_report(results, out_docs, stage)
    logger.info('pilot analysis complete')
    return results
