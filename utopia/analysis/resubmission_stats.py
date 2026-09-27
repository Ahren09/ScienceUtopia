"""Resubmission-cascade statistics for the paper's review-crisis section.

Computes, from a completed run's checkpoints (paper review_history = one entry
per venue attempt): submission inflation, cascade depth, rejection recycling,
time-to-publication by cohort, reviewer load and concentration, score drift,
venue stratification by realized submission volume, and (for the influx2x2
factorial) per-year attempt composition, the recycling multiplier
M_t = total submission events_t / first submissions_t, and researcher/reviewer
population denominators.

Population semantics: checkpoints are END-of-year states (post Phase-5
attrition cull), so "active in year t" = still active at the end of t. Two
reviewer denominators are reported because get_available_reviewers() does NOT
filter is_active: 'active' (is_active & can_review, the headline denominator)
and 'pool' (all can_review agents, the actual sampling pool, as robustness).

Handles no-resubmission runs (influx2x2 cells A/B): resubmission-conditional
statistics degrade to counts of zero / NaN rather than erroring.

Run:
    python -m utopia.analysis.resubmission_stats
"""

from utopia.utils.data_utils import write_json as write_json_file

from utopia.analysis.statistics import bootstrap_ci
from utopia.utils.data_utils import read_json
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu




def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float('nan')


def _checkpoint(directory, year):
    from utopia.analysis.release import checkpoint_path
    return read_json(checkpoint_path(directory, year))


def compute_stats(run_dir: str, num_years: int) -> dict:
    final = _checkpoint(run_dir, num_years)
    papers = final['paper_tracker']['papers']
    histories = [p['review_history'] for p in papers if p.get('review_history')]

    # --- submission events (every venue attempt) per year ---
    subs_per_year = Counter(e['year'] for h in histories for e in h)
    years = list(range(1, num_years + 1))
    counts = [subs_per_year.get(y, 0) for y in years]
    yoy = [(b - a) / a * 100 for a, b in zip(counts, counts[1:]) if a > 0]

    # --- cascade depth among papers rejected at least once ---
    # All non-final attempts are rejections; a final 'reject' status adds one.
    rejected_attempts = [len(p['review_history']) for p in papers if p.get('review_history')
                         and (len(p['review_history']) > 1 or p['status'] == 'reject')]
    ci = bootstrap_ci(np.array(rejected_attempts))

    # --- rejection recycling: rejections followed by another attempt ---
    resubmitted = sum(len(h) - 1 for h in histories)
    final_rejects = sum(1 for p in papers if p['status'] == 'reject' and p.get('review_history'))
    recycle_rate = _ratio(resubmitted, resubmitted + final_rejects) * 100

    # --- time-to-publication (accepted papers), early vs late first-submission cohorts ---
    t_by_cohort = {'early': [], 'late': []}
    for p in papers:
        h = p.get('review_history')
        if not h or p['status'] != 'accept':
            continue
        t = h[-1]['year'] - h[0]['year'] + 1
        if h[0]['year'] <= 3:
            t_by_cohort['early'].append(t)
        elif 6 <= h[0]['year'] <= 8:  # cap at year 8 to limit right-censoring
            t_by_cohort['late'].append(t)
    mw_time = mannwhitneyu(t_by_cohort['early'], t_by_cohort['late'])

    # --- censoring-fair time-to-publication: equal K-round observation window ---
    # Naive conditional-on-acceptance means are biased by right-censoring (late
    # cohorts lack time for long cascades to resolve). Instead, follow every
    # paper for exactly K rounds from first submission and compare cumulative
    # acceptance within k = 1..K. K=3 fits the horizon for both cohorts
    # (latest late-cohort start is year 8; 8 + K - 1 = num_years).
    K = 3
    window = {c: {'n': 0, 'accepted_times': []} for c in ('early', 'late')}
    for p in papers:
        h = p.get('review_history')
        if not h:
            continue
        first = h[0]['year']
        cohort = 'early' if first <= 3 else 'late' if 6 <= first <= 8 else None
        if cohort is None:
            continue
        window[cohort]['n'] += 1
        if p['status'] == 'accept':
            t = h[-1]['year'] - first + 1
            if t <= K:
                window[cohort]['accepted_times'].append(t)
    window_stats = {}
    for c, w in window.items():
        times = np.array(w['accepted_times'])
        window_stats[c] = {
            'n_papers': w['n'],
            'acceptance_within_k_pct': {k: float((times <= k).sum() / w['n'] * 100)
                                        for k in range(1, K + 1)},
            'mean_time_accepted_within_K': float(times.mean()) if len(times) else float('nan'),
        }
    mw_window = mannwhitneyu(window['early']['accepted_times'],
                             window['late']['accepted_times'])

    # --- years from first submission to acceptance (accepted papers) ---
    t_accept = Counter()
    for p in papers:
        h = p.get('review_history')
        if h and p['status'] == 'accept':
            t_accept[h[-1]['year'] - h[0]['year'] + 1] += 1

    # --- prior rejections carried by resubmitted papers accepted in year y ---
    # Rising means the acceptance price for recycled papers grows over time.
    resub_prior = defaultdict(list)
    for p in papers:
        h = p.get('review_history')
        if h and p['status'] == 'accept' and len(h) > 1:
            resub_prior[h[-1]['year']].append(len(h) - 1)
    resub_prior_stats = {
        y: {'mean': float(np.mean(v)), 'n': len(v),
            'sem': float(np.std(v, ddof=1) / np.sqrt(len(v)))}
        for y, v in sorted(resub_prior.items())}

    # --- first-attempt acceptance rate by year, and first-vs-resubmission score gap ---
    # Both underpin the compositional explanation of the cohort speedup: venues accept a
    # fixed fraction of a pool increasingly padded with lower-scoring resubmissions.
    first_attempt = {y: [0, 0] for y in years}  # year -> [accepted_on_first, first_submissions]
    attempt_scores = {'first': [], 'resub': []}
    for p in papers:
        h = p.get('review_history')
        if not h:
            continue
        for i, e in enumerate(h):
            s = [r['overall_score'] for r in e['reviews']]
            if s:
                attempt_scores['first' if i == 0 else 'resub'].append(float(np.mean(s)))
        y = h[0]['year']
        first_attempt[y][1] += 1
        if p['status'] == 'accept' and len(h) == 1:
            first_attempt[y][0] += 1
    first_attempt_by_year = {y: {'accepted_on_first': a, 'first_submissions': n,
                                 'rate_pct': a / n * 100 if n else float('nan')}
                             for y, (a, n) in first_attempt.items()}
    # No-resubmission runs have an empty resub list; mannwhitneyu would raise.
    if attempt_scores['first'] and attempt_scores['resub']:
        mw_scores_p = float(mannwhitneyu(attempt_scores['first'], attempt_scores['resub']).pvalue)
    else:
        mw_scores_p = float('nan')

    # --- attempt composition and recycling multiplier per year (influx2x2) ---
    # Attempt index = position in review_history: 0 = first submission,
    # 1 = first resubmission, >=2 = second-or-later resubmission.
    composition = {y: {'first': 0, 'resub_1st': 0, 'resub_2plus': 0} for y in years}
    for h in histories:
        for i, e in enumerate(h):
            bucket = 'first' if i == 0 else 'resub_1st' if i == 1 else 'resub_2plus'
            if e['year'] in composition:
                composition[e['year']][bucket] += 1
    recycling_multiplier = {
        y: (subs_per_year.get(y, 0) / composition[y]['first']
            if composition[y]['first'] else float('nan'))
        for y in years}

    # --- reviewer load and concentration ---
    load_per_reviewer_year = Counter((r['reviewer_id'], e['year'])
                                     for h in histories for e in h for r in e['reviews'])
    load_by_year = defaultdict(list)
    for (_, y), n in load_per_reviewer_year.items():
        load_by_year[y].append(n)
    mean_load = {y: float(np.mean(load_by_year[y])) if load_by_year[y] else float('nan') for y in years}

    total_load = Counter()
    for (rid, _), n in load_per_reviewer_year.items():
        total_load[rid] += n
    pubs = Counter(p['author_id'] for p in papers if p['status'] == 'accept')
    top10 = {a for a, _ in pubs.most_common(max(1, len(pubs) // 10))}
    top10_load = [total_load[a] for a in top10 if a in total_load]
    burden_ratio = float(np.mean(top10_load) / np.mean(list(total_load.values())))

    # --- mean review score by year (descriptive drift) ---
    scores_by_year = defaultdict(list)
    for h in histories:
        for e in h:
            for r in e['reviews']:
                scores_by_year[e['year']].append(r['overall_score'])
    score_drift = {y: float(np.mean(v)) for y, v in sorted(scores_by_year.items())}

    # --- venue stratification by ROUND-1 submission volume (top vs bottom half) ---
    # Grouping by cumulative realized volume is circular (venues that grew end up
    # classified as high-volume); rank venues by their year-1 volume instead, and
    # report absolute counts and shares of total alongside percentage growth so
    # small-base effects are visible.
    subs_per_conf_year = defaultdict(Counter)
    for h in histories:
        for e in h:
            subs_per_conf_year[e['conference']][e['year']] += 1
    ranked = sorted(subs_per_conf_year, key=lambda c: subs_per_conf_year[c][1], reverse=True)
    top_half = set(ranked[:len(ranked) // 2])
    total_y1 = sum(c[1] for c in subs_per_conf_year.values())
    total_yT = sum(c[num_years] for c in subs_per_conf_year.values())
    venue_groups, prior_attempts = {}, {'high': [], 'low': []}
    for group, confs in (('high', top_half), ('low', set(ranked) - top_half)):
        y1 = sum(subs_per_conf_year[c][1] for c in confs)
        yT = sum(subs_per_conf_year[c][num_years] for c in confs)
        venue_groups[group] = {
            'year1_submissions': y1,
            f'year{num_years}_submissions': yT,
            'growth_pct': (yT - y1) / y1 * 100 if y1 else float('nan'),
            'share_year1_pct': _ratio(y1, total_y1) * 100,
            f'share_year{num_years}_pct': _ratio(yT, total_yT) * 100,
        }
    for p in papers:
        h = p.get('review_history')
        if h and p['status'] == 'accept':
            group = 'high' if h[-1]['conference'] in top_half else 'low'
            prior_attempts[group].append(len(h) - 1)
    mw_tier = mannwhitneyu(prior_attempts['high'], prior_attempts['low'])

    # --- per-year checkpoint pass: funding-median submission split AND
    #     researcher/reviewer population denominators (single load per year) ---
    new_subs = defaultdict(int)  # (agent, year) -> new papers
    for p in papers:
        if p.get('review_history'):
            new_subs[(p['author_id'], p['review_history'][0]['year'])] += 1
    rate = {'low': [], 'high': []}
    population_by_year = {}          # all can_author agents (gross, incl. culled)
    active_by_year = {}              # is_active & can_author (end-of-year)
    active_reviewer_pool_by_year = {}  # is_active & can_review
    reviewer_pool_by_year = {}       # all can_review (the actual sampling pool)
    for y in years:
        cp = _checkpoint(run_dir, y)
        all_agents = cp['ecosystem_data']['agents']
        population_by_year[y] = sum(1 for a in all_agents if a.get('can_author'))
        active_by_year[y] = sum(1 for a in all_agents
                                if a.get('can_author') and a.get('is_active'))
        reviewer_pool_by_year[y] = sum(1 for a in all_agents if a.get('can_review'))
        active_reviewer_pool_by_year[y] = sum(1 for a in all_agents
                                              if a.get('can_review') and a.get('is_active'))
        agents = [a for a in all_agents if isinstance(a.get('resources'), (int, float))]
        med = float(np.median([a['resources'] for a in agents]))
        for a in agents:
            rate['low' if a['resources'] < med else 'high'].append(new_subs.get((a['id'], y), 0))

    # Reviews assigned per year / reviewer denominator. Numerator counts every
    # individual review written that year (3 per submission event, typically).
    reviews_per_year = Counter()
    for h in histories:
        for e in h:
            reviews_per_year[e['year']] += len(e['reviews'])
    reviews_per_active_reviewer = {
        y: (reviews_per_year.get(y, 0) / active_reviewer_pool_by_year[y]
            if active_reviewer_pool_by_year[y] else float('nan')) for y in years}
    reviews_per_pool_member = {
        y: (reviews_per_year.get(y, 0) / reviewer_pool_by_year[y]
            if reviewer_pool_by_year[y] else float('nan')) for y in years}

    return {
        'run_dir': run_dir, 'num_years': num_years,
        'n_agents': len(_checkpoint(run_dir, 1)
                        ['ecosystem_data']['agents']),
        'n_papers': len(papers),
        'submissions_per_year': dict(zip(years, counts)),
        'mean_yoy_inflation_pct': float(np.mean(yoy)),
        'total_growth_pct': _ratio(counts[-1] - counts[0], counts[0]) * 100,
        'cascade_depth_mean': float(np.mean(rejected_attempts)),
        'cascade_depth_ci95': ci,
        'n_rejected_papers': len(rejected_attempts),
        'recycle_rate_pct': recycle_rate,
        'time_to_pub_mean_early_y1_3': float(np.mean(t_by_cohort['early'])),
        'time_to_pub_mean_late_y6_8': float(np.mean(t_by_cohort['late'])),
        'time_to_pub_mannwhitney_p': float(mw_time.pvalue),
        'time_to_pub_equal_window': {'K': K, 'early': window_stats['early'],
                                     'late': window_stats['late'],
                                     'mannwhitney_p': float(mw_window.pvalue)},
        'time_to_acceptance_hist': {int(t): n for t, n in sorted(t_accept.items())},
        'resub_prior_rejections_by_accept_year': resub_prior_stats,
        'first_attempt_acceptance_by_year': first_attempt_by_year,
        'score_first_vs_resub': {'mean_first': (float(np.mean(attempt_scores['first']))
                                                if attempt_scores['first'] else float('nan')),
                                 'mean_resub': (float(np.mean(attempt_scores['resub']))
                                                if attempt_scores['resub'] else float('nan')),
                                 'n_first': len(attempt_scores['first']),
                                 'n_resub': len(attempt_scores['resub']),
                                 'mannwhitney_p': mw_scores_p},
        'attempt_composition_by_year': composition,
        'recycling_multiplier_by_year': recycling_multiplier,
        'population_by_year': population_by_year,
        'active_researchers_by_year': active_by_year,
        'active_reviewer_pool_by_year': active_reviewer_pool_by_year,
        'reviewer_pool_by_year': reviewer_pool_by_year,
        'reviews_assigned_by_year': {y: reviews_per_year.get(y, 0) for y in years},
        'reviews_per_active_reviewer_by_year': reviews_per_active_reviewer,
        'reviews_per_pool_member_by_year': reviews_per_pool_member,
        'mean_reviews_per_reviewer_year': mean_load,
        'reviewer_load_growth_pct': _ratio(mean_load[num_years] - mean_load[1], mean_load[1]) * 100,
        'burden_ratio_top10pct_authors': burden_ratio,
        'mean_review_score_by_year': score_drift,
        'venue_volume_split': {'grouped_by': 'year1_submission_volume',
                               'high': venue_groups['high'], 'low': venue_groups['low'],
                               'prior_attempts_high': float(np.mean(prior_attempts['high'])),
                               'prior_attempts_low': float(np.mean(prior_attempts['low'])),
                               'mannwhitney_p': float(mw_tier.pvalue)},
        'new_submissions_per_agent_year': {'low_funded': float(np.mean(rate['low'])),
                                           'high_funded': float(np.mean(rate['high']))},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--num_years', type=int, default=10)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    stats = compute_stats(args.run_dir, args.num_years)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'resubmission_stats.json'
    write_json_file(stats, path, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"✅ Saved resubmission stats to: {path}")


if __name__ == '__main__':
    main()
