"""Shared statistical estimators with explicit sampling units."""

import json
import numpy as np
import pandas as pd


def hierarchical_bootstrap(contrasts: pd.DataFrame, n_boot=10000, seed=0):
    """Two-level bootstrap: resample seeds, then institutions within seed.

    Returns rows per (outcome, contrast): mean, 95% CI, fraction of seeds with
    the same direction, cluster counts.
    """
    rng = np.random.default_rng(seed)
    out = []
    for (outcome, contrast), df in contrasts.groupby(["outcome", "contrast"]):
        seeds = df["seed"].unique()
        by_seed = {s: df[df["seed"] == s]["diff"].values for s in seeds}
        point = float(np.mean([np.mean(v) for v in by_seed.values()]))
        seed_means = [float(np.mean(v)) for v in by_seed.values()]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            if len(seeds) > 1:
                chosen = rng.choice(seeds, size=len(seeds), replace=True)
            else:
                chosen = seeds
            vals = []
            for s in chosen:
                v = by_seed[s]
                vals.append(np.mean(v[rng.integers(0, len(v), len(v))]))
            boots[b] = np.mean(vals)
        pooled_sd = float(df["diff"].std(ddof=1)) if len(df) > 1 else np.nan
        # Two-sided bootstrap p: how often the resampled mean crosses zero,
        # floored at 1/n_boot (cannot resolve smaller p from n_boot draws).
        p_boot = 2.0 * min(float(np.mean(boots <= 0)), float(np.mean(boots >= 0)))
        p_boot = float(min(1.0, max(p_boot, 1.0 / n_boot)))
        out.append(
            {
                "outcome": outcome,
                "contrast": contrast,
                "mean_diff": point,
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
                "p_boot": p_boot,
                "smd": point / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan,
                "n_seeds": len(seeds),
                "n_institution_blocks": len(df),
                "seed_direction_agreement": float(
                    np.mean(np.sign(seed_means) == np.sign(point))
                )
                if point != 0
                else np.nan,
                "per_seed_means": json.dumps(
                    {str(s): round(float(np.mean(v)), 5) for s, v in by_seed.items()}
                ),
            }
        )
    return pd.DataFrame(out)


def benjamini_hochberg(pvals):
    """BH-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank_idx in range(n - 1, -1, -1):
        i = order[rank_idx]
        val = min(prev, p[i] * n / (rank_idx + 1))
        adj[i] = val
        prev = val
    return adj


def bootstrap_mean_ci(values, n_boot=2000, seed=0):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return (None, None, None)
    if len(v) == 1:
        return (float(v[0]), None, None)
    means = _bootstrap_means(v, n_boot, seed)
    return (
        float(v.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bootstrap_ci(values, n_boot=10000, seed=0):
    means = _bootstrap_means(values, n_boot, seed)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_means(values, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)


def weighted_slope(y, x, w) -> tuple:
    """WLS slope+intercept for y = b0 + b1*x with weights w."""
    y, x, w = (np.asarray(v, dtype=float) for v in (y, x, w))
    W = w.sum()
    xbar, ybar = (w * x).sum() / W, (w * y).sum() / W
    b1 = (w * (x - xbar) * (y - ybar)).sum() / (w * (x - xbar) ** 2).sum()
    return float(b1), float(ybar - b1 * xbar)


def paired_boot(values: np.ndarray, n_boot=10000, seed=0, stat=np.mean) -> dict:
    """Paper-level bootstrap of a paired per-paper statistic vector."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    point = float(stat(v))
    return _paired_boot_summary(point, boots, len(v), n_boot)


def boot_weighted_slope(
    df: pd.DataFrame, ycol, xcol, wcol, n_boot=10000, seed=0
) -> dict:
    d = df.dropna(subset=[ycol, xcol, wcol]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    point = weighted_slope(d[ycol], d[xcol], d[wcol])[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        boots[b] = weighted_slope(
            d[ycol].values[idx], d[xcol].values[idx], d[wcol].values[idx]
        )[0]
    return _paired_boot_summary(point, boots, len(d), n_boot)


def _paired_boot_summary(point, boots, sample_size, n_boot):
    """Summarize paired resamples with the original finite-resolution p-value floor."""
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "estimate": float(point),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "p_boot": float(min(1.0, max(p, 1.0 / n_boot))),
        "n": int(sample_size),
        "se_boot": float(boots.std(ddof=1)),
    }


def cluster_boot_mean(
    values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int
) -> dict:
    """Bootstrap the mean of `values` resampling whole clusters."""
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        stats.append(np.nanmean(values[sel]))
    stats = np.array(stats)
    est = float(np.nanmean(values))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {
        "estimate": est,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_boot": float(min(1.0, p)),
        "n_clusters": int(len(uniq)),
    }


def twoway_boot_mean(
    values: np.ndarray,
    papers: np.ndarray,
    reviewers: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Two-way cluster bootstrap: multiplicative Poisson-like weights from
    independent paper and reviewer resampling (weighted mean statistic)."""
    up, ur = np.unique(papers), np.unique(reviewers)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        wp = dict(zip(up, rng.multinomial(len(up), np.ones(len(up)) / len(up))))
        wr = dict(zip(ur, rng.multinomial(len(ur), np.ones(len(ur)) / len(ur))))
        w = np.array([wp[p] * wr[r] for p, r in zip(papers, reviewers)], dtype=float)
        if w.sum() == 0:
            continue
        stats.append(np.nansum(values * w) / w.sum())
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    p = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {
        "estimate": float(np.nanmean(values)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_boot": float(min(1.0, p)),
    }


def wilson_ci(k, n, z=1.96, *, clip_lower=False):
    """Wilson score interval, retaining the caller's lower-bound clipping policy."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lower = max(0.0, center - half) if clip_lower else center - half
    return lower, center + half
