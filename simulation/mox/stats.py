"""Paired bootstrap confidence intervals over the evaluation set.

The unit of resampling is the example: a method's per-example outcome (correct,
confidence, etc.) is compared to the baseline's on the same example, and the
paired difference is bootstrapped. This is the same discipline the P-JEPA
revision used; it keeps the CIs honest about the example-level variance that
dominates a small toy.
"""
from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                        seed: int = 0, alpha: float = 0.05):
    """CI on mean(a - b). Returns (mean, lo, hi, frac_a_ge_b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "mean": float(d.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "frac_a_gt_b": float((a > b).mean()),
        "n": int(n),
    }


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic. ``label`` is
    1 for the positive class (here: the answer is correct). Higher ``score``
    should indicate the positive class. Ties are handled by average rank."""
    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(bool)
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    sum_pos = ranks[label].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_ci(score, label, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label).astype(int)
    rng = np.random.default_rng(seed)
    n = len(score)
    point = auroc(score, label)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = auroc(score[idx], label[idx])
        if not np.isnan(a):
            boots.append(a)
    boots = np.asarray(boots)
    return {
        "auroc": float(point),
        "ci_lo": float(np.quantile(boots, alpha / 2)),
        "ci_hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(n),
    }


def paired_auroc_diff_ci(score_a, score_b, label, n_boot: int = 5000, seed: int = 0,
                         alpha: float = 0.05):
    """CI on AUROC(score_a) - AUROC(score_b), bootstrapping examples jointly."""
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    label = np.asarray(label).astype(int)
    rng = np.random.default_rng(seed)
    n = len(label)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = auroc(score_a[idx], label[idx])
        db = auroc(score_b[idx], label[idx])
        if not (np.isnan(da) or np.isnan(db)):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    return {
        "diff": float(auroc(score_a, label) - auroc(score_b, label)),
        "ci_lo": float(np.quantile(diffs, alpha / 2)),
        "ci_hi": float(np.quantile(diffs, 1 - alpha / 2)),
        "n": int(n),
    }


def mean_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci_lo": float(np.quantile(boots, alpha / 2)),
        "ci_hi": float(np.quantile(boots, 1 - alpha / 2)),
        "n": int(n),
    }
