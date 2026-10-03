"""Statistical utilities for rigorous optimizer comparison.

- Wilcoxon signed-rank test for pairwise algorithm comparison across paired
  trials (paired by seed/initializer).
- Holm-Bonferroni step-down correction for families of comparisons.
- Summary statistics (median, IQR, std) used by all benchmark reports.

Falls back gracefully when scipy is unavailable (Holm still works; Wilcoxon
returns NaN with a warning instead of failing the whole benchmark).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False


def summarize(values: Sequence[float]) -> Dict[str, float]:
    """Median/IQR/std/min of a trial vector."""
    v = np.asarray(list(values), dtype=np.float64)
    return {
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "best": float(np.min(v)),
        "mean": float(np.mean(v)),
        "n": int(len(v)),
    }


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Tuple[float, str]:
    """Two-sided Wilcoxon signed-rank p-value between paired trial scores.

    Pairs are (x[i], y[i]) for the same trial/seed. Returns (p_value, note).
    If all differences are zero or scipy is missing, returns (nan, note).
    """
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("Paired comparison requires equal-length samples.")
    diff = x - y
    if np.all(diff == 0):
        return float("nan"), "identical"
    if not _SCIPY:
        return float("nan"), "scipy-missing"
    if len(x) < 6:
        return float("nan"), "n<6"
    try:
        res = _scipy_stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue), "ok"
    except ValueError as exc:  # pragma: no cover
        return float("nan"), f"error:{exc}"


def holm_adjust(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down), NaN-safe."""
    p = np.asarray(list(pvals), dtype=np.float64)
    adjusted = np.full_like(p, np.nan)
    valid_idx = np.where(~np.isnan(p))[0]
    if valid_idx.size == 0:
        return adjusted
    pv = p[valid_idx]
    order = np.argsort(pv)
    m = len(pv)
    running_max = -np.inf
    out = np.empty_like(pv)
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pv[idx])
        running_max = max(running_max, adj)
        out[idx] = running_max
    adjusted[valid_idx] = out
    return adjusted


def significance_stars(p: float) -> str:
    """Conventional star annotation for an (adjusted) p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
