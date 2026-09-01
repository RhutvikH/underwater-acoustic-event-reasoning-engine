from __future__ import annotations

import numpy as np


def mcnemar_p(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """McNemar exact-ish mid-p on paired correctness vs gold y."""
    ca = a == y
    cb = b == y
    b01 = int(np.sum((~ca) & cb))
    b10 = int(np.sum(ca & (~cb)))
    n = b01 + b10
    if n == 0:
        return 1.0
    # binomial two-sided
    from scipy.stats import binomtest

    return float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue)


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    stats = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    reject = [False] * m
    for i, idx in enumerate(order):
        if pvalues[idx] <= alpha / (m - i):
            reject[idx] = True
        else:
            break
    return reject
