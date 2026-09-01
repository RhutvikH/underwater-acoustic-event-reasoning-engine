"""Calibration and detection metrics used in Objective 1 / non-negotiable 6."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from uaere.types import FloatArray


def ece(y: FloatArray, p: FloatArray, n_bins: int = 10) -> float:
    """Expected calibration error. Identity: ECE ∈ [0, 1]."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    w = np.zeros(n_bins)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(m):
            continue
        acc[i] = np.mean(y[m])
        conf[i] = np.mean(p[m])
        w[i] = np.mean(m)
    return float(np.sum(w * np.abs(acc - conf)))


def mce(y: FloatArray, p: FloatArray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if np.any(m):
            gaps.append(abs(float(np.mean(y[m]) - np.mean(p[m]))))
    return float(max(gaps) if gaps else 0.0)


def brier(y: FloatArray, p: FloatArray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(np.mean((y - p) ** 2))


def reliability_bins(
    y: FloatArray, p: FloatArray, n_bins: int = 10
) -> tuple[FloatArray, FloatArray, FloatArray]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        cnt[i] = np.sum(m)
        if np.any(m):
            acc[i] = np.mean(y[m])
    return centres, acc, cnt


def roc_auc(y: FloatArray, p: FloatArray) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))
