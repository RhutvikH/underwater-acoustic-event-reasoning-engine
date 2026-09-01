"""Post-hoc temperature / vector scaling on a held-out calibration split."""

from __future__ import annotations

import numpy as np

from uaere.types import FloatArray


def temperature_scale(logits: FloatArray, y: FloatArray) -> float:
    """Grid-search T > 0 that minimises NLL of softmax(logits / T)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.25, 8.0, 24):
        p = _softmax(z / t)
        nll = _nll(p, y)
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def vector_scale(logits: FloatArray, y: FloatArray) -> FloatArray:
    """Per-class temperature (diagonal)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    k = z.shape[1]
    scales = np.ones(k)
    for j in range(k):
        best, best_nll = 1.0, float("inf")
        for t in np.geomspace(0.25, 8.0, 16):
            s = scales.copy()
            s[j] = t
            p = _softmax(z / s)
            nll = _nll(p, y)
            if nll < best_nll:
                best, best_nll = float(t), nll
        scales[j] = best
    return scales


def _softmax(z: FloatArray) -> FloatArray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _nll(p: FloatArray, y: FloatArray) -> float:
    rows = np.arange(len(y))
    return float(-np.mean(np.log(p[rows, y] + 1e-12)))
