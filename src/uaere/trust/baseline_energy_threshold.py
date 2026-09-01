"""Conventional competitor: STFT-band energy ≷ θ.

This file is the ONLY production-adjacent place a fixed energy threshold is
allowed. Eval imports it; the live pipeline must not.
"""

from __future__ import annotations

import numpy as np

from uaere.representation.l0_dsp import extract_l0
from uaere.types import FloatArray, WindowRecord


class EnergyThresholdBaseline:
    def __init__(self, theta: float | None = None) -> None:
        self.theta = theta

    def fit(self, records: list[WindowRecord]) -> float:
        energies = np.array([extract_l0(r.waveform, r.sample_rate).energy for r in records])
        labels = np.array([1.0 if r.event_present else 0.0 for r in records])
        # sweep unique energies as candidate θ, pick Youden on train
        order = np.argsort(energies)
        e = energies[order]
        y = labels[order]
        best_j, best_t = -1.0, float(np.median(e))
        for i in range(len(e)):
            t = e[i]
            pred = energies >= t
            tp = np.mean(pred & (labels == 1))
            tn = np.mean((~pred) & (labels == 0))
            # Youden-like: TPR + TNR - 1 approximated via class-conditional
            tpr = np.sum(pred & (labels == 1)) / max(np.sum(labels == 1), 1)
            tnr = np.sum((~pred) & (labels == 0)) / max(np.sum(labels == 0), 1)
            j = tpr + tnr - 1.0
            if j > best_j:
                best_j, best_t = j, float(t)
            _ = tp, tn, y
        self.theta = best_t
        return best_t

    def scores(self, records: list[WindowRecord]) -> FloatArray:
        return np.array(
            [extract_l0(r.waveform, r.sample_rate).energy for r in records],
            dtype=np.float64,
        )

    def predict(self, records: list[WindowRecord]) -> FloatArray:
        if self.theta is None:
            raise RuntimeError("fit() the energy-threshold baseline first")
        return (self.scores(records) >= self.theta).astype(np.float64)
