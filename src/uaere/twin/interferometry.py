"""Ambient-noise interferometry atlas (Claerbout–Wapenaar → UASN).

On non-event frames, cross-correlate two nodes' waveforms. Weak peak ⇒
learned death zone. No Bellhop, no SSP inversion.
"""

from __future__ import annotations

import numpy as np

from uaere.types import FloatArray


def peak_coherence(a: FloatArray, b: FloatArray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na = float(np.linalg.norm(a) + 1e-12)
    nb = float(np.linalg.norm(b) + 1e-12)
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    g = np.fft.irfft(fa * np.conj(fb), n=n)
    return float(np.max(np.abs(g)) / (na * nb))


class InterferometricAtlas:
    def __init__(self, node_ids: list[str], alpha: float = 0.2, dead_thresh: float = 0.04) -> None:
        self.alpha = float(alpha)
        self.dead_thresh = float(dead_thresh)
        self.strength: dict[tuple[str, str], float] = {}
        ids = list(node_ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                self.strength[self._key(a, b)] = 1.0  # open until evidence

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def update(self, a: str, wa: FloatArray, b: str, wb: FloatArray, is_event: bool) -> None:
        if is_event or a == b:
            return
        s = peak_coherence(wa, wb)
        k = self._key(a, b)
        prev = self.strength.get(k, s)
        self.strength[k] = (1.0 - self.alpha) * prev + self.alpha * s

    def is_dead(self, a: str, b: str) -> bool:
        return self.strength.get(self._key(a, b), 1.0) < self.dead_thresh

    def link_strength(self, a: str, b: str) -> float:
        return float(self.strength.get(self._key(a, b), 1.0))
