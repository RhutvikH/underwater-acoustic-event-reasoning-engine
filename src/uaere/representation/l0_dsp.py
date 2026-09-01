"""Always-on L0: energy, ZCR, spectral centroid/bandwidth/flatness, noise residual.

The L0→L1 admit is a logistic of this *vector*, never `if energy > θ`.
"""

from __future__ import annotations

import numpy as np

from uaere.types import FloatArray, L0Features


def extract_l0(x: FloatArray, fs: int, noise_psd: FloatArray | None = None) -> L0Features:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    energy = float(np.mean(x**2))
    zc = np.abs(np.diff(np.signbit(x).astype(np.int8))).sum()
    zcr = float(zc / max(n - 1, 1) * fs / 2.0)

    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    p = spec**2
    p_sum = float(np.sum(p) + 1e-12)
    centroid = float(np.sum(freqs * p) / p_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p) / p_sum))
    geo = float(np.exp(np.mean(np.log(p + 1e-18))))
    arith = float(np.mean(p) + 1e-18)
    flatness = geo / arith

    if noise_psd is None or len(noise_psd) != len(p):
        residual = float(np.mean(p))
    else:
        residual = float(np.mean(np.maximum(p - noise_psd, 0.0)))

    vec = np.array(
        [energy, zcr, centroid, bandwidth, flatness, residual],
        dtype=np.float64,
    )
    return L0Features(
        energy=energy,
        zcr=zcr,
        centroid_hz=centroid,
        bandwidth_hz=bandwidth,
        flatness=flatness,
        noise_residual=residual,
        vector=vec,
    )


def l0_admit_score(features: L0Features, weights: FloatArray | None = None, bias: float = -1.2) -> float:
    """Logistic of φ0. Default weights use shape (flatness, centroid), not energy alone."""
    w = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.array([2.0, 0.4, 0.002, 0.001, -3.0, 4.0], dtype=np.float64)
    )
    z = float(w @ features.vector + bias)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
