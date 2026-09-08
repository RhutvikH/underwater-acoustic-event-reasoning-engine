"""Precision-weighted predictive coding wake (Rao–Ballard / Friston → UASN).

The generative model is the twin's Knudsen/Wenz ambient for the current
sea-state. Surprise is quadratic prediction error with precision Π high
where the ocean *should* be quiet. Storms match the model (low surprise,
high energy). A distant tug is a structured residual (high surprise, low
energy). This is the wake. It is not `if energy > theta`.
"""

from __future__ import annotations

import numpy as np

from uaere.twin.environment import knudsen_psd
from uaere.types import EnvironmentState, FloatArray


def log_spectrum(x: FloatArray, fs: int) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    p = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2 + 1e-18
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, np.log(p)


def precision(freqs: FloatArray, state: EnvironmentState) -> FloatArray:
    """Π_k = 1 / predicted_psd_k. High precision in bands the model says are quiet."""
    pred = knudsen_psd(freqs, state.sea_state)
    lift = 1.0 + 8.0 * state.turbulence * (freqs / max(float(freqs.max()), 1.0)) ** 2
    return 1.0 / (pred * lift + 1e-8)


def surprise(x: FloatArray, fs: int, state: EnvironmentState) -> float:
    """S = (1/K) e^T Π e, plus a harmonic peakiness bonus in 10–400 Hz."""
    freqs, logp = log_spectrum(x, fs)
    pred = knudsen_psd(freqs, state.sea_state)
    lift = 1.0 + 8.0 * state.turbulence * (freqs / max(float(freqs.max()), 1.0)) ** 2
    log_pred = np.log(pred * lift + 1e-18)
    e = logp - log_pred
    pi = precision(freqs, state)
    s_env = float(np.mean(pi * e * e))
    band = (freqs >= 10.0) & (freqs <= 400.0)
    if np.any(band):
        low = np.exp(logp[band])
        peakiness = float(np.max(low) / (np.mean(low) + 1e-12))
    else:
        peakiness = 1.0
    return s_env + float(np.log(peakiness + 1e-6))


def surprise_admit(S: float, center: float = 8.0, scale: float = 1.5) -> float:
    """Map surprise to (0, 1) for the L0 gate. Center is not an energy threshold."""
    z = (float(S) - center) / scale
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))
