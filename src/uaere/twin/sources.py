"""Class-conditional source signatures. Procedural, not copyrighted recordings.

These are deliberately *spectrally* distinct so the trust engine can beat a
pure energy threshold: high-energy geophony is not a vessel, and a distant
tug is a vessel even at moderate energy.
"""

from __future__ import annotations

import numpy as np

from uaere.types import EventClass, FloatArray


def synthesize_source(
    event_class: EventClass,
    n: int,
    fs: int,
    rng: np.random.Generator,
    range_m: float = 400.0,
) -> FloatArray:
    t = np.arange(n) / fs
    if event_class is EventClass.TUG:
        # strong blade-rate harmonics ~ 18 Hz with low-freq machinery
        blade = 18.0
        x = (
            0.6 * np.sin(2 * np.pi * blade * t)
            + 0.4 * np.sin(2 * np.pi * 2 * blade * t)
            + 0.25 * np.sin(2 * np.pi * 3 * blade * t)
            + 0.15 * _band_noise(n, fs, 40, 400, rng)
        )
    elif event_class is EventClass.TANKER:
        x = 0.7 * _band_noise(n, fs, 20, 180, rng) + 0.2 * np.sin(2 * np.pi * 12.0 * t)
    elif event_class is EventClass.CARGO:
        x = 0.55 * _band_noise(n, fs, 30, 800, rng) + 0.2 * np.sin(2 * np.pi * 25.0 * t)
    elif event_class is EventClass.PASSENGER:
        x = 0.5 * _band_noise(n, fs, 80, 2500, rng) + 0.15 * np.sin(2 * np.pi * 60.0 * t)
    elif event_class is EventClass.BIOLOGICAL:
        # upsweep click-train caricature
        f0 = 400 + 200 * np.sin(2 * np.pi * 2.0 * t)
        x = 0.4 * np.sin(2 * np.pi * np.cumsum(f0) / fs)
        pulses = (np.sin(2 * np.pi * 4.0 * t) > 0.7).astype(np.float64)
        x = x * pulses
    elif event_class is EventClass.GEOPHONY:
        x = _band_noise(n, fs, 800, 6000, rng)
    elif event_class is EventClass.ARTIFACT:
        x = np.zeros(n)
        clicks = rng.integers(0, n, size=12)
        x[clicks] = rng.choice([-1.0, 1.0], size=12)
    else:  # REJECT / ambient placeholder; renderer overlays twin ambient
        x = 0.05 * rng.standard_normal(n)
    x = x / (np.max(np.abs(x)) + 1e-12)
    # range only scales source before propagate(); keep unit-ish here
    return x.astype(np.float64) * (400.0 / max(range_m, 1.0)) ** 0.15


def _band_noise(
    n: int, fs: int, f_lo: float, f_hi: float, rng: np.random.Generator
) -> FloatArray:
    w = rng.standard_normal(n)
    spec = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = ((freqs >= f_lo) & (freqs <= f_hi)).astype(np.float64)
    spec *= band
    y = np.fft.irfft(spec, n=n)
    return y / (np.std(y) + 1e-12)
