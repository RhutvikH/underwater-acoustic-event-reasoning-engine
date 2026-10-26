"""Class-conditional source signatures. Procedural, not copyrighted recordings.

These are deliberately *spectrally* distinct so the trust engine can beat a
pure energy threshold: high-energy geophony is not an event, a distant tug
is still a vessel, and a mysticete moan is not a click train.
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
        # generic odontocete click-train caricature
        x = _click_train(t, n, fs, rate_hz=8.0, f_lo=800.0, f_hi=5000.0, rng=rng)
    elif event_class is EventClass.WHALE:
        # mysticete moan: slow FM 80–400 Hz (humpback-ish, 1 s window)
        f0 = 120.0 + 180.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
        x = 0.7 * np.sin(2 * np.pi * np.cumsum(f0) / fs)
        x += 0.2 * np.sin(2 * np.pi * 2.0 * np.cumsum(f0) / fs)
        x += 0.08 * _band_noise(n, fs, 60, 500, rng)
    elif event_class is EventClass.DOLPHIN:
        x = _click_train(t, n, fs, rate_hz=40.0, f_lo=2000.0, f_hi=8000.0, rng=rng)
    elif event_class is EventClass.SPERM_WHALE:
        # slow loud clicks ~2 Hz, 2–4 kHz
        x = _click_train(t, n, fs, rate_hz=2.0, f_lo=1500.0, f_hi=4500.0, rng=rng)
        x *= 1.3
    elif event_class is EventClass.SEAL:
        f0 = 250.0 + 80.0 * np.sin(2 * np.pi * 3.0 * t)
        x = 0.55 * np.sin(2 * np.pi * np.cumsum(f0) / fs)
        x += 0.25 * _band_noise(n, fs, 100, 900, rng)
    elif event_class is EventClass.FISH:
        # toadfish / croaker chorus: 200–600 Hz pulses
        env = (np.sin(2 * np.pi * 6.0 * t) > 0.2).astype(np.float64)
        x = 0.5 * np.sin(2 * np.pi * 280.0 * t) * env
        x += 0.2 * _band_noise(n, fs, 180, 700, rng)
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


def _click_train(
    t: FloatArray,
    n: int,
    fs: int,
    rate_hz: float,
    f_lo: float,
    f_hi: float,
    rng: np.random.Generator,
) -> FloatArray:
    pulse = (np.sin(2 * np.pi * rate_hz * t) > 0.75).astype(np.float64)
    carrier = np.sin(2 * np.pi * 0.5 * (f_lo + f_hi) * t)
    return 0.55 * carrier * pulse + 0.12 * _band_noise(n, fs, f_lo, f_hi, rng)


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
