"""Geometric spreading + Thorp absorption + 2-path surface bounce.

Not a full Bellhop PE. Documented Urick-class approximation for TRL-4.
"""

from __future__ import annotations

import numpy as np

from uaere.twin.environment import thorp_absorption_db_km
from uaere.types import EnvironmentState, FloatArray


def _delay_filter(delay_s: float, fs: int, n_taps: int = 64) -> FloatArray:
    """Fractional delay via windowed sinc."""
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / fs
    x = (t - delay_s) * fs * np.pi
    h = np.sinc(x / np.pi) * np.hanning(n_taps)
    s = np.sum(h)
    return h / (s if abs(s) > 1e-12 else 1.0)


def propagate(
    source: FloatArray,
    src_xyz: tuple[float, float, float],
    hydro_xyz: tuple[float, float, float],
    state: EnvironmentState,
    fs: int,
    surface_loss_db: float = 6.0,
) -> FloatArray:
    """Return received waveform (still pre-hydrophone)."""
    src = np.asarray(source, dtype=np.float64)
    dx = src_xyz[0] - hydro_xyz[0]
    dy = src_xyz[1] - hydro_xyz[1]
    dz = src_xyz[2] - hydro_xyz[2]
    r = math_dist(dx, dy, dz)
    r = max(r, 1.0)
    c = state.ssp_m_s
    spreading_db = 20.0 * np.log10(r)  # spherical
    # mean-frequency absorption ~ 1 kHz representative for ship noise
    abs_db = thorp_absorption_db_km(1.0) * (r / 1000.0)
    gain = 10.0 ** (-(spreading_db + abs_db) / 20.0)

    delay_s = r / c
    # keep delay as integer samples plus residual (we already pay the range in gain)
    n_delay = int(round(delay_s * fs))
    n_delay = int(np.clip(n_delay, 0, fs // 2))
    direct = np.pad(src * gain, (n_delay, 0))[: len(src)]

    # surface image source
    img = (src_xyz[0], src_xyz[1], -src_xyz[2])
    r2 = math_dist(img[0] - hydro_xyz[0], img[1] - hydro_xyz[1], img[2] - hydro_xyz[2])
    r2 = max(r2, 1.0)
    spreading2 = 20.0 * np.log10(r2)
    abs2 = thorp_absorption_db_km(1.0) * (r2 / 1000.0)
    gain2 = 10.0 ** (-(spreading2 + abs2 + surface_loss_db) / 20.0)
    n2 = int(np.clip(round((r2 / c) * fs), 0, fs // 2))
    bounce = np.pad(src * gain2, (n2, 0))[: len(src)]
    # π phase flip on surface
    recv = direct - bounce
    return recv.astype(np.float64)


def math_dist(dx: float, dy: float, dz: float) -> float:
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def apply_ice_keel_scatter(recv: FloatArray, strength: float, rng: np.random.Generator) -> FloatArray:
    """Optional extra low-frequency scatterer. Off by default. Not a paper claim."""
    if strength <= 0:
        return recv
    n = len(recv)
    lf = rng.standard_normal(n)
    # crude 1/f lift
    spec = np.fft.rfft(lf)
    freqs = np.fft.rfftfreq(n)
    spec *= 1.0 / np.maximum(freqs, 1e-3)
    scatter = np.fft.irfft(spec, n=n)
    scatter /= np.std(scatter) + 1e-12
    return recv + strength * 0.05 * scatter
