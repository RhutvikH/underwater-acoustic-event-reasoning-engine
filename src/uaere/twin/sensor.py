"""Hydrophone + ADC + three fault classes. Oracle health lives here."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from uaere.types import FaultClass, FloatArray, SensorHealth


class Hydrophone:
    def __init__(
        self,
        f_lo_hz: float = 20.0,
        f_hi_hz: float = 7000.0,
        adc_bits: int = 16,
        noise_floor_db: float = -80.0,
        fs: int = 16_000,
    ) -> None:
        self.f_lo_hz = f_lo_hz
        self.f_hi_hz = f_hi_hz
        self.adc_bits = adc_bits
        self.noise_floor_db = noise_floor_db
        self.fs = fs
        lo = max(f_lo_hz / (fs / 2.0), 1e-4)
        hi = min(f_hi_hz / (fs / 2.0), 0.99)
        self._sos = butter(4, [lo, hi], btype="band", output="sos")

    def filter(self, x: FloatArray) -> FloatArray:
        return sosfilt(self._sos, np.asarray(x, dtype=np.float64))


def apply_sensor(
    x: FloatArray,
    hydro: Hydrophone,
    health: SensorHealth,
    rng: np.random.Generator,
) -> tuple[FloatArray, SensorHealth]:
    """Apply hydrophone, ADC, and faults. Returns (observation, oracle health)."""
    y = hydro.filter(x)
    # electronics noise floor
    sigma = 10.0 ** (health.noise_floor_db / 20.0)
    y = y + sigma * rng.standard_normal(len(y))

    fault = health.fault
    if fault is FaultClass.BIAS:
        y = y + 0.15
    elif fault is FaultClass.DROPOUT:
        mask = rng.random(len(y)) < 0.08
        y = y.copy()
        y[mask] = 0.0
    elif fault is FaultClass.GAIN_WANDER:
        wander = 1.0 + 0.25 * np.sin(2 * np.pi * np.arange(len(y)) / max(len(y), 1))
        y = y * wander

    # clock drift: resample by ppm (tiny linear interpolation)
    if abs(health.drift_ppm) > 1e-6:
        n = len(y)
        t = np.arange(n)
        t_src = t * (1.0 + health.drift_ppm * 1e-6)
        t_src = np.clip(t_src, 0, n - 1)
        y = np.interp(t, t_src, y)

    # ADC quantize + clip
    full = 1.0
    levels = 2 ** (health.adc_bits - 1)
    clipped = bool(np.any(np.abs(y) > full))
    y = np.clip(y, -full, full)
    q = np.round(y * levels) / levels
    oracle = SensorHealth(
        noise_floor_db=health.noise_floor_db,
        clipping=clipped or health.clipping,
        drift_ppm=health.drift_ppm,
        adc_bits=health.adc_bits,
        fault=fault,
    )
    return q.astype(np.float64), oracle


def noisy_health_estimate(oracle: SensorHealth, rng: np.random.Generator) -> SensorHealth:
    """What the trust engine is allowed to see. Never the exact oracle."""
    return SensorHealth(
        noise_floor_db=oracle.noise_floor_db + float(rng.normal(0, 3.0)),
        clipping=oracle.clipping if rng.random() > 0.05 else (not oracle.clipping),
        drift_ppm=oracle.drift_ppm + float(rng.normal(0, 0.5)),
        adc_bits=oracle.adc_bits,
        fault=oracle.fault if rng.random() > 0.1 else FaultClass.NONE,
    )
