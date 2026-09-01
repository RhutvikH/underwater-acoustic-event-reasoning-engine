"""Environment submodel: Mackenzie SSP, Thorp absorption, Knudsen wind noise.

Citations (documented for the paper, not imported):
- Mackenzie, J. Acoust. Soc. Am. 70(3), 1981.
- Thorp, J. Acoust. Soc. Am. 42, 1967 (absorption).
- Knudsen / Wenz deep-water ambient noise curves.
"""

from __future__ import annotations

import numpy as np

from uaere.types import EnvironmentState, FloatArray


def mackenzie_ssp(temperature_c: float, salinity_psu: float, depth_m: float) -> float:
    """Sound speed (m/s). Mackenzie 1981 nine-term equation."""
    t = temperature_c
    s = salinity_psu
    d = depth_m
    return (
        1448.96
        + 4.591 * t
        - 5.304e-2 * t**2
        + 2.374e-4 * t**3
        + 1.340 * (s - 35.0)
        + 1.630e-2 * d
        + 1.675e-7 * d**2
        - 1.025e-2 * t * (s - 35.0)
        - 7.139e-13 * t * d**3
    )


def thorp_absorption_db_km(f_khz: float) -> float:
    """Thorp absorption in dB/km. f in kHz."""
    f = max(float(f_khz), 1e-6)
    f2 = f * f
    return 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003


def knudsen_psd(freq_hz: FloatArray, sea_state: int) -> FloatArray:
    """One-sided ambient PSD (linear power per Hz), relative units.

    Sea-state 0–6 lifts the level ~5 dB per step and slightly reddens.
    Turbulence is applied by the caller as a high-frequency lift.
    """
    f = np.maximum(np.asarray(freq_hz, dtype=np.float64), 1.0)
    ss = int(np.clip(sea_state, 0, 6))
    level_db = 44.0 + 5.0 * ss - 17.0 * np.log10(f / 1000.0)
    return 10.0 ** (level_db / 10.0)


class EnvironmentModel:
    def state(
        self,
        temperature_c: float = 12.0,
        salinity_psu: float = 34.0,
        depth_m: float = 40.0,
        sea_state: int = 2,
        snr_db: float = 10.0,
        turbulence: float = 0.1,
    ) -> EnvironmentState:
        ssp = mackenzie_ssp(temperature_c, salinity_psu, depth_m)
        return EnvironmentState(
            temperature_c=temperature_c,
            salinity_psu=salinity_psu,
            depth_m=depth_m,
            sea_state=int(np.clip(sea_state, 0, 6)),
            snr_db=snr_db,
            turbulence=float(np.clip(turbulence, 0.0, 1.0)),
            ssp_m_s=ssp,
        )

    def ambient_noise(
        self,
        n: int,
        fs: int,
        state: EnvironmentState,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Coloured ambient via frequency-domain shaping of white noise."""
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = knudsen_psd(freqs, state.sea_state)
        # turbulence lifts > 2 kHz (flow-acoustic stress, not a new module)
        lift = 1.0 + 8.0 * state.turbulence * (freqs / max(fs / 2.0, 1.0)) ** 2
        shaped = spec * np.sqrt(psd * lift)
        noise = np.fft.irfft(shaped, n=n)
        # unit-variance then scale later against source by SNR
        std = np.std(noise) + 1e-12
        return (noise / std).astype(np.float64)
