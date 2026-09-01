"""Render labelled windows from a YAML scenario. Deterministic under seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from uaere.seed import rng as make_rng
from uaere.twin.environment import EnvironmentModel
from uaere.twin.propagate import apply_ice_keel_scatter, propagate
from uaere.twin.sensor import Hydrophone, apply_sensor, noisy_health_estimate
from uaere.twin.sources import synthesize_source
from uaere.types import (
    WINDOW_SECONDS,
    WORKING_FS,
    EventClass,
    FaultClass,
    SensorHealth,
    WindowRecord,
)

SCENARIO_DIR = Path(__file__).resolve().parents[3] / "configs" / "twin"


def load_scenario(name: str) -> dict:
    path = SCENARIO_DIR / f"{name}.yaml"
    if not path.exists():
        # built-in defaults so tests do not depend on repo layout
        return _builtin_scenarios()[name]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _builtin_scenarios() -> dict[str, dict]:
    return {
        "busy_strait": {
            "temperature_c": 11.0,
            "salinity_psu": 31.0,
            "depth_m": 45.0,
            "sea_state": 2,
            "turbulence": 0.15,
            "ice_keel": 0.0,
            "mix": {
                "tug": 0.18,
                "cargo": 0.18,
                "tanker": 0.14,
                "passenger": 0.14,
                "geophony": 0.16,
                "reject": 0.14,
                "artifact": 0.06,
            },
        },
        "calm_coastal": {
            "temperature_c": 16.0,
            "salinity_psu": 33.0,
            "depth_m": 20.0,
            "sea_state": 1,
            "turbulence": 0.05,
            "ice_keel": 0.0,
            "mix": {
                "tug": 0.15,
                "cargo": 0.15,
                "tanker": 0.1,
                "passenger": 0.15,
                "geophony": 0.05,
                "reject": 0.35,
                "artifact": 0.05,
            },
        },
        "high_sea_state": {
            "temperature_c": 8.0,
            "salinity_psu": 34.0,
            "depth_m": 60.0,
            "sea_state": 5,
            "turbulence": 0.45,
            "ice_keel": 0.0,
            "mix": {
                "tug": 0.1,
                "cargo": 0.1,
                "tanker": 0.08,
                "passenger": 0.08,
                "geophony": 0.4,
                "reject": 0.18,
                "artifact": 0.06,
            },
        },
        "sensor_fault_injection": {
            "temperature_c": 12.0,
            "salinity_psu": 34.0,
            "depth_m": 40.0,
            "sea_state": 2,
            "turbulence": 0.1,
            "ice_keel": 0.0,
            "fault_rate": 0.35,
            "mix": {
                "tug": 0.2,
                "cargo": 0.2,
                "tanker": 0.15,
                "passenger": 0.15,
                "geophony": 0.1,
                "reject": 0.1,
                "artifact": 0.1,
            },
        },
        "ice_keel": {
            "temperature_c": 0.5,
            "salinity_psu": 32.0,
            "depth_m": 80.0,
            "sea_state": 3,
            "turbulence": 0.2,
            "ice_keel": 0.6,
            "mix": {
                "tug": 0.15,
                "cargo": 0.15,
                "tanker": 0.15,
                "passenger": 0.05,
                "geophony": 0.25,
                "reject": 0.2,
                "artifact": 0.05,
            },
        },
    }


class TwinRenderer:
    def __init__(self, scenario: str = "busy_strait", seed: int = 0, fs: int = WORKING_FS) -> None:
        self.scenario_name = scenario
        builtins = _builtin_scenarios()
        yaml_path = SCENARIO_DIR / f"{scenario}.yaml"
        if yaml_path.exists():
            self.cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        else:
            self.cfg = builtins.get(scenario, builtins["busy_strait"])
        self.seed = seed
        self.fs = fs
        self.env = EnvironmentModel()
        self.hydro = Hydrophone(fs=fs)
        self.n = int(fs * WINDOW_SECONDS)

    def render_dataset(self, n_windows: int) -> list[WindowRecord]:
        rng = make_rng(self.seed)
        mix = self.cfg["mix"]
        names = list(mix.keys())
        probs = np.array([mix[k] for k in names], dtype=np.float64)
        probs = probs / probs.sum()
        out: list[WindowRecord] = []
        for i in range(n_windows):
            name = str(rng.choice(names, p=probs))
            cls = EventClass(name)
            rec = self._one(cls, rng, i)
            out.append(rec)
        return out

    def _one(self, cls: EventClass, rng: np.random.Generator, idx: int) -> WindowRecord:
        sea = int(self.cfg.get("sea_state", 2))
        # Distant vessels get lower SNR so energy-threshold misses them;
        # geophony in high sea state is loud so energy-threshold false-alarms.
        if cls in {EventClass.TUG, EventClass.TANKER} and rng.random() < 0.4:
            snr_db = float(rng.uniform(0.0, 6.0))
            range_m = float(rng.uniform(800.0, 1600.0))
        elif cls is EventClass.GEOPHONY:
            snr_db = float(rng.uniform(12.0, 22.0))
            range_m = float(rng.uniform(50.0, 200.0))
        elif cls is EventClass.REJECT:
            snr_db = float(rng.uniform(-5.0, 5.0))
            range_m = 1000.0
        else:
            snr_db = float(rng.uniform(6.0, 16.0))
            range_m = float(rng.uniform(200.0, 700.0))

        state = self.env.state(
            temperature_c=float(self.cfg.get("temperature_c", 12.0)),
            salinity_psu=float(self.cfg.get("salinity_psu", 34.0)),
            depth_m=float(self.cfg.get("depth_m", 40.0)),
            sea_state=sea,
            snr_db=snr_db,
            turbulence=float(self.cfg.get("turbulence", 0.1)),
        )
        src = synthesize_source(cls, self.n, self.fs, rng, range_m=range_m)
        src_xyz = (range_m, 0.0, max(5.0, state.depth_m * 0.3))
        hydro_xyz = (0.0, 0.0, state.depth_m)
        recv = propagate(src, src_xyz, hydro_xyz, state, self.fs)
        recv = apply_ice_keel_scatter(recv, float(self.cfg.get("ice_keel", 0.0)), rng)
        ambient = self.env.ambient_noise(self.n, self.fs, state, rng)
        recv = _mix_snr(recv, ambient, snr_db, cls)

        fault = FaultClass.NONE
        fault_rate = float(self.cfg.get("fault_rate", 0.05))
        if rng.random() < fault_rate:
            fault = rng.choice(list(FaultClass))
            if fault is FaultClass.NONE:
                fault = FaultClass.BIAS
        health = SensorHealth(
            noise_floor_db=-80.0 + (4.0 if fault is not FaultClass.NONE else 0.0),
            clipping=cls is EventClass.ARTIFACT,
            drift_ppm=float(rng.normal(0, 0.2)),
            adc_bits=16,
            fault=fault,
        )
        obs, oracle = apply_sensor(recv, self.hydro, health, rng)
        est = noisy_health_estimate(oracle, rng)
        present = cls not in {EventClass.REJECT, EventClass.GEOPHONY, EventClass.ARTIFACT}
        cause = {
            EventClass.TUG: "cause.vessel.tug",
            EventClass.CARGO: "cause.vessel.cargo",
            EventClass.TANKER: "cause.vessel.tanker",
            EventClass.PASSENGER: "cause.vessel.passenger",
            EventClass.BIOLOGICAL: "cause.taxon.odontocete",
            EventClass.GEOPHONY: "cause.env.wind_waves",
            EventClass.ARTIFACT: "cause.sensor.clipping",
            EventClass.REJECT: "cause.env.ambient",
        }[cls]
        return WindowRecord(
            waveform=obs,
            sample_rate=self.fs,
            environment=state,
            health_oracle=oracle,
            health_estimate=est,
            event_present=present,
            event_class=cls if present else EventClass.REJECT,
            cause_id=cause,
            is_artifact=cls is EventClass.ARTIFACT or oracle.clipping,
            source_id=f"{self.scenario_name}-{idx}",
            t0_s=float(idx),
        )


def _mix_snr(sig: np.ndarray, noise: np.ndarray, snr_db: float, cls: EventClass) -> np.ndarray:
    if cls is EventClass.REJECT:
        return noise * 0.3
    ps = np.mean(sig**2) + 1e-12
    pn = np.mean(noise**2) + 1e-12
    target = ps / (10.0 ** (snr_db / 10.0))
    noise_scaled = noise * np.sqrt(target / pn)
    mixed = sig + noise_scaled
    peak = np.max(np.abs(mixed)) + 1e-12
    return mixed / max(peak, 1.0)
