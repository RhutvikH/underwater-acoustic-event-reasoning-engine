"""Dataset adapters. DeepShip / ShipsEar are optional; twin-synthetic always works."""

from __future__ import annotations

from pathlib import Path

from uaere.data.wavutil import read_wav, resample_linear
from uaere.twin.render import TwinRenderer
from uaere.types import (
    WINDOW_SECONDS,
    WORKING_FS,
    EnvironmentState,
    EventClass,
    SensorHealth,
    WindowRecord,
)

_CLASS_FROM_FOLDER = {
    "cargo": EventClass.CARGO,
    "passengership": EventClass.PASSENGER,
    "passenger": EventClass.PASSENGER,
    "tanker": EventClass.TANKER,
    "tug": EventClass.TUG,
    "oil tanker": EventClass.TANKER,
}


class TwinReplayAdapter:
    def __init__(self, scenario: str = "busy_strait", seed: int = 0, n_windows: int = 256) -> None:
        self.scenario = scenario
        self.seed = seed
        self.n_windows = n_windows

    def load(self) -> list[WindowRecord]:
        return TwinRenderer(self.scenario, seed=self.seed).render_dataset(self.n_windows)

    def summarize(self, records: list[WindowRecord] | None = None) -> dict[str, int]:
        recs = records if records is not None else self.load()
        counts: dict[str, int] = {}
        for r in recs:
            counts[r.event_class.value] = counts.get(r.event_class.value, 0) + 1
        return counts


class DeepShipAdapter:
    """Expects class folders of WAV files. Time split by filename sort (lexical ~ time)."""

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None

    def available(self) -> bool:
        return bool(self.root and self.root.exists())

    def load(self) -> list[WindowRecord]:
        if not self.available():
            return []
        recs: list[WindowRecord] = []
        for folder in sorted(self.root.iterdir()):
            key = folder.name.lower().replace("_", "").replace("-", "")
            mapped = None
            for k, v in _CLASS_FROM_FOLDER.items():
                if k.replace(" ", "") in key:
                    mapped = v
                    break
            if mapped is None or not folder.is_dir():
                continue
            for wav in sorted(folder.glob("*.wav")):
                recs.extend(self._windows(wav, mapped))
        return recs

    def _windows(self, path: Path, cls: EventClass) -> list[WindowRecord]:
        x, fs = read_wav(path)
        x = resample_linear(x, fs, WORKING_FS)
        n = int(WORKING_FS * WINDOW_SECONDS)
        hop = n // 2
        out: list[WindowRecord] = []
        t = 0
        i = 0
        env = EnvironmentState()
        health = SensorHealth()
        while t + n <= len(x):
            out.append(
                WindowRecord(
                    waveform=x[t : t + n],
                    sample_rate=WORKING_FS,
                    environment=env,
                    health_oracle=health,
                    health_estimate=health,
                    event_present=True,
                    event_class=cls,
                    cause_id=f"cause.vessel.{cls.value}",
                    is_artifact=False,
                    source_id=path.stem,
                    t0_s=t / WORKING_FS,
                )
            )
            t += hop
            i += 1
            if i > 10_000:
                break
        return out


class ShipsEarAdapter(DeepShipAdapter):
    """Same windowing; folder names mapped through the marine ontology."""
