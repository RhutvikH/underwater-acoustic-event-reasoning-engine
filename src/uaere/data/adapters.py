"""Dataset adapters. DeepShip / ShipsEar are optional; twin-synthetic always works.

Fish clips are a separate loader. A missing fish root is an error for the
fish commands. It must not fall back to the twin.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from uaere.data.split import FileGroup, recording_locked_split
from uaere.data.wavutil import read_wav, resample_linear
from uaere.twin.render import TwinRenderer
from uaere.types import (
    WINDOW_SECONDS,
    WORKING_FS,
    FISH_LABEL_ALIASES,
    NEGATIVE_LABELS,
    EnvironmentState,
    EventClass,
    SensorHealth,
    WindowRecord,
    event_class_for_label,
)

# ToadFishFinder v4, Pamlico Sound. CC0. Not shipped in this repository.
FISH_DATA_DOI = "10.5061/dryad.gtht76hr9"
# Boatwhistle band is under 1 kHz; this default is the trust environment
# when the release has no CTD. It is not a measurement.
FISH_IMPUTED_ENV = EnvironmentState()
MIN_FILES_PER_SPECIES = 40
# Explicit segment markers only. A bare trailing integer is not enough.
_CLIP_SUFFIX = re.compile(r"[_-](?:clip|seg|win|part)\d+$", re.IGNORECASE)
# bw_BB_201351190.170422052002_000.3870833 and t_RB_335831108.180529020002_011.6336667
# The middle token is one hydrophone deployment. The last token is a clip time.
_TOADFISH_RECORDING = re.compile(
    r"^[A-Za-z]+_([A-Za-z0-9]+_\d+\.\d+)_\d+\.\d+$",
    re.IGNORECASE,
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


class WatkinsAdapter:
    """Optional species-folder WAVs, all mapped to EventClass.BIOLOGICAL.

    Not a species-SOTA trainer. Academic use of Watkins Best-of only.
    Missing root ⇒ load() returns [].
    """

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root) if root else None

    def available(self) -> bool:
        return bool(self.root and self.root.exists())

    def load(self) -> list[WindowRecord]:
        if not self.available():
            return []
        recs: list[WindowRecord] = []
        env = EnvironmentState()
        health = SensorHealth()
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            for wav in sorted(folder.glob("*.wav"))[:80]:
                recs.extend(self._windows(wav, env, health))
        return recs

    def _windows(self, path: Path, env: EnvironmentState, health: SensorHealth) -> list[WindowRecord]:
        from uaere.data.wavutil import read_wav, resample_linear
        from uaere.types import WORKING_FS, WINDOW_SECONDS

        x, fs = read_wav(path)
        x = resample_linear(x, fs, WORKING_FS)
        n = int(WORKING_FS * WINDOW_SECONDS)
        hop = n
        out: list[WindowRecord] = []
        t = 0
        i = 0
        while t + n <= len(x) and i < 8:
            out.append(
                WindowRecord(
                    waveform=x[t : t + n],
                    sample_rate=WORKING_FS,
                    environment=env,
                    health_oracle=health,
                    health_estimate=health,
                    event_present=True,
                    event_class=EventClass.BIOLOGICAL,
                    cause_id="cause.taxon.odontocete",
                    is_artifact=False,
                    source_id=path.stem,
                    t0_s=t / WORKING_FS,
                )
            )
            t += hop
            i += 1
        return out


def canonical_fish_label(folder_name: str) -> str:
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    return FISH_LABEL_ALIASES.get(key, key)


def recording_group_id(path: Path) -> str:
    """Deployment id when the ToadFishFinder name has one.

    ``bw_BB_<id>_<time>`` and ``ot_BB_<id>_<time>`` share ``BB_<id>``, so both
    classes of that deployment stay on one side of the split. Otherwise an
    explicit ``_clipN`` suffix is stripped, and a bare trailing number is kept.
    """
    match = _TOADFISH_RECORDING.match(path.stem)
    if match:
        return match.group(1)
    stem = _CLIP_SUFFIX.sub("", path.stem) or path.stem
    return f"{path.parent.name}/{stem}"


def _wavs_in(folder: Path) -> list[Path]:
    found = list(folder.glob("*.wav")) + list(folder.glob("*.WAV"))
    return sorted(found)


class FishCallAdapter:
    """Species folders of real WAVs. One centre 1 s window per file.

    `bwhistle/` is *Opsanus tau*. `other/` is the Pamlico negative class.
    Any other folder name is a species label. Species with fewer than
    `min_files` source files are dropped and listed in the manifest.
    A missing root raises FileNotFoundError. It does not synthesize ships.
    """

    def __init__(
        self,
        root: str | Path,
        seed: int = 0,
        max_per_class: int | None = 3000,
        min_files: int = MIN_FILES_PER_SPECIES,
    ) -> None:
        self.root = Path(root) if root else None
        self.seed = int(seed)
        self.max_per_class = max_per_class
        self.min_files = int(min_files)

    def available(self) -> bool:
        return bool(self.root and self.root.exists())

    def discover(self) -> dict:
        if not self.available():
            raise FileNotFoundError(
                "Fish recordings were not found. Download wavclips_TFv4.zip from "
                f"https://doi.org/{FISH_DATA_DOI} (CC0), unzip it, and pass "
                "--root to the folder that contains bwhistle/ and other/. "
                "This command does not fall back to the synthetic twin."
            )
        assert self.root is not None
        class_dirs = self._class_dirs(self.root)
        if not class_dirs:
            raise FileNotFoundError(
                f"No WAV class folders under {self.root}. Expected bwhistle/ and other/ "
                "after unzipping wavclips_TFv4.zip."
            )
        grouped: dict[str, dict[str, list[Path]]] = {}
        for label, folder in class_dirs:
            for wav in _wavs_in(folder):
                gid = recording_group_id(wav)
                grouped.setdefault(label, {}).setdefault(gid, []).append(wav)
        dropped: dict[str, int] = {}
        groups: list[FileGroup] = []
        for label, by_id in grouped.items():
            n_files = sum(len(paths) for paths in by_id.values())
            if n_files < self.min_files:
                dropped[label] = n_files
                continue
            for gid, paths in sorted(by_id.items()):
                groups.append(FileGroup(gid, label, tuple(sorted(paths))))
        if not groups:
            raise FileNotFoundError(
                f"No class has {self.min_files} or more source files under {self.root}. "
                f"Dropped: {dropped or 'none'}."
            )
        labels = sorted({g.label for g in groups})
        if not any(lb not in NEGATIVE_LABELS for lb in labels):
            raise FileNotFoundError(
                "Fish data has a negative class only. Add a species folder "
                "(bwhistle/ or <species>/*.wav) with enough files."
            )
        if not any(lb in NEGATIVE_LABELS for lb in labels):
            raise FileNotFoundError(
                "Fish data has no 'other' (or reject) folder. "
                "A call-only set cannot train a detector."
            )
        shared = len({g.group_id for g in groups}) < len(groups)
        multi = any(len(g.paths) > 1 for g in groups)
        deployment = any("/" not in g.group_id for g in groups)
        split_unit = "recording" if (shared or multi or deployment) else "file"
        return {
            "groups": groups,
            "dropped_under_min_files": dropped,
            "split_unit": split_unit,
            "labels_before_cap": labels,
            "doi": FISH_DATA_DOI,
            "min_files": self.min_files,
            "environment_imputed": True,
            "imputed_sea_state": FISH_IMPUTED_ENV.sea_state,
        }

    def load_split(
        self,
    ) -> tuple[list[WindowRecord], list[WindowRecord], list[WindowRecord], dict]:
        found = self.discover()
        groups: list[FileGroup] = found["groups"]
        train_g, val_g, test_g = recording_locked_split(
            groups,
            seed=self.seed,
            max_per_class=self.max_per_class,
        )
        meta = {k: v for k, v in found.items() if k != "groups"}
        meta["cap_unit"] = "files"
        meta["split_note"] = (
            "Deployment ids parsed from the file name stay on one side, and both "
            "classes of that deployment stay with them. The cap counts files. A "
            "deployment is kept whole, so the count can stop short of the cap. "
            "This is not a second estuary."
            if found["split_unit"] == "recording"
            else (
                "Every file is its own group. Station and time were not in the "
                "names, so this is not a held-out day and not a second estuary."
            )
        )
        meta["seed"] = self.seed
        meta["max_per_class"] = self.max_per_class
        meta["n_groups"] = {
            "train": len(train_g),
            "val": len(val_g),
            "test": len(test_g),
        }
        train = self._materialise(train_g)
        val = self._materialise(val_g)
        test = self._materialise(test_g)
        meta["n_windows"] = {"train": len(train), "val": len(val), "test": len(test)}
        meta["counts"] = {
            "train": _count(train),
            "val": _count(val),
            "test": _count(test),
        }
        return train, val, test, meta

    def load(self) -> list[WindowRecord]:
        train, val, test, _ = self.load_split()
        return train + val + test

    def _class_dirs(self, root: Path) -> list[tuple[str, Path]]:
        hits: list[Path] = []
        for folder in [root, *root.rglob("*")]:
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            if folder.name == "__MACOSX":
                continue
            if _wavs_in(folder):
                hits.append(folder)
        # Keep only leaves: a folder that contains WAVs and is not a parent of another hit.
        leaves = []
        for folder in hits:
            if any(other != folder and folder in other.parents for other in hits):
                continue
            leaves.append(folder)
        return [(canonical_fish_label(p.name if p != root else p.name), p) for p in leaves]

    def _materialise(self, groups: list[FileGroup]) -> list[WindowRecord]:
        out: list[WindowRecord] = []
        for g in groups:
            for path in g.paths:
                out.append(self._one(path, g.label, g.group_id))
        return out

    def _one(self, path: Path, label: str, group_id: str) -> WindowRecord:
        x, fs = read_wav(path)
        x = resample_linear(x, fs, WORKING_FS)
        n = int(WORKING_FS * WINDOW_SECONDS)
        if len(x) >= n:
            start = (len(x) - n) // 2
            clip = np.asarray(x[start : start + n], dtype=np.float64)
        else:
            clip = np.pad(np.asarray(x, dtype=np.float64), (0, n - len(x)))
        cls = event_class_for_label(label)
        present = label not in NEGATIVE_LABELS
        if label == EventClass.OPSANUS_TAU.value:
            cause = "cause.taxon.opsanus_tau"
        elif label in NEGATIVE_LABELS:
            cause = "cause.env.ambient"
        else:
            cause = f"cause.taxon.{label}"
        health = SensorHealth()
        return WindowRecord(
            waveform=clip,
            sample_rate=WORKING_FS,
            environment=FISH_IMPUTED_ENV,
            health_oracle=health,
            health_estimate=health,
            event_present=present,
            event_class=cls if present else EventClass.OTHER,
            cause_id=cause,
            is_artifact=False,
            source_id=group_id,
            t0_s=0.0,
            taxon=label,
            environment_imputed=True,
        )


def _count(records: list[WindowRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        name = r.taxon or r.event_class.value
        counts[name] = counts.get(name, 0) + 1
    return counts
