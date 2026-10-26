"""Folder-per-species recordings. Watkins tape ids stay on one side of the split.

A missing root is an error. This loader does not synthesize ships and does
not require a negative class. FishSounds is not a training set.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from uaere.data.adapters import FISH_IMPUTED_ENV, recording_group_id
from uaere.data.split import FileGroup, label_stratified_recording_split
from uaere.data.wavutil import read_wav_centre, resample_linear
from uaere.types import (
    WINDOW_SECONDS,
    SensorHealth,
    WindowRecord,
    event_class_for_label,
)

WATKINS_CITATION = (
    "Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution. "
    "Academic use, not commercial."
)
# Click-specialist names. Aliased porpoise audio is not a whistle class.
_ULTRASONIC_NAME = re.compile(r"porpoise|phocoena|phocoenoides|vaquita", re.IGNORECASE)
# 83093034 → tape 83093. 6102500A and 6102500B share tape 61025.
_TAPE_TOKEN = re.compile(r"^(\d{5})[0-9A-Za-z]+$")

SPECIES_MIN_FILES = 40
SPECIES_MIN_GROUPS = 8
SPECIES_MIN_CLASSES = 10
SPECIES_MAX_PER_CLASS = 500
BAND_SAMPLE_FILES = 8
BAND_POWER_FRACTION = 0.50


class SpeciesDataError(FileNotFoundError):
    """The species tree is missing or does not clear the declared floors."""


def slug_label(name: str) -> str:
    text = name.strip().lower().replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def watkins_tape_id(path: Path) -> str | None:
    """Five-digit catalog tape. The rest of the token is the cut, not a new recording."""
    for part in reversed(path.stem.split("_")):
        match = _TAPE_TOKEN.match(part)
        if match:
            return match.group(1)
    return None


def _wavs_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.wav")) + sorted(folder.glob("*.WAV"))


def class_dirs(root: Path) -> list[tuple[str, Path]]:
    hits: list[Path] = []
    for folder in [root, *root.rglob("*")]:
        if not folder.is_dir() or folder.name.startswith(".") or folder.name == "__MACOSX":
            continue
        if _wavs_in(folder):
            hits.append(folder)
    leaves = [
        folder
        for folder in hits
        if not any(other != folder and folder in other.parents for other in hits)
    ]
    return [(slug_label(folder.name), folder) for folder in leaves]


def load_group_overrides(root: Path) -> dict[str, str]:
    path = root / "groups.csv"
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "file" not in reader.fieldnames or "group_id" not in reader.fieldnames:
            raise SpeciesDataError(f"{path} must have columns file,group_id")
        for row in reader:
            key = (row.get("file") or "").strip()
            gid = (row.get("group_id") or "").strip()
            if not key or not gid:
                continue
            out[key] = gid
            out[Path(key).name] = gid
    return out


def group_for(path: Path, root: Path, overrides: dict[str, str]) -> tuple[str, str]:
    """Return (group_id, how). how is csv, tape, or file."""
    rel = str(path.relative_to(root))
    if rel in overrides:
        return overrides[rel], "csv"
    if path.name in overrides:
        return overrides[path.name], "csv"
    tape = watkins_tape_id(path)
    if tape:
        return tape, "tape"
    return recording_group_id(path), "file"


def _centre(x: np.ndarray, fs: int, seconds: float) -> np.ndarray:
    n = int(fs * seconds)
    if n <= 0 or len(x) <= n:
        return np.asarray(x, dtype=np.float64)
    start = (len(x) - n) // 2
    return np.asarray(x[start : start + n], dtype=np.float64)


def power_above(x: np.ndarray, fs: int, cutoff_hz: float) -> float:
    if fs / 2.0 <= cutoff_hz:
        return 0.0
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    total = float(spec.sum()) + 1e-18
    return float(spec[freqs >= cutoff_hz].sum() / total)


def _band_fraction(paths: tuple[Path, ...], cutoff_hz: float, seed: int) -> float | None:
    if not paths:
        return None
    rng = np.random.default_rng(seed)
    order = list(paths)
    rng.shuffle(order)
    fractions = []
    for path in order[:BAND_SAMPLE_FILES]:
        try:
            x, fs = read_wav_centre(path, seconds=1.0)
        except (OSError, ValueError):
            continue
        clip = _centre(x, fs, 1.0)
        if len(clip) < 16:
            continue
        fractions.append(power_above(clip, fs, cutoff_hz))
    if not fractions:
        return None
    return float(np.median(fractions))


class SpeciesFolderAdapter:
    """One folder per species. No negative class is required.

    ``require_recording_ids`` is for the paper suite: a tree whose names do
    not carry a tape id, and that has no groups.csv, is refused. A custom
    tree may still train with ``split_unit=file``, and the report says so.
    """

    def __init__(
        self,
        root: str | Path,
        seed: int = 0,
        max_per_class: int | None = SPECIES_MAX_PER_CLASS,
        min_files: int = SPECIES_MIN_FILES,
        min_groups: int = SPECIES_MIN_GROUPS,
        min_classes: int = SPECIES_MIN_CLASSES,
        working_fs: int = 16_000,
        cutoff_hz: float = 7_000.0,
        require_recording_ids: bool = False,
    ) -> None:
        self.root = Path(root) if root else None
        self.seed = int(seed)
        self.max_per_class = max_per_class
        self.min_files = int(min_files)
        self.min_groups = int(min_groups)
        self.min_classes = int(min_classes)
        self.working_fs = int(working_fs)
        self.cutoff_hz = float(cutoff_hz)
        self.require_recording_ids = bool(require_recording_ids)

    def available(self) -> bool:
        return bool(self.root and self.root.exists())

    def discover(self) -> dict:
        if not self.available():
            raise SpeciesDataError(
                "Species recordings were not found. Pass --root to a folder of "
                "<species>/*.wav, or unpack the Watkins full-cuts archive under data/watkins. "
                "This command does not fall back to the synthetic twin or the toadfish head."
            )
        assert self.root is not None
        overrides = load_group_overrides(self.root)
        dirs = class_dirs(self.root)
        if not dirs:
            raise SpeciesDataError(
                f"No WAV class folders under {self.root}. Expected one folder per species."
            )
        grouped: dict[str, dict[str, list[Path]]] = {}
        how_counts = {"csv": 0, "tape": 0, "file": 0}
        skipped_ultrasonic: list[str] = []
        for label, folder in dirs:
            if _ULTRASONIC_NAME.search(label):
                skipped_ultrasonic.append(label)
                continue
            for wav in _wavs_in(folder):
                gid, how = group_for(wav, self.root, overrides)
                how_counts[how] = how_counts.get(how, 0) + 1
                grouped.setdefault(label, {}).setdefault(gid, []).append(wav)
        if not grouped:
            raise SpeciesDataError(
                f"No eligible species folders under {self.root}. "
                "Porpoise names are excluded because their clicks sit above this front end."
            )
        groups: list[FileGroup] = []
        rows = []
        for label, by_id in sorted(grouped.items()):
            n_files = sum(len(paths) for paths in by_id.values())
            n_groups = len(by_id)
            reason = ""
            if n_files < self.min_files:
                reason = "below_min_files"
            elif n_groups < self.min_groups:
                reason = "below_min_groups"
            else:
                for gid, paths in sorted(by_id.items()):
                    groups.append(FileGroup(gid, label, tuple(sorted(paths))))
            rows.append(
                {
                    "label": label,
                    "n_files": n_files,
                    "n_groups": n_groups,
                    "eligible_counts": reason == "",
                    "count_reason": reason,
                }
            )
        n_ok = sum(1 for row in rows if row["eligible_counts"])
        identified = how_counts["csv"] + how_counts["tape"]
        total = sum(how_counts.values())
        recording_ids = bool(overrides) or (total > 0 and identified / total >= 0.5)
        if self.require_recording_ids and not recording_ids:
            raise SpeciesDataError(
                "Recording ids were not found in the file names and groups.csv is missing. "
                "Splitting cuts from the same tape into train and test is refused. "
                "Add groups.csv with columns file,group_id, or use the Watkins tape names."
            )
        if n_ok < self.min_classes:
            raise SpeciesDataError(
                f"{n_ok} species clear min_files={self.min_files} and min_groups={self.min_groups}. "
                f"Need {self.min_classes}. Floors are not lowered. FishSounds is not used to pad."
            )
        split_unit = "recording" if recording_ids else "file"
        return {
            "groups": groups,
            "rows": rows,
            "how_counts": how_counts,
            "recording_ids": recording_ids,
            "split_unit": split_unit,
            "overrides": bool(overrides),
            "working_fs": self.working_fs,
            "cutoff_hz": self.cutoff_hz,
            "skipped_ultrasonic": skipped_ultrasonic,
        }

    def apply_band(self, found: dict) -> dict:
        """Drop species whose sampled energy sits above the working band. Not an accuracy filter."""
        groups: list[FileGroup] = found["groups"]
        by_label: dict[str, list[FileGroup]] = {}
        for g in groups:
            by_label.setdefault(g.label, []).append(g)
        kept: list[FileGroup] = []
        band_rows = []
        for row in found["rows"]:
            label = row["label"]
            if not row["eligible_counts"]:
                band_rows.append({**row, "band_ok": False, "power_above": None})
                continue
            paths: list[Path] = []
            for g in by_label.get(label, []):
                paths.extend(g.paths)
            frac = _band_fraction(tuple(paths), self.cutoff_hz, self.seed)
            ok = frac is not None and frac <= BAND_POWER_FRACTION
            band_rows.append({**row, "band_ok": ok, "power_above": frac})
            if ok:
                kept.extend(by_label.get(label, []))
        n_band = sum(1 for row in band_rows if row["band_ok"])
        if n_band < self.min_classes:
            raise SpeciesDataError(
                f"{n_band} species sit inside {self.cutoff_hz:.0f} Hz at {self.working_fs} Hz. "
                f"Need {self.min_classes}. This is the band floor, not a test-set drop."
            )
        out = dict(found)
        out["groups"] = kept
        out["rows"] = band_rows
        out["n_band"] = n_band
        return out

    def load_split(
        self,
    ) -> tuple[list[WindowRecord], list[WindowRecord], list[WindowRecord], dict]:
        found = self.apply_band(self.discover())
        groups: list[FileGroup] = found["groups"]
        train_g, val_g, test_g = label_stratified_recording_split(
            groups, seed=self.seed, max_per_class=self.max_per_class
        )
        meta = {k: v for k, v in found.items() if k != "groups"}
        meta["cap_unit"] = "files"
        meta["seed"] = self.seed
        meta["max_per_class"] = self.max_per_class
        meta["min_files"] = self.min_files
        meta["min_groups"] = self.min_groups
        meta["min_classes"] = self.min_classes
        meta["citation"] = WATKINS_CITATION
        meta["split_note"] = (
            "Tape or groups.csv ids stay on one side. The split is stratified per species "
            "by file mass. This is call-library identification, not a raw soundscape."
            if found["split_unit"] == "recording"
            else (
                "No tape id or groups.csv was found, so each file is its own group. "
                "This is not a held-out recording."
            )
        )
        meta["n_groups"] = {"train": len(train_g), "val": len(val_g), "test": len(test_g)}
        train = self._materialise(train_g)
        val = self._materialise(val_g)
        test = self._materialise(test_g)
        meta["n_windows"] = {"train": len(train), "val": len(val), "test": len(test)}
        meta["counts"] = {"train": _count(train), "val": _count(val), "test": _count(test)}
        meta["labels"] = sorted({r.taxon for r in train + val + test})
        return train, val, test, meta

    def load(self) -> list[WindowRecord]:
        train, val, test, _ = self.load_split()
        return train + val + test

    def _materialise(self, groups: list[FileGroup]) -> list[WindowRecord]:
        out: list[WindowRecord] = []
        for g in groups:
            for path in g.paths:
                out.append(self._one(path, g.label, g.group_id))
        return out

    def _one(self, path: Path, label: str, group_id: str) -> WindowRecord:
        x, fs = read_wav_centre(path, seconds=WINDOW_SECONDS, pad_s=0.05)
        x = resample_linear(x, fs, self.working_fs)
        n = int(self.working_fs * WINDOW_SECONDS)
        if len(x) >= n:
            start = (len(x) - n) // 2
            clip = np.asarray(x[start : start + n], dtype=np.float64)
        else:
            clip = np.pad(np.asarray(x, dtype=np.float64), (0, n - len(x)))
        health = SensorHealth()
        return WindowRecord(
            waveform=clip,
            sample_rate=self.working_fs,
            environment=FISH_IMPUTED_ENV,
            health_oracle=health,
            health_estimate=health,
            event_present=True,
            event_class=event_class_for_label(label),
            cause_id=f"cause.taxon.{label}",
            is_artifact=False,
            source_id=group_id,
            t0_s=0.0,
            taxon=label,
            environment_imputed=True,
        )


def _count(records: list[WindowRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        name = rec.taxon or rec.event_class.value
        counts[name] = counts.get(name, 0) + 1
    return counts


def load_species_split(
    root: str | Path,
    *,
    seed: int = 0,
    max_per_class: int | None = SPECIES_MAX_PER_CLASS,
    min_files: int = SPECIES_MIN_FILES,
    min_groups: int = SPECIES_MIN_GROUPS,
    min_classes: int = SPECIES_MIN_CLASSES,
    require_recording_ids: bool = False,
) -> tuple[list[WindowRecord], list[WindowRecord], list[WindowRecord], dict]:
    """Try 16 kHz, then one 32 kHz fallback. The first pass that clears the floors wins.

    The paper suite sets ``require_recording_ids``. A custom tree may train
    with one file as one group, and the meta says that is not a held-out recording.
    """
    errors: list[str] = []
    for fs, cutoff in ((16_000, 7_000.0), (32_000, 14_000.0)):
        adapter = SpeciesFolderAdapter(
            root,
            seed=seed,
            max_per_class=max_per_class,
            min_files=min_files,
            min_groups=min_groups,
            min_classes=min_classes,
            working_fs=fs,
            cutoff_hz=cutoff,
            require_recording_ids=require_recording_ids,
        )
        try:
            return adapter.load_split()
        except SpeciesDataError as exc:
            errors.append(f"{fs} Hz: {exc}")
    raise SpeciesDataError(
        "No working rate cleared the species floors. "
        + " | ".join(errors)
        + " Floors are not lowered. FishSounds is not used to pad."
    )


def inventory_species(
    root: str | Path,
    seed: int = 0,
    min_files: int = SPECIES_MIN_FILES,
    min_groups: int = SPECIES_MIN_GROUPS,
    min_classes: int = SPECIES_MIN_CLASSES,
) -> dict:
    """Count floors first, then the 16 kHz band, then the 32 kHz fallback. No model is fit."""
    base = SpeciesFolderAdapter(
        root,
        seed=seed,
        max_per_class=None,
        min_files=min_files,
        min_groups=min_groups,
        min_classes=1,
        require_recording_ids=False,
    )
    found = base.discover()
    passes = []
    chosen = None
    last_rows = found["rows"]
    for fs, cutoff in ((16_000, 7_000.0), (32_000, 14_000.0)):
        probe = SpeciesFolderAdapter(
            root,
            seed=seed,
            max_per_class=None,
            min_files=min_files,
            min_groups=min_groups,
            min_classes=1,
            working_fs=fs,
            cutoff_hz=cutoff,
            require_recording_ids=False,
        )
        try:
            banded = probe.apply_band(found)
            labels = sorted({g.label for g in banded["groups"]})
            error = None
            last_rows = banded["rows"]
        except SpeciesDataError as exc:
            labels = []
            error = str(exc)
            banded = {"rows": found["rows"]}
        entry = {
            "working_fs": fs,
            "cutoff_hz": cutoff,
            "n_eligible": len(labels),
            "labels": labels,
            "error": error,
        }
        passes.append(entry)
        if chosen is None and len(labels) >= min_classes and found["recording_ids"]:
            chosen = {
                "working_fs": fs,
                "cutoff_hz": cutoff,
                "labels": labels,
                "n_eligible": len(labels),
            }
            break
    return {
        "root": str(root),
        "citation": WATKINS_CITATION,
        "floors": {
            "min_files": min_files,
            "min_groups": min_groups,
            "min_classes": min_classes,
            "band_power_fraction": BAND_POWER_FRACTION,
        },
        "recording_ids": found["recording_ids"],
        "how_counts": found["how_counts"],
        "split_unit": found["split_unit"],
        "rows": last_rows,
        "skipped_ultrasonic": found.get("skipped_ultrasonic") or [],
        "passes": passes,
        "chosen": chosen,
        "note": (
            "Eligible species are chosen by file count, recording count, and band energy. "
            "Accuracy is not consulted. Call-library cuts, not a raw soundscape."
        ),
    }


def write_inventory(root: str | Path, path: str | Path, **kwargs) -> dict:
    report = inventory_species(root, **kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
