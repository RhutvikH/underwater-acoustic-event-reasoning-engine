"""Turn the Watkins full-cuts zip into data/species/<species>/*.wav.

Audio stays under data/, which git ignores. This does not download the zip
and does not decide which species clear the floors.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from uaere.data.species import slug_label

_GENERIC = {
    "wav",
    "wavs",
    "audio",
    "cuts",
    "full_cuts",
    "all_cuts",
    "best_of",
    "bestof",
    "data",
    "watkins",
}
_SAMPLE = re.compile(r"sample_([A-Za-z0-9]+)_(\d{4})_(\d{5}[0-9A-Za-z]+)", re.IGNORECASE)


def camel_slug(name: str) -> str:
    text = name.replace("'", "").replace("’", "")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    return slug_label(text)


def species_from_member(name: str) -> str | None:
    """Species folder, not the year folder.

    The full-cuts zip is ``.../<Species>/<year>/<tapeid>.wav``. A flat
    ``sample_<Name>_<year>_<tape>.wav`` name is accepted too.
    """
    path = PurePosixPath(name)
    if path.suffix.lower() != ".wav":
        return None
    if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
        return None
    parents = []
    for part in path.parts[:-1]:
        if slug_label(part) in _GENERIC or part.lower().endswith(".zip"):
            continue
        if re.fullmatch(r"\d{4}", part):
            continue
        if "watkins" in part.lower() and "database" in part.lower():
            continue
        parents.append(part)
    if parents:
        return camel_slug(parents[-1])
    match = _SAMPLE.search(path.stem)
    if match:
        return camel_slug(match.group(1))
    return None


def unpack_watkins(zip_path: str | Path, dest: str | Path = "data/species") -> dict:
    """Extract WAV cuts only. Existing files are left in place."""
    zpath = Path(zip_path)
    if not zpath.is_file():
        raise FileNotFoundError(f"Watkins zip not found at {zpath}")
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_existing = 0
    unnamed: list[str] = []
    per_species: dict[str, int] = {}
    with zipfile.ZipFile(zpath) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            label = species_from_member(info.filename)
            if label is None:
                if info.filename.lower().endswith(".wav"):
                    unnamed.append(info.filename)
                continue
            target = root / label / Path(info.filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            per_species[label] = per_species.get(label, 0) + 1
            if target.exists() and target.stat().st_size > 0:
                skipped_existing += 1
                continue
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            written += 1
    wav_members = written + skipped_existing
    if wav_members == 0:
        raise RuntimeError(f"No species WAV members found in {zpath}. Unnamed wavs: {unnamed[:8]}")
    if unnamed and len(unnamed) > wav_members:
        raise RuntimeError(
            "Most WAV members had no species folder and no sample_<Name>_<year>_<tape> name. "
            f"Examples: {unnamed[:8]}. Refusing to drop them into one class."
        )
    return {
        "dest": str(root),
        "written": written,
        "skipped_existing": skipped_existing,
        "n_species_dirs": len(per_species),
        "per_species": dict(sorted(per_species.items())),
        "unnamed_wavs": unnamed[:20],
        "n_unnamed_wavs": len(unnamed),
    }
