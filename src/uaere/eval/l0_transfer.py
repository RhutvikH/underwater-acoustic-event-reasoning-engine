"""How the twin Knudsen admit map treats real fish clips.

The center and the 0.15 cutoff are not refit here. The report is the
reason a species checkpoint does not let that cutoff replace its label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uaere.data.adapters import FishCallAdapter
from uaere.data.split import recording_locked_split
from uaere.representation.predictive import surprise, surprise_admit, surprise_at_admit
from uaere.types import WindowRecord

ADMIT_THRESHOLD = 0.15


def summarize_surprise(records: list[WindowRecord], threshold: float = ADMIT_THRESHOLD) -> dict:
    """Quantiles of S and how often the twin sigmoid would sleep. No new threshold is fit."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for rec in records:
        s = float(surprise(rec.waveform, rec.sample_rate, rec.environment))
        admit = float(surprise_admit(s))
        name = rec.taxon or rec.event_class.value
        by_label.setdefault(name, []).append((s, admit))
    labels = {}
    for name, rows in sorted(by_label.items()):
        s_arr = np.array([row[0] for row in rows], dtype=np.float64)
        a_arr = np.array([row[1] for row in rows], dtype=np.float64)
        q = np.quantile(s_arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        n_below = int(np.sum(a_arr < threshold))
        labels[name] = {
            "n": int(s_arr.size),
            "s_p05": float(q[0]),
            "s_p25": float(q[1]),
            "s_p50": float(q[2]),
            "s_p75": float(q[3]),
            "s_p95": float(q[4]),
            "n_admit_below": n_below,
            "n_admit_at_or_above": int(s_arr.size - n_below),
        }
    return {"n": len(records), "by_label": labels}


def audit_fish_root(
    root: str | Path = "data/toadfish",
    seed: int = 0,
    max_per_class: int = 3000,
    min_files: int = 40,
) -> dict:
    """Deployment-locked split, same arguments as the fish suite. Does not refit the logistic."""
    adapter = FishCallAdapter(root, seed=seed, max_per_class=max_per_class, min_files=min_files)
    found = adapter.discover()
    train_g, val_g, test_g = recording_locked_split(
        found["groups"], seed=seed, max_per_class=max_per_class
    )
    splits = {}
    for name, groups in (("train", train_g), ("val", val_g), ("test", test_g)):
        records = adapter._materialise(groups)
        splits[name] = summarize_surprise(records)
        del records
    return {
        "center": 8.0,
        "scale": 1.5,
        "admit_threshold": ADMIT_THRESHOLD,
        "surprise_at_threshold": surprise_at_admit(ADMIT_THRESHOLD),
        "seed": seed,
        "max_per_class": max_per_class,
        "split_unit": found.get("split_unit"),
        "n_groups": {"train": len(train_g), "val": len(val_g), "test": len(test_g)},
        "note": (
            "Measured on the imputed sea state. Center and scale were not refit, "
            "and this cutoff was not used to choose the logistic or the trust thresholds. "
            "A species checkpoint records S and does not sleep on it."
        ),
        "splits": splits,
    }


def write_l0_audit(
    path: str | Path = "artifacts/paper/fish_l0_surprise.json",
    root: str | Path = "data/toadfish",
    seed: int = 0,
    max_per_class: int = 3000,
    min_files: int = 40,
) -> dict:
    report = audit_fish_root(root, seed=seed, max_per_class=max_per_class, min_files=min_files)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
