"""Time-aware splits, and file-level splits for real hydrophone clips.

A window split of one recording leaks the same call into train and test.
Real fish clips are grouped by recording id, then the whole group goes to
one side. Stratify on the group's label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uaere.types import NEGATIVE_LABELS, WindowRecord


def time_aware_split(
    records: list[WindowRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.2,
) -> tuple[list[WindowRecord], list[WindowRecord], list[WindowRecord]]:
    ordered = sorted(records, key=lambda r: (r.source_id, r.t0_s))
    n = len(ordered)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = ordered[n - n_test :] if n_test else []
    val = ordered[n - n_test - n_val : n - n_test] if n_val else []
    train = ordered[: n - n_test - n_val]
    return train, val, test


@dataclass(frozen=True)
class FileGroup:
    """One source recording. Every path shares a label and must share a split."""

    group_id: str
    label: str
    paths: tuple[Path, ...]


def stratified_group_split(
    groups: list[FileGroup],
    seed: int = 0,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_per_class: int | None = None,
) -> tuple[list[FileGroup], list[FileGroup], list[FileGroup]]:
    """60/20/20 by default, stratified, seed-locked. Caps count groups, not windows."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[FileGroup]] = {}
    for g in groups:
        by_label.setdefault(g.label, []).append(g)
    train: list[FileGroup] = []
    val: list[FileGroup] = []
    test: list[FileGroup] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        if max_per_class is not None:
            items = items[: int(max_per_class)]
        n = len(items)
        if n == 0:
            continue
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 5:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        while n_test + n_val >= n and (n_test > 0 or n_val > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])
    return train, val, test


def _files(groups: list[FileGroup]) -> int:
    return sum(len(g.paths) for g in groups)


def recording_locked_split(
    groups: list[FileGroup],
    seed: int = 0,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_per_class: int | None = None,
) -> tuple[list[FileGroup], list[FileGroup], list[FileGroup]]:
    """60/20/20 by file mass. A shared ``group_id`` is never cut across sides.

    ``max_per_class`` caps files. A group is added whole or not at all, so a
    multi-clip recording is not sliced to hit the cap. The first group of a
    class is kept even when it is larger than the cap.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[FileGroup]] = {}
    for g in groups:
        by_label.setdefault(g.label, []).append(g)
    kept: list[FileGroup] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        if max_per_class is None:
            kept.extend(items)
            continue
        n = 0
        cap = int(max_per_class)
        for g in items:
            sz = len(g.paths)
            if n >= cap:
                break
            if n > 0 and n + sz > cap:
                continue
            kept.append(g)
            n += sz
    clusters: dict[str, list[FileGroup]] = {}
    for g in kept:
        clusters.setdefault(g.group_id, []).append(g)
    buckets: dict[str, list[list[FileGroup]]] = {"pos": [], "neg": []}
    for gs in clusters.values():
        positive = any(g.label not in NEGATIVE_LABELS for g in gs)
        buckets["pos" if positive else "neg"].append(gs)
    train: list[FileGroup] = []
    val: list[FileGroup] = []
    test: list[FileGroup] = []
    for key in ("neg", "pos"):
        items = buckets[key]
        rng.shuffle(items)
        n_files = sum(_files(gs) for gs in items)
        if n_files == 0:
            continue
        n_test = int(round(n_files * test_frac))
        n_val = int(round(n_files * val_frac))
        got_test = 0
        got_val = 0
        train_here: list[list[FileGroup]] = []
        val_here: list[list[FileGroup]] = []
        test_here: list[list[FileGroup]] = []
        for gs in items:
            sz = _files(gs)
            if n_test > 0 and (got_test == 0 or got_test + sz <= n_test):
                test_here.append(gs)
                got_test += sz
            elif n_val > 0 and (got_val == 0 or got_val + sz <= n_val):
                val_here.append(gs)
                got_val += sz
            else:
                train_here.append(gs)
        if not train_here and (val_here or test_here):
            donor = val_here if val_here else test_here
            train_here.append(donor.pop())
        for bucket, dest in ((train_here, train), (val_here, val), (test_here, test)):
            for gs in bucket:
                dest.extend(gs)
    return train, val, test


def _split_clusters_by_mass(
    clusters: list[list[FileGroup]],
    rng: np.random.Generator,
    val_frac: float,
    test_frac: float,
) -> tuple[list[FileGroup], list[FileGroup], list[FileGroup]]:
    """One stratum. Whole clusters stay on one side. File mass fills test, then val."""
    items = list(clusters)
    rng.shuffle(items)
    n_files = sum(_files(gs) for gs in items)
    train: list[FileGroup] = []
    val: list[FileGroup] = []
    test: list[FileGroup] = []
    if n_files == 0:
        return train, val, test
    n_test = int(round(n_files * test_frac))
    n_val = int(round(n_files * val_frac))
    got_test = 0
    got_val = 0
    train_here: list[list[FileGroup]] = []
    val_here: list[list[FileGroup]] = []
    test_here: list[list[FileGroup]] = []
    for gs in items:
        sz = _files(gs)
        if n_test > 0 and (got_test == 0 or got_test + sz <= n_test):
            test_here.append(gs)
            got_test += sz
        elif n_val > 0 and (got_val == 0 or got_val + sz <= n_val):
            val_here.append(gs)
            got_val += sz
        else:
            train_here.append(gs)
    if not train_here and (val_here or test_here):
        donor = val_here if val_here else test_here
        train_here.append(donor.pop())
    for bucket, dest in ((train_here, train), (val_here, val), (test_here, test)):
        for gs in bucket:
            dest.extend(gs)
    return train, val, test


def label_stratified_recording_split(
    groups: list[FileGroup],
    seed: int = 0,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    max_per_class: int | None = None,
) -> tuple[list[FileGroup], list[FileGroup], list[FileGroup]]:
    """Per-label 60/20/20 by file mass. A shared group_id is never cut across sides.

    Used for a multi-species folder. The two-class fish split stays
    ``recording_locked_split``, which only stratifies call versus negative.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[FileGroup]] = {}
    for g in groups:
        by_label.setdefault(g.label, []).append(g)
    kept: list[FileGroup] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        if max_per_class is None:
            kept.extend(items)
            continue
        n = 0
        cap = int(max_per_class)
        for g in items:
            sz = len(g.paths)
            if n >= cap:
                break
            if n > 0 and n + sz > cap:
                continue
            kept.append(g)
            n += sz
    clusters: dict[str, list[FileGroup]] = {}
    for g in kept:
        clusters.setdefault(g.group_id, []).append(g)
    by_stratum: dict[str, list[list[FileGroup]]] = {}
    for gs in clusters.values():
        counts: dict[str, int] = {}
        for g in gs:
            counts[g.label] = counts.get(g.label, 0) + len(g.paths)
        stratum = sorted(counts, key=lambda name: (-counts[name], name))[0]
        by_stratum.setdefault(stratum, []).append(gs)
    train: list[FileGroup] = []
    val: list[FileGroup] = []
    test: list[FileGroup] = []
    for label in sorted(by_stratum):
        tr, va, te = _split_clusters_by_mass(by_stratum[label], rng, val_frac, test_frac)
        train.extend(tr)
        val.extend(va)
        test.extend(te)
    return train, val, test
