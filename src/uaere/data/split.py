"""Time-aware splits: later-in-time recordings are the test set (UATR protocol)."""

from __future__ import annotations

from uaere.types import WindowRecord


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
