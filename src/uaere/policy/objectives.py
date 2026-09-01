"""F(π) = (miss, fa, E, L, ECE, −X) evaluated on a twin replay of trust scores."""

from __future__ import annotations

import numpy as np

from uaere.math.energy import compute_energy_j
from uaere.math.metrics import ece
from uaere.policy.runtime_gate import RuntimeGate
from uaere.types import (
    DeviceProfile,
    ExecutionLevel,
    ObjectiveVector,
    PolicyVector,
    TrustScore,
)


def evaluate_policy(
    policy: PolicyVector,
    trusts: list[TrustScore],
    y_event: np.ndarray,
    y_cause_hit: np.ndarray,
    profile: DeviceProfile,
    macs_by_level: dict[int, float],
) -> ObjectiveVector:
    gate = RuntimeGate(policy)
    levels = [gate.admit(t)[0] for t in trusts]
    pred = np.array(
        [
            int(lv >= ExecutionLevel.L1 and t.wake_confidence >= 0.5)
            for t, lv in zip(trusts, levels, strict=True)
        ]
    )
    # miss / fa on event presence using the gated wake decision
    y = np.asarray(y_event, dtype=int)
    miss = float(np.mean((y == 1) & (pred == 0))) if np.any(y == 1) else 0.0
    fa = float(np.mean((y == 0) & (pred == 1))) if np.any(y == 0) else 0.0
    energies = [
        compute_energy_j(lv, macs_by_level, profile, tx_bits=256 if lv is ExecutionLevel.L4 else 0)
        for lv in levels
    ]
    e_mean = float(np.mean(energies))
    # latency proxy: 1e-7 s per MAC at this profile
    lat = []
    for lv in levels:
        macs = sum(macs_by_level[k] for k in range(int(lv) + 1) if k in macs_by_level)
        lat.append(macs / (profile.cpu_mhz * 1e6))
    p99 = float(np.quantile(lat, 0.99)) if lat else 0.0
    conf = np.array([t.event_trust for t in trusts])
    ece_v = ece(y.astype(float), conf)
    hits = []
    for i, lv in enumerate(levels):
        if y[i] == 1:
            hits.append(1.0 if (lv >= ExecutionLevel.L3 and y_cause_hit[i]) else 0.0)
    xcov = float(np.mean(hits)) if hits else 0.0
    return ObjectiveVector(
        miss=miss,
        fa=fa,
        energy_j_s=e_mean,
        latency_s=p99,
        ece=ece_v,
        explanation_coverage=xcov,
    )
