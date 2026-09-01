"""Runtime gate: the only discrete decisions in the pipeline come from π, not from energy."""

from __future__ import annotations

import numpy as np

from uaere.types import ExecutionLevel, PolicyVector, TrustScore


class RuntimeGate:
    def __init__(self, policy: PolicyVector) -> None:
        self.policy = policy

    def admit(self, trust: TrustScore) -> tuple[ExecutionLevel, str]:
        t = trust.event_trust
        p = self.policy
        if t < p.tau1:
            return ExecutionLevel.L0, "below_tau1_sleep"
        if t < p.tau2:
            return ExecutionLevel.L1, "tau1_l1_trust_only"
        if t < p.tau3:
            return ExecutionLevel.L2, "tau2_l2_classify"
        level = ExecutionLevel.L3
        reason = "tau3_l3_causal"
        if p.tau_collab_lo <= t <= p.tau_collab_hi:
            level = ExecutionLevel.L4
            reason = "collab_band_l4"
        return level, reason


def chebyshev_select(
    front: list[tuple[PolicyVector, np.ndarray]],
    lambdas: np.ndarray | None = None,
) -> PolicyVector:
    import numpy as np

    if not front:
        raise ValueError("empty Pareto front")
    F = np.stack([f for _, f in front])
    z_star = F.min(axis=0)
    z_nad = F.max(axis=0)
    span = np.maximum(z_nad - z_star, 1e-12)
    lam = lambdas if lambdas is not None else np.ones(F.shape[1])
    scores = []
    for row in F:
        scores.append(np.max(lam * (row - z_star) / span))
    return front[int(np.argmin(scores))][0]
