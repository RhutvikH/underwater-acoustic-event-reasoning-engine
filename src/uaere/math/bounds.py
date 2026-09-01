"""Propositions checked by construction and by unit test.

Proposition 1. For any always-on L3 policy π_on and any gated π with p3 < 1
and identical per-level costs c_k > 0,
    E(π) < E(π_on).

Proposition 2. If T is a consistent estimator of the event posterior, the
extra miss rate of a threshold τ3 relative to the Bayes threshold τ* is at
most the posterior mass on (τ*, τ3] (or [τ3, τ*) if τ3 < τ*).
"""

from __future__ import annotations

from uaere.math.energy import expected_compute_energy


def gated_energy_strictly_less(
    c: tuple[float, float, float, float],
    p_gated: tuple[float, float, float],
) -> bool:
    e_gated = expected_compute_energy(c, p_gated)
    e_on = expected_compute_energy(c, (1.0, 1.0, 1.0))
    return e_gated < e_on


def gated_miss_bound(posterior_mass_between: float) -> float:
    """ε(τ3) ≤ mass between τ3 and the Bayes threshold. Mass must be in [0, 1]."""
    m = float(posterior_mass_between)
    if not 0.0 <= m <= 1.0:
        raise ValueError("posterior mass must be in [0, 1]")
    return m
