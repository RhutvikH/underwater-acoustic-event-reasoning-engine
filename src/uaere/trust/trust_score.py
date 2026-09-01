"""T(e) = σ(w_c C_wake + w_h κ(h) + w_s ρ(s,x) − w_u (u_a + u_e)).

No constant θ lives here.
"""

from __future__ import annotations

import numpy as np

from uaere.math.dirichlet import aleatoric_uncertainty, dirichlet_mean, epistemic_uncertainty
from uaere.trust.evidential_head import EvidentialHead
from uaere.types import (
    REJECT_INDEX,
    EnvironmentState,
    FaultClass,
    FloatArray,
    SensorHealth,
    TrustScore,
)


def _sigma(z: float) -> float:
    z = float(np.clip(z, -30, 30))
    return 1.0 / (1.0 + np.exp(-z))


def health_consistency(h: SensorHealth) -> float:
    """κ(h) ∈ [0, 1]. Physically implausible observations score low."""
    k = 1.0
    if h.clipping:
        k *= 0.35
    if h.fault is not FaultClass.NONE:
        k *= 0.45
    if h.adc_bits < 12:
        k *= 0.7
    if abs(h.drift_ppm) > 5:
        k *= 0.8
    return float(np.clip(k, 0.0, 1.0))


def env_consistency(mel: FloatArray, state: EnvironmentState) -> float:
    """ρ(s,x): support for a *non-ambient* cause given the current sea state.

    Vessels are low-frequency-heavy. Wind/rain geophony is high-frequency.
    High sea-state *reduces* this score, because the environment can already
    explain energy in the upper bands. That is the opposite of a wake-word
    energy detector, which false-alarms in storms.
    """
    n = mel.shape[0]
    low = float(np.mean(mel[: max(n // 4, 1)]))
    high = float(np.mean(mel[3 * n // 4 :]))
    contrast = low - high
    score = _sigma(3.0 * contrast - 0.45 * state.sea_state - 1.5 * state.turbulence)
    return float(np.clip(score, 0.0, 1.0))


class TrustEngine:
    def __init__(
        self,
        head: EvidentialHead,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.head = head
        self.w = weights or {
            "wake": 2.2,
            "health": 0.8,
            "env": 1.1,
            "uncertainty": 1.4,
            "bias": -0.6,
        }

    def score(
        self,
        feats: FloatArray,
        mel: FloatArray,
        health: SensorHealth,
        state: EnvironmentState,
    ) -> TrustScore:
        alpha = self.head.alpha(feats)
        p = dirichlet_mean(alpha)
        u_e = float(epistemic_uncertainty(alpha))
        u_a = float(aleatoric_uncertainty(p))
        c_wake = float(1.0 - p[REJECT_INDEX])
        kappa = health_consistency(health)
        rho = env_consistency(mel, state)
        z = (
            self.w["wake"] * c_wake
            + self.w["health"] * kappa
            + self.w["env"] * rho
            - self.w["uncertainty"] * (u_a + u_e)
            + self.w["bias"]
        )
        t = _sigma(z)
        return TrustScore(
            wake_confidence=c_wake,
            event_trust=t,
            p=p,
            alpha=alpha,
            u_aleatoric=u_a,
            u_epistemic=u_e,
            health_consistency=kappa,
            env_consistency=rho,
            components={
                "wake": c_wake,
                "health": kappa,
                "env": rho,
                "u_a": u_a,
                "u_e": u_e,
                "logit": z,
            },
        )
