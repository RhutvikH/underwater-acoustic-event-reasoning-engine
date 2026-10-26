"""Dummy / oracle event head for the live twin demo.

The logistic Mel head is a real (weak) classifier: on a short twin warmup it
often dumps mass on `reject`, so the GUI never reaches L3. This oracle peeks
at the twin's gold class — allowed in a digital twin — and emits a peaked
Dirichlet plus high T(e) so the swarm, KG, and gold rings actually show.

Swap it for the trained head with `uaere demo --model logistic`.
"""

from __future__ import annotations

import numpy as np

from uaere.math.dirichlet import aleatoric_uncertainty, dirichlet_mean, epistemic_uncertainty
from uaere.types import (
    CLASS_INDEX,
    N_EVENT_CLASSES,
    REJECT_INDEX,
    EventClass,
    EnvironmentState,
    SensorHealth,
    TrustScore,
    is_acoustic_event,
)
from uaere.trust.trust_score import health_consistency


def oracle_trust(
    cls: EventClass,
    health: SensorHealth,
    state: EnvironmentState,
    snr_db: float = 10.0,
) -> TrustScore:
    present = is_acoustic_event(cls)
    alpha = np.ones(N_EVENT_CLASSES, dtype=np.float64) * 1.15
    if present:
        idx = CLASS_INDEX.get(cls, CLASS_INDEX[EventClass.BIOLOGICAL])
        alpha[idx] = 48.0
        # Range/SNR scales confidence: close = L3, mid = collab band, far = sleep.
        strength = float(np.clip((snr_db + 8.0) / 32.0, 0.0, 1.0))
        wake = 0.18 + 0.78 * strength
        trust = 0.14 + 0.78 * strength
        label = cls if strength >= 0.28 else EventClass.REJECT
    else:
        alpha[REJECT_INDEX] = 48.0
        wake = 0.10
        trust = 0.11
        label = EventClass.REJECT
    p = dirichlet_mean(alpha)
    kappa = health_consistency(health)
    return TrustScore(
        wake_confidence=wake,
        event_trust=float(np.clip(trust * kappa, 0.0, 1.0)),
        p=p,
        alpha=alpha,
        u_aleatoric=float(aleatoric_uncertainty(p)),
        u_epistemic=float(epistemic_uncertainty(alpha)),
        health_consistency=kappa,
        env_consistency=0.75 if present else 0.25,
        components={"oracle": 1.0, "present": float(present)},
        label=label,
    )
