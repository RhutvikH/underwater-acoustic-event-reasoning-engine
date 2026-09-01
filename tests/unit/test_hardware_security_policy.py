from __future__ import annotations

import numpy as np
import pytest

from uaere.hardware.orchestrator import Orchestrator
from uaere.hardware.profiles import PROFILES
from uaere.math.energy import compute_energy_j
from uaere.policy.baselines import always_l3
from uaere.policy.nsga2 import nsga2
from uaere.policy.runtime_gate import RuntimeGate
from uaere.security.hal import SecureHAL, TamperError
from uaere.types import ExecutionLevel, PolicyVector, TrustScore


def _trust(t: float) -> TrustScore:
    p = np.array([0.7, 0.05, 0.05, 0.05, 0.15])
    return TrustScore(
        wake_confidence=0.85,
        event_trust=t,
        p=p,
        alpha=p * 10 + 1,
        u_aleatoric=0.3,
        u_epistemic=0.2,
        health_consistency=1.0,
        env_consistency=0.8,
    )


def test_gate_levels_increase_with_trust():
    g = RuntimeGate(PolicyVector(0.2, 0.4, 0.7))
    assert g.admit(_trust(0.1))[0] is ExecutionLevel.L0
    assert g.admit(_trust(0.3))[0] is ExecutionLevel.L1
    assert g.admit(_trust(0.5))[0] is ExecutionLevel.L2
    assert g.admit(_trust(0.9))[0] is ExecutionLevel.L3


def test_gd32_refused_when_security_on():
    orch = Orchestrator(PolicyVector(0.2, 0.4, 0.6, device_id="gd32vf103", require_authenticated=True))
    with pytest.raises(PermissionError):
        orch.place(ExecutionLevel.L2)


def test_stm32_allowed():
    orch = Orchestrator(PolicyVector(0.2, 0.4, 0.6, device_id="stm32l476", require_authenticated=True))
    p = orch.place(ExecutionLevel.L2)
    assert p.has_secure_element


def test_five_profiles_positive_energy():
    macs = {0: 1e5, 1: 1e6, 2: 5e5, 3: 2e3}
    assert len(PROFILES) >= 5
    assert "raspberry_pi4" in PROFILES
    assert "raspberry_pi_zero2" in PROFILES
    for p in PROFILES.values():
        assert compute_energy_j(ExecutionLevel.L1, macs, p) > 0


def test_secure_boot_tamper():
    fw = b"firmware"
    hal = SecureHAL(fw, b"model")
    hal.secure_boot(fw)
    with pytest.raises(TamperError):
        hal.secure_boot(b"nope")
    tag = hal.authenticate_inference(b"win", b"out")
    assert hal.verify_inference(b"win", b"out", tag)
    nonce, blob = hal.encrypt(b"hello")
    assert hal.decrypt(nonce, blob) == b"hello"


def test_nsga_returns_front():
    def ev(pi: PolicyVector):
        from uaere.types import ObjectiveVector

        # cheaper if taus are larger
        e = (1.0 - pi.tau3) + 0.1 * pi.tau1
        return ObjectiveVector(miss=pi.tau3, fa=1 - pi.tau1, energy_j_s=e, latency_s=e, ece=0.1, explanation_coverage=pi.tau3)

    front = nsga2(ev, seed=0, pop_size=12, n_gen=4)
    assert len(front) >= 1
    _ = always_l3()
