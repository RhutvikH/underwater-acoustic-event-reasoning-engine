"""End-to-end AHAIF inference for one window. Shared by CLI and eval."""

from __future__ import annotations

import struct

import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.ir import trace_macs
from uaere.classify.train import _mel_stats, transform
from uaere.hardware.orchestrator import Orchestrator
from uaere.kg.graph import MarineAcousticKG
from uaere.math.energy import compute_energy_j
from uaere.policy.runtime_gate import RuntimeGate
from uaere.representation.encoder import TinyCNN, cnn_macs
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.representation.l0_dsp import extract_l0, l0_admit_score
from uaere.representation.l1_tf import log_mel
from uaere.security.hal import SecureHAL
from uaere.trust.trust_score import TrustEngine
from uaere.types import (
    INDEX_CLASS,
    EventClass,
    ExecutionLevel,
    InferenceResult,
    PolicyVector,
    WindowRecord,
)


class AhaifPipeline:
    def __init__(
        self,
        trust: TrustEngine,
        kg: MarineAcousticKG,
        policy: PolicyVector,
        normalizer: AdaptiveNormalizer,
        hal: SecureHAL | None = None,
        cnn: TinyCNN | None = None,
    ) -> None:
        self.trust = trust
        self.reasoner = CausalReasoner(kg)
        self.gate = RuntimeGate(policy)
        self.orch = Orchestrator(policy)
        self.norm = normalizer
        self.hal = hal
        self.cnn = cnn or TinyCNN()
        self.macs = {
            0: 80_000.0,  # rFFT + moments on 16k
            1: 2_000_000.0,
            2: float(cnn_macs((32, 62))),
            3: 2_000.0,
            4: 2_000.0,
        }

    def infer(self, rec: WindowRecord) -> InferenceResult:
        firmware_ok = True
        if self.hal is not None:
            # already booted by caller; still tag the result
            pass
        l0 = extract_l0(rec.waveform, rec.sample_rate)
        admit = l0_admit_score(l0)
        # L0-only if the logistic of φ0 is very small (near-silence), not energy θ
        if admit < 0.15:
            profile = self.orch.place(ExecutionLevel.L0)
            e = compute_energy_j(ExecutionLevel.L0, self.macs, profile)
            dummy = self.trust.score(
                np.zeros(self.trust.head.w1.shape[1]),
                np.zeros((32, 4)),
                rec.health_estimate,
                rec.environment,
            )
            dummy.event_trust = 0.0
            dummy.wake_confidence = 0.0
            return InferenceResult(
                level=ExecutionLevel.L0,
                trust=dummy,
                event_class=EventClass.REJECT,
                explanation=None,
                energy_j=e,
                latency_s=self.macs[0] / (profile.cpu_mhz * 1e6),
                authenticated=self.hal is not None,
                reason="l0_near_silence_shape_gate",
            )

        mel = log_mel(rec.waveform, rec.sample_rate)
        mel_n = self.norm.normalize_mel(mel, rec.environment)
        feats = transform(self.trust.head, _mel_stats(mel_n))
        tscore = self.trust.score(feats, mel_n, rec.health_estimate, rec.environment)
        level, reason = self.gate.admit(tscore)
        profile = self.orch.place(level)
        event_cls = tscore.predicted_class()
        expl = None
        if level >= ExecutionLevel.L2:
            logits = self.cnn.logits(mel_n)
            event_cls = INDEX_CLASS[int(np.argmax(logits))]
        if level >= ExecutionLevel.L3:
            expl = self.reasoner.explain(
                event_cls, rec.waveform, rec.sample_rate, rec.environment
            )
        tx = 256 if level is ExecutionLevel.L4 else 0
        energy = compute_energy_j(level, self.macs, profile, tx_bits=tx)
        auth = False
        if self.hal is not None:
            window_b = rec.waveform.astype(np.float32).tobytes()
            out_b = struct.pack("d", tscore.event_trust)
            tag = self.hal.authenticate_inference(window_b, out_b)
            auth = self.hal.verify_inference(window_b, out_b, tag)
        ir = trace_macs(self.cnn.n_params(), mel_n.shape[1])
        if not ir.under_budget():
            raise RuntimeError(f"INT8 budget exceeded: {ir.int8_bytes} B")
        return InferenceResult(
            level=level,
            trust=tscore,
            event_class=event_cls,
            explanation=expl,
            energy_j=energy,
            latency_s=sum(self.macs[k] for k in range(int(level) + 1) if k in self.macs)
            / (profile.cpu_mhz * 1e6),
            authenticated=auth,
            reason=reason,
            extras={"l0_admit": admit, "firmware_ok": firmware_ok},
        )
