"""Canonical types for AHAIF. Every module speaks this vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


WORKING_FS = 16_000
WINDOW_SECONDS = 1.0
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 32
N_EVENT_CLASSES = 5  # cargo, passenger, tanker, tug, reject


class EventClass(str, Enum):
    REJECT = "reject"
    CARGO = "cargo"
    PASSENGER = "passenger"
    TANKER = "tanker"
    TUG = "tug"
    BIOLOGICAL = "biological"
    GEOPHONY = "geophony"
    ARTIFACT = "artifact"


DEEPSHIP_LABELS: tuple[EventClass, ...] = (
    EventClass.CARGO,
    EventClass.PASSENGER,
    EventClass.TANKER,
    EventClass.TUG,
    EventClass.REJECT,
)

# Classifier / Dirichlet head operates on this ordered label set.
CLASS_INDEX: dict[EventClass, int] = {c: i for i, c in enumerate(DEEPSHIP_LABELS)}
INDEX_CLASS: dict[int, EventClass] = {i: c for c, i in CLASS_INDEX.items()}
REJECT_INDEX = CLASS_INDEX[EventClass.REJECT]


class ExecutionLevel(int, Enum):
    L0 = 0  # always-on DSP
    L1 = 1  # Mel + trust
    L2 = 2  # TinyML classifier
    L3 = 3  # causal KG reasoning
    L4 = 4  # collaborative neighbor-wake


class FaultClass(str, Enum):
    NONE = "none"
    BIAS = "bias"
    DROPOUT = "dropout"
    GAIN_WANDER = "gain_wander"


class ModelVariant(str, Enum):
    TINY = "tiny"
    SMALL = "small"


@dataclass(frozen=True)
class EnvironmentState:
    """Twin / live environment state *s* in the formulation."""

    temperature_c: float = 12.0
    salinity_psu: float = 34.0
    depth_m: float = 40.0
    sea_state: int = 2
    snr_db: float = 10.0
    turbulence: float = 0.1
    ssp_m_s: float = 1490.0

    def as_vector(self) -> FloatArray:
        return np.array(
            [
                self.temperature_c,
                self.salinity_psu,
                self.depth_m,
                float(self.sea_state),
                self.snr_db,
                self.turbulence,
                self.ssp_m_s,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SensorHealth:
    """Oracle health *h* from the twin. The trust engine sees a noisy estimate."""

    noise_floor_db: float = -80.0
    clipping: bool = False
    drift_ppm: float = 0.0
    adc_bits: int = 16
    fault: FaultClass = FaultClass.NONE

    def as_vector(self) -> FloatArray:
        return np.array(
            [
                self.noise_floor_db,
                1.0 if self.clipping else 0.0,
                self.drift_ppm,
                float(self.adc_bits),
                0.0 if self.fault is FaultClass.NONE else 1.0,
            ],
            dtype=np.float64,
        )


@dataclass
class TrustScore:
    """Calibrated event-level wake trust. There is no fixed energy threshold here."""

    wake_confidence: float
    event_trust: float
    p: FloatArray
    alpha: FloatArray
    u_aleatoric: float
    u_epistemic: float
    health_consistency: float
    env_consistency: float
    components: dict[str, float] = field(default_factory=dict)

    def predicted_class(self) -> EventClass:
        return INDEX_CLASS[int(np.argmax(self.p))]


@dataclass
class L0Features:
    energy: float
    zcr: float
    centroid_hz: float
    bandwidth_hz: float
    flatness: float
    noise_residual: float
    vector: FloatArray


@dataclass
class Explanation:
    event_type: str
    cause_id: str
    cause_label: str
    chain: list[str]
    sentence: str
    score: float
    counterfactual_verified: bool
    rejected_causes: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    level: ExecutionLevel
    trust: TrustScore
    event_class: EventClass
    explanation: Explanation | None
    energy_j: float
    latency_s: float
    authenticated: bool
    reason: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class WindowRecord:
    """One labelled 1 s window used by adapters, twin replay, and eval."""

    waveform: FloatArray
    sample_rate: int
    environment: EnvironmentState
    health_oracle: SensorHealth
    health_estimate: SensorHealth
    event_present: bool
    event_class: EventClass
    cause_id: str
    is_artifact: bool
    source_id: str = ""
    t0_s: float = 0.0


@dataclass(frozen=True)
class PolicyVector:
    """Decision vector π = (τ1, τ2, τ3, m, d) plus collaborative band."""

    tau1: float
    tau2: float
    tau3: float
    model: ModelVariant = ModelVariant.TINY
    device_id: str = "stm32l476"
    tau_collab_lo: float = 0.35
    tau_collab_hi: float = 0.55
    require_authenticated: bool = True

    def as_array(self) -> FloatArray:
        return np.array(
            [self.tau1, self.tau2, self.tau3, self.tau_collab_lo, self.tau_collab_hi],
            dtype=np.float64,
        )


@dataclass
class ObjectiveVector:
    miss: float
    fa: float
    energy_j_s: float
    latency_s: float
    ece: float
    explanation_coverage: float

    def as_array(self) -> FloatArray:
        # last objective is maximised, stored negated for min-form
        return np.array(
            [
                self.miss,
                self.fa,
                self.energy_j_s,
                self.latency_s,
                self.ece,
                -self.explanation_coverage,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    isa: str
    cpu_mhz: float
    ram_kb: float
    flash_kb: float
    idle_mw: float
    active_mw: float
    nj_per_mac_int8: float
    tx_nj_per_bit: float
    has_secure_element: bool
    has_secure_boot: bool
    has_tpm: bool
    has_npu: bool
    npu_mac_per_s: float = 0.0
    supports_l2: bool = True
    supports_l3: bool = True
