"""Cheap multi-hydrophone swarm: one physical event, many nodes, event-level fusion.

This is the specialty of the invention. Literature scores *nodes* (is this
sensor malicious?). We score *events*, and cheap neighbours confirm a blip
before anyone pays for L3 causal reasoning.

A single expensive hydrophone is not the product. A field of low-cost nodes
that wake each other is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.train import _mel_stats, extract_features, train_evidential, transform
from uaere.hardware.profiles import PROFILES, load_profile
from uaere.kg.graph import load_kg
from uaere.math.energy import compute_energy_j
from uaere.policy.runtime_gate import RuntimeGate
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.representation.l0_dsp import extract_l0, l0_admit_score
from uaere.representation.l1_tf import log_mel
from uaere.seed import rng as make_rng
from uaere.twin.environment import EnvironmentModel
from uaere.twin.network import AcousticNode, NetworkTwin, tx_energy_j
from uaere.twin.propagate import propagate
from uaere.twin.render import TwinRenderer, _mix_snr
from uaere.twin.sensor import Hydrophone, apply_sensor, noisy_health_estimate
from uaere.twin.sources import synthesize_source
from uaere.trust.trust_score import TrustEngine
from uaere.types import (
    WORKING_FS,
    WINDOW_SECONDS,
    EnvironmentState,
    EventClass,
    ExecutionLevel,
    FaultClass,
    PolicyVector,
    SensorHealth,
)


MACS = {0: 8e4, 1: 2e6, 2: 4e5, 3: 2e3, 4: 2e3}

# Cheap nodes default to Cortex-M / ESP32-class profiles, cycling so a field
# is heterogeneous — that is how you deploy a lot of them inexpensively.
_CHEAP_PROFILES = ("stm32l476", "esp32s3", "raspberry_pi_zero2")


@dataclass
class NodeView:
    node_id: str
    xyz: tuple[float, float, float]
    profile: str
    battery_frac: float
    level: int
    wake: float
    trust: float
    event_class: str
    explanation: str | None
    energy_j: float
    authenticated: bool
    woke_neighbors: list[str]
    confirmations: int
    snr_db: float
    reason: str


@dataclass
class SwarmTick:
    t: float
    scenario: str
    environment: dict
    source: dict
    nodes: list[NodeView]
    links: list[dict]
    kpis: dict


class SwarmField:
    """N cheap hydrophones sharing one ocean. One source, many ears."""

    def __init__(
        self,
        n_nodes: int = 8,
        scenario: str = "busy_strait",
        seed: int = 0,
        extent_m: float = 1200.0,
        battery_j: float = 10.0 * 3600.0,
    ) -> None:
        self.n_nodes = int(n_nodes)
        self.scenario = scenario
        self.seed = seed
        self.extent_m = extent_m
        self.rng = make_rng(seed)
        self.env_model = EnvironmentModel()
        self.hydro = Hydrophone(fs=WORKING_FS)
        self.n = int(WORKING_FS * WINDOW_SECONDS)
        self.twin = TwinRenderer(scenario, seed=seed)
        self.cfg = self.twin.cfg
        self.norm = AdaptiveNormalizer(n_bins=32)
        self.kg = load_kg()
        self.reasoner = CausalReasoner(self.kg)
        self.policy = PolicyVector(
            tau1=0.25,
            tau2=0.45,
            tau3=0.70,
            tau_collab_lo=0.40,
            tau_collab_hi=0.70,
            device_id="stm32l476",
            require_authenticated=False,  # mixed cheap field; auth is per-profile
        )
        self.gate = RuntimeGate(self.policy)
        self.head = None
        self.engine: TrustEngine | None = None
        self.t = 0.0
        self.kpis = {
            "ticks": 0,
            "true_events": 0,
            "detections": 0,
            "false_alarms": 0,
            "collab_wakes": 0,
            "explanations": 0,
            "joules": 0.0,
        }
        nodes = []
        for i in range(self.n_nodes):
            xyz = (
                float(self.rng.uniform(0, extent_m)),
                float(self.rng.uniform(0, extent_m)),
                float(self.rng.uniform(8.0, 40.0)),
            )
            profile = _CHEAP_PROFILES[i % len(_CHEAP_PROFILES)]
            nodes.append(
                AcousticNode(
                    node_id=f"n{i}",
                    xyz=xyz,
                    battery_j=battery_j,
                    profile_id=profile,
                )
            )
        self.net = NetworkTwin(nodes, sink_xyz=(extent_m / 2, extent_m / 2, 0.0))

    def warmup(self, n_train: int = 80) -> None:
        recs = TwinRenderer(self.scenario, seed=self.seed + 17).render_dataset(n_train)
        bun = extract_features(recs, self.norm)
        self.head = train_evidential(bun, steps=120, seed=self.seed)
        self.engine = TrustEngine(self.head)

    def step(self) -> SwarmTick:
        if self.engine is None or self.head is None:
            self.warmup()
        assert self.engine is not None and self.head is not None
        mix = self.cfg.get("mix", {"tug": 0.2, "reject": 0.3, "geophony": 0.2, "cargo": 0.3})
        names = list(mix.keys())
        probs = np.array([float(mix[k]) for k in names], dtype=np.float64)
        probs = probs / probs.sum()
        name = str(self.rng.choice(names, p=probs))
        cls = EventClass(name)
        present = cls in {EventClass.TUG, EventClass.CARGO, EventClass.TANKER, EventClass.PASSENGER}
        src_xyz = (
            float(self.rng.uniform(50, self.extent_m - 50)),
            float(self.rng.uniform(50, self.extent_m - 50)),
            float(self.rng.uniform(5.0, 25.0)),
        )
        state = self.env_model.state(
            temperature_c=float(self.cfg.get("temperature_c", 12.0)),
            salinity_psu=float(self.cfg.get("salinity_psu", 34.0)),
            depth_m=float(self.cfg.get("depth_m", 40.0)),
            sea_state=int(self.cfg.get("sea_state", 2)),
            snr_db=10.0,
            turbulence=float(self.cfg.get("turbulence", 0.1)),
        )
        source = synthesize_source(cls, self.n, WORKING_FS, self.rng, range_m=400.0)

        # Pass 1: every cheap node hears the *same* physical event at its range.
        hears: dict[str, dict] = {}
        for node in self.net.nodes.values():
            hears[node.node_id] = self._hear(node, source, src_xyz, cls, state)

        # Pass 2: local gate. Uncertain nodes request neighbour confirmation
        # *before* anyone runs L3. That is the cheap-swarm specialty.
        links: list[dict] = []
        confirmations: dict[str, int] = {nid: 0 for nid in hears}
        woke: dict[str, list[str]] = {nid: [] for nid in hears}
        for nid, h in hears.items():
            tscore = h["trust"]
            if self.policy.tau_collab_lo <= tscore.event_trust <= self.policy.tau_collab_hi:
                nbrs = self._nearest(nid, k=2)
                for nb in nbrs:
                    cost = self.net.collab_wake_cost(nid, nb, bits=256)
                    self.net.nodes[nid].spend(cost)
                    self.kpis["joules"] += cost
                    self.kpis["collab_wakes"] += 1
                    links.append({"src": nid, "dst": nb, "kind": "collab_wake"})
                    woke[nid].append(nb)
                    if hears[nb]["trust"].wake_confidence >= 0.45:
                        confirmations[nid] += 1

        # Pass 3: escalate. Neighbour confirmations *raise* event trust so a
        # borderline blip can buy L3 only after the field agrees it is real.
        views: list[NodeView] = []
        any_detect = False
        any_fa = False
        for nid, h in hears.items():
            node = self.net.nodes[nid]
            tscore = h["trust"]
            fused = float(
                np.clip(tscore.event_trust + 0.12 * confirmations[nid], 0.0, 1.0)
            )
            tscore.event_trust = fused
            level, reason = self.gate.admit(tscore)
            if confirmations[nid] >= 1 and level.value < ExecutionLevel.L3.value:
                if fused >= self.policy.tau3:
                    level, reason = ExecutionLevel.L3, "collab_confirmed_l3"
            expl = None
            event_cls = tscore.predicted_class()
            if level.value >= ExecutionLevel.L3.value:
                rec = h["record"]
                expl = self.reasoner.explain(
                    rec.event_class if present else event_cls,
                    rec.waveform,
                    rec.sample_rate,
                    rec.environment,
                )
                self.kpis["explanations"] += 1
            profile = load_profile(node.profile_id)
            tx_bits = 256 if woke[nid] else 0
            energy = compute_energy_j(level, MACS, profile, tx_bits=tx_bits)
            node.spend(energy)
            self.kpis["joules"] += energy
            wake = tscore.wake_confidence
            detect = wake >= 0.5 and level.value >= ExecutionLevel.L1.value
            if present and detect:
                any_detect = True
            if (not present) and detect:
                any_fa = True
            auth = profile.has_secure_element and profile.has_secure_boot
            views.append(
                NodeView(
                    node_id=nid,
                    xyz=node.xyz,
                    profile=node.profile_id,
                    battery_frac=node.residual_j / max(node.battery_j, 1e-9),
                    level=int(level),
                    wake=float(wake),
                    trust=float(tscore.event_trust),
                    event_class=event_cls.value,
                    explanation=expl.sentence if expl else None,
                    energy_j=float(energy),
                    authenticated=auth,
                    woke_neighbors=woke[nid],
                    confirmations=confirmations[nid],
                    snr_db=float(h["snr_db"]),
                    reason=reason,
                )
            )

        self.kpis["ticks"] += 1
        if present:
            self.kpis["true_events"] += 1
            if any_detect:
                self.kpis["detections"] += 1
        elif any_fa:
            self.kpis["false_alarms"] += 1
        self.t += 1.0
        tick = SwarmTick(
            t=self.t,
            scenario=self.scenario,
            environment={
                "sea_state": state.sea_state,
                "ssp_m_s": state.ssp_m_s,
                "temperature_c": state.temperature_c,
                "turbulence": state.turbulence,
            },
            source={
                "xyz": list(src_xyz),
                "class": cls.value,
                "present": present,
                "active": True,
            },
            nodes=views,
            links=links,
            kpis=dict(self.kpis),
        )
        self._last = tick
        return tick

    def snapshot(self) -> dict:
        tick = getattr(self, "_last", None)
        if tick is None:
            tick = self.step()
            self._last = tick
        return tick_to_json(tick)

    def _hear(self, node: AcousticNode, source, src_xyz, cls, state: EnvironmentState) -> dict:
        r = dist(node.xyz, src_xyz)
        snr_db = float(np.clip(22.0 - 20.0 * np.log10(max(r, 1.0) / 80.0), -8.0, 24.0))
        recv = propagate(source, src_xyz, node.xyz, state, WORKING_FS)
        ambient = self.env_model.ambient_noise(self.n, WORKING_FS, state, self.rng)
        mixed = _mix_snr(recv, ambient, snr_db, cls)
        health = SensorHealth(fault=FaultClass.NONE)
        obs, oracle = apply_sensor(mixed, self.hydro, health, self.rng)
        est = noisy_health_estimate(oracle, self.rng)
        from uaere.types import WindowRecord

        rec = WindowRecord(
            waveform=obs,
            sample_rate=WORKING_FS,
            environment=state,
            health_oracle=oracle,
            health_estimate=est,
            event_present=cls
            in {EventClass.TUG, EventClass.CARGO, EventClass.TANKER, EventClass.PASSENGER},
            event_class=cls
            if cls in {EventClass.TUG, EventClass.CARGO, EventClass.TANKER, EventClass.PASSENGER}
            else EventClass.REJECT,
            cause_id="",
            is_artifact=False,
            source_id=node.node_id,
        )
        l0 = extract_l0(obs, WORKING_FS)
        if l0_admit_score(l0) < 0.12:
            # near-silence at this node (too far): cheap reject
            dummy = self.engine.score(  # type: ignore[union-attr]
                np.zeros(self.head.w1.shape[1]),  # type: ignore[union-attr]
                np.zeros((32, 4)),
                est,
                state,
            )
            dummy.event_trust = 0.05
            dummy.wake_confidence = 0.05
            return {"record": rec, "trust": dummy, "snr_db": snr_db}
        mel = log_mel(obs, WORKING_FS)
        mel_n = self.norm.normalize_mel(mel, state)
        feats = transform(self.head, _mel_stats(mel_n))  # type: ignore[arg-type]
        tscore = self.engine.score(feats, mel_n, est, state)  # type: ignore[union-attr]
        return {"record": rec, "trust": tscore, "snr_db": snr_db}

    def _nearest(self, nid: str, k: int = 2) -> list[str]:
        src = self.net.nodes[nid]
        others = []
        for oid, n in self.net.nodes.items():
            if oid == nid:
                continue
            others.append((dist(src.xyz, n.xyz), oid))
        others.sort()
        return [oid for _, oid in others[:k]]


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))


def tick_to_json(tick: SwarmTick) -> dict:
    return {
        "t": tick.t,
        "scenario": tick.scenario,
        "environment": tick.environment,
        "source": tick.source,
        "nodes": [n.__dict__ for n in tick.nodes],
        "links": tick.links,
        "kpis": tick.kpis,
    }
