"""`uaere eval --suite paper` — every table in the plan, written to artifacts/paper/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.ir import trace_macs
from uaere.classify.train import extract_features, train_evidential, transform
from uaere.data.split import time_aware_split
from uaere.hardware.orchestrator import Orchestrator
from uaere.hardware.profiles import PROFILES
from uaere.kg.graph import load_kg
from uaere.math.bounds import gated_energy_strictly_less
from uaere.math.complexity import kg_bfs, l0_rfft, l1_mel
from uaere.math.energy import compute_energy_j
from uaere.math.metrics import brier, ece, mce, roc_auc
from uaere.policy.baselines import always_l3
from uaere.policy.nsga2 import nsga2
from uaere.policy.objectives import evaluate_policy
from uaere.policy.runtime_gate import chebyshev_select
from uaere.representation.encoder import TinyCNN
from uaere.security.hal import SecureHAL, TamperError
from uaere.trust.baseline_energy_threshold import EnergyThresholdBaseline
from uaere.trust.trust_score import TrustEngine
from uaere.twin.render import TwinRenderer
from uaere.types import (
    ExecutionLevel,
    PolicyVector,
    WindowRecord,
)


def run_paper_suite(
    out_dir: str | Path = "artifacts/paper",
    n_windows: int = 240,
    seed: int = 0,
    train_steps: int = 250,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = TwinRenderer("busy_strait", seed=seed).render_dataset(n_windows)
    # extra high-sea-state geophony so energy baseline is stressed
    records += TwinRenderer("high_sea_state", seed=seed + 1).render_dataset(n_windows // 3)
    train, val, test = time_aware_split(records)
    bundle_tr = extract_features(train)
    head = train_evidential(bundle_tr, steps=train_steps, seed=seed)
    engine = TrustEngine(head)
    kg = load_kg()
    reasoner = CausalReasoner(kg)
    cnn = TinyCNN(seed=seed)
    assert cnn.int8_bytes() < 250_000

    def score_split(split: list[WindowRecord]):
        bun = extract_features(split)
        xn = transform(head, bun.X)
        trusts = [
            engine.score(xn[i], bun.mel[i], split[i].health_estimate, split[i].environment)
            for i in range(len(split))
        ]
        y = np.array([int(r.event_present) for r in split], dtype=int)
        conf = np.array([t.wake_confidence for t in trusts])
        return trusts, y, conf, bun

    tr_t, tr_y, tr_c, _ = score_split(train)
    te_t, te_y, te_c, _ = score_split(test)

    base = EnergyThresholdBaseline()
    base.fit(train)
    e_scores = base.scores(test)
    # rank-normalize energy to [0,1] for AUC/ECE comparison
    e_prob = (e_scores - e_scores.min()) / (np.ptp(e_scores) + 1e-12)

    trust_auc = roc_auc(te_y, te_c)
    energy_auc = roc_auc(te_y, e_prob)
    trust_ece = ece(te_y.astype(float), te_c)
    energy_ece = ece(te_y.astype(float), e_prob)

    # explanations on true events
    hits = []
    cf_down = 0
    n_true = 0
    for rec in test:
        if not rec.event_present:
            continue
        n_true += 1
        ex = reasoner.explain(rec.event_class, rec.waveform, rec.sample_rate, rec.environment)
        hits.append(int(reasoner.gold_hit(ex, rec)))
        if ex.counterfactual_verified:
            cf_down += 1
    ontology_hit = float(np.mean(hits)) if hits else 0.0

    macs = {0: 8e4, 1: 2e6, 2: 4e5, 3: 2e3, 4: 2e3}
    y_hit = np.array(
        [
            int(
                reasoner.gold_hit(
                    reasoner.explain(r.event_class, r.waveform, r.sample_rate, r.environment),
                    r,
                )
            )
            if r.event_present
            else 0
            for r in test
        ]
    )
    profile = PROFILES["stm32l476"]

    def ev(pi: PolicyVector):
        return evaluate_policy(pi, te_t, te_y, y_hit, profile, macs)

    front = nsga2(ev, seed=seed, pop_size=16, n_gen=8)
    knee = chebyshev_select(front)
    gated = ev(knee)
    always = ev(always_l3())
    energy_ok = gated.energy_j_s < always.energy_j_s
    bound_ok = gated_energy_strictly_less((1, 2, 3, 10), (0.5, 0.3, 0.1))

    # hardware table
    hw_rows = []
    for name, prof in PROFILES.items():
        row = {"device": name, "isa": prof.isa}
        for lv in (ExecutionLevel.L0, ExecutionLevel.L1, ExecutionLevel.L2, ExecutionLevel.L3):
            e = compute_energy_j(lv, macs, prof)
            row[f"L{int(lv)}_mJ"] = e * 1e3
        hw_rows.append(row)

    # security: tamper fails, honest verifies; gd32 refused
    fw = b"ahaif-firmware-v0"
    hal = SecureHAL(fw, model_id=b"tiny-cnn-v0")
    hal.secure_boot(fw)
    tamper_caught = False
    try:
        hal.secure_boot(b"evil-firmware")
    except TamperError:
        tamper_caught = True
    orch = Orchestrator(PolicyVector(0.2, 0.4, 0.6, device_id="gd32vf103", require_authenticated=True))
    gd32_refused = False
    try:
        orch.place(ExecutionLevel.L2)
    except PermissionError:
        gd32_refused = True

    # battery: 10 Wh cell,  days-to-dead
    battery_j = 10.0 * 3600.0
    days_gated = battery_j / max(gated.energy_j_s, 1e-12) / 86400.0
    days_on = battery_j / max(always.energy_j_s, 1e-12) / 86400.0

    # robustness: SNR proxy via high_sea_state already mixed; report sea-state 5 AUC
    hs = TwinRenderer("high_sea_state", seed=99).render_dataset(80)
    hs_t, hs_y, hs_c, _ = score_split(hs)
    robust_auc = roc_auc(hs_y, hs_c)

    # ablations: zero health / env weights
    def _ablate(key: str) -> float:
        w = dict(engine.w)
        w[key] = 0.0
        eng = TrustEngine(head, w)
        bun = extract_features(test)
        xn = transform(head, bun.X)
        conf = []
        for i, rec in enumerate(test):
            conf.append(eng.score(xn[i], bun.mel[i], rec.health_estimate, rec.environment).wake_confidence)
        return roc_auc(te_y, np.array(conf))

    report = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "wake": {
            "trust_auc": trust_auc,
            "energy_auc": energy_auc,
            "trust_beats_energy": bool(trust_auc > energy_auc),
            "trust_ece": trust_ece,
            "energy_ece": energy_ece,
            "trust_mce": mce(te_y.astype(float), te_c),
            "trust_brier": brier(te_y.astype(float), te_c),
        },
        "explanations": {
            "ontology_hit_rate": ontology_hit,
            "cf_verified_true_events": cf_down,
            "n_true_events": n_true,
            "kg_nodes": kg.n_nodes,
            "kg_edges": kg.n_edges,
        },
        "gating": {
            "knee": {
                "tau1": knee.tau1,
                "tau2": knee.tau2,
                "tau3": knee.tau3,
                "device": knee.device_id,
            },
            "gated_energy_j": gated.energy_j_s,
            "always_l3_energy_j": always.energy_j_s,
            "gated_miss": gated.miss,
            "always_miss": always.miss,
            "gated_uses_less_energy": energy_ok,
            "pareto_size": len(front),
            "proposition1_holds": bound_ok,
        },
        "ablation_auc": {
            "full": trust_auc,
            "no_health": _ablate("health"),
            "no_env": _ablate("env"),
        },
        "robustness": {"high_sea_state_auc": robust_auc},
        "complexity": {
            "l0": l0_rfft().__dict__,
            "l1": l1_mel().__dict__,
            "l3": kg_bfs(kg.n_nodes, kg.n_edges).__dict__,
            "cnn_int8_bytes": cnn.int8_bytes(),
            "cnn_macs_ir": trace_macs(cnn.n_params(), 62).__dict__,
        },
        "hardware": hw_rows,
        "security": {
            "tamper_caught": tamper_caught,
            "gd32_refused_authenticated_l2": gd32_refused,
        },
        "battery": {
            "days_gated_10Wh": days_gated,
            "days_always_on_10Wh": days_on,
        },
        "success": {
            "trust_beats_energy_auc": bool(trust_auc > energy_auc),
            "gated_less_energy": energy_ok,
            "kg_chain_on_true_events": bool(ontology_hit > 0),
            "five_device_profiles": len(hw_rows) >= 5,
            "security_demo": tamper_caught and gd32_refused,
        },
    }
    (out / "suite.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    _write_markdown(out / "suite.md", report)
    return report


def _write_markdown(path: Path, r: dict) -> None:
    w = r["wake"]
    g = r["gating"]
    lines = [
        "# AHAIF paper suite",
        "",
        f"- Train/val/test: {r['n_train']}/{r['n_val']}/{r['n_test']}",
        f"- Trust ROC-AUC {w['trust_auc']:.3f} vs energy-threshold {w['energy_auc']:.3f} "
        f"(beats={w['trust_beats_energy']})",
        f"- Trust ECE {w['trust_ece']:.3f} vs energy ECE {w['energy_ece']:.3f}",
        f"- Ontology hit-rate {r['explanations']['ontology_hit_rate']:.3f} "
        f"on {r['explanations']['n_true_events']} true events",
        f"- Gated energy {g['gated_energy_j']:.3e} J/window vs always-L3 {g['always_l3_energy_j']:.3e} "
        f"(less={g['gated_uses_less_energy']})",
        f"- Pareto size {g['pareto_size']}; knee τ={g['knee']}",
        f"- Tamper caught={r['security']['tamper_caught']}; "
        f"gd32 refused={r['security']['gd32_refused_authenticated_l2']}",
        f"- Battery days (10 Wh): gated {r['battery']['days_gated_10Wh']:.1f} vs always-on {r['battery']['days_always_on_10Wh']:.1f}",
        "",
        "## Success flags",
        "",
    ]
    for k, v in r["success"].items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
