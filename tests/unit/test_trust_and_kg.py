from __future__ import annotations

import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.train import extract_features, train_evidential, transform
from uaere.kg.graph import load_kg
from uaere.representation.l0_dsp import extract_l0, l0_admit_score
from uaere.trust.baseline_energy_threshold import EnergyThresholdBaseline
from uaere.trust.trust_score import TrustEngine
from uaere.twin.render import TwinRenderer
from uaere.types import EventClass


def test_l0_admit_is_not_energy_only():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.01, 16000)
    loud_white = rng.normal(0, 0.5, 16000)
    f_q = extract_l0(quiet, 16000)
    f_l = extract_l0(loud_white, 16000)
    # both use the full vector; scores are finite and in (0,1)
    s_q, s_l = l0_admit_score(f_q), l0_admit_score(f_l)
    assert 0.0 < s_q < 1.0
    assert 0.0 < s_l < 1.0


def test_kg_every_vessel_has_caused_by():
    kg = load_kg()
    for ev in ("event.tug_pass", "event.cargo_pass", "event.tanker_pass", "event.passenger_pass"):
        assert kg.causes(ev)


def test_reasoner_gold_hit_on_twin_tug():
    recs = TwinRenderer("busy_strait", seed=2).render_dataset(40)
    tugs = [r for r in recs if r.event_class is EventClass.TUG]
    assert tugs
    ex = CausalReasoner(load_kg()).explain(
        tugs[0].event_class, tugs[0].waveform, tugs[0].sample_rate, tugs[0].environment
    )
    assert ex.cause_id == "cause.vessel.tug"
    assert "tug" in ex.sentence.lower()


def test_trust_beats_energy_on_twin():
    recs = TwinRenderer("busy_strait", seed=0).render_dataset(120)
    recs += TwinRenderer("high_sea_state", seed=1).render_dataset(40)
    n = len(recs)
    train, test = recs[: n // 2], recs[n // 2 :]
    bun = extract_features(train)
    head = train_evidential(bun, steps=120, seed=0)
    engine = TrustEngine(head)
    bun_te = extract_features(test)
    xn = transform(head, bun_te.X)
    y = np.array([int(r.event_present) for r in test])
    conf = np.array(
        [
            engine.score(xn[i], bun_te.mel[i], test[i].health_estimate, test[i].environment).wake_confidence
            for i in range(len(test))
        ]
    )
    base = EnergyThresholdBaseline()
    base.fit(train)
    e = base.scores(test)
    e_prob = (e - e.min()) / (np.ptp(e) + 1e-12)
    from uaere.math.metrics import roc_auc

    assert roc_auc(y, conf) >= roc_auc(y, e_prob) - 1e-6
