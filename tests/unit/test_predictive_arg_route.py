from __future__ import annotations

import numpy as np

from uaere.causal.argumentation import Attack, grounded_extension
from uaere.causal.reasoner import CausalReasoner
from uaere.kg.graph import load_kg
from uaere.representation.predictive import surprise, surprise_admit
from uaere.twin.environment import EnvironmentModel
from uaere.twin.interferometry import InterferometricAtlas
from uaere.twin.network import AcousticNode, NetworkTwin
from uaere.twin.routing import report_to_sink
from uaere.twin.sources import synthesize_source
from uaere.types import EventClass, EnvironmentState


def test_tug_more_surprising_than_storm():
    rng = np.random.default_rng(0)
    env = EnvironmentModel()
    calm = env.state(sea_state=2, turbulence=0.05)
    storm = env.state(sea_state=5, turbulence=0.45)
    tug = synthesize_source(EventClass.TUG, 16000, 16000, rng, range_m=1200.0)
    geo = synthesize_source(EventClass.GEOPHONY, 16000, 16000, rng, range_m=80.0)
    s_tug = surprise(tug, 16000, calm)
    s_storm = surprise(geo, 16000, storm)
    assert s_tug > s_storm
    assert surprise_admit(s_tug) > surprise_admit(s_storm)


def test_dung_rejects_rain_in_calm_sea():
    kg = load_kg()
    r = CausalReasoner(kg)
    rng = np.random.default_rng(1)
    wav = synthesize_source(EventClass.TUG, 16000, 16000, rng)
    state = EnvironmentState(sea_state=2)
    expl = r.explain(EventClass.TUG, wav, 16000, state)
    assert "rain" not in expl.cause_id
    joined = " ".join(expl.why_not).lower() + expl.sentence.lower()
    assert "rain" in joined or "attacks" in " ".join(expl.attacks_fired)


def test_grounded_extension_unit():
    res = grounded_extension(
        ["cause.vessel.tug", "cause.env.rain"],
        [Attack("cause.env.rain", "cause.vessel.tug", unless_sea_state_ge=3)],
        sea_state=2,
    )
    assert "cause.vessel.tug" in res.rejected
    res2 = grounded_extension(
        ["cause.vessel.tug", "cause.env.rain"],
        [Attack("cause.env.rain", "cause.vessel.tug", unless_sea_state_ge=3)],
        sea_state=5,
    )
    assert "cause.vessel.tug" in res2.accepted


def test_interferometry_uncorrelated_is_dead():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16000)
    b = rng.standard_normal(16000)
    atlas = InterferometricAtlas(["n0", "n1"], alpha=1.0, dead_thresh=0.2)
    atlas.update("n0", a, "n1", b, is_event=False)
    assert atlas.is_dead("n0", "n1")


def test_sink_path_skips_dead_hop():
    nodes = [
        AcousticNode("n0", (0.0, 0.0, 10.0), 1000.0, "stm32l476"),
        AcousticNode("n1", (100.0, 0.0, 10.0), 1000.0, "stm32l476"),
        AcousticNode("n2", (0.0, 100.0, 10.0), 1000.0, "stm32l476"),
    ]
    net = NetworkTwin(nodes, sink_xyz=(200.0, 200.0, 0.0))
    atlas = InterferometricAtlas(["n0", "n1", "n2"], dead_thresh=0.5)
    atlas.strength[("n0", "n1")] = 0.0
    out = report_to_sink(net, atlas, "n0")
    assert out["delivered"]
    hops = out["hops"]
    pairs = list(zip(hops, hops[1:]))
    assert ("n0", "n1") not in pairs and ("n1", "n0") not in pairs
