from __future__ import annotations

from uaere.hardware.profiles import load_profile
from uaere.swarm.field import SwarmField


def test_swarm_one_source_many_ears():
    field = SwarmField(n_nodes=6, seed=3, extent_m=800.0)
    field.warmup(n_train=48)
    tick = field.step()
    assert len(tick.nodes) == 6
    assert tick.source["class"]
    # Heterogeneous cheap field
    profiles = {n.profile for n in tick.nodes}
    assert len(profiles) >= 2
    jsonable = [n.__dict__ for n in tick.nodes]
    assert jsonable[0]["node_id"].startswith("n")


def test_pi_profiles_load():
    p = load_profile("raspberry_pi_zero2")
    assert p.supports_l3
    assert p.cpu_mhz >= 1000
