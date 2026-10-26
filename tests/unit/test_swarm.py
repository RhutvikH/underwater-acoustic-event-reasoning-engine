from __future__ import annotations

from uaere.hardware.profiles import load_profile
from uaere.swarm.field import SwarmField
from uaere.types import EventClass, is_acoustic_event


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


def test_oracle_pam_field_reaches_l3_on_animals():
    field = SwarmField(n_nodes=6, scenario="pam_field", seed=1, extent_m=600.0, model="oracle")
    field.warmup()
    saw_animal = False
    saw_l3 = False
    for _ in range(12):
        tick = field.step()
        cls = EventClass(tick.source["class"])
        if not is_acoustic_event(cls):
            continue
        saw_animal = True
        l3 = [n for n in tick.nodes if n.level >= 3]
        if l3:
            saw_l3 = True
            assert any(n.explanation and n.accepted for n in l3)
            sinks = [lk for lk in tick.links if lk.get("kind") == "sink_path"]
            srcs = {lk["src"] for lk in sinks if lk["src"] != "__sink__"}
            assert len(srcs) <= 1
        if saw_l3:
            break
    assert saw_animal
    assert saw_l3


def test_collab_wake_links_on_uncertain_nodes():
    field = SwarmField(n_nodes=8, scenario="pam_field", seed=7, extent_m=1400.0, model="oracle")
    field.warmup()
    saw_purple = False
    for _ in range(16):
        tick = field.step()
        if any(lk.get("kind") == "collab_wake" for lk in tick.links):
            saw_purple = True
            break
    assert saw_purple


def test_pi_profiles_load():
    p = load_profile("raspberry_pi_zero2")
    assert p.supports_l3
    assert p.cpu_mhz >= 1000
