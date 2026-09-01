from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from uaere.types import EventClass

_ONTOLOGY = Path(__file__).with_name("ontology.yaml")

CLASS_TO_EVENT = {
    EventClass.TUG: "event.tug_pass",
    EventClass.CARGO: "event.cargo_pass",
    EventClass.TANKER: "event.tanker_pass",
    EventClass.PASSENGER: "event.passenger_pass",
    EventClass.BIOLOGICAL: "event.biological",
    EventClass.GEOPHONY: "event.geophony",
    EventClass.ARTIFACT: "event.artifact",
    EventClass.REJECT: "event.ambient",
}


class MarineAcousticKG:
    def __init__(self, graph: nx.MultiDiGraph, meta: dict[str, dict]) -> None:
        self.g = graph
        self.meta = meta

    def event_node(self, cls: EventClass) -> str:
        return CLASS_TO_EVENT[cls]

    def causes(self, event_id: str) -> list[tuple[str, dict]]:
        out = []
        if event_id not in self.g:
            return out
        for _, dst, data in self.g.out_edges(event_id, data=True):
            if data.get("rel") == "caused_by":
                out.append((dst, self.meta.get(dst, {})))
        return out

    def cues(self, node_id: str) -> dict[str, Any]:
        return dict(self.meta.get(node_id, {}).get("cues") or {})

    def sea_state_min(self, node_id: str) -> int:
        return int(self.meta.get(node_id, {}).get("sea_state_min") or 0)

    def to_turtle(self) -> str:
        lines = ["@prefix ma: <https://uaere.local/marine#> .", ""]
        for nid, meta in self.meta.items():
            lines.append(f"ma:{nid.replace('.', '_')} a ma:{meta.get('type', 'Node')} ;")
            lines.append(f'  ma:label "{meta.get("label", nid)}" .')
        for u, v, data in self.g.edges(data=True):
            rel = data.get("rel", "related")
            lines.append(
                f"ma:{u.replace('.', '_')} ma:{rel} ma:{v.replace('.', '_')} ."
            )
        return "\n".join(lines) + "\n"

    @property
    def n_nodes(self) -> int:
        return self.g.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.g.number_of_edges()


def load_kg(path: str | Path | None = None) -> MarineAcousticKG:
    p = Path(path) if path else _ONTOLOGY
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    g = nx.MultiDiGraph()
    meta: dict[str, dict] = {}
    for node in raw["nodes"]:
        nid = node["id"]
        g.add_node(nid, **node)
        meta[nid] = node
    for e in raw["edges"]:
        g.add_edge(e["src"], e["dst"], rel=e["rel"])
    return MarineAcousticKG(g, meta)
