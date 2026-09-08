"""Report-to-sink on the interferometric atlas.

Edge weight = TX energy / residual battery. Dead (interferometric) hops
have infinite cost. Not Euclidean shortest path; not REAM Q-learning.
"""

from __future__ import annotations

import heapq

from uaere.twin.interferometry import InterferometricAtlas
from uaere.twin.network import AcousticNode, NetworkTwin, tx_energy_j


SINK_ID = "__sink__"


def report_to_sink(
    net: NetworkTwin,
    atlas: InterferometricAtlas,
    src_id: str,
    bits: int = 2048,
) -> dict:
    """Dijkstra. Returns hops, joules, delivered, or undeliverable reason."""
    if src_id not in net.nodes:
        return {"delivered": False, "reason": "unknown_src", "hops": [], "joules": 0.0}
    src = net.nodes[src_id]
    if src.residual_j <= 0:
        return {"delivered": False, "reason": "src_dead", "hops": [], "joules": 0.0}

    ids = list(net.nodes.keys()) + [SINK_ID]

    def pos(nid: str) -> tuple[float, float, float]:
        if nid == SINK_ID:
            return net.sink_xyz
        return net.nodes[nid].xyz

    def residual(nid: str) -> float:
        if nid == SINK_ID:
            return 1e9
        return net.nodes[nid].residual_j

    def weight(a: str, b: str) -> float:
        if a != SINK_ID and b != SINK_ID and atlas.is_dead(a, b):
            return float("inf")
        pa, pb = pos(a), pos(b)
        r = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2) ** 0.5
        e = tx_energy_j(bits, r)
        q = residual(b)
        if q <= 0 and b != SINK_ID:
            return float("inf")
        return e * (1.0 + 1.0 / (q / max(net.nodes[a].battery_j if a != SINK_ID else 1.0, 1e-9) + 1e-3))

    dist = {n: float("inf") for n in ids}
    prev: dict[str, str | None] = {n: None for n in ids}
    dist[src_id] = 0.0
    heap = [(0.0, src_id)]
    seen: set[str] = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u == SINK_ID:
            break
        for v in ids:
            if v == u:
                continue
            w = weight(u, v)
            if w == float("inf"):
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[SINK_ID] == float("inf"):
        return {"delivered": False, "reason": "undeliverable_shadow_or_void", "hops": [], "joules": 0.0}

    hops = []
    cur: str | None = SINK_ID
    while cur is not None:
        hops.append(cur)
        cur = prev[cur]
    hops.reverse()
    # spend TX on each real hop from the sender of that hop
    joules = 0.0
    for a, b in zip(hops[:-1], hops[1:], strict=True):
        pa, pb = pos(a), pos(b)
        r = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2) ** 0.5
        e = tx_energy_j(bits, r)
        joules += e
        if a != SINK_ID:
            net.nodes[a].spend(e)
    return {
        "delivered": True,
        "reason": "ok",
        "hops": hops,
        "joules": joules,
        "src": src_id,
        "kind": "sink_path",
    }
