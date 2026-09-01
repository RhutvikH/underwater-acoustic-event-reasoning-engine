"""Multi-node residual energy + WHOI Micro-Modem-class TX cost."""

from __future__ import annotations

from dataclasses import dataclass, field


# Joules per bit vs range, calibrated to published WHOI Micro-Modem order-of-magnitude
# figures (tens of nJ–µJ/bit depending on rate/range). We use a conservative µJ/bit.
def tx_energy_j(n_bits: int, range_m: float) -> float:
    uj_per_bit = 0.8 * (1.0 + range_m / 1000.0) ** 2
    return n_bits * uj_per_bit * 1e-6


@dataclass
class AcousticNode:
    node_id: str
    xyz: tuple[float, float, float]
    battery_j: float
    profile_id: str
    residual_j: float = field(init=False)

    def __post_init__(self) -> None:
        self.residual_j = self.battery_j

    def spend(self, joules: float) -> None:
        self.residual_j = max(0.0, self.residual_j - joules)


class NetworkTwin:
    def __init__(self, nodes: list[AcousticNode], sink_xyz: tuple[float, float, float]) -> None:
        self.nodes = {n.node_id: n for n in nodes}
        self.sink_xyz = sink_xyz

    def collab_wake_cost(self, src_id: str, dst_id: str, bits: int = 256) -> float:
        a = self.nodes[src_id]
        b = self.nodes[dst_id]
        r = (
            (a.xyz[0] - b.xyz[0]) ** 2
            + (a.xyz[1] - b.xyz[1]) ** 2
            + (a.xyz[2] - b.xyz[2]) ** 2
        ) ** 0.5
        return tx_energy_j(bits, r)
