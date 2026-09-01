"""Energy identities.

E_compute = c0 + p1 c1 + p2 c2 + p3 c3
"""

from __future__ import annotations

from uaere.types import DeviceProfile, ExecutionLevel


def expected_compute_energy(
    c: tuple[float, float, float, float],
    admit_prob: tuple[float, float, float],
) -> float:
    """c = (c0,c1,c2,c3) joules/window; admit_prob = (p1,p2,p3)."""
    c0, c1, c2, c3 = c
    p1, p2, p3 = admit_prob
    return float(c0 + p1 * c1 + p2 * c2 + p3 * c3)


def macs_energy_j(n_macs: float, profile: DeviceProfile) -> float:
    return float(n_macs) * profile.nj_per_mac_int8 * 1e-9


def time_s_from_macs(n_macs: float, profile: DeviceProfile) -> float:
    if profile.has_npu and profile.npu_mac_per_s > 0:
        return float(n_macs) / profile.npu_mac_per_s
    # ~1 MAC / cycle on CMSIS-NN-class cores as a conservative bound
    cycles = float(n_macs)
    return cycles / (profile.cpu_mhz * 1e6)


def compute_energy_j(
    level: ExecutionLevel,
    macs_by_level: dict[int, float],
    profile: DeviceProfile,
    tx_bits: int = 0,
) -> float:
    total_macs = sum(macs_by_level[k] for k in range(int(level) + 1) if k in macs_by_level)
    e = macs_energy_j(total_macs, profile)
    t = time_s_from_macs(total_macs, profile)
    e += (profile.active_mw * 1e-3) * t
    if tx_bits:
        e += tx_bits * profile.tx_nj_per_bit * 1e-9
    return float(e)
