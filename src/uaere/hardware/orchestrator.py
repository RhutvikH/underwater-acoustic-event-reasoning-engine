"""Map pipeline levels onto device profiles under energy, latency, security."""

from __future__ import annotations

from uaere.hardware.profiles import PROFILES, load_profile
from uaere.types import DeviceProfile, ExecutionLevel, PolicyVector


class Orchestrator:
    def __init__(self, policy: PolicyVector) -> None:
        self.policy = policy

    def place(self, level: ExecutionLevel) -> DeviceProfile:
        profile = load_profile(self.policy.device_id)
        if self.policy.require_authenticated and not _attestable(profile):
            raise PermissionError(
                f"device {profile.device_id} cannot host authenticated TinyML "
                "(no secure element / secure boot)"
            )
        if level >= ExecutionLevel.L2 and not profile.supports_l2:
            raise PermissionError(f"{profile.device_id} does not support L2")
        if level >= ExecutionLevel.L3 and not profile.supports_l3:
            raise PermissionError(f"{profile.device_id} does not support L3")
        return profile

    def cheapest_attestable(self, level: ExecutionLevel) -> DeviceProfile:
        candidates = []
        for p in PROFILES.values():
            if self.policy.require_authenticated and not _attestable(p):
                continue
            if level >= ExecutionLevel.L2 and not p.supports_l2:
                continue
            if level >= ExecutionLevel.L3 and not p.supports_l3:
                continue
            candidates.append(p)
        if not candidates:
            raise PermissionError("no attestable device can host this level")
        return min(candidates, key=lambda p: p.nj_per_mac_int8)


def _attestable(p: DeviceProfile) -> bool:
    return p.has_secure_element and p.has_secure_boot
