"""Raspberry Pi / cheap-Linux edge node.

The scientific pipeline is numpy and already fits a Pi 4 or Pi Zero 2 W.
This module is the *deployable* loop: listen (WAV or twin), infer, optionally
UDP-broadcast a collab-wake so a field of Pis behaves like the swarm twin.

On a laptop, `simulate_pi_field` runs N in-process nodes to prove the same
protocol before SD-card flashing.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass

from uaere.hardware.profiles import load_profile
from uaere.pipeline import AhaifPipeline
from uaere.types import WindowRecord


COLLAB_PORT = 7946


@dataclass
class EdgeNode:
    node_id: str
    profile_id: str
    pipeline: AhaifPipeline
    bind: str = "0.0.0.0"
    port: int = COLLAB_PORT

    def infer(self, rec: WindowRecord) -> dict:
        result = self.pipeline.infer(rec)
        return {
            "node_id": self.node_id,
            "profile": self.profile_id,
            "level": int(result.level),
            "wake": result.trust.wake_confidence,
            "trust": result.trust.event_trust,
            "event_class": result.event_class.value,
            "explanation": result.explanation.sentence if result.explanation else None,
            "energy_j": result.energy_j,
            "reason": result.reason,
            "authenticated": result.authenticated,
        }

    def should_collab(self, trust: float, lo: float = 0.4, hi: float = 0.7) -> bool:
        return lo <= trust <= hi

    def broadcast_wake(self, payload: dict, broadcast: str = "255.255.255.255") -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        msg = json.dumps({"type": "collab_wake", **payload}).encode("utf-8")
        sock.sendto(msg, (broadcast, self.port))
        sock.close()


def simulate_pi_field(pipeline: AhaifPipeline, records: list[WindowRecord], n: int = 4) -> list[dict]:
    """Laptop stand-in for a row of Pi Zero nodes sharing one recording."""
    out = []
    for i, rec in enumerate(records[:n]):
        node = EdgeNode(node_id=f"pi-{i}", profile_id="raspberry_pi_zero2", pipeline=pipeline)
        load_profile(node.profile_id)  # fail fast if unknown
        out.append(node.infer(rec))
    return out
