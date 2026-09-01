"""Tiny IR for INT8 MAC accounting. Orchestrator must not need TensorFlow."""

from __future__ import annotations

from dataclasses import dataclass

from uaere.representation.encoder import cnn_macs
from uaere.types import N_MELS


@dataclass
class ModelIR:
    name: str
    n_params: int
    int8_bytes: int
    macs_per_window: int

    def under_budget(self, bytes_limit: int = 250_000) -> bool:
        return self.int8_bytes <= bytes_limit


def trace_macs(n_params: int, mel_frames: int, c1: int = 8, c2: int = 16) -> ModelIR:
    macs = cnn_macs((N_MELS, mel_frames), c1=c1, c2=c2)
    return ModelIR(
        name="tiny-dw-cnn",
        n_params=n_params,
        int8_bytes=n_params,
        macs_per_window=macs,
    )
