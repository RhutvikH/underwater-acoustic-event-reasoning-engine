"""Pydantic configs. Frozen experiment hashes go in paper/experiment_protocol.md."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    sample_rate: int = 16_000
    window_s: float = 1.0
    hop_s: float = 0.5
    n_mels: int = 32
    n_fft: int = 512
    hop_length: int = 256
    n_classes: int = 5
    deepship_root: str | None = None
    shipsear_root: str | None = None


class ModelConfig(BaseModel):
    hidden: int = 32
    cnn_channels: tuple[int, int] = (8, 16)
    adaptation_alpha: float = 0.05
    trust_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "wake": 2.2,
            "health": 0.8,
            "env": 1.1,
            "uncertainty": 1.4,
            "bias": -0.6,
        }
    )


class TwinConfig(BaseModel):
    scenario: str = "busy_strait"
    n_nodes: int = 4
    seed: int = 0
    duration_s: float = 30.0


class EvalConfig(BaseModel):
    seeds: tuple[int, ...] = (0, 1, 2)
    n_bootstrap: int = 1000
    alpha: float = 0.05
    reliability_bins: int = 10


class AhaifConfig(BaseModel):
    seed: int = 0
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    twin: TwinConfig = Field(default_factory=TwinConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_config(path: str | Path | None = None) -> AhaifConfig:
    if path is None:
        return AhaifConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AhaifConfig.model_validate(raw)
