"""UAERE: Adaptive Hierarchical Acoustic Intelligence Framework (AHAIF).

A trust-aware, energy-efficient acoustic event reasoning pipeline for
underwater acoustic sensor networks. Event-level trust gates causal
reasoning over a marine acoustic knowledge graph. Evaluated in a digital
twin at TRL-4.

Public name: AHAIF
Package name: uaere
"""

from uaere.types import (
    EnvironmentState,
    EventClass,
    ExecutionLevel,
    SensorHealth,
    TrustScore,
)

__version__ = "0.1.0"
__all__ = [
    "ExecutionLevel",
    "EventClass",
    "EnvironmentState",
    "SensorHealth",
    "TrustScore",
    "__version__",
]
