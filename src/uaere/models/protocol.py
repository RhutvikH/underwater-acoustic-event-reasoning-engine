"""Backbone protocol: any model that maps Mel-stats → Dirichlet evidence."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from uaere.types import FloatArray


class Backbone(Protocol):
    name: str

    def fit(self, X: FloatArray, y: FloatArray) -> None: ...

    def evidence(self, x: FloatArray) -> FloatArray: ...

    def predict_label(self, x: FloatArray) -> int:
        e = np.asarray(self.evidence(x), dtype=np.float64)
        return int(np.argmax(e))
