"""Dense → ReLU → evidence, or a fitted logistic mapped to a Dirichlet.

α = 1 + τ p̂  preserves the Dirichlet mean while giving a concentration that
shrinks as the classifier becomes uncertain.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from uaere.math.dirichlet import strengths_from_evidence
from uaere.types import N_EVENT_CLASSES, FloatArray


class EvidentialHead:
    def __init__(self, in_dim: int, n_classes: int = N_EVENT_CLASSES, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        hidden = 32
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.w1 = rng.normal(0, 0.2, size=(hidden, in_dim))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 0.2, size=(n_classes, hidden))
        self.b2 = np.zeros(n_classes)
        self.sklearn: LogisticRegression | None = None
        self.concentration = 20.0
        self._feat_mu = np.zeros(in_dim)
        self._feat_sd = np.ones(in_dim)

    def evidence(self, feats: FloatArray) -> FloatArray:
        h = np.maximum(self.w1 @ feats + self.b1, 0.0)
        return self.w2 @ h + self.b2

    def alpha(self, feats: FloatArray) -> FloatArray:
        x = np.asarray(feats, dtype=np.float64).reshape(-1)
        if self.sklearn is not None:
            p = np.ones(self.n_classes) / self.n_classes
            proba = self.sklearn.predict_proba(x.reshape(1, -1))[0]
            for i, c in enumerate(self.sklearn.classes_):
                p[int(c)] = float(proba[i])
            return 1.0 + self.concentration * p
        return strengths_from_evidence(self.evidence(x))

    def parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]
