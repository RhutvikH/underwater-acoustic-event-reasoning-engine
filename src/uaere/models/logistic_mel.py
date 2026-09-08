"""Default backbone: class-balanced logistic → Dirichlet lift."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from uaere.classify.train import extract_features, transform
from uaere.trust.evidential_head import EvidentialHead
from uaere.types import FloatArray, N_EVENT_CLASSES, WindowRecord


class LogisticMelBackbone:
    name = "logistic_mel"

    def __init__(self, seed: int = 0, steps: int = 200) -> None:
        self.seed = seed
        self.steps = steps
        self.head = EvidentialHead(in_dim=8, n_classes=N_EVENT_CLASSES, seed=seed)

    def fit_records(self, records: list[WindowRecord]) -> EvidentialHead:
        from uaere.classify.train import train_evidential

        bun = extract_features(records)
        self.head = train_evidential(bun, steps=self.steps, seed=self.seed)
        return self.head

    def fit(self, X: FloatArray, y: FloatArray) -> None:
        clf = LogisticRegression(
            max_iter=max(self.steps, 80),
            random_state=self.seed,
            solver="lbfgs",
            class_weight="balanced",
        )
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
        xn = (X - mu) / sd
        clf.fit(xn, y)
        self.head = EvidentialHead(in_dim=xn.shape[1], seed=self.seed)
        self.head._feat_mu = mu
        self.head._feat_sd = sd
        self.head.sklearn = clf

    def evidence(self, x: FloatArray) -> FloatArray:
        xn = transform(self.head, np.asarray(x, dtype=np.float64).reshape(1, -1))[0]
        return self.head.alpha(xn)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            backbone=np.array(self.name),
            w1=self.head.w1,
            b1=self.head.b1,
            w2=self.head.w2,
            b2=self.head.b2,
            feat_mu=getattr(self.head, "_feat_mu", np.zeros(1)),
            feat_sd=getattr(self.head, "_feat_sd", np.ones(1)),
        )
