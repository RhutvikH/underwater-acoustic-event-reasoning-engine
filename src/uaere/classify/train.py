"""Train evidential MLP on Mel statistics. Fast CPU path for TRL-4 experiments.

The CNN provides MAC counts and a second head; the paper's reported trust
scores come from this calibrated evidential MLP on φ1 summaries, which is
what actually runs at L1 on a Cortex-M.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.representation.l1_tf import log_mel
from uaere.trust.evidential_head import EvidentialHead
from uaere.types import CLASS_INDEX, EventClass, FloatArray, WindowRecord


@dataclass
class FeatureBundle:
    X: FloatArray
    y: FloatArray
    mel: list[FloatArray]
    records: list[WindowRecord]


def _mel_stats(mel: FloatArray) -> FloatArray:
    # mean, std, max over time, plus low/mid/high band means
    mu = mel.mean(axis=1)
    sd = mel.std(axis=1)
    mx = mel.max(axis=1)
    n = mel.shape[0]
    bands = np.array(
        [
            mel[: n // 3].mean(),
            mel[n // 3 : 2 * n // 3].mean(),
            mel[2 * n // 3 :].mean(),
        ]
    )
    return np.concatenate([mu, sd, mx, bands])


def extract_features(
    records: list[WindowRecord],
    normalizer: AdaptiveNormalizer | None = None,
) -> FeatureBundle:
    xs = []
    ys = []
    mels = []
    norm = normalizer or AdaptiveNormalizer(n_bins=32)
    for r in records:
        mel = log_mel(r.waveform, r.sample_rate)
        # running noise from mean-over-time of linear-ish energy
        psd = np.exp(mel).mean(axis=1)
        norm.update_noise(psd, is_event=r.event_present)
        mel_n = norm.normalize_mel(mel, r.environment)
        xs.append(_mel_stats(mel_n))
        ys.append(CLASS_INDEX.get(r.event_class, CLASS_INDEX[EventClass.REJECT]))
        mels.append(mel_n)
    return FeatureBundle(X=np.stack(xs), y=np.array(ys, dtype=int), mel=mels, records=records)


def train_evidential(
    bundle: FeatureBundle,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
) -> EvidentialHead:
    """Fit a multinomial logistic map, then lift p̂ to a Dirichlet via α = 1 + τp̂.

    `steps` maps to sklearn `max_iter`. `lr` is unused (kept for CLI compat).
    """
    from sklearn.linear_model import LogisticRegression

    x = bundle.X
    y = bundle.y
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
    xn = (x - mu) / sd
    head = EvidentialHead(in_dim=xn.shape[1], seed=seed)
    head._feat_mu = mu
    head._feat_sd = sd
    clf = LogisticRegression(
        max_iter=max(int(steps), 80),
        random_state=int(seed),
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(xn, y)
    head.sklearn = clf
    _ = lr
    return head


def transform(head: EvidentialHead, X: FloatArray) -> FloatArray:
    mu = getattr(head, "_feat_mu", 0.0)
    sd = getattr(head, "_feat_sd", 1.0)
    return (X - mu) / sd
