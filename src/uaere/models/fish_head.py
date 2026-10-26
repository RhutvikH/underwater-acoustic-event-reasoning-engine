"""Fish-call head: logistic Mel statistics, saved with joblib, used as the label.

The six-way ship head is a different checkpoint. This one names the classes
it was trained on, and the negative class (usually ``other``) is the reject
mass inside wake confidence.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from uaere.classify.train import _mel_stats, mel_feature_rows, transform
from uaere.data.adapters import FISH_DATA_DOI
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.representation.l1_tf import log_mel
from uaere.trust.evidential_head import EvidentialHead
from uaere.trust.trust_score import TrustEngine
from uaere.types import (
    NEGATIVE_LABELS,
    EnvironmentState,
    FloatArray,
    SensorHealth,
    TrustScore,
    WindowRecord,
)

DEFAULT_FISH_CHECKPOINT = Path("artifacts/models/fish_head.joblib")
_BAND: tuple[float, float] | None = None


def toadfish_band() -> tuple[float, float]:
    """Frequency bounds stored on ``cause.taxon.opsanus_tau``. Not a new band."""
    global _BAND
    if _BAND is None:
        from uaere.kg.graph import load_kg

        cues = load_kg().cues("cause.taxon.opsanus_tau")
        _BAND = (float(cues["f_lo"]), float(cues["f_hi"]))
    return _BAND


def band_features(waveform: FloatArray, sample_rate: int) -> np.ndarray:
    """Power, flatness, and period peak inside the stored boatwhistle band.

    The three numbers are read from the waveform. The classifier label is not
    an input. Bounds come from the knowledge-graph node.
    """
    f_lo, f_hi = toadfish_band()
    x = np.asarray(waveform, dtype=np.float64)
    fs = int(sample_rate)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    total = float(spec.sum()) + 1e-12
    band = (freqs >= f_lo) & (freqs <= f_hi)
    frac = float(spec[band].sum() / total)
    sb = spec[band]
    flat = (
        float(np.exp(np.mean(np.log(sb + 1e-12))) / (np.mean(sb) + 1e-12))
        if sb.size
        else 1.0
    )
    ac = np.fft.irfft(spec)
    ac = ac / (ac[0] + 1e-12)
    lo = max(1, int(fs / f_hi))
    hi = min(len(ac) - 1, int(round(fs / f_lo)))
    peak = float(np.max(ac[lo:hi])) if hi > lo else 0.0
    return np.array([frac, flat, peak], dtype=np.float64)


def fish_feature_rows(
    records: list[WindowRecord],
) -> tuple[np.ndarray, list]:
    """Mel statistics plus the three boatwhistle-band numbers."""
    stats, mels = mel_feature_rows(records)
    extra = np.vstack([band_features(r.waveform, r.sample_rate) for r in records])
    return np.hstack([stats, extra]), mels


def vector_for_head(
    head: EvidentialHead,
    waveform: FloatArray,
    sample_rate: int,
    mel_n: np.ndarray,
) -> np.ndarray:
    """Match a saved fish head. Ship heads and older fish checkpoints stay on mel stats."""
    stats = _mel_stats(np.asarray(mel_n, dtype=np.float64))
    if getattr(head, "labels", None) is None or getattr(head, "negative_index", None) is None:
        return stats
    extra = band_features(waveform, sample_rate)
    full = np.concatenate([stats, extra])
    width = int(np.asarray(head._feat_mu).shape[0])
    if full.shape[0] == width:
        return full
    if stats.shape[0] == width:
        return stats
    raise ValueError(
        f"fish head expects {width} features, mel stats are {stats.shape[0]}, "
        f"with the boatwhistle band {full.shape[0]}"
    )


def label_of(record: WindowRecord) -> str:
    return record.taxon or record.event_class.value


def ordered_labels(records: list[WindowRecord]) -> tuple[str, ...]:
    names = sorted({label_of(r) for r in records})
    positives = [n for n in names if n not in NEGATIVE_LABELS]
    negatives = [n for n in names if n in NEGATIVE_LABELS]
    if not positives or not negatives:
        raise ValueError(
            f"fish head needs a call class and a negative class; got {names}"
        )
    return tuple(positives + negatives)


def fit_fish_head(
    records: list[WindowRecord],
    steps: int = 200,
    seed: int = 0,
) -> EvidentialHead:
    labels = ordered_labels(records)
    index = {name: i for i, name in enumerate(labels)}
    x, _ = fish_feature_rows(records)
    y = np.array([index[label_of(r)] for r in records], dtype=int)
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-6
    xn = (x - mu) / sd
    clf = LogisticRegression(
        max_iter=max(int(steps), 80),
        random_state=int(seed),
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(xn, y)
    neg_name = "other" if "other" in labels else next(n for n in labels if n in NEGATIVE_LABELS)
    head = EvidentialHead(in_dim=xn.shape[1], n_classes=len(labels), seed=seed)
    head._feat_mu = mu
    head._feat_sd = sd
    head.sklearn = clf
    head.labels = labels
    head.negative_index = index[neg_name]
    return head


def save_fish_checkpoint(
    head: EvidentialHead,
    path: str | Path,
    meta: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if head.labels is None or head.sklearn is None or head.negative_index is None:
        raise ValueError("fish checkpoint requires a fitted head with labels")
    blob = {
        "kind": "fish_logistic_mel",
        "sklearn": head.sklearn,
        "feat_mu": np.asarray(head._feat_mu, dtype=np.float64),
        "feat_sd": np.asarray(head._feat_sd, dtype=np.float64),
        "labels": list(head.labels),
        "negative_index": int(head.negative_index),
        "concentration": float(head.concentration),
        "sample_rate": 16_000,
        "window_s": 1.0,
        "doi": FISH_DATA_DOI,
        "features": "mel_stats+opsanus_band",
        "band_hz": list(toadfish_band()),
        "meta": meta or {},
    }
    joblib.dump(blob, path)
    return path


def load_fish_checkpoint(path: str | Path) -> tuple[EvidentialHead, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No fish checkpoint at {path}. Train one with "
            "`uaere train --dataset fish --root <unzipped wavclips>`. "
            "Predict will not train a ship model in its place."
        )
    blob = joblib.load(path)
    if blob.get("kind") != "fish_logistic_mel":
        raise ValueError(f"{path} is not a fish_logistic_mel checkpoint")
    labels = tuple(blob["labels"])
    mu = np.asarray(blob["feat_mu"], dtype=np.float64)
    head = EvidentialHead(in_dim=int(mu.shape[0]), n_classes=len(labels), seed=0)
    head._feat_mu = mu
    head._feat_sd = np.asarray(blob["feat_sd"], dtype=np.float64)
    head.sklearn = blob["sklearn"]
    head.labels = labels
    head.negative_index = int(blob["negative_index"])
    head.concentration = float(blob.get("concentration", 20.0))
    return head, dict(blob.get("meta") or {})


def score_waveform(
    head: EvidentialHead,
    waveform: FloatArray,
    sample_rate: int,
    environment: EnvironmentState,
    health: SensorHealth,
) -> TrustScore:
    mel = log_mel(waveform, sample_rate)
    mel_n = AdaptiveNormalizer(32).normalize_mel(mel, environment)
    feats = transform(head, vector_for_head(head, waveform, sample_rate, mel_n))
    return TrustEngine(head).score(feats, mel_n, health, environment)
