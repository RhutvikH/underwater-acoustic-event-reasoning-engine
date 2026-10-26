"""Multi-species head. Logistic is the baseline. An MLP is the candidate.

Validation macro-F1 picks the winner. The test split is not an input here.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

from uaere.representation.l1_tf import log_mel
from uaere.trust.evidential_head import EvidentialHead
from uaere.trust.trust_score import TrustEngine
from uaere.types import EnvironmentState, FloatArray, SensorHealth, TrustScore, WindowRecord

DEFAULT_SPECIES_CHECKPOINT = Path("artifacts/models/species_head.joblib")
N_MELS_SPECIES = 64
N_FRAMES_SPECIES = 32
KINDS = ("species_mlp_mel", "species_logistic_mel", "species_logistic_summary")


def label_of(record: WindowRecord) -> str:
    return record.taxon or record.event_class.value


def ordered_labels(records: list[WindowRecord]) -> tuple[str, ...]:
    names = sorted({label_of(r) for r in records})
    if len(names) < 2:
        raise ValueError(f"species head needs at least two labels; got {names}")
    return tuple(names)


def mel_image(waveform: FloatArray, sample_rate: int) -> np.ndarray:
    """64-band log-mel pooled to 32 frames, then instance-normalised. Shape (2048,)."""
    mel = log_mel(np.asarray(waveform, dtype=np.float64), int(sample_rate), n_mels=N_MELS_SPECIES)
    frames = mel.shape[1]
    if frames <= 0:
        img = np.zeros((N_MELS_SPECIES, N_FRAMES_SPECIES), dtype=np.float64)
    else:
        edges = np.linspace(0, frames, N_FRAMES_SPECIES + 1).astype(int)
        cols = []
        for i in range(N_FRAMES_SPECIES):
            stop = max(int(edges[i + 1]), int(edges[i]) + 1)
            cols.append(mel[:, int(edges[i]) : stop].mean(axis=1))
        img = np.stack(cols, axis=1)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img.reshape(-1)


def mel_summary(waveform: FloatArray, sample_rate: int) -> np.ndarray:
    """Per-band mean, standard deviation, and max. The weak baseline."""
    mel = log_mel(np.asarray(waveform, dtype=np.float64), int(sample_rate), n_mels=N_MELS_SPECIES)
    return np.concatenate([mel.mean(axis=1), mel.std(axis=1), mel.max(axis=1)])


def _matrix(records: list[WindowRecord], kind: str) -> tuple[np.ndarray, np.ndarray]:
    fn = mel_summary if kind == "summary" else mel_image
    rows = [fn(r.waveform, r.sample_rate) for r in records]
    x = np.vstack(rows)
    y = np.array([label_of(r) for r in records], dtype=object)
    return x, y


def _standardise(x: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mu is None:
        mu = x.mean(axis=0)
        sd = x.std(axis=0) + 1e-6
    return (x - mu) / sd, mu, sd


def _macro_f1(clf, x: np.ndarray, y_idx: np.ndarray, n_labels: int) -> float:
    pred = np.asarray(clf.predict(x)).astype(int)
    return float(
        f1_score(y_idx, pred, labels=list(range(n_labels)), average="macro", zero_division=0)
    )


def _as_head(
    clf,
    labels: tuple[str, ...],
    mu: np.ndarray,
    sd: np.ndarray,
    seed: int,
    feature: str,
) -> EvidentialHead:
    index = {name: i for i, name in enumerate(labels)}
    # sklearn classes_ may be the string labels. Map them onto 0..K-1 for alpha().
    if getattr(clf, "classes_", None) is not None and not np.issubdtype(np.asarray(clf.classes_).dtype, np.integer):
        clf.classes_ = np.array([index[str(c)] for c in clf.classes_], dtype=int)
    head = EvidentialHead(in_dim=int(mu.shape[0]), n_classes=len(labels), seed=seed)
    head._feat_mu = np.asarray(mu, dtype=np.float64)
    head._feat_sd = np.asarray(sd, dtype=np.float64)
    head.sklearn = clf
    head.labels = labels
    head.negative_index = None
    head.feature = feature
    return head


def _fit_note(clf) -> dict:
    """Iteration count, and whether the solver stopped before the cap."""
    raw = getattr(clf, "n_iter_", None)
    cap = int(getattr(clf, "max_iter", 0) or 0)
    if raw is None:
        return {"n_iter": None, "converged": None, "max_iter": cap or None}
    n_iter = int(np.max(np.atleast_1d(raw)))
    return {
        "n_iter": n_iter,
        "converged": bool(n_iter < cap) if cap else None,
        "max_iter": cap or None,
    }


def _fit_logistic(x: np.ndarray, y_idx: np.ndarray, seed: int, steps: int) -> LogisticRegression:
    clf = LogisticRegression(
        max_iter=max(int(steps), 200),
        random_state=int(seed),
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(x, y_idx)
    return clf


def _fit_mlp(x: np.ndarray, y_idx: np.ndarray, seed: int, alpha: float) -> MLPClassifier:
    """One hidden layer of 256. Balance uses sample_weight.

    sklearn 1.9 MLPClassifier has no ``class_weight`` argument. Balanced
    sample weights are the same reweighting. Early stopping holds out an
    inner slice of these training rows. A tiny training matrix that cannot
    fill that slice is fit without early stopping, and the flag is stored
    on the estimator as ``uaere_early_stopping``.
    """
    weights = compute_sample_weight("balanced", y_idx)

    def _make(early: bool) -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            alpha=float(alpha),
            early_stopping=early,
            validation_fraction=0.1,
            max_iter=400,
            random_state=int(seed),
        )

    clf = _make(True)
    try:
        clf.fit(x, y_idx, sample_weight=weights)
        clf.uaere_early_stopping = True
    except ValueError:
        clf = _make(False)
        clf.fit(x, y_idx, sample_weight=weights)
        clf.uaere_early_stopping = False
    return clf


def select_species_head(
    train: list[WindowRecord],
    val: list[WindowRecord],
    steps: int = 2000,
    seed: int = 0,
) -> tuple[EvidentialHead, dict]:
    """Fit the declared candidates on train. Pick by validation macro-F1."""
    if not val:
        raise ValueError("species model choice needs a non-empty validation split")
    labels = ordered_labels(train)
    index = {name: i for i, name in enumerate(labels)}
    y_train = np.array([index[label_of(r)] for r in train], dtype=int)
    y_val = np.array([index[label_of(r)] for r in val], dtype=int)
    n_labels = len(labels)

    candidates: list[tuple] = []

    print(f"species fit: {len(train)} train, {len(val)} val, {n_labels} labels", flush=True)
    print("species fit: summary features", flush=True)
    x_sum, _ = _matrix(train, "summary")
    v_sum, _ = _matrix(val, "summary")
    xs, mu_s, sd_s = _standardise(x_sum)
    vs, _, _ = _standardise(v_sum, mu_s, sd_s)
    print("species fit: logistic summary", flush=True)
    log_s = _fit_logistic(xs, y_train, seed, steps)
    candidates.append(
        ("species_logistic_summary", "summary", log_s, mu_s, sd_s, _macro_f1(log_s, vs, y_val, n_labels), x_sum, v_sum)
    )

    print("species fit: image features", flush=True)
    x_img, _ = _matrix(train, "image")
    v_img, _ = _matrix(val, "image")
    xi, mu_i, sd_i = _standardise(x_img)
    vi, _, _ = _standardise(v_img, mu_i, sd_i)
    print("species fit: logistic image", flush=True)
    log_i = _fit_logistic(xi, y_train, seed, steps)
    candidates.append(
        ("species_logistic_mel", "image", log_i, mu_i, sd_i, _macro_f1(log_i, vi, y_val, n_labels), x_img, v_img)
    )
    for alpha in (1e-4, 1e-3):
        print(f"species fit: mlp alpha={alpha}", flush=True)
        mlp = _fit_mlp(xi, y_train, seed, alpha)
        score = _macro_f1(mlp, vi, y_val, n_labels)
        candidates.append(("species_mlp_mel", "image", mlp, mu_i, sd_i, score, x_img, v_img))

    def _preference(row: tuple) -> tuple:
        score = float(row[5])
        kind = row[0]
        mlp = 1 if kind == "species_mlp_mel" else 0
        alpha = getattr(row[2], "alpha", None)
        # Exact tie: MLP over logistic, then the smaller alpha, then the image logistic.
        small_alpha = -float(alpha) if mlp and alpha is not None else 0.0
        image = 1 if kind == "species_logistic_mel" else 0
        return (score, mlp, small_alpha, image)

    kind, feature, clf, mu, sd, score, _x_raw, _v_raw = max(candidates, key=_preference)
    head = _as_head(clf, labels, mu, sd, seed, feature)
    report = {
        "winner": kind,
        "feature": feature,
        "alpha": None if getattr(clf, "alpha", None) is None else float(clf.alpha),
        "early_stopping": bool(getattr(clf, "uaere_early_stopping", False)),
        "val_macro_f1": float(score),
        **_fit_note(clf),
        "candidates": [
            {
                "kind": row[0],
                "feature": row[1],
                "val_macro_f1": float(row[5]),
                "alpha": None if getattr(row[2], "alpha", None) is None else float(row[2].alpha),
                "early_stopping": bool(getattr(row[2], "uaere_early_stopping", False)),
                "class_balance": "sample_weight" if row[0] == "species_mlp_mel" else "class_weight",
                **_fit_note(row[2]),
            }
            for row in candidates
        ],
        "labels": list(labels),
        # Popped before any JSON write. Test scores may be computed from these fits.
        # They are not allowed to change `winner`.
        "_models": [
            {
                "kind": row[0],
                "feature": row[1],
                "clf": row[2],
                "mu": row[3],
                "sd": row[4],
                "alpha": None if getattr(row[2], "alpha", None) is None else float(row[2].alpha),
            }
            for row in candidates
        ],
    }
    return head, report


def save_species_checkpoint(
    head: EvidentialHead,
    path: str | Path,
    meta: dict | None = None,
    selection: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if head.labels is None or head.sklearn is None:
        raise ValueError("species checkpoint requires a fitted head with labels")
    kind = (selection or {}).get("winner") or "species_logistic_mel"
    if kind not in KINDS:
        raise ValueError(f"unknown species checkpoint kind {kind}")
    blob = {
        "kind": kind,
        "feature": (selection or {}).get("feature") or ("summary" if "summary" in kind else "image"),
        "sklearn": head.sklearn,
        "feat_mu": np.asarray(head._feat_mu, dtype=np.float64),
        "feat_sd": np.asarray(head._feat_sd, dtype=np.float64),
        "labels": list(head.labels),
        "negative_index": None,
        "concentration": float(head.concentration),
        "sample_rate": int((meta or {}).get("working_fs") or 16_000),
        "window_s": 1.0,
        "selection": selection or {},
        "meta": meta or {},
    }
    joblib.dump(blob, path)
    return path


def load_species_checkpoint(path: str | Path) -> tuple[EvidentialHead, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No species checkpoint at {path}. Train one with "
            "`uaere train --dataset species --root <species folders>`. "
            "Predict will not train a ship model in its place."
        )
    blob = joblib.load(path)
    if blob.get("kind") not in KINDS:
        raise ValueError(f"{path} is not a species checkpoint")
    labels = tuple(blob["labels"])
    mu = np.asarray(blob["feat_mu"], dtype=np.float64)
    head = EvidentialHead(in_dim=int(mu.shape[0]), n_classes=len(labels), seed=0)
    head._feat_mu = mu
    head._feat_sd = np.asarray(blob["feat_sd"], dtype=np.float64)
    head.sklearn = blob["sklearn"]
    head.labels = labels
    head.negative_index = None
    head.feature = str(blob.get("feature") or ("summary" if "summary" in str(blob.get("kind")) else "image"))
    head.concentration = float(blob.get("concentration", 20.0))
    meta = dict(blob.get("meta") or {})
    meta["kind"] = blob["kind"]
    meta["feature"] = blob.get("feature")
    meta["selection"] = blob.get("selection") or {}
    meta["sample_rate"] = int(blob.get("sample_rate") or meta.get("working_fs") or 16_000)
    meta["working_fs"] = meta["sample_rate"]
    return head, meta


def species_features(
    head: EvidentialHead,
    waveform: FloatArray,
    sample_rate: int,
    feature: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised species vector plus a 32-band mel for the trust engine."""
    which = feature or getattr(head, "feature", None) or "image"
    raw = mel_summary(waveform, sample_rate) if which == "summary" else mel_image(waveform, sample_rate)
    mu = np.asarray(head._feat_mu, dtype=np.float64)
    sd = np.asarray(head._feat_sd, dtype=np.float64)
    xn = (raw - mu) / sd
    mel = log_mel(np.asarray(waveform, dtype=np.float64), int(sample_rate), n_mels=32)
    return xn, mel


def score_waveform(
    head: EvidentialHead,
    waveform: FloatArray,
    sample_rate: int,
    environment: EnvironmentState,
    health: SensorHealth,
    feature: str | None = None,
) -> TrustScore:
    xn, mel = species_features(head, waveform, sample_rate, feature=feature)
    return TrustEngine(head).score(xn, mel, health, environment)
