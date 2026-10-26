"""Score one WAV with a saved checkpoint. Never retrains on the twin."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.data.wavutil import read_wav, resample_linear
from uaere.kg.graph import load_kg
from uaere.models.fish_head import DEFAULT_FISH_CHECKPOINT, load_fish_checkpoint
from uaere.models.fish_head import score_waveform as score_fish
from uaere.models.species_head import DEFAULT_SPECIES_CHECKPOINT, KINDS, load_species_checkpoint
from uaere.models.species_head import score_waveform as score_species
from uaere.representation.predictive import surprise, surprise_admit
from uaere.types import WORKING_FS, WINDOW_SECONDS, EnvironmentState, SensorHealth


def default_checkpoint() -> Path:
    """Species head when it has been trained, otherwise the toadfish head."""
    if DEFAULT_SPECIES_CHECKPOINT.exists():
        return DEFAULT_SPECIES_CHECKPOINT
    return Path(DEFAULT_FISH_CHECKPOINT)


def prepare_window(wav_path: str | Path, sample_rate: int = WORKING_FS) -> tuple[np.ndarray, EnvironmentState]:
    """Centre 1 s at the checkpoint rate. No peak normalisation."""
    x, fs = read_wav(wav_path)
    rate = int(sample_rate)
    x = resample_linear(x, fs, rate)
    n = int(rate * WINDOW_SECONDS)
    if len(x) >= n:
        start = (len(x) - n) // 2
        x = np.asarray(x[start : start + n], dtype=np.float64)
    else:
        x = np.pad(np.asarray(x, dtype=np.float64), (0, n - len(x)))
    return x, EnvironmentState()


def predict_wav(
    wav_path: str | Path,
    checkpoint: str | Path | None = None,
) -> dict:
    path = Path(checkpoint) if checkpoint else default_checkpoint()
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one with `uaere train --dataset species` "
            "or `uaere train --dataset fish`. Predict will not train a ship model in its place."
        )
    kind = joblib.load(path).get("kind")
    if kind in KINDS:
        head, meta = load_species_checkpoint(path)
        rate = int(meta.get("sample_rate") or WORKING_FS)
        x, env = prepare_window(wav_path, rate)
        tscore = score_species(head, x, rate, env, SensorHealth(), feature=meta.get("feature"))
        kg = load_kg()
        if tscore.taxon:
            kg.ensure_taxon(tscore.taxon)
    elif kind == "fish_logistic_mel":
        head, meta = load_fish_checkpoint(path)
        rate = WORKING_FS
        x, env = prepare_window(wav_path, rate)
        tscore = score_fish(head, x, rate, env, SensorHealth())
        kg = load_kg()
    else:
        raise ValueError(f"{path} is not a species or fish checkpoint")
    expl = CausalReasoner(kg).explain(
        tscore.predicted_class(), x, rate, env, taxon=tscore.taxon
    )
    S = surprise(x, rate, env)
    return {
        "file": str(wav_path),
        "listen": str(wav_path),
        "checkpoint": str(path),
        "labels": list(head.labels or ()),
        "sample_rate": rate,
        "surprise": S,
        "surprise_admit": surprise_admit(S),
        "l0_veto": False,
        "wake_confidence": tscore.wake_confidence,
        "event_trust": tscore.event_trust,
        "class": tscore.predicted_name(),
        "sentence": expl.sentence,
        "why": expl.why,
        "why_not": expl.why_not,
        "cause_id": expl.cause_id,
        "environment_imputed": True,
        "train_meta": {k: v for k, v in meta.items() if k != "_models"},
    }
