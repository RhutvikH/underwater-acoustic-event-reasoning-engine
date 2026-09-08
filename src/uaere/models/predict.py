"""Load a wav and emit class, C_wake, T(e), optional KG sentence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.train import _mel_stats, extract_features, train_evidential, transform
from uaere.data.wavutil import read_wav, resample_linear
from uaere.kg.graph import load_kg
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.representation.l1_tf import log_mel
from uaere.representation.predictive import surprise, surprise_admit
from uaere.trust.trust_score import TrustEngine
from uaere.twin.render import TwinRenderer
from uaere.types import (
    WORKING_FS,
    EnvironmentState,
    EventClass,
    FaultClass,
    SensorHealth,
    WindowRecord,
)


def predict_wav(
    wav_path: str | Path,
    head=None,
    engine: TrustEngine | None = None,
) -> dict:
    x, fs = read_wav(wav_path)
    x = resample_linear(x, fs, WORKING_FS)
    n = WORKING_FS
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    x = x[:n]
    env = EnvironmentState()
    rec = WindowRecord(
        waveform=x,
        sample_rate=WORKING_FS,
        environment=env,
        health_oracle=SensorHealth(),
        health_estimate=SensorHealth(fault=FaultClass.NONE),
        event_present=False,
        event_class=EventClass.REJECT,
        cause_id="",
        is_artifact=False,
        source_id=Path(wav_path).stem,
    )
    if head is None or engine is None:
        recs = TwinRenderer("busy_strait", seed=0).render_dataset(80)
        bun = extract_features(recs)
        head = train_evidential(bun, steps=120, seed=0)
        engine = TrustEngine(head)
    S = surprise(x, WORKING_FS, env)
    mel = log_mel(x, WORKING_FS)
    mel_n = AdaptiveNormalizer(32).normalize_mel(mel, env)
    feats = transform(head, _mel_stats(mel_n))
    tscore = engine.score(feats, mel_n, rec.health_estimate, env)
    expl = CausalReasoner(load_kg()).explain(tscore.predicted_class(), x, WORKING_FS, env)
    return {
        "file": str(wav_path),
        "surprise": S,
        "surprise_admit": surprise_admit(S),
        "wake_confidence": tscore.wake_confidence,
        "event_trust": tscore.event_trust,
        "class": tscore.predicted_class().value,
        "sentence": expl.sentence,
        "why": expl.why,
        "why_not": expl.why_not,
    }
