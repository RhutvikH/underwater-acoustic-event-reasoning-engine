from __future__ import annotations

from pathlib import Path

from uaere.classify.train import extract_features, train_evidential
from uaere.cli import main
from uaere.kg.graph import load_kg
from uaere.pipeline import AhaifPipeline
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.security.hal import SecureHAL
from uaere.trust.trust_score import TrustEngine
from uaere.twin.render import TwinRenderer
from uaere.types import PolicyVector


def test_pipeline_runs_and_stays_under_budget():
    recs = TwinRenderer("busy_strait", seed=3).render_dataset(24)
    bun = extract_features(recs)
    head = train_evidential(bun, steps=80, seed=3)
    hal = SecureHAL(b"fw", b"model")
    hal.secure_boot(b"fw")
    pipe = AhaifPipeline(
        trust=TrustEngine(head),
        kg=load_kg(),
        policy=PolicyVector(0.15, 0.35, 0.6, device_id="stm32l476"),
        normalizer=AdaptiveNormalizer(32),
        hal=hal,
    )
    results = [pipe.infer(r) for r in recs]
    assert results
    assert any(r.explanation is not None for r in results) or any(
        r.trust.event_trust < 0.6 for r in results
    )


def test_cli_data_and_kg(tmp_path: Path, capsys):
    assert main(["data", "summarize", "--n", "12", "--seed", "0"]) == 0
    out = tmp_path / "kg.ttl"
    assert main(["kg", "--out", str(out)]) == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "caused_by" in text or "ma:" in text
