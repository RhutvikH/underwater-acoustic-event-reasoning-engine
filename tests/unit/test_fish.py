"""Fish head: real clips decide the label. The twin is not a fallback."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from uaere.cli import main
from uaere.data.adapters import FishCallAdapter, recording_group_id
from uaere.data.wavutil import read_wav, write_wav
from uaere.kg.graph import load_kg
from uaere.models.fish_head import fit_fish_head, load_fish_checkpoint, save_fish_checkpoint, score_waveform
from uaere.pipeline import AhaifPipeline
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.swarm.field import SwarmField
from uaere.trust.trust_score import TrustEngine
from uaere.trust.evidential_head import EvidentialHead
from uaere.types import (
    WORKING_FS,
    EnvironmentState,
    EventClass,
    PolicyVector,
    SensorHealth,
    WindowRecord,
)


def _tone(path: Path, freq: float, n: int = 16000, fs: int = 16000, noise: float = 0.0) -> None:
    t = np.arange(n) / fs
    rng = np.random.default_rng(int(freq) + n)
    x = 0.5 * np.sin(2 * np.pi * freq * t)
    if noise:
        x = x + rng.normal(0, noise, n)
    write_wav(path, x.astype(np.float64), fs)


def _tree(root: Path, n_each: int = 6) -> None:
    bw = root / "bwhistle"
    other = root / "other"
    bw.mkdir(parents=True)
    other.mkdir()
    for i in range(n_each):
        _tone(bw / f"boat_{i}.wav", 180.0, noise=0.01)
        _tone(other / f"noise_{i}.wav", 2200.0, noise=0.2)
    # Two cuts of one recording, and two files that only share a numeric tail.
    _tone(bw / "siteA_clip1.wav", 180.0)
    _tone(bw / "siteA_clip2.wav", 180.0)
    _tone(bw / "call_0001.wav", 190.0)
    _tone(bw / "call_0002.wav", 190.0)


def test_recording_group_keeps_clip_index_together_and_file_ids_apart():
    assert recording_group_id(Path("bwhistle/siteA_clip1.wav")) == recording_group_id(
        Path("bwhistle/siteA_clip2.wav")
    )
    assert recording_group_id(Path("bwhistle/call_0001.wav")) != recording_group_id(
        Path("bwhistle/call_0002.wav")
    )
    deployment = "BB_201351190.170422052002"
    assert (
        recording_group_id(Path(f"bwhistleclips/bw_{deployment}_000.3870833.wav"))
        == deployment
    )
    assert (
        recording_group_id(Path(f"otherclips/ot_{deployment}_004.1000000.wav"))
        == deployment
    )
    assert (
        recording_group_id(Path("bwhistleclips/t_RB_335831108.180529020002_011.6336667.wav"))
        == "RB_335831108.180529020002"
    )


def test_split_puts_one_recording_on_one_side(tmp_path: Path):
    root = tmp_path / "fish"
    _tree(root)
    rare = root / "croaker"
    rare.mkdir()
    _tone(rare / "one.wav", 400.0)
    train, val, test, meta = FishCallAdapter(
        root, seed=0, max_per_class=None, min_files=2
    ).load_split()
    assert "croaker" in meta["dropped_under_min_files"]
    assert meta["split_unit"] == "recording"
    sides = {"train": train, "val": val, "test": test}
    where = []
    for name, rows in sides.items():
        ids = {r.source_id for r in rows}
        if "bwhistle/siteA" in ids:
            where.append(name)
            assert sum(r.source_id == "bwhistle/siteA" for r in rows) == 2
    assert len(where) == 1
    all_ids = [r.source_id for r in train + val + test]
    assert "bwhistle/call_0001" in all_ids
    assert "bwhistle/call_0002" in all_ids
    # Both classes of one Pamlico deployment stay on the same side.
    dep = "BB_201351190.170422052002"
    (root / "bwhistleclips").mkdir()
    (root / "otherclips").mkdir()
    _tone(root / "bwhistleclips" / f"bw_{dep}_000.1000000.wav", 180.0)
    _tone(root / "otherclips" / f"ot_{dep}_001.2000000.wav", 2200.0, noise=0.2)
    for i in range(6):
        token = f"CB_10000000{i}.17010100000{i}"
        _tone(root / "bwhistleclips" / f"bw_{token}_000.1000000.wav", 180.0)
        _tone(root / "otherclips" / f"ot_{token}_000.2000000.wav", 2100.0, noise=0.2)
    train2, val2, test2, meta2 = FishCallAdapter(
        root, seed=1, max_per_class=None, min_files=2
    ).load_split()
    assert meta2["split_unit"] == "recording"
    sides2 = [train2, val2, test2]
    hits = [i for i, rows in enumerate(sides2) if any(r.source_id == dep for r in rows)]
    assert hits == [hits[0]]
    assert sum(r.source_id == dep for r in sides2[hits[0]]) == 2


def test_checkpoint_round_trip_and_pipeline_ignores_cnn(tmp_path: Path, monkeypatch):
    root = tmp_path / "fish"
    _tree(root, n_each=6)
    train, _val, test, meta = FishCallAdapter(
        root, seed=1, max_per_class=None, min_files=1
    ).load_split()
    head = fit_fish_head(train, steps=80, seed=1)
    path = save_fish_checkpoint(head, tmp_path / "fish_head.joblib", meta=meta)
    loaded, got = load_fish_checkpoint(path)
    assert got["split_unit"] == meta["split_unit"]
    assert loaded.labels == ("opsanus_tau", "other")
    rec = next(r for r in test if r.taxon == "opsanus_tau")
    direct = score_waveform(loaded, rec.waveform, rec.sample_rate, rec.environment, rec.health_estimate)
    pipe = AhaifPipeline(
        trust=TrustEngine(loaded),
        kg=load_kg(),
        policy=PolicyVector(0.05, 0.1, 0.2, device_id="stm32l476"),
        normalizer=AdaptiveNormalizer(32),
    )

    def boom(*_a, **_k):
        raise AssertionError("TinyCNN must not name the event")

    pipe.cnn.logits = boom  # type: ignore[method-assign]
    import uaere.pipeline as pipeline_mod

    # The twin cutoff would sleep. A species head still reports its own class.
    monkeypatch.setattr(pipeline_mod, "surprise_admit", lambda _s: 0.0)
    result = pipe.infer(rec)
    assert result.reason != "l0_low_surprise"
    assert result.extras["l0_veto"] is False
    assert result.event_class == direct.predicted_class()
    assert result.extras["taxon"] == direct.taxon
    assert result.event_class in {EventClass.OPSANUS_TAU, EventClass.OTHER}


def test_boatwhistle_band_is_an_independent_check():
    from uaere.causal.reasoner import CausalReasoner

    reasoner = CausalReasoner(load_kg())
    t = np.arange(16000) / 16000
    tone = 0.5 * np.sin(2 * np.pi * 180.0 * t)
    low = reasoner.explain(
        EventClass.OPSANUS_TAU, tone, 16000, EnvironmentState(), taxon="opsanus_tau"
    )
    assert low.cause_id == "cause.taxon.opsanus_tau"
    assert "cues match" in low.why
    assert "boatwhistle band leads" in low.why
    high = 0.5 * np.sin(2 * np.pi * 3000.0 * t)
    hi = reasoner.explain(
        EventClass.OPSANUS_TAU, high, 16000, EnvironmentState(), taxon="opsanus_tau"
    )
    assert hi.cause_id == "cause.taxon.opsanus_tau"
    assert "cues match" in hi.why
    assert "does not lead" in hi.why


def test_ship_head_still_sleeps_on_low_surprise(monkeypatch):
    head = EvidentialHead(in_dim=8, seed=0)
    assert head.labels is None
    pipe = AhaifPipeline(
        trust=TrustEngine(head),
        kg=load_kg(),
        policy=PolicyVector(0.15, 0.35, 0.6, device_id="stm32l476"),
        normalizer=AdaptiveNormalizer(32),
    )
    import uaere.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "surprise_admit", lambda _s: 0.0)
    health = SensorHealth()
    rec = WindowRecord(
        waveform=np.zeros(16000),
        sample_rate=16000,
        environment=EnvironmentState(),
        health_oracle=health,
        health_estimate=health,
        event_present=True,
        event_class=EventClass.TUG,
        cause_id="",
        is_artifact=False,
    )
    result = pipe.infer(rec)
    assert result.reason == "l0_low_surprise"
    assert result.event_class is EventClass.REJECT
    assert result.extras["l0_veto"] is True


def test_swarm_shows_the_classifier_not_the_tape(tmp_path: Path):
    root = tmp_path / "fish"
    _tree(root, n_each=4)
    train, _val, _test, meta = FishCallAdapter(
        root, seed=0, max_per_class=8, min_files=1
    ).load_split()
    head = fit_fish_head(train, steps=40, seed=0)
    ckpt = save_fish_checkpoint(head, tmp_path / "fish_head.joblib", meta=meta)
    field = SwarmField(
        n_nodes=3,
        model="fish",
        replay_root=str(root),
        checkpoint=str(ckpt),
        seed=0,
    )
    field.warmup()
    ops = next(r for r in field._replay if r.taxon == "opsanus_tau")
    field._replay = [ops]
    field._replay_i = 0

    def always_other(_feats, _mel, _health, _state):
        from uaere.types import TrustScore

        p = np.array([0.05, 0.95])
        return TrustScore(
            wake_confidence=0.92,
            event_trust=0.88,
            p=p,
            alpha=np.array([1.2, 30.0]),
            u_aleatoric=0.05,
            u_epistemic=0.05,
            health_consistency=1.0,
            env_consistency=0.8,
            label=EventClass.OTHER,
            taxon="other",
        )

    assert field.engine is not None
    field.engine.score = always_other  # type: ignore[method-assign]
    tick = field.step()
    assert tick.source["class"] == "opsanus_tau"
    assert tick.source["replay"] is True
    assert {n.event_class for n in tick.nodes} == {"other"}


def test_float_upa_is_not_full_scale_normalised(tmp_path: Path):
    path = tmp_path / "upa.wav"
    wavfile.write(path, 24_000, np.array([12_345.0, -8_000.0, 100.0], dtype=np.float32))
    x, fs = read_wav(path)
    assert fs == 24_000
    assert abs(float(x[0]) - 12_345.0) < 1.0
    ipath = tmp_path / "pcm32.wav"
    wavfile.write(ipath, 16_000, np.array([2**30, -(2**30)], dtype=np.int32))
    xi, fsi = read_wav(ipath)
    assert fsi == 16_000
    assert abs(float(xi[0]) - 0.5) < 1e-5


def test_predict_refuses_missing_checkpoint(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _tone(tmp_path / "x.wav", 200.0)
    code = main(["predict", "--wav", "x.wav", "--checkpoint", str(tmp_path / "missing.joblib")])
    err = capsys.readouterr().err
    assert code == 2
    assert "will not train" in err.lower()
    assert not (tmp_path / "artifacts").exists()


def test_fish_eval_missing_root_exits_2(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["eval", "--suite", "fish", "--root", str(tmp_path / "empty")])
    assert code == 2
    assert "twin" in capsys.readouterr().err.lower()
    assert list(tmp_path.glob("**/*")) == []


def test_demo_audio_endpoint_returns_wav():
    from uaere.demo.server import serve_demo

    srv = serve_demo(n_nodes=2, scenario="pam_field", model="oracle", port=18765, seed=1)
    try:
        for _ in range(40):
            if srv.state.field._audio.get("n0") is not None:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("demo never stored a window")
        with urllib.request.urlopen("http://127.0.0.1:18765/api/audio?node=n0", timeout=5) as resp:
            body = resp.read(12)
            ctype = resp.headers["Content-Type"]
        assert ctype == "audio/wav"
        assert body[:4] == b"RIFF"
    finally:
        srv.stop()


def test_reasoner_names_toadfish_not_a_species_list():
    kg = load_kg()
    assert kg.causes("event.toadfish_boatwhistle")
    assert any(cid == "cause.taxon.opsanus_tau" for cid, _ in kg.causes("event.toadfish_boatwhistle"))
    # A one-line score so the import of EnvironmentState stays honest if a future edit needs it.
    assert EnvironmentState().sea_state == 2
    assert SensorHealth().fault.value == "none"
