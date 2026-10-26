"""Species head: recording-locked folders, no twin fallback, wake is max(p)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.cli import main
from uaere.data.species import SpeciesDataError, SpeciesFolderAdapter, watkins_tape_id
from uaere.data.split import FileGroup, label_stratified_recording_split
from uaere.data.watkins_unpack import species_from_member
from uaere.data.wavutil import read_wav, read_wav_centre, write_wav
from uaere.demo.server import resolve_model
from uaere.kg.graph import load_kg
from uaere.models.species_head import (
    KINDS,
    load_species_checkpoint,
    save_species_checkpoint,
    select_species_head,
)
from uaere.pipeline import AhaifPipeline
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.trust.evidential_head import EvidentialHead
from uaere.trust.trust_score import TrustEngine
from uaere.types import EnvironmentState, EventClass, PolicyVector, SensorHealth, TrustScore


class _Proba:
    def __init__(self, row: list[float]) -> None:
        self._p = np.asarray(row, dtype=np.float64)
        self.classes_ = np.arange(len(row))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile(self._p, (len(np.atleast_2d(x)), 1))


def _tone(path: Path, freq: float) -> None:
    fs = 16_000
    t = np.arange(1600) / fs
    write_wav(path, (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float64), fs)


def _species_tree(root: Path, n_tapes: int = 6) -> None:
    for label, freq, base in (("humpback_whale", 180.0, 10_000), ("sperm_whale", 900.0, 20_000)):
        folder = root / label
        folder.mkdir(parents=True)
        for i in range(n_tapes):
            tape = f"{base + i:05d}01"
            _tone(folder / f"sample_{label}_{1960 + i}_{tape}.wav", freq)


def test_centre_read_matches_the_middle_of_a_long_pcm_file(tmp_path: Path):
    fs = 8000
    n = fs * 5
    t = np.arange(n) / fs
    x = (0.25 * np.sin(2 * np.pi * 80.0 * t)).astype(np.float64)
    path = tmp_path / "long.wav"
    write_wav(path, x, fs)
    full, fs_full = read_wav(path)
    centre, fs_c = read_wav_centre(path, seconds=1.0)
    assert fs_full == fs and fs_c == fs
    start = (len(full) - fs) // 2
    assert len(centre) == fs
    assert np.allclose(centre, full[start : start + fs])


def test_tape_token_binds_ab_cuts_and_catalog_cuts():
    assert watkins_tape_id(Path("sample_AtlanticSpottedDolphin_1961_6102500A.wav")) == "61025"
    assert watkins_tape_id(Path("sample_AtlanticSpottedDolphin_1961_6102500B.wav")) == "61025"
    assert watkins_tape_id(Path("cut_83093034.wav")) == "83093"
    assert species_from_member("sample_AtlanticSpottedDolphin_1961_6102500A.wav") == "atlantic_spotted_dolphin"
    assert species_from_member("Humpback Whale/anything.wav") == "humpback_whale"
    assert (
        species_from_member(
            "Watkins_Marine_Mammal_Sound_Database/LeopardSeal/1966/66084014.wav"
        )
        == "leopard_seal"
    )
    assert (
        species_from_member(
            "Watkins_Marine_Mammal_Sound_Database/Short_Finned(Pacific)PilotWhale/1980/8000101.wav"
        )
        == "short_finned_pacific_pilot_whale"
    )


def test_one_tape_cannot_land_on_two_sides():
    groups = [FileGroup("61025", "a", (Path("a_6102500A.wav"), Path("a_6102500B.wav")))]
    for i in range(8):
        groups.append(FileGroup(f"{70000 + i}", "a", (Path(f"a_{i}.wav"),)))
        groups.append(FileGroup(f"{80000 + i}", "b", (Path(f"b_{i}.wav"),)))
    train, val, test = label_stratified_recording_split(groups, seed=0, max_per_class=None)
    hits = []
    for name, rows in (("train", train), ("val", val), ("test", test)):
        if any(g.group_id == "61025" for g in rows):
            hits.append(name)
    assert hits == [hits[0]]
    kept = [g for g in {"train": train, "val": val, "test": test}[hits[0]] if g.group_id == "61025"]
    assert [p.name for p in kept[0].paths] == ["a_6102500A.wav", "a_6102500B.wav"]


def test_adapter_keeps_61025_together(tmp_path: Path):
    root = tmp_path / "species"
    folder = root / "atlantic_spotted_dolphin"
    other = root / "bowhead_whale"
    folder.mkdir(parents=True)
    other.mkdir()
    _tone(folder / "sample_AtlanticSpottedDolphin_1961_6102500A.wav", 400.0)
    _tone(folder / "sample_AtlanticSpottedDolphin_1961_6102500B.wav", 420.0)
    for i in range(6):
        _tone(folder / f"sample_AtlanticSpottedDolphin_1970_{71000 + i:05d}11.wav", 400.0)
        _tone(other / f"sample_BowheadWhale_1970_{81000 + i:05d}11.wav", 120.0)
    train, val, test, meta = SpeciesFolderAdapter(
        root,
        seed=0,
        max_per_class=None,
        min_files=2,
        min_groups=2,
        min_classes=2,
        require_recording_ids=True,
    ).load_split()
    assert meta["split_unit"] == "recording"
    hits = [i for i, rows in enumerate((train, val, test)) if any(r.source_id == "61025" for r in rows)]
    assert hits == [hits[0]]
    assert sum(r.source_id == "61025" for r in (train, val, test)[hits[0]]) == 2


def test_nine_class_tree_is_refused(tmp_path: Path):
    root = tmp_path / "species"
    for i in range(9):
        folder = root / f"species_{i:02d}"
        folder.mkdir(parents=True)
        _tone(folder / f"sample_S_1961_{90000 + i:05d}01.wav", 200.0 + 10 * i)
    (root / "harbor_porpoise").mkdir()
    _tone(root / "harbor_porpoise" / "sample_P_1961_1000101.wav", 2000.0)
    adapter = SpeciesFolderAdapter(
        root, max_per_class=None, min_files=1, min_groups=1, min_classes=10
    )
    try:
        adapter.discover()
    except SpeciesDataError as exc:
        text = str(exc)
    else:
        raise AssertionError("9 species cleared a 10-class floor")
    assert "Need 10" in text
    assert "Floors are not lowered" in text


def test_checkpoint_round_trip_and_pipeline_uses_the_head(tmp_path: Path, monkeypatch):
    root = tmp_path / "species"
    _species_tree(root)
    train, _val, test, meta = SpeciesFolderAdapter(
        root,
        seed=0,
        max_per_class=None,
        min_files=4,
        min_groups=4,
        min_classes=2,
    ).load_split()
    head, selection = select_species_head(train, _val, steps=40, seed=0)
    assert selection["winner"] in KINDS
    assert head.negative_index is None
    assert head.labels == ("humpback_whale", "sperm_whale")
    path = save_species_checkpoint(
        head,
        tmp_path / "species_head.joblib",
        meta=meta,
        selection={k: v for k, v in selection.items() if k != "_models"},
    )
    loaded, got = load_species_checkpoint(path)
    assert got["kind"] in KINDS
    assert loaded.labels == head.labels
    assert loaded.feature in {"image", "summary"}
    rec = next(r for r in test if r.taxon == "humpback_whale")
    pipe = AhaifPipeline(
        trust=TrustEngine(loaded),
        kg=load_kg(),
        policy=PolicyVector(0.05, 0.1, 0.2, device_id="stm32l476"),
        normalizer=AdaptiveNormalizer(32),
    )

    def boom(*_a, **_k):
        raise AssertionError("TinyCNN must not name the event")

    pipe.cnn.logits = boom  # type: ignore[method-assign]

    def disagree(_feats, _mel, _health, _state):
        return TrustScore(
            wake_confidence=0.91,
            event_trust=0.88,
            p=np.array([0.08, 0.92]),
            alpha=np.array([2.0, 20.0]),
            u_aleatoric=0.05,
            u_epistemic=0.05,
            health_consistency=1.0,
            env_consistency=0.5,
            label=EventClass.FISH,
            taxon="sperm_whale",
        )

    pipe.trust.score = disagree  # type: ignore[method-assign]
    import uaere.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "surprise_admit", lambda _s: 0.0)
    result = pipe.infer(rec)
    assert result.extras["l0_veto"] is False
    assert result.reason != "l0_low_surprise"
    assert result.extras["taxon"] == "sperm_whale"
    assert result.extras["taxon"] != rec.taxon
    assert result.event_class is EventClass.FISH
    assert result.explanation is not None
    assert result.explanation.cause_id == "cause.taxon.sperm_whale"
    assert "cues match" not in result.explanation.sentence.lower()
    assert "No frequency band is stored" in result.explanation.sentence


def test_species_wake_is_max_prob_and_ship_and_toadfish_stay():
    health = SensorHealth()
    state = EnvironmentState()
    species = EvidentialHead(in_dim=4, n_classes=3, seed=0)
    species.labels = ("a", "b", "c")
    species.negative_index = None
    species.sklearn = _Proba([0.1, 0.2, 0.7])
    got = TrustEngine(species).score(np.zeros(4), np.zeros((8, 2)), health, state)
    # Wake is the maximum Dirichlet class probability, not class index 5.
    assert abs(got.wake_confidence - float(np.max(got.p))) < 1e-9
    assert abs(got.wake_confidence - (1.0 + 20.0 * 0.7) / (3.0 + 20.0)) < 1e-9
    assert got.taxon == "c"

    fish = EvidentialHead(in_dim=4, n_classes=2, seed=0)
    fish.labels = ("opsanus_tau", "other")
    fish.negative_index = 1
    fish.sklearn = _Proba([0.25, 0.75])
    fish_score = TrustEngine(fish).score(np.zeros(4), np.zeros((8, 2)), health, state)
    assert abs(fish_score.wake_confidence - (1.0 - fish_score.p[1])) < 1e-9
    assert abs(fish_score.wake_confidence - (1.0 + 20.0 * 0.25) / (2.0 + 20.0)) < 1e-9

    ship = EvidentialHead(in_dim=4, n_classes=6, seed=0)
    assert ship.labels is None
    ship.sklearn = _Proba([0.1, 0.1, 0.1, 0.1, 0.2, 0.4])
    ship_score = TrustEngine(ship).score(np.zeros(4), np.zeros((8, 2)), health, state)
    assert abs(ship_score.wake_confidence - (1.0 - ship_score.p[5])) < 1e-9


def test_name_only_cause_does_not_invent_a_band():
    kg = load_kg()
    before = kg.n_nodes
    kg.ensure_taxon("atlantic_spotted_dolphin")
    assert kg.n_nodes == before + 2
    expl = CausalReasoner(kg).explain(
        EventClass.FISH,
        np.zeros(1600),
        16_000,
        EnvironmentState(),
        taxon="atlantic_spotted_dolphin",
    )
    assert expl.cause_id == "cause.taxon.atlantic_spotted_dolphin"
    assert "cues match" not in expl.why.lower()
    assert "No frequency band is stored" in expl.why
    fresh = load_kg()
    toad = CausalReasoner(fresh).explain(
        EventClass.OPSANUS_TAU,
        np.zeros(1600),
        16_000,
        EnvironmentState(),
        taxon="opsanus_tau",
    )
    assert toad.cause_id == "cause.taxon.opsanus_tau"
    assert "cues match" in toad.why


def test_missing_species_root_exits_2(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code = main(["train", "--dataset", "species", "--root", str(tmp_path / "missing")])
    err = capsys.readouterr().err.lower()
    assert code == 2
    assert "twin" in err
    code = main(["eval", "--suite", "species", "--root", str(tmp_path / "missing")])
    assert code == 2
    assert list(tmp_path.glob("artifacts/**/*")) == []


def test_auto_prefers_species_checkpoint(tmp_path: Path, monkeypatch):
    species = tmp_path / "species_head.joblib"
    fish = tmp_path / "fish_head.joblib"
    joblib.dump({"kind": "species_mlp_mel"}, species)
    joblib.dump({"kind": "fish_logistic_mel"}, fish)
    monkeypatch.setattr("uaere.models.species_head.DEFAULT_SPECIES_CHECKPOINT", species)
    monkeypatch.setattr("uaere.demo.server.DEFAULT_FISH_CHECKPOINT", fish)
    assert resolve_model("auto", None) == "species"
    species.unlink()
    assert resolve_model("auto", None) == "fish"
