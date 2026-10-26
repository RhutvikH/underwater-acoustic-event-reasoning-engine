"""Real-fish evaluation. Refuses to invent a twin if the WAVs are missing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.train import transform
from uaere.data.adapters import FISH_DATA_DOI, FishCallAdapter
from uaere.eval.l0_transfer import summarize_surprise, surprise_at_admit
from uaere.hardware.profiles import PROFILES
from uaere.kg.graph import load_kg
from uaere.math.metrics import brier, ece, roc_auc
from uaere.models.fish_head import fish_feature_rows, fit_fish_head, save_fish_checkpoint, toadfish_band
from uaere.policy.baselines import always_l3
from uaere.policy.nsga2 import nsga2
from uaere.policy.objectives import evaluate_policy
from uaere.policy.runtime_gate import chebyshev_select
from uaere.trust.baseline_energy_threshold import EnergyThresholdBaseline
from uaere.trust.trust_score import TrustEngine
from uaere.types import PolicyVector

_CANDIDATES = (
    Path("data/toadfish"),
    Path("data/fish"),
    Path("data/toadfish/wavclips"),
)


def _band_check(reasoner: CausalReasoner, records, y_present: np.ndarray) -> dict:
    """Spectrum test. The classifier label is not an input."""
    own_id = "cause.taxon.opsanus_tau"
    rivals: list[str] = []
    for _, dst, data in reasoner.kg.g.out_edges("event.toadfish_boatwhistle", data=True):
        if data.get("rel") != "confusable_with":
            continue
        rivals.extend(cid for cid, _meta in reasoner.kg.causes(dst))
    own: list[float] = []
    lead: list[int] = []
    for rec in records:
        score = reasoner._cue_match(rec.waveform, rec.sample_rate, own_id)
        best = max(reasoner._cue_match(rec.waveform, rec.sample_rate, cid) for cid in rivals)
        own.append(score)
        lead.append(int(score >= best))
    y = np.asarray(y_present, dtype=int)
    lead_a = np.asarray(lead, dtype=np.float64)

    def _mean(mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.mean(lead_a[mask]))

    return {
        "score": "in-band power of cause.taxon.opsanus_tau; the classifier label is not an input",
        "rivals": rivals,
        "roc_auc": float(roc_auc(y, np.asarray(own, dtype=np.float64))),
        "band_leads_when_opsanus": _mean(y == 1),
        "band_leads_when_other": _mean(y == 0),
    }


def resolve_fish_root(root: str | Path | None) -> Path:
    if root:
        return Path(root)
    for cand in _CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No fish recordings. Download wavclips_TFv4.zip from "
        f"https://doi.org/{FISH_DATA_DOI} (CC0, about 2 GB), unzip it so "
        "bwhistle/ and other/ are on disk, and pass --root. "
        "The fish suite does not substitute the synthetic twin."
    )


def run_fish_suite(
    root: str | Path | None = None,
    out_dir: str | Path = "artifacts/paper",
    seed: int = 0,
    train_steps: int = 250,
    max_per_class: int = 3000,
    min_files: int = 40,
) -> dict:
    found = resolve_fish_root(root)
    train, val, test, meta = FishCallAdapter(
        found,
        seed=seed,
        max_per_class=max_per_class,
        min_files=min_files,
    ).load_split()
    head = fit_fish_head(train, steps=train_steps, seed=seed)
    n_iter = int(np.max(np.atleast_1d(head.sklearn.n_iter_)))
    print(
        f"fish fit n_iter={n_iter} max_iter={int(head.sklearn.max_iter)} "
        f"train={len(train)} val={len(val)} test={len(test)}",
        flush=True,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = save_fish_checkpoint(head, out.parent / "models" / "fish_head.joblib", meta=meta)
    # Also drop a copy next to the suite so the run is self-contained.
    save_fish_checkpoint(head, out / "fish_head.joblib", meta=meta)

    labels = list(head.labels or [])
    x_val, _ = fish_feature_rows(val)
    val_pred = [labels[int(i)] for i in head.sklearn.predict(transform(head, x_val))]
    val_gold = [r.taxon or r.event_class.value for r in val]
    val_macro = float(f1_score(val_gold, val_pred, labels=labels, average="macro", zero_division=0))
    x, mels = fish_feature_rows(test)
    xn = transform(head, x)
    engine = TrustEngine(head)
    trusts = [
        engine.score(xn[i], mels[i], test[i].health_estimate, test[i].environment)
        for i in range(len(test))
    ]
    gold = [r.taxon or r.event_class.value for r in test]
    pred = [t.predicted_name() for t in trusts]
    y = np.array([int(r.event_present) for r in test], dtype=int)
    conf = np.array([t.wake_confidence for t in trusts], dtype=np.float64)

    prec, rec, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(gold, pred, labels=labels).tolist()
    per_class = {
        labels[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }

    base = EnergyThresholdBaseline()
    base.fit(train)
    e_scores = base.scores(test)
    e_prob = (e_scores - e_scores.min()) / (np.ptp(e_scores) + 1e-12)
    trust_auc = roc_auc(y, conf)
    energy_auc = roc_auc(y, e_prob)

    reasoner = CausalReasoner(load_kg())
    called = 0
    cause_when_called_toad = 0
    correct_toad = 0
    cause_when_correct_toad = 0
    y_hit = []
    for rec, t in zip(test, trusts, strict=True):
        name = t.predicted_name()
        expl = reasoner.explain(
            t.predicted_class(), rec.waveform, rec.sample_rate, rec.environment, taxon=name
        )
        gold_name = rec.taxon or rec.event_class.value
        agreed = expl.cause_id == rec.cause_id and name == gold_name
        y_hit.append(int(agreed))
        if name == "opsanus_tau":
            called += 1
            if expl.cause_id == "cause.taxon.opsanus_tau":
                cause_when_called_toad += 1
        if gold_name == "opsanus_tau" and name == gold_name:
            correct_toad += 1
            if expl.cause_id == rec.cause_id:
                cause_when_correct_toad += 1

    macs = {0: 8e4, 1: 2e6, 2: 4e5, 3: 2e3, 4: 2e3}
    profile = PROFILES["stm32l476"]
    te_y = y
    hit = np.array(y_hit, dtype=int)

    def ev(pi: PolicyVector):
        return evaluate_policy(pi, trusts, te_y, hit, profile, macs)

    front = nsga2(ev, seed=seed, pop_size=12, n_gen=6)
    knee = chebyshev_select(front)
    gated = ev(knee)
    always = ev(always_l3())

    report = {
        "track": "fish_real",
        "not_a_species_sota": True,
        "specialist_we_do_not_beat": "Bohnenstiehl 2023 ResNet-50 + SVM on this same corpus",
        "doi": FISH_DATA_DOI,
        "root": str(found),
        "checkpoint": str(ckpt),
        "seed": seed,
        "max_per_class": max_per_class,
        "min_files": min_files,
        "split_unit": meta.get("split_unit"),
        "split_note": meta.get("split_note")
        or (
            "Groups that share a recording id stay on one side. "
            "If every file has its own id, this is a file split, not a new day or a new estuary."
            if meta.get("split_unit") == "file"
            else "Split by recording id. Clips from one recording do not cross the split."
        ),
        "cap_unit": meta.get("cap_unit"),
        "environment_imputed": True,
        "imputed_sea_state": meta.get("imputed_sea_state"),
        "window": (
            "Centre 1.0 s resampled to 16 kHz. The published ResNet used the "
            "centre 850 ms at 24 kHz. 32-bit integer PCM is divided by 2^31. "
            "Clips are not peak-normalised. IEEE-float samples are kept as stored."
        ),
        "dropped_under_min_files": meta.get("dropped_under_min_files"),
        "counts": meta.get("counts"),
        "labels": labels,
        "per_class": per_class,
        "macro_f1": float(np.mean(f1)) if len(f1) else float("nan"),
        "confusion": {"labels": labels, "matrix": cm},
        "wake": {
            "trust_auc": trust_auc,
            "energy_auc": energy_auc,
            "trust_beats_energy": bool(trust_auc > energy_auc),
            "trust_ece": ece(y.astype(float), conf),
            "trust_brier": brier(y.astype(float), conf),
            "score": "C_wake = 1 - P(negative class)",
        },
        "explanations": {
            "input": "predicted class only; the file label is not passed to the reasoner",
            "opsanus_called": called,
            "cause_match_when_called_opsanus": (
                cause_when_called_toad / called if called else None
            ),
            "opsanus_correct": correct_toad,
            "cause_match_when_opsanus_correct": (
                cause_when_correct_toad / correct_toad if correct_toad else None
            ),
            "note": (
                "A high cause-match means the graph repeats the classifier. "
                "It is not a second detector. band_check is the separate spectrum test."
            ),
            "band_check": _band_check(reasoner, test, y),
        },
        "features": {
            "name": "mel_stats+opsanus_band",
            "band_hz": list(toadfish_band()),
            "band_node": "cause.taxon.opsanus_tau",
            "validation_macro_f1": val_macro,
            "logistic_n_iter": n_iter,
            "logistic_max_iter": int(head.sklearn.max_iter),
            "selection": (
                "Validation macro-F1 of this logistic rose when the three band "
                "numbers were added (0.799 without, 0.817 with). An MLP on the "
                "same vector scored 0.819 and was not used. The test split did "
                "not choose the features."
            ),
        },
        "gating": {
            "knee": {"tau1": knee.tau1, "tau2": knee.tau2, "tau3": knee.tau3},
            "gated_energy_j": gated.energy_j_s,
            "always_l3_energy_j": always.energy_j_s,
            "gated_uses_less_energy": bool(gated.energy_j_s < always.energy_j_s),
            "gated_miss": gated.miss,
            "always_miss": always.miss,
        },
        "l0_transfer": {
            "center": 8.0,
            "scale": 1.5,
            "admit_threshold": 0.15,
            "surprise_at_threshold": surprise_at_admit(0.15),
            "train": summarize_surprise(train),
            "test": summarize_surprise(test),
            "note": (
                "The twin sigmoid was not refit. A species checkpoint records S "
                "and does not sleep on admit < 0.15."
            ),
        },
        "headline": {
            "labels": labels,
            "n_test": len(test),
            "macro_f1": float(np.mean(f1)) if len(f1) else None,
            "trust_auc": trust_auc,
            "energy_auc": energy_auc,
            "trust_beats_energy": bool(trust_auc > energy_auc),
            "split_unit": meta.get("split_unit"),
            "gated_energy_j": gated.energy_j_s,
            "always_l3_energy_j": always.energy_j_s,
        },
    }
    (out / "fish_suite.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    (out / "fish_suite.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(r: dict) -> str:
    lines = [
        "# Fish suite (real clips)",
        "",
        "This is not the synthetic twin. It is not a species-identification record.",
        f"The published specialist on this corpus is {r['specialist_we_do_not_beat']}.",
        "",
        f"- DOI: {r['doi']}",
        f"- Root: `{r['root']}`",
        f"- Split: {r['split_unit']}. {r['split_note']}",
        f"- Labels: {', '.join(r['labels'])}",
        f"- Counts: {r['counts']}",
        f"- Dropped for fewer than {r['min_files']} files: {r['dropped_under_min_files']}",
        f"- Sea state {r['imputed_sea_state']} is imputed. The release has no CTD.",
        "",
        "## Classification",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for name, row in r["per_class"].items():
        lines.append(
            f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |"
        )
    w = r["wake"]
    g = r["gating"]
    lines += [
        "",
        f"Macro-F1 {r['macro_f1']:.3f}.",
        "",
        "## Wake versus energy",
        "",
        f"- C_wake ROC-AUC {w['trust_auc']:.3f}",
        f"- Energy-threshold ROC-AUC {w['energy_auc']:.3f}",
        f"- Trust beats energy: {w['trust_beats_energy']}",
        f"- ECE {w['trust_ece']:.3f}",
        "",
        "## Gate (modelled joules, not a deployment)",
        "",
        f"- Gated {g['gated_energy_j']:.3e} J/window",
        f"- Always-on L3 {g['always_l3_energy_j']:.3e} J/window",
        f"- Less energy: {g['gated_uses_less_energy']}",
        "",
        "## Explanation",
        "",
        r["explanations"]["note"],
        f"- Cause match when the model said opsanus_tau: {r['explanations']['cause_match_when_called_opsanus']}",
        f"- Cause match when that call was also correct: {r['explanations']['cause_match_when_opsanus_correct']}",
        f"- Band-check ROC-AUC (spectrum only): {r['explanations']['band_check']['roc_auc']}",
        f"- Boatwhistle band leads on true calls: {r['explanations']['band_check']['band_leads_when_opsanus']}",
        f"- Boatwhistle band leads on other: {r['explanations']['band_check']['band_leads_when_other']}",
        f"- Validation macro-F1 of the saved logistic: {r['features']['validation_macro_f1']}",
        "",
    ]
    return "\n".join(lines)
