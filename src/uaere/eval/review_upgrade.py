"""Validation-chosen fish gates. Does not overwrite the retired fish suite.

The previous ``fish_suite.json`` knee was fit on the test split. This runner
fits each seed on that seed's training windows, chooses every threshold on
that seed's validation windows, and scores the frozen rule once on test.
Checkpoints under ``artifacts/models`` are left untouched.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from uaere.causal.reasoner import CausalReasoner
from uaere.classify.train import transform
from uaere.data.adapters import FISH_DATA_DOI, FISH_IMPUTED_ENV, FishCallAdapter
from uaere.eval.fish_suite import _band_check, resolve_fish_root
from uaere.eval.gate_compare import (
    FISH_DEVICE_ID,
    FISH_MACS_BY_LEVEL,
    HAND_SET_WEIGHTS,
    METHOD_CONCENTRATION,
    apply_explanation_policy,
    energy_parts,
    energy_saving_pct,
    epistemic_from_concentration,
    fixed_policy_row,
    relifted_trusts,
    score_array,
    select_abstention_tau,
    select_explanation_tau,
    abstention_metrics,
)
from uaere.hardware.profiles import PROFILES
from uaere.kg.graph import load_kg
from uaere.math.energy import compute_energy_j
from uaere.math.metrics import brier, ece, roc_auc
from uaere.models.fish_head import fish_feature_rows, fit_fish_head, toadfish_band
from uaere.policy.nsga2 import nsga2
from uaere.policy.objectives import evaluate_policy
from uaere.policy.runtime_gate import chebyshev_select
from uaere.trust.trust_score import TrustEngine
from uaere.types import ExecutionLevel, TrustScore, WindowRecord

PROTOCOL_SEEDS = (0, 1, 2)
SNR_DB = (20, 10, 5, 0)
CONCENTRATION_GRID = (5.0, 10.0, 20.0, 40.0)
WEIGHT_SCALES = (0.5, 1.0, 2.0)
COST_SCALES = (0.5, 1.0, 2.0, 5.0)


def jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonify(payload), indent=2), encoding="utf-8")


def _ids(records: list[WindowRecord]) -> set[str]:
    return {r.source_id for r in records}


def _names(trusts: list[TrustScore]) -> list[str]:
    return [t.predicted_name() for t in trusts]


def _gold(records: list[WindowRecord]) -> list[str]:
    return [r.taxon or r.event_class.value for r in records]


def _present(records: list[WindowRecord]) -> np.ndarray:
    return np.array([int(r.event_present) for r in records], dtype=int)


def _score_split(head, records: list[WindowRecord]) -> list[TrustScore]:
    features, mels = fish_feature_rows(records)
    scaled = transform(head, features)
    engine = TrustEngine(head)
    return [
        engine.score(scaled[i], mels[i], records[i].health_estimate, records[i].environment)
        for i in range(len(records))
    ]


def _spread(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "n_unique_at_1e6": int(np.unique(np.round(array, 6)).size),
    }


def _selection_brief(selected: dict) -> dict:
    chosen = selected["chosen"]
    return {
        "tau": selected["tau"],
        "percentile": selected["percentile"],
        "selected_on": "validation",
        "energy_j": chosen.get("energy_j"),
        "explanation_false_alarm": chosen.get("explanation_false_alarm"),
        "explanation_miss": chosen.get("explanation_miss"),
        "chebyshev": chosen.get("chebyshev"),
    }


def _gated_row(
    method: str,
    kind: str,
    val_trusts,
    test_trusts,
    y_val,
    y_test,
    pred,
    gold,
    labels,
    parts_l1,
    parts_l3,
    weights: dict | None = None,
) -> tuple[dict, dict]:
    val_scores = score_array(val_trusts, kind, weights)
    test_scores = score_array(test_trusts, kind, weights)
    selected = select_explanation_tau(val_scores, y_val, float(parts_l3["total_j"]), float(parts_l1["total_j"]))
    row = apply_explanation_policy(
        test_scores,
        y_test,
        pred,
        gold,
        labels,
        selected["tau"],
        parts_skip=parts_l1,
        parts_explain=parts_l3,
        always_l3_j=float(parts_l3["total_j"]),
    )
    row["method"] = method
    row["score"] = kind
    row["tau_percentile"] = selected["percentile"]
    row["tau_selected_on"] = "validation"
    row["validation_selection"] = _selection_brief(selected)
    if weights is not None:
        row["weights"] = weights
    return row, selected


def _validation_curve(selected: dict, always_l3_j: float) -> list[dict]:
    curve = []
    for row in selected["grid"]:
        curve.append(
            {
                "percentile": row["percentile"],
                "tau": row["tau"],
                "estimated_energy_j": row["energy_j"],
                "estimated_energy_savings_vs_always_l3_pct": energy_saving_pct(row["energy_j"], always_l3_j),
                "explanation_false_alarm_on_negatives": row["explanation_false_alarm"],
                "explanation_recall_on_positives": row["explanation_recall"],
                "additional_processing_rate": row["additional_processing_rate"],
                "scored_on": "validation",
                "same_split_as_selection": True,
            }
        )
    return curve


def _cause_report(reasoner: CausalReasoner, records, trusts) -> tuple[np.ndarray, dict]:
    hits = []
    called = 0
    called_match = 0
    for rec, trust in zip(records, trusts, strict=True):
        name = trust.predicted_name()
        expl = reasoner.explain(
            trust.predicted_class(),
            rec.waveform,
            rec.sample_rate,
            rec.environment,
            taxon=name,
        )
        gold = rec.taxon or rec.event_class.value
        hits.append(int(expl.cause_id == rec.cause_id and name == gold))
        if name == "opsanus_tau":
            called += 1
            if expl.cause_id == "cause.taxon.opsanus_tau":
                called_match += 1
    return np.asarray(hits, dtype=int), {
        "opsanus_called": called,
        "cause_id_is_opsanus_when_model_says_opsanus": (called_match / called) if called else None,
        "structurally_determined": True,
        "note": (
            "event.toadfish_boatwhistle has one caused_by edge, to "
            "cause.taxon.opsanus_tau. Matching that id restates the graph. "
            "It is not evidence that the graph improves the label."
        ),
    }


def _add_noise(waveform: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal = np.asarray(waveform, dtype=np.float64)
    power = float(np.mean(signal * signal))
    if power <= 1e-18:
        return signal.copy()
    sigma = math.sqrt(power / (10.0 ** (float(snr_db) / 10.0)))
    return signal + rng.normal(0.0, sigma, size=signal.shape)


def _snr_block(head, records, tau: float, parts_l1, parts_l3, labels) -> list[dict]:
    """White noise added before features. The head and τ stay frozen."""
    from dataclasses import replace

    rows = []
    y = _present(records)
    gold = _gold(records)
    for snr_db in SNR_DB:
        rng = np.random.default_rng(10_000 + int(snr_db))
        noisy = [replace(rec, waveform=_add_noise(rec.waveform, snr_db, rng)) for rec in records]
        trusts = _score_split(head, noisy)
        scores = score_array(trusts, "trust")
        pred = _names(trusts)
        row = apply_explanation_policy(
            scores,
            y,
            pred,
            gold,
            labels,
            tau,
            parts_skip=parts_l1,
            parts_explain=parts_l3,
            always_l3_j=float(parts_l3["total_j"]),
        )
        conf = np.array([t.wake_confidence for t in trusts], dtype=np.float64)
        row.update(
            {
                "method": "trust",
                "snr_db": snr_db,
                "noise": "white",
                "derived_evaluation": True,
                "not_a_field_recording": True,
                "head": "frozen seed-0 logistic",
                "tau_source": "clean validation",
                "bands_retuned": False,
                "wake_roc_auc": float(roc_auc(y, conf)),
                "wake_ece": float(ece(y.astype(float), conf)),
            }
        )
        rows.append(row)
    return rows


def _mean_block(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(array.mean())
    if array.size < 2:
        return {"n": int(array.size), "mean": mean, "std": 0.0, "ci95": [mean, mean]}
    std = float(array.std(ddof=1))
    from scipy.stats import t

    half = float(t.ppf(0.975, array.size - 1) * std / math.sqrt(array.size))
    return {"n": int(array.size), "mean": mean, "std": std, "ci95": [mean - half, mean + half]}


def _paired(a, b) -> dict:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    diff = left - right
    out = {
        "n": int(left.size),
        "mean_diff": float(diff.mean()) if diff.size else None,
        "std_diff": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
        "test": "paired two-sided t-test on seeds",
    }
    if left.size < 2 or np.allclose(diff, 0.0):
        out["pvalue"] = None
        out["note"] = "The paired difference is zero." if left.size >= 2 else "Fewer than two seeds."
        return out
    from scipy.stats import ttest_rel

    out["pvalue"] = float(ttest_rel(left, right).pvalue)
    out["note"] = "Three seeds is a small sample. The p-value is descriptive."
    return out


def _energy_model(profile) -> dict:
    e0 = compute_energy_j(ExecutionLevel.L0, FISH_MACS_BY_LEVEL, profile)
    e1 = compute_energy_j(ExecutionLevel.L1, FISH_MACS_BY_LEVEL, profile)
    e2 = compute_energy_j(ExecutionLevel.L2, FISH_MACS_BY_LEVEL, profile)
    e3 = compute_energy_j(ExecutionLevel.L3, FISH_MACS_BY_LEVEL, profile)
    return {
        "device_id": FISH_DEVICE_ID,
        "source": "datasheet cost model stm32l476 via compute_energy_j. Not a measured current.",
        "measured_power": False,
        "underwater_deployment": False,
        "macs_by_level": dict(FISH_MACS_BY_LEVEL),
        "e_level_j": {"L0": e0, "L1": e1, "L2": e2, "L3": e3},
        "e_a_j": e1,
        "e_g_j": 0.0,
        "e_g_note": (
            "Trust is a sigmoid of quantities Level 1 already computed. "
            "No separate trust MAC count exists, so E_g is not added again."
        ),
        "e_p_when_explaining_j": e3 - e1,
        "e_level3_graph_only_j": e3 - e2,
        "level2_share_of_additional_path": (e2 - e1) / (e3 - e1),
        "level2_note": (
            "Selecting Level 3 in compute_energy_j also charges Level 2. "
            "The fish label is the logistic at Level 1. Level 2 does not supply that label."
        ),
        "arrival_rate_in_model": False,
        "arrival_rate_note": (
            "Energy is joules per 1 s window. The same arrival rate multiplies every "
            "policy and cancels in the savings percentage. No arrival process was fitted."
        ),
        "formula": "E_i = E_a,i + E_g,i + E_p,i",
        "savings_formula": "100 * (E_always_L3 - E_method) / E_always_L3",
    }


def run_one_seed(root: Path, seed: int, train_steps: int, max_per_class: int, min_files: int, profile) -> dict:
    train, val, test, meta = FishCallAdapter(
        root,
        seed=seed,
        max_per_class=max_per_class,
        min_files=min_files,
    ).load_split()
    leaks = {
        "train_val": len(_ids(train) & _ids(val)),
        "train_test": len(_ids(train) & _ids(test)),
        "val_test": len(_ids(val) & _ids(test)),
    }
    print(f"seed {seed} windows train={len(train)} val={len(val)} test={len(test)} leaks={leaks}", flush=True)
    head = fit_fish_head(train, steps=train_steps, seed=seed)
    n_iter = int(np.max(np.atleast_1d(head.sklearn.n_iter_)))
    print(
        f"seed {seed} fit n_iter={n_iter} max_iter={int(head.sklearn.max_iter)}",
        flush=True,
    )
    val_trusts = _score_split(head, val)
    test_trusts = _score_split(head, test)
    stored = np.array([t.event_trust for t in val_trusts], dtype=np.float64)
    rebuilt = score_array(val_trusts, "trust")
    if float(np.max(np.abs(stored - rebuilt))) > 1e-9:
        raise RuntimeError("trust rebuild does not match TrustEngine.event_trust")
    lifted_method = relifted_trusts(
        val_trusts,
        float(head.concentration),
        head.negative_index,
        source_concentration=float(head.concentration),
    )
    if float(np.max(np.abs(score_array(lifted_method, "trust") - rebuilt))) > 1e-8:
        raise RuntimeError("relift at the method concentration does not match T(e)")

    labels = list(head.labels or [])
    y_val = _present(val)
    y_test = _present(test)
    pred = _names(test_trusts)
    gold = _gold(test)
    val_pred = _names(val_trusts)
    val_gold = _gold(val)
    from sklearn.metrics import f1_score

    val_macro = float(f1_score(val_gold, val_pred, labels=labels, average="macro", zero_division=0))
    conf = np.array([t.wake_confidence for t in test_trusts], dtype=np.float64)
    parts_l0 = energy_parts(ExecutionLevel.L0, FISH_MACS_BY_LEVEL, profile)
    parts_l1 = energy_parts(ExecutionLevel.L1, FISH_MACS_BY_LEVEL, profile)
    parts_l3 = energy_parts(ExecutionLevel.L3, FISH_MACS_BY_LEVEL, profile)
    if abs(float(parts_l3["total_j"]) - compute_energy_j(ExecutionLevel.L3, FISH_MACS_BY_LEVEL, profile)) > 1e-12:
        raise RuntimeError("energy parts do not sum to compute_energy_j")

    always_cheap = fixed_policy_row(
        pred, gold, labels, y_test, explain_all=False, parts_skip=parts_l1, parts_explain=parts_l3, always_l3_j=float(parts_l3["total_j"])
    )
    always_cheap["method"] = "always_l1"
    always_costly = fixed_policy_row(
        pred, gold, labels, y_test, explain_all=True, parts_skip=parts_l1, parts_explain=parts_l3, always_l3_j=float(parts_l3["total_j"])
    )
    always_costly["method"] = "always_l3"

    specs = (
        ("confidence", "wake", None),
        ("uncertainty", "uncertainty", None),
        ("confidence_uncertainty", "confidence_uncertainty", None),
        ("trust", "trust", None),
        ("trust_no_environment", "trust", {**HAND_SET_WEIGHTS, "env": 0.0}),
        ("trust_no_uncertainty", "trust", {**HAND_SET_WEIGHTS, "uncertainty": 0.0}),
    )
    rows = {"always_l1": always_cheap, "always_l3": always_costly}
    selected = {}
    for name, kind, weights in specs:
        row, sel = _gated_row(
            name, kind, val_trusts, test_trusts, y_val, y_test, pred, gold, labels, parts_l1, parts_l3, weights
        )
        rows[name] = row
        selected[name] = sel
    no_graph = fixed_policy_row(
        pred, gold, labels, y_test, explain_all=False, parts_skip=parts_l1, parts_explain=parts_l3, always_l3_j=float(parts_l3["total_j"])
    )
    no_graph["method"] = "full_trust_level3_never"
    no_graph["note"] = (
        "T(e) is still computed. Level 3 is never run, so the knowledge graph "
        "adds no label and no joules. The graph is not an input of T(e)."
    )
    rows["full_trust_level3_never"] = no_graph
    anchor = rows["always_l1"]["macro_f1"]
    for row in rows.values():
        if abs(row["macro_f1"] - anchor) > 1e-9:
            raise RuntimeError(f"{row['method']} changed macro-F1; the explanation gate must not")

    def _abstain(method: str, kind: str) -> dict:
        val_scores = score_array(val_trusts, kind)
        test_scores = score_array(test_trusts, kind)
        sel = select_abstention_tau(val_scores, val_pred, val_gold, labels)
        metrics = abstention_metrics(pred, gold, test_scores >= sel["tau"], labels)
        metrics.update(
            {
                "method": method,
                "score": kind,
                "action": "abstain_if_score_lt_tau",
                "tau": sel["tau"],
                "tau_percentile": sel["percentile"],
                "tau_selected_on": "validation",
                "note": (
                    "Separate baseline. Refusing a label is not the explanation gate "
                    "and is not the macro-F1 of the logistic."
                ),
            }
        )
        return metrics

    abstention = {
        "trust": _abstain("abstain_on_trust", "trust"),
        "confidence": _abstain("abstain_on_confidence", "wake"),
    }

    entropy_row, entropy_sel = _gated_row(
        "entropy", "entropy", val_trusts, test_trusts, y_val, y_test, pred, gold, labels, parts_l1, parts_l3
    )
    uncertainty_scores = score_array(test_trusts, "uncertainty")
    entropy_scores = score_array(test_trusts, "entropy")
    wake_scores = score_array(test_trusts, "wake")

    def _unit_ece(score: np.ndarray) -> dict:
        outside = float(np.mean((score < 0.0) | (score > 1.0)))
        return {"ece": float(ece(y_test.astype(float), np.clip(score, 0.0, 1.0))), "fraction_outside_unit_interval": outside}

    k = int(len(test_trusts[0].p))
    u_e = np.array([t.u_epistemic for t in test_trusts], dtype=np.float64)
    u_a = np.array([t.u_aleatoric for t in test_trusts], dtype=np.float64)
    rho = np.array([t.env_consistency for t in test_trusts], dtype=np.float64)
    kappa = np.array([t.health_consistency for t in test_trusts], dtype=np.float64)

    reasoner = CausalReasoner(load_kg())
    _hits, cause = _cause_report(reasoner, test, test_trusts)
    band = _band_check(reasoner, test, y_test)

    weight_rows = []
    for name in ("wake", "health", "env", "uncertainty"):
        for scale in WEIGHT_SCALES:
            weights = dict(HAND_SET_WEIGHTS)
            weights[name] = HAND_SET_WEIGHTS[name] * scale
            scores = score_array(val_trusts, "trust", weights)
            sel = select_explanation_tau(scores, y_val, float(parts_l3["total_j"]), float(parts_l1["total_j"]))
            applied = apply_explanation_policy(
                scores,
                y_val,
                val_pred,
                val_gold,
                labels,
                sel["tau"],
                parts_skip=parts_l1,
                parts_explain=parts_l3,
                always_l3_j=float(parts_l3["total_j"]),
            )
            applied.update(
                {
                    "scaled_weight": name,
                    "scale": scale,
                    "weights": weights,
                    "tau_percentile": sel["percentile"],
                    "scored_on": "validation",
                    "same_split_as_selection": True,
                }
            )
            weight_rows.append(applied)
    nominal_taus = {row["tau"] for row in weight_rows if row["scale"] == 1.0}
    if len(nominal_taus) != 1 or abs(float(next(iter(nominal_taus))) - float(rows["trust"]["tau"])) > 1e-9:
        raise RuntimeError("a unit weight scale did not reproduce the trust threshold")

    def _cost_family(family: str) -> list[dict]:
        found_rows = []
        for scale in COST_SCALES:
            parts_hi = energy_parts(ExecutionLevel.L3, FISH_MACS_BY_LEVEL, profile, **{family: scale})
            scores = score_array(val_trusts, "trust")
            sel = select_explanation_tau(scores, y_val, float(parts_hi["total_j"]), float(parts_l1["total_j"]))
            applied = apply_explanation_policy(
                scores,
                y_val,
                val_pred,
                val_gold,
                labels,
                sel["tau"],
                parts_skip=parts_l1,
                parts_explain=parts_hi,
                always_l3_j=float(parts_hi["total_j"]),
            )
            applied.update(
                {
                    "family": family,
                    "scale": scale,
                    "tau_percentile": sel["percentile"],
                    "e_explain_j": parts_hi["total_j"],
                    "scored_on": "validation",
                    "same_split_as_selection": True,
                    "source": "scaled datasheet model, not a new measurement",
                }
            )
            found_rows.append(applied)
        return found_rows

    cost_sensitivity = {
        "additional_path_scale": _cost_family("additional_path_scale"),
        "level3_scale": _cost_family("level3_scale"),
    }
    for family_rows in cost_sensitivity.values():
        unit = next(row for row in family_rows if row["scale"] == 1.0)
        if abs(float(unit["tau"]) - float(rows["trust"]["tau"])) > 1e-9:
            raise RuntimeError("unit cost scale moved the validation threshold")

    concentration_rows = []
    for concentration in CONCENTRATION_GRID:
        lifted = relifted_trusts(
            val_trusts,
            concentration,
            head.negative_index,
            source_concentration=float(head.concentration),
        )
        scores = score_array(lifted, "trust")
        sel = select_explanation_tau(scores, y_val, float(parts_l3["total_j"]), float(parts_l1["total_j"]))
        applied = apply_explanation_policy(
            scores,
            y_val,
            _names(lifted),
            val_gold,
            labels,
            sel["tau"],
            parts_skip=parts_l1,
            parts_explain=parts_l3,
            always_l3_j=float(parts_l3["total_j"]),
        )
        applied.update(
            {
                "concentration": concentration,
                "is_the_reported_method": concentration == METHOD_CONCENTRATION,
                "u_epistemic": epistemic_from_concentration(k, concentration),
                "tau_percentile": sel["percentile"],
                "scored_on": "validation",
                "same_split_as_selection": True,
                "head_refit": False,
            }
        )
        concentration_rows.append(applied)

    # Optional staircase. Three thresholds, chosen on validation, scored once on test.
    # Not comparable to the one-τ table: it may stop at Level 0 and it optimises the
    # six-objective vector used by the old suite.
    hit_val, _ = _cause_report(reasoner, val, val_trusts)
    hit_test = _hits

    def _eval_val(policy):
        return evaluate_policy(policy, val_trusts, y_val, hit_val, profile, FISH_MACS_BY_LEVEL)

    front = nsga2(_eval_val, seed=seed, pop_size=12, n_gen=6)
    knee = chebyshev_select(front)
    held = evaluate_policy(knee, test_trusts, y_test, hit_test, profile, FISH_MACS_BY_LEVEL)
    staircase = {
        "comparable_to_one_threshold_table": False,
        "thresholds_selected_on": "validation",
        "search": "NSGA-II pop 12, gen 6, then Chebyshev on the six-objective front",
        "tau1": knee.tau1,
        "tau2": knee.tau2,
        "tau3": knee.tau3,
        "test_energy_j": held.energy_j_s,
        "test_detection_miss": held.miss,
        "test_detection_false_alarm": held.fa,
        "test_explanation_coverage": held.explanation_coverage,
        "note": (
            "Detection miss uses level >= L1 and C_wake >= 0.5. It is not macro-F1. "
            "A Level-0 choice skips the logistic in this cost model. "
            "The retired fish_suite knee was fit on test and is not this row."
        ),
    }

    snr = None
    if seed == 0:
        print("seed 0 SNR perturbations", flush=True)
        snr = _snr_block(head, test, rows["trust"]["tau"], parts_l1, parts_l3, labels)

    from uaere.eval.gate_compare import classification_report

    report = classification_report(pred, gold, labels)
    print(
        f"seed {seed} macro_f1={report['macro_f1']:.4f} trust_tau={rows['trust']['tau']:.4f} "
        f"l3_rate={rows['trust']['level3_rate']:.3f} "
        f"savings={rows['trust']['estimated_energy_savings_vs_always_l3_pct']:.2f}",
        flush=True,
    )
    return {
        "seed": seed,
        "train_steps": train_steps,
        "logistic_n_iter": n_iter,
        "logistic_max_iter": int(head.sklearn.max_iter),
        "concentration": float(head.concentration),
        "negative_index": int(head.negative_index),
        "labels": labels,
        "split": {
            "unit": meta.get("split_unit"),
            "note": meta.get("split_note"),
            "counts": meta.get("counts"),
            "n_windows": meta.get("n_windows"),
            "n_groups": meta.get("n_groups"),
            "group_overlaps": leaks,
            "cap_unit": meta.get("cap_unit"),
        },
        "validation_macro_f1": val_macro,
        "classification": {
            **report,
            "wake_roc_auc": float(roc_auc(y_test, conf)),
            "wake_ece": float(ece(y_test.astype(float), conf)),
            "wake_brier": float(brier(y_test.astype(float), conf)),
            "score": "C_wake = 1 - P(other) after the Dirichlet lift",
            "presegmented_events": True,
            "end_to_end_discovery": False,
        },
        "components": {
            "u_epistemic": _spread(u_e),
            "u_epistemic_formula": epistemic_from_concentration(k, float(head.concentration)),
            "u_epistemic_constant": bool(np.allclose(u_e, u_e[0])),
            "u_aleatoric": _spread(u_a),
            "env_consistency": _spread(rho),
            "health_consistency": _spread(kappa),
            "environment_imputed": True,
            "imputed_environment": {
                "temperature_c": FISH_IMPUTED_ENV.temperature_c,
                "salinity_psu": FISH_IMPUTED_ENV.salinity_psu,
                "depth_m": FISH_IMPUTED_ENV.depth_m,
                "sea_state": FISH_IMPUTED_ENV.sea_state,
                "turbulence": FISH_IMPUTED_ENV.turbulence,
            },
        },
        "energy_identity": {"L0": parts_l0, "L1": parts_l1, "L3": parts_l3},
        "table1_test": rows,
        "abstention_test": abstention,
        "uncertainty_comparison_test": {
            "dirichlet_inverted_sum": rows["uncertainty"],
            "entropy_of_dirichlet_mean": entropy_row,
            "ece": {
                "wake": _unit_ece(wake_scores),
                "dirichlet_inverted_sum": _unit_ece(uncertainty_scores),
                "entropy_explain_score": _unit_ece(entropy_scores),
                "trust": _unit_ece(score_array(test_trusts, "trust")),
            },
            "entropy_validation_selection": _selection_brief(entropy_sel),
            "note": (
                "Both uncertainty scores are thresholded with the same validation rule. "
                "A lower ECE is not by itself a better gate."
            ),
        },
        "knowledge_graph": {"cause_match": cause, "band_check": band},
        "tau_sensitivity_validation": _validation_curve(selected["trust"], float(parts_l3["total_j"])),
        "abstention_is_not_in_tau_curve": True,
        "weight_sensitivity_validation": weight_rows,
        "cost_sensitivity_validation": cost_sensitivity,
        "concentration_sensitivity_validation": concentration_rows,
        "nsga_staircase_validation_selected": staircase,
        "snr_test_frozen_seed0_policy": snr,
    }


def _column(seed_reports: list[dict], method: str, key: str) -> list[float]:
    return [float(report["table1_test"][method][key]) for report in seed_reports]


def aggregate(seed_reports: list[dict]) -> dict:
    methods = list(seed_reports[0]["table1_test"])
    metric_keys = (
        "macro_f1",
        "estimated_energy_j",
        "estimated_energy_savings_vs_always_l3_pct",
        "additional_processing_rate",
        "explanation_recall_on_positives",
        "explanation_false_alarm_on_negatives",
        "error_rate_among_accepted",
        "abstention_rate",
    )
    table = {}
    paired = {}
    for method in methods:
        table[method] = {key: _mean_block(_column(seed_reports, method, key)) for key in metric_keys}
        if method != "trust":
            paired[method] = {
                key: _paired(_column(seed_reports, "trust", key), _column(seed_reports, method, key))
                for key in (
                    "estimated_energy_j",
                    "additional_processing_rate",
                    "explanation_false_alarm_on_negatives",
                    "explanation_recall_on_positives",
                    "macro_f1",
                )
            }
    labels = seed_reports[0]["labels"]
    per_class = {
        label: _mean_block(report["classification"]["per_class_recall"][label] for report in seed_reports)
        for label in labels
    }
    return {
        "seeds": [report["seed"] for report in seed_reports],
        "classification_macro_f1": _mean_block(report["classification"]["macro_f1"] for report in seed_reports),
        "wake_roc_auc": _mean_block(report["classification"]["wake_roc_auc"] for report in seed_reports),
        "wake_ece": _mean_block(report["classification"]["wake_ece"] for report in seed_reports),
        "per_class_recall": per_class,
        "band_check_roc_auc": _mean_block(
            report["knowledge_graph"]["band_check"]["roc_auc"] for report in seed_reports
        ),
        "table1": table,
        "paired_trust_minus_baseline": paired,
        "paired_test": "Two-sided paired t-test across seeds. n=3. Not a substitute for a larger repeat.",
    }


def _markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Review upgrade (validation-chosen gates)",
        "",
        "Numbers in this note are copied from the JSON written beside it.",
        "Energy is the stm32l476 datasheet model, not a measured current.",
        "The explanation gate does not change the logistic label.",
        "",
        f"Macro-F1 mean {summary['classification_macro_f1']['mean']:.4f} "
        f"± {summary['classification_macro_f1']['std']:.4f}.",
        f"Wake ROC-AUC mean {summary['wake_roc_auc']['mean']:.4f} "
        f"± {summary['wake_roc_auc']['std']:.4f}.",
        f"Band-check ROC-AUC mean {summary['band_check_roc_auc']['mean']:.4f} "
        f"± {summary['band_check_roc_auc']['std']:.4f}.",
        "",
        "| Method | Macro-F1 | Energy J | Savings % | Level-3 rate | Explain recall | Explain FA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, block in summary["table1"].items():
        lines.append(
            "| {method} | {f1:.4f} ± {f1s:.4f} | {e:.6f} | {s:.2f} | {r:.3f} | {rec:.3f} | {fa:.3f} |".format(
                method=method,
                f1=block["macro_f1"]["mean"],
                f1s=block["macro_f1"]["std"],
                e=block["estimated_energy_j"]["mean"],
                s=block["estimated_energy_savings_vs_always_l3_pct"]["mean"],
                r=block["additional_processing_rate"]["mean"],
                rec=block["explanation_recall_on_positives"]["mean"],
                fa=block["explanation_false_alarm_on_negatives"]["mean"],
            )
        )
    lines += [
        "",
        "Abstention rows are in the JSON under `abstention_test`. They are not this table.",
        "Sensitivity rows are validation curves. They were not re-picked on the test set.",
        "",
    ]
    return "\n".join(lines)


def run_review_upgrade(
    root: str | Path | None = None,
    out_dir: str | Path = "artifacts/paper/review_upgrade",
    seeds: tuple[int, ...] = PROTOCOL_SEEDS,
    train_steps: int = 2000,
    max_per_class: int = 3000,
    min_files: int = 40,
) -> dict:
    found = resolve_fish_root(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[FISH_DEVICE_ID]
    seed_reports = []
    for seed in seeds:
        report = run_one_seed(found, int(seed), int(train_steps), int(max_per_class), int(min_files), profile)
        _dump(out / f"seed_{seed}.json", report)
        seed_reports.append(report)
    payload = {
        "track": "review_upgrade",
        "title": "Uncertainty-Aware Event-Level Trust Gating for Energy-Efficient Underwater Acoustic Event Classification",
        "doi": FISH_DATA_DOI,
        "root": str(found),
        "seeds": list(seeds),
        "train_steps": int(train_steps),
        "max_per_class": int(max_per_class),
        "min_files": int(min_files),
        "features": {"name": "mel_stats+opsanus_band", "band_hz": list(toadfish_band())},
        "trust": {
            "formula": "T(e) = sigmoid(w_c C_wake + w_h κ + w_s ρ - w_u (u_a + u_e) + b)",
            "weights": dict(HAND_SET_WEIGHTS),
            "weights_sum_to_one": False,
            "weights_learned": False,
            "is_calibrated_probability": False,
            "convex_combination_used": False,
            "knowledge_graph_inside_trust": False,
            "implemented_action": "Level 3 explanation if score >= τ, otherwise Level 1. The label is still emitted.",
            "abstention": "A separate baseline. It is not this action.",
        },
        "energy_model": _energy_model(profile),
        "retired": {
            "artifact": "artifacts/paper/fish_suite.json",
            "reason": "That knee was selected on the test split. Do not quote its taus or its 3.82 mJ as this result.",
        },
        "seed_reports": seed_reports,
        "summary": aggregate(seed_reports),
        "checkpoints_written": False,
    }
    _dump(out / "review_upgrade.json", payload)
    (out / "review_upgrade.md").write_text(_markdown(payload), encoding="utf-8")
    print(f"wrote {out / 'review_upgrade.json'}", flush=True)
    return payload
