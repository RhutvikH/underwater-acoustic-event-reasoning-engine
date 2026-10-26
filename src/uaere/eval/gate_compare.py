"""One-threshold gates chosen on one array and scored on another.

The fish label is the logistic argmax. These gates do not change it. A high
score pays for Level-3 explanation. A low score stops at Level 1 and still
emits the label. That is the implemented action. Abstention is a separate
baseline and is labelled as such.

T(e) is the hand-set sigmoid in ``TrustEngine``. It is a gate score, not a
calibrated probability that the label is correct, and it is not a convex
combination of its inputs. The weights do not sum to 1.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from uaere.math.energy import compute_energy_j
from uaere.types import ExecutionLevel, TrustScore

# Same MAC budget the fish suite used. Differences versus that suite are the
# policy, not a new device model.
FISH_MACS_BY_LEVEL: dict[int, float] = {0: 8e4, 1: 2e6, 2: 4e5, 3: 2e3, 4: 2e3}
FISH_DEVICE_ID = "stm32l476"

# Hand-set logit weights. Not learned. Not a simplex.
HAND_SET_WEIGHTS: dict[str, float] = {
    "wake": 2.2,
    "health": 0.8,
    "env": 1.1,
    "uncertainty": 1.4,
    "bias": -0.6,
}

# Validation grid. The test set is not an argument of the selector.
PERCENTILES: tuple[int, ...] = tuple(range(5, 100, 5))
ABSTAIN_TOKEN = "__abstain__"
# Stored Dirichlet lift. A sweep re-lifts the same logistic probabilities.
METHOD_CONCENTRATION = 20.0


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -30.0, 30.0))
    return 1.0 / (1.0 + math.exp(-z))


def trust_logit(trust: TrustScore, weights: dict[str, float] | None = None) -> float:
    """Same linear part as ``TrustEngine.score``. The graph is not an input."""
    w = HAND_SET_WEIGHTS if weights is None else weights
    return (
        w["wake"] * float(trust.wake_confidence)
        + w["health"] * float(trust.health_consistency)
        + w["env"] * float(trust.env_consistency)
        - w["uncertainty"] * (float(trust.u_aleatoric) + float(trust.u_epistemic))
        + w["bias"]
    )


def epistemic_from_concentration(n_classes: int, concentration: float) -> float:
    """u_e = K / (K + c) when every class is lifted as α = 1 + c p."""
    return float(n_classes) / (float(n_classes) + float(concentration))


def logistic_proba_from_alpha(alpha: np.ndarray, concentration: float = METHOD_CONCENTRATION) -> np.ndarray:
    """Invert α = 1 + c p. Used only to re-lift a stored head at another c."""
    p = (np.asarray(alpha, dtype=np.float64) - 1.0) / float(concentration)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return np.ones(p.shape[0], dtype=np.float64) / max(p.shape[0], 1)
    return p / total


def dirichlet_lift(p_logistic: np.ndarray, concentration: float) -> tuple[np.ndarray, float, float]:
    """Return the Dirichlet mean, u_a, and u_e. u_e does not depend on p."""
    p = np.asarray(p_logistic, dtype=np.float64)
    alpha = 1.0 + float(concentration) * p
    total = float(alpha.sum())
    mean = alpha / total
    u_e = float(mean.shape[0]) / total
    u_a = 1.0 - float(np.max(mean))
    return mean, u_a, u_e


def entropy_explain_score(p: np.ndarray) -> float:
    """1 − H(p) / log K. High means peaked, so the gate may explain.

    This is a confidence proxy built from the Dirichlet mean. It is not a
    second trained uncertainty model.
    """
    prob = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    prob = prob / float(prob.sum())
    h = float(-np.sum(prob * np.log(prob)))
    hmax = float(np.log(prob.shape[0]))
    if hmax <= 0.0:
        return 1.0
    return 1.0 - h / hmax


def uncertainty_explain_score(trust: TrustScore) -> float:
    """Invert u. Large uncertainty must not be the high side of the gate.

    With a fixed concentration, u_e is the same on every window. u_a is low
    when the Dirichlet mean is peaked on either class, so a confident negative
    scores high here. That is the baseline, not a bug to be patched.
    """
    return 1.0 - (float(trust.u_aleatoric) + float(trust.u_epistemic))


def wake_from_mean(p_hat: np.ndarray, negative_index: int | None) -> float:
    if negative_index is None:
        return float(np.max(p_hat))
    return float(1.0 - p_hat[int(negative_index)])


def score_array(
    trusts: list[TrustScore],
    kind: str,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """High means 'pay for Level 3'. ``kind`` selects which scalar is thresholded."""
    out = np.empty(len(trusts), dtype=np.float64)
    w = dict(HAND_SET_WEIGHTS if weights is None else weights)
    if kind == "confidence_uncertainty":
        w["health"] = 0.0
        w["env"] = 0.0
    for i, trust in enumerate(trusts):
        if kind == "wake":
            out[i] = float(trust.wake_confidence)
        elif kind == "uncertainty":
            out[i] = uncertainty_explain_score(trust)
        elif kind == "entropy":
            out[i] = entropy_explain_score(trust.p)
        elif kind in {"trust", "confidence_uncertainty"}:
            out[i] = sigmoid(trust_logit(trust, w))
        else:
            raise ValueError(f"unknown score kind {kind}")
    return out


def relifted_trusts(
    trusts: list[TrustScore],
    concentration: float,
    negative_index: int | None,
    source_concentration: float = METHOD_CONCENTRATION,
) -> list[TrustScore]:
    """Same logistic probabilities, another fixed concentration. Health and ρ stay."""
    lifted: list[TrustScore] = []
    for trust in trusts:
        p_log = logistic_proba_from_alpha(trust.alpha, source_concentration)
        p_hat, u_a, u_e = dirichlet_lift(p_log, concentration)
        c_wake = wake_from_mean(p_hat, negative_index)
        clone = TrustScore(
            wake_confidence=c_wake,
            event_trust=0.0,
            p=p_hat,
            alpha=1.0 + float(concentration) * p_log,
            u_aleatoric=u_a,
            u_epistemic=u_e,
            health_consistency=trust.health_consistency,
            env_consistency=trust.env_consistency,
            components=dict(trust.components),
            label=trust.label,
            taxon=trust.taxon,
        )
        clone.event_trust = sigmoid(trust_logit(clone))
        lifted.append(clone)
    return lifted


def energy_parts(
    level: ExecutionLevel,
    macs_by_level: dict[int, float],
    profile,
    *,
    additional_path_scale: float = 1.0,
    level3_scale: float = 1.0,
) -> dict[str, float | bool]:
    """Split ``compute_energy_j`` into front end, gate, and extra path.

    E_a is Level 0, or Levels 0 and 1 once the logistic runs. E_g is 0: trust
    is arithmetic on quantities Level 1 already produced, and this profile has
    no separate trust MAC count. E_p is the remainder through the selected
    level. Selecting Level 3 therefore includes the Level-2 budget, because
    ``compute_energy_j`` sums every level up to the one selected. At scale 1
    the three parts sum to ``compute_energy_j(level)``.
    """
    if int(level) >= 4:
        raise ValueError("the one-threshold comparison does not select Level 4")
    e0 = compute_energy_j(ExecutionLevel.L0, macs_by_level, profile)
    e1 = compute_energy_j(ExecutionLevel.L1, macs_by_level, profile)
    e2 = compute_energy_j(ExecutionLevel.L2, macs_by_level, profile)
    e3 = compute_energy_j(ExecutionLevel.L3, macs_by_level, profile)
    e3_scaled = e2 + float(level3_scale) * (e3 - e2)
    if int(level) <= 0:
        e_a, e_p, level3_only = e0, 0.0, 0.0
    elif int(level) == 1:
        e_a, e_p, level3_only = e1, 0.0, 0.0
    elif int(level) == 2:
        e_a = e1
        e_p = (e2 - e1) * float(additional_path_scale)
        level3_only = 0.0
    else:
        e_a = e1
        e_p = (e3_scaled - e1) * float(additional_path_scale)
        level3_only = float(level3_scale) * (e3 - e2)
    return {
        "e_a_j": float(e_a),
        "e_g_j": 0.0,
        "e_p_j": float(e_p),
        "total_j": float(e_a + e_p),
        "e_level3_only_j": float(level3_only),
        "includes_level2_budget": bool(int(level) >= 3),
    }


def energy_saving_pct(energy_j: float, baseline_j: float) -> float:
    """100 × (Ē_baseline − Ē) / Ē_baseline. Negative means the method costs more."""
    if baseline_j == 0.0:
        raise ValueError("energy baseline is 0")
    return 100.0 * (float(baseline_j) - float(energy_j)) / float(baseline_j)


def _chebyshev(rows: list[dict], keys: tuple[str, ...], tie_key: str) -> dict:
    if not rows:
        raise ValueError("empty threshold grid")
    grid = np.array([[float(row[key]) for key in keys] for row in rows], dtype=np.float64)
    span = np.maximum(grid.max(axis=0) - grid.min(axis=0), 1e-12)
    norm = (grid - grid.min(axis=0)) / span
    cheb = norm.max(axis=1)
    tie = np.array([float(row[tie_key]) for row in rows], dtype=np.float64)
    order = np.lexsort((np.arange(len(rows)), tie, cheb))
    chosen = dict(rows[int(order[0])])
    chosen["chebyshev"] = float(cheb[int(order[0])])
    return chosen


def _explanation_row(
    explain: np.ndarray,
    y_event: np.ndarray,
    e_explain: float,
    e_skip: float,
) -> dict[str, float | bool]:
    explain = np.asarray(explain, dtype=bool)
    y = np.asarray(y_event, dtype=int)
    rate = float(np.mean(explain)) if explain.size else 0.0
    energy = (1.0 - rate) * float(e_skip) + rate * float(e_explain)
    pos = y == 1
    neg = y == 0
    recall = float(np.mean(explain[pos])) if np.any(pos) else 0.0
    false_alarm = float(np.mean(explain[neg])) if np.any(neg) else 0.0
    return {
        "energy_j": float(energy),
        "explanation_false_alarm": false_alarm,
        "explanation_miss": 1.0 - recall,
        "explanation_recall": recall,
        "additional_processing_rate": rate,
        "has_positives": bool(np.any(pos)),
        "has_negatives": bool(np.any(neg)),
    }


def select_explanation_tau(
    scores: np.ndarray,
    y_event: np.ndarray,
    e_explain: float,
    e_skip: float,
) -> dict:
    """Pick τ on this array only. Percentiles 5, 10, …, 95. Chebyshev knee.

    Objectives, all minimised: energy, explanation false-alarm rate on
    negatives, and one minus explanation recall on positives. Ties break
    toward lower energy, then the earlier percentile.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y_event, dtype=int).reshape(-1)
    if scores.shape[0] != y.shape[0]:
        raise ValueError("scores and labels differ in length")
    rows: list[dict] = []
    for percentile in PERCENTILES:
        tau = float(np.quantile(scores, percentile / 100.0))
        row = _explanation_row(scores >= tau, y, e_explain, e_skip)
        row["tau"] = tau
        row["percentile"] = percentile
        rows.append(row)
    chosen = _chebyshev(
        rows,
        ("energy_j", "explanation_false_alarm", "explanation_miss"),
        "energy_j",
    )
    return {"tau": float(chosen["tau"]), "percentile": int(chosen["percentile"]), "chosen": chosen, "grid": rows}


def classification_report(pred, gold, labels: list[str]) -> dict:
    pred_a = np.asarray(pred)
    gold_a = np.asarray(gold)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold_a, pred_a, labels=list(labels), zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }
    matrix = confusion_matrix(gold_a, pred_a, labels=list(labels)).tolist()
    correct = pred_a == gold_a
    return {
        "macro_f1": float(np.mean(f1)) if len(f1) else float("nan"),
        "accuracy": float(np.mean(correct)) if correct.size else float("nan"),
        "error_rate": float(np.mean(~correct)) if correct.size else float("nan"),
        "per_class": per_class,
        "per_class_recall": {name: row["recall"] for name, row in per_class.items()},
        "confusion": {"labels": list(labels), "matrix": matrix},
    }


def apply_explanation_policy(
    scores: np.ndarray,
    y_event: np.ndarray,
    pred,
    gold,
    labels: list[str],
    tau: float | None,
    *,
    explain: np.ndarray | None = None,
    parts_skip: dict,
    parts_explain: dict,
    always_l3_j: float,
) -> dict:
    """Score a frozen explain-or-stop rule. Classification uses ``pred`` as given."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if explain is None:
        if tau is None:
            raise ValueError("tau is required when explain is not supplied")
        explain = scores >= float(tau)
    explain = np.asarray(explain, dtype=bool)
    costs = _explanation_row(
        explain,
        y_event,
        float(parts_explain["total_j"]),
        float(parts_skip["total_j"]),
    )
    rate = float(costs["additional_processing_rate"])
    report = classification_report(pred, gold, labels)
    e_a = (1.0 - rate) * float(parts_skip["e_a_j"]) + rate * float(parts_explain["e_a_j"])
    e_p = (1.0 - rate) * float(parts_skip["e_p_j"]) + rate * float(parts_explain["e_p_j"])
    e_l3 = (1.0 - rate) * float(parts_skip["e_level3_only_j"]) + rate * float(parts_explain["e_level3_only_j"])
    total = float(costs["energy_j"])
    return {
        "action": "level3_if_score_ge_tau",
        "tau": None if tau is None else float(tau),
        "macro_f1": report["macro_f1"],
        "accuracy": report["accuracy"],
        "error_rate_all_predictions": report["error_rate"],
        "error_rate_among_accepted": report["error_rate"],
        "abstention_rate": 0.0,
        "per_class_recall": report["per_class_recall"],
        "per_class": report["per_class"],
        "estimated_energy_j": total,
        "e_a_j": e_a,
        "e_g_j": 0.0,
        "e_p_j": e_p,
        "e_level3_only_j": e_l3,
        "includes_level2_budget_when_explaining": bool(parts_explain["includes_level2_budget"]),
        "estimated_energy_savings_vs_always_l3_pct": energy_saving_pct(total, always_l3_j),
        "additional_processing_rate": rate,
        "explanation_recall_on_positives": float(costs["explanation_recall"]),
        "explanation_false_alarm_on_negatives": float(costs["explanation_false_alarm"]),
        "level3_rate": rate,
    }


def abstention_metrics(pred, gold, accept: np.ndarray, labels: list[str]) -> dict:
    """Refusals are not a class in the macro average. They count as misses.

    ``macro_f1_accepted`` is only the windows that were labelled.
    ``macro_f1_refusals_as_errors`` scores every window, with a refusal
    matching no gold label. The two numbers are not interchangeable.
    """
    pred_a = np.asarray(pred)
    gold_a = np.asarray(gold)
    accept = np.asarray(accept, dtype=bool)
    shown = np.where(accept, pred_a, ABSTAIN_TOKEN)
    full = classification_report(shown, gold_a, labels)
    if not np.any(accept):
        accepted_f1 = None
        accepted_error = None
    else:
        accepted = classification_report(pred_a[accept], gold_a[accept], labels)
        accepted_f1 = accepted["macro_f1"]
        accepted_error = accepted["error_rate"]
    return {
        "abstention_rate": float(1.0 - np.mean(accept)) if accept.size else float("nan"),
        "macro_f1_accepted": accepted_f1,
        "macro_f1_refusals_as_errors": full["macro_f1"],
        "error_rate_among_accepted": accepted_error,
        "error_rate_full": full["error_rate"],
        "per_class_recall_refusals_as_errors": full["per_class_recall"],
    }


def select_abstention_tau(scores: np.ndarray, pred, gold, labels: list[str]) -> dict:
    """Validation rule for the abstention baseline. Not the explanation gate.

    Minimises abstention rate, error among accepted labels, and one minus
    full-set macro-F1 with refusals counted as errors. Ties break toward
    keeping more labels.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    rows: list[dict] = []
    for percentile in PERCENTILES:
        tau = float(np.quantile(scores, percentile / 100.0))
        metrics = abstention_metrics(pred, gold, scores >= tau, labels)
        if metrics["error_rate_among_accepted"] is None:
            continue
        rows.append(
            {
                "tau": tau,
                "percentile": percentile,
                "abstention_rate": float(metrics["abstention_rate"]),
                "accepted_error": float(metrics["error_rate_among_accepted"]),
                "full_miss": 1.0 - float(metrics["macro_f1_refusals_as_errors"]),
                "metrics": metrics,
            }
        )
    if not rows:
        raise ValueError("every abstention threshold rejected all validation windows")
    chosen = _chebyshev(rows, ("abstention_rate", "accepted_error", "full_miss"), "abstention_rate")
    return {"tau": float(chosen["tau"]), "percentile": int(chosen["percentile"]), "chosen": chosen, "grid": rows}


def fixed_policy_row(
    pred,
    gold,
    labels: list[str],
    y_event: np.ndarray,
    *,
    explain_all: bool,
    parts_skip: dict,
    parts_explain: dict,
    always_l3_j: float,
) -> dict:
    n = len(gold)
    explain = np.ones(n, dtype=bool) if explain_all else np.zeros(n, dtype=bool)
    row = apply_explanation_policy(
        np.zeros(n),
        y_event,
        pred,
        gold,
        labels,
        tau=None,
        explain=explain,
        parts_skip=parts_skip,
        parts_explain=parts_explain,
        always_l3_j=always_l3_j,
    )
    row["tau"] = None
    row["action"] = "always_level3" if explain_all else "always_level1"
    return row
