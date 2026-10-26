"""The review gate is the sigmoid already in TrustEngine, chosen on one array."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from uaere.config.schema import ModelConfig
from uaere.eval.gate_compare import (
    FISH_MACS_BY_LEVEL,
    HAND_SET_WEIGHTS,
    METHOD_CONCENTRATION,
    abstention_metrics,
    apply_explanation_policy,
    dirichlet_lift,
    energy_parts,
    energy_saving_pct,
    epistemic_from_concentration,
    logistic_proba_from_alpha,
    relifted_trusts,
    score_array,
    select_explanation_tau,
    sigmoid,
    trust_logit,
    uncertainty_explain_score,
)
from uaere.hardware.profiles import PROFILES
from uaere.math.dirichlet import aleatoric_uncertainty, dirichlet_mean, epistemic_uncertainty
from uaere.math.energy import compute_energy_j
from uaere.trust.evidential_head import EvidentialHead
from uaere.trust.trust_score import TrustEngine
from uaere.types import EnvironmentState, ExecutionLevel, SensorHealth


def _head() -> EvidentialHead:
    features = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    target = np.array([0, 0, 1, 1])
    clf = LogisticRegression(max_iter=80)
    clf.fit(features, target)
    head = EvidentialHead(in_dim=2, n_classes=2, seed=0)
    head.sklearn = clf
    head.labels = ("call", "other")
    head.negative_index = 1
    head._feat_mu = np.zeros(2)
    head._feat_sd = np.ones(2)
    return head


def _trusts():
    head = _head()
    engine = TrustEngine(head)
    mel = np.linspace(-0.2, 0.8, 32)
    health = SensorHealth()
    state = EnvironmentState()
    rows = []
    for feat in (np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.2, 0.8])):
        rows.append(engine.score(feat, mel, health, state))
    return rows


def test_sigmoid_matches_trust_engine_and_the_hand_set_weights():
    trusts = _trusts()
    engine = TrustEngine(_head())
    assert engine.w == HAND_SET_WEIGHTS
    assert ModelConfig().trust_weights == HAND_SET_WEIGHTS
    assert abs(sum(HAND_SET_WEIGHTS.values()) - 1.0) > 0.1
    for trust in trusts:
        assert trust.event_trust == sigmoid(trust_logit(trust))
        assert abs(score_array([trust], "trust")[0] - trust.event_trust) < 1e-12


def test_zeroing_a_weight_matches_a_second_engine():
    head = _head()
    feat = np.array([1.0, 0.0])
    mel = np.linspace(-0.4, 0.3, 32)
    full = TrustEngine(head).score(feat, mel, SensorHealth(), EnvironmentState())
    silenced = dict(HAND_SET_WEIGHTS)
    silenced["env"] = 0.0
    other = TrustEngine(head, weights=silenced).score(feat, mel, SensorHealth(), EnvironmentState())
    assert abs(other.event_trust - sigmoid(trust_logit(full, silenced))) < 1e-12
    assert other.event_trust != full.event_trust


def test_threshold_is_a_quantile_of_the_array_it_is_given():
    scores = np.array([0.1] * 20 + [0.9] * 20)
    labels = np.array([0] * 20 + [1] * 20)
    selected = select_explanation_tau(scores, labels, e_explain=5.0, e_skip=1.0)
    quantiles = [float(np.quantile(scores, q / 100.0)) for q in range(5, 100, 5)]
    assert any(np.isclose(selected["tau"], q) for q in quantiles)
    # The separating quantile explains positives and withholds negatives.
    explain = scores >= selected["tau"]
    assert explain[labels == 1].all()
    assert not explain[labels == 0].any()
    mixed = np.concatenate([scores, np.zeros(40)])
    mixed_y = np.concatenate([labels, np.array([0] * 20 + [1] * 20)])
    leaked = select_explanation_tau(mixed, mixed_y, e_explain=5.0, e_skip=1.0)
    leaked_quantiles = [float(np.quantile(mixed, q / 100.0)) for q in range(5, 100, 5)]
    assert any(np.isclose(leaked["tau"], q) for q in leaked_quantiles)
    assert not any(np.isclose(selected["tau"], q) and np.isclose(leaked["tau"], q) for q in (0.0,))


def test_energy_parts_match_compute_energy_j_at_level_0_and_level_3():
    profile = PROFILES["stm32l476"]
    for level in (ExecutionLevel.L0, ExecutionLevel.L3):
        parts = energy_parts(level, FISH_MACS_BY_LEVEL, profile)
        assert abs(parts["e_a_j"] + parts["e_g_j"] + parts["e_p_j"] - parts["total_j"]) < 1e-12
        assert abs(parts["total_j"] - compute_energy_j(level, FISH_MACS_BY_LEVEL, profile)) < 1e-12
    level3 = energy_parts(ExecutionLevel.L3, FISH_MACS_BY_LEVEL, profile)
    level1 = energy_parts(ExecutionLevel.L1, FISH_MACS_BY_LEVEL, profile)
    assert level3["includes_level2_budget"]
    assert level3["e_p_j"] > level3["e_level3_only_j"]
    assert level1["e_p_j"] == 0.0
    assert abs(energy_saving_pct(4.0, 6.0) - 100.0 * (6.0 - 4.0) / 6.0) < 1e-12


def test_fixed_concentration_makes_epistemic_uncertainty_constant():
    assert abs(epistemic_from_concentration(2, 20.0) - 2.0 / 22.0) < 1e-12
    for probability in (np.array([0.9, 0.1]), np.array([0.5, 0.5]), np.array([0.2, 0.8])):
        _mean, _aleatoric, epistemic = dirichlet_lift(probability, METHOD_CONCENTRATION)
        assert abs(epistemic - 2.0 / 22.0) < 1e-12
    trusts = _trusts()
    values = [t.u_epistemic for t in trusts]
    assert max(values) - min(values) < 1e-12
    # Confident on either class lowers aleatoric uncertainty. The inverted score rises.
    peaked = dirichlet_lift(np.array([0.95, 0.05]), 20.0)
    flat = dirichlet_lift(np.array([0.5, 0.5]), 20.0)
    assert peaked[1] < flat[1]


def test_relift_at_the_method_concentration_preserves_the_score():
    trusts = _trusts()
    lifted = relifted_trusts(trusts, METHOD_CONCENTRATION, negative_index=1)
    assert np.allclose(score_array(lifted, "trust"), score_array(trusts, "trust"))
    recovered = logistic_proba_from_alpha(trusts[0].alpha, METHOD_CONCENTRATION)
    mean = dirichlet_mean(1.0 + METHOD_CONCENTRATION * recovered)
    assert np.allclose(mean, trusts[0].p)
    assert abs(float(epistemic_uncertainty(trusts[0].alpha)) - trusts[0].u_epistemic) < 1e-12
    assert abs(float(aleatoric_uncertainty(trusts[0].p)) - trusts[0].u_aleatoric) < 1e-12


def test_low_score_stops_at_level_1_and_abstention_is_a_different_action():
    scores = np.array([0.2, 0.4, 0.9, 0.95])
    y = np.array([0, 1, 1, 0])
    pred = ["other", "call", "call", "other"]
    gold = ["other", "call", "call", "other"]
    labels = ["call", "other"]
    profile = PROFILES["stm32l476"]
    skip = energy_parts(ExecutionLevel.L1, FISH_MACS_BY_LEVEL, profile)
    spend = energy_parts(ExecutionLevel.L3, FISH_MACS_BY_LEVEL, profile)
    row = apply_explanation_policy(
        scores, y, pred, gold, labels, tau=0.5, parts_skip=skip, parts_explain=spend, always_l3_j=float(spend["total_j"])
    )
    assert row["action"] == "level3_if_score_ge_tau"
    assert row["abstention_rate"] == 0.0
    assert row["additional_processing_rate"] == 0.5
    assert row["macro_f1"] == 1.0
    refused = abstention_metrics(pred, gold, scores >= 0.85, labels)
    assert refused["abstention_rate"] == 0.5
    assert refused["error_rate_among_accepted"] == 0.0
    assert refused["macro_f1_accepted"] == 1.0
    assert refused["macro_f1_refusals_as_errors"] < refused["macro_f1_accepted"]


def test_uncertainty_score_is_high_when_the_prediction_is_peaked():
    trusts = _trusts()
    # The inverted sum is smaller when aleatoric uncertainty is larger.
    ordered = sorted(trusts, key=lambda t: t.u_aleatoric)
    assert uncertainty_explain_score(ordered[0]) > uncertainty_explain_score(ordered[-1])
