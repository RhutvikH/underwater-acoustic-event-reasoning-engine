from __future__ import annotations

from uaere.eval.evaluator import EvaluationConditions, Evaluator


def test_conditions_are_frozen_protocol():
    c = EvaluationConditions()
    assert c.detection_score == "wake_confidence"
    assert c.min_trust_auc >= 0.7
    assert "energy" not in c.detection_score


def test_evaluator_writes_verdict(tmp_path):
    c = EvaluationConditions(n_windows=48, train_steps=80, seeds=(0,))
    v = Evaluator(c).run(out_dir=tmp_path)
    assert (tmp_path / "verdict.json").exists()
    assert (tmp_path / "verdict.md").exists()
    names = {x["name"] for x in v.checks}
    assert "trust_beats_energy" in names
    assert "tiny_int8_budget" in names
    # TinyCNN budget and security demos must hold even on a short run.
    by = {x["name"]: x["pass"] for x in v.checks}
    assert by["tiny_int8_budget"]
    assert by["tamper_caught"]
    assert by["unattestable_refused"]
