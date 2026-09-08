"""Named evaluator with frozen pass/fail conditions.

`uaere evaluate` is the faculty-facing entry. It runs the paper suite, then
grades the report against EvaluationConditions. Changing a threshold here
is a protocol change: update docs/EVALUATION.md in the same commit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from uaere.eval.suite import run_paper_suite


@dataclass(frozen=True)
class EvaluationConditions:
    """Pass/fail gates for a TRL-4 lab run. Camera-ready uses seeds (0,1,2)."""

    name: str = "paper-trl4-v1"
    n_windows: int = 160
    train_steps: int = 200
    seeds: tuple[int, ...] = (0,)
    val_frac: float = 0.15
    test_frac: float = 0.20
    scenarios: tuple[str, ...] = ("busy_strait", "high_sea_state")
    detection_score: str = "wake_confidence"  # never band energy
    min_trust_auc: float = 0.70
    require_trust_beats_energy: bool = True
    max_trust_ece: float = 0.45
    min_ontology_hit: float = 0.50
    require_gated_less_energy: bool = True
    min_device_profiles: int = 5
    require_tamper_caught: bool = True
    require_unattestable_refused: bool = True
    max_cnn_int8_bytes: int = 250_000
    alpha: float = 0.05
    n_bootstrap: int = 1000
    notes: str = (
        "Twin-synthetic TRL-4. Do not quote as ocean TRL-7. "
        "Detection ROC uses C_wake, not T(e), not energy."
    )


@dataclass
class Verdict:
    passed: bool
    conditions: dict[str, Any]
    checks: list[dict[str, Any]]
    report_path: str
    failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Evaluator:
    """Runs the paper suite and grades it. One object, one protocol version."""

    def __init__(self, conditions: EvaluationConditions | None = None) -> None:
        self.conditions = conditions or EvaluationConditions()

    def run(self, out_dir: str | Path = "artifacts/eval") -> Verdict:
        c = self.conditions
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Primary seed is conditions.seeds[0]; extra seeds are documented for
        # camera-ready, not averaged in this TRL-4 evaluator.
        seed = int(c.seeds[0])
        report = run_paper_suite(
            out_dir=out / "suite",
            n_windows=c.n_windows,
            seed=seed,
            train_steps=c.train_steps,
        )
        checks: list[dict[str, Any]] = []

        def add(name: str, ok: bool, observed: Any, rule: str) -> None:
            checks.append({"name": name, "pass": bool(ok), "observed": observed, "rule": rule})

        w = report["wake"]
        add(
            "trust_auc_floor",
            w["trust_auc"] >= c.min_trust_auc,
            w["trust_auc"],
            f"C_wake ROC-AUC >= {c.min_trust_auc}",
        )
        add(
            "trust_beats_energy",
            (w["trust_auc"] > w["energy_auc"]) if c.require_trust_beats_energy else True,
            {"trust": w["trust_auc"], "energy": w["energy_auc"]},
            "trust AUC > Youden energy-threshold AUC",
        )
        add(
            "trust_ece_ceiling",
            w["trust_ece"] <= c.max_trust_ece,
            w["trust_ece"],
            f"ECE <= {c.max_trust_ece}",
        )
        add(
            "ontology_hit",
            report["explanations"]["ontology_hit_rate"] >= c.min_ontology_hit,
            report["explanations"]["ontology_hit_rate"],
            f"top KG cause matches gold on >= {c.min_ontology_hit} of true events",
        )
        add(
            "gated_energy",
            report["gating"]["gated_uses_less_energy"] if c.require_gated_less_energy else True,
            {
                "gated": report["gating"]["gated_energy_j"],
                "always": report["gating"]["always_l3_energy_j"],
            },
            "knee energy < always-on L3 energy (matched detector)",
        )
        add(
            "device_profiles",
            len(report["hardware"]) >= c.min_device_profiles,
            len(report["hardware"]),
            f">= {c.min_device_profiles} DeviceProfile rows",
        )
        add(
            "tamper_caught",
            report["security"]["tamper_caught"] if c.require_tamper_caught else True,
            report["security"]["tamper_caught"],
            "mutated firmware fails secure_boot",
        )
        add(
            "unattestable_refused",
            report["security"]["gd32_refused_authenticated_l2"]
            if c.require_unattestable_refused
            else True,
            report["security"]["gd32_refused_authenticated_l2"],
            "gd32vf103 refused for authenticated L2",
        )
        add(
            "tiny_int8_budget",
            report["complexity"]["cnn_int8_bytes"] <= c.max_cnn_int8_bytes,
            report["complexity"]["cnn_int8_bytes"],
            f"TinyCNN INT8 bytes <= {c.max_cnn_int8_bytes}",
        )
        add(
            "proposition1",
            report["gating"]["proposition1_holds"],
            True,
            "E(gated) < E(always-on) for p3<1, c_k>0",
        )

        failed = [x["name"] for x in checks if not x["pass"]]
        verdict = Verdict(
            passed=len(failed) == 0,
            conditions=asdict(c),
            checks=checks,
            report_path=str(out / "suite" / "suite.json"),
            failed=failed,
        )
        (out / "verdict.json").write_text(
            json.dumps(verdict.to_dict(), indent=2, default=float), encoding="utf-8"
        )
        (out / "verdict.md").write_text(_verdict_md(verdict), encoding="utf-8")
        return verdict


def _verdict_md(v: Verdict) -> str:
    lines = [
        f"# Evaluator verdict — {'PASS' if v.passed else 'FAIL'}",
        "",
        f"Protocol: `{v.conditions['name']}`",
        f"Suite JSON: `{v.report_path}`",
        "",
        "| Check | Pass | Observed | Rule |",
        "|-------|------|----------|------|",
    ]
    for c in v.checks:
        obs = c["observed"]
        if isinstance(obs, float):
            obs_s = f"{obs:.4f}"
        else:
            obs_s = json.dumps(obs, default=float)
        lines.append(f"| {c['name']} | {c['pass']} | `{obs_s}` | {c['rule']} |")
    if v.failed:
        lines += ["", "## Failed", ""] + [f"- {n}" for n in v.failed]
    lines.append("")
    return "\n".join(lines)
