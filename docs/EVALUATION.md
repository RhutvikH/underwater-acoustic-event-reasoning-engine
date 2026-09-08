# Evaluation conditions

This is the contract an **evaluator** (faculty, reviewer, or you at 2 a.m.
before a deadline) uses. The code that enforces it is
`src/uaere/eval/evaluator.py`. Changing a number in that file **and not
here** is a protocol bug.

## Command

```bash
source .venv/bin/activate
uaere evaluate --out artifacts/eval
```

Optional overrides (still recorded in `verdict.json`):

```bash
uaere evaluate --out artifacts/eval --n-windows 160 --seed 0 --steps 200
```

Exit code **0** = all checks passed. **1** = at least one failed; read
`artifacts/eval/verdict.md`.

`uaere eval --suite paper` still writes the raw tables without grading.
Prefer `uaere evaluate` when you need a yes/no.

## Protocol version

`EvaluationConditions.name = "paper-trl4-v1"`

| Condition | Value | Why |
|-----------|-------|-----|
| Working sample rate | 16 kHz | edge cost; DeepShip 32 kHz is resampled |
| Window | 1.0 s | one inference quantum |
| Scenarios | `busy_strait` + extra `high_sea_state` | storms must be able to fool energy |
| Split | time-aware, val 15%, test 20% | later windows = test (UATR convention) |
| Seeds | `(0,)` in CI; camera-ready `{0,1,2}` | no silent RNG |
| Train steps | 200 (logistic `max_iter`) | evidential lift, not a giant net |
| Detection score | **`wake_confidence` (`C_wake`)** | never band energy, never raw `T(e)` for ROC |
| Baseline | Youden energy threshold on **train only** | the thing we claim to replace |
| Gating baseline | always-on L3 (`tau=0`) | Objective 3 |
| Device under test for joules | `stm32l476` unless stated | MCU-class hydrophone |
| Security | tamper fails boot; `gd32vf103` refused for auth L2 | non-negotiable 5 |
| TinyCNN INT8 | ≤ 250 KB | actually 546 B today |
| α, bootstrap | 0.05, 1000 | Holm–Bonferroni on {AUC, ECE, energy} when 3 seeds are run |

## Pass / fail gates (the evaluator)

| Check | Pass if |
|-------|---------|
| `trust_auc_floor` | `C_wake` ROC-AUC ≥ **0.70** |
| `trust_beats_energy` | trust AUC **>** energy-threshold AUC |
| `trust_ece_ceiling` | ECE ≤ **0.45** |
| `ontology_hit` | top KG cause matches gold on ≥ **50%** of true events |
| `gated_energy` | knee J/window **<** always-on L3 |
| `device_profiles` | ≥ **5** hardware rows |
| `tamper_caught` | mutated firmware raises `TamperError` |
| `unattestable_refused` | authenticated L2 on `gd32vf103` raises `PermissionError` |
| `tiny_int8_budget` | CNN INT8 bytes ≤ 250_000 |
| `proposition1` | analytic `E(pi) < E(always-on L3)` for `p3 < 1` |

These floors are **lab honesty**, not the camera-ready point estimates
(we have seen trust AUC ≈ 0.92 on seed 0). If a PR drops below a floor,
it does not merge.

## What we are **not** allowed to do in an evaluation

1. Tune the energy threshold on the test split.
2. Report energy as the AHAIF detection score.
3. Quote twin-synthetic AUC as DeepShip / ocean performance.
4. Average away a failed security check.
5. Change `min_trust_auc` to make a broken head pass.

## Outputs

```
artifacts/eval/
  verdict.json      machine-readable checks
  verdict.md        table for the report appendix
  suite/suite.json  full paper suite
  suite/suite.md
```

## Labels (so ROC is not fiction)

- `event_present = True` only for tug, cargo, tanker, passenger.
- `geophony` and `artifact` map to Dirichlet **reject**.
- That is why a loud storm can have **low** `C_wake` and
  **high** energy — the baseline’s failure mode.

## Camera-ready extras (not yet a hard gate)

- Repeat `uaere evaluate --seed 0` then `--seed 1` then `--seed 2`.
- McNemar on paired correctness, Wilcoxon on ECE/energy, Holm on the family
  (`src/uaere/eval/stats.py`).
- DeepShip SHA-256 in `paper/experiment_protocol.md` when the bits exist.

## Faculty mapping

| Capstone objective | Check |
|--------------------|--------|
| O1 calibrated event-level trust vs energy | `trust_beats_energy`, `trust_ece_ceiling` |
| O2 KG chains | `ontology_hit` |
| O3 gated energy | `gated_energy`, `proposition1` |
| O4 TRL-4 e2e | whole verdict + hardware + security |
