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
Prefer `uaere evaluate` when you need a yes/no on the **synthetic** track.

`uaere eval --suite review` is the real-clip table used by the manuscript. It is not graded by the floors below. `uaere eval --suite fish` is the older single-seed writer. A finished honest run of either exits 0. A missing WAV root exits 2 and does not substitute the twin. Do not lower `min_trust_auc` to make the synthetic track look better. The preserved `fish_suite.json` also holds `l0_transfer`: how often the twin surprise sigmoid (center 8, cutoff 0.15) would have slept. That cutoff is not refit, and a species checkpoint does not use it as the label. To regenerate only that audit, without refitting the logistic:

```
.venv/bin/python -c "from uaere.eval.l0_transfer import write_l0_audit; write_l0_audit()"
```

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
6. Quote the test-tuned knee in `artifacts/paper/fish_suite.json` as the fish energy result. That file is preserved. The fish table in the manuscript comes from `artifacts/paper/review_upgrade/`.
7. Call `T(e)` a calibrated probability that the label is correct. Detection ROC uses `C_wake`. The explanation gate uses a validation-chosen threshold on a score.

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

## Fish track (real clips, separate from the floors above)

The manuscript table is the review suite. It runs seeds 0, 1, and 2. `--seed` is ignored. `--steps` is floored at 2000. A default `--out artifacts/paper` is redirected to `artifacts/paper/review_upgrade`. The command does not write a checkpoint and does not overwrite `fish_suite.json`.

```bash
uaere eval --suite review --root data/toadfish --out artifacts/paper/review_upgrade --max-per-class 3000 --min-files 40 --steps 2000
```

Writes `artifacts/paper/review_upgrade/review_upgrade.json`, `review_upgrade.md`, and `seed_0.json`, `seed_1.json`, `seed_2.json`.

`uaere eval --suite fish` is the older single-seed writer. It still fits NSGA-II on the test scores and can overwrite `artifacts/paper/fish_suite.json` and `artifacts/models/fish_head.joblib`. Do not re-run it to refresh the paper. The knee already stored in `fish_suite.json` was selected on the test split. Leave that file as the preserved seed-0 classification record. Do not quote its gated joules as the review result.

| Item | Rule |
|------|------|
| Data | ToadFishFinder `bwhistle/` and `other/` (DOI 10.5061/dryad.gtht76hr9). Pre-segmented candidates, not event discovery in a continuous stream. Not FishSounds. |
| Split | `recording_locked_split`, one seed at a time, 60/20/20 by file mass, cap 3000 files per class. A deployment stays on one side. Report the leak count. |
| Classifier | Class-balanced logistic, `class_weight="balanced"`, lbfgs, 102 features (`mel_stats+opsanus_band`). The label is the argmax. The gate does not change it. |
| Score | Detection ROC uses `C_wake = 1 - P(other)`. The explanation gate uses a separate score. `T(e)` is not that ROC score and is not a calibrated probability. |
| Threshold | One `tau` per score, chosen on validation percentiles 5, 10, …, 95 by a Chebyshev knee. Frozen, then scored once on test. The same rule for confidence, inverted uncertainty, confidence–uncertainty, and trust. |
| Action | Level-3 cost if `score >= tau`, otherwise Level 1. The label is still emitted. Abstention is a separate baseline: report its rate, the error among accepted labels, and full-set macro-F1 with refusals counted as errors. |
| Energy | `stm32l476` datasheet model. `E = E_a + E_g + E_p` with `E_g = 0` and `E_a` equal to the Level-1 cost. Savings are estimates. They are not measured power. |
| Explanation | The graph is outside `T(e)`. Cause-match repeats the single toadfish `caused_by` edge. `band_check` does not read the label. If the graph does not move the label or the gate, say so. |
| Loss | If a gate does not beat confidence after the threshold is retuned, write that down. Do not retune bands or weights on the test files. |

Three-seed test macro-F1 is 0.844 ± 0.004 (seeds 0.839, 0.848, 0.844). The 95% t interval is [0.833, 0.854]. Wake ROC-AUC is 0.908 ± 0.009. Wake ECE is 0.043 ± 0.017. Seed 0 reproduces the preserved suite: macro-F1 0.839, wake AUC 0.898, ECE 0.059, Brier 0.120, and the logistic stopped at iteration 64. Energy-threshold AUC 0.656 is that preserved seed-0 detector only. Seeds 1 and 2 do not have an energy-threshold AUC. Validation chose the three band features on seed 0 (macro-F1 0.817 with them, 0.799 without). Cause-match is 1.00 because the graph repeats the predicted class. Band-check ROC-AUC is 0.630 ± 0.053. The Mel-only test macro-F1 0.827 remains `artifacts/paper/fish_suite_melstats.json`.

On these files the trust gate, the confidence gate, the confidence–uncertainty gate, and the ablations that zero the environment weight or the uncertainty weight select the same test windows. Mean modelled energy is 5.779 mJ per window. The saving against always-Level-3 is 10.4% ± 0.7%. The Level-3 rate is 0.355 ± 0.041. Explanation recall is 0.662 and the explanation false-alarm rate is 0.039. The uncertainty-only gate and the entropy gate match each other and are worse: mean saving 8.5%, explanation false-alarm rate about 0.45. Always-Level-1 saves 16.2% and explains nothing. The graph-only joules, Level 3 minus Level 2, are 0.0052 mJ. About 99.5% of the additional-path cost is the Level-2 budget, which the label does not execute.

Abstention on the trust score is not the 0.844 classifier. Seed rates are 0.591, 0.627, and 0.488. Full-set macro-F1 with refusals counted as errors is 0.409, 0.403, and 0.423. On seed 0 that scoring gives `other` recall 0.

The synthetic ontology hit-rate of 1.00 and the twin’s factor-of-about-32 energy ratio are the appendix in `artifacts/paper/suite.json`. They are not this table. White-noise rows in the review JSON are derived perturbations of the seed-0 test waveforms. They are not new field recordings. The threshold used on those rows is the clean validation threshold. The head is not refit.

## Species track (call library, separate from both tables above)

```bash
uaere eval --suite species --root data/species --out artifacts/paper --seed 0
```

Writes `artifacts/paper/species_suite.json`, `species_suite.md`, and `artifacts/models/species_head.joblib`. It does not rewrite `fish_suite.json` or `fish_head.joblib`.

| Item | Rule |
|------|------|
| Data | `data/species/<species>/*.wav`, or the Watkins full-cuts archive unpacked into that tree. Not FishSounds. |
| What a hit means | The cut was already isolated and identified. This is not a soundscape detector. |
| Split | Per species, recording-locked, seed 0, 60/20/20 by file mass. A tape id never crosses a side. Cap 500 files, whole recordings. |
| Choice | Validation macro-F1. Test is scored once and does not drop a species or change `alpha`. |
| Headline | Test macro-F1. Accuracy is also stored. A majority-class baseline is the dumb comparison. There is no energy-threshold baseline, because this set has no ambient class. |
| Wake | Maximum class probability. Not `REJECT_INDEX`. |
| Explanation | One name-only cause per trained label, `cause.taxon.<label>`. No invented hertz. The sentence does not say the cues match. |
| Bar | Macro-F1 ≥ 0.70 and ≥ 15 test files for every reported species. Registered before the score. A miss still exits 0. Missing audio exits 2. |
| Seeds | Seed 0 is the run. Seeds 1 and 2 are named in the JSON as not run. |

Seed 0 result in `artifacts/paper/species_suite.json`: test macro-F1 0.281, accuracy 0.285, majority-class accuracy 0.062, ECE of the maximum probability 0.529. Validation chose `species_mlp_mel` at `alpha` 1e-4 (validation macro-F1 0.292) over the image logistic (0.291), the MLP at `alpha` 1e-3 (0.291), and the summary logistic (0.284). Logistic fits used `max_iter` 2000 and converged (summary at iteration 497, image at 494). The MLP used `max_iter` 400 and early stopping, and both alphas stopped at iteration 51. Every reported species has at least 15 test files. The macro-F1 bar of 0.70 was not met. Seeds 1 and 2 were not run. `artifacts/paper/species_suite_capped.json` keeps the earlier 250-iteration result (test macro-F1 0.275). Do not paste the toadfish macro-F1 into this table.

The missed bar is the result of this protocol. Striped dolphin and walrus are one test tape each (`85012`, 264 cuts, 222 named sperm whale; `83001`, 150 cuts, none correct). Common dolphin (`58013`, 105 cuts) and short-finned Pacific pilot whale (`91005`, 131 cuts) are also one test tape. The mean of the other fourteen F1 scores is 0.321, so deleting the two zero rows is not a new result. A second feature vector, fit on this split and scored on validation only, stayed at 0.298 (log-mel image) and 0.308 (spectral shape plus click rate). The two-class toadfish table is a separate task: three-seed macro-F1 0.844 ± 0.004 in `artifacts/paper/review_upgrade/`. Do not place the Watkins 0.281 beside that mean as the same score. Energy-threshold AUC 0.656 is the preserved seed-0 fish detector only. This Watkins table is the 16-way stress test. A random clip split of the same archive is a different, leakier task. BEANS scores a random split of the small Watkins best-of subset (31 species). Date-and-location splits of the same archive are a harder task (Aguiar et al., Journal of Marine Science and Engineering, 2021). Do not quote the random-split number as this suite. Seeds 1 and 2 of Watkins were not run.

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
