# Experiment protocol (lab notebook)

Two tracks. Do not mix their numbers in one table.

## Commands

Synthetic gating appendix:

```bash
PYTHONPATH=src python -m uaere eval --suite paper --out artifacts/paper --n-windows 240 --seed 0
```

Real fish clips (the manuscript table). Seeds 0, 1, and 2. `--seed` is ignored. `--steps` is floored at 2000. The default output directory is redirected to `artifacts/paper/review_upgrade`. The command does not save a checkpoint.

```bash
PYTHONPATH=src python -m uaere eval --suite review --root data/toadfish --out artifacts/paper/review_upgrade --max-per-class 3000 --min-files 40 --steps 2000
```

Each seed fits on its own training split, chooses every gate threshold on its own validation split, and scores the test split once. Report the mean, the sample standard deviation (ddof = 1), and a 95% t interval. A paired t-test across three seeds is descriptive. The fish command exits 0 after a completed run. It exits 2 if the WAVs are missing. It does not render the twin instead.

The older writer is still available and must not be used to refresh the paper. It searches NSGA-II on the **test** scores and writes `artifacts/paper/fish_suite.json` plus `artifacts/models/fish_head.joblib`:

```bash
PYTHONPATH=src python -m uaere eval --suite fish --root data/toadfish --out artifacts/paper --max-per-class 3000 --min-files 40 --seed 0 --steps 2000
```

The knee already in `fish_suite.json` was selected on the test split. Keep the file. Do not quote its gated joules as the review result. Seed 0 of the review command reproduced that file’s macro-F1.

## Data

| Corpus | Role | Access |
|--------|------|--------|
| ToadFishFinder v4, Pamlico Sound | species label: `opsanus_tau` vs `other` | https://doi.org/10.5061/dryad.gtht76hr9 CC0, `wavclips_TFv4.zip`. Not in git. |
| Twin-synthetic `busy_strait` + `high_sea_state` | gating appendix only | generated, seed-locked |
| DeepShip (Irfan et al. 2021) | not in the numbers below | email authors |
| ShipsEar | not in the numbers below | public release |
| FishSounds | not a training set here | index of fish sounds |

Fish split: `recording_locked_split`, seed 0, 60/20/20 by file mass. The cap is 3000 files per class, and a deployment is kept whole (`recording_locked_split`). Names `bw_<station>_<id>_<time>.wav` and `ot_<station>_<id>_<time>.wav` (and a `t_` prefix) share `<station>_<id>`. Both classes of that id stay on one side. That is not a second estuary. A name without that pattern keeps an explicit `_clipN` / `_segN` / `_winN` / `_partN` together; otherwise one file is one group.

Fish window: centre 1.0 s, resampled 24 kHz → 16 kHz. The published ResNet-50 used the centre 850 ms (samples 6000–26400 of the 1.35 s snippet). 32-bit integer PCM is divided by 2^31. IEEE-float samples are kept as stored. The training copy is not peak-normalised.

L0 on the fish track: surprise S is the Knudsen residual already used on the twin, with the twin admit map (center 8, scale 1.5, sleep below 0.15). That map is not refit on the fish files. A species checkpoint records S and does not replace its label when admit is below 0.15. The counts are `artifacts/paper/fish_l0_surprise.json`, from `uaere.eval.l0_transfer.write_l0_audit` on the same deployment-locked split. That command does not refit the logistic.

Imputed environment at train and at test: 12 °C, 34 PSU, 40 m, sea state 2. The release has no CTD. Sea state 2 is a constant, not a measurement.

Fish features: 99 log-Mel statistics plus three waveform numbers inside the band stored on `cause.taxon.opsanus_tau` (80 and 500 Hz). Seed-0 validation macro-F1 chose those three numbers (0.799 without, 0.817 with). An MLP on the same vector scored 0.819 and was not used. The test split was scored once, after that choice. The seed-0 logistic stopped at iteration 64 of `max_iter` 2000. The solver uses `class_weight="balanced"`. The Mel-only test table remains `artifacts/paper/fish_suite_melstats.json` (macro-F1 0.827).

Fish knowledge graph: hand-written. `event.toadfish_boatwhistle` is caused by `cause.taxon.opsanus_tau`. Confusable events are the sciaenid chorus, wind-wave geophony, and a tug passage, using the bands already stored on those causes. The graph is not an input of `T(e)`. Cause-match repeats the predicted class and is 1.00 on every seed. `band_check` does not read the label. Mean ROC-AUC over the three test splits is 0.630 ± 0.053 (seed 0 was 0.587, and the boatwhistle band led on 0.196 of the true calls in that split).

Fish gate, implemented in `src/uaere/eval/gate_compare.py`. One threshold per score. Level-3 cost if the score is at least `tau`, otherwise Level 1. The logistic label is emitted either way, so macro-F1 does not move. `tau` is a validation percentile in {5, 10, …, 95}, chosen by a Chebyshev knee of energy, explanation false alarms, and one minus explanation recall. The same rule is used for wake confidence, inverted uncertainty `1 - (u_a + u_e)`, the confidence–uncertainty logit, and full `T(e)`. Weights are the hand-set values wake 2.2, health 0.8, environment 1.1, uncertainty 1.4, bias −0.6. They do not sum to 1. Dirichlet concentration is the fixed value 20, so epistemic uncertainty is 2/22 on every two-class window. Health is 1 and environment consistency is constant on this corpus, because the Mel contrast is read after instance-norm plus a FiLM bias that depends only on the imputed environment. A threshold on `T(e)` therefore ranks the same windows as a threshold on `C_wake` once `tau` is retuned. The review run shows that equality. An uncertainty-only gate does not: it explains confident negatives.

Energy identity: `E = E_a + E_g + E_p` on the `stm32l476` profile. `E_a` is the Level-1 cost, 0.005408 J. `E_g` is 0. `E_p` when explaining is `E(Level 3) - E(Level 1)` = 0.0010452 J, of which the graph-only piece `E(Level 3) - E(Level 2)` is 5.2e-6 J. The rest of `E_p` is the Level-2 MAC budget. Level 2 does not supply the label. Savings are datasheet estimates. They are not measured power. The synthetic factor-of-about-32 ratio in `suite.json` is a different policy and is not this saving.

Measured review result, copied from `artifacts/paper/review_upgrade/review_upgrade.json`. Macro-F1 0.844 ± 0.004, 95% t interval [0.833, 0.854]. Wake ROC-AUC 0.908 ± 0.009. Wake ECE 0.043 ± 0.017. The shared trust/confidence policy uses 5.779 mJ per window, saves 10.4% ± 0.7% against always-Level-3, explains 0.355 ± 0.041 of windows, recalls 0.662 of boatwhistles at Level 3, and false-alarms on 0.039 of negatives. Always-Level-1 saves 16.2% and explains nothing. The uncertainty and entropy gates save about 8.5% with an explanation false-alarm rate near 0.45. Abstention is a separate baseline and lowers full-set macro-F1 into the 0.40s. White noise at 20, 10, 5, and 0 dB was added to the frozen seed-0 test waveforms only. Those rows are derived. Temperature scaling and Cohen’s kappa were not run. Seeds 1 and 2 have no energy-threshold AUC. The preserved seed-0 energy-threshold AUC is 0.656.

Ship-track split (appendix only): time-aware, later windows are test (`time_aware_split`).

## Species call library

A third track. Do not mix it with the toadfish table or the synthetic appendix.

```bash
PYTHONPATH=src python -m uaere eval --suite species --root data/species --out artifacts/paper --seed 0
```

Corpus: Watkins Marine Mammal Sound Database, Woods Hole Oceanographic Institution, academic use, not commercial. Unpacked to `data/species/<species>/*.wav`. Not FishSounds. Not a continuous estuary. The eval refuses the run when tape ids are missing (`groups.csv` with columns `file,group_id` is the override). Custom `uaere train --dataset species` may still fit a tree of ordinary files and must say the split is not recording-held-out.

Eligibility, before any score: ≥ 10 species, ≥ 40 files, ≥ 8 independent tapes, cap 500 files by whole recordings, seed 0. Band: 16 kHz and 7 kHz, then one 32 kHz / 14 kHz fallback for this head only. Porpoise names are excluded. The ship twin and the toadfish head stay at 16 kHz.

Split: `label_stratified_recording_split`, 60/20/20 by file mass inside each species. A shared tape id stays on one side. Validation chooses the model. Test is scored once.

Models, same split: class-balanced logistic on 64-band summaries; class-balanced logistic on a 64×32 log-mel; sklearn MLP (256) at `alpha` 1e-4 and 1e-3 with balanced sample weights (sklearn 1.9 has no MLP `class_weight`). Logistic `max_iter` is 2000 (`lbfgs`). MLP `max_iter` is 400, with early stopping on an inner tenth of the training rows. Species train and species eval use at least 2000 logistic iterations. An explicit `--steps` above 2000 is kept. Winner = higher validation macro-F1. Tie: MLP, then the smaller alpha. No PyTorch. Checkpoint `artifacts/models/species_head.joblib`. The fish checkpoint is not refit.

Headline: test macro-F1. Also store per-class precision, recall, F1, support, confusion, majority-class baseline, every candidate's test macro-F1 (report only), and ECE of the maximum class probability. Demonstration bar, fixed before reading the test score: macro-F1 ≥ 0.70 and ≥ 15 test files per reported species. Write the real numbers if the bar is missed. Do not drop a species after seeing the test. Seeds 1 and 2 are not this run. Exit 0 after a completed species run. Exit 2 if the audio or the floors are missing.

Knowledge graph: one name-only cause `cause.taxon.<label>` per trained label. No invented frequency. The sentence names the classifier's species. It is not a second detector.

Seed 0 was run. The current `artifacts/paper/species_suite.json` and `species_suite.md` are the converged fit. Test macro-F1 0.281, accuracy 0.285, majority accuracy 0.062 (`humpback_whale`), ECE 0.529. Chosen on validation: `species_mlp_mel` at `alpha` 1e-4, validation macro-F1 0.292, 51 iterations with early stopping. The image logistic converged at iteration 494 of 2000 (validation macro-F1 0.291). The summary logistic converged at iteration 497 (validation macro-F1 0.284). Support was at least 15 files for every species. Fin whale test F1 is 0.767 (support 97) and killer whale is 0.692 (support 98). Striped dolphin and walrus are 0.000 (supports 264 and 150). The macro-F1 bar of 0.70 was not met. Seeds 1 and 2 were not run. Species were not dropped after the test score. `artifacts/paper/species_suite_capped.json` keeps the earlier stop at 250 logistic iterations (test macro-F1 0.275, image logistic).

How to read 0.281 in the manuscript. The two-class toadfish table is a different task (three-seed macro-F1 0.844 ± 0.004, wake ROC-AUC 0.908 ± 0.009). Energy-threshold AUC 0.656 is the preserved seed-0 fish detector only. Watkins is the stress test. Do not place 0.281 beside the toadfish mean as the same score. The cuts are already 1–2 s, so a louder crop is the same window. Striped dolphin, walrus, common dolphin, and short-finned Pacific pilot whale each have one test tape (`85012` 264 cuts, `83001` 150, `58013` 105, `91005` 131). Striped dolphin accounts for 222 sperm-whale predictions. Removing the two zero F1 scores from the average leaves 0.321. Fin whale recall is 97/97, and 98% of those files are sampled below 8 kHz (median 600 Hz), so that row includes the empty upper spectrum. Killer whale (median native rate 21.9 kHz, 11 test tapes, F1 0.692) is the in-band species with tape support. Striped dolphin’s inventory power above 7 kHz is 0.085. A validation-only feature check stayed at macro-F1 0.30, so this protocol is not refit. A random clip split is not this result. BEANS scores a random split of the Watkins best-of cuts. Date-and-location splits of the same archive are a harder task (Aguiar et al., Journal of Marine Science and Engineering, 2021).

## Baselines

Synthetic appendix (`--suite paper`):

- Fixed STFT-band energy threshold, θ from Youden on train (`baseline_energy_threshold.py` only).
- Always-on L3.
- The NSGA-II staircase, searched on the twin replay.

Fish review (`--suite review`), same splits and the same datasheet costs:

- Always Level 1 (never explain).
- Always Level 3.
- One validation-chosen threshold on wake confidence.
- One validation-chosen threshold on inverted uncertainty.
- One validation-chosen threshold on the confidence–uncertainty logit.
- One validation-chosen threshold on `T(e)`.
- Ablations that zero the environment weight or the uncertainty weight, and a run that computes `T(e)` and then never pays for Level 3.
- Entropy of the Dirichlet mean, as a second uncertainty score.
- Abstention, reported separately from the explanation gate.
- An NSGA-II staircase searched on validation only, reported separately because it can stop at Level 0.

The threshold is not chosen on the test set. Bands are not retuned after the test score. Species are not dropped.

## Hardware

Software-defined profiles in `src/uaere/hardware/profiles.py`. No silicon. Optional future: Renode Cortex-M cycle check of L0 within 2×.

## Security tests

- Tampered firmware fails `secure_boot`.
- Honest HMAC verifies.
- `gd32vf103` is refused for authenticated L2.

## Embargo

Do not put `artifacts/` or trained weights on a public remote until the provisional is filed.
