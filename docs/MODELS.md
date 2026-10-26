# Models (pluggable backbones)

Any backbone that implements `fit(X, y)` and `evidence(x) -> alpha`
can sit under the Dirichlet trust engine.

| Name | Class | What it is |
|------|-------|------------|
| `logistic_mel` (default) | `LogisticMelBackbone` | six-way ship head: class-balanced logistic on Mel statistics, lifted to Dirichlet `alpha = 1 + tau p` |
| `fish_logistic_mel` | `fit_fish_head` | same lift, labels taken from the fish folders (`opsanus_tau`, `other`, …). Features are the 99 Mel statistics plus three waveform numbers from the band stored on `cause.taxon.opsanus_tau`. Saved with joblib. This name is the event class. |
| `species_logistic_summary` | `select_species_head` | class-balanced logistic on 64-band mean, standard deviation, and max. Baseline only. |
| `species_logistic_mel` | `select_species_head` | class-balanced logistic on a 64×32 log-mel image. Baseline. |
| `species_mlp_mel` | `select_species_head` | sklearn MLP, one hidden layer of 256, on that same image. `alpha` is 1e-4 or 1e-3. |

Register a new one in `src/uaere/models/registry.py`:

```python
BACKBONES["my_net"] = MyBackbone
```

`MyBackbone` must provide `.name`, `.fit` / `.fit_records`, `.evidence`.

TinyCNN remains a MAC-count / L2 head; it is not the default detector.

## Commands

```bash
uaere train --backbone logistic_mel --dataset twin --n 200 --out artifacts/models
uaere train --dataset fish --root data/toadfish --out artifacts/models --seed 0
uaere train --dataset species --root data/species --out artifacts/models --seed 0
uaere predict --wav path/to/clip.wav --checkpoint artifacts/models/species_head.joblib
```

Twin checkpoint: `artifacts/models/last.npz` (inspection weights; `uaere predict` does not load it).
Fish checkpoint: `artifacts/models/fish_head.joblib`. Species checkpoint: `artifacts/models/species_head.joblib`. `predict` loads the species file when it exists, otherwise the fish file. A missing file exits 2. It does not train a ship model instead.

The species winner is the candidate with the higher validation macro-F1. An exact tie prefers the MLP, and a tie between the two MLPs prefers the smaller `alpha`. The test split is not an input to that choice. On the seed-0 Watkins run the MLP at `alpha` 1e-4 won (validation macro-F1 0.292, 51 iterations, early stopping). Test macro-F1 of that checkpoint is 0.281. The logistic candidates converged inside `max_iter` 2000 (summary at iteration 497, image at 494). See `docs/DATASETS.md`. sklearn 1.9 `MLPClassifier` has no `class_weight` argument, so the MLP is balanced with `sample_weight`. The logistic candidates use `class_weight="balanced"`. Early stopping holds out an inner tenth of the training rows. The checkpoint is not refit on train+validation, so the saved file matches the test score. Wake for this head is the maximum Dirichlet class probability. The toadfish head still uses `1 - P(other)`. The ship head still uses `1 - P(reject)`.

The waveform is not peak-normalised. Integer PCM is divided by full scale (`2^31` or `32768`). Float samples stay as stored. The window is the centre 1.0 s at the head's working rate.

The toadfish logistic reads those Mel statistics and three numbers taken from the waveform inside the stored boatwhistle band (80–500 Hz on `cause.taxon.opsanus_tau`): the share of power in that band, spectral flatness inside it, and the autocorrelation peak between those two periods. The solver is lbfgs with `class_weight="balanced"`. Validation macro-F1 on the seed-0 recording-locked split was 0.799 without those three numbers and 0.817 with them. An MLP on the same vector scored 0.819 and was not used. The test files did not choose the features or the estimator. The saved seed-0 logistic stopped at iteration 64 of `max_iter` 2000. Its test macro-F1 is 0.839 and its wake AUC is 0.898. Seeds 1 and 2 of the same protocol, in `artifacts/paper/review_upgrade/`, reach test macro-F1 0.848 and 0.844. The three-seed mean is 0.844 ± 0.004. That review command does not overwrite `artifacts/models/fish_head.joblib`. The earlier Mel-only fit is `artifacts/models/fish_head_melstats.joblib`, and its test macro-F1 of 0.827 is `artifacts/paper/fish_suite_melstats.json`.

The fish explanation gate reads `T(e)`, a hand-set sigmoid. The graph is not an input. On this corpus the health term, the environment term, and the epistemic uncertainty are constant, so a validation-chosen threshold on `T(e)` selects the same windows as a threshold on wake confidence. The review JSON is the record of that equality. Do not describe the sigmoid as beating a confidence threshold on these files. Dirichlet concentration is the fixed value 20. It is not a learned Sensoy loss.

TinyCNN is still constructed so its INT8 size can be checked against the MAC budget. Its logits are not the label. On the fish path the label is `TrustScore.predicted_name()` from the saved head. The random `w1`/`b1` matrices inside `EvidentialHead` are unused once `sklearn` is set.
