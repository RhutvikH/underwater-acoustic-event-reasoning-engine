# Datasets

Honest inventory. We did **not** collect a private ocean cruise.

| Name | In this git repo? | How you get it | Labels we use | Licence note |
|------|-------------------|----------------|---------------|--------------|
| Twin-synthetic | generated at runtime | `uaere twin` / `TwinRenderer` | ships (`busy_strait`) or animals (`pam_field`: whale, dolphin, sperm whale, seal, fish) + storm/artifact | ours; procedural, no copyrighted WAV |
| DeepShip | **adapter only** | email authors (Irfan et al. 2021) | 4 vessel classes | paper's terms |
| ShipsEar | **adapter only** | public ShipsEar release | ships + some natural | paper's terms |
| Watkins Best-of | **adapter only** | WHOI academic download | all mapped to `biological` (not 32-way species SOTA) | personal/academic; commercial WHOI use prohibited |
| ToadFishFinder v4 | **not in git** (`data/` is ignored) | Dryad, below | `opsanus_tau` vs `other` | CC0 |
| DCLDE, MobySound, SanctSound, FishSounds, … | links only | see [MARINE_DATASETS.md](MARINE_DATASETS.md) | not used to train species ID | — |

## ToadFishFinder (the fish head)

Public candidate set of oyster toadfish (*Opsanus tau*) boatwhistles from Pamlico Sound, North Carolina, 2016–2018.

- Paper: Bohnenstiehl, *Ecological Informatics* 77, 102268 (2023). https://doi.org/10.1016/j.ecoinf.2023.102268
- Data: https://doi.org/10.5061/dryad.gtht76hr9 (CC0). Dataset page: https://datadryad.org/dataset/doi:10.5061/dryad.gtht76hr9
- Use `wavclips_TFv4.zip` (about 2 GB). Folders `bwhistle/` and `other/`. Do not download the two `.mat` files for this pipeline.
- Zenodo record 8083696 matches the title and hosts the reader notebooks only. It is not the audio.
- Clips are 1.35 s, 24 kHz, 32-bit, pressure-corrected µPa. The loader takes the centre 1.0 s and resamples to 16 kHz. Their ResNet used the centre 850 ms.
- After unzip the folders are `wavclips/bwhistleclips/` and `wavclips/otherclips/` (also accepted: `bwhistle/` and `other/`).
- Names look like `bw_BB_201351190.170422052002_000.3870833.wav`. The middle token is one hydrophone deployment. Both classes of that deployment stay on one side of the split. That is not a second estuary: every deployment in this zip is Pamlico Sound.
- WAV format in this release is 32-bit integer PCM, 24 kHz, mono. Samples are full-scale normalised (divide by 2^31). Relative level across files is kept. The browser copy is peak-limited separately.
- Sea state, temperature, and salinity are not in the release. Training and test both use an imputed sea state of 2.
- Do not commit the zip or the WAVs.

```bash
# unzip so bwhistle/ and other/ sit under the root you pass
uaere train --dataset fish --root data/toadfish --max-per-class 3000 --min-files 40 --seed 0
uaere eval --suite review --root data/toadfish --out artifacts/paper/review_upgrade --max-per-class 3000 --min-files 40 --steps 2000
uaere predict --wav some_clip.wav --checkpoint artifacts/models/fish_head.joblib
```

`train --dataset fish` writes `artifacts/models/fish_head.joblib`. A missing root is an error. It does not synthesize ships. `predict` loads that file and does not retrain. Another species folder `<species>/*.wav` is kept only when it has at least `--min-files` source files (default 40). Until then the reported task is two classes: `opsanus_tau` and `other`. `other` is ambient or non-call, not a fish species.

The published specialist on this corpus is a ResNet-50 plus an SVM. This logistic head is not that model, and the paper must not say it beats it.

The knowledge graph for this call is hand-written in `src/uaere/kg/ontology.yaml`. It was not learned from the clips. `event.toadfish_boatwhistle` has one cause, `cause.taxon.opsanus_tau`, with the stored band 80–500 Hz, and it is marked confusable with a sciaenid chorus, wind-wave noise, and a tug, using the bands already stored on those causes. Missing or contradictory edges are not inferred: the reasoner ranks the `caused_by` neighbours of the predicted event, and this event has one. The logistic reads the 99 Mel statistics plus three waveform numbers from that 80–500 Hz band, with `class_weight="balanced"`. The graph is not an input of the trust score.

The paper table is `artifacts/paper/review_upgrade/review_upgrade.json` (seeds 0, 1, and 2). Test macro-F1 is 0.844 ± 0.004. Wake ROC-AUC is 0.908 ± 0.009. Seed 0 matches the preserved `artifacts/paper/fish_suite.json` classifier: macro-F1 0.839, wake AUC 0.898, ECE 0.059. Energy-threshold AUC 0.656 is that seed-0 file only. Cause-match is 1.00 because the graph repeats the predicted class. The label-free band check has ROC-AUC 0.630 ± 0.053 (seed 0 was 0.587). After a validation-chosen threshold, the trust gate selects the same windows as a confidence threshold. Modelled saving against always-Level-3 is 10.4% ± 0.7% on the STM32L476 datasheet profile. That figure is an estimate. The older fish-suite knee was fit on the test split. Do not quote its joules as this table. The Mel-only test macro-F1 0.827 remains `artifacts/paper/fish_suite_melstats.json`. Clips are pre-segmented, so this split does not demonstrate event discovery in a continuous stream. Sea state is the imputed constant 2.

## Train

```bash
uaere train --dataset twin --backbone logistic_mel --n 400 --out artifacts/models
uaere train --dataset watkins --root /path/to/watkins --backbone logistic_mel
uaere train --dataset fish --root data/toadfish --out artifacts/models
```

Twin and Watkins write `artifacts/models/last.npz` (the logistic weights live in the paper suite, which retrains). That npz is not a fish checkpoint. If Watkins `--root` is empty, training falls back to the twin. Fish does not.

## Predict

```bash
uaere predict --wav path/to/clip.wav
```

Requires `artifacts/models/fish_head.joblib`. Prints `class`, `surprise`, `C_wake`, `T(e)`, `why`, `why_not`. The class is the fish head's argmax. A missing checkpoint exits 2.

## What we refuse to claim

99% closed-set species ID on Watkins. That number is occupied
(WhaleNet, MT-Resformer). Watkins in this repo is “this is animal, not
ship/storm.” FishSounds is an index, not a training set for species ID.
The fish result is `opsanus_tau` versus `other` on the Pamlico candidate set.

## Species folders (the ten-class head)

Public fish audio does not contain ten species with enough independent recordings. FishSounds is an index (about one cut per species) and is not a training set. The corpus that can clear ten classes is the Watkins Marine Mammal Sound Database (Woods Hole Oceanographic Institution): isolated identified cuts, academic use, not commercial. Credit that name in every table.

```text
data/species/<species>/*.wav
data/species/groups.csv    # optional columns: file,group_id
```

A Watkins cut id such as `6102500A` / `6102500B` is tape `61025`. `83093034` is tape `83093`. Cuts from one tape stay on one side. If the names do not carry a tape id and `groups.csv` is absent, `uaere eval --suite species` stops. `uaere train --dataset species` may still fit, and the checkpoint says the split is one file per group, which is not a held-out recording.

Floors, fixed before any accuracy number: at least 10 species, 40 files, and 8 independent recordings. Cap 500 files per species, whole recordings, seed 0. Porpoise names are out (clicks above the working band). Other species stay only when the median energy above the cutoff, on up to 8 centre-1 s clips, is at most half. First pass is 16 kHz with a 7 kHz cutoff. If that pass has fewer than 10 species, one fallback uses 32 kHz and 14 kHz for this head only. The ship twin and the toadfish head stay at 16 kHz. Floors are not lowered.

```bash
uaere train --dataset species --root data/species --out artifacts/models \
  --min-classes 10 --min-files 40 --min-groups 8 --max-per-class 500 --seed 0
uaere eval --suite species --root data/species --out artifacts/paper --seed 0
uaere predict --wav clip.wav --checkpoint artifacts/models/species_head.joblib
```

`--max-per-class` defaults to 500 on this dataset and to 3000 on the fish dataset. A missing root, fewer than `--min-classes`, or a species under the file or group floor exits 2. There is no twin fallback and no toadfish fallback.

Seed 0, 16 kHz, recording-locked, from `artifacts/paper/species_suite.json`. This is call-library identification, not detection in a continuous soundscape. The demonstration bar was macro-F1 at least 0.70 and at least 15 test files for every reported species. Every species cleared the support bar. The macro-F1 did not.

| Item | Value |
|------|-------|
| Eligible species | 16. Spinner dolphin missed the 7 kHz band rule (fraction 0.529). Porpoises were excluded by name. |
| Windows | train 3307, validation 1318, test 1490 |
| Chosen model | `species_mlp_mel`, `alpha` 1e-4 (validation macro-F1 0.292, 51 iterations, early stopping). Image logistic validation macro-F1 was 0.291. |
| Test macro-F1 | 0.281 |
| Test accuracy | 0.285 |
| Majority class | `humpback_whale`, accuracy 0.062, macro-F1 0.007 |
| ECE of the maximum probability | 0.529 |
| Demonstration bar | not met |

The chosen checkpoint is that MLP. Test F1 is 0.767 for fin whale (support 97) and 0.692 for killer whale (support 98). Striped dolphin is 0.000 (support 264) and walrus is 0.000 (support 150). Logistic `max_iter` was 2000. The summary logistic converged at iteration 497. The image logistic converged at iteration 494. MLP `max_iter` was 400 with early stopping, and both alphas stopped at iteration 51. Validation macro-F1 chose the MLP at `alpha` 1e-4 (0.292) over the image logistic (0.291), the MLP at `alpha` 1e-3 (0.291), and the summary logistic (0.284). The test column did not change the winner, the penalty, or the species list. Seeds 1 and 2 were not run. That species run did not refit the toadfish checkpoint. `artifacts/paper/species_suite_capped.json` keeps the earlier 250-iteration run (test macro-F1 0.275, image logistic). Full precision, the confusion matrix, and every per-class row are in `artifacts/paper/species_suite.json`.

### Why this table does not get another fit

The cuts are already about as long as the window. Median duration is 1.67 s for striped dolphin, 1.75 s for killer whale, and 1.80 s for walrus. On mid-length files, the loudest 1 s and the centre 1 s have the same RMS, so a peak crop is the same input.

Four species meet the test set as one tape: striped dolphin `85012` (264 cuts), walrus `83001` (150), common dolphin `58013` (105), and short-finned Pacific pilot whale `91005` (131). Of the 264 striped-dolphin cuts, 222 are named sperm whale. Walrus has no correct test cut. Taking those two zeros out of the average leaves macro-F1 0.321. They stay in the table.

The recording rate is part of the score. Of the fin whale files, 98% are sampled below 8 kHz and the median rate is 600 Hz, so a 16 kHz mel is empty above a few hundred hertz. Fin whale test recall is 97/97. Killer whale is the in-band case with many tapes: median native rate 21.9 kHz, 11 test tapes, test F1 0.692. Striped dolphin is digitized near 81.6 kHz, and the inventory fraction of power above 7 kHz is 0.085, so the upper band of these tapes does not hold a species cue.

A check on the validation split only, with the test set left unscored, stayed near the published fit: log-mel image macro-F1 0.298, and a spectral-shape plus click-rate vector 0.308. Ten broadband species on that same split scored about 0.25. A five-way family label (baleen, beluga, delphinid, pinniped, sperm) reached about 0.52. The species number in the paper is the tape-held-out test macro-F1 0.281, seed 0 only. The two-class toadfish result is a separate table: three-seed macro-F1 0.844 ± 0.004 in `artifacts/paper/review_upgrade/`. Do not compare the two aggregates as the same task.
