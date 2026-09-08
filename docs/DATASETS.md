# Datasets

Honest inventory. We did **not** collect a private ocean cruise.

| Name | In this git repo? | How you get it | Labels we use | Licence note |
|------|-------------------|----------------|---------------|--------------|
| Twin-synthetic | generated at runtime | `uaere twin` / `TwinRenderer` | tug, cargo, tanker, passenger, geophony, artifact, reject, optional biological | ours; procedural, no copyrighted WAV |
| DeepShip | **adapter only** | email authors (Irfan et al. 2021) | 4 vessel classes | paper's terms |
| ShipsEar | **adapter only** | public ShipsEar release | ships + some natural | paper's terms |
| Watkins Best-of | **adapter only** | WHOI academic download | all mapped to `biological` (not 32-way species SOTA) | personal/academic; commercial WHOI use prohibited |
| DCLDE 2026, MobySound | not wired | — | future | — |

## Train

```bash
uaere train --dataset twin --backbone logistic_mel --n 400 --out artifacts/models
uaere train --dataset watkins --root /path/to/watkins --backbone logistic_mel
```

If Watkins `--root` is empty, training falls back to twin.

## Predict

```bash
uaere twin --n 1 --out /tmp/ahaif_clip
uaere predict --wav /tmp/ahaif_clip/busy_strait-0.wav
```

Prints `class`, `surprise`, `C_wake`, `T(e)`, `why`, `why_not`.

## What we refuse to claim

99% closed-set species ID on Watkins. That number is occupied
(WhaleNet, MT-Resformer). Our biological class is “this is animal, not
ship/storm,” then the KG names odontocete as a cause.
