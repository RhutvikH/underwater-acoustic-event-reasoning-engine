# Experiment protocol (lab notebook)

Frozen for the paper. Hashes of public datasets are filled in when the bits arrive.

## Command

```bash
PYTHONPATH=src python -m uaere eval --suite paper --out artifacts/paper --n-windows 240 --seed 0
```

Seeds `{0,1,2}` for the camera-ready tables. α = 0.05, Holm–Bonferroni on the family (AUC, ECE, energy).

## Data

| Corpus | Role | Access |
|--------|------|--------|
| Twin-synthetic `busy_strait` + `high_sea_state` | always-on CI / primary until DeepShip lands | generated, seed-locked |
| DeepShip (Irfan et al. 2021) | O4 primary if released | email authors; SHA-256 TBD |
| ShipsEar | domain-shift / KG coverage | public; SHA-256 TBD |

Split: time-aware, later windows are test (`uaere.data.split.time_aware_split`).

## Baselines

- Fixed STFT-band energy threshold, θ from Youden on train (`baseline_energy_threshold.py` only).
- Always-on L3.
- Energy-then-classify (no L3).

## Hardware

Software-defined profiles in `src/uaere/hardware/profiles.py`. No silicon. Optional future: Renode Cortex-M cycle check of L0 within 2×.

## Security tests

- Tampered firmware fails `secure_boot`.
- Honest HMAC verifies.
- `gd32vf103` is refused for authenticated L2.

## Embargo

Do not put `artifacts/` or trained weights on a public remote until the provisional is filed.
