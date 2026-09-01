# Adaptive Hierarchical Acoustic Intelligence Framework (AHAIF)

Trust-aware, energy-efficient **event-level** acoustic reasoning for underwater
acoustic sensor networks. Capstone product from `docs/review_1.pdf`, built so
that every line of `docs/non_negotiables.txt` is true of the same system.

**TRL-4. Simulation only. Code is embargoed until a patent provisional is filed.**

Team: Rhutvik Hegde (23BCI0079), Kalyani Manoj (23BCI0081). Guide: Dr. S. M. Satapathy, VIT.

## What it is

A hierarchy, not a wake-word pipeline:

| Level | What runs | When |
|-------|-----------|------|
| L0 | DSP moments (energy, ZCR, centroid, flatness, residual) | always |
| L1 | log-Mel + evidential Dirichlet trust \(T(e)\) | φ0 logistic admits |
| L2 | Tiny depthwise CNN classifier | \(T \ge \tau_2\) |
| L3 | Marine acoustic KG + counterfactual explanation | \(T \ge \tau_3\) |
| L4 | Collaborative neighbour-wake | uncertain band |

There is **no** `if energy > θ` in the live pipeline. The conventional energy
threshold lives only in `src/uaere/trust/baseline_energy_threshold.py` and is
imported by eval.

## Non-negotiables → code

| # | Requirement | Where |
|---|-------------|--------|
| 1 | Hierarchical framework + mathematically defined multi-objective optimisation | `src/uaere/policy/`, `src/uaere/math/` |
| 2 | Dynamic Acoustic Trust and Wake Confidence + adaptive multi-level representation | `src/uaere/trust/`, `src/uaere/representation/` |
| 3 | Marine Acoustic Knowledge Graph + causal reasoning | `src/uaere/kg/`, `src/uaere/causal/` |
| 4 | Underwater acoustic digital twin | `src/uaere/twin/` |
| 5 | Hardware-aware orchestration + secure HAL (Cortex-M, ESP32, RISC-V, FPGA, NPU, TPM, SE, secure boot, authenticated TinyML, AEAD) | `src/uaere/hardware/`, `src/uaere/security/` |
| 6 | Formulation, complexity, calibration, stats, ablation, robustness, battery, hardware sim | `src/uaere/eval/`, `src/uaere/math/`, `paper/` |

## How to run (copy-paste)

Python 3.11+. No PyTorch, no TensorFlow.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

uaere data summarize --n 64 --seed 0
uaere twin --scenario busy_strait --n 16 --out artifacts/twin
uaere kg --out artifacts/kg/marine_acoustic.ttl
uaere train --n 200 --steps 200 --out artifacts/models
uaere eval --suite paper --n-windows 160 --seed 0 --out artifacts/paper
python -m pytest tests -q
```

The command that matters for the paper is `uaere eval --suite paper`.
It writes `artifacts/paper/suite.md` and `suite.json`. A reference run
(seed 0) gave trust ROC-AUC **0.916** vs energy-threshold **0.069**,
ontology hit-rate **1.00**, and ~32× less energy than always-on L3.

**Full operator manual (every flag, the Python API, troubleshooting):**
[`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)

If `uaere` is not on `PATH`: `export PYTHONPATH=src` and
`python -m uaere …`.

## Documentation (paper + patent)

| Document | Role |
|----------|------|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | How to run and use |
| [`paper/manuscript.md`](paper/manuscript.md) | Full research paper draft, with results |
| [`paper/complete_specification.md`](paper/complete_specification.md) | Patent complete specification (claims 1–19) |
| [`paper/invention_disclosure.md`](paper/invention_disclosure.md) | Short disclosure |
| [`paper/claim_chart.md`](paper/claim_chart.md) | vs the 35-paper literature review |
| [`paper/formulation.md`](paper/formulation.md) | Numbered equations ≡ code |
| [`paper/experiment_protocol.md`](paper/experiment_protocol.md) | Freeze a camera-ready run |
| [`docs/INDEX.md`](docs/INDEX.md) | Map of every doc |

Working title: *Event-Level Trust-Gated Causal Reasoning for
Energy-Constrained Underwater Acoustic Sensor Networks.* Target: IEEE IoT
Journal or Expert Systems with Applications. SDG 9 and 14.

## Tests

```bash
PYTHONPATH=src python -m pytest tests
bash scripts/ci.sh
```

CI greps for leaked `if energy >` outside the baseline file.

## DeepShip

The pipeline does not require 47 h on day one. Point `data.deepship_root` at a
class-folder dump when you have it (`DeepShipAdapter`). Until then, the twin
is the lab.

## Licence

Proprietary / embargoed. See `LICENSE`.
