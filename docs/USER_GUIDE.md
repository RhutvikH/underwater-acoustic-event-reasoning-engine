# AHAIF user guide — how to run and use the system

This is the operator manual. The science and claims live in
`paper/manuscript.md` and `paper/complete_specification.md`.

**What you are running.** A trust-aware, energy-gated underwater acoustic
event reasoner (package name `uaere`, public name AHAIF). It is a **TRL-4
simulation**. There is no hydrophone and no MCU firmware image.
Every joule, millisecond, and security check is produced by the digital twin
and a software-defined hardware abstraction. The browser demo draws that
simulation. Its play button is the 1 s window a node scored, for headphones.

**Hard rule.** The live pipeline never does `if energy > threshold`. That
baseline exists only for papers, in
`src/uaere/trust/baseline_energy_threshold.py`.

---

## 0. What you get after a successful run

| Command | What it produces |
|---------|------------------|
| `uaere eval --suite paper` | `artifacts/paper/suite.json` + `suite.md` — synthetic gating appendix |
| `uaere eval --suite review` | `artifacts/paper/review_upgrade/review_upgrade.json` — toadfish seeds 0, 1, and 2. This is the manuscript table. |
| `uaere eval --suite fish` | `artifacts/paper/fish_suite.json` + `fish_suite.md` — preserved single-seed toadfish run. Do not re-run it over that file. |
| `uaere twin` | 1-second WAV clips from the digital twin |
| `uaere data summarize` | class counts of a twin draw |
| `uaere train --dataset twin` | `artifacts/models/last.npz` (not loaded by `predict`) |
| `uaere train --dataset fish` | `artifacts/models/fish_head.joblib` |
| `uaere kg` | Turtle export of the marine acoustic knowledge graph |
| `pytest tests` | unit and integration tests, including the fish checkpoint path |

The paper suite prints five booleans. All five must be `true` for a green lab
notebook entry:

```json
{
  "trust_beats_energy_auc": true,
  "gated_less_energy": true,
  "kg_chain_on_true_events": true,
  "five_device_profiles": true,
  "security_demo": true
}
```

---

## 1. Install (once)

Python **3.11 or newer**. No CUDA, no PyTorch, no TensorFlow.

```bash
cd underwater-acoustic-event-reasoning-engine
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Check the CLI:

```bash
uaere --version
# 0.1.0
uaere --help
```

If `uaere` is not on `PATH` (some editable installs):

```bash
export PYTHONPATH=src
python -m uaere --help
```

---

## 2. Five-minute first run

From the repo root, with the venv active:

```bash
# 0. Live swarm GUI (the presentation). Browser: http://127.0.0.1:8765/
uaere demo --nodes 8 --port 8765

# 1. See what the twin thinks a busy strait sounds like (class mix)
uaere data summarize --n 64 --seed 0

# 2. Render 16 labelled 1 s WAVs you can listen to
uaere twin --scenario busy_strait --n 16 --out artifacts/twin --seed 0

# 3. Export the knowledge graph
uaere kg --out artifacts/kg/marine_acoustic.ttl

# 4. Train the ship head on twin-synthetic windows (writes last.npz, not a fish checkpoint)
uaere train --n 200 --steps 200 --seed 0 --out artifacts/models

# 5. Synthetic gating appendix
uaere eval --suite paper --n-windows 160 --seed 0 --out artifacts/paper

# 6. Real fish head, after wavclips_TFv4.zip is unzipped (see docs/DATASETS.md)
uaere train --dataset fish --root data/toadfish --max-per-class 3000 --seed 0 --out artifacts/models
uaere eval --suite review --root data/toadfish --out artifacts/paper/review_upgrade --max-per-class 3000 --min-files 40 --steps 2000
uaere predict --wav data/toadfish/wavclips/bwhistleclips/<one-file>.wav

# 7. Species head, after data/species/<species>/*.wav exists (see docs/DATASETS.md)
uaere train --dataset species --root data/species --out artifacts/models --seed 0
uaere eval --suite species --root data/species --out artifacts/paper --seed 0
```

Step 5 takes on the order of **10–40 seconds** on a laptop. Then open:

- `artifacts/paper/suite.md` — human summary
- `artifacts/paper/suite.json` — machine-readable tables

A reference run (seed 0, 160 windows + high-sea-state extra) produced:

| Metric | Value |
|--------|-------|
| Trust ROC-AUC | 0.916 |
| Energy-threshold ROC-AUC | 0.069 |
| Trust ECE | 0.328 |
| Ontology hit-rate | 1.00 (21/21 true events) |
| Gated energy | 0.208 mJ / window |
| Always-on L3 energy | 6.66 mJ / window |
| 10 Wh battery life (gated vs always-on) | ~2003 d vs ~63 d |

Those numbers are **twin-synthetic**. They are the gating appendix, not the
toadfish result and not DeepShip. The manuscript fish table is
`artifacts/paper/review_upgrade/review_upgrade.md`. Three recording-locked
seeds give test macro-F1 0.844 ± 0.004. Wake ROC-AUC is 0.908 ± 0.009.
The trust explanation gate matches a confidence threshold on these files:
mean modelled energy 5.779 mJ per window, saving 10.4% ± 0.7% against
always-Level-3 on the STM32L476 datasheet profile. That saving is an
estimate. Seed 0 of the same classifier is macro-F1 0.839, preserved in
`artifacts/paper/fish_suite.json`. The knee in that older file was selected
on the test split. Do not quote its 3.82 mJ figure as the review result.
The Mel-only copy is `artifacts/paper/fish_suite_melstats.md` (macro-F1 0.827).
The species table is `artifacts/paper/species_suite.md`. Seed 0 test macro-F1
on the 16 Watkins species that cleared the floors is 0.281. That misses the
pre-registered bar of 0.70. Watkins seeds 1 and 2 were not run. The checkpoint
is `artifacts/models/species_head.joblib`. Species train and species eval use
a logistic budget of at least 2000 iterations. The review suite floors
`--steps` at 2000. An explicit `--steps` above 2000 is kept. The argparse
default of 250 still applies to the paper suite and to `--suite fish`.

---

## 3. CLI reference

### `uaere eval`

Runs the frozen paper protocol (`src/uaere/eval/suite.py`).

```bash
uaere evaluate --out artifacts/eval

# Raw tables without pass/fail (still useful):
uaere eval --suite paper \
  --out artifacts/paper \
  --n-windows 240 \
  --seed 0 \
  --steps 250
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--suite` | `paper` | `paper` (synthetic), `review` (toadfish seeds 0–2), `fish` (older single-seed writer), or `species` (one folder per species) |
| `--out` | `artifacts/paper` | JSON + Markdown destination. `review` redirects the default to `artifacts/paper/review_upgrade`. |
| `--n-windows` | 240 | twin windows from `busy_strait` (the suite also adds `n/3` high-sea-state windows) |
| `--seed` | 0 | numpy Generator seed. `review` ignores it and always runs 0, 1, and 2. Watkins remains seed 0. |
| `--steps` | 250 | logistic `max_iter` for the paper suite and `--suite fish`. `review` floors this at 2000. Species train and species eval use at least 2000. |

Exit code `0` if every success flag is true, else `1`.

**How to read `suite.json`**

- `wake.trust_auc` / `wake.energy_auc` — Objective 1. Detection score is
  wake confidence `C_wake = 1 - p_hat_reject`, **not**
  band energy.
- `wake.trust_ece` — calibration of that score vs `event_present`.
- `explanations.ontology_hit_rate` — Objective 2. Top KG cause matches the
  twin’s `cause_id`.
- `gating.*` — Objective 3 of the **synthetic** appendix. NSGA-II Pareto
  knee vs always-on L3 on twin windows. This block is not the toadfish
  gate. The toadfish comparison is one validation-chosen threshold per
  score, written by `--suite review`.

**How to read `review_upgrade.json`**

- `summary.classification_macro_f1` — logistic macro-F1, mean and sample
  standard deviation over seeds 0, 1, and 2. The explanation gate does not
  change this number.
- `summary` rows for trust, confidence, confidence–uncertainty,
  trust-without-environment, and trust-without-uncertainty — same test
  mask on this corpus after the threshold is retuned.
- Uncertainty-only and entropy rows — a worse explanation policy. They
  send confident negatives to Level 3.
- Energy fields — `stm32l476` datasheet joules. `E_g` is 0. The Level-3
  path includes the unused Level-2 MAC budget.
- `abstention` — a separate baseline. It is not the macro-F1 above.
- SNR rows — white noise added to the seed-0 test waveforms before
  features. Derived data. The head and the clean-validation threshold
  stay frozen.
- `hardware` — millijoules per level on five ISA profiles.
- `security.tamper_caught` — mutated firmware fails secure boot.
- `security.gd32_refused_authenticated_l2` — RISC-V profile without a
  secure element is refused when the policy requires attestation.
- `battery` — Coulomb count from per-window joules on a 10 Wh cell,
  assuming continuous 1 Hz inference.

### `uaere twin`

```bash
uaere twin --scenario busy_strait --n 32 --seed 0 --out artifacts/twin
```

Scenarios (YAML in `configs/twin/`, builtins if the file is missing):

| Name | What it stresses |
|------|------------------|
| `busy_strait` | mixed shipping, sea-state 2 (DeepShip-like) |
| `high_sea_state` | sea-state 5, lots of geophony — energy baseline false-alarms |
| `sensor_fault_injection` | 35% bias / dropout / gain-wander |
| `calm_coastal` | builtin only, quieter mix |
| `ice_keel` | builtin only, optional scatterer; **not a paper claim** |
| `pam_field` | **demo default** — whale, dolphin, sperm whale, seal, fish + storm |

Each WAV is 16 kHz, mono, 1 second. The filename is `{scenario}-{index}.wav`.
Labels are **not** in the WAV; they live on `WindowRecord` objects if you use
the Python API.

### `uaere data summarize`

```bash
uaere data summarize --scenario busy_strait --n 64 --seed 0
```

Prints a JSON histogram of `event_class` values, e.g.

```json
{
  "tug": 12,
  "cargo": 10,
  "reject": 9,
  "geophony": 8,
  "...": 0
}
```

`geophony` and `artifact` are mapped to the Dirichlet **reject** class during
training (they are not vessel events). `event_present` is true only for
tug / cargo / tanker / passenger.

### `uaere train`

```bash
uaere train --scenario busy_strait --n 200 --steps 200 --seed 0 --out artifacts/models
```

Fits a class-balanced multinomial logistic on Mel statistics, then lifts
`p_hat` to a Dirichlet via `alpha = 1 + tau * p_hat`
(`tau = 20`). `--dataset twin` (the default) and `--dataset watkins` write
`artifacts/models/last.npz`. The paper suite **retrains** its own ship head;
that npz is for inspection.

```bash
uaere train --dataset fish --root data/toadfish --max-per-class 3000 --min-files 40 --seed 0
```

Fish training writes `artifacts/models/fish_head.joblib` and returns before
the npz path. A missing `--root` exits 2. It does not fall back to ships.

### `uaere predict`

```bash
uaere predict --wav path/to/clip.wav --checkpoint artifacts/models/fish_head.joblib
```

Loads the joblib file and scores the centre 1 s at that checkpoint's sample rate. With no `--checkpoint`, a species checkpoint is used when `artifacts/models/species_head.joblib` exists, otherwise the fish checkpoint. It does not retrain. A missing checkpoint exits 2.

### `uaere demo` (presentation GUI + Unity feed)

```bash
uaere demo --nodes 8 --port 8765
# default model is auto: species_head.joblib if it exists, else fish_head.joblib, else the ship logistic
uaere demo --model oracle --scenario pam_field   # twin illustration; gold class, not a paper result
uaere demo --model fish --replay data/toadfish --checkpoint artifacts/models/fish_head.joblib
uaere demo --model species --replay data/species --checkpoint artifacts/models/species_head.joblib
```

Open http://127.0.0.1:8765/ — live cheap-sensor field, neighbour-wake
links, click-to-inspect. The class on a node is the classifier's decision.
Press play to hear that node's 1 s window in headphones. That is not sound
into the water. `--model oracle` is how you show the scripted animal rings.
Unity polls `/api/state` (see `unity/AHAIF/`). `GET /api/audio?node=n0`
returns the scored window as 16-bit WAV at the window's sample rate.

### `uaere swarm`

One-shot JSON of N nodes sharing one event (no browser).

```bash
uaere swarm --nodes 8 --ticks 5 --seed 0
```

### `uaere edge`

Laptop stand-in for Raspberry Pi Zero 2 W nodes.

```bash
uaere edge --n 4 --seed 0
```

### `uaere kg`

```bash
uaere kg --out artifacts/kg/marine_acoustic.ttl
```

Turtle dump of the 20-node / 21-edge marine acoustic graph. Source of truth
is `src/uaere/kg/ontology.yaml`.

---

## 4. Python API — infer one window

This is the path a future firmware port would call, minus the HAL.

```python
from uaere.causal.reasoner import CausalReasoner
from uaere.classify.train import extract_features, train_evidential
from uaere.kg.graph import load_kg
from uaere.pipeline import AhaifPipeline
from uaere.representation.env_norm import AdaptiveNormalizer
from uaere.security.hal import SecureHAL
from uaere.trust.trust_score import TrustEngine
from uaere.twin.render import TwinRenderer
from uaere.types import PolicyVector

# 1. Draw a labelled lab set and fit trust
recs = TwinRenderer("busy_strait", seed=0).render_dataset(80)
head = train_evidential(extract_features(recs), steps=120, seed=0)

# 2. Boot the secure HAL (simulated). Tampering this image must fail.
fw, model_id = b"ahaif-firmware-v0", b"tiny-cnn-v0"
hal = SecureHAL(fw, model_id)
hal.secure_boot(fw)

# 3. Policy π: thresholds on event-trust T(e), not on energy
policy = PolicyVector(
    tau1=0.20, tau2=0.40, tau3=0.70,
    device_id="stm32l476",
    require_authenticated=True,
)

pipe = AhaifPipeline(
    trust=TrustEngine(head),
    kg=load_kg(),
    policy=policy,
    normalizer=AdaptiveNormalizer(n_bins=32),
    hal=hal,
)

result = pipe.infer(recs[0])
print(result.level, result.event_class, result.reason, result.energy_j)
print(result.trust.wake_confidence, result.trust.event_trust)
if result.explanation:
    print(result.explanation.sentence)
    print(result.explanation.chain)
```

`InferenceResult` fields you will actually use:

| Field | Meaning |
|-------|---------|
| `level` | `L0`…`L4` actually executed |
| `trust.wake_confidence` | `C_wake`, detection score |
| `trust.event_trust` | `T(e)`, what the **gate** sees |
| `trust.u_aleatoric` / `u_epistemic` | Dirichlet uncertainties |
| `event_class` | label from the trained head (`predicted_class`). TinyCNN is a MAC budget, not this name. |
| `explanation.sentence` | one-line cause chain (only if L3+) |
| `energy_j` | modelled compute (+ TX if L4) on the placed device |
| `authenticated` | HMAC of `(model_id ‖ window ‖ output)` verified |
| `reason` | why that level was chosen |

**Placing on a device that cannot attest.** If `require_authenticated=True`
and `device_id="gd32vf103"`, `pipe.infer` raises `PermissionError`. That is
the orchestration claim, not a crash.

**Tamper demo.**

```python
from uaere.security.hal import TamperError
try:
    hal.secure_boot(b"evil-firmware")
except TamperError:
    print("boot refused")
```

---

## 5. Tests and the energy-threshold guard

```bash
source .venv/bin/activate
python -m pytest tests -q
bash scripts/ci.sh
```

`scripts/ci.sh` runs pytest **and** greps `src/uaere` for `if energy >`
outside the baseline file. If that grep hits, the build is red: someone
re-introduced a wake-word.

---

## 6. Using DeepShip or ShipsEar (optional)

The pipeline does not need 47 hours of DeepShip to run. When you have a
class-folder dump (`Cargo/`, `Tanker/`, `Passenger/` / `Passenger ship/`,
`Tug/` of WAVs):

```python
from uaere.data.adapters import DeepShipAdapter, ShipsEarAdapter
from uaere.data.split import time_aware_split

recs = DeepShipAdapter("/path/to/DeepShip").load()
train, val, test = time_aware_split(recs)   # later-in-time = test
```

`ShipsEarAdapter` is the same windowing with a different root. Real
recordings have a default `EnvironmentState` and healthy `SensorHealth`
because those corpora do not ship twin oracles. Trust’s `kappa(h)` and
`rho(s,x)` are then near-constant; detection still uses `C_wake`.

Put the path in `configs/default.yaml` as `data.deepship_root` when you freeze
a camera-ready run, and record the SHA-256 in
`paper/experiment_protocol.md`.

---

## 7. Interpreting levels (the hierarchy you are gating)

```
hydrophone window (1 s, 16 kHz)
        │
        ▼
 L0  rFFT moments (always computed) + predictive-coding surprise S
        │  ship/twin head: surprise_admit(S); sleep if admit < 0.15
        │  (ocean matches Knudsen model — still not "energy > theta")
        │  fish checkpoint: S is recorded; the species label is the classifier
        ▼
 L1  log-Mel + FiLM env-norm + evidential Dirichlet → C_wake, T(e)
        │
        ▼
     RuntimeGate(pi) on T(e)
        │
        ├─ T < tau1 → stay L0
        ├─ T < tau2 → L1 only
        ├─ T < tau3 → L2 TinyCNN
        ├─ T >= tau3, T outside collab band → L3 KG + WHY/WHY-NOT
        └─ T >= tau3 and tau_collab_lo <= T <= tau_collab_hi → L4 (replaces L3)
```

π itself is **not** typed by a human for the paper run. NSGA-II searches a
population of threshold vectors against the twin replay; the Chebyshev knee
of the non-dominated front is what `suite.json` reports.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No module named uaere` | venv not active or package not installed | `source .venv/bin/activate && pip install -e .` |
| `python -m uaere` fails | old tree without `__main__.py` | pull latest; use `uaere` console script |
| `uaere eval` exit 1 | a success flag is false | read `artifacts/paper/suite.json`; usually AUC or energy |
| Trust AUC ≈ 0.5 | too few windows / unlucky seed | `--n-windows 160` or higher; try `--seed 1` |
| Energy AUC ≪ 0.5 | expected on `high_sea_state` | loud geophony is labelled *not* a vessel; that is the baseline’s failure mode |
| `PermissionError` on infer | authenticated policy + `gd32vf103` | set `require_authenticated=False` or pick `stm32l476` |
| Tests slow | TinyCNN is a numpy nested loop | 24 tests are ~25 s on a laptop; that is normal |
| Want to listen to clips | twin WAVs are float scaled to int16 | any player that accepts 16 kHz mono WAV |

---

## 9. What not to do

- Do **not** push this tree to a public remote before a provisional patent is
  filed (`LICENSE`, `paper/invention_disclosure.md`).
- Do **not** add an LLM, dashboard, or always-on L3 “for the demo”. That
  destroys the energy claim and creates prior art noise.
- Do **not** quote twin-synthetic AUC as DeepShip performance.
- Do **not** quote the 3.82 mJ knee in `fish_suite.json` as the toadfish
  energy result. The manuscript uses the review suite.
- Do **not** say the trust score beats a confidence threshold on the
  Pamlico files. The review run shows the same windows.
- Do **not** call modelled joules a measured underwater deployment.
- Do **not** treat the ice-keel YAML as a contribution. It is a stress
  scenario, not a claim.

---

## 10. Where to read next

| If you want… | Open |
|--------------|------|
| The research paper | `paper/manuscript.md` |
| The patent complete specification | `paper/complete_specification.md` |
| Numbered equations ≡ code | `paper/formulation.md` |
| Claim chart vs 35 papers | `paper/claim_chart.md` |
| How a run is frozen | `paper/experiment_protocol.md` |
| Capstone slides | `docs/review_1.pdf` |
| Non-negotiable constraints | `docs/non_negotiables.txt` |
