# AHAIF user guide — how to run and use the system

This is the operator manual. The science and claims live in
`paper/manuscript.md` and `paper/complete_specification.md`.

**What you are running.** A trust-aware, energy-gated underwater acoustic
event reasoner (package name `uaere`, public name AHAIF). It is a **TRL-4
simulation**. There is no hydrophone, no MCU firmware image, and no web UI.
Every joule, millisecond, and security check is produced by the digital twin
and a software-defined hardware abstraction.

**Hard rule.** The live pipeline never does `if energy > threshold`. That
baseline exists only for papers, in
`src/uaere/trust/baseline_energy_threshold.py`.

---

## 0. What you get after a successful run

| Command | What it produces |
|---------|------------------|
| `uaere eval --suite paper` | `artifacts/paper/suite.json` + `suite.md` — every table the capstone/paper needs |
| `uaere twin` | 1-second WAV clips from the digital twin |
| `uaere data summarize` | class counts of a twin draw |
| `uaere train` | `artifacts/models/evidential_head.npz` |
| `uaere kg` | Turtle export of the marine acoustic knowledge graph |
| `pytest tests` | 24 unit/integration tests |

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
# 1. See what the twin thinks a busy strait sounds like (class mix)
uaere data summarize --n 64 --seed 0

# 2. Render 16 labelled 1 s WAVs you can listen to
uaere twin --scenario busy_strait --n 16 --out artifacts/twin --seed 0

# 3. Export the knowledge graph
uaere kg --out artifacts/kg/marine_acoustic.ttl

# 4. Train the evidential head on twin-synthetic windows
uaere train --n 200 --steps 200 --seed 0 --out artifacts/models

# 5. Run the paper evaluation (the one that matters)
uaere eval --suite paper --n-windows 160 --seed 0 --out artifacts/paper
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

Those numbers are **twin-synthetic, not DeepShip**. Quote them as TRL-4 lab
results, not as ocean-deployment performance.

---

## 3. CLI reference

### `uaere eval`

Runs the frozen paper protocol (`src/uaere/eval/suite.py`).

```bash
uaere eval --suite paper \
  --out artifacts/paper \
  --n-windows 240 \
  --seed 0 \
  --steps 250
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--suite` | `paper` | only `paper` exists today |
| `--out` | `artifacts/paper` | JSON + Markdown destination |
| `--n-windows` | 240 | twin windows from `busy_strait` (the suite also adds `n/3` high-sea-state windows) |
| `--seed` | 0 | numpy Generator seed; camera-ready uses `{0,1,2}` |
| `--steps` | 250 | logistic-regression `max_iter` for the evidential lift |

Exit code `0` if every success flag is true, else `1`.

**How to read `suite.json`**

- `wake.trust_auc` / `wake.energy_auc` — Objective 1. Detection score is
  wake confidence \(C_{\mathrm{wake}}=1-\hat p_{\mathrm{reject}}\), **not**
  band energy.
- `wake.trust_ece` — calibration of that score vs `event_present`.
- `explanations.ontology_hit_rate` — Objective 2. Top KG cause matches the
  twin’s `cause_id`.
- `gating.*` — Objective 3. NSGA-II Pareto knee vs always-on L3. Miss rate
  is matched by construction (same detector, less compute).
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
\(\hat p\) to a Dirichlet via \(\alpha = 1 + \tau\hat p\)
(\(\tau=20\)). Writes `evidential_head.npz`. The paper suite **retrains**
its own head; this command is for inspection and for the Python API below.

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
| `trust.wake_confidence` | \(C_{\mathrm{wake}}\), detection score |
| `trust.event_trust` | \(T(e)\), what the **gate** sees |
| `trust.u_aleatoric` / `u_epistemic` | Dirichlet uncertainties |
| `event_class` | predicted label (CNN if L2+, else argmax of Dirichlet) |
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
because those corpora do not ship twin oracles. Trust’s \(\kappa(h)\) and
\(\rho(s,x)\) are then near-constant; detection still uses \(C_{\mathrm{wake}}\).

Put the path in `configs/default.yaml` as `data.deepship_root` when you freeze
a camera-ready run, and record the SHA-256 in
`paper/experiment_protocol.md`.

---

## 7. Interpreting levels (the hierarchy you are gating)

```
hydrophone window (1 s, 16 kHz)
        │
        ▼
 L0  rFFT moments — always on
        │  logistic of the *vector* (energy, ZCR, centroid, flatness, residual)
        │  admit < 0.15  →  sleep (near-silence), still not "energy > θ"
        ▼
 L1  log-Mel + FiLM env-norm + evidential Dirichlet → C_wake, T(e)
        │
        ▼
     RuntimeGate(π) on T(e)
        │
        ├─ T < τ1 → stay L0
        ├─ T < τ2 → L1 only
        ├─ T < τ3 → L2 TinyCNN
        ├─ T ≥ τ3 → L3 KG + counterfactual
        └─ τ_collab_lo ≤ T ≤ τ_collab_hi → also L4 neighbour-wake TX
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
