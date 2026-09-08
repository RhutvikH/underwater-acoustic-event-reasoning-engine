# Developer handbook (no AI required)

This is how a human extends AHAIF. If a step is not in this file, it is
not a supported extension. Keep this file honest when you change the
architecture.

## 0. Getting started on *this* repo

```bash
git clone <private-url> underwater-acoustic-event-reasoning-engine
cd underwater-acoustic-event-reasoning-engine
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest tests -q
uaere demo --nodes 6 --port 8765 --seconds 8
```

You are ready when pytest is green and http://127.0.0.1:8765/ loads.
Do **not** commit `.venv/`, `artifacts/`, or `__pycache__/`.

### Layout (memorise this)

```
src/uaere/           the product
  types.py           vocabulary (EventClass, PolicyVector, DeviceProfile)
  math/              equations; unit-tested identities
  twin/              ocean lab (SSP, Thorp, hydrophone, sources)
  swarm/             N cheap nodes, one source, L4 fusion
  representation/    L0 DSP, L1 Mel, env-norm, TinyCNN
  trust/             Dirichlet + T(e)  |  baseline_energy_threshold.py ONLY here
  classify/          train logistic→Dirichlet, MAC IR
  kg/ + causal/      ontology.yaml + reasoner
  policy/            NSGA-II + RuntimeGate
  hardware/ + security/
  pipeline.py        one-window firmware-shaped API
  eval/              suite + Evaluator
  demo/              GUI (static/) + HTTP
  edge/              Pi / UDP
  cli.py             uaere …
tests/               pytest
docs/                humans
paper/               claims + manuscript
unity/AHAIF/         visualisation only
configs/             YAML scenarios
```

A 1-second window walks:

`WindowRecord → surprise_admit → (optional) log_mel + env_norm → TrustEngine
→ RuntimeGate → TinyCNN? → CausalReasoner? → Orchestrator + SecureHAL
→ InferenceResult`

The swarm does that **per node**, then fuses confirmations.

## 1. Laws you may not break

1. **No `if energy >` in production.** Only
   `src/uaere/trust/baseline_energy_threshold.py`. `scripts/ci.sh` greps
   for leaks. Detection ROC uses `wake_confidence`.
2. **No silent RNG.** `uaere.seed.rng(seed)` / hydra-less configs with an
   explicit seed.
3. **Equations live in `math/`.** If you change `T(e)`, change
   `docs/MATH.md` in the same commit.
4. **TinyCNN INT8 ≤ 250 KB.**
5. **Security on is the default policy story.** Turning it off is an
   ablation, not a shortcut.
6. **Do not add an LLM, a public dashboard SaaS, or a second physics
   engine in Unity.** Unity *draws* `/api/state`.
7. **Do not open-source until a provisional is filed.**

## 2. How to modify common things

### Add a twin scenario

1. Copy `configs/twin/busy_strait.yaml` to `configs/twin/my_scene.yaml`.
2. Set `sea_state`, `mix` (keys must be `EventClass` values).
3. Either add the same dict to `_builtin_scenarios()` in
   `src/uaere/twin/render.py` **or** keep the YAML; `TwinRenderer("my_scene")`
   loads YAML if the file exists.
4. Run `uaere twin --scenario my_scene --n 8`.
5. Add a line to `docs/USER_GUIDE.md` scenarios table.

### Add a vessel / event class

1. Add a member to `EventClass` in `types.py`.
2. If it is a *vessel event*, add it to `DEEPSHIP_LABELS` **or** keep it
   mapped to `REJECT` via `CLASS_INDEX.get(..., REJECT)` — decide
   consciously. Changing `N_EVENT_CLASSES` retrains the Dirichlet head.
3. Add a synthesizer branch in `twin/sources.py`.
4. Add KG nodes/edges in `kg/ontology.yaml` (`caused_by`, cue bands).
5. Tests: one window of the new class gets a `caused_by` hop.

### Add a DeviceProfile (including a new Pi)

1. Append to `PROFILES` in `hardware/profiles.py` with ISA, MHz, RAM,
   `nj_per_mac_int8`, security bits, `supports_l2/l3`.
2. Cite the datasheet in the module docstring (order-of-magnitude is fine).
3. `tests/unit/test_hardware_security_policy.py` already asserts
   `len(PROFILES) >= 5`. Add a `load_profile("your_id")` smoke test.
4. If it has **no** SE, it is a negative control like `gd32vf103`:
   authenticated placement must fail.

### Add a KG relation or cause

Edit `src/uaere/kg/ontology.yaml` only. Keys: `id`, `type`, `label`,
optional `cues: {f_lo, f_hi, blade_hz}`, `sea_state_min`.
Edges: `{src, rel, dst}` with `rel` in
`radiates | caused_by | confusable_with | requires_env | incompatible_with | observed_as`.
Export: `uaere kg --out artifacts/kg/marine_acoustic.ttl`.

### Change trust weights

`TrustEngine` defaults in `trust/trust_score.py`. Ablations zero a key
(`health`, `env`, `uncertainty`). If you add a term, add it to
`T(e)` in `docs/MATH.md` and to the evaluator
ablation section if it is a claimed contribution.

### Change the gate

`PolicyVector` in `types.py`, staircase in `policy/runtime_gate.py`.
NSGA-II objectives in `policy/objectives.py` — six numbers, last is
**negated** explanation coverage (we minimise).

### Change the GUI

Files: `src/uaere/demo/static/{index.html,style.css,app.js}`.
Server: `src/uaere/demo/server.py` (`GET /api/state`, `POST /api/pause|/api/step`).
Keep the JSON schema stable; Unity depends on it (`docs/UNITY.md`).

### Change evaluation floors

`EvaluationConditions` in `eval/evaluator.py` **and**
`docs/EVALUATION.md` in the **same commit**. Then
`uaere evaluate --out artifacts/eval`.

## 3. Tests you must run

```bash
python -m pytest tests -q
bash scripts/ci.sh
```

Add tests next to the behaviour:

| You changed | Test file |
|-------------|-----------|
| math identity | `tests/unit/test_math.py` |
| twin / seed | `tests/unit/test_twin.py` |
| trust / KG | `tests/unit/test_trust_and_kg.py` |
| HAL / profiles | `tests/unit/test_hardware_security_policy.py` |
| swarm | `tests/unit/test_swarm.py` |
| evaluator | `tests/unit/test_evaluator.py` |
| CLI / pipeline | `tests/integration/test_pipeline_and_cli.py` |

A test that needs DeepShip must **skip** if the folder is missing.

## 4. Style

- Python 3.11+, stdlib + the pins in `pyproject.toml`. No PyTorch.
- Ruff: `E,F,I,UP,B` (line length 100; E501 ignored).
- Public functions: types from `types.py`, no `Any` in new APIs.
- Comments: only non-obvious constraints (e.g. “never oracle health”).

## 5. Debugging a bad AUC

1. `uaere data summarize --n 64` — do you still have both vessels and
   geophony?
2. Print `wake_confidence` vs `event_present` on ten windows.
3. Confirm you did not score ROC on `event_trust` or energy.
4. Confirm `high_sea_state` is still mixed into `run_paper_suite`.
5. Do not lower `min_trust_auc` until you understand the regression.

## 6. What “done” means for a change

- [ ] pytest green
- [ ] `scripts/ci.sh` green (energy-threshold grep)
- [ ] If protocol: `docs/EVALUATION.md` updated
- [ ] If API JSON: `docs/UNITY.md` schema still true
- [ ] If claim language: `paper/complete_specification.md` still maps to
      the new symbol names
