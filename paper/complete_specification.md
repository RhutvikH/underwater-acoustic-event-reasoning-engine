# Complete specification (draft)

**Title of the invention**  
Adaptive hierarchical acoustic intelligence for event-level trust-gated
causal reasoning in underwater acoustic sensor networks

**Inventors**  
Rhutvik Hegde; Kalyani Manoj

**Applicant (intended)**  
Vellore Institute of Technology

**Status**  
Internal draft of a complete specification, prepared for an Indian
provisional followed by a complete specification and optional PCT.
This document is **not** a filed instrument. Do not publish the
accompanying source until a provisional has been accorded a filing date.

**Best mode**  
The source tree `src/uaere/` as of this draft, running 1 s windows at
16 kHz through the digital twin and the software-defined hardware
abstraction. Public name: AHAIF. Package name: `uaere`.

**Correspondence to code**  
Every numbered embodiment below names the Python module that reduces it
to practice. If this specification and the code diverge, the code is the
best mode and this file must be amended before filing.

---

## A. Technical field

The invention relates to underwater acoustic sensor networks, TinyML
inference at energy-constrained edge nodes, calibrated uncertainty
estimation, knowledge-graph causal explanation of acoustic events, and
hardware-aware secure execution of hierarchical inference.

## B. Background

UASNs conventionally wake a classifier when short-time spectral energy
exceeds a predetermined threshold. That rule false-alarms on wind, rain,
and clipping, and misses distant but structurally distinctive sources.

Trust models in distributed sensing score **nodes** (faulty, Byzantine,
compromised). They do not answer whether a particular acoustic window is
a genuine event.

Deep classifiers for ship-radiated noise maximise class accuracy and do
not emit a calibrated event-level admit/reject, a causal explanation, or
an energy policy.

Audio knowledge graphs exist for terrestrial sound and typically require
foundation models. Multi-objective optimisation in WSNs is applied to
routing, not to gating a hierarchy of acoustic inference levels.
Hardware-aware TinyML and secure boot exist separately.

No disclosed system combines: environment-conditioned multi-level
acoustic representations; a dynamic event-level trust and wake-confidence
engine without a predetermined energy threshold; a marine acoustic
knowledge graph with a counterfactual verification; a multi-objective
program that selects an execution level; and a secure hardware
abstraction that maps levels onto heterogeneous device profiles
(ARM Cortex-M, ESP32, RISC-V, FPGA, NPU) including TPM-style
attestation and authenticated TinyML inference.

A claim chart against a 35-paper review is `paper/claim_chart.md`.

## C. Objects of the invention

1. To detect underwater acoustic events without a fixed energy threshold.
2. To score trust at **event** granularity, using spectral, temporal,
   sensor-health, and environmental evidence, with calibrated
   uncertainty.
3. To explain admitted events with a compact marine knowledge graph and
   a counterfactual cue-mask.
4. To spend energy on explanation only when a multi-objective policy
   admits the corresponding hierarchical level.
5. To execute that hierarchy on a hardware abstraction that can refuse
   unattestable devices and tampered firmware.
6. To validate the method in a digital twin of marine acoustics, sensors,
   and collaborative nodes, at TRL-4, without requiring a cruise.

## D. Summary of the invention

In one aspect the invention is a **method** comprising the steps of
independent claim 1 below.

In another aspect the invention is a **system** comprising a digital
twin, one or more edge nodes, and a sink, implementing that method
(independent claim 2).

In a further aspect the invention is a **computer-readable medium**
storing instructions that, when executed, perform the method.

Preferred embodiments include an evidential Dirichlet head; FiLM /
instance-norm conditioned on sound-speed and sea-state; NSGA-II search
of the Pareto set of policies; collaborative neighbour-wake in an
uncertain trust band; five illustrative device profiles; and a secure
HAL providing secure boot, a secure element, TPM PCR quotes,
HMAC-authenticated inference, and AES-GCM communication.

---

## E. Brief description of the drawings (textual)

**Figure 1 — Hierarchical pipeline (best mode).**

```
Digital twin  →  observation x, state s, health estimate h
                     │
 L0 always-on DSP φ0 (rFFT moments)
                     │ logistic of the φ0 *vector* (not energy ≷ θ)
 L1 log-Mel + env-norm + evidential Dirichlet
                     │ C_wake, T(e), u_a, u_e
         Multi-objective gate π*(T)
                     │
          L2 TinyCNN     L3 MAG + CF     L4 collab TX
                     │
         Orchestrator + Secure HAL → device profile d
```

**Figure 2 — Policy staircase** on \(T(e)\) with thresholds
\(\tau_1<\tau_2<\tau_3\) and a collaborative band
\([\tau_{\mathrm{collab,lo}},\tau_{\mathrm{collab,hi}}]\).

**Figure 3 — Knowledge graph fragment.** VesselClass —radiates→ EventType
—caused_by→ VesselClass; Environment incompatible_with selected events;
SensorArtifact caused_by clipping / dropout / gain wander.

**Figure 4 — Pareto front** of \(F(\pi)\) and the Chebyshev knee used at
runtime.

These figures are embodied by `src/uaere/pipeline.py`,
`policy/runtime_gate.py`, `kg/ontology.yaml`, and `eval/suite.py`.

---

## F. Detailed description

### Embodiment 1 — Signals and twin (best mode of the laboratory)

Windows are \(N=f_s\cdot 1\,\mathrm{s}\) samples, \(f_s=16\,\mathrm{kHz}\)
working rate (DeepShip native 32 kHz is resampled by the adapter). The
twin (`src/uaere/twin/`) produces \(x\), oracle health \(h^\star\), and
environment \(s\):

- Mackenzie nine-term sound speed \(c(T,S,z)\).
- Thorp absorption \(\alpha(f)\) dB/km.
- Knudsen-style ambient PSD lifted by sea-state and turbulence.
- Spherical spreading plus a surface image path with a \(\pi\) phase
  inversion and a configurable surface-loss.
- Hydrophone bandpass, equivalent noise, ADC, optional faults.
- A noisy health estimate \(h\) shown to the trust engine
  (`noisy_health_estimate`).
- Optional ice-keel scatterer (embodiment, **not** an independent
  claim).

Collaborative sensing energy uses a range-dependent µJ/bit table of
Micro-Modem class (`twin/network.py`).

### Embodiment 2 — Adaptive multi-level representation

\(\phi_0\): energy, ZCR, centroid, bandwidth, flatness, residual
(`representation/l0_dsp.py`). L0→L1 admit is
\(\sigma(w^\top\phi_0+b)\) with default \(w\) using shape features. This
is expressly **not** `if energy > θ`.

\(\phi_1\): log-Mel, 32 bands (`l1_tf.py`), instance-norm and FiLM from
\(s\) (`env_norm.py`), noise-PSD EMA with time constant \(\alpha\)
(config default 0.05).

\(\phi_2\): depthwise-separable CNN (`encoder.py`) with an owned IR
(`classify/ir.py`) so MAC counts do not depend on TensorFlow Lite. INT8
size must remain \(\le 250\,\mathrm{KB}\); the reference net is 546 B.

### Embodiment 3 — Dynamic trust (no predetermined energy threshold)

Dirichlet strengths \(\alpha=f_\theta(\phi_1)+1>0\), or the equivalent
lift \(\alpha=1+\tau\hat p\) from a fitted multinomial logistic
(`trust/evidential_head.py`, `classify/train.py`). Mean, aleatoric and
epistemic uncertainties, and \(C_{\mathrm{wake}}=1-\hat p_{\mathrm{reject}}\)
follow Sensoy-style identities (`math/dirichlet.py`).

Health consistency \(\kappa(h)\) and environmental consistency
\(\rho(s,x)\) (low-frequency contrast penalised by sea-state and
turbulence) enter

\[
T(e)=\sigma(w_c C_{\mathrm{wake}}+w_h\kappa+w_s\rho-w_u(u_a+u_e)+b).
\]

Default weights are in `TrustEngine`. Ablation zeros individual weights.
Post-hoc temperature / vector scaling lives in `trust/calibrate.py`.

The **only** module permitted to compare energy to a constant is
`trust/baseline_energy_threshold.py`, and it is an evaluation competitor,
not part of the claimed detector. Continuous integration greps the tree
for leaked `if energy >`.

### Embodiment 4 — Marine acoustic knowledge graph and counterfactual

The graph is authored in `kg/ontology.yaml` (20 nodes, 21 edges in the
best mode) and loaded as a NetworkX MultiDiGraph with a Turtle exporter.
Relations and cue bands are as in that file.

`causal/reasoner.py` ranks `caused_by` neighbours by cue-band energy
fraction times an environment prior (`sea_state_min`), renders a chain
and a sentence, and verifies the top cause by attenuating its cue band
by 20:1. Unverified causes are listed on the `Explanation` object. The
inventors do not claim do-calculus identification.

### Embodiment 5 — Multi-objective policy and runtime gate

Decision vector \(\pi=(\tau_1,\tau_2,\tau_3,m,d)\) plus a collaborative
band and an authenticated-inference flag (`types.PolicyVector`).
Objectives \(F(\pi)\) (`policy/objectives.py`): miss, false-alarm,
joules/window, p99 latency, ECE, negated explanation coverage.

Search: NSGA-II, seed-locked, implemented in `policy/nsga2.py` without a
third-party MOEA library. Runtime: Chebyshev knee
(`policy/runtime_gate.chebyshev_select`) and staircase
(`RuntimeGate.admit`).

Energy identity \(E=c_0+p_1c_1+p_2c_2+p_3c_3\) and Proposition 1 are
`math/energy.py` and `math/bounds.py`.

### Embodiment 6 — Hardware abstraction and secure HAL

Device profiles (`hardware/profiles.py`) as in the table of the
manuscript §4.6. `Orchestrator.place(level)` enforces:

- authenticated flag ⇒ `has_secure_element ∧ has_secure_boot`;
- L2/L3 support flags (the RISC-V profile does not host L3);
- otherwise the named profile is used; `cheapest_attestable` is provided.

Secure HAL (`security/hal.py`):

- **Secure boot.** SHA-256(firmware) must match the root hash and the
  SE signature. Mismatch raises `TamperError` (fail-closed).
- **Secure element.** Ed25519 key never exported; `sign` / `verify`.
- **TPM.** PCR extend of the model digest; `quote(nonce)`.
- **Authenticated TinyML.** `HMAC-SHA256(k_session, SHA256(model_id‖x‖y))`.
- **Secure communication.** AES-GCM; session key 32 bytes.

These are protocol simulations. The claims are on the *method of
requiring* them as a placement constraint, not on a new cryptosystem.

### Embodiment 7 — End-to-end pipeline

`AhaifPipeline.infer(WindowRecord) -> InferenceResult`
(`pipeline.py`) is the firmware-shaped entry point: L0 shape-gate, L1
trust, `RuntimeGate`, optional CNN and reasoner, orchestrator placement,
HMAC tag. CLI `uaere eval|twin|train|data|kg` never raises
`NotImplementedError` on a supported path.

### Embodiment 8 — Public-data adapters

`DeepShipAdapter` and `ShipsEarAdapter` window class-folder WAVs at 1 s
hop 0.5 s, resample to 16 kHz, and split later-in-time as test. They are
optional. The twin is sufficient for TRL-4 reduction to practice.

---

## G. Claims

**Claim 1 (method, independent).**  
A computer-implemented method for underwater acoustic event reasoning,
comprising:

(a) extracting a plurality of environment-conditioned acoustic
representations of increasing computational cost from a hydrophone
window and an environment state;

(b) computing a calibrated event-level wake-trust score from said
representations and a sensor-health estimate, **without applying a
predetermined spectral-energy threshold as a wake rule**;

(c) selecting an execution level from a hierarchy by solving, or by
applying a precomputed solution of, a multi-objective program whose
objectives include at least a miss rate, a false-alarm rate, and an
energy cost, and optionally latency, calibration error, and explanation
coverage;

(d) when the selected level includes causal reasoning, generating an
explanation chain over a marine acoustic knowledge graph and verifying
at least one candidate cause by masking spectral cues associated with
that cause and testing whether the masked observation continues to
support the same event type;

(e) executing the selected level on a hardware abstraction that maps
levels to device profiles and that, when a security constraint is
active, requires authenticated inference and refuses placement on a
profile lacking a secure element or secure boot.

**Claim 2 (system, independent).**  
A system comprising:

(a) an underwater acoustic digital twin configured to simulate marine
ambient noise, acoustic propagation, hydrophone and analogue-to-digital
behaviour including faults, residual node energy, and optional
collaborative transmissions;

(b) at least one edge node configured to perform the method of claim 1;
and

(c) a sink configured to verify an authenticated-inference tag and to
decrypt an authenticated-encrypted payload carrying a classification,
an explanation, or an attestation quote.

**Claim 3.** The method of claim 1, wherein the wake-trust score is
obtained from a Dirichlet evidential distribution over event classes
including a reject class, and wherein wake confidence is one minus the
reject mass.

**Claim 4.** The method of claim 3, wherein aleatoric and epistemic
uncertainties derived from said Dirichlet distribution are inputs to
the event-level trust score that feeds step (c).

**Claim 5.** The method of claim 1, wherein at least one representation
is instance-normalised with affine parameters that are a function of
the environment state, said state including a sound-speed value and a
sea-state index.

**Claim 6.** The method of claim 1, wherein a first, always-on
representation comprises spectral shape descriptors in addition to
energy, and a logistic of that vector, not a comparison of energy to a
constant, admits computation of a second representation.

**Claim 7.** The method of claim 1, wherein the multi-objective program
is solved offline by a Pareto evolutionary algorithm, including NSGA-II,
and a scalarisation, including weighted Chebyshev, selects a policy at
runtime.

**Claim 8.** The method of claim 1, further comprising transmitting a
neighbour-wake message when the event-level trust lies in a predetermined
uncertain band, the transmission energy being accounted in the energy
objective.

**Claim 9.** The method of claim 1, wherein the knowledge graph includes
node types for event types, vessel classes, biological taxa,
environmental processes, and sensor artefacts, and relations including
caused-by, radiates, confusable-with, requires-environment, and
incompatible-with.

**Claim 10.** The method of claim 1, wherein device profiles include at
least two of: an ARM Cortex-M profile, an ESP32 profile, a RISC-V
profile, an FPGA profile, and a neural-processing-unit profile, each
profile supplying at least a MAC energy, a RAM budget, and a security
capability bitmap.

**Claim 11.** The method of claim 1, wherein authenticated inference
comprises an HMAC over a digest of a model identifier, the hydrophone
window, and the inference output, keyed by a session key unwrapped by a
secure element.

**Claim 12.** The method of claim 1, wherein secure boot comprises
verifying a signature over a hash of a firmware image against a root
key and refusing execution on mismatch.

**Claim 13.** The method of claim 1, wherein a simulated TPM extends a
platform configuration register with a digest of a model and emits a
quote over a nonce for verification by the sink of claim 2.

**Claim 14.** The method of claim 1, wherein communication of inference
results uses authenticated encryption, including AES-GCM.

**Claim 15.** The method of claim 1, wherein the sensor-health estimate
includes at least one of clipping, clock drift, analogue-to-digital
resolution, and a fault class selected from bias, dropout, and gain
wander, and wherein a consistency score of that estimate down-weights
event trust.

**Claim 16.** The method of claim 1, wherein environmental consistency
scores support for a non-ambient cause by comparing low-frequency and
high-frequency energy and reducing the score as sea-state or turbulence
increases.

**Claim 17.** The system of claim 2, wherein the digital twin generates
labelled windows whose generating cause identifiers are the same
identifiers used as gold for explanation evaluation.

**Claim 18.** The method of claim 1, performed entirely in simulation on
public acoustic corpora, digital-twin synthetic corpora, or both.

**Claim 19 (computer-readable medium).**  
A non-transitory computer-readable medium storing instructions that,
when executed by a processor, cause the processor to perform the method
of any one of claims 1 and 3–18.

---

## H. Abstract of the disclosure

A hierarchical method and system for underwater acoustic event reasoning
computes a calibrated event-level wake-trust score from
environment-conditioned representations and sensor health without a
predetermined energy threshold, selects an execution level by a
multi-objective program over miss, false-alarm and energy, explains
admitted events on a marine acoustic knowledge graph with a
counterfactual cue-mask, and executes the selected level on a hardware
abstraction that can require authenticated TinyML inference and refuse
unattestable or tampered devices. A digital twin provides TRL-4
validation.

---

## I. Industrial applicability

Passive acoustic monitoring of shipping, marine mammals, and illegal
extraction; energy-constrained UASN nodes; laboratory certification of
edge acoustic firmware before a cruise.

## J. Filing checklist (not part of the specification)

1. Keep the git remote private.
2. Freeze `git rev-parse HEAD` and dataset SHA-256 into
   `paper/experiment_protocol.md`.
3. Have institute IP cell / counsel convert this draft into the official
   Form 2 complete specification; **do not file this markdown as-is**.
4. File provisional, then complete, then (optional) PCT.
5. Only then open-source or arXiv the manuscript
   (`paper/manuscript.md`).
