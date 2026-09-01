# Event-Level Trust-Gated Causal Reasoning for Energy-Constrained Underwater Acoustic Sensor Networks

**Working manuscript (TRL-4, simulation).** Do not post to arXiv or a public
GitHub until a patent provisional has been filed.

Rhutvik Hegde (23BCI0079), Kalyani Manoj (23BCI0081)  
Faculty guide: Dr. S. M. Satapathy  
Vellore Institute of Technology — B.Tech. Capstone  
Target venues: *IEEE Internet of Things Journal*; *Expert Systems with Applications*  
SDG 9 (Industry, Innovation and Infrastructure), SDG 14 (Life Below Water)  
Technology readiness: experimental proof of concept (TRL-3) → validated in lab (TRL-4)

---

## Abstract

Underwater acoustic sensor networks (UASNs) typically decide whether to
“wake” on an acoustic event using a fixed spectral-energy threshold, and
existing trust models in this space evaluate whether a *sensor node* is
compromised or faulty rather than whether an *individual acoustic event* is
genuine. We address that event-level gap with the Adaptive Hierarchical
Acoustic Intelligence Framework (AHAIF): (i) environment-conditioned
multi-level representations; (ii) a Dynamic Acoustic Trust and Wake
Confidence Engine that emits a calibrated per-event score from an evidential
Dirichlet head, sensor-health consistency, and environmental consistency,
without a predetermined energy threshold; (iii) a compact Marine Acoustic
Knowledge Graph that produces a human-readable causal chain, verified by a
feature-mask counterfactual; (iv) a multi-objective escalation policy,
searched with NSGA-II, that invokes expensive causal reasoning only for
high-trust events; and (v) a hardware-aware secure execution abstraction
covering simulated ARM Cortex-M, ESP32, RISC-V, FPGA, and NPU profiles,
with secure boot, a secure element, TPM-style attestation, HMAC-authenticated
TinyML inference, and AES-GCM communication. All modules are evaluated in
an underwater acoustic digital twin at TRL-4. On a seed-locked twin
replay, event-level wake confidence achieves ROC-AUC 0.916 against 0.069
for a Youden-tuned energy threshold (the latter false-alarms on high-energy
geophony); gated execution uses 0.208 mJ/window versus 6.66 mJ/window for
always-on causal reasoning at matched miss rate; every true event receives
a KG chain whose top cause matches the generating ontology; tampered
firmware is refused; and a non-attestable RISC-V profile is refused for
authenticated L2 placement.

**Index terms.** Underwater acoustic sensor networks, event-level trust,
evidential deep learning, knowledge graphs, causal explanation, TinyML,
digital twin, multi-objective optimisation, secure inference.

---

## 1. Introduction

Passive acoustic monitoring at the edge is energy-dominated. A node that
classifies and explains every spectrogram dies; a node that waits for
band-energy to cross a hand-set threshold wakes on breaking waves, rain,
and clipping, and sleeps through a distant tug. The first failure wastes
joules. The second misses the event the network was deployed to catch.

The literature on UASN *trust* does not solve this. Cognition-difference
and Bayesian fusion methods score **nodes** (is this hydrophone
compromised?). DeepShip-style networks score **classes** (cargo versus
tanker) with no notion of whether the window should have been admitted at
all, and with no causal explanation. Multi-objective protocols such as
MO-CRP optimise **routing**. Digital twins such as DTNA / DTEAR allocate
**network resources**. iKnow-audio builds a knowledge graph for
**terrestrial** sound with foundation-model backends that will not fit an
STM32. None of these gates an expensive marine causal reasoner on a
calibrated *event-level* trust score under a hardware-aware security
constraint.

This paper’s product is therefore not “a better classifier.” It is a
**hierarchy** whose discrete decisions are the output of a mathematically
defined multi-objective program. We call it AHAIF. Four capstone
objectives, all demonstrated with numbers, follow the review in
`docs/review_1.pdf`:

1. A calibrated event-level trust model versus a fixed-threshold baseline
   (ROC, false-alarm behaviour, ECE).
2. A compact causal knowledge graph and readable explanation chains.
3. A confidence-gated escalation policy with an accuracy–energy trade-off
   against always-on reasoning.
4. End-to-end validation in simulation on public-protocol data adapters
   (DeepShip-shaped) and a digital twin, consistent with TRL-4.

Six non-negotiable engineering constraints — hierarchical multi-objective
optimisation, dynamic trust plus adaptive representation, MAG + causal
reasoning, a digital twin, a secure hardware abstraction, and
publication-grade analysis — are satisfied by the *same* codebase, not by
parallel prototypes.

**Contributions.**

- Event-level wake trust \(T(e)\) and \(C_{\mathrm{wake}}\), with no
  constant energy threshold in the live detector.
- Adaptive multi-level features \(\phi_0,\phi_1,\phi_2\) conditioned on a
  twin environment state \(s\) (SSP, sea-state, turbulence, SNR).
- A marine acoustic KG plus a cheap counterfactual band-mask; we do not
  claim Pearl identification.
- NSGA-II over \(\pi=(\tau_1,\tau_2,\tau_3,m,d)\) with a Chebyshev knee at
  runtime, and a proven energy strict-inequality for \(p_3<1\).
- Software-defined device profiles (Cortex-M4, ESP32-S3, RV32IMC, iCE40,
  Ethos-U55) and a protocol-level secure HAL.
- A reproducible `uaere eval --suite paper` command that writes every
  table in this manuscript.

---

## 2. Related work and gap

A 35-paper review (`docs/review_1_lit_rev.pdf`, charted in
`paper/claim_chart.md`) clusters as follows.

**Digital twins for UASNs** (Song et al. DTNA; DTEAR) separate local
optimisation from global synchronisation, or couple twins to depth-based
routing. They are network controllers. They do not reason about a blip.

**Node-level trust** (cognition-difference dynamic trust) reweights
Bayesian fusion when a *sensor* looks faulty. An honest hydrophone that
heard wind still receives a high node score.

**Passive acoustic classification** (Kammegne et al.; DeepShip SCAE;
dense CNNs; Swin mask-modelling) maximises accuracy on ship-radiated
noise. Transformers will not run on 128 KB RAM. None emit event-level
trust or a causal chain.

**Uncertainty-aware sonar** (Bayesian DL, VI-PANN) estimates aleatoric
and epistemic uncertainty on the *label*. We reuse the evidential
Dirichlet construction (Sensoy et al., NeurIPS 2018) but **feed
uncertainty into the gate**, and we do it with a TinyML-sized head.

**Audio knowledge graphs** (iKnow-audio) structure terrestrial sound with
large models. Marine event types, vessel classes, geophony, and sensor
artefacts are absent; so is energy-aware escalation.

**TinyML acoustic monitors** (Arduino Nano 33 BLE, ESP-Now PAWSNs) prove
INT8 inference at ~0.45 W on land, with handcrafted MFCCs that degrade in
multipath. They have no marine KG and no secure HAL.

**Multi-objective WSN protocols** (MO-CRP / NSGA-II cluster routing)
optimise energy, load, and coverage of *packets*. We optimise miss,
false-alarm, joules, latency, ECE, and explanation coverage of
*inference levels*.

**Gap (restated).** Nobody scores the event, explains it on a marine KG,
and pays for that explanation only when a calibrated trust score and a
hardware/security policy say so.

---

## 3. Problem formulation

Let \(x\in\mathbb{R}^{N}\) be a 1 s hydrophone window at \(f_s=16\,\mathrm{kHz}\).
Let \(s\) be the environment state (temperature, salinity, depth, sea-state,
SNR, turbulence, sound speed) and \(h\) a (noisy) sensor-health estimate
(noise floor, clipping, drift, ADC bits, fault class). A label
\(y\in\{0,1\}\) indicates vessel-event presence; a class
\(c\in\{\mathrm{cargo},\mathrm{passenger},\mathrm{tanker},\mathrm{tug},\mathrm{reject}\}\)
is the Dirichlet support.

A **policy** \(\pi=(\tau_1,\tau_2,\tau_3,m,d)\) maps a scalar event trust
\(T(e)\) to an execution level \(L\in\{0,1,2,3,4\}\) and a device profile
\(d\). The program is

\[
\min_\pi F(\pi)=\big(f_{\mathrm{miss}},\,f_{\mathrm{fa}},\,E,\,L_{p99},\,\mathrm{ECE},\,-X\big)
\]

subject to energy and latency budgets, authenticated inference when the
security flag is set, and explanation coverage \(X\) on true events.

**Proposition 1.** Write \(E_{\mathrm{compute}}=c_0+p_1 c_1+p_2 c_2+p_3 c_3\)
with \(c_k>0\). For any always-on L3 policy (\(p_1=p_2=p_3=1\)) and any
gated policy with \(p_3<1\), \(E(\pi)<E(\pi_{\mathrm{on}})\). Implemented
and unit-tested as `gated_energy_strictly_less`.

**Proposition 2.** If \(T\) is a consistent estimator of the event
posterior, the extra miss rate of a threshold \(\tau_3\) relative to the
Bayes threshold is at most the posterior mass between them
(`gated_miss_bound`).

There is **no** predetermined energy threshold in this program. Any
\(\tau_k\) is a *decision variable* of \(\pi\), not a sensor constant.

---

## 4. The AHAIF pipeline

### 4.1 Adaptive multi-level representation

- \(\phi_0(x,s)\): band energy, zero-crossing rate, spectral centroid,
  bandwidth, flatness, noise-residual. \(O(N\log N)\) via rFFT. Always
  on. A logistic of the **vector** (default weights use flatness and
  residual, not energy alone) admits L1. Near-silence sleeps.
- \(\phi_1(x,s)\): 32-band log-Mel (STFT 512, hop 256), instance-normalised
  and FiLM-modulated by \(\gamma(s),\beta(s)\). A running PSD EMA
  \(\hat n_t=(1-\alpha)\hat n_{t-1}+\alpha\,\mathrm{PSD}(x_t)\) updates
  on non-event frames. CQT is an ablation (`log_cqt_proxy`).
- \(\phi_2(x,s)\): depthwise-separable TinyCNN, 546 parameters, 484 kMAC /
  window, 546 B INT8 — under the 250 KB budget by two orders of magnitude.

### 4.2 Dynamic Acoustic Trust and Wake Confidence Engine

An evidential head maps Mel statistics to Dirichlet strengths. In the
reference implementation the head is a class-balanced multinomial logistic
lifted by \(\alpha=1+\tau\hat p\) (\(\tau=20\)), which preserves the
Dirichlet mean while giving a concentration that shrinks as the classifier
becomes uncertain. Then

\[
\hat p=\frac{\alpha}{\sum_i\alpha_i},\quad
u_{\mathrm{e}}=\frac{K}{\sum_i\alpha_i},\quad
u_{\mathrm{a}}=1-\max_k\hat p_k,\quad
C_{\mathrm{wake}}=1-\hat p_{\mathrm{reject}}.
\]

Health consistency \(\kappa(h)\in[0,1]\) down-weights clipping, faults,
shallow ADCs, and large clock drift. Environmental consistency
\(\rho(s,x)\) is support for a *non-ambient* cause: low-frequency-heavy
spectra in calm water score high; high-frequency geophony in high
sea-state scores low — the opposite of an energy wake. Event trust is

\[
T(e)=\sigma\big(w_c C_{\mathrm{wake}}+w_h\kappa(h)+w_s\rho(s,x)-w_u(u_{\mathrm{a}}+u_{\mathrm{e}})+b\big).
\]

**Detection ROC uses \(C_{\mathrm{wake}}\).** The gate uses \(T(e)\).
Confusing the two is how a storm fools an energy detector and how a
well-calibrated reject class does not.

The energy-threshold baseline (`EnergyThresholdBaseline`) sweeps STFT-band
energy on the train split, picks Youden’s \(J\), and is imported **only**
by eval.

### 4.3 Marine acoustic KG and causal reasoner

A hand-authored graph \(G=(V,E)\) with \(|V|=20\), \(|E|=21\) (Turtle
export via `uaere kg`) has node types EventType, VesselClass, Taxon,
Environment, SensorArtifact and relations `radiates`, `caused_by`,
`confusable_with`, `requires_env`, `incompatible_with`, `observed_as`.
Each cause carries spectral cues \((f_{\mathrm{lo}},f_{\mathrm{hi}})\)
and optional `sea_state_min`.

For a classified event \(\hat y\) the reasoner ranks `caused_by`
neighbours by cue-band energy fraction times an environment prior, emits a
3–6 step chain and a one-sentence gloss, and runs the counterfactual

\[
\mathrm{CF}(c,\hat y)=\mathbf{1}\big[\text{energy in cues of }c\text{ drops after a 20:1 mask}\big].
\]

This is a feature-mask verification, not a structural causal model
identification procedure. We state that limitation in the claims.

### 4.4 Escalation policy

`RuntimeGate` is a staircase on \(T(e)\):

| Condition | Level |
|-----------|-------|
| \(T<\tau_1\) | L0 sleep |
| \(\tau_1\le T<\tau_2\) | L1 trust only |
| \(\tau_2\le T<\tau_3\) | L2 TinyCNN |
| \(T\ge\tau_3\) | L3 KG |
| \(\tau_{\mathrm{collab,lo}}\le T\le\tau_{\mathrm{collab,hi}}\) | L4 TX neighbour-wake |

Offline, a compact NSGA-II (population 16, 8 generations in the paper
suite; no pymoo dependency) evaluates \(F(\pi)\) on the twin replay.
Runtime selects the weighted-Chebyshev knee of the non-dominated front.

### 4.5 Digital twin (the TRL-4 laboratory)

Five submodels, each a pure function:

1. **Environment.** Mackenzie (1981) sound speed; Thorp absorption;
   Knudsen/Wenz wind-noise PSD; turbulence as a high-frequency lift
   (flow-acoustic stress without a new module).
2. **Propagation.** Spherical spreading + absorption + a 2-path surface
   image with a \(\pi\) phase flip. Urick-class, not Bellhop.
3. **Sensor.** 20 Hz–7 kHz hydrophone, ADC quantisation, clipping, clock
   drift, faults {bias, dropout, gain wander}. The trust engine sees a
   *noisy* health estimate, never the oracle (except in an ablation).
4. **Network energy.** WHOI Micro-Modem-class µJ/bit versus range;
   residual battery on each node.
5. **Collaborative sensing.** Optional L4 TX.

Source signatures are procedural (blade-rate harmonics for tugs, reddened
broadband for tankers, high-frequency geophony, click artefacts) so the
repository contains no copyrighted recordings. Distant vessels are mixed
at low SNR; storms are mixed loud. That is how an energy threshold is
forced to reveal its failure mode.

### 4.6 Hardware-aware orchestration and secure HAL

Non-negotiable 5 allows “hardware simulators **or software-defined
hardware emulation**.” We do the latter.

| Profile | ISA | Security | nJ/MAC INT8 |
|---------|-----|----------|-------------|
| `stm32l476` | Cortex-M4 @ 80 MHz | SE + secure boot | 2.5 |
| `esp32s3` | Xtensa LX7 @ 240 MHz | SE + secure boot | 1.8 |
| `gd32vf103` | RV32IMC @ 108 MHz | **none** (negative control) | 3.2 |
| `ice40up5k` | FPGA DSP | bitstream auth | 1.1 |
| `ethos_u55` | M55 + NPU | TrustZone + SE + TPM | 0.15 |

The orchestrator refuses to place L2/L3 on a non-attestable profile when
`require_authenticated=True`. That experiment is for the *orchestration*
claim, not a cryptography claim.

Secure HAL (protocol simulation): staged SHA-256 boot over an
Ed25519-signed firmware root; SE-held model key (never exported); TPM PCR
extend of the model hash and `quote(nonce)`; HMAC-SHA256 authenticated
inference over `hash(model_id ‖ window ‖ output)`; AES-GCM payloads with
a 256-bit session key.

---

## 5. Experimental protocol

**Command.** `uaere eval --suite paper --n-windows 160 --seed 0`  
**Split.** Time-aware: later windows are test (`time_aware_split`).  
**Train / val / test (reference).** 138 / 32 / 43 windows, plus a
high-sea-state robustness draw of 80 windows.  
**Seeds.** `{0}` in this draft; camera-ready `{0,1,2}` with Holm–Bonferroni
on the family (AUC, ECE, energy), \(\alpha=0.05\).  
**Primary detection score.** \(C_{\mathrm{wake}}\).  
**Primary baseline.** Youden energy threshold on train.  
**Gating baseline.** Always-on L3 (`tau1=tau2=tau3=0`).  
**Code identity tests.** Dirichlet \(\alpha>1\), \(\sum\hat p=1\), ECE in
\([0,1]\), Proposition 1, twin determinism under seed, tamper-closed boot,
gd32 refusal.

Adapters for DeepShip and ShipsEar are implemented but were **not** used
for the numbers below (full DeepShip is author-gated). Those numbers must
not be quoted as ocean-deployment performance.

---

## 6. Results

### 6.1 Objective 1 — event-level trust versus energy

| Detector | ROC-AUC | ECE | MCE | Brier |
|----------|---------|-----|-----|-------|
| AHAIF \(C_{\mathrm{wake}}\) | **0.916** | **0.328** | 0.328 | 0.358 |
| Energy threshold (Youden) | 0.069 | 0.571 | — | — |

The energy AUC below 0.5 is not a plotting error. High-energy windows in
the mix are predominantly *geophony* (sea-state 5), labelled
`event_present=0`. A fixed energy wake therefore ranks negatives above
positives — exactly the operational failure the introduction claimed.
Trust uses spectral shape and a reject class, so distant tugs (low energy,
low-frequency harmonics) outrank storms.

On a pure `high_sea_state` draw, trust AUC is 0.978.

Calibration (ECE 0.328) is the weakest Objective-1 number. The Dirichlet
lift is not yet temperature-scaled at eval time; that is the first
camera-ready fix (`trust/calibrate.py` already implements a grid-search
temperature).

### 6.2 Objective 2 — explanations

| Quantity | Value |
|----------|-------|
| KG nodes / edges | 20 / 21 |
| True events in test | 21 |
| Ontology hit-rate (top cause \(=\) generating `cause_id`) | **1.00** |
| Counterfactual-verified true events | 21 |

A typical chain:

`observation cues → event.tug_pass → cause.vessel.tug → env sea_state=2 → cf=verified`

Sentence: *“tug is the top cause of tug pass in sea-state 2; counterfactual
verified.”*

This hit-rate is expected on twin-synthetic data whose labels *are* the
ontology. It is a correctness check of the reasoner, not a claim of
open-ocean explanation quality. A 50-chain author faithfulness study
(Cohen’s \(\kappa\)) is specified in the protocol and not yet run.

### 6.3 Objective 3 — gated energy

| Policy | Energy (J / window) | Miss | Pareto size |
|--------|---------------------|------|-------------|
| Chebyshev knee \(\tau\approx(0.60,0.70,0.83)\) | **\(2.08\times10^{-4}\)** | 0.488 | 16 |
| Always-on L3 | \(6.66\times10^{-3}\) | 0.488 | — |

Miss is **matched by construction**: both policies use the same
\(C_{\mathrm{wake}}\ge 0.5\) detector; the gate only skips L2/L3
*compute*. Energy drops by a factor of ~32. Proposition 1 holds on the
analytic costs. A 10 Wh cell at 1 Hz inference lasts ~2003 days gated
versus ~63 days always-on (Peukert exponent = 1 in this draft).

The absolute miss of 0.488 is a small-test-set / decision-threshold
artefact (\(0.5\) on \(C_{\mathrm{wake}}\) is not the Youden point of
trust). Camera-ready tables should report the full ROC, not a single
operating point, for miss.

### 6.4 Objective 4 — hardware, security, complexity

**Complexity (computed, not guessed).**

| Stage | Big-O | FLOPs / window | Peak RAM |
|-------|-------|----------------|----------|
| L0 rFFT + moments | \(O(N\log N)\) | \(1.12\times10^6\) | 256 kB |
| L1 Mel + IN | \(O(N_{\mathrm{fr}} N_{\mathrm{fft}} B)\) | \(9.99\times10^5\) | 7.8 kB |
| L2 TinyCNN | MAC tracer | \(4.84\times10^5\) MAC | 546 B INT8 |
| L3 KG BFS | \(O(\|V\|+\|E\|)\) | 41 | 2.6 kB |

**Energy per level (mJ / window), software-defined profiles.**

| Device | L0 | L1 | L2 | L3 |
|--------|----|----|----|----|
| stm32l476 | 0.208 | 5.41 | 6.45 | 6.45 |
| esp32s3 | 0.157 | 4.09 | 4.88 | 4.88 |
| gd32vf103 | 0.265 | 6.89 | 8.21 | 8.22 |
| ice40up5k | 0.130 | 3.37 | 4.02 | 4.02 |
| ethos_u55 | 0.012 | 0.313 | 0.373 | 0.373 |

**Security.** Tampered firmware is refused (`tamper_caught=true`).
Authenticated L2 placement on `gd32vf103` is refused
(`gd32_refused_authenticated_l2=true`). Honest HMAC verifies; AES-GCM
round-trips.

**Ablation note.** Zeroing \(w_h\) or \(w_s\) did not move detection AUC
in this run, because ROC is computed on \(C_{\mathrm{wake}}\) (the
Dirichlet reject mass), not on \(T(e)\). Health and environment terms
affect *gating* via \(T(e)\). A camera-ready ablation must report both
the ROC of \(C_{\mathrm{wake}}\) and the energy of the resulting gate
when those weights are zeroed.

### 6.5 Statistical protocol (as specified, partially executed)

McNemar on paired correctness, Wilcoxon on ECE/energy across seeds,
1000-bootstrap CIs, Holm–Bonferroni on {AUC, ECE, energy}. The reference
JSON is a single seed. Repeat with `--seed 1` and `--seed 2` before
submission; `EvalConfig.seeds` is already `(0,1,2)`.

---

## 7. Limitations

1. **Twin, not ocean.** Procedural sources plus Urick-class propagation
   do not replace DeepShip, ShipsEar, or a cruise. TRL-4 is the claim.
2. **Energy AUC is scenario-dependent.** On a vessel-only mix, a tuned
   energy threshold will look less foolish. The high-sea-state mix is the
   operationally honest stress, and we will also report the calm mix.
3. **Compact KG.** Twenty nodes cannot represent a soundscape. Cue
   frequencies are author-specified.
4. **Analytical hardware.** Datasheet-order nJ/MAC, not Renode cycles,
   not silicon. A Cortex-M L0 cycle check is listed as optional.
5. **ECE.** 0.328 needs post-hoc scaling and a reliability diagram in the
   camera-ready PDF.
6. **TinyCNN** is a numpy nested loop used for MAC counts and a second
   head; the reported detector is the logistic–Dirichlet lift on Mel
   statistics, which is what actually fits L1 on a Cortex-M.
7. We do not claim ice-keel link prediction, PINN AUV control, GAN
   denoisers, or MARL swarms.

---

## 8. Conclusion

AHAIF replaces a wake-word with a hierarchy whose only discrete decisions
come from a multi-objective policy on calibrated event-level trust. In a
seed-locked digital twin, that trust ranks vessel events above storm
noise (AUC 0.916 vs 0.069), every true event is explained on a marine
KG, gated execution cuts modelled energy by ~32× at matched miss, and a
secure HAL refuses both tampered firmware and unattestable silicon. The
implementation is the manuscript: equations live in `src/uaere/math/`,
the independent claim lives in `paper/complete_specification.md`, and
`uaere eval --suite paper` regenerates every table.

---

## References (selected)

1. Irfan et al., “DeepShip: An underwater acoustic benchmark dataset…,”
   *Expert Systems with Applications*, 2021.
2. Sensoy, Kaplan, Kandemir, “Evidential deep learning to quantify
   classification uncertainty,” NeurIPS 2018.
3. Mackenzie, “Nine-term equation for sound speed in the oceans,”
   *J. Acoust. Soc. Am.*, 1981.
4. Thorp, “Analytic description of the low-frequency attenuation
   coefficient,” *J. Acoust. Soc. Am.*, 1967.
5. Song et al., “A Digital Twin-based Intelligent Network Architecture
   for UASNs,” arXiv:2410.20151 / 2408.06208, 2024.
6. Traer, Norman-Haignere, McDermott, “Causal inference in environmental
   sound recognition,” 2021.
7. iKnow-audio, EMNLP 2025.
8. Full 35-paper table: `docs/review_1_lit_rev.pdf` and
   `paper/claim_chart.md`.

---

## Reproducibility appendix

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uaere eval --suite paper --n-windows 160 --seed 0 --out artifacts/paper
python -m pytest tests -q
```

Operator manual: `docs/USER_GUIDE.md`.  
Patent specification: `paper/complete_specification.md`.  
Equation list: `paper/formulation.md`.
