# Uniqueness and patentability memo

**This is not legal advice.** A patent is granted by an office, not by
this file. What follows is a technical novelty analysis against
`docs/review_1.pdf`, `docs/review_1_lit_rev.pdf`, `docs/non_negotiables.txt`,
and a 2024–2026 literature sweep.

## What we actually claim

The independent method claim (see `paper/complete_specification.md`
Claim 1) is a **combination**:

1. Environment-conditioned hierarchical representations.
2. **Event-level** wake trust **without** a predetermined energy threshold.
3. Multi-objective selection of an execution level (miss, FA, energy, …).
4. Marine KG explanation + cue-mask counterfactual, paid for only when
   admitted.
5. Hardware abstraction that can require authenticated TinyML and refuse
   unattestable devices.
6. (Dependent / swarm specialty) a **field of cheap heterogeneous nodes**
   that confirm the *same event* before anyone buys L3.

## Closest 2024–2026 art, and why it does not swallow the claim

| Work | What it trusts / reasons | Distinction |
|------|--------------------------|-------------|
| Riemannian VAE UWSN trust (IEEE TMC 2026) | **Malicious nodes** in a latent space | Node reputation, not event admit/reject |
| Federated DRL trust in UASNs (IEEE TMC 2024) | Insider / compromised nodes | Same: node-level |
| Q-learning trust QLTM (2025) | Energy/data/comms evidence that a *neighbour is hostile* | Hostile-node FSM, not acoustic-event gating |
| DLtrust dual-layer DT (IEEE 2026) | Digital twin for **attack** detection (LSTM + cloud DT) | Twin is a security observer of nodes, not a lab for event reasoning |
| Channel-based trust (IEEE IoT-J 2022) | Packet forwarding vs channel HMM | Communications trust |
| ST-GRSR (2026) | Detecting *adversary networks* with a GNN | Inverse problem (find a hidden net), not explain a hydrophone clip |
| AUV KG + RAG/LLM (arXiv 2507.20370) | Mission knowledge + LLM for robot behaviour | Not marine *acoustic event* explanation; uses an LLM we deliberately exclude |
| iKnow-audio (EMNLP 2025) | Terrestrial audio KG + foundation models | Wrong domain, wrong compute class |
| DeepShip classifiers | Class accuracy on ships | No trust gate, no KG, no swarm confirm |
| DTNA / DTEAR | Network resource twins | Routing / RAPD, not event reasoning |
| MO-CRP NSGA-II | Cluster **routing** objectives | We NSGA-II *inference levels* |

**Pattern.** Every UASN “trust” paper in this sweep still answers
“is this *sensor* honest?” AHAIF answers “is this *blip* genuine, and
should the field spend joules explaining it?” That is the gap named in
`docs/review_1.pdf` slide “Research Gap” and it still holds in 2026.

## What is *not* novel (do not claim)

- Mackenzie SSP, Thorp absorption, Knudsen/Wenz curves.
- Evidential Dirichlet heads (Sensoy et al. 2018).
- NSGA-II (Deb).
- Secure boot, HMAC, AES-GCM, TPM quotes as cryptography.
- TinyML on Cortex-M / ESP32 as a hardware class.
- The idea of a digital twin in general.

Claim the **system and the event-level gating method**, including the
cheap-swarm confirmation step.

## Patentability (India / PCT, technical view)

| Criterion | Assessment |
|-----------|------------|
| Novelty | Combination not disclosed in the 35-paper review nor in the 2026 node-trust / DT-security papers above. **File before public GitHub / arXiv.** |
| Inventive step | Non-obvious to a UASN routing person to gate *causal acoustic explanation* on *event* trust with neighbour confirmation. A classifier person would just train a bigger net. |
| Industrial application | Cheap UASN fields, shipping/PAM, lab certification of firmware. |
| Sufficiency | Best mode is the `src/uaere/` tree; equations in `paper/formulation.md`. |
| Risk | Examiner may treat “software on a known MCU” as excluded unless hardware-aware orchestration and the twin+swarm structure are foregrounded (non-negotiable 5). Keep device profiles and authenticated-inference *placement constraint* in Claim 1(e) and Claim 10. |
| Risk | Twin-only TRL-4 does not kill a method claim; it limits the *examples*. |
| Cannot promise | Grant, claim breadth after office actions, or freedom-to-operate vs unpublished applications. |

**Recommended order.** Provisional (this spec) → keep repo private →
complete specification → optional PCT → then paper (`paper/manuscript.md`).

## Trace to the capstone documents

| Document requirement | Present in code? |
|----------------------|------------------|
| review_1.pdf three modules (trust, KG, gated energy) | `trust/`, `kg/`+`causal/`, `policy/` |
| review_1.pdf Obj 1–4 | `uaere eval --suite paper` |
| NN1 hierarchical + MOO, not wake-word | `policy/`, CI grep forbids `if energy >` |
| NN2 dynamic trust + adaptive representation | `trust/`, `representation/env_norm.py` |
| NN3 MAG + causal | `kg/ontology.yaml`, `causal/reasoner.py` |
| NN4 digital twin + collaborative sensing | `twin/`, `swarm/field.py` (now first-class) |
| NN5 HAL: M, ESP32, RV, FPGA, NPU, TPM, SE, boot, auth TinyML, AEAD | `hardware/profiles.py`, `security/hal.py` + Pi profiles |
| NN6 math, complexity, calibration, ablation, robustness, battery, HW sim | `math/`, `eval/suite.py` |
