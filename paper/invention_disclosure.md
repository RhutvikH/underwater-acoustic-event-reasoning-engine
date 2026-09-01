# Invention disclosure — Adaptive Hierarchical Acoustic Intelligence Framework

**Title.** Event-level trust-gated causal reasoning for energy-constrained underwater acoustic sensor networks.

**Inventors.** Rhutvik Hegde; Kalyani Manoj.

**Assignee (intended).** Vellore Institute of Technology.

**Status.** Draft for a provisional specification. Do not publish this repository until filed.

## Field

Underwater acoustic sensor networks; TinyML; calibrated uncertainty; knowledge-graph causal explanation; hardware-aware secure inference.

## Problem

UASNs wake on a *fixed spectral-energy threshold*. Existing trust models score *nodes* (compromised / faulty), not *events* (is this blip genuine?). Causal knowledge graphs exist for terrestrial audio, not marine events. Always-on explanation is too expensive for edge batteries.

## Independent claim (method)

A computer-implemented method for underwater acoustic event reasoning comprising:

1. extracting environment-conditioned multi-level acoustic representations from a hydrophone window and a digital-twin environment state;
2. computing a calibrated event-level wake-trust score from those representations and a sensor-health estimate, **without applying a predetermined energy threshold**;
3. selecting an execution level from a hierarchy by solving a multi-objective program over miss rate, false-alarm rate, energy, latency, and explanation coverage;
4. when the selected level includes causal reasoning, generating an explanation chain over a marine acoustic knowledge graph and verifying it by a feature-mask counterfactual;
5. executing the selected level on a hardware abstraction that maps levels to device profiles and that requires authenticated inference when a security constraint is active.

## Independent claim (system)

A system comprising an underwater acoustic digital twin, one or more edge nodes implementing the method, and a sink that verifies authenticated TinyML tags and AES-GCM payloads.

## Dependent claims (selected)

- Evidential Dirichlet head producing aleatoric and epistemic uncertainty that *feed the gate*.
- Environment-conditioned instance normalisation using twin SSP / sea-state / turbulence.
- NSGA-II (or equivalent) Pareto policy with Chebyshev knee selection at runtime.
- Collaborative neighbour-wake when trust lies in an uncertain band.
- Device profiles for ARM Cortex-M, ESP32, RISC-V, FPGA DSP, and NPU.
- Secure boot hash chain, secure-element-held model key, TPM PCR quote, HMAC-authenticated inference, AES-GCM communication.
- Twin-injected hydrophone faults as health features.
- Refusal to place L2/L3 on a non-attestable profile when the security constraint is on.

## What we are **not** claiming

Ice-keel link prediction, PINN AUV control, phase-aware GAN denoisers, continuous-space MARL. Those appear in the literature review as possible future work only.

## Best mode

The implementation in `src/uaere/` on public/twin-synthetic 1 s windows at 16 kHz, TinyCNN INT8 < 250 KB, software-defined hardware emulation (non-negotiable 5).

## Dates

Conception: capstone review 2026-08-31. Reduction to practice: this repository.
