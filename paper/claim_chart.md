# Claim chart versus `docs/review_1_lit_rev.pdf`

Each row: the paper does **not** disclose the combination “event-level trust-gated causal escalation on a marine KG under a hardware-aware secure HAL.”

| # | Prior art (lit review) | Event-level trust | Marine causal KG | Trust-gated energy hierarchy | Secure HAL + device orchestration |
|---|------------------------|-------------------|------------------|------------------------------|-----------------------------------|
| 1 | DTNA UASN twin | no (network RAPD/CMFD) | no | no | no |
| 2 | DTEAR routing twin | no | no | routing, not inference | no |
| 3 | Cognition-difference dynamic trust | **node**-level | no | fusion weights | no |
| 4 | Glider anomaly logs | mechanical | no | no | no |
| 5 | MEMS hydrophone review | hardware only | no | no | no |
| 6 | TinyML Arduino acoustic | terrestrial MFCC | no | no | no TPM/SE |
| 7 | EdgeEvolution MCU NAS | HIL NAS | no | no acoustics | no |
| 8 | DRL-TinyEdge | venue selection | no | not event trust | no |
| 9 | TinyML PAWSN ~0.45 W | leaf radio | no | no | ESP-Now |
| 10 | VLF Σ-Δ front-end | acquisition | no | no | no |
| 11 | MO-CRP NSGA-II | cluster *routing* | no | different objective | no |
| 12–15 | MARL AUV search | path planning | no | no | no |
| 16–18 | Bayesian DL sonar / VI-PANN | uncertainty for *class* | no | not a gate | too heavy for edge |
| 19 | Causal sound (cognitive science) | human | no deployable KG | no | no |
| 20 | iKnow-audio | terrestrial AKG | terrestrial | no energy gate | foundation models |
| 21–25 | DeepShip classifiers / GANs / dense CNN | accuracy only | no | no | no |
| 26–28 | PINN / SADH / PNN | physics, other modalities | no | no | no |
| 29 | Sonar+AI review | review | no | no | no |
| 30–32 | Kalman / TOA / BlueBuzz | tracking, comms | no | no | modem |
| 33–35 | CoralBuddy / UWA LSTM / ocean RS | platforms / channels | no | no | no |

**Closest art and distinction.** (3) scores sensors, we score events. (11) multi-objective *routes packets*, we multi-objective *admit inference levels*. (20) is a terrestrial audio KG with foundation models; ours is a compact marine KG with a counterfactual band-mask, TinyML-legal. (16) puts uncertainty on the label; we put it into \(T(e)\) and then into \(\pi\).
