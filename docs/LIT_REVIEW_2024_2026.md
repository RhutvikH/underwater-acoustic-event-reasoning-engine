# Literature and patents, 2024–2026

This is the dated review the capstone 35-row PDF is not. `docs/review_1_lit_rev.pdf` has **no years**, mixed ResearchGate links, and **zero patents**. This file answers: what is in-window, what is older on purpose, who is better or worse than AHAIF and **on which question**, and what we should change.

**In-window** means publisher year, conference year, or patent grant/publication year in 2024–2026.  
**Not counted here:** Mackenzie, Thorp, Sensoy, DeepShip, Dung, NSGA-II. Those stay in §0 so nobody thinks we hid them.

We do **not** rank AHAIF’s twin `C_wake` AUC against DeepShip classification accuracy or against malicious-node detection rate. Those are different labels.

---

## How to read a row

| Verdict | Meaning |
|---------|---------|
| **They are stronger** | Better evidence or a bigger public result on *their* task |
| **We are stronger** | We do a job they do not do (event admit, surprise wake, gated marine WHY/WHY-NOT) |
| **Different question** | Same words (“trust”, “twin”, “KG”) for a different object |

Numeric AHAIF numbers in this file are the seed-0 paper-suite run (`artifacts/paper/suite.json`): `C_wake` ROC-AUC 0.916 vs Youden energy 0.069, matched miss 0.488, modelled energy ~32× below always-on L3. That comparison is **only vs our energy baseline**.

---

## 0. Foundations (out of window — keep citing)

These are physics, uncertainty, and datasets. They are not 2024–2026 novelty.

| Work | Year | Why we still cite it | Link |
|------|------|----------------------|------|
| Mackenzie nine-term sound speed | 1981 | Twin `c(T,S,z)` in `twin/environment.py` | https://doi.org/10.1121/1.386920 |
| Thorp absorption | 1967 | Twin Thorp dB/km | https://doi.org/10.1121/1.1910566 |
| Knudsen / Wenz ambient | 1948 / 1962 | Generative spectrum for surprise | Wenz, JASA 34(12), 1962 — https://doi.org/10.1121/1.1909155 |
| Sensoy, Kaplan, Kandemir Dirichlet evidential DL | 2018 | `alpha = 1 + tau p_hat` lift | https://arxiv.org/abs/1806.01768 · NeurIPS https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html |
| Irfan et al. DeepShip | 2021 | Public 4-class ship corpus; adapter only | https://doi.org/10.1016/j.eswa.2021.115270 |
| Traer, Norman-Haignere, McDermott causal environmental sound | 2021 | Human causal sound, not a marine KG | publisher PDF: Cognition 214:104627 |
| Signori, Campagnaro, Nissen, Zorzi channel-based UAN trust | 2022 | Closest *communications* trust; packet/channel, not a blip | IEEE IoT-J 9(20) — https://ieeexplore.ieee.org/document/9778240 |
| Deb et al. NSGA-II | 2002 | We NSGA-II *inference levels*, not routing | IEEE TEC 6(2) |
| Dung abstract argumentation | 1995 | WHY/WHY-NOT grounded extension | AI Journal |
| Claerbout / Wapenaar interferometry | 1968 / 2004 | Ambient-noise Green’s functions as death-zone atlas | seismology prior, not a UASN claim |

Do not put these in a 2024–2026 bibliography and pretend they are new.

---

## 1. In-window corpus (36)

### A. UASN / UWSN “trust” (they score nodes)

**Pattern.** Every paper in this cluster answers “is this *sensor* honest / hostile / Sybil?” AHAIF answers “is this *one-second blip* a genuine event, and should the field spend joules explaining it?”

**1. Song, Huangfu, Guo, Liu, Cui, Shen — Digital Twin-based Network Architecture (DTNA) for UASNs**  
IEEE TMC 24(9), 2025 (preprint 2024).  
https://doi.org/10.1109/TMC.2025.3555640 · https://arxiv.org/abs/2410.20151  

Local RAPD resource allocation; global CMFD MARL data-patching and TNSD slicing. Lab-pool plus Aqua-Sim.  
**They are stronger** as a *network-control* twin (tank + simulator). **Different question:** RAPD/CMFD, not hydrophone-event reasoning.  
**Improve:** keep our twin as an event lab; do not compete on RAPD. Optionally log Aqua-Sim-class delay if we ever claim network control.

**2. Xia, Wei et al. — Riemannian VAE trust for UWSNs**  
IEEE TMC 25(1):117–131, 2026 (online 18 Jul 2025).  
https://doi.org/10.1109/TMC.2025.3590741 · https://ieeexplore.ieee.org/document/11085120/  

Trust evidence in a Riemannian latent space; reported ~94% malicious-node detection.  
**Different question:** node reputation. We have no node-adversary label.  
**Improve:** treat a node-trust score as an *input* to `kappa(h)` / HAL, not a replacement for `T(e)`.

**3. He, Han, Li, Taleb, Wang, Yu — Federated DRL trust in UASNs**  
IEEE TMC 23(5):5150–5161, **issue May 2024**; IEEE Xplore online 4 Aug **2023**.  
https://doi.org/10.1109/TMC.2023.3301825  

Communication / energy / data evidence into local DRL predictors with federated aggregation vs insiders.  
**Borderline year** (DOI 2023, issue 2024). **Different question:** neighbour hostility. Same inventors as patent **32**.  
**Improve:** same as (2) — compose, do not leaderboard.

**4. AFRTM — attention-weighted federated DRL trust**  
IEEE TMC 25(6):8947–8961, June 2026 (pub. 12 Jan 2026).  
https://doi.org/10.1109/TMC.2026.3651511  

Environmental + behavioural evidence, DRL local trust, attention-weighted federated aggregation.  
**Different question:** still node trust / energy of *trust estimation*.  
**Improve:** if we federate anything, federate Dirichlet strengths, not node reputation.

**5. Hosseinzadeh et al. — QLTM hostile-node FSM**  
Ad Hoc Networks 178:103918, 2025.  
https://doi.org/10.1016/j.adhoc.2025.103918  

Energy/data/comms → trust / distrust / uncertain FSM.  
**Different question:** hostile neighbour. Uncertain band is a *node* state; our collab band is *event* `T(e)`.  
**Improve:** keep L4 as event confirmation, not a distrust FSM.

**6. Mai et al. — DLtrust dual-layer DT**  
IEEE GLOBECOM 2025 (8–12 Dec 2025; Xplore 19 Mar 2026). **Not “IEEE 2026 journal.”**  
https://doi.org/10.1109/GLOBECOM59602.2025.11432209  

Local DT LSTM for conventional attacks; Cloud DT weights trust evidence + RL.  
**They are stronger** on *attack-detection* simulation. **Different question:** twin as security observer of nodes.  
**Improve:** do not rename our twin a “security DT.”

**7. Jiang, Zhou, Luo, Cui, Wang, Song — hybrid trust evidence + TabTransformer**  
IEEE TNSE, 2025.  
https://doi.org/10.1109/TNSE.2025.3539320  

Attack detection and deployment via hybrid trust; TabTransformer on trust features; malicious-node types.  
**Different question:** node/attack.  
**Improve:** none for the wake; optional HAL input.

**8. Zhu, Boukerche, Li, Yang — TECTM traffic-aware edge trust**  
ICC 2024, Denver.  
https://doi.org/10.1109/ICC51166.2024.10622709  

Traffic + environment; AUVs as edge ML for cluster trust.  
**Different question:** malicious-node detection with AUVs we do not have.  
**Improve:** do not add AUVs to the claim.

**9. Zhu, Boukerche, Long, Yang — Design guidelines on UWSN trust**  
IEEE Communications Surveys & Tutorials 26(4):2547–2576, 2024.  
https://doi.org/10.1109/COMST.2024.3389728  

Taxonomy: weighted sum, logic, probability, ML. Future work still framed as *node* reliability.  
**They are stronger** as a map of the field. Confirms the gap: event-level acoustic trust is not a surveyed category.  
**Improve:** cite this survey in the manuscript related-work, then state the gap in one sentence.

**10. Sybil mitigation via multi-attribute trust cognition**  
IEEE Sensors Journal 26(12), 15 June 2026 (pub. 7 May 2026).  
https://doi.org/10.1109/JSEN.2026.3687587  

TOPSIS + AHP across PHY/MAC/routing; ~16% PDR gain in sim; **shallow-sea field test**.  
**They are stronger** on a sea trial for *Sybil/PDR*. **Different question:** identity attacks, not event genuineness.  
**Improve:** we have no field trial; do not pretend the GUI is one.

**11. “Trust in the Deep” — authenticating acoustic underwater communication**  
IEEE VCC 2025.  
https://doi.org/10.1109/VCC67261.2025.11351213  

Survey of crypto vs physical-layer authentication for *messages and entities*.  
**Different question:** authenticating a packet/speaker, not admitting a clip. Our HMAC/AES-GCM HAL is this cluster, and is **not** the invention.  
**Improve:** keep HAL as dependent claim; do not cite this as event-trust prior.

**12. Sun, Shen, Yuan, Xie, Wang, Liu — ST-GRSR**  
Defence Technology, in press 28 May 2026.  
https://www.sciencedirect.com/science/article/pii/S2214914726001625  

Spatiotemporal GNN (GraphSAGE + GRU) to detect that an *adversary network exists* (>90% on NS-3 AquaSim).  
**Different question:** inverse graph (“is there a hidden net?”), not “explain this hydrophone window.”  
**Improve:** none for L3.

**13. Hussain, Li, Hussain et al. — EETAUV trust + energy AUV scheme**  
Sensors 25(286), 2025.  
https://doi.org/10.3390/s25010286  

Energy, void-hole, localization, phantom-node trust for 6G IoUT.  
**Different question:** routing/security of AUV-aided nets.  
**Improve:** skip.

**14. ETrust — ensemble trust for ocean cloud–edge nets**  
IEEE Internet of Things Journal 12(1):18–29, 1 Jan 2025 (online 16 Sep 2024).  
https://doi.org/10.1109/JIOT.2024.3460737  

Nodes ship behaviour/comms evidence to cluster → edge → cloud ensemble.  
**Different question:** cloud-edge *node* trust. We refuse cloud LLMs in the reasoner.  
**Improve:** keep L3 on-node.

---

### B. Knowledge graphs and explanation

**15. Olvera, Wang, Stamatiadis, Richard, Essid — iKnow-audio**  
EMNLP 2025 (Suzhou), ACL pp. 34683–34700.  
https://doi.org/10.18653/v1/2025.emnlp-main.1759 · https://aclanthology.org/2025.emnlp-main.1759/  

Audio-centric KG (semantic/causal/taxonomic) refining CLAP on ESC-50, UrbanSound8K, TUT2017, FSD50K, AudioSet, DCASE17.  
**They are stronger** on public-benchmark KG scale and foundation-model backends. **We are stronger** on TinyML-legal marine events + Dung WHY-NOT + energy gate. Wrong domain, wrong compute class.  
**Improve:** grow the marine ontology beyond 20 nodes / 21 edges; do **not** import CLAP.

**16. Grimaldi et al. — KG + RAG for multi-agent underwater missions**  
arXiv:2507.20370, 27 Jul 2025.  
https://doi.org/10.48550/arXiv.2507.20370  

GPT-3.5 + KG + taxonomy: 100% mission validation on 20 Stonefish missions vs 21% LLM-only. Geometric primitives, not vessel/geophony.  
**They are stronger** at LLM-grounded *mission* planning. We **exclude** LLMs in the reasoner on purpose.  
**Improve:** do not add RAG. Optionally log WHY-NOT coverage the way they log BT completeness.

---

### C. TinyML, hydrophone silicon, crypto

**17. Roy, Das, Mondal — TinyML acoustic event detection in PAWSNs**  
Journal of Telecommunications and Information Technology 100(2):69–77, 2025.  
https://doi.org/10.26636/jtit.2025.2.2084 · https://jtit.pl/jtit/article/view/2084  

MCU AED; only above-threshold events are transmitted. ESC-50, Grove mics, ESP32/XBee.  
**They are stronger** on measured terrestrial MCU watts. **Worse for us as prior:** energy threshold *is* the wake, on land.  
**Improve:** keep surprise; if we ever flash ESP32, report joules like they do, not datasheet nJ/MAC.

**18. Baciu et al. — TinyML attestation**  
Future Internet, 2025.  
https://doi.org/10.3390/fi17020085  

Hardware-aware secure TinyML / attestation patterns.  
**They are stronger** as crypto/attestation literature. Our HAL is a **protocol simulation** (non-negotiable 5), not silicon.  
**Improve:** say that in the spec; do not imply Renode-free chips.

**19. SwiftGuard AES for underwater comms**  
ICTC 2024.  
https://doi.org/10.1109/ICTC62082.2024.10827451  

AES for acoustic links. **Not novel as a claim** (we already listed AES-GCM as non-claim).  
**Improve:** keep AEAD as HAL plumbing.

---

### D. Underwater acoustic target recognition (we lose on DeepShip %)

These papers classify ships. AHAIF’s TinyCNN is untrained in the paper suite; detection is `C_wake`, not class accuracy. **Do not quote 0.916 AUC as if it were 98% DeepShip.**

**20. Chu, Hou, Duan, Xu, Hu — CSGKD**  
IEEE Journal of Oceanic Engineering 50(4):3145–3159, Oct 2025.  
https://doi.org/10.1109/JOE.2025.3586648  

Channel–spatial global knowledge distillation; 82.37% on oceanic sets; +8.87% on a lightweight student.  
**They are stronger** at compressing a UATR teacher onto an edge student.  
**Improve:** if we train TinyCNN, distill like this; still gate on surprise/`T(e)`, not on the student’s softmax.

**21. Gao, Chen, Liu, Zhang, Zhang — heterogeneous spectral attention UATR**  
Signal Processing 246:110601, Sep 2026.  
https://doi.org/10.1016/j.sigpro.2026.110601  

Time-frequency Transformer + CNN; **98.26% ShipsEar, 95.58% DeepShip** at low SNR.  
**They are stronger** on public UATR accuracy. We do not have a competing number.  
**Improve:** run `uaere train --dataset` on DeepShip when the files exist; report class metrics *separately* from `C_wake` ROC.

**22. Zhao, Chen, Lu, Cheng, Chen, Li, Alkayem — PR-Conv / MobilePR-ConvNet**  
Sensors 25(22):7007, 17 Nov 2025.  
https://doi.org/10.3390/s25227007  

**98.58% DeepShip, 97.82% ShipsEar**, 94% fewer params than a conventional net; 77.8% at 10 dB SNR.  
**They are stronger** as a lightweight classifier. Closest *engineering* cousin to TinyCNN, still no trust gate / KG / swarm.  
**Improve:** steal width-scaling ideas for the IR; do not claim their %.

**23. Deng, Hong — UATR-DIFF-Transformer**  
OCEANS 2025 Brest; journal version Ocean Engineering 341:122668, Dec 2025.  
https://doi.org/10.1109/OCEANS58557.2025.11104348 · https://doi.org/10.1016/j.oceaneng.2025.122668  

Differential attention + PolyLoss; **90.26% ShipsEar, 97.84% DeepShip**; 68.64% at −10 dB; fewer FLOPs.  
**They are stronger** on low-SNR class accuracy.  
**Improve:** surprise already aims at structure-in-noise; measure tug-vs-storm on *their* SNR grid if we ever leave the twin.

**24. Jiang, Zheng, Wang, Zhang — entropy feature combination**  
Acta Oceanologica Sinica 45:227–245, 2026 (pub. 15 May 2026).  
https://doi.org/10.1007/s13131-025-2565-2  

PE + envelope entropy + spectral entropy + cheap time/frequency stats; RF/AdaBoost ~86% on DeepShip.  
**Different (and weaker as a net)** than (21)–(23), but **closer to L0 DSP**. Our φ0 already has centroid/flatness; we do **not** wake on those.  
**Improve:** keep φ0 as features, not as admit.

**25. Enhanced UATR via multidimensional structured acoustic features**  
AIPCVT 2025 (12–14 Dec 2025).  
https://ieeexplore.ieee.org/document/11405520/  

Graphical multi-domain fusion; 92% on DeepShip-main (proof-of-concept).  
**They are stronger** on that split. Still accuracy-only.  
**Improve:** skip unless we need another DeepShip baseline.

**26. Hu, Chu, Dou, Liu, Liu, Qi — lightweight UATR *localization* via KD**  
IEEE JOE 50(2):1429–1442, 2025.  
https://doi.org/10.1109/JOE.2025.3538928  

Student 98.68% fewer params, 87.4% faster, 97.55% → 96.48% localization accuracy.  
**Different question:** range/place, not event class or trust.  
**Improve:** TDOA-as-trust is idea 20 in `IDEAS.md`; not this increment.

**27. Wang, Jiang, Wang, Shi, Xu, Yan — marine mammal call classification**  
IEEE JOE 50(4):2642–2660, 2025.  
https://doi.org/10.1109/JOE.2025.3559196  

Mammal-call CNN; higher accuracy / less train time than named baselines.  
**Context only.** We map Watkins folders to `biological`, not 32-way SOTA. **Do not compete.**  
**Improve:** keep `species_head` optional and silent in the paper tables.

**28. Broadband passive sonar track-before-detect on raw hydrophone data**  
IEEE JOE 50(4):3106–3116, 2025.  
https://doi.org/10.1109/JOE.2025.3573066  

VAR ambient + heavy-tailed model + Bernoulli TkBD; real data detection range 250 m → 390 m.  
**They are stronger** on *field* detection range. **Different question:** existence/bearing of a target, not class+explanation+gate.  
**Improve:** our surprise is a cheap cousin of “structured residual vs ambient”; we still lack a range trial.

---

### E. Twins, routing, field acoustics (they win on water)

**29. Yu, Qiao — 6D digital twin for AUVs**  
Journal of Field Robotics 43(6):3727–3740, 2026.  
https://doi.org/10.1002/rob.70220  

Physical / virtual / virtual-native / data / services / comms; UE5 cyberspace; offline + online twin.  
**They are stronger** as a vehicle twin with a real AUV loop. Ours is a hydrophone *event* lab; Unity only draws `/api/state`.  
**Improve:** do not claim Unity is a 6D DT.

**30. Deo, Venkateshwaran, Jaiman — physics-guided URN digital twin**  
Ocean Engineering 358:125461, 15 Jun 2026; arXiv:2509.25730.  
https://doi.org/10.1016/j.oceaneng.2026.125461 · https://arxiv.org/abs/2509.25730  

30M Bellhop3D pairs, Salish Sea, <2 dB, 800× speedup, voyage noise mitigation.  
**They are stronger** on 3D TL with uncertainty. We rejected Bellhop-as-invention; our interferometric atlas is the *measured* alternative.  
**Improve:** cite as the honest Bellhop-scale twin we are *not*; keep Claerbout-style `Ghat`.

**31. ILMR — interference-avoidance low-latency cross-layer multipath**  
IEEE JOE 51(1):659–675, Jan 2026.  
https://doi.org/10.1109/JOE.2025.3621805  

Lake + Zhoushan sea trial; ~30% lower latency, ~5% PDR.  
**They are stronger** on field routing. Our Dijkstra is on a learned death-zone graph after L3, not a MAC-layer protocol.  
**Improve:** do not quote PDR.

**32. On-board TL decision aid (spreading vs Bellhop vs ML tree)**  
IEEE JOE 50(3):1668–1675, 2025.  
https://doi.org/10.1109/JOE.2024.3498007  

SV3 Wave Glider + REMUS 100, 25 kHz modem, Southern California 2023; ML matches Bellhop at 0.57 ms vs 764 ms.  
**They are stronger** on at-sea TL for *comms placement*.  
**Improve:** our surface-image + Thorp is closer to their “simple approximation”; say so.

**33. DTEAR — digital-twin-enhanced adaptive routing**  
Preprints.org, 6 Nov 2025. **Not peer-reviewed.**  
https://doi.org/10.20944/preprints202511.0328.v1  

Routing twin. **Label: preprint.** Closest in spirit to claim-chart row 2.  
**Different question:** packets, not events.

---

### F. Patents (in-window instruments)

None of these is event-level trust-gated marine causal reasoning.

**34. CN117082492B — UASN trust management based on federated DRL**  
Inventors He Yu, Han Guangjie, Wang Hao, Jiang Jinfang; Hohai University. Filed 2023-08-09, granted **2025-06-03**.  
https://patents.google.com/patent/CN117082492B/en  

Same family as paper **3**. Node/insider trust with global/local controllers.  
**Different object:** nodes. File before they (or we) blur “trust” in an Indian/PCT claim.

**35. US20250284017A1 — software-defined hydrophone with ML/TinyML**  
Filed 22 May 2025, published **11 Sep 2025**, pending.  
https://patents.google.com/patent/US20250284017A1/en  

Controller may host a TinyML source classifier on the hydrophone.  
**They are closer on hardware.** No event-trust gate, no marine KG, no multi-objective level select.  
**Improve:** HAL already talks to device profiles; this is why Claim 1 must stay a *combination*, not “TinyML on a hydrophone.”

**36. CN119646670B — lightweight underwater-acoustic target recognition**  
Institute of Acoustics, CAS. Filed 18 Feb 2025, granted **13 May 2025**.  
https://patents.google.com/patent/CN119646670B/en  

End-to-end lightweight waveform UATR, tested on ShipsEar and DeepShip.  
**They are stronger** as a granted UATR system patent. Still class-accuracy, not gated explanation.  
**Improve:** do not file a claims set that looks like “a CNN on DeepShip.”

---

## 2. Scoreboard (honest)

| Axis | Winner | Why |
|------|--------|-----|
| Malicious-node detection | Riemannian VAE / DLtrust / QLTM / AFRTM | That is their label; we do not report it |
| Network-control twin | DTNA | Tank + Aqua-Sim |
| Public audio KG | iKnow-audio | ESC-50-scale + CLAP |
| LLM mission KG | Grimaldi | Stonefish 100% vs 21% |
| DeepShip / ShipsEar class % | PR-Conv, DIFF-Transformer, spectral attention | 95–98%; we did not run those sets in the paper suite |
| Field routing / PDR | ILMR, Sybil TOPSIS | lake/sea |
| 3D TL twin | Deo–Jaiman URN twin | 30M Bellhop3D |
| AUV twin | Yu 6D DT | real vehicle loop |
| TinyML watts on a mic | Roy PAWSN | measured ESP32; terrestrial energy threshold |
| Hydrophone+TinyML patent | US20250284017A1, CN119646670B | granted/pending silicon stories |
| Event admit vs energy (twin) | **AHAIF** | 0.916 vs 0.069 `C_wake` ROC, same miss, ~32× modelled compute |
| Surprise wake (not energy θ) | **AHAIF** | in-window TinyML AED still thresholds energy |
| Marine WHY / WHY-NOT on a cheap KG | **AHAIF** | iKnow is terrestrial+foundation; Grimaldi is LLM missions |
| Learned death zones then sink path | **AHAIF** | DTNA/DTEAR route packets; we route an *explanation* on `Ghat` |
| Combination in Claim 1 | **not found in this set** | event trust + no energy θ + MOO levels + marine KG + HAL |

---

## 3. How we improve (without changing the invention)

1. **Run the DeepShip / ShipsEar adapters** that already exist. Report class metrics *beside* `C_wake` ROC. Review Objective 4 asked for this; the paper suite still uses the twin.
2. **Calibrate** the Dirichlet head (`trust/calibrate.py`). ECE 0.328 is the weakest Objective-1 number.
3. **Compose** a node-trust score (VAE/QLTM-style) into `kappa(h)` / HAL. Do not replace `T(e)`.
4. **Grow the KG** past 20 nodes, or evaluate WHY-NOT on a public audio KG split (not CLAP).
5. **Measure joules** on a Pi / Nucleo the way Roy and DTNA measure, instead of datasheet nJ/MAC.
6. **Seeds {0,1,2}** as the protocol already says. n_test=43, one seed, is not a camera-ready table.

Do **not**: claim Watkins species SOTA; stack Bellhop as the invention; add an LLM reasoner; file Claim 1 as “computer program per se” without the hydrophone/HAL technical effect.

---

## 4. Borderline, excluded, and date notes

| Item | Handling |
|------|----------|
| He et al. TMC DOI 2023 / issue 2024 | Counted as **3** with the year conflict written out |
| DTEAR Preprints.org Nov 2025 | Counted as **33**, labeled preprint |
| UNIQUENESS.md “DLtrust IEEE 2026” | Wrong; GLOBECOM **2025** (entry **6**) |
| Channel-based trust IEEE IoT-J 2022 | Foundation (§0), not in-window |
| PLAN warship acoustic database / NEC foundation-model press | News, not papers |
| GBSR GNN secure routing ICSPCC 2023 | Conference 2023; Xplore Jan 2024 — excluded as 2023 |
| WhaleNet / MT-Resformer mammal SOTA | Crowded 2024–2026; we refuse that claim (see **27**) |

The capstone PDF (`docs/review_1_lit_rev.pdf`) remains the historical 35-row table. This file is the dated 2024–2026 review with links.

---

## 5. One-sentence gap

In-window UASN work trusts **nodes**, UATR work maximises **class %**, KG work is **terrestrial or LLM-mission**, and twins control **networks or vehicles**. Nobody in this list gates a **marine causal sentence** on **event-level** surprise/`T(e)` without an energy threshold, on a field of cheap hydrophones, under a secure HAL. That is still the research-gap slide in `docs/review_1.pdf`. It is not a promise that a patent office will agree.
