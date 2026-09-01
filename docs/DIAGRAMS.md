# Diagrams, flowcharts, and Gantt

Use these on slides. They match the running code.

---

## Gantt — capstone (Aug 2026 → Apr 2027)

```mermaid
gantt
    title AHAIF capstone
    dateFormat  YYYY-MM-DD
    axisFormat  %b
    section Foundation
    Literature + non-negotiables freeze     :done, a1, 2026-08-01, 2026-08-31
    Math identities + package scaffold      :done, a2, 2026-09-01, 2026-09-21
    section Twin and sensing
    Digital twin (SSP, Thorp, hydrophone)   :done, b1, 2026-09-08, 2026-10-12
    Cheap-swarm collaborative field         :done, b2, 2026-09-22, 2026-10-26
    section Intelligence
    Adaptive representation L0–L2           :done, c1, 2026-09-15, 2026-10-20
    Event-level trust engine                :done, c2, 2026-10-01, 2026-11-02
    Marine KG + counterfactual              :done, c3, 2026-10-15, 2026-11-16
    NSGA-II gating policy                   :done, c4, 2026-11-01, 2026-11-30
    section Hardware
    Device profiles + secure HAL            :done, d1, 2026-11-10, 2026-12-15
    Raspberry Pi edge runtime               :done, d2, 2026-12-01, 2026-12-22
    section Demonstration
    Presentation GUI                        :done, e1, 2026-12-08, 2027-01-12
    Unity visualisation layer               :done, e2, 2026-12-15, 2027-01-20
    section Publication
    Paper suite freeze (3 seeds)            :f1, 2027-01-15, 2027-02-20
    Provisional patent file                 :milestone, m1, 2027-02-28, 1d
    Manuscript to Q1 / Scopus               :f2, 2027-03-01, 2027-04-15
    Viva / Unity live demo                  :milestone, m2, 2027-04-20, 1d
```

---

## System context

```mermaid
flowchart LR
    subgraph OceanTwin [Digital twin]
      Env[SSP / sea-state / Thorp]
      Prop[Spreading + bounce]
      Src[One physical source]
    end
    subgraph Field [Cheap hydrophone field]
      N1[Node STM32]
      N2[Node ESP32]
      N3[Node Pi Zero]
    end
    subgraph AHAIF
      T[Event trust]
      G[MOO gate]
      KG[Marine KG]
    end
    Unity[Unity / Browser GUI]
    Src --> Prop --> N1 & N2 & N3
    Env --> Prop
    N1 -- L4 wake --> N2
    N2 -- confirm --> T
    T --> G --> KG
    G --> Unity
```

---

## Per-window pipeline

```mermaid
flowchart TD
    X[1 s hydrophone window] --> L0[L0 shape DSP]
    L0 -->|logistic of vector not energy θ| Q{Near silence?}
    Q -->|yes| S[Sleep]
    Q -->|no| L1[L1 Mel + env-norm + Dirichlet]
    L1 --> C[C_wake and T of e]
    C --> P{Policy π}
    P -->|T below τ1| S
    P -->|uncertain band| L4[L4 neighbour wake]
    L4 --> Fuse[Event-level fusion]
    Fuse --> P2{Now above τ3?}
    P -->|T high| L2[L2 TinyCNN]
    L2 --> L3[L3 KG + counterfactual]
    P2 -->|yes| L3
    L3 --> HAL[Secure HAL + device profile]
```

---

## Swarm specialty (the series of cheap sensors)

```mermaid
sequenceDiagram
    participant Src as Ship / storm
    participant A as Cheap node A
    participant B as Cheap node B
    participant C as Cheap node C
    Src->>A: loud structured / or loud geophony
    Src->>B: quieter copy
    Src->>C: quieter copy
    A->>A: T(e) in uncertain band
    A->>B: L4 collab-wake (256 bit)
    A->>C: L4 collab-wake
    B->>A: C_wake
    C->>A: C_wake
    alt neighbours confirm
        A->>A: pay L3, emit KG sentence
    else neighbours disagree
        A->>A: stay asleep — storm rejected
    end
```

This is **not** “A trusts B as a node.” B can be honest and still hear
only wind. Confirmation is of the *event*.

---

## Hardware placement

```mermaid
flowchart LR
    L[Selected level] --> O[Orchestrator]
    O -->|needs SE+boot| OK[stm32 / esp32 / ethos / Pi with software HMAC]
    O -->|security on, no SE| X[Refuse gd32vf103]
```

---

## Work breakdown (WBS)

```
AHAIF
├── Twin (environment, propagate, sensor, network)
├── Swarm field (N cheap nodes, event fusion)
├── Intelligence (φ, trust, KG, policy)
├── Hardware HAL (7 profiles including two Pis)
├── Demo (GUI + Unity JSON)
└── Paper / patent pack
```
