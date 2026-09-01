# AHAIF in plain language

Read this once. Afterward you should be able to **explain the project to a
faculty panel** and **run the demo** without being an acoustics expert.

---

## The one-sentence pitch

A field of cheap underwater microphones that **ask each other whether a
sound is real** before they spend battery explaining it — instead of one
expensive box that wakes up every time the ocean is loud.

## Why that is hard

The ocean is noisy. Wind, rain, and snapping shrimp make a lot of energy
in the water. A traditional sensor uses a rule like:

> “If the sound is louder than 12, wake the computer.”

That rule:

- wakes on storms (wastes battery),
- sleeps through a quiet tugboat far away (misses the event).

Existing “trust” research for underwater networks asks a different
question: *is this sensor hacked?* We ask: *is this blip a genuine
event?* Those are not the same.

## The analogy that carries the whole talk

Think of **smoke alarms in a cheap apartment block**, not a single
museum-grade detector.

1. Every unit has a tiny alarm (a cheap hydrophone + a tiny chip).
2. One unit is unsure — maybe toast, maybe fire.
3. It pings two neighbours (a few extra millijoules of acoustic modem).
4. If they also smell smoke, *one* unit runs the expensive “what kind of
   fire?” checklist (our knowledge graph).
5. If they do not, everyone goes back to sleep.

That neighbour-check is **event-level confirmation**. It is the
specialty. Deploying *many* inexpensive nodes is the point; a single
perfect hydrophone is not.

## The five layers (say them in order)

| Layer | Plain words | When it runs |
|-------|-------------|--------------|
| L0 | Cheap listening: “is anything shaped like a sound at all?” | Always (microwatts) |
| L1 | Trust score: “how sure are we this is a real event?” | If L0 says maybe |
| L2 | Tiny classifier: tug / cargo / tanker / passenger | If trust is high enough |
| L3 | Knowledge graph: “tug, because of blade-rate harmonics, sea-state 2; rain is impossible here.” | Only for high-trust, often after neighbours agree |
| L4 | Ask a neighbour to listen too | When trust is in the *uncertain* band |

There is **no** “if loud then classify.” Loud storms stay at L0/L1.
Quiet but structured ship noise can still climb the ladder.

## Words you will be asked

**Digital twin.** A video-game-accurate *physics lab* of the ocean
(temperature → sound speed, distance → quieter copies, broken
hydrophones). We validate in this lab so we do not need a research
vessel for TRL-4. Unity is the 3D *skin*; Python is the physics.

**Event-level trust.** A number \(T(e)\) and a wake confidence
\(C_{\mathrm{wake}}\) for *this one-second clip*, not for the sensor’s
lifetime reputation.

**Knowledge graph.** A small map of causes (tug, rain, clipping) and
how they relate. Not ChatGPT. A sentence comes out, and we check it by
muting the frequencies that cause should have used.

**Multi-objective policy.** A search that picks thresholds so we miss
few ships, cry wolf rarely, and spend few joules. NSGA-II finds a
front of compromises; we pick the “knee.”

**TRL-4.** Lab validation. Not a deployed ocean product.

**Patentable combination (honest version).** Others do node-trust, or
ship classifiers, or terrestrial audio graphs, or network digital
twins. The combination “event trust → neighbour confirmation → gated
marine causal explanation, on a field of cheap heterogeneous nodes,
under a secure hardware abstraction” is not in those papers. A lawyer
must still file. We cannot promise a grant.

## How to run it, even if you have never used Python

1. Open a terminal.
2. Paste:

```bash
cd /home/kira/Documents/underwater-acoustic-event-reasoning-engine
source .venv/bin/activate
uaere demo --nodes 8 --port 8765
```

3. Open a browser at **http://127.0.0.1:8765/**
4. You should see an overhead ocean, coloured nodes, an amber source,
   magenta lines when neighbours are asked, and a gold ring when a node
   pays for an explanation.
5. Click a node. Read \(C_{\mathrm{wake}}\), \(T(e)\), battery, sentence.
6. Ctrl-C in the terminal stops it.

**If the venv does not exist:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Paper numbers (no GUI):**

```bash
uaere eval --suite paper --n-windows 160 --seed 0
```

**Unity:** leave `uaere demo` running, open the Unity project, add the
scripts under `unity/AHAIF/`, press Play. Unity only *draws* the twin.

**Raspberry Pi:** the same Python runs on a Pi 4 / Pi Zero 2 W. On a
laptop, `uaere edge --n 4` pretends four Pis inferred four windows.
See `docs/RASPBERRY_PI.md`.

## How to explain a live demo in 60 seconds

> “Each dot is a cheap hydrophone. The amber pulse is one real sound.
> Nodes that are sure stay quiet or explain. Nodes that are unsure
> light a magenta line to a neighbour — that is L4. When the field
> agrees, one node spends battery on a knowledge-graph sentence. Storms
> are loud but unstructured, so they rarely get that sentence. That is
> why a swarm of cheap sensors beats one energy threshold.”

## What not to say

- Do not say it has been tested in the deep ocean. It has not.
- Do not say the GUI *is* the digital twin. The twin is the physics
  engine; the GUI is the window.
- Do not say we invented underwater sound-speed equations (Mackenzie,
  Thorp are 1960s–80s). We *use* them inside a new *system*.
- Do not say “AI chatbot.” There is no LLM in the loop on purpose.

## Map of the rest of the binder

| If they ask… | Point to… |
|--------------|-----------|
| How do I operate it? | `docs/USER_GUIDE.md` |
| What is the science? | `paper/manuscript.md` |
| What would we patent? | `paper/complete_specification.md` |
| Is this actually new? | `docs/UNIQUENESS.md` |
| Time plan | Gantt in `docs/DIAGRAMS.md` |
| Hardware | `docs/RASPBERRY_PI.md` |
