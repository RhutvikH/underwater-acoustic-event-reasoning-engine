# Raspberry Pi and real-hardware path

The specialty is **many cheap nodes**, not one rack GPU. A Pi 4 or Pi
Zero 2 W can run the *same* numpy pipeline we use in the twin. Cortex-M
and ESP32 remain the TinyML *cost models* for patent HAL claims; the Pi
is how you actually put a node in a bucket of water this semester.

## What runs where

| Platform | Role | How |
|----------|------|-----|
| Laptop | Twin + GUI + Unity feed | `uaere demo` |
| Laptop | Pretend N Pis | `uaere edge --n 4` |
| Raspberry Pi 4 | Full L0–L3 node | Python 3.11+, `pip install -e .`, `uaere edge` |
| Pi Zero 2 W | Cheap node, L0–L3 still fits | same, slower |
| STM32 / ESP32 | Modelled in `hardware/profiles.py` | energy/latency tables; optional future INT8 flash |

Profiles: `raspberry_pi4`, `raspberry_pi_zero2` in
`src/uaere/hardware/profiles.py`.

## Simulate a Pi field on the laptop (do this in the viva if you have no boards)

```bash
source .venv/bin/activate
uaere edge --n 4 --seed 0
```

You get four JSON inferences as if four Zero 2 W boards heard four
windows. Combined with `uaere demo --nodes 8` you show the swarm
visually *and* the Pi-class runtime.

## Flash a real Pi (when you have one)

1. Raspberry Pi OS 64-bit, Python 3.11+.
2. Copy this repo (private). `python3 -m venv .venv && pip install -e .`
3. USB or I2S MEMS mic later; today the node can ingest twin WAVs:

```bash
uaere twin --n 8 --out /home/pi/clips
uaere edge --n 1
```

4. Neighbour wake uses UDP broadcast port **7946**
   (`uaere.edge.runtime.COLLAB_PORT`). Two Pis on one Ethernet switch
   or one Wi-Fi AP: when \(T(e)\) is in the uncertain band, `broadcast_wake`
   sends `{"type":"collab_wake",...}`. That is the same L4 the GUI draws
   in magenta.

5. Do **not** enable `require_authenticated=True` on a stock Pi unless
   you accept software HMAC only (no SE). The orchestrator will still
   HMAC inside `SecureHAL` if you construct one; the *placement*
   constraint is what refuses `gd32vf103`.

## Power realism

Pi 4 active ~3.5 W. A Zero 2 W ~0.9 W. The *paper’s* 0.2 mJ L0 figure
is the STM32 listening budget, which is why the swarm mixes STM32/ESP32
*models* with Pi *deployables*. Say this sentence in the viva:

> “The joule tables are for the MCU-class hydrophone we would actually
> solder. The Pi is the laboratory node that already runs the identical
> algorithm, today.”

## What “simulatable such that it can run on actual hardware” means here

- Algorithm = numpy, no CUDA, no PyTorch.
- Memory = TinyCNN 546 B + logistic head ≪ 512 MB.
- I/O = WAV now, ALSA later.
- Networking = UDP JSON, not a custom PHY.
- If a board appears, you do not rewrite AHAIF; you point
  `EdgeNode.pipeline` at a live recorder.
