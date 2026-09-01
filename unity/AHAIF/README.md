# AHAIF Unity digital twin (visualization layer)

Unity does **not** re-simulate Mackenzie sound speed or Thorp absorption.
Python `uaere demo` *is* the scientific twin. Unity is the 3D stage you
stand in front of during a viva / investor demo.

## Protocol

`GET http://127.0.0.1:8765/api/state` → JSON

```json
{
  "t": 12,
  "scenario": "busy_strait",
  "environment": {"sea_state": 2, "ssp_m_s": 1491.2},
  "source": {"xyz": [400, 200, 12], "class": "tug", "present": true},
  "nodes": [
    {
      "node_id": "n0", "xyz": [120, 80, 18], "profile": "stm32l476",
      "battery_frac": 0.99, "level": 3, "wake": 0.91, "trust": 0.74,
      "event_class": "tug", "explanation": "tug is the top cause...",
      "energy_j": 0.006, "confirmations": 2, "woke_neighbors": ["n1"]
    }
  ],
  "links": [{"src": "n0", "dst": "n1", "kind": "collab_wake"}],
  "kpis": {"detections": 3, "collab_wakes": 4, "joules": 0.04}
}
```

`xyz` is metres in the twin: X east, Y north, Z depth (positive down).
Scale by `metresToUnits` (default 0.01 → 12 m world for a 1200 m field).

## Drop-in (Unity 2021.3 LTS or newer)

1. Terminal: `uaere demo --nodes 8 --port 8765`
2. New 3D URP project. Create empty GameObject `AHAIFTwin`.
3. Copy `Scripts/*.cs` into `Assets/AHAIF/`.
4. Add `TwinClient`, `SwarmFieldView` on `AHAIFTwin`.
5. Press Play. Nodes, source, and magenta collab links appear live.

CORS is open (`Access-Control-Allow-Origin: *`) so the Unity editor on
the same machine can poll localhost.

## What maps to the paper

| Unity object | AHAIF concept |
|--------------|---------------|
| Sphere per node, colour by profile | cheap heterogeneous field |
| Sphere scale ∝ wake confidence | event-level \(C_{\mathrm{wake}}\) |
| Gold ring when level ≥ 3 | KG explanation paid for |
| Magenta line | L4 neighbour-wake (the swarm specialty) |
| Amber source | one physical event many ears hear |
