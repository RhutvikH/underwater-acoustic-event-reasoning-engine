# Unity guide

Python is the scientific digital twin. Unity is a **3D window** onto
`GET http://127.0.0.1:8765/api/state`. If Unity is not running the
Python demo, you will see an empty scene. That is expected.

You need Unity **2021.3 LTS or newer** (URP or Built-in). No asset store
packages.

## 0. Start the twin (always first)

```bash
cd underwater-acoustic-event-reasoning-engine
source .venv/bin/activate
uaere demo --nodes 8 --port 8765
```

Leave this terminal open. Browser check: http://127.0.0.1:8765/ should
show the same field Unity will draw. JSON check:

http://127.0.0.1:8765/api/state

CORS is `*` so the Editor on localhost may poll.

## 1. Create a Unity project

1. Unity Hub → New project → **3D (URP)** or **3D Built-in**.
2. Name it `AHAIF-Demo`. Save it **outside** this git repo (Unity’s
   `Library/` is huge). You will only copy our scripts in.
3. Open the empty scene.

## 2. Import AHAIF scripts

From this repo copy:

```
unity/AHAIF/Scripts/TwinClient.cs
unity/AHAIF/Scripts/SwarmFieldView.cs
```

into `Assets/AHAIF/` inside the Unity project (create the folder).
Wait for the compiler (bottom-right spinner). Errors about
`UnityWebRequest` mean you are on a very old Unity — use 2021.3+.

## 3. Wire the scene

1. Hierarchy → Create Empty. Name it `AHAIFTwin`.
2. Inspector → Add Component → `Twin Client`.
3. Add Component → `Swarm Field View`.
4. On `Swarm Field View`, drag the same GameObject into `client` (or
   leave empty; `Awake` finds `TwinClient` on the same object).
5. `Twin Client` → Url = `http://127.0.0.1:8765/api/state`
6. `Poll Seconds` = `0.4`
7. `Metres To Units` = `0.01` (a 1200 m field becomes ~12 Unity units).
8. Add a Directional Light if the template did not. Place the camera at
   `(6, 8, -6)` looking at origin, or use the Game view’s free camera.

## 4. Press Play

You should see:

| You see | It means |
|---------|----------|
| Amber sphere that pulses | acoustic **source** (one physical event) |
| Several smaller spheres | cheap **nodes** (STM32 / ESP32 / Pi) |
| Sphere scale changing | `C_wake` — surer nodes look bigger |
| Slight extra scale / pulse | node reached **L3** (paid for an explanation) |
| Magenta line | **L4 neighbour-wake** (the swarm specialty) |

If the scene is empty: the Python demo is not running, or the URL port
differs, or a firewall blocked localhost. Watch Console for
`AHAIF JSON` warnings.

## 5. Make it look like a viva stage (optional, 10 minutes)

- Create a large Plane at y = 0, material dark teal: the sea surface.
  Nodes have **negative Y** (depth). Move the plane to y = 0.2 if they
  clip.
- Fog: Window → Rendering → Lighting → Environment → Fog on, colour
  `#0a2c3c`, density 0.04.
- Camera: add a simple orbit script, or just Frame Selected on
  `AHAIFTwin`.
- UI TextMeshPro: poll `TwinClient.Latest.kpis` if you want on-canvas
  detection counts. Not required; the browser GUI already shows KPIs.

## 6. JSON schema (do not break this)

`xyz` is metres: **X east, Y north, Z depth positive down**.
Unity mapping in `SwarmFieldView.ToU`:

```
Unity.x =  X * metresToUnits
Unity.y = -Z * metresToUnits    // depth goes down
Unity.z =  Y * metresToUnits
```

Fields Unity currently reads: `t`, `source.xyz`, `source.class`,
`nodes[].node_id|xyz|wake|level|profile`, `links[].src|dst`.

Extra fields (`explanation`, `trust`, `battery_frac`, `kpis`) are in
the JSON for your own UI; the two shipped scripts do not require them.

`JsonUtility` does not parse arbitrary nested dictionaries. We used
`[Serializable]` classes in `TwinClient.cs`. If you add a field, add it
**there** with the **same JSON name** (or `[FormerlySerializedAs]`).

## 7. Two laptops

Machine A (twin): `uaere demo --port 8765` and allow the port.
Machine B (Unity): set Url to `http://<A-LAN-ip>:8765/api/state`.
Same Wi-Fi. This is how you project Unity on a hall projector while a
laptop under the table runs physics.

## 8. What Unity is not

- Not a second Thorp/Mackenzie implementation.
- Not the evaluator (`uaere evaluate`).
- Not firmware. Raspberry Pi runs Python `uaere.edge`.

When the physics change, restart `uaere demo`. Unity will follow on the
next poll. You do not rebuild the Unity project unless you changed C#.
