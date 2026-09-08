# Models (pluggable backbones)

Any backbone that implements `fit(X, y)` and `evidence(x) -> alpha`
can sit under the Dirichlet trust engine.

| Name | Class | What it is |
|------|-------|------------|
| `logistic_mel` (default) | `LogisticMelBackbone` | class-balanced logistic on Mel statistics, lifted to Dirichlet `alpha = 1 + tau p` |

Register a new one in `src/uaere/models/registry.py`:

```python
BACKBONES["my_net"] = MyBackbone
```

`MyBackbone` must provide `.name`, `.fit` / `.fit_records`, `.evidence`.

TinyCNN remains a MAC-count / L2 head; it is not the default detector.

## Commands

```bash
uaere train --backbone logistic_mel --dataset twin --n 200 --out artifacts/models
uaere predict --wav path/to/clip.wav
```

Checkpoint: `artifacts/models/last.npz`.
