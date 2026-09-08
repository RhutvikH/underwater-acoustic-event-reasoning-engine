# Which guide to open

Start here if you are lost. Read **one** row, not all of them.

| I am… | Open this | Then maybe |
|-------|-----------|------------|
| Showing this in a viva in ten minutes | [BEGINNER.md](BEGINNER.md) | run `uaere demo` |
| A teammate who has never used Python | [BEGINNER.md](BEGINNER.md) | [USER_GUIDE.md](USER_GUIDE.md) §1–2 |
| Running CLI / Python API / troubleshooting | [USER_GUIDE.md](USER_GUIDE.md) | — |
| **Changing the code** (no AI) | [DEVELOPER.md](DEVELOPER.md) | tests in `tests/` |
| Putting the twin on a Unity stage | [UNITY.md](UNITY.md) | `unity/AHAIF/README.md` |
| Grading a run / writing the results table | [EVALUATION.md](EVALUATION.md) | `uaere evaluate` |
| Asking “is this actually new?” | [UNIQUENESS.md](UNIQUENESS.md) | [LIT_REVIEW_2024_2026.md](LIT_REVIEW_2024_2026.md) · `paper/claim_chart.md` |
| 2024–2026 papers and patents (with links) | [LIT_REVIEW_2024_2026.md](LIT_REVIEW_2024_2026.md) | [UNIQUENESS.md](UNIQUENESS.md) |
| Filing IP | `paper/complete_specification.md` | `paper/invention_disclosure.md` |
| Writing the journal paper | `paper/manuscript.md` | `paper/formulation.md` |
| Flashing a Raspberry Pi | [RASPBERRY_PI.md](RASPBERRY_PI.md) | [DEVELOPER.md](DEVELOPER.md) § edge |
| Need a Gantt or flowchart for slides | [DIAGRAMS.md](DIAGRAMS.md) | — |
| Faculty constraints | [non_negotiables.txt](non_negotiables.txt) | [review_1.pdf](review_1.pdf) |
| All equations (GitHub math) | [MATH.md](MATH.md) | `src/uaere/math/` |
| What data we have | [DATASETS.md](DATASETS.md) | `uaere train` / `predict` |
| Which model / how to plug one | [MODELS.md](MODELS.md) | `src/uaere/models/` |
| 50 unique ideas (ranked) | [IDEAS.md](IDEAS.md) | — |
| Map of every file | [INDEX.md](INDEX.md) | — |

## One command each

```bash
source .venv/bin/activate
uaere demo --nodes 8 --port 8765          # GUI + Unity JSON
uaere evaluate --out artifacts/eval       # frozen evaluator (pass/fail)
python -m pytest tests -q                 # developer gate
```
