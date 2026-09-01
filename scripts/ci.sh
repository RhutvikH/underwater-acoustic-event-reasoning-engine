#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests
python -c "from uaere.cli import main; raise SystemExit(main(['data','summarize','--n','8']))"
# Guard: no fixed energy wake in the live pipeline.
if grep -R --include='*.py' -n 'if energy >' src/uaere | grep -v baseline_energy_threshold; then
  echo "fixed energy threshold leaked into production code" >&2
  exit 1
fi
echo CI_OK
