#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${WOK_SIM_PYTHON:-$project_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3.11"
fi

"$python_bin" -m pytest -q
"$python_bin" -m wok_sim.cli smoke-test --config configs/test.yaml --seed 1

