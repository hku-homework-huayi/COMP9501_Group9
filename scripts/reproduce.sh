#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
experiment_output="${1:-results/main}"
.venv/bin/python scripts/download_data.py
.venv/bin/python -m pytest -q
.venv/bin/python -u scripts/run_experiments.py --phase all --output "$experiment_output"
.venv/bin/python scripts/verify_results.py --output "$experiment_output"
.venv/bin/python scripts/build_report.py --output "$experiment_output"
