#!/usr/bin/env bash
# Usage: ./reproduce.sh [figures|experiments|all]   (default: figures)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
stage="${1:-figures}"

if [[ "$stage" == experiments || "$stage" == all ]]; then
    "$PY" -m experiments.run_detection
    "$PY" -m experiments.run_rca
    "$PY" -m experiments.run_sensitivity detection
    "$PY" -m experiments.run_sensitivity rca
    "$PY" -m experiments.run_scalability
    "$PY" -m experiments.run_single_window
fi

if [[ "$stage" == figures || "$stage" == all ]]; then
    "$PY" -m figures.fig1_framework
    "$PY" -m figures.fig2_method
    "$PY" -m figures.fig3_fig4
    "$PY" -m figures.fig5_sensitivity
    "$PY" -m figures.fig6_scalability
    "$PY" -m figures.fig7_single_window
    "$PY" -m figures.table_per_fault
fi
