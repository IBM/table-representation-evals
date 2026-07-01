#!/bin/bash
# Wrapper for run_experiments.py that ensures benchmark_env is active.
# The orchestrator then handles per-approach conda envs automatically.
#
# Usage: bash run.sh <run_config_name> [options]
#   bash run.sh schema_linking
#   bash run.sh schema_linking --results-dir results_testing
#   bash run.sh schema_linking --stop-on-error

set -e
eval "$(conda shell.bash hook)"
conda activate benchmark_env
export PYTHONPATH="$(pwd):$PYTHONPATH"
python run_experiments.py "$@"
