#!/bin/bash
set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

## Need to activate conda environment
ENV_NAME="$1"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python run_experiments.py test_before_commit --results-dir "results_testing/$TIMESTAMP" --stop-on-error

echo "-------------------------------------"
echo "Completed the run_test_before_commit.sh script"
echo "All tests passed successfully! Note that some tests were run on reduced dataset sizes."
