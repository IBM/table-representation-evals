#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

export PYTHONPATH=$(pwd):$PYTHONPATH

conda activate benchmark_env
python run_experiments.py paper_benchmark_env

conda activate benchmark_env_gritlm
python run_experiments.py paper_gritlm

conda activate benchmark_env_hytrel
python run_experiments.py paper_hytrel

conda activate benchmark_env_tabicl
python run_experiments.py paper_tabicl
