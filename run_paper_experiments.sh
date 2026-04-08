#!/bin/bash
set -e # exit on error

# initialize conda
eval "$(conda shell.bash hook)"

#============================
#run sentence transformer
#============================
./run_benchmark.sh paper_minilm benchmark_env

./run_benchmark.sh paper_granite-r2 benchmark_env

#============================
#run GritLM Experiments
#============================
./run_benchmark.sh paper_gritlm benchmark_env_gritlm

#============================
#run XGBoost as Baseline
#============================
./run_benchmark.sh paper_XGBoost benchmark_env

#============================
#run Tabula Experiments
#============================
./run_benchmark.sh paper_tabula_row benchmark_env
./run_benchmark.sh paper_tabula_pred benchmark_env   

#============================
#run hytrel Experiments
#============================
./run_benchmark.sh paper_hytrel benchmark_env_hytrel

#============================
#run sap rpt oss Experiments
#============================
./run_benchmark.sh paper_sap-rpt-1_row benchmark_env
./run_benchmark.sh paper_sap-rpt-1_pred benchmark_env

#============================
#run tabpfn Experiments
#============================
./run_benchmark.sh paper_tabpfn_row benchmark_env
./run_benchmark.sh paper_tabpfn_pred benchmark_env

#============================
#run tabicl Experiments
#============================
./run_benchmark.sh paper_tabicl_row benchmark_env_tabicl
./run_benchmark.sh paper_tabicl_pred benchmark_env_tabicl
