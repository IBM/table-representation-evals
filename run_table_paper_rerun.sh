#!/bin/bash
set -e

eval "$(conda shell.bash hook)"

#=========================================
# Sentence transformer: MiniLM
#=========================================
./run_benchmark.sh table_paper_rerun_minilm benchmark_env
./run_benchmark.sh table_paper_rerun_minilm_headers benchmark_env

#=========================================
# Sentence transformer: Granite-R2
#=========================================
./run_benchmark.sh table_paper_rerun_granite benchmark_env
./run_benchmark.sh table_paper_rerun_granite_headers benchmark_env

#=========================================
# GritLM
#=========================================
./run_benchmark.sh table_paper_rerun_gritlm benchmark_env_gritlm
./run_benchmark.sh table_paper_rerun_gritlm_headers benchmark_env_gritlm

#=========================================
# HyTrel
#=========================================
./run_benchmark.sh table_paper_rerun_hytrel benchmark_env_hytrel
./run_benchmark.sh table_paper_rerun_hytrel_headers benchmark_env_hytrel

#=========================================
# Hashing
#=========================================
./run_benchmark.sh table_paper_rerun_hashing benchmark_env
./run_benchmark.sh table_paper_rerun_hashing_headers benchmark_env

#=========================================
# TF-IDF
#=========================================
./run_benchmark.sh table_paper_rerun_tfidf benchmark_env
./run_benchmark.sh table_paper_rerun_tfidf_headers benchmark_env

#=========================================
# Gather results
#=========================================
echo "-------------------------------------"
echo "Completed the experiments, gathering the results next"

python benchmark_src/results_processing/gather_results.py results_folder_name=results
