#!/bin/bash
# Runs the full table-level paper sweep (AyeenP, TaDA 2026 workshop paper).
# Each approach has two configs:
#   table_paper_<approach>.yaml          - all 3 tasks at table_row_limit=100
#                                          (transformer approaches sweep serialization
#                                          markdown/csv inside)
#   table_paper_<approach>_headers.yaml  - table_retrieval only at table_row_limit=0
#                                          (markdown only -- csv@0 is redundant)
#
# All runs land in /results/<approach>/<config_hash>/.../ under
# benchmark_output_dir=table_paper_experiments.

set -e

eval "$(conda shell.bash hook)"

#============================
# Sentence Transformer (MiniLM)
#============================
./run_benchmark.sh table_paper_minilm benchmark_env
./run_benchmark.sh table_paper_minilm_headers benchmark_env

#============================
# Sentence Transformer (Granite-R2)
#============================
./run_benchmark.sh table_paper_granite-r2 benchmark_env
./run_benchmark.sh table_paper_granite-r2_headers benchmark_env

#============================
# GritLM
#============================
./run_benchmark.sh table_paper_gritlm benchmark_env_gritlm
./run_benchmark.sh table_paper_gritlm_headers benchmark_env_gritlm

#============================
# HyTrel
#============================
./run_benchmark.sh table_paper_hytrel benchmark_env_hytrel
./run_benchmark.sh table_paper_hytrel_headers benchmark_env_hytrel

#============================
# Hashing BoW baseline
#============================
./run_benchmark.sh table_paper_hashing benchmark_env
./run_benchmark.sh table_paper_hashing_headers benchmark_env

#============================
# TF-IDF BoW baseline
#============================
./run_benchmark.sh table_paper_tfidf benchmark_env
./run_benchmark.sh table_paper_tfidf_headers benchmark_env
