#!/bin/bash
set -e

RESULTS_FOLDER_NAME="results_testing" 

## Need to activate conda environment
ENV_NAME="$1"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

##################################################
########## Test row similarity search ############
##################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=row_similarity_search  dataset_name=Amazon-Google  

##################################################
########## Test more similar than ################
##################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=more_similar_than  dataset_name=wikidata_books@@column_naming::original

##################################################
########## Test predictive ML     ################
##################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=predictive_ml  dataset_name=credit-g
python benchmark_src/run_benchmark.py experiment=test_before_commit task=predictive_ml  dataset_name=anneal
python benchmark_src/run_benchmark.py experiment=test_before_commit task=predictive_ml  dataset_name=healthcare_insurance_expenses 



echo "-------------------------------------"
echo "Completed the test experiments, gathering the results next"

python benchmark_src/results_processing/gather_results.py results_folder_name=$RESULTS_FOLDER_NAME

echo "-------------------------------------"
echo "Completed the run_test_before_commit.sh script"