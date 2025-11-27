#!/bin/bash
set -e

RESULTS_FOLDER_NAME="results_testing" 

export PYTHONPATH=$(pwd):$PYTHONPATH

## Need to activate conda environment
ENV_NAME="$1"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Format: YYYYMMDD_HHMMSS
CURRENT_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

##################################################
########## Test row similarity search ############
##################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=row_similarity_search  dataset_name=Amazon-Google  benchmark_output_dir=$CURRENT_TIMESTAMP

##################################################
########## Test column similarity search ############
##################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=column_similarity_search  dataset_name=valentine  benchmark_output_dir=$CURRENT_TIMESTAMP
python benchmark_src/run_benchmark.py experiment=test_before_commit task=column_similarity_search  dataset_name=opendata  benchmark_output_dir=$CURRENT_TIMESTAMP

##################################################
########## Test more similar than ################
##################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=more_similar_than  dataset_name=wikidata_books@@column_naming::original benchmark_output_dir=$CURRENT_TIMESTAMP

####################################################################################################
########## Test predictive ML (binary / multi-class classification and regression)    ##############
####################################################################################################
python benchmark_src/run_benchmark.py experiment=test_before_commit task=predictive_ml  dataset_name=credit-g benchmark_output_dir=$CURRENT_TIMESTAMP
python benchmark_src/run_benchmark.py experiment=test_before_commit task=predictive_ml  dataset_name=anneal benchmark_output_dir=$CURRENT_TIMESTAMP
python benchmark_src/run_benchmark.py experiment=test_before_commit task=predictive_ml  dataset_name=healthcare_insurance_expenses benchmark_output_dir=$CURRENT_TIMESTAMP

echo "-------------------------------------"
echo "Completed the test experiments, gathering the results next"

python benchmark_src/results_processing/gather_results.py results_folder_name=$RESULTS_FOLDER_NAME

echo "-------------------------------------"
echo "Completed the run_test_before_commit.sh script"