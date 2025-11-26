#!/bin/bash
set -e

# Set OpenMP library path for XGBoost on macOS
export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/libomp/21.1.2/lib:$DYLD_LIBRARY_PATH"


export PYTHONPATH=$(pwd):$PYTHONPATH

# Access the first command-line argument using $1
EXPERIMENT_NAME="$1"
ENV_NAME="$2"

RESULTS_FOLDER_NAME="results" 

# Check if an argument was provided
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <name_of_experiment_yaml_file> (has to be created in approaches/configs/experiment)"
    exit 1
fi

# Check if an argument was provided
if [ -z "$ENV_NAME" ]; then
    echo "Usage: $0 <name_of_environment> Please give the name of your conda environment for the benchmark project."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Pass the EXPERIMENT_NAME to Awk using -v
python benchmark_src/extract_config.py experiment="$EXPERIMENT_NAME" | \
awk -v experiment_var="$EXPERIMENT_NAME" '
BEGIN {
    # Initialize a variable to track if any command failed
    exit_on_error = 0
}
/^TASK:/ {
    # Extract the task name
    current_task = substr($0, 6)
    print "Processing task:", current_task
}
/^DATASET:/ {
    # Extract the datasets string
    current_dataset = substr($0, 9)
}
/^VARIATION:/ {
    variation = substr($0, 11)
    print "  Running benchmark for variation:", variation

    # Execute benchmark script for this variation
    command = "python benchmark_src/run_benchmark.py experiment=" experiment_var \
              " task=\"" current_task "\"" \
              " dataset_name=\"" variation "\""

    result = system(command)
    if (result != 0) {
        exit_on_error = 1
        exit 1
    }
}
END {
    if (exit_on_error == 1) {
        exit 1 # Ensure awk exits with error if any command failed
    }
}
'

echo "-------------------------------------"
echo "Completed the experiments, gathering the results next"

python benchmark_src/results_processing/gather_results.py results_folder_name=$RESULTS_FOLDER_NAME

echo "-------------------------------------"
echo "Completed the run_benchmark.sh script"