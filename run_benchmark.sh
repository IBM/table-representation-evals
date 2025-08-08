#!/bin/bash
set -e

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
/^DATASETS:/ {
    # Extract the datasets string
    datasets_string = substr($0, 10)
    # Split the datasets string by space
    split(datasets_string, datasets_array, " ")

    # Loop through the datasets for the current task
    for (i in datasets_array) {
        dataset = datasets_array[i]
        print "  Running benchmark for dataset:", dataset
        # Execute the benchmark script for each task/dataset combination
        # Use the Awk variable 'experiment_var' in the system command
        command = "python benchmark_src/run_benchmark.py experiment=" experiment_var " task=\"" current_task "\" dataset_name=\"" dataset "\""
        result = system(command)
        if (result != 0) {
                    exit_on_error = 1 # Set flag if command fails
                    break # Exit the inner loop (datasets)
                }
            }
    if (exit_on_error == 1) {
        exit 1 # Exit awk if a command failed in the loop
    }
}
END {
    if (exit_on_error == 1) {
        exit 1 # Ensure awk exits with error if any command failed
    }
}
'

python benchmark_src/utils/gather_results.py results_folder_name=$RESULTS_FOLDER_NAME

echo "-------------------------------------"
echo "Completed the run_benchmark.sh script"