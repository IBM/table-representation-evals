#!/bin/bash
set -euo pipefail  # More strict error handling

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FOLDER_NAME="results"

# Set OpenMP library path for XGBoost on macOS
export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/libomp/21.1.2/lib:$DYLD_LIBRARY_PATH"

# Function to display usage
usage() {
    echo "Usage: $0 <experiment_name> <conda_env_name>"
    echo "  experiment_name: Name of experiment YAML file in approaches/configs/experiment/"
    echo "  conda_env_name: Name of conda environment for the benchmark"
    echo ""
    echo "Example: $0 hytrel_test embedding-benchmark"
    exit 1
}

# Parse command line arguments
EXPERIMENT_NAME="${1:-}"
ENV_NAME="${2:-}"

# Validate arguments
if [[ -z "$EXPERIMENT_NAME" || -z "$ENV_NAME" ]]; then
    usage
fi

# Check if experiment file exists
EXPERIMENT_FILE="approaches/configs/experiment/${EXPERIMENT_NAME}.yaml"
if [[ ! -f "$EXPERIMENT_FILE" ]]; then
    echo "Error: Experiment file '$EXPERIMENT_FILE' not found!"
    exit 1
fi

echo "🚀 Starting benchmark with experiment: $EXPERIMENT_NAME"
echo "📁 Working directory: $SCRIPT_DIR"
echo "🔧 Using conda environment: $ENV_NAME"

# Activate conda environment
eval "$(conda shell.bash hook)"
if ! conda activate "$ENV_NAME" 2>/dev/null; then
    echo "Error: Failed to activate conda environment '$ENV_NAME'"
    echo "Available environments:"
    conda env list
    exit 1
fi

echo "✅ Conda environment activated: $ENV_NAME"

# Function to run benchmark for a single task/dataset
run_benchmark() {
    local experiment="$1"
    local task="$2"
    local dataset="$3"
    
    echo "  🔄 Running: $task/$dataset"
    if python benchmark_src/run_benchmark.py experiment="$experiment" task="$task" dataset_name="$dataset"; then
        echo "  ✅ Completed: $task/$dataset"
        return 0
    else
        echo "  ❌ Failed: $task/$dataset"
        return 1
    fi
}

# Extract and run benchmarks
echo "📋 Extracting benchmark configuration..."
if ! python benchmark_src/extract_config.py experiment="$EXPERIMENT_NAME" | \
awk -v experiment_var="$EXPERIMENT_NAME" '
BEGIN {
    exit_on_error = 0
    total_tasks = 0
    completed_tasks = 0
}
/^TASK:/ {
    current_task = substr($0, 6)
    print "📊 Processing task:", current_task
    total_tasks++
}
/^DATASETS:/ {
    datasets_string = substr($0, 10)
    split(datasets_string, datasets_array, " ")
    
    for (i = 1; i in datasets_array; i++) {
        dataset = datasets_array[i]
        print "  🔄 Running benchmark for dataset:", dataset
        command = "python benchmark_src/run_benchmark.py experiment=" experiment_var " task=\"" current_task "\" dataset_name=\"" dataset "\""
        result = system(command)
        if (result != 0) {
            print "  ❌ Failed:", dataset
            exit_on_error = 1
            break
        } else {
            print "  ✅ Completed:", dataset
            completed_tasks++
        }
    }
    if (exit_on_error == 1) {
        exit 1
    }
}
END {
    print "📈 Summary: " completed_tasks "/" total_tasks " tasks completed"
    if (exit_on_error == 1) {
        exit 1
    }
}'; then
    echo "✅ All benchmarks completed successfully!"
else
    echo "❌ Some benchmarks failed!"
    exit 1
fi

# Gather results
echo "📊 Gathering results..."
if python benchmark_src/utils/gather_results.py results_folder_name="$RESULTS_FOLDER_NAME"; then
    echo "✅ Results gathered successfully!"
else
    echo "⚠️  Warning: Failed to gather results"
fi

echo "🎉 Benchmark completed!"
echo "📁 Results available in: results/$RESULTS_FOLDER_NAME/"