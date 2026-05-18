#!/bin/bash
set -e

# Set OpenMP library path for XGBoost on macOS
export DYLD_LIBRARY_PATH="/opt/homebrew/Cellar/libomp/21.1.2/lib:$DYLD_LIBRARY_PATH"

export PYTHONPATH=$(pwd):$PYTHONPATH

EXPERIMENT_NAME="$1"
ENV_NAME="$2"
RESULTS_FOLDER_NAME="results"

# ----------------------------
# Argument checks
# ----------------------------
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <name_of_experiment_yaml_file>"
    exit 1
fi

if [ -z "$ENV_NAME" ]; then
    echo "Usage: $0 <name_of_environment>"
    exit 1
fi

# ----------------------------
# Activate conda environment
# ----------------------------
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ----------------------------
# Retry wrapper (LIVE OUTPUT)
# ----------------------------
run_with_retry() {
    local cmd="$1"
    local max_retries=3
    local attempt=1

    while [ $attempt -le $max_retries ]; do
        echo "    Attempt $attempt / $max_retries"

        logfile=$(mktemp)

        # Run command: show output LIVE + save to logfile
        bash -c "$cmd" 2>&1 | tee "$logfile"
        status=${PIPESTATUS[0]}

        if [ $status -eq 0 ]; then
            echo "    Success"
            rm -f "$logfile"
            return 0
        fi

        # Detect recursion error from FULL log
        if grep -qiE "RecursionError|maximum recursion depth exceeded" "$logfile"; then
            echo "    ⚠️ Recursion depth error detected, retrying..."
        else
            echo "    ❌ Non-retryable error encountered (see output above)"
            rm -f "$logfile"
            return 1
        fi

        rm -f "$logfile"
        attempt=$((attempt + 1))
        sleep 2
    done

    echo "    ❌ Failed after $max_retries attempts"
    return 1
}

# ----------------------------
# Main loop (parse config)
# ----------------------------
current_task=""

python benchmark_src/extract_config.py experiment="$EXPERIMENT_NAME" | \
while IFS= read -r line; do

    if [[ "$line" == TASK:* ]]; then
        current_task="${line#TASK:}"
        echo "Processing task: $current_task"

    elif [[ "$line" == VARIATION:* ]]; then
        variation="${line#VARIATION:}"

        echo "  Running benchmark for variation: $variation"

        cmd="python benchmark_src/run_benchmark.py \
            experiment=$EXPERIMENT_NAME \
            task=\"$current_task\" \
            dataset_name=\"$variation\""

        run_with_retry "$cmd" || exit 1
    fi

done

# ----------------------------
# Gather results
# ----------------------------
echo "-------------------------------------"
echo "Completed the experiments, gathering the results next"

python benchmark_src/results_processing/gather_results.py \
    results_folder_name=$RESULTS_FOLDER_NAME

echo "-------------------------------------"
echo "Completed the run_benchmark.sh script"