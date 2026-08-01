#!/bin/bash
set -e

# Sets up a dedicated conda env for the Pneuma standalone table_retrieval comparison.
# See standalone_comparisons/README.md for why this lives outside the approaches/ plugin system.
# Run from the repository root: bash standalone_comparisons/pneuma/setup.sh

ENV_FILE=".env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "No .env found at '$ENV_FILE'. Run this script from the repository root, after"
    echo "setup_benchmark.sh has already created .env."
    exit 1
fi

update_env() {
    local key="$1"
    local value="$2"

    if grep -q "^${key}=" "$ENV_FILE"; then
        sed -i.bak "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        rm -f "${ENV_FILE}.bak"
    else
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# -----------------------------
# Prompt for OPENAI_API_KEY
# -----------------------------
echo ""
echo "Pneuma (OpenAI backend) needs an OPENAI_API_KEY to summarize tables and judge query matches."
echo ""

current_key=$(grep "^OPENAI_API_KEY=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- || true)

if [[ -n "$current_key" ]]; then
    read -rp "OPENAI_API_KEY already exists in $ENV_FILE. Do you want to update it? (y/N) " update_key
    if [[ "$update_key" =~ ^[Yy]$ ]]; then
        read -rp "Enter your OPENAI_API_KEY: " openai_key
        update_env "OPENAI_API_KEY" "$openai_key"
    fi
else
    read -rp "Enter your OPENAI_API_KEY: " openai_key
    update_env "OPENAI_API_KEY" "$openai_key"
fi

# -----------------------------
# Create the conda env (Pneuma requires Python 3.12)
# -----------------------------
eval "$(conda shell.bash hook)"

conda info --envs | grep -q "^benchmark_env_pneuma " || conda create -n benchmark_env_pneuma python=3.12 --yes
conda activate benchmark_env_pneuma

# benchmark_src (dataset loading + shared retrieval metrics) only -- this comparison doesn't use
# the approaches/ plugin system, so `approaches` itself doesn't need to be installed.
pip install -r reqs_benchmark.txt
pip install -e .

pip install pneuma

conda deactivate

echo ""
echo "Done. Run the comparison with:"
echo "  conda activate benchmark_env_pneuma"
echo "  python standalone_comparisons/pneuma/run_pneuma_table_retrieval.py --dataset <name> --results-dir <path>"
