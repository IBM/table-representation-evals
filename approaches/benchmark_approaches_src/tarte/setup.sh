#!/usr/bin/env bash
# Setup script for the TARTE embedding approach.
# Run from the project root inside the benchmark_env conda environment:
#   conda activate benchmark_env
#   bash approaches/benchmark_approaches_src/tarte/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARTE_AI_DIR="${SCRIPT_DIR}/tarte-ai"

# tarte-ai's own pyproject.toml depends on "fasttext", which builds Meta's
# fastText bindings from source and commonly fails without a C++ toolchain.
# We install tarte-ai with --no-deps and install its other dependencies
# explicitly, pulling fasttext from conda-forge (prebuilt binary) instead.
echo "=== Installing tarte-ai (vendored copy at ${TARTE_AI_DIR}) ==="
if [ ! -f "${TARTE_AI_DIR}/pyproject.toml" ]; then
    echo "ERROR: tarte-ai not found at ${TARTE_AI_DIR}."
    exit 1
fi
pip install --no-deps -e "${TARTE_AI_DIR}"

echo ""
echo "=== Installing tarte-ai dependencies ==="
pip install numpy pandas scipy scikit-learn torch torcheval huggingface_hub \
    catboost tabpfn xgboost skrub pyarrow fastparquet

echo ""
echo "=== Installing fasttext (prebuilt binary via conda-forge) ==="
conda install -y -c conda-forge fasttext=0.9.2

echo ""
echo "=== TARTE setup complete ==="
echo ""
echo "On first use, TARTE will automatically download two files via HuggingFace Hub:"
echo "  1. tarte_pretrained_weights.pt + tarte_pretrained_configs.json"
echo "     (~500 MB, from huggingface.co/inria-soda/tarte)"
echo "  2. cc.en.300.bin (FastText English model)"
echo "     (~4.2 GB, from huggingface.co/hi-paris/fastText)"
echo ""
echo "Both are cached locally after the first download."
