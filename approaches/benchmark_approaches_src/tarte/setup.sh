#!/usr/bin/env bash
# Setup script for the TARTE embedding approach.
# Run from the project root inside the embedding-benchmark conda environment:
#   conda activate embedding-benchmark
#   bash approaches/benchmark_approaches_src/tarte/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMODULE_DIR="${SCRIPT_DIR}/tarte-ai"

echo "=== Installing tarte-ai (patched fork via submodule) ==="
if [ ! -f "${SUBMODULE_DIR}/pyproject.toml" ]; then
    echo "ERROR: submodule not initialised. Run:"
    echo "  git submodule update --init approaches/benchmark_approaches_src/tarte/tarte-ai"
    exit 1
fi
pip install -e "${SUBMODULE_DIR}"

echo ""
echo "=== Installing fasttext-wheel (pre-built bindings for Meta fastText) ==="
pip install fasttext-wheel

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
