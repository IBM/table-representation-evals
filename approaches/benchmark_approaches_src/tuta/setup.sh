#!/bin/bash
# Setup script for the TUTA approach.
# Run from the repository root: bash approaches/benchmark_approaches_src/tuta/setup.sh
#
# What this script does:
#   1. Clones the TUTA repository into tuta_src/ (model source code only).
#   2. Prints download instructions for the pretrained weights.
#
# Pretrained weight variants (Google Drive, manual download required):
#   TUTA:          https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g
#   TUTA-explicit: https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW
#   TUTA-base:     https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l
#
# After downloading, place the .bin file at the path you set in
#   approaches/configs/approach/tuta.yaml -> checkpoint_path
set -e

echo "=== Setting up TUTA approach ==="

pushd . > /dev/null
cd approaches/benchmark_approaches_src/tuta

TUTA_SRC="tuta_src"

# ------------------------------------------------------------------
# 1. Clone TUTA repository
# ------------------------------------------------------------------
if [ ! -d "$TUTA_SRC" ] || [ -z "$(ls -A "$TUTA_SRC" 2>/dev/null)" ]; then
    echo "Cloning TUTA repository..."
    git clone https://github.com/microsoft/TUTA_table_understanding.git "$TUTA_SRC"
else
    echo "TUTA source already present at $TUTA_SRC"
fi

# Verify the model package is importable
if [ ! -f "$TUTA_SRC/tuta/model/backbones.py" ]; then
    echo "ERROR: Expected $TUTA_SRC/tuta/model/backbones.py not found."
    echo "Check the TUTA repo structure."
    exit 1
fi
echo "TUTA model source verified."

popd > /dev/null

# ------------------------------------------------------------------
# 2. Print weight download instructions
# ------------------------------------------------------------------
echo ""
echo "=== Manual step required: download pretrained weights ==="
echo ""
echo "Choose one of the following TUTA variants and download the .bin file:"
echo ""
echo "  TUTA (recommended):"
echo "    https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g"
echo ""
echo "  TUTA-explicit:"
echo "    https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW"
echo ""
echo "  TUTA-base:"
echo "    https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l"
echo ""
echo "Place the downloaded file at:"
echo "  approaches/benchmark_approaches_src/tuta/tuta_src/tuta_model.bin"
echo ""
echo "Then update approaches/configs/approach/tuta.yaml:"
echo "  checkpoint_path: \"approaches/benchmark_approaches_src/tuta/tuta_src/tuta_model.bin\""
echo "  target: \"tuta\"   # or tuta_explicit / base to match the variant"
echo ""
echo "=== TUTA setup complete (weights still needed) ==="
