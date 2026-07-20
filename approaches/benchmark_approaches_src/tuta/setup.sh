#!/bin/bash
# Setup script for the TUTA approach.
# Run from the repository root: bash approaches/benchmark_approaches_src/tuta/setup.sh
#
# What this script does:
#   1. Clones the TUTA repository into tuta_src/ (model source code only).
#   2. Downloads the pretrained "tuta" checkpoint from Google Drive via gdown.
#   3. Writes the resolved checkpoint path into configs/approaches/tuta.yaml.
#
# Other checkpoint variants (Google Drive):
#   TUTA-explicit: https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW
#   TUTA-base:     https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l
# To use one of these instead, download it manually to tuta_src/tuta.bin and set
# target accordingly in tuta.yaml.
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

# ------------------------------------------------------------------
# 2. Download pretrained weights (TUTA variant) via gdown
# ------------------------------------------------------------------
CKPT_FILE="$TUTA_SRC/tuta.bin"
GDRIVE_ID="1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g"

if [ -f "$CKPT_FILE" ]; then
    echo "Pretrained weights already present at $CKPT_FILE"
else
    echo "Downloading TUTA checkpoint..."
    pip install -q gdown
    gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$CKPT_FILE"
fi

popd > /dev/null

# ------------------------------------------------------------------
# 3. Write the resolved checkpoint_path into the approach YAML
# ------------------------------------------------------------------
YAML_PATH="configs/approaches/tuta.yaml"
REL_CKPT_PATH="approaches/benchmark_approaches_src/tuta/tuta_src/tuta.bin"
ABS_CKPT_FILE="$(pwd)/$REL_CKPT_PATH"

if [ ! -f "$ABS_CKPT_FILE" ]; then
    echo "WARNING: Expected checkpoint at $ABS_CKPT_FILE but file not found. Check the download step above."
else
    sed -i "s|^checkpoint_path:.*|checkpoint_path: \"${REL_CKPT_PATH}\"|" "$YAML_PATH"
    echo "Updated checkpoint_path in $YAML_PATH -> $REL_CKPT_PATH"
fi

echo "=== TUTA setup complete ==="
