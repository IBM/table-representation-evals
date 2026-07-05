#!/bin/bash
# Setup script for the TaBERT approach.
# Run from the repository root: bash approaches/benchmark_approaches_src/tabert/setup.sh
set -e

echo "=== Setting up TaBERT approach ==="

pushd .
cd approaches/benchmark_approaches_src/tabert

# ------------------------------------------------------------------
# 1. Ensure the tabert_src submodule is checked out
# ------------------------------------------------------------------
TABERT_DIR="tabert_src"
if [ ! -d "$TABERT_DIR" ] || [ -z "$(ls -A "$TABERT_DIR" 2>/dev/null)" ]; then
    echo "Cloning TaBERT repository..."
    git clone https://github.com/facebookresearch/TaBERT.git "$TABERT_DIR"
else
    echo "TaBERT source already present."
fi

# ------------------------------------------------------------------
# 2. Install TaBERT and its dependencies
# ------------------------------------------------------------------
echo "Installing TaBERT package..."
pip install --editable "$TABERT_DIR"

# torch-scatter is required for column-aggregation in TaBERT.
# fairseq is NOT required – its only use (distributed_utils) is already stubbed out.
pip install torch-scatter

# ------------------------------------------------------------------
# 3. Download pre-trained checkpoint (TaBERT_Base_K=1)
# ------------------------------------------------------------------
CKPT_DIR="$TABERT_DIR/pretrained_weights"
MODEL_DIR="$CKPT_DIR/tabert_base_k1"
MODEL_BIN="$MODEL_DIR/model.bin"
GDRIVE_ID="1-pdtksj9RzC4yEqdrJQaZu4-dIEXZbM9"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_BIN" ]; then
    echo "Pre-trained weights already present at $MODEL_BIN"
else
    echo "Downloading TaBERT_Base_(K=1) checkpoint..."
    pip install -q gdown

    TAR_FILE="$CKPT_DIR/TaBERT_Base_K1.tar.gz"
    gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O "$TAR_FILE"

    echo "Extracting checkpoint..."
    tar -xzf "$TAR_FILE" -C "$CKPT_DIR"
    rm "$TAR_FILE"

    # Locate the extracted model.bin (folder name may vary) and normalise to MODEL_DIR
    EXTRACTED_BIN=$(find "$CKPT_DIR" -name "model.bin" | head -1)
    if [ -z "$EXTRACTED_BIN" ]; then
        echo "ERROR: Could not find model.bin after extraction. Contents of $CKPT_DIR:"
        ls -R "$CKPT_DIR"
        exit 1
    fi

    EXTRACTED_DIR=$(dirname "$EXTRACTED_BIN")
    if [ "$EXTRACTED_DIR" != "$MODEL_DIR" ]; then
        mv "$EXTRACTED_DIR"/* "$MODEL_DIR"/
        rmdir "$EXTRACTED_DIR" 2>/dev/null || true
    fi

    echo "Checkpoint saved to $MODEL_BIN"
fi

# ------------------------------------------------------------------
# 4. Write the resolved model_path into the approach YAML so the
#    benchmark picks it up without any manual editing.
# ------------------------------------------------------------------
# Path relative to the repo root (what Hydra / get_original_cwd sees)
popd  # back to repo root

REPO_ROOT="$(pwd)"
YAML_PATH="approaches/configs/approach/tabert.yaml"
REL_MODEL_PATH="approaches/benchmark_approaches_src/tabert/tabert_src/pretrained_weights/tabert_base_k1/model.bin"
ABS_MODEL_BIN="$REPO_ROOT/$REL_MODEL_PATH"

if [ ! -f "$ABS_MODEL_BIN" ]; then
    echo "WARNING: Expected model at $ABS_MODEL_BIN but file not found. Check the extraction step above."
else
    # Replace the model_path line (handles both '~' and an existing path)
    sed -i "s|^model_path:.*|model_path: \"${REL_MODEL_PATH}\"|" "$YAML_PATH"
    echo "Updated model_path in $YAML_PATH -> $REL_MODEL_PATH"
fi

echo "=== TaBERT setup complete ==="
