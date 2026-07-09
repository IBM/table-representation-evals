#!/bin/bash
# Setup script for the TABBIE approach.
# Run from the repository root: bash approaches/benchmark_approaches_src/tabbie/setup.sh
#
# What this script does:
#   1. Clones the TABBIE repository (for CLS .npy files and source reference).
#   2. Checks for the pretrained 'mix' model weights; if missing, prompts for a
#      Google Drive download URL and fetches them (see step 2 below).
#   3. Updates approaches/configs/approach/tabbie.yaml with resolved paths.
#
# Requirements:
#   - git
#   - run inside benchmark_env (already has torch/transformers via reqs_benchmark.txt)
#
# If the weights aren't already present, the script will ask you to open
# https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0 ,
# start (then cancel) downloading 'mix.tar.gz' in your browser, and paste the
# resolved download URL — same pattern as hytrel/setup.sh's checkpoint download.
# If you already have mix.tar.gz/mix.tar, place it at
# approaches/benchmark_approaches_src/tabbie/tabbie_src/model/ to skip the prompt.
set -e

echo "=== Setting up TABBIE approach ==="

pushd .
cd approaches/benchmark_approaches_src/tabbie

TABBIE_SRC="tabbie_src"

# ------------------------------------------------------------------
# 1. Clone TABBIE repository (for clscol.npy, clsrow.npy, bert vocab)
# ------------------------------------------------------------------
if [ ! -d "$TABBIE_SRC" ] || [ -z "$(ls -A "$TABBIE_SRC" 2>/dev/null)" ]; then
    echo "Cloning TABBIE repository..."
    git clone https://github.com/SFIG611/tabbie.git "$TABBIE_SRC"
else
    echo "TABBIE source already present at $TABBIE_SRC"
fi

CLS_COL="$TABBIE_SRC/data/clscol.npy"
CLS_ROW="$TABBIE_SRC/data/clsrow.npy"

if [ ! -f "$CLS_COL" ] || [ ! -f "$CLS_ROW" ]; then
    echo "ERROR: CLS token files not found after clone:"
    echo "  Expected: $CLS_COL"
    echo "  Expected: $CLS_ROW"
    echo "Check the TABBIE repo structure."
    exit 1
fi
echo "CLS token files present."

# ------------------------------------------------------------------
# 2. Locate pretrained weights (mix variant); download interactively if
#    not already present at $MODEL_DIR.
#    Source: https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0
#    File:   mix.tar or mix.tar.gz  (~600 MB)
# ------------------------------------------------------------------
MODEL_DIR="$TABBIE_SRC/model"
MODEL_EXTRACTED="$MODEL_DIR/mix"

mkdir -p "$MODEL_DIR"

# Accept either mix.tar or mix.tar.gz
if [ -f "$MODEL_DIR/mix.tar.gz" ]; then
    MODEL_ARCHIVE="$MODEL_DIR/mix.tar.gz"
    TAR_FLAGS="-xzf"
elif [ -f "$MODEL_DIR/mix.tar" ]; then
    MODEL_ARCHIVE="$MODEL_DIR/mix.tar"
    TAR_FLAGS="-xf"
else
    MODEL_ARCHIVE=""
fi

if [ -f "$MODEL_EXTRACTED/config.json" ] && { [ -f "$MODEL_EXTRACTED/weights.th" ] || [ -f "$MODEL_EXTRACTED/best.th" ]; }; then
    echo "Pretrained weights already present at $MODEL_EXTRACTED"
else
    if [ -z "$MODEL_ARCHIVE" ]; then
        echo "--------------- Please take action: --------------------------------------"
        echo "The pretrained weights (mix.tar.gz, ~600 MB) are hosted on Google Drive and"
        echo "can't be fetched with a plain URL:"
        echo "  https://drive.google.com/drive/folders/1vAMv09j-VlWHKd5djiRGuC16yb-lhJO0"
        echo "Open that folder, start downloading 'mix.tar.gz', then cancel the download"
        echo "in your browser and copy the resolved download URL (browser download manager"
        echo "or dev tools network tab)."
        echo "----------------------------------------------------------------------------"

        read -p "Paste the resolved download URL for mix.tar.gz: " URL
        if [[ -z "$URL" ]]; then
            echo "No URL entered. Exiting."
            exit 1
        fi

        MODEL_ARCHIVE="$MODEL_DIR/mix.tar.gz"
        TAR_FLAGS="-xzf"
        echo "Downloading mix.tar.gz..."
        curl --http1.1 -L -o "$MODEL_ARCHIVE" "$URL"
    else
        echo "Archive found at $MODEL_ARCHIVE"
    fi

    echo "Extracting weights..."
    # Extract to a scratch dir so $MODEL_EXTRACTED is never among the items being moved.
    EXTRACT_TMP=$(mktemp -d)
    tar $TAR_FLAGS "$MODEL_ARCHIVE" -C "$EXTRACT_TMP"

    EXTRACTED_CFG=$(find "$EXTRACT_TMP" -name "config.json" | head -1)
    if [ -z "$EXTRACTED_CFG" ]; then
        echo "ERROR: config.json not found after extraction. Contents of $EXTRACT_TMP:"
        ls -R "$EXTRACT_TMP"
        rm -rf "$EXTRACT_TMP"
        exit 1
    fi

    mkdir -p "$MODEL_EXTRACTED"
    mv "$(dirname "$EXTRACTED_CFG")"/* "$MODEL_EXTRACTED"/
    rm -rf "$EXTRACT_TMP"

    echo "Weights extracted to $MODEL_EXTRACTED"
fi

popd  # back to repo root

# ------------------------------------------------------------------
# 3. Update tabbie.yaml with resolved relative paths
# ------------------------------------------------------------------
REPO_ROOT="$(pwd)"
YAML_PATH="configs/approaches/tabbie.yaml"

REL_ARCHIVE="approaches/benchmark_approaches_src/tabbie/tabbie_src/model/mix/model.tar.gz"
REL_CLS_COL="approaches/benchmark_approaches_src/tabbie/tabbie_src/data/clscol.npy"
REL_CLS_ROW="approaches/benchmark_approaches_src/tabbie/tabbie_src/data/clsrow.npy"

# The archive inside the extracted directory is the AllenNLP model.tar.gz
# (the mix.tar.gz contains a model.tar.gz as the AllenNLP archive)
INNER_ARCHIVE="$REPO_ROOT/approaches/benchmark_approaches_src/tabbie/tabbie_src/model/mix/model.tar.gz"

if [ ! -f "$INNER_ARCHIVE" ]; then
    # Fallback: the extracted directory itself may be the archive contents
    # In that case use weights.th + config.json directly via a re-pack
    WEIGHTS="$REPO_ROOT/approaches/benchmark_approaches_src/tabbie/tabbie_src/model/mix/weights.th"
    CONFIG="$REPO_ROOT/approaches/benchmark_approaches_src/tabbie/tabbie_src/model/mix/config.json"
    # Accept best.th (mix variant) or weights.th (freq/other variants)
    BEST_TH="$REPO_ROOT/approaches/benchmark_approaches_src/tabbie/tabbie_src/model/mix/best.th"
    if [ -f "$BEST_TH" ]; then
        WEIGHTS="$BEST_TH"
        WTS_FILENAME="best.th"
    fi
    if [ -f "$WEIGHTS" ] && [ -f "$CONFIG" ]; then
        echo "Repacking extracted weights into model.tar.gz for archive loading..."
        tar -czf "$INNER_ARCHIVE" \
            -C "$REPO_ROOT/approaches/benchmark_approaches_src/tabbie/tabbie_src/model/mix" \
            "$WTS_FILENAME" config.json
        echo "Created $INNER_ARCHIVE"
    else
        echo "WARNING: Could not locate weights.th or config.json."
        echo "Please set archive_path in tabbie.yaml manually."
    fi
fi

if [ -f "$INNER_ARCHIVE" ]; then
    sed -i "s|^archive_path:.*|archive_path: \"${REL_ARCHIVE}\"|" "$YAML_PATH"
fi
sed -i "s|^cls_col_path:.*|cls_col_path: \"${REL_CLS_COL}\"|" "$YAML_PATH"
sed -i "s|^cls_row_path:.*|cls_row_path: \"${REL_CLS_ROW}\"|" "$YAML_PATH"

echo "Updated paths in $YAML_PATH"
echo "=== TABBIE setup complete ==="
