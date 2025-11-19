#!/bin/bash
set -e 

# Setup script for HyTrel approach
echo "Setting up HyTrel approach..."

pushd .
cd approaches/benchmark_approaches_src/hytrel


# =================================================
# Clone the HyTrel repository if it doesn't exist
# =================================================
HYTREL_DIR="hytrel_src"
if [ ! -d "$HYTREL_DIR" ] || [ -z "$(ls -A "$HYTREL_DIR")" ]; then
    echo "Cloning HyTrel repository..."
    git clone https://github.com/awslabs/hypergraph-tabular-lm.git $HYTREL_DIR
else
    echo "HyTrel repository already exists, updating..."
    cd $HYTREL_DIR
    git pull
    cd ..
fi
# =================================================
# Download the checkpoints (if not already done)
# =================================================
# Destination folder (already exists)
DEST_PATH="hytrel_src/checkpoints"

shopt -s nullglob dotglob
# Get all files/folders in DEST_PATH
contents=("$DEST_PATH"/*)


# Check if the directory contains exactly one file called readme.txt
if [ ${#contents[@]} -eq 1 ] && [ "$(basename "${contents[0]}")" = "readme.txt" ]; then
    echo "$DEST_PATH contains only readme.txt, need to download the hytrel checkpoints"

    # 1. Prompt user for the temporary download URL
    echo "--------------- Please take action: --------------------------------------"
    echo "Checkpoints and data can be found here:"
    echo "Link: https://1drv.ms/f/s!Ap3OTapL7f2GgZY3mrF8nmHv17TRyg?e=kZE4Ox"
    echo "Pwd: hytrel"
    echo "Please start and stop the download in your browser and copy the download link"

    read -p "Paste the OneDrive temporary download URL for ckpt_data.zip: " URL
    if [[ -z "$URL" ]]; then
        echo "No URL entered. Exiting."
        exit 1
    fi

    # 2. Download the zip file directly into the destination folder
    ZIP_FILE="$DEST_PATH/ckpt_data.zip"
    echo "Downloading ckpt_data.zip..."
    curl --http1.1 -L -o "$ZIP_FILE" "$URL"

    # 3. Unzip the contents into the same folder
    echo "Unzipping ckpt_data.zip into $DEST_PATH..."
    unzip -o "$ZIP_FILE" -d "$DEST_PATH"

    # 4. Remove the zip file after extraction
    rm "$ZIP_FILE"

    echo "Done! Data saved in: $DEST_PATH"
else
    echo "Hytrel Checkpoints were already downloaded"
    exit 1
fi


popd

# Install additional dependencies for HyTrel
echo "Installing HyTrel dependencies..."

# Core dependencies (some may already be installed)
pip uninstall torch
pip install torch==2.8.0

pip install torch-geometric
pip install transformers
pip install scikit-learn
pip install xgboost

# Additional dependencies required for HyTrel integration
pip install pytorch_lightning  # For checkpoint loading
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html # For PyTorch Geometric operations
pip install torchmetrics       # For evaluation metrics

echo "HyTrel setup automatic part completed!"

