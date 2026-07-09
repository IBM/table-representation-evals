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

# vanilla_table_bert.py and vertical/vertical_attention_table_bert.py both unconditionally
# import `fairseq.distributed_utils` at module load time, but only call it inside an
# `if args.multi_gpu:` branch of their own training code, which our single-GPU
# embedding-only usage never reaches. fairseq==0.8.0 (the version TaBERT was built
# against) can't be installed on a modern stack - importing it pulls in its entire
# package tree, which uses long-removed numpy APIs (e.g. `np.float`) - so make the
# import optional in both files instead of installing fairseq at all.
python3 - <<EOF
files = [
    "$TABERT_DIR/table_bert/vanilla_table_bert.py",
    "$TABERT_DIR/table_bert/vertical/vertical_attention_table_bert.py",
]
old = "from fairseq import distributed_utils"
# Broad except: an installed-but-unusable fairseq (e.g. built against numpy APIs long
# since removed, like np.float) fails with AttributeError deep in its own import chain,
# not ImportError, when fairseq itself is present but broken.
new = "try:\n    from fairseq import distributed_utils\nexcept Exception:\n    distributed_utils = None"
for p in files:
    with open(p) as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new, 1)
        with open(p, "w") as f:
            f.write(content)
        print(f"Patched {p} to make the fairseq import optional.")
    else:
        print(f"{p} already patched (or import line not found) - skipping.")
EOF

# table_bert/dataset.py, table_bert/vanilla_table_bert.py, and table_bert/vertical/dataset.py
# all use the bare `np.int`/`np.bool` aliases as dtype= arguments, which numpy removed
# entirely (they're not just deprecated) in modern releases - replace with the equivalent
# builtin types, matching numpy's own suggested fix.
python3 - <<EOF
import re

files = [
    "$TABERT_DIR/table_bert/dataset.py",
    "$TABERT_DIR/table_bert/vanilla_table_bert.py",
    "$TABERT_DIR/table_bert/vertical/dataset.py",
]
for p in files:
    with open(p) as f:
        content = f.read()
    new_content = re.sub(r"\bnp\.int\b", "int", content)
    new_content = re.sub(r"\bnp\.bool\b", "bool", new_content)
    if new_content != content:
        with open(p, "w") as f:
            f.write(new_content)
        print(f"Patched {p}: replaced removed np.int/np.bool aliases.")
    else:
        print(f"{p}: no removed numpy aliases found - skipping.")
EOF

# ------------------------------------------------------------------
# 2. Install TaBERT and its dependencies
# ------------------------------------------------------------------
echo "Installing TaBERT package..."
pip install --editable "$TABERT_DIR"

# Remaining plain pip dependencies from tabert_src/scripts/env.yml (torch-scatter,
# fairseq, and pytorch_pretrained_bert are handled separately above/below since they
# each need special treatment on a modern stack).
pip install cython ujson msgpack h5py pyzmq redis spacy

# table_bert imports BERT classes from `pytorch_pretrained_bert` by default (its own
# README: "TaBERT will use pytorch-pretrained-bert by default"), falling back to modern
# transformers' long-removed flat module layout (transformers.tokenization_bert etc.)
# only if that import fails - so without it, table_bert fails to import entirely.
# The frozen PyPI release (`pytorch-pretrained-bert`, e.g. 0.6.2) predates fields like
# `layer_norm_eps` that table_bert/config.py hardcodes into its BertConfig(...) call, so
# it must be this exact pre-rename transformers commit instead (per tabert_src/scripts/env.yml).
pip install "git+https://github.com/huggingface/transformers.git@372a5c1ceec49b52c503707e9657bfaae7c236a0#egg=pytorch_pretrained_bert"

# torch-scatter is required for column-aggregation in TaBERT.
# fairseq is NOT installed - its only import is patched to be optional above.
# Pin torch and install torch-scatter from the matching prebuilt-wheel index (mirrors
# hytrel/setup.sh) - a plain `pip install torch-scatter` tries to build from source, and
# pip's isolated build environment can't see torch, so the build fails on `import torch`.
pip uninstall -y torch
pip install torch==2.8.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

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
# Path relative to the repo root (what cfg.project_root resolves against)
popd  # back to repo root

REPO_ROOT="$(pwd)"
YAML_PATH="configs/approaches/tabert.yaml"
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
