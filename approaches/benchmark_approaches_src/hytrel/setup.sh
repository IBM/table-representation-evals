#!/bin/bash

# Setup script for HyTrel approach
echo "Setting up HyTrel approach..."

# Clone the HyTrel repository if it doesn't exist
HYTREL_DIR="hytrel_src"
if [ ! -d "$HYTREL_DIR" ]; then
    echo "Cloning HyTrel repository..."
    git clone https://github.com/awslabs/hypergraph-tabular-lm.git $HYTREL_DIR
else
    echo "HyTrel repository already exists, updating..."
    cd $HYTREL_DIR
    git pull
    cd ..
fi

# Install additional dependencies for HyTrel
echo "Installing HyTrel dependencies..."
# Core dependencies (some may already be installed)
pip install torch-geometric
pip install transformers
pip install scikit-learn
pip install xgboost

# Additional dependencies required for HyTrel integration
pip install pytorch_lightning  # For checkpoint loading
pip install torch_scatter      # For PyTorch Geometric operations
pip install torchmetrics       # For evaluation metrics

echo "HyTrel setup completed!"

