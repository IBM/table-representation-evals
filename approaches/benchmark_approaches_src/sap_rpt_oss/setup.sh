#!/bin/bash
# Setup script for SAP RPT-1-OSS approach

set -e

# Install SAP RPT-1-OSS from GitHub
# Note: The package requires torch>=2.7.0 but PyTorch 2.7.0 doesn't exist yet
# We install with --no-deps and then install compatible versions
echo "Installing SAP RPT-1-OSS (with workaround for PyTorch version requirement)..."
pip install --no-deps git+https://github.com/SAP-samples/sap-rpt-1-oss

# Install compatible dependencies (skipping torch as it's already installed)
echo "Installing compatible dependencies..."
pip install "transformers>=4.52.4" "torcheval>=0.0.7" "scikit-learn>=1.6.1" "pandas>=2.2.3" "pyarrow>=20.0.0" "huggingface_hub"

echo "SAP RPT-1-OSS installed successfully!"
echo "Note: You may need to log in to Hugging Face to download model checkpoints:"
echo "  huggingface-cli login"
echo ""
echo "Note: The package requires torch>=2.7.0 but we're using torch 2.2.2."
echo "If you encounter issues, you may need to upgrade PyTorch when 2.7.0 becomes available."

