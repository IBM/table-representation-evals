#!/bin/bash
# TabDPT Setup Script
# https://github.com/layer6ai-labs/TabDPT-inference

set -e

echo "Installing TabDPT..."
pip install git+https://github.com/layer6ai-labs/TabDPT-inference.git

echo "TabDPT installation complete."
echo "Model weights are downloaded automatically on first use from HuggingFace."
