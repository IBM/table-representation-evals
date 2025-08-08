#!/bin/bash

# TabPFN Setup Script
# This script installs TabPFN and its dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing TabPFN..."
pip install tabpfn

echo "TabPFN installation complete!"
echo "TabPFN will automatically download model weights on first use."
echo "Alternatively, run python scripts/download_all_models.py" 