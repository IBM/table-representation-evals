#!/bin/bash
set -e 

# TabuLA-8B Setup Script
# This script installs all required dependencies for the TabuLA-8B approach

pushd .

cd approaches/benchmark_approaches_src/tabula_8b

echo "Installing TabuLA-8B dependencies..."

# Install requirements from requirements.txt
pip install -r requirements.txt

# Install tableshift from git repository (no dependencies)
echo "Installing tableshift from git repository..."
pip install --no-deps git+https://github.com/mlfoundations/tableshift.git

echo "TabuLA-8B setup complete!"

popd