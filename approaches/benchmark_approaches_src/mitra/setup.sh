#!/bin/bash
set -e

# MITRA Setup Script
# This script installs the required dependencies for the MITRA approach
# Based on instructions from: https://huggingface.co/autogluon/mitra-classifier

echo "Installing MITRA dependencies..."

# Install uv package manager
pip install uv

# Install AutoGluon with MITRA support
uv pip install autogluon.tabular[mitra]

echo "MITRA dependencies installation complete!"
echo "MITRA model will be automatically downloaded from Hugging Face Hub on first use."

# Made with Bob
