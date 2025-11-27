#!/bin/bash
set -e

# TabICL Setup Script
# This script installs the local tabicl package for the TabICL approach

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


TABICL_LOCAL_PATH="$SCRIPT_DIR/tabicl"

if [ ! -d "$TABICL_LOCAL_PATH" ]; then
  echo "Error: Local tabicl folder not found at $TABICL_LOCAL_PATH"
  exit 1
fi

echo "Installing local tabicl package from $TABICL_LOCAL_PATH..."
pip install -e "$TABICL_LOCAL_PATH"

echo "TabICL local package installation complete!" 