#!/bin/bash
# Setup script for ConTextTab approach

set -e

# Initialize submodule if needed
if [ ! -d "contexttab_src" ]; then
    git submodule update --init --recursive
fi

# Install ConTextTab
pip install -e contexttab_src/
