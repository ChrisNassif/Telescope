#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the repository root
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

echo "=== Packaging and Publishing telescope_llm_text_detection ==="

# Clean up previous build files
echo "Cleaning up previous build files..."
rm -rf build/ dist/ *.egg-info/ llm_text_detectors.egg-info/

# Ensure build dependencies are installed
echo "Upgrading build tools (build, twine)..."
pip install --upgrade build twine

# Build the package
echo "Building package distribution..."
python -m build

# Check built archives with twine
echo "Running twine package validation checks..."
python -m twine check dist/*

# Determine repository
REPOSITORY=""
if [[ "$1" == "--test" ]]; then
    REPOSITORY="--repository testpypi"
    echo "Publishing to TestPyPI..."
else
    echo "Publishing to official PyPI..."
fi

# Upload package
python -m twine upload $REPOSITORY dist/*

echo "Package successfully built and uploaded!"
