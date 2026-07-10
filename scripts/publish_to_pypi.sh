#!/usr/bin/env bash

set -e

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"


rm -rf build/ dist/ *.egg-info/ llm_text_detectors.egg-info/

pip install --upgrade build twine

python -m build
python -m twine check dist/*

REPOSITORY=""
if [[ "$1" == "--test" ]]; then
    REPOSITORY="--repository testpypi"
    echo "Publishing to TestPyPI..."
else
    echo "Publishing to official PyPI..."
fi

python -m twine upload --verbose $REPOSITORY dist/*

echo "Package successfully built and uploaded!"
