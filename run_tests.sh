#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
RESULTS_DIR="$SCRIPT_DIR/test-results"

# Activate virtualenv
source "$VENV/bin/activate"

# Ensure pytest and pytest-junit reporter are available
pip install --quiet pytest

# Create output directory
mkdir -p "$RESULTS_DIR"

echo "Running tests..."
python -m pytest tests/ \
    --junit-xml="$RESULTS_DIR/results.xml" \
    -v \
    "$@"

echo ""
echo "XML report written to: $RESULTS_DIR/results.xml"