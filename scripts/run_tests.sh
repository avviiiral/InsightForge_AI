#!/usr/bin/env bash
# Runs the full test suite with coverage.
# Usage: ./scripts/run_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "Running InsightForge-AI test suite..."
pytest tests/ -v --cov=app --cov=backend --cov-report=term-missing
