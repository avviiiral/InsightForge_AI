#!/usr/bin/env bash
# Starts the InsightForge-AI Streamlit frontend.
# Usage: ./scripts/run_frontend.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

echo "Starting InsightForge-AI Streamlit dashboard on http://localhost:8501"
streamlit run frontend/streamlit_app.py
