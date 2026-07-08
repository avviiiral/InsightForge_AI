#!/usr/bin/env bash
# Starts the InsightForge-AI FastAPI backend.
# Usage: ./scripts/run_backend.sh
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

HOST="${BACKEND_HOST:-0.0.0.0}"
PORT="${BACKEND_PORT:-8000}"

echo "Starting InsightForge-AI backend on http://${HOST}:${PORT}"
uvicorn backend.main:app --host "$HOST" --port "$PORT" --reload
