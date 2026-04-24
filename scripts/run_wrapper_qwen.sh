#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.qwen}"
WRAPPER_PORT="${WRAPPER_PORT:-8101}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE"
  echo "Create it first, for example:"
  echo "  cp $ROOT_DIR/.env.qwen.example $ROOT_DIR/.env.qwen"
  exit 1
fi

if [ ! -d "$ROOT_DIR/.venv" ]; then
  echo "Missing virtualenv: $ROOT_DIR/.venv"
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"
set -a
source "$ENV_FILE"
set +a

export VLM_WRAPPER_PORT="$WRAPPER_PORT"

echo "Starting Qwen wrapper on http://localhost:${VLM_WRAPPER_PORT}"
echo "Using env: $ENV_FILE"

exec python "$ROOT_DIR/vlm_wrapper.py"
