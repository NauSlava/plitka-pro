#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install python3." >&2
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null

# Ensure dependencies
"$VENV_DIR/bin/python" -m pip install --upgrade replicate requests pillow >/dev/null

# Try to export token from .env if not set (safe sourcing)
if [ -z "${REPLICATE_API_TOKEN:-}" ] && [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_ROOT/.env"
  set +a
fi

if [ -z "${REPLICATE_API_TOKEN:-}" ]; then
  echo "WARNING: REPLICATE_API_TOKEN is not set. Export it before running." >&2
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/replicate_gui.py"


