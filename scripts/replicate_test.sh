#!/usr/bin/env bash
set -euo pipefail

# По умолчанию можно переопределить через переменную окружения MODEL_REF
# Обновлено под актуальную версию v4.5.10 (LoRA Loading Fix)
MODEL_REF="${MODEL_REF:-nauslava/plitka-pro-project:v4.5.10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install python3." >&2
  exit 1
fi

# Create venv if missing
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

# Ensure pip exists and up to date
"$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null

# Ensure replicate is installed in venv
if ! "$VENV_DIR/bin/python" -c "import replicate" >/dev/null 2>&1; then
  echo "Installing 'replicate' into $VENV_DIR ..."
  "$VENV_DIR/bin/python" -m pip install --upgrade replicate >/dev/null
fi

if [ -z "${REPLICATE_API_TOKEN:-}" ]; then
  echo "WARNING: REPLICATE_API_TOKEN is not set. Export it before running." >&2
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/cli/replicate_test.py" --model "$MODEL_REF" "$@"


