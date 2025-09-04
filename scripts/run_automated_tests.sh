#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

echo "🚀 Запуск автоматического тестирования v4.5.01 - Critical Architecture Fixes"
echo "=" * 80

# Проверяем Python
if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ python3 не найден. Установите python3." >&2
  exit 1
fi

# Создаем виртуальное окружение если нужно
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "📦 Создание виртуального окружения в $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

# Обновляем pip
"$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null

# Устанавливаем зависимости
echo "📦 Установка зависимостей..."
"$VENV_DIR/bin/python" -m pip install --upgrade replicate requests >/dev/null

# Проверяем API токен
if [ -z "${REPLICATE_API_TOKEN:-}" ] && [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PROJECT_ROOT/.env"
  set +a
fi

if [ -z "${REPLICATE_API_TOKEN:-}" ]; then
  echo "❌ REPLICATE_API_TOKEN не установлен!" >&2
  echo "💡 Установите токен в переменной окружения или в файле .env" >&2
  exit 1
fi

echo "✅ API токен найден"
echo "🎯 Запуск автоматического тестирования..."

# Запускаем тестирование
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/test_v4.5.01_automated.py"
