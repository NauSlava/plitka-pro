#!/bin/bash
# Простой скрипт для запуска GUI тестера

echo "🚀 Запуск GUI тестера Plitka Pro v4.5.09..."

# Проверяем наличие виртуального окружения
if [ -f ".venv/bin/python" ]; then
  echo "📱 Запуск GUI через виртуальное окружение..."
  .venv/bin/python scripts/gui/replicate_gui.py
elif command -v python3 >/dev/null 2>&1; then
  echo "⚠️ Виртуальное окружение не найдено, используем системный python3"
  echo "📱 Запуск GUI..."
  python3 scripts/gui/replicate_gui.py
else
  echo "❌ python3 не найден. Установите python3." >&2
  exit 1
fi
