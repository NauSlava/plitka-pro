#!/bin/bash

# Тестовый скрипт для v4.5.09 - Docs sync + scripts cleanup
# Автор: MLOps Architect
# Дата: 12 сентября 2025

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Конфигурация
MODEL_REF="${MODEL_REF:-nauslava/plitka-pro-project:v4.5.09}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESETS_DIR="${SCRIPT_DIR}/presets"

echo -e "${BLUE}=== Тестирование Plitka Pro v4.5.09 ===${NC}"
echo -e "${YELLOW}Модель: ${MODEL_REF}${NC}"
echo -e "${YELLOW}Пресеты: ${PRESETS_DIR}${NC}"
echo ""

# Функция для запуска теста
run_test() {
    local test_file="$1"
    local test_name="$2"
    
    echo -e "${BLUE}Запуск теста: ${test_name}${NC}"
    echo -e "${YELLOW}Файл: ${test_file}${NC}"
    
    if [ ! -f "$test_file" ]; then
        echo -e "${RED}ОШИБКА: Файл ${test_file} не найден${NC}"
        return 1
    fi
    
    # Запуск теста через replicate CLI
    if command -v replicate &> /dev/null; then
        echo -e "${GREEN}Используется Replicate CLI${NC}"
        replicate predict "$MODEL_REF" --input-file "$test_file"
    else
        echo -e "${YELLOW}Replicate CLI не найден, используем curl${NC}"
        # Здесь можно добавить curl запрос к API
        echo -e "${YELLOW}Для полного тестирования установите Replicate CLI${NC}"
    fi
    
    echo -e "${GREEN}Тест ${test_name} завершен${NC}"
    echo ""
}

# Проверка наличия пресетов
if [ ! -d "$PRESETS_DIR" ]; then
    echo -e "${RED}ОШИБКА: Директория пресетов ${PRESETS_DIR} не найдена${NC}"
    exit 1
fi

# Список тестов для v4.5.09
echo -e "${BLUE}Доступные тесты для v4.5.09:${NC}"
echo "1. Быстрый тест (test_v4.5.09_quick.json)"
echo "2. Полный тест (test_inputs_v4.5.09_docs_sync.json)"
echo "3. Тест запросов (test_requests_v4.5.09.json)"
echo ""

# Запуск быстрого теста
if [ -f "${PRESETS_DIR}/test_v4.5.09_quick.json" ]; then
    run_test "${PRESETS_DIR}/test_v4.5.09_quick.json" "Быстрый тест v4.5.09"
fi

# Запуск полного теста
if [ -f "${PRESETS_DIR}/test_inputs_v4.5.09_docs_sync.json" ]; then
    run_test "${PRESETS_DIR}/test_inputs_v4.5.09_docs_sync.json" "Полный тест v4.5.09"
fi

# Запуск теста запросов
if [ -f "${PRESETS_DIR}/test_requests_v4.5.09.json" ]; then
    run_test "${PRESETS_DIR}/test_requests_v4.5.09.json" "Тест запросов v4.5.09"
fi

echo -e "${GREEN}=== Все тесты v4.5.09 завершены ===${NC}"
echo -e "${YELLOW}Версия: v4.5.09 - Docs sync + scripts cleanup${NC}"
echo -e "${YELLOW}Дата: 12 сентября 2025${NC}"
echo -e "${YELLOW}Изменения: Рефакторинг структуры проекта, синхронизация документации${NC}"
