# 🧪 Документация Тестового Скрипта - Plitka Pro Project

## 📋 Общая Информация

**Файл**: `scripts/replicate_test.py`  
**Версия**: v1.2.0  
**Дата создания**: 2 сентября 2025  
**Совместимость**: Plitka Pro v4.4.56+  
**Автор**: Plitka Pro Development Team  

---

## 🎯 Назначение

Тестовый скрипт `replicate_test.py` - это комплексная система тестирования для модели Plitka Pro, специально адаптированная для:

- **Color Grid Adapter**: Тестирование всех паттернов и размеров гранул
- **ControlNet Integration**: Автоматическое и ручное включение
- **Адаптивные параметры**: Автоматическая настройка steps/guidance по количеству цветов
- **Мониторинг**: Отслеживание логов и статистики в реальном времени

---

## 🏗️ Архитектура

### **Основные Компоненты**

1. **`_parse_args()`**: Парсинг аргументов командной строки
2. **`_get_color_grid_test_presets()`**: Специальные тесты для Color Grid Adapter
3. **`_run_color_grid_comprehensive_test()`**: Комплексное тестирование всех возможностей
4. **`_run_single()`**: Запуск одиночного теста с мониторингом
5. **`_resolve_model_and_version()`**: Разрешение версии модели

### **Версии по Умолчанию**

- **Модель**: `nauslava/plitka-pro-project:v4.4.56`
- **Файл пресетов**: `test_inputs_v4.4.56.json`
- **Polling интервал**: 6 секунд
- **Startup timeout**: 7 минут
- **Total timeout**: 25 минут

---

## 🚀 Использование

### **1. Комплексное Тестирование Color Grid Adapter**

```bash
python3 scripts/replicate_test.py --test-color-grid
```

**Что тестируется:**
- Все паттерны (random, granular_medium, granular_small)
- Все размеры гранул (small, medium, large)
- Автоматическое включение ControlNet
- Статистика успешности тестов

### **2. Тест Конкретного Пресета**

```bash
python3 scripts/replicate_test.py --preset color_grid_mix3
```

**Доступные пресеты:**
- `color_grid_basic`: 1 цвет (random паттерн)
- `color_grid_mix2`: 2 цвета (granular + medium гранулы)
- `color_grid_mix3`: 3 цвета (granular + medium гранулы)
- `color_grid_mix4`: 4 цвета (granular + small гранулы)
- `color_grid_granular_test`: Специальный тест granular паттерна
- `color_grid_controlnet_manual`: Ручное включение ControlNet
- `color_grid_edge_cases`: Экстремальный тест 10 цветов
- `color_grid_quality_comparison`: Сравнение с базовыми параметрами
- `legacy_compatibility`: Совместимость со старыми параметрами

### **3. Тест с Ручными Параметрами**

```bash
python3 scripts/replicate_test.py \
  --prompt "50% red, 30% black, 20% white" \
  --steps 30 \
  --guidance 8.0 \
  --lora_scale 0.7
```

### **4. Batch Тестирование**

```bash
python3 scripts/replicate_test.py --batch test_inputs_v4.4.56.json
```

---

## 🎨 Color Grid Adapter Тесты

### **Автоматический Выбор Паттернов**

| Количество цветов | Паттерн | Размер гранул | Steps | Guidance |
|------------------|---------|---------------|-------|----------|
| **1 цвет** | `random` | `large` | 20 | 7.0 |
| **2 цвета** | `granular` | `medium` | 25 | 7.5 |
| **3 цвета** | `granular` | `medium` | 30 | 8.0 |
| **4+ цвета** | `granular` | `small` | 35+ | 8.5+ |

### **Автоматическое Включение ControlNet**

- **1 цвет**: ControlNet отключен
- **2+ цвета**: ControlNet включается автоматически
- **Ручное управление**: `--use_controlnet` для принудительного включения

---

## 📊 Мониторинг и Логирование

### **Реальное Время**

- **Startup мониторинг**: Отслеживание загрузки модели
- **Color Grid Adapter логи**: Специальные логи с эмодзи 🎨
- **ControlNet логи**: Мониторинг активации с эмодзи 🔗
- **Потоковое логирование**: Дозаполнение logs.txt во время выполнения

### **Статистика**

- **Процент успеха**: Автоматический расчет по результатам тестов
- **Детальные результаты**: Статус каждого теста с описанием
- **Время выполнения**: Мониторинг производительности

---

## 🔧 Технические Детали

### **Зависимости**

```python
import argparse      # Парсинг аргументов
import json         # Работа с JSON пресетами
import os           # Файловые операции
import sys          # Системные функции
import time         # Таймеры и задержки
import replicate    # Replicate API клиент
```

### **Структура Файлов**

```
scripts/
├── replicate_test.py              # Основной тестовый скрипт (v1.2.0)
├── test_inputs_v4.4.56.json      # Пресеты для v4.4.56
└── test_inputs_v4.4.39.json      # Пресеты для v4.4.39 (legacy)
```

---

## 📈 Примеры Вывода

### **Запуск Комплексного Теста**

```bash
🎨 === COMPREHENSIVE COLOR GRID ADAPTER TEST ===

📊 Тестируем паттерны: ['random', 'granular_medium', 'granular_small']

🧪 Тестируем паттерн: random
📝 Описание: Тест random паттерна (1 цвет)
🎨 Color Grid Adapter анализ:
   - Промпт: 100% red
   - Количество цветов: 1
   - Ожидаемый паттерн: random
   - Ожидаемый размер гранул: large
   - ControlNet: ручной

✅ Тест паттерна random пройден
```

### **Статистика Результатов**

```bash
📊 === ИТОГОВАЯ СТАТИСТИКА COLOR GRID ADAPTER ===
Всего тестов: 6
Успешных: 6
Провальных: 0
Процент успеха: 100.0%

📋 Детальные результаты:
  ✅ random: Тест random паттерна (1 цвет)
  ✅ granular_medium: Тест granular паттерна с medium гранулами (2 цвета)
  ✅ granular_small: Тест granular паттерна с small гранулами (4 цвета)
  ✅ small_granules: Тест small гранул (3 цвета)
  ✅ medium_granules: Тест medium гранул (2 цвета)
  ✅ large_granules: Тест large гранул (1 цвет)
```

---

## 🚨 Устранение Неполадок

### **Частые Проблемы**

1. **"Preset file not found"**
   - Проверьте наличие файла `test_inputs_v4.4.56.json`
   - Убедитесь, что скрипт запускается из корневой директории проекта

2. **"REPLICATE_API_TOKEN is not set"**
   - Создайте файл `.env` с `REPLICATE_API_TOKEN=your_token`
   - Или установите переменную окружения

3. **"Startup timeout exceeded"**
   - Увеличьте `--startup-timeout` для медленных серверов
   - Проверьте стабильность интернет-соединения

### **Отладка**

```bash
# Подробное логирование
python3 scripts/replicate_test.py --test-color-grid --poll-seconds 3

# Тест конкретной модели
python3 scripts/replicate_test.py --model nauslava/plitka-pro-project:v4.4.56 --preset color_grid_basic
```

---

## 📚 Связанная Документация

- **[Project.md](Project.md)**: Архитектура проекта
- **[Changelog.md](Changelog.md)**: История изменений
- **[QuickStart.md](QuickStart.md)**: Быстрый старт
- **[TESTING_INSTRUCTIONS_v4.4.39.md](TESTING_INSTRUCTIONS_v4.4.39.md)**: Инструкции по тестированию

---

## 🎉 Заключение

Тестовый скрипт `replicate_test.py` v1.2.0 предоставляет комплексную систему тестирования для Plitka Pro v4.4.56+ с полной поддержкой Color Grid Adapter и автоматического ControlNet.

**Ключевые преимущества:**
- ✅ Автоматическое тестирование всех паттернов и размеров гранул
- ✅ Интеллектуальный выбор параметров по сложности промпта
- ✅ Мониторинг в реальном времени с детальными логами
- ✅ Статистика успешности и производительности
- ✅ Совместимость с legacy версиями

---

*Документ обновлен: 2 сентября 2025*  
*Версия скрипта: v1.2.0*  
*Совместимость: Plitka Pro v4.4.56+*
