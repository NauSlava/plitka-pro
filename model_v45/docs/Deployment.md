# Инструкции по Развертыванию - Модель "nauslava/rubber-tile-lora-v45"

## Обзор Развертывания

Модель v45 развертывается с использованием Cog Framework на платформе Replicate с полной поддержкой НАШЕЙ обученной модели.

## Предварительные Требования

### Системные Требования
- **OS:** Linux (Ubuntu 20.04+)
- **CUDA:** 12.4
- **Python:** 3.11
- **GPU:** NVIDIA с поддержкой CUDA
- **Memory:** Минимум 16 GB RAM, 12 GB VRAM

### Установленные Компоненты
- **Docker:** 20.10+
- **Cog:** 0.16.2+
- **Git:** Последняя версия

## Структура Проекта

```
model_v45/
├── cog.yaml                    # Конфигурация Cog
├── predict.py                  # Основной код
├── requirements.txt            # Python зависимости
├── model_files/               # Файлы обученной модели
│   ├── rubber-tile-lora-v4_sdxl_lora.safetensors
│   └── rubber-tile-lora-v4_sdxl_embeddings.safetensors
└── docs/                      # Документация
```

## Этапы Развертывания

### 1. Подготовка Окружения

#### Установка Cog
```bash
# Установка Cog
curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
chmod +x /usr/local/bin/cog

# Проверка установки
cog --version
```

#### Подготовка Python
```bash
# Создание виртуального окружения
python3.11 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Подготовка Файлов Модели

#### Проверка Файлов
```bash
# Проверка наличия файлов
ls -la model_files/
du -h model_files/*

# Проверка целостности
file model_files/*.safetensors
```

#### Структура Файлов
```
model_files/
├── rubber-tile-lora-v4_sdxl_lora.safetensors      # 97.3 MB
└── rubber-tile-lora-v4_sdxl_embeddings.safetensors # 8.1 KB
```

### 3. Локальное Тестирование

#### Сборка Образа
```bash
# Сборка Docker образа
cog build

# Проверка сборки
docker images | grep rubber-tile-lora-v45
```

#### Тестирование Предсказания
```bash
# Локальный тест
cog predict -i prompt="100% white"

# Тест с параметрами
cog predict -i prompt="50% red, 50% black" -i seed=12345
```

### 4. Публикация на Replicate

#### Авторизация
```bash
# Вход в Replicate
cog login

# Проверка авторизации
cog whoami
```

#### Публикация
```bash
# Push на Replicate
cog push r8.im/nauslava/rubber-tile-lora-v45

# Проверка статуса
cog status
```

## Конфигурация Cog

### cog.yaml
```yaml
build:
  gpu: true
  cuda: "12.4"
  python_version: "3.11"
  python_requirements: "requirements.txt"
  system_packages:
    - "libgl1-mesa-glx"
    - "libglib2.0-0"
  cog_runtime: true

predict: "predict.py:Predictor"

image: "r8.im/nauslava/rubber-tile-lora-v45:latest"
```

### Переменные Окружения
```yaml
env:
  HF_HOME: "/tmp/hf_home"
  HF_DATASETS_CACHE: "/tmp/hf_datasets_cache"
  HF_MODELS_CACHE: "/tmp/hf_models_cache"
  TRANSFORMERS_CACHE_MIGRATION_DISABLE: "1"
  HF_HUB_CACHE_MIGRATION_DISABLE: "1"
  HF_HUB_DISABLE_TELEMETRY: "1"
  HF_HUB_DISABLE_PROGRESS_BARS: "1"
  TRANSFORMERS_VERBOSITY: "error"
  TOKENIZERS_PARALLELISM: "false"
```

## Команды Развертывания

### Полный Цикл
```bash
# 1. Подготовка
cog build

# 2. Локальное тестирование
cog predict -i prompt="100% white"

# 3. Публикация
cog push r8.im/nauslava/rubber-tile-lora-v45

# 4. Проверка статуса
cog status
```

### Отладка
```bash
# Детальная сборка
cog build --debug

# Логи сборки
cog build --verbose

# Проверка зависимостей
cog build --check
```

## Мониторинг Развертывания

### Логи Сборки
```bash
# Просмотр логов
cog logs

# Реальные логи
docker logs $(docker ps -q --filter ancestor=rubber-tile-lora-v45)
```

### Статус Модели
```bash
# Проверка статуса
cog status

# Информация о модели
cog info
```

## Устранение Проблем

### Частые Ошибки

#### Ошибка CUDA
```bash
# Проверка CUDA
nvidia-smi
nvcc --version

# Переустановка CUDA
sudo apt-get install cuda-12-4
```

#### Ошибка Памяти
```bash
# Проверка памяти
free -h
nvidia-smi

# Очистка Docker
docker system prune -a
```

#### Ошибка Зависимостей
```bash
# Обновление pip
pip install --upgrade pip

# Переустановка зависимостей
pip install -r requirements.txt --force-reinstall
```

### Диагностика
```bash
# Проверка системы
cog doctor

# Проверка Docker
docker info

# Проверка GPU
cog run nvidia-smi
```

## Безопасность

### Проверка Файлов
```bash
# Проверка безопасности
cog security-scan

# Проверка зависимостей
cog dependency-check
```

### Переменные Окружения
```bash
# Проверка переменных
env | grep -E "(HF_|TRANSFORMERS_)"

# Установка переменных
export HF_HOME="/tmp/hf_home"
export HF_HUB_DISABLE_TELEMETRY="1"
```

## Производительность

### Оптимизации
- **VAE Slicing:** Включено
- **VAE Tiling:** Включено
- **Memory Management:** Автоматическое
- **FP16:** Половинная точность

### Метрики
- **Время инициализации:** ~30-60 секунд
- **Время генерации:** ~20-40 секунд
- **Использование памяти:** ~8-12 GB VRAM

## Обновление Модели

### Процесс Обновления
```bash
# 1. Изменение кода
# 2. Обновление версии в cog.yaml
# 3. Сборка нового образа
cog build

# 4. Тестирование
cog predict -i prompt="100% white"

# 5. Публикация
cog push r8.im/nauslava/rubber-tile-lora-v45
```

### Версионирование
```yaml
# В cog.yaml
image: "r8.im/nauslava/rubber-tile-lora-v45:v4.5.1"
```

## Заключение

Развертывание модели v45 требует **внимания к деталям** и **правильной последовательности действий**. Ключевым фактором успеха является **использование НАШИХ обученных файлов** и **правильная конфигурация Cog**.

### Критические Факторы
1. **Правильные файлы модели:** LoRA и Textual Inversion
2. **Корректная конфигурация:** CUDA, Python, зависимости
3. **Локальное тестирование:** Перед публикацией
4. **Мониторинг:** Логи и статус

### Рекомендации
- Всегда тестируйте локально перед публикацией
- Используйте правильные версии зависимостей
- Мониторьте использование ресурсов
- Ведите журнал изменений
