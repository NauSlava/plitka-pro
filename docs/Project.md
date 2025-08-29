# 🏗️ Plitka Pro - Системная Архитектура

## 📋 Обзор Проекта

**Plitka Pro** - это высокопроизводительная MLOps-система для генерации фотореалистичных изображений резиновой плитки с математически точным контролем цветовых пропорций. Система развернута на платформе Replicate.com и использует гибридную архитектуру на базе Stable Diffusion XL (SDXL) с локальными LoRA и Textual Inversion адаптерами.

### 🎯 Ключевые Цели
- **Основная**: Генерация изображений резиновых плиток с точным контролем цветовых пропорций через API
- **Техническая**: Создание отказоустойчивой и высокопроизводительной системы на Replicate.com
- **Бизнесовая**: Обеспечение стабильного API для интеграции с внешними системами

## 🏛️ Архитектурная Диаграмма

```mermaid
graph TB
    A[Client Request] --> B[Replicate API Gateway]
    B --> C[Plitka Pro v4.3.50]
    
    C --> D[SDXL Base Model]
    C --> E[Local LoRA Adapter]
    C --> F[Local Textual Inversion]
    C --> G[ControlNet Models]
    
    E --> H[model_files/rubber-tile-lora-v4_sdxl_lora.safetensors]
    F --> I[model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors]
    
    G --> J[Canny ControlNet]
    G --> K[SoftEdge ControlNet]
    
    C --> M[Image Generation Pipeline]
    M --> N[Preview Image 512x512]
    M --> O[Final Image 1024x1024]
    M --> P[Color Map Debug]
    
    subgraph "ControlNet Selection Logic"
        Q[Angle Input] --> R{Angle Check}
        R -->|0° or 90°| S[Canny ControlNet]
        R -->|45° or 135°| T[SoftEdge ControlNet]
    end
    
    subgraph "Local Model Loading"
        W[Local LoRA File] --> X{File Exists?}
        X -->|Yes| Y[Load Local LoRA (strength 0.7)]
        X -->|No| Z[Use Base SDXL]
        
        AA[Local TI File] --> BB{File Exists?}
        BB -->|Yes| CC[Load Local TI (dual-encoder format)]
        BB -->|No| DD[Skip TI Loading]
    end
```

## 🔧 Технологический Стек

### Основные Компоненты
- **Платформа**: Replicate.com
- **Инструмент развертывания**: Cog
- **Среда выполнения**: Docker (CUDA 12.4, Python 3.11, PyTorch 2.4.0)
- **Основная библиотека**: `diffusers==0.27.2`

### Модели и Адаптеры
- **Базовая модель**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Локальные файлы**:
  - LoRA: `model_files/rubber-tile-lora-v4_sdxl_lora.safetensors` (97MB)
  - Textual Inversion: `model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors` (8.1KB)
- **ControlNet модели**:
  - Canny: `diffusers/controlnet-canny-sdxl-1.0`
  - SoftEdge: `lllyasviel/control_v11p_sd15_softedge`
- **Scheduler**: `DPMSolverMultistepScheduler`

### Спецификация Кастомной Модели
- **Формат обучения**: Eden LoRA Trainer
- **Триггеры активации**: `ohwx_rubber_tile <s0><s1>` (обязательны в промпте)
- **Сила LoRA**: 0.7 для модели (как в обучении)
- **Ограничения**: Модель обучена только на ракурсе "вид сверху" (0°)
- **Структура данных**: 79 изображений (29 монохромных + 50 полихромных)
- **Аннотации**: Стандартизированный формат `ohwx_rubber_tile <s0><s1>, [PERCENT]% [COLOR] rubber tile`

## 📡 API Контракт

### Входные Параметры
```json
{
  "params_json": "string" // JSON с бизнес-параметрами
}
```

### Структура params_json
```json
{
  "colors": [
    {"name": "red", "proportion": 0.7},
    {"name": "blue", "proportion": 0.3}
  ],
  "angle": 0,
  "quality": "preview|standard|high",
  "seed": 42,
  "overrides": {
    "num_inference_steps_preview": 20,
    "num_inference_steps_final": 40,
    "guidance_scale": 7.0,
    "negative_prompt": "blurry, low quality"
  }
}
```

### Выходные Файлы
- `preview.png` - Быстрая превью версия (512x512)
- `final.png` - Финальное качественное изображение (1024x1024)
- `colormap.png` - Карта цветов для отладки

### Профили Качества
| Профиль | Preview Steps | Final Steps | Preview Size | Final Size | Guidance Scale |
|---------|---------------|-------------|--------------|------------|----------------|
| preview | 20 | 40 | 512x512 | 1024x1024 | 7.0 |
| standard | 20 | 40 | 512x512 | 1024x1024 | 7.0 |
| high | 30 | 60 | 512x512 | 1024x1024 | 7.0 |

## 🎛️ Матрица Угол → ControlNet

| Угол | ControlNet | Приоритет | Применение |
|------|------------|-----------|------------|
| 0° | Без ControlNet | Основной | Вид сверху |
| 90° | Canny | Основной | Вид сверху (повернутый) |
| 45° | SoftEdge | Основной | Диагональный вид |
| 135° | SoftEdge | Основной | Диагональный вид |

## 🚀 Стратегия Развертывания

### "Local Files First" Архитектура
- **Принцип**: Приоритет локальных файлов модели над внешними источниками
- **Преимущества**: 
  - Стабильность и независимость от внешних сервисов
  - Контроль версий и совместимости
  - Быстрая загрузка без сетевых задержек
- **Fallback**: Автоматический переход на базовый SDXL при отсутствии локальных файлов

### Runtime Model Loading
```python
# Автоматический fallback: локальный файл → базовый SDXL
if os.path.exists(lora_path):
    # Загрузка LoRA с силой 0.7 (как в обучении Eden)
    pipe.set_adapters(["rubber-tile-lora-v4_sdxl_lora"], adapter_weights=[0.7])
    pipe.fuse_lora()
    logger.info("✅ Local LoRA loaded with strength 0.7")
else:
    logger.warning("⚠️ Local LoRA not found, using base SDXL")
```

## ⚡ Производительность и Оптимизации

### Оптимизации Памяти
- **VAE CPU Fallback**: VAE перемещается на CPU для decode операций
- **Memory Management**: Агрессивная очистка с `torch.cuda.empty_cache()` и `gc.collect()`
- **Resource Monitoring**: Мониторинг использования GPU памяти

### Оптимизации Генерации
- **VAE Slicing**: Включено для экономии VRAM
- **VAE Tiling**: Включено для больших изображений
- **Torch Compile**: Автоматически для PyTorch 2.4+
- **CUDA Optimizations**: Специфичные для GPU настройки

### Метрики Производительности
- **TTFB**: Time To First Byte (цель: <30 секунд)
- **Generation Time**: Preview + Final (цель: <60 секунд)
- **Memory Usage**: VRAM consumption (цель: <80% от доступной)

## 🔒 Безопасность и Надежность

### Принципы Безопасности
- **Zero Trust**: Проверка всех внешних зависимостей
- **Configuration Integrity**: Синхронизация между конфигурационными файлами
- **Error Handling**: Комплексная обработка ошибок с fallback механизмами

### Отказоустойчивость
- **Fallback Mechanisms**: Автоматический переход на резервные модели
- **Error Recovery**: Восстановление после критических ошибок
- **Graceful Degradation**: Снижение качества при недоступности ресурсов

## 📊 Мониторинг и Логирование

### Система Логирования
- **Setup Logging**: Время инициализации моделей
- **Generation Logging**: Время генерации preview/final
- **Error Logging**: Детальные сообщения об ошибках
- **Performance Logging**: Метрики производительности

### Мониторинг Ресурсов
- **GPU Memory**: Мониторинг использования VRAM
- **Generation Time**: Время выполнения операций
- **Error Rate**: Частота возникновения ошибок
- **Success Rate**: Процент успешных генераций

## 🏷️ Версионирование

### Текущая Версия
- **Версия**: v4.3.60 "TEXTUAL INVERSION TOKEN FIX & UNIFIED TOK"
- **URL**: https://replicate.com/nauslava/plitka-pro-project:v4.3.60
- **Статус**: Активная

### Архитектурные Улучшения
- **Исправленный размер крошки**: Восстановлен через полные промпты из рабочей модели v45
- **Полные промпты**: "ohwx_rubber_tile <s0><s1>, [цвета], photorealistic rubber tile, high quality, detailed texture"
- **Восстановленный префикс**: Ключевой идентификатор "ohwx_rubber_tile" для правильной активации
- **Восстановленный негативный промпт**: "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract"
- **100% совместимость с v45**: Точное копирование всех параметров рабочей модели
- **Максимальное качество**: Фотореалистичность без артефактов и размытости
- **Оптимизированная генерация**: Один проход вместо двух
- **ControlNet для пропорций**: Автоматическое применение для 2-4 цветов
- **Умная логика выбора**: Автоматический выбор метода генерации
- **Callback для preview**: Извлечение промежуточного результата

### Процесс Развертывания
```bash
# Сборка
cog build

# Публикация
cog push r8.im/nauslava/plitka-pro:v4.3.50

# Тестирование
# Через Replicate API или веб-интерфейс
```

## 📁 Структура Проекта

```
plitka-pro-project/
├── predict.py                    # Основная логика генерации (664 строк)
├── cog.yaml                     # Конфигурация Cog
├── requirements.txt             # Зависимости Python
├── Dockerfile                  # Конфигурация Docker
├── model_files/                # Локальные файлы модели
│   ├── rubber-tile-lora-v4_sdxl_lora.safetensors        # LoRA адаптер (97MB)
│   └── rubber-tile-lora-v4_sdxl_embeddings.safetensors  # Textual Inversion (8.1KB)
├── docs/                       # Документация
│   ├── Project.md              # Этот документ
│   ├── TaskTracker.md          # Трекер задач
│   ├── Changelog.md            # Журнал изменений
│   └── Project_Architecture_Synthesis.md # Архитектурный синтез
└── WEB_INPUTS_v4.3.18.md      # Тестовые запросы
```

## 🔄 Жизненный Цикл Разработки

### Протокол "Verifiable Change"
1. **Анализ**: Изучение текущего состояния и планирование изменений
2. **Согласование**: Представление плана и получение подтверждения
3. **Исполнение**: Реализация изменений с обязательным документированием

### Языковая Дисциплина
- **Внутренние рассуждения**: Английский (для технической точности)
- **Коммуникация и документация**: Русский (для ясности понимания)

### Принципы Качества
- **KISS**: Keep It Simple, Stupid
- **SOLID**: Принципы объектно-ориентированного проектирования
- **DRY**: Don't Repeat Yourself
- **Zero Trust**: Проверка всех зависимостей

## 🎯 Текущее Состояние и Планы

### Текущий Статус
- ✅ **Развертывание**: Успешно на Replicate.com
- ✅ **Локальные модели**: LoRA и Textual Inversion файлы интегрированы
- ✅ **ControlNet**: Полная интеграция для управления ракурсом
- ✅ **API**: Стабильная работа с комплексными запросами

### Нерешенные Проблемы
- 🔄 **Мониторинг производительности**: Анализ производительности в production
- 🔄 **Оптимизация скорости**: Улучшение скорости генерации для сложных запросов
- 🔄 **Тестовое покрытие**: Расширение тестов для различных комбинаций

### Следующие Шаги
1. **Мониторинг**: Анализ производительности в production
2. **Оптимизация**: Улучшение скорости генерации для сложных многоцветных запросов
3. **Тестирование**: Расширение тестового покрытия для различных цветовых комбинаций и углов обзора
4. **Документация**: Поддержание актуальности документации

## 📚 Дополнительная Документация

- **TaskTracker.md**: Детальный трекер задач и прогресса
- **Changelog.md**: Полная история изменений проекта
- **Project_Architecture_Synthesis.md**: Архитектурный синтез проекта
- **WEB_INPUTS_v4.3.18.md**: Примеры тестовых запросов

---

*Документ обновлен: v4.3.57 "NEGATIVE PROMPT RESTORATION & COMPLETE v45 COMPATIBILITY"*
*Последнее обновление: 27 декабря 2024*


