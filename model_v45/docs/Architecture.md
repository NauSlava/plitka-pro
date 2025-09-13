# Архитектура Модели "nauslava/rubber-tile-lora-v45"

## Обзор Архитектуры

Модель v45 построена на основе **НАШЕЙ обученной модели** с использованием современных технологий машинного обучения для генерации изображений резиновой плитки.

## Базовая Структура

```
Stable Diffusion XL (SDXL) 1.0
├── Base Model: stabilityai/stable-diffusion-xl-base-1.0
├── LoRA Adapters: rubber-tile-lora-v4_sdxl_lora.safetensors (ОБУЧЕННЫЕ!)
├── Textual Inversion: rubber-tile-lora-v4_sdxl_embeddings.safetensors (ОБУЧЕННЫЕ!)
├── Trigger Tokens: <s0><s1> (ОБУЧЕННЫЕ!)
├── Scheduler: DPMSolverMultistepScheduler
└── VAE: Оптимизированный для SDXL
```

## Ключевые Компоненты

### 1. Базовая Модель
- **Модель:** Stable Diffusion XL 1.0
- **Размер:** ~6.9 GB
- **Формат:** FP16 (половинная точность)
- **Оптимизация:** Hugging Face Diffusers

### 2. LoRA Адаптеры (ОБУЧЕННЫЕ!)
- **Файл:** `rubber-tile-lora-v4_sdxl_lora.safetensors`
- **Размер:** 97.3 MB
- **Сила:** 0.7
- **Состояние:** Fused (объединен с базовой моделью)
- **Слой:** Attention слои SDXL

### 3. Textual Inversion (ОБУЧЕННЫЕ!)
- **Файл:** `rubber-tile-lora-v4_sdxl_embeddings.safetensors`
- **Размер:** 8.1 KB
- **Формат:** SDXL dual-encoder
- **Токены:** `<s0>`, `<s1>`
- **Активация:** `ohwx_rubber_tile`

### 4. Планировщик
- **Тип:** DPMSolverMultistepScheduler
- **Алгоритм:** dpmsolver++
- **Сигмы:** Karras
- **Преимущества:** Быстрая сходимость, стабильность

### 5. VAE (Variational Autoencoder)
- **Оптимизации:** Slicing, Tiling
- **Формат:** FP16
- **Совместимость:** SDXL 1.0

## Процесс Инициализации

### Этап 1: Определение Устройства
```python
if torch.cuda.is_available():
    self.device = "cuda"
    best_gpu = max(range(torch.cuda.device_count()), 
                  key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    torch.cuda.set_device(best_gpu)
else:
    self.device = "cpu"
```

### Этап 2: Загрузка SDXL
```python
self.pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    resume_download=False
)
```

### Этап 3: Загрузка LoRA
```python
self.pipe.set_adapters(["rubber-tile-lora-v4_sdxl_lora"], adapter_weights=[0.7])
self.pipe.fuse_lora()
```

### Этап 4: Загрузка Textual Inversion
```python
self.pipe.load_textual_inversion(ti_path, token="<s0>")
```

### Этап 5: Настройка Планировщика
```python
self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    self.pipe.scheduler.config,
    algorithm_type="dpmsolver++",
    use_karras_sigmas=True
)
```

### Этап 6: VAE Оптимизации
```python
self.pipe.vae.enable_slicing()
self.pipe.vae.enable_tiling()
```

## Обработка Промптов

### Пользовательский Уровень
```
"{процент}% {цвет}, {процент}% {цвет}, ..."
```

### Системный Уровень
```
"ohwx_rubber_tile <s0><s1>, {процент}% {цвет}, {процент}% {цвет}, ..., photorealistic rubber tile, high quality, detailed texture, professional photography, sharp focus"
```

### Функция Преобразования
```python
def _build_prompt(self, color_description: str) -> str:
    base_prompt = "ohwx_rubber_tile <s0><s1>"
    full_prompt = f"{base_prompt}, {color_description}"
    
    quality_descriptors = [
        "photorealistic rubber tile",
        "high quality",
        "detailed texture",
        "professional photography",
        "sharp focus"
    ]
    
    full_prompt += ", " + ", ".join(quality_descriptors)
    return full_prompt
```

## Процесс Генерации

### 1. Валидация Входа
- Проверка формата промпта
- Установка сида
- Подготовка негативного промпта

### 2. Преобразование Промпта
- Добавление базового промпта
- Включение токенов активации
- Добавление качественных дескрипторов

### 3. Генерация Изображения
```python
result = self.pipe(
    prompt=full_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=7.0,
    width=1024,
    height=1024,
    generator=torch.Generator(device=self.device).manual_seed(seed)
)
```

### 4. Постобработка
- Очистка GPU памяти
- Сборка мусора
- Возврат результата

## Оптимизации Производительности

### Память
- **VAE Slicing:** Разделение больших изображений
- **VAE Tiling:** Использование тайлов
- **Автоматическая очистка:** torch.cuda.empty_cache()

### Скорость
- **FP16:** Половинная точность
- **Fused LoRA:** Объединение с базовой моделью
- **Оптимизированный планировщик:** DPMSolver++

### Стабильность
- **Seed Control:** Воспроизводимые результаты
- **Guidance Scale:** 7.0 (оптимальный баланс)
- **20 шагов:** Оптимальное качество/скорость

## Переменные Окружения

### Hugging Face
```bash
HF_HOME=/tmp/hf_home
HF_DATASETS_CACHE=/tmp/hf_datasets_cache
HF_MODELS_CACHE=/tmp/hf_models_cache
```

### Оптимизации
```bash
TRANSFORMERS_CACHE_MIGRATION_DISABLE=1
HF_HUB_CACHE_MIGRATION_DISABLE=1
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_PROGRESS_BARS=1
TRANSFORMERS_VERBOSITY=error
TOKENIZERS_PARALLELISM=false
```

## Совместимость

### Системные Требования
- **OS:** Linux (Ubuntu 20.04+)
- **CUDA:** 12.4
- **Python:** 3.11
- **GPU:** NVIDIA с поддержкой CUDA
- **Memory:** Минимум 16 GB RAM, 12 GB VRAM

### Библиотеки
- **PyTorch:** 2.0.0+
- **Diffusers:** 0.27.0+
- **Transformers:** 4.47.0+
- **Accelerate:** 0.27.0+

## Безопасность

### Файлы Модели
- **Формат:** safetensors (безопасный)
- **Проверка:** Целостность файлов
- **Доступ:** Только для авторизованных пользователей

### API
- **Валидация:** Строгая проверка входных данных
- **Ограничения:** Rate limiting, размер промпта
- **Логирование:** Детальное отслеживание

## Мониторинг

### Метрики Производительности
- **Время инициализации:** ~30-60 секунд
- **Время генерации:** ~20-40 секунд
- **Использование памяти:** ~8-12 GB VRAM

### Логи
- **Уровень:** INFO
- **Формат:** Структурированный
- **Содержание:** Все этапы обработки

## Расширяемость

### Возможные Улучшения
- **ControlNet:** Добавление масок
- **Множественные ракурсы:** Обучение на других углах
- **Дополнительные стили:** Различные типы плитки

### Ограничения
- **Ракурс:** Только вид сверху
- **Размер:** Фиксированный 1024x1024
- **Формат:** Только резиновая плитка

## Заключение

Архитектура модели v45 представляет собой **оптимальное сочетание** НАШЕЙ обученной модели с современными технологиями оптимизации. Ключевым фактором успеха является **использование проверенных LoRA и Textual Inversion файлов**, которые обеспечивают максимальное качество генерации при минимальных ресурсах.
