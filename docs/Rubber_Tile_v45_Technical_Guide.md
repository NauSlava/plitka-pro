# Детальное Техническое Руководство: Модель "nauslava/rubber-tile-lora-v45"

## Обзор Модели

**Модель:** ````nauslava/rubber-tile-lora-v45`  
**Версия:** `763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30`  
**Платформа:** Replicate  
**Статус:** Приватная модель  
**Дата создания:** 2025-08-09T10:39:51.812255Z  
**Cog версия:** 0.16.2  

## API Спецификация

### Входные Параметры

#### 1. **prompt** (обязательный)
- **Тип:** string
- **Описание:** "Описание цветов и пропорций, например: 50% red, 30% black, 20% pearl"
- **Порядок:** 0
- **Формат:** Простой процентный формат для WEB-интерфейса
- **Примеры:**
  - `"50% red, 30% black, 20% pearl"`
  - `"100% white"`
  - `"60% blue, 40% green"`
- **Важно:** Внутри модели этот промпт преобразуется в полный формат с токенами

#### 2. **negative_prompt** (опциональный)
- **Тип:** string
- **Значение по умолчанию:** `"blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract"`
- **Порядок:** 1
- **Описание:** "Дополнительный негативный промпт."

#### 3. **seed** (опциональный)
- **Тип:** integer
- **Значение по умолчанию:** -1
- **Порядок:** 2
- **Описание:** "Сид для воспроизводимости. Оставьте -1 для случайного."

### Выходные Данные
- **Тип:** string (URI)
- **Формат:** Изображение (PNG/JPG)
- **Размер:** 1024x1024 пикселей

## Архитектура Модели

### Базовая Структура
```
Stable Diffusion XL (SDXL) 1.0
├── Base Model: stabilityai/stable-diffusion-xl-base-1.0
├── LoRA Adapters: rubber-tile-lora-v4_sdxl_lora.safetensors
├── Textual Inversion: rubber-tile-lora-v4_sdxl_embeddings.safetensors
├── Trigger Tokens: <s0><s1>
├── Scheduler: DPMSolverMultistepScheduler
└── VAE: Оптимизированный для SDXL
```

### Ключевые Компоненты

#### 1. **Stable Diffusion XL Pipeline**
- **Тип:** `StableDiffusionXLPipeline`
- **Версия:** SDXL 1.0
- **Точность:** float16
- **Вариант:** fp16
- **Безопасность:** safetensors

#### 2. **LoRA Адаптеры**
- **Файл:** `rubber-tile-lora-v4_sdxl_lora.safetensors`
- **Сила:** 0.7
- **Состояние:** Fused (объединен с базовой моделью)
- **Формат:** safetensors

#### 3. **Textual Inversion Embeddings**
- **Файл:** `rubber-tile-lora-v4_sdxl_embeddings.safetensors`
- **Токены:** `<s0>`, `<s1>`
- **Формат:** SDXL dual-encoder (clip_g, clip_l)
- **Формат:** safetensors

#### 4. **Планировщик**
- **Тип:** `DPMSolverMultistepScheduler`
- **Алгоритм:** dpmsolver++
- **Сигмы:** Karras
- **Оптимизация:** Многошаговый алгоритм

#### 5. **VAE Оптимизации**
- **Slicing:** Включено
- **Tiling:** Включено
- **CPU Fallback:** Для decode операций

## Логика Генерации

### Обработка Промптов

#### WEB-интерфейс (Входной формат)
```
"{процент}% {цвет}, {процент}% {цвет}, ..."
```

#### Внутренний формат (После преобразования)
```
"ohwx_rubber_tile <s0><s1>, {процент}% {цвет}, {процент}% {цвет}, ..."
```

#### Примеры Промптов

**WEB-интерфейс (что вводит пользователь):**
```python
# Один цвет
"100% white"

# Два цвета
"50% red, 50% black"

# Три цвета
"40% blue, 35% green, 25% yellow"

# Четыре цвета
"30% red, 25% blue, 25% green, 20% black"
```

**Внутренний формат (что генерирует модель):**
```python
# Один цвет
"ohwx_rubber_tile <s0><s1>, 100% white, photorealistic rubber tile, high quality, detailed texture"

# Два цвета
"ohwx_rubber_tile <s0><s1>, 50% red, 50% black, photorealistic rubber tile, high quality, detailed texture"

# Три цвета
"ohwx_rubber_tile <s0><s1>, 40% blue, 35% green, 25% yellow, photorealistic rubber tile, high quality, detailed texture"

# Четыре цвета
"ohwx_rubber_tile <s0><s1>, 30% red, 25% blue, 25% green, 20% black, photorealistic rubber tile, high quality, detailed texture"
```

#### Поддерживаемые Цвета
- **Основные:** red, blue, green, yellow, white, black
- **Дополнительные:** orange, purple, pink, brown, gray
- **Специальные:** pearl, metallic, neon

### Настройки Генерации

#### Параметры по Умолчанию
```python
generation_params = {
    "num_inference_steps": 20,
    "guidance_scale": 7.0,
    "width": 1024,
    "height": 1024,
    "generator": torch.Generator(device=device).manual_seed(seed)
}
```

#### Обработка Seed
```python
if seed == -1:
    seed = random.randint(0, 999999999)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## Техническая Реализация

### Инициализация Модели

```python
def setup(self):
    """Инициализация модели при запуске сервера."""
    
    # Определение устройства
    if torch.cuda.is_available():
        self.device = "cuda"
        # Выбор GPU с наибольшей памятью
        best_gpu = max(range(torch.cuda.device_count()), 
                      key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        torch.cuda.set_device(best_gpu)
    else:
        self.device = "cpu"
    
    # Загрузка SDXL pipeline
    self.pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        resume_download=False
    )
    
    # Перемещение на GPU
    self.pipe = self.pipe.to(self.device)
    
    # Загрузка LoRA
    lora_path = "/src/model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
    self.pipe.set_adapters(["rubber-tile-lora-v4_sdxl_lora"], adapter_weights=[0.7])
    self.pipe.fuse_lora()
    
    # Загрузка Textual Inversion
    ti_path = "/src/model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors"
    self.pipe.load_textual_inversion(ti_path, token="<s0>")
    
    # Настройка планировщика
    self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        self.pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )
    
    # Оптимизации VAE
    self.pipe.vae.enable_slicing()
    self.pipe.vae.enable_tiling()
```

### Основная Функция Генерации

```python
def predict(self, prompt: str, negative_prompt: str = None, seed: int = -1):
    """Генерация изображения резиновой плитки."""
    
    # Стандартный негативный промпт
    if negative_prompt is None:
        negative_prompt = "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract"
    
    # Установка сида
    if seed == -1:
        seed = random.randint(0, 999999999)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Преобразование простого промпта в полный формат
    full_prompt = f"ohwx_rubber_tile <s0><s1>, {prompt}, photorealistic rubber tile, high quality, detailed texture"
    
    # Генерация изображения
    result = self.pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=20,
        guidance_scale=7.0,
        width=1024,
        height=1024,
        generator=torch.Generator(device=self.device).manual_seed(seed)
    )
    
    return result.images[0]
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

### requirements.txt
```
torch>=2.0.0
diffusers>=0.27.0
transformers>=4.47.0
accelerate>=0.27.0
safetensors>=0.4.0
Pillow>=10.0.0
numpy>=1.24.0
```

## Переменные Окружения

### Оптимизация Производительности
```bash
# Кэш Hugging Face
HF_HOME=/tmp/hf_home
HF_DATASETS_CACHE=/tmp/hf_datasets_cache
HF_MODELS_CACHE=/tmp/hf_models_cache

# Отключение миграции кэша
TRANSFORMERS_CACHE_MIGRATION_DISABLE=1
HF_HUB_CACHE_MIGRATION_DISABLE=1

# Отключение телеметрии
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_PROGRESS_BARS=1

# Настройки загрузки
HF_HUB_ENABLE_HF_TRANSFER=1
HF_HUB_DOWNLOAD_TIMEOUT=500

# Уровень логирования
TRANSFORMERS_VERBOSITY=error
TOKENIZERS_PARALLELISM=false
```

## Ограничения и Особенности

### Ограничения Модели
1. **Угол обзора:** Только вид сверху (0°)
2. **Формат промпта:** Строго процентный формат
3. **Размер изображения:** Фиксированный 1024x1024
4. **Количество цветов:** До 4 цветов в одной плитке

### Особенности Генерации
1. **Консистентность:** Одинаковые результаты при одинаковом seed
2. **Качество:** Оптимизировано для резиновой плитки
3. **Скорость:** 20 шагов генерации для оптимального баланса
4. **Память:** Эффективное использование GPU

### Поддерживаемые Сценарии
- Генерация одноцветной плитки
- Генерация многоцветной плитки с пропорциями
- Воспроизводимые результаты через seed
- Быстрая генерация для предварительного просмотра

## Рекомендации по Использованию

### Лучшие Практики
1. **Формат промптов:** Используйте точный процентный формат
2. **Seed управление:** Используйте фиксированный seed для воспроизводимости
3. **Негативный промпт:** Используйте стандартный для лучшего качества
4. **Цветовые комбинации:** Не более 4 цветов для оптимального результата

### Примеры Использования

**WEB-интерфейс (что вводит пользователь):**
```python
# Простой запрос
{
    "prompt": "100% white",
    "seed": 12345
}

# Сложный запрос
{
    "prompt": "40% red, 30% blue, 20% green, 10% black",
    "negative_prompt": "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract",
    "seed": -1
}
```

**Внутреннее преобразование (что происходит в модели):**
```python
# Простой запрос
{
    "prompt": "ohwx_rubber_tile <s0><s1>, 100% white, photorealistic rubber tile, high quality, detailed texture",
    "seed": 12345
}

# Сложный запрос
{
    "prompt": "ohwx_rubber_tile <s0><s1>, 40% red, 30% blue, 20% green, 10% black, photorealistic rubber tile, high quality, detailed texture",
    "negative_prompt": "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract",
    "seed": -1
}
```

## Заключение

Модель "nauslava/rubber-tile-lora-v45" представляет собой **оптимизированное решение** для генерации изображений резиновой плитки с использованием:

- **Простой и эффективной архитектуры** SDXL + LoRA
- **Стандартизированного формата промптов** с процентным описанием цветов
- **Оптимизированных настроек генерации** для стабильной работы
- **Эффективного управления ресурсами** и памятью

Эта модель демонстрирует **успешную реализацию** специализированной генерации изображений с **умным разделением интерфейса и логики**:

### Ключевые Особенности:
1. **WEB-интерфейс:** Простой процентный формат для пользователей
2. **Внутренняя логика:** Полный формат с токенами для модели
3. **Автоматическое преобразование:** Прозрачное для пользователя
4. **Полная архитектура:** LoRA + Textual Inversion для максимального качества

## Ключевые Особенности Реализации

**Модель v45 использует полную архитектуру: LoRA + Textual Inversion!**

### Архитектурная Полнота
- ✅ **LoRA:** `rubber-tile-lora-v4_sdxl_lora.safetensors` (сила 0.7)
- ✅ **Textual Inversion:** `rubber-tile-lora-v4_sdxl_embeddings.safetensors`
- ✅ **Специальные токены:** Используются (`<s0>`, `<s1>`)
- ✅ **Правильные промпты:** `ohwx_rubber_tile <s0><s1>, [цвета]`

### Преимущества Такого Подхода
1. **Полнота:** Использование всех обученных компонентов
2. **Точность:** Соответствие обучению модели
3. **Качество:** Максимальное качество генерации
4. **Совместимость:** Полная совместимость с обученной моделью
