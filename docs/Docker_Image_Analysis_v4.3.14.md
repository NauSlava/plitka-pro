# 🔍 Анализ Docker образа Plitka Pro v4.3.14

**Образ**: `r8.im/nauslava/plitka-pro-project:v4.3.14`  
**SHA256**: `bd6ac4dccfe234b38d50a424288a57c1dea1de2f632362efb9bba6107aec3550`  
**Дата создания**: 25 августа 2024  
**Размер**: 25.4GB  

---

## 🏗️ Архитектура системы

### **Основные компоненты:**
- **Base Model**: Stable Diffusion XL (SDXL) 1.0
- **Fine-tuning**: LoRA + Textual Inversion
- **Control**: ControlNet для геометрического контроля
- **Pipeline**: StableDiffusionXLControlNetPipeline

### **Структура файлов:**
```
/
├── cog.yaml                    # Конфигурация Cog
├── predict.py                  # Основная логика генерации
├── requirements.txt            # Зависимости Python
├── model_files/               # Модели LoRA и TI
│   ├── rubber-tile-lora-v4_sdxl_lora.safetensors      # LoRA веса (97MB)
│   └── rubber-tile-lora-v4_sdxl_embeddings.safetensors # TI эмбеддинги (8KB)
├── docs/                      # Документация
├── references/                # Референсные изображения
└── test_*.py                  # Тестовые скрипты
```

---

## ⚙️ Конфигурация (cog.yaml)

### **Версия**: v4.3.14 - RETURN FIX & ACCELERATE REMOVAL
```yaml
build:
  gpu: true
  python_version: "3.11"
  python_requirements: "requirements.txt"
  cog_runtime: true
  run:
    - mkdir -p /src/model_files
    - mkdir -p /src/refs
predict: "predict.py:OptimizedPredictor"
```

### **Особенности:**
- ✅ Поддержка GPU
- ✅ Python 3.11
- ✅ Автоматическое создание директорий для моделей
- ✅ Использование OptimizedPredictor класса

---

## 📦 Зависимости (requirements.txt)

### **Основные библиотеки:**
```txt
diffusers==0.24.0              # Pipeline для генерации
transformers==4.35.2           # Модели Hugging Face
accelerate==0.24.1             # Оптимизация производительности
peft==0.7.1                    # Parameter-Efficient Fine-Tuning
safetensors==0.4.2             # Безопасное хранение весов
huggingface-hub==0.19.4        # Доступ к моделям HF
opencv-python-headless==4.10.0.84  # Обработка изображений
numpy==1.26.4                  # Численные вычисления
Pillow==10.3.0                 # Работа с изображениями
torch>=2.4.0                   # PyTorch
torchvision>=0.19.0            # Vision модели PyTorch
psutil>=5.9.0                  # Мониторинг системных ресурсов
```

---

## 🧠 Основная логика генерации (predict.py)

### **Класс**: `OptimizedPredictor`

### **Ключевые методы:**

#### **1. setup() - Инициализация модели**
```python
def setup(self, weights: Optional[Path] = None) -> None:
    # Выбор лучшего устройства (GPU/CPU)
    self.device_info = select_best_device()
    
    # Загрузка SDXL pipeline
    self.pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_REPO_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
        requires_safety_checker=False,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Загрузка LoRA весов
    lora_path = "./model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
    self.pipe.load_lora_weights(lora_path)
    self.pipe.fuse_lora()
    
    # Загрузка Textual Inversion
    ti_path = "./model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors"
    self.pipe.load_textual_inversion(ti_path, token="<s0>")
```

#### **2. _build_prompt() - Построение промпта**
```python
def _build_prompt(self, colors: List[Dict[str, Any]]) -> str:
    prompt_parts = ["ohwx_rubber_tile <s0><s1>"]
    
    # Добавление цветов с пропорциями
    if colors:
        color_desc = []
        for color_info in colors:
            name = color_info.get("name", "").lower()
            proportion = color_info.get("proportion", 0)
            if proportion > 0:
                percentage = int(proportion)
                color_desc.append(f"{percentage}% {name}")
        
        if color_desc:
            prompt_parts.append(", ".join(color_desc))
    
    # Добавление дескрипторов качества
    prompt_parts.extend([
        "photorealistic rubber tile",
        "high quality",
        "detailed texture"
    ])
    
    return ", ".join(prompt_parts)
```

#### **3. predict() - Основная генерация**
```python
def predict(self, params_json: str) -> List[Path]:
    # Парсинг параметров
    params = self._parse_params_json(params_json)
    colors = params.get("colors", [])
    angle = int(params.get("angle", 0))
    quality = str(params.get("quality", "standard"))
    seed = int(params.get("seed", -1))
    overrides = params.get("overrides", {})
    
    # Настройки качества
    if quality == "preview":
        steps_preview, steps_final = 35, 40
        guidance_scale_default = 6.5
    elif quality == "high":
        steps_preview, steps_final = 40, 60
        guidance_scale_default = 6.0
    else:  # standard
        steps_preview, steps_final = 35, 50
        guidance_scale_default = 6.5
    
    # Построение промпта
    base_prompt = self._build_prompt(colors)
    
    # Определение необходимости ControlNet
    should_use_controlnet, reason = self._should_use_controlnet(angle)
    
    # Генерация preview (512x512)
    preview_result = self.pipe(
        prompt=base_prompt,
        negative_prompt="object, blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, smooth, flat",
        width=512, height=512,
        num_inference_steps=steps_preview,
        guidance_scale=guidance_scale,
        generator=generator
    )
    
    # Генерация финального изображения (1024x1024)
    final_result = self.pipe(
        prompt=base_prompt,
        negative_prompt="object, blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, smooth, flat",
        width=1024, height=1024,
        num_inference_steps=steps_final,
        guidance_scale=guidance_scale,
        generator=generator
    )
    
    return [preview_path, final_path, colormap_path]
```

---

## 🎨 Логика промптов

### **Базовый формат промпта:**
```
ohwx_rubber_tile <s0><s1>, [цвета], photorealistic rubber tile, high quality, detailed texture
```

### **Примеры промптов:**

#### **Монохромная плитка:**
```
ohwx_rubber_tile <s0><s1>, 100% black, photorealistic rubber tile, high quality, detailed texture
```

#### **Двухцветная плитка:**
```
ohwx_rubber_tile <s0><s1>, 60% red, 40% blue, photorealistic rubber tile, high quality, detailed texture
```

#### **Трехцветная плитка:**
```
ohwx_rubber_tile <s0><s1>, 50% black, 30% white, 20% red, photorealistic rubber tile, high quality, detailed texture
```

### **Negative prompt:**
```
object, blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, smooth, flat
```

---

## 🔧 ControlNet логика

### **Условия использования ControlNet:**
```python
def _should_use_controlnet(self, angle: int) -> Tuple[bool, str]:
    angle_norm = int(angle) % 180
    
    # Угол 0° - единственный надежный без ControlNet
    if angle_norm == 0:
        return False, "Угол 0° (вид сверху) - единственный надежный ракурс без ControlNet"
    
    # Для других углов - использовать ControlNet
    if angle_norm in (45, 135):
        return True, f"Угол {angle}° требует ControlNet для геометрического контроля"
    elif angle_norm in (30, 60, 90, 120, 150):
        return True, f"Угол {angle}° требует ControlNet для стабильной генерации"
    else:
        return True, f"Угол {angle}° требует ControlNet (нестандартный ракурс)"
```

### **Выбор ControlNet по углу:**
- **0°**: Без ControlNet (базовая модель)
- **45°, 135°**: SoftEdge ControlNet
- **90°**: Canny ControlNet
- **Другие углы**: Автоматический выбор подходящего

---

## 📊 Профили качества

### **Preview (быстрое):**
- **Шаги**: 35 (preview), 40 (final)
- **Размер**: 512x512 (preview), 1024x1024 (final)
- **Guidance Scale**: 6.5
- **Время**: ~15-30 секунд

### **Standard (стандартное):**
- **Шаги**: 35 (preview), 50 (final)
- **Размер**: 512x512 (preview), 1024x1024 (final)
- **Guidance Scale**: 6.5
- **Время**: ~30-45 секунд

### **High (высокое):**
- **Шаги**: 40 (preview), 60 (final)
- **Размер**: 512x512 (preview), 1024x1024 (final)
- **Guidance Scale**: 6.0
- **Время**: ~45-60 секунд

---

## 🎯 Основные параметры

### **Обязательные параметры:**
- **colors**: Массив цветов с пропорциями
- **angle**: Угол укладки (0-180°)
- **quality**: "preview", "standard", "high"
- **seed**: Seed для воспроизводимости (-1 для случайного)

### **Опциональные параметры (overrides):**
- **use_controlnet**: Принудительное включение/отключение
- **guidance_scale**: Масштаб руководства (1.0-20.0)
- **num_inference_steps**: Количество шагов генерации
- **negative_prompt**: Кастомный негативный промпт

### **Поддерживаемые цвета:**
- black, white, red, green, blue
- gray, brown, yellow, orange, purple, pink

---

## 🚀 Оптимизации

### **Управление памятью:**
- **Lazy Loading**: ControlNet модели загружаются только при необходимости
- **GPU Memory Management**: Автоматическая очистка CUDA кэша
- **Resource Monitoring**: Мониторинг использования ресурсов в реальном времени

### **Производительность:**
- **LoRA Fusion**: Объединение LoRA весов для ускорения
- **Device Optimization**: Автоматический выбор лучшего устройства
- **Memory Fraction Control**: Ограничение использования памяти GPU

---

## 📝 Формат запроса

### **JSON структура:**
```json
{
  "params_json": "{\"colors\":[{\"name\":\"black\",\"proportion\":60},{\"name\":\"white\",\"proportion\":40}],\"angle\":0,\"quality\":\"standard\",\"seed\":42}"
}
```

### **Примеры запросов:**

#### **Базовая плитка:**
```json
{
  "colors": [{"name": "black", "proportion": 100}],
  "angle": 0,
  "quality": "standard",
  "seed": 42
}
```

#### **Плитка с ControlNet:**
```json
{
  "colors": [{"name": "red", "proportion": 70}, {"name": "blue", "proportion": 30}],
  "angle": 45,
  "quality": "high",
  "seed": 123
}
```

---

## 🔍 Ключевые особенности v4.3.14

### **Архитектурные решения:**
1. **Lazy Loading Architecture**: ControlNet модели загружаются только при необходимости
2. **Resource Management**: Комплексное управление GPU памятью
3. **Error Handling**: Улучшенная обработка ошибок загрузки моделей
4. **Performance Optimization**: Оптимизация для Replicate платформы

### **Ограничения:**
1. **Угол 0°**: Единственный надежный ракурс без ControlNet
2. **ControlNet Dependency**: Другие углы требуют ControlNet для качества
3. **Memory Requirements**: Минимум 14GB VRAM для полного pipeline

### **Преимущества:**
1. **High Quality**: Профессиональное качество генерации
2. **Flexibility**: Поддержка различных цветовых комбинаций
3. **Stability**: Стабильная работа на Replicate
4. **Performance**: Оптимизированная производительность

---

## 📊 Сравнение с текущей версией

### **v4.3.14 (исследуемая):**
- ✅ Lazy Loading Architecture
- ✅ Resource Management
- ✅ ControlNet Integration
- ✅ LoRA + Textual Inversion
- ⚠️ Устаревшие версии библиотек

### **v4.3.43 (текущая):**
- ✅ Обновленные библиотеки (torch 2.4.0, diffusers 0.26.3)
- ✅ Улучшенная обработка предупреждений
- ✅ Оптимизированные промпты
- ✅ Современная архитектура

---

## 🎯 Выводы и рекомендации

### **Сильные стороны v4.3.14:**
1. **Проверенная архитектура**: Стабильная работа на Replicate
2. **Lazy Loading**: Эффективное использование памяти
3. **ControlNet Integration**: Гибкость в генерации различных ракурсов
4. **Resource Management**: Хорошее управление ресурсами

### **Области для улучшения:**
1. **Обновление библиотек**: Переход на современные версии
2. **Оптимизация промптов**: Улучшение качества генерации
3. **Error Handling**: Более детальная обработка ошибок
4. **Performance Monitoring**: Расширенный мониторинг производительности

### **Рекомендации по использованию:**
1. **Для production**: Использовать текущую версию v4.3.43
2. **Для исследования**: v4.3.14 полезна для понимания архитектуры
3. **Для обучения**: Изучить подходы к управлению ресурсами и Lazy Loading
4. **Для развития**: Взять за основу проверенные решения по ControlNet

---

**Дата анализа**: 26 декабря 2024  
**Аналитик**: AI Assistant  
**Статус**: Завершено ✅**
