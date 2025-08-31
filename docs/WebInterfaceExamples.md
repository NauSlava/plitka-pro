# Web Interface Examples - Plitka Pro Project

## Текущая версия: v4.4.37-pre

### URL для тестирования:
**https://replicate.com/nauslava/plitka-pro-project/versions/7aededdaee1d74437a4f55a4671b71835eadd9b67d3f1e3c3682388115c929c6**

---

## 📋 Параметры API v4.4.37-pre:

### Входные параметры:
- **`prompt`** (обязательный): `string` - описание цветов резиновой плитки
- **`negative_prompt`** (опциональный): `string` - негативный промпт. Если не задан, подставляется анти‑мозаичный список по умолчанию.
- **`seed`** (опциональный): `integer` - сид для воспроизводимости (по умолчанию: -1 = случайный)

### Выходные файлы:
1. **`preview.png`** - уменьшенная версия (512x512)
2. **`final.png`** - полная версия (1024x1024)
3. **`colormap.png`** - легенда по входным пропорциям (256×256)

---

## 🎨 Примеры запросов для v4.4.37-pre:

### 1. Простые одноцветные запросы:

#### Белая плитка:
```json
{
  "prompt": "100% white"
}
```

#### Красная плитка:
```json
{
  "prompt": "100% red"
}
```

#### Синяя плитка:
```json
{
  "prompt": "100% blue"
}
```

#### Зеленая плитка:
```json
{
  "prompt": "100% green"
}
```

#### Черная плитка:
```json
{
  "prompt": "100% black"
}
```

### 2. Многоцветные запросы:

#### Красная и белая плитка:
```json
{
  "prompt": "50% red, 50% white"
}
```

#### Красная и синяя плитка:
```json
{
  "prompt": "70% red, 30% blue"
}
```

#### Трехцветная плитка:
```json
{
  "prompt": "30% red, 40% blue, 30% green"
}
```

#### Четырехцветная плитка:
```json
{
  "prompt": "25% red, 25% blue, 25% green, 25% yellow"
}
```

### 3. Запросы с негативным промптом:

#### С базовым негативным промптом:
```json
{
  "prompt": "100% red",
  "negative_prompt": "blurry, low quality, deformed"
}
```

#### С расширенным негативным промптом:
```json
{
  "prompt": "100% blue",
  "negative_prompt": "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, painting, drawing, text, sketch, low resolution"
}
```

### 4. Запросы с сидом для воспроизводимости:

#### С фиксированным сидом:
```json
{
  "prompt": "70% red, 30% blue",
  "seed": 42
}
```

#### С другим сидом:
```json
{
  "prompt": "50% green, 50% white",
  "seed": 123
}
```

### 5. Полные запросы (все параметры):

#### Полный запрос с всеми параметрами:
```json
{
  "prompt": "50% red, 30% green, 20% blue",
  "negative_prompt": "mosaic, checkerboard, grid, patchwork, tiled, square blocks, seams, borders, rectangles, collage",
  "seed": 123
}
```

#### Другой полный запрос:
```json
{
  "prompt": "40% white, 30% red, 20% blue, 10% green",
  "negative_prompt": "blurry, low quality, deformed, watermark",
  "seed": 456
}
```

### 6. Специальные эффекты:

#### Глянцевая белая плитка:
```json
{
  "prompt": "100% white, glossy finish"
}
```

#### Матовая черная плитка:
```json
{
  "prompt": "100% black, matte texture"
}
```

#### Крапчатая красная плитка:
```json
{
  "prompt": "70% red, 30% white, speckled pattern"
}
```

---

## 🔧 Технические детали v4.4.37-pre:

### Исправления в этой версии:
- ✅ Качество: steps=35, guidance=6.7 по умолчанию
- ✅ LoRA weight=0.75 с `set_adapters`/`fuse_lora`
- ✅ Анти‑мозаика: отключено VAE tiling + расширенный негативный промпт по умолчанию
- ✅ Colormap: легенда из входных пропорций (не заглушка)
- ✅ Производительность: попытка xFormers/SDPA, channels_last, cudnn.benchmark

### Совместимость:
- **diffusers**: 0.28.2
- **torch**: 2.2.0
- **transformers**: 4.39.3
- **accelerate**: 0.29.3
- **huggingface_hub**: 0.22.2

### Ожидаемые результаты:
- Реалистичная резиновая плитка
- Правильные цвета согласно промпту
- Детальная текстура поверхности
- Профессиональное качество изображения
- Использование НАШИХ обученных токенов `<s0><s1>`

---

## 🚀 Готов к тестированию!

**Модель v4.4.37-pre опубликована. Негативный промпт подставляется автоматически, при необходимости переопределяйте.**
