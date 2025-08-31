# 🚀 Быстрый старт с Plitka Pro

## Ссылка на модель
**https://replicate.com/nauslava/plitka-pro-project:v4.4.06**

## Минимальный рабочий пример

Скопируйте этот JSON в поле `prompt`:

```json
{"prompt": "100% white"}
```

## Что вы получите
- **preview.png** - быстрое превью (512x512)
- **final.png** - финальное изображение (1024x1024)  
- **colormap.png** - карта цветов для отладки

## Основные параметры

### Простой промпт
```json
{
  "prompt": "100% white"
}
```

### Многоцветная плитка
```json
{
  "prompt": "50% red, 30% green, 20% blue"
}
```

### Полный запрос
```json
{
  "prompt": "70% red, 30% blue",
  "negative_prompt": "blurry, worst quality, low quality, deformed, watermark",
  "seed": 123
}
```

## ⚠️ Важное ограничение

**Угол обзора**: Только 0° (вид сверху)
- Модель обучена исключительно на top-down изображениях
- Любые другие углы (45°, 90°) дадут непредсказуемые результаты

## Поддерживаемые цвета
- `black`, `white`, `red`, `green`, `blue`, `yellow`, `gray`, `brown`, `orange`, `purple`, `pink`

## Примеры для копирования

### Быстрый тест
```json
{"prompt": "100% blue"}
```

### Двухцветная плитка
```json
{"prompt": "60% red, 40% yellow"}
```

### Многоцветная плитка
```json
{"prompt": "40% black, 30% red, 20% blue, 10% yellow"}
```

### С негативным промптом
```json
{
  "prompt": "80% green, 20% gray",
  "negative_prompt": "blurry, worst quality, low quality, deformed, watermark",
  "seed": 42
}
```

## Советы
1. Используйте простые промпты для быстрого тестирования
2. Используйте `seed` для воспроизводимости результатов
3. Пропорции автоматически нормализуются
4. Модель генерирует только вид сверху (0°)
5. Качество оптимизировано для быстрой генерации (20 шагов)

## Подробная документация
См. `docs/WebInterfaceExamples.md` для полного руководства.

---

*Версия: v4.4.06 "RUBBER TILE GENERATION MODEL (СОВМЕСТИМЫЕ ВЕРСИИ БИБЛИОТЕК)"*
