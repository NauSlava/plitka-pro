# Справочник API - Модель "nauslava/rubber-tile-lora-v45"

## Обзор API

Модель v45 предоставляет простой и эффективный API для генерации изображений резиновой плитки с использованием НАШЕЙ обученной модели.

## Endpoint

```
POST /predict
```

## Входные Параметры

### Обязательные

#### `prompt` (string)
- **Описание:** Описание цветов и пропорций плитки
- **Формат:** Простой процентный формат для WEB-интерфейса
- **Примеры:**
  - `"50% red, 30% black, 20% pearl"`
  - `"100% white"`
  - `"60% blue, 40% green"`

### Опциональные

#### `negative_prompt` (string)
- **Описание:** Что НЕ должно появляться в изображении
- **По умолчанию:** `"blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, painting, drawing, text, sketch, low resolution"`
- **Пример:** `"text, watermark, blurry"`

#### `seed` (integer)
- **Описание:** Сид для воспроизводимости результатов
- **По умолчанию:** `-1` (случайный)
- **Диапазон:** `0` до `999999999`

## Выходные Данные

### Успешный ответ
- **Тип:** Изображение
- **Формат:** PNG/JPG
- **Размер:** 1024x1024 пикселей
- **URI:** Ссылка на сгенерированное изображение

### Ошибка
```json
{
  "error": "Описание ошибки",
  "details": "Детали ошибки"
}
```

## Примеры Запросов

### Простой запрос
```json
{
  "prompt": "100% white"
}
```

### Запрос с дополнительными параметрами
```json
{
  "prompt": "50% red, 30% black, 20% pearl",
  "negative_prompt": "text, watermark",
  "seed": 12345
}
```

### Сложный запрос
```json
{
  "prompt": "40% blue, 35% green, 25% yellow",
  "negative_prompt": "blurry, low quality, deformed, 3d render",
  "seed": 98765
}
```

## Внутренняя Обработка

### Преобразование Промпта
Модель автоматически преобразует простой пользовательский промпт в полный формат:

**Вход:** `"50% red, 30% black, 20% pearl"`

**Внутренний формат:**
```
"ohwx_rubber_tile <s0><s1>, 50% red, 30% black, 20% pearl, photorealistic rubber tile, high quality, detailed texture, professional photography, sharp focus"
```

### Использование НАШЕЙ Обученной Модели
- **LoRA:** `rubber-tile-lora-v4_sdxl_lora.safetensors` (сила 0.7)
- **Textual Inversion:** `rubber-tile-lora-v4_sdxl_embeddings.safetensors`
- **Токены активации:** `<s0><s1>`

## Параметры Генерации

### Фиксированные параметры
- **Шаги:** 20
- **Guidance Scale:** 7.0
- **Размер:** 1024x1024
- **Формат:** float16

### Оптимизации
- **VAE Slicing:** Включено
- **VAE Tiling:** Включено
- **Memory Management:** Автоматическое

## Ограничения

### Ракурс
- **Поддерживается:** Только вид сверху (0°)
- **Причина:** Модель обучена только на этом ракурсе

### Размеры
- **Фиксированный размер:** 1024x1024 пикселей
- **Изменение:** Не поддерживается

### Цвета
- **Поддерживаются:** Все стандартные цвета
- **Формат:** Процентное описание

## Коды Ошибок

### Общие ошибки
- **400:** Неверный формат промпта
- **500:** Внутренняя ошибка сервера
- **503:** Сервис недоступен

### Специфичные ошибки
- **422:** Неподдерживаемый ракурс
- **413:** Слишком длинный промпт
- **429:** Превышен лимит запросов

## Мониторинг

### Метрики
- **Время генерации:** ~20-40 секунд
- **Использование памяти:** ~8-12 GB VRAM
- **Время инициализации:** ~30-60 секунд

### Логи
- **Уровень:** INFO
- **Формат:** Структурированный JSON
- **Хранение:** Временное (в контейнере)

## Безопасность

### Валидация
- **Промпт:** Проверка длины и формата
- **Параметры:** Валидация диапазонов
- **Файлы:** Проверка безопасности

### Ограничения
- **Максимальная длина промпта:** 1000 символов
- **Запрещенные символы:** HTML, SQL инъекции
- **Rate Limiting:** 10 запросов в минуту

## Примеры Интеграции

### Python
```python
import requests

url = "https://api.replicate.com/v1/predictions"
headers = {
    "Authorization": "Token YOUR_API_TOKEN",
    "Content-Type": "application/json"
}

data = {
    "version": "763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30",
    "input": {
        "prompt": "50% red, 50% black"
    }
}

response = requests.post(url, headers=headers, json=data)
```

### JavaScript
```javascript
const response = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: {
        'Authorization': 'Token YOUR_API_TOKEN',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        version: '763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30',
        input: {
            prompt: '50% red, 50% black'
        }
    })
});
```

### cURL
```bash
curl -X POST \
  -H "Authorization: Token YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30",
    "input": {
        "prompt": "50% red, 50% black"
    }
  }' \
  https://api.replicate.com/v1/predictions
```

## Поддержка

### Документация
- **Основная:** Этот файл
- **Примеры:** `Examples.md`
- **Архитектура:** `Architecture.md`

### Контакты
- **Разработчик:** nauslava
- **Платформа:** Replicate
- **Статус:** Приватная модель
