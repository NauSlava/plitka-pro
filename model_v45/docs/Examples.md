# Примеры Использования - Модель "nauslava/rubber-tile-lora-v45"

## Обзор Примеров

Эта страница содержит **практические примеры** использования модели v45 для генерации изображений резиновой плитки с использованием НАШЕЙ обученной модели.

## Базовые Примеры

### 1. Одноцветная Плитка

#### Простой запрос
```json
{
  "prompt": "100% white"
}
```

#### Результат
- **Изображение:** Белая резиновая плитка
- **Качество:** Максимальное
- **Стиль:** Фотореалистичный

#### Внутренний промпт
```
"ohwx_rubber_tile <s0><s1>, 100% white, photorealistic rubber tile, high quality, detailed texture, professional photography, sharp focus"
```

### 2. Двухцветная Плитка

#### Запрос
```json
{
  "prompt": "50% red, 50% black"
}
```

#### Результат
- **Изображение:** Красно-черная плитка
- **Пропорции:** 50/50
- **Качество:** Максимальное

#### Внутренний промпт
```
"ohwx_rubber_tile <s0><s1>, 50% red, 50% black, photorealistic rubber tile, high quality, detailed texture, professional photography, sharp focus"
```

### 3. Трехцветная Плитка

#### Запрос
```json
{
  "prompt": "40% blue, 35% green, 25% yellow"
}
```

#### Результат
- **Изображение:** Сине-зелено-желтая плитка
- **Пропорции:** 40/35/25
- **Качество:** Максимальное

## Продвинутые Примеры

### 4. С Негативным Промптом

#### Запрос
```json
{
  "prompt": "60% purple, 40% pink",
  "negative_prompt": "blurry, low quality, deformed, text, watermark"
}
```

#### Результат
- **Изображение:** Фиолетово-розовая плитка
- **Качество:** Улучшенное (без дефектов)
- **Стиль:** Чистый фотореалистичный

### 5. С Фиксированным Сидом

#### Запрос
```json
{
  "prompt": "70% orange, 30% brown",
  "seed": 12345
}
```

#### Результат
- **Изображение:** Оранжево-коричневая плитка
- **Воспроизводимость:** 100% при том же сиде
- **Качество:** Максимальное

### 6. Сложная Цветовая Схема

#### Запрос
```json
{
  "prompt": "30% navy, 25% teal, 20% cyan, 15% sky blue, 10% light blue"
}
```

#### Результат
- **Изображение:** Многоцветная синяя плитка
- **Пропорции:** 30/25/20/15/10
- **Качество:** Максимальное

## Примеры Интеграции

### 7. Python API

#### Базовый запрос
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
result = response.json()
print(f"Статус: {result['status']}")
```

#### Запрос с параметрами
```python
data = {
    "version": "763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30",
    "input": {
        "prompt": "60% blue, 40% green",
        "negative_prompt": "blurry, low quality",
        "seed": 98765
    }
}
```

### 8. JavaScript/Node.js

#### Базовый запрос
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

const result = await response.json();
console.log(`Статус: ${result.status}`);
```

#### Запрос с параметрами
```javascript
const data = {
    version: '763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30',
    input: {
        prompt: '60% blue, 40% green',
        negative_prompt: 'blurry, low quality',
        seed: 98765
    }
};
```

### 9. cURL

#### Базовый запрос
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

#### Запрос с параметрами
```bash
curl -X POST \
  -H "Authorization: Token YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30",
    "input": {
        "prompt": "60% blue, 40% green",
        "negative_prompt": "blurry, low quality",
        "seed": 98765
    }
  }' \
  https://api.replicate.com/v1/predictions
```

## Примеры Промптов

### Цветовые Комбинации

#### Монохромные
- `"100% white"` - Белая плитка
- `"100% black"` - Черная плитка
- `"100% red"` - Красная плитка
- `"100% blue"` - Синяя плитка

#### Двухцветные
- `"50% red, 50% black"` - Красно-черная
- `"60% blue, 40% green"` - Сине-зеленая
- `"70% yellow, 30% orange"` - Желто-оранжевая
- `"80% purple, 20% pink"` - Фиолетово-розовая

#### Трехцветные
- `"40% red, 35% blue, 25% green"` - Красно-сине-зеленая
- `"50% yellow, 30% orange, 20% red"` - Желто-оранжево-красная
- `"45% blue, 35% purple, 20% cyan"` - Сине-фиолетово-голубая

#### Многоцветные
- `"30% red, 25% blue, 20% green, 15% yellow, 10% purple"`
- `"25% navy, 20% teal, 20% cyan, 20% sky blue, 15% light blue"`

### Специальные Эффекты

#### Металлические
- `"60% silver, 40% gray"` - Серебристо-серая
- `"70% gold, 30% yellow"` - Золотисто-желтая
- `"50% bronze, 50% brown"` - Бронзово-коричневая

#### Пастельные
- `"50% pastel pink, 30% pastel blue, 20% pastel yellow"`
- `"60% pastel green, 40% pastel purple"`

## Примеры Обработки Ошибок

### 10. Обработка Ошибок в Python

```python
import requests
import time

def generate_tile(prompt, max_retries=3):
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": "Token YOUR_API_TOKEN",
        "Content-Type": "application/json"
    }
    
    data = {
        "version": "763970a6702524ff048dd130709e255581912c8a4ab516db4520788f70b84e30",
        "input": {"prompt": prompt}
    }
    
    for attempt in range(max_retries):
        try:
            # Создание предсказания
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            prediction = response.json()
            
            # Ожидание завершения
            while prediction['status'] not in ['succeeded', 'failed']:
                time.sleep(1)
                response = requests.get(prediction['urls']['get'], headers=headers)
                prediction = response.json()
            
            if prediction['status'] == 'succeeded':
                return prediction['output']
            else:
                print(f"Ошибка генерации: {prediction.get('error', 'Неизвестная ошибка')}")
                
        except requests.exceptions.RequestException as e:
            print(f"Ошибка запроса (попытка {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
            continue
    
    raise Exception("Не удалось сгенерировать изображение после всех попыток")

# Использование
try:
    result = generate_tile("50% red, 50% black")
    print(f"Изображение сгенерировано: {result}")
except Exception as e:
    print(f"Ошибка: {e}")
```

## Примеры Тестирования

### 11. Локальное Тестирование

#### Тест базовой функциональности
```bash
# Простой тест
cog predict -i prompt="100% white"

# Тест с параметрами
cog predict -i prompt="50% red, 50% black" -i seed=12345

# Тест с негативным промптом
cog predict -i prompt="60% blue, 40% green" -i negative_prompt="blurry, low quality"
```

#### Тест производительности
```bash
# Тест времени генерации
time cog predict -i prompt="100% white"

# Тест использования памяти
cog predict -i prompt="100% white" 2>&1 | grep -i memory
```

## Заключение

Эти примеры демонстрируют **широкие возможности** модели v45 и **простоту интеграции** в различные приложения. Ключевым фактором успеха является **использование НАШЕЙ обученной модели**, которая обеспечивает максимальное качество генерации.

### Рекомендации по Использованию
1. **Начните с простых промптов** для понимания возможностей
2. **Используйте негативные промпты** для улучшения качества
3. **Применяйте фиксированные сиды** для воспроизводимости
4. **Тестируйте локально** перед интеграцией
5. **Обрабатывайте ошибки** для надежности приложения
