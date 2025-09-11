# 🔌 API Integration Guide - Plitka Pro v4.5.05

**Версия:** v4.5.05  
**Дата:** 4 сентября 2025  
**Статус:** ✅ Production Ready  
**Новое:** Поддержка кодовых слов цветов (RED, BLUE, GRSGRN, etc.)

## 📋 Содержание

1. [Обзор API](#обзор-api)
2. [Аутентификация](#аутентификация)
3. [Базовые параметры](#базовые-параметры)
4. [Цветовая система](#цветовая-система)
5. [Примеры запросов](#примеры-запросов)
6. [Обработка ответов](#обработка-ответов)
7. [Ошибки и их решения](#ошибки-и-их-решения)
8. [Лучшие практики](#лучшие-практики)

---

## 🌐 Обзор API

### **Endpoint**
```
https://api.replicate.com/v1/predictions
```

### **Модель**
```
r8.im/nauslava/plitka-pro-project:v4.5.05
```

**Альтернативные форматы:**
```
r8.im/nauslava/plitka-pro-project:latest
r8.im/nauslava/plitka-pro-project@sha256:[новый_хеш_после_публикации]
```

**Примечание:** После публикации v4.5.05 будет доступен новый хеш. Используйте тег версии для простоты.

### **Тип запроса**
```
POST
```

### **Content-Type**
```
application/json
```

---

## 🔐 Аутентификация

### **API Token**
```bash
Authorization: Token YOUR_REPLICATE_API_TOKEN
```

### **Получение токена**
1. Зарегистрируйтесь на [replicate.com](https://replicate.com)
2. Перейдите в [Account Settings](https://replicate.com/account/api-tokens)
3. Создайте новый API токен
4. Сохраните токен в безопасном месте

### **Пример заголовков**
```http
POST /v1/predictions HTTP/1.1
Host: api.replicate.com
Authorization: Token r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Content-Type: application/json
```

---

## ⚙️ Базовые параметры

### **Обязательные параметры**

| Параметр | Тип | Описание | Пример |
|----------|-----|----------|---------|
| `version` | string | Версия модели | `"r8.im/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929"` |
| `input` | object | Входные параметры | См. ниже |

### **Входные параметры (input)**

| Параметр | Тип | Обязательный | По умолчанию | Описание |
|----------|-----|--------------|--------------|----------|
| `prompt` | string | ✅ | - | Промпт с цветами и описанием |
| `negative_prompt` | string | ❌ | `"blurry, low quality"` | Негативный промпт |
| `num_inference_steps` | integer | ❌ | `25` | Количество шагов генерации |
| `guidance_scale` | float | ❌ | `7.5` | Сила следования промпту |
| `seed` | integer | ❌ | `null` | Сид для воспроизводимости |
| `colormap` | string | ❌ | `"random"` | Тип паттерна colormap |
| `granule_size` | string | ❌ | `"medium"` | Размер гранул |

---

## 🎨 Цветовая система

### **Кодовые слова цветов (РЕКОМЕНДУЕТСЯ)**

| № | Название цвета (Рус) | Кодовое слово (ENG) | RGB | Hex |
|---|---------------------|---------------------|-----|-----|
| 1 | Бежевый | `BEIGE` | (245, 245, 220) | #F5F5DC |
| 2 | Бело-зеленый | `WHTGRN` | (240, 255, 240) | #F0FFF0 |
| 3 | Белый | `WHITE` | (255, 255, 255) | #FFFFFF |
| 4 | Бирюзовый | `TURQSE` | (64, 224, 208) | #40E0D0 |
| 5 | Голубой | `SKYBLUE` | (135, 206, 235) | #87CEEB |
| 6 | Желтый | `YELLOW` | (255, 255, 0) | #FFFF00 |
| 7 | Жемчужный | `PEARL` | (240, 248, 255) | #F0F8FF |
| 8 | Зеленая трава | `GRSGRN` | (34, 139, 34) | #228B22 |
| 9 | Зеленое яблоко | `GRNAPL` | (0, 128, 0) | #008000 |
| 10 | Изумрудный | `EMERALD` | (0, 128, 0) | #008000 |
| 11 | Коричневый | `BROWN` | (139, 69, 19) | #8B4513 |
| 12 | Красный | `RED` | (255, 0, 0) | #FF0000 |
| 13 | Лосось | `SALMON` | (250, 128, 114) | #FA8072 |
| 14 | Оранжевый | `ORANGE` | (255, 165, 0) | #FFA500 |
| 15 | Песочный | `SAND` | (244, 164, 96) | #F4A460 |
| 16 | Розовый | `PINK` | (255, 192, 203) | #FFC0CB |
| 17 | Салатовый | `LIMEGRN` | (50, 205, 50) | #32CD32 |
| 18 | Светло-зеленый | `LTGREEN` | (144, 238, 144) | #90EE90 |
| 19 | Светло-серый | `LTGRAY` | (192, 192, 192) | #C0C0C0 |
| 20 | Серый | `GRAY` | (128, 128, 128) | #808080 |
| 21 | Синий | `BLUE` | (0, 0, 255) | #0000FF |
| 22 | Сиреневый | `LILAC` | (200, 162, 200) | #C8A2C8 |
| 23 | Темно-зеленый | `DKGREEN` | (0, 100, 0) | #006400 |
| 24 | Темно-серый | `DKGRAY` | (64, 64, 64) | #404040 |
| 25 | Темно-синий | `DKBLUE` | (0, 0, 139) | #00008B |
| 26 | Терракот | `TERCOT` | (205, 92, 92) | #CD5C5C |
| 27 | Фиолетовый | `VIOLET` | (238, 130, 238) | #EE82EE |
| 28 | Хаки | `KHAKI` | (240, 230, 140) | #F0E68C |
| 29 | Чёрный | `BLACK` | (0, 0, 0) | #000000 |

### **Формат промпта**

#### **Структура промпта**
```
ohwx_rubber_tile <s0><s1> [ПРОЦЕНТ]% [КОДОВОЕ_СЛОВО], [ПРОЦЕНТ]% [КОДОВОЕ_СЛОВО], [ПАТТЕРН], [КАЧЕСТВО]
```

#### **Обязательные элементы**
- `ohwx_rubber_tile` - триггер модели
- `<s0><s1>` - токены Textual Inversion
- Цвета в формате `XX% КОДОВОЕ_СЛОВО` (рекомендуется)
- `rubber tile` - описание объекта (опционально)

#### **Примеры цветовых комбинаций с кодовыми словами**

**Монохром:**
```
ohwx_rubber_tile <s0><s1> 100% RED
ohwx_rubber_tile <s0><s1> 100% EMERALD
```

**Два цвета:**
```
ohwx_rubber_tile <s0><s1> 70% RED, 30% BLUE
ohwx_rubber_tile <s0><s1> 60% GRSGRN, 40% YELLOW
```

**Три цвета:**
```
ohwx_rubber_tile <s0><s1> 50% WHITE, 30% GRAY, 20% BLACK
ohwx_rubber_tile <s0><s1> 40% RED, 35% BLUE, 25% YELLOW
```

**Четыре цвета:**
```
ohwx_rubber_tile <s0><s1> 25% RED, 25% BLUE, 25% GRSGRN, 25% YELLOW
ohwx_rubber_tile <s0><s1> 20% WHITE, 20% GRAY, 20% BLACK, 20% RED, 20% BLUE
```

#### **Fallback формат (совместимость)**
```
ohwx_rubber_tile <s0><s1> 70% red, 30% blue  # Старый формат
ohwx_rubber_tile <s0><s1> 60% green, 40% yellow  # Старый формат
```

**Примечание:** Кодовые слова обеспечивают максимальную точность и надежность парсинга цветов.

---

## 📝 Примеры запросов

### **1. Базовый запрос (Python)**

```python
import requests
import json

# Конфигурация
API_TOKEN = "r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MODEL_VERSION = "r8.im/nauslava/plitka-pro-project:v4.5.05"
API_URL = "https://api.replicate.com/v1/predictions"

# Заголовки
headers = {
    "Authorization": f"Token {API_TOKEN}",
    "Content-Type": "application/json"
}

# Параметры запроса
data = {
    "version": MODEL_VERSION,
    "input": {
        "prompt": "ohwx_rubber_tile <s0><s1> 70% RED, 30% BLUE, grid pattern, high quality",
        "negative_prompt": "blurry, low quality, artifacts",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": 42,
        "colormap": "random",
        "granule_size": "medium"
    }
}

# Отправка запроса
response = requests.post(API_URL, headers=headers, json=data)

if response.status_code == 201:
    prediction = response.json()
    print(f"✅ Запрос создан: {prediction['id']}")
else:
    print(f"❌ Ошибка: {response.status_code} - {response.text}")
```

### **2. Асинхронный запрос (Python)**

```python
import asyncio
import aiohttp
import json

async def generate_rubber_tile():
    API_TOKEN = "r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    MODEL_VERSION = "r8.im/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929"
    
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "version": MODEL_VERSION,
        "input": {
            "prompt": "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, grid pattern",
            "num_inference_steps": 30,
            "guidance_scale": 8.0
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=data
        ) as response:
            if response.status == 201:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"API Error: {response.status} - {error_text}")

# Запуск
result = asyncio.run(generate_rubber_tile())
print(result)
```

### **3. JavaScript/Node.js**

```javascript
const axios = require('axios');

const API_TOKEN = 'r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx';
const MODEL_VERSION = 'r8.im/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929';

const generateRubberTile = async () => {
    try {
        const response = await axios.post(
            'https://api.replicate.com/v1/predictions',
            {
                version: MODEL_VERSION,
                input: {
                    prompt: 'ohwx_rubber_tile <s0><s1> 70% red, 30% blue rubber tile, random pattern, high quality',
                    negative_prompt: 'blurry, low quality, artifacts',
                    num_inference_steps: 25,
                    guidance_scale: 7.5,
                    seed: 123,
                    colormap: 'random',
                    granule_size: 'medium'
                }
            },
            {
                headers: {
                    'Authorization': `Token ${API_TOKEN}`,
                    'Content-Type': 'application/json'
                }
            }
        );
        
        console.log('✅ Запрос создан:', response.data.id);
        return response.data;
    } catch (error) {
        console.error('❌ Ошибка:', error.response?.data || error.message);
        throw error;
    }
};

generateRubberTile();
```

### **4. cURL**

```bash
curl -X POST \
  https://api.replicate.com/v1/predictions \
  -H "Authorization: Token r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "r8.im/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929",
    "input": {
      "prompt": "ohwx_rubber_tile <s0><s1> 60% red, 40% blue rubber tile, random pattern, high quality",
      "negative_prompt": "blurry, low quality, artifacts",
      "num_inference_steps": 25,
      "guidance_scale": 7.5,
      "seed": 42,
      "colormap": "random",
      "granule_size": "medium"
    }
  }'
```

---

## 📊 Обработка ответов

### **Структура ответа**

```json
{
  "id": "prediction_id",
  "status": "starting|processing|succeeded|failed|canceled",
  "created_at": "2025-09-04T12:00:00Z",
  "started_at": "2025-09-04T12:00:05Z",
  "completed_at": "2025-09-04T12:01:30Z",
  "input": {
    "prompt": "ohwx_rubber_tile <s0><s1> 60% red, 40% blue rubber tile, random pattern, high quality",
    "negative_prompt": "blurry, low quality, artifacts",
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42,
    "colormap": "random",
    "granule_size": "medium"
  },
  "output": [
    "https://replicate.delivery/pbxt/.../final.png",
    "https://replicate.delivery/pbxt/.../colormap.png"
  ],
  "error": null,
  "logs": "=== GENERATION START: 2025-09-04 12:00:00 ===\n...",
  "metrics": {
    "predict_time": 85.2
  }
}
```

### **Мониторинг статуса**

```python
import time
import requests

def wait_for_completion(prediction_id, api_token, timeout=300):
    """Ожидание завершения генерации"""
    
    headers = {"Authorization": f"Token {api_token}"}
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            prediction = response.json()
            status = prediction["status"]
            
            print(f"Статус: {status}")
            
            if status == "succeeded":
                return prediction
            elif status == "failed":
                raise Exception(f"Генерация failed: {prediction.get('error', 'Unknown error')}")
            elif status == "canceled":
                raise Exception("Генерация отменена")
            
            time.sleep(2)  # Проверяем каждые 2 секунды
        else:
            raise Exception(f"Ошибка API: {response.status_code}")
    
    raise Exception("Timeout: генерация не завершилась в отведенное время")

# Использование
try:
    result = wait_for_completion(prediction_id, API_TOKEN)
    print("✅ Генерация завершена!")
    print(f"Изображения: {result['output']}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
```

---

## ⚠️ Ошибки и их решения

### **HTTP коды ошибок**

| Код | Описание | Решение |
|-----|----------|---------|
| 400 | Bad Request | Проверьте формат запроса и параметры |
| 401 | Unauthorized | Проверьте API токен |
| 402 | Payment Required | Пополните баланс аккаунта |
| 403 | Forbidden | Проверьте права доступа |
| 404 | Not Found | Проверьте версию модели |
| 429 | Too Many Requests | Уменьшите частоту запросов |
| 500 | Internal Server Error | Повторите запрос позже |

### **Частые ошибки**

#### **1. Неверный промпт**
```json
{
  "error": "Invalid prompt format"
}
```
**Решение:** Убедитесь, что промпт содержит `ohwx_rubber_tile <s0><s1>` и правильные цвета.

#### **2. Неподдерживаемый цвет**
```json
{
  "error": "Color 'green' not found in color table"
}
```
**Решение:** Используйте цвета из таблицы выше (например, `grsgrn` вместо `green`).

#### **3. Неверные пропорции**
```json
{
  "error": "Color proportions must sum to 100%"
}
```
**Решение:** Убедитесь, что сумма процентов цветов равна 100%.

#### **4. Неверная версия модели**
```json
{
  "title": "Invalid version or not permitted",
  "detail": "The specified version does not exist (or perhaps you don't have permission to use it?)",
  "status": 422
}
```
**Решение:** Используйте полный хеш версии:
```python
# Вместо:
MODEL_VERSION = "r8.im/nauslava/plitka-pro-project:v4.5.03"

# Используйте:
MODEL_VERSION = "r8.im/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929"
```

### **Обработка ошибок (Python)**

```python
def handle_api_error(response):
    """Обработка ошибок API"""
    
    if response.status_code == 200:
        return response.json()
    
    error_data = response.json() if response.content else {}
    error_message = error_data.get('detail', f'HTTP {response.status_code}')
    
    if response.status_code == 401:
        raise Exception("❌ Неверный API токен")
    elif response.status_code == 402:
        raise Exception("❌ Недостаточно средств на балансе")
    elif response.status_code == 429:
        raise Exception("❌ Слишком много запросов. Попробуйте позже")
    elif response.status_code == 400:
        raise Exception(f"❌ Неверный запрос: {error_message}")
    else:
        raise Exception(f"❌ Ошибка API: {error_message}")
```

---

## 🎯 Лучшие практики

### **1. Оптимизация параметров**

#### **Для быстрой генерации:**
```json
{
  "num_inference_steps": 20,
  "guidance_scale": 7.0,
  "colormap": "random"
}
```

#### **Для высокого качества:**
```json
{
  "num_inference_steps": 40,
  "guidance_scale": 9.0,
  "colormap": "granular"
}
```

#### **Для воспроизводимости:**
```json
{
  "seed": 42,
  "num_inference_steps": 25,
  "guidance_scale": 7.5
}
```

### **2. Управление ресурсами**

```python
import asyncio
import aiohttp
from asyncio import Semaphore

class RubberTileGenerator:
    def __init__(self, api_token, max_concurrent=3):
        self.api_token = api_token
        self.semaphore = Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt, **kwargs):
        async with self.semaphore:  # Ограничиваем количество одновременных запросов
            return await self._make_request(prompt, **kwargs)
    
    async def _make_request(self, prompt, **kwargs):
        # Реализация запроса
        pass

# Использование
async def main():
    async with RubberTileGenerator(API_TOKEN, max_concurrent=2) as generator:
        tasks = [
            generator.generate("ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile"),
            generator.generate("ohwx_rubber_tile <s0><s1> 30% red, 30% blue, 40% dkgreen rubber tile"),
            generator.generate("ohwx_rubber_tile <s0><s1> 100% yellow rubber tile")
        ]
        
        results = await asyncio.gather(*tasks)
        return results
```

### **3. Кэширование результатов**

```python
import hashlib
import json
import os
from functools import lru_cache

class CachedGenerator:
    def __init__(self, api_token, cache_dir="./cache"):
        self.api_token = api_token
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, input_params):
        """Создание ключа кэша на основе параметров"""
        key_string = json.dumps(input_params, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key):
        """Загрузка из кэша"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, cache_key, result):
        """Сохранение в кэш"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        with open(cache_file, 'w') as f:
            json.dump(result, f)
    
    def generate(self, **input_params):
        """Генерация с кэшированием"""
        cache_key = self._get_cache_key(input_params)
        
        # Проверяем кэш
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            print("✅ Результат загружен из кэша")
            return cached_result
        
        # Генерируем новый результат
        result = self._make_api_request(input_params)
        
        # Сохраняем в кэш
        self._save_to_cache(cache_key, result)
        print("✅ Результат сохранен в кэш")
        
        return result
```

### **4. Валидация входных данных**

```python
import re
from typing import List, Dict, Any

class PromptValidator:
    """Валидатор промптов для Plitka Pro"""
    
    VALID_COLORS = {
        'BEIGE', 'WHTGRN', 'WHITE', 'TURQSE', 'SKYBLUE', 'YELLOW', 'PEARL',
        'GRSGRN', 'GRNAPL', 'EMERALD', 'BROWN', 'RED', 'SALMON', 'ORANGE',
        'SAND', 'PINK', 'LIMEGRN', 'LTGREEN', 'LTGRAY', 'GRAY', 'BLUE',
        'LILAC', 'DKGREEN', 'DKGRAY', 'DKBLUE', 'TERCOT', 'VIOLET', 'KHAKI', 'BLACK'
    }
    
    VALID_PATTERNS = {'random', 'grid', 'radial', 'granular'}
    VALID_GRANULE_SIZES = {'small', 'medium', 'large'}
    
    @classmethod
    def validate_prompt(cls, prompt: str) -> Dict[str, Any]:
        """Валидация промпта"""
        errors = []
        warnings = []
        
        # Проверка обязательных элементов
        if 'ohwx_rubber_tile' not in prompt:
            errors.append("Промпт должен содержать 'ohwx_rubber_tile'")
        
        if '<s0><s1>' not in prompt:
            errors.append("Промпт должен содержать '<s0><s1>'")
        
        if 'rubber tile' not in prompt:
            errors.append("Промпт должен содержать 'rubber tile'")
        
        # Извлечение и валидация цветов
        color_pattern = r'(\d+)%\s+([A-Z]+)'
        color_matches = re.findall(color_pattern, prompt.upper())
        
        if not color_matches:
            errors.append("Промпт должен содержать цвета в формате 'XX% COLOR'")
        else:
            total_percentage = 0
            for percentage, color in color_matches:
                total_percentage += int(percentage)
                
                if color not in cls.VALID_COLORS:
                    errors.append(f"Неподдерживаемый цвет: {color}")
            
            if total_percentage != 100:
                errors.append(f"Сумма процентов должна быть 100%, получено: {total_percentage}%")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'colors_found': [color for _, color in color_matches],
            'total_percentage': sum(int(p) for p, _ in color_matches)
        }
    
    @classmethod
    def validate_parameters(cls, **params) -> Dict[str, Any]:
        """Валидация параметров"""
        errors = []
        
        # Проверка num_inference_steps
        if 'num_inference_steps' in params:
            steps = params['num_inference_steps']
            if not isinstance(steps, int) or steps < 10 or steps > 100:
                errors.append("num_inference_steps должен быть целым числом от 10 до 100")
        
        # Проверка guidance_scale
        if 'guidance_scale' in params:
            scale = params['guidance_scale']
            if not isinstance(scale, (int, float)) or scale < 1.0 or scale > 20.0:
                errors.append("guidance_scale должен быть числом от 1.0 до 20.0")
        
        # Проверка colormap
        if 'colormap' in params:
            if params['colormap'] not in cls.VALID_PATTERNS:
                errors.append(f"colormap должен быть одним из: {', '.join(cls.VALID_PATTERNS)}")
        
        # Проверка granule_size
        if 'granule_size' in params:
            if params['granule_size'] not in cls.VALID_GRANULE_SIZES:
                errors.append(f"granule_size должен быть одним из: {', '.join(cls.VALID_GRANULE_SIZES)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

# Использование
validator = PromptValidator()

# Валидация промпта
prompt = "ohwx_rubber_tile <s0><s1> 60% red, 40% blue rubber tile, random pattern"
result = validator.validate_prompt(prompt)

if result['valid']:
    print("✅ Промпт валиден")
else:
    print("❌ Ошибки в промпте:", result['errors'])

# Валидация параметров
params = {
    'num_inference_steps': 25,
    'guidance_scale': 7.5,
    'colormap': 'random',
    'granule_size': 'medium'
}
param_result = validator.validate_parameters(**params)

if param_result['valid']:
    print("✅ Параметры валидны")
else:
    print("❌ Ошибки в параметрах:", param_result['errors'])
```

---

## 📚 Дополнительные ресурсы

### **Полезные ссылки**
- [Replicate API Documentation](https://replicate.com/docs)
- [Plitka Pro Model Page](https://replicate.com/nauslava/plitka-pro-project)
- [GitHub Repository](https://github.com/nauslava/plitka-pro-project)

### **Поддержка**
- **Документация:** `docs/` папка в проекте
- **GUI тестер:** `python3 scripts/replicate_gui.py`
- **Примеры:** `scripts/test_inputs_*.json`

### **Версии модели**
- **v4.5.03** (текущая) - Критические исправления + GUI улучшения
- **v4.5.02** - Исправления архитектуры + colormap паттерны
- **v4.5.01** - Критические исправления архитектуры

---

**Руководство подготовлено:** 4 сентября 2025  
**Версия модели:** v4.5.03  
**Статус:** ✅ Production Ready
