# 🚀 Quick API Start - Plitka Pro v4.5.03

**Быстрый старт для интеграции с API модели Plitka Pro**

## ⚡ Быстрый старт

### 1. Получите API токен
```bash
# Зарегистрируйтесь на replicate.com
# Создайте токен в Account Settings
export REPLICATE_API_TOKEN="r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. Базовый запрос (Python)
```python
import requests

# Конфигурация
API_TOKEN = "r8_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MODEL_VERSION = "r8.im/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929"

# Запрос
response = requests.post(
    "https://api.replicate.com/v1/predictions",
    headers={"Authorization": f"Token {API_TOKEN}"},
    json={
        "version": MODEL_VERSION,
        "input": {
            "prompt": "ohwx_rubber_tile <s0><s1> 60% red, 40% blue rubber tile, random pattern, high quality",
            "num_inference_steps": 25,
            "guidance_scale": 7.5
        }
    }
)

if response.status_code == 201:
    prediction = response.json()
    print(f"✅ Запрос создан: {prediction['id']}")
else:
    print(f"❌ Ошибка: {response.text}")
```

### 3. Мониторинг результата
```python
import time

def wait_for_result(prediction_id, api_token):
    while True:
        response = requests.get(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers={"Authorization": f"Token {api_token}"}
        )
        
        prediction = response.json()
        status = prediction["status"]
        
        if status == "succeeded":
            return prediction["output"]  # Список URL изображений
        elif status == "failed":
            raise Exception(f"Ошибка: {prediction.get('error')}")
        
        time.sleep(2)

# Использование
images = wait_for_result(prediction["id"], API_TOKEN)
print(f"Изображения: {images}")
```

## 🎨 Цвета (краткая таблица)

| Цвет | Код | Цвет | Код |
|------|-----|------|-----|
| Красный | RED | Синий | BLUE |
| Зеленая трава | GRSGRN | Желтый | YELLOW |
| Темно-зеленый | DKGREEN | Белый | WHITE |
| Черный | BLACK | Серый | GRAY |
| Оранжевый | ORANGE | Фиолетовый | VIOLET |

**Полная таблица:** [API_Integration_Guide_v4.5.03.md](API_Integration_Guide_v4.5.03.md)

## 📝 Примеры промптов

```python
# Монохром
"ohwx_rubber_tile <s0><s1> 100% red rubber tile, high quality"

# Два цвета
"ohwx_rubber_tile <s0><s1> 60% red, 40% blue rubber tile, random pattern"

# Три цвета
"ohwx_rubber_tile <s0><s1> 40% red, 30% blue, 30% dkgreen rubber tile, grid pattern"

# Четыре цвета
"ohwx_rubber_tile <s0><s1> 30% red, 30% blue, 20% dkgreen, 20% yellow rubber tile, granular pattern"
```

## ⚙️ Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `num_inference_steps` | 25 | Шаги генерации (10-100) |
| `guidance_scale` | 7.5 | Сила следования промпту (1.0-20.0) |
| `colormap` | "random" | Паттерн: random, grid, radial, granular |
| `granule_size` | "medium" | Размер: small, medium, large |
| `seed` | null | Сид для воспроизводимости |

## 🔗 Полезные ссылки

- **Полное руководство:** [API_Integration_Guide_v4.5.03.md](API_Integration_Guide_v4.5.03.md)
- **Модель:** https://replicate.com/nauslava/plitka-pro-project@sha256:04b491be4545f7099a84688ac19c1fdf387b01c9e234593bd9f707abaf658929
- **Replicate API:** https://replicate.com/docs

---

**Версия:** v4.5.03  
**Статус:** ✅ Production Ready
