# 🧱 Plitka Pro - Генератор фотorealistic резиновой плитки

**Plitka Pro** - это MLOps-система для генерации фотorealistic изображений резиновой плитки с использованием Stable Diffusion XL (SDXL) + ControlNet + LoRA + Textual Inversion.

## 🚀 Быстрый старт

### Веб-интерфейс (рекомендуется)
**https://replicate.com/nauslava/plitka-pro**

### Минимальный пример
```json
{"params_json": "{\"colors\":[{\"name\":\"BLACK\",\"proportion\":0.7},{\"name\":\"RED\",\"proportion\":0.3}],\"angle\":0,\"quality\":\"preview\"}"}
```

## 📚 Документация

- **[🚀 Быстрый старт](docs/QuickStart.md)** - минимальные примеры для начала работы
- **[📖 Полное руководство](docs/WebInterfaceExamples.md)** - подробные примеры и описание всех параметров
- **[🎨 Интерактивные примеры](docs/web_examples.html)** - HTML страница с копируемыми примерами
- **[🏗️ Архитектура проекта](docs/Project.md)** - техническая документация
- **[📋 Трекер задач](docs/TaskTracker.md)** - статус разработки
- **[📝 Журнал изменений](docs/Changelog.md)** - история версий
- **[🚀 Развертывание на Replicate](docs/ReplicateDeployment.md)** - инструкция по обходу Docker Hub

## ✨ Возможности

- **Точный контроль цветов** - математически точные пропорции цветовых областей
- **Множественные углы укладки** - 0°, 45°, 90°, 135° с автоматическим выбором ControlNet
- **Три уровня качества** - preview, standard, high
- **Воспроизводимость** - фиксированные seed для повторяемых результатов
- **Быстрая генерация** - параллельная генерация preview и final изображений

## 🎨 Поддерживаемые цвета

### По имени
- `BLACK`, `WHITE`, `RED`, `GREEN`, `BLUE`, `YELLOW`, `GRAY`, `BROWN`

### По HEX коду
- Любые HEX коды: `{"hex": "#FF0000", "proportion": 0.5}`

## ⚙️ Параметры

### Обязательные
- `colors` - массив цветов с пропорциями
- `angle` - угол укладки (0, 45, 90, 135)
- `quality` - качество генерации (preview, standard, high)

### Опциональные
- `seed` - для воспроизводимости
- `overrides` - кастомные настройки (guidance_scale, negative_prompt)

## 📤 Выходные файлы

1. **preview.png** - быстрое превью (512x512)
2. **final.png** - финальное изображение (1024x1024)
3. **colormap.png** - карта цветов для отладки

## 🔧 Технические детали

- **Базовая модель**: Stable Diffusion XL 1.0
- **ControlNet**: Canny (0°/90°), Lineart (45°/135°)
- **LoRA**: Специализированные веса для резиновой плитки
- **Textual Inversion**: Двухтокеновая система для улучшения качества
- **Оптимизации**: VAE slicing/tiling, CUDA optimizations

## 🚀 Развертывание

### Локальная разработка
```bash
# Клонирование
git clone https://github.com/NauSlava/lora-training.git
cd lora-training/plitka-pro

# Установка зависимостей
pip install -r requirements.txt

# Локальный тест
cog predict -i params_json='{"colors":[...], "angle":45}'
```

### Replicate
```bash
# Публикация через GitHub Container Registry
cog push ghcr.io/nauslava/plitka-pro:v4.1.4
```

## 📊 Статус проекта

- **Версия**: v4.1.4
- **Статус**: ✅ Готов к продакшену
- **Последние исправления**: Устранены проблемы с генерацией, отключен torch.compile

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License - см. [LICENSE](../LICENSE) файл в корне репозитория.

---

**Примечание**: Этот проект является частью репозитория [lora-training](https://github.com/NauSlava/lora-training) и использует общую инфраструктуру для LoRA обучения и развертывания.
