# Развертывание Plitka Pro на Replicate

## Обход Docker Hub (решение проблем с авторизацией)

### Проблема
- Docker Hub персональный план имеет ограничения
- Ошибка `authentication required - access token has insufficient scopes`
- Не удается запушить образ в Docker Hub

### Решение: Прямое развертывание через Replicate

#### Шаг 1: Подготовка кода
```bash
# Убедитесь, что код готов к развертыванию
git add .
git commit -m "Ready for Replicate deployment v4.5.10"
```

#### Шаг 2: Создание модели на Replicate
1. **Зайдите на [replicate.com](https://replicate.com)**
2. **Войдите в аккаунт `nauslava`**
3. **Нажмите "Create" → "Model"**

#### Шаг 3: Настройка модели
1. **Название**: `plitka-pro`
2. **Описание**: `AI model for generating rubber tile images from colored crumbs using SDXL + ControlNet + LoRA`
3. **Видимость**: `Public`

#### Шаг 4: Загрузка кода
1. **Загрузите код напрямую** через веб-интерфейс Replicate (предпочтительно)
2. GitHub-поток не используется без явного разрешения владельца проекта

#### Шаг 5: Конфигурация модели
1. **Укажите `predict.py` как основной файл**
2. **Настройте зависимости** (requirements.txt)
3. **Укажите Python версию**: 3.11
4. **Настройте GPU**: CUDA 12.4

#### Шаг 6: Загрузка весов
1. **SDXL Base**: `stabilityai/stable-diffusion-xl-base-1.0`
2. **ControlNet Canny**: `diffusers/controlnet-canny-sdxl-1.0`
3. **LoRA**: `nauslava/plitka-pro-lora`
4. **Textual Inversion**: `nauslava/plitka-pro-ti`

### Примечание о GitHub
В соответствии с правилами проекта, публикация в GitHub и связанные реестры запрещены без явного разрешения владельца.

### Проверка развертывания

После успешного развертывания:

1. **Проверьте URL модели**: `r8.im/nauslava/plitka-pro-project:v4.4.43`
2. **Протестируйте API** через веб-интерфейс
3. **Проверьте логи** на предмет ошибок
4. **Убедитесь в корректности генерации** изображений

### Поддержка

Если возникнут проблемы:
1. **Проверьте логи** в Replicate
2. **Убедитесь в корректности** всех зависимостей
3. **Проверьте размер** загружаемых весов
4. **Обратитесь к документации** Replicate

---

**Примечание**: Этот метод обходит ограничения Docker Hub и позволяет развернуть модель напрямую на Replicate.

**Версия**: v4.4.52
