# Примеры запросов для Web интерфейса Replicate

## Правильный формат запроса

**ВАЖНО:** Используйте простой JSON без двойного экранирования:

```json
{
  "params_json": "{\"colors\":[{\"name\":\"BLACK\",\"proportion\":0.7},{\"name\":\"RED\",\"proportion\":0.3}],\"angle\":0,\"quality\":\"standard\",\"seed\":123456}"
}
```

## Примеры запросов

### 1. Черно-красная плитка (70% черного, 30% красного)
```json
{
  "params_json": "{\"colors\":[{\"name\":\"BLACK\",\"proportion\":0.7},{\"name\":\"RED\",\"proportion\":0.3}],\"angle\":0,\"quality\":\"standard\",\"seed\":123456}"
}
```

### 2. Зелено-желтая плитка (60% зеленого, 40% желтого)
```json
{
  "params_json": "{\"colors\":[{\"name\":\"GREEN\",\"proportion\":0.6},{\"name\":\"YELLOW\",\"proportion\":0.4}],\"angle\":45,\"quality\":\"high\",\"seed\":42}"
}
```

### 3. Синяя плитка (100% синего)
```json
{
  "params_json": "{\"colors\":[{\"name\":\"BLUE\",\"proportion\":1.0}],\"angle\":90,\"quality\":\"standard\",\"seed\":999}"
}
```

### 4. Трехцветная плитка (40% белого, 35% серого, 25% коричневого)
```json
{
  "params_json": "{\"colors\":[{\"name\":\"WHITE\",\"proportion\":0.4},{\"name\":\"GRAY\",\"proportion\":0.35},{\"name\":\"BROWN\",\"proportion\":0.25}],\"angle\":30,\"quality\":\"high\",\"seed\":777}"
}
```

### 5. Черно-белая плитка (15% черного, 85% белого) - как в референсе
```json
{
  "params_json": "{\"colors\":[{\"name\":\"BLACK\",\"proportion\":0.15},{\"name\":\"WHITE\",\"proportion\":0.85}],\"angle\":0,\"quality\":\"standard\",\"seed\":123456}"
}
```

### 6. Красная плитка с высоким качеством
```json
{
  "params_json": "{\"colors\":[{\"name\":\"RED\",\"proportion\":1.0}],\"angle\":0,\"quality\":\"high\",\"seed\":42,\"overrides\":{\"guidance_scale\":8.0}}"
}
```

### 7. Минимальный запрос (случайные цвета)
```json
{
  "params_json": "{\"colors\":[{\"name\":\"BLUE\",\"proportion\":0.6},{\"name\":\"YELLOW\",\"proportion\":0.4}],\"angle\":45,\"quality\":\"high\",\"seed\":42,\"overrides\":{\"guidance_scale\":8.0}}"
}
```

## Параметры

### colors
Массив объектов с цветами и их пропорциями:
- `name`: название цвета (BLACK, WHITE, RED, GREEN, BLUE, YELLOW, GRAY, BROWN)
- `proportion`: доля цвета от 0.0 до 1.0

### angle
Угол поворота текстуры (0-360 градусов)

### quality
- `preview`: быстрое предварительное изображение
- `standard`: стандартное качество (по умолчанию)
- `high`: высокое качество

### seed
Семя для воспроизводимости результатов (-1 для случайного)

### overrides (опционально)
Дополнительные параметры:
- `guidance_scale`: сила следования промпту (по умолчанию 7.5)
- `negative_prompt`: негативный промпт

## Поддерживаемые цвета

- **BLACK** - черный
- **WHITE** - белый  
- **RED** - красный
- **GREEN** - зеленый
- **BLUE** - синий
- **YELLOW** - желтый
- **GRAY** - серый
- **BROWN** - коричневый

## Советы по использованию

1. **Пропорции цветов** должны в сумме давать 1.0 или меньше
2. **Seed** - используйте одинаковые значения для воспроизводимости
3. **Angle** - влияет на направление текстуры
4. **Quality** - "high" дает лучшее качество, но дольше генерирует
5. **Guidance_scale** - значения 7-9 обычно дают хорошие результаты
