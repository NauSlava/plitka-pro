# 🚀 Быстрый старт с Plitka Pro

## Ссылка на модель
**https://replicate.com/nauslava/plitka-pro**

## Минимальный рабочий пример

Скопируйте этот JSON в поле `params_json`:

```json
{"params_json": "{\"colors\":[{\"name\":\"BLACK\",\"proportion\":0.7},{\"name\":\"RED\",\"proportion\":0.3}],\"angle\":0,\"quality\":\"preview\"}"}
```

## Что вы получите
- **preview.png** - быстрое превью (512x512)
- **final.png** - финальное изображение (1024x1024)  
- **colormap.png** - карта цветов для отладки

## Основные параметры

### Цвета
```json
"colors": [
  {"name": "BLACK", "proportion": 0.7},
  {"name": "RED", "proportion": 0.3}
]
```

### Углы укладки
- `0` - горизонтальная
- `45` - диагональная  
- `90` - вертикальная
- `135` - диагональная

### Качество
- `"preview"` - быстро (16/24 steps)
- `"standard"` - нормально (20/30 steps)
- `"high"` - качественно (24/40 steps)

## Поддерживаемые цвета
- `BLACK`, `WHITE`, `RED`, `GREEN`, `BLUE`, `YELLOW`, `GRAY`, `BROWN`
- Или HEX коды: `{"hex": "#FF0000", "proportion": 0.5}`

## Примеры для копирования

### Быстрый тест
```json
{"params_json": "{\"colors\":[{\"name\":\"BLUE\",\"proportion\":0.6},{\"name\":\"YELLOW\",\"proportion\":0.4}],\"angle\":45,\"quality\":\"preview\",\"seed\":42}"}
```

### Высокое качество
```json
{"params_json": "{\"colors\":[{\"name\":\"GREEN\",\"proportion\":0.8},{\"name\":\"GRAY\",\"proportion\":0.2}],\"angle\":90,\"quality\":\"high\",\"seed\":999}"}
```

### Многоцветная плитка
```json
{"params_json": "{\"colors\":[{\"name\":\"BLACK\",\"proportion\":0.4},{\"name\":\"RED\",\"proportion\":0.3},{\"name\":\"BLUE\",\"proportion\":0.2},{\"name\":\"YELLOW\",\"proportion\":0.1}],\"angle\":0,\"quality\":\"standard\",\"seed\":555}"}
```

## Советы
1. Начните с `quality: "preview"` для быстрого тестирования
2. Используйте `seed` для воспроизводимости результатов
3. Пропорции автоматически нормализуются
4. Для диагональных углов (45°, 135°) используется Lineart ControlNet
5. Для горизонтальных/вертикальных углов (0°, 90°) используется Canny ControlNet

## Подробная документация
См. `docs/WebInterfaceExamples.md` для полного руководства.
