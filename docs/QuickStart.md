# 🚀 Быстрый старт с Plitka Pro

## Ссылка на модель
**https://replicate.com/nauslava/plitka-pro-project:v4.3.60**

## Минимальный рабочий пример

Скопируйте этот JSON в поле `params_json`:

```json
{"params_json": "{\"colors\":[{\"name\":\"black\",\"proportion\":0.7},{\"name\":\"red\",\"proportion\":0.3}],\"angle\":0,\"quality\":\"preview\"}"}
```

## Что вы получите
- **preview.png** - быстрое превью (512x512)
- **final.png** - финальное изображение (1024x1024)  
- **colormap.png** - карта цветов для отладки

## Основные параметры

### Цвета
```json
"colors": [
  {"name": "black", "proportion": 0.7},
  {"name": "red", "proportion": 0.3}
]
```

### Углы укладки
- `0` - горизонтальная
- `45` - диагональная  
- `90` - вертикальная
- `135` - диагональная

### Качество
- `"preview"` - быстро (20/40 steps)
- `"standard"` - нормально (20/40 steps)
- `"high"` - качественно (30/60 steps)

## Поддерживаемые цвета
- `black`, `white`, `red`, `green`, `blue`, `yellow`, `gray`, `brown`, `orange`, `purple`, `pink`
- Или HEX коды: `{"hex": "#FF0000", "proportion": 0.5}`

## Примеры для копирования

### Быстрый тест
```json
{"params_json": "{\"colors\":[{\"name\":\"blue\",\"proportion\":0.6},{\"name\":\"yellow\",\"proportion\":0.4}],\"angle\":45,\"quality\":\"preview\",\"seed\":42}"}
```

### Высокое качество
```json
{"params_json": "{\"colors\":[{\"name\":\"green\",\"proportion\":0.8},{\"name\":\"gray\",\"proportion\":0.2}],\"angle\":90,\"quality\":\"high\",\"seed\":999}"}
```

### Многоцветная плитка
```json
{"params_json": "{\"colors\":[{\"name\":\"black\",\"proportion\":0.4},{\"name\":\"red\",\"proportion\":0.3},{\"name\":\"blue\",\"proportion\":0.2},{\"name\":\"yellow\",\"proportion\":0.1}],\"angle\":0,\"quality\":\"standard\",\"seed\":555}"}
```

## Советы
1. Начните с `quality: "preview"` для быстрого тестирования
2. Используйте `seed` для воспроизводимости результатов
3. Пропорции автоматически нормализуются
4. Для диагональных углов (45°, 135°) используется SoftEdge ControlNet
5. Для горизонтальных/вертикальных углов (0°, 90°) используется Canny ControlNet

## Подробная документация
См. `docs/WebInterfaceExamples.md` для полного руководства.

---

*Версия: v4.3.60 "TEXTUAL INVERSION TOKEN FIX & UNIFIED TOK"*
