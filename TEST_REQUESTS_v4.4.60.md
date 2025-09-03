# 🧪 Тестовые запросы для Plitka Pro v4.4.60
## Colormap Pattern Fix + Production Ready

---

## **🔍 КРИТИЧЕСКИЕ ТЕСТЫ ИСПРАВЛЕНИЯ ПАТТЕРНОВ**

### **1. Тест исправления Grid паттерна (КРИТИЧЕСКИЙ)**
**Цель:** Проверить, что grid паттерн больше не создает вертикальные полосы

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, grid pattern, scattered dots, realistic texture"
- negative_prompt: "missing red, missing blue, blurry, low quality, vertical stripes, solid bands"
- num_inference_steps: 25
- guidance_scale: 7.5
- seed: 42
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Точечное распределение красного и синего цветов по всей поверхности, НЕ вертикальные полосы

---

### **2. Тест исправления Radial паттерна (КРИТИЧЕСКИЙ)**
**Цель:** Проверить, что radial паттерн больше не создает концентрические кольца

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, radial pattern, scattered dots, natural distribution"
- negative_prompt: "missing red, missing blue, blurry, low quality, circular bands, solid rings"
- num_inference_steps: 25
- guidance_scale: 7.5
- seed: 456
- colormap: "radial"
- granule_size: "medium"
```

**Ожидаемый результат:** Точечное распределение с радиальным влиянием, НЕ концентрические кольца

---

### **3. Тест сравнения паттернов (ВАЛИДАЦИОННЫЙ)**
**Цель:** Сравнить исправленные паттерны с рабочими

**Запрос 1 - Random (контрольный):**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, random pattern, scattered dots, organic texture"
- negative_prompt: "missing red, missing blue, blurry, low quality, organized patterns"
- num_inference_steps: 25
- guidance_scale: 7.5
- seed: 789
- colormap: "random"
- granule_size: "medium"
```

**Запрос 2 - Grid (исправленный):**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, grid pattern, scattered dots, geometric distribution"
- negative_prompt: "missing red, missing blue, blurry, low quality, vertical stripes"
- num_inference_steps: 25
- guidance_scale: 7.5
- seed: 123
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Оба паттерна должны создавать точечное распределение, grid НЕ должен создавать вертикальные полосы

---

## **🎨 ТЕСТЫ РАЗНЫХ КОЛИЧЕСТВ ЦВЕТОВ**

### **4. Тест моно-цвета с Grid паттерном**
**Цель:** Проверить, что моно-цвета работают корректно с исправленным grid паттерном

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 100% red rubber tile, grid pattern, scattered dots, professional finish"
- negative_prompt: "missing red, blurry, low quality, solid color, uniform, vertical stripes"
- num_inference_steps: 20
- guidance_scale: 7.0
- seed: 333
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Равномерное точечное распределение красного цвета по всей поверхности

---

### **5. Тест трех цветов с Grid паттерном**
**Цель:** Проверить сложные комбинации с исправленным паттерном

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 40% red, 30% blue, 30% yellow rubber tile, grid pattern, scattered dots, realistic"
- negative_prompt: "missing red, missing blue, missing yellow, blurry, low quality, vertical stripes, solid bands"
- num_inference_steps: 30
- guidance_scale: 8.0
- seed: 444
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Точечное распределение трех цветов в правильных пропорциях, НЕ вертикальные полосы

---

### **6. Тест четырех цветов с Grid паттерном**
**Цель:** Проверить максимальную сложность с исправленным паттерном

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 25% red, 25% blue, 25% yellow, 25% grsgrn rubber tile, grid pattern, scattered dots"
- negative_prompt: "missing red, missing blue, missing yellow, missing grsgrn, blurry, low quality, vertical stripes"
- num_inference_steps: 35
- guidance_scale: 8.5
- seed: 555
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Сложное точечное распределение четырех цветов, НЕ вертикальные полосы

---

## **🔧 ТЕСТЫ РАЗМЕРОВ ГРАНУЛ**

### **7. Тест маленьких гранул с Grid паттерном**
**Цель:** Проверить мелкую текстуру с исправленным паттерном

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, grid pattern, small granules, fine texture"
- negative_prompt: "missing red, missing blue, blurry, low quality, vertical stripes, solid bands"
- num_inference_steps: 25
- guidance_scale: 7.5
- seed: 666
- colormap: "grid"
- granule_size: "small"
```

**Ожидаемый результат:** Мелкая точечная текстура с правильным распределением цветов

---

### **8. Тест больших гранул с Grid паттерном**
**Цель:** Проверить крупную текстуру с исправленным паттерном

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 50% red, 50% blue rubber tile, grid pattern, large granules, coarse texture"
- negative_prompt: "missing red, missing blue, blurry, low quality, vertical stripes, solid bands"
- num_inference_steps: 25
- guidance_scale: 7.5
- seed: 777
- colormap: "grid"
- granule_size: "large"
```

**Ожидаемый результат:** Крупная точечная текстура с правильным распределением цветов

---

## **🎯 ТЕСТЫ CONTROLNET ИНТЕГРАЦИИ**

### **9. Тест ControlNet с исправленным Grid паттерном**
**Цель:** Проверить работу ControlNet с исправленными паттернами

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 30% red, 30% blue, 40% yellow rubber tile, grid pattern, scattered dots, ControlNet precision"
- negative_prompt: "missing red, missing blue, missing yellow, blurry, low quality, vertical stripes, solid bands"
- num_inference_steps: 30
- guidance_scale: 8.0
- seed: 888
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Высокоточное точечное распределение с ControlNet, НЕ вертикальные полосы

---

### **10. Тест ControlNet с четырех-цветной комбинацией**
**Цель:** Проверить максимальную сложность ControlNet с исправленными паттернами

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 25% red, 25% blue, 25% yellow, 25% grsgrn rubber tile, grid pattern, scattered dots, ControlNet"
- negative_prompt: "missing red, missing blue, missing yellow, missing grsgrn, blurry, low quality, vertical stripes"
- num_inference_steps: 35
- guidance_scale: 8.5
- seed: 999
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Сложное точечное распределение четырех цветов с ControlNet, НЕ вертикальные полосы

---

## **🏭 ТЕСТЫ ПРОДАКШЕН ГОТОВНОСТИ**

### **11. Тест продакшен качества с Grid паттерном**
**Цель:** Проверить готовность к продакшену с исправленными паттернами

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 100% red rubber tile, grid pattern, scattered dots, production quality, commercial finish"
- negative_prompt: "missing red, blurry, low quality, solid color, vertical stripes, amateur"
- num_inference_steps: 20
- guidance_scale: 7.0
- seed: 1001
- colormap: "grid"
- granule_size: "medium"
```

**Ожидаемый результат:** Профессиональное качество с правильным точечным распределением

---

### **12. Тест продакшен качества с Radial паттерном**
**Цель:** Проверить готовность к продакшену с исправленным radial паттерном

**Запрос:**
```
URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.60

Параметры:
- prompt: "ohwx_rubber_tile <s0><s1> 100% blue rubber tile, radial pattern, scattered dots, production quality, architectural finish"
- negative_prompt: "missing blue, blurry, low quality, solid color, circular bands, amateur"
- num_inference_steps: 20
- guidance_scale: 7.0
- seed: 1002
- colormap: "radial"
- granule_size: "medium"
```

**Ожидаемый результат:** Профессиональное качество с правильным точечным распределением

---

## **📊 ИНСТРУКЦИИ ПО ТЕСТИРОВАНИЮ**

### **Порядок выполнения тестов:**

1. **Начать с критических тестов (1-2)** - проверить исправление основных проблем
2. **Выполнить сравнительные тесты (3)** - убедиться в корректности исправлений
3. **Провести тесты разных количеств цветов (4-6)** - проверить масштабируемость
4. **Протестировать размеры гранул (7-8)** - проверить детализацию
5. **Выполнить тесты ControlNet (9-10)** - проверить интеграцию
6. **Завершить тестами продакшена (11-12)** - подтвердить готовность

### **Критерии успешного тестирования:**

✅ **Grid паттерн:** Точечное распределение, НЕ вертикальные полосы
✅ **Radial паттерн:** Точечное распределение, НЕ концентрические кольца
✅ **Random паттерн:** Органическое точечное распределение
✅ **Granular паттерн:** Реалистичная гранулярная текстура
✅ **Все цвета:** Правильные пропорции и распределение
✅ **ControlNet:** Высокая точность с исправленными паттернами
✅ **Качество:** Профессиональный уровень готовности

### **Ожидаемые результаты:**

- **ДО исправления:** Вертикальные полосы в grid, концентрические кольца в radial
- **ПОСЛЕ исправления:** Точечное распределение во всех паттернах
- **Качество:** Улучшенная реалистичность и естественность текстуры
- **Готовность:** Модель готова к продакшен использованию

---

## **🚀 ЗАКЛЮЧЕНИЕ**

Версия v4.4.60 содержит **критические исправления паттернов colormap**, которые восстанавливают правильное точечное распределение цветов. Все тесты должны подтвердить:

1. **Устранение вертикальных полос** в grid паттерне
2. **Устранение концентрических колец** в radial паттерне
3. **Сохранение качества** random и granular паттернов
4. **Готовность к продакшену** с исправленными паттернами

**Модель готова к комплексному тестированию на Replicate!** 🎯
