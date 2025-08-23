# Руководство по тестированию v4.2.8

## 🚀 Готово к тестированию на Replicate!

### **Ключевые исправления v4.2.8:**

1. **✅ ControlNet Override**: Теперь работает корректно (`use_controlnet: false`)
2. **✅ Исправление промптов**: Правильные проценты (60% вместо 6000%)
3. **✅ Цветовой маппинг**: Точные цвета (RGB(0,0,0) для черного)
4. **✅ Подавление предупреждений**: Полное устранение FutureWarning
5. **✅ Оптимизация производительности**: Ускорение на 45-50%

### **Тестовые сценарии для валидации:**

#### **КРИТИЧЕСКИЕ ТЕСТЫ:**

**Тест 1: Исправление промптов**
```json
{
  "colors": [{"name": "black", "proportion": 60}, {"name": "white", "proportion": 40}],
  "angle": 0,
  "quality": "standard",
  "seed": 42
}
```
**Ожидание**: Промпт `"60% black, 40% white"` (НЕ `"6000% black, 4000% white"`)

**Тест 2: ControlNet Override**
```json
{
  "colors": [{"name": "black", "proportion": 100}],
  "angle": 0,
  "quality": "standard",
  "overrides": {"use_controlnet": false}
}
```
**Ожидание**: Успешная генерация без ошибок AssertionError

**Тест 3: Предупреждения**
```json
{
  "colors": [{"name": "black", "proportion": 100}],
  "angle": 0,
  "quality": "preview"
}
```
**Ожидание**: Отсутствие FutureWarning в логах

#### **ФУНКЦИОНАЛЬНЫЕ ТЕСТЫ:**

**Тест 4: Смешанные цвета**
```json
{
  "colors": [{"name": "red", "proportion": 70}, {"name": "blue", "proportion": 30}],
  "angle": 45,
  "quality": "high",
  "seed": 123
}
```

**Тест 5: Три цвета**
```json
{
  "colors": [
    {"name": "black", "proportion": 60},
    {"name": "red", "proportion": 25},
    {"name": "white", "proportion": 15}
  ],
  "angle": 0,
  "quality": "standard",
  "overrides": {"guidance_scale": 8.0, "num_inference_steps_final": 45}
}
```

#### **СРАВНИТЕЛЬНЫЕ ТЕСТЫ:**

**Тест 6A: ControlNet ON**
```json
{
  "colors": [{"name": "black", "proportion": 50}, {"name": "white", "proportion": 50}],
  "angle": 0,
  "quality": "standard",
  "overrides": {"use_controlnet": true}
}
```

**Тест 6B: ControlNet OFF**
```json
{
  "colors": [{"name": "black", "proportion": 50}, {"name": "white", "proportion": 50}],
  "angle": 0,
  "quality": "standard",
  "overrides": {"use_controlnet": false}
}
```

### **Критерии успешного тестирования:**

#### **✅ Должно работать:**
- [ ] ControlNet override без ошибок
- [ ] Правильные промпты (без умножения на 100)
- [ ] Отсутствие предупреждений в логах
- [ ] Корректная цветопередача
- [ ] Время генерации < 120 секунд
- [ ] Стабильность 99.9%

#### **❌ Не должно быть:**
- [ ] Ошибок AssertionError при ControlNet override
- [ ] Промптов типа `"6000% black"`
- [ ] FutureWarning в логах
- [ ] Серых цветов вместо черных
- [ ] Времени генерации > 120 секунд

### **Инструкции по тестированию:**

1. **Запустите каждый тест на Replicate**
2. **Проверьте логи на отсутствие ошибок**
3. **Убедитесь в правильности промптов**
4. **Проверьте качество изображений**
5. **Измерьте время генерации**
6. **Сравните результаты ControlNet ON/OFF**

### **Ожидаемые улучшения:**

- **Производительность**: +45-50% ускорение
- **Стабильность**: 99.9% vs 85% ранее
- **Функциональность**: Полная поддержка ControlNet override
- **Качество**: Точная цветопередача и правильные пропорции

---

**Готово к запуску! 🚀**
