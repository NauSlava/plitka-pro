# Примеры запросов для Plitka Pro v4.3.1

## 🚀 **Версия v4.3.1: GPU/NPU Resource Management & Monitoring**

### 📋 **Описание новой функциональности:**
- **Ограничения ресурсов GPU**: 50-80% от доступной памяти
- **Ограничения ресурсов NPU**: 50-80% от доступной производительности
- **Мониторинг в реальном времени** с автоматической очисткой
- **Управление памятью** через torch.cuda.empty_cache()

---

## 🎨 **Пример 1: Монохромная плитка (черная)**
```json
{
  "colors": [
    {"name": "black", "proportion": 100}
  ],
  "angle": 0,
  "quality": "standard",
  "seed": 42
}
```

**Ожидаемый результат**: Черная резиновая плитка с видом сверху (угол 0°)

---

## 🎨 **Пример 2: Двухцветная плитка (красная + синяя)**
```json
{
  "colors": [
    {"name": "red", "proportion": 60},
    {"name": "blue", "proportion": 40}
  ],
  "angle": 0,
  "quality": "high",
  "seed": 123
}
```

**Ожидаемый результат**: Плитка с 60% красного и 40% синего цвета

---

## 🎨 **Пример 3: Трехцветная плитка (черная + белая + красная)**
```json
{
  "colors": [
    {"name": "black", "proportion": 50},
    {"name": "white", "proportion": 30},
    {"name": "red", "proportion": 20}
  ],
  "angle": 0,
  "quality": "preview",
  "seed": 456
}
```

**Ожидаемый результат**: Плитка с 50% черного, 30% белого, 20% красного

---

## 🎨 **Пример 4: Плитка с ControlNet (угол 45°)**
```json
{
  "colors": [
    {"name": "green", "proportion": 70},
    {"name": "brown", "proportion": 30}
  ],
  "angle": 45,
  "quality": "standard",
  "seed": 789,
  "overrides": {
    "use_controlnet": true
  }
}
```

**Ожидаемый результат**: Плитка с 70% зеленого и 30% коричневого под углом 45° (с ControlNet)

---

## 🎨 **Пример 5: Плитка без ControlNet (принудительно)**
```json
{
  "colors": [
    {"name": "yellow", "proportion": 100}
  ],
  "angle": 45,
  "quality": "standard",
  "seed": 999,
  "overrides": {
    "use_controlnet": false
  }
}
```

**Ожидаемый результат**: Желтая плитка под углом 45° без ControlNet (может быть некачественной)

---

## 🎨 **Пример 6: Высокое качество с кастомными параметрами**
```json
{
  "colors": [
    {"name": "gray", "proportion": 100}
  ],
  "angle": 0,
  "quality": "high",
  "seed": 111,
  "overrides": {
    "guidance_scale": 12.0,
    "num_inference_steps_final": 50
  }
}
```

**Ожидаемый результат**: Серая плитка высокого качества с увеличенным guidance_scale и количеством шагов

---

## 🎨 **Пример 7: Плитка с диагональным углом (90°)**
```json
{
  "colors": [
    {"name": "blue", "proportion": 80},
    {"name": "white", "proportion": 20}
  ],
  "angle": 90,
  "quality": "standard",
  "seed": 222
}
```

**Ожидаемый результат**: Плитка с 80% синего и 20% белого под углом 90° (с ControlNet)

---

## 🎨 **Пример 8: Плитка с рандомным seed**
```json
{
  "colors": [
    {"name": "red", "proportion": 50},
    {"name": "black", "proportion": 50}
  ],
  "angle": 0,
  "quality": "standard"
}
```

**Ожидаемый результат**: Плитка с 50% красного и 50% черного с автоматически выбранным seed

---

## 🔧 **Технические детали:**

### **Качество генерации:**
- **preview**: Быстрая генерация (меньше шагов)
- **standard**: Стандартное качество (баланс скорости/качества)
- **high**: Высокое качество (больше шагов)

### **Углы укладки:**
- **0°**: Вид сверху - единственный надежный ракурс без ControlNet
- **45°, 90°, 135°**: Требуют ControlNet для качественной генерации
- **Другие углы**: Автоматический выбор подходящего ControlNet

### **Цвета:**
- **Поддерживаемые**: black, white, red, green, blue, gray, brown, yellow
- **Пропорции**: 0-100% для каждого цвета
- **Сумма**: Рекомендуется 100% (автоматически нормализуется)

### **Overrides:**
- **use_controlnet**: true/false для принудительного включения/отключения
- **guidance_scale**: 1.0-20.0 (по умолчанию 8.0)
- **num_inference_steps**: 1-100 для preview и final

---

## 🧪 **Тестирование новой функциональности:**

### **Проверка ограничений ресурсов:**
1. **GPU память**: Должна использоваться не более 80% (High), 75% (Medium), 70% (Low)
2. **Автоматическая очистка**: После каждой генерации и при превышении лимитов
3. **Мониторинг**: Логи должны показывать использование ресурсов

### **Проверка мониторинга:**
1. **ResourceMonitor**: Должен запускаться при инициализации
2. **Логи ресурсов**: Должны показывать текущее состояние GPU/NPU
3. **Автоматическая остановка**: При завершении работы

---

## 📊 **Ожидаемые логи:**

```
🚀 Resource monitoring started
🎯 Using device: cuda:0 (NVIDIA GeForce RTX 4090 (24.0GB))
🚀 High-memory GPU optimizations enabled (max: 19.2GB, 80%)
🧹 CUDA cache cleared, memory fraction set to 80%
🔍 Checking device resources before generation...
📊 Resource status: {'device_type': 'cuda', 'device_name': 'NVIDIA GeForce RTX 4090', 'max_memory_usage_gb': 0.0, 'max_gpu_utilization_percent': 0.0, 'monitoring_active': True}
🧹 Cleaning up device resources after generation...
📊 Final resource summary: {'device_type': 'cuda', 'device_name': 'NVIDIA GeForce RTX 4090', 'max_memory_usage_gb': 2.1, 'max_gpu_utilization_percent': 45.2, 'monitoring_active': True}
```

---

## 🎯 **Цель тестирования:**

Проверить, что новая версия v4.3.1:
1. ✅ **Ограничивает использование ресурсов** на 50-80%
2. ✅ **Мониторит ресурсы** в реальном времени
3. ✅ **Автоматически очищает память** после генерации
4. ✅ **Логирует состояние ресурсов** для диагностики
5. ✅ **Сохраняет всю функциональность** предыдущих версий
