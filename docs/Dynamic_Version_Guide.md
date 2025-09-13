# 🚀 Руководство по динамической системе версий

## 📋 **Обзор**

Динамическая система версий автоматически определяет текущую версию проекта из git тегов и исключает необходимость жесткого кодирования версий в коде.

## 🔄 **Процесс публикации**

### **Автоматическая публикация (рекомендуется):**
```bash
python3 scripts/publish_version.py
```

Этот скрипт:
1. ✅ Автоматически определяет версию из git
2. ✅ Обновляет информацию о версии
3. ✅ Собирает Docker образ (`cog build`)
4. ✅ Публикует на Replicate (`cog push`)

### **Ручное обновление версии:**
```bash
python3 scripts/update_version_info.py
```

## 📁 **Структура файлов**

```
project/
├── version_info.json          # Автоматически создается
├── version_config.json        # Конфигурация версий
├── version_manager.py         # Менеджер версий
└── scripts/
    ├── update_version_info.py # Обновление версии
    └── publish_version.py     # Публикация
```

## 🔧 **Использование в коде**

### **Получение текущей версии:**
```python
from version_manager import get_current_version

version = get_current_version()
print(f"Текущая версия: {version}")
```

### **Получение текущего хеша:**
```python
from version_manager import get_current_hash

hash_value = get_current_hash()
print(f"Текущий хеш: {hash_value}")
```

### **Получение полной информации:**
```python
from version_manager import get_current_version_info

info = get_current_version_info()
print(f"Версия: {info['version']}")
print(f"Хеш: {info['hash']}")
print(f"Источник: {info['source']}")
```

## 🎯 **Стратегия определения версии**

Система использует следующий приоритет:

1. **Переменные окружения** (`CURRENT_VERSION`, `CURRENT_DOCKER_HASH`)
2. **Файл version_info.json** (создается при публикации)
3. **Конфигурация** (`version_config.json`)

## ✅ **Преимущества**

- ✅ **Автоматизация:** Нет необходимости вручную обновлять версии
- ✅ **Надежность:** Множественные источники определения версии
- ✅ **Гибкость:** Легко переключаться между источниками
- ✅ **Консистентность:** Все компоненты используют одну версию
- ✅ **Отказоустойчивость:** Fallback стратегии на всех уровнях

## 🧪 **Тестирование**

### **Проверка текущей версии:**
```bash
python3 -c "from version_manager import get_current_version; print(get_current_version())"
```

### **Проверка GUI тестера:**
```bash
python3 scripts/replicate_gui.py
```

### **Проверка predict.py:**
```bash
python3 -c "from predict import MODEL_VERSION; print(MODEL_VERSION)"
```

## 🚨 **Важные замечания**

1. **Git теги:** Система использует git теги для определения версии
2. **Docker образы:** Хеш определяется из локальных Docker образов
3. **Replicate:** Версии должны быть активированы на Replicate
4. **Fallback:** Всегда есть резервные стратегии определения версии

## 📞 **Поддержка**

При возникновении проблем:
1. Проверьте `version_info.json`
2. Убедитесь, что git теги установлены
3. Проверьте Docker образы
4. Обратитесь к отчету `docs/reports/Dynamic_Version_System_Report.md`

