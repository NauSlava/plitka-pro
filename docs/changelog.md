# Changelog

## [2025-01-20] - Версия [v4.1.1] - Исправление размеров эмбеддингов в Textual Inversion

### Исправлено
- **Критическое исправление**: Устранена ошибка `ValueError: clip_g embedding shape does not match text_encoder embedding size`
  - Исправлена логика проверки размеров эмбеддингов в `_install_sdxl_textual_inversion_dual()`
  - Добавлена корректная проверка размерности эмбеддингов вместо сравнения форм
  - Добавлен `.squeeze(0)` для удаления batch dimension при установке эмбеддингов
- **Образ обновлен**: `r8.im/nauslava/plitka-pro:thin-ti-fix-v2` с исправленной логикой TI

### Технические детали
- **predict.py**: Исправлена проверка `emb_g_cast.shape[-1] != emb_layer_g.weight.shape[-1]`
- **SDXL TI Support**: Корректная обработка размеров dual-encoder эмбеддингов
- **Error Handling**: Улучшена диагностика ошибок с детальными сообщениями о размерах

## [2025-01-20] - Версия [v4.1.0] - Исправление загрузки Textual Inversion для SDXL

### Исправлено
- **Критическое исправление**: Устранена ошибка загрузки Textual Inversion для SDXL dual-encoder моделей
  - Добавлен fallback механизм для ручной установки `clip_g` и `clip_l` тензоров
  - Реализован метод `_install_sdxl_textual_inversion_dual()` для корректной обработки SDXL TI формата
  - Исправлена ошибка `ValueError: Loaded state dictonary is incorrect: {'clip_g': tensor(...), 'clip_l': tensor(...)}`
- **Оптимизация сборки**: Образ успешно запушен как `r8.im/nauslava/plitka-pro:thin-ti-fix`
  - Использован "thin" подход без предварительной загрузки весов
  - Модели загружаются в runtime для экономии места на диске

### Технические детали
- **predict.py**: Добавлен try-except блок для стандартной загрузки TI с fallback на ручную установку
- **SDXL TI Support**: Корректная обработка dual-encoder формата с отдельными `clip_g` и `clip_l` тензорами
- **Build Strategy**: Переход на "thin by default" для локальной разработки

## [2025-01-20] - Версия [v4.0.0] - Оптимизация сборки и ControlNet синхронизация

### Добавлено
- **ControlNet Lineart**: Добавлена поддержка Lineart ControlNet для диагональных углов (45°/135°)
- **Угловая логика**: Реализован `select_controlnet_by_angle()` для динамического выбора ControlNet
- **Graceful Fallback**: Модели загружаются из локального кэша или напрямую с Hugging Face Hub

### Изменено
- **cog.yaml**: Упрощен до минимальной конфигурации без предварительной загрузки весов
- **Dockerfile**: Переименован в `Dockerfile.legacy` и депрекирован
- **Build Strategy**: Внедрен "thin" подход для экономии места на диске

### Исправлено
- **401 Unauthorized**: Удалена загрузка gated модели `controlnet-lineart-sdxl-1.0`
- **Disk Space**: Решена проблема недостатка места (150GB) через оптимизацию сборки
- **YAML Syntax**: Исправлена ошибка `Additional property else is not allowed`
