# Тестирование модели на Replicate (v4.4.39)

Этот гайд описывает запуск тестов через Replicate с учётом долгого старта (2–5 минут) и авто‑остановки при ошибках запуска.

## Требования
- Установите переменную окружения `REPLICATE_API_TOKEN` (или добавьте в `.env`).
- Установите пакет: `pip install replicate`.

## Скрипт тестирования
Используйте `scripts/replicate_test.py`. Он:
- Создаёт prediction для `nauslava/plitka-pro-project:v4.4.39`.
- Поллит статус и логи каждые 6 секунд.
- Ожидает старта до 7 минут. Признаки старта: `setup completed` или статус `processing`.
- При критической ошибке до старта — отменяет prediction.
- Печатает логи и итоговые результаты (preview/final/colormap).

## Запуск одного теста (preset)
```bash
python3 scripts/replicate_test.py --model nauslava/plitka-pro-project:v4.4.39 --preset basic
```
Доступные пресеты: `basic`, `mix2`, `mix3`, `fast`, `hq`, `antimosaic`, `controlnet` (см. `scripts/test_inputs_v4.4.39.json`).

## Запуск батча пресетов
```bash
python3 scripts/replicate_test.py --model nauslava/plitka-pro-project:v4.4.39 --batch scripts/test_inputs_v4.4.39.json
```

## Переопределение входов
```bash
python3 scripts/replicate_test.py \
  --model nauslava/plitka-pro-project:v4.4.39 \
  --prompt "60% red, 40% white" --seed 12345 --steps 35 --guidance 6.7 --lora_scale 0.75
```

## Тайм‑ауты
- Ожидание старта: 7 минут (`--startup-timeout`).
- Общий тайм‑аут: 25 минут (`--total-timeout`).
- Интервал опроса: 6 секунд (`--poll-seconds`).

## Признаки ошибок запуска (авто‑остановка)
До старта в логах: `Traceback`, `CRITICAL`, `RuntimeError`, `CUDA error`, `device-side assert`, `❌`, `ERROR:predict:`.

## Ожидаемые артефакты
JSON с URL‑ссылками на: `preview.png`, `final.png`, `colormap.png`.
