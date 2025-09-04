#!/usr/bin/env python3
"""
Универсальный тестер для любой версии Plitka Pro
Автоматически определяет версию и запускает соответствующие тесты
"""

import os
import json
import time
import sys
import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def extract_version_from_cog_yaml() -> str:
    """Извлекает версию из cog.yaml"""
    try:
        cog_yaml_path = project_root / "cog.yaml"
        if not cog_yaml_path.exists():
            return "unknown"
            
        with open(cog_yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Ищем строку с версией в комментарии
        for line in content.split('\n'):
            if line.strip().startswith('# Version:'):
                version_part = line.split('Version:')[1].strip()
                return version_part.split()[0]  # Берем первое слово после "Version:"
            
            # Альтернативный поиск по тегу образа
            if 'image:' in line and 'plitka-pro-project:' in line:
                version_part = line.split('plitka-pro-project:')[1].strip()
                return version_part
                
        return "unknown"
    except Exception as e:
        print(f"⚠️ Ошибка чтения cog.yaml: {e}")
        return "unknown"

def get_presets_file_for_version(version: str) -> Path:
    """Возвращает путь к файлу пресетов для указанной версии"""
    scripts_dir = project_root / "scripts"
    
    # Маппинг версий к файлам пресетов
    version_mapping = {
        "v4.5.01": "test_inputs_v4.5.01_critical_fixes.json",
        "v4.4.61": "test_inputs_v4.4.61_extended.json",
        "v4.4.60": "test_inputs_v4.4.60_extended.json",
        "v4.4.59": "test_inputs_v4.4.59.json",
        "v4.4.58": "test_inputs_v4.4.58.json",
        "v4.4.56": "test_inputs_v4.4.56.json",
        "v4.4.45": "test_inputs_v4.4.45.json",
        "v4.4.39": "test_inputs_v4.4.39.json",
    }
    
    # Получаем имя файла для версии
    filename = version_mapping.get(version, "test_inputs_v4.4.39.json")
    presets_path = scripts_dir / filename
    
    # Проверяем существование файла
    if presets_path.exists():
        return presets_path
    else:
        # Fallback к базовому файлу
        fallback_path = scripts_dir / "test_inputs_v4.4.39.json"
        if fallback_path.exists():
            print(f"⚠️ Файл пресетов для версии {version} не найден, используем базовый")
            return fallback_path
        else:
            # Создаем пустой файл пресетов
            empty_presets = {
                "default": {
                    "prompt": "ohwx_rubber_tile <s0><s1> 100% red rubber tile, high quality",
                    "negative_prompt": "blurry, low quality",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "seed": 42,
                    "colormap": "random",
                    "granule_size": "medium",
                    "description": "Базовый пресет для тестирования"
                }
            }
            with open(fallback_path, 'w', encoding='utf-8') as f:
                json.dump(empty_presets, f, indent=2, ensure_ascii=False)
            print(f"📁 Создан базовый файл пресетов: {fallback_path}")
            return fallback_path

def load_env_token() -> Optional[str]:
    """Загружает API токен из переменных окружения или .env файла"""
    token = os.getenv("REPLICATE_API_TOKEN")
    if token:
        return token
    
    env_path = project_root / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("REPLICATE_API_TOKEN="):
                        _, value = line.split("=", 1)
                        value = value.strip().strip('"').strip("'")
                        if value:
                            return value
        except Exception:
            pass
    return None

def load_presets(version: str) -> Dict[str, Dict[str, Any]]:
    """Загружает пресеты для указанной версии"""
    presets_path = get_presets_file_for_version(version)
    
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Presets JSON must be an object mapping preset names to input dicts")
            return data
    except Exception as e:
        print(f"❌ Ошибка загрузки пресетов: {e}")
        return {}

def compare_versions(version1: str, version2: str) -> int:
    """Сравнивает версии. Возвращает -1 если version1 < version2, 0 если равны, 1 если version1 > version2"""
    def version_tuple(v):
        # Убираем 'v' и разбиваем на части
        v = v.replace('v', '')
        parts = v.split('.')
        return tuple(int(part) for part in parts)
    
    try:
        v1_tuple = version_tuple(version1)
        v2_tuple = version_tuple(version2)
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    except (ValueError, IndexError):
        # Если не можем распарсить, сравниваем как строки
        return -1 if version1 < version2 else (1 if version1 > version2 else 0)

def load_color_table() -> List[str]:
    """Загружает таблицу цветов из файла colors_table.txt"""
    try:
        color_file = project_root / "colors_table.txt"
        if not color_file.exists():
            # Fallback к стандартным цветам
            return ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PINK", "WHITE", "BLACK", "GRAY", "BROWN"]
        
        with open(color_file, "r", encoding="utf-8") as f:
            colors = [line.strip().upper() for line in f if line.strip()]
        return colors
    except Exception as e:
        print(f"⚠️ Ошибка загрузки таблицы цветов: {e}")
        # Fallback к стандартным цветам
        return ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PINK", "WHITE", "BLACK", "GRAY", "BROWN"]

def validate_preset_for_version(preset_name: str, preset_data: dict, version: str) -> tuple[bool, List[str]]:
    """Валидирует пресет на соответствие требованиям указанной версии"""
    errors = []
    
    # Загружаем таблицу цветов
    valid_colors = load_color_table()
    
    try:
        # Проверяем наличие prompt
        if 'prompt' not in preset_data:
            errors.append("❌ Отсутствует поле 'prompt'")
            return False, errors
        
        # Проверяем обязательные поля в зависимости от версии
        if compare_versions(version, "v4.5.01") >= 0:
            required_fields = ['seed', 'num_inference_steps', 'guidance_scale']
            optional_fields = ['colormap', 'granule_size', 'negative_prompt']
        else:
            # Для старых версий
            required_fields = ['seed', 'steps', 'guidance', 'lora_scale']
            optional_fields = ['use_controlnet', 'description']
    
        for field in required_fields:
            if field not in preset_data:
                errors.append(f"❌ Отсутствует обязательное поле '{field}'")
        
        # Проверяем опциональные поля
        for field in optional_fields:
            if field not in preset_data:
                errors.append(f"⚠️ Отсутствует опциональное поле '{field}'")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"❌ Критическая ошибка валидации: {e}")
        return False, errors

def run_single_test(client, preset_name: str, preset_data: dict, version: str) -> dict:
    """Запускает один тест и возвращает результат"""
    print(f"\n🧪 ТЕСТ: {preset_name}")
    print("=" * 60)
    
    # Валидируем пресет
    is_valid, errors = validate_preset_for_version(preset_name, preset_data, version)
    if not is_valid:
        print("❌ Пресет не прошел валидацию:")
        for error in errors:
            print(f"  {error}")
        return {
            "name": preset_name,
            "status": "validation_failed",
            "errors": errors,
            "duration": 0
        }
    
    print("✅ Пресет прошел валидацию")
    
    # Показываем параметры теста
    print(f"📝 Prompt: {preset_data.get('prompt', 'N/A')[:80]}...")
    print(f"🎲 Seed: {preset_data.get('seed', 'N/A')}")
    
    if version >= "v4.5.01":
        print(f"🔄 Steps: {preset_data.get('num_inference_steps', 'N/A')}")
        print(f"📏 Guidance: {preset_data.get('guidance_scale', 'N/A')}")
        print(f"🎨 Colormap: {preset_data.get('colormap', 'N/A')}")
        print(f"🔍 Granule Size: {preset_data.get('granule_size', 'N/A')}")
    else:
        print(f"🔄 Steps: {preset_data.get('steps', 'N/A')}")
        print(f"📏 Guidance: {preset_data.get('guidance', 'N/A')}")
        print(f"🔧 LoRA Scale: {preset_data.get('lora_scale', 'N/A')}")
    
    start_time = time.time()
    
    try:
        # Создаем предсказание
        print(f"\n🚀 Запуск генерации...")
        pred = client.predictions.create(
            model=f"nauslava/plitka-pro-project:{version}",
            input=preset_data
        )
        
        print(f"📋 Prediction ID: {pred.id}")
        
        # Ждем завершения
        print("⏳ Ожидание завершения...")
        pred.wait()
        
        duration = time.time() - start_time
        
        if pred.status == "succeeded":
            print(f"✅ Тест завершен успешно за {duration:.1f}с")
            return {
                "name": preset_name,
                "status": "success",
                "duration": duration,
                "prediction_id": pred.id,
                "output": pred.output
            }
        else:
            print(f"❌ Тест завершился с ошибкой: {pred.status}")
            return {
                "name": preset_name,
                "status": "failed",
                "duration": duration,
                "prediction_id": pred.id,
                "error": pred.error
            }
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Ошибка выполнения теста: {e}")
        return {
            "name": preset_name,
            "status": "error",
            "duration": duration,
            "error": str(e)
        }

def run_tests(version: str, test_names: List[str] = None, max_tests: int = None) -> List[dict]:
    """Запускает тесты для указанной версии"""
    print(f"🎯 УНИВЕРСАЛЬНОЕ ТЕСТИРОВАНИЕ {version}")
    print("=" * 80)
    
    # Проверяем API токен
    api_token = load_env_token()
    if not api_token:
        print("❌ REPLICATE_API_TOKEN не найден!")
        print("💡 Установите токен в переменной окружения или в файле .env")
        return []
    
    # Проверяем доступность replicate
    try:
        import replicate
        client = replicate.Client(api_token=api_token)
        print("✅ Replicate клиент инициализирован")
    except ImportError:
        print("❌ Модуль 'replicate' не установлен!")
        print("💡 Установите: pip install replicate")
        return []
    except Exception as e:
        print(f"❌ Ошибка инициализации Replicate: {e}")
        return []
    
    # Загружаем пресеты
    presets = load_presets(version)
    if not presets:
        print("❌ Пресеты не загружены!")
        return []
    
    print(f"📁 Загружено пресетов: {len(presets)}")
    
    # Определяем тесты для запуска
    if test_names:
        available_tests = [name for name in test_names if name in presets]
        print(f"🎯 Указанных тестов найдено: {len(available_tests)}")
    else:
        # Выбираем первые несколько тестов
        available_tests = list(presets.keys())
        if max_tests:
            available_tests = available_tests[:max_tests]
        print(f"🎯 Тестов для запуска: {len(available_tests)}")
    
    if not available_tests:
        print("❌ Тесты не найдены!")
        return []
    
    # Запускаем тесты
    results = []
    total_start_time = time.time()
    
    for test_name in available_tests:
        preset_data = presets[test_name]
        result = run_single_test(client, test_name, preset_data, version)
        results.append(result)
        
        # Небольшая пауза между тестами
        time.sleep(2)
    
    total_duration = time.time() - total_start_time
    
    # Анализируем результаты
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"✅ Успешных тестов: {len(successful)}")
    print(f"❌ Неудачных тестов: {len(failed)}")
    print(f"⏱️ Общее время: {total_duration:.1f}с")
    
    # Детальные результаты
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['name']}: {result['status']} ({result['duration']:.1f}с)")
    
    # Сохраняем результаты
    save_results(results, total_duration, version)
    
    return results

def save_results(results: List[dict], total_duration: float, version: str) -> None:
    """Сохраняет результаты тестирования"""
    try:
        # Создаем папку для результатов
        results_dir = project_root / "replicate_runs" / f"{version}_universal_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем файл результатов
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"test_results_{timestamp}.json"
        
        test_summary = {
            "version": version,
            "test_type": "Universal Testing",
            "timestamp": timestamp,
            "total_duration": total_duration,
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r["status"] == "success"]),
            "failed_tests": len([r for r in results if r["status"] != "success"]),
            "results": results
        }
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(test_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены: {results_file}")
        
    except Exception as e:
        print(f"⚠️ Ошибка сохранения результатов: {e}")

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Универсальный тестер для Plitka Pro")
    parser.add_argument("--version", "-v", help="Версия для тестирования (по умолчанию определяется автоматически)")
    parser.add_argument("--tests", "-t", nargs="+", help="Конкретные тесты для запуска")
    parser.add_argument("--max", "-m", type=int, help="Максимальное количество тестов")
    parser.add_argument("--list", "-l", action="store_true", help="Показать доступные тесты")
    
    args = parser.parse_args()
    
    try:
        # Определяем версию
        if args.version:
            version = args.version
        else:
            version = extract_version_from_cog_yaml()
        
        if version == "unknown":
            print("❌ Не удалось определить версию!")
            print("💡 Укажите версию вручную: --version v4.5.01")
            return 1
        
        print(f"🎯 Версия для тестирования: {version}")
        
        # Если запрошен список тестов
        if args.list:
            presets = load_presets(version)
            if presets:
                print(f"\n📋 Доступные тесты для {version}:")
                for i, name in enumerate(presets.keys(), 1):
                    print(f"  {i}. {name}")
            else:
                print("❌ Пресеты не найдены!")
            return 0
        
        # Запускаем тесты
        results = run_tests(version, args.tests, args.max)
        
        if not results:
            print("\n❌ Тестирование не выполнено!")
            return 1
        
        # Проверяем успешность
        successful = [r for r in results if r["status"] == "success"]
        if len(successful) == len(results):
            print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            return 0
        else:
            print(f"\n⚠️ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ!")
            print(f"✅ Успешно: {len(successful)}/{len(results)}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Тестирование прервано пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
