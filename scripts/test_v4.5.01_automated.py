#!/usr/bin/env python3
"""
Автоматический тестер для v4.5.01 - Critical Architecture Fixes
Тестирует критические исправления: AttributeError, валидация colormap, ControlNet методы
"""

import os
import json
import time
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

def load_presets() -> Dict[str, Dict[str, Any]]:
    """Загружает пресеты для v4.5.01"""
    presets_path = project_root / "scripts" / "test_inputs_v4.5.01_critical_fixes.json"
    
    if not presets_path.exists():
        print(f"❌ Файл пресетов не найден: {presets_path}")
        return {}
    
    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Presets JSON must be an object mapping preset names to input dicts")
            return data
    except Exception as e:
        print(f"❌ Ошибка загрузки пресетов: {e}")
        return {}

def validate_preset(preset_name: str, preset_data: dict) -> tuple[bool, List[str]]:
    """Валидирует пресет на соответствие требованиям v4.5.01"""
    errors = []
    
    try:
        # Проверяем наличие prompt
        if 'prompt' not in preset_data:
            errors.append("❌ Отсутствует поле 'prompt'")
            return False, errors
        
        prompt = preset_data['prompt']
        
        # Проверяем обязательные поля для v4.5.01
        required_fields = ['seed', 'num_inference_steps', 'guidance_scale']
        for field in required_fields:
            if field not in preset_data:
                errors.append(f"❌ Отсутствует обязательное поле '{field}'")
        
        # Проверяем опциональные поля
        optional_fields = ['colormap', 'granule_size', 'negative_prompt']
        for field in optional_fields:
            if field not in preset_data:
                errors.append(f"⚠️ Отсутствует опциональное поле '{field}'")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"❌ Критическая ошибка валидации: {e}")
        return False, errors

def run_single_test(client, preset_name: str, preset_data: dict) -> dict:
    """Запускает один тест и возвращает результат"""
    print(f"\n🧪 ТЕСТ: {preset_name}")
    print("=" * 60)
    
    # Валидируем пресет
    is_valid, errors = validate_preset(preset_name, preset_data)
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
    print(f"🔄 Steps: {preset_data.get('num_inference_steps', 'N/A')}")
    print(f"📏 Guidance: {preset_data.get('guidance_scale', 'N/A')}")
    print(f"🎨 Colormap: {preset_data.get('colormap', 'N/A')}")
    print(f"🔍 Granule Size: {preset_data.get('granule_size', 'N/A')}")
    
    start_time = time.time()
    
    try:
        # Создаем предсказание
        print(f"\n🚀 Запуск генерации...")
        pred = client.predictions.create(
            model="nauslava/plitka-pro-project:v4.5.01",
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

def run_critical_tests() -> List[dict]:
    """Запускает критические тесты для v4.5.01"""
    print("🎯 АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ v4.5.01 - Critical Architecture Fixes")
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
    presets = load_presets()
    if not presets:
        print("❌ Пресеты не загружены!")
        return []
    
    print(f"📁 Загружено пресетов: {len(presets)}")
    
    # Фильтруем критические тесты
    critical_tests = [
        "🔴 MONO COLOR - BASIC FUNCTIONALITY",
        "🔴🔵 DUO COLOR - ATTRIBUTEERROR FIX", 
        "🔴🔵🔵 TRIO COLOR - CONTROLNET METHODS",
        "🌈 MULTI COLOR - COMPLEX CONTROLNET",
        "🎨 RGBA COLORMAP VALIDATION"
    ]
    
    available_tests = [name for name in critical_tests if name in presets]
    print(f"🎯 Критических тестов найдено: {len(available_tests)}")
    
    if not available_tests:
        print("❌ Критические тесты не найдены в пресетах!")
        return []
    
    # Запускаем тесты
    results = []
    total_start_time = time.time()
    
    for test_name in available_tests:
        preset_data = presets[test_name]
        result = run_single_test(client, test_name, preset_data)
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
    save_results(results, total_duration)
    
    return results

def save_results(results: List[dict], total_duration: float) -> None:
    """Сохраняет результаты тестирования"""
    try:
        # Создаем папку для результатов
        results_dir = project_root / "replicate_runs" / "v4.5.01_automated_tests"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем файл результатов
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"test_results_{timestamp}.json"
        
        test_summary = {
            "version": "v4.5.01",
            "test_type": "Critical Architecture Fixes",
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
    try:
        results = run_critical_tests()
        
        if not results:
            print("\n❌ Тестирование не выполнено!")
            return 1
        
        # Проверяем успешность
        successful = [r for r in results if r["status"] == "success"]
        if len(successful) == len(results):
            print("\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
            print("✅ Критические исправления v4.5.01 работают корректно")
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
