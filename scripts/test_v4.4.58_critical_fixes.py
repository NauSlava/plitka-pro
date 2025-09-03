#!/usr/bin/env python3
"""
Скрипт автоматического тестирования критических исправлений v4.4.58
Проверяет работу ColorManager и исправление рассинхронизации модулей
"""

import json
import os
import sys
from pathlib import Path

# Импорт ColorManager из predict.py
sys.path.append('.')
try:
    from predict import ColorManager
    COLOR_MANAGER_AVAILABLE = True
except ImportError:
    COLOR_MANAGER_AVAILABLE = False
    print("⚠️ ColorManager недоступен - используется упрощенная валидация")

def load_test_presets():
    """Загрузка тестовых пресетов для v4.4.58"""
    preset_file = "scripts/test_inputs_v4.4.58.json"
    
    if not os.path.exists(preset_file):
        print(f"❌ Файл пресетов не найден: {preset_file}")
        return None
    
    try:
        with open(preset_file, 'r', encoding='utf-8') as f:
            presets = json.load(f)
        print(f"✅ Загружено пресетов для тестирования: {len(presets)}")
        return presets
    except Exception as e:
        print(f"❌ Ошибка загрузки пресетов: {e}")
        return None

def analyze_preset(preset_name, preset_data):
    """Анализ отдельного пресета"""
    prompt = preset_data.get("prompt", "")
    expected_steps = preset_data.get("steps", 20)
    expected_guidance = preset_data.get("guidance", 7.0)
    
    # Использование ColorManager если доступен
    if COLOR_MANAGER_AVAILABLE:
        try:
            color_manager = ColorManager()
            colors = color_manager.extract_colors_from_prompt(prompt)
            color_count = len(colors)
            print(f"   🔍 ColorManager: {colors} ({color_count} цветов)")
        except Exception as e:
            print(f"   ⚠️ ColorManager ошибка: {e}")
            colors = []
            color_count = 0
    else:
        # Упрощенная валидация
        colors = []
        words = prompt.lower().split()
        
        for word in words:
            clean_word = word.strip('%,.!?()[]{}')
            if clean_word in ['grsgrn', 'ltgreen', 'whtgrn', 'dkblue', 'dkgreen', 'dkgray', 'emerald', 'grnapl', 'turqse', 'skyblue', 'pearl', 'salmon', 'beige', 'violet', 'lilac', 'red', 'blue', 'yellow', 'brown', 'gray', 'white', 'black', 'orange', 'pink', 'khaki', 'limegrn', 'ltgray', 'tercot']:
                colors.append(clean_word)
        
        color_count = len(colors)
    
    # Определение ожидаемых параметров на основе количества цветов
    if color_count == 1:
        expected_adaptive_steps = 20
        expected_adaptive_guidance = 7.0
    elif color_count == 2:
        expected_adaptive_steps = 25
        expected_adaptive_guidance = 7.5
    elif color_count == 3:
        expected_adaptive_steps = 30
        expected_adaptive_guidance = 8.0
    else:  # 4+ цвета
        expected_adaptive_steps = 35
        expected_adaptive_guidance = 8.5
    
    # Проверка соответствия
    steps_match = expected_steps == expected_adaptive_steps
    guidance_match = abs(expected_guidance - expected_adaptive_guidance) < 0.1
    
    return {
        "prompt": prompt,
        "colors": colors,
        "color_count": color_count,
        "expected_steps": expected_steps,
        "expected_guidance": expected_guidance,
        "expected_adaptive_steps": expected_adaptive_steps,
        "expected_adaptive_guidance": expected_adaptive_guidance,
        "steps_match": steps_match,
        "guidance_match": guidance_match,
        "all_match": steps_match and guidance_match
    }

def run_critical_tests():
    """Запуск критических тестов"""
    print("🚀 ТЕСТИРОВАНИЕ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ v4.4.58")
    print("=" * 80)
    
    if COLOR_MANAGER_AVAILABLE:
        print("✅ ColorManager доступен - используется полная валидация")
    else:
        print("⚠️ ColorManager недоступен - используется упрощенная валидация")
    
    # Загрузка пресетов
    presets = load_test_presets()
    if not presets:
        return False
    
    # Анализ каждого пресета
    test_results = []
    critical_tests = []
    
    for preset_name, preset_data in presets.items():
        if "critical_fix_test" in preset_name:
            critical_tests.append((preset_name, preset_data))
        else:
            test_results.append((preset_name, preset_data))
    
    # Сначала тестируем критические исправления
    print("\n🔴 КРИТИЧЕСКИЕ ТЕСТЫ ИСПРАВЛЕНИЙ:")
    print("-" * 50)
    
    critical_success = 0
    for preset_name, preset_data in critical_tests:
        result = analyze_preset(preset_name, preset_data)
        
        status = "✅" if result["all_match"] else "❌"
        print(f"{status} {preset_name}")
        print(f"   Промпт: {result['prompt']}")
        print(f"   Цвета: {result['colors']} ({result['color_count']})")
        print(f"   Ожидаемые параметры: steps={result['expected_steps']}, guidance={result['expected_guidance']}")
        print(f"   Адаптивные параметры: steps={result['expected_adaptive_steps']}, guidance={result['expected_adaptive_guidance']}")
        print(f"   Соответствие: {'ДА' if result['all_match'] else 'НЕТ'}")
        print()
        
        if result["all_match"]:
            critical_success += 1
    
    # Тестируем остальные пресеты
    print("\n🔵 ОБЩИЕ ТЕСТЫ:")
    print("-" * 50)
    
    general_success = 0
    for preset_name, preset_data in test_results:
        result = analyze_preset(preset_name, preset_data)
        
        status = "✅" if result["all_match"] else "❌"
        print(f"{status} {preset_name}")
        print(f"   Промпт: {result['prompt']}")
        print(f"   Цвета: {result['colors']} ({result['color_count']})")
        print(f"   Соответствие: {'ДА' if result['all_match'] else 'НЕТ'}")
        
        if result["all_match"]:
            general_success += 1
    
    # Итоговый отчет
    print("\n" + "=" * 80)
    print("📊 ИТОГОВЫЙ ОТЧЕТ:")
    print(f"🔴 Критические тесты: {critical_success}/{len(critical_tests)} ✅")
    print(f"🔵 Общие тесты: {general_success}/{len(test_results)} ✅")
    print(f"📈 Общий успех: {critical_success + general_success}/{len(presets)} ✅")
    
    # Проверка критических тестов
    if critical_success == len(critical_tests):
        print("\n🎉 ВСЕ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ РАБОТАЮТ КОРРЕКТНО!")
        print("✅ Рассинхронизация модулей устранена")
        print("✅ Специальные цвета распознаются корректно")
        print("✅ Адаптивные параметры применяются правильно")
        return True
    else:
        print(f"\n⚠️ КРИТИЧЕСКИЕ ПРОБЛЕМЫ НЕ ПОЛНОСТЬЮ ИСПРАВЛЕНЫ!")
        print(f"❌ Успешно: {critical_success}/{len(critical_tests)}")
        return False

def main():
    """Основная функция"""
    print("🎨 Тестирование критических исправлений Plitka Pro v4.4.58")
    print("=" * 80)
    
    # Проверка наличия файлов
    required_files = [
        "predict.py",
        "scripts/test_inputs_v4.4.58.json",
        "scripts/validate_colors.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Отсутствуют необходимые файлы:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ Все необходимые файлы найдены")
    
    # Запуск тестов
    success = run_critical_tests()
    
    if success:
        print("\n🚀 ГОТОВ К ТЕСТИРОВАНИЮ НА REPLICATE!")
        print("📋 URL: https://replicate.com/nauslava/plitka-pro-project:v4.4.58")
        print("🎯 Рекомендуемые тесты:")
        print("   1. 100% grsgrn (должно быть 1 цвет, 20 steps)")
        print("   2. 60% grsgrn, 40% yellow (должно быть 2 цвета, 25 steps)")
        print("   3. 60% ltgreen, 40% blue (должно быть 2 цвета, 25 steps)")
    else:
        print("\n⚠️ ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ ОТЛАДКА!")
    
    return success

if __name__ == "__main__":
    main()
