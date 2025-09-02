#!/usr/bin/env python3
"""
Скрипт валидации названий цветов для проекта Plitka Pro
Проверяет корректность названий цветов согласно таблице соответствий
"""

import json
import re
import os
from typing import List, Dict, Set

# Таблица соответствий русских и английских названий цветов
COLOR_TABLE = {
    "Бежевый": "BEIGE",
    "Бело-зеленый": "WHTGRN", 
    "Белый": "WHITE",
    "Бирюзовый": "TURQSE",
    "Голубой": "SKYBLUE",
    "Желтый": "YELLOW",
    "Жемчужный": "PEARL",
    "Зеленая трава": "GRSGRN",
    "Зеленое яблоко": "GRNAPL",
    "Изумрудный": "EMERALD",
    "Коричневый": "BROWN",
    "Красный": "RED",
    "Лосось": "SALMON",
    "Оранжевый": "ORANGE",
    "Песочный": "SAND",
    "Розовый": "PINK",
    "Салатовый": "LIMEGRN",
    "Светло-зеленый": "LTGREEN",
    "Светло-серый": "LTGRAY",
    "Серый": "GRAY",
    "Синий": "BLUE",
    "Сиреневый": "LILAC",
    "Темно-зеленый": "DKGREEN",
    "Темно-серый": "DKGRAY",
    "Темно-синий": "DKBLUE",
    "Терракот": "TERCOT",
    "Фиолетовый": "VIOLET",
    "Хаки": "KHAKI",
    "Чёрный": "BLACK"
}

# Допустимые названия цветов (в нижнем регистре)
VALID_COLORS = {color.lower() for color in COLOR_TABLE.values()}

def extract_colors_from_prompt(prompt: str) -> List[str]:
    """Извлекает названия цветов из промпта"""
    # Паттерн для поиска цветов: "XX% color_name" или "color_name"
    color_pattern = r'(\d+)%\s+([a-zA-Z]+)'
    matches = re.findall(color_pattern, prompt)
    
    colors = []
    for percent, color_name in matches:
        colors.append(color_name.lower().strip())
    
    return colors

def validate_presets_file(file_path: str) -> Dict[str, List[str]]:
    """Валидирует файл пресетов"""
    print(f"🔍 Валидация файла: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            presets = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return {}
    
    errors = []
    warnings = []
    valid_presets = 0
    
    for preset_name, preset_data in presets.items():
        if "prompt" not in preset_data:
            continue
            
        prompt = preset_data["prompt"]
        colors = extract_colors_from_prompt(prompt)
        
        # Проверка каждого цвета
        for color in colors:
            if color not in VALID_COLORS:
                errors.append(f"❌ {preset_name}: недопустимый цвет '{color}' в промпте '{prompt}'")
        
        # Проверка на дублирование цветов
        if len(colors) != len(set(colors)):
            warnings.append(f"⚠️ {preset_name}: дублирование цветов в промпте '{prompt}'")
        
        valid_presets += 1
    
    print(f"✅ Валидировано пресетов: {valid_presets}")
    print(f"❌ Ошибок: {len(errors)}")
    print(f"⚠️ Предупреждений: {len(warnings)}")
    
    return {
        "errors": errors,
        "warnings": warnings,
        "valid_presets": valid_presets
    }

def validate_colors_table():
    """Проверяет корректность таблицы цветов"""
    print("🔍 Валидация таблицы цветов...")
    
    # Проверка на дублирование
    english_colors = list(COLOR_TABLE.values())
    if len(english_colors) != len(set(english_colors)):
        print("❌ Обнаружены дублирующиеся английские названия цветов")
        return False
    
    # Проверка на пустые значения
    for russian, english in COLOR_TABLE.items():
        if not english or not russian:
            print(f"❌ Пустое значение: {russian} -> {english}")
            return False
    
    print(f"✅ Таблица цветов корректна: {len(COLOR_TABLE)} цветов")
    return True

def main():
    """Основная функция"""
    print("🎨 Валидация названий цветов для проекта Plitka Pro")
    print("=" * 60)
    
    # Валидация таблицы цветов
    if not validate_colors_table():
        return
    
    print("\n📋 Допустимые названия цветов:")
    for i, color in enumerate(sorted(VALID_COLORS), 1):
        print(f"  {i:2d}. {color}")
    
    print(f"\n📁 Поиск файлов пресетов...")
    
    # Поиск файлов пресетов
    scripts_dir = "scripts"
    preset_files = []
    
    if os.path.exists(scripts_dir):
        for file in os.listdir(scripts_dir):
            if file.startswith("test_inputs_") and file.endswith(".json"):
                preset_files.append(os.path.join(scripts_dir, file))
    
    if not preset_files:
        print("❌ Файлы пресетов не найдены")
        return
    
    print(f"✅ Найдено файлов пресетов: {len(preset_files)}")
    
    # Валидация каждого файла
    total_errors = 0
    total_warnings = 0
    total_presets = 0
    
    for preset_file in preset_files:
        print(f"\n{'='*60}")
        result = validate_presets_file(preset_file)
        
        if result:
            total_errors += len(result["errors"])
            total_warnings += len(result["warnings"])
            total_presets += result["valid_presets"]
            
            # Вывод ошибок
            for error in result["errors"]:
                print(error)
            
            # Вывод предупреждений
            for warning in result["warnings"]:
                print(warning)
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ:")
    print(f"✅ Всего пресетов: {total_presets}")
    print(f"❌ Всего ошибок: {total_errors}")
    print(f"⚠️ Всего предупреждений: {total_warnings}")
    
    if total_errors == 0:
        print("🎉 Все названия цветов корректны!")
    else:
        print("❌ Обнаружены ошибки в названиях цветов")

if __name__ == "__main__":
    main()
