#!/usr/bin/env python3
"""
Тест парсинга цветов для проверки исправлений
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from color_manager import ColorManager

def test_color_parsing():
    """Тестирует парсинг цветов"""
    print("🧪 Тестирование парсинга цветов...")
    
    # Создаем ColorManager
    color_manager = ColorManager()
    
    # Тестовые промпты
    test_prompts = [
        "70% red, 30% blue rubber tile, grid pattern",
        "60% red, 40% blue",
        "50% white, 30% gray, 20% black",
        "100% yellow",
        "25% red, 25% blue, 25% green, 25% yellow"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Промпт: {prompt}")
        
        # Тест extract_colors_from_prompt
        colors = color_manager.extract_colors_from_prompt(prompt)
        print(f"🎨 Извлеченные цвета: {colors}")
        
        # Тест валидации
        is_valid = color_manager.validate_colors(colors)
        print(f"✅ Валидация: {'ПРОЙДЕНА' if is_valid else 'НЕ ПРОЙДЕНА'}")
        
        # Проверяем, что все цвета есть в таблице
        for color in colors:
            if color not in color_manager.valid_colors:
                print(f"❌ ОШИБКА: Цвет '{color}' не найден в таблице!")
            else:
                print(f"✅ Цвет '{color}' найден в таблице")

if __name__ == "__main__":
    test_color_parsing()
