#!/usr/bin/env python3
"""
Тест парсинга цветов с кодовыми словами
"""

from color_manager import ColorManager

def test_coded_colors():
    """Тестирует парсинг цветов с кодовыми словами"""
    print("🧪 Тестирование парсинга цветов с кодовыми словами...")
    
    # Создаем ColorManager
    color_manager = ColorManager()
    
    # Тестовые промпты с кодовыми словами
    test_prompts = [
        "ohwx_rubber_tile <s0><s1> 70% RED, 30% BLUE, grid pattern",
        "60% GRSGRN, 40% YELLOW",
        "50% WHITE, 30% GRAY, 20% BLACK",
        "100% EMERALD",
        "25% RED, 25% BLUE, 25% GRSGRN, 25% YELLOW",
        "70% red, 30% blue rubber tile",  # Старый формат для fallback
        "50% WHITE, 50% WHITE"  # Дублирование
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Тест {i}: {prompt}")
        
        # Тест extract_colors_from_prompt
        colors = color_manager.extract_colors_from_prompt(prompt)
        print(f"🎨 Извлеченные цвета: {colors}")
        
        # Тест валидации
        is_valid = color_manager.validate_colors(colors)
        print(f"✅ Валидация: {'ПРОЙДЕНА' if is_valid else 'НЕ ПРОЙДЕНА'}")
        
        # Тест количества цветов
        color_count = color_manager.get_color_count(prompt)
        print(f"📊 Количество цветов: {color_count}")
        
        # Тест парсинга процентов (симулируем как в predict.py)
        print("🔍 Тестирование парсинга процентов...")
        import re
        
        # Создаем паттерн для поиска кодовых слов цветов
        color_codes = '|'.join(color_manager.valid_colors)
        
        # Ищем паттерны: число% КОДОВОЕ_СЛОВО_ЦВЕТА
        percent_pattern = rf'(\d+(?:\.\d+)?)\s*%\s*({color_codes})\b'
        matches = re.findall(percent_pattern, prompt, re.IGNORECASE)
        
        print(f"📊 Найдено совпадений: {len(matches)}")
        for percent_str, color_code in matches:
            print(f"  ✅ {percent_str}% {color_code.upper()}")

if __name__ == "__main__":
    test_coded_colors()
