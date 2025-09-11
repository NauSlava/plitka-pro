#!/usr/bin/env python3
"""
Тест с реальным промптом из логов
"""

from color_manager import ColorManager

def test_real_prompt():
    """Тестирует парсинг реального промпта из логов"""
    print("🧪 Тестирование реального промпта из логов...")
    
    # Создаем ColorManager
    color_manager = ColorManager()
    
    # Реальный промпт из логов
    real_prompt = "ohwx_rubber_tile <s0><s1> 70% red, 30% blue rubber tile, grid pattern, precise color control"
    
    print(f"📝 Реальный промпт: {real_prompt}")
    
    # Тест extract_colors_from_prompt
    colors = color_manager.extract_colors_from_prompt(real_prompt)
    print(f"🎨 Извлеченные цвета: {colors}")
    
    # Тест валидации
    is_valid = color_manager.validate_colors(colors)
    print(f"✅ Валидация: {'ПРОЙДЕНА' if is_valid else 'НЕ ПРОЙДЕНА'}")
    
    # Проверяем количество цветов
    color_count = color_manager.get_color_count(real_prompt)
    print(f"📊 Количество цветов: {color_count}")
    
    # Тестируем парсинг процентов
    print("\n🔍 Тестирование парсинга процентов...")
    
    # Симулируем парсинг как в _parse_percent_colors
    parts = [p.strip() for p in real_prompt.split(',') if p.strip()]
    print(f"📝 Части промпта: {parts}")
    
    for p in parts:
        if '%' in p:
            try:
                percent_str, name = p.split('%', 1)
                percent = float(percent_str.strip())
                color_name = name.strip()
                print(f"  📊 {percent}% - '{color_name}'")
                
                # Улучшенный парсинг названия цвета
                words = color_name.lower().split()
                found_color = None
                for word in words:
                    if word in color_manager.valid_colors:
                        found_color = word.upper()
                        break
                
                if found_color:
                    print(f"    ✅ Найден цвет: {found_color}")
                else:
                    print(f"    ❌ Цвет не найден в таблице: {color_name}")
                    
            except Exception as e:
                print(f"    ❌ Ошибка парсинга '{p}': {e}")

if __name__ == "__main__":
    test_real_prompt()
