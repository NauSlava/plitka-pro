#!/usr/bin/env python3
"""
Тестовый скрипт для проверки парсера цветов
"""

def test_parse_percent_colors():
    """Тестирует парсер процентов цветов"""
    
    def _parse_percent_colors(simple_prompt: str):
        """Простенький парсер строк вида '60% red, 40% white' → список цветов и долей [0..1]."""
        parts = [p.strip() for p in simple_prompt.split(',') if p.strip()]
        result = []
        for p in parts:
            try:
                percent_str, name = p.split('%', 1)
                percent = float(percent_str.strip())
                color_name = name.strip()
                if color_name.lower().startswith(('of ', ' ')):
                    color_name = color_name.split()[-1]
                result.append({"name": color_name, "proportion": max(0.0, min(1.0, percent / 100.0))})
            except Exception:
                continue
        # Нормализация, если сумма не 1.0
        total = sum(c["proportion"] for c in result) or 1.0
        for c in result:
            c["proportion"] = c["proportion"] / total
        return result
    
    # Тестовые случаи из пресетов
    test_cases = [
        "100% red",
        "60% red, 40% white", 
        "50% red, 30% black, 20% white",
        "25% red, 25% blue, 25% green, 25% yellow",
        "50% red, 50% white",
        "50% black, 50% white",
        "10% red, 10% blue, 10% green, 10% yellow, 10% purple, 10% orange, 10% pink, 10% brown, 10% gray, 10% cyan"
    ]
    
    print("🧪 ТЕСТИРОВАНИЕ ПАРСЕРА ЦВЕТОВ")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Тест {i}: '{test_case}'")
        
        try:
            result = _parse_percent_colors(test_case)
            
            # Проверяем сумму пропорций
            total_proportion = sum(c["proportion"] for c in result)
            total_percent = total_proportion * 100
            
            print(f"  ✅ Результат парсинга:")
            for color in result:
                percent = color["proportion"] * 100
                print(f"    • {color['name']}: {percent:.1f}%")
            
            print(f"  📊 Сумма пропорций: {total_proportion:.3f} ({total_percent:.1f}%)")
            
            if abs(total_proportion - 1.0) < 0.001:
                print(f"  ✅ Сумма = 100% (корректно)")
            else:
                print(f"  ❌ Сумма ≠ 100% (ошибка!)")
                
        except Exception as e:
            print(f"  ❌ Ошибка парсинга: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")

if __name__ == "__main__":
    test_parse_percent_colors()
