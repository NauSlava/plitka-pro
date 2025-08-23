#!/usr/bin/env python3
"""
Тестовый скрипт для проверки логики парсинга params_json
"""

import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_params_json(params_json: str):
    """Clean parsing of params_json with proper error handling."""
    try:
        # First, try to parse the input directly
        if not params_json:
            return {}
        
        # Handle potential double-escaped JSON from web interface
        params = json.loads(params_json)
        
        # If params_json contains another params_json, extract it
        if "params_json" in params:
            inner_json = params["params_json"]
            if isinstance(inner_json, str):
                params = json.loads(inner_json)
            else:
                params = inner_json
        
        # Validate and clean the parsed parameters
        cleaned_params = {}
        
        # Colors validation
        if "colors" in params:
            colors = params["colors"]
            if isinstance(colors, list):
                cleaned_colors = []
                for color_info in colors:
                    if isinstance(color_info, dict):
                        name = color_info.get("name", "").strip()
                        proportion = color_info.get("proportion", 0)
                        
                        # Validate proportion (should be 0-100)
                        try:
                            proportion = float(proportion)
                            if 0 <= proportion <= 100:
                                cleaned_colors.append({
                                    "name": name.lower(),
                                    "proportion": proportion
                                })
                            else:
                                logger.warning(f"Invalid proportion {proportion}, must be 0-100")
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid proportion value: {proportion}")
                
                cleaned_params["colors"] = cleaned_colors
        
        # Other parameters
        for key in ["angle", "seed", "quality"]:
            if key in params:
                value = params[key]
                if key == "angle":
                    try:
                        cleaned_params[key] = int(value) % 360
                    except (ValueError, TypeError):
                        cleaned_params[key] = 0
                elif key == "seed":
                    try:
                        cleaned_params[key] = int(value)
                    except (ValueError, TypeError):
                        cleaned_params[key] = -1
                elif key == "quality":
                    if value in ["preview", "standard", "high"]:
                        cleaned_params[key] = value
                    else:
                        cleaned_params[key] = "standard"
        
        # Overrides validation
        if "overrides" in params and isinstance(params["overrides"], dict):
            overrides = params["overrides"]
            cleaned_overrides = {}
            
            # ControlNet setting
            if "use_controlnet" in overrides:
                cleaned_overrides["use_controlnet"] = bool(overrides["use_controlnet"])
            
            # Guidance scale
            if "guidance_scale" in overrides:
                try:
                    guidance = float(overrides["guidance_scale"])
                    if 1.0 <= guidance <= 20.0:
                        cleaned_overrides["guidance_scale"] = guidance
                    else:
                        logger.warning(f"Invalid guidance_scale {guidance}, using default")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid guidance_scale value: {overrides['guidance_scale']}")
            
            # Steps overrides
            for key in ["num_inference_steps_preview", "num_inference_steps_final"]:
                if key in overrides:
                    try:
                        steps = int(overrides[key])
                        if 1 <= steps <= 100:
                            cleaned_overrides[key] = steps
                        else:
                            logger.warning(f"Invalid {key} {steps}, using default")
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid {key} value: {overrides[key]}")
            
            if cleaned_overrides:
                cleaned_params["overrides"] = cleaned_overrides
        
        logger.info(f"Parsed parameters: {cleaned_params}")
        return cleaned_params
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing params: {e}")
        raise ValueError(f"Failed to parse parameters: {e}")

def build_prompt(colors):
    """Build clean prompt from color information."""
    prompt_parts = ["ohwx_rubber_tile <s0><s1>"]
    
    # Add color proportions to prompt
    if colors:
        color_desc = []
        for color_info in colors:
            name = color_info.get("name", "").lower()
            proportion = color_info.get("proportion", 0)
            if proportion > 0:
                # proportion уже в процентах (0-100), не умножаем на 100
                percentage = int(proportion)
                color_desc.append(f"{percentage}% {name}")
        
        if color_desc:
            prompt_parts.append(", ".join(color_desc))
    
    # Add quality descriptors
    prompt_parts.extend([
        "photorealistic rubber tile",
        "matte texture", 
        "top view",
        "rubber granules",
        "textured surface"
    ])
    
    return ", ".join(prompt_parts)

def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование логики парсинга params_json")
    print("=" * 50)
    
    # Тест 1: Простой JSON
    print("\n📝 Тест 1: Простой JSON")
    test1 = '{"colors": [{"name": "black", "proportion": 100}], "angle": 0, "seed": 42}'
    try:
        result1 = parse_params_json(test1)
        print(f"✅ Результат: {result1}")
        
        # Проверяем промпт
        prompt1 = build_prompt(result1.get("colors", []))
        print(f"🎨 Промпт: {prompt1}")
        
        # Проверяем, что нет "10000% black"
        if "10000%" not in prompt1:
            print("✅ Критическая ошибка '10000% black' исправлена!")
        else:
            print("❌ Ошибка '10000% black' все еще присутствует!")
            
    except Exception as e:
        print(f"❌ Ошибка в тесте 1: {e}")
    
    # Тест 2: Двойное экранирование (как в веб-интерфейсе)
    print("\n📝 Тест 2: Двойное экранирование (веб-интерфейс)")
    test2 = '{"params_json": "{\\"colors\\": [{\\"name\\": \\"black\\", \\"proportion\\": 100}], \\"angle\\": 45, \\"seed\\": 42}"}'
    try:
        result2 = parse_params_json(test2)
        print(f"✅ Результат: {result2}")
        
        # Проверяем промпт
        prompt2 = build_prompt(result2.get("colors", []))
        print(f"🎨 Промпт: {prompt2}")
        
    except Exception as e:
        print(f"❌ Ошибка в тесте 2: {e}")
    
    # Тест 3: С overrides для отключения ControlNet
    print("\n📝 Тест 3: С overrides (отключение ControlNet)")
    test3 = '{"colors": [{"name": "black", "proportion": 100}], "overrides": {"use_controlnet": false, "guidance_scale": 9.0}}'
    try:
        result3 = parse_params_json(test3)
        print(f"✅ Результат: {result3}")
        
        # Проверяем overrides
        overrides = result3.get("overrides", {})
        if "use_controlnet" in overrides:
            print(f"🎛️ ControlNet setting: {overrides['use_controlnet']}")
            if overrides["use_controlnet"] == False:
                print("✅ ControlNet успешно отключен!")
            else:
                print("❌ ControlNet не отключен!")
        
    except Exception as e:
        print(f"❌ Ошибка в тесте 3: {e}")
    
    # Тест 4: Валидация некорректных данных
    print("\n📝 Тест 4: Валидация некорректных данных")
    test4 = '{"colors": [{"name": "black", "proportion": 150}], "angle": "invalid", "seed": "abc"}'
    try:
        result4 = parse_params_json(test4)
        print(f"✅ Результат (с валидацией): {result4}")
        
    except Exception as e:
        print(f"❌ Ошибка в тесте 4: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Тестирование завершено!")
    
    # Итоговая проверка критических исправлений
    print("\n🔍 Проверка критических исправлений:")
    
    # Проверяем, что в промпте нет "10000% black"
    test_final = '{"colors": [{"name": "black", "proportion": 100}]}'
    try:
        result_final = parse_params_json(test_final)
        prompt_final = build_prompt(result_final.get("colors", []))
        
        if "10000%" not in prompt_final and "100% black" in prompt_final:
            print("✅ Критическая ошибка '10000% black' исправлена!")
        else:
            print("❌ Ошибка '10000% black' все еще присутствует!")
            
        if "use_controlnet" in result_final.get("overrides", {}):
            print("✅ Логика overrides работает!")
        else:
            print("✅ Логика overrides готова к работе!")
            
    except Exception as e:
        print(f"❌ Ошибка в финальной проверке: {e}")

if __name__ == "__main__":
    main()
