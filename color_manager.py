#!/usr/bin/env python3
"""
Отдельный модуль для управления цветами без зависимости от PyTorch
"""

import logging
from typing import List, Set, Dict, Any
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColorManager:
    """Централизованное управление цветами для устранения рассинхронизации модулей"""
    
    def __init__(self):
        # Загружаем цвета из файла colors_table.txt
        self.valid_colors = self._load_colors_from_file()
        
        # Таблица соответствий русских и английских названий цветов
        self.color_table = {
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
        
        # RGB значения для всех цветов
        self.color_rgb_map = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "gray": (128, 128, 128),
            "grey": (128, 128, 128),
            "brown": (139, 69, 19),
            "orange": (255, 165, 0),
            "pink": (255, 192, 203),
            "beige": (245, 245, 220),
            "dkblue": (0, 0, 139),
            "dkgray": (64, 64, 64),
            "dkgreen": (0, 100, 0),
            "emerald": (0, 128, 0),
            "grnapl": (0, 128, 0),
            "grsgrn": (34, 139, 34),
            "khaki": (240, 230, 140),
            "lilac": (200, 162, 200),
            "limegrn": (50, 205, 50),
            "ltgray": (192, 192, 192),
            "ltgreen": (144, 238, 144),
            "pearl": (240, 248, 255),
            "salmon": (250, 128, 114),
            "sand": (244, 164, 96),
            "skyblue": (135, 206, 235),
            "tercot": (205, 92, 92),
            "turqse": (64, 224, 208),
            "violet": (238, 130, 238),
            "whtgrn": (240, 255, 240)
        }
    
    def _load_colors_from_file(self) -> Set[str]:
        """Загружает цвета из файла colors_table.txt"""
        try:
            colors = set()
            with open("colors_table.txt", "r", encoding="utf-8") as f:
                for line in f:
                    color = line.strip().upper()
                    if color:
                        colors.add(color.lower())  # Сохраняем в нижнем регистре для сравнения
            logger.info(f"✅ Загружено {len(colors)} цветов из colors_table.txt")
            return colors
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки colors_table.txt: {e}")
            # Fallback к встроенной таблице
            return {
                "beige", "black", "blue", "brown", "dkblue", "dkgray", "dkgreen",
                "emerald", "gray", "grnapl", "grsgrn", "khaki", "lilac", "limegrn",
                "ltgray", "ltgreen", "orange", "pearl", "pink", "red", "salmon",
                "sand", "skyblue", "tercot", "turqse", "violet", "white", "whtgrn", "yellow"
            }
    
    def extract_colors_from_prompt(self, prompt: str) -> List[str]:
        """Единая функция для извлечения цветов из промпта"""
        colors = []
        prompt_lower = prompt.lower()
        
        # Сначала ищем цвета по точному совпадению
        for color in self.valid_colors:
            if color in prompt_lower:
                # Проверяем, что это отдельное слово, а не часть другого слова
                pattern = r'\b' + re.escape(color) + r'\b'
                if re.search(pattern, prompt_lower):
                    colors.append(color)
        
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_colors = []
        for color in colors:
            if color not in seen:
                seen.add(color)
                unique_colors.append(color)
        
        return unique_colors
    
    def get_color_rgb(self, color_name: str) -> tuple:
        """Получение RGB значения для цвета"""
        return self.color_rgb_map.get(color_name.lower(), (127, 127, 127))
    
    def validate_colors(self, colors: List[str]) -> bool:
        """Валидация списка цветов"""
        return all(color.lower() in self.valid_colors for color in colors)
    
    def get_color_count(self, prompt: str) -> int:
        """Получение количества цветов в промпте"""
        return len(self.extract_colors_from_prompt(prompt))
