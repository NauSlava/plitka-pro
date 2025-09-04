# МУЛЬТИМОДАЛЬНЫЙ CONTROLNET для Plitka Pro v4.4.60+
# Вариант 3: Комплексный подход с несколькими типами ControlNet

import numpy as np
from PIL import Image, ImageFilter
import logging

logger = logging.getLogger(__name__)

class MultimodalControlNet:
    """Мультимодальный ControlNet с адаптивным выбором на основе сложности промпта"""
    
    def __init__(self):
        self.controlnet_types = {
            "t2i_color": "Text-to-Image Color Control",
            "color_grid": "Color Grid Pattern Control", 
            "shuffle": "Color Shuffle Control",
            "softedge": "Soft Edge Control",
            "canny": "Canny Edge Control"
        }
    
    def select_optimal_controlnet(self, color_count: int) -> list:
        """
        Выбирает оптимальную комбинацию ControlNet на основе сложности промпта
        
        Args:
            color_count: Количество цветов в промпте
            
        Returns:
            Список типов ControlNet для применения
        """
        if color_count == 1:
            return None  # Базовая модель справляется
        elif color_count == 2:
            return ["t2i_color"]  # Только цветовой контроль
        elif color_count == 3:
            return ["t2i_color", "shuffle"]  # Цвет + перемешивание
        elif color_count == 4:
            return ["t2i_color", "color_grid", "shuffle"]  # Цвет + сетка + перемешивание
        else:  # 5+ цветов
            return ["t2i_color", "color_grid", "shuffle", "softedge"]  # Полный контроль
    
    def create_control_image(self, controlnet_type: str, base_image: Image.Image, prompt: str) -> Image.Image:
        """
        Создает специализированную контрольную карту для конкретного типа ControlNet
        
        Args:
            controlnet_type: Тип ControlNet
            base_image: Базовое изображение
            prompt: Текстовый промпт
            
        Returns:
            Оптимизированная контрольная карта
        """
        try:
            if controlnet_type == "t2i_color":
                # Цветовой контроль: усиливаем цветовые различия
                return self._enhance_color_control(base_image)
            elif controlnet_type == "color_grid":
                # Сеточный контроль: создаем геометрическую структуру
                return self._create_grid_control(base_image)
            elif controlnet_type == "shuffle":
                # Контроль перемешивания: добавляем случайность
                return self._create_shuffle_control(base_image)
            elif controlnet_type == "softedge":
                # Мягкие края: сглаживаем границы
                return self._create_softedge_control(base_image)
            elif controlnet_type == "canny":
                # Canny края: четкие границы
                return self._create_canny_control(base_image)
            else:
                return base_image
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания контрольной карты для {controlnet_type}: {e}")
            return base_image
    
    def _enhance_color_control(self, image: Image.Image) -> Image.Image:
        """Усиливает цветовые различия для лучшего цветового контроля"""
        try:
            # Конвертируем в RGB для цветовых операций
            rgb_image = image.convert('RGB')
            
            # Увеличиваем насыщенность
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(rgb_image)
            enhanced = enhancer.enhance(1.5)
            
            # Увеличиваем контраст
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(1.3)
            
            # Конвертируем обратно в grayscale
            return enhanced.convert('L')
        except Exception as e:
            logger.warning(f"⚠️ Ошибка усиления цветового контроля: {e}")
            return image
    
    def _create_grid_control(self, image: Image.Image) -> Image.Image:
        """Создает геометрическую структуру для сеточного контроля"""
        try:
            # Создаем сеточный паттерн
            width, height = image.size
            grid_image = Image.new('L', (width, height), 255)
            
            # Рисуем вертикальные линии
            for x in range(0, width, 64):
                for y in range(height):
                    grid_image.putpixel((x, y), 0)
            
            # Рисуем горизонтальные линии
            for y in range(0, height, 64):
                for x in range(width):
                    grid_image.putpixel((x, y), 0)
            
            # Комбинируем с исходным изображением
            combined = Image.blend(image, grid_image, 0.3)
            return combined
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания сеточного контроля: {e}")
            return image
    
    def _create_shuffle_control(self, image: Image.Image) -> Image.Image:
        """Создает контрольную карту с элементами случайности"""
        try:
            # Добавляем случайные точки и линии
            width, height = image.size
            shuffle_image = image.copy()
            
            # Случайные точки
            for _ in range(100):
                x = np.random.randint(0, width)
                y = np.random.randint(0, height)
                intensity = np.random.randint(0, 255)
                shuffle_image.putpixel((x, y), intensity)
            
            # Случайные линии
            for _ in range(20):
                x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
                x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
                intensity = np.random.randint(0, 255)
                
                # Рисуем линию
                points = self._get_line_points(x1, y1, x2, y2)
                for px, py in points:
                    if 0 <= px < width and 0 <= py < height:
                        shuffle_image.putpixel((px, py), intensity)
            
            return shuffle_image
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания контроля перемешивания: {e}")
            return image
    
    def _create_softedge_control(self, image: Image.Image) -> Image.Image:
        """Создает контрольную карту с мягкими краями"""
        try:
            # Применяем мягкие края
            softedge = image.filter(ImageFilter.EDGE_ENHANCE)
            softedge = softedge.filter(ImageFilter.SMOOTH)
            
            # Смешиваем с исходным изображением
            combined = Image.blend(image, softedge, 0.4)
            return combined
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания мягких краев: {e}")
            return image
    
    def _create_canny_control(self, image: Image.Image) -> Image.Image:
        """Создает контрольную карту с четкими краями (Canny-like)"""
        try:
            # Применяем несколько фильтров для создания четких краев
            edge1 = image.filter(ImageFilter.FIND_EDGES)
            edge2 = image.filter(ImageFilter.EDGE_ENHANCE)
            
            # Комбинируем края
            combined = Image.blend(edge1, edge2, 0.5)
            
            # Увеличиваем контраст
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(combined)
            enhanced = enhancer.enhance(2.0)
            
            return enhanced
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания Canny контроля: {e}")
            return image
    
    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> list:
        """Алгоритм Брезенхэма для рисования линий"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx > dy:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            y = y1
            for x in range(x1, x2 + 1):
                points.append((x, y))
                if y != y2:
                    if y < y2:
                        y += 1
                    else:
                        y -= 1
        else:
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            x = x1
            for y in range(y1, y2 + 1):
                points.append((x, y))
                if x != x2:
                    if x < x2:
                        x += 1
                    else:
                        x -= 1
        
        return points
    
    def apply_multi_controlnet(self, prompt: str, controlnets: list, base_image: Image.Image) -> dict:
        """
        Применяет несколько ControlNet последовательно для максимальной точности
        
        Args:
            prompt: Текстовый промпт
            controlnets: Список типов ControlNet
            base_image: Базовое изображение
            
        Returns:
            Словарь с параметрами для pipeline
        """
        if not controlnets or not base_image:
            return None
        
        try:
            # Создаем множественные контрольные карты
            control_images = []
            for controlnet_type in controlnets:
                control_image = self.create_control_image(controlnet_type, base_image, prompt)
                control_images.append(control_image)
            
            # Создаем комбинированную контрольную карту
            combined_hint = self._create_combined_control_hint(control_images)
            
            # Применяем каждый ControlNet с соответствующими весами
            pipe_kwargs = {}
            
            # Основная контрольная карта
            pipe_kwargs["image"] = combined_hint
            
            # Настройки для разных типов ControlNet
            if "t2i_color" in controlnets:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.8
            elif "color_grid" in controlnets:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.9
            elif "shuffle" in controlnets:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.7
            elif "softedge" in controlnets:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.85
            elif "canny" in controlnets:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.75
            
            # Если несколько ControlNet, используем средний вес
            if len(controlnets) > 1:
                pipe_kwargs["controlnet_conditioning_scale"] = 0.8
            
            return pipe_kwargs
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка применения мульти ControlNet: {e}")
            return None
    
    def _create_combined_control_hint(self, control_images: list) -> Image.Image:
        """
        Создает комбинированную контрольную карту из нескольких источников
        
        Args:
            control_images: Список контрольных изображений
            
        Returns:
            Комбинированная контрольная карта
        """
        try:
            if not control_images:
                return None
            
            # Берем первое изображение как основу
            base_image = control_images[0]
            if len(control_images) == 1:
                return base_image
            
            # Комбинируем несколько контрольных карт
            combined = Image.new('L', base_image.size, 0)
            for i, img in enumerate(control_images):
                if img is not None:
                    # Нормализуем и добавляем с весами
                    weight = 1.0 / len(control_images)
                    img_array = np.array(img.convert('L')).astype(np.float32) * weight
                    combined_array = np.array(combined).astype(np.float32)
                    combined_array += img_array
                    combined = Image.fromarray(np.clip(combined_array, 0, 255).astype(np.uint8))
            
            return combined
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка создания комбинированной контрольной карты: {e}")
            return control_images[0] if control_images else None

# Пример использования
if __name__ == "__main__":
    # Инициализация логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание экземпляра мультимодального ControlNet
    mm_controlnet = MultimodalControlNet()
    
    # Пример выбора оптимального ControlNet
    color_counts = [1, 2, 3, 4, 5]
    for count in color_counts:
        selected = mm_controlnet.select_optimal_controlnet(count)
        print(f"Цветов: {count} -> ControlNet: {selected}")
    
    print("\n🎯 Мультимодальный ControlNet готов к интеграции в Plitka Pro!")
