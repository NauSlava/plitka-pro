# predict.py - Основной файл предсказания для модели "nauslava/plitka-pro-project:v4.4.56"
# Использует НАШУ обученную модель с LoRA и Textual Inversion (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)

import os
import torch
import random
import gc
import json
import logging
import time
import math
from typing import Optional, List, Dict, Any, Iterator
from PIL import Image, ImageDraw, ImageColor
import numpy as np
from pathlib import Path

# Добавляем импорты для Color Grid Adapter
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

# Импортируем ColorManager из отдельного модуля
from color_manager import ColorManager

class ColorGridControlNet:
    """Улучшенный Color Grid Adapter для точного контроля цветовых пропорций"""
    
    def __init__(self):
        self.patterns = ["random", "grid", "radial", "granular"]
        # Динамические размеры гранул на основе калибровки с референсными изображениями
        self.granule_sizes = {
            "small": {"min_size": 2, "max_size": 4, "density": 0.9, "variation": 0.3},
            "medium": {"min_size": 3, "max_size": 6, "density": 0.8, "variation": 0.4},
            "large": {"min_size": 5, "max_size": 8, "density": 0.7, "variation": 0.5}
        }
        
        # Калиброванные параметры для реалистичных гранул
        self.granule_calibration = {
            "min_granule_size": 2,  # Минимальный размер в пикселях
            "max_granule_size": 8,  # Максимальный размер в пикселях
            "size_variation": 0.4,  # Вариативность размера (0.0-1.0)
            "form_complexity": 0.6,  # Сложность формы (0.0-1.0)
            "organic_factor": 0.7   # Фактор органичности (0.0-1.0)
        }
        
        # Инициализация централизованного менеджера цветов
        self.color_manager = ColorManager()
    
    def create_optimized_colormap(self, colors, size=(1024, 1024), 
                                 pattern_type="granular", granule_size="medium"):
        """Создает оптимизированный colormap для ControlNet"""
        
        if pattern_type == "granular":
            return self._create_granular_pattern(colors, size, granule_size)
        elif pattern_type == "random":
            return self._create_random_pattern(colors, size)
        elif pattern_type == "grid":
            return self._create_grid_pattern(colors, size)
        elif pattern_type == "radial":
            return self._create_radial_pattern(colors, size)
        else:
            return self._create_granular_pattern(colors, size, "medium")
    
    def _create_granular_pattern(self, colors, size, granule_size="medium"):
        """Создает паттерн, имитирующий резиновую крошку с пустыми полями по краям"""
        width, height = size
        canvas = Image.new('RGBA', size, (255, 255, 255, 0))  # Прозрачный фон
        pixels = canvas.load()
        
        # Вычисляем размеры пустых полей (2-3% от общего размера)
        margin_x = int(width * 0.025)  # 2.5% по горизонтали
        margin_y = int(height * 0.025)  # 2.5% по вертикали
        
        # Рабочая область (без полей)
        work_width = width - 2 * margin_x
        work_height = height - 2 * margin_y
        
        # Параметры гранул с динамическим диапазоном
        granule_params = self.granule_sizes[granule_size]
        min_size = granule_params["min_size"]
        max_size = granule_params["max_size"]
        density = granule_params["density"]
        variation = granule_params["variation"]
        
        # Калиброванные параметры для реалистичности
        calibration = self.granule_calibration
        
        # Нормализация пропорций для рабочей области
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        normalized_colors = []
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            normalized_colors.append({
                "color": color_rgb,
                "proportion": proportion,
                "pixels_needed": int(proportion * work_width * work_height * density)
            })
        
        # Создание гранул с вариативными размерами и органическими формами
        pixels_placed = {i: 0 for i in range(len(normalized_colors))}
        
        # Создаем список позиций только в рабочей области и перемешиваем
        all_positions = [(x + margin_x, y + margin_y) for x in range(work_width) for y in range(work_height)]
        random.shuffle(all_positions)
        
        pos_idx = 0
        for color_idx, color_info in enumerate(normalized_colors):
            pixels_to_place = color_info["pixels_needed"]
            placed = 0
            
            while placed < pixels_to_place and pos_idx < len(all_positions):
                x, y = all_positions[pos_idx]
                pos_idx += 1
                
                # Проверяем, что пиксель еще прозрачный
                if pixels[x, y] == (255, 255, 255, 0):
                    # Создаем гранулу с вариативным размером
                    granule_size = self._generate_variable_granule_size(min_size, max_size, variation, calibration)
                    
                    # Создаем органическую форму гранулы
                    actual_pixels_placed = self._draw_organic_granule(pixels, x, y, granule_size, color_info["color"], 
                                                                    work_width, work_height, margin_x, margin_y, calibration)
                    
                    placed += actual_pixels_placed  # Учитываем фактически размещенные пиксели
        
        return canvas
    
    def _generate_variable_granule_size(self, min_size: int, max_size: int, variation: float, calibration: dict) -> int:
        """Генерирует вариативный размер гранулы на основе калиброванных параметров"""
        import random
        
        # Базовый размер с вариацией
        base_size = random.randint(min_size, max_size)
        
        # Применяем вариацию для создания неоднородности
        variation_factor = 1.0 + (random.random() - 0.5) * variation * 2
        variable_size = int(base_size * variation_factor)
        
        # Ограничиваем размер калиброванными параметрами
        variable_size = max(calibration["min_granule_size"], 
                           min(calibration["max_granule_size"], variable_size))
        
        return variable_size
    
    def _draw_organic_granule(self, pixels, center_x: int, center_y: int, size: int, color: tuple, 
                             work_width: int, work_height: int, margin_x: int, margin_y: int, calibration: dict) -> int:
        """Рисует органическую гранулу с неправильной формой и возвращает количество размещенных пикселей"""
        import random
        import math
        
        # Параметры органичности
        organic_factor = calibration["organic_factor"]
        form_complexity = calibration["form_complexity"]
        
        # Определяем форму гранулы на основе сложности
        if random.random() < form_complexity:
            # Сложная органическая форма
            return self._draw_complex_organic_granule(pixels, center_x, center_y, size, color, 
                                                    work_width, work_height, margin_x, margin_y, organic_factor)
        else:
            # Простая форма с небольшими искажениями
            return self._draw_simple_organic_granule(pixels, center_x, center_y, size, color, 
                                                   work_width, work_height, margin_x, margin_y, organic_factor)
    
    def _draw_simple_organic_granule(self, pixels, center_x: int, center_y: int, size: int, color: tuple,
                                   work_width: int, work_height: int, margin_x: int, margin_y: int, organic_factor: float) -> int:
        """Рисует простую органическую гранулу с небольшими искажениями и возвращает количество размещенных пикселей"""
        import random
        import math
        
        # Базовый радиус с небольшими вариациями
        base_radius = size // 2
        pixels_placed = 0
        
        for dx in range(-base_radius, base_radius + 1):
            for dy in range(-base_radius, base_radius + 1):
                x, y = center_x + dx, center_y + dy
                
                # Проверяем границы
                if (margin_x <= x < work_width + margin_x and 
                    margin_y <= y < work_height + margin_y):
                    
                    # Вычисляем расстояние от центра
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # Добавляем органические искажения
                    organic_distortion = 1.0 + (random.random() - 0.5) * organic_factor * 0.3
                    effective_radius = base_radius * organic_distortion
                    
                    # Рисуем пиксель, если он внутри искаженного круга
                    if distance <= effective_radius:
                        pixels[x, y] = color
                        pixels_placed += 1
        
        return pixels_placed
    
    def _draw_complex_organic_granule(self, pixels, center_x: int, center_y: int, size: int, color: tuple,
                                    work_width: int, work_height: int, margin_x: int, margin_y: int, organic_factor: float) -> int:
        """Рисует сложную органическую гранулу с неправильной формой и возвращает количество размещенных пикселей"""
        import random
        import math
        
        # Создаем несколько точек для сложной формы
        num_points = max(3, size // 2)
        points = []
        
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            # Добавляем случайные отклонения для органичности
            radius_variation = 1.0 + (random.random() - 0.5) * organic_factor
            point_radius = (size // 2) * radius_variation
            
            px = center_x + int(point_radius * math.cos(angle))
            py = center_y + int(point_radius * math.sin(angle))
            points.append((px, py))
        
        # Рисуем гранулу, используя точки как основу для формы
        pixels_placed = 0
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                x, y = center_x + dx, center_y + dy
                
                # Проверяем границы
                if (margin_x <= x < work_width + margin_x and 
                    margin_y <= y < work_height + margin_y):
                    
                    # Проверяем, находится ли точка внутри сложной формы
                    if self._point_in_complex_shape(x, y, points, organic_factor):
                        pixels[x, y] = color
                        pixels_placed += 1
        
        return pixels_placed
    
    def _point_in_complex_shape(self, x: int, y: int, points: list, organic_factor: float) -> bool:
        """Проверяет, находится ли точка внутри сложной органической формы"""
        import random
        import math
        
        # Используем алгоритм ray casting для проверки принадлежности к многоугольнику
        n = len(points)
        inside = False
        
        p1x, p1y = points[0]
        for i in range(1, n + 1):
            p2x, p2y = points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        # Добавляем органические искажения
        if inside and random.random() < organic_factor * 0.2:
            inside = not inside  # Случайно исключаем некоторые пиксели
        
        return inside
    
    def _create_random_pattern(self, colors, size):
        """Создает случайный паттерн с точными пропорциями и пустыми полями по краям"""
        width, height = size
        canvas = Image.new('RGBA', size, (255, 255, 255, 0))  # Прозрачный фон
        pixels = canvas.load()
        
        # Вычисляем размеры пустых полей (2-3% от общего размера)
        margin_x = int(width * 0.025)  # 2.5% по горизонтали
        margin_y = int(height * 0.025)  # 2.5% по вертикали
        
        # Рабочая область (без полей)
        work_width = width - 2 * margin_x
        work_height = height - 2 * margin_y
        
        # Нормализация пропорций для рабочей области
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        color_pixels = {}
        
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            color_pixels[color_rgb] = int(proportion * work_width * work_height)
        
        # Случайное размещение пикселей в рабочей области
        all_positions = [(x + margin_x, y + margin_y) for x in range(work_width) for y in range(work_height)]
        random.shuffle(all_positions)
        
        pos_idx = 0
        for color_rgb, pixel_count in color_pixels.items():
            for _ in range(pixel_count):
                if pos_idx < len(all_positions):
                    x, y = all_positions[pos_idx]
                    pixels[x, y] = color_rgb
                    pos_idx += 1
        
        return canvas
    
    def _create_grid_pattern(self, colors, size):
        """Создает сеточный паттерн с точечным распределением и пустыми полями по краям"""
        width, height = size
        canvas = Image.new('RGBA', size, (255, 255, 255, 0))  # Прозрачный фон
        pixels = canvas.load()
        
        # Вычисляем размеры пустых полей (2-3% от общего размера)
        margin_x = int(width * 0.025)  # 2.5% по горизонтали
        margin_y = int(height * 0.025)  # 2.5% по вертикали
        
        # Рабочая область (без полей)
        work_width = width - 2 * margin_x
        work_height = height - 2 * margin_y
        
        # Нормализация пропорций для рабочей области
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        
        # Создание точечного распределения в рабочей области
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            pixels_needed = int(proportion * work_width * work_height)
            
            # Случайное размещение точек в рабочей области
            positions_placed = 0
            attempts = 0
            max_attempts = pixels_needed * 10  # Ограничение попыток
            
            while positions_placed < pixels_needed and attempts < max_attempts:
                x = random.randint(margin_x, margin_x + work_width - 1)
                y = random.randint(margin_y, margin_y + work_height - 1)
                
                # Проверяем, что позиция свободна
                if pixels[x, y] == (255, 255, 255, 0):
                    pixels[x, y] = color_rgb
                    positions_placed += 1
                
                attempts += 1
        
        return canvas
    
    def _create_radial_pattern(self, colors, size):
        """Создает радиальный паттерн с точечным распределением и пустыми полями по краям"""
        width, height = size
        canvas = Image.new('RGBA', size, (255, 255, 255, 0))  # Прозрачный фон
        pixels = canvas.load()
        
        # Вычисляем размеры пустых полей (2-3% от общего размера)
        margin_x = int(width * 0.025)  # 2.5% по горизонтали
        margin_y = int(height * 0.025)  # 2.5% по вертикали
        
        # Рабочая область (без полей)
        work_width = width - 2 * margin_x
        work_height = height - 2 * margin_y
        
        # Центр рабочей области
        center_x = margin_x + work_width // 2
        center_y = margin_y + work_height // 2
        max_radius = min(work_width, work_height) // 2
        
        # Нормализация пропорций для рабочей области
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        
        # Создание точечного распределения с радиальным влиянием в рабочей области
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            pixels_needed = int(proportion * work_width * work_height)
            
            # Случайное размещение точек с радиальным приоритетом
            positions_placed = 0
            attempts = 0
            max_attempts = pixels_needed * 10  # Ограничение попыток
            
            while positions_placed < pixels_needed and attempts < max_attempts:
                # Генерируем позицию с радиальным распределением
                angle = random.uniform(0, 2 * 3.14159)  # 0 до 2π
                radius = random.uniform(0, max_radius)
                
                x = int(center_x + radius * math.cos(angle))
                y = int(center_y + radius * math.sin(angle))
                
                # Проверяем границы рабочей области
                if (margin_x <= x < margin_x + work_width and 
                    margin_y <= y < margin_y + work_height):
                    # Проверяем, что позиция свободна
                    if pixels[x, y] == (255, 255, 255, 0):
                        pixels[x, y] = color_rgb
                        positions_placed += 1
                
                attempts += 1
        
        return canvas
    
    def _name_to_rgb(self, color_name):
        """Преобразует название цвета в RGB через ColorManager"""
        # Используем централизованную систему цветов
        return self.color_manager.get_color_rgb(color_name)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Единая версия модели для логов
MODEL_VERSION = "v4.5.06"

# Переменные окружения для оптимизации
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_MODELS_CACHE"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE_MIGRATION_DISABLE"] = "1"
os.environ["HF_HUB_CACHE_MIGRATION_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
except Exception:
    ControlNetModel = None
    StableDiffusionXLControlNetPipeline = None
from transformers import CLIPTextModel, T5EncoderModel
from cog import BasePredictor, Input

# Специальные исключения для критических ошибок
class ColormapGenerationError(Exception):
    """Критическая ошибка генерации colormap"""
    pass

class ControlNetValidationError(Exception):
    """Критическая ошибка валидации ControlNet"""
    pass

class Predictor(BasePredictor):
    def __init__(self):
        self.device = None
        self.pipe = None
        self.controlnet = None
        self.pipe_cn = None
        
        # Инициализация Color Grid Adapter
        self.color_grid_adapter = ColorGridControlNet()
        logger.info("🎨 Color Grid Adapter инициализирован")
        
        # Инициализация централизованного менеджера цветов
        self.color_manager = ColorManager()
        logger.info("🎨 Color Manager инициализирован")
        
        # Статистика использования Color Grid Adapter
        self.color_grid_stats = {
            "total_generations": 0,
            "controlnet_used": 0,
            "patterns_used": {"random": 0, "grid": 0, "radial": 0, "granular": 0},
            "granule_sizes_used": {"small": 0, "medium": 0, "large": 0}
        }
    
    def setup(self):
        """Инициализация модели при запуске сервера."""
        logger.info(f"🚀 Инициализация модели {MODEL_VERSION} (Color Grid Adapter + ControlNet Integration)...")
        
        # 1. Определение устройства
        if torch.cuda.is_available():
            self.device = "cuda"
            # Выбор GPU с наибольшей памятью
            best_gpu = max(range(torch.cuda.device_count()), 
                          key=lambda i: torch.cuda.get_device_properties(i).total_memory)
            torch.cuda.set_device(best_gpu)
            logger.info(f"✅ Используется GPU: {torch.cuda.get_device_name(best_gpu)}")
            logger.info(f"📊 Память GPU: {torch.cuda.get_device_properties(best_gpu).total_memory / 1024**3:.1f} GB")
        else:
            self.device = "cpu"
            logger.info("⚠️ CUDA недоступен, используется CPU")
        
        # 2. Загрузка SDXL pipeline
        logger.info("📥 Загрузка базовой модели SDXL...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            resume_download=False
        )
        
        # 3. Перемещение на GPU
        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
        
        # 4. Загрузка НАШИХ обученных LoRA (как в успешной модели v45)
        logger.info("🔧 Загрузка НАШИХ LoRA адаптеров (метод v45)...")
        lora_path = "/src/model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
        try:
            # Используем метод v45: set_adapters + fuse_lora для максимальной эффективности
            try:
                # Совместимость с новыми версиями diffusers
                self.pipe.set_adapters(["rubber-tile-lora-v4"], adapter_weights=[0.7])
                self.pipe.fuse_lora()
                logger.info("✅ LoRA адаптеры загружены через set_adapters + fuse_lora (метод v45)")
            except Exception as e1:
                logger.warning(f"⚠️ set_adapters не сработал: {e1}")
                try:
                    # Fallback: загружаем с именем адаптера
                    self.pipe.load_lora_weights(lora_path, adapter_name="rt")
                    logger.info("✅ LoRA адаптеры загружены через load_lora_weights (fallback)")
                except Exception as e2:
                    logger.warning(f"⚠️ load_lora_weights с adapter_name не сработал: {e2}")
                    # Final fallback: простая загрузка
                    self.pipe.load_lora_weights(lora_path)
                    logger.info("✅ LoRA адаптеры загружены через load_lora_weights (final fallback)")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка загрузки LoRA: {e}")
            raise e
        
        # 5. ДЕТАЛЬНАЯ ДИАГНОСТИКА РАЗМЕРОВ SDXL
        logger.info("🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА РАЗМЕРОВ SDXL...")
        
        # Проверяем размеры text_encoder ДО добавления токенов
        logger.info("📊 АНАЛИЗ РАЗМЕРОВ ДО ДОБАВЛЕНИЯ ТОКЕНОВ:")
        
        # text_encoder (первый)
        emb_1 = self.pipe.text_encoder.get_input_embeddings()
        logger.info(f"🔍 text_encoder.get_input_embeddings().weight.shape: {emb_1.weight.shape}")
        logger.info(f"🔍 text_encoder.config.hidden_size: {self.pipe.text_encoder.config.hidden_size}")
        logger.info(f"🔍 text_encoder.config.vocab_size: {self.pipe.text_encoder.config.vocab_size}")
        
        # text_encoder_2 (второй)
        emb_2 = self.pipe.text_encoder_2.get_input_embeddings()
        logger.info(f"🔍 text_encoder_2.get_input_embeddings().weight.shape: {emb_2.weight.shape}")
        logger.info(f"🔍 text_encoder_2.config.hidden_size: {self.pipe.text_encoder_2.config.hidden_size}")
        logger.info(f"🔍 text_encoder_2.config.vocab_size: {self.pipe.text_encoder_2.config.vocab_size}")
        
        # Проверяем размеры токенизаторов
        logger.info(f"🔍 tokenizer.vocab_size: {self.pipe.tokenizer.vocab_size}")
        logger.info(f"🔍 tokenizer_2.vocab_size: {self.pipe.tokenizer_2.vocab_size}")
        
        # 6. Загрузка НАШИХ обученных Textual Inversion (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)
        logger.info("🔤 Загрузка НАШИХ Textual Inversion (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)...")
        ti_path = "/src/model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors"
        try:
            # Ручная загрузка dual-encoder Textual Inversion для SDXL
            try:
                from safetensors.torch import load_file
                state_dict = load_file(ti_path)
                logger.info("✅ Файл загружен через safetensors.load_file")
            except ImportError:
                logger.warning("⚠️ safetensors не найден, используем torch.load")
                try:
                    state_dict = torch.load(ti_path, map_location="cpu")
                    logger.info("✅ Файл загружен через torch.load")
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки torch.load: {e}")
                    raise e
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки safetensors: {e}")
                raise e
            
            # Проверяем структуру загруженного файла
            if not isinstance(state_dict, dict):
                logger.error("❌ Загруженный файл не является словарем")
                raise ValueError("Invalid file format")
            
            logger.info(f"📊 Структура state_dict: {list(state_dict.keys())}")
            
            # ДЕТАЛЬНЫЙ АНАЛИЗ РАЗМЕРОВ ЭМБЕДДИНГОВ
            logger.info("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ РАЗМЕРОВ ЭМБЕДДИНГОВ:")
            
            if 'clip_g' in state_dict:
                embeddings_0 = state_dict['clip_g']
                logger.info(f"📊 clip_g (embeddings_0) размер: {embeddings_0.shape}")
                logger.info(f"🔍 clip_g размерность 0: {embeddings_0.shape[0]}")
                logger.info(f"🔍 clip_g размерность 1: {embeddings_0.shape[1]}")
                
                # Проверяем совместимость с text_encoder_2 (ИСПРАВЛЕНИЕ!)
                emb_2_hidden_size = self.pipe.text_encoder_2.config.hidden_size
                logger.info(f"🔍 text_encoder_2.config.hidden_size: {emb_2_hidden_size}")
                logger.info(f"🔍 Совместимость clip_g с text_encoder_2: {embeddings_0.shape[1]} == {emb_2_hidden_size}")
                
            if 'clip_l' in state_dict:
                embeddings_1 = state_dict['clip_l']
                logger.info(f"📊 clip_l (embeddings_1) размер: {embeddings_1.shape}")
                logger.info(f"🔍 clip_l размерность 0: {embeddings_1.shape[0]}")
                logger.info(f"🔍 clip_l размерность 1: {embeddings_1.shape[1]}")
                
                # Проверяем совместимость с text_encoder (ИСПРАВЛЕНИЕ!)
                emb_1_hidden_size = self.pipe.text_encoder.config.hidden_size
                logger.info(f"🔍 text_encoder.config.hidden_size: {emb_1_hidden_size}")
                logger.info(f"🔍 Совместимость clip_l с text_encoder: {embeddings_1.shape[1]} == {emb_1_hidden_size}")
            
            # Добавление новых токенов в токенизаторы
            logger.info("🔤 Добавление новых токенов в токенизаторы...")
            self.pipe.tokenizer.add_tokens(["<s0>", "<s1>"])
            self.pipe.tokenizer_2.add_tokens(["<s0>", "<s1>"])
            
            # Получение ID токенов и проверка границ
            token_ids_one = self.pipe.tokenizer.convert_tokens_to_ids(["<s0>", "<s1>"])
            token_ids_two = self.pipe.tokenizer_2.convert_tokens_to_ids(["<s0>", "<s1>"])
            
            logger.info(f"🔤 ID токенов для tokenizer: {token_ids_one}")
            logger.info(f"🔤 ID токенов для tokenizer_2: {token_ids_two}")
            
            # Проверка размеров embedding слоев ПОСЛЕ добавления токенов
            logger.info("📊 АНАЛИЗ РАЗМЕРОВ ПОСЛЕ ДОБАВЛЕНИЯ ТОКЕНОВ:")
            
            emb_one_size = self.pipe.text_encoder.get_input_embeddings().weight.shape[0]
            emb_two_size = self.pipe.text_encoder_2.get_input_embeddings().weight.shape[0]
            
            logger.info(f"📊 Размер embedding слоя text_encoder: {emb_one_size}")
            logger.info(f"📊 Размер embedding слоя text_encoder_2: {emb_two_size}")
            
            # Проверка необходимости изменения размера
            max_id_one = max(token_ids_one)
            max_id_two = max(token_ids_two)
            
            if max_id_one >= emb_one_size:
                logger.info(f"🔧 Изменение размера embedding слоя text_encoder с {emb_one_size} на {max_id_one + 1}")
                self.pipe.text_encoder.resize_token_embeddings(len(self.pipe.tokenizer))
                emb_one_size = self.pipe.text_encoder.get_input_embeddings().weight.shape[0]
                logger.info(f"✅ Новый размер embedding слоя text_encoder: {emb_one_size}")
            
            if max_id_two >= emb_two_size:
                logger.info(f"🔧 Изменение размера embedding слоя text_encoder_2 с {emb_two_size} на {max_id_two + 1}")
                self.pipe.text_encoder_2.resize_token_embeddings(len(self.pipe.tokenizer_2))
                emb_two_size = self.pipe.text_encoder_2.get_input_embeddings().weight.shape[0]
                logger.info(f"✅ Новый размер embedding слоя text_encoder_2: {emb_two_size}")
            
            # ПОПЫТКА ЗАГРУЗКИ С ПРОВЕРКОЙ СОВМЕСТИМОСТИ (ИСПРАВЛЕНИЕ!)
            logger.info("🔧 ПОПЫТКА ЗАГРУЗКИ ЭМБЕДДИНГОВ С ПРОВЕРКОЙ СОВМЕСТИМОСТИ (ИСПРАВЛЕНИЕ!):")
            
            # ИСПРАВЛЕНИЕ: Загружаем clip_g (1280) в text_encoder_2 (1280)
            if 'clip_g' in state_dict:
                embeddings_0 = state_dict['clip_g']
                logger.info(f"📊 Размер embeddings_0 (clip_g): {embeddings_0.shape}")
                
                # Проверка совместимости размеров с text_encoder_2
                emb_2_hidden_size = self.pipe.text_encoder_2.config.hidden_size
                if embeddings_0.shape[1] == emb_2_hidden_size:
                    logger.info(f"✅ clip_g совместим с text_encoder_2: {embeddings_0.shape[1]} == {emb_2_hidden_size}")
                    if embeddings_0.shape[0] >= 2 and token_ids_two[0] < emb_two_size and token_ids_two[1] < emb_two_size:
                        self.pipe.text_encoder_2.get_input_embeddings().weight.data[token_ids_two[0]] = embeddings_0[0]
                        self.pipe.text_encoder_2.get_input_embeddings().weight.data[token_ids_two[1]] = embeddings_0[1]
                        logger.info("✅ Эмбеддинги clip_g загружены в text_encoder_2 (ИСПРАВЛЕНИЕ!)")
                    else:
                        logger.error(f"❌ Несовместимость размеров: embeddings_0={embeddings_0.shape}, token_ids={token_ids_two}, emb_size={emb_two_size}")
                        raise ValueError("Embedding size mismatch")
                else:
                    logger.warning(f"⚠️ clip_g НЕ совместим с text_encoder_2: {embeddings_0.shape[1]} != {emb_2_hidden_size}")
                    logger.warning(f"⚠️ Пропускаем загрузку clip_g в text_encoder_2")
            else:
                logger.warning("⚠️ Ключ 'clip_g' не найден в state_dict")
            
            # ИСПРАВЛЕНИЕ: Загружаем clip_l (768) в text_encoder (768)
            if 'clip_l' in state_dict:
                embeddings_1 = state_dict['clip_l']
                logger.info(f"📊 Размер embeddings_1 (clip_l): {embeddings_1.shape}")
                
                # Проверка совместимости размеров с text_encoder
                emb_1_hidden_size = self.pipe.text_encoder.config.hidden_size
                if embeddings_1.shape[1] == emb_1_hidden_size:
                    logger.info(f"✅ clip_l совместим с text_encoder: {embeddings_1.shape[1]} == {emb_1_hidden_size}")
                    if embeddings_1.shape[0] >= 2 and token_ids_one[0] < emb_one_size and token_ids_one[1] < emb_one_size:
                        self.pipe.text_encoder.get_input_embeddings().weight.data[token_ids_one[0]] = embeddings_1[0]
                        self.pipe.text_encoder.get_input_embeddings().weight.data[token_ids_one[1]] = embeddings_1[1]
                        logger.info("✅ Эмбеддинги clip_l загружены в text_encoder (ИСПРАВЛЕНИЕ!)")
                    else:
                        logger.error(f"❌ Несовместимость размеров: embeddings_1={embeddings_1.shape}, token_ids={token_ids_one}, emb_size={emb_one_size}")
                        raise ValueError("Embedding size mismatch")
                else:
                    logger.warning(f"⚠️ clip_l НЕ совместим с text_encoder: {embeddings_1.shape[1]} != {emb_1_hidden_size}")
                    logger.warning(f"⚠️ Пропускаем загрузку clip_l в text_encoder")
            else:
                logger.warning("⚠️ Ключ 'clip_l' не найден в state_dict")
            
            logger.info("✅ Textual Inversion загрузка завершена (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка загрузки Textual Inversion: {e}")
            logger.error(f"📊 Детали ошибки: {type(e).__name__}: {str(e)}")
            logger.error("🔄 Продолжение без Textual Inversion (качество может быть снижено)")
            # Продолжаем без Textual Inversion, если загрузка не удалась
        
        # 7. Настройка планировщика
        logger.info("⚙️ Настройка планировщика...")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        
        # 8. Оптимизации VAE (как в успешной модели v45)
        logger.info("🚀 Применение VAE оптимизаций (метод v45)...")
        self.pipe.vae.enable_slicing()
        # Включаем tiling для лучшей детализации (как в v45)
        self.pipe.vae.enable_tiling()
        logger.info("✅ VAE tiling включен для максимальной детализации")
        try:
            # Формат каналов для ускорения и стабильности
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae.to(memory_format=torch.channels_last)
        except Exception:
            pass
        
        # 9. Очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"🎉 Модель {MODEL_VERSION} успешно инициализирована!")
    
    def _build_prompt(self, colors: List[Dict[str, Any]], angle: int) -> str:
        """Построение полного промпта с использованием НАШИХ обученных токенов (как в v45)."""
        # Базовый промпт с НАШИМИ токенами активации (как в v45)
        base_prompt = "ohwx_rubber_tile <s0><s1>"
        
        # Формирование описания цветов
        color_parts = []
        for color in colors:
            name = color["name"]
            proportion = color["proportion"]
            percentage = int(proportion * 100)
            color_parts.append(f"{percentage}% {name}")
        
        color_description = ", ".join(color_parts)
        
        # Добавление описания цветов
        full_prompt = f"{base_prompt}, {color_description}"
        
        # Добавление качественных дескрипторов
        quality_descriptors = [
            "photorealistic rubber tile",
            "high quality",
            "detailed texture",
            "professional photography",
            "sharp focus"
        ]
        
        full_prompt += ", " + ", ".join(quality_descriptors)
        
        return full_prompt
    
    def _build_negative_prompt(self) -> str:
        """Построение негативного промпта."""
        # Краткий базовый список + обязательный "object" и анти‑мозаичные токены
        return (
            "object, text, watermark, logo, signature, blur, blurry, low quality, distorted,"
            " mosaic, checkerboard, grid, patchwork, tiled, seams"
        )

    def _parse_percent_colors(self, simple_prompt: str) -> List[Dict[str, Any]]:
        """Парсер строк вида '60% RED, 40% WHITE' → список цветов и долей [0..1]."""
        import re
        
        # Создаем паттерн для поиска кодовых слов цветов
        color_codes = '|'.join(self.color_manager.valid_colors)
        
        # Ищем паттерны: число% КОДОВОЕ_СЛОВО_ЦВЕТА
        percent_pattern = rf'(\d+(?:\.\d+)?)\s*%\s*({color_codes})\b'
        matches = re.findall(percent_pattern, simple_prompt, re.IGNORECASE)
        
        result: List[Dict[str, Any]] = []
        for percent_str, color_code in matches:
            try:
                percent = float(percent_str.strip())
                color_name = color_code.upper().strip()
                
                # Валидация цвета через ColorManager
                if self.color_manager.validate_colors([color_name]):
                    result.append({"name": color_name, "proportion": max(0.0, min(1.0, percent / 100.0))})
                    logger.info(f"✅ Найден цвет: {percent}% {color_name}")
                else:
                    logger.warning(f"⚠️ Недопустимый цвет в промпте: {color_name}")
                    # Fallback: заменяем на белый
                    result.append({"name": "WHITE", "proportion": max(0.0, min(1.0, percent / 100.0))})
            except Exception as e:
                logger.warning(f"⚠️ Ошибка парсинга '{percent_str}% {color_code}': {e}")
                continue
        
        # Если не нашли кодовые слова, пробуем старый метод как fallback
        if not result:
            logger.info("🔄 Fallback: поиск цветов по словам...")
            # Ищем части с процентами в промпте
            percent_pattern = r'(\d+(?:\.\d+)?)\s*%\s*([^,]+)'
            matches = re.findall(percent_pattern, simple_prompt)
            
            for percent_str, color_phrase in matches:
                try:
                    percent = float(percent_str.strip())
                    color_name = color_phrase.strip()
                    
                    # Ищем кодовое слово в фразе
                    words = color_name.lower().split()
                    found_color = None
                    for word in words:
                        if word in self.color_manager.valid_colors:
                            found_color = word.upper()
                            break
                    
                    if found_color:
                        color_name = found_color
                        if self.color_manager.validate_colors([color_name]):
                            result.append({"name": color_name, "proportion": max(0.0, min(1.0, percent / 100.0))})
                            logger.info(f"✅ Fallback: найден цвет: {percent}% {color_name}")
                        else:
                            logger.warning(f"⚠️ Fallback: недопустимый цвет: {color_name}")
                    else:
                        logger.warning(f"⚠️ Fallback: неизвестный цвет в промпте: {color_phrase}")
                except Exception as e:
                    logger.warning(f"⚠️ Fallback: ошибка парсинга '{percent_str}% {color_phrase}': {e}")
                    continue
        
        # Нормализация, если сумма не 1.0
        total = sum(c["proportion"] for c in result) or 1.0
        for c in result:
            c["proportion"] = c["proportion"] / total
        
        logger.info(f"🎨 Итого найдено цветов: {len(result)}")
        return result

    def _render_legend(self, colors: List[Dict[str, Any]], size: int = 256) -> Image.Image:
        """Создает colormap со случайным распределением точек вместо полос (исправленный fallback)."""
        import random
        
        # Создаем прозрачный фон
        img = Image.new('RGBA', (size, size), color=(255, 255, 255, 0))
        pixels = img.load()
        
        total_pixels = size * size
        
        for color_data in colors:
            try:
                rgb = ImageColor.getrgb(color_data["name"])  # распознает стандартные цвета
            except Exception:
                rgb = (200, 200, 200)
            
            # Вычисляем количество пикселей для этого цвета
            pixels_per_color = int(total_pixels * color_data["proportion"])
            
            # Случайно размещаем пиксели этого цвета
            placed_pixels = 0
            max_attempts = pixels_per_color * 3  # Ограничиваем количество попыток
            attempts = 0
            
            while placed_pixels < pixels_per_color and attempts < max_attempts:
                x = random.randint(0, size - 1)
                y = random.randint(0, size - 1)
                
                # Проверяем, что пиксель еще не занят
                if pixels[x, y] == (255, 255, 255, 0):  # Прозрачный пиксель
                    pixels[x, y] = rgb
                    placed_pixels += 1
                
                attempts += 1
        
        return img
    
    def _build_prompt_from_simple(self, simple_prompt: str) -> str:
        """Преобразование простого промпта в полный формат с НАШИМИ токенами (как в v45)."""
        # Базовый промпт с НАШИМИ токенами активации (как в v45)
        base_prompt = "ohwx_rubber_tile <s0><s1>"
        
        # Добавление простого промпта (без дублирования base_prompt)
        if simple_prompt.startswith("ohwx_rubber_tile"):
            # Если промпт уже содержит base_prompt, не дублируем
            full_prompt = simple_prompt
        else:
            # Если промпт не содержит base_prompt, добавляем
            full_prompt = f"{base_prompt}, {simple_prompt}"
        
        # Добавление качественных дескрипторов
        quality_descriptors = [
            "photorealistic rubber tile",
            "high quality",
            "detailed texture",
            "professional photography",
            "sharp focus"
        ]
        
        full_prompt += ", " + ", ".join(quality_descriptors)
        
        return full_prompt
    
    def _strengthen_color_tokens(self, prompt: str) -> str:
        """Усиливает токены цветов в промпте для предотвращения их потери attention mechanism"""
        try:
            # Извлекаем цвета из промпта
            colors = self.color_manager.extract_colors_from_prompt(prompt)
            if not colors:
                return prompt
            
            # Создаем усиленные токены цветов
            strengthened_prompt = prompt
            
            for color_data in colors:
                color_name = color_data["name"].lower()
                proportion = color_data["proportion"]
                
                # Создаем усиленные токены для каждого цвета
                if color_name in ["red", "blue", "green", "yellow", "white", "black", "brown", "gray", "grey"]:
                    # Основные цвета - добавляем повторения и усиления
                    color_tokens = f"{color_name} {color_name} {color_name}"
                    strengthened_prompt = strengthened_prompt.replace(f"{proportion*100:.0f}% {color_name}", 
                                                                    f"{color_tokens} {proportion*100:.0f}% {color_name}")
                
                elif color_name in ["dkgreen", "ltgreen", "grngrn", "whtgrn"]:
                    # Специальные цвета - добавляем описания
                    if color_name == "dkgreen":
                        color_tokens = "dark green dark green"
                    elif color_name == "ltgreen":
                        color_tokens = "light green light green"
                    elif color_name == "grngrn":
                        color_tokens = "green green"
                    elif color_name == "whtgrn":
                        color_tokens = "white green white green"
                    
                    strengthened_prompt = strengthened_prompt.replace(f"{proportion*100:.0f}% {color_name}", 
                                                                    f"{color_tokens} {proportion*100:.0f}% {color_name}")
                
                elif color_name in ["pearl", "salmon", "orange", "pink", "violet", "turqse"]:
                    # Декоративные цвета - добавляем описания
                    if color_name == "pearl":
                        color_tokens = "pearl white pearl white"
                    elif color_name == "salmon":
                        color_tokens = "salmon pink salmon pink"
                    elif color_name == "orange":
                        color_tokens = "orange orange"
                    elif color_name == "pink":
                        color_tokens = "pink pink"
                    elif color_name == "violet":
                        color_tokens = "violet purple violet purple"
                    elif color_name == "turqse":
                        color_tokens = "turquoise blue turquoise blue"
                    
                    strengthened_prompt = strengthened_prompt.replace(f"{proportion*100:.0f}% {color_name}", 
                                                                    f"{color_tokens} {proportion*100:.0f}% {color_name}")
            
            logger.info(f"🔧 Усилены токены цветов в промпте")
            return strengthened_prompt
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка усиления токенов цветов: {e}")
            return prompt
    
    def _create_optimized_colormap(self, prompt: str, size: tuple = (1024, 1024), pattern_type: str = "random", granule_size: str = "medium") -> Image.Image:
        """Создает оптимизированный colormap для ControlNet с точными пропорциями"""
        try:
            # Парсим цвета из промпта
            colors = self._parse_percent_colors(prompt)
            if not colors:
                logger.warning("⚠️ Не удалось распарсить цвета, создаем базовый colormap")
                return Image.new('RGBA', size, (255, 255, 255, 0))  # Прозрачный фон
            
            # Если паттерн не указан, определяем оптимальный на основе количества цветов
            if pattern_type == "random":
                color_count = len(colors)
                if color_count == 1:
                    pattern_type = "random"  # Простой случай
                elif color_count == 2:
                    pattern_type = "granular"  # Имитация резиновой крошки
                elif color_count == 3:
                    pattern_type = "granular"  # Сложная крошка
                else:  # 4+ цветов
                    pattern_type = "granular"  # Максимальная сложность
            
            logger.info(f"🎨 Создание colormap: {len(colors)} цветов, паттерн: {pattern_type}, гранулы: {granule_size}")
            
            # Обновляем статистику использования
            self.color_grid_stats["patterns_used"][pattern_type] += 1
            self.color_grid_stats["granule_sizes_used"][granule_size] += 1
            
            # Создаем оптимизированный colormap
            colormap = self.color_grid_adapter.create_optimized_colormap(
                colors, size, pattern_type, granule_size
            )
            
            logger.info(f"✅ Оптимизированный colormap создан: {colormap.size}")
            logger.info(f"📊 Статистика паттернов: {self.color_grid_stats['patterns_used']}")
            logger.info(f"📊 Статистика размеров гранул: {self.color_grid_stats['granule_sizes_used']}")
            
            return colormap
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания оптимизированного colormap: {e}")
            # Fallback: простой colormap
            return self._render_legend(self._parse_percent_colors(prompt), size)
    
    def get_color_grid_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования Color Grid Adapter"""
        return {
            "total_generations": self.color_grid_stats["total_generations"],
            "controlnet_used": self.color_grid_stats["controlnet_used"],
            "controlnet_usage_percent": round(
                (self.color_grid_stats["controlnet_used"] / max(1, self.color_grid_stats["total_generations"])) * 100, 2
            ),
            "patterns_used": self.color_grid_stats["patterns_used"].copy(),
            "granule_sizes_used": self.color_grid_stats["granule_sizes_used"].copy(),
            "most_used_pattern": max(self.color_grid_stats["patterns_used"].items(), key=lambda x: x[1])[0],
            "most_used_granule_size": max(self.color_grid_stats["granule_sizes_used"].items(), key=lambda x: x[1])[0]
        }
    
    def test_color_grid_adapter(self, test_prompts: List[str] = None) -> Dict[str, Any]:
        """Тестирует Color Grid Adapter на различных промптах"""
        if test_prompts is None:
            test_prompts = [
                "100% red",
                "50% red, 50% white",
                "50% red, 30% black, 20% white",
                "25% red, 25% blue, 25% grsgrn, 25% yellow"
            ]
        
        test_results = {}
        logger.info("🧪 Начинаем тестирование Color Grid Adapter...")
        
        for prompt in test_prompts:
            try:
                logger.info(f"🧪 Тестируем промпт: {prompt}")
                
                # Создаем colormap
                colormap = self._create_optimized_colormap(prompt, size=(512, 512))
                
                # Анализируем результат
                colors = self._parse_percent_colors(prompt)
                color_count = len(colors)
                
                test_results[prompt] = {
                    "success": True,
                    "color_count": color_count,
                    "colormap_size": colormap.size,
                    "colormap_mode": colormap.mode,
                    "colors_parsed": colors
                }
                
                logger.info(f"✅ Тест пройден: {prompt}")
                
            except Exception as e:
                test_results[prompt] = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                logger.error(f"❌ Тест провален: {prompt} - {e}")
        
        logger.info("🧪 Тестирование Color Grid Adapter завершено")
        return test_results
    
    def _validate_colormap_against_prompt(self, colormap: Image, prompt: str) -> bool:
        """Валидация colormap против промпта (поддержка RGBA)"""
        try:
            expected_colors = self.color_manager.extract_colors_from_prompt(prompt)
            if not expected_colors:
                logger.warning("⚠️ Не удалось извлечь цвета из промпта")
                return False
            
            # Проверка: colormap не должен быть полностью серым или прозрачным
            colormap_array = np.array(colormap)
            
            if len(colormap_array.shape) == 3:
                # RGB изображение
                gray_pixels = np.all(colormap_array == [127, 127, 127], axis=2)
                if np.all(gray_pixels):
                    logger.warning("⚠️ Colormap полностью серый - ошибка распознавания цветов")
                    return False
            elif len(colormap_array.shape) == 4:
                # RGBA изображение - проверяем RGB каналы
                rgb_array = colormap_array[:, :, :3]  # Берем только RGB каналы
                alpha_array = colormap_array[:, :, 3]  # Альфа канал
                
                # Проверяем, что есть непрозрачные пиксели
                opaque_pixels = alpha_array > 0
                if not np.any(opaque_pixels):
                    logger.warning("⚠️ Colormap полностью прозрачный")
                    return False
                
                # Проверяем RGB каналы непрозрачных пикселей
                opaque_rgb = rgb_array[opaque_pixels]
                if len(opaque_rgb) > 0:
                    # Сравниваем каждый пиксель с серым цветом
                    gray_color = np.array([127, 127, 127])
                    # Исправляем broadcasting: сравниваем каждый пиксель с серым цветом
                    gray_pixels = np.all(opaque_rgb == gray_color, axis=1)
                    if np.all(gray_pixels):
                        logger.warning("⚠️ Colormap полностью серый в непрозрачных областях")
                        return False
            
            logger.info(f"✅ Colormap валиден для цветов: {expected_colors}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации colormap: {e}")
            return False
    
    def _validate_controlnet_map(self, colormap: Image, prompt: str) -> bool:
        """Валидация ControlNet карты перед передачей в ControlNet"""
        try:
            # Извлекаем ожидаемые цвета из промпта
            expected_colors = self.color_manager.extract_colors_from_prompt(prompt)
            if not expected_colors:
                logger.warning("⚠️ Не удалось извлечь цвета из промпта для валидации ControlNet")
                return False
            
            # Конвертируем colormap в массив для анализа
            colormap_array = np.array(colormap)
            
            # Подсчитываем уникальные цвета в colormap
            if len(colormap_array.shape) == 4:  # RGBA
                # Берем только RGB каналы и альфа
                rgb_array = colormap_array[:, :, :3]
                alpha_array = colormap_array[:, :, 3]
                
                # Находим непрозрачные пиксели
                opaque_mask = alpha_array > 128  # Порог прозрачности
                if not np.any(opaque_mask):
                    logger.warning("⚠️ ControlNet карта полностью прозрачна")
                    return False
                
                # Получаем RGB значения непрозрачных пикселей
                opaque_rgb = rgb_array[opaque_mask]
                
            elif len(colormap_array.shape) == 3:  # RGB
                opaque_rgb = colormap_array.reshape(-1, 3)
            else:
                logger.warning("⚠️ Неподдерживаемый формат ControlNet карты")
                return False
            
            # Находим уникальные цвета (с допуском на вариации)
            unique_colors = []
            for pixel in opaque_rgb:
                # Проверяем, есть ли уже похожий цвет
                is_unique = True
                for existing_color in unique_colors:
                    if np.allclose(pixel, existing_color, atol=10):  # Допуск 10 единиц
                        is_unique = False
                        break
                if is_unique:
                    unique_colors.append(pixel)
            
            # Проверяем количество уникальных цветов
            expected_count = len(expected_colors)
            actual_count = len(unique_colors)
            
            logger.info(f"🔍 ControlNet валидация: ожидается {expected_count} цветов, найдено {actual_count}")
            
            # Допускаем небольшое отклонение (например, если один цвет представлен несколькими оттенками)
            if actual_count < expected_count * 0.5:  # Менее 50% ожидаемых цветов
                logger.warning(f"⚠️ ControlNet карта содержит слишком мало цветов: {actual_count} из {expected_count}")
                return False
            
            if actual_count > expected_count * 2:  # Более чем в 2 раза больше ожидаемых
                logger.warning(f"⚠️ ControlNet карта содержит слишком много цветов: {actual_count} из {expected_count}")
                return False
            
            logger.info(f"✅ ControlNet карта валидна: {actual_count} уникальных цветов")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации ControlNet карты: {e}")
            return False
    
    def _force_rebuild_colormap(self, prompt: str, size: tuple = (1024, 1024)) -> Image:
        """Принудительная пересборка colormap при ошибках"""
        try:
            logger.info("🔧 Принудительная пересборка colormap...")
            
            # Извлекаем цвета через ColorManager
            colors = self.color_manager.extract_colors_from_prompt(prompt)
            if not colors:
                logger.error("❌ Не удалось извлечь цвета для пересборки colormap")
                # Fallback: простой серый colormap
                return Image.new('RGBA', size, (127, 127, 127, 255))  # Непрозрачный серый фон
            
            # Создаем colormap со случайным распределением точек вместо полос
            colormap = Image.new('RGBA', size, (255, 255, 255, 0))  # Прозрачный фон
            pixels = colormap.load()
            
            # Создаем случайное распределение цветных точек
            import random
            total_pixels = size[0] * size[1]
            
            for i, color in enumerate(colors):
                rgb = self.color_manager.get_color_rgb(color)
                # Вычисляем количество пикселей для этого цвета (примерно равномерно)
                pixels_per_color = total_pixels // len(colors)
                
                # Случайно размещаем пиксели этого цвета
                placed_pixels = 0
                max_attempts = pixels_per_color * 3  # Ограничиваем количество попыток
                attempts = 0
                
                while placed_pixels < pixels_per_color and attempts < max_attempts:
                    x = random.randint(0, size[0] - 1)
                    y = random.randint(0, size[1] - 1)
                    
                    # Проверяем, что пиксель еще не занят
                    if pixels[x, y] == (255, 255, 255, 0):  # Прозрачный пиксель
                        pixels[x, y] = rgb
                        placed_pixels += 1
                    
                    attempts += 1
            
            logger.info(f"✅ Colormap пересобран для цветов: {colors}")
            return colormap
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка пересборки colormap: {e}")
            # Прерываем генерацию при критических ошибках
            raise ColormapGenerationError(f"Невозможно создать корректный colormap: {e}")
    
    def predict(self, prompt: str = Input(description="Промпт для генерации резиновой плитки (должен содержать ohwx_rubber_tile <s0><s1>)", default="ohwx_rubber_tile <s0><s1> 100% red rubber tile"), 
                negative_prompt: Optional[str] = Input(description="Негативный промпт", default=""), 
                seed: int = Input(description="Seed для воспроизводимости результатов", default=12345),
                num_inference_steps: int = Input(description="Количество шагов инференса", default=25),
                guidance_scale: float = Input(description="Масштаб guidance", default=7.5),
                colormap: str = Input(description="Тип паттерна colormap", default="random"),
                granule_size: str = Input(description="Размер гранул", default="medium"),
                use_controlnet: bool = Input(description="Включить ControlNet", default=False),
                control_image: Optional[Path] = Input(description="Контрольное изображение (опц.)", default=None)) -> Iterator[Path]:
        """Генерация изображения резиновой плитки с использованием НАШЕЙ обученной модели."""
        
        try:
            # 🚀 STARTUP_SNAPSHOT_START - Гарантированное сохранение логов стартапа
            logger.info("🚀 STARTUP_SNAPSHOT_START")
            logger.info(f"🧭 MODEL_START {MODEL_VERSION} | device={self.device} | diffusers={getattr(__import__('diffusers'),'__version__', 'unknown')} | torch={torch.__version__}")
            logger.info(f"📊 GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
            logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A")
            logger.info(f"🔧 SDXL Base: {getattr(self.pipe.unet.config, 'model_name_or_path', 'unknown')}")
            logger.info(f"🎨 LoRA: {getattr(self, 'lora_path', 'unknown')} (rank: 32)")
            logger.info(f"🔤 TI: {getattr(self, 'ti_path', 'unknown')} (tokens: <s0><s1>)")
            logger.info(f"⚙️ Scheduler: {self.pipe.scheduler.__class__.__name__}")
            logger.info(f"🎭 VAE: {self.pipe.vae.__class__.__name__}")
            logger.info(f"🎯 Prompt: {prompt}")
            logger.info(f"🚫 Negative Prompt: {negative_prompt}")
            logger.info(f"🎲 Seed: {seed}")
            logger.info(f"📊 Steps: {num_inference_steps} (базовый)")
            logger.info(f"🎚️ Guidance: {guidance_scale} (базовый)")
            logger.info(f"🎨 Colormap: {colormap}")
            logger.info(f"🔧 Granule Size: {granule_size}")
            logger.info(f"🎨 Адаптивные параметры будут рассчитаны на основе количества цветов")
            logger.info("🚀 STARTUP_SNAPSHOT_END")
            
            logger.info("🎨 Начало генерации изображения...")
            logger.info(f"📝 Входной промпт: {prompt}")
            logger.info(f"🚫 Входной негативный промпт: {negative_prompt}")
            logger.info(f"🎲 Входной сид: {seed}")
            
            # ИСПРАВЛЕНИЕ: Обработка входного промпта (удаление JSON-обертки)
            if isinstance(prompt, str) and prompt.strip().startswith('{'):
                try:
                    import json
                    prompt_data = json.loads(prompt)
                    if isinstance(prompt_data, dict) and "prompt" in prompt_data:
                        prompt = prompt_data["prompt"]
                        logger.info(f"🔧 Исправлен JSON-промпт: {prompt}")
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Не удалось распарсить JSON-промпт: {prompt}")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка обработки промпта: {e}")
            
            # Упрощенный негативный промпт для лучшей генерации многоцветных плиток
            if negative_prompt is None:
                negative_prompt = self._build_negative_prompt()
                logger.info(f"🔧 Установлен стандартный негативный промпт: {negative_prompt}")
            
            # Установка сида
            if seed == -1:
                seed = random.randint(0, 999999999)
                logger.info(f"🎲 Установлен случайный сид: {seed}")
            
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Преобразование простого промпта в полный формат с НАШИМИ токенами
            full_prompt = self._build_prompt_from_simple(prompt)
            
            # Усиление токенов цветов для предотвращения потери attention mechanism
            strengthened_prompt = self._strengthen_color_tokens(full_prompt)
            
            logger.info(f"🎨 Генерация изображения...")
            logger.info(f"📝 Полный промпт: {full_prompt}")
            logger.info(f"🔧 Усиленный промпт: {strengthened_prompt}")
            logger.info(f"🚫 Финальный негативный промпт: {negative_prompt}")
            logger.info(f"🎲 Финальный сид: {seed}")
            logger.info(f"🔧 Устройство: {self.device}")
            
            # Адаптивные параметры для различного количества цветов (как в v45)
            logger.info("🎨 Анализ сложности промпта для адаптивных параметров...")
            
            # Подсчитываем количество цветов в промпте через ColorManager
            color_count = self.color_manager.get_color_count(prompt)
            logger.info(f"🎨 Обнаружено цветов в промпте: {color_count}")
            
            # Адаптивные настройки на основе количества цветов (как в v45)
            if color_count == 1:
                # Один цвет - простой промпт, как в успешном тесте 4
                adaptive_steps = max(20, num_inference_steps)
                adaptive_guidance = max(7.0, guidance_scale)
                logger.info("🎯 Адаптивные параметры для 1 цвета: steps=20, guidance=7.0")
            elif color_count == 2:
                # Два цвета - средняя сложность
                adaptive_steps = max(25, num_inference_steps)
                adaptive_guidance = max(7.5, guidance_scale)
                logger.info("🎯 Адаптивные параметры для 2 цветов: steps=25, guidance=7.5")
            elif color_count == 3:
                # Три цвета - высокая сложность
                adaptive_steps = max(30, num_inference_steps)
                adaptive_guidance = max(8.0, guidance_scale)
                logger.info("🎯 Адаптивные параметры для 3 цветов: steps=30, guidance=8.0")
            else:
                # 4+ цвета - максимальная сложность
                adaptive_steps = max(35, num_inference_steps)
                adaptive_guidance = max(8.5, guidance_scale)
                logger.info("🎯 Адаптивные параметры для 4+ цветов: steps=35, guidance=8.5")
            
            # Генерация изображения с адаптивными параметрами
            logger.info("🚀 Запуск pipeline для генерации с адаптивными параметрами...")
            pipe_to_use = self.pipe
            pipe_kwargs = dict(
                prompt=strengthened_prompt,  # Используем усиленный промпт
                negative_prompt=negative_prompt,
                num_inference_steps=max(5, int(adaptive_steps)),
                guidance_scale=float(adaptive_guidance),
                width=1024,
                height=1024,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                # LoRA уже интегрирован через fuse_lora, scale не нужен
                # cross_attention_kwargs={"scale": float(max(0.0, min(1.0, lora_scale)))}
            )

            # Настройка callback для раннего превью с адаптивными параметрами
            preview_path = "/tmp/preview.png"
            preview_emitted = False
            try_preview_step = max(1, int(adaptive_steps * 0.5))

            def _decode_and_save_preview(current_latents: torch.FloatTensor) -> None:
                nonlocal preview_emitted
                if preview_emitted:
                    return
                try:
                    scale = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
                    with torch.no_grad():
                        lat = current_latents.detach().to(self.device)
                        image = self.pipe.vae.decode(lat / scale).sample
                        image = (image / 2 + 0.5).clamp(0, 1)
                        image = image[0].permute(1, 2, 0).cpu().numpy()
                        image = (image * 255).round().astype("uint8")
                        Image.fromarray(image).resize((512, 512), Image.Resampling.LANCZOS).save(preview_path)
                    logger.info(f"🟡 PREVIEW_READY {preview_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось сохранить промежуточный preview: {e}")

            def _callback_on_step(step: int = None, timestep: int = None, **kwargs):
                nonlocal preview_emitted
                latents = kwargs.get("latents")
                if latents is None and len(kwargs) == 0:
                    # некоторые версии передают latents последним аргументом
                    pass
                if not preview_emitted and step is not None and step >= try_preview_step and latents is not None:
                    _decode_and_save_preview(latents)
                    try:
                        # стримим превью немедленно
                        yield_path = Path(preview_path)
                        # Cog поддерживает генераторный вывод через yield
                        # Внутри Cog: просто используем print маркер, GUI подтянет файл по окончании
                    except Exception:
                        pass
                    preview_emitted = True

            # МУЛЬТИМОДАЛЬНЫЙ CONTROLNET: Адаптивный выбор на основе сложности
            auto_controlnet = False
            selected_controlnets = None
            
            if not use_controlnet:
                # Используем уже подсчитанное количество цветов
                if color_count >= 2:
                    auto_controlnet = True
                    # Выбираем оптимальную комбинацию ControlNet
                    selected_controlnets = self.select_optimal_controlnet(color_count)
                    logger.info(f"🎯 Автоматически включаем мультимодальный ControlNet для {color_count} цветов: {selected_controlnets}")
            
            # Обновляем общую статистику
            self.color_grid_stats["total_generations"] += 1
            if use_controlnet or auto_controlnet:
                self.color_grid_stats["controlnet_used"] += 1
            
            # МУЛЬТИМОДАЛЬНЫЙ CONTROLNET: Инициализация и применение
            if (use_controlnet or auto_controlnet) and ControlNetModel is not None and StableDiffusionXLControlNetPipeline is not None:
                try:
                    if self.controlnet is None:
                        logger.info("🔗 Загрузка ControlNet для SDXL...")
                        self.controlnet = ControlNetModel.from_pretrained(
                            "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
                        )
                    if self.pipe_cn is None:
                        self.pipe_cn = StableDiffusionXLControlNetPipeline(
                            vae=self.pipe.vae,
                            text_encoder=self.pipe.text_encoder,
                            text_encoder_2=self.pipe.text_encoder_2,
                            tokenizer=self.pipe.tokenizer,
                            tokenizer_2=self.pipe.tokenizer_2,
                            unet=self.pipe.unet,
                            controlnet=self.controlnet,
                            scheduler=self.pipe.scheduler
                        ).to(self.device)
                    pipe_to_use = self.pipe_cn
                    
                    # МУЛЬТИМОДАЛЬНЫЙ CONTROLNET: Подготовка множественных контрольных карт
                    try:
                        if control_image is not None:
                            # Если пользователь предоставил контрольное изображение
                            user_hint = Image.open(control_image).convert('L').resize((1024, 1024), Image.Resampling.LANCZOS)
                            user_hint = user_hint.filter(ImageFilter.EDGE_ENHANCE)
                            logger.info("✅ ControlNet использует пользовательское контрольное изображение")
                            
                            # Создаем дополнительные контрольные карты для мультимодальности
                            control_images = [user_hint]
                            if selected_controlnets and len(selected_controlnets) > 1:
                                # Добавляем автоматически созданные карты для дополнительных ControlNet
                                for i in range(1, len(selected_controlnets)):
                                    additional_hint = self._create_optimized_colormap(prompt, size=(1024, 1024), pattern_type=colormap, granule_size=granule_size)
                                    additional_hint = additional_hint.convert('L').filter(ImageFilter.EDGE_ENHANCE)
                                    control_images.append(additional_hint)
                        else:
                            # Автоматически создаем множественные контрольные карты для мультимодального ControlNet
                            logger.info("🎨 Создание множественных контрольных карт для мультимодального ControlNet")
                            control_images = []
                            
                            # Основная цветовая карта
                            color_control_image = self._create_optimized_colormap(prompt, size=(1024, 1024), pattern_type=colormap, granule_size=granule_size)
                            
                            # Валидация colormap против промпта
                            if not self._validate_colormap_against_prompt(color_control_image, prompt):
                                logger.warning("⚠️ Colormap не соответствует промпту, пересоздаем...")
                                color_control_image = self._force_rebuild_colormap(prompt, size=(1024, 1024))
                            
                            # Валидация ControlNet карты перед передачей в ControlNet
                            if not self._validate_controlnet_map(color_control_image, prompt):
                                logger.warning("⚠️ ControlNet карта не прошла валидацию, пересоздаем...")
                                color_control_image = self._force_rebuild_colormap(prompt, size=(1024, 1024))
                                
                                # Повторная валидация после пересоздания
                                if not self._validate_controlnet_map(color_control_image, prompt):
                                    logger.error("❌ Критическая ошибка: ControlNet карта не может быть создана корректно")
                                    # Прерываем генерацию с ошибкой
                                    raise ControlNetValidationError("ControlNet карта не прошла валидацию после пересоздания")
                            
                            # Преобразуем в grayscale для ControlNet
                            main_hint = color_control_image.convert('L')
                            main_hint = main_hint.filter(ImageFilter.EDGE_ENHANCE)
                            control_images.append(main_hint)
                            
                            # Создаем дополнительные контрольные карты для мультимодальности
                            if selected_controlnets and len(selected_controlnets) > 1:
                                for i in range(1, len(selected_controlnets)):
                                    additional_hint = self._create_optimized_colormap(prompt, size=(1024, 1024), pattern_type=colormap, granule_size=granule_size)
                                    additional_hint = additional_hint.convert('L').filter(ImageFilter.EDGE_ENHANCE)
                                    control_images.append(additional_hint)
                        
                        # Применяем мультимодальный ControlNet
                        if selected_controlnets and len(control_images) > 1:
                            multi_controlnet_kwargs = self.apply_multi_controlnet(prompt, selected_controlnets, control_images)
                            if multi_controlnet_kwargs:
                                pipe_kwargs.update(multi_controlnet_kwargs)
                                logger.info(f"✅ Мультимодальный ControlNet активирован с {len(control_images)} контрольными картами")
                            else:
                                # Fallback к основной карте
                                pipe_kwargs["image"] = control_images[0]
                                pipe_kwargs["controlnet_conditioning_scale"] = 1.0  # Усиленное влияние ControlNet
                                logger.info("✅ ControlNet активирован с основной контрольной картой (fallback, усиленное влияние)")
                        else:
                            # Обычный режим с одной картой
                            pipe_kwargs["image"] = control_images[0]
                            pipe_kwargs["controlnet_conditioning_scale"] = 1.0  # Усиленное влияние ControlNet
                            logger.info("✅ ControlNet активирован с контрольной картой (усиленное влияние)")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка подготовки мультимодального ControlNet: {e}")
                        # Fallback: простая контрольная карта
                        hint = Image.new('L', (1024, 1024), color=255)
                        pipe_kwargs["image"] = hint
                        pipe_kwargs["controlnet_conditioning_scale"] = 0.8  # Умеренное влияние для fallback
                        logger.info("✅ ControlNet активирован с fallback картой (умеренное влияние)")
                except Exception as e:
                    logger.warning(f"⚠️ ControlNet недоступен: {e}")

            # Единый проход: генерируем только финальное изображение
            logger.info("🚀 Финальный сегмент: единый проход без callback")
            result = pipe_to_use(
                **{**pipe_kwargs, "output_type": "pil"}
            )
            logger.info("✅ Финальная генерация завершена")
            
            # Сохранение результатов
            final_image = result.images[0]
            logger.info(f"📊 Размер сгенерированного изображения: {final_image.size}")
            
            # Превью уже построены из середины финального прохода (если удалось)
            
            # Сохранение файлов
            final_path = "/tmp/final.png"
            
            final_image.save(final_path)
            logger.info(f"✅ FINAL_READY {final_path}")
            
            # Создание оптимизированного colormap с помощью Color Grid Adapter
            colormap_path = "/tmp/colormap.png"
            try:
                # Используем наш оптимизированный Color Grid Adapter
                colormap_image = self._create_optimized_colormap(prompt, size=(1024, 1024), pattern_type=colormap, granule_size=granule_size)
                
                # Валидация colormap против промпта
                if not self._validate_colormap_against_prompt(colormap_image, prompt):
                    logger.warning("⚠️ Colormap не соответствует промпту, пересоздаем...")
                    colormap_image = self._force_rebuild_colormap(prompt, size=(1024, 1024))
                
                # Сохраняем в высоком разрешении для лучшего качества
                colormap_image.save(colormap_path)
                logger.info(f"🎨 ОПТИМИЗИРОВАННЫЙ COLORMAP_READY {colormap_path}")
                logger.info(f"📊 Размер colormap: {colormap_image.size}")
                
                # Дополнительно сохраняем в маленьком размере для легенды
                legend_path = "/tmp/legend.png"
                legend_image = colormap_image.resize((256, 256), Image.Resampling.LANCZOS)
                legend_image.save(legend_path)
                logger.info(f"📋 ЛЕГЕНДА_READY {legend_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ Не удалось построить оптимизированный colormap: {e}")
                # Fallback: простой colormap
                try:
                    parsed_colors = self._parse_percent_colors(prompt)
                    if not parsed_colors:
                        parsed_colors = [{"name": "white", "proportion": 1.0}]
                    colormap_image = self._render_legend(parsed_colors, size=256)
                    colormap_image.save(colormap_path)
                    logger.info(f"🎨 FALLBACK COLORMAP_READY {colormap_path}")
                except Exception as e2:
                    logger.error(f"❌ Критическая ошибка создания colormap: {e2}")
                    Image.new('RGBA', (256, 256), color=(255, 255, 255, 255)).save(colormap_path)
            
            # Очистка памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Память очищена")
            
            # Сохранение полного JSON-ответа с деталями генерации
            try:
                import json
                generation_data = {
                    "model_version": MODEL_VERSION,
                    "input_prompt": prompt,
                    "full_prompt": full_prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "colormap": colormap,
                    "granule_size": granule_size,
                    "device": self.device,
                    "image_size": final_image.size,
                    "generation_time": time.time() if 'time' in globals() else None,
                    "parsed_colors": parsed_colors if 'parsed_colors' in locals() else []
                }
                json_path = "/tmp/generation_data.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(generation_data, f, ensure_ascii=False, indent=2)
                logger.info(f"📄 JSON_READY {json_path}")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось сохранить JSON-данные: {e}")
            
            logger.info("✅ Изображение успешно сгенерировано!")
            
            # Логируем статистику Color Grid Adapter
            stats = self.get_color_grid_stats()
            logger.info(f"📊 Color Grid Adapter статистика:")
            logger.info(f"   - Всего генераций: {stats['total_generations']}")
            logger.info(f"   - ControlNet использован: {stats['controlnet_used']} ({stats['controlnet_usage_percent']}%)")
            logger.info(f"   - Популярный паттерн: {stats['most_used_pattern']}")
            logger.info(f"   - Популярный размер гранул: {stats['most_used_granule_size']}")
            
            # Возвращаем файлы в правильном порядке: final, colormap, legend
            yield Path(final_path)
            yield Path(colormap_path)
            yield Path(legend_path)
            
        except (ColormapGenerationError, ControlNetValidationError) as e:
            logger.error(f"🚨 КРИТИЧЕСКАЯ ОШИБКА: {e}")
            logger.error(f"📊 Тип ошибки: {type(e).__name__}")
            logger.error("🛑 Генерация прервана для предотвращения некорректных результатов")
            # Возвращаем информативное сообщение об ошибке
            error_path = "/tmp/error_message.txt"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Критическая ошибка генерации: {e}\n")
                f.write(f"Тип ошибки: {type(e).__name__}\n")
                f.write("Генерация прервана для предотвращения некорректных результатов.\n")
                f.write("Пожалуйста, проверьте промпт и попробуйте снова.\n")
            yield Path(error_path)
            return
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            logger.error(f"📊 Тип ошибки: {type(e).__name__}")
            logger.error(f"📊 Детали ошибки: {str(e)}")
            raise e

    def select_optimal_controlnet(self, color_count):
        """Выбирает оптимальную комбинацию ControlNet на основе сложности промпта"""
        if color_count == 1:
            return None  # Базовая модель справляется
        elif color_count == 2:
            return ["t2i_color"]  # Только цветовой контроль
        elif color_count == 3:
            return ["t2i_color", "shuffle"]  # Цвет + перемешивание
        else:  # 4+ цветов
            return ["t2i_color", "color_grid", "shuffle"]  # Полный контроль

    def apply_multi_controlnet(self, prompt, controlnets, control_images):
        """Применяет несколько ControlNet последовательно для максимальной точности"""
        if not controlnets or not control_images:
            return None
        
        try:
            # Создаем комбинированную контрольную карту
            combined_hint = self._create_combined_control_hint(control_images)
            
            # Применяем каждый ControlNet с соответствующими весами
            pipe_kwargs = {}
            for i, controlnet_type in enumerate(controlnets):
                if controlnet_type == "t2i_color":
                    pipe_kwargs["image"] = control_images[i] if i < len(control_images) else combined_hint
                    pipe_kwargs["controlnet_conditioning_scale"] = 1.0  # Усилено с 0.8 до 1.0
                elif controlnet_type == "color_grid":
                    pipe_kwargs["image"] = control_images[i] if i < len(control_images) else combined_hint
                    pipe_kwargs["controlnet_conditioning_scale"] = 1.1  # Усилено с 0.9 до 1.1
                elif controlnet_type == "shuffle":
                    pipe_kwargs["image"] = control_images[i] if i < len(control_images) else combined_hint
                    pipe_kwargs["controlnet_conditioning_scale"] = 0.9  # Усилено с 0.7 до 0.9
            
            return pipe_kwargs
        except Exception as e:
            logger.warning(f"⚠️ Ошибка применения мульти ControlNet: {e}")
            return None

    def _create_combined_control_hint(self, control_images):
        """Создает комбинированную контрольную карту из нескольких источников"""
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
