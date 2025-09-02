# predict.py - Основной файл предсказания для модели "nauslava/plitka-pro-project:v4.4.56"
# Использует НАШУ обученную модель с LoRA и Textual Inversion (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)

import os
import torch
import random
import gc
import json
import logging
import time
from typing import Optional, List, Dict, Any, Iterator
from PIL import Image, ImageDraw, ImageColor
import numpy as np
from pathlib import Path

# Добавляем импорты для Color Grid Adapter
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

class ColorManager:
    """Централизованное управление цветами для устранения рассинхронизации модулей"""
    
    def __init__(self):
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
        
        # Допустимые названия цветов (в нижнем регистре)
        self.valid_colors = {color.lower() for color in self.color_table.values()}
        
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
    
    def extract_colors_from_prompt(self, prompt: str) -> List[str]:
        """Единая функция для извлечения цветов из промпта"""
        colors = []
        words = prompt.lower().split()
        
        for word in words:
            # Убираем знаки препинания и проценты
            clean_word = word.strip('%,.!?()[]{}')
            if clean_word in self.valid_colors:
                colors.append(clean_word)
        
        return colors
    
    def get_color_rgb(self, color_name: str) -> tuple:
        """Получение RGB значения для цвета"""
        return self.color_rgb_map.get(color_name.lower(), (127, 127, 127))
    
    def validate_colors(self, colors: List[str]) -> bool:
        """Валидация списка цветов"""
        return all(color in self.valid_colors for color in colors)
    
    def get_color_count(self, prompt: str) -> int:
        """Получение количества цветов в промпте"""
        return len(self.extract_colors_from_prompt(prompt))

class ColorGridControlNet:
    """Улучшенный Color Grid Adapter для точного контроля цветовых пропорций"""
    
    def __init__(self):
        self.patterns = ["random", "grid", "radial", "granular"]
        self.granule_sizes = {
            "small": {"size_range": (2, 4), "density": 0.9},
            "medium": {"size_range": (3, 6), "density": 0.8},
            "large": {"size_range": (5, 8), "density": 0.7}
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
        """Создает паттерн, имитирующий резиновую крошку"""
        width, height = size
        canvas = Image.new('RGB', size, (255, 255, 255))
        pixels = canvas.load()
        
        # Параметры гранул
        granule_params = self.granule_sizes[granule_size]
        min_size, max_size = granule_params["size_range"]
        density = granule_params["density"]
        
        # Нормализация пропорций
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        normalized_colors = []
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            normalized_colors.append({
                "color": color_rgb,
                "proportion": proportion,
                "pixels_needed": int(proportion * width * height * density)
            })
        
        # Создание гранул
        pixels_placed = {i: 0 for i in range(len(normalized_colors))}
        
        for _ in range(int(width * height * density)):
            # Выбор цвета на основе пропорций
            available_colors = [i for i, color_info in enumerate(normalized_colors) 
                              if pixels_placed[i] < color_info["pixels_needed"]]
            
            if not available_colors:
                break
            
            color_idx = random.choice(available_colors)
            color_info = normalized_colors[color_idx]
            
            # Создание гранулы
            granule_size = random.randint(min_size, max_size)
            x = random.randint(0, width - granule_size)
            y = random.randint(0, height - granule_size)
            
            # Размещение гранулы
            for dx in range(granule_size):
                for dy in range(granule_size):
                    if (0 <= x + dx < width and 0 <= y + dy < height and
                        pixels[x + dx, y + dy] == (255, 255, 255)):  # Только пустые пиксели
                        pixels[x + dx, y + dy] = color_info["color"]
                        pixels_placed[color_idx] += 1
        
        return canvas
    
    def _create_random_pattern(self, colors, size):
        """Создает случайный паттерн с точными пропорциями"""
        width, height = size
        canvas = Image.new('RGB', size, (255, 255, 255))
        pixels = canvas.load()
        
        # Нормализация пропорций
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        color_pixels = {}
        
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            color_pixels[color_rgb] = int(proportion * width * height)
        
        # Случайное размещение пикселей
        all_positions = [(x, y) for x in range(width) for y in range(height)]
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
        """Создает сеточный паттерн с точными пропорциями"""
        width, height = size
        canvas = Image.new('RGB', size, (255, 255, 255))
        pixels = canvas.load()
        
        # Нормализация пропорций
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        x_pos = 0
        
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            color_width = int(proportion * width)
            
            # Заполнение вертикальной полосы
            for x in range(x_pos, min(x_pos + color_width, width)):
                for y in range(height):
                    pixels[x, y] = color_rgb
            
            x_pos += color_width
        
        return canvas
    
    def _create_radial_pattern(self, colors, size):
        """Создает радиальный паттерн с точными пропорциями"""
        width, height = size
        canvas = Image.new('RGB', size, (255, 255, 255))
        pixels = canvas.load()
        
        center_x, center_y = width // 2, height // 2
        max_radius = max(center_x, center_y)
        
        # Нормализация пропорций
        total_proportion = sum(color.get("proportion", 0) for color in colors)
        current_radius = 0
        
        for color in colors:
            proportion = color.get("proportion", 0) / total_proportion
            color_rgb = self._name_to_rgb(color.get("name", "white"))
            radius_increment = int(proportion * max_radius)
            
            # Заполнение кольца
            for r in range(current_radius, current_radius + radius_increment):
                for x in range(width):
                    for y in range(height):
                        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                        if r <= dist < r + 1:
                            pixels[x, y] = color_rgb
            
            current_radius += radius_increment
        
        return canvas
    
    def _name_to_rgb(self, color_name):
        """Преобразует название цвета в RGB через ColorManager"""
        # Используем централизованную систему цветов
        return self.color_manager.get_color_rgb(color_name)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Единая версия модели для логов
MODEL_VERSION = "v4.4.58"

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
        """Простенький парсер строк вида '60% red, 40% white' → список цветов и долей [0..1]."""
        parts = [p.strip() for p in simple_prompt.split(',') if p.strip()]
        result: List[Dict[str, Any]] = []
        for p in parts:
            try:
                percent_str, name = p.split('%', 1)
                percent = float(percent_str.strip())
                color_name = name.strip()
                if color_name.lower().startswith(('of ', ' ')):
                    color_name = color_name.split()[-1]
                
                # Валидация цвета через ColorManager
                if self.color_manager.validate_colors([color_name]):
                    result.append({"name": color_name, "proportion": max(0.0, min(1.0, percent / 100.0))})
                else:
                    logger.warning(f"⚠️ Недопустимый цвет в промпте: {color_name}")
                    # Fallback: заменяем на белый
                    result.append({"name": "white", "proportion": max(0.0, min(1.0, percent / 100.0))})
            except Exception:
                continue
        # Нормализация, если сумма не 1.0
        total = sum(c["proportion"] for c in result) or 1.0
        for c in result:
            c["proportion"] = c["proportion"] / total
        return result

    def _render_legend(self, colors: List[Dict[str, Any]], size: int = 256) -> Image.Image:
        """Строим простую легенду/colormap из входных пропорций (горизонтальные полосы)."""
        img = Image.new('RGB', (size, size), color='white')
        draw = ImageDraw.Draw(img)
        y = 0
        for c in colors:
            h = max(1, int(size * c["proportion"]))
            try:
                rgb = ImageColor.getrgb(c["name"])  # распознает стандартные цвета
            except Exception:
                rgb = (200, 200, 200)
            draw.rectangle([0, y, size, min(size, y + h)], fill=rgb)
            y += h
        # Подгоняем последнюю полосу до края
        if y < size and colors:
            try:
                rgb_last = ImageColor.getrgb(colors[-1]["name"])
            except Exception:
                rgb_last = (200, 200, 200)
            draw.rectangle([0, y, size, size], fill=rgb_last)
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
    
    def _create_optimized_colormap(self, prompt: str, size: tuple = (1024, 1024)) -> Image.Image:
        """Создает оптимизированный colormap для ControlNet с точными пропорциями"""
        try:
            # Парсим цвета из промпта
            colors = self._parse_percent_colors(prompt)
            if not colors:
                logger.warning("⚠️ Не удалось распарсить цвета, создаем базовый colormap")
                return Image.new('RGB', size, (255, 255, 255))
            
            # Определяем оптимальный паттерн на основе количества цветов
            color_count = len(colors)
            if color_count == 1:
                pattern_type = "random"  # Простой случай
            elif color_count == 2:
                pattern_type = "granular"  # Имитация резиновой крошки
            elif color_count == 3:
                pattern_type = "granular"  # Сложная крошка
            else:  # 4+ цветов
                pattern_type = "granular"  # Максимальная сложность
            
            # Определяем размер гранул на основе сложности
            if color_count <= 2:
                granule_size = "medium"
            else:
                granule_size = "small"  # Меньшие гранулы для сложных комбинаций
            
            logger.info(f"🎨 Создание colormap: {color_count} цветов, паттерн: {pattern_type}, гранулы: {granule_size}")
            
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
        """Валидация colormap против промпта"""
        try:
            expected_colors = self.color_manager.extract_colors_from_prompt(prompt)
            if not expected_colors:
                logger.warning("⚠️ Не удалось извлечь цвета из промпта")
                return False
            
            # Простая проверка: colormap не должен быть полностью серым
            colormap_array = np.array(colormap)
            if len(colormap_array.shape) == 3:
                # RGB изображение
                gray_pixels = np.all(colormap_array == [127, 127, 127], axis=2)
                if np.all(gray_pixels):
                    logger.warning("⚠️ Colormap полностью серый - ошибка распознавания цветов")
                    return False
            
            logger.info(f"✅ Colormap валиден для цветов: {expected_colors}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка валидации colormap: {e}")
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
                return Image.new('RGB', size, (127, 127, 127))
            
            # Создаем простой colormap с правильными цветами
            colormap = Image.new('RGB', size, (255, 255, 255))
            pixels = colormap.load()
            
            # Размещаем цвета в простом паттерне
            for i, color in enumerate(colors):
                rgb = self.color_manager.get_color_rgb(color)
                # Разделяем изображение на секции по цветам
                start_x = (i * size[0]) // len(colors)
                end_x = ((i + 1) * size[0]) // len(colors)
                for x in range(start_x, end_x):
                    for y in range(size[1]):
                        pixels[x, y] = rgb
            
            logger.info(f"✅ Colormap пересобран для цветов: {colors}")
            return colormap
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка пересборки colormap: {e}")
            # Fallback: простой серый colormap
            return Image.new('RGB', size, (127, 127, 127))
    
    def predict(self, prompt: str = Input(description="Описание цветов резиновой плитки", default="100% red"), 
                negative_prompt: Optional[str] = Input(description="Негативный промпт", default=None), 
                seed: int = Input(description="Сид для воспроизводимости", default=-1),
                steps: int = Input(description="Число шагов", default=20),
                guidance: float = Input(description="Guidance scale", default=7.0),
                lora_scale: float = Input(description="Сила LoRA (0.0-1.0)", default=0.7),
                use_controlnet: bool = Input(description="Включить ControlNet SoftEdge (требует control_image)", default=False),
                control_image: Optional[Path] = Input(description="Контрольное изображение (опц.) для SoftEdge", default=None)) -> Iterator[Path]:
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
            logger.info(f"📊 Steps: {steps} (базовый)")
            logger.info(f"🎚️ Guidance: {guidance} (базовый)")
            logger.info(f"🔧 LoRA Scale: {lora_scale} (базовый)")
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
            
            logger.info(f"🎨 Генерация изображения...")
            logger.info(f"📝 Полный промпт: {full_prompt}")
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
                adaptive_steps = 20
                adaptive_guidance = 7.0
                logger.info("🎯 Адаптивные параметры для 1 цвета: steps=20, guidance=7.0")
            elif color_count == 2:
                # Два цвета - средняя сложность
                adaptive_steps = 25
                adaptive_guidance = 7.5
                logger.info("🎯 Адаптивные параметры для 2 цветов: steps=25, guidance=7.5")
            elif color_count == 3:
                # Три цвета - высокая сложность
                adaptive_steps = 30
                adaptive_guidance = 8.0
                logger.info("🎯 Адаптивные параметры для 3 цветов: steps=30, guidance=8.0")
            else:
                # 4+ цвета - максимальная сложность
                adaptive_steps = 35
                adaptive_guidance = 8.5
                logger.info("🎯 Адаптивные параметры для 4+ цветов: steps=35, guidance=8.5")
            
            # Генерация изображения с адаптивными параметрами
            logger.info("🚀 Запуск pipeline для генерации с адаптивными параметрами...")
            pipe_to_use = self.pipe
            pipe_kwargs = dict(
                prompt=full_prompt,
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

            # Автоматическое включение ControlNet для сложных промптов (2+ цветов)
            auto_controlnet = False
            if not use_controlnet:
                # Используем уже подсчитанное количество цветов
                if color_count >= 2:
                    auto_controlnet = True
                    logger.info(f"🎯 Автоматически включаем ControlNet для {color_count} цветов")
            
            # Обновляем общую статистику
            self.color_grid_stats["total_generations"] += 1
            if use_controlnet or auto_controlnet:
                self.color_grid_stats["controlnet_used"] += 1
            
            # ControlNet lazy init (включая автоматическое включение)
            if (use_controlnet or auto_controlnet) and ControlNetModel is not None and StableDiffusionXLControlNetPipeline is not None:
                try:
                    if self.controlnet is None:
                        logger.info("🔗 Загрузка ControlNet SoftEdge для SDXL...")
                        self.controlnet = ControlNetModel.from_pretrained(
                            "diffusers/controlnet-softedge-sdxl-1.0", torch_dtype=torch.float16
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
                    
                    # Подготовка контрольной карты для ControlNet
                    try:
                        if control_image is not None:
                            # Если пользователь предоставил контрольное изображение
                            from PIL import ImageFilter
                            hint = Image.open(control_image).convert('L').resize((1024, 1024), Image.Resampling.LANCZOS)
                            hint = hint.filter(ImageFilter.EDGE_ENHANCE)
                            logger.info("✅ ControlNet использует пользовательское контрольное изображение")
                        else:
                            # Автоматически создаем оптимизированную контрольную карту
                            logger.info("🎨 Создание автоматической контрольной карты для ControlNet")
                            color_control_image = self._create_optimized_colormap(prompt, size=(1024, 1024))
                            
                            # Валидация colormap против промпта
                            if not self._validate_colormap_against_prompt(color_control_image, prompt):
                                logger.warning("⚠️ Colormap не соответствует промпту, пересоздаем...")
                                color_control_image = self._force_rebuild_colormap(prompt, size=(1024, 1024))
                            
                            # Преобразуем в grayscale для ControlNet
                            hint = color_control_image.convert('L')
                            
                            # Применяем edge enhancement для лучшего контроля
                            hint = hint.filter(ImageFilter.EDGE_ENHANCE)
                            logger.info("✅ ControlNet использует автоматически созданную контрольную карту")
                        
                        pipe_kwargs["image"] = hint
                        logger.info("✅ ControlNet активирован с контрольной картой")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Ошибка подготовки контрольной карты: {e}")
                        # Fallback: простая контрольная карта
                        hint = Image.new('L', (1024, 1024), color=255)
                        pipe_kwargs["image"] = hint
                        logger.info("✅ ControlNet активирован с fallback картой")
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
                colormap_image = self._create_optimized_colormap(prompt, size=(1024, 1024))
                
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
                    Image.new('RGB', (256, 256), color='white').save(colormap_path)
            
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
                    "steps": steps,
                    "guidance": guidance,
                    "lora_scale": lora_scale,
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
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            logger.error(f"📊 Тип ошибки: {type(e).__name__}")
            logger.error(f"📊 Детали ошибки: {str(e)}")
            raise e
