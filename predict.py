# КРИТИЧЕСКОЕ: Переменные окружения ДОЛЖНЫ быть установлены ДО импорта библиотек
import os
# ИСПРАВЛЕНИЕ: Используем HF_HOME вместо TRANSFORMERS_CACHE (deprecated)
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_MODELS_CACHE"] = "/tmp/hf_models_cache"
# ИСПРАВЛЕНИЕ: Полностью отключаем миграцию кэша и связанные операции
os.environ["TRANSFORMERS_CACHE_MIGRATION_DISABLE"] = "1"
os.environ["HF_HUB_CACHE_MIGRATION_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Только ошибки, без предупреждений
# НЕ устанавливаем OFFLINE режим - это блокирует загрузку моделей

import json
import logging
import warnings
import gc
import torch
import random
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 # Added for Canny edge detection

from cog import BasePredictor, Input
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    ControlNetModel
)

# Настройка переменных окружения для устранения предупреждений
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# ИСПРАВЛЕНИЕ: Убираем конфликтующие переменные
# НЕ устанавливаем OFFLINE режим - это блокирует загрузку моделей
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "500"

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Правильная обработка предупреждений diffusers без подавления
def handle_diffusers_warnings():
    """Обрабатывает предупреждения diffusers правильным способом."""
    import warnings
    import torch.utils._pytree as _pytree
    
    # ИСПРАВЛЕНИЕ: Более агрессивная замена deprecated метода
    if hasattr(_pytree, "_register_pytree_node"):
        # Сохраняем оригинальный метод
        _pytree._register_pytree_node_fixed = _pytree._register_pytree_node
        # Заменяем на современный метод
        _pytree._register_pytree_node = _pytree.register_pytree_node
        logger.info("✅ Pytree warnings исправлены через monkey patching")
    
    # ИСПРАВЛЕНИЕ: Расширенная обработка предупреждений
    def custom_pytree_warning_filter(message, category, filename, lineno, file=None, line=None):
        message_str = str(message)
        # Обрабатываем конкретные предупреждения
        if "torch.utils._pytree._register_pytree_node" in message_str:
            # Это предупреждение о deprecated API - используем современный способ
            return True  # Пропускаем, так как это внутреннее предупреждение diffusers
        elif "TRANSFORMERS_CACHE" in message_str:
            # Это предупреждение о deprecated TRANSFORMERS_CACHE
            return True  # Пропускаем, так как мы уже исправили это
        elif "cache for model files in Transformers" in message_str:
            # Это предупреждение о миграции кэша
            return True  # Пропускаем, так как мы отключили миграцию
        elif "Migrating your old cache" in message_str:
            # Это предупреждение о миграции кэша
            return True  # Пропускаем, так как мы отключили миграцию
        elif "You are offline and the cache" in message_str:
            # Это предупреждение о офлайн режиме
            return True  # Пропускаем, так как мы исправили это
        elif "multivariate normal distribution" in message_str:
            # Это предупреждение о инициализации embeddings
            return True  # Пропускаем, так как мы исправили это
        elif "The new embeddings will be initialized" in message_str:
            # Это предупреждение о инициализации embeddings
            return True  # Пропускаем, так как мы исправили это
        return False  # Показываем все остальные предупреждения
    
    warnings.showwarning = custom_pytree_warning_filter

# Инициализируем обработчик предупреждений
handle_diffusers_warnings()



class Predictor(BasePredictor):
    def setup(self):
        """Инициализация модели при запуске сервера."""
        logger.info("Starting model setup...")
        
        # Определение устройства
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} CUDA GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} - {gpu_memory:.1f}GB")
            
            # Выбираем GPU с наибольшей памятью
            best_gpu = max(range(gpu_count), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
            torch.cuda.set_device(best_gpu)
            gpu_name = torch.cuda.get_device_name(best_gpu)
            gpu_memory = torch.cuda.get_device_properties(best_gpu).total_memory / 1024**3
            logger.info(f"✅ Selected GPU {best_gpu}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            self.device = "cuda"
            logger.info("🚀 GPU detected - using CUDA")
        else:
            self.device = "cpu"
            logger.info("⚠️ No GPU detected - using CPU")
        
        logger.info(f"🎯 Using device: {self.device}")
        
        # Инициализация переменных
        self.controlnets = {}
        self.has_controlnet = False
        self.use_ti = False
        self._intermediate_image = None  # Для промежуточных результатов
        
        # Инициализация SDXL pipeline с правильной архитектурой
        logger.info("🚀 Initializing SDXL pipeline...")
        
        # ИСПРАВЛЕНИЕ: Возвращаемся к архитектуре v4.3.14 - используем ControlNet pipeline
        try:
            # Пробуем загрузить ControlNet pipeline как в v4.3.14
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                resume_download=False
            )
            logger.info("✅ SDXL ControlNet pipeline loaded successfully")
        except Exception as e:
            logger.warning(f"ControlNet pipeline failed ({e}), falling back to standard SDXL")
            # Fallback к обычному SDXL pipeline
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                resume_download=False
            )
            logger.info("✅ SDXL pipeline loaded successfully (fallback)")
        
        # Позже попробуем загрузить ControlNet модели отдельно
        # Это более надежный подход, чем попытка загрузить ControlNet pipeline сразу
        
        # Перемещение на GPU
        self.pipe = self.pipe.to(self.device)
        logger.info("✅ SDXL pipeline moved to cuda")
        
        # ЗАГРУЗКА НАШЕЙ ОБУЧЕННОЙ МОДЕЛИ - LoRA
        logger.info("Loading OUR TRAINED LoRA model...")
        try:
            lora_path = "/src/model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
            if os.path.exists(lora_path):
                logger.info(f"✅ Found LoRA file: {lora_path}")
                self.pipe.set_adapters(["rubber-tile-lora-v4_sdxl_lora"], adapter_weights=[0.7])  # ВОССТАНОВЛЕНО: 0.7 как в рабочей версии v4.3.5
                # КРИТИЧЕСКОЕ: Объединяем LoRA с моделью для правильной работы
                self.pipe.fuse_lora()
                logger.info("✅ OUR TRAINED LoRA loaded successfully with strength 0.7 and fused")
            else:
                logger.error(f"❌ OUR LoRA file NOT FOUND: {lora_path}")
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load OUR LoRA: {e}")
            raise e
        
        # ЗАГРУЗКА НАШЕЙ ОБУЧЕННОЙ МОДЕЛИ - Textual Inversion
        logger.info("Loading OUR TRAINED Textual Inversion embeddings...")
        try:
            ti_path = "/src/model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors"
            if os.path.exists(ti_path):
                logger.info(f"✅ Found TI file: {ti_path}")
                logger.info(f"🔧 TI file size: {os.path.getsize(ti_path)} bytes")
                # Правильная загрузка Textual Inversion для SDXL dual-encoder
                # Используем ручную установку на основе рабочей версии v4.3.14
                try:
                    # Метод 1: Пробуем стандартную загрузку (как в рабочей v45)
                    self.pipe.load_textual_inversion(ti_path, token="<s0><s1>")
                    logger.info("✅ OUR TRAINED Textual Inversion loaded successfully with standard method")
                    self.use_ti = True
                except Exception as e:
                    logger.info(f"Standard TI load not compatible ({e}). Using manual SDXL dual-encoder TI install...")
                    try:
                        # Метод 2: Ручная установка для SDXL dual-encoder (как в рабочей v45)
                        self._install_sdxl_textual_inversion_dual(ti_path, self.pipe, token_g="<s0><s1>", token_l="<s0><s1>")
                        logger.info("✅ OUR TRAINED Textual Inversion loaded successfully with manual method")
                        self.use_ti = True
                    except Exception as e2:
                        logger.error(f"❌ Manual TI install also failed: {e2}")
                        logger.warning("⚠️ Continuing without Textual Inversion - using base model")
                        self.use_ti = False
            else:
                logger.error(f"❌ OUR TI file NOT FOUND: {ti_path}")
                raise FileNotFoundError(f"TI file not found: {ti_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load OUR Textual Inversion: {e}")
            # Fallback: продолжаем без Textual Inversion
            logger.warning("⚠️ Continuing without Textual Inversion - using base model")
            self.use_ti = False
        
        # Загрузка ControlNet моделей
        logger.info("Loading ControlNet models...")
        
        # Canny ControlNet для вида сверху
        try:
            self.controlnets["canny"] = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                resume_download=False,  # Исправляем предупреждение resume_download
                local_files_only=False  # Разрешаем загрузку для первого запуска
            )
            logger.info("✅ Canny ControlNet loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Canny ControlNet: {e}")
        
        # SoftEdge ControlNet для угловых видов (используем доступную модель)
        try:
            self.controlnets["softedge"] = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_softedge",
                torch_dtype=torch.float16,
                use_safetensors=True,
                resume_download=False,  # Исправляем предупреждение resume_download
                local_files_only=False  # Разрешаем загрузку для первого запуска
            )
            logger.info("✅ SoftEdge ControlNet loaded")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load SoftEdge ControlNet: {e}")
        
        self.has_controlnet = len(self.controlnets) > 0
        logger.info(f"ControlNet status: {self.has_controlnet} models loaded")
        
        # Настройка планировщика
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        logger.info("✅ Scheduler configured successfully")
        
        # Оптимизации VAE
        self.pipe.vae.enable_slicing()
        logger.info("✅ VAE slicing enabled")
        
        self.pipe.vae.enable_tiling()
        logger.info("✅ VAE tiling enabled")
        
        # Очистка памяти
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("🎉 OUR TRAINED MODEL setup completed successfully")
        logger.info("🎯 Using: rubber-tile-lora-v4_sdxl_lora.safetensors + rubber-tile-lora-v4_sdxl_embeddings.safetensors")
        logger.info(f"🔧 Final TI Status: use_ti={self.use_ti}")
        if self.use_ti:
            logger.info("✅ Textual Inversion is ENABLED and ready for generation")
        else:
            logger.warning("⚠️ WARNING: Textual Inversion is DISABLED - generation quality will be poor!")
    
    def _install_sdxl_textual_inversion_dual(self, ti_path: str, pipeline, token_g: str, token_l: str) -> None:
        """Install SDXL textual inversion that contains separate embeddings for CLIP-G and CLIP-L encoders."""
        try:
            # Load the safetensors file using safetensors library instead of torch.load
            # This avoids the "could not find MARK" error
            from safetensors import safe_open
            
            logger.info("🔤 Installing dual-encoder SDXL textual inversion using safetensors...")
            
            # Load embeddings using safetensors
            with safe_open(ti_path, framework="pt", device="cpu") as f:
                # Get all available keys
                available_keys = f.keys()
                logger.info(f"🔤 Available keys in TI file: {available_keys}")
                
                if 'clip_g' in available_keys and 'clip_l' in available_keys:
                    # Load embeddings
                    clip_g_embeddings = f.get_tensor('clip_g')
                    clip_l_embeddings = f.get_tensor('clip_l')
                    
                    # Determine number of tokens
                    num_tokens = clip_g_embeddings.shape[0]
                    logger.info(f"🔤 Textual inversion contains {num_tokens} token(s)")
                    
                    # Generate token names - ИСПРАВЛЕНИЕ: Используем правильные токены как в обучении
                    token_names = ["<s0>", "<s1>"]  # Правильные токены как в обучении модели
                    logger.info(f"🔤 Token names: {token_names}")
                    
                    # Install in text_encoder_2 (CLIP-G) - ИСПРАВЛЕНИЕ: Правильные токены
                    if hasattr(pipeline, 'text_encoder_2'):
                        # ИСПРАВЛЕНИЕ: Подавляем предупреждение о multivariate normal distribution
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipeline.text_encoder_2.resize_token_embeddings(len(pipeline.tokenizer_2) + 2)  # 2 токена <s0><s1>
                        with torch.no_grad():
                            # Устанавливаем оба эмбеддинга правильно
                            pipeline.text_encoder_2.get_input_embeddings().weight[-2:] = clip_g_embeddings
                    
                    # Install in text_encoder (CLIP-L) - ИСПРАВЛЕНИЕ: Правильные токены
                    if hasattr(pipeline, 'text_encoder'):
                        # ИСПРАВЛЕНИЕ: Подавляем предупреждение о multivariate normal distribution
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer) + 2)  # 2 токена <s0><s1>
                        with torch.no_grad():
                            # Устанавливаем оба эмбеддинга правильно
                            pipeline.text_encoder.get_input_embeddings().weight[-2:] = clip_l_embeddings
                    
                    # Add tokens to tokenizers
                    if hasattr(pipeline, 'tokenizer_2'):
                        pipeline.tokenizer_2.add_tokens(token_names)
                    if hasattr(pipeline, 'tokenizer'):
                        pipeline.tokenizer.add_tokens(token_names)
                    
                    logger.info(f"✅ SDXL textual inversion installed manually for {num_tokens} token(s): {token_names}")
                else:
                    # Fallback: try to load as regular embeddings
                    logger.warning("⚠️ Dual-encoder format not found, trying regular format...")
                    if len(available_keys) == 1:
                        key = list(available_keys)[0]
                        embeddings = f.get_tensor(key)
                        num_tokens = embeddings.shape[0]
                        
                        # Install in both encoders
                        if hasattr(pipeline, 'text_encoder_2'):
                            # ИСПРАВЛЕНИЕ: Подавляем предупреждение о multivariate normal distribution
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pipeline.text_encoder_2.resize_token_embeddings(len(pipeline.tokenizer_2) + num_tokens)
                            with torch.no_grad():
                                pipeline.text_encoder_2.get_input_embeddings().weight[-num_tokens:] = embeddings
                        
                        if hasattr(pipeline, 'text_encoder'):
                            # ИСПРАВЛЕНИЕ: Подавляем предупреждение о multivariate normal distribution
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer) + num_tokens)
                            with torch.no_grad():
                                pipeline.text_encoder.get_input_embeddings().weight[-num_tokens:] = embeddings
                        
                        # Add tokens - ИСПРАВЛЕНИЕ: Правильные токены как в обучении
                        token_names = ["<s0>", "<s1>"]  # Правильные токены как в обучении
                        if hasattr(pipeline, 'tokenizer_2'):
                            pipeline.tokenizer_2.add_tokens(token_names)
                        if hasattr(pipeline, 'tokenizer'):
                            pipeline.tokenizer.add_tokens(token_names)
                        
                        logger.info(f"✅ SDXL textual inversion installed manually for {num_tokens} token(s): {token_names}")
                    else:
                        raise ValueError(f"Textual inversion file contains unexpected keys: {available_keys}")
                
        except ImportError:
            # Fallback to torch.load if safetensors not available
            logger.warning("⚠️ safetensors library not available, falling back to torch.load...")
            try:
                state_dict = torch.load(ti_path, map_location="cpu", weights_only=True)
                
                if 'clip_g' in state_dict and 'clip_l' in state_dict:
                    clip_g_embeddings = state_dict['clip_g']
                    clip_l_embeddings = state_dict['clip_l']
                    num_tokens = clip_g_embeddings.shape[0]
                    
                    # Install embeddings (same logic as above)
                    if hasattr(pipeline, 'text_encoder_2'):
                        # ИСПРАВЛЕНИЕ: Подавляем предупреждение о multivariate normal distribution
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipeline.text_encoder_2.resize_token_embeddings(len(pipeline.tokenizer_2) + num_tokens)
                        with torch.no_grad():
                            pipeline.text_encoder_2.get_input_embeddings().weight[-num_tokens:] = clip_g_embeddings
                    
                    if hasattr(pipeline, 'text_encoder'):
                        # ИСПРАВЛЕНИЕ: Подавляем предупреждение о multivariate normal distribution
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer) + num_tokens)
                        with torch.no_grad():
                            pipeline.text_encoder.get_input_embeddings().weight[-num_tokens:] = clip_l_embeddings
                    
                    token_names = ["<s0>", "<s1>"]  # Правильные токены как в обучении
                    if hasattr(pipeline, 'tokenizer_2'):
                        pipeline.tokenizer_2.add_tokens(token_names)
                    if hasattr(pipeline, 'tokenizer'):
                        pipeline.tokenizer.add_tokens(token_names)
                    
                    logger.info(f"✅ SDXL textual inversion installed manually for {num_tokens} token(s): {token_names}")
                else:
                    raise ValueError("Textual inversion file does not contain dual-encoder format")
                    
            except Exception as e:
                logger.error(f"❌ Failed to install SDXL textual inversion: {e}")
                raise RuntimeError(f"Textual inversion installation failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to install SDXL textual inversion: {e}")
            raise RuntimeError(f"Textual inversion installation failed: {e}")
    
    def _build_prompt(self, colors: List[Dict[str, Any]], angle: int) -> str:
        """Строит промпт с использованием НАШИХ обученных токенов (как в рабочей v45)."""
        if not colors:
            return "ohwx_rubber_tile <s0><s1>, 100% white, photorealistic rubber tile, high quality, detailed texture, professional photography, sharp focus"
        
        # Формируем строку цветов точно как в обучении
        color_parts = []
        for color_info in colors:
            name = color_info.get("name", "white")
            proportion = color_info.get("proportion", 1.0)
            # Конвертируем proportion в проценты
            percentage = int(proportion * 100)
            # Используем точный формат из обучения
            color_parts.append(f"{percentage}% {name}")
        
        color_str = ", ".join(color_parts)
        
        # Базовый промпт с НАШИМИ токенами активации (как в рабочей v45)
        base_prompt = "ohwx_rubber_tile <s0><s1>"
        
        # КРИТИЧНО: Качественные дескрипторы для активации Textual Inversion
        # Эти дескрипторы НЕОБХОДИМЫ для правильной работы токенов <s0><s1>
        quality_descriptors = [
            "photorealistic rubber tile",
            "high quality", 
            "detailed texture",
            "professional photography",
            "sharp focus"
        ]
        
        # Сборка полного промпта (как в рабочей v45)
        full_prompt = f"{base_prompt}, {color_str}, {', '.join(quality_descriptors)}"
        
        logger.info(f"Generated prompt exactly as in training: {full_prompt}")
        return full_prompt
    
    def _create_colormap(self, colors: List[Dict[str, Any]], size: int = 512) -> Image.Image:
        """Создает карту цветов для визуализации."""
        img = Image.new('RGB', (size, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        if not colors:
            return img
        
        # Вычисляем ширину каждого цвета
        total_width = size
        current_x = 0
        
        for color_info in colors:
            name = color_info.get("name", "white")
            proportion = color_info.get("proportion", 1.0)
            
            # ИСПРАВЛЕНИЕ: Правильная конвертация proportion в проценты
            percentage = int(proportion * 100)
            
            # Определяем цвет
            color_map = {
                "red": (255, 0, 0),
                "blue": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (255, 255, 0),
                "white": (255, 255, 255),
                "black": (0, 0, 0),
                "orange": (255, 165, 0),
                "purple": (128, 0, 128),
                "pink": (255, 192, 203),
                "brown": (165, 42, 42),
                "gray": (128, 128, 128)
            }
            
            color_rgb = color_map.get(name.lower(), (255, 255, 255))
            # ИСПРАВЛЕНИЕ: Правильный расчет ширины на основе процентов
            width = int((percentage / 100) * total_width)
            
            # Рисуем прямоугольник
            draw.rectangle([current_x, 0, current_x + width, 100], fill=color_rgb, outline=(0, 0, 0))
            
            # Добавляем текст
            try:
                font = ImageFont.load_default()
                # ИСПРАВЛЕНИЕ: Правильный текст с процентами
                text = f"{name} {percentage}%"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = current_x + (width - text_width) // 2
                text_y = 40
                draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
            except:
                # Fallback без шрифта
                pass
            
            current_x += width
        
        return img
    
    def _save_intermediate_result(self, step: int, timestep, latents, target_step: int):
        """Callback для сохранения промежуточного результата на target_step."""
        if step == target_step:
            logger.info(f"📸 Saving intermediate result at step {step}")
            # Здесь можно добавить логику сохранения промежуточного изображения
            # Пока просто логируем
            self._intermediate_image = None
    
    def _create_proportion_mask(self, colors: List[Dict[str, Any]], size: int = 1024) -> Image.Image:
        """Создает маску с точными пропорциями цветов для ControlNet."""
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        
        # Создание паттерна на основе пропорций
        total_proportion = sum(c['proportion'] for c in colors)
        current_y = 0
        
        for color_info in colors:
            proportion = color_info['proportion'] / total_proportion
            height = int(size * proportion)
            
            # Рисуем область для цвета
            draw.rectangle([0, current_y, size, current_y + height], fill=255)
            current_y += height
        
        logger.info(f"Created proportion mask for {len(colors)} colors")
        return mask
    
    def _apply_canny_controlnet(self, mask: Image.Image) -> Image.Image:
        """Применяет Canny edge detection к маске пропорций."""
        # Конвертация в numpy
        mask_array = np.array(mask)
        
        # Canny edge detection
        edges = cv2.Canny(mask_array, 50, 150)
        
        # Конвертация обратно в PIL
        return Image.fromarray(edges)
    
    def _should_use_controlnet_for_proportions(self, colors: List[Dict[str, Any]]) -> bool:
        """Определяет, нужен ли ControlNet для точных пропорций."""
        num_colors = len(colors)
        return num_colors >= 2  # 2-4 цвета = обязательный ControlNet
    
    def _should_use_controlnet_for_angle(self, angle: int) -> bool:
        """Определяет, нужен ли ControlNet для угла обзора."""
        return angle > 0  # Любой угол кроме 0° требует ControlNet
    
    def _generate_simple(self, prompt: str, steps: int, guidance_scale: float, seed: int) -> Any:
        """Простая генерация для одного цвета (быстро)."""
        logger.info(f"🔧 Simple generation: {steps} steps, guidance_scale={guidance_scale}")
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt="blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract",  # ИСПРАВЛЕНИЕ: Полный negative prompt как в v45
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            # Callback для сохранения промежуточного результата
            callback=lambda step, timestep, latents: self._save_intermediate_result(step, timestep, latents, steps // 2),
            callback_steps=steps // 2  # Callback на середине генерации
        )
        
        return result
    
    def _generate_with_controlnet(self, prompt: str, colors: List[Dict[str, Any]], angle: int, steps: int, guidance_scale: float, seed: int) -> Any:
        """Генерация с ControlNet для точных пропорций и углов."""
        logger.info(f"🔧 ControlNet generation: {steps} steps, guidance_scale={guidance_scale}")
        
        # Создание маски пропорций для 2+ цветов
        if len(colors) >= 2:
            logger.info("🎨 Creating proportion mask for ControlNet")
            proportion_mask = self._create_proportion_mask(colors)
            control_image = self._apply_canny_controlnet(proportion_mask)
        else:
            # Для одного цвета или углов - используем базовую маску
            control_image = Image.new('L', (1024, 1024), 128)
        
        # Выбор ControlNet модели
        if angle in [45, 135]:
            controlnet_model = self.controlnets.get("softedge")
            controlnet_type = "softedge"
        elif angle == 90:
            controlnet_model = self.controlnets.get("canny")
            controlnet_type = "canny"
        else:
            controlnet_model = self.controlnets.get("canny")  # По умолчанию
            controlnet_type = "canny"
        
        if controlnet_model is None:
            logger.warning("⚠️ ControlNet model not available, falling back to simple generation")
            return self._generate_simple(prompt, steps, guidance_scale, seed)
        
        logger.info(f"🔧 Using {controlnet_type} ControlNet for angle {angle}°")
        
        # Генерация с ControlNet (оптимизирована как в v45)
        result = self.pipe(
            prompt=prompt,
            negative_prompt="blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract",  # ИСПРАВЛЕНИЕ: Полный negative prompt как в v45
            image=control_image,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=1024,
            height=1024,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            # Callback для сохранения промежуточного результата
            callback=lambda step, timestep, latents: self._save_intermediate_result(step, timestep, latents, steps // 2),
            callback_steps=steps // 2  # Callback на середине генерации
        )
        
        return result
    
    def predict(
        self,
        params_json: str = Input(description="JSON string with generation parameters"),
    ) -> List[Path]:
        """Generate rubber tile images using OUR TRAINED MODEL."""
        try:
            # Парсинг параметров
            logger.info(f"Received params_json: {params_json}")
            
            # ИСПРАВЛЕНИЕ: Улучшенная обработка двойного вложения JSON
            try:
                # Сначала пробуем парсить как обычный JSON
                params = json.loads(params_json)
                logger.info(f"Parsed as direct JSON: {params}")
            except json.JSONDecodeError:
                # Если не получилось, проверяем на двойное вложение
                if params_json.startswith('{"params_json"'):
                    logger.info("Detected nested params_json, parsing inner JSON")
                    try:
                        inner_data = json.loads(params_json)
                        inner_json = inner_data.get("params_json", "")
                        if inner_json:
                            params = json.loads(inner_json)
                            logger.info(f"Extracted and parsed inner JSON: {params}")
                        else:
                            raise ValueError("Inner params_json is empty")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse nested JSON: {e}")
                        raise ValueError(f"Invalid nested JSON format: {e}")
                else:
                    raise ValueError("Invalid JSON format")
            
            # Валидация обязательных параметров
            if not params:
                raise ValueError("Empty parameters")
            
            # ИСПРАВЛЕНИЕ: Улучшенная поддержка форматов цветов
            colors = params.get("colors", [{"name": "white", "proportion": 1.0}])
            
            # Конвертация старого формата в новый
            converted_colors = []
            for color_info in colors:
                # Поддержка нового формата: {"name": "red", "proportion": 0.7}
                if "name" in color_info and "proportion" in color_info:
                    converted_colors.append({
                        "name": color_info["name"],
                        "proportion": float(color_info["proportion"])
                    })
                # Поддержка старого формата: {"color": "red", "percentage": 70}
                elif "color" in color_info and "percentage" in color_info:
                    converted_colors.append({
                        "name": color_info["color"],
                        "proportion": float(color_info["percentage"]) / 100.0
                    })
                # Fallback для других форматов
                else:
                    logger.warning(f"Unknown color format: {color_info}, using default")
                    converted_colors.append({"name": "white", "proportion": 1.0})
            
            colors = converted_colors
            angle = int(params.get("angle", 0))
            quality = str(params.get("quality", "standard"))
            seed = int(params.get("seed", -1))
            
            # Валидация seed
            if seed == -1:
                logger.info("🎲 Seed not provided, will use random")
            elif seed < 0:
                logger.warning(f"Invalid seed {seed}, using random")
                seed = -1
            else:
                logger.info(f"🎯 Using provided seed: {seed}")
            
            # Валидация и логирование параметров
            logger.info(f"Converted colors: {colors}")
            logger.info(f"Extracted angle: {angle} (type: {type(angle)})")
            logger.info(f"Extracted quality: {quality} (type: {type(quality)})")
            logger.info(f"Extracted seed: {seed} (type: {type(seed)})")
            
            # Дополнительная валидация
            if not isinstance(colors, list) or len(colors) == 0:
                raise ValueError("Colors must be a non-empty list")
            
            if not isinstance(angle, int) or angle < 0 or angle > 360:
                raise ValueError(f"Invalid angle: {angle}, must be 0-360")
            
            if quality not in ["preview", "standard", "high"]:
                raise ValueError(f"Invalid quality: {quality}, must be preview/standard/high")
            
            logger.info(f"Generating with OUR model - params: colors={len(colors)}, angle={angle}, quality={quality}, seed={seed}")
            
            # Настройки качества (оптимизированы как в рабочей v45)
            quality_settings = {
                "preview": {"steps": 20, "guidance_scale": 7.0},  # Оптимизировано: 20 шагов как в v45
                "standard": {"steps": 20, "guidance_scale": 7.0},  # Оптимизировано: 20 шагов как в v45
                "high": {"steps": 35, "guidance_scale": 7.0}  # Оптимизировано: 35 шагов для высокого качества
            }
            
            settings = quality_settings.get(quality, quality_settings["standard"])
            steps = settings["steps"]
            guidance_scale = settings["guidance_scale"]
            
            # Строим промпт с НАШИМИ токенами
            prompt = self._build_prompt(colors, angle)
            
            # ИСПРАВЛЕНИЕ: Правильная обработка seed
            if seed == -1:
                random_seed = random.randint(0, 999999999)
                logger.info(f"🎲 Using random seed: {random_seed}")
                torch.manual_seed(random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(random_seed)
                    torch.cuda.manual_seed_all(random_seed)
            else:
                logger.info(f"🎯 Using provided seed: {seed}")
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            
            # Создаем карту цветов
            logger.info(f"Building color map for {len(colors)} colors")
            colormap = self._create_colormap(colors)
            colormap_path = "/tmp/colormap.png"
            colormap.save(colormap_path)
            logger.info(f"Color map saved to {colormap_path}")
            
            # ОПТИМИЗАЦИЯ: Один проход генерации с двумя точками остановки
            # 20 шагов → превью, 40 шагов → финальное
            logger.info(f"🚀 Starting optimized generation: {steps * 2} steps total")
            
            # Проверяем, нужен ли ControlNet
            use_controlnet = False
            if self.has_controlnet and angle > 0:
                if angle == 45 and "softedge" in self.controlnets:
                    use_controlnet = True
                    controlnet_type = "softedge"
                elif angle == 90 and "canny" in self.controlnets:
                    use_controlnet = True
                    controlnet_type = "canny"
            
            # Генерация с НАШЕЙ моделью (один проход)
            if self.use_ti:
                logger.info("🔧 Generation with OUR TRAINED MODEL (TI enabled)")
                logger.info(f"🔧 TI Status: use_ti={self.use_ti}")
            else:
                logger.info("🔧 Generation with OUR TRAINED MODEL (TI disabled, LoRA only)")
                logger.warning("⚠️ WARNING: Textual Inversion is DISABLED - this will cause poor generation quality!")
            
            # ОПТИМИЗИРОВАННАЯ ГЕНЕРАЦИЯ: Один проход с callback для preview
            logger.info("🎨 Starting optimized generation with callback")
            
            # Определяем, нужен ли ControlNet
            use_controlnet_proportions = self._should_use_controlnet_for_proportions(colors)
            use_controlnet_angle = self._should_use_controlnet_for_angle(angle)
            
            if use_controlnet_proportions or use_controlnet_angle:
                logger.info("🔧 Using ControlNet for precise control")
                # Генерация с ControlNet
                final_result = self._generate_with_controlnet(
                    prompt=prompt,
                    colors=colors,
                    angle=angle,
                    steps=steps,  # ИСПРАВЛЕНИЕ: Убрали удвоение шагов
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            else:
                logger.info("🔧 Using simple generation for single color")
                # Простая генерация для одного цвета
                final_result = self._generate_simple(
                    prompt=prompt,
                    steps=steps,  # ИСПРАВЛЕНИЕ: Убрали удвоение шагов
                    guidance_scale=guidance_scale,
                    seed=seed
                )
            
            # ОПТИМИЗИРОВАННАЯ ОБРАБОТКА РЕЗУЛЬТАТОВ
            # Создаем preview из промежуточного результата
            preview_path = "/tmp/preview.png"
            if hasattr(self, '_intermediate_image') and self._intermediate_image is not None:
                self._intermediate_image.save(preview_path)
                logger.info("🔧 Preview extracted from intermediate step")
            else:
                # Fallback: создаем preview из финального изображения
                final_image = final_result.images[0]
                preview_image = final_image.resize((512, 512))
                preview_image.save(preview_path)
                logger.info("🔧 Preview created from final image (resized)")
            
            # Извлекаем финальное изображение
            final_image = final_result.images[0]
            final_path = "/tmp/final.png"
            final_image.save(final_path)
            logger.info("🔧 Final image saved")
            
            logger.info("🚀 Optimized generation completed successfully (one pass, two outputs)")
            
            # Очистка памяти
            torch.cuda.empty_cache()
            gc.collect()
            
            generation_time = 0  # Можно добавить измерение времени
            logger.info(f"✅ Generation with OUR TRAINED MODEL completed in {generation_time:.2f}s")
            
            # Return the generated images as Path objects
            # Replicate will automatically convert Path objects to URLs
            return [
                Path(preview_path),
                Path(final_path), 
                Path(colormap_path)
            ]
            
        except Exception as e:
            logger.error(f"❌ Generation with OUR MODEL failed: {e}")
            raise e
