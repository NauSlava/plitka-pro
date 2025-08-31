# predict.py - Основной файл предсказания для модели "nauslava/plitka-pro-project:v4.4.36"
# Использует НАШУ обученную модель с LoRA и Textual Inversion (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)

import os
import torch
import random
import gc
import json
import logging
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageColor
import numpy as np
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from transformers import CLIPTextModel, T5EncoderModel
from cog import BasePredictor, Input

class Predictor(BasePredictor):
    def __init__(self):
        self.device = None
        self.pipe = None
    
    def setup(self):
        """Инициализация модели при запуске сервера."""
        logger.info("🚀 Инициализация модели v4.4.37-pre (КАЧЕСТВО/АНТИ-МОЗАИКА/LEGEND)...")
        
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
        
        # 4. Загрузка НАШИХ обученных LoRA (совместимый API)
        logger.info("🔧 Загрузка НАШИХ LoRA адаптеров...")
        lora_path = "/src/model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
        try:
            # Загружаем с именем адаптера и устанавливаем вес 0.75 (из профиля)
            self.pipe.load_lora_weights(lora_path, adapter_name="rt")
            if hasattr(self.pipe, "set_adapters"):
                self.pipe.set_adapters(["rt"], adapter_weights=[0.75])
                try:
                    self.pipe.fuse_lora()
                except Exception:
                    # В некоторых версиях fuse_lora отсутствует — это не критично
                    pass
                logger.info("✅ LoRA адаптеры загружены (weight=0.75)")
            else:
                # Совместимость со старыми diffusers: set_adapters отсутствует
                try:
                    if hasattr(self.pipe, "fuse_lora"):
                        self.pipe.fuse_lora()
                except Exception:
                    pass
                logger.info("ℹ️ set_adapters недоступен в этой версии diffusers; используем дефолтный вес LoRA")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки LoRA: {e}")
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
        
        # 8. Оптимизации VAE
        logger.info("🚀 Применение VAE оптимизаций...")
        self.pipe.vae.enable_slicing()
        # Отключаем tiling, чтобы избежать "пэчворк"-артефактов
        # self.pipe.vae.enable_tiling()
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
        
        logger.info("🎉 Модель v4.4.36 успешно инициализирована (ИСПРАВЛЕНИЕ ПЕРЕПУТАННЫХ РАЗМЕРОВ)!")
    
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
        return (
            "text, watermark, logo, signature, blur, low quality, distorted, object,"
            " blurry, worst quality, deformed, 3d render, cartoon, abstract, painting,"
            " drawing, sketch, low resolution, mosaic, checkerboard, grid, patchwork,"
            " tiled, square blocks, seams, borders, rectangles, collage, large blocks"
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
                result.append({"name": color_name, "proportion": max(0.0, min(1.0, percent / 100.0))})
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
    
    def predict(self, prompt: str = Input(description="Описание цветов резиновой плитки", default="100% red"), 
                negative_prompt: Optional[str] = Input(description="Негативный промпт", default=None), 
                seed: int = Input(description="Сид для воспроизводимости", default=-1)) -> List[Path]:
        """Генерация изображения резиновой плитки с использованием НАШЕЙ обученной модели."""
        
        try:
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
            
            # Стандартный негативный промпт
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
            
            # Генерация изображения
            logger.info("🚀 Запуск pipeline для генерации...")
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=35,
                guidance_scale=6.7,
                width=1024,
                height=1024,
                generator=torch.Generator(device=self.device).manual_seed(seed)
            )
            logger.info("✅ Pipeline завершен успешно")
            
            # Сохранение результатов
            final_image = result.images[0]
            logger.info(f"📊 Размер сгенерированного изображения: {final_image.size}")
            
            # Создание preview (уменьшенная версия)
            preview_image = final_image.resize((512, 512), Image.Resampling.LANCZOS)
            logger.info("✅ Preview изображение создано")
            
            # Сохранение файлов
            final_path = "/tmp/final.png"
            preview_path = "/tmp/preview.png"
            
            final_image.save(final_path)
            preview_image.save(preview_path)
            logger.info("✅ Файлы сохранены")
            
            # Создание colormap (легенды) из входных пропорций
            colormap_path = "/tmp/colormap.png"
            try:
                parsed_colors = self._parse_percent_colors(prompt)
                if not parsed_colors:
                    parsed_colors = [{"name": "white", "proportion": 1.0}]
                colormap_image = self._render_legend(parsed_colors, size=256)
                colormap_image.save(colormap_path)
                logger.info("✅ Colormap создан из входных пропорций")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось построить colormap: {e}")
                Image.new('RGB', (256, 256), color='white').save(colormap_path)
            
            # Очистка памяти
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("🧹 Память очищена")
            
            logger.info("✅ Изображение успешно сгенерировано!")
            
            return [Path(preview_path), Path(final_path), Path(colormap_path)]
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации: {e}")
            logger.error(f"📊 Тип ошибки: {type(e).__name__}")
            logger.error(f"📊 Детали ошибки: {str(e)}")
            raise e
