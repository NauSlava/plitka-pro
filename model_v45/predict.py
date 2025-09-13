# predict.py - Основной файл предсказания для модели "nauslava/rubber-tile-lora-v45"
# Использует НАШУ обученную модель с LoRA и Textual Inversion

import os
import torch
import random
import gc
from typing import Optional
from PIL import Image
import numpy as np

# Переменные окружения для оптимизации
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"
os.environ["HF_MODELS_CACHE"] = "/tmp/hf_models_cache"
os.environ["TRANSFORMERS_CACHE_MIGRATION_DISABLE"] = "1"
os.environ["HF_HUB_CACHE_MIGRATION_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, T5EncoderModel

class Predictor:
    def __init__(self):
        self.device = None
        self.pipe = None
        self.setup()
    
    def setup(self):
        """Инициализация модели при запуске сервера."""
        print("🚀 Инициализация модели v45...")
        
        # 1. Определение устройства
        if torch.cuda.is_available():
            self.device = "cuda"
            # Выбор GPU с наибольшей памятью
            best_gpu = max(range(torch.cuda.device_count()), 
                          key=lambda i: torch.cuda.get_device_properties(i).total_memory)
            torch.cuda.set_device(best_gpu)
            print(f"✅ Используется GPU: {torch.cuda.get_device_name(best_gpu)}")
        else:
            self.device = "cpu"
            print("⚠️ CUDA недоступен, используется CPU")
        
        # 2. Загрузка SDXL pipeline
        print("📥 Загрузка базовой модели SDXL...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            resume_download=False
        )
        
        # 3. Перемещение на GPU
        self.pipe = self.pipe.to(self.device)
        
        # 4. Загрузка НАШИХ обученных LoRA
        print("🔧 Загрузка НАШИХ LoRA адаптеров...")
        lora_path = "/src/model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
        self.pipe.set_adapters(["rubber-tile-lora-v4_sdxl_lora"], adapter_weights=[0.7])
        self.pipe.fuse_lora()
        print("✅ LoRA адаптеры загружены и объединены")
        
        # 5. Загрузка НАШИХ обученных Textual Inversion
        print("🔤 Загрузка НАШИХ Textual Inversion...")
        ti_path = "/src/model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors"
        self.pipe.load_textual_inversion(ti_path, token="<s0>")
        print("✅ Textual Inversion загружен")
        
        # 6. Настройка планировщика
        print("⚙️ Настройка планировщика...")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        
        # 7. Оптимизации VAE
        print("🚀 Применение VAE оптимизаций...")
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # 8. Очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("🎉 Модель v45 успешно инициализирована!")
    
    def _build_prompt(self, color_description: str) -> str:
        """Построение полного промпта с использованием НАШИХ обученных токенов."""
        # Базовый промпт с НАШИМИ токенами активации
        base_prompt = "ohwx_rubber_tile <s0><s1>"
        
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
    
    def predict(self, prompt: str, negative_prompt: Optional[str] = None, seed: int = -1) -> Image.Image:
        """Генерация изображения резиновой плитки с использованием НАШЕЙ обученной модели."""
        
        # Стандартный негативный промпт
        if negative_prompt is None:
            negative_prompt = "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, painting, drawing, text, sketch, low resolution"
        
        # Установка сида
        if seed == -1:
            seed = random.randint(0, 999999999)
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Преобразование простого промпта в полный формат с НАШИМИ токенами
        full_prompt = self._build_prompt(prompt)
        
        print(f"🎨 Генерация изображения...")
        print(f"📝 Промпт: {full_prompt}")
        print(f"🚫 Негативный промпт: {negative_prompt}")
        print(f"🎲 Сид: {seed}")
        
        # Генерация изображения
        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.0,
            width=1024,
            height=1024,
            generator=torch.Generator(device=self.device).manual_seed(seed)
        )
        
        # Очистка памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("✅ Изображение успешно сгенерировано!")
        return result.images[0]
