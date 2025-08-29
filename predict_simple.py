# predict_simple.py - Упрощенная версия без управления памятью и GPU/NPU мониторинга
import os
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
import cv2

from diffusers import (
    StableDiffusionXLPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# УБРАНО: ResourceMonitor класс - упрощена архитектура

# УБРАНО: ОПТИМИЗАЦИЯ ДЛЯ MULTI-GPU И NPU - упрощена архитектура

# УБРАНО: ИСПРАВЛЕНО v4.3.15: ACCELERATE ПОЛНОСТЬЮ УДАЛЕН - упрощена архитектура

# УБРАНО: Фильтры предупреждений - упрощена архитектура

class OptimizedPredictor:
    """Упрощенный предиктор для генерации резиновых плиток."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.has_controlnet = False
        self.controlnet_canny = None
        self.controlnet_softedge = None
        self.controlnet_lineart = None
        
    def setup(self, weights: Optional[str] = None) -> None:
        """Упрощенная инициализация модели."""
        logger.info("Starting model setup...")
        
        # Простое определение устройства
        if torch.cuda.is_available():
            logger.info(f"Found {torch.cuda.device_count()} CUDA GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} - {props.total_memory / (1024**3):.1f}GB")
            logger.info(f"✅ Selected GPU 0: {torch.cuda.get_device_properties(0).name} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB)")
            logger.info("🚀 GPU detected - using CUDA")
        else:
            logger.info("⚠️ No CUDA GPU found - using CPU")
            self.device = "cpu"
        
        logger.info(f"🎯 Using device: {self.device}")
        
        # УБРАНО: Resource monitoring - упрощена архитектура
        
        # УБРАНО: Lazy Loading Architecture - упрощена архитектура
        
        # УБРАНО: Memory management - упрощена архитектура
        
        # Инициализация SDXL pipeline
        logger.info("🚀 Initializing SDXL pipeline...")
        
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            
            # Перемещение на устройство
            self.pipe = self.pipe.to(self.device)
            logger.info(f"✅ SDXL pipeline moved to {self.device}")
            
            # Загрузка LoRA
            logger.info("Loading LoRA weights...")
            self.pipe.load_lora_weights("nauslava/plitka-pro-lora-v4")
            logger.info("✅ LoRA weights loaded successfully")
            
            # Загрузка Textual Inversion
            logger.info("Loading Textual Inversion embeddings...")
            try:
                self.pipe.load_textual_inversion("nauslava/plitka-pro-textual-inversion-v4")
                logger.info("✅ Textual Inversion loaded successfully")
            except Exception as e:
                logger.warning(f"Standard TI load failed ({e}). Falling back to manual SDXL dual-encoder TI install...")
                self._install_sdxl_textual_inversion_dual("nauslava/plitka-pro-textual-inversion-v4", self.pipe, token_g="<s0>", token_l="<s0>")
                logger.info("✅ Textual Inversion loaded with manual method")
            
            # Настройка scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            logger.info("✅ Scheduler configured successfully")
            
            # VAE оптимизации
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            logger.info("✅ VAE slicing enabled")
            logger.info("✅ VAE tiling enabled")
            
            logger.info("🎉 Model setup completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            raise e
    
    def _install_sdxl_textual_inversion_dual(self, ti_path: str, pipeline, token_g: str, token_l: str) -> None:
        """Install SDXL textual inversion that contains separate embeddings for CLIP-G and CLIP-L encoders."""
        logger.info("🔤 Installing dual-encoder SDXL textual inversion...")
        
        try:
            # Load the textual inversion embeddings
            from safetensors import safe_open
            
            # Load embeddings from the model path
            model_path = ti_path
            if not os.path.exists(model_path):
                # Try to download from Hugging Face Hub
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(repo_id=ti_path)
            
            # Load the embeddings
            with safe_open(os.path.join(model_path, "learned_embeds.safetensors"), framework="pt", device="cpu") as f:
                string_to_param = {key: value for key, value in f.get_tensor("string_to_param").items()}
            
            # Extract token names
            token_names = list(string_to_param.keys())
            logger.info(f"🔤 Textual inversion contains {len(token_names)} token(s)")
            logger.info(f"🔤 Token names: {token_names}")
            
            # Install embeddings for both CLIP-G and CLIP-L
            pipeline.load_textual_inversion(model_path, weight_name="learned_embeds.safetensors")
            
            logger.info(f"✅ SDXL textual inversion installed manually for {len(token_names)} token(s): {token_names}")
            
        except Exception as e:
            logger.error(f"❌ Failed to install SDXL textual inversion: {e}")
            raise e
    
    def _build_prompt(self, colors: List[Dict[str, Any]]) -> str:
        """Build prompt from color specifications."""
        if not colors:
            return "ohwx_rubber_tile <s0><s1>, 100% white"
        
        color_parts = []
        for color_info in colors:
            name = color_info.get("name", "white")
            proportion = color_info.get("proportion", 100)
            color_parts.append(f"{proportion}% {name}")
        
        color_str = ", ".join(color_parts)
        return f"ohwx_rubber_tile <s0><s1>, {color_str}, photorealistic rubber tile, matte texture, top view, rubber granules, textured surface"
    
    def _should_use_controlnet(self, angle: int) -> Tuple[bool, str]:
        """Determine if ControlNet should be used based on angle."""
        if angle == 0:
            return False, "Угол 0° (вид сверху) - единственный надежный ракурс без ControlNet"
        else:
            return True, f"Угол {angle}° требует ControlNet для геометрического контроля"
    
    def predict(self, params_json: str) -> List[str]:
        """Упрощенная функция предсказания."""
        start_time = time.time()
        
        try:
            # Парсинг параметров
            params = json.loads(params_json)
            
            colors = params.get("colors", [])
            angle = int(params.get("angle", 0))
            seed = int(params.get("seed", -1))
            quality = str(params.get("quality", "standard"))
            overrides: Dict[str, Any] = params.get("overrides", {}) or {}
            
            logger.info(f"Generating with params: colors={len(colors)}, angle={angle}, quality={quality}, seed={seed}")
            
            # УБРАНО: Resource status logging - упрощена архитектура
            
            # УБРАНО: ИСПРАВЛЕННЫЕ ПРОФИЛИ КАЧЕСТВА v4.3.15 - упрощена архитектура
            # Простые параметры качества
            if quality == "preview":
                steps_preview, steps_final = 25, 30
                size_preview, size_final = (512, 512), (1024, 1024)
                guidance_scale_default = 5.5
            elif quality == "high":
                steps_preview, steps_final = 50, 80
                size_preview, size_final = (512, 512), (1024, 1024)
                guidance_scale_default = 5.0
            else:  # standard
                steps_preview, steps_final = 40, 60
                size_preview, size_final = (512, 512), (1024, 1024)
                guidance_scale_default = 5.5
            
            # Apply overrides
            num_inference_steps_preview = int(overrides.get("num_inference_steps_preview", steps_preview))
            num_inference_steps_final = int(overrides.get("num_inference_steps_final", steps_final))
            guidance_scale = float(overrides.get("guidance_scale", guidance_scale_default))
            
            # Build clean prompt
            base_prompt = self._build_prompt(colors)
            logger.info(f"Generated prompt: {base_prompt}")
            
            negative_prompt = overrides.get(
                "negative_prompt",
                "object, blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, smooth, flat",
            )
            
            # Generator
            generator = torch.manual_seed(seed) if seed != -1 else torch.Generator(device=self.device)
            if seed == -1:
                seed = generator.seed()
            
            # Create color map
            colormap_path = "/tmp/colormap.png"
            logger.info(f"Building color map for {len(colors)} colors")
            colormap_img = self._build_color_map(colors, size_final, colormap_path)
            logger.info(f"Color map saved to {colormap_path}")
            
            # УБРАНО: ControlNet logic - упрощена архитектура
            use_controlnet = False
            control_preview = None
            control_final = None
            
            # Generate preview
            preview_start = time.time()
            logger.info(f"Generating preview with {num_inference_steps_preview} steps, guidance_scale={guidance_scale}")
            
            # УБРАНО: ControlNet status logging - упрощена архитектура
            
            # УБРАНО: Управление памятью - упрощена архитектура
            
            # Генерируем preview с параметрами preview
            preview_params = {
                "prompt": base_prompt,
                "negative_prompt": negative_prompt,
                "width": size_preview[0],
                "height": size_preview[1],
                "num_inference_steps": num_inference_steps_preview,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            
            # Add ControlNet image if enabled and available
            if use_controlnet and control_preview is not None and self.has_controlnet:
                preview_params["image"] = control_preview
                logger.info("✅ Using ControlNet for preview generation")
            
            logger.info("🔧 Preview generation с полным pipeline на GPU")
            with torch.no_grad():
                preview = self.pipe(**preview_params).images[0]
            
            logger.info("🔧 Preview generation завершен успешно")
            
            # Generate final
            final_start = time.time()
            logger.info(f"Generating final image with {num_inference_steps_final} steps")
            
            # Prepare generation parameters
            gen_params = {
                "prompt": base_prompt,
                "negative_prompt": negative_prompt,
                "width": size_final[0],
                "height": size_final[1],
                "num_inference_steps": num_inference_steps_final,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            
            # Add ControlNet image if enabled and available
            if use_controlnet and control_final is not None and self.has_controlnet:
                gen_params["image"] = control_final
                logger.info("✅ Using ControlNet for final generation")
            else:
                logger.info("ℹ️ Final generation without ControlNet")
            
            # УБРАНО: Управление памятью - упрощена архитектура
            
            # Генерируем final с параметрами final
            with torch.no_grad():
                final = self.pipe(**gen_params).images[0]
            
            logger.info("🔧 Final generation завершен успешно")
            
            # Сохраняем изображения
            preview_path = Path("/tmp/preview.png")
            final_path = Path("/tmp/final.png")
            colormap_path = Path("/tmp/colormap.png")
            
            preview.save(preview_path)
            final.save(final_path)
            colormap_img.save(colormap_path)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Generation completed in {total_time:.2f}s")
            
            return [str(preview_path), str(final_path), str(colormap_path)]
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def _build_color_map(self, colors: List[Dict[str, Any]], size: Tuple[int, int], output_path: str) -> Image.Image:
        """Build a simple color map image."""
        width, height = size
        
        # Create a simple color map
        if not colors:
            # Default white
            color_map = Image.new('RGB', (width, height), (255, 255, 255))
        else:
            # Create gradient based on colors
            color_map = Image.new('RGB', (width, height))
            pixels = color_map.load()
            
            for y in range(height):
                for x in range(width):
                    # Simple color mixing based on proportions
                    r, g, b = 0, 0, 0
                    total_proportion = sum(color.get("proportion", 0) for color in colors)
                    
                    for color_info in colors:
                        proportion = color_info.get("proportion", 0) / total_proportion
                        color_name = color_info.get("name", "white").lower()
                        
                        # Simple color mapping
                        if color_name == "red":
                            r += int(255 * proportion)
                        elif color_name == "green":
                            g += int(255 * proportion)
                        elif color_name == "blue":
                            b += int(255 * proportion)
                        elif color_name == "yellow":
                            r += int(255 * proportion)
                            g += int(255 * proportion)
                        elif color_name == "purple":
                            r += int(128 * proportion)
                            b += int(255 * proportion)
                        elif color_name == "orange":
                            r += int(255 * proportion)
                            g += int(128 * proportion)
                        elif color_name == "black":
                            pass  # Keep black
                        else:  # white or unknown
                            r += int(255 * proportion)
                            g += int(255 * proportion)
                            b += int(255 * proportion)
                    
                    pixels[x, y] = (min(255, r), min(255, g), min(255, b))
        
        return color_map
