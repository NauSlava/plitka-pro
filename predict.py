# predict.py - Optimized Version with Dual Pipeline Architecture
from cog import BasePredictor, Input, Path
from typing import List, Dict, Any, Tuple, Optional
import json
import os
import random
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import cv2

import torch
from safetensors.torch import load_file as load_safetensors
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
)

# Максимально агрессивное подавление предупреждений
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*")

# CUDA Memory Optimization
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Suppress specific warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Absolute cache paths inside the container
WEIGHTS_ROOT = "/weights"
if os.path.isdir(WEIGHTS_ROOT):
    SDXL_CACHE_DIR = os.path.join(WEIGHTS_ROOT, "sdxl-cache")
    CONTROLNET_CACHE_DIR = os.path.join(WEIGHTS_ROOT, "controlnet-cache")
else:
    SDXL_CACHE_DIR = "/src/sdxl-cache"
    CONTROLNET_CACHE_DIR = "/src/controlnet-cache"
REFS_DIR = "/src/model_files/refs"

# ControlNet local subfolders
CONTROLNET_CANNY_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-canny-sdxl-1.0")
CONTROLNET_HED_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-hed-sdxl-1.0")
CONTROLNET_SOFTEDGE_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-softedge-sdxl-1.0")
CONTROLNET_LINEART_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-lineart-sdxl-1.0")

# HF repo ids for fallback
SDXL_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_CANNY_REPO_ID = "diffusers/controlnet-canny-sdxl-1.0"
CONTROLNET_LINEART_REPO_ID = "diffusers/controlnet-lineart-sdxl-1.0"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def average_color_from_image(image_path: str) -> Tuple[int, int, int]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(x) for x in mean.tolist())


def sample_color_for_name(color_name: str) -> Tuple[int, int, int]:
    """Улучшенный цветовой маппинг с точными цветами"""
    # Сначала пробуем найти цвет в референсах
    color_folder = os.path.join(REFS_DIR, color_name)
    if os.path.isdir(color_folder):
        candidates: List[str] = []
        for fn in os.listdir(color_folder):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(os.path.join(color_folder, fn))
        if candidates:
            choice = random.choice(candidates)
            return average_color_from_image(choice)
    
    # Улучшенный fallback с точными цветами
    color_map = {
        "black": (0, 0, 0),           # Истинно черный
        "white": (255, 255, 255),     # Истинно белый
        "red": (255, 0, 0),           # Яркий красный
        "blue": (0, 0, 255),          # Яркий синий
        "green": (0, 255, 0),         # Яркий зеленый
        "yellow": (255, 255, 0),      # Яркий желтый
        "gray": (128, 128, 128),      # Средний серый
        "brown": (139, 69, 19),       # Коричневый
        "dark green": (0, 100, 0),    # Темно-зеленый
        "orange": (255, 165, 0),      # Оранжевый
        "purple": (128, 0, 128),      # Фиолетовый
        "pink": (255, 192, 203),      # Розовый
    }
    return color_map.get(color_name.lower(), (127, 127, 127))


def build_color_map(colors: List[Dict[str, Any]], size: Tuple[int, int], out_path: str) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (width, height))
    
    # Normalize proportions
    props = [max(0.0, float(c.get("proportion", 0))) for c in colors]
    total = sum(props) or 1.0
    props = [p / total for p in props]

    x0 = 0
    for c, p in zip(colors, props):
        w = int(round(p * width))
        if w <= 0:
            continue
        name = c.get("name")
        hex_color = c.get("hex")
        if hex_color:
            rgb = hex_to_rgb(hex_color)
        elif name:
            rgb = sample_color_for_name(name)
        else:
            rgb = (127, 127, 127)
        block = Image.new("RGB", (w, height), rgb)
        canvas.paste(block, (x0, 0))
        x0 += w

    canvas.save(out_path)
    return canvas


def canny_edge_from_image(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def select_controlnet_by_angle(
    angle: int,
    cn_canny: ControlNetModel,
    cn_softedge: Optional[ControlNetModel],
    cn_lineart: Optional[ControlNetModel],
) -> ControlNetModel:
    angle_norm = int(angle) % 180
    # Prefer canny for 0/90
    if angle_norm in (0, 90):
        return cn_canny
    # For diagonals prefer lineart if available, then softedge/HED, else fallback to canny
    if cn_lineart is not None:
        return cn_lineart
    if cn_softedge is not None:
        return cn_softedge
    return cn_canny


class OptimizedPredictor(BasePredictor):
    def setup(self):
        """Оптимизированная настройка модели с lazy loading архитектурой"""
        logger.info("🚀 Starting optimized model setup with lazy loading...")
        
        # Загружаем только базовые компоненты
        logger.info("📦 Loading shared components...")
        self._load_shared_components() # Pass a default repo ID for shared components
        
        logger.info("�� Loading ControlNet models...")
        self._load_controlnet_models() # Pass default repo IDs for controlnet models
        
        # Инициализируем пайплайны как None для lazy loading
        self.controlnet_pipeline = None
        self.base_pipeline = None
        
        logger.info("✅ Model setup completed with lazy loading architecture")
        logger.info("💡 Pipelines will be created on-demand to optimize memory usage")

    def _get_controlnet_pipeline(self):
        """Lazy loading для ControlNet пайплайна"""
        if self.controlnet_pipeline is None:
            logger.info("🎨 Creating ControlNet pipeline on-demand...")
            # Очищаем память перед созданием нового пайплайна
            torch.cuda.empty_cache()
            
            base_model_src = SDXL_CACHE_DIR if os.path.isdir(SDXL_CACHE_DIR) else SDXL_REPO_ID
            shared_components = self._load_shared_components()
            self.controlnet_pipeline = self._create_controlnet_pipeline(base_model_src, shared_components)
            logger.info("✅ ControlNet pipeline created and cached")
        return self.controlnet_pipeline

    def _get_base_pipeline(self):
        """Lazy loading для базового SDXL пайплайна"""
        if self.base_pipeline is None:
            logger.info("🎯 Creating base SDXL pipeline on-demand...")
            # Очищаем память перед созданием нового пайплайна
            torch.cuda.empty_cache()
            
            base_model_src = SDXL_CACHE_DIR if os.path.isdir(SDXL_CACHE_DIR) else SDXL_REPO_ID
            shared_components = self._load_shared_components()
            self.base_pipeline = self._create_base_pipeline(base_model_src, shared_components)
            logger.info("✅ Base SDXL pipeline created and cached")
        return self.base_pipeline

    def _load_shared_components(self) -> Dict[str, Any]:
        """Load shared components (LoRA, TI) that will be used by both pipelines."""
        logger.info("🔧 Loading LoRA weights...")
        lora_path = os.path.join(WEIGHTS_ROOT, "ohwx_rubber_tile_lora.safetensors")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")
        
        logger.info("🔤 Loading Textual Inversion embeddings...")
        ti_path = os.path.join(WEIGHTS_ROOT, "ohwx_rubber_tile_ti.safetensors")
        if not os.path.exists(ti_path):
            raise FileNotFoundError(f"Textual Inversion embeddings not found at {ti_path}")
        
        return {
            "lora_path": lora_path,
            "ti_path": ti_path
        }

    def _load_controlnet_models(self):
        """Load ControlNet models for edge detection and line control."""
        logger.info("🎯 Loading ControlNet models...")
        
        # Resolve model sources
        canny_model_src = CONTROLNET_CANNY_DIR if os.path.isdir(CONTROLNET_CANNY_DIR) else CONTROLNET_CANNY_REPO_ID
        softedge_path = CONTROLNET_HED_DIR if os.path.isdir(CONTROLNET_HED_DIR) else (
            CONTROLNET_SOFTEDGE_DIR if os.path.isdir(CONTROLNET_SOFTEDGE_DIR) else None
        )
        
        self.controlnet_models = {}
        
        # Load Canny ControlNet
        try:
            self.controlnet_models["canny"] = ControlNetModel.from_pretrained(
                canny_model_src,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            logger.info("✅ Canny ControlNet loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Canny ControlNet: {e}")
            self.controlnet_models["canny"] = None
        
        # Load SoftEdge/HED ControlNet
        if softedge_path:
            try:
                self.controlnet_models["softedge"] = ControlNetModel.from_pretrained(
                    softedge_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                logger.info("✅ SoftEdge ControlNet loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load SoftEdge ControlNet: {e}")
                self.controlnet_models["softedge"] = None
        else:
            self.controlnet_models["softedge"] = None
        
        # Load Lineart ControlNet
        try:
            self.controlnet_models["lineart"] = ControlNetModel.from_pretrained(
                CONTROLNET_LINEART_REPO_ID,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            logger.info("✅ Lineart ControlNet loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Lineart ControlNet: {e}")
            self.controlnet_models["lineart"] = None

    def _create_controlnet_pipeline(self, base_model_src: str, shared_components: Dict[str, Any]) -> StableDiffusionXLControlNetPipeline:
        """Create ControlNet pipeline with optimizations."""
        logger.info("🎨 Creating ControlNet pipeline...")
        
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model_src,
            controlnet=self.controlnet_models['canny'],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        
        # Apply shared components
        self._apply_shared_components(pipeline, shared_components)
        
        # Optimizations
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        
        return pipeline

    def _create_base_pipeline(self, base_model_src: str, shared_components: Dict[str, Any]) -> StableDiffusionXLPipeline:
        """Create base SDXL pipeline without ControlNet."""
        logger.info("🎯 Creating base SDXL pipeline...")
        
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_src,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        
        # Apply shared components
        self._apply_shared_components(pipeline, shared_components)
        
        # Optimizations
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        
        return pipeline

    def _apply_shared_components(self, pipeline, shared_components: Dict[str, Any]):
        """Apply LoRA and Textual Inversion to pipeline."""
        # Load LoRA
        if 'lora_path' in shared_components:
            pipeline.load_lora_weights(shared_components['lora_path'])
            pipeline.fuse_lora()
        
        # Load Textual Inversion
        if 'ti_path' in shared_components:
            try:
                pipeline.load_textual_inversion(shared_components['ti_path'], token="<s0>")
                logger.info("✅ Standard TI load successful")
            except Exception as e:
                logger.warning(f"⚠️ Standard TI load failed ({e}). Using manual SDXL dual-encoder TI install...")
                self._install_sdxl_textual_inversion_dual(
                    shared_components['ti_path'], 
                    pipeline, 
                    token_g="<s0>", 
                    token_l="<s0>"
                )
                logger.info("✅ Manual TI installation completed")

    def predict(
        self,
        params_json: str = Input(description="Business-oriented parameters JSON: colors, angle, seed, quality, overrides")
    ) -> List[Path]:
        """Generate preview/final images using optimized dual pipeline architecture."""
        start_time = time.time()
        try:
            # Parse parameters
            params = json.loads(params_json) if params_json else {}
            if "params_json" in params:
                inner_json = params["params_json"]
                params = json.loads(inner_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid params_json: {e}")

        colors = params.get("colors", [])
        angle = int(params.get("angle", 0))
        seed = int(params.get("seed", -1))
        quality = str(params.get("quality", "standard"))
        overrides: Dict[str, Any] = params.get("overrides", {}) or {}
        use_controlnet = overrides.get("use_controlnet", True)

        logger.info(f"🎨 Generating with params: colors={len(colors)}, angle={angle}, quality={quality}, seed={seed}, controlnet={use_controlnet}")

        # Quality profiles
        if quality == "preview":
            steps_preview, steps_final = 16, 24
            size_preview, size_final = (512, 512), (1024, 1024)
        elif quality == "high":
            steps_preview, steps_final = 24, 40
            size_preview, size_final = (512, 512), (1024, 1024)
        else:  # standard
            steps_preview, steps_final = 20, 30
            size_preview, size_final = (512, 512), (1024, 1024)

        num_inference_steps_preview = int(overrides.get("num_inference_steps_preview", steps_preview))
        num_inference_steps_final = int(overrides.get("num_inference_steps_final", steps_final))
        guidance_scale = float(overrides.get("guidance_scale", 7.5))

        # Build optimized prompt
        prompt_parts = ["ohwx_rubber_tile <s0><s1>"]
        
        if colors:
            color_desc = []
            for color_info in colors:
                name = color_info.get("name", "").lower()
                proportion = color_info.get("proportion", 0)
                if proportion > 0:
                    percentage = int(proportion)  # Правильные проценты
                    color_desc.append(f"{percentage}% {name}")
            
            if color_desc:
                prompt_parts.append(", ".join(color_desc))
        
        prompt_parts.extend([
            "photorealistic rubber tile",
            "matte texture", 
            "top view",
            "rubber granules",
            "textured surface"
        ])
        
        base_prompt = ", ".join(prompt_parts)
        negative_prompt = overrides.get(
            "negative_prompt",
            "object, blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, smooth, flat",
        )

        # Generator
        generator = torch.manual_seed(seed) if seed != -1 else torch.Generator(device="cuda")
        if seed == -1:
            seed = generator.seed()

        # Create color map
        colormap_path = "/tmp/colormap.png"
        logger.info(f"🎨 Building color map for {len(colors)} colors")
        colormap_img = build_color_map(colors, size_final, colormap_path)
        logger.info(f"✅ Color map saved to {colormap_path}")

        # Choose pipeline based on ControlNet setting
        if use_controlnet:
            logger.info("🎯 Using ControlNet pipeline")
            return self._generate_with_controlnet(
                base_prompt, negative_prompt, colormap_img,
                guidance_scale, num_inference_steps_preview, seed
            )
        else:
            logger.info("🎯 Using base SDXL pipeline")
            return self._generate_without_controlnet(
                base_prompt, negative_prompt,
                guidance_scale, num_inference_steps_preview, seed
            )

    def _generate_with_controlnet(self, prompt: str, negative_prompt: str, image: Image.Image, 
                                 guidance_scale: float, num_inference_steps: int, seed: int) -> Image.Image:
        """Generate image using ControlNet pipeline with lazy loading."""
        # Получаем пайплайн через lazy loading
        pipeline = self._get_controlnet_pipeline()
        
        # Select appropriate ControlNet model based on angle
        controlnet_model = self._select_controlnet_model(0)  # Default to Canny for now
        
        if controlnet_model is None:
            logger.warning("⚠️ No ControlNet model available, falling back to base pipeline")
            return self._generate_without_controlnet(prompt, negative_prompt, guidance_scale, num_inference_steps, seed)
        
        # Generate control image
        control_image = self._create_control_image(image, controlnet_model)
        
        # Generate image
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        )
        
        return result.images[0]

    def _select_controlnet_model(self, angle: int):
        """Select appropriate ControlNet model based on angle."""
        if angle in [0, 90, 180, 270]:
            return self.controlnet_models.get("canny")
        elif angle in [30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330]:
            return self.controlnet_models.get("lineart")
        else:
            return self.controlnet_models.get("softedge") or self.controlnet_models.get("canny")

    def _create_control_image(self, image: Image.Image, controlnet_model) -> Image.Image:
        """Create control image using appropriate ControlNet model."""
        if "canny" in str(controlnet_model):
            return canny_edge_from_image(image, 80, 160)
        elif "lineart" in str(controlnet_model):
            # For lineart, we'll use a simple edge detection for now
            return canny_edge_from_image(image, 100, 200)
        else:
            # Default to canny
            return canny_edge_from_image(image, 80, 160)

    def _generate_without_controlnet(self, prompt: str, negative_prompt: str, 
                                   guidance_scale: float, num_inference_steps: int, seed: int) -> Image.Image:
        """Generate image using base SDXL pipeline with lazy loading."""
        # Получаем пайплайн через lazy loading
        pipeline = self._get_base_pipeline()
        
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        )
        
        return result.images[0]

    def _install_sdxl_textual_inversion_dual(self, ti_path: str, pipeline, token_g: str, token_l: str) -> None:
        """Install SDXL textual inversion that contains separate embeddings for CLIP-G and CLIP-L encoders."""
        try:
            # Load the safetensors file
            state_dict = load_safetensors(ti_path)
            
            # Check if it's dual-encoder format
            if 'clip_g' in state_dict and 'clip_l' in state_dict:
                logger.info("🔤 Installing dual-encoder SDXL textual inversion...")
                
                # Get embeddings
                clip_g_embeddings = state_dict['clip_g']
                clip_l_embeddings = state_dict['clip_l']
                
                # Determine number of tokens
                num_tokens = clip_g_embeddings.shape[0]
                logger.info(f"🔤 Textual inversion contains {num_tokens} token(s)")
                
                # Generate token names
                token_names = [f"<s{i}>" for i in range(num_tokens)]
                logger.info(f"🔤 Token names: {token_names}")
                
                # Install in text_encoder_2 (CLIP-G)
                if hasattr(pipeline, 'text_encoder_2'):
                    pipeline.text_encoder_2.resize_token_embeddings(len(pipeline.tokenizer_2) + num_tokens)
                    with torch.no_grad():
                        pipeline.text_encoder_2.get_input_embeddings().weight[-num_tokens:] = clip_g_embeddings
                
                # Install in text_encoder (CLIP-L)
                if hasattr(pipeline, 'text_encoder'):
                    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer) + num_tokens)
                    with torch.no_grad():
                        pipeline.text_encoder.get_input_embeddings().weight[-num_tokens:] = clip_l_embeddings
                
                # Add tokens to tokenizers
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


# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True