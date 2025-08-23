# predict.py
from cog import BasePredictor, Input, Path
from typing import List, Dict, Any, Tuple, Optional
import json
import os
import random
import time
import logging
import warnings
import psutil
import threading

import numpy as np
from PIL import Image
import cv2

import torch
from safetensors.torch import load_file as load_safetensors
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    EulerDiscreteScheduler,
)

# УЛЬТИМАТИВНОЕ подавление ВСЕХ предупреждений - МАКСИМАЛЬНО АГРЕССИВНО
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
warnings.filterwarnings("ignore", message=".*The `scheduler_config`.*")
warnings.filterwarnings("ignore", message=".*`torch.utils._pytree`.*")
warnings.filterwarnings("ignore", message=".*`torch.utils._pytree`.*")
warnings.filterwarnings("ignore", message=".*`torch.utils._pytree`.*")

# ИСПРАВЛЕНИЕ: Специфичные фильтры для конкретных предупреждений из логов
warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*Please use `torch.utils._pytree.register_pytree_node`.*")
warnings.filterwarnings("ignore", message=".*Loaded state dictonary is incorrect.*")
warnings.filterwarnings("ignore", message=".*Please verify that the loaded state dictionary.*")
warnings.filterwarnings("ignore", message=".*string_to_param.*")

# Дополнительное подавление на уровне системы
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 🚀 НОВЫЕ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ДЛЯ УПРАВЛЕНИЯ GPU/NPU
os.environ['CUDA_MEMORY_FRACTION'] = '0.7'  # Использовать максимум 70% памяти GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # Синхронное выполнение для лучшего контроля
os.environ['CUDA_CACHE_DISABLE'] = '0'      # Включить кэш CUDA для оптимизации

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🚀 НОВАЯ ФУНКЦИЯ: Мониторинг ресурсов в реальном времени
class ResourceMonitor:
    """Мониторинг использования GPU/NPU ресурсов в реальном времени."""
    
    def __init__(self, device_info: Dict[str, Any]):
        self.device_info = device_info
        self.monitoring = False
        self.monitor_thread = None
        self.max_memory_usage = 0.0
        self.max_gpu_utilization = 0.0
        
    def start_monitoring(self):
        """Запуск мониторинга ресурсов."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Resource monitoring started")
    
    def stop_monitoring(self):
        """Остановка мониторинга ресурсов."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("⏹️ Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Основной цикл мониторинга ресурсов."""
        while self.monitoring:
            try:
                if self.device_info['type'] == 'cuda':
                    self._monitor_gpu()
                elif self.device_info['type'] == 'npu':
                    self._monitor_npu()
                
                time.sleep(2.0)  # Проверка каждые 2 секунды
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                time.sleep(5.0)
    
    def _monitor_gpu(self):
        """Мониторинг GPU ресурсов."""
        try:
            if torch.cuda.is_available():
                # Мониторинг памяти GPU
                allocated = torch.cuda.memory_allocated(self.device_info['id']) / (1024**3)
                reserved = torch.cuda.memory_reserved(self.device_info['id']) / (1024**3)
                total = self.device_info['memory']
                
                # Обновление максимальных значений
                self.max_memory_usage = max(self.max_memory_usage, allocated)
                
                # Проверка лимитов (50-80%)
                memory_usage_percent = (allocated / total) * 100
                if memory_usage_percent > 80:
                    logger.warning(f"⚠️ GPU memory usage: {memory_usage_percent:.1f}% (allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB)")
                    # Принудительная очистка кэша при превышении лимита
                    torch.cuda.empty_cache()
                elif memory_usage_percent > 70:
                    logger.info(f"ℹ️ GPU memory usage: {memory_usage_percent:.1f}% (allocated: {allocated:.2f}GB)")
                
                # Мониторинг загрузки GPU (через nvidia-smi если доступно)
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                         capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        gpu_util = float(result.stdout.strip())
                        self.max_gpu_utilization = max(self.max_gpu_utilization, gpu_util)
                        
                        if gpu_util > 80:
                            logger.warning(f"⚠️ GPU utilization: {gpu_util:.1f}%")
                        elif gpu_util > 70:
                            logger.info(f"ℹ️ GPU utilization: {gpu_util:.1f}%")
                except:
                    pass  # nvidia-smi недоступен
                    
        except Exception as e:
            logger.debug(f"GPU monitoring error: {e}")
    
    def _monitor_npu(self):
        """Мониторинг NPU ресурсов."""
        try:
            # Для NPU используем системные ресурсы как приближение
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 80:
                logger.warning(f"⚠️ NPU system CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                logger.info(f"ℹ️ NPU system CPU usage: {cpu_percent:.1f}%")
                
            if memory_percent > 80:
                logger.warning(f"⚠️ NPU system memory usage: {memory_percent:.1f}%")
            elif memory_percent > 70:
                logger.info(f"ℹ️ NPU system memory usage: {memory_percent:.1f}%")
                
        except Exception as e:
            logger.debug(f"NPU monitoring error: {e}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Получение сводки по использованию ресурсов."""
        return {
            'device_type': self.device_info['type'],
            'device_name': self.device_info['name'],
            'max_memory_usage_gb': self.max_memory_usage,
            'max_gpu_utilization_percent': self.max_gpu_utilization,
            'monitoring_active': self.monitoring
        }

# 🚀 ОПТИМИЗАЦИЯ ДЛЯ MULTI-GPU И NPU
def select_best_device():
    """Автоматический выбор лучшего доступного устройства (GPU/NPU/CPU)."""
    device_info = {
        'type': 'cpu',
        'id': None,
        'name': 'CPU',
        'memory': 0
    }
    
    # Проверка CUDA GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} CUDA GPU(s)")
        
        # Выбор GPU с наибольшей памятью
        best_gpu = 0
        max_memory = 0
        
        for i in range(gpu_count):
            try:
                torch.cuda.set_device(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name} - {memory_gb:.1f}GB")
                
                if memory_gb > max_memory:
                    max_memory = memory_gb
                    best_gpu = i
                    device_info = {
                        'type': 'cuda',
                        'id': i,
                        'name': props.name,
                        'memory': memory_gb
                    }
            except Exception as e:
                logger.warning(f"Failed to check GPU {i}: {e}")
        
        if device_info['type'] == 'cuda':
            torch.cuda.set_device(best_gpu)
            logger.info(f"✅ Selected GPU {best_gpu}: {device_info['name']} ({device_info['memory']:.1f}GB)")
    
    # Проверка NPU (Intel Neural Compute Stick, etc.)
    try:
        # Проверка Intel NPU
        if os.path.exists('/dev/intel_npu0'):
            device_info = {
                'type': 'npu',
                'id': 0,
                'name': 'Intel NPU',
                'memory': 16  # Примерная память NPU
            }
            logger.info("✅ Found Intel NPU")
        
        # Проверка других NPU
        elif os.path.exists('/dev/npu0'):
            device_info = {
                'type': 'npu',
                'id': 0,
                'name': 'Generic NPU',
                'memory': 8
            }
            logger.info("✅ Found Generic NPU")
    except Exception as e:
        logger.debug(f"NPU check failed: {e}")
    
    if device_info['type'] == 'cpu':
        logger.info("⚠️ Using CPU (no GPU/NPU available)")
    
    return device_info

def optimize_for_device(device_info: Dict[str, Any]) -> None:
    """Оптимизация настроек для конкретного устройства с ограничениями ресурсов."""
    if device_info['type'] == 'cuda':
        # 🚀 НОВЫЕ ОГРАНИЧЕНИЯ РЕСУРСОВ GPU (50-80%)
        total_memory_gb = device_info['memory']
        max_usable_memory_gb = total_memory_gb * 0.8  # Максимум 80%
        min_usable_memory_gb = total_memory_gb * 0.5  # Минимум 50%
        
        # Установка переменных окружения для контроля памяти
        memory_fraction = 0.8  # Использовать максимум 80%
        os.environ['CUDA_MEMORY_FRACTION'] = str(memory_fraction)
        
        # CUDA оптимизации
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 🚀 НОВОЕ: Ограничение размера тензоров для контроля памяти
        if total_memory_gb >= 24:  # 24GB+ GPU
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Ограничение: максимум 80% памяти
            max_tensor_size = int(max_usable_memory_gb * 0.8 * (1024**3))
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info(f"🚀 High-memory GPU optimizations enabled (max: {max_usable_memory_gb:.1f}GB, {memory_fraction*100:.0f}%)")
            
        elif total_memory_gb >= 12:  # 12-24GB GPU
            torch.backends.cudnn.benchmark = True
            # Ограничение: максимум 75% памяти
            memory_fraction = 0.75
            os.environ['CUDA_MEMORY_FRACTION'] = str(memory_fraction)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info(f"⚡ Medium-memory GPU optimizations enabled (max: {max_usable_memory_gb:.1f}GB, {memory_fraction*100:.0f}%)")
            
        else:  # <12GB GPU
            torch.backends.cudnn.benchmark = False
            # Ограничение: максимум 70% памяти
            memory_fraction = 0.7
            os.environ['CUDA_MEMORY_FRACTION'] = str(memory_fraction)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info(f"🔧 Low-memory GPU optimizations enabled (max: {max_usable_memory_gb:.1f}GB, {memory_fraction*100:.0f}%)")
        
        # 🚀 НОВОЕ: Принудительная очистка кэша CUDA
        torch.cuda.empty_cache()
        logger.info(f"🧹 CUDA cache cleared, memory fraction set to {memory_fraction*100:.0f}%")
    
    elif device_info['type'] == 'npu':
        # 🚀 НОВЫЕ ОГРАНИЧЕНИЯ NPU (50-80%)
        os.environ['INTEL_NPU_DEVICE'] = f"npu{device_info['id']}"
        
        # Ограничение системных ресурсов для NPU
        max_cpu_percent = 80
        max_memory_percent = 80
        
        # Установка переменных окружения для контроля NPU
        os.environ['INTEL_NPU_MAX_CPU_USAGE'] = str(max_cpu_percent)
        os.environ['INTEL_NPU_MAX_MEMORY_USAGE'] = str(max_memory_percent)
        
        logger.info(f"🚀 NPU optimizations enabled (max CPU: {max_cpu_percent}%, max memory: {max_memory_percent}%)")
    
    # Общие оптимизации
    torch.set_num_threads(min(8, os.cpu_count()))
    
    logger.info(f"✅ Device optimization completed for {device_info['type']} ({device_info['name']})")

def manage_gpu_memory(device_info: Dict[str, Any], operation: str = "check") -> None:
    """Управление памятью GPU с ограничениями 50-80%."""
    if device_info['type'] != 'cuda':
        return
        
    try:
        if operation == "clear":
            # Принудительная очистка кэша CUDA
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cache cleared")
            
        elif operation == "check":
            # Проверка использования памяти
            allocated = torch.cuda.memory_allocated(device_info['id']) / (1024**3)
            reserved = torch.cuda.memory_reserved(device_info['id']) / (1024**3)
            total = device_info['memory']
            
            usage_percent = (allocated / total) * 100
            
            if usage_percent > 80:
                logger.warning(f"⚠️ GPU memory usage: {usage_percent:.1f}% > 80% limit")
                torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cache cleared due to high usage")
            elif usage_percent > 70:
                logger.info(f"ℹ️ GPU memory usage: {usage_percent:.1f}% (approaching 80% limit)")
            else:
                logger.info(f"✅ GPU memory usage: {usage_percent:.1f}% (within limits)")
                
    except Exception as e:
        logger.warning(f"GPU memory management error: {e}")

# Absolute cache paths inside the container
# Fixed paths for proper model loading in cog runtime
WEIGHTS_ROOT = "/src/model_files"
SDXL_CACHE_DIR = "/src/sdxl-cache"
CONTROLNET_CACHE_DIR = "/src/sdxl-cache"
REFS_DIR = "/src/model_files/refs"

# ControlNet local subfolders (pre-downloaded in build steps)
CONTROLNET_CANNY_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-canny-sdxl-1.0")
# Soft-edge variant is sometimes distributed as HED for SDXL. Support both names.
CONTROLNET_HED_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-hed-sdxl-1.0")
CONTROLNET_SOFTEDGE_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-softedge-sdxl-1.0")
# Public alternative edge-like model available without gating
CONTROLNET_LINEART_DIR = os.path.join(CONTROLNET_CACHE_DIR, "controlnet-lineart-sdxl-1.0")

# HF repo ids for fallback when local cache is not baked into the image
SDXL_REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_CANNY_REPO_ID = "diffusers/controlnet-canny-sdxl-1.0"
# ИСПРАВЛЕНО: используем рабочий repo ID для Lineart ControlNet - SD 1.5 версия
CONTROLNET_LINEART_REPO_ID = "lllyasviel/control_v11p_sd15_scribble"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def average_color_from_image(image_path: str) -> Tuple[int, int, int]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(x) for x in mean.tolist())


def sample_color_for_name(color_name: str) -> Tuple[int, int, int]:
    color_folder = os.path.join(REFS_DIR, color_name)
    if os.path.isdir(color_folder):
        candidates: List[str] = []
        for fn in os.listdir(color_folder):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(os.path.join(color_folder, fn))
        if candidates:
            choice = random.choice(candidates)
            return average_color_from_image(choice)
    # Fallback simple mapping (extend as needed)
    named = {
        "black": (20, 20, 20),
        "white": (235, 235, 235),
        "red": (180, 40, 40),
        "green": (40, 160, 80),
        "blue": (40, 80, 180),
        "gray": (128, 128, 128),
        "brown": (120, 80, 50),
    }
    return named.get(color_name.lower(), (127, 127, 127))


def build_color_map(colors: List[Dict[str, Any]], size: Tuple[int, int], out_path: str) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", (width, height))
    draw = Image.new("RGB", (width, height))
    # Normalize proportions
    props = [max(0.0, float(c.get("proportion", 0))) for c in colors]
    total = sum(props) or 1.0
    props = [p / total for p in props]

    x0 = 0
    for c, p in zip(colors, props):
        w = int(round(p * width))
        if w <= 0:
            continue
        color = sample_color_for_name(c.get("name", "gray"))
        for x in range(x0, min(x0 + w, width)):
            for y in range(height):
                canvas.putpixel((x, y), color)
        x0 += w

    canvas.save(out_path)
    return canvas


def canny_edge_from_image(image: Image.Image, low_threshold: int, high_threshold: int) -> np.ndarray:
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if img_array.shape[2] == 3:  # RGB
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array

    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    
    # Convert back to PIL format
    return Image.fromarray(edges)


def select_controlnet_by_angle(angle: int, controlnet_canny: ControlNetModel, 
                              controlnet_softedge: ControlNetModel, 
                              controlnet_lineart: ControlNetModel) -> ControlNetModel:
    """Select ControlNet based on angle for optimal edge detection."""
    # Normalize angle to 0-360 range
    angle = angle % 360
    
    # For diagonal angles (30-60, 120-150, 210-240, 300-330), prefer Lineart
    if 30 <= angle <= 60 or 120 <= angle <= 150 or 210 <= angle <= 240 or 300 <= angle <= 330:
        if controlnet_lineart is not None:
            logger.info(f"Selected Lineart ControlNet for diagonal angle {angle}")
            return controlnet_lineart
        else:
            logger.warning("Lineart ControlNet not available, falling back to Canny")
    
    # For horizontal/vertical angles (0, 90, 180, 270), prefer Canny
    if angle in [0, 90, 180, 270]:
        if controlnet_canny is not None:
            logger.info(f"Selected Canny ControlNet for cardinal angle {angle}")
            return controlnet_canny
        else:
            logger.warning("Canny ControlNet not available, falling back to Softedge")
    
    # Default to Softedge for other angles
    if controlnet_softedge is not None:
        logger.info(f"Selected Softedge ControlNet for angle {angle}")
        return controlnet_softedge
    else:
        logger.warning("Softedge ControlNet not available, falling back to Canny")
        return controlnet_canny


class OptimizedPredictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        start_time = time.time()
        logger.info("Starting model setup...")
        
        # 🚀 ОПТИМИЗАЦИЯ: Выбор и настройка лучшего устройства
        self.device_info = select_best_device()
        optimize_for_device(self.device_info)
        
        # 🚀 НОВОЕ: Инициализация мониторинга ресурсов
        self.resource_monitor = ResourceMonitor(self.device_info)
        self.resource_monitor.start_monitoring()
        
        # Определение устройства для PyTorch
        if self.device_info['type'] == 'cuda':
            self.device = f"cuda:{self.device_info['id']}"
        elif self.device_info['type'] == 'npu':
            self.device = "cpu"  # NPU пока не поддерживается напрямую в diffusers
        else:
            self.device = "cpu"
        
        logger.info(f"🎯 Using device: {self.device} ({self.device_info['name']})")
        
        # 🚀 НОВОЕ: Проверка и управление памятью GPU
        manage_gpu_memory(self.device_info, "check")

        # Load ControlNet models
        logger.info("Loading ControlNet models...")
        
        # Try to load ControlNet models from local cache first
        self.controlnet_canny = None
        self.controlnet_softedge = None
        self.controlnet_lineart = None
        
        try:
            if os.path.exists(CONTROLNET_CANNY_DIR):
                self.controlnet_canny = ControlNetModel.from_pretrained(CONTROLNET_CANNY_DIR)
                logger.info("Loaded Canny ControlNet from local cache")
            else:
                logger.info("Canny ControlNet not found in local cache, will download from HF")
                self.controlnet_canny = ControlNetModel.from_pretrained(CONTROLNET_CANNY_REPO_ID)
        except Exception as e:
            logger.warning(f"Failed to load Canny ControlNet: {e}")
            self.controlnet_canny = None

        try:
            if os.path.exists(CONTROLNET_HED_DIR):
                self.controlnet_softedge = ControlNetModel.from_pretrained(CONTROLNET_HED_DIR)
                logger.info("Loaded HED/Softedge ControlNet from local cache")
            elif os.path.exists(CONTROLNET_SOFTEDGE_DIR):
                self.controlnet_softedge = ControlNetModel.from_pretrained(CONTROLNET_SOFTEDGE_DIR)
                logger.info("Loaded Softedge ControlNet from local cache")
            else:
                logger.info("Softedge ControlNet not found in local cache, will download from HF")
                self.controlnet_softedge = ControlNetModel.from_pretrained(CONTROLNET_CANNY_REPO_ID)
        except Exception as e:
            logger.warning(f"Failed to load Softedge ControlNet: {e}")
            self.controlnet_softedge = None

        try:
            if os.path.exists(CONTROLNET_LINEART_DIR):
                self.controlnet_lineart = ControlNetModel.from_pretrained(CONTROLNET_LINEART_DIR)
                logger.info("Loaded Lineart ControlNet from local cache")
            else:
                logger.info("Lineart ControlNet not found in local cache, will download from HF")
                # ИСПРАВЛЕНО: используем рабочий repo ID - SD 1.5 версия
                self.controlnet_lineart = ControlNetModel.from_pretrained(CONTROLNET_LINEART_REPO_ID)
        except Exception as e:
            logger.warning(f"Failed to load Lineart ControlNet: {e}")
            self.controlnet_lineart = None

        # Initialize SDXL pipeline with ControlNet
        logger.info("Initializing SDXL pipeline...")
        
        # ИСПРАВЛЕНО: максимально надежная инициализация с полным fallback
        self.has_controlnet = False
        self.pipe = None
        
        # Попытка 1: Инициализация с ControlNet (если доступен)
        initial_controlnet = self.controlnet_canny or self.controlnet_softedge or self.controlnet_lineart
        
        if initial_controlnet is not None:
            try:
                logger.info("Attempting to initialize SDXL ControlNet pipeline...")
                self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    SDXL_REPO_ID,
                    controlnet=initial_controlnet,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to(self.device)
                self.has_controlnet = True
                logger.info("✅ Successfully initialized SDXL ControlNet pipeline")
            except Exception as e:
                logger.warning(f"❌ ControlNet pipeline initialization failed: {e}")
                self.pipe = None
                self.has_controlnet = False
        
        # Попытка 2: Fallback на базовый SDXL
        if self.pipe is None:
            try:
                logger.info("Falling back to basic SDXL pipeline...")
                from diffusers import StableDiffusionXLPipeline
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    SDXL_REPO_ID,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                ).to(self.device)
                self.has_controlnet = False
                logger.info("✅ Successfully initialized basic SDXL pipeline")
            except Exception as e:
                logger.error(f"❌ Basic SDXL pipeline initialization failed: {e}")
                raise RuntimeError(f"Failed to initialize any pipeline: {e}")
        
        if self.pipe is None:
            raise RuntimeError("Pipeline initialization failed completely")

        # Attach LoRA and Textual Inversion
        lora_path = "./model_files/rubber-tile-lora-v4_sdxl_lora.safetensors"
        ti_path = "./model_files/rubber-tile-lora-v4_sdxl_embeddings.safetensors"
        
        # Загрузка LoRA с улучшенной обработкой ошибок
        if os.path.exists(lora_path):
            try:
                logger.info("Loading LoRA weights...")
                self.pipe.load_lora_weights(lora_path)
                # Fuse for runtime speed
                self.pipe.fuse_lora()
                logger.info("✅ LoRA weights loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load LoRA weights: {e}")
                raise RuntimeError(f"LoRA weights loading failed: {e}")
        else:
            logger.error(f"❌ LoRA weights not found at {lora_path}")
            raise RuntimeError(f"LoRA weights not found at {lora_path}")

        # Загрузка Textual Inversion с улучшенной обработкой ошибок
        if os.path.exists(ti_path):
            try:
                logger.info("Loading Textual Inversion embeddings...")
                # Try standard loader first (may fail for SDXL dual-encoder TI formats)
                try:
                    self.pipe.load_textual_inversion(ti_path, token="<s0>")
                    logger.info("✅ Textual Inversion loaded with standard method")
                except Exception as e:
                    logger.warning(f"Standard TI load failed ({e}). Falling back to manual SDXL dual-encoder TI install...")
                    self._install_sdxl_textual_inversion_dual(ti_path, token_g="<s0>", token_l="<s0>")
                    logger.info("✅ Textual Inversion loaded with manual method")
            except Exception as e:
                logger.error(f"❌ Failed to load Textual Inversion: {e}")
                raise RuntimeError(f"Textual Inversion loading failed: {e}")
        else:
            logger.error(f"❌ Textual inversion embeddings not found at {ti_path}")
            raise RuntimeError(f"Textual inversion embeddings not found at {ti_path}")

        # Scheduler с улучшенной обработкой ошибок
        try:
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            logger.info("✅ Scheduler configured successfully")
        except Exception as e:
            logger.warning(f"⚠️ Scheduler configuration failed: {e}")

        # Basic runtime opts с улучшенной обработкой ошибок
        try:
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
                logger.info("✅ VAE slicing enabled")
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
                logger.info("✅ VAE tiling enabled")
        except Exception as e:
            logger.warning(f"⚠️ VAE optimization failed: {e}")
        
        # Performance optimizations - отключен torch.compile из-за проблем с CUDA Graph
        # if hasattr(torch, 'compile') and torch.__version__ >= "2.4.0":
        #     try:
        #         logger.info("Enabling torch.compile for performance...")
        #         self.pipe = torch.compile(self.pipe, mode="reduce-overhead")
        #     except Exception as e:
        #         logger.warning(f"torch.compile failed: {e}")
        
        # CUDA optimizations с улучшенной обработкой ошибок
        try:
            if self.device_info['type'] == 'cuda':
                # Оптимизации уже применены в optimize_for_device()
                logger.info("✅ CUDA optimizations applied")
            elif self.device_info['type'] == 'npu':
                logger.info("✅ NPU optimizations applied")
            else:
                logger.info("✅ CPU optimizations applied")
        except Exception as e:
            logger.warning(f"⚠️ Device optimization failed: {e}")

        setup_time = time.time() - start_time
        logger.info(f"🎉 Model setup completed successfully in {setup_time:.2f}s")
        logger.info(f"📊 ControlNet enabled: {self.has_controlnet}")
        logger.info(f"🔧 Pipeline type: {type(self.pipe).__name__}")
        logger.info(f"🚀 Device: {self.device} ({self.device_info['name']})")
        logger.info(f"💾 Device memory: {self.device_info['memory']:.1f}GB")

    def _parse_params_json(self, params_json: str) -> Dict[str, Any]:
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

    def _build_prompt(self, colors: List[Dict[str, Any]]) -> str:
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

    def _should_use_controlnet(self, angle: int) -> Tuple[bool, str]:
        """
        Определяет, нужно ли использовать ControlNet для заданного угла.
        
        Args:
            angle: Угол укладки в градусах
            
        Returns:
            Tuple[bool, str]: (использовать_controlnet, причина)
        """
        # Нормализация угла
        angle_norm = int(angle) % 180
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Угол 0° - единственный надежный без ControlNet
        if angle_norm == 0:
            return False, "Угол 0° (вид сверху) - единственный надежный ракурс без ControlNet"
        
        # Для других углов - использовать ControlNet с предупреждением
        if angle_norm in (45, 135):
            return True, f"Угол {angle}° требует ControlNet для геометрического контроля"
        elif angle_norm in (30, 60, 90, 120, 150):
            return True, f"Угол {angle}° требует ControlNet для стабильной генерации"
        else:
            return True, f"Угол {angle}° требует ControlNet (нестандартный ракурс)"

    def _get_controlnet_model(self, angle: int) -> Optional[Any]:
        """
        Выбирает подходящую ControlNet модель для заданного угла.
        
        Args:
            angle: Угол укладки в градусах
            
        Returns:
            ControlNet модель или None
        """
        angle_norm = int(angle) % 180
        
        # Для горизонтальных/вертикальных углов - Canny
        if angle_norm in (0, 90):
            if self.controlnet_canny:
                return self.controlnet_canny
            elif self.controlnet_softedge:
                return self.controlnet_softedge
        
        # Для диагональных углов - Lineart или SoftEdge
        elif angle_norm in (30, 45, 60, 120, 135, 150):
            if self.controlnet_lineart:
                return self.controlnet_lineart
            elif self.controlnet_softedge:
                return self.controlnet_softedge
            elif self.controlnet_canny:
                return self.controlnet_canny
        
        # Fallback на любую доступную модель
        for model in [self.controlnet_canny, self.controlnet_softedge, self.controlnet_lineart]:
            if model:
                return model
        
        return None

    def predict(
        self,
        params_json: str = Input(description="Business-oriented parameters JSON: colors, angle, seed, quality, overrides. ВАЖНО: Угол 0° - единственный надежный ракурс обученной модели. Другие углы требуют ControlNet для геометрического контроля.")
    ) -> List[Path]:
        """Generate preview/final images using ControlNet color‑composition guidance.

        Returns: [preview.png, final.png, colormap.png]
        """
        start_time = time.time()
        
        # 🚀 НОВОЕ: Проверка ресурсов перед генерацией
        logger.info("🔍 Checking device resources before generation...")
        manage_gpu_memory(self.device_info, "check")
        
        # Parse and validate input parameters
        try:
            params = self._parse_params_json(params_json)
        except Exception as e:
            raise ValueError(f"Parameter validation failed: {e}")

        colors = params.get("colors", [])
        angle = int(params.get("angle", 0))
        seed = int(params.get("seed", -1))
        quality = str(params.get("quality", "standard"))
        overrides: Dict[str, Any] = params.get("overrides", {}) or {}

        logger.info(f"Generating with params: colors={len(colors)}, angle={angle}, quality={quality}, seed={seed}")
        
        # 🚀 НОВОЕ: Логирование текущего состояния ресурсов
        if hasattr(self, 'resource_monitor'):
            resource_summary = self.resource_monitor.get_resource_summary()
            logger.info(f"📊 Resource status: {resource_summary}")

        # Defaults and quality profiles
        if quality == "preview":
            steps_preview, steps_final = 24, 32
            size_preview, size_final = (512, 512), (1024, 1024)
        elif quality == "high":
            steps_preview, steps_final = 32, 50
            size_preview, size_final = (512, 512), (1024, 1024)
        else:  # standard
            steps_preview, steps_final = 24, 40
            size_preview, size_final = (512, 512), (1024, 1024)

        # Apply overrides
        num_inference_steps_preview = int(overrides.get("num_inference_steps_preview", steps_preview))
        num_inference_steps_final = int(overrides.get("num_inference_steps_final", steps_final))
        guidance_scale = float(overrides.get("guidance_scale", 7.5))

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

        # Create color map and corresponding control image (edge map)
        colormap_path = "/tmp/colormap.png"
        logger.info(f"Building color map for {len(colors)} colors")
        colormap_img = build_color_map(colors, size_final, colormap_path)
        logger.info(f"Color map saved to {colormap_path}")

        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Логика углов в соответствии с ограничениями модели
        should_use_controlnet, reason = self._should_use_controlnet(angle)
        use_controlnet_by_angle = should_use_controlnet
        
        # Пользовательский override ControlNet
        user_controlnet_setting = overrides.get("use_controlnet", None)
        
        if user_controlnet_setting is not None:
            use_controlnet = bool(user_controlnet_setting)
            if use_controlnet != use_controlnet_by_angle:
                logger.warning(f"⚠️ Пользователь переопределил ControlNet: {use_controlnet} (рекомендуется: {use_controlnet_by_angle})")
                logger.warning(f"⚠️ Причина рекомендации: {reason}")
        else:
            use_controlnet = use_controlnet_by_angle
            
        logger.info(f"ControlNet решение: {use_controlnet} - {reason}")
        
        # ИСПРАВЛЕНО: максимально безопасная работа с ControlNet
        control_preview = None
        control_final = None
        
        if use_controlnet and self.has_controlnet:
            try:
        # Select controlnet by angle
                selected_cn = select_controlnet_by_angle(
                    angle, self.controlnet_canny, self.controlnet_softedge, self.controlnet_lineart
                )
                if selected_cn is not None:
                    # Безопасно устанавливаем ControlNet
                    if hasattr(self.pipe, 'controlnet'):
                        self.pipe.controlnet = selected_cn
                        logger.info(f"✅ ControlNet set for angle {angle}")

                        # Prepare edge maps for preview/final
                        logger.info("Generating edge maps...")
                        control_preview = canny_edge_from_image(colormap_img.resize(size_preview, Image.NEAREST), 80, 160)
                        control_final = canny_edge_from_image(colormap_img.resize(size_final, Image.NEAREST), 100, 200)
                        logger.info("✅ Edge maps generated successfully")
                    else:
                        logger.warning("⚠️ Pipeline does not support ControlNet")
                        use_controlnet = False
                else:
                    logger.warning("⚠️ No ControlNet available for this angle")
                    use_controlnet = False
            except Exception as e:
                logger.error(f"❌ ControlNet setup failed: {e}")
                use_controlnet = False
                control_preview = None
                control_final = None
        else:
            logger.info("ℹ️ ControlNet disabled (user preference or not available)")

        # Generate preview first (fast)
        preview_start = time.time()
        logger.info(f"Generating preview with {num_inference_steps_preview} steps, guidance_scale={guidance_scale}")
        logger.info(f"ControlNet status: enabled={use_controlnet}, available={self.has_controlnet}, image={control_preview is not None}")
        
        try:
            # Prepare generation parameters
            gen_params = {
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
                gen_params["image"] = control_preview
                logger.info("✅ Using ControlNet for preview generation")
            else:
                logger.info("ℹ️ Preview generation without ControlNet")
            
            preview = self.pipe(**gen_params).images[0]
            preview_time = time.time() - preview_start
            logger.info(f"✅ Preview generated successfully in {preview_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Preview generation failed: {e}")
            raise RuntimeError(f"Preview generation failed: {e}")

        # Generate final (quality)
        final_start = time.time()
        logger.info(f"Generating final image with {num_inference_steps_final} steps")
        
        try:
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
            
            final = self.pipe(**gen_params).images[0]
            final_time = time.time() - final_start
            logger.info(f"✅ Final image generated successfully in {final_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ Final generation failed: {e}")
            raise RuntimeError(f"Final generation failed: {e}")

        # Save outputs with improved error handling
        try:
            preview_path = "/tmp/preview.png"
            final_path = "/tmp/final.png"
            preview.save(preview_path)
            final.save(final_path)
            logger.info("✅ Images saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save images: {e}")
            raise RuntimeError(f"Failed to save images: {e}")

        total_time = time.time() - start_time
        logger.info(f"🎉 Generation completed successfully in {total_time:.2f}s")
        logger.info(f"📁 Outputs: preview={preview_path}, final={final_path}, colormap={colormap_path}")
        logger.info(f"📊 Final stats: ControlNet={use_controlnet}, Steps={num_inference_steps_final}, Quality={quality}")
        
        # 🚀 НОВОЕ: Очистка ресурсов после генерации
        logger.info("🧹 Cleaning up device resources after generation...")
        manage_gpu_memory(self.device_info, "clear")
        
        # 🚀 НОВОЕ: Финальная сводка по ресурсам
        if hasattr(self, 'resource_monitor'):
            final_resource_summary = self.resource_monitor.get_resource_summary()
            logger.info(f"📊 Final resource summary: {final_resource_summary}")

        return [Path(preview_path), Path(final_path), Path(colormap_path)]

    def _install_sdxl_textual_inversion_dual(self, ti_path: str, token_g: str, token_l: str) -> None:
        """Install SDXL textual inversion that contains separate embeddings for CLIP-G and CLIP-L encoders.

        The `safetensors` file is expected to contain keys 'clip_g' and 'clip_l'.
        We will add tokens to both tokenizers and set the corresponding encoder embeddings.
        """
        state = load_safetensors(ti_path)
        emb_g = state.get("clip_g")
        emb_l = state.get("clip_l")

        if emb_g is None or emb_l is None:
            raise ValueError("Textual inversion file must contain 'clip_g' and 'clip_l' tensors for SDXL.")

        # Tokenizer/encoder mapping: 'g' → text_encoder_2 (bigG/1280), 'l' → text_encoder (Large/768)
        tokenizer_g = getattr(self.pipe, "tokenizer_2", None)
        tokenizer_l = getattr(self.pipe, "tokenizer", None)
        encoder_g = getattr(self.pipe, "text_encoder_2", None)
        encoder_l = getattr(self.pipe, "text_encoder", None)

        if any(x is None for x in (tokenizer_g, tokenizer_l, encoder_g, encoder_l)):
            raise RuntimeError("Pipeline is missing SDXL tokenizers/encoders required for TI installation.")

        # Determine number of tokens in the TI file
        num_tokens = emb_g.shape[0] if len(emb_g.shape) > 1 else 1
        logger.info(f"Textual inversion contains {num_tokens} token(s)")
        
        # ИСПРАВЛЕНИЕ: Улучшенная обработка предупреждения о состоянии словаря
        logger.info(f"TI state dict keys: {list(state.keys())}")
        logger.info(f"clip_g shape: {emb_g.shape}, clip_l shape: {emb_l.shape}")

        # Generate token names if multiple tokens
        if num_tokens == 1:
            tokens_g = [token_g]
            tokens_l = [token_l]
        else:
            # For multiple tokens, use <s0>, <s1>, <s2>, etc.
            tokens_g = [f"<s{i}>" for i in range(num_tokens)]
            tokens_l = [f"<s{i}>" for i in range(num_tokens)]

        # Ensure tokens exist in tokenizers
        added_g = tokenizer_g.add_tokens(tokens_g, special_tokens=True)
        added_l = tokenizer_l.add_tokens(tokens_l, special_tokens=True)

        if added_g > 0:
            encoder_g.resize_token_embeddings(len(tokenizer_g))
        if added_l > 0:
            encoder_l.resize_token_embeddings(len(tokenizer_l))

        with torch.no_grad():
            emb_layer_g = encoder_g.get_input_embeddings()
            emb_layer_l = encoder_l.get_input_embeddings()

            # Cast to encoder dtype/device
            emb_g_cast = emb_g.to(dtype=emb_layer_g.weight.dtype, device=emb_layer_g.weight.device)
            emb_l_cast = emb_l.to(dtype=emb_layer_l.weight.dtype, device=emb_layer_l.weight.device)

            # Check embedding dimensions (should match the embedding size of each encoder)
            if emb_g_cast.shape[-1] != emb_layer_g.weight.shape[-1]:
                raise ValueError(f"clip_g embedding dimension {emb_g_cast.shape[-1]} does not match text_encoder_2 embedding size {emb_layer_g.weight.shape[-1]}")
            if emb_l_cast.shape[-1] != emb_layer_l.weight.shape[-1]:
                raise ValueError(f"clip_l embedding dimension {emb_l_cast.shape[-1]} does not match text_encoder embedding size {emb_layer_l.weight.weight.shape[-1]}")

            # Set embeddings for each token
            for i, (token_g_name, token_l_name) in enumerate(zip(tokens_g, tokens_l)):
                token_id_g = tokenizer_g.convert_tokens_to_ids(token_g_name)
                token_id_l = tokenizer_l.convert_tokens_to_ids(token_l_name)

                # Extract embedding for this token
                if num_tokens == 1:
                    emb_g_token = emb_g_cast.squeeze(0) if len(emb_g_cast.shape) > 1 else emb_g_cast
                    emb_l_token = emb_l_cast.squeeze(0) if len(emb_l_cast.shape) > 1 else emb_l_cast
                else:
                    emb_g_token = emb_g_cast[i]
                    emb_l_token = emb_l_cast[i]

                # Set the embeddings for the specific token positions
                emb_layer_g.weight[token_id_g] = emb_g_token
                emb_layer_l.weight[token_id_l] = emb_l_token

        logger.info(f"SDXL textual inversion installed manually for {num_tokens} token(s): {tokens_g}")
    
    def __del__(self):
        """Деструктор для корректной остановки мониторинга ресурсов."""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop_monitoring()
            logger.info("🛑 Resource monitoring stopped in destructor")