# test_predict.py - Тесты для модели "nauslava/rubber-tile-lora-v45"

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import numpy as np

# Добавление пути к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predict import Predictor

class TestPredictor(unittest.TestCase):
    """Тесты для класса Predictor."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Мокаем torch.cuda для тестов
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()
        
        # Мокаем StableDiffusionXLPipeline
        self.pipeline_patcher = patch('predict.StableDiffusionXLPipeline')
        self.mock_pipeline_class = self.pipeline_patcher.start()
        
        # Создаем мок pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline
        
        # Создаем экземпляр Predictor
        self.predictor = Predictor()
    
    def tearDown(self):
        """Очистка после тестов."""
        self.cuda_patcher.stop()
        self.pipeline_patcher.stop()
    
    def test_device_selection_cpu(self):
        """Тест выбора CPU устройства."""
        # Проверяем, что при недоступности CUDA используется CPU
        self.assertEqual(self.predictor.device, "cpu")
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_properties')
    def test_device_selection_gpu(self, mock_get_props, mock_device_count, mock_cuda_available):
        """Тест выбора GPU устройства."""
        # Мокаем свойства GPU
        mock_get_props.side_effect = [
            Mock(total_memory=8 * 1024**3),  # 8 GB
            Mock(total_memory=16 * 1024**3)   # 16 GB
        ]
        
        # Пересоздаем Predictor с доступным CUDA
        predictor = Predictor()
        self.assertEqual(predictor.device, "cuda")
    
    def test_pipeline_initialization(self):
        """Тест инициализации pipeline."""
        # Проверяем, что pipeline был создан с правильными параметрами
        self.mock_pipeline_class.from_pretrained.assert_called_once_with(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            resume_download=False
        )
    
    def test_lora_loading(self):
        """Тест загрузки LoRA адаптеров."""
        # Проверяем, что LoRA был загружен
        self.mock_pipeline.set_adapters.assert_called_once_with(
            ["rubber-tile-lora-v4_sdxl_lora"], 
            adapter_weights=[0.7]
        )
        self.mock_pipeline.fuse_lora.assert_called_once()
    
    def test_textual_inversion_loading(self):
        """Тест загрузки Textual Inversion."""
        # Проверяем, что Textual Inversion был загружен
        self.mock_pipeline.load_textual_inversion.assert_called_once()
    
    def test_scheduler_configuration(self):
        """Тест настройки планировщика."""
        # Проверяем, что планировщик был настроен
        self.assertIsNotNone(self.predictor.pipe.scheduler)
    
    def test_vae_optimizations(self):
        """Тест VAE оптимизаций."""
        # Проверяем, что VAE оптимизации были применены
        self.mock_pipeline.vae.enable_slicing.assert_called_once()
        self.mock_pipeline.vae.enable_tiling.assert_called_once()
    
    def test_build_prompt_simple(self):
        """Тест построения простого промпта."""
        color_desc = "100% white"
        result = self.predictor._build_prompt(color_desc)
        
        # Проверяем базовую структуру
        self.assertIn("ohwx_rubber_tile <s0><s1>", result)
        self.assertIn("100% white", result)
        self.assertIn("photorealistic rubber tile", result)
        self.assertIn("high quality", result)
        self.assertIn("detailed texture", result)
    
    def test_build_prompt_complex(self):
        """Тест построения сложного промпта."""
        color_desc = "50% red, 30% black, 20% pearl"
        result = self.predictor._build_prompt(color_desc)
        
        # Проверяем все компоненты
        self.assertIn("ohwx_rubber_tile <s0><s1>", result)
        self.assertIn("50% red, 30% black, 20% pearl", result)
        self.assertIn("photorealistic rubber tile", result)
        self.assertIn("professional photography", result)
        self.assertIn("sharp focus", result)
    
    def test_build_prompt_quality_descriptors(self):
        """Тест качественных дескрипторов в промпте."""
        color_desc = "100% blue"
        result = self.predictor._build_prompt(color_desc)
        
        # Проверяем наличие всех качественных дескрипторов
        expected_descriptors = [
            "photorealistic rubber tile",
            "high quality",
            "detailed texture",
            "professional photography",
            "sharp focus"
        ]
        
        for descriptor in expected_descriptors:
            self.assertIn(descriptor, result)
    
    @patch('torch.manual_seed')
    @patch('torch.Generator')
    def test_predict_with_seed(self, mock_generator_class, mock_manual_seed):
        """Тест предсказания с установкой сида."""
        # Мокаем генератор
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict с фиксированным сидом
        seed = 12345
        result = self.predictor.predict("100% white", seed=seed)
        
        # Проверяем, что сид был установлен
        mock_manual_seed.assert_called_with(seed)
        mock_generator.manual_seed.assert_called_with(seed)
    
    def test_predict_default_negative_prompt(self):
        """Тест стандартного негативного промпта."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict без негативного промпта
        result = self.predictor.predict("100% white")
        
        # Проверяем, что был использован стандартный негативный промпт
        expected_negative = "blurry, worst quality, low quality, deformed, watermark, 3d render, cartoon, abstract, painting, drawing, text, sketch, low resolution"
        self.mock_pipeline.assert_called_with(
            prompt=unittest.mock.ANY,
            negative_prompt=expected_negative,
            num_inference_steps=20,
            guidance_scale=7.0,
            width=1024,
            height=1024,
            generator=unittest.mock.ANY
        )
    
    def test_predict_custom_negative_prompt(self):
        """Тест пользовательского негативного промпта."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict с пользовательским негативным промптом
        custom_negative = "custom negative prompt"
        result = self.predictor.predict("100% white", negative_prompt=custom_negative)
        
        # Проверяем, что был использован пользовательский негативный промпт
        self.mock_pipeline.assert_called_with(
            prompt=unittest.mock.ANY,
            negative_prompt=custom_negative,
            num_inference_steps=20,
            guidance_scale=7.0,
            width=1024,
            height=1024,
            generator=unittest.mock.ANY
        )
    
    def test_predict_generation_parameters(self):
        """Тест параметров генерации."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict
        result = self.predictor.predict("100% white")
        
        # Проверяем параметры генерации
        call_args = self.mock_pipeline.call_args
        self.assertEqual(call_args[1]['num_inference_steps'], 20)
        self.assertEqual(call_args[1]['guidance_scale'], 7.0)
        self.assertEqual(call_args[1]['width'], 1024)
        self.assertEqual(call_args[1]['height'], 1024)
    
    def test_predict_return_value(self):
        """Тест возвращаемого значения."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict
        result = self.predictor.predict("100% white")
        
        # Проверяем, что возвращается правильное изображение
        self.assertEqual(result, mock_image)
    
    @patch('gc.collect')
    def test_memory_cleanup(self, mock_gc_collect):
        """Тест очистки памяти."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict
        result = self.predictor.predict("100% white")
        
        # Проверяем, что была вызвана очистка памяти
        mock_gc_collect.assert_called_once()

class TestPromptBuilding(unittest.TestCase):
    """Тесты для построения промптов."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        self.predictor = Predictor()
    
    def test_prompt_structure(self):
        """Тест структуры промпта."""
        color_desc = "60% blue, 40% green"
        result = self.predictor._build_prompt(color_desc)
        
        # Проверяем порядок компонентов
        parts = result.split(", ")
        self.assertEqual(parts[0], "ohwx_rubber_tile <s0><s1>")
        self.assertEqual(parts[1], "60% blue, 40% green")
        self.assertIn("photorealistic rubber tile", parts[2:])
    
    def test_prompt_quality_descriptors_order(self):
        """Тест порядка качественных дескрипторов."""
        color_desc = "100% red"
        result = self.predictor._build_prompt(color_desc)
        
        # Проверяем, что качественные дескрипторы идут в правильном порядке
        expected_order = [
            "photorealistic rubber tile",
            "high quality",
            "detailed texture",
            "professional photography",
            "sharp focus"
        ]
        
        # Находим позиции дескрипторов
        positions = []
        for descriptor in expected_order:
            pos = result.find(descriptor)
            self.assertNotEqual(pos, -1, f"Дескриптор '{descriptor}' не найден")
            positions.append(pos)
        
        # Проверяем, что они идут в правильном порядке
        for i in range(1, len(positions)):
            self.assertGreater(positions[i], positions[i-1], 
                             f"Дескриптор '{expected_order[i]}' идет раньше '{expected_order[i-1]}'")

if __name__ == '__main__':
    # Запуск тестов
    unittest.main(verbosity=2)
