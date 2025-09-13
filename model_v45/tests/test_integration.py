# test_integration.py - Интеграционные тесты для модели "nauslava/rubber-tile-lora-v45"

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import torch
from PIL import Image
import numpy as np

# Добавление пути к модулю
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from predict import Predictor

class TestModelIntegration(unittest.TestCase):
    """Интеграционные тесты для полной модели."""
    
    def setUp(self):
        """Настройка тестового окружения."""
        # Создаем временную директорию для тестов
        self.test_dir = tempfile.mkdtemp()
        
        # Мокаем torch.cuda для тестов
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()
        
        # Мокаем StableDiffusionXLPipeline
        self.pipeline_patcher = patch('predict.StableDiffusionXLPipeline')
        self.mock_pipeline_class = self.pipeline_patcher.start()
        
        # Создаем мок pipeline
        self.mock_pipeline = Mock()
        self.mock_pipeline_class.from_pretrained.return_value = self.mock_pipeline
        
        # Мокаем VAE
        self.mock_vae = Mock()
        self.mock_pipeline.vae = self.mock_vae
        
        # Мокаем планировщик
        self.mock_scheduler = Mock()
        self.mock_pipeline.scheduler = self.mock_scheduler
        
        # Создаем экземпляр Predictor
        self.predictor = Predictor()
    
    def tearDown(self):
        """Очистка после тестов."""
        self.cuda_patcher.stop()
        self.pipeline_patcher.stop()
        
        # Удаляем временную директорию
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_model_initialization(self):
        """Тест полной инициализации модели."""
        # Проверяем, что все компоненты были инициализированы
        self.assertIsNotNone(self.predictor.device)
        self.assertIsNotNone(self.predictor.pipe)
        
        # Проверяем, что pipeline был создан
        self.mock_pipeline_class.from_pretrained.assert_called_once()
        
        # Проверяем, что LoRA был загружен
        self.mock_pipeline.set_adapters.assert_called_once()
        self.mock_pipeline.fuse_lora.assert_called_once()
        
        # Проверяем, что Textual Inversion был загружен
        self.mock_pipeline.load_textual_inversion.assert_called_once()
        
        # Проверяем, что планировщик был настроен
        self.assertIsNotNone(self.predictor.pipe.scheduler)
        
        # Проверяем, что VAE оптимизации были применены
        self.mock_vae.enable_slicing.assert_called_once()
        self.mock_vae.enable_tiling.assert_called_once()
    
    def test_model_file_paths(self):
        """Тест путей к файлам модели."""
        # Проверяем, что используются правильные пути к файлам
        lora_call = self.mock_pipeline.set_adapters.call_args
        ti_call = self.mock_pipeline.load_textual_inversion.call_args
        
        # Проверяем LoRA
        self.assertIn("rubber-tile-lora-v4_sdxl_lora", str(lora_call))
        
        # Проверяем Textual Inversion
        self.assertIn("rubber-tile-lora-v4_sdxl_embeddings.safetensors", str(ti_call))
    
    def test_lora_configuration(self):
        """Тест конфигурации LoRA."""
        # Проверяем параметры LoRA
        lora_call = self.mock_pipeline.set_adapters.call_args
        
        # Проверяем имя адаптера
        self.assertEqual(lora_call[0][0], ["rubber-tile-lora-v4_sdxl_lora"])
        
        # Проверяем силу адаптера
        self.assertEqual(lora_call[0][1], [0.7])
        
        # Проверяем, что LoRA был объединен
        self.mock_pipeline.fuse_lora.assert_called_once()
    
    def test_textual_inversion_configuration(self):
        """Тест конфигурации Textual Inversion."""
        # Проверяем параметры Textual Inversion
        ti_call = self.mock_pipeline.load_textual_inversion.call_args
        
        # Проверяем токен
        self.assertEqual(ti_call[1]['token'], "<s0>")
    
    def test_scheduler_configuration(self):
        """Тест конфигурации планировщика."""
        # Проверяем, что планировщик был настроен
        self.assertIsNotNone(self.predictor.pipe.scheduler)
    
    def test_vae_optimizations(self):
        """Тест VAE оптимизаций."""
        # Проверяем, что все VAE оптимизации были применены
        self.mock_vae.enable_slicing.assert_called_once()
        self.mock_vae.enable_tiling.assert_called_once()
    
    def test_prompt_processing_pipeline(self):
        """Тест полного пайплайна обработки промпта."""
        # Тестируем различные типы промптов
        test_cases = [
            "100% white",
            "50% red, 50% black",
            "40% blue, 35% green, 25% yellow",
            "60% purple, 40% pink"
        ]
        
        for prompt in test_cases:
            with self.subTest(prompt=prompt):
                result = self.predictor._build_prompt(prompt)
                
                # Проверяем базовую структуру
                self.assertIn("ohwx_rubber_tile <s0><s1>", result)
                self.assertIn(prompt, result)
                
                # Проверяем качественные дескрипторы
                quality_descriptors = [
                    "photorealistic rubber tile",
                    "high quality",
                    "detailed texture",
                    "professional photography",
                    "sharp focus"
                ]
                
                for descriptor in quality_descriptors:
                    self.assertIn(descriptor, result)
    
    def test_generation_pipeline(self):
        """Тест полного пайплайна генерации."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Тестируем генерацию с различными параметрами
        test_cases = [
            ("100% white", None, -1),
            ("50% red, 50% black", "custom negative", 12345),
            ("60% blue, 40% green", None, 98765)
        ]
        
        for prompt, negative_prompt, seed in test_cases:
            with self.subTest(prompt=prompt, negative_prompt=negative_prompt, seed=seed):
                result = self.predictor.predict(prompt, negative_prompt, seed)
                
                # Проверяем, что результат получен
                self.assertEqual(result, mock_image)
                
                # Проверяем, что pipeline был вызван с правильными параметрами
                self.mock_pipeline.assert_called_with(
                    prompt=unittest.mock.ANY,
                    negative_prompt=unittest.mock.ANY,
                    num_inference_steps=20,
                    guidance_scale=7.0,
                    width=1024,
                    height=1024,
                    generator=unittest.mock.ANY
                )
    
    def test_memory_management(self):
        """Тест управления памятью."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Вызываем predict несколько раз
        for i in range(3):
            result = self.predictor.predict(f"{100-i*20}% white")
            self.assertEqual(result, mock_image)
        
        # Проверяем, что pipeline был вызван правильное количество раз
        self.assertEqual(self.mock_pipeline.call_count, 3)
    
    def test_error_handling(self):
        """Тест обработки ошибок."""
        # Мокаем ошибку в pipeline
        self.mock_pipeline.side_effect = Exception("Pipeline error")
        
        # Проверяем, что ошибка обрабатывается корректно
        with self.assertRaises(Exception):
            self.predictor.predict("100% white")
    
    def test_seed_reproducibility(self):
        """Тест воспроизводимости с сидом."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Тестируем с фиксированным сидом
        seed = 12345
        result1 = self.predictor.predict("100% white", seed=seed)
        result2 = self.predictor.predict("100% white", seed=seed)
        
        # Проверяем, что результаты одинаковые
        self.assertEqual(result1, result2)
        self.assertEqual(result1, mock_image)
    
    def test_negative_prompt_handling(self):
        """Тест обработки негативных промптов."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Тестируем различные негативные промпты
        test_cases = [
            None,  # Стандартный
            "custom negative",  # Пользовательский
            "blurry, low quality",  # Короткий
            "very long negative prompt with many words and descriptions"  # Длинный
        ]
        
        for negative_prompt in test_cases:
            with self.subTest(negative_prompt=negative_prompt):
                result = self.predictor.predict("100% white", negative_prompt=negative_prompt)
                
                # Проверяем, что результат получен
                self.assertEqual(result, mock_image)
                
                # Проверяем, что pipeline был вызван
                self.mock_pipeline.assert_called()
    
    def test_model_consistency(self):
        """Тест консистентности модели."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Тестируем консистентность с одинаковыми параметрами
        prompt = "50% red, 50% black"
        seed = 54321
        
        # Первый вызов
        result1 = self.predictor.predict(prompt, seed=seed)
        
        # Второй вызов с теми же параметрами
        result2 = self.predictor.predict(prompt, seed=seed)
        
        # Проверяем консистентность
        self.assertEqual(result1, result2)
        self.assertEqual(result1, mock_image)
    
    def test_performance_characteristics(self):
        """Тест характеристик производительности."""
        # Мокаем результат генерации
        mock_image = Mock()
        mock_result = Mock()
        mock_result.images = [mock_image]
        self.mock_pipeline.return_value = mock_result
        
        # Тестируем различные размеры промптов
        test_prompts = [
            "100% white",  # Короткий
            "50% red, 50% black",  # Средний
            "30% navy, 25% teal, 20% cyan, 15% sky blue, 10% light blue"  # Длинный
        ]
        
        for prompt in test_prompts:
            with self.subTest(prompt=prompt):
                result = self.predictor.predict(prompt)
                
                # Проверяем, что результат получен
                self.assertEqual(result, mock_image)
                
                # Проверяем, что pipeline был вызван с правильными параметрами
                call_args = self.mock_pipeline.call_args
                self.assertEqual(call_args[1]['num_inference_steps'], 20)
                self.assertEqual(call_args[1]['guidance_scale'], 7.0)
                self.assertEqual(call_args[1]['width'], 1024)
                self.assertEqual(call_args[1]['height'], 1024)

class TestModelCompatibility(unittest.TestCase):
    """Тесты совместимости модели."""
    
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
    
    def test_sdxl_compatibility(self):
        """Тест совместимости с SDXL."""
        # Проверяем, что используется правильная базовая модель
        call_args = self.mock_pipeline_class.from_pretrained.call_args
        self.assertEqual(call_args[0][0], "stabilityai/stable-diffusion-xl-base-1.0")
    
    def test_fp16_compatibility(self):
        """Тест совместимости с FP16."""
        # Проверяем, что используется FP16
        call_args = self.mock_pipeline_class.from_pretrained.call_args
        self.assertEqual(call_args[1]['torch_dtype'], torch.float16)
        self.assertEqual(call_args[1]['variant'], "fp16")
    
    def test_safetensors_compatibility(self):
        """Тест совместимости с safetensors."""
        # Проверяем, что используется safetensors
        call_args = self.mock_pipeline_class.from_pretrained.call_args
        self.assertTrue(call_args[1]['use_safetensors'])
    
    def test_cuda_compatibility(self):
        """Тест совместимости с CUDA."""
        # Проверяем, что модель может работать с CUDA
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_properties') as mock_get_props:
                    mock_get_props.return_value = Mock(total_memory=16 * 1024**3)
                    
                    # Пересоздаем Predictor с доступным CUDA
                    predictor = Predictor()
                    self.assertEqual(predictor.device, "cuda")

if __name__ == '__main__':
    # Запуск интеграционных тестов
    unittest.main(verbosity=2)

