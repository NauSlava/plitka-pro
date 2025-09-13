#!/usr/bin/env python3
"""
Модуль для управления версиями и хешами проекта Plitka Pro
"""

import json
import os
import subprocess
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime


class VersionManager:
    """Менеджер версий для проекта Plitka Pro"""
    
    def __init__(self, config_path: str = "version_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Загружает конфигурацию версий из файла"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"⚠️ Ошибка загрузки конфигурации: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Возвращает конфигурацию по умолчанию"""
        return {
            "version_mappings": {},
            "default_version": "v4.5.07",
            "fallback_strategy": {
                "primary": "replicate_api",
                "secondary": "version_config",
                "tertiary": "docker_images"
            },
            "replicate_model": "nauslava/plitka-pro-project"
        }
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """Получает информацию о версии"""
        return self.config.get("version_mappings", {}).get(version)
    
    def get_replicate_hash(self, version: str) -> Optional[str]:
        """Получает хеш Replicate для версии"""
        version_info = self.get_version_info(version)
        return version_info.get("replicate_hash") if version_info else None
    
    def get_full_hash(self, version: str) -> Optional[str]:
        """Получает полный хеш для версии"""
        version_info = self.get_version_info(version)
        return version_info.get("full_hash") if version_info else None
    
    def get_default_version(self) -> str:
        """Получает версию по умолчанию"""
        return self.config.get("default_version", "v4.5.07")
    
    def get_available_versions(self) -> list:
        """Получает список доступных версий"""
        return list(self.config.get("version_mappings", {}).keys())
    
    def get_docker_image_hash(self, version: str) -> Optional[str]:
        """Получает хеш Docker образа для указанной версии"""
        try:
            result = subprocess.run(
                ["docker", "images", "--format", "{{.ID}}", f"r8.im/nauslava/plitka-pro-project:{version}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()[:8]  # Возвращаем первые 8 символов
            return None
        except Exception as e:
            print(f"⚠️ Ошибка получения Docker хеша: {e}")
            return None
    
    def get_replicate_version_hash(self, version: str) -> Optional[str]:
        """Получает хеш версии с Replicate API"""
        try:
            import os
            import replicate
            from dotenv import load_dotenv
            
            load_dotenv()
            client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
            
            # Получаем все версии модели
            model_info = client.models.get(self.config.get("replicate_model", "nauslava/plitka-pro-project"))
            versions = model_info.versions.list()
            
            # Ищем версию по хешу из конфигурации
            config_hash = self.get_replicate_hash(version)
            if config_hash:
                for ver in versions:
                    if config_hash in ver.id:
                        return config_hash
                # Если версия не найдена на Replicate, но есть в конфигурации
                print(f"⚠️ Версия {version} с хешем {config_hash} не найдена на Replicate")
                return None
            
            # Если не найдена, возвращаем latest
            latest = model_info.latest_version
            if latest:
                return latest.id[:8]
            
            return None
        except Exception as e:
            print(f"⚠️ Ошибка получения Replicate хеша: {e}")
            return None
    
    def get_optimal_hash(self, version: str) -> str:
        """Получает оптимальный хеш для версии (гибридный подход)"""
        print(f"🔍 Поиск хеша для версии: {version}")
        
        # Стратегия 1: Конфигурация версий (приоритет для локальных версий)
        config_hash = self.get_replicate_hash(version)
        if config_hash:
            print(f"✅ Найден хеш из конфигурации: {config_hash}")
            # Проверяем, есть ли эта версия на Replicate
            replicate_hash = self.get_replicate_version_hash(version)
            if replicate_hash and replicate_hash == config_hash:
                print(f"✅ Версия подтверждена на Replicate: {replicate_hash}")
                return replicate_hash
            elif replicate_hash:
                print(f"⚠️ Конфликт хешей: конфиг={config_hash}, replicate={replicate_hash}")
                return config_hash  # Приоритет конфигурации
            else:
                print(f"⚠️ Версия {version} не найдена на Replicate, используем конфигурацию")
                return config_hash
        
        # Стратегия 2: Replicate API (только если нет в конфигурации)
        replicate_hash = self.get_replicate_version_hash(version)
        if replicate_hash:
            print(f"✅ Найден Replicate хеш: {replicate_hash}")
            return replicate_hash
        
        # Стратегия 3: Docker образы
        docker_hash = self.get_docker_image_hash(version)
        if docker_hash:
            print(f"✅ Найден Docker хеш: {docker_hash}")
            return docker_hash
        
        # Fallback: используем хеш latest версии из конфигурации
        fallback_hash = self.get_replicate_hash(self.get_default_version())
        if fallback_hash:
            print(f"⚠️ Используем fallback хеш latest версии: {fallback_hash}")
            return fallback_hash
        
        # Последний fallback: hardcoded (не рекомендуется)
        print("⚠️ Используем hardcoded fallback хеш")
        return "ca94ea46"
    
    def update_version_config(self, version: str, replicate_hash: str, full_hash: str, 
                            description: str, status: str = "stable"):
        """Обновляет конфигурацию версии"""
        if "version_mappings" not in self.config:
            self.config["version_mappings"] = {}
        
        self.config["version_mappings"][version] = {
            "replicate_hash": replicate_hash,
            "full_hash": full_hash,
            "description": description,
            "published_date": "2025-01-11",  # Можно сделать динамическим
            "status": status
        }
        
        # Сохраняем конфигурацию
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ Конфигурация версии {version} обновлена")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения конфигурации: {e}")
    
    def get_version_status(self, version: str) -> str:
        """Получает статус версии"""
        version_info = self.get_version_info(version)
        return version_info.get("status", "unknown") if version_info else "unknown"
    
    def is_latest_version(self, version: str) -> bool:
        """Проверяет, является ли версия latest"""
        return self.get_version_status(version) == "latest"
    
    def get_current_version_from_env(self) -> Optional[str]:
        """Получает текущую версию из переменных окружения"""
        return os.getenv('CURRENT_VERSION')
    
    def get_current_hash_from_env(self) -> Optional[str]:
        """Получает текущий хеш из переменных окружения"""
        return os.getenv('CURRENT_DOCKER_HASH') or os.getenv('CURRENT_GIT_HASH')
    
    def get_version_info_from_file(self) -> Optional[Dict]:
        """Получает информацию о версии из version_info.json"""
        version_info_file = Path('version_info.json')
        try:
            if version_info_file.exists():
                with open(version_info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка чтения version_info.json: {e}")
        return None
    
    def get_dynamic_version_info(self) -> Dict:
        """Получает динамическую информацию о версии (приоритет: env -> file -> config)"""
        # Приоритет 1: Переменные окружения
        env_version = self.get_current_version_from_env()
        env_hash = self.get_current_hash_from_env()
        
        if env_version and env_hash:
            return {
                "version": env_version,
                "hash": env_hash,
                "source": "environment"
            }
        
        # Приоритет 2: Файл version_info.json
        file_info = self.get_version_info_from_file()
        if file_info:
            return {
                "version": file_info.get("version"),
                "hash": file_info.get("docker_hash") or file_info.get("git_hash"),
                "source": "file"
            }
        
        # Приоритет 3: Конфигурация
        default_version = self.get_default_version()
        config_hash = self.get_replicate_hash(default_version)
        
        return {
            "version": default_version,
            "hash": config_hash,
            "source": "config"
        }


# Глобальный экземпляр менеджера версий
version_manager = VersionManager()


def get_optimal_hash(version: str) -> str:
    """Удобная функция для получения оптимального хеша"""
    return version_manager.get_optimal_hash(version)


def get_version_info(version: str) -> Optional[Dict]:
    """Удобная функция для получения информации о версии"""
    return version_manager.get_version_info(version)


def get_current_version() -> str:
    """Получает текущую версию (динамически)"""
    dynamic_info = version_manager.get_dynamic_version_info()
    return dynamic_info.get("version", "unknown")


def get_current_hash() -> str:
    """Получает текущий хеш (динамически)"""
    dynamic_info = version_manager.get_dynamic_version_info()
    return dynamic_info.get("hash", "unknown")


def get_current_version_info() -> Dict:
    """Получает полную информацию о текущей версии"""
    return version_manager.get_dynamic_version_info()


if __name__ == "__main__":
    # Тестирование модуля
    vm = VersionManager()
    
    print("=== Тестирование VersionManager ===")
    print(f"Доступные версии: {vm.get_available_versions()}")
    print(f"Версия по умолчанию: {vm.get_default_version()}")
    
    test_version = "v4.5.07"
    print(f"\nТестирование версии: {test_version}")
    print(f"Информация о версии: {vm.get_version_info(test_version)}")
    print(f"Replicate хеш: {vm.get_replicate_hash(test_version)}")
    print(f"Оптимальный хеш: {vm.get_optimal_hash(test_version)}")
