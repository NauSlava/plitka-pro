#!/usr/bin/env python3
"""
Скрипт для автоматического обновления информации о версии и хеше
Используется при публикации на Replicate
"""

import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def get_git_version():
    """Получает версию из git тегов"""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except Exception as e:
        print(f"⚠️ Ошибка получения git версии: {e}")
        return "unknown"

def get_git_commit_hash():
    """Получает хеш коммита из git"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except Exception as e:
        print(f"⚠️ Ошибка получения git хеша: {e}")
        return "unknown"

def get_docker_image_hash(version):
    """Получает хеш Docker образа для указанной версии"""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.ID}}", f"r8.im/nauslava/plitka-pro-project:{version}"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:8]
        return None
    except Exception as e:
        print(f"⚠️ Ошибка получения Docker хеша: {e}")
        return None

def update_version_info():
    """Обновляет информацию о версии и хеше"""
    
    # Получаем информацию из git
    git_version = get_git_version()
    git_hash = get_git_commit_hash()
    
    print(f"🔍 Git версия: {git_version}")
    print(f"🔍 Git хеш: {git_hash}")
    
    # Определяем версию для использования
    if git_version.startswith('v'):
        current_version = git_version
    else:
        # Если нет тега, используем версию из cog.yaml
        try:
            with open('cog.yaml', 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.strip().startswith('# Version:'):
                        current_version = line.split(':')[1].strip()
                        break
                else:
                    current_version = "v4.5.06"  # fallback
        except:
            current_version = "v4.5.06"  # fallback
    
    print(f"📋 Текущая версия: {current_version}")
    
    # Получаем Docker хеш
    docker_hash = get_docker_image_hash(current_version)
    if docker_hash:
        print(f"🐳 Docker хеш: {docker_hash}")
    else:
        print("⚠️ Docker хеш не найден")
    
    # Создаем файл с информацией о версии
    version_info = {
        "version": current_version,
        "git_version": git_version,
        "git_hash": git_hash,
        "docker_hash": docker_hash,
        "updated_at": datetime.now().isoformat(),
        "build_info": {
            "python_version": sys.version,
            "platform": sys.platform
        }
    }
    
    # Сохраняем в version_info.json
    with open('version_info.json', 'w', encoding='utf-8') as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Информация о версии сохранена в version_info.json")
    
    # Обновляем .env файл
    env_file = Path('.env')
    if env_file.exists():
        # Читаем существующий .env
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        # Обновляем или добавляем переменные
        env_lines = env_content.split('\n')
        updated_lines = []
        
        # Переменные для обновления
        env_vars = {
            'CURRENT_VERSION': current_version,
            'CURRENT_GIT_HASH': git_hash,
            'CURRENT_DOCKER_HASH': docker_hash or 'unknown'
        }
        
        # Обрабатываем существующие строки
        for line in env_lines:
            if '=' in line and not line.strip().startswith('#'):
                key = line.split('=')[0].strip()
                if key in env_vars:
                    updated_lines.append(f"{key}={env_vars[key]}")
                    del env_vars[key]
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        # Добавляем новые переменные
        for key, value in env_vars.items():
            updated_lines.append(f"{key}={value}")
        
        # Записываем обновленный .env
        with open(env_file, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"✅ .env файл обновлен с переменными версии")
    
    return version_info

if __name__ == "__main__":
    print("=== Обновление информации о версии ===")
    version_info = update_version_info()
    print("\n📋 Итоговая информация:")
    for key, value in version_info.items():
        print(f"  {key}: {value}")

