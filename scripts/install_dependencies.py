#!/usr/bin/env python3
"""
Скрипт для установки зависимостей GUI скрипта replicate_gui.py
"""

import subprocess
import sys

def install_package(package):
    """Устанавливает пакет через pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} успешно установлен")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Ошибка установки {package}")
        return False

def main():
    print("🔧 Установка зависимостей для GUI скрипта...")
    
    # Основные зависимости
    packages = [
        "pyyaml>=6.0.1",
        "replicate",
        "pillow",
        "numpy"
    ]
    
    success_count = 0
    for package in packages:
        print(f"\n📦 Устанавливаю {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Результат: {success_count}/{len(packages)} пакетов установлено")
    
    if success_count == len(packages):
        print("🎉 Все зависимости установлены! GUI скрипт готов к работе.")
    else:
        print("⚠️ Некоторые зависимости не установлены. GUI скрипт может работать с ограничениями.")

if __name__ == "__main__":
    main()
