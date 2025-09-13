#!/usr/bin/env python3
"""
Скрипт для автоматической публикации версии с обновлением информации
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Выполняет команду с описанием"""
    print(f"\n🔄 {description}...")
    print(f"Команда: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} - успешно")
        if result.stdout:
            print(f"Вывод: {result.stdout}")
    else:
        print(f"❌ {description} - ошибка")
        print(f"Ошибка: {result.stderr}")
        return False
    
    return True

def update_version_info():
    """Обновляет информацию о версии"""
    print("\n📋 Обновление информации о версии...")
    
    try:
        # Запускаем скрипт обновления версии
        result = subprocess.run(
            [sys.executable, "scripts/update_version_info.py"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print("✅ Информация о версии обновлена")
            print(f"Вывод: {result.stdout}")
        else:
            print(f"❌ Ошибка обновления версии: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Ошибка запуска скрипта обновления: {e}")
        return False
    
    return True

def build_and_push():
    """Собирает и публикует образ"""
    print("\n🚀 Сборка и публикация образа...")
    
    # Обновляем информацию о версии
    if not update_version_info():
        print("❌ Не удалось обновить информацию о версии")
        return False
    
    # Собираем образ
    if not run_command("cog build", "Сборка Docker образа"):
        return False
    
    # Публикуем образ
    if not run_command("cog push", "Публикация на Replicate"):
        return False
    
    print("\n✅ Публикация завершена успешно!")
    return True

def main():
    """Основная функция"""
    print("=== Автоматическая публикация версии ===")
    
    # Проверяем, что мы в корне проекта
    if not Path("cog.yaml").exists():
        print("❌ Ошибка: cog.yaml не найден. Запустите скрипт из корня проекта.")
        return 1
    
    # Проверяем, что cog установлен
    try:
        result = subprocess.run(["cog", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Ошибка: cog не установлен или не доступен")
            return 1
    except FileNotFoundError:
        print("❌ Ошибка: cog не найден в PATH")
        return 1
    
    # Выполняем сборку и публикацию
    if build_and_push():
        print("\n🎉 Публикация завершена успешно!")
        print("\n📋 Следующие шаги:")
        print("1. Проверьте статус на Replicate")
        print("2. Активируйте новую версию, если необходимо")
        print("3. Протестируйте через GUI тестер")
        return 0
    else:
        print("\n❌ Публикация завершилась с ошибками")
        return 1

if __name__ == "__main__":
    sys.exit(main())

