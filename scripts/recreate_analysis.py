#!/usr/bin/env python3
"""
Скрипт для пересоздания файлов анализа логов с исправленной классификацией
"""

import json
import os
from pathlib import Path
from datetime import datetime
import re

def parse_log_file(log_file_path):
    """Парсит файл лога и возвращает структурированные данные"""
    logs = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Разбиваем на отдельные записи логов
    log_entries = content.split('\n')
    
    for entry in log_entries:
        if not entry.strip():
            continue
            
        # Парсим временную метку и содержимое
        # Формат: [2025-09-12T20:23:20.410284] [INFO] content
        timestamp_match = re.match(r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.\d+\] \[(INFO|ERROR|WARNING)\] (.+)', entry)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1).replace('T', ' ')
            log_level = timestamp_match.group(2)
            content = timestamp_match.group(3)
            
            # Определяем тип лога с исправленной логикой
            log_type = "info"  # По умолчанию
            
            # Игнорируем неправильную классификацию в исходных файлах
            # Определяем тип только по содержимому с исправленной логикой
            if "▲" in content or "warning" in content.lower():
                log_type = "warning"
            elif "❌" in content or "error" in content.lower() or ("ошибка" in content and "❌" not in content):
                log_type = "error"
            elif "версия" in content.lower() and ("не найдена" in content or "неизвестна" in content):
                log_type = "version_issue"
            elif "пресет" in content.lower() and "содержит ошибки" in content:
                log_type = "preset_error"
            else:
                # Для успешных сообщений (✅) всегда info, независимо от уровня в файле
                log_type = "info"
            
            logs.append({
                "timestamp": timestamp_str,
                "content": content,
                "type": log_type
            })
    
    return logs

def create_session_analysis(log_file_path, logs):
    """Создает файл анализа сессии"""
    filename = Path(log_file_path).name
    session_id = filename.replace('.log', '')
    
    # Извлекаем информацию о сессии из имени файла
    parts = session_id.split('_')
    version = parts[1] if len(parts) > 1 else "unknown"
    hash_id = parts[2] if len(parts) > 2 else "unknown"
    
    # Подсчитываем статистику
    stats = {
        "total_logs": len(logs),
        "warnings": sum(1 for log in logs if log["type"] == "warning"),
        "errors": sum(1 for log in logs if log["type"] == "error"),
        "version_issues": sum(1 for log in logs if log["type"] == "version_issue"),
        "preset_errors": sum(1 for log in logs if log["type"] == "preset_error")
    }
    
    # Определяем время начала и окончания
    if logs:
        start_time = logs[0]["timestamp"]
        end_time = logs[-1]["timestamp"]
        
        # Простой расчет продолжительности
        try:
            start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
            duration = str(end_dt - start_dt)
        except:
            duration = "unknown"
    else:
        start_time = "unknown"
        end_time = "unknown"
        duration = "unknown"
    
    session_info = {
        "session_id": session_id,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "version": version,
        "hash_id": hash_id
    }
    
    analysis_data = {
        "session_info": session_info,
        "statistics": stats,
        "all_logs": logs,
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    return analysis_data

def main():
    """Основная функция"""
    logs_dir = Path("docs/reports/gui_logs/realtime")
    
    if not logs_dir.exists():
        print("❌ Директория логов не найдена")
        return
    
    # Находим все файлы логов
    log_files = list(logs_dir.glob("*.log"))
    
    if not log_files:
        print("❌ Файлы логов не найдены")
        return
    
    print(f"📊 Найдено {len(log_files)} файлов логов для пересоздания")
    
    for log_file in log_files:
        print(f"🔄 Обрабатываю: {log_file.name}")
        
        # Парсим лог
        logs = parse_log_file(log_file)
        
        # Создаем анализ
        analysis_data = create_session_analysis(log_file, logs)
        
        # Сохраняем файл анализа
        analysis_file = log_file.parent / f"{log_file.stem}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Создан: {analysis_file.name}")
        print(f"  - Логов: {analysis_data['statistics']['total_logs']}")
        print(f"  - Предупреждения: {analysis_data['statistics']['warnings']}")
        print(f"  - Ошибки: {analysis_data['statistics']['errors']}")
        print(f"  - Проблемы с версиями: {analysis_data['statistics']['version_issues']}")
        print(f"  - Ошибки пресетов: {analysis_data['statistics']['preset_errors']}")
        print()

if __name__ == "__main__":
    main()
