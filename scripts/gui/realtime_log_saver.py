#!/usr/bin/env python3
"""
Real-time GUI Log Saver для Plitka Pro
Автоматически сохраняет все логи GUI в реальном времени для анализа
"""

import os
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class RealtimeLogSaver:
    def __init__(self, project_root="/home/papaandrey/plitka-pro-project"):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "docs" / "reports" / "gui_logs" / "realtime"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Файлы для сохранения
        self.session_log_file = None
        self.analysis_file = None
        self.current_session_id = None
        
        # Буфер для логов
        self.log_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Статистика
        self.stats = {
            "total_logs": 0,
            "warnings": 0,
            "errors": 0,
            "version_issues": 0,
            "preset_errors": 0,
            "session_start": None
        }
        
        # Запускаем автосохранение
        self.start_autosave()
    
    def start_new_session(self, version="unknown", hash_id="unknown"):
        """Начинает новую сессию логирования"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_id = f"session_{version}_{hash_id}_{timestamp}"
        
        # Создаем файлы для сессии
        self.session_log_file = self.logs_dir / f"{self.current_session_id}.log"
        self.analysis_file = self.logs_dir / f"{self.current_session_id}_analysis.json"
        
        # Сбрасываем статистику
        self.stats = {
            "total_logs": 0,
            "warnings": 0,
            "errors": 0,
            "version_issues": 0,
            "preset_errors": 0,
            "session_start": datetime.now().isoformat(),
            "version": version,
            "hash_id": hash_id
        }
        
        # Очищаем буфер
        with self.buffer_lock:
            self.log_buffer = []
        
        # Записываем заголовок сессии
        self._write_session_header(version, hash_id)
        
        print(f"🔄 Начата новая сессия логирования: {self.current_session_id}")
    
    def add_log(self, log_content: str, log_type: str = "info"):
        """Добавляет лог в буфер для сохранения"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "content": log_content.strip(),
            "type": log_type,
            "session_id": self.current_session_id
        }
        
        # Анализируем лог на предмет проблем
        self._analyze_log_entry(log_entry)
        
        # Добавляем в буфер
        with self.buffer_lock:
            self.log_buffer.append(log_entry)
            self.stats["total_logs"] += 1
        
        # Записываем в файл немедленно
        self._write_log_entry(log_entry)
    
    def _analyze_log_entry(self, log_entry: Dict[str, Any]):
        """Анализирует лог на предмет проблем"""
        content = log_entry["content"].lower()
        
        # Подсчет предупреждений
        if "▲" in log_entry["content"] or "warning" in content:
            self.stats["warnings"] += 1
            log_entry["type"] = "warning"
        
        # Подсчет ошибок
        if "❌" in log_entry["content"] or "error" in content or ("ошибка" in content and "❌" not in content):
            self.stats["errors"] += 1
            log_entry["type"] = "error"
        
        # Проблемы с версиями
        if "версия" in content and ("не найдена" in content or "неизвестна" in content):
            self.stats["version_issues"] += 1
            log_entry["type"] = "version_issue"
        
        # Ошибки пресетов
        if "пресет" in content and "содержит ошибки" in content:
            self.stats["preset_errors"] += 1
            log_entry["type"] = "preset_error"
    
    def _write_session_header(self, version: str, hash_id: str):
        """Записывает заголовок сессии"""
        if not self.session_log_file:
            return
        
        header = f"""=== GUI Real-time Logs Session ===
Session ID: {self.current_session_id}
Version: {version}
Hash: {hash_id}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
========================================

"""
        
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Записывает лог в файл"""
        if not self.session_log_file:
            return
        
        timestamp = log_entry["timestamp"]
        content = log_entry["content"]
        log_type = log_entry["type"]
        
        # Форматируем лог
        formatted_log = f"[{timestamp}] [{log_type.upper()}] {content}\n"
        
        # Записываем в файл
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_log)
    
    def start_autosave(self):
        """Запускает автосохранение анализа каждые 30 секунд"""
        def autosave_worker():
            while True:
                time.sleep(30)  # Сохраняем каждые 30 секунд
                self._save_analysis()
        
        autosave_thread = threading.Thread(target=autosave_worker, daemon=True)
        autosave_thread.start()
    
    def _save_analysis(self):
        """Сохраняет анализ в JSON файл"""
        if not self.analysis_file:
            return
        
        analysis = {
            "session_info": {
                "session_id": self.current_session_id,
                "start_time": self.stats["session_start"],
                "version": self.stats.get("version", "unknown"),
                "hash_id": self.stats.get("hash_id", "unknown")
            },
            "statistics": self.stats,
            "recent_logs": self.log_buffer[-50:],  # Последние 50 логов
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    def save_final_analysis(self):
        """Сохраняет финальный анализ сессии"""
        if not self.analysis_file:
            return
        
        # Добавляем финальную информацию
        self.stats["session_end"] = datetime.now().isoformat()
        self.stats["total_duration"] = self._calculate_duration()
        
        # Сохраняем полный анализ
        analysis = {
            "session_info": {
                "session_id": self.current_session_id,
                "start_time": self.stats["session_start"],
                "end_time": self.stats["session_end"],
                "duration": self.stats["total_duration"],
                "version": self.stats.get("version", "unknown"),
                "hash_id": self.stats.get("hash_id", "unknown")
            },
            "statistics": self.stats,
            "all_logs": self.log_buffer,  # Все логи
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        with open(self.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # Создаем отчет в Markdown
        self._create_markdown_report()
        
        print(f"✅ Финальный анализ сохранен: {self.analysis_file}")
    
    def _calculate_duration(self):
        """Вычисляет продолжительность сессии"""
        if not self.stats["session_start"]:
            return "unknown"
        
        start_time = datetime.fromisoformat(self.stats["session_start"])
        end_time = datetime.now()
        duration = end_time - start_time
        
        return str(duration)
    
    def _create_markdown_report(self):
        """Создает отчет в формате Markdown"""
        report_file = self.logs_dir / f"{self.current_session_id}_report.md"
        
        report = f"""# GUI Real-time Logs Report

**Сессия:** {self.current_session_id}  
**Версия:** {self.stats.get('version', 'unknown')}  
**Хеш:** {self.stats.get('hash_id', 'unknown')}  
**Начало:** {self.stats.get('session_start', 'unknown')}  
**Окончание:** {self.stats.get('session_end', 'unknown')}  
**Продолжительность:** {self.stats.get('total_duration', 'unknown')}  

## 📊 Статистика

- **Всего логов:** {self.stats['total_logs']}
- **Предупреждения:** {self.stats['warnings']}
- **Ошибки:** {self.stats['errors']}
- **Проблемы с версиями:** {self.stats['version_issues']}
- **Ошибки пресетов:** {self.stats['preset_errors']}

## 📝 Последние логи

"""
        
        # Добавляем последние 20 логов
        recent_logs = self.log_buffer[-20:] if self.log_buffer else []
        for log in recent_logs:
            report += f"- **[{log['type'].upper()}]** {log['content']}\n"
        
        report += f"\n---\n*Отчет создан автоматически {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Отчет создан: {report_file}")

# Глобальный экземпляр для использования в GUI
realtime_log_saver = RealtimeLogSaver()

def get_log_saver():
    """Возвращает экземпляр логгера"""
    return realtime_log_saver

if __name__ == "__main__":
    # Тестирование
    saver = RealtimeLogSaver()
    saver.start_new_session("v4.5.09", "a6fe1706")
    
    # Добавляем тестовые логи
    saver.add_log("Тест 1: Запуск GUI", "info")
    saver.add_log("▲ Предупреждение: Версия не найдена", "warning")
    saver.add_log("х Ошибка: Пресет содержит ошибки", "error")
    
    time.sleep(2)
    saver.save_final_analysis()
