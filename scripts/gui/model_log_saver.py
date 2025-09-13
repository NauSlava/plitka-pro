#!/usr/bin/env python3
"""
Система сохранения логов модели в реальном времени
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class ModelLogSaver:
    """Сохраняет логи модели в реальном времени"""
    
    def __init__(self, base_dir: str = "replicate_runs"):
        self.base_dir = Path(base_dir)
        self.current_session_dir: Optional[Path] = None
        self.current_log_file: Optional[Path] = None
        self.session_info: Dict[str, Any] = {}
        self.log_buffer: list = []
        
    def start_new_session(self, version: str, hash_id: str, prediction_id: str) -> None:
        """Начинает новую сессию логирования модели"""
        try:
            # Создаем папку для сессии
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"model_logs_{version}_{hash_id}_{timestamp}"
            
            # Создаем структуру папок
            hash_dir = self.base_dir / hash_id
            session_dir = hash_dir / session_name
            
            # Создаем папки если не существуют
            session_dir.mkdir(parents=True, exist_ok=True)
            
            self.current_session_dir = session_dir
            self.current_log_file = session_dir / "model_realtime.log"
            
            # Сохраняем информацию о сессии
            self.session_info = {
                "session_id": session_name,
                "version": version,
                "hash_id": hash_id,
                "prediction_id": prediction_id,
                "start_time": datetime.now().isoformat(),
                "log_file": str(self.current_log_file)
            }
            
            # Создаем файл логов с заголовком
            with open(self.current_log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== MODEL REAL-TIME LOGS ===\n")
                f.write(f"Session: {session_name}\n")
                f.write(f"Version: {version}\n")
                f.write(f"Hash: {hash_id}\n")
                f.write(f"Prediction ID: {prediction_id}\n")
                f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
            
            # Сохраняем информацию о сессии
            session_info_file = session_dir / "session_info.json"
            with open(session_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_info, f, ensure_ascii=False, indent=2)
                
            print(f"📁 Начата сессия логирования модели: {session_name}")
            print(f"📄 Файл логов: {self.current_log_file}")
            
        except Exception as e:
            print(f"❌ Ошибка создания сессии логирования модели: {e}")
    
    def add_log(self, log_text: str) -> None:
        """Добавляет лог в реальном времени"""
        if not self.current_log_file:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Добавляем в буфер
            self.log_buffer.append({
                "timestamp": timestamp,
                "content": log_text.strip()
            })
            
            # Записываем в файл
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {log_text}")
                if not log_text.endswith('\n'):
                    f.write('\n')
                    
        except Exception as e:
            print(f"❌ Ошибка записи лога модели: {e}")
    
    def save_final_analysis(self) -> None:
        """Сохраняет финальный анализ сессии"""
        if not self.current_session_dir:
            return
            
        try:
            # Обновляем информацию о сессии
            self.session_info.update({
                "end_time": datetime.now().isoformat(),
                "total_logs": len(self.log_buffer),
                "log_file_size": self.current_log_file.stat().st_size if self.current_log_file.exists() else 0
            })
            
            # Сохраняем финальную информацию
            session_info_file = self.current_session_dir / "session_info.json"
            with open(session_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_info, f, ensure_ascii=False, indent=2)
            
            # Создаем анализ логов
            analysis_file = self.current_session_dir / "logs_analysis.json"
            analysis = {
                "session_info": self.session_info,
                "log_statistics": {
                    "total_logs": len(self.log_buffer),
                    "start_time": self.log_buffer[0]["timestamp"] if self.log_buffer else None,
                    "end_time": self.log_buffer[-1]["timestamp"] if self.log_buffer else None,
                    "log_file_size_bytes": self.current_log_file.stat().st_size if self.current_log_file.exists() else 0
                },
                "all_logs": self.log_buffer,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            
            print(f"📊 Анализ логов модели сохранен: {analysis_file}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения анализа логов модели: {e}")
    
    def get_current_log_path(self) -> Optional[str]:
        """Возвращает путь к текущему файлу логов"""
        return str(self.current_log_file) if self.current_log_file else None

# Глобальный экземпляр для использования в GUI
_model_log_saver: Optional[ModelLogSaver] = None

def get_model_log_saver() -> ModelLogSaver:
    """Возвращает глобальный экземпляр ModelLogSaver"""
    global _model_log_saver
    if _model_log_saver is None:
        _model_log_saver = ModelLogSaver()
    return _model_log_saver

def start_model_logging(version: str, hash_id: str, prediction_id: str) -> None:
    """Начинает логирование модели"""
    saver = get_model_log_saver()
    saver.start_new_session(version, hash_id, prediction_id)

def add_model_log(log_text: str) -> None:
    """Добавляет лог модели"""
    saver = get_model_log_saver()
    saver.add_log(log_text)

def finish_model_logging() -> None:
    """Завершает логирование модели"""
    saver = get_model_log_saver()
    saver.save_final_analysis()

def get_model_log_path() -> Optional[str]:
    """Возвращает путь к текущему файлу логов модели"""
    saver = get_model_log_saver()
    return saver.get_current_log_path()
