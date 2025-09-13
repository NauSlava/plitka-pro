#!/usr/bin/env python3
"""
Анализ логов модели в реальном времени
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def find_model_log_sessions(base_dir: str = "replicate_runs") -> List[Path]:
    """Находит все сессии логирования модели"""
    base_path = Path(base_dir)
    sessions = []
    
    if not base_path.exists():
        return sessions
    
    # Ищем папки с model_logs_
    for hash_dir in base_path.iterdir():
        if hash_dir.is_dir():
            for session_dir in hash_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith("model_logs_"):
                    sessions.append(session_dir)
    
    # Сортируем по времени создания (новые сначала)
    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return sessions

def analyze_session(session_dir: Path) -> Dict[str, Any]:
    """Анализирует одну сессию логирования модели"""
    session_info = {}
    
    # Загружаем информацию о сессии
    session_info_file = session_dir / "session_info.json"
    if session_info_file.exists():
        try:
            with open(session_info_file, 'r', encoding='utf-8') as f:
                session_info = json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки session_info.json: {e}")
    
    # Загружаем анализ логов
    analysis_file = session_dir / "logs_analysis.json"
    if analysis_file.exists():
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                session_info.update(analysis)
        except Exception as e:
            print(f"❌ Ошибка загрузки logs_analysis.json: {e}")
    
    # Проверяем файл логов
    log_file = session_dir / "model_realtime.log"
    if log_file.exists():
        session_info["log_file_size"] = log_file.stat().st_size
        session_info["log_file_exists"] = True
    else:
        session_info["log_file_exists"] = False
    
    return session_info

def create_summary_report(sessions: List[Dict[str, Any]]) -> str:
    """Создает сводный отчет по всем сессиям"""
    if not sessions:
        return "# Model Logs Summary Report\n\n**Нет сессий логирования модели.**\n"
    
    total_sessions = len(sessions)
    total_logs = sum(s.get('log_statistics', {}).get('total_logs', 0) for s in sessions)
    total_size = sum(s.get('log_file_size', 0) for s in sessions)
    
    report = f"""# Model Logs Summary Report

**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Общая статистика

- **Всего сессий:** {total_sessions}
- **Всего логов:** {total_logs}
- **Общий размер логов:** {total_size:,} байт

## 📈 Статистика по сессиям

"""
    
    for i, session in enumerate(sessions, 1):
        session_id = session.get('session_id', 'unknown')
        version = session.get('version', 'unknown')
        hash_id = session.get('hash_id', 'unknown')
        prediction_id = session.get('prediction_id', 'unknown')
        start_time = session.get('start_time', 'unknown')
        end_time = session.get('end_time', 'unknown')
        total_logs = session.get('log_statistics', {}).get('total_logs', 0)
        log_size = session.get('log_file_size', 0)
        
        report += f"""### {i}. {session_id}

- **Версия:** {version}
- **Хеш:** {hash_id}
- **Prediction ID:** {prediction_id}
- **Начало:** {start_time}
- **Окончание:** {end_time}
- **Логов:** {total_logs}
- **Размер:** {log_size:,} байт

"""
    
    return report

def create_issues_report(sessions: List[Dict[str, Any]]) -> str:
    """Создает отчет по проблемам"""
    issues = []
    
    for session in sessions:
        session_id = session.get('session_id', 'unknown')
        logs = session.get('all_logs', [])
        
        # Ищем ошибки в логах
        for log in logs:
            content = log.get('content', '')
            if any(keyword in content.lower() for keyword in ['error', 'ошибка', 'failed', 'exception']):
                issues.append({
                    'session': session_id,
                    'timestamp': log.get('timestamp', 'unknown'),
                    'content': content,
                    'type': 'error'
                })
        
        # Проверяем отсутствие файла логов
        if not session.get('log_file_exists', False):
            issues.append({
                'session': session_id,
                'timestamp': session.get('start_time', 'unknown'),
                'content': 'Файл логов отсутствует',
                'type': 'missing_file'
            })
    
    if not issues:
        return "# Model Logs Issues Report\n\n**Проблем не найдено.**\n"
    
    report = f"""# Model Logs Issues Report

**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🚨 Найденные проблемы

"""
    
    for issue in issues:
        report += f"""### {issue['type'].upper()}

- **Сессия:** {issue['session']}
- **Время:** {issue['timestamp']}
- **Содержимое:** {issue['content']}

"""
    
    return report

def main():
    """Основная функция"""
    print("🔍 Поиск сессий логирования модели...")
    
    # Находим все сессии
    session_dirs = find_model_log_sessions()
    
    if not session_dirs:
        print("❌ Сессии логирования модели не найдены")
        return
    
    print(f"📊 Найдено {len(session_dirs)} сессий")
    
    # Анализируем каждую сессию
    sessions = []
    for session_dir in session_dirs:
        print(f"🔄 Анализирую: {session_dir.name}")
        session_info = analyze_session(session_dir)
        sessions.append(session_info)
    
    # Создаем отчеты
    reports_dir = Path("docs/reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Сводный отчет
    summary_report = create_summary_report(sessions)
    summary_file = reports_dir / "model_logs_summary_report.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"📄 Сводный отчет создан: {summary_file}")
    
    # Отчет по проблемам
    issues_report = create_issues_report(sessions)
    issues_file = reports_dir / "model_logs_issues_report.md"
    with open(issues_file, 'w', encoding='utf-8') as f:
        f.write(issues_report)
    print(f"📄 Отчет по проблемам создан: {issues_file}")
    
    print("✅ Анализ завершен")
    
    # Показываем информацию о последней сессии
    if sessions:
        latest = sessions[0]
        print(f"\n📊 Последняя сессия: {latest.get('session_id', 'unknown')}")
        print(f"  - Версия: {latest.get('version', 'unknown')}")
        print(f"  - Логов: {latest.get('log_statistics', {}).get('total_logs', 0)}")
        print(f"  - Размер: {latest.get('log_file_size', 0):,} байт")
        print(f"  - Файл логов: {'✅' if latest.get('log_file_exists', False) else '❌'}")

if __name__ == "__main__":
    main()
