#!/usr/bin/env python3
"""
GUI Logs Analyzer для Plitka Pro
Анализирует сохраненные логи GUI и создает отчеты
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class GUILogsAnalyzer:
    def __init__(self, project_root="/home/papaandrey/plitka-pro-project"):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "docs" / "reports" / "gui_logs" / "realtime"
        self.reports_dir = self.project_root / "docs" / "reports"
        
    def analyze_all_sessions(self):
        """Анализирует все сессии логирования"""
        if not self.logs_dir.exists():
            print("❌ Директория логов не найдена")
            return
        
        # Находим все файлы анализа
        analysis_files = list(self.logs_dir.glob("*_analysis.json"))
        
        if not analysis_files:
            print("❌ Файлы анализа не найдены")
            return
        
        print(f"📊 Найдено {len(analysis_files)} сессий для анализа")
        
        # Анализируем каждую сессию
        all_sessions = []
        for analysis_file in analysis_files:
            session_data = self._load_session_analysis(analysis_file)
            if session_data:
                all_sessions.append(session_data)
        
        # Создаем сводный отчет
        self._create_summary_report(all_sessions)
        
        # Создаем отчет по проблемам
        self._create_issues_report(all_sessions)
        
        print("✅ Анализ завершен")
    
    def _load_session_analysis(self, analysis_file: Path) -> Dict[str, Any]:
        """Загружает анализ сессии"""
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Ошибка загрузки {analysis_file}: {e}")
            return None
    
    def _create_summary_report(self, sessions: List[Dict[str, Any]]):
        """Создает сводный отчет по всем сессиям"""
        report_file = self.reports_dir / "gui_logs_summary_report.md"
        
        # Статистика по всем сессиям
        total_sessions = len(sessions)
        total_logs = sum(s.get('statistics', {}).get('total_logs', 0) for s in sessions)
        total_warnings = sum(s.get('statistics', {}).get('warnings', 0) for s in sessions)
        total_errors = sum(s.get('statistics', {}).get('errors', 0) for s in sessions)
        total_version_issues = sum(s.get('statistics', {}).get('version_issues', 0) for s in sessions)
        total_preset_errors = sum(s.get('statistics', {}).get('preset_errors', 0) for s in sessions)
        
        report = f"""# GUI Logs Summary Report

**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Всего сессий:** {total_sessions}  

## 📊 Общая статистика

- **Всего логов:** {total_logs}
- **Предупреждения:** {total_warnings}
- **Ошибки:** {total_errors}
- **Проблемы с версиями:** {total_version_issues}
- **Ошибки пресетов:** {total_preset_errors}

## 📈 Статистика по сессиям

"""
        
        for session in sessions:
            session_info = session.get('session_info', {})
            stats = session.get('statistics', {})
            
            report += f"""### {session_info.get('session_id', 'Unknown')}
- **Версия:** {session_info.get('version', 'unknown')}
- **Хеш:** {session_info.get('hash_id', 'unknown')}
- **Начало:** {session_info.get('start_time', 'unknown')}
- **Продолжительность:** {session_info.get('duration', 'unknown')}
- **Логов:** {stats.get('total_logs', 0)}
- **Предупреждения:** {stats.get('warnings', 0)}
- **Ошибки:** {stats.get('errors', 0)}

"""
        
        report += f"""## 🔍 Анализ проблем

### Топ проблем по частоте:
"""
        
        # Анализируем частые проблемы
        problem_counts = {}
        for session in sessions:
            logs = session.get('all_logs', [])
            for log in logs:
                content = log.get('content', '')
                if '▲' in content or 'warning' in content.lower():
                    problem_counts['warnings'] = problem_counts.get('warnings', 0) + 1
                elif '❌' in content or 'error' in content.lower() or ('ошибка' in content.lower() and '❌' not in content):
                    problem_counts['errors'] = problem_counts.get('errors', 0) + 1
                elif 'версия' in content.lower() and 'не найдена' in content.lower():
                    problem_counts['version_issues'] = problem_counts.get('version_issues', 0) + 1
                elif 'пресет' in content.lower() and 'содержит ошибки' in content.lower():
                    problem_counts['preset_errors'] = problem_counts.get('preset_errors', 0) + 1
        
        for problem, count in sorted(problem_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- **{problem}:** {count} раз\n"
        
        report += f"\n---\n*Отчет создан автоматически {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Сводный отчет создан: {report_file}")
    
    def _create_issues_report(self, sessions: List[Dict[str, Any]]):
        """Создает отчет по проблемам"""
        report_file = self.reports_dir / "gui_logs_issues_report.md"
        
        report = f"""# GUI Logs Issues Report

**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

## 🚨 Критические проблемы

"""
        
        # Собираем все проблемы
        all_issues = []
        for session in sessions:
            logs = session.get('all_logs', [])
            for log in logs:
                if log.get('type') in ['error', 'warning', 'version_issue', 'preset_error']:
                    all_issues.append({
                        'session': session.get('session_info', {}).get('session_id', 'unknown'),
                        'timestamp': log.get('timestamp', 'unknown'),
                        'type': log.get('type', 'unknown'),
                        'content': log.get('content', '')
                    })
        
        # Группируем по типам
        issues_by_type = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Создаем отчет по типам
        for issue_type, issues in issues_by_type.items():
            report += f"### {issue_type.upper()}\n\n"
            for issue in issues[:10]:  # Показываем первые 10
                report += f"- **{issue['timestamp']}** [{issue['session']}]: {issue['content']}\n"
            if len(issues) > 10:
                report += f"- ... и еще {len(issues) - 10} проблем\n"
            report += "\n"
        
        report += f"""## 📋 Рекомендации

1. **Мониторинг:** Регулярно проверяйте логи на предмет новых проблем
2. **Автоматизация:** Настройте автоматические уведомления о критических ошибках
3. **Анализ трендов:** Отслеживайте изменения в количестве проблем по версиям
4. **Оптимизация:** Фокусируйтесь на наиболее частых проблемах

---
*Отчет создан автоматически {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Отчет по проблемам создан: {report_file}")
    
    def analyze_latest_session(self):
        """Анализирует последнюю сессию"""
        if not self.logs_dir.exists():
            print("❌ Директория логов не найдена")
            return
        
        # Находим последний файл анализа
        analysis_files = list(self.logs_dir.glob("*_analysis.json"))
        if not analysis_files:
            print("❌ Файлы анализа не найдены")
            return
        
        latest_file = max(analysis_files, key=os.path.getctime)
        session_data = self._load_session_analysis(latest_file)
        
        if not session_data:
            print("❌ Ошибка загрузки последней сессии")
            return
        
        print(f"📊 Анализ последней сессии: {latest_file.name}")
        
        stats = session_data.get('statistics', {})
        print(f"  - Логов: {stats.get('total_logs', 0)}")
        print(f"  - Предупреждения: {stats.get('warnings', 0)}")
        print(f"  - Ошибки: {stats.get('errors', 0)}")
        print(f"  - Проблемы с версиями: {stats.get('version_issues', 0)}")
        print(f"  - Ошибки пресетов: {stats.get('preset_errors', 0)}")

def main():
    """Основная функция"""
    analyzer = GUILogsAnalyzer()
    
    print("🔍 Анализ логов GUI...")
    analyzer.analyze_all_sessions()
    
    print("\n📊 Анализ последней сессии:")
    analyzer.analyze_latest_session()

if __name__ == "__main__":
    main()
