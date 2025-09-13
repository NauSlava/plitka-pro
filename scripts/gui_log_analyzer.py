#!/usr/bin/env python3
"""
GUI Log Analyzer для Plitka Pro
Анализирует логи GUI и сохраняет их для дальнейшего анализа
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

class GUILogAnalyzer:
    def __init__(self, project_root="/home/papaandrey/plitka-pro-project"):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "docs" / "reports" / "gui_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
    def save_gui_logs(self, log_content, version="unknown", hash_id="unknown"):
        """Сохраняет логи GUI в файл для анализа"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gui_logs_v{version}_{hash_id}_{timestamp}.txt"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== GUI Logs Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"Version: {version}\n")
            f.write(f"Hash: {hash_id}\n")
            f.write("=" * 80 + "\n\n")
            f.write(log_content)
        
        print(f"✅ Логи GUI сохранены: {filepath}")
        return filepath
    
    def analyze_gui_logs(self, log_content):
        """Анализирует логи GUI и извлекает проблемы"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "warnings": [],
            "errors": [],
            "version_issues": [],
            "preset_errors": [],
            "statistics": {}
        }
        
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            # Поиск предупреждений
            if "▲" in line and "WARNING" in line.upper():
                analysis["warnings"].append({
                    "line": i + 1,
                    "content": line.strip(),
                    "type": "warning"
                })
            
            # Поиск ошибок
            if "х" in line and "ошибки" in line.lower():
                analysis["errors"].append({
                    "line": i + 1,
                    "content": line.strip(),
                    "type": "error"
                })
            
            # Поиск проблем с версиями
            if "версия" in line.lower() and "не найдена" in line.lower():
                analysis["version_issues"].append({
                    "line": i + 1,
                    "content": line.strip(),
                    "type": "version_issue"
                })
            
            # Поиск ошибок пресетов
            if "пресет" in line.lower() and "содержит ошибки" in line.lower():
                analysis["preset_errors"].append({
                    "line": i + 1,
                    "content": line.strip(),
                    "type": "preset_error"
                })
        
        # Статистика
        analysis["statistics"] = {
            "total_warnings": len(analysis["warnings"]),
            "total_errors": len(analysis["errors"]),
            "version_issues": len(analysis["version_issues"]),
            "preset_errors": len(analysis["preset_errors"]),
            "total_lines": len(lines)
        }
        
        return analysis
    
    def save_analysis(self, analysis, version="unknown", hash_id="unknown"):
        """Сохраняет анализ в JSON файл"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gui_analysis_v{version}_{hash_id}_{timestamp}.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Анализ GUI сохранен: {filepath}")
        return filepath
    
    def generate_report(self, analysis, version="unknown", hash_id="unknown"):
        """Генерирует отчет по анализу логов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gui_report_v{version}_{hash_id}_{timestamp}.md"
        filepath = self.logs_dir / filename
        
        report = f"""# GUI Logs Analysis Report

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Версия:** {version}  
**Хеш:** {hash_id}  

## 📊 Статистика

- **Всего строк:** {analysis['statistics']['total_lines']}
- **Предупреждения:** {analysis['statistics']['total_warnings']}
- **Ошибки:** {analysis['statistics']['total_errors']}
- **Проблемы с версиями:** {analysis['statistics']['version_issues']}
- **Ошибки пресетов:** {analysis['statistics']['preset_errors']}

## ⚠️ Предупреждения

"""
        
        for warning in analysis["warnings"]:
            report += f"- **Строка {warning['line']}:** {warning['content']}\n"
        
        report += "\n## ❌ Ошибки\n\n"
        
        for error in analysis["errors"]:
            report += f"- **Строка {error['line']}:** {error['content']}\n"
        
        report += "\n## 🔧 Проблемы с версиями\n\n"
        
        for issue in analysis["version_issues"]:
            report += f"- **Строка {issue['line']}:** {issue['content']}\n"
        
        report += "\n## 🎨 Ошибки пресетов\n\n"
        
        for preset_error in analysis["preset_errors"]:
            report += f"- **Строка {preset_error['line']}:** {preset_error['content']}\n"
        
        report += f"\n---\n*Отчет сгенерирован автоматически {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Отчет GUI сохранен: {filepath}")
        return filepath

def main():
    """Основная функция для тестирования"""
    analyzer = GUILogAnalyzer()
    
    # Пример использования
    sample_log = """
    ▲ Версия v4.5.09 не найдена или неизвестна, принудительно используем v4.5.01
    Загружены пресеты из: test_inputs_v4.5.01_critical_fixes.json
    ■ Валидация пресетов...
    ▲ Пресет 'TRIO COLOR - CONTROLNET METHODS' содержит ошибки:
    х Неизвестный цвет 'dkgreen'. Доступные цвета: RED, BLUE, GREEN, YELLOW, ORANGE, PINK, WHITE, BLACK, GRAY, BROWN...
    х Сумма процентов не равна 100%: 70.0%
    """
    
    # Сохранение логов
    log_file = analyzer.save_gui_logs(sample_log, "v4.5.09", "a6fe1706")
    
    # Анализ логов
    analysis = analyzer.analyze_gui_logs(sample_log)
    
    # Сохранение анализа
    analysis_file = analyzer.save_analysis(analysis, "v4.5.09", "a6fe1706")
    
    # Генерация отчета
    report_file = analyzer.generate_report(analysis, "v4.5.09", "a6fe1706")
    
    print(f"\n📁 Файлы сохранены:")
    print(f"  - Логи: {log_file}")
    print(f"  - Анализ: {analysis_file}")
    print(f"  - Отчет: {report_file}")

if __name__ == "__main__":
    main()
