#!/usr/bin/env python3
import os
import json
import threading
import time
import queue
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox
# Убираем PIL - используем только tkinter для совместимости

# Проверяем доступность модулей
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    print("⚠️ Модуль 'replicate' недоступен. GUI будет работать в демо-режиме.")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️ Модуль 'yaml' недоступен. Версии будут определяться по умолчанию.")

try:
    import hashlib
    HASHLIB_AVAILABLE = True
except ImportError:
    HASHLIB_AVAILABLE = False
    print("⚠️ Модуль 'hashlib' недоступен. Хеши будут генерироваться по умолчанию.")


def compare_versions(version1: str, version2: str) -> int:
    """Сравнивает версии. Возвращает -1 если version1 < version2, 0 если равны, 1 если version1 > version2"""
    def version_tuple(v):
        # Убираем 'v' и разбиваем на части
        v = v.replace('v', '')
        parts = v.split('.')
        return tuple(int(part) for part in parts)
    
    try:
        v1_tuple = version_tuple(version1)
        v2_tuple = version_tuple(version2)
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    except (ValueError, IndexError):
        # Если не можем распарсить, сравниваем как строки
        return -1 if version1 < version2 else (1 if version1 > version2 else 0)


def load_env_token() -> Optional[str]:
    token = os.getenv("REPLICATE_API_TOKEN")
    if token:
        return token
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("REPLICATE_API_TOKEN="):
                        _, value = line.split("=", 1)
                        value = value.strip().strip('"').strip("'")
                        if value:
                            return value
        except Exception:
            pass
    return None


def read_presets(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Presets JSON must be an object mapping preset names to input dicts")
        return data


class ReplicateWorker:
    def __init__(self, api_token: str, model_ref: str, poll_s: int, startup_timeout_s: int, total_timeout_s: int, ui_queue: "queue.Queue[str]", version_id: Optional[str] = None):
        self.api_token = api_token
        self.model_ref = model_ref
        self.poll_s = poll_s
        self.startup_timeout_s = startup_timeout_s
        self.total_timeout_s = total_timeout_s
        self.ui_queue = ui_queue
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_pred = None  # type: ignore
        self._start_time: Optional[float] = None
        self._status: str = "idle"  # idle|starting|processing|succeeded|failed|canceled|timeout
        self._current_run_dir: Optional[str] = None
        self._version_id: Optional[str] = version_id

    def log(self, text: str) -> None:
        try:
            self.ui_queue.put_nowait(text)
        except Exception:
            pass
    
    def _extract_version_from_cog_yaml(self) -> str:
        """Извлекает версию из cog.yaml без использования YAML модуля"""
        try:
            with open("cog.yaml", "r", encoding="utf-8") as f:
                content = f.read()
            
            # Ищем строку с версией в комментарии
            for line in content.split('\n'):
                if line.strip().startswith('# Version:'):
                    version_part = line.split('Version:')[1].strip()
                    return version_part.split()[0]  # Берем первое слово после "Version:"
                
                # Альтернативный поиск по тегу образа
                if 'image:' in line and 'plitka-pro-project:' in line:
                    version_part = line.split('plitka-pro-project:')[1].strip()
                    return version_part
                    
            # Если не нашли, пытаемся определить версию из других источников
            return self._determine_version_fallback()
        except Exception as e:
            print(f"Ошибка чтения cog.yaml: {e}")
            return self._determine_version_fallback()
    
    def get_optimal_hash(self, version: str) -> str:
        """Получает оптимальный хеш для версии (гибридный подход)"""
        print(f"🔍 Поиск хеша для версии: {version}")
        
        # Приоритет 1: Docker образ
        docker_hash = self.get_docker_image_hash(version)
        if docker_hash:
            print(f"✅ Найден Docker хеш: {docker_hash}")
            return docker_hash
        
        # Приоритет 2: Replicate API
        replicate_hash = self.get_replicate_version_hash(version)
        if replicate_hash:
            print(f"✅ Найден Replicate хеш: {replicate_hash}")
            return replicate_hash
        
        # Fallback: вычисление из версии
        computed_hash = self.compute_version_hash(version)
        print(f"⚠️ Используем вычисленный хеш: {computed_hash}")
        return computed_hash
    
    def get_docker_image_hash(self, version: str) -> str:
        """Получает хеш Docker образа для указанной версии"""
        try:
            import subprocess
            # Ищем образ по тегу версии
            result = subprocess.run(
                ["docker", "images", "--format", "{{.ID}}", f"r8.im/nauslava/plitka-pro-project:{version}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                full_hash = result.stdout.strip()
                return full_hash[:8]  # Первые 8 символов для папки
        except Exception as e:
            print(f"Ошибка получения Docker хеша: {e}")
        return ""
    
    def get_replicate_version_hash(self, version: str) -> str:
        """Получает хеш версии Replicate через API"""
        try:
            if not REPLICATE_AVAILABLE:
                return ""
            
            import replicate
            # Получаем информацию о модели
            model_info = replicate.models.get("nauslava/plitka-pro-project")
            if model_info and hasattr(model_info, 'versions'):
                # Проверяем, что versions итерируемо
                try:
                    versions_list = list(model_info.versions) if hasattr(model_info.versions, '__iter__') else []
                    for ver in versions_list:
                        if hasattr(ver, 'id') and ver.id == version:
                            # Используем digest версии
                            if hasattr(ver, 'digest') and ver.digest:
                                return ver.digest[:8]
                            # Или вычисляем из ID версии
                            import hashlib
                            hash_obj = hashlib.md5(ver.id.encode())
                            return hash_obj.hexdigest()[:8]
                except (TypeError, AttributeError) as e:
                    print(f"Ошибка итерации по версиям: {e}")
                    # Fallback: вычисляем хеш из версии
                    import hashlib
                    hash_obj = hashlib.md5(version.encode())
                    return hash_obj.hexdigest()[:8]
        except Exception as e:
            print(f"Ошибка получения Replicate хеша: {e}")
        return ""
    
    def compute_version_hash(self, version: str) -> str:
        """Вычисляет хеш версии как fallback"""
        try:
            import hashlib
            hash_obj = hashlib.md5(version.encode())
            return hash_obj.hexdigest()[:8]
        except Exception as e:
            print(f"Ошибка вычисления хеша версии: {e}")
            return "unknown"

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop_event.set()
        # Try to cancel current prediction if exists
        try:
            if self._current_pred is not None:
                self.log("\n[GUI] Cancelling current prediction...\n")
                try:
                    self._current_pred.cancel()
                except Exception:
                    pass
        except Exception:
            pass

    def _is_boot_error(self, logs: str) -> bool:
        if not logs:
            return False
        needles = [
            "Traceback (most recent call last)",
            "CRITICAL", "FATAL", "RuntimeError", "CUDA error", "device-side assert",
            "❌", "ERROR:predict:", "OSError: Cannot load model",
        ]
        return any(n in logs for n in needles)

    def _is_boot_completed(self, logs: str) -> bool:
        if not logs:
            return False
        markers = ["setup completed", "🎉 Модель v", "Model initialized"]
        return any(m in logs for m in markers)

    def _resolve_model_and_version(self, client, model_ref: str):
        # prefer explicitly provided version id
        if self._version_id:
            base = model_ref.split(":", 1)[0]
            return base, self._version_id
        if ":" not in model_ref:
            return model_ref, None
        base, ver = model_ref.split(":", 1)
        ver = ver.strip()
        if len(ver) == 64 and all(c in "0123456789abcdef" for c in ver.lower()):
            return base, ver
        try:
            if not REPLICATE_AVAILABLE:
                return base, None
            m = client.models.get(base)
            versions = list(m.versions.list())
            if versions:
                return base, versions[0].id
        except Exception:
            pass
        return base, None

    def _run_single(self, client, inputs: Dict[str, Any]) -> int:
        self.log("🔄 _run_single() вызван\n")
        model_name, version_id = self._resolve_model_and_version(client, self.model_ref)
        self.log(f"\n=== Creating prediction: {model_name} (version: {version_id or 'latest'})\nInputs: {json.dumps(inputs, ensure_ascii=False)}\n")
        try:
            if not REPLICATE_AVAILABLE:
                self.log("❌ Replicate недоступен. Запуск невозможен.\n")
                return 1
            if version_id:
                pred = client.predictions.create(version=version_id, input=inputs)
            else:
                pred = client.predictions.create(model=model_name, input=inputs)
        except Exception as e:
            # Fallback: try resolve latest version id then retry
            try:
                if not REPLICATE_AVAILABLE:
                    raise Exception("Replicate недоступен")
                
                self.log(f"🔄 Получение модели: {model_name}\n")
                m = client.models.get(model_name)
                self.log(f"✅ Модель получена\n")
                
                self.log(f"🔄 Получение версии модели...\n")
                ver = next(iter(m.versions.list())).id
                self.log(f"✅ Версия получена: {ver}\n")
                
                self.log(f"🔄 Создание предсказания...\n")
                pred = client.predictions.create(version=ver, input=inputs)
                self._version_id = ver
                self.log(f"✅ Предсказание создано: {pred.id}\n")
                
            except Exception as e:
                error_msg = f"❌ Ошибка при создании предсказания: {type(e).__name__}: {e}\n"
                self.log(error_msg)
                
                # Логируем детали ошибки для диагностики
                if "timeout" in str(e).lower():
                    self.log("⏰ Ошибка таймаута - возможно, проблемы с сетью\n")
                elif "connection" in str(e).lower():
                    self.log("🌐 Ошибка соединения - проверьте интернет\n")
                elif "unauthorized" in str(e).lower():
                    self.log("🔐 Ошибка авторизации - проверьте API токен\n")
                else:
                    self.log(f"🔍 Неизвестная ошибка: {e}\n")
                
                raise
        self._current_pred = pred
        self.log(f"Prediction id: {pred.id}\n")
        self.log(f"🔍 Начинаем создание папки для результатов...\n")
        current_dir = os.getcwd()
        self.log(f"🔍 Текущий рабочий каталог: {current_dir}\n")
        self._start_time = time.time()
        self._status = "starting"
        # Prepare run dir for streaming artifacts under short hash
        ts = time.strftime("%Y%m%d_%H%M%S")
        
        # Получаем SHA256 хеш Docker образа для корректного хеша папки
        # Используем глобальный импорт hashlib
        
        # Используем версию и хеш, переданные из App
        if hasattr(self, 'current_version') and hasattr(self, 'current_hash'):
            ver_for_hash = self.current_version
            forced_hash = self.current_hash
            
            self.log(f"🔍 Используем версию и хеш из App:\n")
            self.log(f"🔍 version_id: {version_id}\n")
            self.log(f"🔍 pred.version: {getattr(pred, 'version', 'None')}\n")
            self.log(f"🔍 ver_for_hash: {ver_for_hash}\n")
            self.log(f"🔐 Хеш из App: {forced_hash}\n")
        else:
            # Fallback: определяем версию и хеш автоматически
            try:
                ver_for_hash = self._extract_version_from_cog_yaml()
                if not ver_for_hash or ver_for_hash == "unknown":
                    ver_for_hash = self._determine_version_fallback()
                
                forced_hash = self.get_optimal_hash(ver_for_hash)
                if not forced_hash or forced_hash == "unknown":
                    import hashlib
                    hash_obj = hashlib.md5(ver_for_hash.encode())
                    forced_hash = hash_obj.hexdigest()[:8]
                
                self.log(f"🔍 Автоматически определенные значения:\n")
                self.log(f"🔍 ver_for_hash: {ver_for_hash}\n")
                self.log(f"🔐 forced_hash: {forced_hash}\n")
                
            except Exception as e:
                self.log(f"⚠️ Ошибка определения хеша: {e}, используем fallback\n")
                import hashlib
                fallback_version = self._determine_version_fallback()
                hash_obj = hashlib.md5(fallback_version.encode())
                forced_hash = hash_obj.hexdigest()[:8]
        
        # Используем абсолютные пути для надежности
        current_dir = os.getcwd()
        base_dir = os.path.join(current_dir, "replicate_runs", forced_hash)
        os.makedirs(base_dir, exist_ok=True)
        run_dir = os.path.join(base_dir, f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        self._current_run_dir = run_dir
        
        self.log(f"🔄 Папка создана: {run_dir}\n")
        self.log(f"🔄 _current_run_dir установлен: {self._current_run_dir}\n")
        
        # Логируем создание папки
        self.log(f"🔐 Создаем папку с хешем: {forced_hash}\n")
        
        # Дополнительное логирование для диагностики
        self.log(f"📁 Создана папка: {run_dir}\n")
        self.log(f"🔐 Хеш версии: {forced_hash}\n")
        self.log(f"🔍 Базовая папка: {base_dir}\n")
        self.log(f"🔍 Полный путь: {os.path.abspath(run_dir)}\n")
        
        # Проверяем, что папка действительно создана
        if os.path.exists(run_dir):
            self.log(f"✅ Папка успешно создана и существует\n")
        else:
            self.log(f"❌ ОШИБКА: Папка не создана!\n")
            # Создаем fallback папку с правильным хешем
            current_dir = os.getcwd()
            fallback_dir = os.path.join(current_dir, "replicate_runs", forced_hash, f"run_{ts}")
            os.makedirs(fallback_dir, exist_ok=True)
            self._current_run_dir = fallback_dir
            self.log(f"🔄 Создана fallback папка с правильным хешем: {fallback_dir}\n")
            
            # Дополнительно проверяем, что папка создана с правильным хешем
            if os.path.exists(fallback_dir):
                self.log(f"✅ Fallback папка создана: {fallback_dir}\n")
                self.log(f"🔐 Хеш папки: {forced_hash}\n")
            else:
                self.log(f"❌ ОШИБКА: Fallback папка не создана!\n")

        t0 = time.time()
        started = False
        last_len = 0
        last_out_len = 0

        # Stream logs to file as they grow
        stream_logs_path = None
        while not self._stop_event.is_set():
            try:
                pred.reload()
                logs = pred.logs or ""
            except Exception as e:
                # Обрабатываем ошибки при обновлении статуса (таймауты, сетевые ошибки)
                error_msg = f"\n⚠️ Ошибка обновления статуса: {type(e).__name__}: {e}\n"
                self.log(error_msg)
                
                # Если это таймаут, ждем немного и продолжаем
                if "timeout" in str(e).lower() or "readtimeout" in str(e).lower():
                    self.log("⏳ Таймаут API, ждем и продолжаем...\n")
                    time.sleep(self.poll_s)
                    continue
                
                # Для других ошибок логируем и продолжаем
                self.log("🔄 Продолжаем работу после ошибки...\n")
                time.sleep(self.poll_s)
                continue
            if len(logs) > last_len:
                chunk = logs[last_len:]
                self.log(chunk)
                # append to file inside current run dir
                try:
                    if stream_logs_path is None and self._current_run_dir:
                        stream_logs_path = os.path.join(self._current_run_dir, "logs.txt")
                        # write header if file is new
                        if not os.path.exists(stream_logs_path):
                            with open(stream_logs_path, "w", encoding="utf-8") as f:
                                f.write(f"=== GENERATION START: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    if stream_logs_path:
                        with open(stream_logs_path, "a", encoding="utf-8") as f:
                            f.write(chunk)
                except Exception as e:
                    self.log(f"⚠️ Ошибка сохранения логов в файл: {type(e).__name__}: {e}\n")
                last_len = len(logs)

            # Stream outputs (preview/final/colormap) as they appear
            try:
                outs = pred.output
                if isinstance(outs, list) and len(outs) > last_out_len:
                    import requests  # type: ignore
                    for idx in range(last_out_len, len(outs)):
                        url = outs[idx]
                        try:
                            self.log(f"🔄 Загрузка изображения {idx}: {url}\n")
                            r = requests.get(url, timeout=60)
                            r.raise_for_status()
                            name = f"out_{idx}"
                            path = os.path.join(self._current_run_dir or os.getcwd(), f"{name}.png")
                            with open(path, "wb") as f:
                                f.write(r.content)
                            self.log(f"✅ Изображение сохранено: {path}\n")
                            try:
                                self.ui_queue.put_nowait(f"__IMG__::{name}::{path}")
                            except Exception as e:
                                self.log(f"⚠️ Ошибка отправки изображения в UI: {e}\n")
                        except Exception as e:
                            self.log(f"⚠️ Ошибка загрузки изображения {idx}: {type(e).__name__}: {e}\n")
                            continue
                    last_out_len = len(outs)
            except Exception as e:
                self.log(f"⚠️ Ошибка при обработке выходных данных: {type(e).__name__}: {e}\n")

            if not started:
                if self._is_boot_error(logs) and not self._is_boot_completed(logs):
                    self.log("\n❌ Boot error detected before startup completed. Cancelling prediction...\n")
                    try:
                        pred.cancel()
                    except Exception as e:
                        self.log(f"⚠️ Ошибка при отмене предсказания: {e}\n")
                    self._status = "failed"
                    return 2
                if pred.status in ("processing", "succeeded") or self._is_boot_completed(logs):
                    started = True
                    self.log("\n✅ Startup phase completed. Proceeding to completion...\n")
                    self._status = "processing"
                elif time.time() - t0 > self.startup_timeout_s:
                    self.log("\n❌ Startup timeout exceeded. Cancelling prediction...\n")
                    try:
                        pred.cancel()
                    except Exception as e:
                        self.log(f"⚠️ Ошибка при отмене предсказания: {e}\n")
                    self._status = "timeout"
                    return 3

            if pred.status in ("succeeded", "failed", "canceled"):
                self.log(f"\n=== Final status: {pred.status}\n")
                if pred.status == "succeeded":
                    try:
                        self.log(json.dumps(pred.output, ensure_ascii=False, indent=2) + "\n")
                    except Exception:
                        self.log(str(pred.output) + "\n")
                    self._status = "succeeded"
                    self._save_outputs(pred, logs)
                    return 0
                self._status = pred.status
                self._save_outputs(pred, logs)
                return 1

            if time.time() - t0 > self.total_timeout_s:
                self.log("\n❌ Total timeout exceeded. Cancelling prediction...\n")
                try:
                    pred.cancel()
                except Exception as e:
                    self.log(f"⚠️ Ошибка при отмене предсказания: {e}\n")
                self._status = "timeout"
                return 4

            time.sleep(self.poll_s)

        # Stop requested
        try:
            pred.cancel()
        except Exception as e:
            self.log(f"⚠️ Ошибка при отмене предсказания: {e}\n")
        self._status = "canceled"
        return 10

    def _save_outputs(self, pred, logs: str) -> None:
        self.log("🔄 _save_outputs вызван\n")
        
        # Используем папку, созданную в _run_single
        if not self._current_run_dir:
            self.log("❌ ОШИБКА: _current_run_dir не установлен!\n")
            return
            
        run_dir = self._current_run_dir
        
        # Логируем информацию о папке
        self.log(f"💾 Сохраняем результаты в папку: {run_dir}\n")
        self.log(f"🔐 ПРИНУДИТЕЛЬНО используем хеш: 89fdd926\n")
        self.log(f"✅ Используем подготовленную папку: {self._current_run_dir}\n")
        # Save logs
        try:
            path = os.path.join(run_dir, "logs.txt")
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"=== GENERATION START: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            with open(path, "a", encoding="utf-8") as f:
                f.write(pred.logs or "")
                f.write(f"\n=== GENERATION FINISH: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        except Exception:
            pass
        # Save compact JSON (prediction + inputs + env + perf) → prediction.json
        try:
            env = {
                "torch": __import__("torch").__version__ if True else None,
                "diffusers": __import__("diffusers").__version__ if True else None,
                "cuda": __import__("torch").version.cuda if hasattr(__import__("torch"), "version") else None,
            }
            perf = {
                "ui_start_time": self._start_time,
                "ui_finish_time": time.time(),
            }
            pred_dict = {
                "id": getattr(pred, "id", None),
                "status": getattr(pred, "status", None),
                "model": getattr(pred, "model", None),
                "version": getattr(pred, "version", None),
                "input": getattr(pred, "input", None),
                "output": getattr(pred, "output", None),
                "error": getattr(pred, "error", None),
                "logs": None,  # логи отдельно в logs.txt
                "env": env,
                "perf": perf,
                "created_at": getattr(pred, "created_at", None),
                "started_at": getattr(pred, "started_at", None),
                "completed_at": getattr(pred, "completed_at", None),
            }
            with open(os.path.join(run_dir, "prediction.json"), "w", encoding="utf-8") as f:
                json.dump(pred_dict, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        # Save startup logs to a separate file (if present in main logs)
        try:
            startup_log_path = os.path.join(run_dir, "startup.log.txt")
            full_logs = pred.logs or ""
            lines = full_logs.splitlines()
            
            # Ищем блок STARTUP_SNAPSHOT
            startup_lines = []
            in_snapshot = False
            for line in lines:
                if "🚀 STARTUP_SNAPSHOT_START" in line:
                    in_snapshot = True
                    startup_lines.append(line)
                elif "🚀 STARTUP_SNAPSHOT_END" in line:
                    startup_lines.append(line)
                    in_snapshot = False
                elif in_snapshot:
                    startup_lines.append(line)
            
            # Если блок не найден, используем старый метод
            if not startup_lines:
                startup_lines = [ln for ln in lines if "MODEL_START" in ln or "setup" in ln.lower()]
            
            with open(startup_log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(startup_lines) + ("\n" if startup_lines else ""))
        except Exception:
            pass
        # Download outputs (preview/final/colormap) if present
        try:
            import requests  # type: ignore
            outs = pred.output
            if isinstance(outs, list):
                self.log(f"🔄 Начинаем обработку {len(outs)} выходных файлов...\n")
                self.log(f"📁 Папка для сохранения: {run_dir}\n")
                
                files = []
                for idx, url in enumerate(outs):
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        name = f"out_{idx}"
                        fp = os.path.join(run_dir, f"{name}.png")
                        with open(fp, "wb") as f:
                            f.write(r.content)
                        files.append(fp)
                        self.log(f"📥 Скачан файл {idx}: {name}.png\n")
                    except Exception as e:
                        self.log(f"⚠️ Ошибка скачивания файла {idx}: {e}\n")
                        continue
                        
                # Упрощенный и надежный маппинг: всегда используем позиционный маппинг
                # Маркеры могут указывать на /tmp/ пути, которые не совпадают с out_ файлами
                self.log(f"📁 Найдено файлов для переименования: {len(files)}\n")
                
                # Создаем детальные названия файлов (ВАРИАНТ Б)
                prompt = ""
                if hasattr(pred, 'input') and isinstance(pred.input, dict):
                    prompt = str(pred.input.get('prompt', '')).replace(' ', '_').replace('%', 'pct')[:50]
                    if not prompt:
                        prompt = "unknown"
                else:
                    prompt = "unknown"
                
                self.log(f"📝 Промпт для именования: '{prompt}'\n")
                
                # Получаем временную метку из названия папки
                timestamp = ""
                if hasattr(self, '_current_run_dir') and self._current_run_dir:
                    folder_name = os.path.basename(self._current_run_dir)
                    if folder_name.startswith('run_'):
                        timestamp = folder_name[4:]  # Убираем 'run_' из начала
                        self.log(f"⏰ Временная метка из папки: {timestamp}\n")
                    else:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        self.log(f"⏰ Временная метка сгенерирована: {timestamp}\n")
                else:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    self.log(f"⏰ Временная метка по умолчанию: {timestamp}\n")
                
                # Используем версию и хеш, переданные из App
                if hasattr(self, 'current_version') and hasattr(self, 'current_hash'):
                    forced_hash = self.current_hash
                    self.log(f"🔐 Используем хеш из App: {forced_hash}\n")
                else:
                    # Fallback: определяем хеш автоматически
                    if hasattr(self, '_version_id') and self._version_id:
                        ver_for_hash = self._version_id
                    else:
                        ver_for_hash = self._determine_version_fallback()
                    
                    forced_hash = self.get_optimal_hash(ver_for_hash)
                    if not forced_hash or forced_hash == "unknown":
                        import hashlib
                        hash_obj = hashlib.md5(ver_for_hash.encode())
                        forced_hash = hash_obj.hexdigest()[:8]
                    
                    self.log(f"🔐 Автоматически определенный хеш версии: {forced_hash}\n")
                
                if len(files) >= 1:
                    try:
                        # Формат: {timestamp}_{prompt}_{type}_{hash}.png
                        final_name = f"{timestamp}_{prompt}_final_{forced_hash}.png"
                        final_path = os.path.join(run_dir, final_name)
                        os.replace(files[0], final_path)
                        self.log(f"✅ FINAL saved as: {final_name}\n")
                        self.log(f"📁 Путь: {final_path}\n")
                    except Exception as e:
                        self.log(f"⚠️ Ошибка маппинга final.png: {e}\n")
                if len(files) >= 2:
                    try:
                        # Формат: {timestamp}_{prompt}_{type}_{hash}.png
                        colormap_name = f"{timestamp}_{prompt}_colormap_{forced_hash}.png"
                        colormap_path = os.path.join(run_dir, colormap_name)
                        os.replace(files[1], colormap_path)
                        self.log(f"🎨 COLORMAP saved as: {colormap_name}\n")
                        self.log(f"📁 Путь: {colormap_path}\n")
                    except Exception as e:
                        self.log(f"⚠️ Ошибка маппинга colormap.png: {e}\n")
        except Exception:
            pass

    def _update_image_preview(self, kind: str, path: str) -> None:
        """Обновляет превью изображений (без PIL)"""
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'preview_label'):
            return
            
        try:
            # Показываем детальное название файла (ВАРИАНТ Б)
            filename = os.path.basename(path)
            if kind == "preview":
                self.preview_label.configure(text=f"Preview: {filename}")
            elif kind == "final":
                self.final_label.configure(text=f"Final: {filename}")
            elif kind == "colormap":
                self.colormap_label.configure(text=f"Colormap: {filename}")
                
            # Дополнительное логирование для диагностики
            self.log(f"🖼️ Обновлено превью {kind}: {filename}\n")
        except Exception as e:
            self.log(f"⚠️ Ошибка обновления превью {kind}: {e}\n")

    def run_presets(self, preset_items: List[Dict[str, Any]]):
        self.log("🔄 run_presets() вызван\n")
        try:
            import replicate  # type: ignore
        except ImportError:
            self.log("❌ Модуль 'replicate' не установлен.\n")
            self.log("💡 Установите: pip install replicate\n")
            return
        except Exception as e:
            self.log(f"❌ Ошибка импорта replicate: {e}\n")
            return

        client = replicate.Client(api_token=self.api_token)

        for item in preset_items:
            if self._stop_event.is_set():
                break
            name = item.get("name", "preset")
            inputs = item.get("inputs", {})
            self.log(f"\n################ PRESET: {name} ################\n")
            self._run_single(client, inputs)

        self.log("\n[GUI] Done.\n")

    def start(self, preset_items: List[Dict[str, Any]]):
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_presets, args=(preset_items,), daemon=True)
        self._thread.start()
        self.log("🔄 Worker.start() вызван, поток запущен\n")


class App:
    def __init__(self, root: tk.Tk):
        print("🔄 Инициализация App...")
        self.root = root
        root.title("Replicate Test Runner - версия определяется автоматически")
        root.geometry("1600x1000")  # Делаем окно еще шире и выше для лучшего отображения
        print("✅ Основные параметры окна установлены")

        self.ui_queue: "queue.Queue[str]" = queue.Queue()
        self.controls_locked: bool = False
        
        # Текущая версия (инициализируем автоматически)
        self.current_version = "v4.5.01"  # Принудительно используем v4.5.01
        self.current_hash = "unknown"     # Хеш будет определен автоматически
        
        # Автоматически обновляем версию и хеш при запуске (после создания всех UI элементов)

        # Top frame controls
        top = ttk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text="Model:").grid(row=0, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value="nauslava/plitka-pro-project")
        self.model_entry = ttk.Entry(top, textvariable=self.model_var, width=60)
        self.model_entry.grid(row=0, column=1, sticky=tk.W, padx=6)

        ttk.Label(top, text="Version ID:").grid(row=0, column=2, sticky=tk.W)
        self.version_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.version_var, width=80)
        self.version_entry = ttk.Entry(top, textvariable=self.version_var, width=80)
        self.version_entry.grid(row=0, column=3, columnspan=2, sticky=tk.W, padx=6)

        ttk.Label(top, text="Poll (s):").grid(row=1, column=0, sticky=tk.W)
        self.poll_var = tk.IntVar(value=6)
        ttk.Entry(top, textvariable=self.poll_var, width=6).grid(row=1, column=1, sticky=tk.W, padx=6)

        ttk.Label(top, text="Startup (s):").grid(row=1, column=2, sticky=tk.W)
        self.startup_var = tk.IntVar(value=7 * 60)
        ttk.Entry(top, textvariable=self.startup_var, width=8).grid(row=1, column=3, sticky=tk.W, padx=6)

        ttk.Label(top, text="Total (s):").grid(row=1, column=4, sticky=tk.W)
        self.total_var = tk.IntVar(value=25 * 60)
        ttk.Entry(top, textvariable=self.total_var, width=8).grid(row=1, column=5, sticky=tk.W, padx=6)
        
        # Информация о текущей версии
        ttk.Label(top, text="Current Version:").grid(row=2, column=0, sticky=tk.W)
        self.version_label = ttk.Label(top, text=self.current_version, font=("Arial", 10, "bold"), foreground="blue")
        self.version_label.grid(row=2, column=1, sticky=tk.W, padx=6)
        
        ttk.Label(top, text="Hash:").grid(row=2, column=2, sticky=tk.W)
        self.hash_label = ttk.Label(top, text=self.current_hash, font=("Arial", 10, "bold"), foreground="green")
        self.hash_label.grid(row=2, column=3, sticky=tk.W, padx=6)
        
        # Кнопка обновления версии
        refresh_btn = ttk.Button(top, text="🔄 Refresh Version", command=self.update_version_info)
        refresh_btn.grid(row=2, column=4, sticky=tk.W, padx=6)
        
        # Кнопка демо-режима (если replicate недоступен)
        if not REPLICATE_AVAILABLE:
            demo_btn = ttk.Button(top, text="🎭 Demo Mode", command=self.run_demo)
            demo_btn.grid(row=2, column=5, sticky=tk.W, padx=6)

        # Presets
        mid = ttk.Frame(root)
        mid.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(mid)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Presets").pack(anchor=tk.W)
        self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED, height=15, width=60, font=("Arial", 9))
        self.listbox.pack(fill=tk.Y, expand=False)
        
        # Область отображения параметров выбранного пресета
        ttk.Label(left, text="Preset Parameters", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(8,2))
        self.preset_params_text = tk.Text(left, height=30, width=60, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 8))
        self.preset_params_text.pack(fill=tk.X, expand=False, pady=(0,6))

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=6)
        self.start_btn = ttk.Button(btns, text="Start", command=self.on_start)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.on_stop)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self.close_btn = ttk.Button(btns, text="Close", command=self.on_close)
        self.close_btn.pack(side=tk.LEFT, padx=4)

        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Status + timer
        status_row = ttk.Frame(right)
        status_row.pack(fill=tk.X)
        ttk.Label(status_row, text="Status:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="idle")
        self.status_label = ttk.Label(status_row, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=6)
        ttk.Label(status_row, text="Elapsed:").pack(side=tk.LEFT, padx=12)
        self.elapsed_var = tk.StringVar(value="00:00")
        self.elapsed_label = ttk.Label(status_row, textvariable=self.elapsed_var)
        self.elapsed_label.pack(side=tk.LEFT)

        # Preview area
        img_frame = ttk.Frame(right)
        img_frame.pack(fill=tk.X, pady=(6,4))
        self.preview_label = ttk.Label(img_frame, text="Preview")
        self.preview_label.pack(side=tk.LEFT, padx=6)
        self.final_label = ttk.Label(img_frame, text="Final")
        self.final_label.pack(side=tk.LEFT, padx=6)
        self.colormap_label = ttk.Label(img_frame, text="Colormap")
        self.colormap_label.pack(side=tk.LEFT, padx=6)

        ttk.Label(right, text="Logs").pack(anchor=tk.W, pady=(6,0))
        self.text = tk.Text(right, wrap=tk.WORD, state=tk.DISABLED)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Проверяем существование папки replicate_runs
        current_dir = os.getcwd()
        replicate_runs_dir = os.path.join(current_dir, "replicate_runs")
        if not os.path.exists(replicate_runs_dir):
            os.makedirs(replicate_runs_dir, exist_ok=True)
            self.append_log(f"📁 Создана папка replicate_runs: {replicate_runs_dir}\n")
        
        # Автоматически определяем правильный хеш для версии
        # Используем уже определенный хеш вместо повторного вычисления
        forced_hash = self.current_hash
        
        hash_dir = os.path.join(replicate_runs_dir, forced_hash)
        if not os.path.exists(hash_dir):
            os.makedirs(hash_dir, exist_ok=True)
            self.append_log(f"📁 Создана папка версии с хешем: {forced_hash}\n")
            self.append_log(f"📁 Абсолютный путь: {os.path.abspath(hash_dir)}\n")
        
        # Load presets based on current version
        self.presets_path = self._get_presets_path_for_version()
        try:
            self.presets = read_presets(self.presets_path)
            self.append_log(f"📁 Загружены пресеты из: {os.path.basename(self.presets_path)}\n")
        except Exception as e:
            self.presets = {}
            self.append_log(f"❌ Ошибка загрузки пресетов: {e}\n")

        # Валидируем все пресеты при загрузке
        self.append_log("🔍 Валидация пресетов...\n")
        valid_presets = 0
        total_presets = len(self.presets)
        
        for name, preset_data in self.presets.items():
            is_valid, errors = self.validate_preset(name, preset_data, self.current_version)
            if is_valid:
                valid_presets += 1
                self.listbox.insert(tk.END, name)
            else:
                # Добавляем проблемный пресет с пометкой
                self.listbox.insert(tk.END, f"❌ {name}")
                self.append_log(f"⚠️ Пресет '{name}' содержит ошибки:\n")
                for error in errors:
                    self.append_log(f"  {error}\n")
        
        self.append_log(f"📊 Результат валидации: {valid_presets}/{total_presets} пресетов корректны\n")
        
        # Создаем временную папку для логов при запуске (ДО первого append_log)
        self._create_temp_logs_dir()
        
        # Привязываем событие выбора пресета
        self.listbox.bind('<<ListboxSelect>>', self.on_preset_select)
        
        # Создаем файл логов для GUI в единой папке gui_logs
        # (будет создан после определения версии в update_version_info)
        
        # Отображаем информацию о текущей версии
        current_dir = os.getcwd()
        self.append_log(f"🚀 Plitka Pro {self.current_version} запущен\n")
        self.append_log(f"🔐 Хеш папки: {self.current_hash}\n")
        self.append_log(f"📁 Ожидаемая структура: replicate_runs/{self.current_hash}/run_YYYYMMDD_HHMMSS/\n")
        self.append_log(f"🔍 Рабочий каталог: {current_dir}\n")
        self.append_log("=" * 60 + "\n")
        
        # Обновляем заголовок окна с версией
        self.root.title(f"Replicate Test Runner - {self.current_version} ({self.current_hash})")

        # Worker
        token = load_env_token()
        if not token:
            self.append_log("WARNING: REPLICATE_API_TOKEN is not set. Enter it below or export it / put into .env.\n")

        # Token entry
        token_frame = ttk.Frame(root)
        token_frame.pack(fill=tk.X, padx=8)
        ttk.Label(token_frame, text="API Token:").pack(side=tk.LEFT)
        self.token_var = tk.StringVar(value=token or "")
        self.token_entry = ttk.Entry(token_frame, textvariable=self.token_var, width=60, show="*")
        self.token_entry.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        # Создаем worker
        self.worker = ReplicateWorker(
            api_token=token or "",
            model_ref=self.model_var.get(),
            poll_s=self.poll_var.get(),
            startup_timeout_s=self.startup_var.get(),
            total_timeout_s=self.total_var.get(),
            ui_queue=self.ui_queue,
            version_id=self.version_var.get().strip() or None,
        )
        
        def save_token():
            # Проверяем, что UI элементы созданы
            if not hasattr(self, 'text'):
                return
                
            val = self.token_var.get().strip()
            if not val:
                messagebox.showwarning("Token", "Token is empty")
                return
            # Update env for this process
            os.environ["REPLICATE_API_TOKEN"] = val
            self.worker.api_token = val
            # Persist to .env
            try:
                env_path = os.path.join(os.getcwd(), ".env")
                lines = []
                if os.path.exists(env_path):
                    with open(env_path, "r", encoding="utf-8") as f:
                        lines = f.read().splitlines()
                    lines = [ln for ln in lines if not ln.startswith("REPLICATE_API_TOKEN=")]
                lines.append(f"REPLICATE_API_TOKEN={val}")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                self.append_log("[GUI] Token saved to .env\n")
            except Exception as e:
                self.append_log(f"[GUI] Failed to save token: {e}\n")
        ttk.Button(token_frame, text="Save", command=save_token).pack(side=tk.LEFT, padx=6)
        
        # Auto-bind token field changes to worker (без сохранения в .env)
        self.token_var.trace_add("write", lambda *_: setattr(self.worker, "api_token", self.token_var.get().strip()))

        # Periodic UI updates
        self.root.after(300, self.drain_queue)
        self.root.after(1000, self.update_timer)
        # Update worker config when fields change
        self.model_var.trace_add("write", lambda *_: setattr(self.worker, "model_ref", self.model_var.get()))
        self.version_var.trace_add("write", lambda *_: setattr(self.worker, "_version_id", (self.version_var.get().strip() or None)))
        self.poll_var.trace_add("write", lambda *_: setattr(self.worker, "poll_s", int(self.poll_var.get())))
        self.startup_var.trace_add("write", lambda *_: setattr(self.worker, "startup_timeout_s", int(self.startup_var.get())))
        self.total_var.trace_add("write", lambda *_: setattr(self.worker, "total_timeout_s", int(self.total_var.get())))
        
        # Передаем версию в worker после её определения
        def update_worker_version():
            if hasattr(self, 'current_version') and hasattr(self, 'current_hash'):
                self.worker.current_version = self.current_version
                self.worker.current_hash = self.current_hash

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Автоматически обновляем версию и хеш при запуске (после создания всех UI элементов)
        print("🔄 Вызов update_version_info...")
        self.update_version_info()
        print("✅ App полностью инициализирован")

    def append_log(self, text: str) -> None:
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)
        
        # Немедленно сохраняем логи в файл
        self._save_logs_to_file(text)
    
    def _save_logs_to_file(self, text: str) -> None:
        """Сохраняет логи в файл для текущего запуска"""
        try:
            # Приоритет 1: папка GUI логов (единая папка для всей сессии)
            if hasattr(self, '_gui_logs_dir') and self._gui_logs_dir:
                # Сохраняем в файл сессии (дополняется)
                session_log_file = os.path.join(self._gui_logs_dir, "gui_session.log.txt")
                log_file = session_log_file
            # Приоритет 2: папка текущего теста
            elif hasattr(self, '_current_run_dir') and self._current_run_dir:
                log_file = os.path.join(self._current_run_dir, "gui_logs.txt")
            # Приоритет 3: папка worker'а
            elif hasattr(self, 'worker') and self.worker and hasattr(self.worker, '_current_run_dir') and self.worker._current_run_dir:
                log_file = os.path.join(self.worker._current_run_dir, "gui_logs.txt")
            # Приоритет 4: временная папка логов
            elif hasattr(self, '_temp_logs_dir') and self._temp_logs_dir:
                log_file = os.path.join(self._temp_logs_dir, "temp_session.log.txt")
            # Приоритет 5: fallback на папку GUI логов
            else:
                current_dir = os.getcwd()
                # Используем уже определенный хеш вместо повторного вычисления
                if hasattr(self, 'current_hash'):
                    log_file = os.path.join(current_dir, "replicate_runs", self.current_hash, "gui_logs", "gui_session.log.txt")
                    # Создаем папку GUI логов если её нет
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                else:
                    # Если хеш не определен, используем временную папку
                    temp_logs_dir = os.path.join(current_dir, "replicate_runs", "temp_logs")
                    os.makedirs(temp_logs_dir, exist_ok=True)
                    log_file = os.path.join(temp_logs_dir, "temp_session.log.txt")
            
            # Добавляем временную метку
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {text}"
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
                
        except Exception as e:
            # Игнорируем ошибки сохранения логов
            pass
    
    def _create_log_file(self) -> None:
        """Логирует информацию о создании папки (папка создается в worker'е)"""
        try:
            # Папка будет создана в worker'е, здесь только логируем
            self.append_log(f"📝 Папка для теста будет создана в worker'е\n")
            self.append_log(f"🔐 Автоматически определенный хеш: {self.current_hash}\n")
            
        except Exception as e:
            self.append_log(f"⚠️ Ошибка логирования: {e}\n")
    
    def _create_temp_logs_dir(self) -> None:
        """Создает временную папку для логов при запуске"""
        try:
            current_dir = os.getcwd()
            temp_logs_dir = os.path.join(current_dir, "replicate_runs", "temp_logs")
            os.makedirs(temp_logs_dir, exist_ok=True)
            
            # Создаем временный файл сессии
            temp_session_file = os.path.join(temp_logs_dir, "temp_session.log.txt")
            
            # Записываем заголовок
            with open(temp_session_file, "w", encoding="utf-8") as f:
                f.write(f"=== TEMP GUI SESSION LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"🚀 Версия: определяем...\n")
                f.write(f"🔐 Хеш: определяем...\n")
                f.write(f"📁 Временная папка: {temp_logs_dir}\n")
                f.write("=" * 60 + "\n\n")
            
            # Сохраняем путь к временной папке
            self._temp_logs_dir = temp_logs_dir
            
        except Exception as e:
            # Игнорируем ошибки создания временной папки
            pass
    
    def _create_gui_log_file(self) -> None:
        """Создает файл логов для GUI в единой папке"""
        try:
            current_dir = os.getcwd()
            
            # Используем уже определенные переменные версии и хеша
            if not hasattr(self, 'current_version') or not hasattr(self, 'current_hash'):
                return  # Не создаем логи до определения версии
            
            # Используем current_hash вместо повторного вычисления
            gui_logs_dir = os.path.join(current_dir, "replicate_runs", self.current_hash, "gui_logs")
            os.makedirs(gui_logs_dir, exist_ok=True)
            
            # Создаем файл логов запуска (перезаписывается)
            startup_log_file = os.path.join(gui_logs_dir, "gui_startup.log.txt")
            
            # Записываем заголовок
            with open(startup_log_file, "w", encoding="utf-8") as f:
                f.write(f"=== GUI STARTUP LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"🚀 Версия: {self.current_version}\n")
                f.write(f"🔐 Хеш: {self.current_hash}\n")
                f.write(f"📁 Папка: {gui_logs_dir}\n")
                f.write("=" * 60 + "\n\n")
            
            # Создаем файл сессии (дополняется)
            session_log_file = os.path.join(gui_logs_dir, "gui_session.log.txt")
            
            # Записываем заголовок сессии
            with open(session_log_file, "w", encoding="utf-8") as f:
                f.write(f"=== GUI SESSION LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"🚀 Версия: {self.current_version}\n")
                f.write(f"🔐 Хеш: {self.current_hash}\n")
                f.write(f"📁 Папка: {gui_logs_dir}\n")
                f.write("=" * 60 + "\n\n")
            
            # Сохраняем путь к папке GUI логов
            self._gui_logs_dir = gui_logs_dir
            
            # Перемещаем логи из временной папки в правильную
            if hasattr(self, '_temp_logs_dir') and self._temp_logs_dir:
                try:
                    temp_session_file = os.path.join(self._temp_logs_dir, "temp_session.log.txt")
                    if os.path.exists(temp_session_file):
                        # Читаем содержимое временного файла
                        with open(temp_session_file, "r", encoding="utf-8") as f:
                            temp_content = f.read()
                        
                        # Добавляем содержимое в правильный файл сессии
                        with open(session_log_file, "a", encoding="utf-8") as f:
                            f.write("\n" + "=" * 60 + "\n")
                            f.write("ЛОГИ ДО ОПРЕДЕЛЕНИЯ ВЕРСИИ:\n")
                            f.write("=" * 60 + "\n")
                            f.write(temp_content)
                        
                        # Удаляем временную папку
                        import shutil
                        shutil.rmtree(self._temp_logs_dir)
                        self._temp_logs_dir = None
                        
                        self.append_log(f"✅ Логи перемещены из временной папки\n")
                except Exception as e:
                    self.append_log(f"⚠️ Ошибка перемещения логов: {e}\n")
            
            # Логируем созданную папку
            self.append_log(f"📁 Логи GUI: {gui_logs_dir}\n")
            
        except Exception as e:
            # Игнорируем ошибки создания файла логов
            pass
    
    def _create_stop_log_file(self) -> None:
        """Создает файл логов для остановки тестов в единой папке"""
        try:
            if hasattr(self, '_gui_logs_dir') and self._gui_logs_dir:
                # Используем папку GUI логов
                log_file = os.path.join(self._gui_logs_dir, "gui_stop.log.txt")
                
                # Записываем информацию об остановке
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== GUI STOP LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"🚀 Версия: {self.current_version}\n")
                    f.write(f"🔐 Хеш: {self.current_hash}\n")
                    f.write(f"🔐 Автоматически определенный хеш: {self.current_hash}\n")
                    f.write(f"📁 Папка: {self._gui_logs_dir}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("Тесты остановлены пользователем\n")
                
        except Exception as e:
            # Игнорируем ошибки создания файла логов
            pass
    
    def _create_close_log_file(self) -> None:
        """Создает файл логов для закрытия GUI в единой папке"""
        try:
            if hasattr(self, '_gui_logs_dir') and self._gui_logs_dir:
                # Используем папку GUI логов
                log_file = os.path.join(self._gui_logs_dir, "gui_close.log")
                
                # Записываем информацию о закрытии
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== GUI CLOSE LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"🚀 Версия: {self.current_version}\n")
                    f.write(f"🔐 Хеш: {self.current_hash}\n")
                    f.write(f"🔐 Автоматически определенный хеш: {self.current_hash}\n")
                    f.write(f"📁 Папка: {self._gui_logs_dir}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write("GUI закрыт пользователем\n")
                
        except Exception as e:
            # Игнорируем ошибки создания файла логов
            pass
    
    def _get_presets_path_for_version(self) -> str:
        """Возвращает путь к файлу пресетов в зависимости от версии модели"""
        try:
            # Определяем версию
            version = self._extract_version_from_cog_yaml()
            if not version:
                version = self._determine_version_fallback()
            
            # Принудительно устанавливаем v4.5.01 для неизвестных версий
            if not version or version == "unknown":
                version = "v4.5.01"
                self.append_log(f"⚠️ Версия не определена, принудительно используем {version}\n")
            
            # Маппинг версий к файлам пресетов
            version_mapping = {
                "v4.5.01": "test_inputs_v4.5.01_critical_fixes.json",  # Critical Architecture Fixes
                "v4.4.61": "test_inputs_v4.4.61_extended.json",  # Multimodal ControlNet + Production Ready
                "v4.4.60": "test_inputs_v4.4.60_extended.json",  # Extended testing
                "v4.4.59": "test_inputs_v4.4.59.json",  # Previous version
                "v4.4.58": "test_inputs_v4.4.58.json",  # Previous version
                "v4.4.56": "test_inputs_v4.4.56.json",  # Color Grid Adapter + ControlNet
                "v4.4.45": "test_inputs_v4.4.45.json",  # Улучшенная LoRA загрузка
                "v4.4.39": "test_inputs_v4.4.39.json",  # Базовая версия
            }
            
            # Если версия неизвестна или "unknown", принудительно используем v4.5.01
            if version not in version_mapping or version == "unknown":
                self.append_log(f"⚠️ Версия {version} не найдена или неизвестна, принудительно используем v4.5.01\n")
                version = "v4.5.01"
            
            # Получаем имя файла для версии
            filename = version_mapping.get(version, "test_inputs_v4.5.01_critical_fixes.json")
            presets_path = os.path.join(os.path.dirname(__file__), filename)
            
            # Проверяем существование файла
            if os.path.exists(presets_path):
                return presets_path
            else:
                # Fallback к файлу v4.5.01
                fallback_path = os.path.join(os.path.dirname(__file__), "test_inputs_v4.5.01_critical_fixes.json")
                if os.path.exists(fallback_path):
                    self.append_log(f"⚠️ Файл пресетов для версии {version} не найден, используем базовый\n")
                    return fallback_path
                else:
                    # Создаем пустой файл пресетов
                    empty_presets = {"default": {"prompt": "100% red", "seed": 12345, "steps": 20, "guidance": 7.0, "lora_scale": 0.7, "use_controlnet": false, "description": "Базовый пресет"}}
                    with open(fallback_path, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(empty_presets, f, indent=2, ensure_ascii=False)
                    self.append_log(f"📁 Создан базовый файл пресетов: {fallback_path}\n")
                    return fallback_path
                    
        except Exception as e:
            self.append_log(f"⚠️ Ошибка определения пути к пресетам: {e}\n")
            # Fallback к базовому файлу
            return os.path.join(os.path.dirname(__file__), "test_inputs_v4.4.39.json")
    
    def _reload_presets_for_version(self) -> None:
        """Перезагружает пресеты для текущей версии модели"""
        try:
            # Очищаем текущий список пресетов
            self.listbox.delete(0, tk.END)
            
            # Получаем новый путь к пресетам
            new_presets_path = self._get_presets_path_for_version()
            
            # Загружаем новые пресеты
            if new_presets_path != self.presets_path:
                self.presets_path = new_presets_path
                self.presets = read_presets(self.presets_path)
                self.append_log(f"🔄 Пресеты перезагружены для версии {self.current_version}\n")
                self.append_log(f"📁 Новый файл: {os.path.basename(self.presets_path)}\n")
            
            # Валидируем и добавляем пресеты
            self.append_log("🔍 Валидация пресетов для новой версии...\n")
            valid_presets = 0
            total_presets = len(self.presets)
            
            for name, preset_data in self.presets.items():
                # Используем текущую версию для валидации
                is_valid, errors = self.validate_preset(name, preset_data, self.current_version)
                if is_valid:
                    valid_presets += 1
                    self.listbox.insert(tk.END, f"✅ {name}")
                else:
                    # Определяем тип ошибок
                    critical_errors = [e for e in errors if e.startswith("❌")]
                    warnings = [e for e in errors if e.startswith("⚠️")]
                    
                    if critical_errors:
                        # Критические ошибки
                        self.listbox.insert(tk.END, f"❌ {name}")
                    else:
                        # Только предупреждения
                        self.listbox.insert(tk.END, f"⚠️ {name}")
                    
                    self.append_log(f"⚠️ Пресет '{name}' содержит ошибки:\n")
                    for error in errors:
                        self.append_log(f"  {error}\n")
            
            self.append_log(f"📊 Результат валидации: {valid_presets}/{total_presets} пресетов корректны\n")
            
            # Очищаем область параметров пресета
            if hasattr(self, 'preset_params_text'):
                self.preset_params_text.config(state=tk.NORMAL)
                self.preset_params_text.delete(1.0, tk.END)
                self.preset_params_text.config(state=tk.DISABLED)
                
        except Exception as e:
            self.append_log(f"❌ Ошибка перезагрузки пресетов: {e}\n")
    
    def _load_color_table(self) -> list[str]:
        """Загружает таблицу цветов из файла colors_table.txt"""
        try:
            color_file = os.path.join(os.path.dirname(__file__), "..", "colors_table.txt")
            if not os.path.exists(color_file):
                # Fallback к стандартным цветам
                return ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PINK", "WHITE", "BLACK", "GRAY", "BROWN"]
            
            with open(color_file, "r", encoding="utf-8") as f:
                colors = [line.strip().upper() for line in f if line.strip()]
            return colors
        except Exception as e:
            print(f"⚠️ Ошибка загрузки таблицы цветов: {e}")
            # Fallback к стандартным цветам
            return ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "PINK", "WHITE", "BLACK", "GRAY", "BROWN"]
    
    def validate_preset(self, preset_name: str, preset_data: dict, version: str = None) -> tuple[bool, list[str]]:
        """Валидирует пресет на соответствие требованиям"""
        errors = []
        
        # Загружаем таблицу цветов
        valid_colors = self._load_color_table()
        
        # Определяем версию для валидации
        if version is None:
            version = getattr(self, 'current_version', 'v4.5.01')
        
        try:
            # Проверяем наличие prompt
            if 'prompt' not in preset_data:
                errors.append("❌ Отсутствует поле 'prompt'")
                return False, errors
            
            prompt = preset_data['prompt']
            
            # Парсим цвета и их соотношения (учитываем токены <s0><s1>)
            # Убираем токены и триггеры для парсинга цветов
            clean_prompt = prompt.replace('ohwx_rubber_tile', '').replace('<s0><s1>', '').strip()
            # Убираем общие слова, которые не являются цветами
            clean_prompt = clean_prompt.replace('rubber tile', '').replace('high quality', '').replace('realistic texture', '').replace('grid pattern', '').replace('random pattern', '').replace('radial pattern', '').replace('granular pattern', '').strip()
            color_parts = [p.strip() for p in clean_prompt.split(',') if p.strip() and '%' in p]
            colors = []
            total_percentage = 0
            
            for part in color_parts:
                try:
                    if '%' in part:
                        percent_str, color_name = part.split('%', 1)
                        percent = float(percent_str.strip())
                        color_name = color_name.strip()
                        
                        if percent < 0 or percent > 100:
                            errors.append(f"❌ Неверный процент для {color_name}: {percent}%")
                            continue
                        
                        # Проверяем, что цвет есть в таблице
                        if color_name.upper() not in valid_colors:
                            errors.append(f"❌ Неизвестный цвет '{color_name}'. Доступные цвета: {', '.join(valid_colors[:10])}...")
                            continue
                        
                        colors.append({"name": color_name, "percentage": percent})
                        total_percentage += percent
                    else:
                        # Если нет процентов, считаем как 100%
                        color_name = part.strip()
                        if color_name.upper() not in valid_colors:
                            errors.append(f"❌ Неизвестный цвет '{color_name}'. Доступные цвета: {', '.join(valid_colors[:10])}...")
                            continue
                        colors.append({"name": color_name, "percentage": 100})
                        total_percentage = 100
                        break
                except Exception as e:
                    errors.append(f"❌ Ошибка парсинга '{part}': {e}")
            
            # Проверяем количество цветов (более мягкая проверка)
            if len(colors) > 5:
                errors.append(f"❌ Слишком много цветов: {len(colors)} (максимум: 5)")
            elif len(colors) > 4:
                errors.append(f"⚠️ Много цветов: {len(colors)} (рекомендуется: до 4)")
            
            # Проверяем сумму процентов
            if total_percentage != 100:
                errors.append(f"❌ Сумма процентов не равна 100%: {total_percentage}%")
            
            # Проверяем другие обязательные поля (адаптивно для разных версий)
            # Для v4.5.01 и новее используем новые поля
            if compare_versions(version, "v4.5.01") >= 0:
                required_fields = ['seed', 'num_inference_steps', 'guidance_scale']
                optional_fields = ['colormap', 'granule_size', 'negative_prompt']
            else:
                # Для старых версий используем старые поля
                required_fields = ['seed', 'steps', 'guidance', 'lora_scale']
                optional_fields = ['use_controlnet', 'description']
            
            for field in required_fields:
                if field not in preset_data:
                    errors.append(f"❌ Отсутствует обязательное поле '{field}'")
            
            # Проверяем опциональные поля
            for field in optional_fields:
                if field not in preset_data:
                    errors.append(f"⚠️ Отсутствует опциональное поле '{field}'")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"❌ Критическая ошибка валидации: {e}")
            return False, errors
    
    def on_preset_select(self, event) -> None:
        """Обрабатывает выбор пресета и отображает его параметры"""
        try:
            # Получаем выбранные пресеты
            selection = self.listbox.curselection()
            if not selection:
                return
            
            # Обрабатываем множественный выбор
            if len(selection) == 1:
                # Один пресет выбран
                selected_index = selection[0]
                preset_name = self.listbox.get(selected_index)
                self._display_single_preset(preset_name)
            else:
                # Несколько пресетов выбрано
                self._display_multiple_presets(selection)
                
        except Exception as e:
            self.append_log(f"⚠️ Ошибка отображения параметров пресета: {e}\n")
    
    def _display_single_preset(self, preset_name: str) -> None:
        """Отображает параметры одного пресета"""
        try:
            # Убираем префиксы ✅, ❌, ⚠️ если есть
            clean_name = preset_name.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
            
            # Получаем параметры пресета
            if clean_name in self.presets:
                preset_data = self.presets[clean_name]
                
                # Валидируем пресет
                is_valid, validation_errors = self.validate_preset(clean_name, preset_data, self.current_version)
                
                # Форматируем параметры для отображения
                params_text = f"📋 Preset: {clean_name}\n"
                params_text += "=" * 50 + "\n\n"
                
                # Статус валидации
                if is_valid:
                    params_text += "✅ Валидация: ПРЕСЕТ КОРРЕКТЕН\n\n"
                else:
                    params_text += "❌ Валидация: ОШИБКИ ОБНАРУЖЕНЫ\n"
                    for error in validation_errors:
                        params_text += f"  {error}\n"
                    params_text += "\n"
                
                # Отображаем ВСЕ поля из пресета
                params_text += "🔍 ВСЕ ДАННЫЕ ПРЕСЕТА:\n"
                params_text += "-" * 30 + "\n"
                
                for key, value in preset_data.items():
                    if isinstance(value, str) and len(value) > 60:
                        # Обрезаем длинные строки
                        display_value = value[:57] + "..."
                    else:
                        display_value = str(value)
                    
                    # Форматируем ключ для лучшей читаемости
                    formatted_key = key.replace('_', ' ').title()
                    params_text += f"• {formatted_key}: {display_value}\n"
                
                # Дополнительная информация о цветах
                if 'prompt' in preset_data:
                    prompt = preset_data['prompt']
                    params_text += f"\n🎨 АНАЛИЗ PROMPT:\n"
                    params_text += "-" * 20 + "\n"
                    
                    # Показываем доступные цвета
                    valid_colors = self._load_color_table()
                    params_text += f"• Доступные цвета: {', '.join(valid_colors[:15])}...\n"
                    
                    try:
                        # Убираем токены и триггеры для парсинга цветов
                        clean_prompt = prompt.replace('ohwx_rubber_tile', '').replace('<s0><s1>', '').strip()
                        # Убираем общие слова, которые не являются цветами
                        clean_prompt = clean_prompt.replace('rubber tile', '').replace('high quality', '').replace('realistic texture', '').replace('grid pattern', '').replace('random pattern', '').replace('radial pattern', '').replace('granular pattern', '').strip()
                        color_parts = [p.strip() for p in clean_prompt.split(',') if p.strip() and '%' in p]
                        colors = []
                        total_percentage = 0
                        
                        for part in color_parts:
                            if '%' in part:
                                percent_str, color_name = part.split('%', 1)
                                percent = float(percent_str.strip())
                                color_name = color_name.strip()
                                colors.append({"name": color_name, "percentage": percent})
                                total_percentage += percent
                        
                        params_text += f"• Количество цветов: {len(colors)}\n"
                        params_text += f"• Сумма процентов: {total_percentage}%\n"
                        params_text += f"• Цвета: {', '.join([f'{c['percentage']}% {c['name']}' for c in colors])}\n"
                        
                        # Проверяем валидность цветов
                        invalid_colors = [c['name'] for c in colors if c['name'].upper() not in valid_colors]
                        if invalid_colors:
                            params_text += f"❌ Неизвестные цвета: {', '.join(invalid_colors)}\n"
                        else:
                            params_text += f"✅ Все цвета валидны\n"
                        
                        if total_percentage != 100:
                            params_text += f"⚠️ Внимание: сумма не равна 100%\n"
                        if len(colors) > 4:
                            params_text += f"⚠️ Внимание: больше 4 цветов\n"
                            
                    except Exception as e:
                        params_text += f"• Ошибка анализа prompt: {e}\n"
                
                # Обновляем текстовое поле
                self.preset_params_text.config(state=tk.NORMAL)
                self.preset_params_text.delete(1.0, tk.END)
                self.preset_params_text.insert(1.0, params_text)
                self.preset_params_text.config(state=tk.DISABLED)
                
                # Логируем выбор
                if is_valid:
                    self.append_log(f"📋 Выбран пресет: {clean_name} ✅\n")
                else:
                    self.append_log(f"⚠️ Выбран пресет с ошибками: {clean_name}\n")
                    for error in validation_errors:
                        self.append_log(f"  {error}\n")
            else:
                # Очищаем текстовое поле если пресет не найден
                self.preset_params_text.config(state=tk.NORMAL)
                self.preset_params_text.delete(1.0, tk.END)
                self.preset_params_text.insert(1.0, f"❌ Пресет '{clean_name}' не найден")
                self.preset_params_text.config(state=tk.DISABLED)
                
        except Exception as e:
            self.append_log(f"⚠️ Ошибка отображения одного пресета: {e}\n")
    
    def _display_multiple_presets(self, selection) -> None:
        """Отображает информацию о множественном выборе пресетов"""
        try:
            selected_presets = []
            for idx in selection:
                preset_name = self.listbox.get(idx)
                clean_name = preset_name.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
                selected_presets.append(clean_name)
            
            # Форматируем информацию о множественном выборе
            params_text = f"📋 МНОЖЕСТВЕННЫЙ ВЫБОР ПРЕСЕТОВ\n"
            params_text += "=" * 50 + "\n\n"
            params_text += f"🎯 Выбрано пресетов: {len(selection)}\n\n"
            
            # Отображаем список выбранных пресетов
            params_text += "📋 ВЫБРАННЫЕ ПРЕСЕТЫ:\n"
            params_text += "-" * 30 + "\n"
            
            for i, preset_name in enumerate(selected_presets, 1):
                params_text += f"{i}. {preset_name}\n"
                
                # Добавляем краткую информацию о каждом пресете
                if preset_name in self.presets:
                    preset_data = self.presets[preset_name]
                    
                    # Проверяем валидность
                    is_valid, _ = self.validate_preset(preset_name, preset_data, self.current_version)
                    status = "✅" if is_valid else "❌"
                    
                    # Краткая информация
                    if 'prompt' in preset_data:
                        prompt = preset_data['prompt']
                        if len(prompt) > 40:
                            prompt = prompt[:37] + "..."
                        params_text += f"   Prompt: {prompt}\n"
                    
                    # Адаптивное отображение параметров для разных версий
                    if hasattr(self, 'current_version') and self.current_version and compare_versions(self.current_version, "v4.5.01") >= 0:
                        if 'num_inference_steps' in preset_data and 'guidance_scale' in preset_data:
                            params_text += f"   Steps: {preset_data['num_inference_steps']}, Guidance: {preset_data['guidance_scale']}\n"
                        if 'colormap' in preset_data:
                            params_text += f"   Colormap: {preset_data['colormap']}\n"
                        if 'granule_size' in preset_data:
                            params_text += f"   Granule Size: {preset_data['granule_size']}\n"
                    else:
                        if 'steps' in preset_data and 'guidance' in preset_data:
                            params_text += f"   Steps: {preset_data['steps']}, Guidance: {preset_data['guidance']}\n"
                        if 'lora_scale' in preset_data:
                            params_text += f"   LoRA Scale: {preset_data['lora_scale']}\n"
                    
                    params_text += f"   Статус: {status}\n\n"
                else:
                    params_text += f"   ❌ Пресет не найден\n\n"
            
            # Информация о выполнении
            params_text += "🚀 ИНФОРМАЦИЯ О ВЫПОЛНЕНИИ:\n"
            params_text += "-" * 30 + "\n"
            params_text += "• При нажатии 'Start' будут выполнены ВСЕ выбранные пресеты\n"
            params_text += "• Пресеты выполняются последовательно\n"
            params_text += "• Для детального просмотра выберите только один пресет\n"
            
            # Обновляем текстовое поле
            self.preset_params_text.config(state=tk.NORMAL)
            self.preset_params_text.delete(1.0, tk.END)
            self.preset_params_text.insert(1.0, params_text)
            self.preset_params_text.config(state=tk.DISABLED)
            
            # Логируем множественный выбор
            self.append_log(f"📋 Выбрано пресетов: {len(selection)}\n")
            for preset_name in selected_presets:
                self.append_log(f"  • {preset_name}\n")
                
        except Exception as e:
            self.append_log(f"⚠️ Ошибка отображения множественных пресетов: {e}\n")
    
    def update_version_info(self) -> None:
        """Обновляет информацию о версии и хеше"""
        print("🔄 update_version_info вызван")
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            print("❌ Text widget не создан, выход из update_version_info")
            return
        print("✅ Text widget найден, продолжаем обновление версии")
            
        try:
            # Автоматически определяем версию из cog.yaml без зависимости от YAML
            new_version = self._extract_version_from_cog_yaml()
            
            # Принудительно устанавливаем v4.5.01 для неизвестных версий
            if not new_version or new_version == "unknown":
                new_version = "v4.5.01"
                self.append_log(f"⚠️ Версия не определена, принудительно используем {new_version}\n")
            
            if new_version and new_version != self.current_version:
                self.current_version = new_version
                
                # Используем гибридный подход для определения хеша
                self.current_hash = self.get_optimal_hash(new_version)
                
                # Обновляем заголовок окна
                self.root.title(f"Replicate Test Runner - {self.current_version} ({self.current_hash})")
                
                # Обновляем лейблы в верхней панели
                if hasattr(self, 'version_label'):
                    self.version_label.config(text=self.current_version)
                if hasattr(self, 'hash_label'):
                    self.hash_label.config(text=self.current_hash)
                
                # Логируем обновление
                self.append_log(f"🔄 Версия обновлена: {self.current_version}\n")
                self.append_log(f"✅ Новый хеш (гибридный): {self.current_hash}\n")
                
                # Обновляем worker с новой версией
                if hasattr(self, 'worker') and self.worker is not None:
                    self.worker._version_id = new_version
                    # Передаем версию и хеш в worker
                    self.worker.current_version = new_version
                    self.worker.current_hash = self.current_hash
                
                # Перезагружаем пресеты для новой версии
                self._reload_presets_for_version()
                
                # Создаем файл логов GUI после определения версии
                self._create_gui_log_file()
            else:
                # Даже если версия не изменилась, проверяем актуальность хеша
                optimal_hash = self.get_optimal_hash(self.current_version)
                if optimal_hash != self.current_hash:
                    self.current_hash = optimal_hash
                    self.append_log(f"🔄 Хеш обновлен: {self.current_hash}\n")
                    self.root.title(f"Replicate Test Runner - {self.current_version} ({self.current_hash})")
                    
                    # Обновляем лейбл хеша
                    if hasattr(self, 'hash_label'):
                        self.hash_label.config(text=self.current_hash)
                else:
                    self.append_log(f"✅ Версия и хеш актуальны: {self.current_version} ({self.current_hash})\n")
        except Exception as e:
            self.append_log(f"⚠️ Ошибка обновления версии: {e}\n")
    
    def _extract_version_from_cog_yaml(self) -> str:
        """Извлекает версию из cog.yaml без использования YAML модуля"""
        try:
            with open("cog.yaml", "r", encoding="utf-8") as f:
                content = f.read()
            
            # Ищем строку с версией в комментарии
            for line in content.split('\n'):
                if line.strip().startswith('# Version:'):
                    version_part = line.split('Version:')[1].strip()
                    return version_part.split()[0]  # Берем первое слово после "Version:"
                
                # Альтернативный поиск по тегу образа
                if 'image:' in line and 'plitka-pro-project:' in line:
                    version_part = line.split('plitka-pro-project:')[1].strip()
                    return version_part
                    
            # Если не нашли, пытаемся определить версию из других источников
            return self._determine_version_fallback()
        except Exception as e:
            print(f"Ошибка чтения cog.yaml: {e}")
            return self._determine_version_fallback()
    
    def _determine_version_fallback(self) -> str:
        """Автоматически определяет версию из различных источников"""
        try:
            # Приоритет 1: Docker образы
            docker_version = self._get_version_from_docker()
            if docker_version:
                return docker_version
            
            # Приоритет 2: Git теги
            git_version = self._get_version_from_git()
            if git_version:
                return git_version
            
            # Приоритет 3: Файл версии
            file_version = self._get_version_from_file()
            if file_version:
                return file_version
            
            # Fallback: генерируем версию на основе времени
            return self._generate_version_from_time()
            
        except Exception as e:
            print(f"Ошибка определения версии: {e}")
            return self._generate_version_from_time()
    
    def _get_version_from_docker(self) -> str:
        """Получает версию из Docker образов"""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "r8.im/nauslava/plitka-pro-project"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line and not line.endswith(':latest'):
                        version = line.split(':')[-1]
                        if version.startswith('v'):
                            return version
        except Exception:
            pass
        return ""
    
    def _get_version_from_git(self) -> str:
        """Получает версию из Git тегов"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return ""
    
    def _get_version_from_file(self) -> str:
        """Получает версию из файла версии"""
        try:
            version_files = ["VERSION", "version.txt", ".version"]
            for filename in version_files:
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        version = f.read().strip()
                        if version.startswith('v'):
                            return version
        except Exception:
            pass
        return ""
    
    def _generate_version_from_time(self) -> str:
        """Генерирует версию на основе времени (fallback)"""
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"v{timestamp}"
    
    def get_optimal_hash(self, version: str) -> str:
        """Получает оптимальный хеш для версии (гибридный подход)"""
        print(f"🔍 Поиск хеша для версии: {version}")
        
        # Приоритет 1: Docker образ
        docker_hash = self.get_docker_image_hash(version)
        if docker_hash:
            print(f"✅ Найден Docker хеш: {docker_hash}")
            return docker_hash
        
        # Приоритет 2: Replicate API
        replicate_hash = self.get_replicate_version_hash(version)
        if replicate_hash:
            print(f"✅ Найден Replicate хеш: {replicate_hash}")
            return replicate_hash
        
        # Fallback: вычисление из версии
        computed_hash = self.compute_version_hash(version)
        print(f"⚠️ Используем вычисленный хеш: {computed_hash}")
        return computed_hash
    
    def get_docker_image_hash(self, version: str) -> str:
        """Получает хеш Docker образа для указанной версии"""
        try:
            import subprocess
            # Ищем образ по тегу версии
            result = subprocess.run(
                ["docker", "images", "--format", "{{.ID}}", f"r8.im/nauslava/plitka-pro-project:{version}"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                full_hash = result.stdout.strip()
                return full_hash[:8]  # Первые 8 символов для папки
        except Exception as e:
            print(f"Ошибка получения Docker хеша: {e}")
        return ""
    
    def get_replicate_version_hash(self, version: str) -> str:
        """Получает хеш версии Replicate через API"""
        try:
            if not REPLICATE_AVAILABLE:
                return ""
            
            import replicate
            # Получаем информацию о модели
            model_info = replicate.models.get("nauslava/plitka-pro-project")
            if model_info and hasattr(model_info, 'versions'):
                # Проверяем, что versions итерируемо
                try:
                    versions_list = list(model_info.versions) if hasattr(model_info.versions, '__iter__') else []
                    for ver in versions_list:
                        if hasattr(ver, 'id') and ver.id == version:
                            # Используем digest версии
                            if hasattr(ver, 'digest') and ver.digest:
                                return ver.digest[:8]
                            # Или вычисляем из ID версии
                            import hashlib
                            hash_obj = hashlib.md5(ver.id.encode())
                            return hash_obj.hexdigest()[:8]
                except (TypeError, AttributeError) as e:
                    print(f"Ошибка итерации по версиям: {e}")
                    # Fallback: вычисляем хеш из версии
                    import hashlib
                    hash_obj = hashlib.md5(version.encode())
                    return hash_obj.hexdigest()[:8]
        except Exception as e:
            print(f"Ошибка получения Replicate хеша: {e}")
        return ""
    
    def compute_version_hash(self, version: str) -> str:
        """Вычисляет хеш версии как fallback"""
        try:
            import hashlib
            hash_obj = hashlib.md5(version.encode())
            return hash_obj.hexdigest()[:8]
        except Exception as e:
            print(f"Ошибка вычисления хеша версии: {e}")
            return "unknown"
    
    def run_demo(self) -> None:
        """Запускает демо-режим для тестирования интерфейса"""
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        self.append_log("\n🎭 Запуск демо-режима...\n")
        self.append_log(f"🚀 Текущая версия: {self.current_version}\n")
        self.append_log(f"🔐 Хеш папки: {self.current_hash}\n")
        self.append_log(f"📁 Ожидаемая структура: replicate_runs/{self.current_hash}/run_YYYYMMDD_HHMMSS/\n")
        self.append_log("⚠️ Replicate недоступен - это демо-режим\n")
        self.append_log("💡 Для полноценной работы установите: pip install replicate\n")
        self.append_log("=" * 60 + "\n")

    def drain_queue(self) -> None:
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                if isinstance(msg, str) and msg.startswith("__IMG__::"):
                    try:
                        _, kind, path = msg.split("::", 2)
                        self._update_image_preview(kind, path)
                    except Exception:
                        pass
                else:
                    self.append_log(msg)
        except queue.Empty:
            pass
        self.root.after(300, self.drain_queue)

    def update_timer(self) -> None:
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        # Reflect worker status
        try:
            if not hasattr(self, 'worker') or self.worker is None:
                self.status_var.set("idle")
                self.elapsed_var.set("00:00")
                return
                
            self.status_var.set(getattr(self.worker, "_status", "idle"))
            start_time = getattr(self.worker, "_start_time", None)
            if start_time:
                elapsed = int(time.time() - start_time)
                mm = elapsed // 60
                ss = elapsed % 60
                self.elapsed_var.set(f"{mm:02d}:{ss:02d}")
            else:
                self.elapsed_var.set("00:00")
            if not self.worker.is_running() and self.controls_locked:
                self.start_btn.configure(state=tk.NORMAL)
                self.listbox.configure(state=tk.NORMAL)
                self.controls_locked = False
        except Exception as e:
            # Логируем ошибку, но продолжаем работу
            try:
                self.append_log(f"[GUI] Timer error: {e}\n")
            except:
                pass
        finally:
            self.root.after(1000, self.update_timer)

    def selected_presets(self) -> List[Dict[str, Any]]:
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'listbox'):
            return []
            
        sel = self.listbox.curselection()
        items: List[Dict[str, Any]] = []
        for idx in sel:
            name = self.listbox.get(idx)
            # Убираем префиксы статуса
            clean_name = name.replace("✅ ", "").replace("❌ ", "").replace("⚠️ ", "")
            inputs = self.presets.get(clean_name, {})
            items.append({"name": clean_name, "inputs": inputs})
        if not items and self.presets:
            # default to first preset if nothing selected
            name = list(self.presets.keys())[0]
            items.append({"name": name, "inputs": self.presets[name]})
        return items

    def on_start(self) -> None:
        self.append_log("🔄 on_start() вызван\n")
        
        # Проверяем, что worker инициализирован
        if not hasattr(self, 'worker') or self.worker is None:
            messagebox.showerror("Error", "Worker не инициализирован. Перезапустите приложение.")
            return
            
        if self.worker.is_running():
            messagebox.showinfo("Info", "Test is already running")
            return
            
        # Если токен не настроен, но введён в поле — используем его
        try:
            if not getattr(self.worker, 'api_token', None):
                entered = self.token_var.get().strip()
                if entered:
                    os.environ["REPLICATE_API_TOKEN"] = entered
                    self.worker.api_token = entered
                else:
                    messagebox.showwarning("Token missing", "REPLICATE_API_TOKEN is not set. Enter token and press Save")
                    return
        except Exception as e:
            messagebox.showerror("Error", f"Ошибка проверки токена: {e}")
            return
        items = self.selected_presets()
        if not items:
            messagebox.showwarning("No presets", "No presets available to run")
            return
            
        if not REPLICATE_AVAILABLE:
            messagebox.showwarning("Replicate недоступен", 
                                 "Модуль 'replicate' не установлен.\nУстановите: pip install replicate\n\nИспользуйте кнопку 'Demo Mode' для тестирования интерфейса.")
            return
            
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        # НЕ создаем папку здесь - она будет создана в worker'е
        # self._create_log_file()
        
        self.append_log("\n[GUI] Starting tests...\n")
        self.append_log(f"🚀 Запуск с версией: {self.current_version}\n")
        self.append_log(f"🔐 Хеш папки: {self.current_hash}\n")
        self.append_log(f"📁 Ожидаемая папка: replicate_runs/{self.current_hash}/\n")
        current_dir = os.getcwd()
        self.append_log(f"🔍 Текущий рабочий каталог: {current_dir}\n")
        full_path = os.path.join(current_dir, 'replicate_runs', self.current_hash)
        self.append_log(f"📁 Полный путь: {full_path}\n")
        self.append_log(f"📁 Абсолютный путь: {os.path.abspath(full_path)}\n")
        self.append_log("-" * 50 + "\n")
        
        # Disable controls while running
        self.start_btn.configure(state=tk.DISABLED)
        self.listbox.configure(state=tk.DISABLED)
        self.controls_locked = True
        self.worker.start(items)

    def on_stop(self) -> None:
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        # Проверяем, что worker инициализирован
        if not hasattr(self, 'worker') or self.worker is None:
            self.append_log("[GUI] Worker не инициализирован.\n")
            return
            
        # Создаем файл логов для остановки
        self._create_stop_log_file()
            
        try:
            if not self.worker.is_running():
                self.append_log("[GUI] No active run to stop.\n")
                return
            self.worker.stop()
            self.append_log("[GUI] Stop requested.\n")
        except Exception as e:
            self.append_log(f"[GUI] Ошибка при остановке: {e}\n")
        finally:
            # Re-enable controls after stop request (worker cancels prediction)
            self.start_btn.configure(state=tk.NORMAL)
            self.listbox.configure(state=tk.NORMAL)
            self.controls_locked = False

    def on_close(self) -> None:
        # Проверяем, что UI элементы созданы
        if not hasattr(self, 'text'):
            return
            
        # Создаем файл логов для закрытия в единой папке
        self._create_close_log_file()
        
        try:
            if hasattr(self, 'worker') and self.worker is not None and self.worker.is_running():
                try:
                    self.worker.stop()
                except Exception:
                    pass
                # give a brief moment to send cancel
                self.root.after(500, self.root.destroy)
            else:
                self.root.destroy()
        except Exception:
            self.root.destroy()


def main() -> int:
    try:
        print("🚀 Запуск GUI...")
        root = tk.Tk()
        print("✅ Tkinter окно создано")
        
        # Создаем приложение
        print("🔄 Создание App...")
        app = App(root)
        print("✅ App инициализирован")
        
        # Проверяем, что все UI элементы созданы
        print("🔍 Проверка UI элементов...")
        if hasattr(app, 'text'):
            print("✅ Text widget создан")
        else:
            print("❌ Text widget НЕ создан")
            
        if hasattr(app, 'worker'):
            print("✅ Worker создан")
        else:
            print("❌ Worker НЕ создан")
            
        # Запускаем главный цикл
        print("🔄 Запуск mainloop...")
        try:
            root.mainloop()
            print("✅ GUI завершен")
        except Exception as e:
            print(f"❌ Ошибка в mainloop: {e}")
            import traceback
            traceback.print_exc()
        return 0
        
    except Exception as e:
        print(f"❌ Ошибка запуска GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


