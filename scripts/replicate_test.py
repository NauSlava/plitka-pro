#!/usr/bin/env python3
"""
Replicate Test Harness for Plitka Pro Project
Version: v1.2.0
Compatible with: Plitka Pro v4.4.56+
Description: Comprehensive testing framework for Color Grid Adapter and ControlNet integration
Author: Plitka Pro Development Team
Date: 2025-09-02
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional


def _load_env_token() -> Optional[str]:
    token = os.getenv("REPLICATE_API_TOKEN")
    if token:
        return token
    # Fallback: minimal .env reader
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate test harness with Color Grid Adapter testing and startup log monitoring")
    parser.add_argument("--model", default="nauslava/plitka-pro-project:v4.4.56", help="Model ref 'owner/name[:version_or_tag]' (default: %(default)s)")
    parser.add_argument("--preset", default=None, help="Preset name from JSON presets file")
    parser.add_argument("--version-id", default=None, help="Explicit 64-hex Replicate version id to use")
    parser.add_argument("--batch", default=None, help="Path to JSON file with presets to run as a batch")

    # Direct inputs (match predict.py)
    parser.add_argument("--prompt", default=None, help="Prompt text, e.g. '60% red, 40% white'")
    parser.add_argument("--negative_prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Seed (-1 for random)")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale")
    parser.add_argument("--lora_scale", type=float, default=None, help="LoRA scale (0.0-1.0)")
    parser.add_argument("--use_controlnet", action="store_true", help="Enable ControlNet SoftEdge")

    # Color Grid Adapter specific
    parser.add_argument("--test-color-grid", action="store_true", help="Run comprehensive Color Grid Adapter tests")
    parser.add_argument("--test-patterns", action="store_true", help="Test all Color Grid Adapter patterns")
    parser.add_argument("--test-granule-sizes", action="store_true", help="Test all granule sizes")

    # Behaviour
    parser.add_argument("--poll-seconds", type=int, default=6, help="Polling interval in seconds (default: %(default)s)")
    parser.add_argument("--startup-timeout", type=int, default=7*60, help="Startup timeout seconds to see setup completion (default: %(default)s)")
    parser.add_argument("--total-timeout", type=int, default=25*60, help="Total timeout seconds for a prediction (default: %(default)s)")
    return parser.parse_args()


def _merge_inputs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def _read_presets(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Presets JSON must be an object mapping preset names to input dicts")
        return data


def _get_color_grid_test_presets() -> Dict[str, Dict[str, Any]]:
    """Возвращает специальные тесты для Color Grid Adapter"""
    return {
        "color_grid_patterns": {
            "random": {
                "prompt": "100% red",
                "seed": 12345,
                "steps": 20,
                "guidance": 7.0,
                "lora_scale": 0.7,
                "use_controlnet": False,
                "description": "Тест random паттерна (1 цвет)"
            },
            "granular_medium": {
                "prompt": "50% red, 50% white",
                "seed": 12345,
                "steps": 25,
                "guidance": 7.5,
                "lora_scale": 0.7,
                "use_controlnet": False,
                "description": "Тест granular паттерна с medium гранулами (2 цвета)"
            },
            "granular_small": {
                "prompt": "25% red, 25% blue, 25% green, 25% yellow",
                "seed": 12345,
                "steps": 35,
                "guidance": 8.5,
                "lora_scale": 0.7,
                "use_controlnet": False,
                "description": "Тест granular паттерна с small гранулами (4 цвета)"
            }
        },
        "color_grid_granule_sizes": {
            "small_granules": {
                "prompt": "50% red, 30% black, 20% white",
                "seed": 12345,
                "steps": 30,
                "guidance": 8.0,
                "lora_scale": 0.7,
                "use_controlnet": False,
                "description": "Тест small гранул (3 цвета)"
            },
            "medium_granules": {
                "prompt": "60% red, 40% white",
                "seed": 12345,
                "steps": 25,
                "guidance": 7.5,
                "lora_scale": 0.7,
                "use_controlnet": False,
                "description": "Тест medium гранул (2 цвета)"
            },
            "large_granules": {
                "prompt": "100% gray",
                "seed": 12345,
                "steps": 20,
                "guidance": 7.0,
                "lora_scale": 0.7,
                "use_controlnet": False,
                "description": "Тест large гранул (1 цвет)"
            }
        }
    }


def _run_color_grid_comprehensive_test(client, model_ref: str, poll_s: int, startup_timeout_s: int, total_timeout_s: int, version_id_arg: Optional[str]) -> int:
    """Запускает комплексный тест Color Grid Adapter"""
    print("\n🎨 === COMPREHENSIVE COLOR GRID ADAPTER TEST ===")
    
    # Тестируем все паттерны
    pattern_presets = _get_color_grid_test_presets()["color_grid_patterns"]
    print(f"\n📊 Тестируем паттерны: {list(pattern_presets.keys())}")
    
    results = {}
    for pattern_name, preset in pattern_presets.items():
        print(f"\n🧪 Тестируем паттерн: {pattern_name}")
        print(f"📝 Описание: {preset['description']}")
        
        rc = _run_single(
            client, model_ref, preset, poll_s, startup_timeout_s, total_timeout_s, version_id_arg
        )
        
        results[pattern_name] = {
            "return_code": rc,
            "success": rc == 0,
            "preset": preset
        }
        
        if rc != 0:
            print(f"❌ Тест паттерна {pattern_name} провален (код: {rc})")
        else:
            print(f"✅ Тест паттерна {pattern_name} пройден")
    
    # Тестируем все размеры гранул
    granule_presets = _get_color_grid_test_presets()["color_grid_granule_sizes"]
    print(f"\n📊 Тестируем размеры гранул: {list(granule_presets.keys())}")
    
    for granule_name, preset in granule_presets.items():
        print(f"\n🧪 Тестируем размер гранул: {granule_name}")
        print(f"📝 Описание: {preset['description']}")
        
        rc = _run_single(
            client, model_ref, preset, poll_s, startup_timeout_s, total_timeout_s, version_id_arg
        )
        
        results[granule_name] = {
            "return_code": rc,
            "success": rc == 0,
            "preset": preset
        }
        
        if rc != 0:
            print(f"❌ Тест размера гранул {granule_name} провален (код: {rc})")
        else:
            print(f"✅ Тест размера гранул {granule_name} пройден")
    
    # Итоговая статистика
    print("\n📊 === ИТОГОВАЯ СТАТИСТИКА COLOR GRID ADAPTER ===")
    total_tests = len(results)
    successful_tests = sum(1 for r in results.values() if r["success"])
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Всего тестов: {total_tests}")
    print(f"Успешных: {successful_tests}")
    print(f"Провальных: {total_tests - successful_tests}")
    print(f"Процент успеха: {success_rate:.1f}%")
    
    # Детальные результаты
    print("\n📋 Детальные результаты:")
    for test_name, result in results.items():
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {test_name}: {result['preset']['description']}")
    
    return 0 if success_rate >= 80 else 1


def _is_boot_error(logs: str) -> bool:
    if not logs:
        return False
    needles = [
        "Traceback (most recent call last)",
        "CRITICAL", "FATAL", "RuntimeError", "CUDA error", "device-side assert",
        "❌", "ERROR:predict:", "OSError: Cannot load model",
    ]
    for n in needles:
        if n in logs:
            return True
    return False


def _is_boot_completed(logs: str) -> bool:
    if not logs:
        return False
    markers = [
        "setup completed",
        "🎉 Модель v",  # localized marker from predict.py
        "Model initialized",  # generic
    ]
    for m in markers:
        if m in logs:
            return True
    return False


def _resolve_model_and_version(client, model_ref: str, version_id_arg: Optional[str]) -> (str, Optional[str]):
    if version_id_arg:
        base = model_ref.split(":", 1)[0]
        return base, version_id_arg
    # Accept 'owner/name' or 'owner/name:versionOrTag'
    if ":" not in model_ref:
        return model_ref, None
    base, ver = model_ref.split(":", 1)
    ver = ver.strip()
    # If looks like a 64-hex version id, use it directly
    if len(ver) == 64 and all(c in "0123456789abcdef" for c in ver.lower()):
        return base, ver
    # Otherwise, fallback to latest version (ignore tag) to avoid API error
    try:
        m = client.models.get(base)
        versions = list(m.versions.list())
        if versions:
            return base, versions[0].id
    except Exception:
        pass
    return base, None


def _run_single(client, model_ref: str, inputs: Dict[str, Any], poll_s: int, startup_timeout_s: int, total_timeout_s: int, version_id_arg: Optional[str]) -> int:
    model_name, version_id = _resolve_model_and_version(client, model_ref, version_id_arg)
    
    # Анализ промпта для Color Grid Adapter
    prompt = inputs.get("prompt", "")
    color_count = len([word for word in prompt.lower().split() 
                       if any(color in word for color in ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'grey', 'black', 'white', 'cyan', 'magenta'])])
    
    print(f"\n=== Creating prediction: {model_name} (version: {version_id or 'latest'})")
    print(f"🎨 Color Grid Adapter анализ:")
    print(f"   - Промпт: {prompt}")
    print(f"   - Количество цветов: {color_count}")
    print(f"   - Ожидаемый паттерн: {'random' if color_count == 1 else 'granular'}")
    print(f"   - Ожидаемый размер гранул: {'medium' if color_count <= 2 else 'small'}")
    print(f"   - ControlNet: {'автоматически' if color_count >= 2 and not inputs.get('use_controlnet') else 'ручной'}")
    print(f"Inputs: {json.dumps(inputs, ensure_ascii=False)}\n")
    
    if version_id:
        pred = client.predictions.create(version=version_id, input=inputs)
    else:
        pred = client.predictions.create(model=model_name, input=inputs)
    print(f"Prediction id: {pred.id}")

    t0 = time.time()
    started = False
    last_logs_len = 0
    last_out_len = 0

    # Prepare run folder for artifacts (logs, outputs)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), f"replicate_runs/run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    def _save_artifacts(final_status: str) -> None:
        try:
            # logs
            with open(os.path.join(run_dir, "logs.txt"), "w", encoding="utf-8") as f:
                f.write(pred.logs or "")
            # meta
            meta = {
                "id": pred.id,
                "status": final_status,
                "model": model_name,
                "version": version_id or "latest",
                "created_at": ts,
                "finished_at": time.strftime("%Y%m%d_%H%M%S"),
                "inputs": inputs,
            }
            with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _download_new_outputs() -> None:
        nonlocal last_out_len
        try:
            outs = pred.output
            if isinstance(outs, list) and len(outs) > last_out_len:
                try:
                    import requests  # type: ignore
                except Exception:
                    return
                for idx in range(last_out_len, len(outs)):
                    url = outs[idx]
                    try:
                        r = requests.get(url, timeout=60)
                        r.raise_for_status()
                        name = ["preview", "final", "colormap"][idx] if idx < 3 else f"out_{idx}"
                        with open(os.path.join(run_dir, f"{name}.png"), "wb") as f:
                            f.write(r.content)
                    except Exception:
                        continue
                last_out_len = len(outs)
        except Exception:
            pass

    while True:
        pred.reload()
        logs = pred.logs or ""
        # print incremental logs
        if len(logs) > last_logs_len:
            sys.stdout.write(logs[last_logs_len:])
            sys.stdout.flush()
            last_logs_len = len(logs)

        # stream outputs as soon as they appear
        _download_new_outputs()

        # Startup monitoring
        if not started:
            if _is_boot_error(logs) and not _is_boot_completed(logs):
                print("\n❌ Boot error detected before startup completed. Cancelling prediction...")
                try:
                    pred.cancel()
                except Exception:
                    pass
                _save_artifacts("boot_failed")
                return 2
            if pred.status in ("processing", "succeeded") or _is_boot_completed(logs):
                started = True
                print("\n✅ Startup phase completed. Proceeding to completion...")
            elif time.time() - t0 > startup_timeout_s:
                print("\n❌ Startup timeout exceeded. Cancelling prediction...")
                try:
                    pred.cancel()
                except Exception:
                    pass
                _save_artifacts("startup_timeout")
                return 3
        
        # Color Grid Adapter monitoring
        if started and "🎨" in logs and "Color Grid Adapter" in logs:
            # Ищем логи Color Grid Adapter
            color_grid_logs = [line for line in logs.split('\n') if "🎨" in line and "Color Grid Adapter" in line]
            if color_grid_logs:
                print(f"\n🎨 Color Grid Adapter логи:")
                for log_line in color_grid_logs[-3:]:  # Показываем последние 3 логи
                    print(f"   {log_line.strip()}")
        
        # ControlNet monitoring
        if started and ("🔗" in logs or "ControlNet" in logs):
            controlnet_logs = [line for line in logs.split('\n') if "🔗" in line or "ControlNet" in line]
            if controlnet_logs:
                print(f"\n🔗 ControlNet логи:")
                for log_line in controlnet_logs[-3:]:  # Показываем последние 3 логи
                    print(f"   {log_line.strip()}")

        # Final states
        if pred.status in ("succeeded", "failed", "canceled"):
            print(f"\n=== Final status: {pred.status}")
            _download_new_outputs()
            _save_artifacts(pred.status)
            if pred.status == "succeeded":
                print("Output:")
                try:
                    print(json.dumps(pred.output, ensure_ascii=False, indent=2))
                except Exception:
                    print(str(pred.output))
                return 0
            else:
                return 1

        if time.time() - t0 > total_timeout_s:
            print("\n❌ Total timeout exceeded. Cancelling prediction...")
            try:
                pred.cancel()
            except Exception:
                pass
            _save_artifacts("total_timeout")
            return 4

        time.sleep(poll_s)


def main() -> int:
    token = _load_env_token()
    if not token:
        print("REPLICATE_API_TOKEN is not set. Put it in environment or .env file.")
        return 5

    try:
        import replicate  # type: ignore
    except Exception:
        print("The 'replicate' package is required. Install with: pip install replicate")
        return 6

    args = _parse_args()
    client = replicate.Client(api_token=token)

    base_inputs: Dict[str, Any] = {}
    if args.prompt is not None:
        base_inputs["prompt"] = args.prompt
    if args.negative_prompt is not None:
        base_inputs["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        base_inputs["seed"] = args.seed
    if args.steps is not None:
        base_inputs["steps"] = args.steps
    if args.guidance is not None:
        base_inputs["guidance"] = args.guidance
    if args.lora_scale is not None:
        base_inputs["lora_scale"] = args.lora_scale
    if args.use_controlnet:
        base_inputs["use_controlnet"] = True

    # Batch mode
    if args.batch:
        presets = _read_presets(args.batch)
        rc_all = 0
        for name, input_spec in presets.items():
            print(f"\n\n################ PRESET: {name} ################")
            merged = _merge_inputs(input_spec, base_inputs)
            rc = _run_single(
                client,
                args.model,
                merged,
                args.poll_seconds,
                args.startup_timeout,
                args.total_timeout,
                args.version_id,
            )
            rc_all = rc_all or rc
        return rc_all

    # Color Grid Adapter comprehensive testing
    if args.test_color_grid:
        return _run_color_grid_comprehensive_test(
            client,
            args.model,
            args.poll_seconds,
            args.startup_timeout,
            args.total_timeout,
            args.version_id,
        )
    
    # Single run (optionally via named preset file)
    if args.preset:
        # default preset file name for current version
        default_file = os.path.join(os.path.dirname(__file__), "test_inputs_v4.4.56.json")
        if not os.path.exists(default_file):
            print(f"Preset file not found: {default_file}")
            return 7
        presets = _read_presets(default_file)
        if args.preset not in presets:
            print(f"Preset '{args.preset}' is not defined in {default_file}")
            return 8
        inputs = _merge_inputs(presets[args.preset], base_inputs)
    else:
        inputs = base_inputs

    if not inputs.get("prompt"):
        inputs["prompt"] = "100% red"

    return _run_single(
        client,
        args.model,
        inputs,
        args.poll_seconds,
        args.startup_timeout,
        args.total_timeout,
        args.version_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())


