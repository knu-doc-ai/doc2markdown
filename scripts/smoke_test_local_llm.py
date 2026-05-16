from __future__ import annotations

"""로컬 오픈웨이트 LLM smoke test."""

import importlib
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.llm_core import LLMConfig, LLMGenerationError, LocalTransformersLLMClient


EXIT_IMPORT_FAILURE = 2
EXIT_LOAD_FAILURE = 3
EXIT_OOM_FAILURE = 4
EXIT_JSON_FAILURE = 5
EXIT_GENERATION_FAILURE = 6


def main() -> int:
    _load_dotenv_if_available()
    config = LLMConfig.from_env()
    print("[Smoke] Local LLM smoke test 시작")
    print(f"[Smoke] model_id={config.model_id}")
    print(f"[Smoke] max_new_tokens={config.max_new_tokens}")
    print(f"[Smoke] enrichment_mode={config.mode}")

    imports = _check_imports()
    if imports is None:
        return EXIT_IMPORT_FAILURE

    torch = imports["torch"]
    _print_torch_runtime(torch)

    client = LocalTransformersLLMClient(config)
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _, model = client._load_model()
    except RuntimeError as error:
        if _is_oom_error(error):
            _print_exception("OOM", error)
            _print_peak_memory(torch)
            return EXIT_OOM_FAILURE
        _print_exception("Load Failure", error)
        return EXIT_LOAD_FAILURE
    except Exception as error:
        _print_exception("Load Failure", error)
        return EXIT_LOAD_FAILURE

    print("[Smoke] 모델 로드 성공")
    _print_model_runtime(model)

    try:
        response = client.generate_json("content_repair", _build_content_repair_payload())
    except LLMGenerationError as error:
        _print_exception("JSON Failure", error)
        _print_peak_memory(torch)
        return EXIT_JSON_FAILURE
    except RuntimeError as error:
        if _is_oom_error(error):
            _print_exception("OOM", error)
            _print_peak_memory(torch)
            return EXIT_OOM_FAILURE
        _print_exception("Generation Failure", error)
        _print_peak_memory(torch)
        return EXIT_GENERATION_FAILURE
    except Exception as error:
        _print_exception("Generation Failure", error)
        _print_peak_memory(torch)
        return EXIT_GENERATION_FAILURE

    print("[Smoke] JSON 생성 성공")
    print(f"[Smoke] response={response}")
    _print_peak_memory(torch)
    return 0


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(ROOT_DIR / ".env")


def _check_imports() -> dict[str, Any] | None:
    modules: dict[str, Any] = {}
    for name in ("torch", "transformers", "accelerate", "huggingface_hub", "safetensors"):
        try:
            module = importlib.import_module(name)
        except Exception as error:
            print(f"[Smoke][Import Failure] {name} import 실패: {error}")
            return None
        modules[name] = module
        version = getattr(module, "__version__", "unknown")
        print(f"[Smoke] {name}={version}")
    return modules


def _print_torch_runtime(torch: Any) -> None:
    cuda_available = torch.cuda.is_available()
    print(f"[Smoke] cuda_available={cuda_available}")
    if not cuda_available:
        return

    device_index = torch.cuda.current_device()
    print(f"[Smoke] cuda_device={torch.cuda.get_device_name(device_index)}")
    props = torch.cuda.get_device_properties(device_index)
    print(f"[Smoke] cuda_total_memory_gb={props.total_memory / 1024**3:.2f}")


def _print_model_runtime(model: Any) -> None:
    print(f"[Smoke] model_device={getattr(model, 'device', 'unknown')}")
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        print("[Smoke] hf_device_map 없음")
        return

    counts = Counter(str(device) for device in device_map.values())
    summary = ", ".join(f"{device}:{count}" for device, count in sorted(counts.items()))
    print(f"[Smoke] hf_device_map_summary={summary}")

    for device_name in ("cuda:0", "cpu", "disk"):
        names = [name for name, device in device_map.items() if str(device) == device_name]
        if not names:
            continue
        examples = ", ".join(names[:5])
        print(f"[Smoke] device_map_{device_name.replace(':', '_')}_count={len(names)} examples={examples}")


def _print_peak_memory(torch: Any) -> None:
    if not torch.cuda.is_available():
        return
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[Smoke] cuda_peak_allocated_gb={peak_gb:.2f}")
    print(f"[Smoke] cuda_peak_reserved_gb={reserved_gb:.2f}")


def _build_content_repair_payload() -> dict[str, Any]:
    return {
        "schema": {"repairs": [{"node_id": "string", "text": "string", "confidence": "float"}]},
        "items": [
            {
                "node_id": "smoke_paragraph_1",
                "text": "한국어문장입 니다",
                "language": "korean",
                "role": "paragraph",
            }
        ],
        "constraint": "공백만 변경. 비공백 문자 시퀀스 완전 동일 유지.",
    }


def _is_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message or "cuda out of memory" in message


def _print_exception(label: str, error: BaseException) -> None:
    message = str(error) or "<empty>"
    error_type = f"{type(error).__module__}.{type(error).__name__}"
    print(f"[Smoke][{label}] type={error_type}")
    print(f"[Smoke][{label}] message={message}")
    print(f"[Smoke][{label}] repr={error!r}")
    print(f"[Smoke][{label}] traceback:")
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stdout)


if __name__ == "__main__":
    os.environ.setdefault("LLM_MAX_NEW_TOKENS", "128")
    os.environ.setdefault("LOCAL_LLM_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")
    os.environ.setdefault("LLM_ENRICHMENT_MODE", "all")
    raise SystemExit(main())
