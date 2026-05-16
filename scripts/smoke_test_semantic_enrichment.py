from __future__ import annotations

"""제목/캡션 후보 재분류용 로컬 LLM smoke test."""

import importlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.assembly.ir import AssemblyElement, AssemblyMeta, AssemblyResult, AssembledDocument, TableRef
from modules.llm_core import LLMConfig, LLMGenerationError, LocalTransformersLLMClient
from modules.llm_enrichment import SemanticEnricher
from modules.llm_response_parser import parse_semantic_response


EXIT_IMPORT_FAILURE = 2
EXIT_OOM_FAILURE = 4
EXIT_JSON_FAILURE = 5
EXIT_GENERATION_FAILURE = 6


def main() -> int:
    _load_dotenv_if_available()
    os.environ["LLM_ENRICHMENT_MODE"] = "semantic"
    os.environ.setdefault("LLM_MAX_NEW_TOKENS", "128")
    os.environ.setdefault("LLM_SEMANTIC_MAX_NEW_TOKENS", "512")
    os.environ.setdefault("LOCAL_LLM_MODEL_ID", "Qwen/Qwen3-1.7B")

    config = LLMConfig.from_env()
    print("[Smoke][Semantic] semantic enrichment smoke test 시작")
    print(f"[Smoke][Semantic] model_id={config.model_id}")
    print(f"[Smoke][Semantic] enrichment_mode={config.mode}")
    print(f"[Smoke][Semantic] max_new_tokens={config.max_new_tokens}")
    print(f"[Smoke][Semantic] semantic_max_new_tokens={config.max_new_tokens_for_task('semantic_enrichment')}")

    imports = _check_imports()
    if imports is None:
        return EXIT_IMPORT_FAILURE

    torch = imports["torch"]
    _print_torch_runtime(torch)

    client = LocalTransformersLLMClient(config)
    payload = _build_semantic_payload()
    print(f"[Smoke][Semantic] candidates={len(payload['candidates'])}, objects={len(payload['objects'])}")
    print(f"[Smoke][Semantic] candidate_stats={payload.get('candidate_stats')}")
    for candidate in payload["candidates"]:
        print(
            "[Smoke][Semantic] candidate "
            f"id={candidate['id']}, hint={candidate.get('semantic_hint')}, "
            f"reason={candidate.get('semantic_reason')}, text={candidate['text']}"
        )

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        response = client.generate_json("semantic_enrichment", payload)
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

    print("[Smoke][Semantic] JSON 생성 성공")
    print(f"[Smoke][Semantic] raw_response={response}")

    parsed = parse_semantic_response(response)
    print(f"[Smoke][Semantic] parsed_decisions={len(parsed.decisions)}")
    print(f"[Smoke][Semantic] parsed_caption_links={len(parsed.caption_links)}")
    for decision in parsed.decisions:
        print(
            "[Smoke][Semantic] decision "
            f"id={decision.id}, kind={decision.kind}, "
            f"heading_level={decision.heading_level}, confidence={decision.confidence}"
        )
    for link in parsed.caption_links:
        print(
            "[Smoke][Semantic] caption_link "
            f"caption_id={link.caption_id}, target_id={link.target_id}, confidence={link.confidence}"
        )

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
            print(f"[Smoke][Semantic][Import Failure] {name} import 실패: {error}")
            return None
        modules[name] = module
        version = getattr(module, "__version__", "unknown")
        print(f"[Smoke][Semantic] {name}={version}")
    return modules


def _build_semantic_payload() -> dict[str, Any]:
    result = AssemblyResult(
        ordered_elements=[
            _candidate("b_heading_1", "text", "1.2 범위 및 제약사항", 1, 1),
            _candidate("b_para_1", "text", "본 문서는 웹 기반 계산기 애플리케이션의 요구사항을 정의합니다.", 1, 2),
            _candidate("b_heading_2", "text", "3.1 테일러 급수 전개 (공학 함수 근사)", 3, 1),
            _candidate("b_heading_3", "text", "5. 비기능 요구사항 (Non-Functional Requirements)", 5, 1),
            _candidate("b_caption_1", "text", "표 1. 계산기 주요 기능", 5, 2),
            _candidate("b_para_2", "text", "사용자는 숫자 버튼을 눌러 입력값을 구성할 수 있습니다.", 5, 3),
        ],
        document=AssembledDocument(
            table_refs=[TableRef(table_id="table_1", page=5, bbox=(80.0, 210.0, 520.0, 360.0))]
        ),
        metadata=AssemblyMeta(stage="normalized"),
    )
    return SemanticEnricher._build_semantic_payload(result)


def _candidate(block_id: str, kind: str, text: str, page: int, order: int) -> AssemblyElement:
    return AssemblyElement(
        id=block_id,
        page=page,
        kind=kind,
        text=text,
        bbox=(80.0, float(80 + order * 30), 520.0, float(105 + order * 30)),
        confidence=0.95,
        column_id=0,
        reading_order=order,
        label="Text",
    )


def _print_torch_runtime(torch: Any) -> None:
    cuda_available = torch.cuda.is_available()
    print(f"[Smoke][Semantic] cuda_available={cuda_available}")
    if not cuda_available:
        return

    device_index = torch.cuda.current_device()
    print(f"[Smoke][Semantic] cuda_device={torch.cuda.get_device_name(device_index)}")
    props = torch.cuda.get_device_properties(device_index)
    print(f"[Smoke][Semantic] cuda_total_memory_gb={props.total_memory / 1024**3:.2f}")


def _print_peak_memory(torch: Any) -> None:
    if not torch.cuda.is_available():
        return
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[Smoke][Semantic] cuda_peak_allocated_gb={peak_gb:.2f}")
    print(f"[Smoke][Semantic] cuda_peak_reserved_gb={reserved_gb:.2f}")


def _is_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message or "cuda out of memory" in message


def _print_exception(label: str, error: BaseException) -> None:
    message = str(error) or "<empty>"
    error_type = f"{type(error).__module__}.{type(error).__name__}"
    print(f"[Smoke][Semantic][{label}] type={error_type}")
    print(f"[Smoke][Semantic][{label}] message={message}")
    print(f"[Smoke][Semantic][{label}] repr={error!r}")
    print(f"[Smoke][Semantic][{label}] traceback:")
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
