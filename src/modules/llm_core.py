from __future__ import annotations

"""선택형 Markdown 보강용 로컬 오픈웨이트 LLM 유틸리티.

운영용 client는 지연 로딩 구조 유지.
테스트에서는 ``generate_json``을 가진 fake client 주입 가능.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ENRICHMENT_MODE = "baseline"
VALID_ENRICHMENT_MODES = frozenset({"baseline", "semantic", "content", "all"})


class LLMClient(Protocol):
    """보강 모듈과 테스트의 최소 client 규약."""

    @property
    def model_id(self) -> str:
        ...

    def generate_json(self, task: str, payload: dict[str, Any]) -> Any:
        ...


class LLMGenerationError(RuntimeError):
    """로컬 모델의 JSON 생성 실패."""


@dataclass(frozen=True)
class LLMConfig:
    model_id: str = DEFAULT_MODEL_ID
    mode: str = DEFAULT_ENRICHMENT_MODE
    max_new_tokens: int = 512
    progress_log_interval: int = 0
    content_batch_size: int = 8
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        mode = os.getenv("LLM_ENRICHMENT_MODE", DEFAULT_ENRICHMENT_MODE).strip().lower()
        if mode not in VALID_ENRICHMENT_MODES:
            mode = DEFAULT_ENRICHMENT_MODE

        return cls(
            model_id=os.getenv("LOCAL_LLM_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID,
            mode=mode,
            max_new_tokens=_env_int("LLM_MAX_NEW_TOKENS", default=512),
            progress_log_interval=_env_int("LLM_PROGRESS_LOG_INTERVAL", default=0),
            content_batch_size=max(1, _env_int("LLM_CONTENT_BATCH_SIZE", default=8)),
            temperature=_env_float("LLM_TEMPERATURE", default=0.0),
        )

    def runs_semantic(self) -> bool:
        return self.mode in {"semantic", "all"}

    def runs_content(self) -> bool:
        return self.mode in {"content", "all"}

    def uses_enrichment(self) -> bool:
        return self.mode != "baseline"


class LocalTransformersLLMClient:
    """Transformers 기반 로컬 LLM client와 JSON 전용 출력 파싱."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()
        self._tokenizer = None
        self._model = None

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def generate_json(self, task: str, payload: dict[str, Any]) -> Any:
        tokenizer, model = self._load_model()
        prompt = self._build_prompt(task, payload)
        messages = [
            {
                "role": "system",
                "content": (
                    "문서-Markdown 후처리 보조 역할. "
                    "반드시 유효한 JSON만 반환. 설명 문장과 markdown fence 제외."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        model_inputs = self._build_model_inputs(tokenizer, messages, model.device)
        input_ids = model_inputs["input_ids"]
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        stopping_criteria = self._build_progress_stopping_criteria(task, input_ids.shape[-1])
        if stopping_criteria is not None:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature

        output_ids = model.generate(**model_inputs, **generation_kwargs)
        generated = output_ids[0][input_ids.shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return parse_json_object(text)

    def _load_model(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as error:  # pragma: no cover - 로컬 환경 의존
            raise LLMGenerationError(f"Transformers 로컬 LLM 의존성 사용 불가: {error}") from error

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def _build_progress_stopping_criteria(self, task: str, prompt_tokens: int) -> Any | None:
        interval = self.config.progress_log_interval
        if interval <= 0:
            return None

        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            return None

        max_new_tokens = self.config.max_new_tokens

        class ProgressLogger(StoppingCriteria):
            def __init__(self):
                self.next_log_at = interval

            def __call__(self, input_ids, scores, **kwargs) -> bool:
                generated_tokens = max(0, int(input_ids.shape[-1]) - prompt_tokens)
                if generated_tokens >= self.next_log_at:
                    print(f"[LLM][Generate] {task}: generated_tokens≈{generated_tokens}/{max_new_tokens}")
                    while self.next_log_at <= generated_tokens:
                        self.next_log_at += interval
                return False

        return StoppingCriteriaList([ProgressLogger()])

    @staticmethod
    def _build_model_inputs(tokenizer: Any, messages: list[dict[str, str]], device: Any) -> dict[str, Any]:
        raw_inputs = LocalTransformersLLMClient._tokenize_messages(tokenizer, messages)
        return LocalTransformersLLMClient._normalize_model_inputs(raw_inputs, device)

    @staticmethod
    def _tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> Any:
        if not hasattr(tokenizer, "apply_chat_template"):
            return tokenizer(
                "\n".join(message["content"] for message in messages),
                return_tensors="pt",
            )

        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

    @staticmethod
    def _normalize_model_inputs(raw_inputs: Any, device: Any) -> dict[str, Any]:
        if hasattr(raw_inputs, "items"):
            return {
                key: LocalTransformersLLMClient._move_to_device(value, device)
                for key, value in raw_inputs.items()
            }

        input_ids = getattr(raw_inputs, "input_ids", None)
        if input_ids is not None:
            return {"input_ids": LocalTransformersLLMClient._move_to_device(input_ids, device)}

        return {"input_ids": LocalTransformersLLMClient._move_to_device(raw_inputs, device)}

    @staticmethod
    def _move_to_device(value: Any, device: Any) -> Any:
        if hasattr(value, "to"):
            return value.to(device)
        return value

    @staticmethod
    def _build_prompt(task: str, payload: dict[str, Any]) -> str:
        return (
            f"작업: {task}\n"
            "요청 schema 정확히 준수. 원문 의미 보존.\n"
            f"입력 JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )


class ContentEnricher:
    """modules.llm_core 기존 import 호환용 모의 객체."""

    def enrich(self, layout_elements, table_results=None, config=None):
        return layout_elements


def parse_json_object(text: str) -> Any:
    """모델 원문 응답에서 JSON 객체/배열 파싱."""
    if not isinstance(text, str) or not text.strip():
        raise LLMGenerationError("로컬 LLM 빈 응답 반환.")

    cleaned = _strip_code_fence(text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidate = _extract_balanced_json(cleaned)
    if candidate is None:
        raise LLMGenerationError("로컬 LLM 응답에 JSON 없음.")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise LLMGenerationError(f"로컬 LLM 응답의 JSON 파싱 실패: {error}") from error


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _extract_balanced_json(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return text[start:start + end]
    return None


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


__all__ = [
    "ContentEnricher",
    "DEFAULT_ENRICHMENT_MODE",
    "DEFAULT_MODEL_ID",
    "VALID_ENRICHMENT_MODES",
    "LLMClient",
    "LLMConfig",
    "LLMGenerationError",
    "LocalTransformersLLMClient",
    "parse_json_object",
]
