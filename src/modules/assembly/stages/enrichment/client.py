from __future__ import annotations

from typing import Any, Protocol

from modules.assembly.stages.enrichment.config import LLMConfig
from modules.assembly.stages.enrichment.json_parser import LLMGenerationError, parse_json_object
from modules.assembly.stages.enrichment.prompts import build_prompt


class LLMClient(Protocol):
    """보강 모듈과 테스트의 최소 client 규약."""

    @property
    def model_id(self) -> str:
        ...

    def generate_json(self, task: str, payload: dict[str, Any]) -> Any:
        ...


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
        prompt = build_prompt(task, payload)
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
        max_new_tokens = self.config.max_new_tokens_for_task(task)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        stopping_criteria = self._build_progress_stopping_criteria(task, input_ids.shape[-1], max_new_tokens)
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

    def _build_progress_stopping_criteria(self, task: str, prompt_tokens: int, max_new_tokens: int) -> Any | None:
        interval = self.config.progress_log_interval
        if interval <= 0:
            return None

        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            return None

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

