from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

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
        self._models: dict[str, tuple[Any, Any]] = {}

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def generate_json(self, task: str, payload: dict[str, Any]) -> Any:
        tokenizer, model = self._load_model(self.config.model_id_for_task(task))
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

    def _load_model(self, model_id: str | None = None):
        resolved_model_id = model_id or self.config.model_id
        if resolved_model_id == self.config.model_id and self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        cached = self._models.get(resolved_model_id)
        if cached is not None:
            return cached

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as error:  # pragma: no cover - 로컬 환경 의존
            raise LLMGenerationError(f"Transformers 로컬 LLM 의존성 사용 불가: {error}") from error

        tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        self._models[resolved_model_id] = (tokenizer, model)
        if resolved_model_id == self.config.model_id:
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


class OpenAICompatibleLLMClient:
    """OpenAI-compatible chat client for local servers such as Ollama or llama.cpp."""

    def __init__(self, config: LLMConfig | None = None, client: Any | None = None):
        self.config = config or LLMConfig.from_env()
        self._client = client
        self._validate_base_url()

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def generate_json(self, task: str, payload: dict[str, Any]) -> Any:
        prompt = build_prompt(task, payload)
        try:
            response = self._load_client().chat.completions.create(
                model=self._request_model_id(task),
                messages=_build_messages(prompt),
                temperature=self.config.temperature,
                max_tokens=self.config.max_new_tokens_for_task(task),
            )
        except Exception as error:
            if error.__class__.__name__ == "APIConnectionError":
                raise LLMGenerationError(self._connection_error_message(task)) from error
            raise
        text = _extract_chat_content(response)
        return parse_json_object(text)

    def _validate_base_url(self) -> None:
        if self.config.backend != "ollama":
            return

        parsed = urlparse(self.config.api_base_url)
        host = parsed.hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise LLMGenerationError(
                "Ollama backend only allows local base URLs. "
                "Set LOCAL_LLM_BASE_URL to http://127.0.0.1:11434/v1."
            )

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except Exception as error:  # pragma: no cover - local environment guard
            raise LLMGenerationError(f"OpenAI-compatible LLM client unavailable: {error}") from error

        self._client = OpenAI(
            base_url=self.config.api_base_url,
            api_key=self.config.api_key,
            timeout=self.config.request_timeout,
        )
        return self._client

    def _request_model_id(self, task: str) -> str:
        model_id = self.config.model_id_for_task(task)
        if self.config.backend != "ollama":
            return model_id
        return _normalize_ollama_hf_model_id(model_id)

    def _connection_error_message(self, task: str) -> str:
        if self.config.backend == "ollama":
            return (
                f"Local Ollama server is unreachable at {self.config.api_base_url}. "
                f"Start it with: ollama run {self._request_model_id(task)}"
            )
        return f"OpenAI-compatible LLM server is unreachable at {self.config.api_base_url}."


class OllamaLLMClient:
    """Native Ollama chat client with thinking disabled and JSON mode enabled."""

    def __init__(self, config: LLMConfig | None = None, transport: Any | None = None):
        self.config = config or LLMConfig.from_env()
        self._transport = transport or self._post_json
        self._validate_base_url()

    @property
    def model_id(self) -> str:
        return self.config.model_id

    def generate_json(self, task: str, payload: dict[str, Any]) -> Any:
        num_predict = self._num_predict(task)
        text = self._generate_text(task, payload, num_predict)
        try:
            return parse_json_object(text)
        except LLMGenerationError as error:
            retry_tokens = max(num_predict * 2, 512)
            if retry_tokens > num_predict:
                text = self._generate_text(task, payload, retry_tokens)
                try:
                    return parse_json_object(text)
                except LLMGenerationError as retry_error:
                    raise self._invalid_json_error(text) from retry_error
            raise self._invalid_json_error(text) from error

    def _generate_text(self, task: str, payload: dict[str, Any], num_predict: int) -> str:
        body = {
            "model": _normalize_ollama_hf_model_id(self.config.model_id_for_task(task)),
            "messages": _build_messages(build_prompt(task, payload)),
            "think": False,
            "format": _ollama_format_schema(task),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": num_predict,
            },
        }
        response = self._transport(self._chat_url(), body, self.config.request_timeout)
        return _extract_ollama_chat_content(response)

    def _num_predict(self, task: str) -> int:
        configured = self.config.max_new_tokens_for_task(task)
        minimum = 512 if task == "content_repair" else 256
        return max(configured, minimum)

    @staticmethod
    def _invalid_json_error(text: str) -> LLMGenerationError:
        excerpt = " ".join(text.strip().split())[:240] if isinstance(text, str) else repr(text)
        return LLMGenerationError(f"Ollama LLM returned invalid JSON. response_excerpt={excerpt!r}")

    def _validate_base_url(self) -> None:
        parsed = urlparse(self.config.api_base_url)
        host = parsed.hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise LLMGenerationError(
                "Ollama backend only allows local base URLs. "
                "Set LOCAL_LLM_BASE_URL to http://127.0.0.1:11434/v1."
            )

    def _chat_url(self) -> str:
        parsed = urlparse(self.config.api_base_url)
        return urlunparse((parsed.scheme, parsed.netloc, "/api/chat", "", "", ""))

    def _post_json(self, url: str, body: dict[str, Any], timeout: float) -> Any:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            model_id = str(body.get("model") or self.config.model_id)
            try:
                response_text = error.read().decode("utf-8", errors="replace")
            except Exception:
                response_text = ""
            excerpt = " ".join(response_text.split())[:240]
            raise LLMGenerationError(
                f"Ollama request failed at {self.config.api_base_url}: "
                f"status={error.code}, model={model_id}, response_excerpt={excerpt!r}"
            ) from error
        except URLError as error:
            model_id = str(body.get("model") or self.config.model_id)
            raise LLMGenerationError(
                f"Local Ollama server is unreachable at {self.config.api_base_url}. "
                f"Start it with: ollama run {model_id}"
            ) from error


def create_llm_client(config: LLMConfig | None = None) -> LLMClient:
    resolved = config or LLMConfig.from_env()
    if resolved.backend == "ollama":
        return OllamaLLMClient(resolved)
    if resolved.backend in {"openai", "openai-compatible"}:
        return OpenAICompatibleLLMClient(resolved)
    return LocalTransformersLLMClient(resolved)


def _build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Document-to-Markdown preprocessing assistant. "
                "Return valid JSON only. Do not include explanations or markdown fences."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _ollama_format_schema(task: str) -> dict[str, Any] | str:
    if task == "semantic_enrichment":
        return {
            "type": "object",
            "properties": {
                "semantic_decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "kind": {"type": "string", "enum": ["text", "heading", "caption", "note"]},
                            "heading_level": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                            "confidence": {"type": "number"},
                        },
                        "required": ["id", "kind", "heading_level", "confidence"],
                        "additionalProperties": False,
                    },
                },
                "caption_links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "caption_id": {"type": "string"},
                            "target_id": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["caption_id", "target_id", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["semantic_decisions", "caption_links"],
            "additionalProperties": False,
        }
    if task == "content_repair":
        return {
            "type": "object",
            "properties": {
                "repairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["node_id", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["repairs"],
            "additionalProperties": False,
        }
    return "json"


def _extract_chat_content(response: Any) -> str:
    choices = _get_attr_or_item(response, "choices")
    if not choices:
        raise LLMGenerationError("OpenAI-compatible LLM returned no choices.")

    first_choice = choices[0]
    message = _get_attr_or_item(first_choice, "message")
    content = _get_attr_or_item(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise LLMGenerationError("OpenAI-compatible LLM returned empty content.")
    return content


def _extract_ollama_chat_content(response: Any) -> str:
    message = _get_attr_or_item(response, "message")
    content = _get_attr_or_item(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise LLMGenerationError("Ollama LLM returned empty content.")
    return content


def _get_attr_or_item(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _normalize_ollama_hf_model_id(model_id: str) -> str:
    if model_id.startswith("hf.co/"):
        return model_id
    if "gguf" in model_id.lower() and "/" in model_id and ":" in model_id:
        return f"hf.co/{model_id}"
    return model_id

