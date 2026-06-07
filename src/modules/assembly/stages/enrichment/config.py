from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_OLLAMA_MODEL_ID = "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0"
DEFAULT_ENRICHMENT_MODE = "baseline"
DEFAULT_BACKEND = "transformers"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OPENAI_API_KEY = "ollama"
DEFAULT_SEMANTIC_BATCH_SIZE = 8
DEFAULT_CONTENT_BATCH_SIZE = 8
DEFAULT_OLLAMA_SEMANTIC_BATCH_SIZE = 32
DEFAULT_OLLAMA_CONTENT_BATCH_SIZE = 2
VALID_ENRICHMENT_MODES = frozenset({"baseline", "semantic", "content", "all"})
VALID_BACKENDS = frozenset({"transformers", "ollama", "openai", "openai-compatible"})


@dataclass(frozen=True)
class LLMConfig:
    model_id: str = DEFAULT_MODEL_ID
    semantic_model_id: str | None = None
    content_model_id: str | None = None
    mode: str = DEFAULT_ENRICHMENT_MODE
    backend: str = DEFAULT_BACKEND
    api_base_url: str = DEFAULT_OPENAI_BASE_URL
    api_key: str = DEFAULT_OPENAI_API_KEY
    request_timeout: float = 60.0
    max_new_tokens: int = 512
    semantic_max_new_tokens: int | None = None
    content_max_new_tokens: int | None = None
    progress_log_interval: int = 0
    semantic_batch_size: int | None = None
    content_batch_size: int | None = None
    content_min_chars: int = 12
    temperature: float = 0.0

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower()
        if mode not in VALID_ENRICHMENT_MODES:
            mode = DEFAULT_ENRICHMENT_MODE
        object.__setattr__(self, "mode", mode)
        if self.semantic_model_id is None:
            object.__setattr__(self, "semantic_model_id", self.model_id)
        if self.content_model_id is None:
            object.__setattr__(self, "content_model_id", self.model_id)
        semantic_batch_size = self.semantic_batch_size
        if semantic_batch_size is None:
            semantic_batch_size = _default_semantic_batch_size(self.backend)
        object.__setattr__(self, "semantic_batch_size", max(1, semantic_batch_size))
        content_batch_size = self.content_batch_size
        if content_batch_size is None:
            content_batch_size = _default_content_batch_size(self.backend)
        object.__setattr__(self, "content_batch_size", max(1, content_batch_size))

    @classmethod
    def from_env(cls) -> "LLMConfig":
        mode = os.getenv("LLM_ENRICHMENT_MODE", DEFAULT_ENRICHMENT_MODE).strip().lower()
        if mode not in VALID_ENRICHMENT_MODES:
            mode = DEFAULT_ENRICHMENT_MODE
        backend = _resolve_backend()
        default_model_id = _default_model_id_for_backend(backend)
        semantic_model_id = _env_first(
            ("LOCAL_LLM_SEMANTIC_MODEL_ID", "LLM_SEMANTIC_MODEL_ID"),
            default=default_model_id,
        )
        content_model_id = _env_first(
            ("LOCAL_LLM_CONTENT_MODEL_ID", "LLM_CONTENT_MODEL_ID"),
            default=default_model_id,
        )
        model_id = _default_model_id_for_mode(mode, semantic_model_id, content_model_id)

        return cls(
            model_id=model_id,
            semantic_model_id=semantic_model_id,
            content_model_id=content_model_id,
            mode=mode,
            backend=backend,
            api_base_url=_env_first(
                ("LOCAL_LLM_BASE_URL", "LLM_BASE_URL"),
                default=DEFAULT_OPENAI_BASE_URL,
            ),
            api_key=_env_first(
                ("LOCAL_LLM_API_KEY", "LLM_API_KEY"),
                default=DEFAULT_OPENAI_API_KEY,
            ),
            request_timeout=_env_float("LLM_REQUEST_TIMEOUT_SECONDS", default=60.0),
            semantic_max_new_tokens=_env_int("LLM_SEMANTIC_MAX_NEW_TOKENS", default=128),
            content_max_new_tokens=_env_int("LLM_CONTENT_MAX_NEW_TOKENS", default=256),
            progress_log_interval=_env_int("LLM_PROGRESS_LOG_INTERVAL", default=0),
            semantic_batch_size=_env_optional_int("LLM_SEMANTIC_BATCH_SIZE"),
            content_batch_size=_env_optional_int("LLM_CONTENT_BATCH_SIZE"),
            content_min_chars=max(0, _env_int("LLM_CONTENT_MIN_CHARS", default=12)),
            temperature=_env_float("LLM_TEMPERATURE", default=0.0),
        )

    def max_new_tokens_for_task(self, task: str) -> int:
        if task == "semantic_enrichment" and self.semantic_max_new_tokens is not None:
            return self.semantic_max_new_tokens
        if task == "content_repair" and self.content_max_new_tokens is not None:
            return self.content_max_new_tokens
        return self.max_new_tokens

    def model_id_for_task(self, task: str) -> str:
        if task == "semantic_enrichment":
            return self.semantic_model_id
        if task == "content_repair":
            return self.content_model_id
        return self.model_id

    def enables_semantic(self) -> bool:
        return self.mode in {"semantic", "all"}

    def enables_content(self) -> bool:
        return self.mode in {"content", "all"}

    def uses_enrichment(self) -> bool:
        return self.mode != "baseline"

def print_enrichment_config(config: LLMConfig) -> None:
    if not config.uses_enrichment():
        print("[Assembly][Config] LLM_ENRICHMENT_MODE=baseline (disabled)")
        return

    print(f"[Assembly][Config] LLM_ENRICHMENT_MODE={config.mode}")
    print(f"[Assembly][Config] LOCAL_LLM_BACKEND={config.backend}")
    if config.enables_semantic():
        print(f"[Assembly][Config] LOCAL_LLM_SEMANTIC_MODEL_ID={config.semantic_model_id}")
        print(f"[Assembly][Config] LLM_SEMANTIC_MAX_NEW_TOKENS={config.max_new_tokens_for_task('semantic_enrichment')}")
        print(f"[Assembly][Config] LLM_SEMANTIC_BATCH_SIZE={config.semantic_batch_size}")
    if config.enables_content():
        print(f"[Assembly][Config] LOCAL_LLM_CONTENT_MODEL_ID={config.content_model_id}")
        print(f"[Assembly][Config] LLM_CONTENT_MAX_NEW_TOKENS={config.max_new_tokens_for_task('content_repair')}")
        print(f"[Assembly][Config] LLM_CONTENT_BATCH_SIZE={config.content_batch_size}")
        print(f"[Assembly][Config] LLM_CONTENT_MIN_CHARS={config.content_min_chars}")
    if config.backend != "transformers":
        print(f"[Assembly][Config] LOCAL_LLM_BASE_URL={config.api_base_url}")


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_first(names: tuple[str, ...], *, default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


def _resolve_backend() -> str:
    raw = _env_first(("LOCAL_LLM_BACKEND", "LLM_BACKEND"), default="")
    backend = raw.strip().lower()
    if backend in VALID_BACKENDS:
        return backend
    return DEFAULT_BACKEND


def _default_model_id_for_backend(backend: str) -> str:
    if backend == "ollama":
        return DEFAULT_OLLAMA_MODEL_ID
    return DEFAULT_MODEL_ID


def _default_model_id_for_mode(mode: str, semantic_model_id: str, content_model_id: str) -> str:
    if mode == "content":
        return content_model_id
    return semantic_model_id


def _default_semantic_batch_size(backend: str) -> int:
    if backend == "ollama":
        return DEFAULT_OLLAMA_SEMANTIC_BATCH_SIZE
    return DEFAULT_SEMANTIC_BATCH_SIZE


def _default_content_batch_size(backend: str) -> int:
    if backend == "ollama":
        return DEFAULT_OLLAMA_CONTENT_BATCH_SIZE
    return DEFAULT_CONTENT_BATCH_SIZE


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
