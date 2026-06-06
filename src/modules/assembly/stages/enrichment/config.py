from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ENRICHMENT_MODE = "baseline"
VALID_ENRICHMENT_MODES = frozenset({"baseline", "semantic", "content", "all"})


@dataclass(frozen=True)
class LLMConfig:
    model_id: str = DEFAULT_MODEL_ID
    mode: str = DEFAULT_ENRICHMENT_MODE
    max_new_tokens: int = 512
    semantic_max_new_tokens: int | None = None
    content_max_new_tokens: int | None = None
    progress_log_interval: int = 0
    content_batch_size: int = 8
    content_min_chars: int = 12
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
            semantic_max_new_tokens=_env_optional_int("LLM_SEMANTIC_MAX_NEW_TOKENS"),
            content_max_new_tokens=_env_optional_int("LLM_CONTENT_MAX_NEW_TOKENS"),
            progress_log_interval=_env_int("LLM_PROGRESS_LOG_INTERVAL", default=0),
            content_batch_size=max(1, _env_int("LLM_CONTENT_BATCH_SIZE", default=8)),
            content_min_chars=max(0, _env_int("LLM_CONTENT_MIN_CHARS", default=12)),
            temperature=_env_float("LLM_TEMPERATURE", default=0.0),
        )

    def max_new_tokens_for_task(self, task: str) -> int:
        if task == "semantic_enrichment" and self.semantic_max_new_tokens is not None:
            return self.semantic_max_new_tokens
        if task == "content_repair" and self.content_max_new_tokens is not None:
            return self.content_max_new_tokens
        return self.max_new_tokens

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
    print(f"[Assembly][Config] LOCAL_LLM_MODEL_ID={config.model_id}")
    print(f"[Assembly][Config] LLM_MAX_NEW_TOKENS={config.max_new_tokens}")
    print(f"[Assembly][Config] LLM_SEMANTIC_MAX_NEW_TOKENS={config.max_new_tokens_for_task('semantic_enrichment')}")
    print(f"[Assembly][Config] LLM_CONTENT_MAX_NEW_TOKENS={config.max_new_tokens_for_task('content_repair')}")
    print(f"[Assembly][Config] LLM_CONTENT_BATCH_SIZE={config.content_batch_size}")
    print(f"[Assembly][Config] LLM_CONTENT_MIN_CHARS={config.content_min_chars}")


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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
