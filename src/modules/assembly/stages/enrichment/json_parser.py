from __future__ import annotations

import json
import re
from typing import Any


class LLMGenerationError(RuntimeError):
    """로컬 모델의 JSON 생성 실패."""


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
