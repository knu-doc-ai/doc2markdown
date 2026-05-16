from __future__ import annotations

"""LLM 응답 경계 parser."""

import math
from dataclasses import dataclass
from typing import Any


ALLOWED_SEMANTIC_KINDS = frozenset({"text", "heading", "caption", "note"})
MIN_CONFIDENCE = 0.5

SEMANTIC_DECISION_KEYS = ("semantic_decisions", "decisions", "blocks")
CAPTION_LINK_KEYS = ("caption_links", "caption_candidate_repairs", "links")
CONTENT_REPAIR_KEYS = ("repairs", "content_repairs", "items")


@dataclass(frozen=True)
class SemanticDecision:
    id: str
    kind: str
    heading_level: int | None
    confidence: float


@dataclass(frozen=True)
class CaptionLink:
    caption_id: str
    target_id: str
    confidence: float


@dataclass(frozen=True)
class ContentRepair:
    node_id: str
    text: str
    confidence: float


@dataclass(frozen=True)
class ParsedSemanticResponse:
    decisions: list[SemanticDecision]
    caption_links: list[CaptionLink]


def parse_semantic_response(response: Any) -> ParsedSemanticResponse:
    decisions: list[SemanticDecision] = []
    for item in _extract_list(response, SEMANTIC_DECISION_KEYS):
        decision = _parse_semantic_decision(item)
        if decision is not None:
            decisions.append(decision)

    caption_links: list[CaptionLink] = []
    for item in _extract_list(response, CAPTION_LINK_KEYS):
        link = _parse_caption_link(item)
        if link is not None:
            caption_links.append(link)

    return ParsedSemanticResponse(decisions=decisions, caption_links=caption_links)


def parse_content_repair(response: Any, node_id: str) -> ContentRepair | None:
    repairs = parse_content_repairs(response)

    for repair in repairs:
        if repair.node_id == node_id:
            return repair

    if len(repairs) == 1:
        return repairs[0]
    return None


def parse_content_repairs(response: Any) -> list[ContentRepair]:
    repairs: list[ContentRepair] = []
    for item in _extract_list(response, CONTENT_REPAIR_KEYS):
        repair = _parse_content_repair(item)
        if repair is not None:
            repairs.append(repair)
    return repairs


def _parse_semantic_decision(item: Any) -> SemanticDecision | None:
    if not isinstance(item, dict):
        return None

    element_id = _normalize_str(item.get("id"))
    kind = _normalize_str(item.get("kind"))
    confidence = _normalize_float(item.get("confidence"))
    if element_id is None or kind is None:
        return None

    normalized_kind = kind.lower()
    if normalized_kind not in ALLOWED_SEMANTIC_KINDS or confidence < MIN_CONFIDENCE:
        return None

    return SemanticDecision(
        id=element_id,
        kind=normalized_kind,
        heading_level=_normalize_int(item.get("heading_level")),
        confidence=confidence,
    )


def _parse_caption_link(item: Any) -> CaptionLink | None:
    if not isinstance(item, dict):
        return None

    caption_id = _normalize_str(item.get("caption_id"))
    target_id = _normalize_str(item.get("target_id"))
    confidence = _normalize_float(item.get("confidence"))
    if caption_id is None or target_id is None or confidence < MIN_CONFIDENCE:
        return None

    return CaptionLink(caption_id=caption_id, target_id=target_id, confidence=confidence)


def _parse_content_repair(item: Any) -> ContentRepair | None:
    if not isinstance(item, dict):
        return None

    node_id = _normalize_str(item.get("node_id"))
    text = _normalize_str(item.get("text"))
    confidence = _normalize_float(item.get("confidence"))
    if node_id is None or text is None or confidence < MIN_CONFIDENCE:
        return None

    return ContentRepair(node_id=node_id, text=text, confidence=confidence)


def _extract_list(response: Any, keys: tuple[str, ...]) -> list[Any]:
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        return []

    for key in keys:
        value = response.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_float(value: Any) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    return normalized if math.isfinite(normalized) else 0.0


__all__ = [
    "ALLOWED_SEMANTIC_KINDS",
    "CaptionLink",
    "ContentRepair",
    "ParsedSemanticResponse",
    "SemanticDecision",
    "parse_content_repair",
    "parse_content_repairs",
    "parse_semantic_response",
]
