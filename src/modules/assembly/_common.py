from __future__ import annotations

"""Assembly 내부 단계가 함께 쓰는 공용 정규화 유틸."""

import re
from typing import Any, Dict, List, Optional, Tuple

from modules.assembly.types import AssemblyElementKind, BBox


class AssemblyCommonMixin:
    """adapter와 후속 단계가 공통으로 쓰는 정규화 유틸 모음."""

    # note/caption 참조 객체 내부의 id 필드 후보
    REF_ID_KEYS: Tuple[str, ...] = ("id", "note_id", "caption_id", "uuid")

    # 다양한 label/type 값을 AssemblyElementKind로 정규화하기 위한 매핑
    KIND_ALIASES: Dict[str, AssemblyElementKind] = {
        "text": "text",
        "paragraph": "text",
        "body": "text",
        "heading": "heading",
        "title": "heading",
        "section_header": "heading",
        "list_item": "list_item",
        "list": "list_item",
        "bullet": "list_item",
        "table": "table",
        "figure": "figure",
        "picture": "figure",
        "image": "figure",
        "caption": "caption",
        "note": "note",
        "footnote": "note",
        "formula": "formula",
        "equation": "formula",
        "quote": "quote",
        "blockquote": "quote",
        "code": "code_block",
        "code_block": "code_block",
        "header": "header",
        "page_header": "header",
        "footer": "footer",
        "page_footer": "footer",
        "page_number": "page_number",
        "noise": "noise",
        "artifact": "noise",
    }

    @classmethod
    def _pick_first(cls, payload: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        """주어진 key 후보 중 먼저 발견되는 값을 반환한다."""
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    @classmethod
    def _coerce_list(cls, value: Any) -> List[Any]:
        """단일 값/tuple/list를 일관된 list로 맞춘다."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @classmethod
    def _normalize_text(cls, value: Any) -> Optional[str]:
        """문자열 양끝 공백과 내부 중복 공백을 정리한다."""
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        return re.sub(r"\s+", " ", text)

    @classmethod
    def _normalize_kind(cls, value: str) -> AssemblyElementKind:
        """upstream label/type 값을 내부 표준 kind로 맞춘다."""
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return cls.KIND_ALIASES.get(normalized, "text")

    @classmethod
    def _normalize_bbox(cls, value: Any) -> Optional[BBox]:
        """여러 bbox 표현을 [x1, y1, x2, y2] 튜플로 정규화한다."""
        if value is None:
            return None

        if isinstance(value, dict):
            if {"x1", "y1", "x2", "y2"}.issubset(value.keys()):
                coords = [value["x1"], value["y1"], value["x2"], value["y2"]]
            elif {"left", "top", "right", "bottom"}.issubset(value.keys()):
                coords = [value["left"], value["top"], value["right"], value["bottom"]]
            elif {"x", "y", "width", "height"}.issubset(value.keys()):
                x = cls._normalize_float(value["x"])
                y = cls._normalize_float(value["y"])
                width = cls._normalize_float(value["width"])
                height = cls._normalize_float(value["height"])
                if None in (x, y, width, height):
                    return None
                coords = [x, y, x + width, y + height]
            else:
                return None
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            coords = list(value)
        else:
            return None

        normalized = [cls._normalize_float(item) for item in coords]
        if any(item is None for item in normalized):
            return None

        return (
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
        )

    @classmethod
    def _normalize_float(cls, value: Any) -> Optional[float]:
        """float 해석이 가능한 값을 float로 맞춘다."""
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_int(cls, value: Any, default: Optional[int] = None) -> Optional[int]:
        """int 해석이 가능한 값을 int로 맞춘다."""
        if value is None or value == "":
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _normalize_str(cls, value: Any) -> Optional[str]:
        """None/빈 문자열을 제외한 문자열 값만 남긴다."""
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    @classmethod
    def _normalize_id_list(cls, value: Any) -> List[str]:
        """여러 id 표현을 문자열 list로 정규화한다."""
        items = cls._coerce_list(value)
        normalized: List[str] = []

        for item in items:
            candidate = cls._normalize_ref_id(item)
            if candidate is not None:
                normalized.append(candidate)

        return normalized

    @classmethod
    def _normalize_ref_id(cls, value: Any) -> Optional[str]:
        """ref id가 dict/원시값 어느 형태든 문자열 id로 정규화한다."""
        if isinstance(value, dict):
            return cls._normalize_str(cls._pick_first(value, cls.REF_ID_KEYS))

        return cls._normalize_str(value)

    @classmethod
    def _looks_like_markdown_table(cls, value: Any) -> bool:
        """문자열이 markdown table 본문인지 느슨하게 판별한다."""
        text = cls._normalize_str(value)
        if text is None:
            return False

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        header_line = lines[0]
        divider_line = lines[1]
        if "|" not in header_line or "|" not in divider_line:
            return False

        divider_chars = (
            divider_line
            .replace("|", "")
            .replace(":", "")
            .replace("-", "")
            .replace(" ", "")
        )
        return divider_chars == ""

    @classmethod
    def _normalize_markdown_table(cls, value: Any) -> Optional[str]:
        """표 markdown이면 정규화된 본문만 반환한다."""
        text = cls._normalize_str(value)
        if text is None or not cls._looks_like_markdown_table(text):
            return None
        return text

    @classmethod
    def _extract_metadata(cls, payload: Dict[str, Any], known_keys: set[str]) -> Dict[str, Any]:
        """이미 소비한 key를 제외한 나머지를 metadata로 보존한다."""
        return {
            key: value
            for key, value in payload.items()
            if key not in known_keys and value is not None
        }

    @classmethod
    def _merge_unique_ids(cls, *values: Any) -> List[str]:
        """여러 id 목록을 순서를 유지하면서 합친다."""
        merged: Dict[str, None] = {}

        for value in values:
            for item in cls._coerce_list(value):
                candidate = cls._normalize_ref_id(item)
                if candidate is not None:
                    merged[candidate] = None

        return list(merged.keys())

    @classmethod
    def _bbox_iou(cls, left: Optional[BBox], right: Optional[BBox]) -> Optional[float]:
        """두 bbox의 IoU를 계산한다."""
        if left is None or right is None:
            return None

        left_x1, left_y1, left_x2, left_y2 = left
        right_x1, right_y1, right_x2, right_y2 = right

        inter_x1 = max(left_x1, right_x1)
        inter_y1 = max(left_y1, right_y1)
        inter_x2 = min(left_x2, right_x2)
        inter_y2 = min(left_y2, right_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        left_area = max(0.0, (left_x2 - left_x1) * (left_y2 - left_y1))
        right_area = max(0.0, (right_x2 - right_x1) * (right_y2 - right_y1))
        union = left_area + right_area - intersection
        if union <= 0:
            return 0.0

        return intersection / union


__all__ = ["AssemblyCommonMixin"]
