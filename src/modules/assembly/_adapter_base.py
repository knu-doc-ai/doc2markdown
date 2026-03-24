from __future__ import annotations

"""Assembly 입력 어댑터가 공유하는 상수와 정규화 유틸."""

import re
from typing import Any, Dict, List, Optional, Tuple

from modules.assembly.ir import AssemblyElement, AssemblyMeta, PageStats, TableRef
from modules.assembly.types import (
    AssemblyAdapterType,
    AssemblyElementKind,
    AssemblySourceType,
    AssemblyStage,
    AssemblyWarningCode,
    BBox,
)


class AssemblyAdapterCommon:
    """layout/table 어댑터가 공통으로 쓰는 기반 클래스."""

    # 복합 payload 안에서 layout 결과를 찾을 때 먼저 보는 최상위 키 후보
    LAYOUT_CONTAINER_KEYS: Tuple[str, ...] = (
        "layout_output",
        "layout_result",
        "layout",
        "vision_output",
        "vision_result",
        "ordered_layout",
    )
    # 복합 payload 안에서 table 결과를 찾을 때 먼저 보는 최상위 키 후보
    TABLE_CONTAINER_KEYS: Tuple[str, ...] = (
        "table_output",
        "table_result",
        "table_results",
        "tables_output",
        "tables_result",
    )
    # layout block 리스트를 찾을 때 순서대로 탐색할 필드명 후보
    ELEMENT_LIST_KEYS: Tuple[str, ...] = (
        "ordered_elements",
        "elements",
        "blocks",
        "items",
        "layout_elements",
        "regions",
    )
    # 페이지 단위 payload 묶음을 찾기 위한 리스트 필드명 후보
    PAGE_LIST_KEYS: Tuple[str, ...] = (
        "pages",
        "page_results",
        "document_pages",
    )
    # 페이지 통계가 별도 구조로 전달될 때 사용하는 필드명 후보
    PAGE_STATS_KEYS: Tuple[str, ...] = (
        "page_stats",
        "page_statistics",
        "statistics",
    )
    # page 번호를 뜻하는 필드명 후보
    PAGE_NUMBER_KEYS: Tuple[str, ...] = (
        "page",
        "page_num",
        "page_number",
    )
    # 복합 payload 안에서 문서 루트 성격의 중첩 컨테이너를 찾기 위한 키 후보
    DOCUMENT_CONTAINER_KEYS: Tuple[str, ...] = ("document",)
    # table 항목 리스트를 찾기 위한 필드명 후보
    TABLE_LIST_KEYS: Tuple[str, ...] = (
        "table_refs",
        "tables",
        "results",
        "table_results",
        "table_ir",
        "items",
    )
    # element 고유 식별자를 찾기 위한 필드명 후보
    ELEMENT_ID_KEYS: Tuple[str, ...] = ("id", "element_id", "block_id", "uuid")
    # table 고유 식별자를 찾기 위한 필드명 후보
    TABLE_ID_KEYS: Tuple[str, ...] = ("table_id", "id", "uuid", "table_key")
    # label/type/class 값을 찾기 위한 필드명 후보
    ELEMENT_LABEL_KEYS: Tuple[str, ...] = (
        "label",
        "type",
        "kind",
        "class",
        "category",
        "role",
    )
    # bbox 좌표를 찾기 위한 필드명 후보
    BBOX_KEYS: Tuple[str, ...] = ("bbox", "box", "bounds", "rect")
    # 텍스트 본문을 찾기 위한 필드명 후보
    TEXT_KEYS: Tuple[str, ...] = ("text", "content", "ocr_text", "raw_text", "value")
    # confidence score를 찾기 위한 필드명 후보
    CONFIDENCE_KEYS: Tuple[str, ...] = ("confidence", "score", "probability")
    # column 번호를 찾기 위한 필드명 후보
    COLUMN_KEYS: Tuple[str, ...] = ("column_id", "column")
    # reading order 인덱스를 찾기 위한 필드명 후보
    READING_ORDER_KEYS: Tuple[str, ...] = ("reading_order", "order", "index")
    # parent 참조를 찾기 위한 필드명 후보
    PARENT_KEYS: Tuple[str, ...] = ("parent_id", "parent", "section_id")
    # provenance block id를 찾기 위한 필드명 후보
    SOURCE_BLOCK_IDS_KEYS: Tuple[str, ...] = ("source_block_ids", "block_ids", "source_blocks")
    # page 통계 필드 후보
    PAGE_WIDTH_KEYS: Tuple[str, ...] = ("width", "page_width")
    PAGE_HEIGHT_KEYS: Tuple[str, ...] = ("height", "page_height")
    LINE_HEIGHT_KEYS: Tuple[str, ...] = ("median_line_height", "line_height", "avg_line_height")
    BODY_FONT_SIZE_KEYS: Tuple[str, ...] = ("body_font_size", "font_size", "avg_font_size")
    COLUMN_COUNT_KEYS: Tuple[str, ...] = ("column_count", "columns", "num_columns")
    # table caption / note 참조를 찾기 위한 필드명 후보
    CAPTION_KEYS: Tuple[str, ...] = ("caption_id", "caption", "caption_ref")
    NOTE_KEYS: Tuple[str, ...] = ("note_ids", "notes", "note_refs")
    # table 구조 자체를 뜻하는 필드명 후보
    TABLE_STRUCTURE_KEYS: Tuple[str, ...] = ("cells", "rows", "columns")
    # figure asset metadata를 찾기 위한 문서 단위 필드명 후보
    FIGURE_ASSET_KEYS: Tuple[str, ...] = (
        "figure_assets_metadata",
        "figure_assets",
        "figure_asset_map",
    )
    # note/caption 참조 객체 내부의 id 필드 후보
    REF_ID_KEYS: Tuple[str, ...] = ("id", "note_id", "caption_id", "uuid")
    # upstream id가 없을 때 사용하는 내부 임시 ID prefix
    ELEMENT_FALLBACK_PREFIX: str = "element"
    TABLE_FALLBACK_PREFIX: str = "table"
    # 현재 어댑터 단계에서 사용하는 warning code 상수
    WARNING_LAYOUT_MISSING_ID: AssemblyWarningCode = "layout_missing_id"
    WARNING_LAYOUT_MISSING_PAGE: AssemblyWarningCode = "layout_missing_page"
    WARNING_TABLE_MISSING_ID: AssemblyWarningCode = "table_missing_id"
    WARNING_TABLE_MISSING_PAGE: AssemblyWarningCode = "table_missing_page"
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
    def _build_adapter_metadata(
        cls,
        stage: AssemblyStage,
        adapter: AssemblyAdapterType,
        source: AssemblySourceType,
        **extra: Any,
    ) -> AssemblyMeta:
        """어댑터 단계 메타데이터를 일관된 구조로 구성한다."""
        return AssemblyMeta(stage=stage, adapter=adapter, source=source, details=extra)

    @classmethod
    def _merge_ref_list(cls, primary: List[Any], secondary: List[Any], id_attr: str) -> List[Any]:
        """같은 id를 가진 ref는 뒤쪽 입력으로 덮어쓰며 병합한다."""
        merged: Dict[str, Any] = {}

        for item in list(primary) + list(secondary):
            ref_id = getattr(item, id_attr, None)
            if ref_id is None:
                continue
            merged[str(ref_id)] = item

        return list(merged.values())

    @classmethod
    def _merge_page_stats(cls, current: PageStats, incoming: PageStats) -> PageStats:
        """명시적 통계와 page 메타데이터를 보수적으로 합친다."""
        return PageStats(
            page=current.page,
            width=current.width if current.width is not None else incoming.width,
            height=current.height if current.height is not None else incoming.height,
            median_line_height=(
                current.median_line_height
                if current.median_line_height is not None
                else incoming.median_line_height
            ),
            body_font_size=(
                current.body_font_size
                if current.body_font_size is not None
                else incoming.body_font_size
            ),
            column_count=current.column_count if current.column_count is not None else incoming.column_count,
            metadata={**incoming.metadata, **current.metadata},
            raw=current.raw if current.raw is not None else incoming.raw,
        )

    @classmethod
    def _pick_first(cls, payload: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    @classmethod
    def _make_element_fallback_id(cls, index: int) -> str:
        """페이지 정보가 없을 때 사용하는 element 임시 ID 규칙."""
        return f"{cls.ELEMENT_FALLBACK_PREFIX}_{index}"

    @classmethod
    def _make_page_element_fallback_id(cls, page: int, index: int) -> str:
        """페이지 내 element용 임시 ID 규칙."""
        return f"p{page}_e{index}"

    @classmethod
    def _make_table_fallback_id(cls, index: int) -> str:
        """table 임시 ID 규칙."""
        return f"{cls.TABLE_FALLBACK_PREFIX}_{index}"

    @classmethod
    def _coerce_list(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @classmethod
    def _normalize_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        if not text:
            return None

        return re.sub(r"\s+", " ", text)

    @classmethod
    def _normalize_kind(cls, value: str) -> AssemblyElementKind:
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return cls.KIND_ALIASES.get(normalized, "text")

    @classmethod
    def _normalize_bbox(cls, value: Any) -> Optional[BBox]:
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
        if value is None or value == "":
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _normalize_int(cls, value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None or value == "":
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _normalize_str(cls, value: Any) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        return text or None

    @classmethod
    def _normalize_id_list(cls, value: Any) -> List[str]:
        items = cls._coerce_list(value)
        normalized: List[str] = []

        for item in items:
            candidate = cls._normalize_ref_id(item)
            if candidate is not None:
                normalized.append(candidate)

        return normalized

    @classmethod
    def _normalize_ref_id(cls, value: Any) -> Optional[str]:
        if isinstance(value, dict):
            return cls._normalize_str(cls._pick_first(value, cls.REF_ID_KEYS))

        return cls._normalize_str(value)

    @classmethod
    def _extract_source_block_ids(cls, payload: Dict[str, Any]) -> List[str]:
        """source block provenance를 표준 source_block_ids 형태로 정규화한다."""
        value = cls._pick_first(payload, cls.SOURCE_BLOCK_IDS_KEYS)
        return cls._normalize_id_list(value)

    @classmethod
    def _extract_metadata(cls, payload: Dict[str, Any], known_keys: set[str]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in known_keys and value is not None
        }

    @classmethod
    def _has_layout_shape(cls, raw: Any) -> bool:
        if isinstance(raw, dict):
            return (
                cls._pick_first(raw, cls.PAGE_LIST_KEYS) is not None
                or cls._pick_first(raw, cls.ELEMENT_LIST_KEYS) is not None
                or cls._looks_like_element_entry(raw)
            )

        return cls._is_layout_sequence(raw)

    @classmethod
    def _has_table_shape(cls, raw: Any) -> bool:
        if isinstance(raw, dict):
            if cls._pick_first(raw, cls.TABLE_LIST_KEYS) is not None:
                return True

            document_payload = cls._pick_first(raw, cls.DOCUMENT_CONTAINER_KEYS)
            if isinstance(document_payload, dict) and cls._pick_first(document_payload, cls.TABLE_LIST_KEYS):
                return True

            return cls._looks_like_table_entry(raw)

        return cls._is_table_sequence(raw)

    @classmethod
    def _is_layout_sequence(cls, raw: Any) -> bool:
        if not isinstance(raw, (list, tuple)) or not raw:
            return False

        return any(isinstance(item, AssemblyElement) or cls._looks_like_element_entry(item) for item in raw)

    @classmethod
    def _is_table_sequence(cls, raw: Any) -> bool:
        if not isinstance(raw, (list, tuple)) or not raw:
            return False

        return all(isinstance(item, TableRef) or cls._looks_like_table_entry(item) for item in raw)

    @classmethod
    def _looks_like_element_entry(cls, raw: Any) -> bool:
        if isinstance(raw, AssemblyElement):
            return True

        if not isinstance(raw, dict):
            return False

        return any(
            key in raw
            for key in (
                *cls.TEXT_KEYS,
                *cls.BBOX_KEYS,
                *cls.ELEMENT_LABEL_KEYS,
            )
        )

    @classmethod
    def _looks_like_table_entry(cls, raw: Any) -> bool:
        if isinstance(raw, TableRef):
            return True

        if not isinstance(raw, dict):
            return False

        return any(
            key in raw
            for key in (
                *cls.TABLE_ID_KEYS,
                *cls.CAPTION_KEYS,
                *cls.NOTE_KEYS,
                *cls.TABLE_STRUCTURE_KEYS,
            )
        )
