from __future__ import annotations

"""Assembly 입력 어댑터가 공유하는 adapter 전용 상수와 유틸."""

from typing import Any, Dict, List, Tuple

from modules.assembly._common import AssemblyCommonMixin
from modules.assembly.ir import AssemblyElement, AssemblyMeta, PageStats, TableRef
from modules.assembly.types import (
    AssemblyAdapterType,
    AssemblySourceType,
    AssemblyStage,
    AssemblyWarningCode,
)


class AssemblyAdapterCommon(AssemblyCommonMixin):
    """layout/table 어댑터가 공통으로 쓰는 adapter 전용 기반 클래스."""

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
    # 표 마크다운 본문을 담는 필드명 후보
    TABLE_MARKDOWN_KEYS: Tuple[str, ...] = (
        "markdown",
        "table_markdown",
        "md",
        "table_md",
    )
    # table crop/image 경로를 담는 필드명 후보
    TABLE_IMAGE_KEYS: Tuple[str, ...] = (
        "crop_path",
        "image_path",
        "table_image_path",
    )
    # figure asset metadata를 찾기 위한 문서 단위 필드명 후보
    FIGURE_ASSET_KEYS: Tuple[str, ...] = (
        "figure_assets_metadata",
        "figure_assets",
        "figure_asset_map",
    )
    # 문서 단위 상위 메타데이터 필드 후보
    DOCUMENT_METADATA_KEYS: Tuple[str, ...] = (
        "file_name",
        "file_type",
        "total_pages",
    )
    # upstream id가 없을 때 사용하는 내부 임시 ID prefix
    ELEMENT_FALLBACK_PREFIX: str = "element"
    TABLE_FALLBACK_PREFIX: str = "table"
    # 현재 어댑터 단계에서 사용하는 warning code 상수
    WARNING_LAYOUT_MISSING_ID: AssemblyWarningCode = "layout_missing_id"
    WARNING_LAYOUT_MISSING_PAGE: AssemblyWarningCode = "layout_missing_page"
    WARNING_TABLE_MISSING_ID: AssemblyWarningCode = "table_missing_id"
    WARNING_TABLE_MISSING_PAGE: AssemblyWarningCode = "table_missing_page"

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
    def _extract_source_block_ids(cls, payload: Dict[str, Any]) -> List[str]:
        """source block provenance를 표준 source_block_ids 형태로 정규화한다."""
        value = cls._pick_first(payload, cls.SOURCE_BLOCK_IDS_KEYS)
        return cls._normalize_id_list(value)

    @classmethod
    def _has_layout_shape(cls, raw: Any) -> bool:
        """입력이 layout payload처럼 생겼는지 판별한다."""
        if isinstance(raw, dict):
            return (
                cls._pick_first(raw, cls.PAGE_LIST_KEYS) is not None
                or cls._pick_first(raw, cls.ELEMENT_LIST_KEYS) is not None
                or cls._looks_like_element_entry(raw)
            )

        return cls._is_layout_sequence(raw)

    @classmethod
    def _has_table_shape(cls, raw: Any) -> bool:
        """입력이 table payload처럼 생겼는지 판별한다."""
        if cls._looks_like_markdown_table(raw):
            return True

        if isinstance(raw, dict):
            return (
                cls._pick_first(raw, cls.TABLE_LIST_KEYS) is not None
                or cls._pick_first(raw, cls.TABLE_MARKDOWN_KEYS) is not None
                or cls._looks_like_table_entry(raw)
            )

        return cls._is_table_sequence(raw)

    @classmethod
    def _is_layout_sequence(cls, raw: Any) -> bool:
        """list/tuple 입력이 layout entry 시퀀스인지 판별한다."""
        if not isinstance(raw, (list, tuple)) or not raw:
            return False

        return any(cls._looks_like_element_entry(item) for item in raw)

    @classmethod
    def _is_table_sequence(cls, raw: Any) -> bool:
        """list/tuple 입력이 table entry 시퀀스인지 판별한다."""
        if not isinstance(raw, (list, tuple)) or not raw:
            return False

        return any(
            cls._looks_like_table_entry(item) or cls._looks_like_markdown_table(item)
            for item in raw
        )

    @classmethod
    def _looks_like_element_entry(cls, raw: Any) -> bool:
        """dict 하나가 layout element처럼 보이는지 판별한다."""
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
        """dict 하나가 table entry처럼 보이는지 판별한다."""
        if isinstance(raw, TableRef):
            return True

        if cls._looks_like_markdown_table(raw):
            return True

        if not isinstance(raw, dict):
            return False

        markdown_candidate = cls._pick_first(raw, cls.TABLE_MARKDOWN_KEYS)
        if cls._looks_like_markdown_table(markdown_candidate):
            return True

        return any(
            key in raw
            for key in (
                *cls.TABLE_ID_KEYS,
                *cls.CAPTION_KEYS,
                *cls.NOTE_KEYS,
                *cls.TABLE_STRUCTURE_KEYS,
            )
        )


__all__ = ["AssemblyAdapterCommon"]
