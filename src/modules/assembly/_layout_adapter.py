from __future__ import annotations

"""layout 출력 전용 Assembly 어댑터."""

from typing import Any, Dict, List, Optional, Tuple

from modules.assembly._adapter_base import AssemblyAdapterCommon
from modules.assembly.ir import (
    AssemblyElement,
    AssemblyResult,
    AssemblyWarning,
    AssembledDocument,
    FigureRef,
    NoteRef,
    PageStats,
    TableRef,
)
from modules.assembly.types import AssemblySourceType


class LayoutAdapterMixin(AssemblyAdapterCommon):
    """layout payload를 표준 IR로 바꾸는 로직 모음."""

    @classmethod
    def from_layout_output(cls, raw: Any) -> AssemblyResult:
        """layout 계열 출력을 Assembly IR로 정규화한다."""
        if isinstance(raw, AssemblyResult):
            return raw

        if raw is None:
            return AssemblyResult(
                metadata=cls._build_adapter_metadata(
                    stage="adapter_seed",
                    adapter="layout",
                    source="empty",
                ),
                raw=raw,
            )

        # `pages` 해석은 아래 helper가 맡는다.
        # 이 함수는 page 통계, block 목록, 제목 후보, provisional ref를 순서대로 조립하는
        # layout seed 조립 흐름에만 집중한다.
        page_stats, page_warnings = cls._extract_page_stats(raw)
        elements, element_warnings = cls._extract_layout_elements(raw)
        title_candidate, title_source_block_ids = cls._infer_title_candidate(elements)
        figure_assets_metadata = cls._extract_figure_assets_metadata(raw)
        table_refs, figure_refs, note_refs = cls._extract_object_refs_from_elements(
            elements,
            figure_assets_metadata,
        )

        return AssemblyResult(
            ordered_elements=elements,
            document=AssembledDocument(
                title_candidate=title_candidate,
                title_source_block_ids=title_source_block_ids,
                table_refs=table_refs,
                figure_refs=figure_refs,
                note_refs=note_refs,
                figure_assets_metadata=figure_assets_metadata,
                raw=raw,
            ),
            page_stats=page_stats,
            warnings=page_warnings + element_warnings,
            metadata=cls._build_adapter_metadata(
                stage="adapter_seed",
                adapter="layout",
                source=cls._infer_layout_source(raw),
                element_count=len(elements),
                page_count=len(page_stats),
            ),
            raw=raw,
        )

    @classmethod
    def _resolve_layout_source(cls, raw: Any) -> Any:
        """복합 payload 안에서 layout 후보를 찾는다."""
        if isinstance(raw, dict):
            # LAYOUT_CONTAINER_KEYS를 순서대로 검사한다.
            nested = cls._pick_first(raw, cls.LAYOUT_CONTAINER_KEYS)
            # 명시적 layout 컨테이너가 있으면 그 값을 바로 반환한다.
            if nested is not None:
                return nested

        # 명시적 컨테이너가 없으면 raw 자체가 layout처럼 생겼는지 본다.
        if cls._has_layout_shape(raw):
            return raw

        return None

    @classmethod
    def _infer_layout_source(cls, raw: Any) -> AssemblySourceType:
        """layout 입력이 어떤 형태로 들어왔는지 메타데이터용으로 추정한다."""
        # 비어 있는 입력은 adapter가 아무것도 소비하지 않았음을 명시적으로 남긴다.
        if raw is None:
            return "empty"

        # element dict 리스트가 직접 들어온 경우다.
        if cls._is_layout_sequence(raw):
            return "direct_list"

        if isinstance(raw, dict):
            # 명시적 layout 컨테이너가 있으면 provenance를 더 구체적으로 남긴다.
            if cls._pick_first(raw, cls.LAYOUT_CONTAINER_KEYS) is not None:
                return "layout_container"
            # 그 외 dict는 raw 자체를 layout payload로 해석한 경우다.
            return "raw"

        return "raw"

    @classmethod
    def _extract_object_refs_from_elements(
        cls,
        elements: List[AssemblyElement],
        figure_assets_metadata: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[TableRef], List[FigureRef], List[NoteRef]]:
        """layout element에서 provisional table/figure/note ref를 추출한다."""
        table_refs: List[TableRef] = []
        figure_refs: List[FigureRef] = []
        note_refs: List[NoteRef] = []

        # layout block를 한 번 순회하면서 후속 단계가 바로 참조할 수 있는
        # provisional ref만 만든다. caption/note 연결 같은 정교한 조립은 뒤 단계의 책임이다.
        for element in elements:
            if element.kind == "table":
                # table spec이 미정이어도 최소 식별자와 위치는 확보해 둔다.
                table_refs.append(
                    TableRef(
                        table_id=element.id,
                        page=element.page,
                        bbox=element.bbox,
                        source_block_ids=[element.id],
                        metadata={"source": "layout_element"},
                        raw=element.raw,
                    )
                )
            elif element.kind == "figure":
                # figure asset 메타가 명시적으로 있으면 우선 사용하고,
                # 없으면 element metadata에서 보조 경로를 찾아본다.
                figure_metadata = figure_assets_metadata.get(element.id, {})
                asset_path = cls._normalize_str(
                    figure_metadata.get("asset_path")
                    or element.metadata.get("asset_path")
                    or element.metadata.get("crop_path")
                )
                figure_refs.append(
                    FigureRef(
                        figure_id=element.id,
                        page=element.page,
                        bbox=element.bbox,
                        asset_path=asset_path,
                        source_block_ids=[element.id],
                        metadata={"source": "layout_element", **figure_metadata},
                        raw=element.raw,
                    )
                )
            elif element.kind == "note":
                # note는 연결 규칙을 아직 확정하지 않고 note ref로만 보존한다.
                note_refs.append(
                    NoteRef(
                        note_id=element.id,
                        page=element.page,
                        bbox=element.bbox,
                        text=element.text,
                        source_block_ids=[element.id],
                        metadata={"source": "layout_element"},
                        raw=element.raw,
                    )
                )

        return table_refs, figure_refs, note_refs

    @classmethod
    def _infer_title_candidate(cls, elements: List[AssemblyElement]) -> Tuple[Optional[str], List[str]]:
        """초기 단계에서 문서 제목 후보를 명시적으로 추출한다."""
        # 입력 block가 없으면 제목 후보도 만들 수 없다.
        if not elements:
            return None, []

        # 현재 단계는 보수적으로 첫 heading을 제목 후보로 사용한다.
        for element in elements:
            if element.kind == "heading" and element.text:
                return element.text, [element.id]

        # heading이 전혀 없으면 첫 text block으로 fallback한다.
        first_text_element = next((element for element in elements if element.text), None)
        if first_text_element is None:
            return None, []

        return first_text_element.text, [first_text_element.id]

    @classmethod
    def _extract_figure_assets_metadata(cls, raw: Any) -> Dict[str, Dict[str, Any]]:
        """figure asset metadata를 명시적 맵 형태로 추출한다."""
        if not isinstance(raw, dict):
            return {}

        # 명시적 figure asset 컨테이너가 있을 때만 사용한다.
        # 아직은 추정 규칙보다 보존을 우선한다.
        candidate = cls._pick_first(raw, cls.FIGURE_ASSET_KEYS)
        if isinstance(candidate, dict):
            return {
                str(key): value
                for key, value in candidate.items()
                if isinstance(value, dict)
            }

        return {}

    @classmethod
    def _extract_layout_elements(
        cls,
        raw: Any,
    ) -> Tuple[List[AssemblyElement], List[AssemblyWarning]]:
        elements: List[AssemblyElement] = []
        warnings: List[AssemblyWarning] = []

        # 가장 단순한 입력은 element dict 리스트 자체가 직접 들어오는 경우다.
        if isinstance(raw, list):
            for index, item in enumerate(raw, start=1):
                element, item_warnings = cls._build_element(
                    item,
                    fallback_page=None,
                    fallback_id=cls._make_element_fallback_id(index),
                )
                if element is not None:
                    elements.append(element)
                warnings.extend(item_warnings)
            return elements, warnings

        if not isinstance(raw, dict):
            return elements, warnings

        # 일반적인 layout payload는 pages 아래에 page별 block 목록을 둔다.
        # 따라서 우선 PAGE_LIST_KEYS를 따라 내려가면서 page 번호와 element를 함께 읽는다.
        pages = cls._coerce_list(cls._pick_first(raw, cls.PAGE_LIST_KEYS))
        for page_index, page_payload in enumerate(pages, start=1):
            if not isinstance(page_payload, dict):
                continue

            page_number = cls._normalize_int(
                cls._pick_first(page_payload, cls.PAGE_NUMBER_KEYS),
                default=page_index,
            )
            page_elements = cls._coerce_list(cls._pick_first(page_payload, cls.ELEMENT_LIST_KEYS))

            for item_index, item in enumerate(page_elements, start=1):
                element, item_warnings = cls._build_element(
                    item,
                    fallback_page=page_number,
                    fallback_id=cls._make_page_element_fallback_id(page_number, item_index),
                )
                if element is not None:
                    elements.append(element)
                warnings.extend(item_warnings)

        # page 컨테이너에서 충분히 읽었다면 그 결과를 우선 사용한다.
        if elements:
            return elements, warnings

        # 일부 upstream은 elements/blocks/items를 top-level에 바로 둔다.
        # pages 구조가 없으면 그 형태를 두 번째 fallback으로 지원한다.
        top_level_elements = cls._coerce_list(cls._pick_first(raw, cls.ELEMENT_LIST_KEYS))
        for index, item in enumerate(top_level_elements, start=1):
            element, item_warnings = cls._build_element(
                item,
                fallback_page=None,
                fallback_id=cls._make_element_fallback_id(index),
            )
            if element is not None:
                elements.append(element)
            warnings.extend(item_warnings)

        if elements:
            return elements, warnings

        # 마지막 fallback은 raw 자체가 단일 element entry처럼 생긴 경우다.
        if cls._looks_like_element_entry(raw):
            element, item_warnings = cls._build_element(
                raw,
                fallback_page=None,
                fallback_id=cls._make_element_fallback_id(1),
            )
            if element is not None:
                elements.append(element)
            warnings.extend(item_warnings)

        return elements, warnings

    @classmethod
    def _extract_page_stats(
        cls,
        raw: Any,
    ) -> Tuple[List[PageStats], List[AssemblyWarning]]:
        stats_by_page: Dict[int, PageStats] = {}
        warnings: List[AssemblyWarning] = []

        if not isinstance(raw, dict):
            return [], warnings

        # 명시적으로 page_stats가 별도 배열로 오면 가장 먼저 읽는다.
        explicit_stats = cls._coerce_list(cls._pick_first(raw, cls.PAGE_STATS_KEYS))
        for index, item in enumerate(explicit_stats, start=1):
            page_stats, item_warnings = cls._build_page_stats(item, fallback_page=index)
            if page_stats is not None:
                stats_by_page[page_stats.page] = page_stats
            warnings.extend(item_warnings)

        # pages 안에 page_stat 메타를 넣은경우.
        pages = cls._coerce_list(cls._pick_first(raw, cls.PAGE_LIST_KEYS))
        for index, item in enumerate(pages, start=1):
            page_stats, item_warnings = cls._build_page_stats(item, fallback_page=index)
            if page_stats is None:
                warnings.extend(item_warnings)
                continue

            # 명시적 page_stats와 page payload 메타가 동시에 있으면
            # 이미 있는 값을 우선하면서 부족한 값만 보수적으로 합친다.
            if page_stats.page in stats_by_page:
                stats_by_page[page_stats.page] = cls._merge_page_stats(
                    stats_by_page[page_stats.page],
                    page_stats,
                )
            else:
                stats_by_page[page_stats.page] = page_stats
            warnings.extend(item_warnings)

        # 후속 단계가 page 순서를 바로 사용할 수 있게 page 번호 기준으로 정렬한다.
        ordered_pages = sorted(stats_by_page)
        return [stats_by_page[page] for page in ordered_pages], warnings

    @classmethod
    def _build_element(
        cls,
        raw: Any,
        fallback_page: Optional[int],
        fallback_id: str,
    ) -> Tuple[Optional[AssemblyElement], List[AssemblyWarning]]:
        warnings: List[AssemblyWarning] = []

        if isinstance(raw, AssemblyElement):
            return raw, warnings

        payload = raw if isinstance(raw, dict) else {"text": raw} # "text": 기본 설정

        element_id = cls._normalize_str(cls._pick_first(payload, cls.ELEMENT_ID_KEYS))
        page = cls._normalize_int(cls._pick_first(payload, cls.PAGE_NUMBER_KEYS))
        if element_id is None:
            element_id = fallback_id
            warnings.append(
                AssemblyWarning(
                    code=cls.WARNING_LAYOUT_MISSING_ID,
                    message="layout element에 id가 없어 임시 id를 부여했습니다.",
                    level="info",
                    page=page if page is not None else fallback_page,
                    element_ids=[element_id],
                    raw=raw,
                )
            )

        if page is None:
            page = 1
            warnings.append(
                AssemblyWarning(
                    code=cls.WARNING_LAYOUT_MISSING_PAGE,
                    message="layout element에 page 정보가 없어 1페이지로 가정했습니다.",
                    level="warning",
                    page=page,
                    element_ids=[element_id],
                    raw=raw,
                )
            )

        label = cls._normalize_str(cls._pick_first(payload, cls.ELEMENT_LABEL_KEYS))
        kind = cls._normalize_kind(label or "text") # "text": 기본 설정
        bbox = cls._normalize_bbox(cls._pick_first(payload, cls.BBOX_KEYS) or payload)
        text = cls._normalize_text(cls._pick_first(payload, cls.TEXT_KEYS))

        element = AssemblyElement(
            id=element_id,
            page=page,
            kind=kind,
            bbox=bbox,
            text=text,
            label=label,
            confidence=cls._normalize_float(cls._pick_first(payload, cls.CONFIDENCE_KEYS)),
            column_id=cls._normalize_int(cls._pick_first(payload, cls.COLUMN_KEYS)),
            reading_order=cls._normalize_int(cls._pick_first(payload, cls.READING_ORDER_KEYS)),
            parent_id=cls._normalize_str(cls._pick_first(payload, cls.PARENT_KEYS)),
            metadata=cls._extract_metadata(
                payload,
                {
                    *cls.ELEMENT_ID_KEYS,
                    *cls.PAGE_NUMBER_KEYS,
                    *cls.ELEMENT_LABEL_KEYS,
                    *cls.BBOX_KEYS,
                    *cls.TEXT_KEYS,
                    *cls.CONFIDENCE_KEYS,
                    *cls.COLUMN_KEYS,
                    *cls.READING_ORDER_KEYS,
                    *cls.PARENT_KEYS,
                },
            ),
            raw=raw,
        )
        return element, warnings

    @classmethod
    def _build_page_stats(
        cls,
        raw: Any,
        fallback_page: int,
    ) -> Tuple[Optional[PageStats], List[AssemblyWarning]]:
        warnings: List[AssemblyWarning] = []

        if isinstance(raw, PageStats):
            return raw, warnings

        if not isinstance(raw, dict):
            return None, warnings

        page = cls._normalize_int(
            cls._pick_first(raw, cls.PAGE_NUMBER_KEYS),
            default=fallback_page,
        )
        if page is None:
            return None, warnings

        page_stats = PageStats(
            page=page,
            width=cls._normalize_float(cls._pick_first(raw, cls.PAGE_WIDTH_KEYS)),
            height=cls._normalize_float(cls._pick_first(raw, cls.PAGE_HEIGHT_KEYS)),
            median_line_height=cls._normalize_float(cls._pick_first(raw, cls.LINE_HEIGHT_KEYS)),
            body_font_size=cls._normalize_float(cls._pick_first(raw, cls.BODY_FONT_SIZE_KEYS)),
            column_count=cls._normalize_int(cls._pick_first(raw, cls.COLUMN_COUNT_KEYS)),
            metadata=cls._extract_metadata(
                raw,
                {
                    *cls.PAGE_NUMBER_KEYS,
                    *cls.PAGE_WIDTH_KEYS,
                    *cls.PAGE_HEIGHT_KEYS,
                    *cls.LINE_HEIGHT_KEYS,
                    *cls.BODY_FONT_SIZE_KEYS,
                    *cls.COLUMN_COUNT_KEYS,
                    "elements",
                    "blocks",
                    "items",
                    "layout_elements",
                    "regions",
                },
            ),
            raw=raw,
        )
        return page_stats, warnings
