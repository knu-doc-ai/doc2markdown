from __future__ import annotations

"""Assembly Normalize / Filter 단계."""

import re
from dataclasses import replace
from math import ceil
from statistics import median
from typing import Any, Dict, List, Optional

from modules.assembly._common import AssemblyCommonMixin
from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    AssembledDocument,
    FigureRef,
    NoteRef,
    PageStats,
    TableRef,
)


class NormalizeFilter(AssemblyCommonMixin):
    """adapter seed 결과를 reading order 직전 상태로 정리한다."""

    # R-ASM-01, R-ASM-02에 대응하는 margin 영역 비율
    TOP_ZONE_RATIO = 0.10
    BOTTOM_ZONE_RATIO = 0.10
    # R-ASM-03에 대응하는 저신뢰 noise 기준
    LOW_CONF_THRESHOLD = 0.50
    LOW_CONF_SHORT_TEXT_MAX = 3
    REPEATED_MARGIN_MIN_PAGES = 2
    REPEATED_MARGIN_PAGE_RATIO = 0.30

    TEXT_REQUIRED_KINDS = frozenset(
        {
            "text",
            "heading",
            "list_item",
            "caption",
            "note",
            "quote",
            "code_block",
            "header",
            "footer",
            "page_number",
        }
    )
    OBJECT_LIKE_KINDS = frozenset({"table", "figure", "formula"})
    LINE_HEIGHT_KINDS = frozenset({"text", "heading", "list_item", "caption", "note", "quote"})
    BODY_TEXT_KINDS = frozenset({"text", "list_item", "caption", "note", "quote"})

    PAGE_NUMBER_PATTERN = re.compile(
        r"^(?:페이지|page)\s*\d+(?:\s*/\s*\d+)?$",
        re.IGNORECASE,
    )
    NON_CONTENT_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)

    @classmethod
    def apply(cls, result: AssemblyResult) -> AssemblyResult:
        """adapter seed 결과를 정규화하고 필터링한다."""
        if not isinstance(result, AssemblyResult):
            return result

        if cls._should_skip_normalize(result.metadata.stage):
            return result

        # R-ASM-01, R-ASM-02, R-ASM-03의 판정에 공통으로 쓰일
        # page 기준값을 먼저 정리한다.
        page_dimensions = cls._build_page_dimensions(result.page_stats, result.ordered_elements)
        repeated_margin_roles = cls._detect_repeated_margin_roles(
            result.ordered_elements,
            page_dimensions,
        )

        normalized_elements: List[AssemblyElement] = []
        filtered_by_reason = cls._empty_filter_buckets()

        for element in result.ordered_elements:
            normalized_element, filter_reason = cls._normalize_element(
                element,
                repeated_margin_roles,
                page_dimensions,
            )
            if normalized_element is None:
                filtered_by_reason.setdefault(filter_reason or "unknown", []).append(element.id)
                continue
            normalized_elements.append(normalized_element)

        element_map = {element.id: element for element in normalized_elements}
        normalized_page_stats = cls._normalize_page_stats(
            result.page_stats,
            normalized_elements,
            page_dimensions,
        )

        title_candidate, title_source_block_ids = cls._infer_title_candidate(normalized_elements)
        if title_candidate is None:
            title_candidate = result.document.title_candidate
            title_source_block_ids = list(result.document.title_source_block_ids)

        normalization_summary = cls._build_normalization_summary(
            input_count=len(result.ordered_elements),
            output_count=len(normalized_elements),
            filtered_by_reason=filtered_by_reason,
        )

        normalized_document = AssembledDocument(
            title_candidate=title_candidate,
            title_source_block_ids=title_source_block_ids,
            children=list(result.document.children),
            sections=list(result.document.sections),
            table_refs=cls._sync_table_refs(result.document.table_refs, element_map),
            figure_refs=cls._sync_figure_refs(result.document.figure_refs, element_map),
            note_refs=cls._sync_note_refs(result.document.note_refs, element_map),
            figure_assets_metadata=dict(result.document.figure_assets_metadata),
            metadata={
                **dict(result.document.metadata),
                "normalize_filter": normalization_summary,
            },
            raw=result.document.raw,
        )

        return AssemblyResult(
            ordered_elements=normalized_elements,
            block_relations=list(result.block_relations),
            document=normalized_document,
            page_stats=normalized_page_stats,
            warnings=list(result.warnings),
            metadata=cls._build_normalized_metadata(result.metadata, normalization_summary),
            raw=result.raw,
        )

    @classmethod
    def _should_skip_normalize(cls, stage: Optional[str]) -> bool:
        """이미 normalize 이후 단계면 다시 처리하지 않는다."""
        return stage in {
            "normalized",
            "structure_assembled",
            "validated",
        }

    @classmethod
    def _empty_filter_buckets(cls) -> Dict[str, List[str]]:
        """후속 디버깅을 위한 기본 필터 버킷을 만든다."""
        return {
            "empty_text": [],
            "header": [],
            "footer": [],
            "page_number": [],
            "low_confidence_noise": [],
            "upstream_noise": [],
        }

    @classmethod
    def _normalize_element(
        cls,
        element: AssemblyElement,
        repeated_margin_roles: Dict[str, str],
        page_dimensions: Dict[int, Dict[str, float]],
    ) -> tuple[Optional[AssemblyElement], Optional[str]]:
        """개별 element를 정규화하고 제외 여부를 판정한다."""
        metadata = dict(element.metadata)
        normalized_text = cls._normalize_text(element.text)
        if normalized_text != element.text:
            metadata["text_normalized"] = True

        # R-ASM-01, R-ASM-02
        # upstream label 또는 반복 margin 패턴으로 header/footer/page_number를 판정한다.
        explicit_role = cls._detect_explicit_margin_role(
            element,
            normalized_text,
            page_dimensions,
        )
        repeated_role = repeated_margin_roles.get(element.id)
        filter_role = explicit_role or repeated_role
        if filter_role is not None:
            metadata["normalized_role"] = filter_role
            metadata["excluded_from_reading_order"] = True
            return None, filter_role

        if element.kind == "noise":
            metadata["excluded_from_reading_order"] = True
            return None, "upstream_noise"

        if element.kind in cls.TEXT_REQUIRED_KINDS and normalized_text is None:
            metadata["excluded_from_reading_order"] = True
            return None, "empty_text"

        # R-ASM-03
        # 짧고 신뢰도 낮은 비내용 조각만 보수적으로 noise로 제거한다.
        if cls._should_filter_low_confidence_noise(
            kind=element.kind,
            text=normalized_text,
            confidence=element.confidence,
        ):
            metadata["excluded_from_reading_order"] = True
            metadata["normalized_role"] = "noise"
            return None, "low_confidence_noise"

        return replace(
            element,
            text=normalized_text,
            metadata=metadata,
        ), None

    @classmethod
    def _detect_explicit_margin_role(
        cls,
        element: AssemblyElement,
        text: Optional[str],
        page_dimensions: Dict[int, Dict[str, float]],
    ) -> Optional[str]:
        """upstream label과 page number 패턴을 우선 적용한다."""
        # R-ASM-01, R-ASM-02
        # 이미 upstream이 header/footer/page_number로 준 경우는 그대로 신뢰한다.
        if element.kind in {"header", "footer", "page_number"}:
            return element.kind

        if text is None or element.bbox is None:
            return None

        # R-ASM-02
        # page number 패턴은 하단 영역에 있을 때만 footer 계열로 간주한다.
        if cls._looks_like_page_number(text) and cls._is_bottom_zone(
            element.page,
            element.bbox[3],
            page_dimensions,
        ):
            return "page_number"

        return None

    @classmethod
    def _looks_like_page_number(cls, text: str) -> bool:
        """페이지 번호 전용 텍스트를 느슨하게 감지한다."""
        compact = text.strip()
        if cls.PAGE_NUMBER_PATTERN.fullmatch(compact):
            return True

        lowered = compact.lower()
        return bool(
            re.fullmatch(r"(?:페이지|page)\s+\d+\s*/\s*\d+", lowered)
            or re.fullmatch(r"\d+\s*/\s*\d+", lowered)
        )

    @classmethod
    def _detect_repeated_margin_roles(
        cls,
        elements: List[AssemblyElement],
        page_dimensions: Dict[int, Dict[str, float]],
    ) -> Dict[str, str]:
        """여러 페이지에서 반복되는 상하단 텍스트를 header/footer로 태깅한다."""
        # R-ASM-01, R-ASM-02
        # 여러 페이지에서 반복되는 상단/하단 문자열을 찾아
        # 본문 조립에서 제외할 margin block으로 판정한다.
        total_pages = max(
            len(page_dimensions),
            len({element.page for element in elements}),
        )
        if total_pages <= 1:
            return {}

        min_pages = max(
            cls.REPEATED_MARGIN_MIN_PAGES,
            ceil(total_pages * cls.REPEATED_MARGIN_PAGE_RATIO),
        )

        candidates: List[Dict[str, Any]] = []
        zone_pages: Dict[str, Dict[str, set[int]]] = {"top": {}, "bottom": {}}

        for element in elements:
            if element.kind in cls.OBJECT_LIKE_KINDS or element.bbox is None:
                continue

            text = cls._normalize_text(element.text)
            if text is None:
                continue

            zone = cls._detect_margin_zone(
                page=element.page,
                y1=element.bbox[1],
                y2=element.bbox[3],
                page_dimensions=page_dimensions,
            )
            if zone is None:
                continue

            fingerprint = cls._fingerprint_margin_text(text)
            if fingerprint is None:
                continue

            zone_pages[zone].setdefault(fingerprint, set()).add(element.page)
            candidates.append(
                {
                    "id": element.id,
                    "zone": zone,
                    "text": text,
                    "fingerprint": fingerprint,
                }
            )

        detected_roles: Dict[str, str] = {}
        for candidate in candidates:
            pages = zone_pages[candidate["zone"]][candidate["fingerprint"]]
            if len(pages) < min_pages:
                continue

            if candidate["zone"] == "top":
                detected_roles[candidate["id"]] = "header"
                continue

            if cls._looks_like_page_number(candidate["text"]):
                detected_roles[candidate["id"]] = "page_number"
            else:
                detected_roles[candidate["id"]] = "footer"

        return detected_roles

    @classmethod
    def _detect_margin_zone(
        cls,
        page: int,
        y1: float,
        y2: float,
        page_dimensions: Dict[int, Dict[str, float]],
    ) -> Optional[str]:
        """상단/하단 margin 영역 후보인지 판정한다."""
        page_height = page_dimensions.get(page, {}).get("height")
        if page_height is None or page_height <= 0:
            return None

        if y1 <= page_height * cls.TOP_ZONE_RATIO:
            return "top"
        if y2 >= page_height * (1 - cls.BOTTOM_ZONE_RATIO):
            return "bottom"
        return None

    @classmethod
    def _fingerprint_margin_text(cls, text: str) -> Optional[str]:
        """페이지 번호 차이만 무시하고 반복 텍스트를 비교한다."""
        normalized = cls._normalize_text(text)
        if normalized is None:
            return None

        lowered = normalized.lower()
        lowered = re.sub(r"\d+", "#", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered or None

    @classmethod
    def _is_bottom_zone(
        cls,
        page: int,
        y2: float,
        page_dimensions: Dict[int, Dict[str, float]],
    ) -> bool:
        """footer 판정용 하단 영역 여부를 계산한다."""
        page_height = page_dimensions.get(page, {}).get("height")
        if page_height is None or page_height <= 0:
            return False
        return y2 >= page_height * (1 - cls.BOTTOM_ZONE_RATIO)

    @classmethod
    def _should_filter_low_confidence_noise(
        cls,
        kind: str,
        text: Optional[str],
        confidence: Optional[float],
    ) -> bool:
        """짧고 신뢰도 낮은 조각만 보수적으로 noise로 제거한다."""
        # R-ASM-03
        # heading/본문을 공격적으로 지우지 않기 위해 매우 짧거나
        # 비문자 위주인 경우에만 noise로 본다.
        if confidence is None or confidence >= cls.LOW_CONF_THRESHOLD:
            return False

        if kind in cls.OBJECT_LIKE_KINDS:
            return False

        compact_text = re.sub(r"\s+", "", text or "")
        if not compact_text:
            return True

        if len(compact_text) <= cls.LOW_CONF_SHORT_TEXT_MAX:
            return True

        return bool(cls.NON_CONTENT_PATTERN.fullmatch(compact_text))

    @classmethod
    def _build_page_dimensions(
        cls,
        page_stats: List[PageStats],
        elements: List[AssemblyElement],
    ) -> Dict[int, Dict[str, float]]:
        """page 높이/너비가 비어 있어도 bbox 기반 추정치를 보완한다."""
        dimensions: Dict[int, Dict[str, float]] = {}

        for stat in page_stats:
            dimensions[stat.page] = {
                "width": stat.width or 0.0,
                "height": stat.height or 0.0,
            }

        for element in elements:
            if element.bbox is None:
                dimensions.setdefault(element.page, {"width": 0.0, "height": 0.0})
                continue

            x2 = float(element.bbox[2])
            y2 = float(element.bbox[3])
            page_dimension = dimensions.setdefault(
                element.page,
                {"width": 0.0, "height": 0.0},
            )
            page_dimension["width"] = max(page_dimension["width"], x2)
            page_dimension["height"] = max(page_dimension["height"], y2)

        return dimensions

    @classmethod
    def _normalize_page_stats(
        cls,
        page_stats: List[PageStats],
        elements: List[AssemblyElement],
        page_dimensions: Dict[int, Dict[str, float]],
    ) -> List[PageStats]:
        """후속 threshold 계산에 필요한 page 통계를 보수적으로 보강한다."""
        # 이후 ReadingOrder/Structure 단계에서 상대 threshold를 안정적으로 쓰도록
        # 누락된 page 통계를 bbox 기반으로 보강한다.
        stats_by_page = {stat.page: stat for stat in page_stats}
        elements_by_page: Dict[int, List[AssemblyElement]] = {}

        for element in elements:
            elements_by_page.setdefault(element.page, []).append(element)

        normalized_stats: List[PageStats] = []
        page_numbers = sorted(set(stats_by_page) | set(elements_by_page) | set(page_dimensions))

        for page in page_numbers:
            current = stats_by_page.get(page, PageStats(page=page))
            metadata = dict(current.metadata)
            dimension = page_dimensions.get(page, {})
            page_elements = elements_by_page.get(page, [])

            width = current.width if current.width is not None else cls._normalize_float(dimension.get("width"))
            height = current.height if current.height is not None else cls._normalize_float(dimension.get("height"))

            median_line_height = current.median_line_height
            if median_line_height is None:
                inferred_line_height = cls._infer_median_height(page_elements, cls.LINE_HEIGHT_KINDS)
                median_line_height = inferred_line_height
                if inferred_line_height is not None:
                    metadata["inferred_median_line_height"] = True

            body_font_size = current.body_font_size
            if body_font_size is None:
                inferred_body_font = cls._infer_median_height(page_elements, cls.BODY_TEXT_KINDS)
                if inferred_body_font is None:
                    inferred_body_font = median_line_height
                elif (
                    median_line_height is not None
                    and inferred_body_font > median_line_height * 1.25
                ):
                    # body font 추정치가 line height보다 과하게 크면
                    # 다중 행 paragraph 높이가 섞였다고 보고 line height로 보정한다.
                    inferred_body_font = median_line_height
                body_font_size = inferred_body_font
                if inferred_body_font is not None:
                    metadata["inferred_body_font_size"] = True

            metadata["active_element_count"] = len(page_elements)

            normalized_stats.append(
                PageStats(
                    page=page,
                    width=width,
                    height=height,
                    median_line_height=median_line_height,
                    body_font_size=body_font_size,
                    column_count=current.column_count,
                    metadata=metadata,
                    raw=current.raw,
                )
            )

        return normalized_stats

    @classmethod
    def _infer_median_height(
        cls,
        elements: List[AssemblyElement],
        allowed_kinds: frozenset[str],
    ) -> Optional[float]:
        """짧은 블록 높이 대역만 사용해 line/body 기준값을 보수적으로 추정한다."""
        heights = sorted(
            [
                float(element.bbox[3] - element.bbox[1])
                for element in elements
                if element.kind in allowed_kinds and element.bbox is not None and element.text
            ]
        )
        if not heights:
            return None

        # 문단 박스는 여러 줄이 한 번에 묶여 높이가 커지기 쉬우므로,
        # 가장 낮은 높이 주변의 작은 밴드만 사용해 기준값을 잡는다.
        base_height = heights[0]
        clustered_heights = [height for height in heights if height <= base_height * 1.5]
        if not clustered_heights:
            clustered_heights = [base_height]

        return float(median(clustered_heights))

    @classmethod
    def _infer_title_candidate(
        cls,
        elements: List[AssemblyElement],
    ) -> tuple[Optional[str], List[str]]:
        """필터링 이후의 유효 element 기준으로 제목 후보를 다시 잡는다."""
        if not elements:
            return None, []

        for element in elements:
            if element.kind == "heading" and element.text:
                return element.text, [element.id]

        first_text_element = next((element for element in elements if element.text), None)
        if first_text_element is None:
            return None, []

        return first_text_element.text, [first_text_element.id]

    @classmethod
    def _sync_table_refs(
        cls,
        table_refs: List[TableRef],
        element_map: Dict[str, AssemblyElement],
    ) -> List[TableRef]:
        """layout table element와 연결된 ref의 좌표/페이지를 다시 맞춘다."""
        synced_refs: List[TableRef] = []
        for table_ref in table_refs:
            source_element = element_map.get(table_ref.table_id)
            if source_element is None:
                synced_refs.append(table_ref)
                continue

            synced_refs.append(
                replace(
                    table_ref,
                    page=source_element.page,
                    bbox=source_element.bbox or table_ref.bbox,
                    metadata={
                        **dict(table_ref.metadata),
                        "normalized_from_element": True,
                    },
                )
            )
        return synced_refs

    @classmethod
    def _sync_figure_refs(
        cls,
        figure_refs: List[FigureRef],
        element_map: Dict[str, AssemblyElement],
    ) -> List[FigureRef]:
        """figure ref도 element 정규화 결과와 좌표를 맞춘다."""
        synced_refs: List[FigureRef] = []
        for figure_ref in figure_refs:
            source_element = element_map.get(figure_ref.figure_id)
            if source_element is None:
                synced_refs.append(figure_ref)
                continue

            synced_refs.append(
                replace(
                    figure_ref,
                    page=source_element.page,
                    bbox=source_element.bbox or figure_ref.bbox,
                    metadata={
                        **dict(figure_ref.metadata),
                        "normalized_from_element": True,
                    },
                )
            )
        return synced_refs

    @classmethod
    def _sync_note_refs(
        cls,
        note_refs: List[NoteRef],
        element_map: Dict[str, AssemblyElement],
    ) -> List[NoteRef]:
        """note ref는 정규화된 text를 반영해 뒤 단계가 바로 쓰게 한다."""
        synced_refs: List[NoteRef] = []
        for note_ref in note_refs:
            source_element = element_map.get(note_ref.note_id)
            if source_element is None:
                synced_refs.append(note_ref)
                continue

            synced_refs.append(
                replace(
                    note_ref,
                    page=source_element.page,
                    bbox=source_element.bbox or note_ref.bbox,
                    text=source_element.text or note_ref.text,
                    metadata={
                        **dict(note_ref.metadata),
                        "normalized_from_element": True,
                    },
                )
            )
        return synced_refs

    @classmethod
    def _build_normalization_summary(
        cls,
        input_count: int,
        output_count: int,
        filtered_by_reason: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """문서 metadata와 result metadata에 함께 남길 요약을 만든다."""
        filtered_counts = {
            reason: len(element_ids)
            for reason, element_ids in filtered_by_reason.items()
            if element_ids
        }
        filtered_ids = {
            reason: element_ids
            for reason, element_ids in filtered_by_reason.items()
            if element_ids
        }

        return {
            "input_element_count": input_count,
            "output_element_count": output_count,
            "filtered_count": input_count - output_count,
            "filtered_counts": filtered_counts,
            "filtered_element_ids": filtered_ids,
        }

    @classmethod
    def _build_normalized_metadata(
        cls,
        previous_metadata: AssemblyMeta,
        normalization_summary: Dict[str, Any],
    ) -> AssemblyMeta:
        """이전 메타데이터를 보존하면서 stage만 normalized로 갱신한다."""
        details = dict(previous_metadata.details)
        details["upstream_stage"] = previous_metadata.stage
        details["normalize_filter"] = normalization_summary

        return AssemblyMeta(
            stage="normalized",
            adapter=previous_metadata.adapter,
            source=previous_metadata.source,
            details=details,
        )


__all__ = ["NormalizeFilter"]
