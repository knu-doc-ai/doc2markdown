from __future__ import annotations

"""Assembly Reading Order Resolver 단계."""

from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Dict, List, Optional

from modules.assembly._common import AssemblyCommonMixin
from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    BlockRelation,
    PageStats,
)
from modules.assembly.normalize_filter import NormalizeFilter


@dataclass
class _ColumnBand:
    """페이지 안의 컬럼 가로 범위를 표현한다."""

    id: int
    left: float
    right: float
    center_x: float
    upstream_ids: tuple[int, ...] = ()


@dataclass
class _PagePlan:
    """한 페이지의 reading order 계산 계획이다."""

    page: int
    page_width: float
    line_height: float
    same_line_threshold: float
    column_gap_threshold: float
    column_count: int
    column_source: str
    bands: List[_ColumnBand]


@dataclass
class _PageEntry:
    """정렬 중에만 쓰는 element 래퍼다."""

    original_index: int
    element: AssemblyElement
    resolved_column_id: int


class ReadingOrderResolver(AssemblyCommonMixin):
    """normalized 결과에 reading order와 column 정보를 확정한다."""

    # R-ASM-04
    # 컬럼 cluster 간 최소 간격은 page width 비율을 기본으로 잡는다.
    COLUMN_GAP_RATIO = 0.08
    # R-ASM-06
    # 같은 줄 band 판정은 page line height의 상대값으로 계산한다.
    SAME_LINE_RATIO = 0.40

    DEFAULT_LINE_HEIGHT = 12.0
    COLUMN_CANDIDATE_MAX_WIDTH_RATIO = 0.75
    SPANNING_MIN_OVERLAP_RATIO = 0.30
    SPANNING_MIN_WIDTH_RATIO = 0.45

    # 컬럼 추정은 본문 계열 블록 중심으로 한다.
    COLUMN_CANDIDATE_KINDS = frozenset(
        {
            "text",
            "heading",
            "list_item",
            "caption",
            "note",
            "quote",
            "code_block",
        }
    )

    @classmethod
    def apply(cls, result: AssemblyResult) -> AssemblyResult:
        """normalized 결과에 읽기 순서를 반영한다."""
        if not isinstance(result, AssemblyResult):
            return result

        if cls._should_skip_resolution(result.metadata.stage):
            return result

        if result.metadata.stage != "normalized":
            result = NormalizeFilter.apply(result)

        page_stats_by_page = {page_stat.page: page_stat for page_stat in result.page_stats}
        elements_by_page: Dict[int, List[AssemblyElement]] = {}
        original_order_map: Dict[str, int] = {}

        for index, element in enumerate(result.ordered_elements):
            elements_by_page.setdefault(element.page, []).append(element)
            original_order_map[element.id] = index

        all_pages = sorted(set(page_stats_by_page) | set(elements_by_page))
        resolved_elements: List[AssemblyElement] = []
        resolved_page_stats: List[PageStats] = []
        page_summaries: List[Dict[str, Any]] = []
        reading_order_index = 1

        for page in all_pages:
            page_elements = list(elements_by_page.get(page, []))
            page_stat = page_stats_by_page.get(page, PageStats(page=page))
            page_plan = cls._build_page_plan(page, page_elements, page_stat)
            ordered_entries = cls._resolve_page_entries(
                page_elements=page_elements,
                page_plan=page_plan,
                original_order_map=original_order_map,
            )

            for entry in ordered_entries:
                metadata = dict(entry.element.metadata)
                metadata["reading_order_resolved"] = True
                metadata["column_assignment_source"] = page_plan.column_source
                if (
                    entry.element.column_id is not None
                    and entry.element.column_id != entry.resolved_column_id
                ):
                    metadata["upstream_column_id"] = entry.element.column_id

                resolved_elements.append(
                    replace(
                        entry.element,
                        column_id=entry.resolved_column_id,
                        reading_order=reading_order_index,
                        metadata=metadata,
                    )
                )
                reading_order_index += 1

            resolved_page_stats.append(cls._build_resolved_page_stats(page_stat, page_plan, ordered_entries))
            page_summaries.append(
                {
                    "page": page,
                    "element_count": len(ordered_entries),
                    "column_count": page_plan.column_count,
                    "column_source": page_plan.column_source,
                    "same_line_threshold": page_plan.same_line_threshold,
                    "column_gap_threshold": page_plan.column_gap_threshold,
                    "column_centers": [band.center_x for band in page_plan.bands],
                }
            )

        next_relations = cls._build_next_relations(resolved_elements)
        reading_order_summary = {
            "page_count": len(all_pages),
            "element_count": len(resolved_elements),
            "next_relation_count": len(next_relations),
            "pages": page_summaries,
        }

        document_metadata = dict(result.document.metadata)
        document_metadata["reading_order"] = reading_order_summary

        return AssemblyResult(
            ordered_elements=resolved_elements,
            block_relations=cls._merge_next_relations(result.block_relations, next_relations),
            document=replace(
                result.document,
                metadata=document_metadata,
            ),
            page_stats=resolved_page_stats,
            warnings=list(result.warnings),
            metadata=cls._build_resolved_metadata(result.metadata, reading_order_summary),
            raw=result.raw,
        )

    @classmethod
    def _should_skip_resolution(cls, stage: Optional[str]) -> bool:
        """이미 reading order 이후 단계라면 다시 계산하지 않는다."""
        return stage in {
            "reading_order_resolved",
            "structure_assembled",
            "validated",
        }

    @classmethod
    def _build_page_plan(
        cls,
        page: int,
        elements: List[AssemblyElement],
        page_stat: PageStats,
    ) -> _PagePlan:
        """페이지별 threshold와 컬럼 계획을 계산한다."""
        # R-ASM-04
        # 페이지 폭, line height, 컬럼 gap 기준을 먼저 정한 뒤
        # upstream 힌트 -> geometry 추정 -> 단일 컬럼 fallback 순으로 계획을 만든다.
        page_width = cls._resolve_page_width(page_stat, elements)
        line_height = (
            page_stat.median_line_height
            or cls._infer_line_height(elements)
            or cls.DEFAULT_LINE_HEIGHT
        )
        same_line_threshold = max(1.0, line_height * cls.SAME_LINE_RATIO)
        column_gap_threshold = max(page_width * cls.COLUMN_GAP_RATIO, line_height * 2.0, 24.0)

        hinted_bands = cls._build_upstream_bands(elements)
        if len(hinted_bands) > 1:
            return _PagePlan(
                page=page,
                page_width=page_width,
                line_height=line_height,
                same_line_threshold=same_line_threshold,
                column_gap_threshold=column_gap_threshold,
                column_count=len(hinted_bands),
                column_source="upstream_hint",
                bands=hinted_bands,
            )

        # 지금 단계의 geometry 추정은 단일/이중 컬럼을 우선 대상으로 둔다.
        hinted_column_count = page_stat.column_count if (page_stat.column_count or 0) > 1 else None
        desired_column_count = hinted_column_count or 2
        inferred_bands = cls._infer_geometry_bands(
            elements=elements,
            page_width=page_width,
            column_gap_threshold=column_gap_threshold,
            desired_column_count=desired_column_count,
        )
        if len(inferred_bands) > 1:
            return _PagePlan(
                page=page,
                page_width=page_width,
                line_height=line_height,
                same_line_threshold=same_line_threshold,
                column_gap_threshold=column_gap_threshold,
                column_count=len(inferred_bands),
                column_source="geometry",
                bands=inferred_bands,
            )

        if hinted_column_count is not None:
            hinted_uniform_bands = cls._build_uniform_bands(page_width, hinted_column_count)
            return _PagePlan(
                page=page,
                page_width=page_width,
                line_height=line_height,
                same_line_threshold=same_line_threshold,
                column_gap_threshold=column_gap_threshold,
                column_count=len(hinted_uniform_bands),
                column_source="page_stats_hint",
                bands=hinted_uniform_bands,
            )

        return _PagePlan(
            page=page,
            page_width=page_width,
            line_height=line_height,
            same_line_threshold=same_line_threshold,
            column_gap_threshold=column_gap_threshold,
            column_count=1,
            column_source="single_column",
            bands=cls._build_uniform_bands(page_width, 1),
        )

    @classmethod
    def _resolve_page_width(
        cls,
        page_stat: PageStats,
        elements: List[AssemblyElement],
    ) -> float:
        """page width가 없으면 bbox 기반으로 보수적으로 추정한다."""
        if page_stat.width is not None and page_stat.width > 0:
            return float(page_stat.width)

        candidates = [
            float(element.bbox[2])
            for element in elements
            if element.bbox is not None
        ]
        if candidates:
            return max(candidates)

        return 1000.0

    @classmethod
    def _infer_line_height(cls, elements: List[AssemblyElement]) -> Optional[float]:
        """bbox 높이 중 작은 쪽 cluster를 line height 근사치로 쓴다."""
        heights = sorted(
            [
                float(element.bbox[3] - element.bbox[1])
                for element in elements
                if element.bbox is not None
                and (element.bbox[3] - element.bbox[1]) > 0
                and element.kind in cls.COLUMN_CANDIDATE_KINDS
            ]
        )
        if not heights:
            return None

        base_height = heights[0]
        clustered = [height for height in heights if height <= base_height * 1.5]
        if not clustered:
            clustered = [base_height]

        return float(median(clustered))

    @classmethod
    def _build_upstream_bands(cls, elements: List[AssemblyElement]) -> List[_ColumnBand]:
        """upstream column_id 힌트가 충분하면 그 범위를 그대로 쓴다."""
        # R-ASM-04
        # upstream이 준 column_id가 여러 개면 geometry 추정보다 먼저 신뢰한다.
        grouped: Dict[int, List[AssemblyElement]] = {}
        for element in elements:
            if element.bbox is None or (element.column_id or 0) <= 0:
                continue
            grouped.setdefault(int(element.column_id), []).append(element)

        if len(grouped) <= 1:
            return []

        raw_bands: List[_ColumnBand] = []
        for upstream_column_id, items in grouped.items():
            raw_bands.append(
                _ColumnBand(
                    id=upstream_column_id,
                    left=min(item.bbox[0] for item in items if item.bbox is not None),
                    right=max(item.bbox[2] for item in items if item.bbox is not None),
                    center_x=sum(
                        (item.bbox[0] + item.bbox[2]) / 2
                        for item in items
                        if item.bbox is not None
                    )
                    / len(items),
                    upstream_ids=(upstream_column_id,),
                )
            )

        raw_bands.sort(key=lambda band: band.center_x)
        normalized_bands: List[_ColumnBand] = []
        for index, band in enumerate(raw_bands, start=1):
            normalized_bands.append(
                _ColumnBand(
                    id=index,
                    left=band.left,
                    right=band.right,
                    center_x=band.center_x,
                    upstream_ids=band.upstream_ids,
                )
            )

        return normalized_bands

    @classmethod
    def _infer_geometry_bands(
        cls,
        elements: List[AssemblyElement],
        page_width: float,
        column_gap_threshold: float,
        desired_column_count: Optional[int],
    ) -> List[_ColumnBand]:
        """bbox 중심값으로 좌우 컬럼 cluster를 추정한다."""
        # R-ASM-04
        # 본문 계열 block의 center_x 간격이 충분히 벌어지면 다단으로 본다.
        candidates = [
            element
            for element in elements
            if cls._is_column_candidate(element, page_width)
        ]
        if len(candidates) < 2:
            return []

        sorted_candidates = sorted(
            candidates,
            key=lambda element: ((element.bbox[0] + element.bbox[2]) / 2, element.bbox[0]),
        )
        split_indices = cls._find_split_indices(
            elements=sorted_candidates,
            column_gap_threshold=column_gap_threshold,
            desired_column_count=desired_column_count,
        )
        if not split_indices:
            return []

        clusters: List[List[AssemblyElement]] = []
        start = 0
        for split_index in split_indices:
            clusters.append(sorted_candidates[start:split_index])
            start = split_index
        clusters.append(sorted_candidates[start:])

        if len(clusters) <= 1:
            return []

        page_right_edge = max(
            page_width,
            max(
                element.bbox[2]
                for element in sorted_candidates
                if element.bbox is not None
            ),
        )
        boundaries: List[float] = [0.0]
        for split_index in split_indices:
            left_center = (
                sorted_candidates[split_index - 1].bbox[0]
                + sorted_candidates[split_index - 1].bbox[2]
            ) / 2
            right_center = (
                sorted_candidates[split_index].bbox[0]
                + sorted_candidates[split_index].bbox[2]
            ) / 2
            boundaries.append((left_center + right_center) / 2)
        boundaries.append(float(page_right_edge))

        bands: List[_ColumnBand] = []
        for index, cluster in enumerate(clusters, start=1):
            if not cluster:
                continue
            bands.append(
                _ColumnBand(
                    id=index,
                    left=boundaries[index - 1],
                    right=boundaries[index],
                    center_x=sum(
                        (element.bbox[0] + element.bbox[2]) / 2
                        for element in cluster
                        if element.bbox is not None
                    )
                    / len(cluster),
                )
            )

        return bands if len(bands) > 1 else []

    @classmethod
    def _is_column_candidate(cls, element: AssemblyElement, page_width: float) -> bool:
        """컬럼 중심 추정에 쓸 만한 본문 계열 블록만 고른다."""
        if element.bbox is None or element.kind not in cls.COLUMN_CANDIDATE_KINDS:
            return False

        width = element.bbox[2] - element.bbox[0]
        if width <= 0:
            return False

        if page_width > 0 and width > page_width * cls.COLUMN_CANDIDATE_MAX_WIDTH_RATIO:
            return False

        return True

    @classmethod
    def _find_split_indices(
        cls,
        elements: List[AssemblyElement],
        column_gap_threshold: float,
        desired_column_count: Optional[int],
    ) -> List[int]:
        """정렬된 center_x 시퀀스에서 컬럼 분리 지점을 찾는다."""
        # R-ASM-04
        # center_x 간 큰 gap를 컬럼 경계 후보로 본다.
        if len(elements) < 2:
            return []

        gaps: List[tuple[float, int]] = []
        for index in range(len(elements) - 1):
            current_center = (elements[index].bbox[0] + elements[index].bbox[2]) / 2
            next_center = (elements[index + 1].bbox[0] + elements[index + 1].bbox[2]) / 2
            gaps.append((next_center - current_center, index + 1))

        if desired_column_count and desired_column_count > 1:
            ranked = sorted(gaps, key=lambda item: item[0], reverse=True)
            selected = [
                split_index
                for gap, split_index in ranked[: desired_column_count - 1]
                if gap >= column_gap_threshold
            ]
            return sorted(selected)

        return []

    @classmethod
    def _build_uniform_bands(cls, page_width: float, column_count: int) -> List[_ColumnBand]:
        """fallback용 균등 분할 컬럼 범위를 만든다."""
        safe_column_count = max(1, column_count)
        safe_width = max(page_width, 1.0)
        bands: List[_ColumnBand] = []
        for index in range(safe_column_count):
            left = safe_width * index / safe_column_count
            right = safe_width * (index + 1) / safe_column_count
            bands.append(
                _ColumnBand(
                    id=index + 1,
                    left=left,
                    right=right,
                    center_x=(left + right) / 2,
                )
            )
        return bands

    @classmethod
    def _resolve_page_entries(
        cls,
        page_elements: List[AssemblyElement],
        page_plan: _PagePlan,
        original_order_map: Dict[str, int],
    ) -> List[_PageEntry]:
        """페이지 안의 element를 column/line 기준으로 정렬한다."""
        entries = [
            _PageEntry(
                original_index=original_order_map.get(element.id, index),
                element=element,
                resolved_column_id=cls._resolve_column_id(element, page_plan),
            )
            for index, element in enumerate(page_elements)
        ]
        if page_plan.column_count <= 1:
            return cls._order_region_entries(entries, page_plan, column_ids=[1])

        # R-ASM-05
        # 다단 페이지는 컬럼별 top-down 정렬을 기본으로 한다.
        # R-ASM-06
        # 다만 spanning block은 region 경계로 먼저 배치한 뒤 컬럼 정렬을 적용한다.
        spanning_entries = [
            entry
            for entry in entries
            if entry.resolved_column_id == 0
        ]
        non_spanning_entries = [
            entry
            for entry in entries
            if entry.resolved_column_id != 0
        ]
        spanning_entries = cls._sort_line_band_entries(
            spanning_entries,
            page_plan.same_line_threshold,
        )

        if not spanning_entries:
            return cls._order_region_entries(
                non_spanning_entries,
                page_plan,
                column_ids=[band.id for band in page_plan.bands],
            )

        regions: Dict[int, List[_PageEntry]] = {index: [] for index in range(len(spanning_entries) + 1)}
        for entry in non_spanning_entries:
            region_index = cls._find_region_index(entry, spanning_entries)
            regions.setdefault(region_index, []).append(entry)

        ordered_entries: List[_PageEntry] = []
        ordered_column_ids = [band.id for band in page_plan.bands]
        for region_index in range(len(spanning_entries) + 1):
            ordered_entries.extend(
                cls._order_region_entries(
                    regions.get(region_index, []),
                    page_plan,
                    column_ids=ordered_column_ids,
                )
            )
            if region_index < len(spanning_entries):
                ordered_entries.append(spanning_entries[region_index])

        return ordered_entries

    @classmethod
    def _resolve_column_id(
        cls,
        element: AssemblyElement,
        page_plan: _PagePlan,
    ) -> int:
        """element를 어느 컬럼에 둘지 확정한다."""
        # R-ASM-04
        # 다단으로 판단된 페이지에서는 중심 x와 spanning 여부로 컬럼을 배정한다.
        if page_plan.column_count <= 1:
            return 1

        if element.bbox is None:
            return 0

        if (element.column_id or 0) > 0:
            for band in page_plan.bands:
                if int(element.column_id) in band.upstream_ids:
                    return band.id

        if cls._is_spanning_element(element, page_plan):
            return 0

        element_center_x = (element.bbox[0] + element.bbox[2]) / 2
        best_band = min(
            page_plan.bands,
            key=lambda band: abs(band.center_x - element_center_x),
        )
        return best_band.id

    @classmethod
    def _is_spanning_element(
        cls,
        element: AssemblyElement,
        page_plan: _PagePlan,
    ) -> bool:
        """여러 컬럼을 가로지르는 블록은 region 경계처럼 취급한다."""
        if element.bbox is None or page_plan.column_count <= 1:
            return False

        element_width = element.bbox[2] - element.bbox[0]
        if element_width <= 0:
            return False

        if (
            page_plan.page_width > 0
            and element_width >= page_plan.page_width * cls.SPANNING_MIN_WIDTH_RATIO
            and cls._crosses_column_boundary(element, page_plan)
        ):
            return True

        overlap_band_count = 0
        for band in page_plan.bands:
            overlap = cls._horizontal_overlap(element.bbox[0], element.bbox[2], band.left, band.right)
            if overlap >= element_width * cls.SPANNING_MIN_OVERLAP_RATIO:
                overlap_band_count += 1

        return overlap_band_count >= 2

    @classmethod
    def _estimate_column_region_start(
        cls,
        entries: List[_PageEntry],
        page_plan: _PagePlan,
    ) -> Optional[float]:
        """실제 다단 본문이 시작되는 대략적인 y 지점을 찾는다."""
        if page_plan.column_count <= 1:
            return None

        column_tops: List[float] = []
        for band in page_plan.bands:
            stable_entries = [
                entry
                for entry in entries
                if entry.resolved_column_id == band.id
                and not cls._crosses_column_boundary(entry.element, page_plan)
            ]
            if not stable_entries:
                continue
            column_tops.append(min(cls._entry_top(entry) for entry in stable_entries))

        if len(column_tops) < 2:
            return None
        return min(column_tops)

    @classmethod
    def _crosses_column_boundary(
        cls,
        element: AssemblyElement,
        page_plan: _PagePlan,
    ) -> bool:
        """컬럼 경계선을 가로지르는 블록인지 본다."""
        if element.bbox is None or page_plan.column_count <= 1:
            return False

        for band in page_plan.bands[:-1]:
            if element.bbox[0] < band.right < element.bbox[2]:
                return True
        return False

    @classmethod
    def _horizontal_overlap(
        cls,
        left_x1: float,
        left_x2: float,
        right_x1: float,
        right_x2: float,
    ) -> float:
        """두 가로 구간의 겹침 길이를 계산한다."""
        return max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))

    @classmethod
    def _find_region_index(
        cls,
        entry: _PageEntry,
        spanning_entries: List[_PageEntry],
    ) -> int:
        """spanning block 아래로 내려갈 때마다 region index를 하나씩 늘린다."""
        entry_top = cls._entry_top(entry)
        region_index = 0
        for spanning_entry in spanning_entries:
            if cls._entry_bottom(spanning_entry) <= entry_top:
                region_index += 1
                continue
            break
        return region_index

    @classmethod
    def _order_region_entries(
        cls,
        entries: List[_PageEntry],
        page_plan: _PagePlan,
        column_ids: List[int],
    ) -> List[_PageEntry]:
        """한 region 안에서는 컬럼별 top-down 정렬을 적용한다."""
        # R-ASM-05
        # 같은 region에서는 왼쪽 컬럼부터 컬럼 내부 y 오름차순으로 읽는다.
        ordered_entries: List[_PageEntry] = []
        for column_id in column_ids:
            column_entries = [
                entry
                for entry in entries
                if entry.resolved_column_id == column_id
            ]
            ordered_entries.extend(
                cls._sort_line_band_entries(column_entries, page_plan.same_line_threshold)
            )

        leftover_entries = [
            entry
            for entry in entries
            if entry.resolved_column_id not in set(column_ids)
        ]
        ordered_entries.extend(
            cls._sort_line_band_entries(leftover_entries, page_plan.same_line_threshold)
        )
        return ordered_entries

    @classmethod
    def _sort_line_band_entries(
        cls,
        entries: List[_PageEntry],
        same_line_threshold: float,
    ) -> List[_PageEntry]:
        """R-ASM-05, R-ASM-06에 따라 line band를 만든 뒤 좌->우로 정렬한다."""
        # R-ASM-05
        # 같은 컬럼 안에서는 기본적으로 y 오름차순을 따른다.
        # R-ASM-06
        # 중심 y 차이가 임계값 이하이면 같은 줄로 보고 x 기준으로 다시 정렬한다.
        sortable_entries = sorted(
            entries,
            key=lambda entry: (
                cls._entry_center_y(entry),
                cls._entry_top(entry),
                cls._entry_left(entry),
                entry.original_index,
            ),
        )
        if not sortable_entries:
            return []

        ordered_entries: List[_PageEntry] = []
        current_band: List[_PageEntry] = []
        current_band_center_y: Optional[float] = None

        for entry in sortable_entries:
            center_y = cls._entry_center_y(entry)
            if current_band_center_y is None:
                current_band = [entry]
                current_band_center_y = center_y
                continue

            if abs(center_y - current_band_center_y) <= same_line_threshold:
                current_band.append(entry)
                current_band_center_y = min(current_band_center_y, center_y)
                continue

            ordered_entries.extend(cls._sort_same_band_entries(current_band))
            current_band = [entry]
            current_band_center_y = center_y

        ordered_entries.extend(cls._sort_same_band_entries(current_band))
        return ordered_entries

    @classmethod
    def _sort_same_band_entries(cls, entries: List[_PageEntry]) -> List[_PageEntry]:
        """같은 line band 안에서는 좌->우를 우선한다."""
        return sorted(
            entries,
            key=lambda entry: (
                cls._entry_left(entry),
                cls._entry_top(entry),
                entry.original_index,
            ),
        )

    @classmethod
    def _entry_left(cls, entry: _PageEntry) -> float:
        """bbox가 없으면 기존 순서를 fallback으로 사용한다."""
        if entry.element.bbox is None:
            return float(entry.original_index)
        return float(entry.element.bbox[0])

    @classmethod
    def _entry_top(cls, entry: _PageEntry) -> float:
        """bbox가 없으면 기존 순서를 fallback으로 사용한다."""
        if entry.element.bbox is None:
            return float(entry.original_index)
        return float(entry.element.bbox[1])

    @classmethod
    def _entry_bottom(cls, entry: _PageEntry) -> float:
        """bbox가 없으면 기존 순서를 fallback으로 사용한다."""
        if entry.element.bbox is None:
            return float(entry.original_index)
        return float(entry.element.bbox[3])

    @classmethod
    def _entry_center_y(cls, entry: _PageEntry) -> float:
        """same-line 판정에 쓸 중심 y를 계산한다."""
        if entry.element.bbox is None:
            return float(entry.original_index)
        return float((entry.element.bbox[1] + entry.element.bbox[3]) / 2)

    @classmethod
    def _build_resolved_page_stats(
        cls,
        page_stat: PageStats,
        page_plan: _PagePlan,
        ordered_entries: List[_PageEntry],
    ) -> PageStats:
        """reading order 단계에서 계산한 컬럼 정보를 page stats에 반영한다."""
        metadata = dict(page_stat.metadata)
        metadata["reading_order"] = {
            "column_count": page_plan.column_count,
            "column_source": page_plan.column_source,
            "same_line_threshold": page_plan.same_line_threshold,
            "column_gap_threshold": page_plan.column_gap_threshold,
            "column_centers": [band.center_x for band in page_plan.bands],
            "ordered_element_ids": [entry.element.id for entry in ordered_entries],
        }

        return replace(
            page_stat,
            width=page_stat.width if page_stat.width is not None else page_plan.page_width,
            median_line_height=(
                page_stat.median_line_height
                if page_stat.median_line_height is not None
                else page_plan.line_height
            ),
            column_count=page_plan.column_count,
            metadata=metadata,
        )

    @classmethod
    def _build_next_relations(cls, ordered_elements: List[AssemblyElement]) -> List[BlockRelation]:
        """최종 reading order를 next edge로도 보존한다."""
        next_relations: List[BlockRelation] = []
        for current, following in zip(ordered_elements, ordered_elements[1:]):
            next_relations.append(
                BlockRelation(
                    type="next",
                    src=current.id,
                    dst=following.id,
                    score=1.0,
                    metadata={
                        "page": current.page,
                        "same_page": current.page == following.page,
                        "reading_order": (current.reading_order, following.reading_order),
                    },
                )
            )
        return next_relations

    @classmethod
    def _merge_next_relations(
        cls,
        existing_relations: List[BlockRelation],
        next_relations: List[BlockRelation],
    ) -> List[BlockRelation]:
        """기존 next relation은 새 계산 결과로 교체하고 나머지는 유지한다."""
        merged_relations = [
            relation
            for relation in existing_relations
            if relation.type != "next"
        ]
        merged_relations.extend(next_relations)
        return merged_relations

    @classmethod
    def _build_resolved_metadata(
        cls,
        previous_metadata: AssemblyMeta,
        reading_order_summary: Dict[str, Any],
    ) -> AssemblyMeta:
        """이전 메타데이터를 보존하면서 stage만 reading_order_resolved로 갱신한다."""
        details = dict(previous_metadata.details)
        details["upstream_stage"] = previous_metadata.stage
        details["reading_order"] = reading_order_summary

        return AssemblyMeta(
            stage="reading_order_resolved",
            adapter=previous_metadata.adapter,
            source=previous_metadata.source,
            details=details,
        )


__all__ = ["ReadingOrderResolver"]
