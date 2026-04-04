from __future__ import annotations

"""Assembly 공개 입력 어댑터 facade."""

from typing import Any, List, Optional, Tuple

from modules.assembly._layout_adapter import LayoutAdapterMixin
from modules.assembly._table_adapter import TableAdapterMixin
from modules.assembly.ir import AssemblyResult, AssembledDocument, TableRef
from modules.assembly.types import (
    FIGURE_REF_ID_ATTR,
    MERGED_METADATA_LAYOUT_KEY,
    MERGED_METADATA_TABLE_KEY,
    NOTE_REF_ID_ATTR,
    TABLE_REF_ID_ATTR,
)


class AssemblyInputAdapter(LayoutAdapterMixin, TableAdapterMixin):
    """
    외부에서 사용하는 Assembly 입력 어댑터 진입점.

    공개 import 경로는 유지하고, 내부 구현만 layout/table/공통 유틸로 분리한다.
    """

    @classmethod
    def from_raw(cls, raw: Any) -> AssemblyResult:
        """단일 payload 안에서 layout/table 출력을 찾아 조립 가능한 초기 IR로 변환한다."""
        if isinstance(raw, AssemblyResult):
            return raw

        if isinstance(raw, AssembledDocument):
            return AssemblyResult(document=raw, raw=raw)

        layout_source = cls._resolve_layout_source(raw)  # layout 후보를 찾는다.
        table_source = cls._resolve_table_source(raw)  # table 후보를 찾는다.

        # 둘 다 찾지 못했고 raw 자체는 비어 있지 않으면, raw를 layout 입력으로 한 번 더 해석해 본다.
        if layout_source is None and table_source is None and raw is not None:
            layout_source = raw

        layout_result = cls.from_layout_output(layout_source)  # layout seed를 만든다.
        table_result = cls.from_table_output(table_source)  # table seed를 만든다.
        return cls._merge_results(layout_result, table_result, raw)  # 두 결과를 병합한다.

    @classmethod
    def from_outputs(cls, layout_output: Any, table_output: Any = None) -> AssemblyResult:
        """layout/table 출력을 명시적으로 받아 조립 seed를 생성한다."""
        return cls.from_raw(
            {
                "layout_output": layout_output,
                "table_output": table_output,
            }
        )

    @classmethod
    def _merge_results(
        cls,
        layout_result: AssemblyResult,
        table_result: AssemblyResult,
        raw: Any,
    ) -> AssemblyResult:
        """layout/table 어댑터 결과를 하나의 AssemblyResult로 합친다."""
        linked_table_refs = cls._link_table_refs(
            layout_result.document.table_refs,
            table_result.document.table_refs,
        )
        merged_document_metadata = {
            **dict(layout_result.document.metadata),
            **dict(table_result.document.metadata),
            MERGED_METADATA_LAYOUT_KEY: dict(layout_result.document.metadata),
            MERGED_METADATA_TABLE_KEY: dict(table_result.document.metadata),
        }

        merged_document = AssembledDocument(
            title_candidate=layout_result.document.title_candidate or table_result.document.title_candidate,
            title_source_block_ids=(
                list(layout_result.document.title_source_block_ids)
                or list(table_result.document.title_source_block_ids)
            ),
            children=list(layout_result.document.children),
            sections=list(layout_result.document.sections),
            table_refs=cls._merge_ref_list(
                layout_result.document.table_refs,
                linked_table_refs,
                id_attr=TABLE_REF_ID_ATTR,
            ),
            figure_refs=cls._merge_ref_list(
                layout_result.document.figure_refs,
                table_result.document.figure_refs,
                id_attr=FIGURE_REF_ID_ATTR,
            ),
            note_refs=cls._merge_ref_list(
                layout_result.document.note_refs,
                table_result.document.note_refs,
                id_attr=NOTE_REF_ID_ATTR,
            ),
            figure_assets_metadata={
                **dict(layout_result.document.figure_assets_metadata),
                **dict(table_result.document.figure_assets_metadata),
            },
            metadata=merged_document_metadata,
            raw=raw,
        )

        return AssemblyResult(
            ordered_elements=list(layout_result.ordered_elements),
            block_relations=list(layout_result.block_relations) + list(table_result.block_relations),
            document=merged_document,
            page_stats=list(layout_result.page_stats),
            warnings=list(layout_result.warnings) + list(table_result.warnings),
            metadata=cls._build_adapter_metadata(
                stage="adapter_seed",
                adapter="merged",
                source="raw",
                layout=layout_result.metadata,
                table=table_result.metadata,
            ),
            raw=raw,
        )

    @classmethod
    def _link_table_refs(
        cls,
        layout_table_refs: List[TableRef],
        table_table_refs: List[TableRef],
    ) -> List[TableRef]:
        """table markdown 결과를 layout table ref와 느슨하게 연결한다."""
        if not layout_table_refs or not table_table_refs:
            return list(table_table_refs)

        remaining_layout = list(layout_table_refs)
        linked_refs: List[TableRef] = []

        for index, table_ref in enumerate(table_table_refs):
            layout_ref, strategy = cls._match_layout_table_ref(
                table_ref,
                remaining_layout,
                index,
            )
            if layout_ref is None or strategy is None:
                linked_refs.append(table_ref)
                continue

            linked_refs.append(cls._merge_linked_table_ref(layout_ref, table_ref, strategy))
            remaining_layout = [
                candidate
                for candidate in remaining_layout
                if candidate.table_id != layout_ref.table_id
            ]

        return linked_refs

    @classmethod
    def _match_layout_table_ref(
        cls,
        table_ref: TableRef,
        layout_candidates: List[TableRef],
        index: int,
    ) -> Tuple[Optional[TableRef], Optional[str]]:
        """table ref와 가장 그럴듯한 layout table ref를 찾는다."""
        if not layout_candidates:
            return None, None

        for candidate in layout_candidates:
            if candidate.table_id == table_ref.table_id:
                return candidate, "table_id"

        table_source_ids = set(table_ref.source_block_ids)
        if table_source_ids:
            for candidate in layout_candidates:
                if table_source_ids.intersection(candidate.source_block_ids):
                    return candidate, "source_block_id"

        table_image_path = cls._normalize_str(
            table_ref.metadata.get("crop_path")
            or table_ref.metadata.get("image_path")
            or table_ref.metadata.get("table_image_path")
        )
        if table_image_path is not None:
            for candidate in layout_candidates:
                candidate_image_path = cls._normalize_str(
                    candidate.metadata.get("crop_path")
                    or candidate.metadata.get("image_path")
                    or candidate.metadata.get("table_image_path")
                )
                if candidate_image_path == table_image_path:
                    return candidate, "crop_path"

        if not table_ref.metadata.get("generated_page") and table_ref.bbox is not None:
            best_candidate: Optional[TableRef] = None
            best_score = 0.0
            for candidate in layout_candidates:
                if candidate.page != table_ref.page:
                    continue
                iou = cls._bbox_iou(candidate.bbox, table_ref.bbox) or 0.0
                if iou > best_score:
                    best_candidate = candidate
                    best_score = iou

            if best_candidate is not None and best_score >= 0.5:
                return best_candidate, "page_bbox"

        if not table_ref.metadata.get("generated_page"):
            same_page_candidates = [
                candidate
                for candidate in layout_candidates
                if candidate.page == table_ref.page
            ]
            if len(same_page_candidates) == 1:
                return same_page_candidates[0], "page_only"

        if index < len(layout_candidates):
            return layout_candidates[index], "document_order"

        return None, None

    @classmethod
    def _merge_linked_table_ref(
        cls,
        layout_ref: TableRef,
        table_ref: TableRef,
        strategy: str,
    ) -> TableRef:
        """layout 위치 정보와 table markdown 내용을 하나의 TableRef로 합친다."""
        metadata = {
            **dict(layout_ref.metadata),
            **dict(table_ref.metadata),
            "layout_table_id": layout_ref.table_id,
            "link_strategy": strategy,
        }
        if table_ref.table_id != layout_ref.table_id:
            metadata["table_output_id"] = table_ref.table_id

        return TableRef(
            table_id=layout_ref.table_id,
            page=layout_ref.page,
            bbox=layout_ref.bbox or table_ref.bbox,
            caption_id=table_ref.caption_id or layout_ref.caption_id,
            note_ids=cls._merge_unique_ids(layout_ref.note_ids, table_ref.note_ids),
            source_block_ids=cls._merge_unique_ids(
                layout_ref.source_block_ids or [layout_ref.table_id],
                table_ref.source_block_ids,
            ),
            metadata=metadata,
            raw={
                "layout": layout_ref.raw,
                "table": table_ref.raw,
            },
        )


def from_layout_output(raw: Any) -> AssemblyResult:
    """외부 layout 출력을 Assembly IR로 적응시키는 공개 함수."""
    return AssemblyInputAdapter.from_layout_output(raw)


def from_table_output(raw: Any) -> AssemblyResult:
    """외부 table 출력을 Assembly IR로 적응시키는 공개 함수."""
    return AssemblyInputAdapter.from_table_output(raw)


def from_outputs(layout_output: Any, table_output: Any = None) -> AssemblyResult:
    """layout/table 출력을 명시적으로 받아 Assembly IR seed로 만든다."""
    return AssemblyInputAdapter.from_outputs(layout_output, table_output)


__all__ = [
    "AssemblyInputAdapter",
    "from_layout_output",
    "from_table_output",
    "from_outputs",
]
