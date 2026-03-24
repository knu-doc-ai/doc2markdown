from __future__ import annotations

"""Assembly 공개 입력 어댑터 facade."""

from typing import Any

from modules.assembly._layout_adapter import LayoutAdapterMixin
from modules.assembly._table_adapter import TableAdapterMixin
from modules.assembly.ir import AssemblyResult, AssembledDocument
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

        layout_source = cls._resolve_layout_source(raw)
        table_source = cls._resolve_table_source(raw)

        if layout_source is None and table_source is None and raw is not None:
            layout_source = raw

        layout_result = cls.from_layout_output(layout_source)
        table_result = cls.from_table_output(table_source)
        return cls._merge_results(layout_result, table_result, raw)

    @classmethod
    def _merge_results(
        cls,
        layout_result: AssemblyResult,
        table_result: AssemblyResult,
        raw: Any,
    ) -> AssemblyResult:
        """layout/table 어댑터 결과를 하나의 AssemblyResult로 합친다."""
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
                table_result.document.table_refs,
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
            metadata={
                MERGED_METADATA_LAYOUT_KEY: dict(layout_result.document.metadata),
                MERGED_METADATA_TABLE_KEY: dict(table_result.document.metadata),
            },
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


def from_layout_output(raw: Any) -> AssemblyResult:
    """외부 layout 출력을 Assembly IR로 적응시키는 공개 함수."""
    return AssemblyInputAdapter.from_layout_output(raw)


def from_table_output(raw: Any) -> AssemblyResult:
    """외부 table 출력을 Assembly IR로 적응시키는 공개 함수."""
    return AssemblyInputAdapter.from_table_output(raw)


__all__ = [
    "AssemblyInputAdapter",
    "from_layout_output",
    "from_table_output",
]
