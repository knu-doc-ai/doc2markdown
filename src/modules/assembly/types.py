from __future__ import annotations

"""Assembly IR에서 공통으로 사용하는 Literal/alias 정의 모듈."""

from typing import Literal, Tuple


# bbox는 [x1, y1, x2, y2] 형식
BBox = Tuple[float, float, float, float]

AssemblyElementKind = Literal[
    "text",
    "heading",
    "list_item",
    "table",
    "figure",
    "caption",
    "note",
    "formula",
    "quote",
    "code_block",
    "header",
    "footer",
    "page_number",
    "noise",
]

BlockRelationType = Literal[
    "next",
    "child_of",
    "caption_of",
    "note_of",
]

AssemblyWarningLevel = Literal[
    "info",
    "warning",
    "error",
]

AssemblyWarningCode = Literal[
    "layout_missing_id",  # layout element에 원본 id가 없어 임시 id를 만든 경우
    "layout_missing_page",  # layout element에 page 정보가 없어 기본 page를 가정한 경우
    "table_missing_id",  # table 결과에 원본 table id가 없어 임시 id를 만든 경우
    "table_missing_page",  # table 결과에 page 정보가 없어 기본 page를 가정한 경우
    "orphan_caption",  # 어떤 object에도 연결되지 않은 caption block을 찾은 경우
    "orphan_note",  # 어떤 object에도 연결되지 않은 note block을 찾은 경우
    "orphan_table",  # caption 없이 남은 table ref를 찾은 경우
    "orphan_figure",  # caption 없이 남은 figure ref를 찾은 경우
    "empty_section",  # body 없이 heading만 남은 section을 찾은 경우
    "relation_conflict",  # next/child/caption/note 관계가 서로 충돌하는 경우
    "structure_orphan_block",  # 구조 트리에 귀속되지 못한 block을 찾은 경우
    "missing_geometry",  # bbox 같은 기하 정보가 빠진 block/ref를 찾은 경우
    "low_confidence_chunk",  # 필터링 후에도 남아 있는 저신뢰 chunk를 표시하는 경우
]

AssemblyStage = Literal[
    "adapter_seed",  # 업스트림 raw를 내부 IR 초안으로만 정규화한 상태
    "normalized",  # 공백 정리, 필터링, 라벨 정규화 등 전처리가 끝난 상태
    "reading_order_resolved",  # legacy: 예전 assembly reading order 단계 결과
    "structure_assembled",  # 문단/리스트/섹션/표 연결 등 구조 조립이 끝난 상태
    "validated",  # orphan/conflict/low-confidence 검증까지 마친 최종 상태
]

AssemblyAdapterType = Literal[
    "layout",  # layout 계열 입력을 정규화한 결과
    "table",  # table 계열 입력을 정규화한 결과
    "merged",  # layout/table 어댑터 결과를 병합한 결과
]

AssemblySourceType = Literal[
    "empty",  # 입력이 비어 있어 빈 결과를 만든 경우
    "raw",  # 별도 래핑 없이 raw payload 자체를 해석한 경우
    "layout_container",  # layout 전용 컨테이너 내부에서 payload를 찾은 경우
    "table_container",  # table 전용 컨테이너 내부에서 payload를 찾은 경우
    "direct_list",  # dict 래핑 없이 리스트 자체가 바로 들어온 경우
]

AssembledNodeType = Literal[
    "section",  # heading 기반 section subtree
    "paragraph_group",  # 여러 text block이 묶인 문단 그룹
    "list_group",  # list / list_item으로 변환될 리스트 그룹
    "table_ref",  # table IR를 가리키는 참조 노드
    "figure_ref",  # figure asset을 가리키는 참조 노드
    "note_ref",  # note block을 가리키는 참조 노드
]

# AssembledNodeType의 각 멤버를 코드에서 재사용하기 위한 상수
ASSEMBLED_NODE_SECTION: AssembledNodeType = "section"
ASSEMBLED_NODE_PARAGRAPH_GROUP: AssembledNodeType = "paragraph_group"
ASSEMBLED_NODE_LIST_GROUP: AssembledNodeType = "list_group"
ASSEMBLED_NODE_TABLE_REF: AssembledNodeType = "table_ref"
ASSEMBLED_NODE_FIGURE_REF: AssembledNodeType = "figure_ref"
ASSEMBLED_NODE_NOTE_REF: AssembledNodeType = "note_ref"

# 병합 단계에서 ref 식별 필드명을 문자열 상수로 고정한다.
TABLE_REF_ID_ATTR = "table_id"
FIGURE_REF_ID_ATTR = "figure_id"
NOTE_REF_ID_ATTR = "note_id"

# 병합된 document.metadata 안에서 어댑터별 메타데이터를 구분하는 키
MERGED_METADATA_LAYOUT_KEY = "layout"
MERGED_METADATA_TABLE_KEY = "table"


__all__ = [
    "BBox",
    "AssemblyElementKind",
    "BlockRelationType",
    "AssemblyWarningLevel",
    "AssemblyWarningCode",
    "AssemblyStage",
    "AssemblyAdapterType",
    "AssemblySourceType",
    "AssembledNodeType",
    "ASSEMBLED_NODE_SECTION",
    "ASSEMBLED_NODE_PARAGRAPH_GROUP",
    "ASSEMBLED_NODE_LIST_GROUP",
    "ASSEMBLED_NODE_TABLE_REF",
    "ASSEMBLED_NODE_FIGURE_REF",
    "ASSEMBLED_NODE_NOTE_REF",
    "TABLE_REF_ID_ATTR",
    "FIGURE_REF_ID_ATTR",
    "NOTE_REF_ID_ATTR",
    "MERGED_METADATA_LAYOUT_KEY",
    "MERGED_METADATA_TABLE_KEY",
]
