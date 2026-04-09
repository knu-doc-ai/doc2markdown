from __future__ import annotations

"""Assembly 내부 표준 IR dataclass 정의 모듈."""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional, TypeAlias

from modules.assembly.types import (
    ASSEMBLED_NODE_FIGURE_REF,
    ASSEMBLED_NODE_LIST_GROUP,
    ASSEMBLED_NODE_NOTE_REF,
    ASSEMBLED_NODE_PARAGRAPH_GROUP,
    ASSEMBLED_NODE_SECTION,
    ASSEMBLED_NODE_TABLE_REF,
    AssemblyAdapterType,
    AssemblyElementKind,
    AssemblySourceType,
    AssemblyStage,
    AssembledNodeType,
    AssemblyWarningCode,
    AssemblyWarningLevel,
    BBox,
    BlockRelationType,
)


def _serialize_value(value: Any) -> Any:
    """dataclass 기반 IR을 사전 형태로 안정적으로 변환"""
    if is_dataclass(value):
        return {
            field_info.name: _serialize_value(getattr(value, field_info.name))
            for field_info in fields(value)
        }

    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    return value


@dataclass
class SerializableIR:
    """공통 직렬화 인터페이스"""

    def to_dict(self) -> Dict[str, Any]:
        return _serialize_value(self)


@dataclass
class AssemblyElement(SerializableIR):
    """
    Assembly 내부에서 사용하는 표준 블록 단위

    다른 팀 출력 스펙이 바뀌더라도 이 객체로 한 번 정규화하면
    이후 단계는 동일한 필드만 바라보도록 만드는 것이 목적
    """

    id: str
    page: int
    kind: AssemblyElementKind = "text"
    bbox: Optional[BBox] = None
    text: Optional[str] = None
    # 외부 모듈 label 원형 보존용. 정규화된 값은 kind에 저장
    label: Optional[str] = None
    # upstream confidence 전달용
    confidence: Optional[float] = None
    # reading order 단계에서 계산될 컬럼 번호
    column_id: Optional[int] = None
    # 최종 읽기 순서 인덱스
    reading_order: Optional[int] = None
    # 구조 조립 후 부모 블록 또는 section 참조용
    parent_id: Optional[str] = None
    # 정식 필드 승격 전 보조 정보 저장용
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 원본 payload 추적용
    raw: Any = None


@dataclass
class TableRef(SerializableIR):
    """표 본문과 표 관련 부속 정보를 느슨하게 연결하기 위한 참조 객체"""

    table_id: str
    page: int
    # AssembledNodeType 중 table_ref 멤버
    type: Literal["table_ref"] = ASSEMBLED_NODE_TABLE_REF
    bbox: Optional[BBox] = None
    caption_id: Optional[str] = None
    # 연결된 주석 블록 ID 목록
    note_ids: List[str] = field(default_factory=list)
    # IR Builder의 source_block_ids로 직접 넘기기 위한 provenance 정보
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class PageStats(SerializableIR):
    """페이지 단위 통계값을 담는 컨테이너"""

    page: int
    width: Optional[float] = None
    height: Optional[float] = None
    # threshold 계산의 기준값
    median_line_height: Optional[float] = None
    # heading 추정 등의 기준값
    body_font_size: Optional[float] = None
    # reading order 단계에서 추정된 컬럼 수
    column_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class BlockRelation(SerializableIR):
    """블록 간 관계를 edge 형태로 표현"""

    type: BlockRelationType
    # 관계 출발 블록 ID
    src: str
    # 관계 도착 블록 ID
    dst: str
    # 규칙 기반 또는 후처리 기반 신뢰도 점수
    score: Optional[float] = None
    # 규칙 ID, 거리값 등 관계 판정 근거 저장용
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 원본 관계 판단 근거 보관용
    raw: Any = None


@dataclass
class ParagraphGroup(SerializableIR):
    """여러 text block을 문단 단위로 묶은 조립 노드"""

    # AssembledNodeType 중 paragraph_group 멤버
    type: Literal["paragraph_group"] = ASSEMBLED_NODE_PARAGRAPH_GROUP
    id: str = ""
    block_ids: List[str] = field(default_factory=list)
    text: Optional[str] = None
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class ListGroupItem(SerializableIR):
    """리스트 항목 후보를 보관하는 조립 단위"""

    block_ids: List[str] = field(default_factory=list)
    text: Optional[str] = None
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class ListGroup(SerializableIR):
    """IR Builder의 list / list_item node 생성에 사용할 리스트 그룹"""

    # AssembledNodeType 중 list_group 멤버
    type: Literal["list_group"] = ASSEMBLED_NODE_LIST_GROUP
    id: str = ""
    ordered: Optional[bool] = None
    items: List[ListGroupItem] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class FigureRef(SerializableIR):
    """그림 asset 및 caption 연결 정보를 보관하는 참조 객체"""

    figure_id: str
    page: int
    # AssembledNodeType 중 figure_ref 멤버
    type: Literal["figure_ref"] = ASSEMBLED_NODE_FIGURE_REF
    bbox: Optional[BBox] = None
    caption_id: Optional[str] = None
    asset_path: Optional[str] = None
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class NoteRef(SerializableIR):
    """독립 note 또는 object 부속 note를 나타내는 참조 객체"""

    note_id: str
    page: int
    # AssembledNodeType 중 note_ref 멤버
    type: Literal["note_ref"] = ASSEMBLED_NODE_NOTE_REF
    bbox: Optional[BBox] = None
    text: Optional[str] = None
    target_id: Optional[str] = None
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class SectionNode(SerializableIR):
    """heading 기준으로 구성된 section subtree"""

    # AssembledNodeType 중 section 멤버
    type: Literal["section"] = ASSEMBLED_NODE_SECTION
    id: str = ""
    level: Optional[int] = None
    title: Optional[str] = None
    heading_block_id: Optional[str] = None
    source_block_ids: List[str] = field(default_factory=list)
    children: List["AssembledNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


AssembledNode: TypeAlias = (
    SectionNode
    | ParagraphGroup
    | ListGroup
    | TableRef
    | FigureRef
    | NoteRef
)


@dataclass
class AssembledDocument(SerializableIR):
    """
    조립된 문서 구조의 루트 객체

    canonical root sequence는 children에 보관하고,
    sections는 heading 기반 탐색을 위한 section index/legacy view로 유지한다.
    """

    # IR Builder가 document.title 후보로 사용할 수 있는 명시적 제목 후보
    title_candidate: Optional[str] = None
    # title 후보를 만든 source block id 목록
    title_source_block_ids: List[str] = field(default_factory=list)
    # root 레벨 조립 노드의 canonical ordered sequence
    children: List[AssembledNode] = field(default_factory=list)
    # heading 기반 section tree index/legacy view
    sections: List[SectionNode] = field(default_factory=list)
    # 문서 차원 table 참조 목록
    table_refs: List[TableRef] = field(default_factory=list)
    # 문서 차원 figure 참조 목록
    figure_refs: List[FigureRef] = field(default_factory=list)
    # 문서 차원 note 참조 목록
    note_refs: List[NoteRef] = field(default_factory=list)
    # figure asset path 등 figure metadata를 명시적으로 전달하기 위한 슬롯
    figure_assets_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class AssemblyWarning(SerializableIR):
    """조립 단계에서 발견한 경고/보류 이슈를 표현"""

    # 규칙/검증 단계 warning code
    code: AssemblyWarningCode
    message: str
    # warning, error, info 등 심각도 구분용
    level: AssemblyWarningLevel = "warning"
    page: Optional[int] = None
    # 관련 블록 추적용
    element_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class AssemblyMeta(SerializableIR):
    """
    AssemblyResult 메타데이터 전용 타입

    stage / adapter / source 같은 공통 상태값은 고정 필드로 두고,
    element_count 같은 보조 정보는 details에 모아둔다.
    """

    # adapter_seed -> normalized -> reading_order_resolved
    # -> structure_assembled -> validated 순으로 진행된다.
    stage: Optional[AssemblyStage] = None
    # 현재 결과를 직접 생성한 어댑터 또는 조립 주체
    adapter: Optional[AssemblyAdapterType] = None
    # 입력이 비어 있었는지, 리스트였는지, 복합 payload 내부 컨테이너였는지 표시
    source: Optional[AssemblySourceType] = None
    # 공통 필드로 승격할 필요가 없는 보조 메타데이터 저장용
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssemblyResult(SerializableIR):
    """Assembly 단계의 표준 반환 객체"""

    # reading order가 반영된 블록 목록
    ordered_elements: List[AssemblyElement] = field(default_factory=list)
    # 블록 간 관계 edge 목록
    block_relations: List[BlockRelation] = field(default_factory=list)
    # 최종 조립 결과 루트
    document: AssembledDocument = field(default_factory=AssembledDocument)
    # 페이지 단위 통계 정보
    page_stats: List[PageStats] = field(default_factory=list)
    # 조립/검증 단계 경고 목록
    warnings: List[AssemblyWarning] = field(default_factory=list)
    # 현재 결과가 어떤 단계/입력/어댑터에서 만들어졌는지 설명하는 구조화 메타데이터
    metadata: AssemblyMeta = field(default_factory=AssemblyMeta)
    raw: Any = None


__all__ = [
    "BBox",
    "AssemblyElementKind",
    "AssembledNodeType",
    "BlockRelationType",
    "AssemblyWarningLevel",
    "AssemblyWarningCode",
    "AssemblyStage",
    "AssemblyAdapterType",
    "AssemblySourceType",
    "SerializableIR",
    "AssemblyElement",
    "ParagraphGroup",
    "ListGroupItem",
    "ListGroup",
    "TableRef",
    "FigureRef",
    "NoteRef",
    "PageStats",
    "BlockRelation",
    "SectionNode",
    "AssembledNode",
    "AssembledDocument",
    "AssemblyWarning",
    "AssemblyMeta",
    "AssemblyResult",
]
