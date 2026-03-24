from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


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
    bbox: Optional[BBox] = None
    caption_id: Optional[str] = None
    # 연결된 주석 블록 ID 목록
    note_ids: List[str] = field(default_factory=list)
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
class AssembledDocument(SerializableIR):
    """
    조립된 문서 구조의 루트 객체

    section/tree의 최종 모양은 아직 확정 전이므로 내부 노드는
    우선 dict 기반으로 열어두고 table 참조만 별도 타입으로 유지
    """

    # heading 기반 section tree 저장용
    sections: List[Dict[str, Any]] = field(default_factory=list)
    # section 바깥 루트 레벨 자식 노드 저장용
    children: List[Dict[str, Any]] = field(default_factory=list)
    # 문서 차원 table 참조 목록
    table_refs: List[TableRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class AssemblyWarning(SerializableIR):
    """조립 단계에서 발견한 경고/보류 이슈를 표현"""

    # 규칙/검증 단계 warning code
    code: str
    message: str
    # warning, error, info 등 심각도 구분용
    level: AssemblyWarningLevel = "warning"
    page: Optional[int] = None
    # 관련 블록 추적용
    element_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class DocumentAssembler:
    """
    Assembly 규칙 엔진의 진입점

    현재 단계에서는 내부 표준 IR 컨테이너만 먼저 고정하고
    실제 Normalize/Reading Order/Structure 조립 로직은 이후 단계에서 확장
    """

    def build(self, raw: Any) -> AssemblyResult:
        """
        아직 조립 로직이 없으므로 입력을 표준 결과 껍데기에 보관해 반환

        이렇게 두면 pipeline import는 바로 살아나고 다음 단계에서
        어댑터와 규칙 엔진을 붙일 때 반환 타입을 다시 바꾸지 않아도 됨
        """
        if isinstance(raw, AssemblyResult):
            return raw

        if isinstance(raw, AssembledDocument):
            return AssemblyResult(document=raw, raw=raw)

        return AssemblyResult(
            document=AssembledDocument(raw=raw),
            metadata={"stage": "ir_bootstrap"},
            raw=raw,
        )


__all__ = [
    "BBox",
    "AssemblyElementKind",
    "BlockRelationType",
    "AssemblyWarningLevel",
    "SerializableIR",
    "AssemblyElement",
    "TableRef",
    "PageStats",
    "BlockRelation",
    "AssembledDocument",
    "AssemblyWarning",
    "AssemblyResult",
    "DocumentAssembler",
]
