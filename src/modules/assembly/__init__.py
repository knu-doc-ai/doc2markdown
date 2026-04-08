"""
Assembly 패키지의 공개 API facade

외부에서는 이 파일만 통해 import하면 되고,
실제 구현은 types / ir / adapters / service 모듈로 분리한다.
"""

from modules.assembly.adapters import AssemblyInputAdapter, from_layout_output, from_table_output
from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    AssemblyWarning,
    AssembledDocument,
    AssembledNode,
    BlockRelation,
    FigureRef,
    ListGroup,
    ListGroupItem,
    PageStats,
    ParagraphGroup,
    NoteRef,
    SerializableIR,
    SectionNode,
    TableRef,
)
from modules.assembly.normalize_filter import NormalizeFilter
from modules.assembly.service import DocumentAssembler
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

__all__ = [
    "BBox",
    "ASSEMBLED_NODE_SECTION",
    "ASSEMBLED_NODE_PARAGRAPH_GROUP",
    "ASSEMBLED_NODE_LIST_GROUP",
    "ASSEMBLED_NODE_TABLE_REF",
    "ASSEMBLED_NODE_FIGURE_REF",
    "ASSEMBLED_NODE_NOTE_REF",
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
    "AssemblyInputAdapter",
    "from_layout_output",
    "from_table_output",
    "NormalizeFilter",
    "DocumentAssembler",
]
