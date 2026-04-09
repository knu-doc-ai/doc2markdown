from __future__ import annotations

"""Markdown Rendering 단계에서 사용하는 결과 IR 정의."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


RenderWarningLevel = Literal[
    "info",
    "warning",
    "error",
]

RenderWarningCode = Literal[
    "renderer_contract_only",
    "unsupported_render_input",
    "invalid_assembly_stage",
    "unsupported_node_type",
    "empty_heading",
    "empty_paragraph",
    "empty_list_item",
    "empty_note",
    "table_crop_fallback",
    "table_placeholder",
    "figure_placeholder",
]


@dataclass
class SerializableRenderIR:
    """Rendering 결과 dataclass를 dict로 직렬화하기 위한 공통 인터페이스."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RenderWarning(SerializableRenderIR):
    """렌더링 단계 경고/보류 정보를 표현한다."""

    code: RenderWarningCode
    message: str
    level: RenderWarningLevel = "warning"
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderStats(SerializableRenderIR):
    """렌더링 단계에서 수집한 단순 통계."""

    input_stage: Optional[str] = None
    ordered_element_count: int = 0
    root_child_count: int = 0
    section_count: int = 0
    table_ref_count: int = 0
    figure_ref_count: int = 0
    note_ref_count: int = 0
    warning_count: int = 0
    placeholder_count: int = 0
    table_fallback_count: int = 0
    rendered_block_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarkdownRenderResult(SerializableRenderIR):
    """Renderer의 최종 반환 계약."""

    markdown: str = ""
    warnings: List[RenderWarning] = field(default_factory=list)
    stats: RenderStats = field(default_factory=RenderStats)


__all__ = [
    "RenderWarningLevel",
    "RenderWarningCode",
    "SerializableRenderIR",
    "RenderWarning",
    "RenderStats",
    "MarkdownRenderResult",
]
