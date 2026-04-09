from __future__ import annotations

"""Markdown Renderer 계약과 입력 검증을 담당하는 서비스."""

from collections.abc import Mapping
from typing import Any

from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    AssemblyWarning,
    AssembledDocument,
    BlockRelation,
    FigureRef,
    ListGroup,
    ListGroupItem,
    NoteRef,
    PageStats,
    ParagraphGroup,
    SectionNode,
    TableRef,
)
from modules.rendering.ir import MarkdownRenderResult, RenderStats, RenderWarning


class MarkdownRenderer:
    """
    Rendering 단계의 공개 진입점.

    현재 단계에서는 계약만 고정한다.
    - 입력: validated AssemblyResult 또는 그 직렬화 dict
    - 출력: markdown / warnings / stats
    - 비목표: 구조 재판단
    """

    def render(self, assembly_result: AssemblyResult | dict[str, Any]) -> MarkdownRenderResult:
        """입력 계약을 검증하고 document.children 기반 Markdown 골격을 만든다."""
        normalized_result = self._coerce_assembly_result(assembly_result)
        self._validate_stage(normalized_result)

        warnings: list[RenderWarning] = []
        render_context = self._build_render_context(normalized_result)
        blocks = self._render_nodes(
            nodes=normalized_result.document.children,
            warnings=warnings,
            render_context=render_context,
        )
        markdown, cleanup_report = self._finalize_markdown("\n\n".join(blocks))

        stats = self._build_stats(
            result=normalized_result,
            warnings=warnings,
            rendered_block_count=self._count_renderable_blocks(normalized_result.document.children),
            cleanup_report=cleanup_report,
        )
        return MarkdownRenderResult(
            markdown=markdown,
            warnings=warnings,
            stats=stats,
        )

    @classmethod
    def _coerce_assembly_result(cls, value: AssemblyResult | dict[str, Any]) -> AssemblyResult:
        """Renderer 입력을 AssemblyResult로 정규화한다."""
        if isinstance(value, AssemblyResult):
            return value

        if isinstance(value, Mapping):
            return cls._assembly_result_from_dict(dict(value))

        raise TypeError(
            "MarkdownRenderer는 AssemblyResult(stage='validated') 또는 그 직렬화 dict만 받습니다."
        )

    @classmethod
    def _validate_stage(cls, result: AssemblyResult) -> None:
        """Rendering이 validated 결과만 받도록 강제한다."""
        stage = result.metadata.stage if result.metadata is not None else None
        if stage != "validated":
            raise ValueError(
                "MarkdownRenderer는 AssemblyResult(stage='validated') 또는 그 직렬화 dict만 받습니다. "
                f"현재 stage={stage!r}"
            )

    @classmethod
    def _build_stats(
        cls,
        result: AssemblyResult,
        warnings: list[RenderWarning],
        rendered_block_count: int,
        cleanup_report: dict[str, Any],
    ) -> RenderStats:
        """현재 입력 구조를 기반으로 기본 통계를 만든다."""
        placeholder_count = sum(
            1
            for warning in warnings
            if warning.code in {"table_placeholder", "figure_placeholder"}
        )
        table_fallback_count = sum(
            1
            for warning in warnings
            if warning.code == "table_crop_fallback"
        )

        return RenderStats(
            input_stage=result.metadata.stage,
            ordered_element_count=len(result.ordered_elements),
            root_child_count=len(result.document.children),
            section_count=len(result.document.sections),
            table_ref_count=len(result.document.table_refs),
            figure_ref_count=len(result.document.figure_refs),
            note_ref_count=len(result.document.note_refs),
            warning_count=len(warnings),
            placeholder_count=placeholder_count,
            table_fallback_count=table_fallback_count,
            rendered_block_count=rendered_block_count,
            metadata={
                "renderer_contract_fixed": True,
                "used_document_children": True,
                "supported_node_types": [
                    "section",
                    "paragraph_group",
                    "list_group",
                    "table_ref",
                    "figure_ref",
                    "note_ref",
                ],
                "document_title_candidate": result.document.title_candidate,
                "render_report": {
                    "cleanup": cleanup_report,
                    "placeholder_count": placeholder_count,
                    "table_fallback_count": table_fallback_count,
                    "warning_code_counts": cls._summarize_warning_codes(warnings),
                },
            },
        )

    @classmethod
    def _count_renderable_blocks(cls, nodes: list[Any]) -> int:
        """현재 단계에서 실제로 렌더링 가능한 block 개수를 센다."""
        count = 0

        for node in nodes:
            if isinstance(node, SectionNode):
                title = cls._normalize_inline_text(node.title)
                count += 1 if title else 0
                count += cls._count_renderable_blocks(node.children)
                continue

            if isinstance(node, ParagraphGroup):
                count += 1 if cls._normalize_block_text(node.text) else 0
                continue

            if isinstance(node, ListGroup):
                if any(cls._normalize_block_text(item.text) for item in node.items):
                    count += 1
                continue

            if isinstance(node, TableRef):
                count += 1
                continue

            if isinstance(node, FigureRef):
                count += 1
                continue

            if isinstance(node, NoteRef):
                text = cls._normalize_block_text(node.text)
                count += 1 if text else 0

        return count

    @staticmethod
    def _summarize_warning_codes(warnings: list[RenderWarning]) -> dict[str, int]:
        """warning code별 개수를 요약한다."""
        summary: dict[str, int] = {}
        for warning in warnings:
            summary[warning.code] = summary.get(warning.code, 0) + 1
        return summary

    @classmethod
    def _render_nodes(
        cls,
        nodes: list[Any],
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
    ) -> list[str]:
        """document.children 순서를 그대로 따라 Markdown block 목록을 만든다."""
        rendered_blocks: list[str] = []

        for node in nodes:
            rendered = cls._render_node(
                node=node,
                warnings=warnings,
                render_context=render_context,
            )
            if rendered:
                rendered_blocks.append(rendered)

        return rendered_blocks

    @classmethod
    def _render_node(
        cls,
        node: Any,
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
    ) -> str:
        """조립 노드 타입별로 기본 Markdown block을 만든다."""
        if isinstance(node, SectionNode):
            return cls._render_section(node, warnings, render_context)

        if isinstance(node, ParagraphGroup):
            return cls._render_paragraph_group(node, warnings)

        if isinstance(node, ListGroup):
            return cls._render_list_group(node, warnings)

        if isinstance(node, TableRef):
            return cls._render_table_ref(node, warnings, render_context)

        if isinstance(node, FigureRef):
            return cls._render_figure_ref(node, warnings, render_context)

        if isinstance(node, NoteRef):
            return cls._render_note_ref(node, warnings, render_context)

        node_type = getattr(node, "type", type(node).__name__)
        node_id = (
            getattr(node, "id", None)
            or getattr(node, "table_id", None)
            or getattr(node, "figure_id", None)
            or getattr(node, "note_id", None)
        )
        warnings.append(
            RenderWarning(
                code="unsupported_node_type",
                message=f"현재 단계에서는 {node_type!r} 렌더링을 아직 지원하지 않습니다.",
                node_id=node_id,
                metadata={"node_type": node_type},
            )
        )
        return ""

    @classmethod
    def _render_section(
        cls,
        section: SectionNode,
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
    ) -> str:
        """section 노드를 heading + child block 묶음으로 렌더링한다."""
        title = cls._normalize_inline_text(section.title)
        blocks: list[str] = []

        if title:
            level = cls._normalize_heading_level(section.level)
            blocks.append(f"{'#' * level} {title}")
        else:
            warnings.append(
                RenderWarning(
                    code="empty_heading",
                    message="section title이 비어 있어 heading 출력은 생략합니다.",
                    node_id=section.id,
                    metadata={"level": section.level},
                )
            )

        child_blocks = cls._render_nodes(
            nodes=section.children,
            warnings=warnings,
            render_context=render_context,
        )
        blocks.extend(child_blocks)
        return "\n\n".join(blocks).strip()

    @classmethod
    def _render_paragraph_group(
        cls,
        paragraph: ParagraphGroup,
        warnings: list[RenderWarning],
    ) -> str:
        """paragraph_group을 일반 문단으로 렌더링한다."""
        text = cls._normalize_block_text(paragraph.text)
        if text:
            return text

        warnings.append(
            RenderWarning(
                code="empty_paragraph",
                message="paragraph_group text가 비어 있어 출력하지 않습니다.",
                node_id=paragraph.id,
                metadata={"block_ids": list(paragraph.block_ids)},
            )
        )
        return ""

    @classmethod
    def _render_list_group(
        cls,
        list_group: ListGroup,
        warnings: list[RenderWarning],
    ) -> str:
        """list_group을 ordered/unordered Markdown list로 렌더링한다."""
        ordered = bool(list_group.ordered)
        counters_by_level: dict[int, int] = {}
        lines: list[str] = []

        for item in list_group.items:
            item_text = cls._normalize_block_text(item.text)
            if not item_text:
                warnings.append(
                    RenderWarning(
                        code="empty_list_item",
                        message="list item text가 비어 있어 출력하지 않습니다.",
                        node_id=(item.block_ids[0] if item.block_ids else list_group.id),
                        metadata={"list_group_id": list_group.id},
                    )
                )
                continue

            indent_level = cls._extract_indent_level(item)
            counters_by_level = {
                level: count
                for level, count in counters_by_level.items()
                if level <= indent_level
            }
            counters_by_level[indent_level] = counters_by_level.get(indent_level, 0) + 1

            marker = f"{counters_by_level[indent_level]}." if ordered else "-"
            lines.extend(
                cls._render_list_item_lines(
                    text=item_text,
                    marker=marker,
                    indent_level=indent_level,
                )
            )

        return "\n".join(lines).strip()

    @classmethod
    def _render_table_ref(
        cls,
        table_ref: TableRef,
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
    ) -> str:
        """table_ref를 markdown table / 이미지 fallback / placeholder 중 하나로 렌더링한다."""
        metadata = table_ref.metadata if isinstance(table_ref.metadata, dict) else {}
        markdown = metadata.get("markdown")
        crop_path = metadata.get("crop_path")
        caption_text = cls._lookup_caption_text_by_id(table_ref.caption_id, render_context)
        caption_block = cls._render_table_caption(caption_text)
        note_blocks = cls._render_attached_notes(
            target_id=table_ref.table_id,
            preferred_note_ids=table_ref.note_ids,
            warnings=warnings,
            render_context=render_context,
            note_style="paragraph",
        )

        if isinstance(markdown, str) and markdown.strip():
            blocks = [markdown.strip()]
            if caption_block:
                blocks.append(caption_block)
            blocks.extend(note_blocks)
            return "\n\n".join(blocks).strip()

        if isinstance(crop_path, str) and crop_path.strip():
            warnings.append(
                RenderWarning(
                    code="table_crop_fallback",
                    message="table markdown이 없어 crop_path 이미지 fallback으로 렌더링했습니다.",
                    node_id=table_ref.table_id,
                    metadata={
                        "table_id": table_ref.table_id,
                        "crop_path": crop_path,
                    },
                )
            )
            blocks = [f"![Table {table_ref.table_id}]({cls._normalize_asset_path(crop_path)})"]
            if caption_block:
                blocks.append(caption_block)
            blocks.extend(note_blocks)
            return "\n\n".join(blocks).strip()

        warnings.append(
            RenderWarning(
                code="table_placeholder",
                message="table markdown과 crop_path가 모두 없어 placeholder를 출력했습니다.",
                node_id=table_ref.table_id,
                metadata={"table_id": table_ref.table_id},
            )
        )
        blocks = [f"[TABLE PLACEHOLDER: {table_ref.table_id}]"]
        if caption_block:
            blocks.append(caption_block)
        blocks.extend(note_blocks)
        return "\n\n".join(blocks).strip()

    @classmethod
    def _render_figure_ref(
        cls,
        figure_ref: FigureRef,
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
    ) -> str:
        """figure_ref를 이미지 + caption 조합으로 렌더링한다."""
        metadata = figure_ref.metadata if isinstance(figure_ref.metadata, dict) else {}
        asset_path = figure_ref.asset_path or metadata.get("crop_path")
        caption_text = cls._lookup_caption_text_by_id(figure_ref.caption_id, render_context)
        caption_block = cls._render_figure_caption(caption_text)
        note_blocks = cls._render_attached_notes(
            target_id=figure_ref.figure_id,
            preferred_note_ids=[],
            warnings=warnings,
            render_context=render_context,
            note_style="paragraph",
        )

        if isinstance(asset_path, str) and asset_path.strip():
            blocks = [f"![Figure {figure_ref.figure_id}]({cls._normalize_asset_path(asset_path)})"]
            if caption_block:
                blocks.append(caption_block)
            blocks.extend(note_blocks)
            return "\n\n".join(blocks).strip()

        warnings.append(
            RenderWarning(
                code="figure_placeholder",
                message="figure asset_path와 crop_path가 모두 없어 placeholder를 출력했습니다.",
                node_id=figure_ref.figure_id,
                metadata={"figure_id": figure_ref.figure_id},
            )
        )
        blocks = [f"[FIGURE PLACEHOLDER: {figure_ref.figure_id}]"]
        if caption_block:
            blocks.append(caption_block)
        blocks.extend(note_blocks)
        return "\n\n".join(blocks).strip()

    @classmethod
    def _render_note_ref(
        cls,
        note_ref: NoteRef,
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
    ) -> str:
        """standalone note 또는 누락된 attached note를 보조 텍스트로 렌더링한다."""
        rendered_attached_note_ids = render_context.get("rendered_attached_note_ids", set())
        if note_ref.note_id in rendered_attached_note_ids:
            return ""

        text = cls._normalize_block_text(note_ref.text)
        if not text:
            warnings.append(
                RenderWarning(
                    code="empty_note",
                    message="note text가 비어 있어 출력하지 않습니다.",
                    node_id=note_ref.note_id,
                    metadata={"target_id": note_ref.target_id},
                )
            )
            return ""

        rendered_attached_note_ids.add(note_ref.note_id)
        if note_ref.target_id:
            return cls._render_paragraph_note(text)
        return cls._render_blockquote_note(text)

    @classmethod
    def _lookup_caption_text_by_id(
        cls,
        caption_id: str | None,
        render_context: dict[str, Any],
    ) -> str:
        """caption_id로 연결된 caption text를 ordered_elements에서 찾는다."""
        if not caption_id:
            return ""

        ordered_element_map = render_context.get("ordered_element_map", {})
        if not isinstance(ordered_element_map, dict):
            return ""

        caption_element = ordered_element_map.get(caption_id)
        if not isinstance(caption_element, AssemblyElement):
            return ""

        return cls._normalize_inline_text(caption_element.text)

    @classmethod
    def _render_table_caption(cls, caption_text: str) -> str:
        """table caption을 Markdown 보조 텍스트로 렌더링한다."""
        if not caption_text:
            return ""
        return f"*{caption_text}*"

    @classmethod
    def _render_figure_caption(cls, caption_text: str) -> str:
        """figure caption을 Markdown 보조 텍스트로 렌더링한다."""
        if not caption_text:
            return ""
        return f"*{caption_text}*"

    @classmethod
    def _render_attached_notes(
        cls,
        target_id: str,
        preferred_note_ids: list[str],
        warnings: list[RenderWarning],
        render_context: dict[str, Any],
        note_style: str,
    ) -> list[str]:
        """object에 연결된 note들을 찾아 Markdown block 목록으로 렌더링한다."""
        attached_notes = cls._resolve_attached_notes(
            target_id=target_id,
            preferred_note_ids=preferred_note_ids,
            render_context=render_context,
        )
        rendered_blocks: list[str] = []

        for note_ref in attached_notes:
            text = cls._normalize_block_text(note_ref.text)
            if not text:
                warnings.append(
                    RenderWarning(
                        code="empty_note",
                        message="attached note text가 비어 있어 출력하지 않습니다.",
                        node_id=note_ref.note_id,
                        metadata={"target_id": target_id},
                    )
                )
                continue

            rendered_attached_note_ids = render_context.get("rendered_attached_note_ids", set())
            rendered_attached_note_ids.add(note_ref.note_id)

            if note_style == "blockquote":
                rendered_blocks.append(cls._render_blockquote_note(text))
            else:
                rendered_blocks.append(cls._render_paragraph_note(text))

        return rendered_blocks

    @classmethod
    def _resolve_attached_notes(
        cls,
        target_id: str,
        preferred_note_ids: list[str],
        render_context: dict[str, Any],
    ) -> list[NoteRef]:
        """target_id 기준으로 붙은 note 목록을 중복 없이 모은다."""
        note_ref_map = render_context.get("note_ref_map", {})
        target_note_map = render_context.get("target_note_map", {})
        resolved_note_ids: list[str] = []

        for note_id in preferred_note_ids:
            if isinstance(note_id, str) and note_id not in resolved_note_ids:
                resolved_note_ids.append(note_id)

        for note_id in target_note_map.get(target_id, []):
            if note_id not in resolved_note_ids:
                resolved_note_ids.append(note_id)

        resolved_notes: list[NoteRef] = []
        for note_id in resolved_note_ids:
            note_ref = note_ref_map.get(note_id)
            if isinstance(note_ref, NoteRef):
                resolved_notes.append(note_ref)

        return resolved_notes

    @staticmethod
    def _render_paragraph_note(text: str) -> str:
        """note를 일반 보조 문단으로 렌더링한다."""
        return text.strip()

    @staticmethod
    def _render_blockquote_note(text: str) -> str:
        """note를 blockquote 형식으로 렌더링한다."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(f"> {line}" for line in lines)

    @staticmethod
    def _render_list_item_lines(
        text: str,
        marker: str,
        indent_level: int,
    ) -> list[str]:
        """list item 한 개를 Markdown line 목록으로 바꾼다."""
        indent = "  " * max(0, indent_level)
        continuation_indent = f"{indent}  "
        text_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not text_lines:
            return []

        rendered_lines = [f"{indent}{marker} {text_lines[0]}"]
        rendered_lines.extend(f"{continuation_indent}{line}" for line in text_lines[1:])
        return rendered_lines

    @staticmethod
    def _normalize_heading_level(level: Any) -> int:
        """heading level을 Markdown에서 안전한 1~4 범위로 정규화한다."""
        if not isinstance(level, int):
            return 1
        return max(1, min(level, 4))

    @staticmethod
    def _extract_indent_level(item: ListGroupItem) -> int:
        """list item metadata에서 indent level을 읽어 온다."""
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        indent_level = metadata.get("indent_level", 0)
        if not isinstance(indent_level, int):
            return 0
        return max(0, indent_level)

    @staticmethod
    def _normalize_inline_text(value: Any) -> str:
        """heading 등 한 줄 텍스트를 Markdown에 넣기 좋은 형태로 정리한다."""
        if not isinstance(value, str):
            return ""

        lines = [line.strip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        compact = " ".join(line for line in lines if line)
        return compact.strip()

    @classmethod
    def _normalize_block_text(cls, value: Any) -> str:
        """문단/리스트 텍스트의 불필요한 줄바꿈과 공백을 줄인다."""
        return cls._normalize_inline_text(value)

    @staticmethod
    def _normalize_asset_path(path: str) -> str:
        """Markdown 이미지 경로에 맞게 구분자를 정리한다."""
        return path.replace("\\", "/").strip()

    @staticmethod
    def _finalize_markdown(markdown: str) -> tuple[str, dict[str, Any]]:
        """기본 개행 규칙과 trailing whitespace를 정리하고 cleanup 통계를 만든다."""
        if not markdown:
            return "", {
                "trimmed_trailing_whitespace_lines": 0,
                "collapsed_blank_lines": 0,
                "removed_edge_blank_lines": 0,
                "input_line_count": 0,
                "output_line_count": 0,
            }

        normalized_markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = normalized_markdown.split("\n")
        trimmed_trailing_whitespace_lines = sum(
            1 for line in raw_lines if line != line.rstrip()
        )
        stripped_lines = [line.rstrip() for line in raw_lines]

        collapsed_lines: list[str] = []
        collapsed_blank_lines = 0
        previous_blank = False
        for line in stripped_lines:
            is_blank = line == ""
            if is_blank and previous_blank:
                collapsed_blank_lines += 1
                continue
            collapsed_lines.append(line)
            previous_blank = is_blank

        leading_blank_lines = 0
        while leading_blank_lines < len(collapsed_lines) and collapsed_lines[leading_blank_lines] == "":
            leading_blank_lines += 1

        trailing_blank_lines = 0
        while trailing_blank_lines < len(collapsed_lines) and collapsed_lines[-(trailing_blank_lines + 1)] == "":
            trailing_blank_lines += 1

        if trailing_blank_lines > 0:
            cleaned_lines = collapsed_lines[leading_blank_lines:-trailing_blank_lines]
        else:
            cleaned_lines = collapsed_lines[leading_blank_lines:]

        cleaned = "\n".join(cleaned_lines)
        cleanup_report = {
            "trimmed_trailing_whitespace_lines": trimmed_trailing_whitespace_lines,
            "collapsed_blank_lines": collapsed_blank_lines,
            "removed_edge_blank_lines": leading_blank_lines + trailing_blank_lines,
            "input_line_count": len(raw_lines),
            "output_line_count": len(cleaned_lines),
        }
        return cleaned, cleanup_report

    @staticmethod
    def _build_render_context(result: AssemblyResult) -> dict[str, Any]:
        """렌더링 중 반복 조회할 문맥 정보를 만든다."""
        note_ref_map = {
            note_ref.note_id: note_ref
            for note_ref in result.document.note_refs
        }
        target_note_map: dict[str, list[str]] = {}
        for note_ref in result.document.note_refs:
            if not note_ref.target_id:
                continue
            target_note_map.setdefault(note_ref.target_id, []).append(note_ref.note_id)

        return {
            "ordered_element_map": {
                element.id: element
                for element in result.ordered_elements
            },
            "note_ref_map": note_ref_map,
            "target_note_map": target_note_map,
            "rendered_attached_note_ids": set(),
        }

    @classmethod
    def _assembly_result_from_dict(cls, data: dict[str, Any]) -> AssemblyResult:
        """직렬화 dict를 AssemblyResult로 복원한다."""
        return AssemblyResult(
            ordered_elements=[
                cls._assembly_element_from_dict(item)
                for item in cls._coerce_list(data.get("ordered_elements"))
            ],
            block_relations=[
                cls._block_relation_from_dict(item)
                for item in cls._coerce_list(data.get("block_relations"))
            ],
            document=cls._document_from_dict(cls._coerce_dict(data.get("document"))),
            page_stats=[
                cls._page_stats_from_dict(item)
                for item in cls._coerce_list(data.get("page_stats"))
            ],
            warnings=[
                cls._assembly_warning_from_dict(item)
                for item in cls._coerce_list(data.get("warnings"))
            ],
            metadata=cls._meta_from_dict(cls._coerce_dict(data.get("metadata"))),
            raw=data.get("raw"),
        )

    @classmethod
    def _document_from_dict(cls, data: dict[str, Any]) -> AssembledDocument:
        """AssembledDocument dict를 복원한다."""
        return AssembledDocument(
            title_candidate=data.get("title_candidate"),
            title_source_block_ids=list(data.get("title_source_block_ids") or []),
            children=[
                cls._assembled_node_from_dict(item)
                for item in cls._coerce_list(data.get("children"))
            ],
            sections=[
                cls._section_node_from_dict(item)
                for item in cls._coerce_list(data.get("sections"))
            ],
            table_refs=[
                cls._table_ref_from_dict(item)
                for item in cls._coerce_list(data.get("table_refs"))
            ],
            figure_refs=[
                cls._figure_ref_from_dict(item)
                for item in cls._coerce_list(data.get("figure_refs"))
            ],
            note_refs=[
                cls._note_ref_from_dict(item)
                for item in cls._coerce_list(data.get("note_refs"))
            ],
            figure_assets_metadata=dict(data.get("figure_assets_metadata") or {}),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _assembled_node_from_dict(cls, data: dict[str, Any]) -> Any:
        """children 안의 조립 노드를 type 기준으로 복원한다."""
        node_type = data.get("type")

        if node_type == "section":
            return cls._section_node_from_dict(data)
        if node_type == "paragraph_group":
            return cls._paragraph_group_from_dict(data)
        if node_type == "list_group":
            return cls._list_group_from_dict(data)
        if node_type == "table_ref":
            return cls._table_ref_from_dict(data)
        if node_type == "figure_ref":
            return cls._figure_ref_from_dict(data)
        if node_type == "note_ref":
            return cls._note_ref_from_dict(data)

        raise ValueError(f"알 수 없는 assembled node type입니다: {node_type!r}")

    @classmethod
    def _section_node_from_dict(cls, data: dict[str, Any]) -> SectionNode:
        """section 노드를 재귀적으로 복원한다."""
        return SectionNode(
            type="section",
            id=str(data.get("id", "")),
            level=data.get("level"),
            title=data.get("title"),
            heading_block_id=data.get("heading_block_id"),
            source_block_ids=list(data.get("source_block_ids") or []),
            children=[
                cls._assembled_node_from_dict(item)
                for item in cls._coerce_list(data.get("children"))
            ],
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _paragraph_group_from_dict(cls, data: dict[str, Any]) -> ParagraphGroup:
        """paragraph_group 노드를 복원한다."""
        return ParagraphGroup(
            type="paragraph_group",
            id=str(data.get("id", "")),
            block_ids=list(data.get("block_ids") or []),
            text=data.get("text"),
            source_block_ids=list(data.get("source_block_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _list_group_from_dict(cls, data: dict[str, Any]) -> ListGroup:
        """list_group 노드를 복원한다."""
        return ListGroup(
            type="list_group",
            id=str(data.get("id", "")),
            ordered=data.get("ordered"),
            items=[
                cls._list_group_item_from_dict(item)
                for item in cls._coerce_list(data.get("items"))
            ],
            source_block_ids=list(data.get("source_block_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _list_group_item_from_dict(cls, data: dict[str, Any]) -> ListGroupItem:
        """list_group item을 복원한다."""
        return ListGroupItem(
            block_ids=list(data.get("block_ids") or []),
            text=data.get("text"),
            source_block_ids=list(data.get("source_block_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _table_ref_from_dict(cls, data: dict[str, Any]) -> TableRef:
        """table_ref 노드를 복원한다."""
        return TableRef(
            table_id=str(data.get("table_id", "")),
            page=int(data.get("page", 1)),
            type="table_ref",
            bbox=cls._bbox_from_value(data.get("bbox")),
            caption_id=data.get("caption_id"),
            note_ids=list(data.get("note_ids") or []),
            source_block_ids=list(data.get("source_block_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _figure_ref_from_dict(cls, data: dict[str, Any]) -> FigureRef:
        """figure_ref 노드를 복원한다."""
        return FigureRef(
            figure_id=str(data.get("figure_id", "")),
            page=int(data.get("page", 1)),
            type="figure_ref",
            bbox=cls._bbox_from_value(data.get("bbox")),
            caption_id=data.get("caption_id"),
            asset_path=data.get("asset_path"),
            source_block_ids=list(data.get("source_block_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _note_ref_from_dict(cls, data: dict[str, Any]) -> NoteRef:
        """note_ref 노드를 복원한다."""
        return NoteRef(
            note_id=str(data.get("note_id", "")),
            page=int(data.get("page", 1)),
            type="note_ref",
            bbox=cls._bbox_from_value(data.get("bbox")),
            text=data.get("text"),
            target_id=data.get("target_id"),
            source_block_ids=list(data.get("source_block_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _assembly_element_from_dict(cls, data: dict[str, Any]) -> AssemblyElement:
        """ordered element를 복원한다."""
        return AssemblyElement(
            id=str(data.get("id", "")),
            page=int(data.get("page", 1)),
            kind=data.get("kind", "text"),
            bbox=cls._bbox_from_value(data.get("bbox")),
            text=data.get("text"),
            label=data.get("label"),
            confidence=data.get("confidence"),
            column_id=data.get("column_id"),
            reading_order=data.get("reading_order"),
            parent_id=data.get("parent_id"),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _block_relation_from_dict(cls, data: dict[str, Any]) -> BlockRelation:
        """block relation을 복원한다."""
        return BlockRelation(
            type=data.get("type", "next"),
            src=str(data.get("src", "")),
            dst=str(data.get("dst", "")),
            score=data.get("score"),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _page_stats_from_dict(cls, data: dict[str, Any]) -> PageStats:
        """page 통계를 복원한다."""
        return PageStats(
            page=int(data.get("page", 1)),
            width=data.get("width"),
            height=data.get("height"),
            median_line_height=data.get("median_line_height"),
            body_font_size=data.get("body_font_size"),
            column_count=data.get("column_count"),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _assembly_warning_from_dict(cls, data: dict[str, Any]) -> AssemblyWarning:
        """assembly warning을 복원한다."""
        return AssemblyWarning(
            code=data.get("code", "missing_geometry"),
            message=data.get("message", ""),
            level=data.get("level", "warning"),
            page=data.get("page"),
            element_ids=list(data.get("element_ids") or []),
            metadata=dict(data.get("metadata") or {}),
            raw=data.get("raw"),
        )

    @classmethod
    def _meta_from_dict(cls, data: dict[str, Any]) -> AssemblyMeta:
        """assembly metadata를 복원한다."""
        return AssemblyMeta(
            stage=data.get("stage"),
            adapter=data.get("adapter"),
            source=data.get("source"),
            details=dict(data.get("details") or {}),
        )

    @staticmethod
    def _coerce_dict(value: Any) -> dict[str, Any]:
        """dict가 아니면 빈 dict로 정규화한다."""
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_list(value: Any) -> list[dict[str, Any]]:
        """list가 아니면 빈 list로 정규화한다."""
        if not isinstance(value, list):
            return []
        return [dict(item) for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _bbox_from_value(value: Any) -> tuple[float, float, float, float] | None:
        """bbox list를 tuple로 정규화한다."""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        return (
            float(value[0]),
            float(value[1]),
            float(value[2]),
            float(value[3]),
        )


__all__ = ["MarkdownRenderer"]
