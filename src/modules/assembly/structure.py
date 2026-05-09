from __future__ import annotations

"""Assembly StructureAssembler 단계."""

import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.assembly._common import AssemblyCommonMixin
from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
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
from modules.assembly.normalize_filter import NormalizeFilter


class StructureAssembler(AssemblyCommonMixin):
    """읽기 순서가 확정된 block을 문서 구조 IR로 조립한다."""

    # R-ASM-09, R-ASM-10
    PARA_MERGE_RATIO = 0.80
    NEW_PARA_RATIO = 1.50
    # R-ASM-07
    HEADING_FONT_RATIO = 1.20
    # R-ASM-13, R-ASM-14, R-ASM-15
    CAPTION_DIST_RATIO = 1.00
    NOTE_DIST_RATIO = 2.50

    DEFAULT_LINE_HEIGHT = 12.0
    DEFAULT_BODY_FONT_SIZE = 12.0
    MIN_INDENT_TOLERANCE = 18.0

    PARAGRAPH_LIKE_KINDS = frozenset({"text", "quote", "code_block", "formula", "caption"})
    TERMINAL_PUNCTUATION = tuple(".!?;:)]}\"'”’")

    ORDERED_LIST_PATTERN = re.compile(
        r"^\s*(?:\d+[.)]|[A-Za-z][.)]|[가-힣][.)]|\(\d+\)|\([A-Za-z]\)|\([가-힣]\))\s+"
    )
    UNORDERED_LIST_PATTERN = re.compile(r"^\s*(?:[-*•◦▪‣·])\s+")
    NUMERIC_HEADING_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+")
    PAREN_HEADING_PATTERN = re.compile(r"^\s*\((\d+|[A-Za-z]|[가-힣])\)\s+")
    KOREAN_HEADING_PATTERN = re.compile(r"^\s*([가-힣]|[A-Za-z])[.)]\s+")
    TABLE_CAPTION_PATTERN = re.compile(r"^\s*(?:table|tbl\.?|표)\s*\d*", re.IGNORECASE)
    FIGURE_CAPTION_PATTERN = re.compile(r"^\s*(?:figure|fig\.?|그림)\s*\d*", re.IGNORECASE)
    NOTE_PATTERN = re.compile(r"^\s*(?:note\b|note:|주\)|주:|※|단위:|source:)", re.IGNORECASE)

    @classmethod
    def apply(cls, result: AssemblyResult) -> AssemblyResult:
        """reading order 결과를 section/list/paragraph/object 구조로 조립한다."""
        if not isinstance(result, AssemblyResult):
            return result

        if cls._should_skip_structure(result.metadata.stage):
            return result

        if result.metadata.stage != "normalized":
            result = NormalizeFilter.apply(result)

        ordered_elements = cls._ensure_reading_order(result.ordered_elements)
        next_relations = cls._build_next_relations(ordered_elements)
        page_stats_by_page = {page_stat.page: page_stat for page_stat in result.page_stats}
        element_map = {element.id: element for element in ordered_elements}

        (
            table_refs,
            figure_refs,
            note_refs,
            caption_target_map,
            note_target_map,
            attachment_summary,
        ) = cls._resolve_object_attachments(
            ordered_elements=ordered_elements,
            element_map=element_map,
            table_refs=result.document.table_refs,
            figure_refs=result.document.figure_refs,
            note_refs=result.document.note_refs,
            page_stats_by_page=page_stats_by_page,
        )

        section_stack: List[SectionNode] = []
        root_children: List[Any] = []
        top_sections: List[SectionNode] = []
        structure_relations: List[BlockRelation] = cls._build_attachment_relations(
            table_refs=table_refs,
            figure_refs=figure_refs,
            note_target_map=note_target_map,
        )
        parent_assignments: Dict[str, str] = {}

        paragraph_index = 1
        list_index = 1
        paragraph_buffer: List[AssemblyElement] = []
        list_buffer: List[AssemblyElement] = []
        anchored_table_ids: Set[str] = set()
        anchored_figure_ids: Set[str] = set()
        anchored_note_ids: Set[str] = set()

        def flush_paragraph_buffer() -> None:
            nonlocal paragraph_index
            if not paragraph_buffer:
                return

            paragraph_node = cls._build_paragraph_group(
                block_ids=list(paragraph_buffer),
                group_index=paragraph_index,
            )
            paragraph_index += 1
            cls._append_node_to_tree(
                node=paragraph_node,
                section_stack=section_stack,
                root_children=root_children,
                relations=structure_relations,
                source_block_ids=paragraph_node.source_block_ids,
            )
            paragraph_buffer.clear()

        def flush_list_buffer() -> None:
            nonlocal list_index
            if not list_buffer:
                return

            list_node = cls._build_list_group(
                block_ids=list(list_buffer),
                group_index=list_index,
                page_stats_by_page=page_stats_by_page,
            )
            list_index += 1
            cls._append_node_to_tree(
                node=list_node,
                section_stack=section_stack,
                root_children=root_children,
                relations=structure_relations,
                source_block_ids=list_node.source_block_ids,
            )
            list_buffer.clear()

        for element in ordered_elements:
            if element.kind == "heading" and element.text:
                # R-ASM-07, R-ASM-08
                # heading block를 section 시작점으로 보고
                # 텍스트 패턴/높이 기반 level을 추정해 section tree를 만든다.
                flush_paragraph_buffer()
                flush_list_buffer()

                section_node = cls._build_section_node(
                    heading=element,
                    page_stat=page_stats_by_page.get(element.page),
                )
                while section_stack and cls._effective_section_level(section_stack[-1]) >= cls._effective_section_level(section_node):
                    section_stack.pop()

                if section_stack:
                    parent_section = section_stack[-1]
                    parent_section.children.append(section_node)
                    structure_relations.append(
                        BlockRelation(
                            type="child_of",
                            src=section_node.id,
                            dst=parent_section.id,
                            score=1.0,
                            metadata={"source": "structure_section_tree"},
                        )
                    )
                else:
                    root_children.append(section_node)
                    top_sections.append(section_node)

                section_stack.append(section_node)
                parent_assignments[element.id] = section_node.id
                structure_relations.append(
                    BlockRelation(
                        type="child_of",
                        src=element.id,
                        dst=section_node.id,
                        score=1.0,
                        metadata={"source": "structure_heading", "role": "heading"},
                    )
                )
                continue

            if element.id in caption_target_map:
                # R-ASM-13, R-ASM-14
                # caption은 독립 paragraph로 남기지 않고 object에 귀속시킨다.
                flush_paragraph_buffer()
                flush_list_buffer()
                parent_assignments[element.id] = caption_target_map[element.id]
                continue

            if element.id in note_target_map:
                # R-ASM-15
                # note도 독립 paragraph로 남기지 않고 object 부속 정보로 처리한다.
                flush_paragraph_buffer()
                flush_list_buffer()
                parent_assignments[element.id] = note_target_map[element.id]
                continue

            if element.kind == "list_item":
                # R-ASM-11
                # 연속 list_item은 list group 후보로 먼저 모은다.
                flush_paragraph_buffer()
                if not list_buffer:
                    list_buffer.append(element)
                    continue

                if cls._should_continue_list(
                    previous=list_buffer[-1],
                    current=element,
                    page_stat=page_stats_by_page.get(element.page),
                ):
                    list_buffer.append(element)
                    continue

                flush_list_buffer()
                list_buffer.append(element)
                continue

            flush_list_buffer()

            if element.kind == "table":
                flush_paragraph_buffer()
                table_node = cls._resolve_table_node(table_refs, element)
                cls._append_node_to_tree(
                    node=table_node,
                    section_stack=section_stack,
                    root_children=root_children,
                    relations=structure_relations,
                    source_block_ids=table_node.source_block_ids or [table_node.table_id],
                )
                if section_stack:
                    parent_assignments[element.id] = section_stack[-1].id
                anchored_table_ids.add(table_node.table_id)
                continue

            if element.kind == "figure":
                flush_paragraph_buffer()
                figure_node = cls._resolve_figure_node(figure_refs, element)
                cls._append_node_to_tree(
                    node=figure_node,
                    section_stack=section_stack,
                    root_children=root_children,
                    relations=structure_relations,
                    source_block_ids=figure_node.source_block_ids or [figure_node.figure_id],
                )
                if section_stack:
                    parent_assignments[element.id] = section_stack[-1].id
                anchored_figure_ids.add(figure_node.figure_id)
                continue

            if element.kind == "note":
                flush_paragraph_buffer()
                note_node = cls._resolve_note_node(note_refs, element)
                cls._append_node_to_tree(
                    node=note_node,
                    section_stack=section_stack,
                    root_children=root_children,
                    relations=structure_relations,
                    source_block_ids=note_node.source_block_ids or [note_node.note_id],
                )
                if section_stack:
                    parent_assignments[element.id] = section_stack[-1].id
                anchored_note_ids.add(note_node.note_id)
                continue

            if not paragraph_buffer:
                paragraph_buffer.append(element)
                continue

            if cls._should_merge_paragraph(
                previous=paragraph_buffer[-1],
                current=element,
                page_stat=page_stats_by_page.get(element.page),
            ):
                # R-ASM-09
                # 같은 column 안에서 간격/들여쓰기가 자연스러우면 같은 문단으로 묶는다.
                paragraph_buffer.append(element)
                continue

            # R-ASM-10
            # 간격이 벌어지거나 패턴이 달라지면 새 문단을 시작한다.
            flush_paragraph_buffer()
            paragraph_buffer.append(element)

        flush_paragraph_buffer()
        flush_list_buffer()

        for table_ref in table_refs:
            if table_ref.table_id in anchored_table_ids:
                continue
            root_children.append(table_ref)

        for figure_ref in figure_refs:
            if figure_ref.figure_id in anchored_figure_ids:
                continue
            root_children.append(figure_ref)

        for note_ref in note_refs:
            if note_ref.note_id in anchored_note_ids or note_ref.target_id is not None:
                continue
            root_children.append(note_ref)

        cls._finalize_sections(top_sections)

        updated_elements = cls._apply_parent_assignments(
            ordered_elements=ordered_elements,
            parent_assignments=parent_assignments,
            caption_target_map=caption_target_map,
            note_target_map=note_target_map,
        )
        structure_summary = cls._build_structure_summary(
            root_children=root_children,
            top_sections=top_sections,
            table_refs=table_refs,
            figure_refs=figure_refs,
            note_refs=note_refs,
            attachment_summary=attachment_summary,
        )

        document_metadata = dict(result.document.metadata)
        document_metadata["structure_assembly"] = structure_summary

        return AssemblyResult(
            ordered_elements=updated_elements,
            block_relations=cls._merge_next_relations(result.block_relations, next_relations) + structure_relations,
            document=replace(
                result.document,
                children=root_children,
                sections=top_sections,
                table_refs=table_refs,
                figure_refs=figure_refs,
                note_refs=note_refs,
                metadata=document_metadata,
            ),
            page_stats=list(result.page_stats),
            warnings=list(result.warnings),
            metadata=cls._build_structure_metadata(result.metadata, structure_summary),
            raw=result.raw,
        )

    @classmethod
    def _should_skip_structure(cls, stage: Optional[str]) -> bool:
        """이미 구조 조립 이후 단계면 그대로 반환한다."""
        return stage in {"structure_assembled", "validated"}

    @classmethod
    def _ensure_reading_order(cls, elements: List[AssemblyElement]) -> List[AssemblyElement]:
        """upstream이 정한 순서를 유지하면서 reading_order 필드를 채운다."""
        if not elements:
            return []

        if all(element.reading_order is not None for element in elements):
            ordered_elements = sorted(
                elements,
                key=lambda element: (
                    element.reading_order,
                    element.page,
                    cls._bbox_top(element),
                    cls._bbox_left(element),
                    element.id,
                ),
            )
        else:
            ordered_elements = list(elements)

        materialized_elements: List[AssemblyElement] = []
        for index, element in enumerate(ordered_elements, start=1):
            if element.reading_order is not None:
                materialized_elements.append(element)
                continue

            metadata = dict(element.metadata)
            metadata["reading_order_source"] = "upstream_sequence"
            materialized_elements.append(
                replace(
                    element,
                    reading_order=index,
                    metadata=metadata,
                )
            )

        return materialized_elements

    @classmethod
    def _build_next_relations(cls, ordered_elements: List[AssemblyElement]) -> List[BlockRelation]:
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
        merged_relations = [
            relation
            for relation in existing_relations
            if relation.type != "next"
        ]
        merged_relations.extend(next_relations)
        return merged_relations

    @classmethod
    def _resolve_object_attachments(
        cls,
        ordered_elements: List[AssemblyElement],
        element_map: Dict[str, AssemblyElement],
        table_refs: List[TableRef],
        figure_refs: List[FigureRef],
        note_refs: List[NoteRef],
        page_stats_by_page: Dict[int, PageStats],
    ) -> tuple[
        List[TableRef],
        List[FigureRef],
        List[NoteRef],
        Dict[str, str],
        Dict[str, str],
        Dict[str, Any],
    ]:
        """caption/note 연결을 기존 ref 우선으로 보정한다."""
        # R-ASM-13, R-ASM-14, R-ASM-15
        # upstream이 이미 준 연결 정보가 있으면 먼저 보존하고,
        # 비어 있을 때만 근접 규칙으로 보수적으로 보정한다.
        caption_elements = [
            element
            for element in ordered_elements
            if element.kind == "caption"
        ]
        note_elements = [
            element
            for element in ordered_elements
            if element.kind == "note"
        ]

        used_caption_ids: Set[str] = set()
        note_target_map: Dict[str, str] = {}
        caption_target_map: Dict[str, str] = {}

        updated_table_refs: List[TableRef] = []
        for table_ref in table_refs:
            page_stat = page_stats_by_page.get(table_ref.page)
            caption_id = table_ref.caption_id or cls._find_caption_candidate(
                object_ref=table_ref,
                object_kind="table",
                candidates=caption_elements,
                used_caption_ids=used_caption_ids,
                page_stat=page_stat,
                element_map=element_map,
            )
            if caption_id is not None:
                used_caption_ids.add(caption_id)
                caption_target_map[caption_id] = table_ref.table_id

            merged_note_ids = list(table_ref.note_ids)
            for note_id in merged_note_ids:
                note_target_map[note_id] = table_ref.table_id

            if not merged_note_ids:
                merged_note_ids = cls._find_note_candidates(
                    object_ref=table_ref,
                    candidates=note_elements,
                    assigned_targets=note_target_map,
                    page_stat=page_stat,
                    element_map=element_map,
                    anchor_element=element_map.get(caption_id) if caption_id else None,
                )
                for note_id in merged_note_ids:
                    note_target_map[note_id] = table_ref.table_id

            updated_table_refs.append(
                replace(
                    table_ref,
                    caption_id=caption_id,
                    note_ids=merged_note_ids,
                    metadata={
                        **dict(table_ref.metadata),
                        "structure_attachment_checked": True,
                    },
                )
            )

        updated_figure_refs: List[FigureRef] = []
        for figure_ref in figure_refs:
            page_stat = page_stats_by_page.get(figure_ref.page)
            caption_id = figure_ref.caption_id or cls._find_caption_candidate(
                object_ref=figure_ref,
                object_kind="figure",
                candidates=caption_elements,
                used_caption_ids=used_caption_ids,
                page_stat=page_stat,
                element_map=element_map,
            )
            if caption_id is not None:
                used_caption_ids.add(caption_id)
                caption_target_map[caption_id] = figure_ref.figure_id

            updated_figure_refs.append(
                replace(
                    figure_ref,
                    caption_id=caption_id,
                    metadata={
                        **dict(figure_ref.metadata),
                        "structure_attachment_checked": True,
                    },
                )
            )

        updated_note_refs: List[NoteRef] = []
        for note_ref in note_refs:
            target_id = note_target_map.get(note_ref.note_id, note_ref.target_id)
            updated_note_refs.append(
                replace(
                    note_ref,
                    target_id=target_id,
                    metadata={
                        **dict(note_ref.metadata),
                        "structure_attachment_checked": True,
                    },
                )
            )

        attachment_summary = {
            "table_count": len(updated_table_refs),
            "figure_count": len(updated_figure_refs),
            "attached_caption_count": len(caption_target_map),
            "attached_note_count": len(note_target_map),
            "caption_target_map": dict(caption_target_map),
            "note_target_map": dict(note_target_map),
        }

        return (
            updated_table_refs,
            updated_figure_refs,
            updated_note_refs,
            caption_target_map,
            note_target_map,
            attachment_summary,
        )

    @classmethod
    def _find_caption_candidate(
        cls,
        object_ref: TableRef | FigureRef,
        object_kind: str,
        candidates: List[AssemblyElement],
        used_caption_ids: Set[str],
        page_stat: Optional[PageStats],
        element_map: Dict[str, AssemblyElement],
    ) -> Optional[str]:
        """object와 가장 가까운 caption block 하나를 보수적으로 연결한다."""
        # R-ASM-13, R-ASM-14
        # object 종류별 caption keyword와 거리/가로 겹침을 함께 본다.
        threshold = cls._caption_threshold(page_stat)
        best_id: Optional[str] = None
        best_score: Optional[tuple[int, float]] = None

        for candidate in candidates:
            if candidate.id in used_caption_ids:
                continue
            if candidate.page != object_ref.page:
                continue
            if candidate.bbox is None or object_ref.bbox is None:
                continue
            if not cls._looks_like_caption_text(candidate.text, object_kind):
                continue
            if cls._horizontal_overlap_ratio(candidate.bbox, object_ref.bbox) < 0.30:
                continue

            position_rank, distance = cls._caption_distance(candidate.bbox, object_ref.bbox)
            if distance > threshold:
                continue

            score = (position_rank, distance)
            if best_score is None or score < best_score:
                best_score = score
                best_id = candidate.id

        if best_id is not None:
            return best_id

        explicit_id = getattr(object_ref, "caption_id", None)
        if explicit_id and explicit_id in element_map:
            return explicit_id
        return explicit_id

    @classmethod
    def _find_note_candidates(
        cls,
        object_ref: TableRef,
        candidates: List[AssemblyElement],
        assigned_targets: Dict[str, str],
        page_stat: Optional[PageStats],
        element_map: Dict[str, AssemblyElement],
        anchor_element: Optional[AssemblyElement],
    ) -> List[str]:
        """table 하단 note block을 짧은 거리 기준으로 연결한다."""
        # R-ASM-15
        # note는 table 또는 caption 바로 아래에 있는 경우만 연결한다.
        if object_ref.bbox is None:
            return list(object_ref.note_ids)

        threshold = cls._note_threshold(page_stat)
        anchor_bbox = anchor_element.bbox if anchor_element and anchor_element.bbox is not None else object_ref.bbox
        best_candidates: List[tuple[float, str]] = []

        for candidate in candidates:
            if candidate.id in assigned_targets:
                continue
            if candidate.page != object_ref.page:
                continue
            if candidate.bbox is None:
                continue
            if not cls._looks_like_note_text(candidate.text):
                continue
            if cls._horizontal_overlap_ratio(candidate.bbox, object_ref.bbox) < 0.30:
                continue

            distance = candidate.bbox[1] - anchor_bbox[3]
            if distance < 0 or distance > threshold:
                continue
            best_candidates.append((distance, candidate.id))

        best_candidates.sort(key=lambda item: item[0])
        return [note_id for _, note_id in best_candidates]

    @classmethod
    def _build_attachment_relations(
        cls,
        table_refs: List[TableRef],
        figure_refs: List[FigureRef],
        note_target_map: Dict[str, str],
    ) -> List[BlockRelation]:
        """caption_of / note_of 관계를 edge로 만든다."""
        # R-ASM-13, R-ASM-14, R-ASM-15
        # 연결이 확정된 caption/note는 graph view에서도 edge로 남긴다.
        relations: List[BlockRelation] = []

        for table_ref in table_refs:
            if table_ref.caption_id:
                relations.append(
                    BlockRelation(
                        type="caption_of",
                        src=table_ref.caption_id,
                        dst=table_ref.table_id,
                        score=1.0,
                        metadata={"source": "structure_attachment", "object_kind": "table"},
                    )
                )
            for note_id in table_ref.note_ids:
                relations.append(
                    BlockRelation(
                        type="note_of",
                        src=note_id,
                        dst=table_ref.table_id,
                        score=1.0,
                        metadata={"source": "structure_attachment", "object_kind": "table"},
                    )
                )

        table_target_ids = {table_ref.table_id for table_ref in table_refs}
        for figure_ref in figure_refs:
            if figure_ref.caption_id:
                relations.append(
                    BlockRelation(
                        type="caption_of",
                        src=figure_ref.caption_id,
                        dst=figure_ref.figure_id,
                        score=1.0,
                        metadata={"source": "structure_attachment", "object_kind": "figure"},
                    )
                )

        # figure note는 전용 ref 필드가 없어서 note_ref.target_id만 갱신해 두고,
        # 여기서는 table에 속하지 않는 note target만 별도 relation으로 남긴다.
        for note_id, target_id in note_target_map.items():
            if target_id in table_target_ids:
                continue
            relations.append(
                BlockRelation(
                    type="note_of",
                    src=note_id,
                    dst=target_id,
                    score=1.0,
                    metadata={"source": "structure_attachment", "object_kind": "figure"},
                )
            )

        return relations

    @classmethod
    def _resolve_table_node(cls, table_refs: List[TableRef], element: AssemblyElement) -> TableRef:
        """table element에 대응하는 ref를 우선 재사용한다."""
        for table_ref in table_refs:
            if table_ref.table_id == element.id:
                return table_ref

        return TableRef(
            table_id=element.id,
            page=element.page,
            bbox=element.bbox,
            source_block_ids=[element.id],
            metadata={"source": "structure_fallback"},
            raw=element.raw,
        )

    @classmethod
    def _resolve_figure_node(cls, figure_refs: List[FigureRef], element: AssemblyElement) -> FigureRef:
        """figure element에 대응하는 ref를 우선 재사용한다."""
        for figure_ref in figure_refs:
            if figure_ref.figure_id == element.id:
                return figure_ref

        return FigureRef(
            figure_id=element.id,
            page=element.page,
            bbox=element.bbox,
            source_block_ids=[element.id],
            metadata={"source": "structure_fallback"},
            raw=element.raw,
        )

    @classmethod
    def _resolve_note_node(cls, note_refs: List[NoteRef], element: AssemblyElement) -> NoteRef:
        """standalone note는 document.note_refs를 재사용한다."""
        for note_ref in note_refs:
            if note_ref.note_id == element.id:
                return note_ref

        return NoteRef(
            note_id=element.id,
            page=element.page,
            bbox=element.bbox,
            text=element.text,
            source_block_ids=[element.id],
            metadata={"source": "structure_fallback"},
            raw=element.raw,
        )

    @classmethod
    def _append_node_to_tree(
        cls,
        node: Any,
        section_stack: List[SectionNode],
        root_children: List[Any],
        relations: List[BlockRelation],
        source_block_ids: List[str],
    ) -> None:
        """현재 section 문맥에 맞게 node를 배치한다."""
        if section_stack:
            # R-ASM-07
            # 현재 활성 section이 있으면 하위 node를 그 section 아래에 귀속시킨다.
            current_section = section_stack[-1]
            current_section.children.append(node)
            for source_block_id in source_block_ids:
                relations.append(
                    BlockRelation(
                        type="child_of",
                        src=source_block_id,
                        dst=current_section.id,
                        score=1.0,
                        metadata={"source": "structure_child_assignment"},
                    )
                )
            return

        root_children.append(node)

    @classmethod
    def _build_paragraph_group(
        cls,
        block_ids: List[AssemblyElement],
        group_index: int,
    ) -> ParagraphGroup:
        """연속 block을 하나의 paragraph_group으로 묶는다."""
        source_block_ids = [block.id for block in block_ids]
        texts = [block.text for block in block_ids if block.text]
        joined_text = " ".join(texts) if texts else None
        kinds = list(dict.fromkeys(block.kind for block in block_ids))

        return ParagraphGroup(
            id=f"paragraph_{group_index}",
            block_ids=source_block_ids,
            text=joined_text,
            source_block_ids=source_block_ids,
            metadata={
                "kinds": kinds,
                "page_range": sorted({block.page for block in block_ids}),
                "column_ids": [block.column_id for block in block_ids],
                "line_count": len(block_ids),
            },
            raw=[block.raw for block in block_ids],
        )

    @classmethod
    def _build_list_group(
        cls,
        block_ids: List[AssemblyElement],
        group_index: int,
        page_stats_by_page: Dict[int, PageStats],
    ) -> ListGroup:
        """연속 list_item block을 하나의 list_group으로 묶는다."""
        # R-ASM-11
        # list item은 우선 group 단위로 모으고, 들여쓰기 레벨은 metadata로 남긴다.
        source_block_ids = [block.id for block in block_ids]
        ordered_flags = [cls._is_ordered_list_item(block.text) for block in block_ids if block.text]
        ordered = all(ordered_flags) if ordered_flags else None

        base_indent = min(cls._bbox_left(block) for block in block_ids)
        first_page_stat = page_stats_by_page.get(block_ids[0].page)
        indent_unit = max(
            cls.MIN_INDENT_TOLERANCE,
            cls._line_height(first_page_stat),
        )

        items: List[ListGroupItem] = []
        for block in block_ids:
            item_indent = cls._bbox_left(block)
            indent_level = max(0, round((item_indent - base_indent) / indent_unit))
            list_marker, stripped_text = cls._split_list_marker(block.text)
            items.append(
                ListGroupItem(
                    block_ids=[block.id],
                    text=stripped_text,
                    source_block_ids=[block.id],
                    metadata={
                        "indent": item_indent,
                        "indent_level": indent_level,
                        "ordered": cls._is_ordered_list_item(block.text),
                        "list_marker": list_marker,
                        "source_text": block.text,
                    },
                    raw=block.raw,
                )
            )

        return ListGroup(
            id=f"list_{group_index}",
            ordered=ordered,
            items=items,
            source_block_ids=source_block_ids,
            metadata={
                "item_count": len(items),
                "base_indent": base_indent,
            },
            raw=[block.raw for block in block_ids],
        )

    @classmethod
    def _build_section_node(
        cls,
        heading: AssemblyElement,
        page_stat: Optional[PageStats],
    ) -> SectionNode:
        """heading block 하나로 section node를 시작한다."""
        level = cls._infer_heading_level(heading, page_stat)
        return SectionNode(
            id=f"section_{heading.id}",
            level=level,
            title=heading.text,
            heading_block_id=heading.id,
            source_block_ids=[heading.id],
            metadata={
                "page": heading.page,
                "column_id": heading.column_id,
                "reading_order": heading.reading_order,
                "heading_level_source": cls._infer_heading_level_source(heading, page_stat),
            },
            raw=heading.raw,
        )

    @classmethod
    def _infer_heading_level(cls, element: AssemblyElement, page_stat: Optional[PageStats]) -> int:
        """문자 패턴과 block 높이를 같이 보고 heading level을 추정한다."""
        # R-ASM-08
        # 숫자/괄호 패턴이 있으면 우선 사용하고, 없으면 body 대비 높이로 보정한다.
        text = cls._normalize_text(element.text) or ""

        numeric_match = cls.NUMERIC_HEADING_PATTERN.match(text)
        if numeric_match:
            return max(1, numeric_match.group(1).count(".") + 1)

        if cls.PAREN_HEADING_PATTERN.match(text):
            return 2

        if cls.KOREAN_HEADING_PATTERN.match(text):
            return 2

        body_font_size = page_stat.body_font_size if page_stat else None
        heading_height = cls._bbox_height(element)
        if heading_height is not None:
            baseline = body_font_size or cls.DEFAULT_BODY_FONT_SIZE
            if heading_height >= baseline * 1.8:
                return 1
            if heading_height >= baseline * cls.HEADING_FONT_RATIO:
                return 2

        return 3

    @classmethod
    def _infer_heading_level_source(cls, element: AssemblyElement, page_stat: Optional[PageStats]) -> str:
        """level 추정 근거를 metadata에 남긴다."""
        text = cls._normalize_text(element.text) or ""
        if cls.NUMERIC_HEADING_PATTERN.match(text):
            return "numeric_pattern"
        if cls.PAREN_HEADING_PATTERN.match(text):
            return "paren_pattern"
        if cls.KOREAN_HEADING_PATTERN.match(text):
            return "korean_pattern"
        if cls._bbox_height(element) is not None and page_stat and page_stat.body_font_size is not None:
            return "height_ratio"
        return "default"

    @classmethod
    def _effective_section_level(cls, section: SectionNode) -> int:
        """level이 비어 있어도 stack 계산이 가능하게 만든다."""
        return section.level if section.level is not None else 99

    @classmethod
    def _shares_column_flow(
        cls,
        previous: AssemblyElement,
        current: AssemblyElement,
        page_stat: Optional[PageStats],
    ) -> bool:
        """column_id가 없더라도 bbox 단서로 같은 흐름인지 판단한다."""
        if previous.column_id is not None and current.column_id is not None:
            return previous.column_id == current.column_id

        if previous.bbox is None or current.bbox is None:
            return previous.column_id == current.column_id

        indent_tolerance = max(cls.MIN_INDENT_TOLERANCE, cls._line_height(page_stat))
        if abs(previous.bbox[0] - current.bbox[0]) <= indent_tolerance:
            return True

        return cls._horizontal_overlap_ratio(previous.bbox, current.bbox) >= 0.5

    @classmethod
    def _should_merge_paragraph(
        cls,
        previous: AssemblyElement,
        current: AssemblyElement,
        page_stat: Optional[PageStats],
    ) -> bool:
        """같은 문단으로 묶을 수 있는지 보수적으로 판단한다."""
        # R-ASM-09, R-ASM-10
        # page/column, 세로 간격, 들여쓰기, 문장 종료 패턴을 같이 보고
        # 같은 문단 유지 여부를 판정한다.
        if previous.kind not in cls.PARAGRAPH_LIKE_KINDS or current.kind not in cls.PARAGRAPH_LIKE_KINDS:
            return False
        if previous.page != current.page:
            return False
        if not cls._shares_column_flow(previous, current, page_stat):
            return False
        if previous.bbox is None or current.bbox is None:
            return False
        if current.kind == "caption":
            return False

        line_height = cls._line_height(page_stat)
        gap = current.bbox[1] - previous.bbox[3]
        if gap > max(line_height * cls.PARA_MERGE_RATIO, 0.0):
            return False

        indent_tolerance = max(cls.MIN_INDENT_TOLERANCE, line_height)
        if abs(previous.bbox[0] - current.bbox[0]) > indent_tolerance:
            return False

        previous_text = cls._normalize_text(previous.text) or ""
        if previous_text.endswith(cls.TERMINAL_PUNCTUATION):
            return False

        if current.text and cls._is_list_like_text(current.text):
            return False

        return True

    @classmethod
    def _should_continue_list(
        cls,
        previous: AssemblyElement,
        current: AssemblyElement,
        page_stat: Optional[PageStats],
    ) -> bool:
        """연속 list_item을 같은 list_group으로 묶을 수 있는지 본다."""
        # R-ASM-11
        # 너무 떨어지지 않고 marker 계열이 같으면 같은 list로 유지한다.
        if previous.page != current.page:
            return False
        if not cls._shares_column_flow(previous, current, page_stat):
            return False
        if previous.bbox is None or current.bbox is None:
            return False

        line_height = cls._line_height(page_stat)
        gap = current.bbox[1] - previous.bbox[3]
        if gap > max(line_height * cls.NEW_PARA_RATIO, cls.MIN_INDENT_TOLERANCE):
            return False

        previous_ordered = cls._is_ordered_list_item(previous.text)
        current_ordered = cls._is_ordered_list_item(current.text)
        if previous_ordered != current_ordered:
            return False

        return True

    @classmethod
    def _apply_parent_assignments(
        cls,
        ordered_elements: List[AssemblyElement],
        parent_assignments: Dict[str, str],
        caption_target_map: Dict[str, str],
        note_target_map: Dict[str, str],
    ) -> List[AssemblyElement]:
        """구조 조립 결과를 element.parent_id와 metadata에 반영한다."""
        updated_elements: List[AssemblyElement] = []
        for element in ordered_elements:
            metadata = dict(element.metadata)
            metadata["structure_assembled"] = True

            parent_id = parent_assignments.get(element.id)
            if element.id in caption_target_map:
                parent_id = caption_target_map[element.id]
                metadata["attached_as"] = "caption"
                metadata["target_id"] = caption_target_map[element.id]
            elif element.id in note_target_map:
                parent_id = note_target_map[element.id]
                metadata["attached_as"] = "note"
                metadata["target_id"] = note_target_map[element.id]
            elif parent_id is not None:
                metadata["section_id"] = parent_id

            updated_elements.append(
                replace(
                    element,
                    parent_id=parent_id or element.parent_id,
                    metadata=metadata,
                )
            )

        return updated_elements

    @classmethod
    def _finalize_sections(cls, sections: List[SectionNode]) -> None:
        """section subtree가 완성된 뒤 provenance를 재계산한다."""
        for section in sections:
            section.children = cls._finalize_section_children(section.children)
            aggregated_ids = [section.heading_block_id] if section.heading_block_id else []

            for child in section.children:
                child_source_ids = cls._extract_node_source_block_ids(child)
                aggregated_ids.extend(child_source_ids)

            section.source_block_ids = cls._merge_unique_ids(aggregated_ids)

    @classmethod
    def _finalize_section_children(cls, children: List[Any]) -> List[Any]:
        """중첩 section도 같은 규칙으로 마무리한다."""
        finalized_children: List[Any] = []
        for child in children:
            if isinstance(child, SectionNode):
                cls._finalize_sections([child])
            finalized_children.append(child)
        return finalized_children

    @classmethod
    def _extract_node_source_block_ids(cls, node: Any) -> List[str]:
        """조립 노드에서 provenance id를 일관되게 꺼낸다."""
        if isinstance(node, SectionNode):
            return list(node.source_block_ids)
        if hasattr(node, "source_block_ids"):
            return list(getattr(node, "source_block_ids"))
        if hasattr(node, "table_id"):
            return [getattr(node, "table_id")]
        if hasattr(node, "figure_id"):
            return [getattr(node, "figure_id")]
        if hasattr(node, "note_id"):
            return [getattr(node, "note_id")]
        return []

    @classmethod
    def _build_structure_summary(
        cls,
        root_children: List[Any],
        top_sections: List[SectionNode],
        table_refs: List[TableRef],
        figure_refs: List[FigureRef],
        note_refs: List[NoteRef],
        attachment_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """metadata에 남길 구조 조립 요약을 만든다."""
        paragraph_count = 0
        list_count = 0
        section_count = 0

        def walk(node: Any) -> None:
            nonlocal paragraph_count, list_count, section_count
            if isinstance(node, SectionNode):
                section_count += 1
                for child in node.children:
                    walk(child)
                return
            if isinstance(node, ParagraphGroup):
                paragraph_count += 1
                return
            if isinstance(node, ListGroup):
                list_count += 1

        for child in root_children:
            walk(child)

        return {
            "root_child_count": len(root_children),
            "top_section_count": len(top_sections),
            "section_count": section_count,
            "paragraph_group_count": paragraph_count,
            "list_group_count": list_count,
            "table_ref_count": len(table_refs),
            "figure_ref_count": len(figure_refs),
            "note_ref_count": len(note_refs),
            "standalone_note_count": len([note_ref for note_ref in note_refs if note_ref.target_id is None]),
            "attached_note_count": len([note_ref for note_ref in note_refs if note_ref.target_id is not None]),
            "attachment": attachment_summary,
        }

    @classmethod
    def _build_structure_metadata(
        cls,
        previous_metadata: AssemblyMeta,
        structure_summary: Dict[str, Any],
    ) -> AssemblyMeta:
        """이전 메타데이터를 보존하면서 stage만 structure_assembled로 바꾼다."""
        details = dict(previous_metadata.details)
        details["upstream_stage"] = previous_metadata.stage
        details["structure_assembly"] = structure_summary

        return AssemblyMeta(
            stage="structure_assembled",
            adapter=previous_metadata.adapter,
            source=previous_metadata.source,
            details=details,
        )

    @classmethod
    def _caption_threshold(cls, page_stat: Optional[PageStats]) -> float:
        """caption 연결에 사용할 최대 거리다."""
        return max(24.0, cls._line_height(page_stat) * cls.CAPTION_DIST_RATIO * 1.5)

    @classmethod
    def _note_threshold(cls, page_stat: Optional[PageStats]) -> float:
        """note 연결에 사용할 최대 거리다."""
        return max(32.0, cls._line_height(page_stat) * cls.NOTE_DIST_RATIO)

    @classmethod
    def _line_height(cls, page_stat: Optional[PageStats]) -> float:
        """없을 때도 안전하게 line height를 돌려준다."""
        if page_stat is None or page_stat.median_line_height is None:
            return cls.DEFAULT_LINE_HEIGHT
        return max(1.0, float(page_stat.median_line_height))

    @classmethod
    def _looks_like_caption_text(cls, text: Optional[str], object_kind: str) -> bool:
        """caption 패턴을 object 종류별로 느슨하게 확인한다."""
        normalized = cls._normalize_text(text)
        if normalized is None:
            return False

        if object_kind == "table":
            return bool(cls.TABLE_CAPTION_PATTERN.match(normalized))
        return bool(cls.FIGURE_CAPTION_PATTERN.match(normalized))

    @classmethod
    def _looks_like_note_text(cls, text: Optional[str]) -> bool:
        """note 패턴을 보수적으로 확인한다."""
        normalized = cls._normalize_text(text)
        if normalized is None:
            return False
        return bool(cls.NOTE_PATTERN.match(normalized))

    @classmethod
    def _caption_distance(
        cls,
        candidate_bbox: Tuple[float, float, float, float],
        object_bbox: Tuple[float, float, float, float],
    ) -> tuple[int, float]:
        """아래쪽 caption을 우선하고, 같으면 가까운 쪽을 선택한다."""
        if candidate_bbox[1] >= object_bbox[3]:
            return 0, candidate_bbox[1] - object_bbox[3]
        return 1, max(0.0, object_bbox[1] - candidate_bbox[3])

    @classmethod
    def _horizontal_overlap_ratio(
        cls,
        left_bbox: Tuple[float, float, float, float],
        right_bbox: Tuple[float, float, float, float],
    ) -> float:
        """두 bbox의 가로 겹침 비율을 구한다."""
        left_width = max(0.0, left_bbox[2] - left_bbox[0])
        right_width = max(0.0, right_bbox[2] - right_bbox[0])
        if left_width <= 0 or right_width <= 0:
            return 0.0
        overlap = max(0.0, min(left_bbox[2], right_bbox[2]) - max(left_bbox[0], right_bbox[0]))
        return overlap / min(left_width, right_width)

    @classmethod
    def _bbox_left(cls, element: AssemblyElement) -> float:
        """bbox가 없으면 0을 돌려준다."""
        if element.bbox is None:
            return 0.0
        return float(element.bbox[0])

    @classmethod
    def _bbox_top(cls, element: AssemblyElement) -> float:
        """bbox가 없으면 0을 돌려준다."""
        if element.bbox is None:
            return 0.0
        return float(element.bbox[1])

    @classmethod
    def _bbox_height(cls, element: AssemblyElement) -> Optional[float]:
        """bbox 높이를 반환한다."""
        if element.bbox is None:
            return None
        return max(0.0, float(element.bbox[3] - element.bbox[1]))

    @classmethod
    def _is_ordered_list_item(cls, text: Optional[str]) -> bool:
        """ordered list marker를 간단히 판별한다."""
        normalized = cls._normalize_text(text)
        if normalized is None:
            return False
        return bool(cls.ORDERED_LIST_PATTERN.match(normalized))

    @classmethod
    def _split_list_marker(cls, text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """list marker와 실제 item 본문을 분리한다."""
        normalized = cls._normalize_text(text)
        if normalized is None:
            return None, None

        ordered_match = cls.ORDERED_LIST_PATTERN.match(normalized)
        if ordered_match:
            marker = ordered_match.group(0).strip()
            stripped_text = normalized[ordered_match.end():].strip()
            return marker, stripped_text or normalized

        unordered_match = cls.UNORDERED_LIST_PATTERN.match(normalized)
        if unordered_match:
            marker = unordered_match.group(0).strip()
            stripped_text = normalized[unordered_match.end():].strip()
            return marker, stripped_text or normalized

        return None, normalized

    @classmethod
    def _is_list_like_text(cls, text: str) -> bool:
        """문단 병합 중 list 시작 후보를 잘못 합치지 않게 막는다."""
        return bool(cls.ORDERED_LIST_PATTERN.match(text) or cls.UNORDERED_LIST_PATTERN.match(text))


__all__ = ["StructureAssembler"]
