from __future__ import annotations

"""Assembly Validator 단계."""

import json
from collections import defaultdict
from dataclasses import replace
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from modules.assembly._common import AssemblyCommonMixin
from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    AssemblyWarning,
    BlockRelation,
    FigureRef,
    ListGroup,
    NoteRef,
    ParagraphGroup,
    SectionNode,
    TableRef,
)
from modules.assembly.normalize_filter import NormalizeFilter
from modules.assembly.structure import StructureAssembler


class AssemblyValidator(AssemblyCommonMixin):
    """structure 결과를 최종 점검하고 warning을 남긴다."""

    LOW_CONF_THRESHOLD = NormalizeFilter.LOW_CONF_THRESHOLD
    GEOMETRY_REQUIRED_KINDS = frozenset(
        {
            "heading",
            "text",
            "list_item",
            "table",
            "figure",
            "caption",
            "note",
            "formula",
            "quote",
            "code_block",
        }
    )

    @classmethod
    def apply(cls, result: AssemblyResult) -> AssemblyResult:
        """구조 조립 결과를 검증하고 `validated` stage로 마감한다."""
        if not isinstance(result, AssemblyResult):
            return result

        if cls._should_skip_validation(result.metadata.stage):
            return result

        if result.metadata.stage != "structure_assembled":
            result = StructureAssembler.apply(result)

        added_warnings = cls._collect_validation_warnings(result)
        merged_warnings = cls._merge_warnings(result.warnings, added_warnings)
        validation_summary = cls._build_validation_summary(
            result=result,
            input_warning_count=len(result.warnings),
            added_warnings=added_warnings,
            output_warnings=merged_warnings,
        )

        document_metadata = dict(result.document.metadata)
        document_metadata["validation"] = validation_summary

        return AssemblyResult(
            ordered_elements=list(result.ordered_elements),
            block_relations=list(result.block_relations),
            document=replace(
                result.document,
                metadata=document_metadata,
            ),
            page_stats=list(result.page_stats),
            warnings=merged_warnings,
            metadata=cls._build_validated_metadata(result.metadata, validation_summary),
            raw=result.raw,
        )

    @classmethod
    def _should_skip_validation(cls, stage: Optional[str]) -> bool:
        """이미 검증이 끝난 결과라면 그대로 반환한다."""
        return stage == "validated"

    @classmethod
    def _collect_validation_warnings(cls, result: AssemblyResult) -> List[AssemblyWarning]:
        """현재 구조 IR에서 파생되는 warning을 수집한다."""
        elements_by_id = {element.id: element for element in result.ordered_elements}
        table_refs = list(result.document.table_refs)
        figure_refs = list(result.document.figure_refs)
        note_refs = list(result.document.note_refs)
        object_ids = cls._collect_object_ids(table_refs, figure_refs, note_refs)
        root_source_ids = cls._collect_source_ids_from_nodes(result.document.children)
        relations_by_type = cls._group_relations(result.block_relations)

        warnings: List[AssemblyWarning] = []
        # 추가 안정화 검증
        # reading order 결과와 next edge가 어긋나는지 본다.
        warnings.extend(
            cls._validate_next_relations(
                ordered_elements=result.ordered_elements,
                next_relations=relations_by_type["next"],
            )
        )
        # 추가 안정화 검증
        # block가 section/tree에 빠지지 않고 귀속됐는지 본다.
        warnings.extend(
            cls._validate_child_relations(
                ordered_elements=result.ordered_elements,
                child_relations=relations_by_type["child_of"],
                root_source_ids=root_source_ids,
            )
        )
        # R-ASM-17
        # orphan caption과 caption 대상 충돌을 점검한다.
        warnings.extend(
            cls._validate_caption_links(
                ordered_elements=result.ordered_elements,
                table_refs=table_refs,
                figure_refs=figure_refs,
                caption_relations=relations_by_type["caption_of"],
                object_ids=object_ids,
            )
        )
        # R-ASM-17
        # orphan note와 note 대상 충돌을 점검한다.
        warnings.extend(
            cls._validate_note_links(
                ordered_elements=result.ordered_elements,
                table_refs=table_refs,
                note_refs=note_refs,
                note_relations=relations_by_type["note_of"],
                object_ids=object_ids,
                elements_by_id=elements_by_id,
            )
        )
        # R-ASM-17
        # caption 없는 table/figure를 orphan object로 유지하면서 warning을 남긴다.
        warnings.extend(
            cls._validate_object_refs(
                table_refs=table_refs,
                figure_refs=figure_refs,
                elements_by_id=elements_by_id,
                caption_relations=relations_by_type["caption_of"],
                note_relations=relations_by_type["note_of"],
            )
        )
        # R-ASM-18
        # heading 뒤에 body가 없는 empty section을 점검한다.
        warnings.extend(cls._validate_sections(result.document.sections))
        # 추가 안정화 검증
        # 후속 단계가 쓰는 bbox 누락을 점검한다.
        warnings.extend(
            cls._validate_geometry(
                ordered_elements=result.ordered_elements,
                table_refs=table_refs,
                figure_refs=figure_refs,
                note_refs=note_refs,
            )
        )
        # R-ASM-03 연장선의 검토용 표시
        # 필터링 후에도 남은 저신뢰 block를 표시한다.
        warnings.extend(cls._validate_low_confidence_chunks(result.ordered_elements))
        return warnings

    @classmethod
    def _validate_next_relations(
        cls,
        ordered_elements: Sequence[AssemblyElement],
        next_relations: Sequence[BlockRelation],
    ) -> List[AssemblyWarning]:
        """reading order와 next relation이 같은 순서를 가리키는지 점검한다."""
        # 추가 안정화 검증
        # 위키 규칙표의 직접 항목은 아니지만 next edge 무결성을 확인한다.
        if len(ordered_elements) <= 1 and not next_relations:
            return []

        expected_pairs = [
            (current.id, following.id)
            for current, following in zip(ordered_elements, ordered_elements[1:])
        ]
        actual_pairs = [(relation.src, relation.dst) for relation in next_relations]

        missing_pairs = [pair for pair in expected_pairs if pair not in actual_pairs]
        extra_pairs = [pair for pair in actual_pairs if pair not in expected_pairs]

        pair_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        for pair in actual_pairs:
            pair_counts[pair] += 1
        duplicate_pairs = [pair for pair, count in pair_counts.items() if count > 1]

        if not missing_pairs and not extra_pairs and not duplicate_pairs:
            return []

        element_ids = cls._merge_unique_ids(
            [src for src, _ in missing_pairs],
            [dst for _, dst in missing_pairs],
            [src for src, _ in extra_pairs],
            [dst for _, dst in extra_pairs],
            [src for src, _ in duplicate_pairs],
            [dst for _, dst in duplicate_pairs],
        )
        page = cls._first_known_page(
            ordered_elements,
            [pair[0] for pair in missing_pairs + extra_pairs + duplicate_pairs],
        )

        return [
            AssemblyWarning(
                code="relation_conflict",
                message="reading order와 next 관계가 서로 일치하지 않습니다.",
                level="warning",
                page=page,
                element_ids=element_ids,
                metadata={
                    "relation_type": "next",
                    "expected_count": len(expected_pairs),
                    "actual_count": len(actual_pairs),
                    "missing_pairs": [list(pair) for pair in missing_pairs],
                    "extra_pairs": [list(pair) for pair in extra_pairs],
                    "duplicate_pairs": [list(pair) for pair in duplicate_pairs],
                },
            )
        ]

    @classmethod
    def _validate_child_relations(
        cls,
        ordered_elements: Sequence[AssemblyElement],
        child_relations: Sequence[BlockRelation],
        root_source_ids: Set[str],
    ) -> List[AssemblyWarning]:
        """본문 block가 section/tree에 일관되게 귀속되었는지 점검한다."""
        # 추가 안정화 검증
        # section 밖 orphan block와 다중 parent 충돌을 점검한다.
        warnings: List[AssemblyWarning] = []
        child_targets = cls._build_relation_targets(child_relations)

        conflicting_ids = [src for src, targets in child_targets.items() if len(targets) > 1]
        if conflicting_ids:
            warnings.append(
                AssemblyWarning(
                    code="relation_conflict",
                    message="하나의 block이 여러 section 부모를 가리키고 있습니다.",
                    level="warning",
                    page=cls._first_known_page(ordered_elements, conflicting_ids),
                    element_ids=conflicting_ids,
                    metadata={
                        "relation_type": "child_of",
                        "targets": {
                            src: sorted(targets)
                            for src, targets in child_targets.items()
                            if src in conflicting_ids
                        },
                    },
                )
            )

        orphan_by_page: DefaultDict[int, List[str]] = defaultdict(list)
        for element in ordered_elements:
            if element.kind in {"caption", "note"}:
                continue

            relation_targets = child_targets.get(element.id, set())
            if element.parent_id is not None and relation_targets and element.parent_id not in relation_targets:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="element.parent_id와 child_of 관계의 부모가 서로 다릅니다.",
                        level="warning",
                        page=element.page,
                        element_ids=[element.id],
                        metadata={
                            "relation_type": "child_of",
                            "parent_id": element.parent_id,
                            "relation_targets": sorted(relation_targets),
                        },
                    )
                )

            if element.id in root_source_ids or relation_targets:
                continue
            orphan_by_page[element.page].append(element.id)

        for page, element_ids in sorted(orphan_by_page.items()):
            warnings.append(
                AssemblyWarning(
                    code="structure_orphan_block",
                    message="구조 트리 어디에도 귀속되지 못한 block이 있습니다.",
                    level="warning",
                    page=page,
                    element_ids=element_ids,
                    metadata={"relation_type": "child_of"},
                )
            )

        return warnings

    @classmethod
    def _validate_caption_links(
        cls,
        ordered_elements: Sequence[AssemblyElement],
        table_refs: Sequence[TableRef],
        figure_refs: Sequence[FigureRef],
        caption_relations: Sequence[BlockRelation],
        object_ids: Set[str],
    ) -> List[AssemblyWarning]:
        """caption 연결과 caption_of 관계가 일관적인지 점검한다."""
        # R-ASM-17
        # 어떤 object에도 연결되지 않은 caption과 잘못 연결된 caption을 점검한다.
        warnings: List[AssemblyWarning] = []
        caption_targets = cls._build_relation_targets(caption_relations)

        for table_ref in table_refs:
            if table_ref.caption_id is not None:
                caption_targets[table_ref.caption_id].add(table_ref.table_id)
        for figure_ref in figure_refs:
            if figure_ref.caption_id is not None:
                caption_targets[figure_ref.caption_id].add(figure_ref.figure_id)

        conflicting_ids = [
            caption_id
            for caption_id, targets in caption_targets.items()
            if len(targets) > 1
        ]
        if conflicting_ids:
            warnings.append(
                AssemblyWarning(
                    code="relation_conflict",
                    message="하나의 caption이 여러 object를 동시에 가리키고 있습니다.",
                    level="warning",
                    page=cls._first_known_page(ordered_elements, conflicting_ids),
                    element_ids=conflicting_ids,
                    metadata={
                        "relation_type": "caption_of",
                        "targets": {
                            caption_id: sorted(caption_targets[caption_id])
                            for caption_id in conflicting_ids
                        },
                    },
                )
            )

        orphan_by_page: DefaultDict[int, List[str]] = defaultdict(list)
        for element in ordered_elements:
            if element.kind != "caption":
                continue

            targets = caption_targets.get(element.id, set())
            if not targets:
                orphan_by_page[element.page].append(element.id)
                continue

            if element.parent_id is not None and element.parent_id not in targets:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="caption block의 parent_id와 caption_of 대상이 다릅니다.",
                        level="warning",
                        page=element.page,
                        element_ids=[element.id],
                        metadata={
                            "relation_type": "caption_of",
                            "parent_id": element.parent_id,
                            "relation_targets": sorted(targets),
                        },
                    )
                )

            missing_targets = sorted(target for target in targets if target not in object_ids)
            if missing_targets:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="caption이 가리키는 object id가 현재 문서 ref 목록에 없습니다.",
                        level="warning",
                        page=element.page,
                        element_ids=[element.id],
                        metadata={
                            "relation_type": "caption_of",
                            "missing_targets": missing_targets,
                        },
                    )
                )

        for page, element_ids in sorted(orphan_by_page.items()):
            warnings.append(
                AssemblyWarning(
                    code="orphan_caption",
                    message="어떤 table/figure에도 연결되지 않은 caption이 있습니다.",
                    level="warning",
                    page=page,
                    element_ids=element_ids,
                )
            )

        return warnings

    @classmethod
    def _validate_note_links(
        cls,
        ordered_elements: Sequence[AssemblyElement],
        table_refs: Sequence[TableRef],
        note_refs: Sequence[NoteRef],
        note_relations: Sequence[BlockRelation],
        object_ids: Set[str],
        elements_by_id: Dict[str, AssemblyElement],
    ) -> List[AssemblyWarning]:
        """note 연결과 note_of 관계가 일관적인지 점검한다."""
        # R-ASM-17 연장선
        # orphan note와 잘못 연결된 table note를 점검한다.
        warnings: List[AssemblyWarning] = []
        note_targets = cls._build_relation_targets(note_relations)

        for table_ref in table_refs:
            for note_id in table_ref.note_ids:
                note_targets[note_id].add(table_ref.table_id)
        for note_ref in note_refs:
            if note_ref.target_id is not None:
                note_targets[note_ref.note_id].add(note_ref.target_id)

        conflicting_ids = [
            note_id
            for note_id, targets in note_targets.items()
            if len(targets) > 1
        ]
        if conflicting_ids:
            warnings.append(
                AssemblyWarning(
                    code="relation_conflict",
                    message="하나의 note가 여러 object를 동시에 가리키고 있습니다.",
                    level="warning",
                    page=cls._first_known_page(ordered_elements, conflicting_ids),
                    element_ids=conflicting_ids,
                    metadata={
                        "relation_type": "note_of",
                        "targets": {
                            note_id: sorted(note_targets[note_id])
                            for note_id in conflicting_ids
                        },
                    },
                )
            )

        orphan_by_page: DefaultDict[int, List[str]] = defaultdict(list)
        for note_ref in note_refs:
            targets = note_targets.get(note_ref.note_id, set())
            note_element = elements_by_id.get(note_ref.note_id)
            page = note_element.page if note_element is not None else note_ref.page

            if not targets:
                orphan_by_page[page].append(note_ref.note_id)
                continue

            if note_element is not None and note_element.parent_id is not None and note_element.parent_id not in targets:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="note block의 parent_id와 note_of 대상이 다릅니다.",
                        level="warning",
                        page=page,
                        element_ids=[note_ref.note_id],
                        metadata={
                            "relation_type": "note_of",
                            "parent_id": note_element.parent_id,
                            "relation_targets": sorted(targets),
                        },
                    )
                )

            missing_targets = sorted(target for target in targets if target not in object_ids)
            if missing_targets:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="note가 가리키는 object id가 현재 문서 ref 목록에 없습니다.",
                        level="warning",
                        page=page,
                        element_ids=[note_ref.note_id],
                        metadata={
                            "relation_type": "note_of",
                            "missing_targets": missing_targets,
                        },
                    )
                )

        for page, element_ids in sorted(orphan_by_page.items()):
            warnings.append(
                AssemblyWarning(
                    code="orphan_note",
                    message="어떤 object에도 연결되지 않은 note가 있습니다.",
                    level="warning",
                    page=page,
                    element_ids=element_ids,
                )
            )

        return warnings

    @classmethod
    def _validate_object_refs(
        cls,
        table_refs: Sequence[TableRef],
        figure_refs: Sequence[FigureRef],
        elements_by_id: Dict[str, AssemblyElement],
        caption_relations: Sequence[BlockRelation],
        note_relations: Sequence[BlockRelation],
    ) -> List[AssemblyWarning]:
        """table/figure ref가 필요한 attachment를 갖고 있는지 점검한다."""
        # R-ASM-17
        # caption 없는 table/figure를 orphan object로 남기고 warning을 추가한다.
        warnings: List[AssemblyWarning] = []
        caption_targets = cls._build_relation_targets(caption_relations)
        note_targets = cls._build_relation_targets(note_relations)

        for table_ref in table_refs:
            if table_ref.caption_id is None:
                warnings.append(
                    AssemblyWarning(
                        code="orphan_table",
                        message="caption이 연결되지 않은 table이 있습니다.",
                        level="warning",
                        page=table_ref.page,
                        element_ids=[table_ref.table_id],
                    )
                )
            elif table_ref.caption_id not in elements_by_id:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="table이 참조한 caption block id를 ordered_elements에서 찾지 못했습니다.",
                        level="warning",
                        page=table_ref.page,
                        element_ids=[table_ref.table_id, table_ref.caption_id],
                        metadata={"relation_type": "caption_of"},
                    )
                )
            elif table_ref.table_id not in caption_targets.get(table_ref.caption_id, set()):
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="table_ref.caption_id와 caption_of 관계가 서로 맞지 않습니다.",
                        level="warning",
                        page=table_ref.page,
                        element_ids=[table_ref.table_id, table_ref.caption_id],
                        metadata={"relation_type": "caption_of"},
                    )
                )

            for note_id in table_ref.note_ids:
                if note_id not in elements_by_id:
                    warnings.append(
                        AssemblyWarning(
                            code="relation_conflict",
                            message="table이 참조한 note block id를 ordered_elements에서 찾지 못했습니다.",
                            level="warning",
                            page=table_ref.page,
                            element_ids=[table_ref.table_id, note_id],
                            metadata={"relation_type": "note_of"},
                        )
                    )
                    continue

                if table_ref.table_id not in note_targets.get(note_id, set()):
                    warnings.append(
                        AssemblyWarning(
                            code="relation_conflict",
                            message="table_ref.note_ids와 note_of 관계가 서로 맞지 않습니다.",
                            level="warning",
                            page=table_ref.page,
                            element_ids=[table_ref.table_id, note_id],
                            metadata={"relation_type": "note_of"},
                        )
                    )

        for figure_ref in figure_refs:
            if figure_ref.caption_id is None:
                warnings.append(
                    AssemblyWarning(
                        code="orphan_figure",
                        message="caption이 연결되지 않은 figure가 있습니다.",
                        level="warning",
                        page=figure_ref.page,
                        element_ids=[figure_ref.figure_id],
                    )
                )
            elif figure_ref.caption_id not in elements_by_id:
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="figure가 참조한 caption block id를 ordered_elements에서 찾지 못했습니다.",
                        level="warning",
                        page=figure_ref.page,
                        element_ids=[figure_ref.figure_id, figure_ref.caption_id],
                        metadata={"relation_type": "caption_of"},
                    )
                )
            elif figure_ref.figure_id not in caption_targets.get(figure_ref.caption_id, set()):
                warnings.append(
                    AssemblyWarning(
                        code="relation_conflict",
                        message="figure_ref.caption_id와 caption_of 관계가 서로 맞지 않습니다.",
                        level="warning",
                        page=figure_ref.page,
                        element_ids=[figure_ref.figure_id, figure_ref.caption_id],
                        metadata={"relation_type": "caption_of"},
                    )
                )

        return warnings

    @classmethod
    def _validate_sections(cls, sections: Sequence[SectionNode]) -> List[AssemblyWarning]:
        """body 없이 heading만 남은 section을 찾는다."""
        # R-ASM-18
        # heading 뒤에 body가 없는 empty section을 찾는다.
        warnings: List[AssemblyWarning] = []
        for section in cls._iter_sections(sections):
            body_child_count = len(
                [child for child in section.children if not isinstance(child, SectionNode)]
            )
            if body_child_count > 0:
                continue

            warnings.append(
                AssemblyWarning(
                    code="empty_section",
                    message="body 없이 heading만 남은 section이 있습니다.",
                    level="warning",
                    page=cls._normalize_int(section.metadata.get("page")),
                    element_ids=cls._merge_unique_ids(section.heading_block_id, section.source_block_ids),
                    metadata={
                        "section_id": section.id,
                        "title": section.title,
                        "child_section_count": len(section.children),
                    },
                )
            )

        return warnings

    @classmethod
    def _validate_geometry(
        cls,
        ordered_elements: Sequence[AssemblyElement],
        table_refs: Sequence[TableRef],
        figure_refs: Sequence[FigureRef],
        note_refs: Sequence[NoteRef],
    ) -> List[AssemblyWarning]:
        """후속 단계가 쓰는 bbox가 비어 있는 block/ref를 모아 경고한다."""
        # 추가 안정화 검증
        # page/id는 앞 단계에서 보정되므로 여기서는 geometry 누락만 본다.
        missing_by_page: DefaultDict[int, List[str]] = defaultdict(list)

        for element in ordered_elements:
            if element.kind not in cls.GEOMETRY_REQUIRED_KINDS or element.bbox is not None:
                continue
            missing_by_page[element.page].append(element.id)

        for ref in list(table_refs) + list(figure_refs) + list(note_refs):
            if ref.bbox is not None:
                continue
            missing_by_page[ref.page].append(
                getattr(ref, "table_id", None)
                or getattr(ref, "figure_id", None)
                or getattr(ref, "note_id", None)
            )

        warnings: List[AssemblyWarning] = []
        for page, element_ids in sorted(missing_by_page.items()):
            normalized_ids = [element_id for element_id in element_ids if element_id]
            warnings.append(
                AssemblyWarning(
                    code="missing_geometry",
                    message="bbox가 비어 있는 block 또는 ref가 있습니다.",
                    level="warning",
                    page=page,
                    element_ids=normalized_ids,
                )
            )

        return warnings

    @classmethod
    def _validate_low_confidence_chunks(
        cls,
        ordered_elements: Sequence[AssemblyElement],
    ) -> List[AssemblyWarning]:
        """필터링 후에도 남은 저신뢰 block를 info 수준으로 남긴다."""
        # R-ASM-03 연장선의 검토용 표시
        # 제거 대상은 아니지만 review가 필요한 chunk를 남긴다.
        low_confidence_by_page: DefaultDict[int, List[str]] = defaultdict(list)

        for element in ordered_elements:
            if element.confidence is None or element.confidence >= cls.LOW_CONF_THRESHOLD:
                continue
            low_confidence_by_page[element.page].append(element.id)

        warnings: List[AssemblyWarning] = []
        for page, element_ids in sorted(low_confidence_by_page.items()):
            warnings.append(
                AssemblyWarning(
                    code="low_confidence_chunk",
                    message="후속 검토가 필요한 저신뢰 block이 남아 있습니다.",
                    level="info",
                    page=page,
                    element_ids=element_ids,
                    metadata={"threshold": cls.LOW_CONF_THRESHOLD},
                )
            )

        return warnings

    @classmethod
    def _group_relations(
        cls,
        relations: Sequence[BlockRelation],
    ) -> DefaultDict[str, List[BlockRelation]]:
        """관계 타입별로 relation을 묶는다."""
        grouped: DefaultDict[str, List[BlockRelation]] = defaultdict(list)
        for relation in relations:
            grouped[relation.type].append(relation)
        return grouped

    @classmethod
    def _build_relation_targets(
        cls,
        relations: Sequence[BlockRelation],
    ) -> DefaultDict[str, Set[str]]:
        """src 기준으로 도착 대상 집합을 만든다."""
        targets: DefaultDict[str, Set[str]] = defaultdict(set)
        for relation in relations:
            targets[relation.src].add(relation.dst)
        return targets

    @classmethod
    def _collect_object_ids(
        cls,
        table_refs: Sequence[TableRef],
        figure_refs: Sequence[FigureRef],
        note_refs: Sequence[NoteRef],
    ) -> Set[str]:
        """object ref id 집합을 만든다."""
        object_ids: Set[str] = set()
        object_ids.update(table_ref.table_id for table_ref in table_refs)
        object_ids.update(figure_ref.figure_id for figure_ref in figure_refs)
        object_ids.update(note_ref.note_id for note_ref in note_refs)
        return object_ids

    @classmethod
    def _collect_source_ids_from_nodes(cls, nodes: Sequence[Any]) -> Set[str]:
        """문서 구조 트리에서 소비된 source block id를 재귀적으로 모은다."""
        source_ids: Set[str] = set()

        for node in nodes:
            if isinstance(node, SectionNode):
                source_ids.update(cls._merge_unique_ids(node.heading_block_id, node.source_block_ids))
                source_ids.update(cls._collect_source_ids_from_nodes(node.children))
                continue

            if isinstance(node, ParagraphGroup):
                source_ids.update(cls._merge_unique_ids(node.block_ids, node.source_block_ids))
                continue

            if isinstance(node, ListGroup):
                source_ids.update(cls._merge_unique_ids(node.source_block_ids))
                for item in node.items:
                    source_ids.update(cls._merge_unique_ids(item.block_ids, item.source_block_ids))
                continue

            if isinstance(node, TableRef):
                source_ids.update(cls._merge_unique_ids(node.table_id, node.source_block_ids))
                continue

            if isinstance(node, FigureRef):
                source_ids.update(cls._merge_unique_ids(node.figure_id, node.source_block_ids))
                continue

            if isinstance(node, NoteRef):
                source_ids.update(cls._merge_unique_ids(node.note_id, node.source_block_ids))
                continue

        return source_ids

    @classmethod
    def _iter_sections(cls, sections: Sequence[SectionNode]) -> Iterable[SectionNode]:
        """section subtree를 평탄하게 순회한다."""
        for section in sections:
            yield section
            for child in section.children:
                if isinstance(child, SectionNode):
                    yield from cls._iter_sections([child])

    @classmethod
    def _first_known_page(
        cls,
        ordered_elements: Sequence[AssemblyElement],
        element_ids: Sequence[str],
    ) -> Optional[int]:
        """주어진 element id 목록에서 가장 먼저 찾는 page를 반환한다."""
        pages_by_id = {element.id: element.page for element in ordered_elements}
        for element_id in element_ids:
            page = pages_by_id.get(element_id)
            if page is not None:
                return page
        return None

    @classmethod
    def _merge_warnings(
        cls,
        existing_warnings: Sequence[AssemblyWarning],
        added_warnings: Sequence[AssemblyWarning],
    ) -> List[AssemblyWarning]:
        """기존 warning에 새 warning을 붙이되 중복은 한 번만 남긴다."""
        merged: List[AssemblyWarning] = []
        seen_keys: Set[Tuple[Any, ...]] = set()

        for warning in list(existing_warnings) + list(added_warnings):
            warning_key = (
                warning.level,
                warning.code,
                warning.page,
                tuple(warning.element_ids),
                warning.message,
                json.dumps(warning.metadata, ensure_ascii=False, sort_keys=True),
            )
            if warning_key in seen_keys:
                continue
            seen_keys.add(warning_key)
            merged.append(warning)

        return merged

    @classmethod
    def _build_validation_summary(
        cls,
        result: AssemblyResult,
        input_warning_count: int,
        added_warnings: Sequence[AssemblyWarning],
        output_warnings: Sequence[AssemblyWarning],
    ) -> Dict[str, Any]:
        """검증 단계 요약을 metadata에 남긴다."""
        added_counts = cls._count_warnings_by_code(added_warnings)
        total_counts = cls._count_warnings_by_code(output_warnings)
        level_counts = cls._count_warnings_by_level(output_warnings)

        return {
            "input_warning_count": input_warning_count,
            "added_warning_count": len(added_warnings),
            "output_warning_count": len(output_warnings),
            "added_warning_counts": added_counts,
            "warning_counts": total_counts,
            "warning_level_counts": level_counts,
            "element_count": len(result.ordered_elements),
            "section_count": len(list(cls._iter_sections(result.document.sections))),
            "root_child_count": len(result.document.children),
            "table_ref_count": len(result.document.table_refs),
            "figure_ref_count": len(result.document.figure_refs),
            "note_ref_count": len(result.document.note_refs),
        }

    @classmethod
    def _count_warnings_by_code(
        cls,
        warnings: Sequence[AssemblyWarning],
    ) -> Dict[str, int]:
        """warning code별 개수를 센다."""
        counts: Dict[str, int] = {}
        for warning in warnings:
            counts[warning.code] = counts.get(warning.code, 0) + 1
        return counts

    @classmethod
    def _count_warnings_by_level(
        cls,
        warnings: Sequence[AssemblyWarning],
    ) -> Dict[str, int]:
        """warning level별 개수를 센다."""
        counts: Dict[str, int] = {}
        for warning in warnings:
            counts[warning.level] = counts.get(warning.level, 0) + 1
        return counts

    @classmethod
    def _build_validated_metadata(
        cls,
        previous_metadata: AssemblyMeta,
        validation_summary: Dict[str, Any],
    ) -> AssemblyMeta:
        """이전 메타데이터를 보존하면서 validated stage를 기록한다."""
        details = dict(previous_metadata.details)
        details["upstream_stage"] = previous_metadata.stage
        details["validation"] = validation_summary

        return AssemblyMeta(
            stage="validated",
            adapter=previous_metadata.adapter,
            source=previous_metadata.source,
            details=details,
        )


__all__ = ["AssemblyValidator"]
