from __future__ import annotations

"""선택형 LLM 기반 Assembly IR 보강.

보수적 변경 원칙 유지.
LLM 출력은 block role 승격 또는 공백 보정에만 사용.
모든 변경에는 provenance metadata 기록, 실패 시 원본 IR 유지.
"""

import re
import time
from dataclasses import dataclass, replace
from typing import Any

from modules.assembly.ir import (
    AssemblyElement,
    AssemblyResult,
    AssemblyWarning,
    AssembledDocument,
    FigureRef,
    ListGroup,
    ListGroupItem,
    ParagraphGroup,
    SectionNode,
    TableRef,
)
from modules.llm_core import LLMClient, LLMConfig, LocalTransformersLLMClient
from modules.llm_response_parser import (
    ALLOWED_SEMANTIC_KINDS,
    CaptionLink,
    ContentRepair,
    SemanticDecision,
    parse_content_repairs,
    parse_semantic_response,
)


SEMANTIC_TASK = "semantic_enrichment"
CONTENT_TASK = "content_repair"
URL_PATTERN = re.compile(r"(?:https?://|www\.|[\w.-]+@[\w.-]+)", re.IGNORECASE)
MARKDOWN_TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


@dataclass(frozen=True)
class _ContentRepairCandidate:
    node_id: str
    text: str
    role: str
    language: str


class SemanticEnricher:
    """구조 조립 전 text block 재분류와 caption-object 연결 보정."""

    def __init__(self, config: LLMConfig | None = None, client: LLMClient | None = None):
        self.config = config or LLMConfig.from_env()
        self.client = client

    def apply(self, result: AssemblyResult) -> AssemblyResult:
        if not self.config.runs_semantic():
            return result

        client = self._client()
        warnings: list[AssemblyWarning] = []
        summary: dict[str, Any] = {
            "semantic_enabled": True,
            "model": client.model_id,
            "enrichment_mode": self.config.mode,
            "decision_count": 0,
            "applied_decision_count": 0,
            "caption_link_count": 0,
            "applied_caption_link_count": 0,
        }

        payload = self._build_semantic_payload(result)
        print(
            f"[LLM][Semantic] 시작: model={client.model_id}, "
            f"candidates={len(payload['candidates'])}, objects={len(payload['objects'])}, "
            f"max_new_tokens={self.config.max_new_tokens}"
        )
        started_at = time.perf_counter()
        try:
            response = client.generate_json(SEMANTIC_TASK, payload)
        except Exception as error:
            print(
                f"[LLM][Semantic] 실패: error={type(error).__name__}, "
                f"message={_format_error(error)}, elapsed={_elapsed_seconds(started_at)}s"
            )
            warnings.append(self._warning("llm_semantic_failed", str(error)))
            return self._with_metadata_and_warnings(result, summary, warnings)

        parsed_response = parse_semantic_response(response)
        summary["decision_count"] = len(parsed_response.decisions)
        summary["caption_link_count"] = len(parsed_response.caption_links)
        print(
            f"[LLM][Semantic] 응답 파싱 완료: decisions={summary['decision_count']}, "
            f"caption_links={summary['caption_link_count']}, elapsed={_elapsed_seconds(started_at)}s"
        )

        updated_elements, applied_decisions = self._apply_decisions(
            result.ordered_elements,
            parsed_response.decisions,
            warnings,
        )
        updated_document, applied_links = self._apply_caption_links(
            result.document,
            updated_elements,
            parsed_response.caption_links,
            warnings,
        )
        summary["applied_decision_count"] = applied_decisions
        summary["applied_caption_link_count"] = applied_links
        print(
            f"[LLM][Semantic] 적용 완료: applied_decisions={applied_decisions}, "
            f"applied_caption_links={applied_links}, warnings={len(warnings)}"
        )

        return AssemblyResult(
            ordered_elements=updated_elements,
            block_relations=list(result.block_relations),
            document=replace(
                updated_document,
                metadata=self._merge_summary(updated_document.metadata, "semantic", summary),
            ),
            page_stats=list(result.page_stats),
            warnings=list(result.warnings) + warnings,
            metadata=result.metadata,
            raw=result.raw,
        )

    def _client(self) -> LLMClient:
        if self.client is None:
            self.client = LocalTransformersLLMClient(self.config)
        return self.client

    @staticmethod
    def _build_semantic_payload(result: AssemblyResult) -> dict[str, Any]:
        candidates = [
            {
                "id": element.id,
                "page": element.page,
                "kind": element.kind,
                "text": element.text,
                "bbox": element.bbox,
                "confidence": element.confidence,
                "column_id": element.column_id,
                "reading_order": element.reading_order,
                "label": element.label,
            }
            for element in result.ordered_elements
            if element.kind in ALLOWED_SEMANTIC_KINDS and element.text
        ]
        objects = [
            {
                "target_id": table_ref.table_id,
                "object_kind": "table",
                "page": table_ref.page,
                "bbox": table_ref.bbox,
                "caption_id": table_ref.caption_id,
            }
            for table_ref in result.document.table_refs
        ] + [
            {
                "target_id": figure_ref.figure_id,
                "object_kind": "figure",
                "page": figure_ref.page,
                "bbox": figure_ref.bbox,
                "caption_id": figure_ref.caption_id,
            }
            for figure_ref in result.document.figure_refs
        ]
        return {
            "schema": {
                "semantic_decisions": [{"id": "string", "kind": "text|heading|caption|note", "heading_level": "int|null", "confidence": "float"}],
                "caption_links": [{"caption_id": "string", "target_id": "string", "confidence": "float"}],
            },
            "candidates": candidates,
            "objects": objects,
            "page_stats": [page_stat.to_dict() for page_stat in result.page_stats],
        }

    def _apply_decisions(
        self,
        elements: list[AssemblyElement],
        decisions: list[SemanticDecision],
        warnings: list[AssemblyWarning],
    ) -> tuple[list[AssemblyElement], int]:
        decisions_by_id = {decision.id: decision for decision in decisions}

        applied = 0
        updated_elements: list[AssemblyElement] = []
        for element in elements:
            decision = decisions_by_id.get(element.id)
            if decision is None:
                updated_elements.append(element)
                continue

            metadata = {
                **dict(element.metadata),
                **self._llm_metadata(SEMANTIC_TASK, decision.confidence),
                "llm_original_kind": element.kind,
            }
            if decision.heading_level is not None:
                metadata["llm_heading_level"] = max(1, min(6, decision.heading_level))

            if element.kind != decision.kind or decision.heading_level is not None:
                applied += 1

            updated_elements.append(replace(element, kind=decision.kind, metadata=metadata))

        unknown_ids = sorted(set(decisions_by_id) - {element.id for element in elements})
        if unknown_ids:
            warnings.append(
                self._warning(
                    "llm_semantic_unknown_block",
                    "LLM이 알 수 없는 block id에 대한 semantic decision 반환.",
                    element_ids=unknown_ids,
                    metadata={"unknown_ids": unknown_ids},
                )
            )
        return updated_elements, applied

    def _apply_caption_links(
        self,
        document: AssembledDocument,
        elements: list[AssemblyElement],
        links: list[CaptionLink],
        warnings: list[AssemblyWarning],
    ) -> tuple[AssembledDocument, int]:
        element_ids = {element.id for element in elements}
        applied = 0
        link_by_target: dict[str, CaptionLink] = {}

        for link in links:
            if link.caption_id not in element_ids:
                warnings.append(
                    self._warning(
                        "llm_caption_link_unknown_caption",
                        "LLM이 알 수 없는 caption block에 대한 caption link 반환.",
                        element_ids=[link.caption_id],
                        metadata={"target_id": link.target_id},
                    )
                )
                continue
            link_by_target[link.target_id] = link

        table_refs = [
            self._apply_link_to_table_ref(table_ref, link_by_target.get(table_ref.table_id))
            for table_ref in document.table_refs
        ]
        figure_refs = [
            self._apply_link_to_figure_ref(figure_ref, link_by_target.get(figure_ref.figure_id))
            for figure_ref in document.figure_refs
        ]

        applied = sum(1 for before, after in zip(document.table_refs, table_refs) if before.caption_id != after.caption_id)
        applied += sum(1 for before, after in zip(document.figure_refs, figure_refs) if before.caption_id != after.caption_id)
        return replace(document, table_refs=table_refs, figure_refs=figure_refs), applied

    def _apply_link_to_table_ref(self, table_ref: TableRef, link: CaptionLink | None) -> TableRef:
        if link is None:
            return table_ref
        return replace(
            table_ref,
            caption_id=link.caption_id,
            metadata={**dict(table_ref.metadata), **self._llm_metadata("caption_candidate_repair", link.confidence)},
        )

    def _apply_link_to_figure_ref(self, figure_ref: FigureRef, link: CaptionLink | None) -> FigureRef:
        if link is None:
            return figure_ref
        return replace(
            figure_ref,
            caption_id=link.caption_id,
            metadata={**dict(figure_ref.metadata), **self._llm_metadata("caption_candidate_repair", link.confidence)},
        )

    def _llm_metadata(self, task: str, confidence: float | None) -> dict[str, Any]:
        return {
            "llm_enriched": True,
            "llm_model": self.config.model_id,
            "llm_task": task,
            "llm_confidence": confidence,
            "llm_enrichment_mode": self.config.mode,
        }

    @staticmethod
    def _merge_summary(metadata: dict[str, Any], key: str, summary: dict[str, Any]) -> dict[str, Any]:
        existing = dict(metadata.get("llm_enrichment") or {})
        existing[key] = summary
        return {**dict(metadata), "llm_enrichment": existing}

    @staticmethod
    def _with_metadata_and_warnings(
        result: AssemblyResult,
        summary: dict[str, Any],
        warnings: list[AssemblyWarning],
    ) -> AssemblyResult:
        document_metadata = SemanticEnricher._merge_summary(result.document.metadata, "semantic", summary)
        return replace(
            result,
            document=replace(result.document, metadata=document_metadata),
            warnings=list(result.warnings) + warnings,
        )

    @staticmethod
    def _warning(
        code: str,
        message: str,
        *,
        element_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssemblyWarning:
        return AssemblyWarning(
            code=code,
            message=message,
            level="warning",
            element_ids=element_ids or [],
            metadata=metadata or {},
        )


class ContentEnricher:
    """구조 조립 후 paragraph/list/heading text 보정."""

    def __init__(self, config: LLMConfig | None = None, client: LLMClient | None = None):
        self.config = config or LLMConfig.from_env()
        self.client = client

    def apply(self, result: AssemblyResult) -> AssemblyResult:
        if not self.config.runs_content():
            return result

        client = self._client()
        warnings: list[AssemblyWarning] = []
        summary = {
            "content_enabled": True,
            "model": client.model_id,
            "enrichment_mode": self.config.mode,
            "content_batch_size": self.config.content_batch_size,
            "llm_candidate_count": 0,
            "batch_count": 0,
            "attempt_count": 0,
            "applied_count": 0,
            "discarded_count": 0,
            "rule_repair_count": 0,
        }

        print(
            f"[LLM][Content] 시작: model={client.model_id}, "
            f"children={len(result.document.children)}, sections={len(result.document.sections)}, "
            f"max_new_tokens={self.config.max_new_tokens}, "
            f"batch_size={self.config.content_batch_size}"
        )
        started_at = time.perf_counter()
        use_section_fallback = not any(isinstance(node, SectionNode) for node in result.document.children) and bool(result.document.sections)
        candidates = self._collect_repair_candidates(result.document.children)
        if use_section_fallback:
            candidates.extend(self._collect_repair_candidates(result.document.sections))
        summary["llm_candidate_count"] = len(candidates)
        print(
            f"[LLM][Content] 후보 수집 완료: llm_candidates={len(candidates)}, "
            f"batch_size={self.config.content_batch_size}"
        )

        repairs_by_node = self._generate_content_repairs(candidates, warnings, summary)
        candidate_ids = {candidate.node_id for candidate in candidates}

        children = [
            self._repair_node(node, warnings, summary, repairs_by_node, candidate_ids)
            for node in result.document.children
        ]
        sections = [node for node in children if isinstance(node, SectionNode)]
        if not sections and result.document.sections:
            sections = [
                self._repair_node(section, warnings, summary, repairs_by_node, candidate_ids)
                for section in result.document.sections
            ]
        document_metadata = SemanticEnricher._merge_summary(result.document.metadata, "content", summary)
        print(
            f"[LLM][Content] 완료: candidates={summary['llm_candidate_count']}, "
            f"batches={summary['batch_count']}, attempts={summary['attempt_count']}, "
            f"applied={summary['applied_count']}, discarded={summary['discarded_count']}, "
            f"rule_repairs={summary['rule_repair_count']}, warnings={len(warnings)}, "
            f"elapsed={_elapsed_seconds(started_at)}s"
        )

        return replace(
            result,
            document=replace(result.document, children=children, sections=sections, metadata=document_metadata),
            warnings=list(result.warnings) + warnings,
        )

    def _client(self) -> LLMClient:
        if self.client is None:
            self.client = LocalTransformersLLMClient(self.config)
        return self.client

    def _repair_node(
        self,
        node: Any,
        warnings: list[AssemblyWarning],
        summary: dict[str, Any],
        repairs_by_node: dict[str, ContentRepair],
        candidate_ids: set[str],
    ) -> Any:
        if isinstance(node, SectionNode):
            title, metadata = self._repair_text(
                node_id=node.id,
                text=node.title,
                metadata=node.metadata,
                warnings=warnings,
                summary=summary,
                role="heading",
                repairs_by_node=repairs_by_node,
                candidate_ids=candidate_ids,
            )
            children = [
                self._repair_node(child, warnings, summary, repairs_by_node, candidate_ids)
                for child in node.children
            ]
            return replace(node, title=title, children=children, metadata=metadata)

        if isinstance(node, ParagraphGroup):
            text, metadata = self._repair_text(
                node_id=node.id,
                text=node.text,
                metadata=node.metadata,
                warnings=warnings,
                summary=summary,
                role="paragraph",
                repairs_by_node=repairs_by_node,
                candidate_ids=candidate_ids,
            )
            return replace(node, text=text, metadata=metadata)

        if isinstance(node, ListGroup):
            items = [
                self._repair_list_item(item, warnings, summary, repairs_by_node, candidate_ids)
                for item in node.items
            ]
            return replace(node, items=items)

        return node

    def _repair_list_item(
        self,
        item: ListGroupItem,
        warnings: list[AssemblyWarning],
        summary: dict[str, Any],
        repairs_by_node: dict[str, ContentRepair],
        candidate_ids: set[str],
    ) -> ListGroupItem:
        node_id = item.block_ids[0] if item.block_ids else "list_item"
        text, metadata = self._repair_text(
            node_id=node_id,
            text=item.text,
            metadata=item.metadata,
            warnings=warnings,
            summary=summary,
            role="list_item",
            repairs_by_node=repairs_by_node,
            candidate_ids=candidate_ids,
        )
        return replace(item, text=text, metadata=metadata)

    def _repair_text(
        self,
        *,
        node_id: str,
        text: Any,
        metadata: dict[str, Any],
        warnings: list[AssemblyWarning],
        summary: dict[str, Any],
        role: str,
        repairs_by_node: dict[str, ContentRepair],
        candidate_ids: set[str],
    ) -> tuple[Any, dict[str, Any]]:
        if not isinstance(text, str) or not text.strip() or self._should_skip_text(text, metadata):
            return text, dict(metadata)

        language = self._detect_language(text)
        if language == "english":
            repaired = self._repair_english_hyphenation(text)
            if repaired != text:
                summary["rule_repair_count"] += 1
                summary["applied_count"] += 1
                return repaired, {
                    **dict(metadata),
                    **self._llm_metadata("english_rule_repair", 1.0),
                    "llm_language": language,
                    "llm_repair_source": "rule",
                }
            return text, dict(metadata)

        if node_id not in candidate_ids:
            return text, dict(metadata)

        repair = repairs_by_node.get(node_id)
        if repair is None:
            return text, dict(metadata)

        if self._non_space_signature(text) != self._non_space_signature(repair.text):
            summary["discarded_count"] += 1
            warnings.append(
                SemanticEnricher._warning(
                    "llm_content_preservation_failed",
                    "LLM content repair가 비공백 문자를 변경하여 결과 폐기.",
                    element_ids=[node_id],
                    metadata={"role": role, "language": language},
                )
            )
            return text, dict(metadata)

        if repair.text != text:
            summary["applied_count"] += 1
            return repair.text, {
                **dict(metadata),
                **self._llm_metadata(CONTENT_TASK, repair.confidence),
                "llm_language": language,
            }

        return text, dict(metadata)

    def _collect_repair_candidates(self, nodes: list[Any]) -> list[_ContentRepairCandidate]:
        candidates: list[_ContentRepairCandidate] = []
        for node in nodes:
            if isinstance(node, SectionNode):
                candidate = self._build_repair_candidate(node.id, node.title, node.metadata, "heading")
                if candidate is not None:
                    candidates.append(candidate)
                candidates.extend(self._collect_repair_candidates(node.children))
            elif isinstance(node, ParagraphGroup):
                candidate = self._build_repair_candidate(node.id, node.text, node.metadata, "paragraph")
                if candidate is not None:
                    candidates.append(candidate)
            elif isinstance(node, ListGroup):
                for item in node.items:
                    node_id = item.block_ids[0] if item.block_ids else "list_item"
                    candidate = self._build_repair_candidate(node_id, item.text, item.metadata, "list_item")
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _build_repair_candidate(
        self,
        node_id: str,
        text: Any,
        metadata: dict[str, Any],
        role: str,
    ) -> _ContentRepairCandidate | None:
        if not isinstance(text, str) or not text.strip() or self._should_skip_text(text, metadata):
            return None

        language = self._detect_language(text)
        if language == "english":
            return None
        return _ContentRepairCandidate(node_id=node_id, text=text, role=role, language=language)

    def _generate_content_repairs(
        self,
        candidates: list[_ContentRepairCandidate],
        warnings: list[AssemblyWarning],
        summary: dict[str, Any],
    ) -> dict[str, ContentRepair]:
        if not candidates:
            return {}

        batches = list(self._chunk_candidates(candidates, self.config.content_batch_size))
        summary["attempt_count"] += len(candidates)
        summary["batch_count"] = len(batches)

        repairs_by_node: dict[str, ContentRepair] = {}
        for batch_index, batch in enumerate(batches, start=1):
            batch_ids = [candidate.node_id for candidate in batch]
            started_at = time.perf_counter()
            print(
                f"[LLM][Content] batch {batch_index}/{len(batches)} 시작: "
                f"items={len(batch)}, node_ids={_format_node_ids(batch_ids)}"
            )
            try:
                response = self._client().generate_json(CONTENT_TASK, self._build_content_payload(batch))
            except Exception as error:
                print(
                    f"[LLM][Content] batch {batch_index}/{len(batches)} 실패: "
                    f"items={len(batch)}, error={type(error).__name__}, "
                    f"message={_format_error(error)}, elapsed={_elapsed_seconds(started_at)}s"
                )
                warnings.append(
                    SemanticEnricher._warning(
                        "llm_content_failed",
                        str(error),
                        element_ids=batch_ids,
                        metadata={"batch_index": batch_index, "batch_size": len(batch)},
                    )
                )
                continue

            repairs = parse_content_repairs(response)
            matched_count = self._store_matched_repairs(batch, repairs, repairs_by_node)
            print(
                f"[LLM][Content] batch {batch_index}/{len(batches)} 완료: "
                f"parsed={len(repairs)}, matched={matched_count}, "
                f"elapsed={_elapsed_seconds(started_at)}s"
            )
        return repairs_by_node

    @staticmethod
    def _build_content_payload(batch: list[_ContentRepairCandidate]) -> dict[str, Any]:
        return {
            "schema": {"repairs": [{"node_id": "string", "text": "string", "confidence": "float"}]},
            "items": [
                {
                    "node_id": candidate.node_id,
                    "text": candidate.text,
                    "language": candidate.language,
                    "role": candidate.role,
                }
                for candidate in batch
            ],
            "constraint": "공백만 변경. 비공백 문자 시퀀스 완전 동일 유지.",
        }

    @staticmethod
    def _chunk_candidates(
        candidates: list[_ContentRepairCandidate],
        batch_size: int,
    ) -> list[list[_ContentRepairCandidate]]:
        return [
            candidates[index:index + batch_size]
            for index in range(0, len(candidates), batch_size)
        ]

    @staticmethod
    def _store_matched_repairs(
        batch: list[_ContentRepairCandidate],
        repairs: list[ContentRepair],
        repairs_by_node: dict[str, ContentRepair],
    ) -> int:
        repair_by_id = {repair.node_id: repair for repair in repairs}
        matched_count = 0
        for candidate in batch:
            repair = repair_by_id.get(candidate.node_id)
            if repair is None and len(batch) == 1 and len(repairs) == 1:
                repair = repairs[0]
            if repair is None:
                continue
            repairs_by_node[candidate.node_id] = repair
            matched_count += 1
        return matched_count

    @staticmethod
    def _should_skip_text(text: str, metadata: dict[str, Any]) -> bool:
        if URL_PATTERN.search(text):
            return True
        if "`" in text:
            return True
        if any(MARKDOWN_TABLE_LINE_PATTERN.match(line) for line in text.splitlines()):
            return True
        kinds = metadata.get("kinds")
        if isinstance(kinds, list) and any(kind in {"code_block", "formula"} for kind in kinds):
            return True
        return False

    @staticmethod
    def _detect_language(text: str) -> str:
        compact = re.sub(r"\s+", "", text)
        if not compact:
            return "unknown"
        hangul = len(re.findall(r"[가-힣]", compact))
        latin = len(re.findall(r"[A-Za-z]", compact))
        total = len(compact)
        if hangul / total >= 0.25:
            return "korean"
        if latin / total >= 0.60:
            return "english"
        return "mixed"

    @staticmethod
    def _repair_english_hyphenation(text: str) -> str:
        return re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)

    @staticmethod
    def _non_space_signature(text: str) -> str:
        return re.sub(r"\s+", "", text)

    def _llm_metadata(self, task: str, confidence: float | None) -> dict[str, Any]:
        return {
            "llm_enriched": True,
            "llm_model": self.config.model_id,
            "llm_task": task,
            "llm_confidence": confidence,
            "llm_enrichment_mode": self.config.mode,
        }


def _elapsed_seconds(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.2f}"


def _format_error(error: BaseException) -> str:
    message = str(error).strip().splitlines()
    if message:
        return message[0]
    return repr(error)


def _format_node_ids(node_ids: list[str], *, limit: int = 3) -> str:
    if len(node_ids) <= limit:
        return ", ".join(node_ids)
    return f"{', '.join(node_ids[:limit])}, ..."


__all__ = ["ContentEnricher", "SemanticEnricher"]
