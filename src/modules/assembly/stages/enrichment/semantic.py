from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import replace
from typing import Any

from modules.assembly.ir import (
    AssemblyElement,
    AssemblyResult,
    AssemblyWarning,
    AssembledDocument,
    FigureRef,
    TableRef,
)
from modules.assembly.stages.enrichment.base import (
    SEMANTIC_TASK,
    URL_PATTERN,
    _BaseEnricher,
    _elapsed_seconds,
    _format_error,
    non_space_signature,
)
from modules.assembly.stages.enrichment.response_parser import (
    ALLOWED_SEMANTIC_KINDS,
    CaptionLink,
    SemanticDecision,
    parse_semantic_response,
)


SEMANTIC_NUMERIC_HEADING_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S+")
SEMANTIC_CAPTION_PATTERN = re.compile(r"^\s*(?:table|tbl\.?|figure|fig\.?|표|그림)\s*\d+", re.IGNORECASE)
SEMANTIC_NOTE_PATTERN = re.compile(r"^\s*(?:note\b|note:|source:|비고|주의|주\s*:)", re.IGNORECASE)
SEMANTIC_TERMINAL_PUNCTUATION = tuple(".!?;:。！？")
SEMANTIC_TITLE_MAX_CHARS = 90
SEMANTIC_TITLE_MAX_WORDS = 12
SEMANTIC_HEADING_HEIGHT_RATIO = 1.15


class SemanticEnricher(_BaseEnricher):
    """구조 조립 전 text block 재분류와 caption-object 연결 보정."""

    def apply(self, result: AssemblyResult) -> AssemblyResult:
        if not self.config.enables_semantic():
            return result

        client = self._client()
        warnings: list[AssemblyWarning] = []
        summary: dict[str, Any] = {
            "semantic_enabled": True,
            "model": client.model_id,
            "enrichment_mode": self.config.mode,
            "eligible_candidate_count": 0,
            "llm_candidate_count": 0,
            "skipped_candidate_count": 0,
            "candidate_reason_counts": {},
            "decision_count": 0,
            "applied_decision_count": 0,
            "caption_link_count": 0,
            "applied_caption_link_count": 0,
        }

        payload = self._build_semantic_payload(result)
        candidate_stats = payload.get("candidate_stats") or {}
        summary["eligible_candidate_count"] = candidate_stats.get("eligible_count", 0)
        summary["llm_candidate_count"] = candidate_stats.get("included_count", len(payload["candidates"]))
        summary["skipped_candidate_count"] = candidate_stats.get("skipped_count", 0)
        summary["candidate_reason_counts"] = candidate_stats.get("reason_counts", {})
        print(
            f"[LLM][Semantic] 시작: model={client.model_id}, "
            f"candidates={len(payload['candidates'])}/{summary['eligible_candidate_count']}, "
            f"skipped={summary['skipped_candidate_count']}, objects={len(payload['objects'])}, "
            f"max_new_tokens={self.config.max_new_tokens_for_task(SEMANTIC_TASK)}"
        )
        if not payload["candidates"]:
            print("[LLM][Semantic] 후보 없음: LLM 호출 건너뜀")
            return self._with_metadata_and_warnings(result, "semantic", summary, warnings)

        started_at = time.perf_counter()
        try:
            response = client.generate_json(SEMANTIC_TASK, payload)
        except Exception as error:
            print(
                f"[LLM][Semantic] 실패: error={type(error).__name__}, "
                f"message={_format_error(error)}, elapsed={_elapsed_seconds(started_at)}s"
            )
            warnings.append(self._warning("llm_semantic_failed", str(error)))
            return self._with_metadata_and_warnings(result, "semantic", summary, warnings)

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

    @staticmethod
    def _build_semantic_payload(result: AssemblyResult) -> dict[str, Any]:
        page_stats_by_page = {page_stat.page: page_stat for page_stat in result.page_stats}
        eligible_elements = [
            element
            for element in result.ordered_elements
            if element.kind in ALLOWED_SEMANTIC_KINDS and element.text
        ]
        candidates = [
            candidate
            for element in eligible_elements
            for candidate in [SemanticEnricher._build_semantic_candidate(element, page_stats_by_page.get(element.page))]
            if candidate is not None
        ]
        reason_counts = Counter(
            candidate.get("semantic_reason", "unknown")
            for candidate in candidates
        )
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
            "candidate_stats": {
                "eligible_count": len(eligible_elements),
                "included_count": len(candidates),
                "skipped_count": max(0, len(eligible_elements) - len(candidates)),
                "reason_counts": dict(reason_counts),
            },
        }

    @staticmethod
    def _build_semantic_candidate(element: AssemblyElement, page_stat: Any | None) -> dict[str, Any] | None:
        text = (element.text or "").strip()
        if not text:
            return None

        reason, hint = SemanticEnricher._semantic_candidate_hint(element, text, page_stat)
        if reason is None:
            return None

        candidate = {
            "id": element.id,
            "page": element.page,
            "kind": element.kind,
            "text": element.text,
            "bbox": element.bbox,
            "confidence": element.confidence,
            "column_id": element.column_id,
            "reading_order": element.reading_order,
            "label": element.label,
            "semantic_hint": hint,
            "semantic_reason": reason,
            "text_length": len(text),
            "non_space_length": len(non_space_signature(text)),
        }

        height_ratio = SemanticEnricher._height_to_body_ratio(element, page_stat)
        if height_ratio is not None:
            candidate["height_to_body_ratio"] = round(height_ratio, 3)
        return candidate

    @staticmethod
    def _semantic_candidate_hint(element: AssemblyElement, text: str, page_stat: Any | None) -> tuple[str | None, str | None]:
        if element.kind in {"heading", "caption", "note"}:
            return "existing_kind", element.kind

        label = (element.label or "").lower()
        if "caption" in label:
            return "label_caption", "caption"
        if "title" in label or "header" in label or "heading" in label:
            return "label_heading", "heading"

        if SEMANTIC_CAPTION_PATTERN.match(text):
            return "caption_pattern", "caption"
        if SEMANTIC_NOTE_PATTERN.match(text):
            return "note_pattern", "note"
        if SEMANTIC_NUMERIC_HEADING_PATTERN.match(text):
            return "numeric_heading_pattern", "heading"

        height_ratio = SemanticEnricher._height_to_body_ratio(element, page_stat)
        if height_ratio is not None and height_ratio >= SEMANTIC_HEADING_HEIGHT_RATIO and SemanticEnricher._is_short_title_like(text):
            return "height_title_like", "heading"

        if SemanticEnricher._is_short_title_like(text):
            return "short_title_like", "heading"

        return None, None

    @staticmethod
    def _is_short_title_like(text: str) -> bool:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return False
        if len(normalized) > SEMANTIC_TITLE_MAX_CHARS:
            return False
        if normalized.endswith(SEMANTIC_TERMINAL_PUNCTUATION):
            return False
        if len(normalized.split()) > SEMANTIC_TITLE_MAX_WORDS:
            return False
        if URL_PATTERN.search(normalized) or "`" in normalized:
            return False
        return True

    @staticmethod
    def _height_to_body_ratio(element: AssemblyElement, page_stat: Any | None) -> float | None:
        if element.bbox is None or page_stat is None:
            return None
        body_font_size = getattr(page_stat, "body_font_size", None)
        if body_font_size is None:
            return None
        try:
            baseline = float(body_font_size)
        except (TypeError, ValueError):
            return None
        if baseline <= 0:
            return None
        height = max(0.0, float(element.bbox[3]) - float(element.bbox[1]))
        return height / baseline

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
