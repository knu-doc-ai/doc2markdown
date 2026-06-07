from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any

from modules.assembly.ir import (
    AssemblyResult,
    AssemblyWarning,
    ListGroup,
    ListGroupItem,
    ParagraphGroup,
    SectionNode,
)
from modules.assembly.stages.enrichment.base import (
    CONTENT_TASK,
    MARKDOWN_TABLE_LINE_PATTERN,
    URL_PATTERN,
    _BaseEnricher,
    _elapsed_seconds,
    _format_error,
    _format_node_ids,
    non_space_signature,
)
from modules.assembly.stages.enrichment.json_parser import LLMGenerationError
from modules.assembly.stages.enrichment.response_parser import ContentRepair, parse_content_repairs


CONTENT_SUMMARY_SAMPLE_LIMIT = 8


@dataclass(frozen=True)
class _ContentRepairCandidate:
    node_id: str
    text: str
    role: str
    language: str
    spacing_suspect: bool


class ContentEnricher(_BaseEnricher):
    """구조 조립 후 paragraph/list/heading text 보정."""

    def apply(self, result: AssemblyResult) -> AssemblyResult:
        if not self.config.enables_content():
            return result

        client = self._client()
        model_id = self.config.model_id_for_task(CONTENT_TASK)
        warnings: list[AssemblyWarning] = []
        summary = {
            "content_enabled": True,
            "model": model_id,
            "enrichment_mode": self.config.mode,
            "content_batch_size": self.config.content_batch_size,
            "content_min_chars": self.config.content_min_chars,
            "content_max_new_tokens": self.config.max_new_tokens_for_task(CONTENT_TASK),
            "llm_candidate_count": 0,
            "batch_count": 0,
            "attempt_count": 0,
            "parsed_count": 0,
            "matched_count": 0,
            "unchanged_count": 0,
            "missing_repair_count": 0,
            "applied_count": 0,
            "discarded_count": 0,
            "rule_repair_count": 0,
            "failed_batch_count": 0,
            "json_parse_failure_count": 0,
            "applied_repair_ids": [],
            "applied_repair_samples": [],
            "missing_repair_ids": [],
            "unchanged_sample_ids": [],
            "discarded_repair_ids": [],
            "discarded_repair_samples": [],
        }

        print(
            f"[LLM][Content] 시작: model={model_id}, "
            f"children={len(result.document.children)}, sections={len(result.document.sections)}, "
            f"max_new_tokens={self.config.max_new_tokens_for_task(CONTENT_TASK)}, "
            f"batch_size={self.config.content_batch_size}, min_chars={self.config.content_min_chars}"
        )
        started_at = time.perf_counter()
        use_section_fallback = not any(isinstance(node, SectionNode) for node in result.document.children) and bool(result.document.sections)
        candidates = self._collect_repair_candidates(result.document.children)
        if use_section_fallback:
            candidates.extend(self._collect_repair_candidates(result.document.sections))
        summary["llm_candidate_count"] = len(candidates)
        print(
            f"[LLM][Content] 후보 수집 완료: llm_candidates={len(candidates)}, "
            f"batch_size={self.config.content_batch_size}, min_chars={self.config.content_min_chars}"
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
        document_metadata = self._merge_summary(result.document.metadata, "content", summary)
        print(
            f"[LLM][Content] 완료: candidates={summary['llm_candidate_count']}, "
            f"batches={summary['batch_count']}, attempts={summary['attempt_count']}, "
            f"parsed={summary['parsed_count']}, matched={summary['matched_count']}, "
            f"missing={summary['missing_repair_count']}, unchanged={summary['unchanged_count']}, "
            f"applied={summary['applied_count']}, discarded={summary['discarded_count']}, "
            f"rule_repairs={summary['rule_repair_count']}, failed_batches={summary['failed_batch_count']}, "
            f"warnings={len(warnings)}, "
            f"elapsed={_elapsed_seconds(started_at)}s"
        )
        self._print_content_diagnostics(summary)

        return replace(
            result,
            document=replace(result.document, children=children, sections=sections, metadata=document_metadata),
            warnings=list(result.warnings) + warnings,
        )

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
                self._record_applied_repair(
                    node_id=node_id,
                    original=text,
                    repaired=repaired,
                    summary=summary,
                    role=role,
                    language=language,
                    source="rule",
                )
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
            self._discard_repair(
                node_id=node_id,
                text=text,
                repair=repair,
                warnings=warnings,
                summary=summary,
                role=role,
                language=language,
                code="llm_content_preservation_failed",
                message="LLM content repair가 비공백 문자를 변경하여 결과 폐기.",
                reason="non_space_signature_changed",
            )
            return text, dict(metadata)

        unsafe_reason = self._unsafe_spacing_repair_reason(text, repair.text)
        if unsafe_reason is not None:
            self._discard_repair(
                node_id=node_id,
                text=text,
                repair=repair,
                warnings=warnings,
                summary=summary,
                role=role,
                language=language,
                code="llm_content_spacing_overcollapsed",
                message="LLM content repair가 정상 공백을 과도하게 삭제하여 결과 폐기.",
                reason=unsafe_reason,
            )
            return text, dict(metadata)

        if repair.text != text:
            summary["applied_count"] += 1
            self._record_applied_repair(
                node_id=node_id,
                original=text,
                repaired=repair.text,
                summary=summary,
                role=role,
                language=language,
                source="llm",
            )
            return repair.text, {
                **dict(metadata),
                **self._llm_metadata(CONTENT_TASK, repair.confidence),
                "llm_language": language,
            }

        summary["unchanged_count"] += 1
        self._append_summary_sample(summary, "unchanged_sample_ids", node_id)
        return text, dict(metadata)

    def _record_applied_repair(
        self,
        *,
        node_id: str,
        original: str,
        repaired: str,
        summary: dict[str, Any],
        role: str,
        language: str,
        source: str,
    ) -> None:
        self._append_summary_sample(summary, "applied_repair_ids", node_id)
        self._append_summary_sample(
            summary,
            "applied_repair_samples",
            {
                "node_id": node_id,
                "source": source,
                "role": role,
                "language": language,
                "original_excerpt": self._excerpt(original),
                "repair_excerpt": self._excerpt(repaired),
            },
        )

    def _discard_repair(
        self,
        *,
        node_id: str,
        text: str,
        repair: ContentRepair,
        warnings: list[AssemblyWarning],
        summary: dict[str, Any],
        role: str,
        language: str,
        code: str,
        message: str,
        reason: str,
    ) -> None:
        summary["discarded_count"] += 1
        self._append_summary_sample(summary, "discarded_repair_ids", node_id)
        self._append_summary_sample(
            summary,
            "discarded_repair_samples",
            {
                "node_id": node_id,
                "original_excerpt": self._excerpt(text),
                "repair_excerpt": self._excerpt(repair.text),
                "discard_reason": reason,
            },
        )
        warnings.append(
            self._warning(
                code,
                message,
                element_ids=[node_id],
                metadata={
                    "role": role,
                    "language": language,
                    "original_excerpt": self._excerpt(text),
                    "repair_excerpt": self._excerpt(repair.text),
                    "discard_reason": reason,
                },
            )
        )

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
        spacing_suspect = self._has_korean_internal_spacing_suspect(text)
        if len(self._non_space_signature(text)) < self.config.content_min_chars and not spacing_suspect:
            return None
        return _ContentRepairCandidate(
            node_id=node_id,
            text=text,
            role=role,
            language=language,
            spacing_suspect=spacing_suspect,
        )

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
                summary["failed_batch_count"] += 1
                if isinstance(error, LLMGenerationError):
                    summary["json_parse_failure_count"] += 1
                print(
                    f"[LLM][Content] batch {batch_index}/{len(batches)} 실패: "
                    f"items={len(batch)}, error={type(error).__name__}, "
                    f"message={_format_error(error)}, elapsed={_elapsed_seconds(started_at)}s"
                )
                warnings.append(
                    self._warning(
                        "llm_content_failed",
                        str(error),
                        element_ids=batch_ids,
                        metadata={"batch_index": batch_index, "batch_size": len(batch)},
                    )
                )
                continue

            repairs = parse_content_repairs(response)
            matched_count, missing_ids = self._store_matched_repairs(batch, repairs, repairs_by_node)
            summary["parsed_count"] += len(repairs)
            summary["matched_count"] += matched_count
            missing_count = max(0, len(batch) - matched_count)
            summary["missing_repair_count"] += missing_count
            for missing_id in missing_ids:
                self._append_summary_sample(summary, "missing_repair_ids", missing_id)
            print(
                f"[LLM][Content] batch {batch_index}/{len(batches)} 완료: "
                f"parsed={len(repairs)}, matched={matched_count}, missing={missing_count}, "
                f"elapsed={_elapsed_seconds(started_at)}s"
            )
            if missing_ids:
                print(
                    f"[LLM][Content] batch {batch_index}/{len(batches)} missing ids: "
                    f"{_format_node_ids(missing_ids, limit=8)}"
                )
        return repairs_by_node

    @staticmethod
    def _build_content_payload(batch: list[_ContentRepairCandidate]) -> dict[str, Any]:
        return {
            "items": [
                {
                    "node_id": candidate.node_id,
                    "text": candidate.text,
                    "non_space_signature": ContentEnricher._non_space_signature(candidate.text),
                    "language": candidate.language,
                    "role": candidate.role,
                    "spacing_suspect": candidate.spacing_suspect,
                }
                for candidate in batch
            ],
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
    ) -> tuple[int, list[str]]:
        repair_by_id = {repair.node_id: repair for repair in repairs}
        matched_count = 0
        missing_ids: list[str] = []
        for candidate in batch:
            repair = repair_by_id.get(candidate.node_id)
            if repair is None and len(batch) == 1 and len(repairs) == 1:
                repair = repairs[0]
            if repair is None:
                missing_ids.append(candidate.node_id)
                continue
            repairs_by_node[candidate.node_id] = repair
            matched_count += 1
        return matched_count, missing_ids

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
        hangul = len(re.findall(r"[\uac00-\ud7a3]", compact))
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

    @classmethod
    def _unsafe_spacing_repair_reason(cls, original: str, repaired: str) -> str | None:
        if original == repaired:
            return None

        original_gap_count = cls._space_gap_count(original)
        if original_gap_count == 0:
            return None

        repaired_gap_count = cls._space_gap_count(repaired)
        removed_contexts = cls._removed_space_contexts(original, repaired)
        if not removed_contexts:
            return None

        if original_gap_count >= 3 and repaired_gap_count == 0:
            return "removed_all_spaces"

        for left, right in removed_contexts:
            if not (cls._is_hangul_syllable(left) and cls._is_hangul_syllable(right)):
                return "removed_non_hangul_boundary_space"

        if original_gap_count >= 8 and len(removed_contexts) / original_gap_count > 0.45:
            return "removed_too_many_spaces"

        return None

    @classmethod
    def _removed_space_contexts(cls, original: str, repaired: str) -> list[tuple[str, str]]:
        signature = cls._non_space_signature(original)
        original_gaps = cls._space_gaps_between_non_space(original)
        repaired_gaps = cls._space_gaps_between_non_space(repaired)
        if len(signature) < 2 or len(original_gaps) != len(repaired_gaps):
            return []

        removed: list[tuple[str, str]] = []
        for index, (original_gap, repaired_gap) in enumerate(zip(original_gaps, repaired_gaps)):
            if original_gap and not repaired_gap:
                removed.append((signature[index], signature[index + 1]))
        return removed

    @staticmethod
    def _space_gap_count(text: str) -> int:
        return sum(1 for gap in ContentEnricher._space_gaps_between_non_space(text) if gap)

    @staticmethod
    def _space_gaps_between_non_space(text: str) -> list[str]:
        gaps: list[str] = []
        previous_non_space_seen = False
        pending_space = ""
        for char in text:
            if char.isspace():
                if previous_non_space_seen:
                    pending_space += char
                continue
            if previous_non_space_seen:
                gaps.append(pending_space)
            previous_non_space_seen = True
            pending_space = ""
        return gaps

    @staticmethod
    def _is_hangul_syllable(char: str) -> bool:
        return "\uac00" <= char <= "\ud7a3"

    @staticmethod
    def _append_summary_sample(summary: dict[str, Any], key: str, value: Any) -> None:
        samples = summary.get(key)
        if not isinstance(samples, list):
            return
        if len(samples) < CONTENT_SUMMARY_SAMPLE_LIMIT:
            samples.append(value)

    @staticmethod
    def _excerpt(text: str, *, limit: int = 120) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit] + "..."

    @staticmethod
    def _print_content_diagnostics(summary: dict[str, Any]) -> None:
        applied_ids = summary.get("applied_repair_ids") or []
        missing_ids = summary.get("missing_repair_ids") or []
        discarded_ids = summary.get("discarded_repair_ids") or []
        unchanged_ids = summary.get("unchanged_sample_ids") or []
        if applied_ids:
            print(f"[LLM][Content] applied ids sample: {_format_node_ids(list(applied_ids), limit=8)}")
        if missing_ids:
            print(f"[LLM][Content] missing ids sample: {_format_node_ids(list(missing_ids), limit=8)}")
        if discarded_ids:
            print(f"[LLM][Content] discarded ids sample: {_format_node_ids(list(discarded_ids), limit=8)}")
        if unchanged_ids:
            print(f"[LLM][Content] unchanged ids sample: {_format_node_ids(list(unchanged_ids), limit=8)}")

    @staticmethod
    def _has_korean_internal_spacing_suspect(text: str) -> bool:
        tokens = [token for token in re.split(r"\s+", text.strip()) if token]
        if len(tokens) < 2:
            return False

        for left, right in zip(tokens, tokens[1:]):
            left_hangul = "".join(re.findall(r"[\uac00-\ud7a3]", left))
            right_hangul = "".join(re.findall(r"[\uac00-\ud7a3]", right))
            if left_hangul and right_hangul and (len(left_hangul) == 1 or len(right_hangul) == 1):
                return True
        return False

    @staticmethod
    def _non_space_signature(text: str) -> str:
        return non_space_signature(text)
