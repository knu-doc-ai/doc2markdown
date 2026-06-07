from __future__ import annotations

"""Assembly IR 단계 파이프라인과 trace 수집."""

import time
from dataclasses import dataclass
from typing import Any, Callable

from modules.assembly.ir import AssemblyResult


@dataclass(frozen=True)
class AssemblyBuildTrace:
    result: AssemblyResult
    stages: dict[str, AssemblyResult]


class AssemblyPipeline:
    """Assembly IR 단계를 순서대로 실행하고 선택적 중간 결과를 보관한다.

    파이프라인 순서:
    1. adapter_seed: layout/table 출력을 초기 AssemblyResult로 변환한다.
    2. normalized: 좌표/텍스트를 정규화하고 노이즈 layout 요소를 필터링한다.
    3. semantic_enriched: normalized block에 선택적 LLM semantic 보강을 적용한다.
    4. structure_assembled: section, paragraph, list, ref 구조를 조립한다.
    5. content_enriched: 조립된 문서 텍스트에 선택적 LLM content 보강을 적용한다.
    6. validated: 최종 문서와 block relation을 검증한다.
    """

    def __init__(
        self,
        assembler: Any,
        *,
        semantic_enricher: Any = None,
        content_enricher: Any = None,
    ):
        self.assembler = assembler
        self.semantic_enricher, self.content_enricher = self._resolve_enrichers(
            semantic_enricher=semantic_enricher,
            content_enricher=content_enricher,
        )

    def build_from_outputs(
        self,
        layout_output: Any,
        table_output: Any = None,
    ) -> AssemblyBuildTrace:
        """layout/table 출력에서 validated AssemblyResult와 단계별 trace를 만든다."""
        stages: dict[str, AssemblyResult] = {}

        seed_result = self._run_stage(
            "adapter_seed",
            lambda: self.assembler.build_seed_from_outputs(layout_output, table_output),
        )
        stages["adapter_seed"] = seed_result

        normalized_result = self._run_stage(
            "NormalizeFilter",
            lambda: self.assembler.normalize(seed_result),
        )
        stages["normalized"] = normalized_result

        semantic_result = self._run_enrichment_stage(
            stage_key="semantic_enriched",
            label="SemanticEnricher",
            result=normalized_result,
            enricher=self.semantic_enricher,
            enabled_method_name="enables_semantic",
            stages=stages,
        )

        structure_result = self._run_stage(
            "StructureAssembler",
            lambda: self.assembler.assemble_structure(semantic_result),
        )
        stages["structure_assembled"] = structure_result

        content_result = self._run_enrichment_stage(
            stage_key="content_enriched",
            label="ContentEnricher",
            result=structure_result,
            enricher=self.content_enricher,
            enabled_method_name="enables_content",
            stages=stages,
        )

        validated_result = self._run_stage(
            "AssemblyValidator",
            lambda: self.assembler.validate(content_result),
        )
        stages["validated"] = validated_result

        return AssemblyBuildTrace(result=validated_result, stages=stages)

    @staticmethod
    def _resolve_enrichers(
        *,
        semantic_enricher: Any = None,
        content_enricher: Any = None,
    ) -> tuple[Any, Any]:
        """주입된 enricher를 쓰거나 환경 설정 기반 기본 enricher를 만든다."""
        if semantic_enricher is not None and content_enricher is not None:
            return semantic_enricher, content_enricher

        from modules.assembly.stages.enrichment import (
            ContentEnricher,
            LLMConfig,
            SemanticEnricher,
            print_enrichment_config,
        )

        config = LLMConfig.from_env()
        print_enrichment_config(config)
        if semantic_enricher is None:
            semantic_enricher = SemanticEnricher(config=config)
        if content_enricher is None:
            content_enricher = ContentEnricher(config=config)
        return semantic_enricher, content_enricher

    def _run_enrichment_stage(
        self,
        *,
        stage_key: str,
        label: str,
        result: AssemblyResult,
        enricher: Any,
        enabled_method_name: str,
        stages: dict[str, AssemblyResult],
    ) -> AssemblyResult:
        """enrichment 단계를 실행하고 활성화된 경우에만 trace에 기록한다."""
        if not self._is_enricher_enabled(enricher, enabled_method_name):
            mode = self._enricher_mode(enricher)
            print(f"[Assembly] {label} 건너뜀: mode={mode}")
            return enricher.apply(result)

        enriched_result = self._run_stage(label, lambda: enricher.apply(result))
        stages[stage_key] = enriched_result
        return enriched_result

    @staticmethod
    def _is_enricher_enabled(enricher: Any, enabled_method_name: str) -> bool:
        config = getattr(enricher, "config", None)
        enabled_method = getattr(config, enabled_method_name, None)
        if callable(enabled_method):
            return bool(enabled_method())
        return True

    @staticmethod
    def _enricher_mode(enricher: Any) -> str:
        config = getattr(enricher, "config", None)
        return str(getattr(config, "mode", "unknown"))

    def _run_stage(self, label: str, action: Callable[[], AssemblyResult]) -> AssemblyResult:
        print(f"[Assembly] {label} 시작")
        started_at = time.perf_counter()
        result = action()
        self._print_stage_summary(label, result, started_at)
        return result

    @staticmethod
    def _print_stage_summary(label: str, result: AssemblyResult, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        metadata = getattr(result, "metadata", None)
        stage = getattr(metadata, "stage", None) or "-"
        elements = getattr(result, "ordered_elements", []) or []
        warnings = getattr(result, "warnings", []) or []
        document = getattr(result, "document", None)

        print(
            f"[Assembly] {label} 완료: "
            f"stage={stage}, elements={len(elements)}, warnings={len(warnings)}, elapsed={elapsed:.2f}s"
        )
        if document is None:
            return

        children = getattr(document, "children", []) or []
        sections = getattr(document, "sections", []) or []
        table_refs = getattr(document, "table_refs", []) or []
        figure_refs = getattr(document, "figure_refs", []) or []
        print(
            f"[Assembly] {label} 문서: "
            f"children={len(children)}, sections={len(sections)}, "
            f"tables={len(table_refs)}, figures={len(figure_refs)}"
        )
