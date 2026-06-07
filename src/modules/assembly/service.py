from __future__ import annotations

"""Assembly IR 생성 공개 진입점을 제공한다."""

from pathlib import Path
from typing import Any

from modules.assembly.adapters import from_outputs as adapter_from_outputs
from modules.assembly.adapters import from_raw as adapter_from_raw
from modules.assembly.ir import AssemblyResult
from modules.assembly.pipeline import AssemblyBuildTrace, AssemblyPipeline
from modules.assembly.stages.contracts import require_assembly_result, require_stage
from modules.assembly.stages.normalize_filter import NormalizeFilter
from modules.assembly.stages.structure import StructureAssembler
from modules.assembly.stages.validation import AssemblyValidator


class DocumentAssembler:
    """raw/layout/table 입력을 Assembly IR로 조립하는 공개 관문이다."""

    def build(self, raw: Any) -> AssemblyResult:
        """raw 입력을 validated AssemblyResult로 만들거나 그대로 돌려준다."""
        if isinstance(raw, AssemblyResult) and raw.metadata.stage == "validated":
            return raw
        return self.validate(self.build_structure(raw))

    @staticmethod
    def build_seed(raw: Any) -> AssemblyResult:
        """공개 raw payload를 adapter_seed AssemblyResult로 돌려준다."""
        if isinstance(raw, AssemblyResult):
            return raw
        return adapter_from_raw(raw)

    @staticmethod
    def build_seed_from_outputs(layout_output: Any, table_output: Any = None) -> AssemblyResult:
        """layout/table 출력을 adapter_seed AssemblyResult로 돌려준다."""
        return adapter_from_outputs(layout_output, table_output)

    def build_normalized(self, raw: Any) -> AssemblyResult:
        """raw 입력을 normalized AssemblyResult로 만들거나 그대로 돌려준다."""
        seed_result = self.build_seed(raw)
        if seed_result.metadata.stage == "normalized":
            return seed_result
        return self.normalize(seed_result)

    def build_structure(self, raw: Any) -> AssemblyResult:
        """raw 입력을 structure_assembled AssemblyResult로 만들거나 그대로 돌려준다."""
        if isinstance(raw, AssemblyResult) and raw.metadata.stage == "structure_assembled":
            return raw
        normalized_result = self.build_normalized(raw)
        return self.assemble_structure(normalized_result)

    def build_from_outputs(self, layout_output: Any, table_output: Any = None) -> AssemblyResult:
        """layout/table 출력에서 validated AssemblyResult를 만든다."""
        return self.build(self.build_seed_from_outputs(layout_output, table_output))

    def build_from_outputs_with_trace(
        self,
        layout_output: Any,
        table_output: Any = None,
        *,
        semantic_enricher: Any = None,
        content_enricher: Any = None,
    ) -> AssemblyBuildTrace:
        """layout/table 출력에서 validated AssemblyResult와 중간 단계 trace를 만든다."""
        return AssemblyPipeline(
            self,
            semantic_enricher=semantic_enricher,
            content_enricher=content_enricher,
        ).build_from_outputs(layout_output, table_output)

    @staticmethod
    def get_output_subdir() -> Path | None:
        """현재 환경에서 assembly가 소유하는 출력 variant 하위 경로를 돌려준다."""
        from modules.assembly.stages.enrichment import LLMConfig

        config = LLMConfig.from_env()
        if not config.uses_enrichment():
            return None
        return Path("llm_enrichment") / config.mode

    @staticmethod
    def normalize(seed_result: AssemblyResult) -> AssemblyResult:
        """adapter_seed 입력에서 normalized 단계를 실행한다."""
        seed_result = require_assembly_result(seed_result, "DocumentAssembler.normalize")
        require_stage(seed_result, "adapter_seed", "DocumentAssembler.normalize")
        return NormalizeFilter.apply(seed_result)

    @staticmethod
    def assemble_structure(normalized_result: AssemblyResult) -> AssemblyResult:
        """normalized 입력에서 structure assembly 단계를 실행한다."""
        normalized_result = require_assembly_result(
            normalized_result,
            "DocumentAssembler.assemble_structure",
        )
        require_stage(normalized_result, "normalized", "DocumentAssembler.assemble_structure")
        return StructureAssembler.apply(normalized_result)

    @staticmethod
    def validate(structure_result: AssemblyResult) -> AssemblyResult:
        """structure_assembled 입력에서 최종 validation 단계를 실행한다."""
        structure_result = require_assembly_result(structure_result, "DocumentAssembler.validate")
        require_stage(structure_result, "structure_assembled", "DocumentAssembler.validate")
        return AssemblyValidator.apply(structure_result)
