from __future__ import annotations

"""Assembly 엔트리포인트 서비스 모듈."""

from typing import Any

from modules.assembly.adapters import AssemblyInputAdapter
from modules.assembly.ir import AssemblyResult


class DocumentAssembler:
    """
    Assembly 규칙 엔진의 진입점

    현재 단계에서는 어댑터 계층으로 입력 스펙을 고정하고
    실제 Normalize/ReadingOrder/Structure 조립 로직은 이후 단계에서 확장
    """

    def build(self, raw: Any) -> AssemblyResult:
        """
        현재 단계에서는 raw 입력을 내부 표준 IR의 초기 시드로만 변환한다.

        다음 단계의 Normalize / ReadingOrderResolver / StructureAssembler /
        Validator는 이 결과를 입력으로 받아 확장되도록 의도한다.
        """
        return AssemblyInputAdapter.from_raw(raw)


__all__ = [
    "DocumentAssembler",
]
