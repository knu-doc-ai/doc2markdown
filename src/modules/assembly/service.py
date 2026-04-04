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

    def build_from_outputs(self, layout_output: Any, table_output: Any = None) -> AssemblyResult:
        """
        layout/table 출력을 명시적으로 받아 초기 Assembly IR을 만든다.

        Layout Analysis와 Table Extraction이 서로 다른 타이밍에 연결될 때
        상위 파이프라인이 raw payload 포맷을 직접 조립하지 않아도 되게 하는 얇은 헬퍼다.
        """
        return AssemblyInputAdapter.from_outputs(layout_output, table_output)


__all__ = [
    "DocumentAssembler",
]
