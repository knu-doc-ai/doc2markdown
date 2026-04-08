from __future__ import annotations

"""Assembly 엔트리포인트 서비스 모듈."""

from typing import Any

from modules.assembly.adapters import AssemblyInputAdapter
from modules.assembly.ir import AssemblyResult
from modules.assembly.normalize_filter import NormalizeFilter


class DocumentAssembler:
    """
    Assembly 규칙 엔진의 진입점

    현재 단계에서는 어댑터 계층으로 입력 스펙을 고정하고
    Normalize 단계까지 연결한다.
    """

    def build(self, raw: Any) -> AssemblyResult:
        """
        raw 입력을 adapter seed로 바꾼 뒤 Normalize / Filter 단계까지 수행한다.
        """
        seed_result = AssemblyInputAdapter.from_raw(raw)
        return NormalizeFilter.apply(seed_result)

    def build_from_outputs(self, layout_output: Any, table_output: Any = None) -> AssemblyResult:
        """
        layout/table 출력을 명시적으로 받아 Normalize 단계까지 수행한다.

        Layout Analysis와 Table Extraction이 서로 다른 타이밍에 연결될 때
        상위 파이프라인이 raw payload 포맷을 직접 조립하지 않아도 되게 하는 얇은 헬퍼다.
        """
        return self.build(
            {
                "layout_output": layout_output,
                "table_output": table_output,
            }
        )


__all__ = [
    "DocumentAssembler",
]
