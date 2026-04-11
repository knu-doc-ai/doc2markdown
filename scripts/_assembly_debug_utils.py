from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from modules.assembly.adapters import AssemblyInputAdapter
from modules.assembly.ir import AssemblyResult
from modules.assembly.normalize_filter import NormalizeFilter
from modules.assembly.reading_order import ReadingOrderResolver
from modules.assembly.structure import StructureAssembler
from modules.assembly.validator import AssemblyValidator


ASSEMBLY_STAGE_OUTPUTS = [
    ("adapter_seed", "01_adapter_seed.json"),
    ("normalized", "02_normalized.json"),
    ("reading_order_resolved", "03_reading_order_resolved.json"),
    ("structure_assembled", "04_structure_assembled.json"),
    ("validated", "05_validated.json"),
]


def serialize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def strip_raw_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_raw_fields(item)
            for key, item in value.items()
            if key != "raw"
        }

    if isinstance(value, list):
        return [strip_raw_fields(item) for item in value]

    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(strip_raw_fields(serialize(payload)), file, ensure_ascii=False, indent=2)


def save_stage_results(write_dir: Path, stage_results: Dict[str, AssemblyResult]) -> Dict[str, str]:
    saved_paths: Dict[str, str] = {}
    write_dir.mkdir(parents=True, exist_ok=True)

    for stage_name, filename in ASSEMBLY_STAGE_OUTPUTS:
        if stage_name not in stage_results:
            continue

        path = write_dir / filename
        save_json(path, stage_results[stage_name])
        saved_paths[stage_name] = str(path)

    return saved_paths


def build_stage_results_from_outputs(
    layout_output: Any,
    table_output: Any = None,
) -> Dict[str, AssemblyResult]:
    seed_result = AssemblyInputAdapter.from_raw(
        {
            "layout_output": layout_output,
            "table_output": table_output,
        }
    )
    normalized_result = NormalizeFilter.apply(seed_result)
    reading_order_result = ReadingOrderResolver.apply(normalized_result)
    structure_result = StructureAssembler.apply(reading_order_result)
    validated_result = AssemblyValidator.apply(structure_result)

    return {
        "adapter_seed": seed_result,
        "normalized": normalized_result,
        "reading_order_resolved": reading_order_result,
        "structure_assembled": structure_result,
        "validated": validated_result,
    }
