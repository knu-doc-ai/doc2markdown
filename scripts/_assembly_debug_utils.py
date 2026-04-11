from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from modules.assembly.adapters import AssemblyInputAdapter
from modules.assembly.ir import AssemblyResult, PageStats
from modules.assembly.normalize_filter import NormalizeFilter
from modules.assembly.reading_order import ReadingOrderResolver, _PageEntry
from modules.assembly.structure import StructureAssembler
from modules.assembly.validator import AssemblyValidator


ASSEMBLY_STAGE_OUTPUTS = [
    ("adapter_seed", "01_adapter_seed.json"),
    ("normalized", "02_normalized.json"),
    ("reading_order_resolved", "03_reading_order_resolved.json"),
    ("structure_assembled", "04_structure_assembled.json"),
    ("validated", "05_validated.json"),
]

# DEFAULT_READING_ORDER_STRATEGY = "default"
DEFAULT_READING_ORDER_STRATEGY = "layout_priority"


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
    strategy: str = DEFAULT_READING_ORDER_STRATEGY,
) -> Dict[str, AssemblyResult]:
    seed_result = AssemblyInputAdapter.from_raw(
        {
            "layout_output": layout_output,
            "table_output": table_output,
        }
    )
    normalized_result = NormalizeFilter.apply(seed_result)

    if strategy == "layout_priority":
        reading_order_result = apply_layout_priority_reading_order(normalized_result)
    else:
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


def apply_layout_priority_reading_order(result: AssemblyResult) -> AssemblyResult:
    if not isinstance(result, AssemblyResult):
        return result

    if ReadingOrderResolver._should_skip_resolution(result.metadata.stage):
        return result

    if result.metadata.stage != "normalized":
        result = NormalizeFilter.apply(result)

    page_stats_by_page = {page_stat.page: page_stat for page_stat in result.page_stats}
    elements_by_page: Dict[int, list] = {}
    original_order_map: Dict[str, int] = {}

    for index, element in enumerate(result.ordered_elements):
        elements_by_page.setdefault(element.page, []).append(element)
        original_order_map[element.id] = index

    all_pages = sorted(set(page_stats_by_page) | set(elements_by_page))
    resolved_elements = []
    resolved_page_stats = []
    page_summaries = []
    reading_order_index = 1

    for page in all_pages:
        page_elements = list(elements_by_page.get(page, []))
        page_stat = page_stats_by_page.get(page, PageStats(page=page))
        page_plan = ReadingOrderResolver._build_page_plan(page, page_elements, page_stat)

        ordered_entries = [
            _PageEntry(
                original_index=original_order_map.get(element.id, index),
                element=element,
                resolved_column_id=ReadingOrderResolver._resolve_column_id(element, page_plan),
            )
            for index, element in enumerate(page_elements)
        ]

        for entry in ordered_entries:
            metadata = dict(entry.element.metadata)
            metadata["reading_order_resolved"] = True
            metadata["column_assignment_source"] = page_plan.column_source
            metadata["reading_order_strategy"] = "layout_sequence_priority"
            if (
                entry.element.column_id is not None
                and entry.element.column_id != entry.resolved_column_id
            ):
                metadata["upstream_column_id"] = entry.element.column_id

            resolved_elements.append(
                replace(
                    entry.element,
                    column_id=entry.resolved_column_id,
                    reading_order=reading_order_index,
                    metadata=metadata,
                )
            )
            reading_order_index += 1

        resolved_page_stats.append(
            ReadingOrderResolver._build_resolved_page_stats(
                page_stat,
                page_plan,
                ordered_entries,
            )
        )
        page_summaries.append(
            {
                "page": page,
                "element_count": len(ordered_entries),
                "column_count": page_plan.column_count,
                "column_source": page_plan.column_source,
                "same_line_threshold": page_plan.same_line_threshold,
                "column_gap_threshold": page_plan.column_gap_threshold,
                "column_centers": [band.center_x for band in page_plan.bands],
                "strategy": "layout_sequence_priority",
            }
        )

    next_relations = ReadingOrderResolver._build_next_relations(resolved_elements)
    reading_order_summary = {
        "page_count": len(all_pages),
        "element_count": len(resolved_elements),
        "next_relation_count": len(next_relations),
        "strategy": "layout_sequence_priority",
        "pages": page_summaries,
    }

    document_metadata = dict(result.document.metadata)
    document_metadata["reading_order"] = reading_order_summary

    return AssemblyResult(
        ordered_elements=resolved_elements,
        block_relations=ReadingOrderResolver._merge_next_relations(
            result.block_relations,
            next_relations,
        ),
        document=replace(result.document, metadata=document_metadata),
        page_stats=resolved_page_stats,
        warnings=list(result.warnings),
        metadata=ReadingOrderResolver._build_resolved_metadata(
            result.metadata,
            reading_order_summary,
        ),
        raw=result.raw,
    )
