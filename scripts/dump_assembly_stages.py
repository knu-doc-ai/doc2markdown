import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "2단 문서 text"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _assembly_debug_utils import (
    ASSEMBLY_STAGE_OUTPUTS,
    DEFAULT_READING_ORDER_STRATEGY,
    build_stage_results_from_outputs,
    save_stage_results,
)


def resolve_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        candidate = Path(raw_output_dir)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    return DEFAULT_OUTPUT_DIR


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compact_text(text: str | None, max_length: int = 56) -> str:
    if not text:
        return ""
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3] + "..."


def element_position(element) -> tuple[str, str]:
    if element.bbox is None:
        return "-", "-"
    return f"{float(element.bbox[0]):.1f}", f"{float(element.bbox[1]):.1f}"


def print_stage_summary(stage_name: str, result, limit: int) -> None:
    print(
        f"\n[{stage_name}] stage={result.metadata.stage} "
        f"elements={len(result.ordered_elements)} "
        f"relations={len(result.block_relations)} "
        f"warnings={len(result.warnings)}"
    )

    for element in result.ordered_elements[:limit]:
        left, top = element_position(element)
        print(
            "  "
            f"{element.id:<18} "
            f"kind={element.kind:<10} "
            f"page={element.page:<2} "
            f"col={str(element.column_id):<4} "
            f"ro={str(element.reading_order):<4} "
            f"left={left:<8} "
            f"top={top:<8} "
            f"text={compact_text(element.text)}"
        )

    if len(result.ordered_elements) > limit:
        print(f"  ... {len(result.ordered_elements) - limit} more elements omitted")


def collect_stage_changes(previous_result, current_result) -> list[str]:
    if previous_result is None:
        return []

    previous_by_id = {element.id: element for element in previous_result.ordered_elements}
    changes: list[str] = []
    tracked_fields = ("column_id", "reading_order", "parent_id")

    for element in current_result.ordered_elements:
        previous_element = previous_by_id.get(element.id)
        if previous_element is None:
            changes.append(f"{element.id}: added")
            continue

        field_changes = []
        for field_name in tracked_fields:
            before = getattr(previous_element, field_name)
            after = getattr(element, field_name)
            if before != after:
                field_changes.append(f"{field_name} {before!r} -> {after!r}")

        if field_changes:
            changes.append(f"{element.id}: " + ", ".join(field_changes))

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load saved layout/table outputs and dump each assembly stage as JSON."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Pipeline output directory containing metadata.json and table_results.json. Defaults to data/output/2단 문서 text.",
    )
    parser.add_argument(
        "--write-dir",
        default=None,
        help="Directory to save per-stage JSON files. Defaults to <output_dir>/debug/assembly_stages.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="How many ordered elements to preview per stage in the console.",
    )
    parser.add_argument(
        "--hide-elements",
        action="store_true",
        help="Only print stage headers and change summaries.",
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "layout_priority"],
        default=DEFAULT_READING_ORDER_STRATEGY,
        help=(
            "Reading-order strategy to use when rebuilding assembly stages. "
            f"Defaults to {DEFAULT_READING_ORDER_STRATEGY}."
        ),
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    layout_path = output_dir / "metadata.json"
    table_path = output_dir / "table_results.json"

    if not layout_path.exists():
        raise FileNotFoundError(f"Layout output not found: {layout_path}")
    if not table_path.exists():
        raise FileNotFoundError(f"Table output not found: {table_path}")

    layout_output = load_json(layout_path)
    table_output = load_json(table_path)

    stage_results = build_stage_results_from_outputs(
        layout_output,
        table_output,
        strategy=args.strategy,
    )

    write_dir = (
        Path(args.write_dir).resolve()
        if args.write_dir
        else output_dir / "debug" / "assembly_stages"
    )
    save_stage_results(write_dir, stage_results)

    print(f"output_dir: {output_dir}")
    print(f"write_dir: {write_dir}")

    previous_result = None
    for stage_name, filename in ASSEMBLY_STAGE_OUTPUTS:
        result = stage_results[stage_name]
        print(f"saved: {write_dir / filename}")
        if not args.hide_elements:
            print_stage_summary(stage_name, result, max(0, args.limit))

        changes = collect_stage_changes(previous_result, result)
        if changes:
            print("  changed-from-previous:")
            for line in changes[: args.limit]:
                print(f"    {line}")
            if len(changes) > args.limit:
                print(f"    ... {len(changes) - args.limit} more changes omitted")

        previous_result = result


if __name__ == "__main__":
    main()
