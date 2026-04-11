import json
import sys
import unittest
from pathlib import Path


# 테스트 실행 위치와 관계없이 src를 import 경로에 추가한다.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.assembly import DocumentAssembler, from_layout_output, from_table_output


FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "assembly"


def load_fixture(name: str):
    """fixture JSON을 읽어 dict/list로 반환한다."""
    with (FIXTURE_DIR / f"{name}.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def build_layout_payload(
    elements,
    *,
    file_name: str = "mixed_layout_case.pdf",
    width: int = 3000,
    height: int = 2895,
):
    return {
        "layout_output": {
            "file_name": file_name,
            "total_pages": 1,
            "pages": [
                {
                    "page_num": 1,
                    "width": width,
                    "height": height,
                    "elements": elements,
                }
            ],
        },
        "table_output": [],
    }


class LayoutAdapterContractTests(unittest.TestCase):
    def test_single_column_fixture_is_normalized_into_layout_ir(self):
        # 단일 컬럼 layout fixture가 기본 Assembly IR로 안정적으로 변환되는지 검증한다.
        raw = load_fixture("single_column")

        result = from_layout_output(raw)

        self.assertEqual(result.metadata.stage, "adapter_seed")
        self.assertEqual(result.metadata.adapter, "layout")
        self.assertEqual(result.metadata.source, "raw")
        self.assertEqual(len(result.ordered_elements), 4)
        self.assertEqual([element.kind for element in result.ordered_elements], ["heading", "text", "text", "list_item"])
        self.assertEqual(result.document.title_candidate, "Document Title")
        self.assertEqual(result.document.title_source_block_ids, ["h1"])
        self.assertEqual(len(result.page_stats), 1)
        self.assertEqual(result.page_stats[0].column_count, 1)
        self.assertEqual(result.warnings, [])

    def test_two_column_fixture_preserves_upstream_column_and_order_hints(self):
        # upstream이 준 column_id / reading_order 힌트를 어댑터가 손상 없이 보존하는지 검증한다.
        raw = load_fixture("two_column")

        result = from_layout_output(raw)

        self.assertEqual([element.column_id for element in result.ordered_elements], [1, 1, 2, 2])
        self.assertEqual([element.reading_order for element in result.ordered_elements], [1, 2, 3, 4])
        self.assertEqual(result.page_stats[0].column_count, 2)

    def test_heading_list_fixture_supports_direct_list_source_and_label_aliases(self):
        # direct list 입력과 label alias 매핑(Title, bullet, list, body)이 기대한 kind로 정규화되는지 검증한다.
        raw = load_fixture("heading_list")

        result = from_layout_output(raw)

        self.assertEqual(result.metadata.source, "direct_list")
        self.assertEqual([element.kind for element in result.ordered_elements], ["heading", "list_item", "list_item", "text"])
        self.assertEqual(result.document.title_candidate, "Section A")
        self.assertEqual(result.document.title_source_block_ids, ["title_1"])

    def test_layout_adapter_emits_fallback_warnings_for_missing_id_and_page(self):
        # id/page가 없는 layout 블록에 대해 fallback 값과 warning을 함께 남기는지 검증한다.
        raw = [{"label": "text", "text": "Loose paragraph"}]

        result = from_layout_output(raw)
        warning_codes = [warning.code for warning in result.warnings]

        self.assertEqual(result.metadata.source, "direct_list")
        self.assertEqual(result.ordered_elements[0].id, "element_1")
        self.assertEqual(result.ordered_elements[0].page, 1)
        self.assertIn("layout_missing_id", warning_codes)
        self.assertIn("layout_missing_page", warning_codes)


class TableAdapterContractTests(unittest.TestCase):
    def test_table_adapter_keeps_minimum_table_reference_and_extra_metadata(self):
        # table spec이 완전하지 않아도 핵심 ref 정보는 표준화하고 나머지 메타데이터는 보존하는지 검증한다.
        raw = load_fixture("table_caption_note")["table_output"]

        result = from_table_output(raw)

        self.assertEqual(result.metadata.stage, "adapter_seed")
        self.assertEqual(result.metadata.adapter, "table")
        self.assertEqual(result.metadata.source, "raw")
        self.assertEqual(len(result.document.table_refs), 1)

        table_ref = result.document.table_refs[0]
        self.assertEqual(table_ref.table_id, "table_1")
        self.assertEqual(table_ref.page, 1)
        self.assertEqual(table_ref.caption_id, "cap_1")
        self.assertEqual(table_ref.note_ids, ["note_1"])
        self.assertEqual(table_ref.source_block_ids, ["table_1"])
        self.assertEqual(table_ref.metadata["rows"], 3)
        self.assertEqual(table_ref.metadata["columns"], 4)
        self.assertEqual(table_ref.metadata["parser"], "stub")

    def test_table_adapter_emits_fallback_warnings_for_missing_id_and_page(self):
        # id/page가 빠진 table 입력에도 fallback 값과 warning이 생성되는지 검증한다.
        result = from_table_output([{"table_id": None}])
        warning_codes = [warning.code for warning in result.warnings]

        self.assertEqual(result.metadata.source, "direct_list")
        self.assertEqual(result.document.table_refs[0].table_id, "table_1")
        self.assertEqual(result.document.table_refs[0].page, 1)
        self.assertIn("table_missing_id", warning_codes)
        self.assertIn("table_missing_page", warning_codes)

    def test_table_adapter_converts_plain_markdown_string_into_seed_ref(self):
        # plain markdown 문자열을 table seed ref로 바꾸고 markdown 본문을 metadata에 보존하는지 검증한다.
        raw = load_fixture("markdown_table_seed")["table_markdown"]

        result = from_table_output(raw)
        warning_codes = [warning.code for warning in result.warnings]

        self.assertEqual(result.metadata.stage, "adapter_seed")
        self.assertEqual(result.metadata.adapter, "table")
        self.assertEqual(result.metadata.source, "raw")
        self.assertEqual(len(result.document.table_refs), 1)

        table_ref = result.document.table_refs[0]
        self.assertEqual(table_ref.table_id, "table_1")
        self.assertEqual(table_ref.page, 1)
        self.assertEqual(table_ref.metadata["content_format"], "markdown")
        self.assertIn("| 카테고리 | 기능 ID |", table_ref.metadata["markdown"])
        self.assertIn("table_missing_id", warning_codes)
        self.assertIn("table_missing_page", warning_codes)

    def test_table_adapter_accepts_mixed_raw_and_markdown_entries_in_one_list(self):
        # 기존 raw table entry와 markdown 문자열이 한 리스트에 섞여 와도 둘 다 seed ref로 변환하는지 검증한다.
        raw = [
            {
                "table_id": "table_raw_1",
                "page": 3,
                "bbox": [10, 20, 200, 120],
                "rows": 2,
                "columns": 2,
            },
            load_fixture("markdown_table_seed")["table_markdown"],
        ]

        result = from_table_output(raw)

        self.assertEqual(result.metadata.source, "direct_list")
        self.assertEqual(len(result.document.table_refs), 2)
        self.assertEqual(result.document.table_refs[0].table_id, "table_raw_1")
        self.assertEqual(result.document.table_refs[0].page, 3)
        self.assertEqual(result.document.table_refs[1].table_id, "table_2")
        self.assertEqual(result.document.table_refs[1].metadata["content_format"], "markdown")
        self.assertIn("| 기본 연산 | FUNC-001 |", result.document.table_refs[1].metadata["markdown"])


class AssemblyServiceContractTests(unittest.TestCase):
    def test_document_assembler_merges_layout_and_table_outputs(self):
        # 통합 payload에서 layout/table 출력을 함께 받아 normalize 단계 결과를 반환하는지 검증한다.
        raw = load_fixture("table_caption_note")

        result = DocumentAssembler().build(raw)

        self.assertEqual(result.metadata.stage, "validated")
        self.assertEqual(result.metadata.adapter, "merged")
        self.assertEqual(result.metadata.source, "raw")
        self.assertEqual(len(result.ordered_elements), 4)
        self.assertEqual([element.id for element in result.ordered_elements], ["intro_1", "table_1", "cap_1", "note_1"])
        self.assertEqual(len(result.document.table_refs), 1)
        self.assertEqual(len(result.document.note_refs), 1)
        self.assertEqual(result.document.note_refs[0].note_id, "note_1")
        self.assertEqual(result.document.table_refs[0].caption_id, "cap_1")
        self.assertEqual(result.document.table_refs[0].note_ids, ["note_1"])

        serialized = result.to_dict()
        self.assertEqual(serialized["document"]["table_refs"][0]["table_id"], "table_1")
        self.assertEqual(serialized["metadata"]["adapter"], "merged")

    def test_document_assembler_build_from_outputs_links_markdown_table_to_layout_ref(self):
        # build_from_outputs가 layout output과 plain markdown table output을 받아 normalize 결과를 반환하는지 검증한다.
        fixture = load_fixture("layout_markdown_link")

        result = DocumentAssembler().build_from_outputs(
            fixture["layout_output"],
            fixture["table_markdown"],
        )

        self.assertEqual(result.metadata.stage, "validated")
        self.assertEqual(result.metadata.adapter, "merged")
        self.assertEqual(len(result.ordered_elements), 1)
        self.assertEqual([element.id for element in result.ordered_elements], ["p1_table_7"])
        self.assertEqual(len(result.document.table_refs), 1)

        table_ref = result.document.table_refs[0]
        self.assertEqual(table_ref.table_id, "p1_table_7")
        self.assertEqual(table_ref.page, 1)
        self.assertEqual(table_ref.bbox, (120.0, 150.0, 880.0, 500.0))
        self.assertEqual(table_ref.metadata["content_format"], "markdown")
        self.assertEqual(table_ref.metadata["crop_path"], "data/output\\sample_layout.pdf\\crops\\p1_table_7.png")
        self.assertEqual(table_ref.metadata["link_strategy"], "document_order")
        self.assertIn("| 카테고리 | 기능 ID |", table_ref.metadata["markdown"])


    def test_document_assembler_keeps_left_column_first_when_top_block_crosses_boundary(self):
        raw = {
            "layout_output": {
                "file_name": "boundary_case.pdf",
                "total_pages": 1,
                "pages": [
                    {
                        "page_num": 1,
                        "width": 3509,
                        "height": 2895,
                        "elements": [
                            {
                                "id": 1,
                                "type": "Text",
                                "bbox": [148.68, 148.62, 1524.73, 1094.45],
                                "confidence": 0.95,
                                "text": "Left intro block that slightly crosses the inferred boundary.",
                            },
                            {
                                "id": 2,
                                "type": "Text",
                                "bbox": [147.62, 1177.04, 907.44, 1475.25],
                                "confidence": 0.95,
                                "text": "Left lower block",
                            },
                            {
                                "id": 13,
                                "type": "Section-header",
                                "bbox": [1796.69, 149.93, 2126.31, 234.29],
                                "confidence": 0.95,
                                "text": "Right heading",
                            },
                            {
                                "id": 15,
                                "type": "Text",
                                "bbox": [1801.04, 1917.92, 3331.48, 2190.83],
                                "confidence": 0.95,
                                "text": "Right body block",
                            },
                        ],
                    }
                ],
            },
            "table_output": [],
        }

        result = DocumentAssembler().build_reading_order(raw)

        self.assertEqual(
            [element.id for element in result.ordered_elements[:4]],
            ["p1_text_1", "p1_text_2", "p1_heading_13", "p1_text_15"],
        )
        self.assertEqual(
            [(element.id, element.column_id, element.reading_order) for element in result.ordered_elements[:4]],
            [
                ("p1_text_1", 1, 1),
                ("p1_text_2", 1, 2),
                ("p1_heading_13", 2, 3),
                ("p1_text_15", 2, 4),
            ],
        )

    def test_document_assembler_uses_gutter_aware_boundary_for_wide_left_block(self):
        raw = {
            "layout_output": {
                "file_name": "gutter_boundary_case.pdf",
                "total_pages": 1,
                "pages": [
                    {
                        "page_num": 1,
                        "width": 3000,
                        "height": 2895,
                        "elements": [
                            {
                                "id": 1,
                                "type": "Text",
                                "bbox": [148.68, 148.62, 1524.73, 1094.45],
                                "confidence": 0.95,
                                "text": "Left intro block that should remain in column 1.",
                            },
                            {
                                "id": 2,
                                "type": "Text",
                                "bbox": [147.62, 1177.04, 907.44, 1475.25],
                                "confidence": 0.95,
                                "text": "Left lower block",
                            },
                            {
                                "id": 13,
                                "type": "Section-header",
                                "bbox": [1796.69, 149.93, 2126.31, 234.29],
                                "confidence": 0.95,
                                "text": "Right heading",
                            },
                            {
                                "id": 15,
                                "type": "Text",
                                "bbox": [1801.04, 1917.92, 3331.48, 2190.83],
                                "confidence": 0.95,
                                "text": "Right body block",
                            },
                        ],
                    }
                ],
            },
            "table_output": [],
        }

        result = DocumentAssembler().build_reading_order(raw)

        self.assertEqual(
            [(element.id, element.column_id, element.reading_order) for element in result.ordered_elements[:4]],
            [
                ("p1_text_1", 1, 1),
                ("p1_text_2", 1, 2),
                ("p1_heading_13", 2, 3),
                ("p1_text_15", 2, 4),
            ],
        )

    def test_document_assembler_orders_spanning_intro_before_two_columns(self):
        raw = build_layout_payload(
            [
                {
                    "id": 1,
                    "type": "Text",
                    "bbox": [120.0, 120.0, 2880.0, 360.0],
                    "confidence": 0.98,
                    "text": "Top intro spanning block",
                },
                {
                    "id": 2,
                    "type": "Text",
                    "bbox": [140.0, 520.0, 1100.0, 760.0],
                    "confidence": 0.98,
                    "text": "Left column first block",
                },
                {
                    "id": 3,
                    "type": "Text",
                    "bbox": [140.0, 860.0, 1100.0, 1120.0],
                    "confidence": 0.98,
                    "text": "Left column second block",
                },
                {
                    "id": 4,
                    "type": "Text",
                    "bbox": [1700.0, 540.0, 2680.0, 800.0],
                    "confidence": 0.98,
                    "text": "Right column first block",
                },
                {
                    "id": 5,
                    "type": "Text",
                    "bbox": [1700.0, 900.0, 2680.0, 1160.0],
                    "confidence": 0.98,
                    "text": "Right column second block",
                },
            ],
            file_name="top_spanning_then_two_columns.pdf",
        )

        result = DocumentAssembler().build_reading_order(raw)

        self.assertEqual(
            [(element.id, element.column_id, element.reading_order) for element in result.ordered_elements[:5]],
            [
                ("p1_text_1", 0, 1),
                ("p1_text_2", 1, 2),
                ("p1_text_3", 1, 3),
                ("p1_text_4", 2, 4),
                ("p1_text_5", 2, 5),
            ],
        )

    def test_document_assembler_orders_two_columns_before_spanning_outro(self):
        raw = build_layout_payload(
            [
                {
                    "id": 1,
                    "type": "Text",
                    "bbox": [140.0, 180.0, 1100.0, 430.0],
                    "confidence": 0.98,
                    "text": "Left column first block",
                },
                {
                    "id": 2,
                    "type": "Text",
                    "bbox": [140.0, 520.0, 1100.0, 770.0],
                    "confidence": 0.98,
                    "text": "Left column second block",
                },
                {
                    "id": 3,
                    "type": "Text",
                    "bbox": [1700.0, 200.0, 2680.0, 450.0],
                    "confidence": 0.98,
                    "text": "Right column first block",
                },
                {
                    "id": 4,
                    "type": "Text",
                    "bbox": [1700.0, 560.0, 2680.0, 810.0],
                    "confidence": 0.98,
                    "text": "Right column second block",
                },
                {
                    "id": 5,
                    "type": "Text",
                    "bbox": [130.0, 1040.0, 2860.0, 1300.0],
                    "confidence": 0.98,
                    "text": "Bottom outro spanning block",
                },
            ],
            file_name="two_columns_then_bottom_spanning.pdf",
        )

        result = DocumentAssembler().build_reading_order(raw)

        self.assertEqual(
            [(element.id, element.column_id, element.reading_order) for element in result.ordered_elements[:5]],
            [
                ("p1_text_1", 1, 1),
                ("p1_text_2", 1, 2),
                ("p1_text_3", 2, 3),
                ("p1_text_4", 2, 4),
                ("p1_text_5", 0, 5),
            ],
        )

    def test_document_assembler_splits_two_column_regions_around_midpage_spanning_block(self):
        raw = build_layout_payload(
            [
                {
                    "id": 1,
                    "type": "Text",
                    "bbox": [140.0, 180.0, 1100.0, 430.0],
                    "confidence": 0.98,
                    "text": "Top left block",
                },
                {
                    "id": 2,
                    "type": "Text",
                    "bbox": [1700.0, 200.0, 2680.0, 450.0],
                    "confidence": 0.98,
                    "text": "Top right block",
                },
                {
                    "id": 3,
                    "type": "Text",
                    "bbox": [130.0, 760.0, 2860.0, 980.0],
                    "confidence": 0.98,
                    "text": "Middle spanning separator",
                },
                {
                    "id": 4,
                    "type": "Text",
                    "bbox": [140.0, 1220.0, 1100.0, 1470.0],
                    "confidence": 0.98,
                    "text": "Bottom left block",
                },
                {
                    "id": 5,
                    "type": "Text",
                    "bbox": [1700.0, 1240.0, 2680.0, 1490.0],
                    "confidence": 0.98,
                    "text": "Bottom right block",
                },
            ],
            file_name="two_columns_spanning_two_columns.pdf",
        )

        result = DocumentAssembler().build_reading_order(raw)

        self.assertEqual(
            [(element.id, element.column_id, element.reading_order) for element in result.ordered_elements[:5]],
            [
                ("p1_text_1", 1, 1),
                ("p1_text_2", 2, 2),
                ("p1_text_3", 0, 3),
                ("p1_text_4", 1, 4),
                ("p1_text_5", 2, 5),
            ],
        )


if __name__ == "__main__":
    unittest.main()
