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


class AssemblyServiceContractTests(unittest.TestCase):
    def test_document_assembler_merges_layout_and_table_outputs(self):
        # 통합 payload에서 layout/table 출력을 함께 받아 하나의 seed result로 병합하는지 검증한다.
        raw = load_fixture("table_caption_note")

        result = DocumentAssembler().build(raw)

        self.assertEqual(result.metadata.stage, "adapter_seed")
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


if __name__ == "__main__":
    unittest.main()
