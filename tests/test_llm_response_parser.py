import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.llm_response_parser import parse_content_repair, parse_content_repairs, parse_semantic_response


class SemanticResponseParserTests(unittest.TestCase):
    def test_top_level_list_is_treated_as_semantic_decisions(self):
        parsed = parse_semantic_response(
            [
                {
                    "id": " b1 ",
                    "kind": "Heading",
                    "heading_level": "2",
                    "confidence": "0.91",
                }
            ]
        )

        self.assertEqual(len(parsed.decisions), 1)
        self.assertEqual(parsed.decisions[0].id, "b1")
        self.assertEqual(parsed.decisions[0].kind, "heading")
        self.assertEqual(parsed.decisions[0].heading_level, 2)
        self.assertEqual(parsed.decisions[0].confidence, 0.91)

    def test_alias_keys_are_supported_for_semantic_and_caption(self):
        parsed = parse_semantic_response(
            {
                "blocks": [{"id": "cap1", "kind": "caption", "confidence": 0.8}],
                "caption_candidate_repairs": [
                    {"caption_id": "cap1", "target_id": "t1", "confidence": "0.84"}
                ],
            }
        )

        self.assertEqual([decision.id for decision in parsed.decisions], ["cap1"])
        self.assertEqual(len(parsed.caption_links), 1)
        self.assertEqual(parsed.caption_links[0].caption_id, "cap1")
        self.assertEqual(parsed.caption_links[0].target_id, "t1")
        self.assertEqual(parsed.caption_links[0].confidence, 0.84)

    def test_invalid_semantic_decisions_are_filtered(self):
        parsed = parse_semantic_response(
            {
                "semantic_decisions": [
                    {"id": "", "kind": "heading", "confidence": 0.9},
                    {"id": "b1", "kind": "table", "confidence": 0.9},
                    {"id": "b2", "kind": "heading", "confidence": 0.49},
                    {"id": "b3", "kind": "heading", "confidence": "bad"},
                    {"id": "b4", "kind": "note", "confidence": 0.5},
                ]
            }
        )

        self.assertEqual([decision.id for decision in parsed.decisions], ["b4"])

    def test_invalid_caption_links_are_filtered(self):
        parsed = parse_semantic_response(
            {
                "caption_links": [
                    {"caption_id": "", "target_id": "t1", "confidence": 0.9},
                    {"caption_id": "cap1", "target_id": "", "confidence": 0.9},
                    {"caption_id": "cap2", "target_id": "t2", "confidence": 0.49},
                    {"caption_id": "cap3", "target_id": "t3", "confidence": "0.5"},
                ]
            }
        )

        self.assertEqual([(link.caption_id, link.target_id) for link in parsed.caption_links], [("cap3", "t3")])

    def test_malformed_semantic_response_returns_empty_result(self):
        for response in (None, {}, {"semantic_decisions": {"not": "list"}}, ["not-a-dict"]):
            parsed = parse_semantic_response(response)
            self.assertEqual(parsed.decisions, [])
            self.assertEqual(parsed.caption_links, [])


class ContentRepairParserTests(unittest.TestCase):
    def test_content_repairs_returns_all_valid_items_for_batch_response(self):
        repairs = parse_content_repairs(
            {
                "repairs": [
                    {"node_id": "p1", "text": "첫 번째 문장", "confidence": 0.9},
                    {"node_id": "p2", "text": "두 번째 문장", "confidence": "0.8"},
                    {"node_id": "p3", "text": "낮은 신뢰도", "confidence": 0.4},
                ]
            }
        )

        self.assertEqual([repair.node_id for repair in repairs], ["p1", "p2"])

    def test_content_repair_prefers_matching_node_id(self):
        repair = parse_content_repair(
            {
                "repairs": [
                    {"node_id": "other", "text": "Other text", "confidence": 0.9},
                    {"node_id": "target", "text": "Fixed text", "confidence": "0.92"},
                ]
            },
            "target",
        )

        self.assertIsNotNone(repair)
        self.assertEqual(repair.node_id, "target")
        self.assertEqual(repair.text, "Fixed text")
        self.assertEqual(repair.confidence, 0.92)

    def test_content_repair_uses_single_valid_fallback(self):
        repair = parse_content_repair(
            {"content_repairs": [{"node_id": "other", "text": "Fallback text", "confidence": 0.9}]},
            "target",
        )

        self.assertIsNotNone(repair)
        self.assertEqual(repair.node_id, "other")
        self.assertEqual(repair.text, "Fallback text")

    def test_content_repair_filters_invalid_items(self):
        repair = parse_content_repair(
            {
                "items": [
                    {"node_id": "", "text": "Missing node", "confidence": 0.9},
                    {"node_id": "target", "text": "", "confidence": 0.9},
                    {"node_id": "target", "text": "Low confidence", "confidence": 0.49},
                ]
            },
            "target",
        )

        self.assertIsNone(repair)

    def test_malformed_content_response_returns_none(self):
        for response in (None, {}, {"repairs": {"not": "list"}}, ["not-a-dict"]):
            self.assertIsNone(parse_content_repair(response, "target"))


if __name__ == "__main__":
    unittest.main()
