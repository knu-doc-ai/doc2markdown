import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    AssembledDocument,
    ParagraphGroup,
    TableRef,
)
from modules.llm_core import LLMConfig
from modules.llm_enrichment import ContentEnricher, SemanticEnricher


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    @property
    def model_id(self):
        return "fake-local-llm"

    def generate_json(self, task, payload):
        self.calls.append((task, payload))
        response = self.responses.get(task, {})
        if isinstance(response, Exception):
            raise response
        return response


def enabled_config(mode):
    return LLMConfig(mode=mode, model_id="fake-local-llm")


class SemanticEnricherTests(unittest.TestCase):
    def test_fake_client_promotes_text_block_to_heading(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(
                    id="b1",
                    page=1,
                    kind="text",
                    text="1. Introduction",
                    bbox=(10.0, 10.0, 300.0, 40.0),
                )
            ],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient(
            {
                "semantic_enrichment": {
                    "semantic_decisions": [
                        {"id": "b1", "kind": "heading", "heading_level": 1, "confidence": 0.91}
                    ]
                }
            }
        )

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)

        self.assertEqual(enriched.ordered_elements[0].kind, "heading")
        self.assertEqual(enriched.ordered_elements[0].metadata["llm_heading_level"], 1)
        self.assertTrue(enriched.ordered_elements[0].metadata["llm_enriched"])

    def test_fake_client_links_caption_candidate_to_table_ref(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(id="t1", page=1, kind="table", bbox=(10.0, 10.0, 300.0, 100.0)),
                AssemblyElement(
                    id="cap1",
                    page=1,
                    kind="text",
                    text="Table 1. Result summary",
                    bbox=(10.0, 110.0, 300.0, 130.0),
                ),
            ],
            document=AssembledDocument(
                table_refs=[TableRef(table_id="t1", page=1, bbox=(10.0, 10.0, 300.0, 100.0))]
            ),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient(
            {
                "semantic_enrichment": {
                    "semantic_decisions": [
                        {"id": "cap1", "kind": "caption", "heading_level": None, "confidence": 0.88}
                    ],
                    "caption_links": [
                        {"caption_id": "cap1", "target_id": "t1", "confidence": 0.84}
                    ],
                }
            }
        )

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)

        self.assertEqual(enriched.ordered_elements[1].kind, "caption")
        self.assertEqual(enriched.document.table_refs[0].caption_id, "cap1")
        self.assertTrue(enriched.document.table_refs[0].metadata["llm_enriched"])


class ContentEnricherTests(unittest.TestCase):
    def test_korean_repair_is_discarded_when_non_space_signature_changes(self):
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(
                        id="paragraph_1",
                        block_ids=["b1"],
                        text="한국어문장입니다",
                    )
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_1", "text": "한국어 문장입니다!", "confidence": 0.9}
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(mode="content", model_id="fake-local-llm"),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, "한국어문장입니다")
        self.assertIn("llm_content_preservation_failed", [warning.code for warning in enriched.warnings])

    def test_english_hyphenation_rule_repairs_split_word_without_model_call(self):
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(
                        id="paragraph_1",
                        block_ids=["b1"],
                        text="This paragraph contains infor- mation.",
                    )
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient({"content_repair": {}})

        enriched = ContentEnricher(config=enabled_config("content"), client=client).apply(result)

        self.assertEqual(enriched.document.children[0].text, "This paragraph contains information.")
        self.assertEqual(client.calls, [])

    def test_korean_repairs_are_requested_in_batches(self):
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(
                        id="paragraph_1",
                        block_ids=["b1"],
                        text="한국어문장입니다",
                    ),
                    ParagraphGroup(
                        id="paragraph_2",
                        block_ids=["b2"],
                        text="두번째문장입니다",
                    ),
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_1", "text": "한국어 문장입니다", "confidence": 0.9},
                        {"node_id": "paragraph_2", "text": "두번째 문장입니다", "confidence": 0.91},
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(
                mode="content",
                model_id="fake-local-llm",
                content_batch_size=8,
            ),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, "한국어 문장입니다")
        self.assertEqual(enriched.document.children[1].text, "두번째 문장입니다")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]["items"]), 2)

    def test_baseline_mode_is_noop(self):
        result = AssemblyResult(
            ordered_elements=[AssemblyElement(id="b1", page=1, kind="text", text="Title")],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient(
            {
                "semantic_enrichment": {
                    "semantic_decisions": [{"id": "b1", "kind": "heading", "confidence": 1.0}]
                }
            }
        )

        enriched = SemanticEnricher(config=LLMConfig(mode="baseline"), client=client).apply(result)

        self.assertIs(enriched, result)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
