import unittest

from tests._helpers import FakeLLMClient

from modules.assembly.ir import (
    AssemblyElement,
    AssemblyMeta,
    AssemblyResult,
    AssembledDocument,
    PageStats,
    ParagraphGroup,
    TableRef,
)
from modules.assembly.stages.enrichment import ContentEnricher, LLMConfig, SemanticEnricher
from modules.assembly.stages.structure import StructureAssembler


def enabled_config(mode):
    return LLMConfig(mode=mode, model_id="fake-local-llm")


class EchoSemanticBatchClient:
    @property
    def model_id(self):
        return "fake-local-llm"

    def __init__(self):
        self.calls = []

    def generate_json(self, task, payload):
        self.calls.append((task, payload))
        return {
            "semantic_decisions": [
                {
                    "id": candidate["id"],
                    "kind": "heading",
                    "heading_level": 1,
                    "confidence": 0.8,
                }
                for candidate in payload.get("candidates", [])
            ],
            "caption_links": [],
        }


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

    def test_semantic_payload_filters_body_text_and_adds_hints(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(
                    id="h1",
                    page=1,
                    kind="text",
                    text="1.2 조사 범위",
                    bbox=(10.0, 10.0, 300.0, 28.0),
                ),
                AssemblyElement(
                    id="p1",
                    page=1,
                    kind="text",
                    text="본 문서는 주요 내용을 설명합니다.",
                    bbox=(10.0, 40.0, 300.0, 56.0),
                ),
                AssemblyElement(
                    id="cap1",
                    page=1,
                    kind="text",
                    text="표 1. 주요 항목",
                    bbox=(10.0, 70.0, 300.0, 86.0),
                ),
                AssemblyElement(
                    id="existing_heading",
                    page=1,
                    kind="heading",
                    text="개요",
                    bbox=(10.0, 100.0, 300.0, 124.0),
                ),
            ],
            page_stats=[PageStats(page=1, body_font_size=12.0)],
            metadata=AssemblyMeta(stage="normalized"),
        )

        payload = SemanticEnricher._build_semantic_payload(result)
        candidates_by_id = {candidate["id"]: candidate for candidate in payload["candidates"]}

        self.assertIn("h1", candidates_by_id)
        self.assertIn("cap1", candidates_by_id)
        self.assertIn("existing_heading", candidates_by_id)
        self.assertNotIn("p1", candidates_by_id)
        self.assertEqual(candidates_by_id["h1"]["semantic_hint"], "heading")
        self.assertEqual(candidates_by_id["h1"]["semantic_reason"], "numeric_heading_pattern")
        self.assertEqual(candidates_by_id["cap1"]["semantic_hint"], "caption")
        self.assertEqual(payload["candidate_stats"]["eligible_count"], 4)
        self.assertEqual(payload["candidate_stats"]["included_count"], 3)
        self.assertEqual(payload["candidate_stats"]["skipped_count"], 1)

    def test_semantic_summary_records_candidate_filter_counts(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(id="h1", page=1, kind="text", text="1.2 조사 범위"),
                AssemblyElement(id="p1", page=1, kind="text", text="본문 문장입니다."),
            ],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient({"semantic_enrichment": {"semantic_decisions": []}})

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)
        summary = enriched.document.metadata["llm_enrichment"]["semantic"]

        self.assertEqual(summary["eligible_candidate_count"], 2)
        self.assertEqual(summary["llm_candidate_count"], 1)
        self.assertEqual(summary["skipped_candidate_count"], 1)

    def test_semantic_requests_candidates_in_batches(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(id="h1", page=1, kind="heading", text="Overview"),
                AssemblyElement(id="h2", page=1, kind="heading", text="Scope"),
                AssemblyElement(id="h3", page=1, kind="heading", text="Risks"),
            ],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = EchoSemanticBatchClient()

        enriched = SemanticEnricher(
            config=LLMConfig(
                mode="semantic",
                model_id="fake-local-llm",
                semantic_batch_size=2,
            ),
            client=client,
        ).apply(result)
        summary = enriched.document.metadata["llm_enrichment"]["semantic"]

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(client.calls[0][1]["candidates"]), 2)
        self.assertEqual(len(client.calls[1][1]["candidates"]), 1)
        self.assertEqual(summary["semantic_batch_size"], 2)
        self.assertEqual(summary["batch_count"], 2)
        self.assertEqual(summary["decision_count"], 3)
        self.assertEqual(summary["failed_batch_count"], 0)

    def test_numeric_heading_rule_promotes_title_like_text_when_llm_omits_it(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(id="h1", page=1, kind="text", text="1.2 Scope and constraints"),
                AssemblyElement(id="p1", page=1, kind="text", text="1.0 version scope is excluded."),
            ],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient({"semantic_enrichment": {"semantic_decisions": []}})

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)
        summary = enriched.document.metadata["llm_enrichment"]["semantic"]
        elements_by_id = {element.id: element for element in enriched.ordered_elements}

        self.assertEqual(elements_by_id["h1"].kind, "heading")
        self.assertEqual(elements_by_id["h1"].metadata["llm_heading_level"], 2)
        self.assertEqual(elements_by_id["p1"].kind, "text")
        self.assertEqual(summary["rule_decision_count"], 1)
        self.assertEqual(summary["applied_decision_count"], 1)

        structured = StructureAssembler.apply(enriched)
        self.assertEqual(structured.document.sections[0].title, "1.2 Scope and constraints")
        self.assertEqual(structured.document.sections[0].level, 2)

    def test_top_level_numeric_rule_requires_child_or_geometry_support(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(id="list_like", page=1, kind="text", text="5. Save file"),
                AssemblyElement(id="h5", page=1, kind="text", text="5. Key Findings"),
                AssemblyElement(id="h51", page=1, kind="text", text="5.1 Survey Results"),
            ],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient({"semantic_enrichment": {"semantic_decisions": []}})

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)
        elements_by_id = {element.id: element for element in enriched.ordered_elements}

        self.assertEqual(elements_by_id["list_like"].kind, "text")
        self.assertEqual(elements_by_id["h5"].kind, "heading")
        self.assertEqual(elements_by_id["h5"].metadata["llm_heading_level"], 1)
        self.assertEqual(elements_by_id["h51"].kind, "heading")
        self.assertEqual(elements_by_id["h51"].metadata["llm_heading_level"], 2)

    def test_non_numeric_heading_rule_uses_strong_layout_signal(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(
                    id="h1",
                    page=1,
                    kind="text",
                    text="Executive Summary",
                    bbox=(10.0, 10.0, 300.0, 35.0),
                ),
                AssemblyElement(id="short", page=1, kind="text", text="Short phrase"),
            ],
            page_stats=[PageStats(page=1, body_font_size=12.0)],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient({"semantic_enrichment": {"semantic_decisions": []}})

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)
        elements_by_id = {element.id: element for element in enriched.ordered_elements}

        self.assertEqual(elements_by_id["h1"].kind, "heading")
        self.assertEqual(elements_by_id["h1"].metadata["llm_heading_level"], 1)
        self.assertEqual(elements_by_id["short"].kind, "text")

    def test_semantic_skips_llm_call_when_no_candidates_remain(self):
        result = AssemblyResult(
            ordered_elements=[
                AssemblyElement(id="p1", page=1, kind="text", text="본문 문장입니다."),
            ],
            document=AssembledDocument(),
            metadata=AssemblyMeta(stage="normalized"),
        )
        client = FakeLLMClient({"semantic_enrichment": AssertionError("호출되면 안 됨")})

        enriched = SemanticEnricher(config=enabled_config("semantic"), client=client).apply(result)
        summary = enriched.document.metadata["llm_enrichment"]["semantic"]

        self.assertEqual(client.calls, [])
        self.assertEqual(summary["llm_candidate_count"], 0)
        self.assertEqual(summary["skipped_candidate_count"], 1)


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
            config=LLMConfig(mode="content", model_id="fake-local-llm", content_min_chars=0),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, "한국어문장입니다")
        summary = enriched.document.metadata["llm_enrichment"]["content"]
        self.assertEqual(summary["discarded_repair_ids"], ["paragraph_1"])
        self.assertEqual(summary["discarded_repair_samples"][0]["node_id"], "paragraph_1")
        self.assertIn("llm_content_preservation_failed", [warning.code for warning in enriched.warnings])
        self.assertIn("repair_excerpt", enriched.warnings[-1].metadata)

    def test_korean_repair_is_discarded_when_spaces_are_overcollapsed(self):
        original = (
            "본 문서는 주요 내용을 설명하기 위해 작성되었습니다. "
            "검토 결과는 다음 단계에 반영됩니다."
        )
        collapsed = original.replace(" ", "")
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(
                        id="paragraph_1",
                        block_ids=["b1"],
                        text=original,
                    )
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_1", "text": collapsed}
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(mode="content", model_id="fake-local-llm", content_min_chars=0),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, original)
        summary = enriched.document.metadata["llm_enrichment"]["content"]
        self.assertEqual(summary["discarded_count"], 1)
        self.assertEqual(summary["discarded_repair_samples"][0]["discard_reason"], "removed_all_spaces")
        self.assertIn("llm_content_spacing_overcollapsed", [warning.code for warning in enriched.warnings])

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
        summary = enriched.document.metadata["llm_enrichment"]["content"]
        self.assertEqual(summary["applied_repair_ids"], ["paragraph_1"])
        self.assertEqual(summary["applied_repair_samples"][0]["source"], "rule")
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
                content_min_chars=0,
            ),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, "한국어 문장입니다")
        self.assertEqual(enriched.document.children[1].text, "두번째 문장입니다")
        summary = enriched.document.metadata["llm_enrichment"]["content"]
        self.assertEqual(summary["applied_repair_ids"], ["paragraph_1", "paragraph_2"])
        self.assertEqual(summary["applied_repair_samples"][0]["source"], "llm")
        self.assertEqual(summary["applied_repair_samples"][0]["original_excerpt"], "한국어문장입니다")
        self.assertEqual(summary["applied_repair_samples"][0]["repair_excerpt"], "한국어 문장입니다")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]["items"]), 2)
        self.assertEqual(client.calls[0][1]["items"][0]["non_space_signature"], "한국어문장입니다")
        self.assertEqual(client.calls[0][1]["items"][1]["non_space_signature"], "두번째문장입니다")

    def test_long_korean_ocr_spacing_repair_applies_with_single_item_batch(self):
        original = (
            "본 문서는 웹 기반 계산기 애플리케이션의 기 능적, 비기능적 요구사항을 정의하기 위해 "
            "작 성되었습니다. React.js 프레임워크를 기반으로 하며, PWA(Progressive Web App) "
            "표준을 준 수하여 다양한 디바이스에서 일관된 사용자 경 험을 제공합니다."
        )
        repaired = (
            "본 문서는 웹 기반 계산기 애플리케이션의 기능적, 비기능적 요구사항을 정의하기 위해 "
            "작성되었습니다. React.js 프레임워크를 기반으로 하며, PWA(Progressive Web App) "
            "표준을 준수하여 다양한 디바이스에서 일관된 사용자 경험을 제공합니다."
        )
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(
                        id="paragraph_2",
                        block_ids=["p1_text_8"],
                        text=original,
                    )
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_2", "text": repaired, "confidence": 0.93}
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(
                mode="content",
                model_id="fake-local-llm",
                content_batch_size=1,
                content_min_chars=20,
            ),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, repaired)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["items"][0]["node_id"], "paragraph_2")

    def test_content_summary_counts_parsed_matched_missing_and_unchanged_repairs(self):
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(id="paragraph_1", block_ids=["b1"], text="기 능적 요구사항"),
                    ParagraphGroup(id="paragraph_2", block_ids=["b2"], text="정상 문장입니다"),
                    ParagraphGroup(id="paragraph_3", block_ids=["b3"], text="작 성되었습니다"),
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_1", "text": "기능적 요구사항"},
                        {"node_id": "paragraph_2", "text": "정상 문장입니다", "confidence": 0.0},
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(
                mode="content",
                model_id="fake-local-llm",
                content_batch_size=8,
                content_min_chars=0,
            ),
            client=client,
        ).apply(result)
        summary = enriched.document.metadata["llm_enrichment"]["content"]

        self.assertEqual(enriched.document.children[0].text, "기능적 요구사항")
        self.assertEqual(enriched.document.children[1].text, "정상 문장입니다")
        self.assertEqual(enriched.document.children[2].text, "작 성되었습니다")
        self.assertIsNone(enriched.document.children[0].metadata["llm_confidence"])
        self.assertEqual(summary["parsed_count"], 2)
        self.assertEqual(summary["matched_count"], 2)
        self.assertEqual(summary["missing_repair_count"], 1)
        self.assertEqual(summary["missing_repair_ids"], ["paragraph_3"])
        self.assertEqual(summary["unchanged_count"], 1)
        self.assertEqual(summary["unchanged_sample_ids"], ["paragraph_2"])
        self.assertEqual(summary["applied_count"], 1)
        self.assertEqual(summary["applied_repair_ids"], ["paragraph_1"])
        self.assertEqual(summary["rule_repair_count"], 0)

    def test_content_repair_skips_text_below_min_chars_threshold(self):
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(
                        id="paragraph_1",
                        block_ids=["b1"],
                        text="짧은문장",
                    )
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_1", "text": "짧은 문장", "confidence": 0.9}
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(
                mode="content",
                model_id="fake-local-llm",
                content_min_chars=20,
            ),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, "짧은문장")
        self.assertEqual(client.calls, [])

    def test_short_korean_internal_spacing_is_requested_below_min_chars_threshold(self):
        result = AssemblyResult(
            document=AssembledDocument(
                children=[
                    ParagraphGroup(id="paragraph_1", block_ids=["b1"], text="기 능적"),
                    ParagraphGroup(id="paragraph_2", block_ids=["b2"], text="외 부"),
                ]
            ),
            metadata=AssemblyMeta(stage="structure_assembled"),
        )
        client = FakeLLMClient(
            {
                "content_repair": {
                    "repairs": [
                        {"node_id": "paragraph_1", "text": "기능적", "confidence": 0.9},
                        {"node_id": "paragraph_2", "text": "외부", "confidence": 0.9},
                    ]
                }
            }
        )

        enriched = ContentEnricher(
            config=LLMConfig(
                mode="content",
                model_id="fake-local-llm",
                content_batch_size=8,
                content_min_chars=20,
            ),
            client=client,
        ).apply(result)

        self.assertEqual(enriched.document.children[0].text, "기능적")
        self.assertEqual(enriched.document.children[1].text, "외부")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["items"][0]["spacing_suspect"], True)
        self.assertEqual(client.calls[0][1]["items"][1]["spacing_suspect"], True)

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
