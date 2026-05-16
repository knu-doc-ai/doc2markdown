import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.llm_core import LLMConfig, LocalTransformersLLMClient


class FakeTensor:
    def __init__(self, shape=(1, 3)):
        self.shape = shape
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeBatchEncoding:
    def __init__(self):
        self.input_ids = FakeTensor()
        self.attention_mask = FakeTensor()

    def items(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }.items()


class FakeGeneratedRow:
    def __getitem__(self, key):
        return ["generated"]


class FakeGeneratedOutput:
    def __getitem__(self, key):
        return FakeGeneratedRow()


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.batch_encoding = FakeBatchEncoding()

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, enable_thinking=False):
        return self.batch_encoding

    def decode(self, generated, skip_special_tokens):
        return '{"ok": true}'


class FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeGeneratedOutput()


class LocalTransformersLLMClientTests(unittest.TestCase):
    def test_generate_json_accepts_batch_encoding_inputs(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = LocalTransformersLLMClient(LLMConfig(model_id="fake-local-llm", max_new_tokens=7))
        client._tokenizer = tokenizer
        client._model = model

        response = client.generate_json("content_repair", {"items": []})

        self.assertEqual(response, {"ok": True})
        self.assertIs(model.generate_kwargs["input_ids"], tokenizer.batch_encoding.input_ids)
        self.assertIs(model.generate_kwargs["attention_mask"], tokenizer.batch_encoding.attention_mask)
        self.assertEqual(model.generate_kwargs["input_ids"].device, "cuda:0")
        self.assertEqual(model.generate_kwargs["attention_mask"].device, "cuda:0")
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 7)

    def test_content_task_uses_content_max_new_tokens(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = LocalTransformersLLMClient(
            LLMConfig(
                model_id="fake-local-llm",
                max_new_tokens=7,
                semantic_max_new_tokens=13,
                content_max_new_tokens=11,
            )
        )
        client._tokenizer = tokenizer
        client._model = model

        client.generate_json("content_repair", {"items": []})

        self.assertEqual(model.generate_kwargs["max_new_tokens"], 11)

    def test_semantic_task_uses_semantic_max_new_tokens(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = LocalTransformersLLMClient(
            LLMConfig(
                model_id="fake-local-llm",
                max_new_tokens=7,
                semantic_max_new_tokens=13,
                content_max_new_tokens=11,
            )
        )
        client._tokenizer = tokenizer
        client._model = model

        client.generate_json("semantic_enrichment", {"candidates": []})

        self.assertEqual(model.generate_kwargs["max_new_tokens"], 13)

    def test_non_content_task_uses_general_max_new_tokens(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = LocalTransformersLLMClient(
            LLMConfig(
                model_id="fake-local-llm",
                max_new_tokens=7,
                content_max_new_tokens=11,
            )
        )
        client._tokenizer = tokenizer
        client._model = model

        client.generate_json("semantic_enrichment", {"items": []})

        self.assertEqual(model.generate_kwargs["max_new_tokens"], 7)

    def test_semantic_prompt_mentions_heading_rules_and_examples(self):
        prompt = LocalTransformersLLMClient._build_prompt("semantic_enrichment", {"candidates": [], "objects": []})

        self.assertIn("semantic_enrichment", prompt)
        self.assertIn("semantic_decisions", prompt)
        self.assertIn("caption_links", prompt)
        self.assertIn("heading_level", prompt)
        self.assertIn("1.2", prompt)
        self.assertIn("Non-Functional Requirements", prompt)

    def test_content_prompt_mentions_ocr_spacing_examples(self):
        prompt = LocalTransformersLLMClient._build_prompt("content_repair", {"items": []})

        self.assertIn("OCR/PDF", prompt)
        self.assertIn("모든 입력 item", prompt)
        self.assertIn("비공백 문자 시퀀스", prompt)
        self.assertIn("기 능적 -> 기능적", prompt)
        self.assertIn("작 성되었습니다 -> 작성되었습니다", prompt)
        self.assertIn("준 수하여 -> 준수하여", prompt)
        self.assertIn("경 험 -> 경험", prompt)


if __name__ == "__main__":
    unittest.main()
