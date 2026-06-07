import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

from tests import _helpers  # noqa: F401

from modules.assembly.stages.enrichment import (
    LLMConfig,
    LocalTransformersLLMClient,
    OllamaLLMClient,
    OpenAICompatibleLLMClient,
    create_llm_client,
)
from modules.assembly.stages.enrichment.prompts import build_prompt


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


class FakeChatCompletions:
    def __init__(self):
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


class FakeOpenAICompatibleClient:
    def __init__(self):
        self.completions = FakeChatCompletions()
        self.chat = type("FakeChat", (), {"completions": self.completions})()


class FakeOllamaTransport:
    def __init__(self):
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append((url, body, timeout))
        return {"message": {"content": '{"ok": true}'}}


class FakeHTTPErrorBody(BytesIO):
    pass


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

    def test_transformers_client_uses_task_specific_cached_model(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        client = LocalTransformersLLMClient(
            LLMConfig(
                model_id="base-local-llm",
                semantic_model_id="semantic-local-llm",
                semantic_max_new_tokens=13,
            )
        )
        client._models["semantic-local-llm"] = (tokenizer, model)

        response = client.generate_json("semantic_enrichment", {"candidates": []})

        self.assertEqual(response, {"ok": True})
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 13)

    def test_semantic_prompt_mentions_heading_rules_and_examples(self):
        prompt = build_prompt("semantic_enrichment", {"candidates": [], "objects": []})

        self.assertIn("semantic_enrichment", prompt)
        self.assertIn("semantic_decisions", prompt)
        self.assertIn("caption_links", prompt)
        self.assertIn("heading_level", prompt)
        self.assertIn("1.2 조사 범위", prompt)
        self.assertIn("3.1 주요 결과 요약", prompt)
        self.assertIn("5. 결론 및 향후 과제", prompt)
        self.assertIn("본 문서는 주요 내용을 설명합니다.", prompt)

    def test_content_prompt_mentions_ocr_spacing_examples(self):
        prompt = build_prompt("content_repair", {"items": []})

        self.assertIn("OCR/PDF", prompt)
        self.assertIn("모든 입력 item", prompt)
        self.assertIn("비공백 문자 시퀀스", prompt)
        self.assertIn("non_space_signature", prompt)
        self.assertIn("confidence를 사용하지 않으므로 반환하지 않음", prompt)
        self.assertIn("닫는 괄호나 문장부호를 삭제하지 말 것", prompt)
        self.assertIn("문단 전체의 정상 띄어쓰기를 제거하지 말 것", prompt)
        self.assertIn("작 성되었습니다 -> 작성되었습니다", prompt)
        self.assertIn("경 험을 -> 경험을", prompt)
        self.assertIn("문 서를 -> 문서를", prompt)
        self.assertIn("내 용을 -> 내용을", prompt)
        self.assertIn("검 토합니다 -> 검토합니다", prompt)
        self.assertIn("자료 분석 -> 자료 분석", prompt)
        self.assertIn("검토 의견 -> 검토 의견", prompt)
        self.assertIn("고정 단어 목록에 의존하지 말고", prompt)


class OpenAICompatibleLLMClientTests(unittest.TestCase):
    def test_generate_json_uses_chat_completion_and_task_token_budget(self):
        fake_client = FakeOpenAICompatibleClient()
        config = LLMConfig(
            backend="openai-compatible",
            model_id="local-model",
            max_new_tokens=7,
            semantic_max_new_tokens=13,
        )
        client = OpenAICompatibleLLMClient(config, client=fake_client)

        response = client.generate_json("semantic_enrichment", {"candidates": []})

        self.assertEqual(response, {"ok": True})
        self.assertEqual(
            fake_client.completions.create_kwargs["model"],
            "local-model",
        )
        self.assertEqual(fake_client.completions.create_kwargs["max_tokens"], 13)
        self.assertEqual(fake_client.completions.create_kwargs["messages"][0]["role"], "system")
        self.assertEqual(fake_client.completions.create_kwargs["messages"][1]["role"], "user")

    def test_factory_selects_ollama_client_for_ollama_backend(self):
        config = LLMConfig(backend="ollama", model_id="hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0")

        self.assertIsInstance(create_llm_client(config), OllamaLLMClient)

    def test_config_defaults_to_transformers_without_explicit_backend(self):
        with patch.dict(
            "os.environ",
            {"LOCAL_LLM_SEMANTIC_MODEL_ID": "ggml-org/Qwen3-0.6B-GGUF:Q4_0"},
            clear=True,
        ):
            config = LLMConfig.from_env()

        self.assertEqual(config.backend, "transformers")
        self.assertEqual(config.model_id_for_task("semantic_enrichment"), "ggml-org/Qwen3-0.6B-GGUF:Q4_0")
        self.assertEqual(config.model_id_for_task("content_repair"), "Qwen/Qwen3-0.6B")
        self.assertEqual(config.request_timeout, 60.0)
        self.assertEqual(config.semantic_batch_size, 8)
        self.assertEqual(config.content_batch_size, 8)

    def test_config_defaults_ollama_batches_by_task(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_LLM_BACKEND": "ollama",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        self.assertEqual(config.backend, "ollama")
        self.assertEqual(config.model_id_for_task("semantic_enrichment"), "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0")
        self.assertEqual(config.model_id_for_task("content_repair"), "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0")
        self.assertEqual(config.request_timeout, 60.0)
        self.assertEqual(config.semantic_batch_size, 32)
        self.assertEqual(config.content_batch_size, 2)

    def test_config_respects_explicit_request_timeout(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_LLM_BACKEND": "ollama",
                "LOCAL_LLM_SEMANTIC_MODEL_ID": "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0",
                "LLM_REQUEST_TIMEOUT_SECONDS": "45",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        self.assertEqual(config.request_timeout, 45.0)

    def test_config_respects_explicit_batch_sizes_for_ollama(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_LLM_BACKEND": "ollama",
                "LOCAL_LLM_SEMANTIC_MODEL_ID": "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0",
                "LOCAL_LLM_CONTENT_MODEL_ID": "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0",
                "LLM_SEMANTIC_BATCH_SIZE": "4",
                "LLM_CONTENT_BATCH_SIZE": "5",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        self.assertEqual(config.backend, "ollama")
        self.assertEqual(config.semantic_batch_size, 4)
        self.assertEqual(config.content_batch_size, 5)

    def test_config_reads_task_specific_model_ids(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_LLM_SEMANTIC_MODEL_ID": "semantic-model",
                "LOCAL_LLM_CONTENT_MODEL_ID": "hf.co/local/content-GGUF:Q4_0",
            },
            clear=True,
        ):
            config = LLMConfig.from_env()

        self.assertEqual(config.model_id_for_task("semantic_enrichment"), "semantic-model")
        self.assertEqual(config.model_id_for_task("content_repair"), "hf.co/local/content-GGUF:Q4_0")
        self.assertEqual(config.model_id_for_task("other"), "semantic-model")
        self.assertEqual(config.backend, "transformers")

    def test_ollama_backend_rejects_non_local_base_url(self):
        config = LLMConfig(
            backend="ollama",
            model_id="hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0",
            api_base_url="https://api.openai.com/v1",
        )

        with self.assertRaisesRegex(RuntimeError, "only allows local base URLs"):
            OllamaLLMClient(config)

    def test_ollama_client_uses_native_chat_with_thinking_disabled_and_json_mode(self):
        transport = FakeOllamaTransport()
        config = LLMConfig(
            backend="ollama",
            model_id="ggml-org/Qwen3-0.6B-GGUF:Q4_0",
            api_base_url="http://127.0.0.1:11434/v1",
            max_new_tokens=7,
            semantic_max_new_tokens=13,
        )
        client = OllamaLLMClient(config, transport=transport)

        response = client.generate_json("semantic_enrichment", {"candidates": []})

        self.assertEqual(response, {"ok": True})
        url, body, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["model"], "hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0")
        self.assertFalse(body["think"])
        self.assertEqual(body["format"]["type"], "object")
        self.assertIn("semantic_decisions", body["format"]["properties"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["num_predict"], 256)

    def test_ollama_client_uses_content_model_override(self):
        transport = FakeOllamaTransport()
        config = LLMConfig(
            backend="ollama",
            model_id="ggml-org/Qwen3-0.6B-GGUF:Q4_0",
            content_model_id="ggml-org/Content-GGUF:Q4_0",
            api_base_url="http://127.0.0.1:11434/v1",
            max_new_tokens=7,
        )
        client = OllamaLLMClient(config, transport=transport)

        response = client.generate_json("content_repair", {"items": []})

        self.assertEqual(response, {"ok": True})
        url, body, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["model"], "hf.co/ggml-org/Content-GGUF:Q4_0")
        self.assertEqual(body["options"]["num_predict"], 512)
        self.assertNotIn("confidence", body["format"]["properties"]["repairs"]["items"]["properties"])
        self.assertEqual(body["format"]["properties"]["repairs"]["items"]["required"], ["node_id", "text"])
        self.assertIn("작 성되었습니다 -> 작성되었습니다", body["messages"][1]["content"])

    def test_ollama_client_reports_http_error_without_unreachable_message(self):
        config = LLMConfig(
            backend="ollama",
            model_id="ggml-org/Qwen3-0.6B-GGUF:Q4_0",
            api_base_url="http://127.0.0.1:11434/v1",
        )
        client = OllamaLLMClient(config)
        error = HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=FakeHTTPErrorBody(b'{"error":"model not found"}'),
        )

        with patch("modules.assembly.stages.enrichment.client.urlopen", side_effect=error):
            with self.assertRaisesRegex(Exception, "status=404.*model=hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0"):
                client.generate_json("content_repair", {"items": []})


if __name__ == "__main__":
    unittest.main()
