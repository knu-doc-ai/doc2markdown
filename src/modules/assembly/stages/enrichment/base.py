from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any

from modules.assembly.ir import AssemblyResult, AssemblyWarning
from modules.assembly.stages.enrichment.client import LLMClient, create_llm_client
from modules.assembly.stages.enrichment.config import LLMConfig


SEMANTIC_TASK = "semantic_enrichment"
CONTENT_TASK = "content_repair"
URL_PATTERN = re.compile(r"(?:https?://|www\.|[\w.-]+@[\w.-]+)", re.IGNORECASE)
MARKDOWN_TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")


class _BaseEnricher:
    def __init__(self, config: LLMConfig | None = None, client: LLMClient | None = None):
        self.config = config or LLMConfig.from_env()
        self.client = client

    def _client(self) -> LLMClient:
        if self.client is None:
            self.client = create_llm_client(self.config)
        return self.client

    def _llm_metadata(self, task: str, confidence: float | None) -> dict[str, Any]:
        return {
            "llm_enriched": True,
            "llm_model": self.config.model_id_for_task(task),
            "llm_task": task,
            "llm_confidence": confidence,
            "llm_enrichment_mode": self.config.mode,
        }

    @staticmethod
    def _merge_summary(metadata: dict[str, Any], key: str, summary: dict[str, Any]) -> dict[str, Any]:
        existing = dict(metadata.get("llm_enrichment") or {})
        existing[key] = summary
        return {**dict(metadata), "llm_enrichment": existing}

    @classmethod
    def _with_metadata_and_warnings(
        cls,
        result: AssemblyResult,
        key: str,
        summary: dict[str, Any],
        warnings: list[AssemblyWarning],
    ) -> AssemblyResult:
        document_metadata = cls._merge_summary(result.document.metadata, key, summary)
        return replace(
            result,
            document=replace(result.document, metadata=document_metadata),
            warnings=list(result.warnings) + warnings,
        )

    @staticmethod
    def _warning(
        code: str,
        message: str,
        *,
        element_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AssemblyWarning:
        return AssemblyWarning(
            code=code,
            message=message,
            level="warning",
            element_ids=element_ids or [],
            metadata=metadata or {},
        )


def _elapsed_seconds(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.2f}"


def _format_error(error: BaseException) -> str:
    message = str(error).strip().splitlines()
    if message:
        return message[0]
    return repr(error)


def _format_node_ids(node_ids: list[str], *, limit: int = 3) -> str:
    if len(node_ids) <= limit:
        return ", ".join(node_ids)
    return f"{', '.join(node_ids[:limit])}, ..."


def non_space_signature(text: str) -> str:
    return re.sub(r"\s+", "", text)
