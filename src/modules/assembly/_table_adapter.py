from __future__ import annotations

"""table 출력 전용 Assembly 어댑터."""

from typing import Any, List, Optional, Tuple

from modules.assembly._adapter_base import AssemblyAdapterCommon
from modules.assembly.ir import AssemblyResult, AssemblyWarning, AssembledDocument, TableRef
from modules.assembly.types import AssemblySourceType


class TableAdapterMixin(AssemblyAdapterCommon):
    """table payload를 표준 IR로 바꾸는 로직 모음."""

    @classmethod
    def from_table_output(cls, raw: Any) -> AssemblyResult:
        """table 계열 출력을 TableRef 목록 중심 IR로 정규화한다."""
        if isinstance(raw, AssemblyResult):
            return raw

        if raw is None:
            return AssemblyResult(
                metadata=cls._build_adapter_metadata(
                    stage="adapter_seed",
                    adapter="table",
                    source="empty",
                ),
                raw=raw,
            )

        table_refs, warnings = cls._extract_table_refs(raw)

        return AssemblyResult(
            document=AssembledDocument(
                table_refs=table_refs,
                metadata={"adapter": "table"},
                raw=raw,
            ),
            warnings=warnings,
            metadata=cls._build_adapter_metadata(
                stage="adapter_seed",
                adapter="table",
                source=cls._infer_table_source(raw),
                table_count=len(table_refs),
            ),
            raw=raw,
        )

    @classmethod
    def _resolve_table_source(cls, raw: Any) -> Any:
        """복합 payload 안에서 table 후보를 찾는다."""
        if isinstance(raw, dict):
            nested = cls._pick_first(raw, cls.TABLE_CONTAINER_KEYS)
            if nested is not None:
                return nested

            document_payload = cls._pick_first(raw, cls.DOCUMENT_CONTAINER_KEYS)
            if cls._has_table_shape(document_payload):
                return document_payload

            if cls._has_table_shape(raw) and not cls._has_layout_shape(raw):
                return raw

        if cls._is_table_sequence(raw):
            return raw

        return None

    @classmethod
    def _infer_table_source(cls, raw: Any) -> AssemblySourceType:
        """table 입력이 어떤 형태로 들어왔는지 메타데이터용으로 추정한다."""
        if raw is None:
            return "empty"

        if cls._is_table_sequence(raw):
            return "direct_list"

        if isinstance(raw, dict):
            if cls._pick_first(raw, cls.TABLE_CONTAINER_KEYS) is not None:
                return "table_container"
            return "raw"

        return "raw"

    @classmethod
    def _extract_table_refs(
        cls,
        raw: Any,
    ) -> Tuple[List[TableRef], List[AssemblyWarning]]:
        table_refs: List[TableRef] = []
        warnings: List[AssemblyWarning] = []

        if isinstance(raw, list):
            for index, item in enumerate(raw, start=1):
                table_ref, item_warnings = cls._build_table_ref(
                    item,
                    fallback_page=None,
                    fallback_id=cls._make_table_fallback_id(index),
                )
                if table_ref is not None:
                    table_refs.append(table_ref)
                warnings.extend(item_warnings)
            return table_refs, warnings

        if not isinstance(raw, dict):
            return table_refs, warnings

        document_payload = cls._pick_first(raw, cls.DOCUMENT_CONTAINER_KEYS)
        if not isinstance(document_payload, dict):
            document_payload = None
        table_entries = cls._coerce_list(cls._pick_first(raw, cls.TABLE_LIST_KEYS))

        if not table_entries and document_payload is not None:
            table_entries = cls._coerce_list(cls._pick_first(document_payload, cls.TABLE_LIST_KEYS))

        if not table_entries and cls._looks_like_table_entry(raw):
            table_entries = [raw]

        for index, item in enumerate(table_entries, start=1):
            table_ref, item_warnings = cls._build_table_ref(
                item,
                fallback_page=None,
                fallback_id=cls._make_table_fallback_id(index),
            )
            if table_ref is not None:
                table_refs.append(table_ref)
            warnings.extend(item_warnings)

        return table_refs, warnings

    @classmethod
    def _build_table_ref(
        cls,
        raw: Any,
        fallback_page: Optional[int],
        fallback_id: str,
    ) -> Tuple[Optional[TableRef], List[AssemblyWarning]]:
        warnings: List[AssemblyWarning] = []

        if isinstance(raw, TableRef):
            return raw, warnings

        payload = raw if isinstance(raw, dict) else {"table_id": raw}

        table_id = cls._normalize_str(cls._pick_first(payload, cls.TABLE_ID_KEYS))
        if table_id is None:
            table_id = fallback_id
            warnings.append(
                AssemblyWarning(
                    code=cls.WARNING_TABLE_MISSING_ID,
                    message="table 결과에 table_id가 없어 임시 id를 부여했습니다.",
                    level="info",
                    page=fallback_page,
                    element_ids=[table_id],
                    raw=raw,
                )
            )

        page = cls._normalize_int(
            cls._pick_first(payload, cls.PAGE_NUMBER_KEYS),
            default=fallback_page,
        )
        if page is None:
            page = 1
            warnings.append(
                AssemblyWarning(
                    code=cls.WARNING_TABLE_MISSING_PAGE,
                    message="table 결과에 page 정보가 없어 1페이지로 가정했습니다.",
                    level="warning",
                    page=page,
                    element_ids=[table_id],
                    raw=raw,
                )
            )

        caption_id = cls._normalize_ref_id(cls._pick_first(payload, cls.CAPTION_KEYS))
        note_ids = cls._normalize_id_list(cls._pick_first(payload, cls.NOTE_KEYS))

        table_ref = TableRef(
            table_id=table_id,
            page=page,
            bbox=cls._normalize_bbox(cls._pick_first(payload, cls.BBOX_KEYS) or payload),
            caption_id=caption_id,
            note_ids=note_ids,
            source_block_ids=cls._extract_source_block_ids(payload),
            metadata=cls._extract_metadata(
                payload,
                {
                    *cls.TABLE_ID_KEYS,
                    *cls.PAGE_NUMBER_KEYS,
                    *cls.BBOX_KEYS,
                    *cls.CAPTION_KEYS,
                    *cls.NOTE_KEYS,
                    *cls.SOURCE_BLOCK_IDS_KEYS,
                },
            ),
            raw=raw,
        )
        return table_ref, warnings
