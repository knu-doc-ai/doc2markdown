import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from dotenv import load_dotenv

from modules.ingestion import FilePreProcessor
from modules.vision_engine import LayoutAnalyzer
from modules.text_extractor import TextExtractor
from modules.assembly.service import DocumentAssembler
from modules.rendering.service import MarkdownRenderer

load_dotenv()


class DocumentToMarkdownPipeline:
    """
    문서 -> Markdown 전체 파이프라인.

    1. 문서 전처리
    2. 레이아웃 분석
    3. 텍스트 추출
    4. 표 Markdown 변환
    5. 문서 조립
    6. Markdown 렌더링 및 저장
    """

    def __init__(
        self,
        preprocessor: Optional[Any] = None,
        vision_engine: Optional[Any] = None,
        text_extractor: Optional[Any] = None,
        table_extractor: Optional[Any] = None,
        assembler: Optional[Any] = None,
        renderer: Optional[Any] = None,
        semantic_enricher: Optional[Any] = None,
        content_enricher: Optional[Any] = None,
        project_root: Optional[str] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
        self.output_base_dir = self.project_root / "data" / "output"
        self.temp_dir = self.project_root / "data" / "temp"
        self.table_extraction_mode = os.getenv("TABLE_EXTRACTION_MODE", "direct").strip().lower()
        self.table_extraction_enabled = _env_bool("TABLE_EXTRACTION_ENABLED", default=True)
        self.force_table_extraction_fallback = _env_bool("TABLE_EXTRACTION_FORCE_FALLBACK")

        self.preprocessor = preprocessor or FilePreProcessor(temp_dir=str(self.temp_dir))
        self.vision_engine = vision_engine or LayoutAnalyzer(output_base_dir=str(self.output_base_dir))
        self.text_extractor = text_extractor or TextExtractor()
        self.table_extractor = None if self._skips_table_extraction() else table_extractor or self._create_table_extractor()
        self.assembler = assembler or DocumentAssembler()
        self.renderer = renderer or MarkdownRenderer()
        self._print_table_config()
        self.semantic_enricher = semantic_enricher
        self.content_enricher = content_enricher

    def run_until_assembly(self, file_path: str) -> Dict[str, Any]:
        """조립 단계까지 실행 및 중간 산출물 저장."""
        resolved_file_path = Path(file_path).resolve()
        output_dir = self._get_output_dir(resolved_file_path)

        print(f"[Pipeline] 시작: {resolved_file_path}")

        print("[Pipeline] 1/6 입력 전처리")
        raw_pages = self.preprocessor.process(str(resolved_file_path))

        print("[Pipeline] 2/6 레이아웃 분석")
        layout_result = self.vision_engine.analyze(raw_pages)
        self._release_vision_gpu_memory()

        print("[Pipeline] 3/6 텍스트 추출")
        try:
            extracted_result = self.text_extractor.extract_text(layout_result, str(resolved_file_path))
        finally:
            if not self._uses_direct_table_extraction() and hasattr(self.text_extractor, "release_model"):
                self.text_extractor.release_model()
        self._save_json(output_dir / "metadata.json", extracted_result)

        print("[Pipeline] 4/6 표 Markdown 변환")
        table_result = self._build_table_results(extracted_result)
        if hasattr(self.table_extractor, "release_model"):
            self.table_extractor.release_model()
        self._save_json(output_dir / "table_results.json", table_result)

        print("[Pipeline] 5/6 Assembly IR build + LLM enrich")
        assembly_result = self._build_assembly_result(extracted_result, table_result)
        assembly_result_dict = self._serialize(assembly_result)
        self._save_json(output_dir / "assembly_result.json", assembly_result_dict)

        print("[Pipeline] Assembly IR 완료")
        return {
            "status": "success",
            "file_path": str(resolved_file_path),
            "output_dir": str(output_dir),
            "layout_result": extracted_result,
            "table_result": table_result,
            "assembly_result": assembly_result_dict,
        }

    def run(self, file_path: str, render_markdown: bool = True) -> Dict[str, Any]:
        """전체 파이프라인 실행 및 최종 Markdown 저장."""
        result = self.run_until_assembly(file_path=file_path)

        self._render_markdown(result, render_markdown=render_markdown)

        self._save_json(Path(result["output_dir"]) / "pipeline_result.json", result)
        self._cleanup_temp_images(self._collect_temp_image_paths(result["layout_result"]))
        print("[Pipeline] 완료")
        return result

    def _build_table_results(self, layout_result: Dict[str, Any]) -> list[Dict[str, Any]]:
        """레이아웃 결과의 표 crop 순회 및 Markdown 표 결과 생성."""
        table_results = []

        for page in layout_result.get("pages", []):
            page_num = page.get("page_num", 1)
            for element in page.get("elements", []):
                if element.get("type") != "Table":
                    continue

                crop_path = element.get("crop_path")
                if not crop_path:
                    continue

                table_id = self._build_table_id(page_num, element.get("id"))
                table_entry = {
                    "table_id": table_id,
                    "page": page_num,
                    "bbox": element.get("bbox"),
                    "crop_path": crop_path,
                    "source_block_ids": [table_id],
                }

                print(f"[Pipeline][Table] 추출 시작: {table_id}")
                if self._skips_table_extraction():
                    table_entry["extraction_error"] = self._table_extraction_skip_reason()
                    print(f"[Pipeline][Table] fallback 사용: {table_id}")
                    table_results.append(table_entry)
                    continue

                try:
                    markdown = self.table_extractor.extract_table(crop_path)
                    if markdown and not markdown.startswith("표 구조를 찾지 못했습니다."):
                        table_entry["markdown"] = markdown
                        print(f"[Pipeline][Table] 추출 완료: {table_id}")
                    else:
                        table_entry["extraction_error"] = markdown
                        print(f"[Pipeline][Table] 추출 실패: {table_id}")
                except Exception as error:
                    table_entry["extraction_error"] = str(error)
                    print(f"[Pipeline][Table] 추출 실패: {table_id}")

                table_results.append(table_entry)

        return table_results

    def _build_assembly_result(self, layout_result: Any, table_result: Any) -> Any:
        """assembly 모듈을 통해 선택적 LLM 보강을 포함한 Assembly IR을 만든다."""
        trace = self.assembler.build_from_outputs_with_trace(
            layout_result,
            table_result,
            semantic_enricher=self.semantic_enricher,
            content_enricher=self.content_enricher,
        )
        return trace.result

    def _render_markdown(self, result: Dict[str, Any], *, render_markdown: bool) -> None:
        if not render_markdown or self.renderer is None:
            result["markdown_result"] = None
            result["saved_paths"] = {}
            return

        print("[Pipeline] 6/6 Markdown rendering")
        output_dir = Path(result["output_dir"])
        markdown_result = self.renderer.render(result["assembly_result"])
        saved_paths = self.renderer.save(markdown_result, output_dir=output_dir)
        result["markdown_result"] = self._serialize(markdown_result)
        result["saved_paths"] = saved_paths

    def _build_table_id(self, page_num: int, raw_element_id: Any) -> str:
        """Assembly 어댑터와 같은 규칙의 table id 생성."""
        if isinstance(raw_element_id, int):
            return f"p{page_num}_table_{raw_element_id}"

        raw_id_str = str(raw_element_id).strip() if raw_element_id is not None else ""
        if raw_id_str.isdigit():
            return f"p{page_num}_table_{raw_id_str}"
        if raw_id_str:
            return raw_id_str
        return f"p{page_num}_table_unknown"

    def _get_output_dir(self, file_path: Path) -> Path:
        """프로젝트 루트 기준 출력 디렉터리 반환."""
        output_dir = self.output_base_dir / file_path.stem
        assembly_output_subdir = self._get_assembly_output_subdir()
        if assembly_output_subdir is not None:
            output_dir = output_dir / assembly_output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _get_assembly_output_subdir(self) -> Path | None:
        provider = getattr(self.assembler, "get_output_subdir", None)
        if not callable(provider):
            return None

        subdir = provider()
        if subdir is None:
            return None
        return Path(subdir)

    def _create_table_extractor(self):
        """환경변수 기준 표 추출 실행 방식 선택."""
        if self._uses_direct_table_extraction():
            table_module = import_module("modules.Table_to_markdown")
            return table_module.TableExtractor()

        table_module = import_module("modules.table_extractor")
        return table_module.TableExtractor()

    def _uses_direct_table_extraction(self) -> bool:
        """표 추출의 현재 프로세스 직접 실행 여부 반환."""
        return self.table_extraction_mode == "direct"

    def _skips_table_extraction(self) -> bool:
        return self.force_table_extraction_fallback or not self.table_extraction_enabled

    def _table_extraction_skip_reason(self) -> str:
        if self.force_table_extraction_fallback:
            return "TABLE_EXTRACTION_FORCE_FALLBACK=true"
        return "TABLE_EXTRACTION_ENABLED=false"

    def _print_table_config(self) -> None:
        print(f"[Pipeline][Config] TABLE_EXTRACTION_MODE={self.table_extraction_mode}")
        print(f"[Pipeline][Config] TABLE_EXTRACTION_ENABLED={self.table_extraction_enabled}")
        print(f"[Pipeline][Config] TABLE_EXTRACTION_FORCE_FALLBACK={self.force_table_extraction_fallback}")

    def _save_json(self, path: Path, payload: Any) -> None:
        """UTF-8 JSON 파일 저장."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            serialized_payload = self._strip_raw_fields(self._serialize(payload))
            json.dump(serialized_payload, file, ensure_ascii=False, indent=2)

    def _release_vision_gpu_memory(self) -> None:
        """레이아웃 분석 후 비전 모델 GPU 메모리 정리."""
        if not torch.cuda.is_available():
            return

        model = getattr(self.vision_engine, "model", None)
        if model is not None:
            try:
                model.to("cpu")
                print("[Pipeline] Vision GPU 메모리 해제 완료")
            except Exception:
                print("[Pipeline] Vision GPU 메모리 해제를 건너뜁니다.")

        torch.cuda.empty_cache()

    def _collect_temp_image_paths(self, ingestion_result: Dict[str, Any]) -> list[Path]:
        """이번 실행에서 생성된 temp 페이지 이미지 경로만 수집."""
        collected_paths = []

        for page in ingestion_result.get("pages", []):
            image_path = page.get("image_path")
            if not image_path:
                continue

            resolved_path = Path(image_path).resolve()
            try:
                resolved_path.relative_to(self.temp_dir.resolve())
            except ValueError:
                continue

            collected_paths.append(resolved_path)

        return collected_paths

    def _cleanup_temp_images(self, image_paths: list[Path]) -> None:
        """이번 실행에서 만든 temp 페이지 이미지만 삭제."""
        deleted_count = 0

        for image_path in image_paths:
            if not image_path.exists():
                continue

            try:
                image_path.unlink()
                deleted_count += 1
            except OSError:
                print(f"[Pipeline] Temp 이미지 삭제를 건너뜁니다: {image_path}")

        if deleted_count > 0:
            print(f"[Pipeline] Temp 이미지 {deleted_count}개 삭제 완료")

    def _serialize(self, value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value

    def _strip_raw_fields(self, value: Any) -> Any:
        """저장용 JSON의 과도한 raw 필드 제거."""
        if isinstance(value, dict):
            return {
                key: self._strip_raw_fields(item)
                for key, item in value.items()
                if key != "raw"
            }

        if isinstance(value, list):
            return [self._strip_raw_fields(item) for item in value]

        return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    sample_file = Path(__file__).resolve().parents[1] / "data" / "raw" / "calculator_srs_final.pdf"

    pipeline = DocumentToMarkdownPipeline()
    if sample_file.exists():
        pipeline.run(str(sample_file))
    else:
        print(f"[Pipeline] 샘플 파일을 찾을 수 없습니다: {sample_file}")
