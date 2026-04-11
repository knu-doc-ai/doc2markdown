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
from modules.assembler import DocumentAssembler
from modules.renderer import MarkdownRenderer

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
        project_root: Optional[str] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]
        self.output_base_dir = self.project_root / "data" / "output"
        self.temp_dir = self.project_root / "data" / "temp"
        self.table_extraction_mode = os.getenv("TABLE_EXTRACTION_MODE", "direct").strip().lower()

        self.preprocessor = preprocessor or FilePreProcessor(temp_dir=str(self.temp_dir))
        self.vision_engine = vision_engine or LayoutAnalyzer(output_base_dir=str(self.output_base_dir))
        self.text_extractor = text_extractor or TextExtractor()
        self.table_extractor = table_extractor or self._create_table_extractor()
        self.assembler = assembler or DocumentAssembler()
        self.renderer = renderer or MarkdownRenderer()

    def run_until_assembly(self, file_path: str) -> Dict[str, Any]:
        """조립 단계까지 실행하고 중간 산출물도 함께 저장한다."""
        resolved_file_path = Path(file_path).resolve()
        output_dir = self._get_output_dir(resolved_file_path)

        print(f"▶ 파이프라인 시작: {resolved_file_path}")

        print("▶ [Step 1] 문서 전처리 중...")
        raw_pages = self.preprocessor.process(str(resolved_file_path))

        print("▶ [Step 2] 레이아웃 분석 중...")
        layout_result = self.vision_engine.analyze(raw_pages)
        self._release_vision_gpu_memory()

        print("▶ [Step 3] 텍스트 추출 중...")
        try:
            extracted_result = self.text_extractor.extract_text(layout_result, str(resolved_file_path))
        finally:
            if not self._uses_direct_table_extraction() and hasattr(self.text_extractor, "release_model"):
                self.text_extractor.release_model()
        self._save_json(output_dir / "metadata.json", extracted_result)

        print("▶ [Step 4] 표 Markdown 변환 중...")
        table_result = self._build_table_results(extracted_result)
        if self._uses_direct_table_extraction() and hasattr(self.text_extractor, "release_model"):
            self.text_extractor.release_model()
        self._save_json(output_dir / "table_results.json", table_result)

        print("▶ [Step 5] 문서 조립 중...")
        assembly_result = self.assembler.build_from_outputs(extracted_result, table_result)
        assembly_result_dict = self._serialize(assembly_result)
        self._save_json(output_dir / "assembly_result.json", assembly_result_dict)

        print("▶ Assembly IR 생성 완료!")
        return {
            "status": "success",
            "file_path": str(resolved_file_path),
            "output_dir": str(output_dir),
            "layout_result": extracted_result,
            "table_result": table_result,
            "assembly_result": assembly_result_dict,
        }

    def run(self, file_path: str, render_markdown: bool = True) -> Dict[str, Any]:
        """전체 파이프라인을 실행하고 최종 Markdown까지 저장한다."""
        result = self.run_until_assembly(file_path=file_path)

        if render_markdown and self.renderer is not None:
            print("▶ [Step 6] Markdown 렌더링 중...")
            output_dir = Path(result["output_dir"])
            markdown_result = self.renderer.render(result["assembly_result"])
            saved_paths = self.renderer.save(markdown_result, output_dir=output_dir)
            result["markdown_result"] = self._serialize(markdown_result)
            result["saved_paths"] = saved_paths
        else:
            result["markdown_result"] = None
            result["saved_paths"] = {}

        self._save_json(Path(result["output_dir"]) / "pipeline_result.json", result)
        self._cleanup_temp_images(self._collect_temp_image_paths(result["layout_result"]))
        print("▶ 파이프라인 변환 완료!")
        return result

    def _build_table_results(self, layout_result: Dict[str, Any]) -> list[Dict[str, Any]]:
        """레이아웃 결과의 표 crop을 순회하며 Markdown 표 결과를 만든다."""
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

                print(f"[Pipeline] 표 추출 시작: {table_id}")
                try:
                    markdown = self.table_extractor.extract_table(crop_path)
                    if markdown and not markdown.startswith("표 구조를 찾지 못했습니다."):
                        table_entry["markdown"] = markdown
                        print(f"[Pipeline] 표 추출 완료: {table_id}")
                    else:
                        table_entry["extraction_error"] = markdown
                        print(f"[Pipeline] 표 추출 실패: {table_id}")
                except Exception as error:
                    table_entry["extraction_error"] = str(error)
                    print(f"[Pipeline] 표 추출 실패: {table_id}")

                table_results.append(table_entry)

        return table_results

    def _build_table_id(self, page_num: int, raw_element_id: Any) -> str:
        """Assembly 어댑터와 같은 규칙의 table id를 만든다."""
        if isinstance(raw_element_id, int):
            return f"p{page_num}_table_{raw_element_id}"

        raw_id_str = str(raw_element_id).strip() if raw_element_id is not None else ""
        if raw_id_str.isdigit():
            return f"p{page_num}_table_{raw_id_str}"
        if raw_id_str:
            return raw_id_str
        return f"p{page_num}_table_unknown"

    def _get_output_dir(self, file_path: Path) -> Path:
        """프로젝트 루트 기준 출력 디렉터리를 반환한다."""
        output_dir = self.output_base_dir / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _create_table_extractor(self):
        """환경변수에 따라 표 추출 실행 방식을 고른다."""
        if self._uses_direct_table_extraction():
            print("[Pipeline] TABLE_EXTRACTION_MODE=direct")
            table_module = import_module("modules.Table_to_markdown")
            return table_module.TableExtractor()

        print("[Pipeline] TABLE_EXTRACTION_MODE=worker")
        table_module = import_module("modules.table_extractor")
        return table_module.TableExtractor()

    def _uses_direct_table_extraction(self) -> bool:
        """표 추출을 현재 프로세스에서 직접 실행할지 여부를 반환한다."""
        return self.table_extraction_mode == "direct"

    def _save_json(self, path: Path, payload: Any) -> None:
        """UTF-8 JSON 파일로 저장한다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            serialized_payload = self._strip_raw_fields(self._serialize(payload))
            json.dump(serialized_payload, file, ensure_ascii=False, indent=2)

    def _release_vision_gpu_memory(self) -> None:
        """레이아웃 분석이 끝난 뒤 비전 모델 GPU 메모리를 정리한다."""
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
        """이번 실행에서 생성된 temp 페이지 이미지 경로만 수집한다."""
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
        """이번 실행에서 만든 temp 페이지 이미지만 삭제한다."""
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
        """저장용 JSON에서는 과도하게 커지는 raw 필드를 제거한다."""
        if isinstance(value, dict):
            return {
                key: self._strip_raw_fields(item)
                for key, item in value.items()
                if key != "raw"
            }

        if isinstance(value, list):
            return [self._strip_raw_fields(item) for item in value]

        return value


if __name__ == "__main__":
    sample_file = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_lg_report.pdf"

    pipeline = DocumentToMarkdownPipeline()
    if sample_file.exists():
        pipeline.run(str(sample_file))
    else:
        print(f"샘플 파일을 찾을 수 없습니다: {sample_file}")
