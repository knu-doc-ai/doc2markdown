import argparse
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_INPUT_PDF = PROJECT_ROOT / "data" / "raw" / "2단 문서 text.pdf"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from _assembly_debug_utils import (
    build_stage_results_from_outputs,
    save_json,
    save_stage_results,
    serialize,
)


load_dotenv()


class LayoutOrderDebugPipeline:
    def __init__(
        self,
        preprocessor: Optional[Any] = None,
        vision_engine: Optional[Any] = None,
        text_extractor: Optional[Any] = None,
        table_extractor: Optional[Any] = None,
        renderer: Optional[Any] = None,
        project_root: Optional[str] = None,
    ):
        try:
            from modules.ingestion import FilePreProcessor
            from modules.renderer import MarkdownRenderer
            from modules.text_extractor import TextExtractor
            from modules.vision_engine import LayoutAnalyzer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Required dependency is missing for the debug pipeline: "
                f"{error.name}. Run this script in the project environment with the full requirements installed."
            ) from error

        self.project_root = Path(project_root).resolve() if project_root else PROJECT_ROOT
        self.output_base_dir = self.project_root / "data" / "output"
        self.temp_dir = self.project_root / "data" / "temp"
        self.table_extraction_mode = os.getenv("TABLE_EXTRACTION_MODE", "direct").strip().lower()

        self.preprocessor = preprocessor or FilePreProcessor(temp_dir=str(self.temp_dir))
        self.vision_engine = vision_engine or LayoutAnalyzer(output_base_dir=str(self.output_base_dir))
        self.text_extractor = text_extractor or TextExtractor()
        self.table_extractor = table_extractor or self._create_table_extractor()
        self.renderer = renderer or MarkdownRenderer()

    def run(
        self,
        file_path: str,
        output_dir: str | None = None,
        render_markdown: bool = True,
    ) -> Dict[str, Any]:
        resolved_file_path = Path(file_path).resolve()
        resolved_output_dir = self._get_output_dir(resolved_file_path, output_dir)
        debug_steps_dir = resolved_output_dir / "debug" / "pipeline_steps"

        print(f"[DebugPipeline] start: {resolved_file_path}")

        print("[DebugPipeline] Step 1/6 ingestion")
        raw_pages = self.preprocessor.process(str(resolved_file_path))
        save_json(debug_steps_dir / "01_ingestion.json", raw_pages)

        print("[DebugPipeline] Step 2/6 layout analysis")
        layout_result = self.vision_engine.analyze(raw_pages)
        save_json(debug_steps_dir / "02_layout_analysis.json", layout_result)
        self._release_vision_gpu_memory()

        print("[DebugPipeline] Step 3/6 text extraction")
        try:
            extracted_result = self.text_extractor.extract_text(layout_result, str(resolved_file_path))
        finally:
            if not self._uses_direct_table_extraction() and hasattr(self.text_extractor, "release_model"):
                self.text_extractor.release_model()

        save_json(resolved_output_dir / "metadata.json", extracted_result)
        save_json(debug_steps_dir / "03_text_extraction.json", extracted_result)

        print("[DebugPipeline] Step 4/6 table extraction")
        table_result = self._build_table_results(extracted_result)
        if self._uses_direct_table_extraction() and hasattr(self.text_extractor, "release_model"):
            self.text_extractor.release_model()

        save_json(resolved_output_dir / "table_results.json", table_result)
        save_json(debug_steps_dir / "04_table_results.json", table_result)

        print("[DebugPipeline] Step 5/6 assembly")
        stage_results = build_stage_results_from_outputs(extracted_result, table_result)
        assembly_stage_paths = save_stage_results(
            resolved_output_dir / "debug" / "assembly_stages",
            stage_results,
        )
        assembly_result = stage_results["validated"]
        save_json(resolved_output_dir / "assembly_result.json", assembly_result)

        pipeline_result: Dict[str, Any] = {
            "status": "success",
            "file_path": str(resolved_file_path),
            "output_dir": str(resolved_output_dir),
            "debug_step_paths": {
                "ingestion": str(debug_steps_dir / "01_ingestion.json"),
                "layout_analysis": str(debug_steps_dir / "02_layout_analysis.json"),
                "text_extraction": str(debug_steps_dir / "03_text_extraction.json"),
                "table_results": str(debug_steps_dir / "04_table_results.json"),
            },
            "assembly_stage_paths": assembly_stage_paths,
            "layout_result": extracted_result,
            "table_result": table_result,
            "assembly_result": serialize(assembly_result),
        }

        if render_markdown and self.renderer is not None:
            print("[DebugPipeline] Step 6/6 markdown rendering")
            markdown_result = self.renderer.render(assembly_result)
            saved_paths = self.renderer.save(markdown_result, output_dir=resolved_output_dir)
            pipeline_result["markdown_result"] = serialize(markdown_result)
            pipeline_result["saved_paths"] = saved_paths
            save_json(debug_steps_dir / "05_markdown_render.json", markdown_result)
        else:
            pipeline_result["markdown_result"] = None
            pipeline_result["saved_paths"] = {}

        save_json(resolved_output_dir / "pipeline_result.json", pipeline_result)
        self._cleanup_temp_images(self._collect_temp_image_paths(layout_result))
        print(f"[DebugPipeline] done: {resolved_output_dir}")
        return pipeline_result

    def _build_table_results(self, layout_result: Dict[str, Any]) -> list[Dict[str, Any]]:
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

                print(f"[DebugPipeline] table extraction: {table_id}")
                try:
                    markdown = self.table_extractor.extract_table(crop_path)
                    if markdown and not markdown.startswith("⚠구조를 찾지 못했습니다"):
                        table_entry["markdown"] = markdown
                    else:
                        table_entry["extraction_error"] = markdown
                except Exception as error:
                    table_entry["extraction_error"] = str(error)

                table_results.append(table_entry)

        return table_results

    def _build_table_id(self, page_num: int, raw_element_id: Any) -> str:
        if isinstance(raw_element_id, int):
            return f"p{page_num}_table_{raw_element_id}"

        raw_id_str = str(raw_element_id).strip() if raw_element_id is not None else ""
        if raw_id_str.isdigit():
            return f"p{page_num}_table_{raw_id_str}"
        if raw_id_str:
            return raw_id_str
        return f"p{page_num}_table_unknown"

    def _get_output_dir(self, file_path: Path, raw_output_dir: str | None) -> Path:
        if raw_output_dir:
            output_dir = Path(raw_output_dir)
            if not output_dir.is_absolute():
                output_dir = (self.project_root / output_dir).resolve()
        else:
            output_dir = self.output_base_dir / file_path.stem

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _create_table_extractor(self):
        if self._uses_direct_table_extraction():
            print("[DebugPipeline] TABLE_EXTRACTION_MODE=direct")
            table_module = import_module("modules.Table_to_markdown")
            return table_module.TableExtractor()

        print("[DebugPipeline] TABLE_EXTRACTION_MODE=worker")
        table_module = import_module("modules.table_extractor")
        return table_module.TableExtractor()

    def _uses_direct_table_extraction(self) -> bool:
        return self.table_extraction_mode == "direct"

    def _release_vision_gpu_memory(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            return

        if not torch.cuda.is_available():
            return

        model = getattr(self.vision_engine, "model", None)
        if model is not None:
            try:
                model.to("cpu")
                print("[DebugPipeline] vision GPU memory released")
            except Exception:
                print("[DebugPipeline] skipped vision GPU release")

        torch.cuda.empty_cache()

    def _collect_temp_image_paths(self, ingestion_result: Dict[str, Any]) -> list[Path]:
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
        for image_path in image_paths:
            if not image_path.exists():
                continue

            try:
                image_path.unlink()
            except OSError:
                print(f"[DebugPipeline] skipped temp cleanup: {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the document pipeline and save debug artifacts for each pipeline and assembly stage."
    )
    parser.add_argument(
        "input_pdf",
        nargs="?",
        default=str(DEFAULT_INPUT_PDF),
        help="Path to the input PDF. Defaults to data/raw/2단 문서 text.pdf.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save outputs. Defaults to data/output/<stem>.",
    )
    parser.add_argument(
        "--no-render-markdown",
        action="store_true",
        help="Skip markdown rendering and only save pipeline/assembly debug artifacts.",
    )
    args = parser.parse_args()

    input_pdf = Path(args.input_pdf)
    if not input_pdf.is_absolute():
        input_pdf = (PROJECT_ROOT / input_pdf).resolve()

    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    try:
        pipeline = LayoutOrderDebugPipeline()
        pipeline.run(
            file_path=str(input_pdf),
            output_dir=args.output_dir,
            render_markdown=not args.no_render_markdown,
        )
    except RuntimeError as error:
        print(f"[DebugPipeline] {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
