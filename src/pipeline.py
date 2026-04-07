from typing import Any, Dict, Optional

# 각 팀원이 개발할 핵심 모듈들을 불러옵니다.
# (아직 해당 파일들에 코드가 없어도 구조상 이렇게 import 합니다)
from modules.ingestion import FilePreProcessor
from modules.vision_engine import LayoutAnalyzer
from modules.text_extractor import TextExtractor
from modules.llm_core import ContentEnricher
from modules.assembler import DocumentAssembler
from modules.renderer import MarkdownRenderer

class DocumentToMarkdownPipeline:
    """
    LG전자 산학협력: 레이아웃 보존형 문서-Markdown 자동 변환 파이프라인

    1. Input Layer: 문서 전처리
    2. Visual Analysis Engine: 레이아웃 분석
    3. Text Extraction: BBox 기반 텍스트 정밀 추출 (PyMuPDF + VARCO OCR)
    4. Enrichment Layer: Assembly 입력 형식으로 정규화
    5. Document Assembly: 문서 구조 IR 조립
    6. Markdown Rendering: 최종 Markdown 생성
    """

    def __init__(
        self,
        preprocessor: Optional[Any] = FilePreProcessor(),
        vision_engine: Optional[Any] = LayoutAnalyzer(),
        text_extractor: Optional[Any] = TextExtractor(),
        content_enricher: Optional[Any] = ContentEnricher(),
        assembler: Optional[Any] = DocumentAssembler(),
        renderer: Optional[Any] = MarkdownRenderer(),
    ):
        # 1. Input Layer (문서 전처리)
        self.preprocessor = preprocessor

        # 2. Visual Analysis Engine (시각 구조 분석)
        self.vision_engine = vision_engine

        # 2.5 Text Extraction (텍스트 추출)
        self.text_extractor = text_extractor
        
        # 3. Enrichment Layer (Assembly 입력 형식 정규화)
        self.content_enricher = content_enricher

        # 4. Document Assembly (문서 구조 IR 조립)
        self.assembler = assembler

        # 5. Markdown Rendering (최종 Markdown 생성)
        self.renderer = renderer

    def run_until_assembly(
        self,
        file_path: str,
        table_results: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assembly IR 단계까지만 실행합니다.
        """
        print(f"🚀 파이프라인 시작: {file_path}")

        # =========================================================
        # 단계 1: Input Layer (문서 섭취 및 전처리)
        # =========================================================
        print("▶ [Step 1] 문서 전처리 중...")
        # PDF를 페이지별 이미지나 기초 데이터로 분할합니다. [cite: 29, 37]
        raw_pages = self.preprocessor.process(file_path)

        # =========================================================
        # 단계 2: Visual Analysis Engine (시각 구조 분석)
        # =========================================================
        print("▶ [Step 2] 레이아웃 분석 및 영역 분리 중...")
        # LayoutLM이나 OCR 등을 통해 제목, 본문, 표, 이미지를 분리합니다. [cite: 30, 36, 38, 41-46]
        layout_elements = self.vision_engine.analyze(raw_pages)

        # =========================================================
        # 단계 2.5: Text Extraction (텍스트 추출 및 이삭줍기)
        # =========================================================
        print("▶ [Step 2.5] BBox 기반 텍스트 추출 중...")
        # 🌟 파이프라인 실행 시점에 받은 file_path를 넘겨줌
        extracted_elements = self.text_extractor.extract_text(layout_elements, file_path)

        # =========================================================
        # 단계 3: Enrichment Layer (Assembly 입력 형식 정규화)
        # =========================================================
        print("▶ [Step 3] Assembly 입력 형식 정규화 중...")
        enrichment_result = self.content_enricher.enrich(
            layout_elements,
            table_results=table_results,
            config=config,
        )

        # =========================================================
        # 단계 4: Document Assembly (문서 구조 IR 조립)
        # =========================================================
        print("▶ [Step 4] 문서 구조 조립 중...")
        assembly_result = self.assembler.build(enrichment_result)
        print("Assembly IR 생성 완료!")

        return {
            "status": "success",
            "ingestion_result": raw_pages,
            "layout_result": layout_elements,
            "extracted_result": extracted_elements,
            "enrichment_result": enrichment_result,
            "assembly_result": self._serialize(assembly_result),
        }

    def run(
        self,
        file_path: str,
        table_results: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        render_markdown: bool = True,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인을 실행합니다.

        기본 동작은 Assembly IR까지 만든 뒤, 마지막에 Markdown Renderer를 통해
        최종 Markdown을 생성하는 것입니다.
        """
        result = self.run_until_assembly(
            file_path=file_path,
            table_results=table_results,
            config=config,
        )

        if render_markdown and self.renderer is not None:
            # =========================================================
            # 단계 5: Markdown Rendering (최종 Markdown 생성)
            # =========================================================
            print("▶ [Step 5] 최종 Markdown 생성 중...")
            markdown_result = self.renderer.render(result["assembly_result"])
            result["markdown_result"] = self._serialize(markdown_result)
        else:
            result["markdown_result"] = None

        result["status"] = "success"
        print("파이프라인 변환 완료!")
        return result

    def _serialize(self, value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value


if __name__ == "__main__":
    sample_file = "../data/raw/sample_lg_report.pdf"

    pipeline = DocumentToMarkdownPipeline()
    # result = pipeline.run(sample_file)
    # print(result)
