import os
from typing import Dict, Any

# 각 팀원이 개발할 핵심 모듈들을 불러옵니다.
# (아직 해당 파일들에 코드가 없어도 구조상 이렇게 import 합니다)
from modules.ingestion import FilePreProcessor
from modules.vision_engine import LayoutAnalyzer
from modules.llm_core import MultiModalConverter
from modules.assembler import MarkdownAssembler

class DocumentToMarkdownPipeline:
    """
    LG전자 산학협력: 레이아웃 보존형 문서-Markdown 자동 변환 파이프라인
    제안서의 4단계 아키텍처를 순차적으로 실행하는 오케스트레이터입니다.
    """
    def __init__(self):
        # 1. Input Layer (문서 전처리)
        self.preprocessor = FilePreProcessor()
        
        # 2. Visual Analysis Engine (시각 구조 분석)
        self.vision_engine = LayoutAnalyzer()
        
        # 3. AI Agent Core (LLM 및 멀티모달 변환)
        self.llm_core = MultiModalConverter()
        
        # 4. Output Layer (마크다운 조립)
        self.assembler = MarkdownAssembler()

    def run(self, file_path: str) -> Dict[str, Any]:
        """
        단일 문서를 입력받아 최종 Markdown 결과물과 에셋을 반환합니다.
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
        # 단계 3: AI Agent Core (LLM 기반 멀티모달 변환)
        # =========================================================
        print("▶ [Step 3] AI 멀티모달 변환 (표 생성, 문맥 정제, Alt-text)...")
        # - 복잡한 표를 MD 표 문법으로 재구성 [cite: 48]
        # - 다단 텍스트를 논리적 흐름으로 정제 [cite: 50, 51]
        # - 이미지 Alt-text 자동 생성 [cite: 52]
        converted_elements = self.llm_core.convert(layout_elements)

        # =========================================================
        # 단계 4: Output Layer (최종 마크다운 조립)
        # =========================================================
        print("▶ [Step 4] 최종 마크다운 조립 중...")
        # 변환된 요소들을 원래 레이아웃 순서에 맞게 합칩니다. [cite: 32, 53]
        final_markdown, assets_path = self.assembler.build(converted_elements)

        print("✅ 파이프라인 변환 완료!")
        
        # 최종 결과물을 딕셔너리 형태로 UI 단에 반환합니다.
        return {
            "markdown_content": final_markdown,
            "assets_folder": assets_path,
            "status": "success"
        }

# 테스트용 코드 (직접 이 파일을 실행했을 때만 작동)
if __name__ == "__main__":
    # 테스트용 가짜 파일 경로
    sample_file = "../data/raw/sample_lg_report.pdf"
    
    # 파이프라인 객체 생성 및 실행
    pipeline = DocumentToMarkdownPipeline()
    # result = pipeline.run(sample_file) # 모듈들이 실제로 구현되면 주석 해제하여 테스트