import os
import pprint

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.ingestion import FilePreProcessor  # 1단계
from src.modules.vision_engine import VisionEngine # 2단계

def test_vision_pipeline(file_path):
    print(f"\n{'='*50}")
    print(f"🔍 비전 엔진 단독 테스트 시작: {os.path.basename(file_path)}")
    print(f"{'='*50}\n")

    # 1. 1단계: 파일 전처리 (PDF -> Images + Raw Text)
    # ------------------------------------------------
    preprocessor = FilePreProcessor()
    print("STEP 1: 파일 전처리 중...")
    ingestion_data = preprocessor.process(file_path)
    
    # 2. 2단계: 비전 분석 (YOLOv8 -> BBox + Image Cropping)
    # ------------------------------------------------
    vision_engine = VisionEngine(output_base_dir="data/output")
    print("\nSTEP 2: 비전 분석 및 객체 추출 중...")
    # 이 과정에서 내부적으로 data/output/[파일명]/crops/ 폴더에 이미지들이 저장됨
    final_output = vision_engine.process_document(ingestion_data)

    # 3. 결과 검증 (JSON 구조 확인)
    # ------------------------------------------------
    print(f"\n{'='*50}")
    print("✅ 테스트 완료! 추출된 데이터 요약")
    print(f"{'='*50}")
    
    for page in final_output["pages"]:
        print(f"\n📄 [Page {page['page_num']}]")
        elements = page.get("elements", [])
        print(f"   - 탐지된 요소 개수: {len(elements)}개")
        
        # 표(Table)와 그림(Figure)이 잘 추출되었는지 확인
        tables = [e for e in elements if e["type"] == "Table"]
        figures = [e for e in elements if e["type"] in ["Picture", "Figure"]]
        
        print(f"   - 📊 추출된 표: {len(tables)}개")
        for t in tables:
            print(f"     └─ 저장 경로: {t['crop_path']}")
            
        print(f"   - 🖼️ 추출된 그림: {len(figures)}개")
        for f in figures:
            print(f"     └─ 저장 경로: {f['crop_path']}")

    print(f"\n📂 최종 JSON 메타데이터 위치: data/output/{final_output['file_name']}/metadata.json")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # 테스트하고 싶은 파일 경로를 넣어주세요 (샘플 PDF나 이미지)
    SAMPLE_PATH = "data/raw/calculator_srs_final.pdf" 
    
    if os.path.exists(SAMPLE_PATH):
        test_vision_pipeline(SAMPLE_PATH)
    else:
        print(f"🚨 파일을 찾을 수 없습니다: {SAMPLE_PATH}")
        print("data/raw/ 폴더에 테스트할 파일을 넣어주세요.")