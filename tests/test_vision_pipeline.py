import os
import sys
import pprint
import cv2  # 시각화를 위해 OpenCV 추가

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.ingestion import FilePreProcessor  # 1단계
from src.modules.vision_engine import VisionEngine # 2단계

def draw_bboxes_from_json(final_output, output_base_dir="data/output"):
    """
    최종 JSON 데이터를 바탕으로 원본 이미지에 BBox를 그리고 저장하는 시각화 함수
    """
    file_name = final_output["file_name"]
    vis_dir = os.path.join(output_base_dir, file_name, "visualized_pages")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\n🎨 Bounding Box 시각화 시작 (저장 경로: {vis_dir})")
    
    for page in final_output["pages"]:
        img_path = page["image_path"]
        page_num = page["page_num"]
        elements = page.get("elements", [])
        
        if not os.path.exists(img_path):
            print(f"   ⚠️ 이미지를 찾을 수 없습니다: {img_path}")
            continue
            
        # OpenCV로 이미지 읽기
        img_cv = cv2.imread(img_path)
        
        for el in elements:
            coords = [int(c) for c in el["bbox"]]
            label = el["type"]
            conf = el["confidence"]
            el_id = el["id"]
            
            # 라벨별로 색상 다르게 지정 (BGR 포맷)
            if label == "Table": 
                color = (0, 255, 0)      # 초록색
            elif label == "Picture" or label == "Figure": 
                color = (255, 0, 0)      # 파란색
            elif label == "Formula": 
                color = (0, 165, 255)    # 주황색
            elif label == "Section-header": 
                color = (200, 0, 200)    # 보라색
            else: 
                color = (150, 150, 150)  # 일반 텍스트 등은 회색
            
            # BBox 그리기 (선 두께 4)
            cv2.rectangle(img_cv, (coords[0], coords[1]), (coords[2], coords[3]), color, 4)
            
            # 텍스트 라벨 달기 (ID, 라벨명, 신뢰도)
            text = f"[{el_id}] {label} {conf:.2f}"
            # 고해상도 이미지에 맞게 폰트 크기(1.2)와 두께(3)를 크게 설정
            cv2.putText(img_cv, text, (coords[0], coords[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
        save_path = os.path.join(vis_dir, f"page_{page_num}_visualized.png")
        cv2.imwrite(save_path, img_cv)
        print(f"   - [Page {page_num}] 시각화 완료: {len(elements)}개 객체 표시")
        
    print(f"✅ 모든 시각화 이미지 생성 완료!\n")


def test_vision_pipeline(file_path):
    print(f"\n{'='*50}")
    print(f"🔍 비전 엔진 단독 테스트 시작: {os.path.basename(file_path)}")
    print(f"{'='*50}\n")

    # 1. 1단계: 파일 전처리
    preprocessor = FilePreProcessor()
    print("STEP 1: 파일 전처리 중...")
    ingestion_data = preprocessor.process(file_path)
    
    # 2. 2단계: 비전 분석
    vision_engine = VisionEngine(output_base_dir="data/output")
    print("\nSTEP 2: 비전 분석 및 객체 추출 중...")
    final_output = vision_engine.process_document(ingestion_data)

    # 3. 결과 검증 로깅
    print(f"\n{'='*50}")
    print("✅ 테스트 완료! 추출된 데이터 요약")
    print(f"{'='*50}")
    
    for page in final_output["pages"]:
        print(f"\n📄 [Page {page['page_num']}]")
        elements = page.get("elements", [])
        print(f"   - 탐지된 요소 개수: {len(elements)}개")
        
        tables = [e for e in elements if e["type"] == "Table"]
        figures = [e for e in elements if e["type"] in ["Picture", "Figure"]]
        
        print(f"   - 📊 추출된 표: {len(tables)}개")
        print(f"   - 🖼️ 추출된 그림: {len(figures)}개")

    print(f"\n📂 최종 JSON 메타데이터 위치: data/output/{final_output['file_name']}/metadata.json")
    
    # ⭐ 4. 눈으로 직접 확인하기 위한 시각화 함수 호출!
    draw_bboxes_from_json(final_output)


if __name__ == "__main__":
    SAMPLE_PATH = "data/raw/calculator_srs_final.pdf" 
    
    if os.path.exists(SAMPLE_PATH):
        test_vision_pipeline(SAMPLE_PATH)
    else:
        print(f"🚨 파일을 찾을 수 없습니다: {SAMPLE_PATH}")
        print("data/raw/ 폴더에 테스트할 파일을 넣어주세요.")