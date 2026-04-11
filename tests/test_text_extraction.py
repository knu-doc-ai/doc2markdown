import os
import json
import sys
import cv2  # 시각화를 위해 OpenCV 추가

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.text_extractor import TextExtractor

def draw_bboxes_from_json(final_output, output_base_dir="data/output"):
    """
    최종 JSON 데이터를 바탕으로 원본 이미지에 BBox를 그리고 저장하는 시각화 함수
    """
    file_name = final_output["file_name"]
    vis_dir = os.path.join(output_base_dir, file_name, "with_text_visualized_pages")
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
            
        save_path = os.path.join(vis_dir, f"page_{page_num}_with_text_visualized.png")
        cv2.imwrite(save_path, img_cv)
        print(f"   - [Page {page_num}] 시각화 완료: {len(elements)}개 객체 표시")
        
    print(f"✅ 모든 시각화 이미지 생성 완료!\n")

def test_text_extraction(pdf_path, json_path):
    if not os.path.exists(json_path):
        print("🚨 metadata.json이 없습니다. 비전 파이프라인을 먼저 돌려주세요.")
        return

    # 2. 비전 엔진 결과물 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 3. 텍스트 추출기 가동!
    # 🌟 수정 포인트: 초기화 시점이 아니라, 실행 시점에 pdf_path를 넘깁니다!
    extractor = TextExtractor()
    enriched_metadata = extractor.extract_text(metadata, pdf_path)

    # 4. 결과 저장 (텍스트가 추가된 최종 완성본)
    output_path = json_path.replace("metadata.json", "metadata_with_text.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_metadata, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 텍스트 추출 완료! 최종 데이터가 저장되었습니다: {output_path}")
    
    draw_bboxes_from_json(enriched_metadata)

if __name__ == "__main__":
    SAMPLE_PATH_LIST = [
        ("data/raw/calculator_srs_final.pdf", "data/output/calculator_srs_final.pdf/metadata.json"),
        ("data/raw/aiReadable.pdf", "data/output/aiReadable.pdf/metadata.json")
    ]
    
    for pdf_path, json_path in SAMPLE_PATH_LIST:
        test_text_extraction(pdf_path, json_path)