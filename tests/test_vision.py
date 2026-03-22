# pip install matplotlib opencv-python ultralytics surya-ocr
# pip install huggingface_hub
# ingestion.py 실행 후 실행하여야 합니다 (PDF에서 이미지 추출 필요)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. OpenCV 엔진 (Baseline - 딥러닝 아님)
# ==========================================
def run_opencv_engine(image_path):
    """전통적인 이미지 처리 방식으로 텍스트/표 블록의 윤곽선을 찾습니다."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이진화 및 팽창(Dilation)으로 글자들을 뭉치게 만듦
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 너무 작은 노이즈 박스 제거
        if w > 20 and h > 10:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
    return cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# ==========================================
# 2. 문서 전용 YOLOv8 엔진 (Public 대체 모델 적용)
# ==========================================
def run_yolo_engine(image_path):
    """일반 사물이 아닌 '문서 레이아웃' 전용으로 파인튜닝된 YOLOv8 모델을 사용합니다."""
    try:
        import cv2
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        
        print("   (Hugging Face에서 공개된 문서 전용 YOLO 가중치를 다운로드 중...)")
        # hantian 개발자가 DocLayNet으로 학습시켜 전체 공개한 YOLOv8 Small 모델
        model_path = hf_hub_download(repo_id="hantian/yolo-doclaynet", filename="yolov8s-doclaynet.pt")
        
        # 다운받은 '문서 전용' 가중치로 YOLO 장전!
        model = YOLO(model_path)
        
        # 추론 실행
        results = model(image_path)
        
        # YOLO 렌더링 결과 이미지 가져오기
        res_plotted = results[0].plot()
        return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    except Exception as e:
        print(f"🚨 문서 전용 YOLO 엔진 실행 중 오류: {e}")
        from PIL import Image
        return Image.open(image_path)

# ==========================================
# 3. Surya 엔진 (최신 버전 API 적용 - Foundation 추가)
# ==========================================
def run_surya_engine(image_path):
    """Marker 내부에서 사용되는 최첨단 문서 레이아웃 분석기입니다."""
    try:
        from surya.layout import LayoutPredictor
        from surya.foundation import FoundationPredictor # ⭐ 파운데이션 모델 임포트 추가!
        
        img = Image.open(image_path).convert("RGB")
        
        print("   (Surya 딥러닝 파운데이션 모델 로딩 중...)")
        # ⭐ 1. 파운데이션(기반) 모델 먼저 생성
        foundation_predictor = FoundationPredictor()
        
        print("   (Surya 레이아웃 모델 로딩 중...)")
        # ⭐ 2. 레이아웃 예측기에 파운데이션 모델을 주입(Inject)하여 생성
        predictor = LayoutPredictor(foundation_predictor)
        
        # 레이아웃 추론
        layout_results = predictor([img])[0]
        
        # PIL을 이용해 이미지 위에 박스 그리기
        draw = ImageDraw.Draw(img)
        for bbox_info in layout_results.bboxes:
            box = bbox_info.bbox
            label = bbox_info.label 
            
            # 레이블에 따라 색상 다르게 지정
            color = "red" if label == "Table" else "blue" if label == "Title" else "green"
            
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], box[1] - 15), label, fill=color)
            
        return np.array(img)
        
    except Exception as e:
        print(f"🚨 Surya 엔진 실행 중 오류: {e}")
        return Image.open(image_path)

# ==========================================
# 통합 테스트 및 시각화 (1x3 Grid)
# ==========================================
def visualize_comparison(image_path):
    print("🚀 3가지 비전 엔진 비교 테스트를 시작합니다...\n")
    
    print("1/3. OpenCV 처리 중...")
    img_cv2 = run_opencv_engine(image_path)
    
    print("2/3. YOLOv8 처리 중...")
    img_yolo = run_yolo_engine(image_path)
    
    print("3/3. Surya (Marker) 처리 중...")
    img_surya = run_surya_engine(image_path)
    
    # Matplotlib으로 3개의 결과 나란히 띄우기
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle("Vision AI Model Bake-off: Document Layout Analysis", fontsize=20)
    
    axes[0].imshow(img_cv2)
    axes[0].set_title("1. OpenCV (Heuristic Baseline)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img_yolo)
    axes[1].set_title("2. YOLOv8 (Fast Object Detection)", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(img_surya)
    axes[2].set_title("3. Surya / Marker (Deep Document Understanding)", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 이전에 ingestion.py 테스트로 만들어둔 이미지를 사용합니다.
    # 만약 파일이 없다면, 방금 전에 만든 PDF에서 추출된 이미지 경로를 직접 넣어주세요.
    test_image_path = "data/temp/calculator_srs_final.pdf_page_3.png" 
    
    import os
    if os.path.exists(test_image_path):
        visualize_comparison(test_image_path)
    else:
        print(f"🚨 테스트용 이미지가 없습니다: {test_image_path}")
        print("ingestion.py를 먼저 실행하거나, 아무 이미지 파일 경로나 지정해주세요.")