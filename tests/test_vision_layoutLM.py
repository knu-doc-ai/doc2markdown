# pip install timm
# pip install transformers
# ingestion.py 실행 후 실행하여야 합니다 (PDF에서 이미지 추출 필요)

import warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run_layoutlm_engine(image_path):
    """
    Microsoft의 LayoutLMv3는 윈도우에서 구동이 매우 어려워,
    Hugging Face의 DETR(Transformers) 기반 문서 레이아웃 파이프라인으로 우회 테스트합니다.
    """
    print("⏳ Hugging Face 파이프라인 모델을 다운로드 중입니다... (최초 1회)")
    try:
        from transformers import pipeline
        
        # MS
        layout_analyzer = pipeline(
            "object-detection", 
            model="microsoft/table-transformer-detection" 
        )
        
        img = Image.open(image_path).convert("RGB")
        results = layout_analyzer(img)
        
        # 결과 시각화
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        for item in results:
            box = item['box']
            label = item['label']
            score = item['score']
            
            # 박스 그리기
            cv2.rectangle(img_cv, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 0, 255), 2)
            # 라벨 텍스트 달기
            cv2.putText(img_cv, f"{label} ({score:.2f})", (box['xmin'], box['ymin'] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"🚨 파이프라인 테스트 중 에러 발생: {e}")
        return Image.open(image_path)

if __name__ == "__main__":
    test_image = "data/temp/calculator_srs_final.pdf_page_3.png"
    result_img = run_layoutlm_engine(test_image)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(result_img)
    plt.title("Microsoft LayoutLM (Document Object Detection)")
    plt.axis('off')
    plt.show()