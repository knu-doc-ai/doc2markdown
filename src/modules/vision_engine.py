import os
import json
import dill
from typing import Dict, Any, List
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class VisionEngine:
    def __init__(self, output_base_dir: str = "data/output"):
        self.output_base_dir = output_base_dir
        # 문서 전용 YOLOv8 가중치 다운로드 및 로드
        print("🚀 [Vision] 문서 레이아웃 전용 YOLO 모델 로드 중...")
        
        try:
            # hantian 개발자의 YOLOv8x 기반 DocLayNet 가중치 다운로드
            model_path = hf_hub_download(
                repo_id="DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet", 
                filename="yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
            )
            self.model = YOLO(model_path)
            print("✅ [Vision] YOLOv8x 모델 로드 완료!")
        except Exception as e:
            print(f"🚨 [Vision] 모델 로드 실패: {e}")
            raise

    def process_document(self, ingestion_data: Dict) -> Dict[str, Any]:
        file_name = ingestion_data["file_name"]
        # 파일별로 독립된 결과 폴더 생성
        doc_output_dir = os.path.join(self.output_base_dir, file_name)
        crop_dir = os.path.join(doc_output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        print(f"👁️ [Vision] '{file_name}' 시각 분석 및 객체 추출 시작...")

        for page in ingestion_data["pages"]:
            img_path = page["image_path"]
            results = self.model(img_path, conf=0.35, imgsz=960)[0]
            
            page_elements = []
            
            # YOLO 탐지 결과 순회
            for i, box in enumerate(results.boxes):
                # 좌표 및 클래스 정보 추출
                coords = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                conf = float(box.conf[0])

                element = {
                    "id": i + 1,
                    "type": label,
                    "bbox": coords,
                    "confidence": conf,
                    "crop_path": None
                }

                # 'Table'이나 'Picture'등은 따로 잘라서 저장
                if label in ["Table", "Picture", "Figure", "Formula"]:
                    crop_name = f"p{page['page_num']}_{label.lower()}_{i+1}.png"
                    save_path = os.path.join(crop_dir, crop_name)
                    
                    with Image.open(img_path) as full_img:
                        cropped_img = full_img.crop(coords)
                        cropped_img.save(save_path)
                    
                    element["crop_path"] = save_path
                
                page_elements.append(element)

            # 읽기 순서 정렬 (Y좌표 우선, 그 다음 X좌표)
            page["elements"] = sorted(page_elements, key=lambda x: (x["bbox"][1], x["bbox"][0]))

        # 메타데이터 파일 생성
        meta_path = os.path.join(doc_output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(ingestion_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ [Vision] 분석 완료! 결과물 저장소: {doc_output_dir}")
        return ingestion_data


class LayoutAnalyzer(VisionEngine):
    """
    기존 VisionEngine을 파이프라인 단계 이름에 맞춰 감싼 어댑터입니다.
    """

    def analyze(self, ingestion_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process_document(ingestion_data)
