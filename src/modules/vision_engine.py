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
    
    def _get_intersection_area(self, box1: List[float], box2: List[float]) -> float:
        """두 Bounding Box가 겹치는 영역의 넓이를 계산합니다."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        return (x_right - x_left) * (y_bottom - y_top)
    
    def _get_box_area(self, box: List[float]) -> float:
        """Bounding Box의 넓이를 계산합니다."""
        return (box[2] - box[0]) * (box[3] - box[1])
    
    def _postprocess_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        [세션 B: 후처리 필터]
        겹치는 박스 중 우선순위와 신뢰도에 따라 유효한 박스만 남깁니다.
        """
        to_remove = set()
        
        # 라벨별 우선순위 가중치 (표와 그림을 최우선으로 보호)
        priority = {
            "Table": 5, 
            "Picture": 4, 
            "Figure": 4, 
            "Formula": 3, 
            "Section-header": 2, 
            "Text": 1, 
            "List-item": 1, 
            "Page-header": 0, 
            "Page-footer": 0
        }

        for i in range(len(elements)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(elements)):
                if j in to_remove:
                    continue

                box1, box2 = elements[i]["bbox"], elements[j]["bbox"]
                inter_area = self._get_intersection_area(box1, box2)
                
                # 겹치는 영역이 없으면 패스
                if inter_area == 0:
                    continue
                
                area1, area2 = self._get_box_area(box1), self._get_box_area(box2)
                min_area = min(area1, area2)
                
                # 두 박스 중 더 작은 박스 면적의 70% 이상이 겹친다면 (포함 관계라면)
                if (inter_area / min_area) > 0.70:
                    type1, type2 = elements[i]["type"], elements[j]["type"]
                    p1, p2 = priority.get(type1, 0), priority.get(type2, 0)
                    
                    # 1순위: 우선순위가 높은 라벨 승리
                    if p1 > p2:
                        to_remove.add(j)
                    elif p2 > p1:
                        to_remove.add(i)
                        break # i가 죽었으므로 j와의 추가 비교 중단
                    else:
                        # 2순위: 라벨 우선순위가 같다면 신뢰도(Confidence) 높은 놈 승리
                        if elements[i]["confidence"] >= elements[j]["confidence"]:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break

        # 살아남은 객체들만 모아서 Y좌표 순서대로 최종 정렬
        valid_elements = [el for idx, el in enumerate(elements) if idx not in to_remove]
        valid_elements = sorted(valid_elements, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        
        for idx, el in enumerate(valid_elements):
            el["id"] = idx + 1
            
        return valid_elements
    
    def process_document(self, ingestion_data: Dict) -> Dict[str, Any]:
        file_name = ingestion_data["file_name"]
        doc_output_dir = os.path.join(self.output_base_dir, file_name)
        crop_dir = os.path.join(doc_output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        print(f"👁️ [Vision] '{file_name}' 시각 분석 및 객체 추출 시작...")

        for page in ingestion_data["pages"]:
            img_path = page["image_path"]
            results = self.model(img_path, conf=0.25, iou=0.45, imgsz=640)[0]
            
            raw_elements = []
            
            # YOLO 탐지 결과 순회
            for i, box in enumerate(results.boxes):
                # 좌표 및 클래스 정보 추출
                coords = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                conf = float(box.conf[0])

                element = {
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
                
                raw_elements.append(element)

                # 2. 후처리 필터 적용 (중복 제거 및 정렬)
            cleaned_elements = self._postprocess_elements(raw_elements)
            page["elements"] = cleaned_elements
            
            print(f"   - {page['page_num']}페이지: 탐지 {len(raw_elements)}개 -> 정제 후 {len(cleaned_elements)}개 요소 확보")

        # 메타데이터 저장
        meta_path = os.path.join(doc_output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(ingestion_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ [Vision] 분석 완료! 결과물 저장: {meta_path}")
        return ingestion_data

class LayoutAnalyzer(VisionEngine):
    """
    기존 VisionEngine을 파이프라인 단계 이름에 맞춰 감싼 어댑터입니다.
    """

    def analyze(self, ingestion_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process_document(ingestion_data)
