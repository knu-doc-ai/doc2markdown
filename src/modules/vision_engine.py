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
    
    def _sort_reading_order(self, elements: List[Dict]) -> List[Dict]:
        """
        중앙선 교차(Midline Spanning)를 이용한 다단 읽기 순서 정렬
        """
        if not elements:
            return []
        
        # 1. 페이지의 X축 정중앙(Midline) 계산
        x_coords = [el["bbox"][0] for el in elements] + [el["bbox"][2] for el in elements]
        midline = (min(x_coords) + max(x_coords)) / 2
        
        # 2. 우선 Y좌표 기준으로 전체 1차 정렬 (위에서 아래로 훑기 위함)
        elements = sorted(elements, key=lambda x: x["bbox"][1])
        
        blocks = []
        current_block = {"left": [], "right": [], "full": []}
        
        for el in elements:
            xmin, ymin, xmax, ymax = el["bbox"]
            
            # 💡 핵심 로직: 박스의 시작점은 중앙선 왼쪽이고, 끝점은 오른쪽이라면? -> 선을 걸쳤다! (통짜/중앙정렬)
            if xmin < midline and xmax > midline:
                # 기존에 읽던 좌/우 단락이 있다면 블록을 마감하고 저장
                if current_block["left"] or current_block["right"]:
                    blocks.append(current_block)
                    current_block = {"left": [], "right": [], "full": []}
                
                # 통짜 요소(제목, 큰 표 등)는 단독 블록으로 바로 저장
                current_block["full"].append(el)
                blocks.append(current_block)
                current_block = {"left": [], "right": [], "full": []}
                
            # 중앙선 왼쪽에만 쏙 들어가 있다면 -> 좌측 단
            elif xmax <= midline:
                current_block["left"].append(el)
                
            # 중앙선 오른쪽에만 쏙 들어가 있다면 -> 우측 단
            else:
                current_block["right"].append(el)
                
        # 마지막으로 남은 찌꺼기 블록 털어내기
        if current_block["left"] or current_block["right"] or current_block["full"]:
            blocks.append(current_block)
            
        # 3. 인간의 읽기 순서대로 최종 조립 (블록 순서대로 -> 좌측 위아래 다 읽고 -> 우측 위아래)
        final_sorted = []
        for block in blocks:
            if block["full"]:
                final_sorted.extend(sorted(block["full"], key=lambda x: x["bbox"][1]))
            if block["left"]:
                final_sorted.extend(sorted(block["left"], key=lambda x: x["bbox"][1]))
            if block["right"]:
                final_sorted.extend(sorted(block["right"], key=lambda x: x["bbox"][1]))
                
        return final_sorted
    
    def _postprocess_elements(self, elements: List[Dict]) -> List[Dict]:
        """
        [후처리 필터] 가짜 그림(HWP 테두리) 제거 + 우선순위 NMS
        """
        # 🛡️ 1. HWP 공문서 종특 방어 (가짜 그림 필터)
        fake_picture_indices = set()
        
        for i, el_i in enumerate(elements):
            # 대상이 그림이나 피규어일 때
            if el_i["type"] in ["Picture", "Figure"]:
                contain_count = 0
                box_i = el_i["bbox"]
                
                # 다른 모든 요소들과 비교해서 내 뱃속에 몇 개나 들어있는지 카운트
                for j, el_j in enumerate(elements):
                    if i == j: continue
                    box_j = el_j["bbox"]
                    inter_area = self._get_intersection_area(box_i, box_j)
                    area_j = self._get_box_area(box_j)
                    
                    # 다른 요소가 이 '그림' 안에 80% 이상 쏙 들어가 있다면
                    if area_j > 0 and (inter_area / area_j) > 0.80:
                        contain_count += 1
                
                # 💡 핵심: 뱃속에 요소가 4개 이상 있다? -> 이건 그림이 아니라 레이아웃 테두리다! 사형!
                if contain_count >= 4:
                    fake_picture_indices.add(i)
                    
        # 가짜 그림을 걸러낸 깨끗한 리스트로 필터링 시작
        filtered_elements = [el for i, el in enumerate(elements) if i not in fake_picture_indices]

        # ⚔️ 2. NMS 중복 제거 (기존 로직 동일)
        to_remove = set()
        priority = {
            "Table": 3, "Picture": 2, "Figure": 2, "Formula": 2, "Text": 2,
            "Section-header": 1, "List-item": 1, "Caption": 1,
            "Page-header": 1, "Page-footer": 1
        }

        for i in range(len(filtered_elements)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(filtered_elements)):
                if j in to_remove:
                    continue

                box1, box2 = filtered_elements[i]["bbox"], filtered_elements[j]["bbox"]
                inter_area = self._get_intersection_area(box1, box2)
                if inter_area == 0:
                    continue
                
                area1, area2 = self._get_box_area(box1), self._get_box_area(box2)
                min_area = min(area1, area2)
                
                if (inter_area / min_area) > 0.70:
                    type1, type2 = filtered_elements[i]["type"], filtered_elements[j]["type"]
                    p1, p2 = priority.get(type1, 0), priority.get(type2, 0)
                    
                    if p1 > p2:
                        to_remove.add(j)
                    elif p2 > p1:
                        to_remove.add(i)
                        break
                    else:
                        if filtered_elements[i]["confidence"] >= filtered_elements[j]["confidence"]:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break

        # 살아남은 객체들 추출 및 다단 읽기 순서 정렬
        valid_elements = [el for idx, el in enumerate(filtered_elements) if idx not in to_remove]
        valid_elements = self._sort_reading_order(valid_elements)
        
        for idx, el in enumerate(valid_elements):
            el["id"] = idx + 1
            
        return valid_elements
    
    def process_document(self, ingestion_data: Dict[str, Any]) -> Dict[str, Any]:
        file_name = ingestion_data["file_name"]
        doc_output_dir = os.path.join(self.output_base_dir, file_name)
        crop_dir = os.path.join(doc_output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        # ⭐ 멀티 스케일 (640 & 960) 앙상블 모드
        print(f"👁️ [Vision] '{file_name}' 분석 시작 (Multi-Scale 앙상블 모드)...")

        for page in ingestion_data["pages"]:
            img_path = page["image_path"]
            raw_elements = []
            
            # 1. 두 가지 해상도로 각각 스캔하여 박스 긁어모으기
            for size in [640, 960]:
                results = self.model(img_path, conf=0.25, iou=0.45, imgsz=size)[0]
                
                for box in results.boxes:
                    coords = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = results.names[cls_id]
                    conf = float(box.conf[0])
                    
                    raw_elements.append({
                        "type": label,
                        "bbox": coords,
                        "confidence": conf,
                        "crop_path": None # 크롭은 생존자만 나중에!
                    })
            
            # 2. 후처리 필터(NMS)로 640과 960의 겹치는 박스 제거 (똑똑한 놈만 생존)
            cleaned_elements = self._postprocess_elements(raw_elements)
            
            # 3. 생존한 객체들만 모아서 최종 이미지 크롭 진행 (디스크 I/O 최적화)
            with Image.open(img_path) as full_img:
                for el in cleaned_elements:
                    if el["type"] in ["Table", "Picture", "Figure", "Formula"]:
                        crop_name = f"p{page['page_num']}_{el['type'].lower()}_{el['id']}.png"
                        save_path = os.path.join(crop_dir, crop_name)
                        
                        cropped_img = full_img.crop(el["bbox"])
                        cropped_img.save(save_path)
                        el["crop_path"] = save_path

            page["elements"] = cleaned_elements
            print(f"   - {page['page_num']}페이지: 스캔 {len(raw_elements)}개 -> 최종 생존 {len(cleaned_elements)}개")

        # 메타데이터 저장
        meta_path = os.path.join(doc_output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(ingestion_data, f, ensure_ascii=False, indent=4)
            
        print(f"✅ [Vision] 멀티 스케일 앙상블 분석 완료! 결과물 저장: {meta_path}")
        return ingestion_data

class LayoutAnalyzer(VisionEngine):
    """
    기존 VisionEngine을 파이프라인 단계 이름에 맞춰 감싼 어댑터입니다.
    """

    def analyze(self, ingestion_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.process_document(ingestion_data)
