import os
import cv2
import re
import torch
import paddle
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from docling.document_converter import DocumentConverter

# ==========================================
# 1. 듀얼 OCR 엔진 초기화
# ==========================================
print("=== 하드웨어 및 스마트 듀얼 엔진 초기화 ===")
# 디텍션(위치 찾기) 및 한글 담당
ocr_kr = PaddleOCR(lang='korean', use_gpu=True, show_log=False, det_db_thresh=0.3)
# 숫자, 특수기호(%, . ,), 영문 담당
ocr_en = PaddleOCR(lang='en', use_gpu=True, show_log=False)
print("✅ 스마트 듀얼 엔진 로드 완료!\n")

def process_table_to_markdown(table_data, page_image: Image.Image):
    img_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    
    max_row = max(c.end_row_offset_idx for c in table_data.table_cells)
    max_col = max(c.end_col_offset_idx for c in table_data.table_cells)
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # 열별 중심점 산출 (어긋남 방지)
    col_centers = {}
    for cell in table_data.table_cells:
        c_idx = cell.start_col_offset_idx
        center_x = (cell.bbox.l + cell.bbox.r) / 2.0
        if c_idx not in col_centers:
            col_centers[c_idx] = []
        col_centers[c_idx].append(center_x)
    avg_col_centers = {k: sum(v)/len(v) for k, v in col_centers.items()}

    table_left = min(cell.bbox.l for cell in table_data.table_cells)
    table_right = max(cell.bbox.r for cell in table_data.table_cells)

    row_cells = {}
    for cell in table_data.table_cells:
        r_idx = cell.start_row_offset_idx
        if r_idx not in row_cells:
            row_cells[r_idx] = []
        row_cells[r_idx].append(cell)

    # 행 단위 루프
    for row_idx in sorted(row_cells.keys()):
        cells = row_cells[row_idx]
        row_top = max(0, min(c.bbox.t for c in cells) - 3)
        row_bottom = max(c.bbox.b for c in cells) + 3
        
        row_img = img_cv[int(row_top):int(row_bottom), int(table_left):int(table_right)]
        if row_img.size == 0 or row_img.shape[0] < 5 or row_img.shape[1] < 5:
            continue

        # 해상도 2배 확대
        row_img_upscaled = cv2.resize(row_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 🚀 1. 한국어 엔진으로 전체 행 스캔
        res_kr = ocr_kr.ocr(row_img_upscaled, cls=False)
        
        if res_kr and res_kr[0]:
            for line in res_kr[0]:
                box, (text_kr, score_kr) = line
                
                # 🚀 2. 동적 엔진 스위칭 (한글 포함 여부 검사)
                has_korean = bool(re.search(r'[가-힣]', text_kr))
                
                if has_korean:
                    # 한글이 포함되어 있다면 한국어 엔진 결과 채택 (예: "현대오토에버", "1위")
                    final_text = text_kr
                else:
                    # 한글이 없다면 (예: "4.7%", "11,655", "LG CNS") -> 영어 엔진 출동
                    # 해당 글자 영역의 박스 좌표 추출
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    
                    # 자를 때 아주 미세한 여백(±2px) 추가
                    x_min = max(0, int(min(xs)) - 2)
                    x_max = min(row_img_upscaled.shape[1], int(max(xs)) + 2)
                    y_min = max(0, int(min(ys)) - 2)
                    y_max = min(row_img_upscaled.shape[0], int(max(ys)) + 2)
                    
                    crop_img = row_img_upscaled[y_min:y_max, x_min:x_max]
                    
                    if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
                        # det=False 옵션: 위치 찾기 생략하고 글자 인식만 수행 (초고속)
                        res_en = ocr_en.ocr(crop_img, det=False, cls=False)
                        
                        if res_en and res_en[0]:
                            # res_en 구조: [[('text', score)]]
                            final_text = res_en[0][0][0]
                        else:
                            final_text = text_kr # 실패 시 원래 텍스트 복구
                    else:
                        final_text = text_kr

                # 3. 데이터 매핑 로직 (원상태로 복원된 텍스트 사용)
                upscaled_center_x = (box[0][0] + box[1][0]) / 2.0
                original_center_x = upscaled_center_x / 2.0
                abs_x = table_left + original_center_x
                
                best_col = min(avg_col_centers.keys(), key=lambda k: abs(avg_col_centers[k] - abs_x))
                
                if grid[row_idx][best_col]:
                    grid[row_idx][best_col] += " " + final_text
                else:
                    grid[row_idx][best_col] = final_text

    # Markdown 변환
    md_lines = []
    for i, row_data in enumerate(grid):
        cleaned_row = [str(item).replace('\n', ' ').strip() for item in row_data]
        md_lines.append("| " + " | ".join(cleaned_row) + " |")
        if i == 0:
            separator = ["---"] * (max_col + 1)
            md_lines.append("|" + "|".join(separator) + "|")

    return "\n".join(md_lines)
# ==========================================
# 4. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    target_image = "example_sheets_4.png"  # ◀◀◀ 테스트할 PDF 파일명을 여기에 적어주세요!
    
    if not os.path.exists(target_image):
        print(f"❌ '{target_image}' 파일을 찾을 수 없습니다.")
        exit()

    print(f"'{target_image}' 문서 처리를 시작합니다...\n")

    converter = DocumentConverter()
    doc_result = converter.convert(target_image)
    tables = doc_result.document.tables
    print(f"✅ 총 {len(tables)}개의 표(Table) 구조 발견!\n")

    original_image = Image.open(target_image).convert("RGB")

    for i, table_item in enumerate(tables):
        print(f"--- [표 {i+1} 처리 결과] ---")
        markdown_output = process_table_to_markdown(table_item.data, original_image)
        print(markdown_output)
        print("\n")