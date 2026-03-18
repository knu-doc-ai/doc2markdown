import os
import cv2
import re
import torch
import paddle
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from docling.document_converter import DocumentConverter

# 사용한 버전
# opencv == 4.6.0.66, numpy < 2.0, paddleocr == 2.7.3, 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118, paddlepaddle-gpu==2.6.1.post117, python 3.10.19

ocr_kr = PaddleOCR(lang='korean', use_gpu=True, show_log=False, det_db_thresh=0.2, det_db_box_thresh=0.1)
ocr_en = PaddleOCR(lang='en', use_gpu=True, show_log=False, det_db_thresh=0.2, det_db_box_thresh=0.1)

def process_table_to_markdown(table_data, page_image: Image.Image):
    img_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
    
    max_row = max(c.end_row_offset_idx for c in table_data.table_cells)
    max_col = max(c.end_col_offset_idx for c in table_data.table_cells)
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # 열별 중심점 매핑
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

    for row_idx in sorted(row_cells.keys()):
        cells = row_cells[row_idx]
        
        row_left = max(0, int(table_left) - 10)
        row_right = min(img_cv.shape[1], int(table_right) + 10)
        row_top = max(0, int(min(c.bbox.t for c in cells)) - 5)
        row_bottom = min(img_cv.shape[0], int(max(c.bbox.b for c in cells)) + 5)
        
        row_img = img_cv[row_top:row_bottom, row_left:row_right]
        if row_img.size == 0 or row_img.shape[0] < 5 or row_img.shape[1] < 5:
            continue

        # 범용 여백 및 2배율 적용
        PAD = 15
        padded_img = cv2.copyMakeBorder(row_img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        row_img_upscaled = cv2.resize(padded_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

        # 1. 두 엔진 모두 독립적으로 스캔 (병렬 처리)
        res_kr = ocr_kr.ocr(row_img_upscaled, cls=False)
        res_en = ocr_en.ocr(row_img_upscaled, cls=False)

        # 현재 행(Row)의 열(Column)별로 데이터를 담을 임시 list 준비
        row_cell_data = {c: {"kr": [], "en": []} for c in avg_col_centers.keys()}

        # 데이터를 list에 담는 내부 함수
        def extract_and_map(ocr_res, lang_key):
            if ocr_res and ocr_res[0]:
                for line in ocr_res[0]:
                    box, (text, score) = line
                    text = str(text).strip()
                    if not text:
                        continue
                    
                    # 좌표 역산
                    upscaled_center_x = (box[0][0] + box[1][0]) / 2.0
                    original_center_x = (upscaled_center_x / 2.0) - PAD
                    abs_x = row_left + original_center_x
                    
                    # 가장 가까운 열 찾아서 list에 추가
                    best_col = min(avg_col_centers.keys(), key=lambda k: abs(avg_col_centers[k] - abs_x))
                    row_cell_data[best_col][lang_key].append(text)

        # 두 엔진의 결과를 각각의 list에 쏟아 붓기
        extract_and_map(res_kr, "kr")
        extract_and_map(res_en, "en")

        # 2. 열별로 최종 결과 판결
        for col in avg_col_centers.keys():
            kr_str = " ".join(row_cell_data[col]["kr"]).strip()
            en_str = " ".join(row_cell_data[col]["en"]).strip()
            
            # 영어 엔진 결과가 '순수 숫자/기호'인지 검사 (콤마, 소수점, %, 괄호 포함)
            is_en_pure_number = bool(re.match(r'^[\d\.\,\%\-\+\s\(\)]+$', en_str)) and len(en_str) > 0
            
            # 판결 로직 (Priority Rules)
            if is_en_pure_number:
                # 규칙 1: 완벽한 숫자면 한국어 엔진 무시하고 무조건 영어 채택
                final_text = en_str
            elif re.search(r'[가-힣]', kr_str):
                # 규칙 2: 숫자가 아닌데 한글이 있으면 한국어 엔진 채택
                final_text = kr_str
            elif en_str:
                # 규칙 3: 한글도 없고 순수 숫자도 아닌 영문(예: LG CNS)이면 영어 엔진 채택
                final_text = en_str
            else:
                # 예비용 (Fallback)
                final_text = kr_str
                
            if final_text:
                if grid[row_idx][col]:
                    grid[row_idx][col] += " " + final_text
                else:
                    grid[row_idx][col] = final_text

    # Markdown 변환
    md_lines = []
    for i, row_data in enumerate(grid):
        cleaned_row = [str(item).replace('\n', ' ').strip() for item in row_data]
        md_lines.append("| " + " | ".join(cleaned_row) + " |")
        if i == 0:
            separator = ["---"] * (max_col + 1)
            md_lines.append("|" + "|".join(separator) + "|")

    return "\n".join(md_lines)

# 4. 메인 실행 블록 (후에 메인과 상호작용위해서 이부분만 고치면 됨)

if __name__ == "__main__":
    target_image = "example_sheets_4.png"  #테스트 파일 명 넣기
    
    if not os.path.exists(target_image):
        print(f"❌ '{target_image}' 파일을 찾을 수 없습니다.")
        exit()

    converter = DocumentConverter()
    doc_result = converter.convert(target_image)
    tables = doc_result.document.tables

    original_image = Image.open(target_image).convert("RGB")
    if not tables:
        print("이미지에서 표를 찾지 못했습니다.")
    else:
        original_image = Image.open(target_image).convert("RGB")
        markdown_output = process_table_to_markdown(tables[0].data, original_image)
        print(markdown_output)
