import os
import cv2
import re
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import easyocr
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

print("=== AI 모델 로드 중 (VARCO, TATR, EasyOCR) ===")
VARCO_MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
varco_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    VARCO_MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="auto"
)
varco_processor = AutoProcessor.from_pretrained(VARCO_MODEL_ID)
varco_model.eval()

TATR_MODEL_ID = "microsoft/table-transformer-structure-recognition"
tatr_processor = DetrImageProcessor.from_pretrained(TATR_MODEL_ID)
tatr_model = TableTransformerForObjectDetection.from_pretrained(TATR_MODEL_ID)
tatr_model.eval()
if torch.cuda.is_available(): tatr_model = tatr_model.to("cuda")

reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
print("AI 모델 로드 완료\n")

# 빈 칸(여백) 판별기
def is_blank_cell(cell_img_cv):
    """이미지가 실질적으로 비어있는지(여백만 있는지) 검사합니다."""
    gray = cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 테두리 노이즈를 피하기 위해 중앙 80% 영역만 검사
    crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
    if crop.size == 0: return True
    # 픽셀 변화가 5 이하면 글씨가 없는 빈 칸
    return np.std(crop) < 5.0

def extract_text_with_varco(cell_img_cv):
    image = Image.fromarray(cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2RGB))
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "<ocr>"}]}]
    inputs = varco_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(varco_model.device, torch.float16)

    with torch.no_grad():
        generate_ids = varco_model.generate(**inputs, max_new_tokens=1024, pad_token_id=varco_processor.tokenizer.eos_token_id)
    
    raw_output = varco_processor.decode(generate_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    cleaned_text = re.sub(r'<bbox>.*?</bbox>', '', raw_output).replace('<char>', '').replace('</char>', '').replace('<|im_end|>', '').replace('</s>', '').strip()
    
    #  LaTeX 수학 기호를 일반 텍스트로 치환
    cleaned_text = cleaned_text.replace('\\times', '×').replace('\\div', '÷').replace('\\pm', '±').replace('\\cdot', '·')
    return cleaned_text

def get_tatr_rows_cols(image: Image.Image):
    inputs = tatr_processor(images=image, return_tensors="pt")
    if torch.cuda.is_available(): inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = tatr_model(**inputs)
        
    target_sizes = torch.tensor([image.size[::-1]])
    results = tatr_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    
    raw_rows, raw_cols = [], []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        box = [round(i, 2) for i in box.tolist()]
        label_name = tatr_model.config.id2label[label.item()]
        
        if label_name in ["table row", "table column header"]:
            if box[3] - box[1] > 10: raw_rows.append(box) 
        elif label_name == "table column":
            if box[2] - box[0] > 10: raw_cols.append(box)
            
    # 쓸데없이 좁은 기둥 삭제
    raw_rows.sort(key=lambda x: x[1])
    rows = []
    for box in raw_rows:
        # 두 행(Row) 사이의 간격이 35px 이하라면 같은 행으로 취급하고 무시!
        if not rows or abs(box[1] - rows[-1][1]) > 10: rows.append(box)

    raw_cols.sort(key=lambda x: x[0])
    cols = []
    for box in raw_cols:
        # 열(Col) 간격도 35px 이하 필터링 적용
        if not cols or abs(box[0] - cols[-1][0]) > 10: cols.append(box)
            
    return rows, cols

def process_table_hybrid(image_path):
    orig_img = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    
    PADDING = 50
    padded_img_pil = ImageOps.expand(orig_img, border=PADDING, fill='white')
    img_cv_padded = cv2.copyMakeBorder(img_cv, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    print("TATR: 표 구조 분석")
    rows, cols = get_tatr_rows_cols(padded_img_pil)
    if not rows or not cols: return "표 구조를 찾지 못했습니다."
        
    print("EasyOCR: 텍스트 레이더 스캔")
    ocr_results = reader.readtext(img_cv_padded)
    text_boxes = []
    for bbox, _, _ in ocr_results:
        xs, ys = [pt[0] for pt in bbox], [pt[1] for pt in bbox]
        text_boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    
    print("3. 교차 검증: 병합 셀 매핑")
    num_rows, num_cols = len(rows), len(cols)
    grid = [[{"row_span": 1, "col_span": 1, "bbox": [], "is_master": True, "text": ""} 
             for _ in range(num_cols)] for _ in range(num_rows)]
    
    for r_idx, r_box in enumerate(rows):
        for c_idx, c_box in enumerate(cols):
            grid[r_idx][c_idx]["bbox"] = [c_box[0], r_box[1], c_box[2], r_box[3]]

    for (tx1, ty1, tx2, ty2) in text_boxes:
        covered_rows, covered_cols = [], []
        for r_idx, r_box in enumerate(rows):
            overlap_h = max(0, min(ty2, r_box[3]) - max(ty1, r_box[1]))
            if overlap_h / (ty2 - ty1 + 1e-5) > 0.3: covered_rows.append(r_idx)
        for c_idx, c_box in enumerate(cols):
            overlap_w = max(0, min(tx2, c_box[2]) - max(tx1, c_box[0]))
            if overlap_w / (tx2 - tx1 + 1e-5) > 0.2: covered_cols.append(c_idx)

        if covered_rows and covered_cols:
            sr, er = min(covered_rows), max(covered_rows)
            sc, ec = min(covered_cols), max(covered_cols)
            rs, cs = (er - sr) + 1, (ec - sc) + 1
            
            if rs > 1 or cs > 1:
                master = grid[sr][sc]
                if rs >= master["row_span"] and cs >= master["col_span"]:
                    master["row_span"], master["col_span"] = rs, cs
                    master["bbox"] = [cols[sc][0], rows[sr][1], cols[ec][2], rows[er][3]]
                    for r in range(sr, er + 1):
                        for c in range(sc, ec + 1):
                            if r == sr and c == sc: continue
                            grid[r][c]["is_master"] = False

    print("4. VARCO: 정밀 텍스트 추출")
    for r in range(num_rows):
        for c in range(num_cols):
            cell = grid[r][c]
            if not cell["is_master"]: continue 
                
            x1, y1, x2, y2 = map(int, cell["bbox"])
            MARGIN = 2
            cell_img = img_cv_padded[max(0, y1-MARGIN):min(img_cv_padded.shape[0], y2+MARGIN), 
                                     max(0, x1-MARGIN):min(img_cv_padded.shape[1], x2+MARGIN)]
            
            # 여백 칸은 VARCO 돌리지 않고 통과
            if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5 or is_blank_cell(cell_img): 
                cell["text"] = ""
                continue

            PAD_INTERNAL = 10
            cell_padded = cv2.copyMakeBorder(cell_img, PAD_INTERNAL, PAD_INTERNAL, PAD_INTERNAL, PAD_INTERNAL, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cell_upscaled = cv2.resize(cell_padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            
            cell["text"] = extract_text_with_varco(cell_upscaled)

    # 5. 마크다운 조립
    md_lines = []
    for r in range(num_rows):
        row_data = []
        for c in range(num_cols):
            cell = grid[r][c]
            if cell["is_master"]:
                text = cell["text"].replace('\n', ' ')
                text = re.sub(r'\s+', ' ', text).strip()
                # 환각 텍스트들 마지막 필터링
                if text in ["VARCO VISION", "VARCOVISION", "xxx", "I", ".", "-", "_"]: text = ""
                row_data.append(text)
            else:
                row_data.append(" ") # 병합 슬레이브
                
        md_lines.append("| " + " | ".join(row_data) + " |")
        if r == 0:
            md_lines.append("|" + "|".join(["---"] * num_cols) + "|")

    return "\n".join(md_lines)

if __name__ == "__main__":
    target_image = "example_sheets_7.png"  
    import time
    start = time.time()
    if os.path.exists(target_image):
        print(f"\n'{target_image}' 파싱 시작...")
        final_md = process_table_hybrid(target_image)
        
        print(final_md) 
        
        with open("final_perfect_markdown.md", "w", encoding="utf-8") as f:
            f.write(final_md)
        print("'final_perfect_markdown.md' 파일이 성공적으로 저장되었습니다") #후에 마크다운 파일 저장안하고 그냥 반환하는 형식으로 변경하면 됨
    else:
        print(f"'{target_image}' 파일을 찾을 수 없습니다.")
    end = time.time()
    print(f"걸린 시간 : {end - start}")
