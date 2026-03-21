import os
import cv2
import re
import torch
import numpy as np
from PIL import Image, ImageOps
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import warnings
import logging

# 경고 메시지 끄기(안써도 됨)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. 모델 로드 (VARCO & TATR)
print("=== VARCO-VISION & TATR 모델 로드 중... ===")
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
if torch.cuda.is_available():
    tatr_model = tatr_model.to("cuda")

print("AI 모델 로드 완료\n")

def extract_text_with_varco(cell_img_cv):
    # VARCO OCR로 텍스트 추출
    image = Image.fromarray(cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2RGB))
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "<ocr>"}]}]
    inputs = varco_processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(varco_model.device, torch.float16)

    with torch.no_grad():
        generate_ids = varco_model.generate(**inputs, max_new_tokens=1024, pad_token_id=varco_processor.tokenizer.eos_token_id)
    
    raw_output = varco_processor.decode(generate_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
    cleaned_text = re.sub(r'<bbox>.*?</bbox>', '', raw_output).replace('<char>', '').replace('</char>', '').replace('<|im_end|>', '').replace('</s>', '').strip()
    return cleaned_text

def get_tatr_grid(image: Image.Image):
    # TATR로 행렬 좌표를 추출하고 중복된 선을 필터링
    inputs = tatr_processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = tatr_model(**inputs)
        
    target_sizes = torch.tensor([image.size[::-1]])
    results = tatr_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    
    raw_rows, raw_cols = [], []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        box = [round(i, 2) for i in box.tolist()]
        label_name = tatr_model.config.id2label[label.item()]
        
        if label_name in ["table row", "table column header"]:
            raw_rows.append(box) 
        elif label_name == "table column":
            raw_cols.append(box)
            
    # 중복된 행/열 제거 로직
    raw_rows.sort(key=lambda x: x[1])
    rows = []
    for box in raw_rows:
        if not rows or abs(box[1] - rows[-1][1]) > 10:
            rows.append(box)

    raw_cols.sort(key=lambda x: x[0])
    cols = []
    for box in raw_cols:
        if not cols or abs(box[0] - cols[-1][0]) > 10:
            cols.append(box)
            
    return rows, cols

def process_table_tatr_varco(image_path):
    orig_img = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)
    
    # 패딩 100 적용, 나중에 이미지에 따라서 바꿔질 가능성 있을수도
    PADDING = 100
    padded_img_pil = ImageOps.expand(orig_img, border=PADDING, fill='white')
    img_cv_padded = cv2.copyMakeBorder(img_cv, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    print("TATR: 표의 구조를 분석 중")
    rows, cols = get_tatr_grid(padded_img_pil)
    
    if not rows or not cols:
        return "표 구조를 찾지 못했습니다."
        
    print(f"발견된 행(Row) 개수: {len(rows)}, 열(Column) 개수: {len(cols)}") # 행 열 확인용, 나중에 지우기
    
    grid = [["" for _ in range(len(cols))] for _ in range(len(rows))]

    for r_idx, row_box in enumerate(rows):
        y_min, y_max = row_box[1], row_box[3]
        for c_idx, col_box in enumerate(cols):
            x_min, x_max = col_box[0], col_box[2]
            
            # 마진 1픽셀 적용
            MARGIN = 1
            left = max(0, int(x_min) - MARGIN)
            right = min(img_cv_padded.shape[1], int(x_max) + MARGIN)
            top = max(0, int(y_min) - MARGIN)
            bottom = min(img_cv_padded.shape[0], int(y_max) + MARGIN)
            
            cell_img = img_cv_padded[top:bottom, left:right]
            
            if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                continue

            # VARCO 내부 여백 및 2배 확대
            PAD_INTERNAL = 10
            cell_padded_internal = cv2.copyMakeBorder(cell_img, PAD_INTERNAL, PAD_INTERNAL, PAD_INTERNAL, PAD_INTERNAL, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cell_img_upscaled = cv2.resize(cell_padded_internal, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            
            extracted_text = extract_text_with_varco(cell_img_upscaled)
            grid[r_idx][c_idx] = extracted_text

    # 마크다운 조립, VARCO 완벽 필터링
    md_lines = []
    for i, row_data in enumerate(grid):
        cleaned_row = []
        for item in row_data:
            # 줄바꿈을 띄어쓰기 하나로 치환
            text = str(item).replace('\n', ' ')
            # 연속된 공백을 하나의 공백으로 깔끔하게 정리
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 오직 'VARCO VISION' 밖에 없을 때만 완벽하게 공백 처리 (비어있는 셀 이미지를 받으면 VARCO VISION으로 return하기 때문)
            if text in ["VARCO VISION", "VARCOVISION"]:
                text = ""
                
            cleaned_row.append(text)
            
        md_lines.append("| " + " | ".join(cleaned_row) + " |")
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    return "\n".join(md_lines)

if __name__ == "__main__":
    target_image = "example_sheets_5.png"  
    
    if os.path.exists(target_image):
        print(f"{target_image} 이미지 탐색...")
        final_md = process_table_tatr_varco(target_image)
        
        print("\n" + "="*50)
        print(final_md) #확인용 출력, 후에 합칠때는 지우기
        print("="*50 + "\n")
        
        with open("final_perfect_markdown.md", "w", encoding="utf-8") as f:
            f.write(final_md)
        print("final_perfect_markdown.md 파일에 저장하였습니다.")
    else:
        print(f" {target_image}' 파일을 찾을 수 없습니다.")
