import json
import logging
import os
import re
import sys

import cv2
import easyocr
import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    DetrImageProcessor,
    LlavaOnevisionForConditionalGeneration,
    TableTransformerForObjectDetection,
)

from modules.shared_ocr import get_shared_varco_components

logging.getLogger("transformers").setLevel(logging.ERROR)

TABLE_EXTRACTION_MODE = os.getenv("TABLE_EXTRACTION_MODE", "direct").strip().lower()

# 최적화 설정 값
BATCH_SIZE = 8                  # VRAM에 맞춰 조절
EASYOCR_CONF_THRESHOLD = 0.98   # 신뢰도 하한선

print("=== AI 모델 로드 중 (VARCO, TATR, EasyOCR) ===")

VARCO_MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
if TABLE_EXTRACTION_MODE == "direct":
    print("[TableOCR] direct 모드: SharedOCR 재사용")
    varco_processor, varco_model = get_shared_varco_components()
else:
    print("[TableOCR] VARCO-VISION OCR 모델 로드 중...")
    varco_processor = AutoProcessor.from_pretrained(VARCO_MODEL_ID)
    varco_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        VARCO_MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    varco_model.eval()
    print("[TableOCR] VARCO 모델 로드 완료!")

TATR_MODEL_ID = "microsoft/table-transformer-structure-recognition"
tatr_processor = DetrImageProcessor.from_pretrained(TATR_MODEL_ID)
tatr_model = TableTransformerForObjectDetection.from_pretrained(TATR_MODEL_ID)
tatr_model.eval()
if torch.cuda.is_available():
    tatr_model = tatr_model.to("cuda")

reader = easyocr.Reader(["ko", "en"], gpu=torch.cuda.is_available())
print("AI 모델 로드 완료\n")


class TableExtractor:
    """표 crop 이미지를 Markdown 문자열로 변환한다."""
    def extract_table(self, image_path):
        return process_table_hybrid(image_path)


def clean_varco_text(raw_output: str) -> str:
    cleaned_text = re.sub(r"<bbox>.*?</bbox>", "", raw_output)
    cleaned_text = cleaned_text.replace("<char>", "").replace("</char>", "")
    cleaned_text = cleaned_text.replace("<|im_end|>", "").replace("</s>", "").strip()
    cleaned_text = cleaned_text.replace("\\times", "×")
    cleaned_text = cleaned_text.replace("\\div", "÷")
    cleaned_text = cleaned_text.replace("\\pm", "±")
    cleaned_text = cleaned_text.replace("\\cdot", "·")
    return cleaned_text


def _run_varco_batch_generation(images: list[Image.Image], batch_size: int = BATCH_SIZE) -> list[str]:
    results = []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "<ocr>"},
                    ],
                }
            ]
            for img in batch_imgs
        ]
        
        inputs = varco_processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(varco_model.device, torch.float16)

        with torch.no_grad():
            generate_ids = varco_model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=varco_processor.tokenizer.eos_token_id,
            )

        for j in range(len(batch_imgs)):
            input_len = len(inputs.input_ids[j])
            raw_output = varco_processor.decode(
                generate_ids[j][input_len:],
                skip_special_tokens=False,
            )
            results.append(clean_varco_text(raw_output))
            
    return results


def is_blank_cell(cell_img_cv):
    gray = cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    if crop.size == 0:
        return True
    return np.std(crop) < 5.0


def get_tatr_rows_cols(image: Image.Image):
    inputs = tatr_processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = tatr_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = tatr_processor.post_process_object_detection(
        outputs, threshold=0.5, target_sizes=target_sizes
    )[0]

    raw_rows, raw_cols = [], []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(item, 2) for item in box.tolist()]
        label_name = tatr_model.config.id2label[label.item()]

        if label_name in ["table row", "table column header"]:
            if box[3] - box[1] > 10:
                raw_rows.append(box)
        elif label_name == "table column":
            if box[2] - box[0] > 10:
                raw_cols.append(box)

    raw_rows.sort(key=lambda item: item[1])
    rows = []
    for box in raw_rows:
        if not rows or abs(box[1] - rows[-1][1]) > 10:
            rows.append(box)

    raw_cols.sort(key=lambda item: item[0])
    cols = []
    for box in raw_cols:
        if not cols or abs(box[0] - cols[-1][0]) > 10:
            cols.append(box)

    return rows, cols


def process_table_hybrid(image_path):
    orig_img = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)

    padding = 50
    padded_img_pil = ImageOps.expand(orig_img, border=padding, fill="white")
    img_cv_padded = cv2.copyMakeBorder(
        img_cv, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    print("1. TATR: 표 구조 분석")
    rows, cols = get_tatr_rows_cols(padded_img_pil)
    if not rows or not cols:
        return "표 구조를 찾지 못했습니다."

    print("2. EasyOCR: 텍스트 스캔 (텍스트 및 신뢰도 확보)")
    ocr_results = reader.readtext(img_cv_padded)
    text_boxes = []
    for bbox, text, conf in ocr_results:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        text_boxes.append({
            "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            "text": text,
            "conf": conf
        })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("3. 교차 검증: 병합 셀 매핑 및 EasyOCR 텍스트 할당")
    num_rows = len(rows)
    num_cols = len(cols)
    grid = [
        [
            {"row_span": 1, "col_span": 1, "bbox": [], "is_master": True, "text": "", "easyocr_text": "", "easyocr_conf": 0.0}
            for _ in range(num_cols)
        ]
        for _ in range(num_rows)
    ]

    cut_y = [rows[0][1]] + [(rows[i][3] + rows[i+1][1]) / 2.0 for i in range(num_rows - 1)] + [rows[-1][3]]
    cut_x = [cols[0][0]] + [(cols[i][2] + cols[i+1][0]) / 2.0 for i in range(num_cols - 1)] + [cols[-1][2]]

    for row_index in range(num_rows):
        for col_index in range(num_cols):
            grid[row_index][col_index]["bbox"] = [cut_x[col_index], cut_y[row_index], cut_x[col_index + 1], cut_y[row_index + 1]]

    # 병합 셀 판별 로직
    for tb in text_boxes:
        tx1, ty1, tx2, ty2 = tb["bbox"]
        text_w = tx2 - tx1
        text_h = ty2 - ty1
        if text_w < 5 or text_h < 5:
            continue

        covered_rows = [r for r in range(num_rows) if max(0, min(ty2, cut_y[r+1]) - max(ty1, cut_y[r])) / text_h > 0.3]
        covered_cols = [c for c in range(num_cols) if max(0, min(tx2, cut_x[c+1]) - max(tx1, cut_x[c])) / text_w > 0.25]

        if covered_rows and covered_cols:
            sr, er = min(covered_rows), max(covered_rows)
            sc, ec = min(covered_cols), max(covered_cols)
            row_span = (er - sr) + 1
            col_span = (ec - sc) + 1

            if row_span > 1 or col_span > 1:
                master = grid[sr][sc]
                if row_span >= master["row_span"] and col_span >= master["col_span"]:
                    master["row_span"] = row_span
                    master["col_span"] = col_span
                    master["bbox"] = [cut_x[sc], cut_y[sr], cut_x[ec + 1], cut_y[er + 1]]
                    for row_index in range(sr, er + 1):
                        for col_index in range(sc, ec + 1):
                            if row_index != sr or col_index != sc:
                                grid[row_index][col_index]["is_master"] = False

    # 마스터 셀에 EasyOCR 텍스트 및 신뢰도 매핑
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            cell = grid[row_index][col_index]
            if not cell["is_master"]:
                continue
            
            cx1, cy1, cx2, cy2 = cell["bbox"]
            matched_texts = []
            matched_confs = []
            
            for tb in text_boxes:
                tx1, ty1, tx2, ty2 = tb["bbox"]
                # 셀 영역과 텍스트 박스 영역의 교차 여부 확인
                overlap_w = max(0, min(tx2, cx2) - max(tx1, cx1))
                overlap_h = max(0, min(ty2, cy2) - max(ty1, cy1))
                tb_area = (tx2 - tx1) * (ty2 - ty1)
                
                # 텍스트 박스 면적의 50% 이상이 셀 안에 포함되어 있으면 해당 셀의 텍스트로 인정
                if tb_area > 0 and (overlap_w * overlap_h) / tb_area > 0.5:
                    matched_texts.append(tb["text"])
                    matched_confs.append(tb["conf"])
                    
            if matched_texts:
                cell["easyocr_text"] = " ".join(matched_texts)
                cell["easyocr_conf"] = sum(matched_confs) / len(matched_confs)

    print(f"4. 조건부 VLM 정밀 텍스트 추출 (EasyOCR Threshold: {EASYOCR_CONF_THRESHOLD})")
    cell_tasks = []
    easyocr_pass_count = 0

    for row_index in range(num_rows):
        for col_index in range(num_cols):
            cell = grid[row_index][col_index]
            if not cell["is_master"]:
                continue

            # 빈 셀 체크
            x1, y1, x2, y2 = map(int, cell["bbox"])
            margin = 2
            cell_img = img_cv_padded[
                max(0, y1 - margin):min(img_cv_padded.shape[0], y2 + margin),
                max(0, x1 - margin):min(img_cv_padded.shape[1], x2 + margin),
            ]

            if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5 or is_blank_cell(cell_img):
                cell["text"] = ""
                continue

            # EasyOCR 조건부 패스: 신뢰도가 높으면 VARCO 생략
            if cell["easyocr_conf"] >= EASYOCR_CONF_THRESHOLD and cell["easyocr_text"].strip():
                cell["text"] = cell["easyocr_text"]
                easyocr_pass_count += 1
                continue

            # VARCO 처리를 위한 전처리 및 대기열 추가
            pad_internal = 10
            cell_padded = cv2.copyMakeBorder(
                cell_img, pad_internal, pad_internal, pad_internal, pad_internal,
                cv2.BORDER_CONSTANT, value=[255, 255, 255],
            )
            cell_upscaled = cv2.resize(
                cell_padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR,
            )
            pil_image = Image.fromarray(cv2.cvtColor(cell_upscaled, cv2.COLOR_BGR2RGB))
            cell_tasks.append((row_index, col_index, pil_image))

    print(f"{easyocr_pass_count}개의 셀은 신뢰도가 높아 바로 확정했습니다")
    if cell_tasks:
        print(f"나머지 {len(cell_tasks)}개의 셀을 병렬 추출합니다...")
        images_to_process = [task[2] for task in cell_tasks]
        extracted_texts = _run_varco_batch_generation(images_to_process, batch_size=BATCH_SIZE)
        
        for (r, c, _), text in zip(cell_tasks, extracted_texts):
            grid[r][c]["text"] = text

    print("5. 마크다운 생성 완료")
    md_lines = []
    for row_index in range(num_rows):
        row_data = []
        for col_index in range(num_cols):
            cell = grid[row_index][col_index]
            if cell["is_master"]:
                text = cell["text"].replace("\n", " ")
                text = re.sub(r"\s+", " ", text).strip()
                if text in ["VARCO VISION", "VARCOVISION", "xxx", "I", ".", "-", "_"]:
                    text = ""
                row_data.append(text)
            else:
                row_data.append(" ")

        md_lines.append("| " + " | ".join(row_data) + " |")
        if row_index == 0:
            md_lines.append("|" + "|".join(["---"] * num_cols) + "|")

    return "\n".join(md_lines)


def _run_cli_extraction(image_path: str, result_path: str) -> int:
    try:
        markdown = process_table_hybrid(image_path)
        payload = {"status": "success", "markdown": markdown}
        exit_code = 0
    except Exception as error:
        payload = {"status": "error", "error": str(error)}
        exit_code = 1

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return exit_code


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--extract":
        sys.exit(_run_cli_extraction(sys.argv[2], sys.argv[3]))

    target_image = "example_sheets_7.png"
    import time

    start = time.time()
    if os.path.exists(target_image):
        print(f"\n'{target_image}' 테스트 시작...")
        final_md = process_table_hybrid(target_image)
        print("\n" + final_md + "\n")

        with open("final_perfect_markdown.md", "w", encoding="utf-8") as file:
            file.write(final_md)
        print("'final_perfect_markdown.md' 파일이 저장되었습니다.")
    else:
        print(f"'{target_image}' 파일을 찾을 수 없습니다.")
    end = time.time()
    print(f"\n걸린 시간: {end - start:.2f}초")
