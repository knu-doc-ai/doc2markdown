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


def _run_varco_generation(image: Image.Image) -> str:
    """VARCO로 셀 텍스트를 추출한다."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "<ocr>"},
            ],
        }
    ]
    inputs = varco_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(varco_model.device, torch.float16)

    with torch.no_grad():
        generate_ids = varco_model.generate(
            **inputs,
            max_new_tokens=1024,
            pad_token_id=varco_processor.tokenizer.eos_token_id,
        )

    raw_output = varco_processor.decode(
        generate_ids[0][len(inputs.input_ids[0]):],
        skip_special_tokens=False,
    )
    cleaned_text = re.sub(r"<bbox>.*?</bbox>", "", raw_output)
    cleaned_text = cleaned_text.replace("<char>", "").replace("</char>", "")
    cleaned_text = cleaned_text.replace("<|im_end|>", "").replace("</s>", "").strip()
    cleaned_text = cleaned_text.replace("\\times", "×")
    cleaned_text = cleaned_text.replace("\\div", "÷")
    cleaned_text = cleaned_text.replace("\\pm", "±")
    cleaned_text = cleaned_text.replace("\\cdot", "·")
    return cleaned_text


def is_blank_cell(cell_img_cv):
    """빈 셀처럼 보이는 영역은 OCR 대상에서 제외한다."""
    gray = cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    if crop.size == 0:
        return True
    return np.std(crop) < 5.0


def extract_text_with_varco(cell_img_cv):
    image = Image.fromarray(cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2RGB))
    return _run_varco_generation(image)


def get_tatr_rows_cols(image: Image.Image):
    inputs = tatr_processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = tatr_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = tatr_processor.post_process_object_detection(
        outputs,
        threshold=0.5,
        target_sizes=target_sizes,
    )[0]

    raw_rows = []
    raw_cols = []
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
        img_cv,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )

    print("1. TATR: 표 구조 분석")
    rows, cols = get_tatr_rows_cols(padded_img_pil)
    if not rows or not cols:
        return "표 구조를 찾지 못했습니다."

    print("2. EasyOCR: 텍스트 레이더 스캔")
    ocr_results = reader.readtext(img_cv_padded)
    text_boxes = []
    for bbox, _, _ in ocr_results:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        text_boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("3. 교차 검증: 병합 셀 매핑")
    num_rows = len(rows)
    num_cols = len(cols)
    grid = [
        [
            {"row_span": 1, "col_span": 1, "bbox": [], "is_master": True, "text": ""}
            for _ in range(num_cols)
        ]
        for _ in range(num_rows)
    ]

    cut_y = [rows[0][1]]
    for index in range(num_rows - 1):
        cut_y.append((rows[index][3] + rows[index + 1][1]) / 2.0)
    cut_y.append(rows[-1][3])

    cut_x = [cols[0][0]]
    for index in range(num_cols - 1):
        cut_x.append((cols[index][2] + cols[index + 1][0]) / 2.0)
    cut_x.append(cols[-1][2])

    for row_index in range(num_rows):
        for col_index in range(num_cols):
            grid[row_index][col_index]["bbox"] = [
                cut_x[col_index],
                cut_y[row_index],
                cut_x[col_index + 1],
                cut_y[row_index + 1],
            ]

    for tx1, ty1, tx2, ty2 in text_boxes:
        text_w = tx2 - tx1
        text_h = ty2 - ty1
        if text_w < 5 or text_h < 5:
            continue

        covered_rows = []
        covered_cols = []

        for row_index in range(num_rows):
            overlap_h = max(0, min(ty2, cut_y[row_index + 1]) - max(ty1, cut_y[row_index]))
            if overlap_h / text_h > 0.3:
                covered_rows.append(row_index)

        for col_index in range(num_cols):
            overlap_w = max(0, min(tx2, cut_x[col_index + 1]) - max(tx1, cut_x[col_index]))
            if overlap_w / text_w > 0.25:
                covered_cols.append(col_index)

        if covered_rows and covered_cols:
            sr = min(covered_rows)
            er = max(covered_rows)
            sc = min(covered_cols)
            ec = max(covered_cols)
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
                            if row_index == sr and col_index == sc:
                                continue
                            grid[row_index][col_index]["is_master"] = False

    print("4. VARCO: 정밀 텍스트 추출")
    for row_index in range(num_rows):
        for col_index in range(num_cols):
            cell = grid[row_index][col_index]
            if not cell["is_master"]:
                continue

            x1, y1, x2, y2 = map(int, cell["bbox"])
            margin = 2
            cell_img = img_cv_padded[
                max(0, y1 - margin):min(img_cv_padded.shape[0], y2 + margin),
                max(0, x1 - margin):min(img_cv_padded.shape[1], x2 + margin),
            ]

            if (
                cell_img.size == 0
                or cell_img.shape[0] < 5
                or cell_img.shape[1] < 5
                or is_blank_cell(cell_img)
            ):
                cell["text"] = ""
                continue

            pad_internal = 10
            cell_padded = cv2.copyMakeBorder(
                cell_img,
                pad_internal,
                pad_internal,
                pad_internal,
                pad_internal,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
            cell_upscaled = cv2.resize(
                cell_padded,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LINEAR,
            )
            cell["text"] = extract_text_with_varco(cell_upscaled)

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
    """워커 프로세스용 표 추출 진입점."""
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
        print(final_md)

        with open("final_perfect_markdown.md", "w", encoding="utf-8") as file:
            file.write(final_md)
        print("'final_perfect_markdown.md' 파일이 저장되었습니다.")
    else:
        print(f"'{target_image}' 파일을 찾을 수 없습니다.")
    end = time.time()
    print(f"걸린 시간: {end - start}")
