import gc
import logging
import os
import re
import warnings
from typing import Any, Dict

import cv2
import fitz
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from modules.shared_ocr import get_shared_varco_components, release_shared_varco_components

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


class TextExtractor:
    """PDF 본문 텍스트를 추출하는 모듈."""

    def __init__(self):
        self.table_extraction_mode = os.getenv("TABLE_EXTRACTION_MODE", "direct").strip().lower()
        self.use_shared_varco = self.table_extraction_mode == "direct"
        self.varco_processor, self.varco_model = self._load_varco_components()

    def _load_varco_components(self):
        """텍스트 추출용 VARCO 모델을 직접 로드한다."""
        if self.use_shared_varco:
            return get_shared_varco_components()

        print("[TextExtractor] VARCO-VISION OCR 모델 로드 중...")
        varco_model_id = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
        processor = AutoProcessor.from_pretrained(varco_model_id)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            varco_model_id,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        model.eval()
        print("[TextExtractor] VARCO 모델 로드 완료!\n")
        return processor, model

    def _clean_text(self, text: str) -> str:
        """줄바꿈과 과한 공백을 정리한다."""
        if not text:
            return ""
        return " ".join(text.replace("\n", " ").split())

    def _is_blank_cell(self, cell_img_cv) -> bool:
        """빈 이미지처럼 보이는 영역은 OCR에서 제외한다."""
        gray = cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        if crop.size == 0:
            return True
        return np.std(crop) < 5.0

    def _extract_with_varco(self, cell_img_cv) -> str:
        """VARCO로 이미지 조각의 텍스트를 읽는다."""
        if (
            cell_img_cv.size == 0
            or cell_img_cv.shape[0] < 5
            or cell_img_cv.shape[1] < 5
            or self._is_blank_cell(cell_img_cv)
        ):
            return ""

        image = Image.fromarray(cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2RGB))
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "<ocr>"},
                ],
            }
        ]
        inputs = self.varco_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.varco_model.device, torch.float16)

        with torch.no_grad():
            generate_ids = self.varco_model.generate(
                **inputs,
                max_new_tokens=1024,
                pad_token_id=self.varco_processor.tokenizer.eos_token_id,
            )

        raw_output = self.varco_processor.decode(
            generate_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=False,
        )
        cleaned_text = re.sub(r"<bbox>.*?</bbox>", "", raw_output)
        cleaned_text = cleaned_text.replace("<char>", "").replace("</char>", "")
        cleaned_text = cleaned_text.replace("<|im_end|>", "").replace("</s>", "").strip()
        cleaned_text = cleaned_text.replace("\\times", "횞")
        cleaned_text = cleaned_text.replace("\\div", "첨")
        cleaned_text = cleaned_text.replace("\\pm", "짹")
        cleaned_text = cleaned_text.replace("\\cdot", "쨌")
        if cleaned_text in ["VARCO VISION", "VARCOVISION", "xxx", "I", ".", "-", "_"]:
            return ""
        return cleaned_text

    def extract_text(self, metadata: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """레이아웃 결과의 bbox를 기준으로 텍스트를 채운다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"원본 PDF 파일을 찾을 수 없습니다: {file_path}")

        print(f"[TextExtractor] '{os.path.basename(file_path)}' 텍스트 추출 시작...")
        doc = fitz.open(file_path)

        for page_data in metadata["pages"]:
            page_num = page_data["page_num"]
            pdf_page = doc[page_num - 1]

            pdf_rect = pdf_page.rect
            img_width, img_height = page_data["width"], page_data["height"]
            scale_x = pdf_rect.width / img_width
            scale_y = pdf_rect.height / img_height

            img_path = page_data.get("image_path")
            img_cv = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None

            fitz_count = 0
            ocr_count = 0

            for element in page_data.get("elements", []):
                text_types = [
                    "Text",
                    "Section-header",
                    "List-item",
                    "Caption",
                    "Page-header",
                    "Page-footer",
                    "Title",
                    "Subtitle",
                ]

                if element["type"] in text_types:
                    x1, y1, x2, y2 = element["bbox"]
                    clip_rect = fitz.Rect(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                    raw_text = pdf_page.get_text("text", clip=clip_rect)
                    cleaned_text = self._clean_text(raw_text)

                    if not cleaned_text and img_cv is not None:
                        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
                        crop_img = img_cv[iy1:iy2, ix1:ix2]
                        pad = 10
                        crop_padded = cv2.copyMakeBorder(
                            crop_img,
                            pad,
                            pad,
                            pad,
                            pad,
                            cv2.BORDER_CONSTANT,
                            value=[255, 255, 255],
                        )
                        crop_upscaled = cv2.resize(
                            crop_padded,
                            None,
                            fx=2.0,
                            fy=2.0,
                            interpolation=cv2.INTER_LINEAR,
                        )

                        raw_text = self._extract_with_varco(crop_upscaled)
                        cleaned_text = self._clean_text(raw_text)
                        if cleaned_text:
                            ocr_count += 1
                    elif cleaned_text:
                        fitz_count += 1

                    element["text"] = cleaned_text
                else:
                    element["text"] = f"[{element['type']} Image]"

            yolo_elements = page_data.get("elements", [])
            missed_elements = self._sweep_missed_texts(pdf_page, yolo_elements, scale_x, scale_y)

            if missed_elements:
                page_data["elements"].extend(missed_elements)
                page_data["elements"] = sorted(page_data["elements"], key=lambda item: item["bbox"][1])
                for index, element in enumerate(page_data["elements"]):
                    element["id"] = index + 1
                print(f"   {page_num}페이지: 누락 텍스트 {len(missed_elements)}개 복구 및 재정렬 완료")

            print(
                f"   - {page_num}페이지: 추출 완료 "
                f"(디지털 텍스트 {fitz_count}개 / OCR 텍스트 {ocr_count}개)"
            )

        doc.close()
        return metadata

    def release_model(self) -> None:
        """텍스트 추출이 끝난 뒤 VARCO 자원을 해제한다."""
        if self.varco_model is None and self.varco_processor is None:
            return

        if self.use_shared_varco:
            self.varco_model = None
            self.varco_processor = None
            release_shared_varco_components()
            return

        self.varco_model = None
        self.varco_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[TextExtractor] VARCO 메모리 해제 완료")

    def _sweep_missed_texts(self, pdf_page, yolo_elements, scale_x, scale_y):
        """YOLO가 놓친 PDF 텍스트 블록을 보강한다."""
        missed_elements = []
        fitz_blocks = pdf_page.get_text("blocks")

        for block in fitz_blocks:
            bx1, by1, bx2, by2, text = block[:5]

            if block[6] == 1 or not text.strip():
                continue

            is_missed = True
            for element in yolo_elements:
                yx1, yy1, yx2, yy2 = element["bbox"]
                px1, py1, px2, py2 = yx1 * scale_x, yy1 * scale_y, yx2 * scale_x, yy2 * scale_y

                inter_x1 = max(bx1, px1)
                inter_y1 = max(by1, py1)
                inter_x2 = min(bx2, px2)
                inter_y2 = min(by2, py2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    is_missed = False
                    break

            if is_missed:
                preview = self._clean_text(text)[:20]
                print(f"   [탐색기] 비전 모델이 놓친 텍스트 발견: '{preview}...'")
                missed_elements.append(
                    {
                        "type": "Text",
                        "bbox": [bx1 / scale_x, by1 / scale_y, bx2 / scale_x, by2 / scale_y],
                        "confidence": 1.0,
                        "text": self._clean_text(text),
                        "crop_path": None,
                    }
                )

        return missed_elements
