import os
import re
import json
import cv2
import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class TextExtractor:
    """
    [하이브리드 텍스트 추출 엔진]
    1. 디지털 PDF: PyMuPDF(fitz)로 100% 정확도와 고속 추출 (1차 스캔)
    2. 스캔본 PDF: 추출된 텍스트가 없으면 VARCO 모델을 활용한 정밀 OCR 수행 (2차 폴백)
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"🚨 PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        # 1. PDF 문서 로드
        self.doc = fitz.open(pdf_path)
        print(f"📖 [TextExtractor] '{os.path.basename(pdf_path)}' 로드 완료 (총 {len(self.doc)}페이지)")

        # 2. VARCO OCR 모델 로드 (조원 코드 이식)
        print("🤖 [TextExtractor] VARCO-VISION OCR 모델 로드 중 (스캔본 대비용)...")
        self.VARCO_MODEL_ID = "NCSOFT/VARCO-VISION-2.0-1.7B-OCR"
        self.varco_processor = AutoProcessor.from_pretrained(self.VARCO_MODEL_ID)
        self.varco_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.VARCO_MODEL_ID, 
            torch_dtype=torch.float16, 
            attn_implementation="sdpa", 
            device_map="auto"
        )
        self.varco_model.eval()
        print("✅ [TextExtractor] VARCO 모델 로드 완료!\n")

    def _clean_text(self, text: str) -> str:
        """추출된 텍스트의 불필요한 줄바꿈이나 공백을 정리합니다."""
        if not text:
            return ""
        return " ".join(text.replace("\n", " ").split())

    def _is_blank_cell(self, cell_img_cv) -> bool:
        """이미지가 실질적으로 비어있는지(여백) 검사하여 환각(Hallucination) 방지"""
        gray = cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        if crop.size == 0: return True
        return np.std(crop) < 5.0

    def _extract_with_varco(self, cell_img_cv) -> str:
        """VARCO 모델을 이용해 이미지 조각에서 텍스트를 읽어냅니다."""
        # 1. 여백 검사 (비어있으면 무시)
        if cell_img_cv.size == 0 or cell_img_cv.shape[0] < 5 or cell_img_cv.shape[1] < 5 or self._is_blank_cell(cell_img_cv):
            return ""

        # 2. VARCO 추론 준비
        image = Image.fromarray(cv2.cvtColor(cell_img_cv, cv2.COLOR_BGR2RGB))
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "<ocr>"}]}]
        
        inputs = self.varco_processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.varco_model.device, torch.float16)

        # 3. 텍스트 생성
        with torch.no_grad():
            generate_ids = self.varco_model.generate(**inputs, max_new_tokens=1024, pad_token_id=self.varco_processor.tokenizer.eos_token_id)
        
        raw_output = self.varco_processor.decode(generate_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
        cleaned_text = re.sub(r'<bbox>.*?</bbox>', '', raw_output).replace('<char>', '').replace('</char>', '').replace('<|im_end|>', '').replace('</s>', '').strip()
        
        # 4. 수식 기호 치환 및 환각 필터링
        cleaned_text = cleaned_text.replace('\\times', '×').replace('\\div', '÷').replace('\\pm', '±').replace('\\cdot', '·')
        if cleaned_text in ["VARCO VISION", "VARCOVISION", "xxx", "I", ".", "-", "_"]: 
            return ""
            
        return cleaned_text

    def extract_text(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        print("🔍 [TextExtractor] 하이브리드(PyMuPDF + VARCO OCR) 텍스트 추출 시작...")
        
        for page_data in metadata["pages"]:
            page_num = page_data["page_num"]
            pdf_page = self.doc[page_num - 1] 
            
            # 스케일링 비율 계산용 변수
            pdf_rect = pdf_page.rect
            img_width, img_height = page_data["width"], page_data["height"]
            scale_x = pdf_rect.width / img_width
            scale_y = pdf_rect.height / img_height
            
            # OCR을 위한 고해상도 전체 이미지 로드 (OpenCV)
            img_path = page_data.get("image_path")
            img_cv = cv2.imread(img_path) if img_path and os.path.exists(img_path) else None
            
            fitz_count = 0
            ocr_count = 0
            
            for el in page_data.get("elements", []):
                text_types = ["Text", "Section-header", "List-item", "Caption", "Page-header", "Page-footer", "Title", "Subtitle"]
                
                if el["type"] in text_types:
                    x1, y1, x2, y2 = el["bbox"]
                    
                    # [1차 시도]: PyMuPDF (fitz) 로 빠른 추출
                    clip_rect = fitz.Rect(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
                    raw_text = pdf_page.get_text("text", clip=clip_rect)
                    cleaned_text = self._clean_text(raw_text)
                    
                    # [2차 폴백]: fitz가 실패했고(스캔본), 이미지가 로드되어 있다면 VARCO OCR 가동!
                    if not cleaned_text and img_cv is not None:
                        # 좌표를 int로 변환하여 이미지 자르기
                        ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])
                        crop_img = img_cv[iy1:iy2, ix1:ix2]
                        
                        # 여백 덧대기 (OCR 인식률 향상 목적, 조원 코드 차용)
                        PAD = 10
                        crop_padded = cv2.copyMakeBorder(crop_img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        # 해상도 2배 뻥튀기
                        crop_upscaled = cv2.resize(crop_padded, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
                        
                        cleaned_text = self._extract_with_varco(crop_upscaled)
                        if cleaned_text:
                            ocr_count += 1
                    else:
                        if cleaned_text:
                            fitz_count += 1

                    el["text"] = cleaned_text
                else:
                    el["text"] = f"[{el['type']} Image]"

            print(f"   - {page_num}페이지: 추출 완료 (디지털 텍스트 {fitz_count}개 / OCR 텍스트 {ocr_count}개)")
            
        return metadata