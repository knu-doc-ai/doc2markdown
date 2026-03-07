import os
from PIL import Image
from docling.document_converter import DocumentConverter
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
from openai import OpenAI

# 1. 초기화: 모델 및 클라이언트 로드 [cite: 10, 21]
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
doc_converter = DocumentConverter()
client = OpenAI(api_key="your api key")

def get_layout_map(image_path):
    """LayoutLMv3를 사용하여 문서의 시각적 영역(bbox)과 라벨을 추출합니다. [cite: 21, 38]"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    outputs = model(**inputs)
    
    # 예측된 라벨과 좌표값 매핑 로직 (단순화 버전)
    # 결과 예시: [{'label': 'table', 'bbox': [100, 200, 500, 400]}, {'label': 'text', ...}]
    predictions = outputs.logits.argmax(-1)
    # ... (실제 구현 시에는 predictions를 좌표 평면상의 영역으로 그룹화하는 후처리가 필요합니다)
    return predictions 

def process_table_with_gpt(table_image_path):
    """표 영역 이미지를 GPT-4o Vision에게 전달하여 MD로 변환합니다. [cite: 23, 48, 52]"""
    # 이미지를 Base64로 인코딩하여 GPT-4o API 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 표 이미지의 병합된 셀을 유지하며 마크다운 표로 변환해줘."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{table_image_path}"}}
                ],
            }
        ]
    )
    return response.choices[0].message.content

def integrated_pipeline(file_path):
    """전체 통합 파이프라인 [cite: 19, 22]"""
    # 1단계: Docling을 통한 전체 텍스트 및 기본 구조 추출 [cite: 21, 53]
    conversion_result = doc_converter.convert(file_path)
    raw_elements = conversion_result.document.export_to_dict() # 영역별 데이터 확보
    
    final_markdown = ""
    
    # 2단계: LayoutLMv3의 좌표 정보를 기준으로 요소 순회 [cite: 31, 47]
    for element in raw_elements['elements']:
        label = element.get('label') # LayoutLMv3가 판단한 라벨 기반
        
        # 표(Table)인 경우 GPT-4o Vision에게 특수 임무 부여 [cite: 23, 48]
        if label == 'table':
            print("표 감지: GPT-4o로 정밀 변환 중...")
            # 실제 구현 시 해당 좌표의 이미지를 크롭(Crop)하여 전달
            table_md = process_table_with_gpt("cropped_table.jpg")
            final_markdown += f"\n{table_md}\n"
        
        # 일반 텍스트인 경우 Docling의 추출 결과 사용 [cite: 51, 54]
        else:
            text_content = element.get('text', '')
            if label == 'header':
                final_markdown += f"\n# {text_content}\n"
            else:
                final_markdown += f"\n{text_content}\n"
                
    return final_markdown

# 실행
if __name__ == "__main__":
    result_md = integrated_pipeline("input_document.pdf")
    with open("final_output.md", "w", encoding="utf-8") as f:
        f.write(result_md)