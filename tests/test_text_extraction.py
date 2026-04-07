import os
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.text_extractor import TextExtractor

def test_text_extraction(pdf_path, json_path):
    if not os.path.exists(json_path):
        print("🚨 metadata.json이 없습니다. 비전 파이프라인을 먼저 돌려주세요.")
        return

    # 2. 비전 엔진 결과물 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 3. 텍스트 추출기 가동!
    extractor = TextExtractor(pdf_path)
    enriched_metadata = extractor.extract_text(metadata)

    # 4. 결과 저장 (텍스트가 추가된 최종 완성본)
    output_path = json_path.replace("metadata.json", "metadata_with_text.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_metadata, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 텍스트 추출 완료! 최종 데이터가 저장되었습니다: {output_path}")
    
    # 5. 결과 살짝 엿보기 (1페이지의 제목/텍스트만 출력)
    print("\n👀 [1페이지 추출 결과 미리보기]")
    for el in enriched_metadata["pages"][0]["elements"][:5]:
        print(f"[{el['id']}] {el['type']}: {el.get('text', '')[:50]}...")

if __name__ == "__main__":
    SAMPLE_PATH_LIST = [("data/raw/calculator_srs_final.pdf", "data/output/calculator_srs_final.pdf/metadata.json"),
                        ("data/raw/aiReadable.pdf", "data/output/aiReadable.pdf/metadata.json")]
    
    for pdf_path, json_path in SAMPLE_PATH_LIST:
        test_text_extraction(pdf_path, json_path)