import os
import fitz
from PIL import Image
from typing import Dict, Any, List

class FilePreProcessor:
    """
    [1단계: Input Layer] 
    사용자가 업로드한 문서(PDF, 이미지)를 AI가 분석하기 좋은 형태(고화질 이미지 + 기초 텍스트)로 
    잘게 쪼개고 전처리하는 모듈입니다.
    """
    def __init__(self, temp_dir: str = "data/temp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def process(self, file_path: str) -> Dict[str, Any]:
        print(f"📥 [Ingestion] 파일 전처리 시작: {file_path}")
        
        self._validate_file(file_path)
        
        file_ext = os.path.splitext(file_path)[-1].lower()
        file_name = os.path.basename(file_path)
        
        processed_data = {
            "file_name": file_name,
            "file_type": file_ext,
            "total_pages": 0,
            "pages": []
        }

        if file_ext == '.pdf':
            processed_data["pages"] = self._process_pdf(file_path, file_name)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            processed_data["pages"] = self._process_image(file_path, file_name)
        
        processed_data["total_pages"] = len(processed_data["pages"])
        
        print(f"✅ [Ingestion] 전처리 완료! 총 {processed_data['total_pages']}장 변환됨.")
        return processed_data

    def _validate_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"🚨 파일을 찾을 수 없습니다: {file_path}")
        
        valid_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
        ext = os.path.splitext(file_path)[-1].lower()
        if ext not in valid_extensions:
            raise ValueError(f"🚨 지원하지 않는 파일 형식입니다. (지원: {valid_extensions})")

    def _process_pdf(self, file_path: str, file_name: str) -> List[Dict[str, Any]]:
        """PyMuPDF(fitz)만 사용하여 텍스트를 추출하고 고화질 이미지로 렌더링합니다."""
        pages_data = []
        doc = fitz.open(file_path)
        
        for page_num_zero_indexed in range(len(doc)):
            page = doc.load_page(page_num_zero_indexed)
            page_num = page_num_zero_indexed + 1
            
            # 1. 텍스트 추출
            raw_text = page.get_text("text").strip()
            
            # 2. 고화질 이미지로 렌더링 (dpi=300 설정으로 Vision 모델에 적합하게 선명도 상향)
            pix = page.get_pixmap(dpi=300)
            image_path = os.path.join(self.temp_dir, f"{file_name}_page_{page_num}.png")
            
            # PyMuPDF의 pixmap을 파일로 직접 저장
            pix.save(image_path)
            
            pages_data.append({
                "page_num": page_num,
                "image_path": image_path,
                "raw_text": raw_text,
                "width": pix.width,
                "height": pix.height
            })
            
        doc.close()
        return pages_data

    def _process_image(self, file_path: str, file_name: str) -> List[Dict[str, Any]]:
        img = Image.open(file_path)
        image_path = os.path.join(self.temp_dir, f"{file_name}_page_1.png")
        img.save(image_path, "PNG")
        
        return [{
            "page_num": 1,
            "image_path": image_path,
            "raw_text": "", 
            "width": img.width,
            "height": img.height
        }]

# ==========================================
# 모듈 테스트용 코드
# ==========================================
if __name__ == "__main__":
    sample_file = "data/raw/test.pdf"
    
    print("이 파일을 직접 실행하면 테스트가 진행됩니다.")
    processor = FilePreProcessor()
    result = processor.process(sample_file)
    print(result)