import os
import sys
import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# src 폴더 내부의 모듈(modules.*)을 직접 import 하므로 Python Path에 추가해 줍니다.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.pipeline import DocumentToMarkdownPipeline

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Document Conversion API", version="1.0.0")

# 배포 환경/동일 EC2 등을 고려하여 환경변수를 통한 CORS 오리진 설정
# 로컬 개발 시 FE(5173), BE(8000) 접근 허용
origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000")
CORS_ORIGINS = [orig.strip() for orig in origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = DocumentToMarkdownPipeline()

@app.post("/ai/convert")
async def convert_document(file: UploadFile = File(...), format: str = Form("markdown")):
    """
    백엔드에서 업로드된 원본 PDF를 수신받고, DocumentToMarkdown 파이프라인을 실행하여
    최종 처리된 Markdown과 추출된 이미지들의 base64 데이터를 반환합니다.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_pdf_path = ""
    try:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_pdf_path = temp_file.name
        
        logger.info(f"Received file: {file.filename}, saved temporarily at {temp_pdf_path}")
        
        # 파이프라인 실행
        result = pipeline.run(file_path=temp_pdf_path)
        
        if result.get("status") != "success":
            logger.error("Pipeline status is not success")
            raise HTTPException(status_code=500, detail="Failed to convert document in pipeline.")
        
        # Pipeline 결과물에서 markdown_result 획득
        md_result = result.get("markdown_result")
        if not md_result:
            logger.warning("Pipeline succeeded but no markdown_result generated.")
            raise HTTPException(status_code=500, detail="Markdown rendering failed inside pipeline.")
        
        # BE 기대 형태: {"markdown": "...", "images": [{"filename": "...", "data": "<base64_string>"}]}
        # md_result는 SerializedRenderIR 형태의 dict이므로 markdown 필드를 추출합니다.
        markdown_text = md_result.get("markdown", "") if isinstance(md_result, dict) else str(md_result)
        images = []
        
        try:
            import re
            import base64
            
            # 마크다운 텍스트 내에서 로컬 이미지 경로 추출, Base64 변환 및 경로 치환 수행
            def process_image_match(match):
                alt_text = match.group(1)
                img_path = match.group(2).strip()
                
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    with open(img_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    filename = os.path.basename(img_path)
                    
                    if not any(img.get("filename") == filename for img in images):
                        images.append({"filename": filename, "data": encoded_string})
                        
                    # BE/FE에서 바로 사용할 수 있도록 로컬 절대 경로를 순수 파일명으로 치환
                    return f"![{alt_text}]({filename})"
                else:
                    logger.warning(f"Image path found in markdown but file not found on disk: {img_path}")
                    return match.group(0)
                    
            markdown_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', process_image_match, markdown_text)
            
        except Exception as img_err:
            logger.error(f"Image extraction block failed: {img_err}", exc_info=True)
            # 이미지 추출 실패시에도 마크다운 변환 프로세스는 완전히 죽지 않도록 예외 처리
        
        logger.info(f"Successfully converted {file.filename}.")
        
        return {
            "markdown": markdown_text,
            "images": images
        }
            
    except Exception as e:
        logger.error(f"Error processing conversion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 임시 파일 삭제
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                logger.error(f"Failed to remove temp file {temp_pdf_path}: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "AI server is running"}

if __name__ == "__main__":
    import uvicorn
    # 배포 시 환경 변수로 HOST 설정 (AWS 인스턴스 내 로컬 통신 위해 0.0.0.0 또는 127.0.0.0 활용)
    host = os.getenv("AI_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("AI_SERVER_PORT", "8001"))
    
    uvicorn.run("main:app", host=host, port=port, reload=True)
