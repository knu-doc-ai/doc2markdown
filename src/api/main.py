import time
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# FastAPI 앱 생성
app = FastAPI(title="LG전자 문서 변환 AI 에이전트 API")

# HTML 템플릿 폴더 연결
templates = Jinja2Templates(directory="src/api/templates")

@app.get("/")
async def serve_frontend(request: Request):
    """메인 웹 페이지(3분할 화면)를 서비스합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/convert")
async def convert_document(file: UploadFile = File(...)):
    """프론트엔드에서 파일을 받아 AI 파이프라인으로 넘기고 결과를 반환합니다."""
    print(f"업로드된 파일 수신: {file.filename}")
    
    # TODO: 여기에 pipeline.run(file) 코드가 들어갈 예정입니다.
    time.sleep(2) # AI 처리 시간 가짜 로딩
    
    # 뼈대용 가짜(Dummy) 마크다운 데이터
    dummy_markdown = f"""# {file.filename} 변환 결과

## 1.1. 추진 배경
이 텍스트는 FastAPI 백엔드에서 생성되어 넘어온 가짜 데이터입니다. 
가운데 에디터에서 내용을 수정하면 오른쪽 화면에 실시간으로 반영됩니다!

![그래프 대체 텍스트](./assets/sample_chart.png)

## 1.2. 기술 스택 비교표
| 구분 | 기술명 | 용도 |
|---|---|---|
| **Vision** | Marker | PDF 영역 분리 및 레이아웃 분석 |
| **LLM** | GPT-4o | 다단 문맥 정제 및 표 변환 |
| **Backend** | FastAPI | 빠르고 가벼운 AI REST API 구축 |
"""
    # 프론트엔드로 변환된 텍스트를 JSON 형태로 반환
    return {"status": "success", "markdown": dummy_markdown}

# 터미널에서 직접 실행할 때 사용하는 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)