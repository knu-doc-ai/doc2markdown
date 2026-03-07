# 📄 [LG전자 산학협력] 비전 AI 기반의 레이아웃 보존형 문서-Markdown 자동 변환 에이전트

> **Development of Layout-Aware Document-to-Markdown AI Agent**
> 
> **2026 SW중심대학 학부생 중심 산학협력 프로젝트 (LG전자)**

## 📌 프로젝트 소개

본 프로젝트는 기업 내 수많은 문서(PDF, 보고서 등)를 LLM 학습 데이터나 기술 블로그로 활용하기 위해 Markdown 형식으로 자동 변환하는 AI 에이전트를 개발합니다. 기존의 단순 텍스트 추출 방식이 가진 한계를 극복하기 위해, 시각적 정보(Layout)와 텍스트 정보(Content)를 결합하여 이해하는 '멀티모달' 접근법을 사용합니다.

**🎯 최종 목표:** 원본 문서의 시각적 레이아웃을 90% 이상 보존하며 배포 가능한 수준의 Markdown 파일을 생성

## 📂 디렉토리 구조 (Directory Structure)

프로젝트는 크게 4단계 레이어(입력 -> 시각 구조 분석 -> AI 에이전트 코어 -> 출력)에 맞추어 모듈화되어 있습니다.

<pre>
doc2markdown/
│
├── data/                       # 📁 데이터 저장소 (Git 업로드 제외)
│   ├── raw/                    # 사용자가 업로드한 원본 PDF 및 이미지
│   ├── temp/                   # 전처리 중 생성되는 임시 파일
│   └── output/                 # 최종 완성된 .md 파일과 assets ZIP 폴더
│
├── src/                        # 💻 메인 소스 코드
│   ├── ui/                     # [4. Output Layer] 웹 인터페이스 관련
│   │   ├── app.py              # Streamlit/Gradio 대시보드 실행 파일
│   │   └── components.py       # 원본-프리뷰 대조 및 수동 보정 UI 컴포넌트
│   │
│   ├── modules/                # 핵심 AI 엔진 모듈
│   │   ├── ingestion.py        # [1. Input Layer] PDF 파싱 및 전처리
│   │   ├── vision_engine.py    # [2. Visual Analysis] 문서 구조 분석 및 영역 분리 (제목, 다단, 표, 그림)
│   │   ├── llm_core.py         # [3. AI Agent Core] 표 변환(MD Table), 본문 정제, Alt-text 생성
│   │   └── assembler.py        # [4. Output Layer] 최종 마크다운 조립
│   │
│   ├── utils/                  # 🛠 공통 유틸리티
│   │   ├── prompts.py          # LLM 프롬프트 템플릿
│   │   ├── eval_metrics.py     # 90% 레이아웃 보존율 검증을 위한 자체 평가지표
│   │   └── config.py           # API 키 및 환경 변수 설정
│   │
│   └── pipeline.py             # 각 모듈을 연결하는 파이프라인 오케스트레이터
│
├── tests/                      # 🧪 단위 테스트 (Unit Tests)
├── .env.example                # 환경 변수 템플릿 (실제 .env는 Git 제외)
├── .gitignore                  # Git 추적 제외 목록
├── requirements.txt            # 의존성 패키지 목록
└── README.md                   # 프로젝트 개요 및 실행 가이드
</pre>

## 🚀 시작하기 (Getting Started)

### 1. 환경 설정

가상환경을 생성하고 활성화한 뒤, 필요한 패키지를 설치합니다.

~~~bash
# 가상환경 생성 및 활성화 (예: conda)
conda create -n lg_agent python=3.10
conda activate lg_agent

# 패키지 설치
pip install -r requirements.txt
~~~

### 2. 환경 변수 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고, 필요한 API 키를 입력합니다.

~~~bash
cp .env.example .env
# .env 파일 내부에 OPENAI_API_KEY, ANTHROPIC_API_KEY 등 입력
~~~

### 3. 애플리케이션 실행
아래 명령어를 통해 웹 대시보드를 실행합니다.

~~~bash
streamlit run src/ui/app.py
~~~