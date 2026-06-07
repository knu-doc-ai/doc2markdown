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
│   │   ├── assembly/stages/enrichment/ # [3. AI Agent Core] Assembly IR LLM 보강
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

# 1. 필수 라이브러리 일괄 설치
pip install -r requirements.txt

# 2. GPU(CUDA 11.8 기준) 사용을 위한 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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

### 4. 참고: 노트북/리눅스 Ollama 설정

2GB VRAM 노트북에서는 `transformers`로 `Qwen/Qwen3-0.6B`를 직접 올리면 CUDA OOM이 날 수 있습니다. PyTorch 로딩, 긴 semantic prompt, 생성 중 KV cache 때문에 실제 요구 메모리가 2GB를 넘을 수 있습니다.

노트북에서는 Ollama의 GGUF 양자화 모델을 먼저 사용합니다. 이 방식은 OpenAI 같은 외부 API가 아니라 내 PC의 로컬 Ollama 서버(`127.0.0.1:11434`)를 호출합니다.

Windows PowerShell에서 Ollama를 설치합니다.

~~~powershell
irm https://ollama.com/install.ps1 | iex
~~~

Linux에서는 공식 설치 스크립트를 사용합니다.

~~~bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
sudo systemctl start ollama
sudo systemctl status ollama
~~~

설치 후 새 터미널을 열고 정상 설치 여부를 확인합니다.

~~~powershell
ollama --version
~~~

semantic/content 테스트용 Qwen3 0.6B Q4_0 GGUF 모델을 받습니다.

~~~powershell
ollama pull hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0
~~~

모델을 한 번 실행해 서버와 모델을 깨웁니다. 대화 프롬프트가 뜨면 `/bye`로 나와도 됩니다.

~~~powershell
ollama run hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0
~~~

서버가 떠 있는지 확인합니다.

~~~powershell
curl http://127.0.0.1:11434/api/tags
~~~

기본 Ollama 로컬 서버를 쓰면 `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_API_KEY`는 따로 적지 않아도 됩니다. 다른 로컬 주소를 써야 할 때만 아래 값을 `.env` 뒤쪽에 주석 해제해서 넣습니다.
기존 `# 6. 로컬 오픈웨이트 LLM 후처리 설정` 부분은 주석처리합니다.

~~~env
LOCAL_LLM_BASE_URL="http://127.0.0.1:11434/v1"
LOCAL_LLM_API_KEY="ollama"
LOCAL_LLM_SEMANTIC_MODEL_ID="hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0"
LOCAL_LLM_CONTENT_MODEL_ID="hf.co/ggml-org/Qwen3-0.6B-GGUF:Q4_0"
~~~

Ollama backend에서는 semantic 후보 30개 안팎을 한 번에 처리하도록 `LLM_SEMANTIC_BATCH_SIZE="32"`를 기본값으로 둡니다. 응답 JSON이 흔들리거나 처리가 너무 느리면 16, 8, 4 순서로 낮춥니다.

content 보정은 semantic보다 배치 크기에 민감합니다. Qwen3 0.6B GGUF에서 `LLM_CONTENT_BATCH_SIZE="16"`처럼 크게 잡으면 JSON은 정상이어도 모델이 대부분 원문을 그대로 반환할 수 있습니다. content/all 모드에서는 `LLM_CONTENT_BATCH_SIZE="2"`부터 시작하고, 빠른 확인이 필요하면 1로 낮춥니다.

content 응답 timeout은 `LLM_REQUEST_TIMEOUT_SECONDS="60"`을 기본으로 사용합니다. 대부분의 정상 batch가 몇 초 안에 끝나고, 오래 걸리는 요청은 JSON 반복이나 비정상 생성으로 빠질 가능성이 크기 때문입니다.
content는 짧은 JSON만 필요하므로 `LLM_CONTENT_MAX_NEW_TOKENS`도 256~512 범위에서 시작합니다. transformers 경로에서 OOM이 난다면 Ollama backend를 사용해야 합니다.
