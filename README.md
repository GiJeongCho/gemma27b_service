# Gemma 3 27B LLM API Server

Gemma 3 27B IT 모델을 On-Premise GPU 환경에서 서빙하는 FastAPI 기반 추론 서버입니다.

## 프로젝트 구조

```
LLM/
├── src/
│   ├── api.py                  # FastAPI 엔트리포인트 (일반/스트리밍 엔드포인트)
│   └── v1/
│       ├── service.py          # 모델 추론 서비스 레이어
│       └── utils/
│           └── model_loader.py # 로컬 모델 로딩 유틸리티
├── test/
│   └── model_infer.py          # 모델 로드 및 추론 테스트 스크립트
├── Dockerfile
├── pyproject.toml
└── README.md
```

## 주요 기능

- **일반 추론** (`POST /generate`): 전체 결과를 한 번에 반환
- **스트리밍 추론** (`POST /generate/stream`): SSE(Server-Sent Events) 방식으로 토큰 단위 실시간 반환
- **헬스 체크** (`GET /health`): 모델 로드 상태, GPU 정보 확인

## 기술 스택

- **모델**: Google Gemma 3 27B IT (`bfloat16`, `device_map=auto`)
- **프레임워크**: FastAPI + Uvicorn
- **추론**: Hugging Face Transformers, PyTorch 2.6+ (CUDA 12.4)
- **패키지 관리**: uv

## 실행 방법

### 로컬 실행

```bash
# 의존성 설치
uv sync

# 서버 실행 (기본 포트: 8092)
uv run uvicorn src.api:app --host 0.0.0.0 --port 8092 --reload
```

### Docker 실행

```bash
# 이미지 빌드
docker build -t llm-server .

# 컨테이너 실행 (GPU 필요)
docker run --gpus all -p 8092:8092 \
  -v /path/to/models:/app/src/resources/model:Z \
  llm-server
```

## API 사용 예시

### 일반 추론

```bash
curl -X POST http://localhost:8092/generate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "회의록을 요약해줘.",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "system_prompt": "당신은 한국어 회의록 요약 전문가입니다."
  }'
```

### 스트리밍 추론

```bash
curl -N -X POST http://localhost:8092/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "회의록을 요약해줘.",
    "max_new_tokens": 512
  }'
```

## 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `GEMMA_MODEL_PATH` | 로컬 모델 디렉토리 경로 | `src/resources/model/models--google--gemma-3-27b-it/snapshots/...` |

## 참고 사항

- 모델 파일(safetensors 등)은 `.gitignore`에 의해 Git 추적에서 제외됩니다.
- GPU VRAM 약 54GB 이상 필요 (bfloat16 기준).
- 모델은 서버 시작 시(lifespan) 자동으로 GPU에 로드됩니다.


## 시작

cd /home/pps-nipa/PoC/fish/LLM
source .venv/bin/activate
(uv run) uvicorn src.api:app --host 0.0.0.0 --port 8092 --reload