import json
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.v1.service import GemmaService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

service = GemmaService()


def kst_now() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Gemma 3 27B 모델을 GPU 메모리에 로드합니다...")
    try:
        service.load()
        logger.info("모델 로드 완료. API 서비스 준비됨.")
    except Exception as e:
        logger.error("모델 로드 실패: %s", e)
    yield


app = FastAPI(
    title="Gemma 3 27B LLM API",
    description="Gemma 3 27B IT 모델 기반 텍스트 생성 API (일반 / 스트리밍)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    message: str = Field(..., description="사용자 입력 메시지")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="최대 생성 토큰 수")
    min_new_tokens: int = Field(1, ge=1, le=4096, description="최소 생성 토큰 수")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="샘플링 온도 (0이면 greedy)")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="누적 확률 기반 nucleus sampling")
    top_k: int = Field(64, ge=0, le=1000, description="상위 K개 토큰만 샘플링 (0이면 비활성)")
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0, description="반복 억제 패널티 (1.0이면 비활성)")
    system_prompt: Optional[str] = Field(None, description="시스템 프롬프트")


@app.get("/health")
def health():
    return {"status": "ok", "model": service.get_status(), "server_time_kst": kst_now()}


@app.post("/generate")
def generate(req: GenerateRequest):
    """전체 결과를 한 번에 반환합니다."""
    if service.model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    request_time = kst_now()
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"[{request_time}] POST /generate")
    print(f"  message: {req.message[:200]}{'...' if len(req.message) > 200 else ''}")
    print(f"  max_new_tokens={req.max_new_tokens}, min_new_tokens={req.min_new_tokens}, temperature={req.temperature}")
    print(f"  top_p={req.top_p}, top_k={req.top_k}, repetition_penalty={req.repetition_penalty}")
    if req.system_prompt:
        print(f"  system_prompt: {req.system_prompt[:100]}{'...' if len(req.system_prompt) > 100 else ''}")

    try:
        result = service.generate(
            message=req.message,
            max_new_tokens=req.max_new_tokens,
            min_new_tokens=req.min_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            system_prompt=req.system_prompt,
        )
    except Exception as e:
        logger.error("생성 실패: %s", e, exc_info=True)
        print(f"[{kst_now()}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    response_time = kst_now()
    total_elapsed = round(time.time() - t0, 2)

    response = {
        **result,
        "timestamp": {
            "request_kst": request_time,
            "response_kst": response_time,
            "total_elapsed_sec": total_elapsed,
        },
    }

    print(f"[{response_time}] 응답 완료 ({total_elapsed}s)")
    print(f"  usage: {result.get('usage', {})}")
    print(f"  text: {result.get('text', '')[:300]}{'...' if len(result.get('text', '')) > 300 else ''}")
    print(f"{'='*60}\n")

    return response


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    """SSE(Server-Sent Events) 스트리밍으로 토큰 단위 반환합니다."""
    if service.model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    def event_generator():
        request_time = kst_now()
        t0 = time.time()
        token_count = 0
        full_text_parts = []

        print(f"\n{'='*60}")
        print(f"[{request_time}] POST /generate/stream")
        print(f"  message: {req.message[:200]}{'...' if len(req.message) > 200 else ''}")
        print(f"  max_new_tokens={req.max_new_tokens}, min_new_tokens={req.min_new_tokens}, temperature={req.temperature}")
        print(f"  top_p={req.top_p}, top_k={req.top_k}, repetition_penalty={req.repetition_penalty}")
        if req.system_prompt:
            print(f"  system_prompt: {req.system_prompt[:100]}{'...' if len(req.system_prompt) > 100 else ''}")

        try:
            for token_text in service.generate_stream(
                message=req.message,
                max_new_tokens=req.max_new_tokens,
                min_new_tokens=req.min_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                system_prompt=req.system_prompt,
            ):
                token_count += 1
                full_text_parts.append(token_text)
                chunk = json.dumps({"token": token_text}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"

        except Exception as e:
            error = json.dumps({"error": str(e)}, ensure_ascii=False)
            print(f"[{kst_now()}] STREAM ERROR: {e}")
            yield f"data: {error}\n\n"

        elapsed = round(time.time() - t0, 2)
        response_time = kst_now()
        done = json.dumps({
            "done": True,
            "usage": {
                "output_tokens": token_count,
                "generation_time": elapsed,
                "tokens_per_sec": round(token_count / elapsed, 1) if elapsed > 0 else 0,
            },
            "timestamp": {
                "request_kst": request_time,
                "response_kst": response_time,
                "total_elapsed_sec": elapsed,
            },
        }, ensure_ascii=False)
        yield f"data: {done}\n\n"

        full_text = "".join(full_text_parts)
        print(f"[{response_time}] 스트리밍 완료 ({elapsed}s, {token_count} tokens)")
        print(f"  text: {full_text[:300]}{'...' if len(full_text) > 300 else ''}")
        print(f"{'='*60}\n")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8092, reload=False)
