# 1. 빌드 스테이지
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. 실행 스테이지
FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV GEMMA_MODEL_PATH=/app/src/resources/model/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a

EXPOSE 8092
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8092"]
