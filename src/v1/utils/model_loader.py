"""Gemma 3 27B 로컬 모델 로드 유틸리티.

서비스에서 import하여 사용:
    from src.v1.utils.model_loader import load_model, DEFAULT_MODEL_PATH
"""

import os
import time
import logging
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

logger = logging.getLogger(__name__)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(
    _UTILS_DIR, "..", "..", "resources", "model",
    "models--google--gemma-3-27b-it",
    "snapshots",
    "005ad3404e59d6023443cb575daa05336842228a",
))


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """로컬에 저장된 모델과 프로세서를 GPU에 로드합니다. 네트워크 접근 없이 동작합니다."""
    logger.info("로컬 모델 로딩 중: %s", model_path)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"모델 디렉토리가 존재하지 않습니다: {model_path}")

    t0 = time.time()

    processor = AutoProcessor.from_pretrained(
        model_path,
        force_download=False,
        local_files_only=True,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        force_download=False,
        local_files_only=True,
    ).eval()

    elapsed = time.time() - t0
    logger.info("모델 로딩 완료 (%.1fs) | device_map: %s", elapsed, model.hf_device_map)
    return model, processor
