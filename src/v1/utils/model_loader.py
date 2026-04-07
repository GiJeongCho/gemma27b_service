"""Gemma 4 31B 로컬 모델 로드 유틸리티.

서비스에서 import하여 사용:
    from src.v1.utils.model_loader import load_model, DEFAULT_MODEL_PATH
"""

import os
import time
import logging
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_SNAPSHOTS_DIR = os.path.abspath(os.path.join(
    _UTILS_DIR, "..", "..", "resources", "model",
    "models--google--gemma-4-31B-it",
    "snapshots",
))

DEFAULT_MODEL_PATH = _SNAPSHOTS_DIR
if os.path.isdir(_SNAPSHOTS_DIR):
    _snaps = os.listdir(_SNAPSHOTS_DIR)
    if _snaps:
        DEFAULT_MODEL_PATH = os.path.join(_SNAPSHOTS_DIR, _snaps[0])


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

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        force_download=False,
        local_files_only=True,
    ).eval()

    elapsed = time.time() - t0
    device_map_info = getattr(model, "hf_device_map", None)
    logger.info("모델 로딩 완료 (%.1fs) | device_map: %s", elapsed, device_map_info)
    return model, processor
