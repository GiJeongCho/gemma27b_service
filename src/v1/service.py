"""Gemma 3 27B 모델 서비스 레이어.

모델 로딩과 추론 로직을 캡슐화합니다.
"""

import os
import time
import logging
from typing import Optional, Generator

import torch
from threading import Thread
from transformers import TextIteratorStreamer

from src.v1.utils.model_loader import load_model, DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


class GemmaService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_path = os.getenv("GEMMA_MODEL_PATH", DEFAULT_MODEL_PATH)

    def load(self):
        """로컬 모델을 GPU에 로드합니다."""
        self.model, self.processor = load_model(self.model_path)

    def _build_inputs(self, message: str, system_prompt: Optional[str] = None):
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": message}],
        })

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        return inputs

    def _check_ready(self):
        if self.model is None or self.processor is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

    def generate(
        self,
        message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """단일 메시지에 대한 추론 (전체 결과를 한 번에 반환)."""
        self._check_ready()

        inputs = self._build_inputs(message, system_prompt)
        input_len = inputs["input_ids"].shape[-1]

        t0 = time.time()
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        elapsed = time.time() - t0

        new_tokens = generation.shape[-1] - input_len
        output_tokens = generation[0][input_len:]
        text = self.processor.decode(output_tokens, skip_special_tokens=True)

        return {
            "text": text,
            "usage": {
                "input_tokens": input_len,
                "output_tokens": new_tokens,
                "generation_time": round(elapsed, 2),
                "tokens_per_sec": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
            },
        }

    def generate_stream(
        self,
        message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """스트리밍 추론 (토큰 단위로 yield)."""
        self._check_ready()

        inputs = self._build_inputs(message, system_prompt)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        thread = Thread(target=self._run_generation, args=(gen_kwargs,))
        thread.start()

        yield from streamer
        thread.join()

    def _run_generation(self, gen_kwargs: dict):
        with torch.inference_mode():
            self.model.generate(**gen_kwargs)

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "device": str(self.model.device) if self.model else None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
