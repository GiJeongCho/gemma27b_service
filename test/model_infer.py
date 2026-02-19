"""Gemma 3 27B 로컬 모델 로드 및 추론 테스트 스크립트."""

import os
import sys
import time
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "src",
    "resources",
    "model",
    "models--google--gemma-3-27b-it",
    "snapshots",
    "005ad3404e59d6023443cb575daa05336842228a",
)
LOCAL_MODEL_PATH = os.path.normpath(LOCAL_MODEL_PATH)


def load_model():
    print(f"모델 경로: {LOCAL_MODEL_PATH}")

    if not os.path.isdir(LOCAL_MODEL_PATH):
        print(f"[오류] 모델 디렉토리가 존재하지 않습니다: {LOCAL_MODEL_PATH}")
        sys.exit(1)

    print("프로세서 로딩 중...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        LOCAL_MODEL_PATH,
        force_download=False,
        local_files_only=True,
    )
    print(f"프로세서 로딩 완료 ({time.time() - t0:.1f}s)")

    print("모델 로딩 중... (시간이 걸릴 수 있습니다)")
    t0 = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        force_download=False,
        local_files_only=True,
    ).eval()
    print(f"모델 로딩 완료 ({time.time() - t0:.1f}s)")
    print(f"device_map: {model.hf_device_map}\n")

    return model, processor


def generate(model, processor, user_message: str, max_new_tokens: int = 256) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": user_message}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    t0 = time.time()
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
    elapsed = time.time() - t0
    new_tokens = generation.shape[-1] - input_len
    print(f"[생성 완료] {new_tokens} tokens / {elapsed:.1f}s ({new_tokens / elapsed:.1f} tok/s)")

    output_tokens = generation[0][input_len:]
    return processor.decode(output_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    model, processor = load_model()

    test_prompt = "안녕하세요! 간단한 자기소개를 해주세요."
    print(f"[테스트 프롬프트] {test_prompt}\n")

    response = generate(model, processor, test_prompt)
    print("-" * 50)
    print(f"[Gemma 3 응답]\n{response}")
    print("-" * 50)
