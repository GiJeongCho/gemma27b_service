
import os
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

HF_TOKEN = "os.getenv("HF_TOKEN")"

MODEL_ID = "google/gemma-3-27b-it"
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


def install_dependencies():
    """필요한 라이브러리가 없으면 설치 안내 메시지를 출력합니다."""
    try:
        import transformers
        import accelerate
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("필요한 라이브러리가 설치되지 않았습니다. 터미널에서 아래 명령어를 실행하세요:")
        print("pip install -U torch transformers accelerate bitsandbytes")
        exit(1)


def download_model():
    print(f"[{MODEL_ID}] 모델 다운로드 시작... (시간이 걸릴 수 있습니다)")
    print(f"저장 경로: {MODEL_CACHE_DIR}\n")

    try:
        AutoProcessor.from_pretrained(
            MODEL_ID, token=HF_TOKEN, cache_dir=MODEL_CACHE_DIR
        )
        print("프로세서 다운로드 완료!")
    except Exception as e:
        print("\n[오류] 모델 접근 권한이 없거나 토큰이 잘못되었습니다.")
        print("Hugging Face 웹사이트에서 모델 사용 동의를 했는지 확인하세요.")
        raise e

    Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        cache_dir=MODEL_CACHE_DIR,
    )
    print("\n모델 다운로드 완료!")
    print(f"저장 위치: {MODEL_CACHE_DIR}")


if __name__ == "__main__":
    install_dependencies()
    download_model()
