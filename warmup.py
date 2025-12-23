#!/usr/bin/env python3
"""
Warmup script to pre-download and cache all models during Docker build.
This reduces cold start time in serverless environments like RunPod.
"""
import os
import sys

print("[WARMUP] Starting model pre-caching...")

# Configuration from environment
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Voxtral-Small-24B-2507")
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1")
SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def download_voxtral():
    """Pre-download Voxtral model and processor"""
    try:
        print(f"[WARMUP] Downloading Voxtral model: {MODEL_ID}")
        from transformers import AutoProcessor, AutoModel

        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        # Download processor
        print("[WARMUP] Downloading Voxtral processor...")
        AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
        print("[WARMUP] ✓ Voxtral processor cached")

        # Download model (metadata only, not weights - too large)
        print("[WARMUP] Downloading Voxtral model config...")
        AutoModel.from_pretrained(MODEL_ID, **kwargs, config_only=True)
        print("[WARMUP] ✓ Voxtral config cached")

        return True
    except Exception as e:
        print(f"[WARMUP] ✗ Failed to cache Voxtral: {e}")
        return False

def download_diarization():
    """Pre-download PyAnnote diarization model"""
    try:
        print(f"[WARMUP] Downloading diarization model: {DIAR_MODEL}")
        from pyannote.audio import Pipeline

        kwargs = {}
        if HF_TOKEN:
            kwargs["use_auth_token"] = HF_TOKEN

        Pipeline.from_pretrained(DIAR_MODEL, **kwargs)
        print("[WARMUP] ✓ Diarization model cached")
        return True
    except Exception as e:
        print(f"[WARMUP] ✗ Failed to cache diarization: {e}")
        return False

def download_sentiment():
    """Pre-download sentiment analysis model"""
    try:
        print(f"[WARMUP] Downloading sentiment model: {SENTIMENT_MODEL}")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        print("[WARMUP] ✓ Sentiment tokenizer cached")

        AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
        print("[WARMUP] ✓ Sentiment model cached")
        return True
    except Exception as e:
        print(f"[WARMUP] ✗ Failed to cache sentiment model: {e}")
        return False

def main():
    print(f"[WARMUP] HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
    print(f"[WARMUP] TORCH_HOME: {os.environ.get('TORCH_HOME', 'not set')}")
    print(f"[WARMUP] HF_TOKEN present: {bool(HF_TOKEN)}")
    print()

    results = {
        "voxtral": download_voxtral(),
        "diarization": download_diarization(),
        "sentiment": download_sentiment(),
    }

    print()
    print("[WARMUP] ===== Summary =====")
    for model, success in results.items():
        status = "✓" if success else "✗"
        print(f"[WARMUP] {status} {model}")

    all_success = all(results.values())
    if all_success:
        print("[WARMUP] All models successfully cached!")
        return 0
    else:
        print("[WARMUP] Some models failed to cache (non-critical)")
        return 0  # Don't fail build, models will download on first use

if __name__ == "__main__":
    sys.exit(main())
