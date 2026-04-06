#!/usr/bin/env python3
"""Pré-téléchargement des modèles dans /app/.cache/huggingface au build Docker."""
import os
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    print("[PRELOAD] ERROR: HF_TOKEN manquant")
    print("[PRELOAD] Passe-le au build : --build-arg HF_TOKEN=hf_xxx")
    print("[PRELOAD] Ou dans RunPod Settings → Build → Environment Variables → HF_TOKEN")
    sys.exit(1)

LLM_MODEL = os.environ.get("PRELOAD_LLM_MODEL", "mistralai/Ministral-8B-Instruct-2410")
WHISPER_MODEL = os.environ.get("PRELOAD_WHISPER_MODEL", "large-v2")
CACHE_DIR = "/app/.cache/huggingface/hub"

print(f"[PRELOAD] HF_TOKEN trouvé ({len(HF_TOKEN)} chars)")
print(f"[PRELOAD] LLM: {LLM_MODEL}")
print(f"[PRELOAD] Whisper: {WHISPER_MODEL}")

# 1) LLM via snapshot_download
print(f"[PRELOAD] Downloading LLM {LLM_MODEL}...")
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=LLM_MODEL,
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
        ignore_patterns=[
            "*.msgpack", "*.h5", "flax_model*", "tf_model*",
            "consolidated*", "original/*", "*.pt", "*.bin",
        ],
    )
    print(f"[PRELOAD] LLM {LLM_MODEL} downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ERROR downloading LLM: {type(e).__name__}: {e}")
    sys.exit(2)

# 2) Whisper via faster-whisper
print(f"[PRELOAD] Downloading Whisper {WHISPER_MODEL}...")
try:
    from faster_whisper import WhisperModel
    m = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        download_root="/app/.cache/huggingface/faster-whisper",
    )
    del m
    print(f"[PRELOAD] Whisper {WHISPER_MODEL} downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ERROR downloading Whisper: {type(e).__name__}: {e}")
    sys.exit(3)

print("[PRELOAD] All models cached successfully")
