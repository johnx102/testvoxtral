#!/usr/bin/env python3
"""Pré-téléchargement des modèles dans /app/.cache/huggingface au build Docker.
- Si un HF_TOKEN est passé via build arg → télécharge les modèles dans l'image
- Sinon → exit 0, les modèles seront téléchargés au 1er cold start (FlashBoot)
"""
import os
import sys

# Essayer plusieurs noms de variable pour trouver le token
HF_TOKEN = ""
token_source = None
for candidate in [
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_ACCESS_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_API_TOKEN",
    "HUGGING_FACE_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
]:
    val = os.environ.get(candidate, "").strip()
    if val:
        HF_TOKEN = val
        token_source = candidate
        break

# Fallback : BuildKit secrets
if not HF_TOKEN:
    for secret_path in ["/run/secrets/hf_token", "/run/secrets/HF_TOKEN"]:
        if os.path.exists(secret_path):
            try:
                with open(secret_path) as f:
                    HF_TOKEN = f.read().strip()
                if HF_TOKEN:
                    token_source = f"secret:{secret_path}"
                    break
            except Exception:
                pass

if not HF_TOKEN:
    print("[PRELOAD] ⚠ Pas de HF_TOKEN au build → modèles téléchargés au 1er cold start")
    print("[PRELOAD] ⚠ (FlashBoot prendra le snapshot après le 1er cold start, ~1s ensuite)")
    sys.exit(0)

print(f"[PRELOAD] ✅ Token trouvé via {token_source} ({len(HF_TOKEN)} chars)")

LLM_MODEL = os.environ.get("PRELOAD_LLM_MODEL", "mistralai/Ministral-8B-Instruct-2410")
WHISPER_MODEL = os.environ.get("PRELOAD_WHISPER_MODEL", "bofenghuang-french")
CACHE_DIR = "/app/.cache/huggingface/hub"

print(f"[PRELOAD] LLM: {LLM_MODEL}")
print(f"[PRELOAD] Whisper: {WHISPER_MODEL}")
print(f"[PRELOAD] Cache dir: {CACHE_DIR}")

# 5) LLM via snapshot_download
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
    print(f"[PRELOAD] ✅ LLM {LLM_MODEL} downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ❌ ERROR downloading LLM: {type(e).__name__}: {e}")
    # Ne pas faire échouer le build — le modèle sera téléchargé au runtime
    sys.exit(0)

# 6) Whisper
print(f"[PRELOAD] Downloading Whisper {WHISPER_MODEL}...")
try:
    if WHISPER_MODEL.lower() in ("bofenghuang-french", "french", "fr-distil"):
        # Cas spécial : modèle bofenghuang français distillé
        # On télécharge uniquement le sous-dossier ctranslate2/
        from huggingface_hub import snapshot_download as hf_snapshot
        hf_snapshot(
            repo_id="bofenghuang/whisper-large-v3-french-distil-dec16",
            local_dir="/app/.cache/whisper-french-distil-dec16",
            allow_patterns="ctranslate2/*",
        )
        print(f"[PRELOAD] ✅ Whisper bofenghuang french distil dec16 downloaded successfully")
    else:
        # Cas standard : faster-whisper télécharge depuis Systran
        from faster_whisper import WhisperModel
        m = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            download_root="/app/.cache/huggingface/faster-whisper",
        )
        del m
        print(f"[PRELOAD] ✅ Whisper {WHISPER_MODEL} downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ❌ ERROR downloading Whisper: {type(e).__name__}: {e}")
    sys.exit(0)

print("[PRELOAD] ✅ All models cached successfully")
