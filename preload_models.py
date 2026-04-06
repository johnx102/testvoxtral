#!/usr/bin/env python3
"""Pré-téléchargement des modèles dans /app/.cache/huggingface au build Docker.
Version diagnostique : log toutes les env vars pertinentes pour identifier
comment RunPod passe (ou non) le token au build.
"""
import os
import sys

print("=" * 70)
print("[PRELOAD] === DIAGNOSTIC ENV VARS (build context) ===")
print("=" * 70)

# 1) Scanner toutes les env vars qui contiennent HF, HUGG, TOKEN, KEY, SECRET
relevant_keys = []
for key in sorted(os.environ.keys()):
    key_upper = key.upper()
    if any(pattern in key_upper for pattern in ("HF", "HUGG", "TOKEN", "KEY", "SECRET")):
        value = os.environ[key]
        # Masquer la valeur : montrer longueur + 4 premiers chars
        if value:
            preview = value[:4] + "..." + f"({len(value)} chars)"
        else:
            preview = "(empty)"
        print(f"[PRELOAD]   {key} = {preview}")
        relevant_keys.append(key)

if not relevant_keys:
    print("[PRELOAD]   (aucune variable HF/HUGG/TOKEN/KEY/SECRET trouvée)")

# 2) Chercher les secrets BuildKit éventuels
print("[PRELOAD] === BUILDKIT SECRETS ===")
for secret_path in [
    "/run/secrets/hf_token",
    "/run/secrets/HF_TOKEN",
    "/run/secrets/HUGGING_FACE_HUB_TOKEN",
    "/run/secrets/hugging_face_token",
]:
    if os.path.exists(secret_path):
        try:
            with open(secret_path) as f:
                content = f.read().strip()
            print(f"[PRELOAD]   {secret_path} exists, {len(content)} chars")
        except Exception as e:
            print(f"[PRELOAD]   {secret_path} exists but unreadable: {e}")
    else:
        print(f"[PRELOAD]   {secret_path} (not found)")

print("=" * 70)

# 3) Essayer plusieurs noms de variable pour trouver le token
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
    "RUNPOD_HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
]:
    val = os.environ.get(candidate, "").strip()
    if val:
        HF_TOKEN = val
        token_source = candidate
        break

# 4) Fallback : lire depuis BuildKit secrets
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
    print("[PRELOAD] ❌ Aucun token trouvé dans aucune variable/secret connue")
    print("[PRELOAD] ℹ Les modèles seront téléchargés au 1er cold start")
    print("[PRELOAD] ℹ Skipping preload (build continue sans pré-cache)")
    sys.exit(0)  # Exit 0 pour que le build continue

print(f"[PRELOAD] ✅ Token trouvé via: {token_source} ({len(HF_TOKEN)} chars)")

LLM_MODEL = os.environ.get("PRELOAD_LLM_MODEL", "mistralai/Ministral-8B-Instruct-2410")
WHISPER_MODEL = os.environ.get("PRELOAD_WHISPER_MODEL", "large-v2")
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

# 6) Whisper via faster-whisper
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
    print(f"[PRELOAD] ✅ Whisper {WHISPER_MODEL} downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ❌ ERROR downloading Whisper: {type(e).__name__}: {e}")
    sys.exit(0)

print("[PRELOAD] ✅ All models cached successfully")
