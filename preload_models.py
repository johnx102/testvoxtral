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
    print("[PRELOAD] ⚠ Pas de HF_TOKEN au build — tentative sans token (modèles non-gated)")
    print("[PRELOAD] ⚠ Si un modèle est gated, il sera téléchargé au 1er cold start")
    # On NE fait PAS sys.exit(0) ici : Qwen et Whisper sont non-gated,
    # ils peuvent être téléchargés sans token. On continue.

print(f"[PRELOAD] ✅ Token trouvé via {token_source} ({len(HF_TOKEN)} chars)")

LLM_MODEL = os.environ.get("PRELOAD_LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
WHISPER_REPO = "bofenghuang/whisper-large-v3-french-distil-dec16"
WHISPER_LOCAL_DIR = "/app/.cache/whisper-french-distil-dec16"
CACHE_DIR = "/app/.cache/huggingface/hub"

print(f"[PRELOAD] LLM: {LLM_MODEL}")
print(f"[PRELOAD] Whisper: {WHISPER_REPO}")
print(f"[PRELOAD] Cache dir: {CACHE_DIR}")

from huggingface_hub import snapshot_download

# 5) LLM via snapshot_download
# Note : les gros modèles (24B+) sont trop volumineux pour le build Docker
# RunPod (~48GB de safetensors). Si le téléchargement échoue (timeout, espace
# disque, modèle gated, etc.), on continue quand même — le LLM sera téléchargé
# au premier cold start et FlashBoot le cachera pour les starts suivants.
print(f"[PRELOAD] Downloading LLM {LLM_MODEL}...")
try:
    snapshot_download(
        repo_id=LLM_MODEL,
        token=HF_TOKEN or None,
        cache_dir=CACHE_DIR,
        ignore_patterns=[
            "*.msgpack", "*.h5", "flax_model*", "tf_model*",
            "consolidated*", "original/*", "*.pt", "*.bin",
        ],
    )
    print(f"[PRELOAD] ✅ LLM {LLM_MODEL} downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ⚠ LLM download failed (will download at runtime): {type(e).__name__}: {e}")
    # NE PAS sys.exit() ici — on continue pour télécharger Whisper

# 6) Whisper bofenghuang français : on télécharge uniquement le sous-dossier ctranslate2/
print(f"[PRELOAD] Downloading {WHISPER_REPO}...")
try:
    snapshot_download(
        repo_id=WHISPER_REPO,
        local_dir=WHISPER_LOCAL_DIR,
        allow_patterns="ctranslate2/*",
    )
    print(f"[PRELOAD] ✅ Whisper downloaded successfully")
except Exception as e:
    print(f"[PRELOAD] ⚠ Whisper download failed: {type(e).__name__}: {e}")

print("[PRELOAD] ✅ Preload terminé (les modèles manquants seront téléchargés au runtime)")
