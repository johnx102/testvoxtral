# Image RunPod officielle
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_JIT=0 \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    APP_VERSION=whisper-ministral8b-v5.2 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Requirements de base
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Transformers + accelerate + bitsandbytes (pour INT4/INT8)
RUN pip install --no-cache-dir --upgrade \
    "transformers>=4.50.0" \
    "accelerate>=1.0.0" \
    "bitsandbytes>=0.43.0" \
    "hf_transfer>=0.1.6"

# faster-whisper pour la transcription
RUN pip install --no-cache-dir "faster-whisper>=1.0.0"

# Flash Attention 2 (gain 20-40% sur génération, facultatif)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "[BUILD] flash-attn install failed, continuing without it"

# Fix PyTorch 2.6+ weights_only
COPY patch_torch_load.py /tmp/patch_torch_load.py
RUN python3 /tmp/patch_torch_load.py && rm /tmp/patch_torch_load.py

# Créer les dossiers de cache DANS l'image
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# Copy app files
COPY main.py /app/main.py

# ─── Variables d'environnement par défaut ────────────────────────────────────
# QUANT_MODE : bf16 (~16GB VRAM, le plus rapide) | bnb8 (~8GB, lent) | bnb4 (~5GB, qualité moindre)
# ⚠ Ministral 8B bf16 = ~16GB, tient sur RTX 4090 24GB avec Whisper (~3GB)
ENV LLM_MODEL_ID="mistralai/Ministral-8B-Instruct-2410" \
    WHISPER_MODEL_SIZE="large-v2" \
    WHISPER_DEVICE="cuda" \
    WHISPER_COMPUTE="float16" \
    MAX_DURATION_S="3600" \
    WITH_SUMMARY_DEFAULT="1" \
    ENABLE_SENTIMENT="1" \
    ENABLE_HOLD_MUSIC_DETECTION="1" \
    ENABLE_TRANSCRIPT_CORRECTION="1" \
    QUANT_MODE="bf16" \
    LOG_LEVEL="INFO"
# Note : HF_TOKEN est passé via les env vars RunPod au runtime (pas ici)
# ⚠ Supprime QUANT_MODE des env vars RunPod Settings pour que la valeur de l'image s'applique

# ─── Pré-téléchargement des modèles dans l'image ─────────────────────────────
# HF_TOKEN doit être passé en build arg : --build-arg HF_TOKEN=hf_xxx
# Dans RunPod : Settings → Build → Environment Variables → HF_TOKEN
# Si HF_TOKEN absent : les modèles seront téléchargés au premier cold start (pas d'erreur build)

ARG HF_TOKEN=""
COPY preload_models.py /tmp/preload_models.py
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "[BUILD] HF_TOKEN trouvé — lancement du pré-téléchargement..." && \
        HF_TOKEN="$HF_TOKEN" \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        PRELOAD_LLM_MODEL="mistralai/Ministral-8B-Instruct-2410" \
        PRELOAD_WHISPER_MODEL="large-v2" \
        python3 /tmp/preload_models.py && \
        du -sh /app/.cache/huggingface && \
        rm /tmp/preload_models.py ; \
    else \
        echo "[BUILD] ⚠ Pas de HF_TOKEN — modèles téléchargés au premier cold start" && \
        rm /tmp/preload_models.py ; \
    fi

# Forcer faster-whisper à utiliser le cache local au runtime
ENV HF_HUB_CACHE=/app/.cache/huggingface/hub

ENTRYPOINT ["python", "-u", "main.py"]
