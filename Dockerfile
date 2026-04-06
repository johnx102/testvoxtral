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
    APP_VERSION=whisper-ministral8b-v5.3 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TORCH_HOME=/app/.cache/torch

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch CUDA 12.8
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Transformers, accelerate, bitsandbytes, hf_transfer
RUN pip install --no-cache-dir --upgrade transformers accelerate bitsandbytes hf_transfer

# Installer faster-whisper
RUN pip install --no-cache-dir faster-whisper

# Flash Attention 2 (gain 20-40% sur génération, best-effort)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "[BUILD] flash-attn install skipped"

# Fix PyTorch 2.6+ weights_only
COPY patch_torch_load.py /tmp/patch_torch_load.py
RUN python3 /tmp/patch_torch_load.py && rm /tmp/patch_torch_load.py

# Créer les dossiers de cache DANS l'image
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# Copy app files
COPY main.py /app/main.py

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

# ─── Pré-téléchargement des modèles dans l'image ─────────────────────────────
# Le script preload_models.py fait un diagnostic complet des env vars au build,
# tente plusieurs noms de token, et télécharge les modèles si possible.
# Si aucun token n'est trouvé, il sort proprement (exit 0) → le build continue
# et les modèles seront téléchargés au 1er cold start.

ARG HF_TOKEN=""
ARG HUGGING_FACE_HUB_TOKEN=""
ARG HUGGINGFACE_HUB_TOKEN=""
ARG HF_ACCESS_TOKEN=""
ARG HUGGINGFACE_TOKEN=""
ARG HUGGING_FACE_TOKEN=""
ARG HF_API_TOKEN=""
ARG HUGGINGFACEHUB_API_TOKEN=""
# CACHE_BUST : changer pour forcer le re-run du step preload
ARG CACHE_BUST="2026-04-06-v6-diagnostic"

COPY preload_models.py /tmp/preload_models.py
RUN echo "[BUILD] CACHE_BUST=$CACHE_BUST" && \
    HF_TOKEN="$HF_TOKEN" \
    HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
    HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
    HF_ACCESS_TOKEN="$HF_ACCESS_TOKEN" \
    HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
    HUGGING_FACE_TOKEN="$HUGGING_FACE_TOKEN" \
    HF_API_TOKEN="$HF_API_TOKEN" \
    HUGGINGFACEHUB_API_TOKEN="$HUGGINGFACEHUB_API_TOKEN" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PRELOAD_LLM_MODEL="mistralai/Ministral-8B-Instruct-2410" \
    PRELOAD_WHISPER_MODEL="large-v2" \
    python3 /tmp/preload_models.py && \
    rm /tmp/preload_models.py && \
    (du -sh /app/.cache/huggingface 2>/dev/null || echo "[BUILD] No preloaded cache")

ENTRYPOINT ["python", "-u", "main.py"]
