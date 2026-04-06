# Image RunPod officielle
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_JIT=0 \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    APP_VERSION=whisper-mistral-v5.0 \
    HF_HOME=/app/.cache/huggingface \
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

# Upgrade transformers pour supporter Mistral3Config (>= 4.48)
RUN pip install --no-cache-dir --upgrade transformers>=4.50.0 accelerate>=1.0.0

# Installer faster-whisper
RUN pip install --no-cache-dir faster-whisper>=1.0.0

# Fix PyTorch 2.6+ weights_only
COPY patch_torch_load.py /tmp/patch_torch_load.py
RUN python3 /tmp/patch_torch_load.py && rm /tmp/patch_torch_load.py

# Créer les dossiers de cache DANS l'image
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# Copy app files
COPY main.py /app/main.py

ENV LLM_MODEL_ID="mistralai/Mistral-Small-3.1-24B-Instruct-2503" \
    WHISPER_MODEL_SIZE="large-v2" \
    WHISPER_DEVICE="cuda" \
    WHISPER_COMPUTE="float16" \
    MAX_DURATION_S="3600" \
    WITH_SUMMARY_DEFAULT="1" \
    ENABLE_SENTIMENT="1" \
    ENABLE_HOLD_MUSIC_DETECTION="1" \
    QUANT_MODE="torchao" \
    LOG_LEVEL="INFO"
# Note : HF_TOKEN est passé via les env vars RunPod au runtime (pas ici)

# ─── Pré-téléchargement des modèles dans l'image ─────────────────────────────
# HF_TOKEN doit être passé en build arg : --build-arg HF_TOKEN=hf_xxx
# Dans RunPod : Settings → Build → Environment Variables → HF_TOKEN

ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
        echo "[BUILD] HF_TOKEN trouvé — pré-téléchargement des modèles..." && \
        HF_TOKEN=$HF_TOKEN python3 -c " \
import os; \
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', ''); \
from huggingface_hub import snapshot_download; \
print('[BUILD] Downloading Mistral Small 3.1 24B...'); \
snapshot_download( \
    repo_id='mistralai/Mistral-Small-3.1-24B-Instruct-2503', \
    token=os.environ['HF_TOKEN'], \
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'], \
); \
print('[BUILD] Downloading Whisper large-v2...'); \
from faster_whisper import WhisperModel; \
WhisperModel('large-v2', device='cpu', compute_type='int8'); \
print('[BUILD] All models cached successfully'); \
" && \
        echo "[BUILD] Modèles pré-téléchargés dans /app/.cache/huggingface"; \
    else \
        echo "[BUILD] Pas de HF_TOKEN — les modèles seront téléchargés au premier cold start"; \
        echo "[BUILD] Pour pré-cacher : ajouter HF_TOKEN dans RunPod Build Settings"; \
    fi

ENTRYPOINT ["python", "-u", "main.py"]
