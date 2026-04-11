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
    APP_VERSION=whisper-bofenghuang-v6.0-qwen25-14b \
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
# On force l'utilisation d'un wheel précompilé (--only-binary) pour éviter la
# compilation depuis source qui peut prendre 30-60 min et faire timeout le build.
# Si aucun wheel n'est disponible pour cette combinaison torch/cuda/python →
# l'install échoue silencieusement et le code Python utilise le fallback (bf16
# sans flash-attn, ~20% plus lent mais fonctionnel).
RUN timeout 300 pip install --no-cache-dir --only-binary=flash-attn flash-attn 2>&1 || \
    echo "[BUILD] flash-attn install skipped (no wheel or timeout — code will fallback)"

# Fix PyTorch 2.6+ weights_only
COPY patch_torch_load.py /tmp/patch_torch_load.py
RUN python3 /tmp/patch_torch_load.py && rm /tmp/patch_torch_load.py

# Créer les dossiers de cache DANS l'image
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch

# Copy app files
COPY main.py /app/main.py

ENV LLM_MODEL_ID="Qwen/Qwen2.5-14B-Instruct" \
    WHISPER_DEVICE="cuda" \
    WHISPER_COMPUTE="float16" \
    MAX_DURATION_S="3600" \
    WITH_SUMMARY_DEFAULT="1" \
    ENABLE_SENTIMENT="1" \
    ENABLE_HOLD_MUSIC_DETECTION="1" \
    ENABLE_TRANSCRIPT_CORRECTION="1" \
    QUANT_MODE="bnb4" \
    LOG_LEVEL="INFO"
# Note : HF_TOKEN est passé via les env vars RunPod au runtime (pas ici)

# ─── Pré-téléchargement des modèles dans l'image (optionnel) ─────────────────
# RunPod Serverless ne propage PAS les Runtime Env Vars vers le build Docker.
# → Pour pré-cacher les modèles dans l'image, il faut soit :
#   1) Build local : docker build --build-arg HF_TOKEN=hf_xxx ...
#   2) BuildKit secret : docker build --secret id=hf_token,env=HF_TOKEN ...
# Sans token → le script sort en exit 0, les modèles sont téléchargés au 1er
# cold start, et FlashBoot prend le snapshot pour les cold starts suivants (~1s).

ARG HF_TOKEN=""
ARG HUGGING_FACE_HUB_TOKEN=""

COPY preload_models.py /tmp/preload_models.py
RUN HF_TOKEN="$HF_TOKEN" \
    HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PRELOAD_LLM_MODEL="Qwen/Qwen2.5-14B-Instruct" \
    python3 /tmp/preload_models.py && \
    rm /tmp/preload_models.py

ENTRYPOINT ["python", "-u", "main.py"]
