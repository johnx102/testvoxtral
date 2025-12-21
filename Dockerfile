FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    APP_VERSION=2025-12-21-vllm-final

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev \
    libsndfile1 libsndfile1-dev \
    git ca-certificates curl wget \
    pkg-config build-essential \
    llvm-14 llvm-14-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV LLVM_CONFIG=/usr/bin/llvm-config-14

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch pour CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

# vLLM avec support Voxtral (>= 0.10.0)
RUN pip install --no-cache-dir vllm>=0.10.0

# Mistral common pour audio (requis pour Voxtral)
RUN pip install --no-cache-dir "mistral_common[audio]>=1.8.1"

# Numpy compatible
RUN pip install --no-cache-dir "numpy>=1.26.0,<2.0.0" llvmlite==0.43.0 numba==0.60.0

# Audio processing
RUN pip install --no-cache-dir \
    scipy==1.12.0 \
    soundfile==0.12.1 \
    librosa==0.10.2 \
    pydub==0.25.1

# HuggingFace
RUN pip install --no-cache-dir \
    huggingface_hub>=0.26.0 \
    safetensors>=0.4.0 \
    accelerate>=0.34.0

# Matplotlib pour pyannote
RUN pip install --no-cache-dir matplotlib==3.8.2

# PyAnnote avec ses dépendances
RUN pip install --no-cache-dir pyannote.audio>=3.3.0

# OpenAI client pour appeler vLLM
RUN pip install --no-cache-dir openai>=1.0.0

# API et utils
RUN pip install --no-cache-dir \
    requests>=2.31.0 \
    runpod>=1.7.0 \
    psutil>=5.9.0

# Forcer numpy < 2.0
RUN pip install --no-cache-dir "numpy>=1.26.0,<2.0.0" --force-reinstall

# Scripts
COPY start.sh /app/start.sh
COPY main.py /app/main.py
RUN chmod +x /app/start.sh

# Variables d'environnement
ENV MAX_DURATION_S="9000" \
    LOG_LEVEL="INFO" \
    PYTHONPATH="/app" \
    HF_HOME="/app/.cache/huggingface" \
    VLLM_PORT="8000" \
    VOXTRAL_MODEL="mistralai/Voxtral-Mini-3B-2507"

# Note: On utilise Voxtral-Mini-3B car il nécessite ~10GB VRAM
# Pour Voxtral-Small-24B, il faut ~55GB VRAM et tensor-parallel-size 2

RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

ENTRYPOINT ["/app/start.sh"]
