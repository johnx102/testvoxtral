FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    APP_VERSION=2025-12-21-v3

# System deps - INCLUT LLVM pour numba
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev \
    libsndfile1 libsndfile1-dev \
    git ca-certificates curl wget \
    pkg-config build-essential \
    llvm-14 llvm-14-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configuration LLVM pour numba
ENV LLVM_CONFIG=/usr/bin/llvm-config-14

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch pour CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Numpy et numba d'abord (versions pré-compilées compatibles)
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    llvmlite==0.43.0 \
    numba==0.60.0

# Audio processing
RUN pip install --no-cache-dir \
    scipy==1.12.0 \
    soundfile==0.12.1 \
    librosa==0.10.2 \
    pydub==0.25.1

# HuggingFace ecosystem
RUN pip install --no-cache-dir \
    huggingface_hub==0.26.2 \
    safetensors==0.4.5 \
    accelerate==0.34.2 \
    sentencepiece==0.2.0 \
    tokenizers==0.20.3

# Transformers avec support Voxtral
RUN pip install --no-cache-dir transformers==4.47.1

# Mistral tokenizer (requis pour Voxtral)
RUN pip install --no-cache-dir mistral-common==1.8.6

# PyAnnote (installation séparée car complexe)
RUN pip install --no-cache-dir \
    pyannote.audio==3.1.1 \
    speechbrain==0.5.16

# API et utils
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    runpod==1.7.0 \
    psutil==5.9.8

# App et scripts
COPY main.py /app/main.py
COPY diagnostic.py /app/diagnostic.py

# Variables d'environnement pour le service
# Note: HF_TOKEN doit être passé au runtime via -e ou RunPod secrets
ENV MAX_DURATION_S="9000" \
    LOG_LEVEL="INFO" \
    PYTHONPATH="/app" \
    HF_HUB_CACHE="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/transformers"

# Créer les dossiers de cache
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers && \
    chmod -R 777 /app/.cache

ENTRYPOINT ["python", "-u", "main.py"]
