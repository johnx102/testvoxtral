FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    APP_VERSION=2025-12-21-fixed

# System deps avec FFmpeg et outils de compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev \
    libsndfile1 git ca-certificates curl \
    pkg-config build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch pour CUDA 12.1 (cohérent avec l'image de base)
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 \
    torchaudio==2.4.0+cu121 \
    torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Requirements (sans torch car déjà installé)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App et scripts
COPY main.py /app/main.py
COPY diagnostic.py /app/diagnostic.py

# Variables d'environnement pour le service
ENV HF_TOKEN="" \
    MAX_DURATION_S="9000" \
    LOG_LEVEL="INFO" \
    PYTHONPATH="/app" \
    HF_HUB_CACHE="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/transformers"

# Créer les dossiers de cache
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers

ENTRYPOINT ["python", "-u", "main.py"]
