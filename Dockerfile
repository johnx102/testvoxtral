FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_JIT=0 \
    APP_VERSION=2025-12-20-stable

# System deps avec FFmpeg pour torchcodec
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg libavformat-dev libavcodec-dev libavutil-dev \
    libsndfile1 git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Torch versions stables compatibles
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0

# Requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY main.py /app/main.py

ENV MODEL_ID="mistralai/Voxtral-Small-24B-2507" \
    MAX_NEW_TOKENS="512" \
    TEMPERATURE="0.0" \
    TOP_P="0.95" \
    HF_TOKEN="" \
    MAX_DURATION_S="1200" \
    DIAR_MODEL="pyannote/speaker-diarization-3.1" \
    WITH_SUMMARY_DEFAULT="1" \
    SENTIMENT_MODEL="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" \
    SENTIMENT_TYPE="zero-shot" \
    ENABLE_SENTIMENT="1" \
    SENTIMENT_DEVICE="-1" \
    ENABLE_DIARIZATION="1" \
    ENABLE_SINGLE_VOICE_DETECTION="0" \
    LOG_LEVEL="INFO"

ENTRYPOINT ["python", "-u", "main.py"]
