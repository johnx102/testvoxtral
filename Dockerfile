# GPU-enabled image for Voxtral Mini (PyTorch) + pyannote diarization on RunPod Serverless
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
# Note: torch/torchaudio pinned to CUDA 12.1 wheels
RUN python3 -m pip install --upgrade pip && \    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 && \    python3 -m pip install -r /app/requirements.txt

# Copy app code
COPY main.py /app/main.py

# Default envs (override at deploy time)
ENV MODEL_ID="mistralai/Voxtral-Mini-3B-2507" \
    MAX_NEW_TOKENS="512" \
    TEMPERATURE="0.0" \
    TOP_P="0.95" \
    HF_TOKEN="" \
    MAX_DURATION_S="1200" \
    DIAR_MODEL="pyannote/speaker-diarization-3.1" \
    WITH_SUMMARY_DEFAULT="1"

# RunPod serverless entrypoint
ENTRYPOINT ["python3", "-u", "main.py"]
