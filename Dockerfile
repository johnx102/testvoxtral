FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_JIT=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg libsndfile1 \
    git ca-certificates \
    build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.1)
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

COPY requirements.txt /app/requirements.txt

# Install deps; if it fails, print the tail of pip log so RunPod UI shows the real error
RUN set -eux; \
    python3 -m pip install -v -r /app/requirements.txt 2>&1 | tee /tmp/pip_install.log; \
    python3 -m pip check

COPY main.py /app/main.py

ENV MODEL_ID="mistralai/Voxtral-Small-24B-2507" \
    HF_TOKEN="" \
    LOG_LEVEL="INFO"

ENTRYPOINT ["python3", "-u", "/app/main.py"]
