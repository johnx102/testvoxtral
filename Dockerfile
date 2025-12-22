FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive     PIP_DISABLE_PIP_VERSION_CHECK=1     PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     HF_HUB_DISABLE_TELEMETRY=1     TRANSFORMERS_NO_ADVISORY_WARNINGS=1     PYTORCH_JIT=0     APP_VERSION=2025-12-22-voxtral-fix04

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv     ffmpeg libsndfile1 git ca-certificates  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.1)
RUN python3 -m pip install --upgrade pip setuptools wheel  && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121     torch torchvision torchaudio

# deps (print full error if anything fails)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -v -r /app/requirements.txt

COPY main.py /app/main.py
ENTRYPOINT ["python3", "-u", "/app/main.py"]
