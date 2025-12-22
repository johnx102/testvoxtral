FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive     PIP_DISABLE_PIP_VERSION_CHECK=1     PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     HF_HUB_DISABLE_TELEMETRY=1     TRANSFORMERS_NO_ADVISORY_WARNINGS=1     PYTORCH_JIT=0

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv     ffmpeg libsndfile1 ca-certificates git  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.1)
RUN python3 -m pip install --upgrade pip setuptools wheel  && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121     torch torchvision torchaudio

COPY requirements.txt /app/requirements.txt
RUN set -e; python3 -m pip install -v -r /app/requirements.txt

COPY main.py /app/main.py

ENV MODEL_ID="mistralai/Voxtral-Small-24B-2507"     MAX_NEW_TOKENS="512"     TEMPERATURE="0.0"     TOP_P="0.95"     HF_TOKEN=""     MAX_DURATION_S="1200"     WITH_SUMMARY_DEFAULT="1"     LOG_LEVEL="INFO"

ENTRYPOINT ["python3", "-u", "/app/main.py"]
