FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive     PIP_DISABLE_PIP_VERSION_CHECK=1     PIP_NO_CACHE_DIR=1     PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     HF_HUB_DISABLE_TELEMETRY=1     TRANSFORMERS_NO_ADVISORY_WARNINGS=1     PYTORCH_JIT=0     APP_VERSION=2025-12-22-voxtral-v10-full

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv     ffmpeg libsndfile1 git ca-certificates     build-essential python3-dev pkg-config  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.1)
RUN python3 -m pip install --upgrade pip setuptools wheel  && python3 -m pip install --index-url https://download.pytorch.org/whl/cu121     torch torchvision torchaudio

COPY requirements.txt /app/requirements.txt

# Install deps; print tail on failure
RUN set -e;     python3 -m pip install -v -r /app/requirements.txt 2>&1 | tee /tmp/pip_install.log ||     (echo "----- pip install FAILED (tail) -----"; tail -n 250 /tmp/pip_install.log; exit 1);     python3 -m pip check || true

COPY main.py /app/main.py

# Defaults
ENV MODEL_ID="mistralai/Voxtral-Small-24B-2507"     HF_TOKEN=""     MAX_NEW_TOKENS="512"     TEMPERATURE="0.0"     TOP_P="0.95"     MAX_DURATION_S="9000"     DIAR_MODEL="pyannote/speaker-diarization-3.1"     WITH_SUMMARY_DEFAULT="1"     ENABLE_SENTIMENT="1"     LOG_LEVEL="INFO"

ENTRYPOINT ["python3", "-u", "/app/main.py"]
