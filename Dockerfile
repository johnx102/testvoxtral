FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=2025-12-21-whisper-final

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    ffmpeg \
    libsndfile1 \
    git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch pour CUDA 12.1 (séparé car index spécial)
RUN pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Dépendances depuis requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Forcer numpy < 2.0 après tout (au cas où une dep l'upgrade)
RUN pip install "numpy>=1.24.0,<2.0.0" --force-reinstall

# Copier le code
COPY main.py /app/main.py

# Variables d'environnement
ENV MAX_DURATION_S="9000" \
    WHISPER_MODEL="large-v3" \
    HF_HOME="/app/.cache/huggingface" \
    WHISPER_CACHE="/app/.cache/whisper"

# Créer les dossiers de cache
RUN mkdir -p /app/.cache/huggingface /app/.cache/whisper && \
    chmod -R 777 /app/.cache

ENTRYPOINT ["python", "-u", "main.py"]
