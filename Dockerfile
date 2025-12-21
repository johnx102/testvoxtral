FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python setup
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# PyTorch avec CUDA 12.1
RUN pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# WhisperX (inclut faster-whisper, pyannote, etc.)
RUN pip install git+https://github.com/m-bain/whisperx.git

# Dépendances supplémentaires
RUN pip install \
    runpod>=1.6.2 \
    requests>=2.31.0 \
    soundfile>=0.12.1 \
    numpy>=1.24.0,<2.0.0

# Pour le résumé - modèle léger
RUN pip install \
    transformers>=4.36.0 \
    sentencepiece>=0.1.99 \
    accelerate>=0.25.0

# Forcer numpy < 2.0
RUN pip install "numpy>=1.24.0,<2.0.0" --force-reinstall

# Copier le code
COPY src/ /app/src/

# Variables d'environnement
ENV WHISPER_MODEL="large-v3" \
    HF_HOME="/app/.cache/huggingface" \
    MAX_DURATION_S="9000"

# Cache directories
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Télécharger le modèle Whisper au build (optionnel mais recommandé)
# RUN python -c "import whisperx; whisperx.load_model('large-v3', device='cpu', compute_type='int8')"

ENTRYPOINT ["python", "-u", "/app/src/handler.py"]
