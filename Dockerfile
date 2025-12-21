FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip tooling
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- Core Python deps (pinned / known-good) ----
# IMPORTANT: each RUN is separate (no accidental "RUN" inside pip line)
RUN python3 -m pip install --no-cache-dir \
    huggingface_hub==0.26.2 \
    safetensors==0.4.5 \
    accelerate==0.34.2 \
    sentencepiece==0.2.0

# Transformers with Voxtral support
RUN python3 -m pip install --no-cache-dir "transformers==4.56.1"

# Voxtral tokenizer dependency (must be >= 1.8.x)
RUN python3 -m pip install --no-cache-dir --upgrade "mistral-common[audio]==1.8.6"

# Pyannote diarization
RUN python3 -m pip install --no-cache-dir "pyannote.audio==3.1.1"

# Audio utils (si ton code en a besoin)
RUN python3 -m pip install --no-cache-dir \
    soundfile==0.12.1 \
    librosa==0.10.2.post1 \
    numpy==1.26.4 \
    scipy==1.10.0 \
    pydub==0.25.1 \
    requests==2.32.0

# RunPod worker
RUN python3 -m pip install --no-cache-dir "runpod==1.7.0"

# Copy app
COPY main.py /app/main.py
COPY diagnostic.py /app/diagnostic.py

# (Optionnel) log versions au build (super utile)
RUN python3 - <<'PY'
import transformers, mistral_common
print("transformers:", transformers.__version__)
print("mistral_common:", mistral_common.__version__)
PY

CMD ["python3", "-u", "main.py"]
