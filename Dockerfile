FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- PyTorch CUDA (pin torchaudio < 2.2 to keep set_audio_backend for pyannote.audio 3.1.x) ----
# IMPORTANT: use cu121 wheels (CUDA 12.1)
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2+cu121 \
    torchaudio==2.1.2+cu121

# ---- Core deps ----
RUN python3 -m pip install --no-cache-dir \
    huggingface_hub==0.26.2 \
    safetensors==0.4.5 \
    accelerate==0.34.2 \
    sentencepiece==0.2.0 \
    soundfile==0.12.1 \
    librosa==0.10.2.post1 \
    numpy==1.26.4 \
    scipy==1.10.0 \
    pydub==0.25.1 \
    requests==2.32.0

# Transformers + Voxtral
RUN python3 -m pip install --no-cache-dir "transformers==4.56.1"
RUN python3 -m pip install --no-cache-dir --upgrade "mistral-common[audio]==1.8.6"

# Pyannote diarization
RUN python3 -m pip install --no-cache-dir "pyannote.audio==3.1.1"

# Re-pin torchaudio after pyannote in case a newer torchaudio is pulled
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    --upgrade --force-reinstall torchaudio==2.1.2+cu121

# RunPod worker
RUN python3 -m pip install --no-cache-dir "runpod==1.7.0"

# Copy app
COPY main.py /app/main.py
COPY diagnostic.py /app/diagnostic.py

# Print versions at build time (useful)
RUN python3 - <<'PY'
import torch, torchaudio, transformers
import mistral_common
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
print("mistral_common:", mistral_common.__version__)
PY

CMD ["python3", "-u", "main.py"]
