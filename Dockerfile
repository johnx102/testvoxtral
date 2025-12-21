FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg git ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- PyTorch CUDA 12.1 stack (new enough for transformers 4.56.x) ---
RUN python3 -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 \
  torchaudio==2.3.1

# Pin NumPy <2.0 (pyannote + deps not compatible with NumPy 2 yet)
RUN python3 -m pip install --no-cache-dir numpy==1.26.4

# Core deps
RUN python3 -m pip install --no-cache-dir \
    huggingface_hub==0.26.2 \
    safetensors==0.4.5 \
    accelerate==0.34.2 \
    sentencepiece==0.2.0 \
    soundfile==0.12.1 \
    librosa==0.10.2.post1 \
    scipy==1.10.0 \
    pydub==0.25.1 \
    requests==2.32.0

# Transformers + Voxtral deps
RUN python3 -m pip install --no-cache-dir transformers==4.56.1
RUN python3 -m pip install --no-cache-dir --upgrade "mistral-common[audio]==1.8.6"

# Pyannote diarization
RUN python3 -m pip install --no-cache-dir pyannote.audio==3.1.1

# Re-force NumPy pin AFTER pyannote install (some deps may try to upgrade it)
RUN python3 -m pip install --no-cache-dir numpy==1.26.4

# RunPod worker
RUN python3 -m pip install --no-cache-dir runpod==1.7.0

COPY main.py /app/main.py
COPY diagnostic.py /app/diagnostic.py

# Never fail build if optional imports behave differently
RUN python3 - <<'PY' || true
try:
    import torch, torchaudio, transformers, numpy
    import mistral_common
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("torchaudio:", torchaudio.__version__)
    print("transformers:", transformers.__version__)
    print("numpy:", numpy.__version__)
    print("mistral_common:", mistral_common.__version__)
except Exception as e:
    print("Version check skipped:", repr(e))
PY

CMD ["python3", "-u", "main.py"]
