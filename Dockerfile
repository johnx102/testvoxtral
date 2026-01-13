# Image RunPod officielle - testée avec TOUS leurs GPUs (A100, RTX 6000, L40S, etc.)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTORCH_JIT=0 \
    TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 \
    APP_VERSION=2025-08-23-02 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

# System deps (ffmpeg pour audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch (CUDA 12.8) - PyTorch 2.7.1 = support TOUS les GPUs (RTX 6000 Ada inclus)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# FIX PyTorch 2.6+ weights_only : Patcher speechbrain/pyannote proprement
# On remplace torch.load(X) par torch.load(X, weights_only=False) avec regex
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    for FILE in "$SITE/speechbrain/utils/checkpoints.py" "$SITE/pyannote/audio/core/io.py"; do \
        if [ -f "$FILE" ]; then \
            python3 -c "
import re, sys
with open('$FILE', 'r') as f:
    content = f.read()
# Skip if already patched
if 'weights_only=False' in content:
    print('[BUILD] Already patched: $FILE')
    sys.exit(0)
# Replace torch.load(args) with torch.load(args, weights_only=False)
# This regex matches simple single-line torch.load calls
patched = re.sub(
    r'torch\.load\(([^)]+)\)',
    r'torch.load(\1, weights_only=False)',
    content
)
with open('$FILE', 'w') as f:
    f.write(patched)
print('[BUILD] Patched: $FILE')
" && echo "[BUILD] Done: $FILE"; \
        fi; \
    done

# Create cache directories (RunPod will mount /workspace)
RUN mkdir -p /workspace/.cache/huggingface /workspace/.cache/torch

# Copy app files
COPY main.py /app/main.py
COPY warmup.py /app/warmup.py

ENV MODEL_ID="mistralai/Voxtral-Small-24B-2507" \
    MAX_NEW_TOKENS="512" \
    TEMPERATURE="0.0" \
    TOP_P="0.95" \
    HF_TOKEN="" \
    MAX_DURATION_S="1200" \
    DIAR_MODEL="pyannote/speaker-diarization-3.1" \
    WITH_SUMMARY_DEFAULT="1" \
    SENTIMENT_MODEL="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" \
    SENTIMENT_TYPE="zero-shot" \
    ENABLE_SENTIMENT="1" \
    SENTIMENT_DEVICE="-1" \
    LOG_LEVEL="INFO"

# Pre-download models at build time
# RunPod injects HF_TOKEN as ENV var during build, so we use it if available
RUN echo "[BUILD] Checking for HF_TOKEN to pre-cache models..." && \
    if [ -n "$HF_TOKEN" ]; then \
        echo "[BUILD] HF_TOKEN found - Pre-caching models..." && \
        python /app/warmup.py || echo "[BUILD] Warmup failed (non-critical)"; \
    else \
        echo "[BUILD] No HF_TOKEN found - Models will download on first use"; \
    fi

ENTRYPOINT ["python", "-u", "main.py"]
