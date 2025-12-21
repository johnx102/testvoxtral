#!/bin/bash
set -e

echo "=== Voxtral Serverless Worker Starting ==="
echo "Model: ${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}"
echo "vLLM Port: ${VLLM_PORT:-8000}"

# Variables par défaut
export VOXTRAL_MODEL=${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-3B-2507}
export VLLM_PORT=${VLLM_PORT:-8000}

# Vérifier HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Some models may not be accessible."
fi

# Lancer vLLM en background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model ${VOXTRAL_MODEL} \
    --tokenizer_mode mistral \
    --config_format mistral \
    --load_format mistral \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    2>&1 | while read line; do echo "[vLLM] $line"; done &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

# Attendre que vLLM soit prêt (max 5 minutes)
echo "Waiting for vLLM to be ready..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "Waiting for vLLM... ($WAITED/$MAX_WAIT seconds)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM did not start in time"
    exit 1
fi

# Lancer le worker Python
echo "Starting RunPod worker..."
exec python -u /app/main.py
