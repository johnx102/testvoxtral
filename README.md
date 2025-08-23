# Voxtral Mini 3B — Transcription + Résumé + Diarization + Humeur (Health2)

Corrections incluses :
- pyannote `.to(torch.device("cuda"))`
- ajout `mistral_common` (Voxtral tokenizer)
- sentiment via **zero-shot** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (safetensors)

## Build & Push
```bash
docker build -t <repo>/voxtral-mini-serverless-diar-sent-health2:latest .
docker push <repo>/voxtral-mini-serverless-diar-sent-health2:latest
```

## Variables d’env
- `MODEL_ID=mistralai/Voxtral-Mini-3B-2507`
- `HF_TOKEN` (si nécessaire, ex. pyannote licence)
- `DIAR_MODEL=pyannote/speaker-diarization-3.1`
- `SENTIMENT_MODEL=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- `SENTIMENT_TYPE=zero-shot`
- `ENABLE_SENTIMENT=1`
- `MAX_DURATION_S=1200`

## Health check
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "input": { "task": "health" } }'
```

## Transcription diarized + résumé
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "transcribe_diarized",
      "audio_url": "https://example.com/mon_audio.wav",
      "language": "fr",
      "with_summary": true,
      "max_new_tokens": 512
    }
  }'
```
