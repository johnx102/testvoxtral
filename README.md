# Voxtral Mini 3B — Transcription + Résumé + Diarization + Humeur (Health3)

**Fixes clés :**
- Sentiment **sur CPU** par défaut (`SENTIMENT_DEVICE=-1`) + `PYTORCH_JIT=0` → évite l'erreur NVRTC/TorchScript `fabs(...)`.
- Diarizer `.to(...)` **robuste** : essaie CUDA (torch.device), sinon garde sur CPU sans crasher.
- `APP_VERSION=2025-08-23-02` renvoyé dans `/health` pour vérifier que la bonne image tourne.

## Health
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "input": { "task": "health" } }'
```
Vérifie que `info.app_version` = **2025-08-23-02** et `sentiment_device` = **cpu**.

## Transcription diarized
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
