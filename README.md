# Voxtral Mini 3B — Transcription + Résumé + Diarization + Humeur (Hardened)

Cette version ajoute un **endpoint de santé** pour diagnostiquer rapidement les soucis
(build/runtime, token HF, accès modèles, support Voxtral dans transformers, etc.).

## Build & Push
```bash
docker build -t <repo>/voxtral-mini-serverless-diar-sent-health:latest .
docker push <repo>/voxtral-mini-serverless-diar-sent-health:latest
```

## Variables d’environnement
- `MODEL_ID=mistralai/Voxtral-Mini-3B-2507`
- `HF_TOKEN` (si requis / licences HF — ex. pyannote)
- `DIAR_MODEL=pyannote/speaker-diarization-3.1`
- `SENTIMENT_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment`
- `ENABLE_SENTIMENT=1`
- `MAX_DURATION_S=1200`
- `LOG_LEVEL=DEBUG` (pour log verbeux au besoin)

## Tests rapides

### 1) Health check (recommandé)
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "input": { "task": "health" } }'
```
**Réponse** : 
- `ok: true/false`
- `info`: versions, device, flags
- `errors`: messages explicites (ex. licence pyannote non acceptée → 401/403)

### 2) Transcription diarized + humeur + résumé
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

## Problèmes fréquents & solutions
- **Transformers sans Voxtral** → `errors.voxtral_load`: mets `transformers@main` (déjà dans requirements) et rebuild.
- **pyannote 401/403** → accepte la licence sur HF + fournis `HF_TOKEN` (read).
- **Build exit 127** → cette image force `python` + `pip`, ajoute libsndfile, ffmpeg, git et installe torch/cu121 avant le reste.
