# Voxtral Mini (3B) — Transcription + Résumé **avec diarization intégrée** (RunPod Serverless)

Cet endpoint **serverless** prend un `.wav` (URL/base64/fichier) et peut :
- **Transcrire** (Voxtral Mini 3B)
- **Résumer** l’audio avec **le même modèle** Voxtral
- **Diariser** (identifier les locuteurs) via **pyannote** puis **transcrire par segment** et **(optionnel) résumer**

---

## 1) Contenu

- `Dockerfile` — CUDA 12.1 + PyTorch 2.3 + Transformers (*main*) + pyannote
- `requirements.txt`
- `main.py` — `runpod.serverless.start({"handler": handler})` avec 3 tâches : `transcribe`, `summary`, `transcribe_diarized`
- `sample_*.json` — exemples d’appels

---

## 2) Build & Push

```bash
docker build -t <repo>/voxtral-mini-serverless-diar:latest .
docker push <repo>/voxtral-mini-serverless-diar:latest
```

---

## 3) Endpoint Serverless RunPod

- **Image** : `<repo>/voxtral-mini-serverless-diar:latest`
- **Handler** : `handler`
- **GPU** : T4 16GB (éco), L4/A10 24GB (plus rapide)
- **Env vars** (ajustez au besoin) :
  - `MODEL_ID=mistralai/Voxtral-Mini-3B-2507`
  - `HF_TOKEN=<token HF>` (si requis par les modèles)
  - `DIAR_MODEL=pyannote/speaker-diarization-3.1`
  - `MAX_NEW_TOKENS=512`
  - `MAX_DURATION_S=1200`
  - `WITH_SUMMARY_DEFAULT=1` (ajoute un résumé dans `transcribe_diarized`)

> ⚠️ Pour pyannote, certains modèles nécessitent d’**accepter la licence** côté Hugging Face et d’utiliser `HF_TOKEN`.

---

## 4) Appels API (runsync)

### a) Ping

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \\
  -H "Authorization: Bearer $RUNPOD_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{ "input": { "ping": true } }'
```

### b) Transcription simple

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \\
  -H "Authorization: Bearer $RUNPOD_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d @sample_transcribe.json
```

### c) Résumé direct

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \\
  -H "Authorization: Bearer $RUNPOD_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d @sample_summary.json
```

### d) Transcription **avec diarization** + (optionnel) résumé

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \\
  -H "Authorization: Bearer $RUNPOD_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d @sample_transcribe_diarized.json
```

**Réponse type :**
```json
{
  "task": "transcribe_diarized",
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.12, "end": 4.85, "text": "Bonjour ..."},
    {"speaker": "SPEAKER_01", "start": 4.90, "end": 9.32, "text": "Oui je vous appelle ..."}
  ],
  "transcript": "SPEAKER_00: Bonjour ...\nSPEAKER_01: Oui je vous appelle ...",
  "summary": "• ...\n• ..."
}
```

---

## 5) Conseils

- **Durée max** : `MAX_DURATION_S` protège contre les fichiers trop longs.
- **Langue** : envoyez `"language": "fr"` si vous savez que c’est du français.
- **GPU** : T4 fonctionne ; L4/A10 améliore la latence. Le 3B tient largement en FP16/BF16.
- **Coût/latence** : la diarization + transcription par segments ajoute du temps de calcul. Ajustez selon vos SLA.

Bon déploiement !
