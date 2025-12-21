# Voxtral Serverless Worker

Service de transcription audio avec diarisation, résumé et analyse de sentiment.

## Architecture

- **vLLM** : Sert le modèle Voxtral pour transcription/résumé/sentiment
- **Pyannote** : Diarisation des speakers
- **RunPod** : Framework serverless

## Modèles utilisés

| Composant | Modèle | VRAM requise |
|-----------|--------|--------------|
| Transcription | Voxtral-Mini-3B | ~10 GB |
| Diarisation | pyannote/speaker-diarization-3.1 | ~2 GB |

**Note** : Pour utiliser Voxtral-Small-24B (meilleure qualité), changez `VOXTRAL_MODEL` et prévoyez ~55 GB de VRAM.

## Configuration RunPod

### Variables d'environnement requises

| Variable | Description | Exemple |
|----------|-------------|---------|
| `HF_TOKEN` | Token HuggingFace (requis) | `hf_xxx...` |
| `VOXTRAL_MODEL` | Modèle Voxtral à utiliser | `mistralai/Voxtral-Mini-3B-2507` |
| `VLLM_PORT` | Port interne vLLM | `8000` |
| `MAX_DURATION_S` | Durée max audio (secondes) | `9000` |

### GPU recommandé

- **Voxtral-Mini-3B** : RTX 3090/4090 (24GB) ou A10 (24GB)
- **Voxtral-Small-24B** : A100 80GB ou 2x A100 40GB

## API

### Endpoint

```
POST /run
```

### Paramètres d'entrée

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "task": "transcribe_diarized",
    "language": "fr",
    "summary": true,
    "sentiment": true
  }
}
```

| Paramètre | Type | Description |
|-----------|------|-------------|
| `audio_url` | string | URL du fichier audio (requis) |
| `task` | string | `transcribe` ou `transcribe_diarized` |
| `language` | string | Code langue ISO (ex: `fr`, `en`) |
| `summary` | boolean | Inclure un résumé |
| `sentiment` | boolean | Inclure l'analyse de sentiment |

### Exemple de réponse

```json
{
  "task": "transcribe_diarized",
  "language": "fr",
  "duration": 179.2,
  "transcriptions": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.5,
      "end": 5.2,
      "text": "Bonjour, je vous appelle concernant..."
    },
    {
      "speaker": "SPEAKER_01", 
      "start": 5.5,
      "end": 12.1,
      "text": "Oui bonjour, comment puis-je vous aider?"
    }
  ],
  "summary": "Résumé: L'appelant contacte le service client...",
  "sentiment": {
    "sentiment": "positif",
    "score": 0.75,
    "emotions": ["satisfaction", "politesse"],
    "satisfaction_client": "satisfait"
  }
}
```

## Build local

```bash
docker build -t voxtral-serverless .

# Test local
docker run --gpus all \
  -e HF_TOKEN="hf_xxx" \
  -e RUNPOD_DEBUG=1 \
  -p 8000:8000 \
  voxtral-serverless
```

## Temps de démarrage

- Premier démarrage : ~5-10 minutes (téléchargement des modèles)
- Démarrages suivants : ~2-3 minutes (chargement depuis cache)

## Formats audio supportés

- WAV (recommandé)
- MP3
- OGG
- Autres formats convertibles par librosa

## Limitations

- Durée max par défaut : 2h30 (9000 secondes)
- Taille max audio : dépend de la RAM disponible
- Concurrent requests : 1 (serverless)
