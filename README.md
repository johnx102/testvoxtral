# WhisperX Worker - Transcription + Diarisation + Résumé + Sentiment

Worker RunPod Serverless basé sur [WhisperX](https://github.com/m-bain/whisperX) avec fonctionnalités additionnelles.

## Fonctionnalitésj

- ✅ **Transcription** - Whisper large-v3 (meilleure qualité)
- ✅ **Alignement** - Timestamps précis au niveau des mots
- ✅ **Diarisation** - Identification des speakers (Pyannote)
- ✅ **Résumé** - Extraction des points clés
- ✅ **Sentiment** - Analyse de la satisfaction client

## Configuration RunPod

### Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `HF_TOKEN` | Token HuggingFace (requis pour diarisation) | - |
| `WHISPER_MODEL` | Modèle Whisper | `large-v3` |
| `BATCH_SIZE` | Taille des batchs | `16` |
| `MAX_DURATION_S` | Durée max audio (secondes) | `9000` |

### GPU recommandé

- Minimum: 16GB VRAM (RTX 4080, A10)
- Recommandé: 24GB+ (RTX 4090, A100)

## API

### Paramètres d'entrée

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `audio_url` / `audio_file` | string | Oui | URL du fichier audio |
| `language` | string | Non | Code langue (ex: `fr`, `en`). Auto-détection si absent |
| `task` | string | Non | `transcribe` ou `transcribe_diarized` |
| `diarization` | bool | Non | Activer la diarisation |
| `align_output` | bool | Non | Activer l'alignement mot-à-mot (défaut: true) |
| `summary` / `with_summary` | bool | Non | Générer un résumé |
| `sentiment` | bool | Non | Analyser le sentiment |
| `batch_size` | int | Non | Taille des batchs (défaut: 16) |
| `min_speakers` | int | Non | Nombre min de speakers |
| `max_speakers` | int | Non | Nombre max de speakers |
| `huggingface_access_token` | string | Non | Token HF (sinon utilise `HF_TOKEN` env) |

### Exemple - Transcription simple

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "language": "fr"
  }
}
```

### Exemple - Transcription avec diarisation

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

### Exemple - Configuration complète

```json
{
  "input": {
    "audio_url": "https://example.com/audio.wav",
    "language": "fr",
    "diarization": true,
    "align_output": true,
    "batch_size": 32,
    "min_speakers": 2,
    "max_speakers": 5,
    "summary": true,
    "sentiment": true
  }
}
```

## Réponse

### Sans diarisation

```json
{
  "transcription": "Texte complet de la transcription...",
  "detected_language": "fr",
  "summary": "Résumé: Points clés de la conversation...",
  "sentiment": {
    "sentiment": "positif",
    "score": 0.72,
    "satisfaction_client": "satisfait",
    "details": {"positif": 5, "negatif": 1}
  }
}
```

### Avec diarisation

```json
{
  "segments": [
    {
      "start": 0.5,
      "end": 5.2,
      "text": "Bonjour, je vous appelle concernant...",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Bonjour", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"},
        {"word": "je", "start": 1.0, "end": 1.1, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 5.5,
      "end": 12.1,
      "text": "Oui bonjour, comment puis-je vous aider?",
      "speaker": "SPEAKER_01"
    }
  ],
  "transcriptions": [...],
  "detected_language": "fr",
  "summary": "Résumé: ...",
  "sentiment": {
    "sentiment": "positif",
    "score": 0.72,
    "satisfaction_client": "satisfait"
  }
}
```

## Build Docker

```bash
docker build -t whisperx-worker .
```

## Test local

```bash
docker run --gpus all \
  -e HF_TOKEN="hf_xxx" \
  -e RUNPOD_DEBUG=1 \
  whisperx-worker
```

## Avantages vs autres solutions

| Feature | Ce worker | Whisper simple | faster-whisper |
|---------|-----------|----------------|----------------|
| Qualité transcription | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Vitesse | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Timestamps mots | ✅ Précis | ❌ | ⚠️ Approximatif |
| Diarisation | ✅ Intégrée | ❌ | ❌ |
| Résumé | ✅ | ❌ | ❌ |
| Sentiment | ✅ | ❌ | ❌ |

## Crédits

- [WhisperX](https://github.com/m-bain/whisperx) - Transcription + alignement
- [Pyannote](https://github.com/pyannote/pyannote-audio) - Diarisation
- [RunPod](https://runpod.io) - Plateforme serverless
