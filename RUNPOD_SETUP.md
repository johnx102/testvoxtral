# Configuration RunPod Serverless - Guide Rapide

## 🚀 Déploiement depuis GitHub

### 1. Variables d'Environnement Requises

Dans la configuration de votre endpoint RunPod, ajoutez ces variables :

```bash
# OBLIGATOIRE - Pour télécharger les modèles
HF_TOKEN=hf_votre_token_huggingface

# Cache persistant (déjà configuré dans le Dockerfile)
HF_HOME=/workspace/.cache/huggingface
TORCH_HOME=/workspace/.cache/torch

# Configuration du modèle (optionnel, valeurs par défaut dans Dockerfile)
MODEL_ID=mistralai/Voxtral-Small-24B-2507
DIAR_MODEL=pyannote/speaker-diarization-3.1
SENTIMENT_MODEL=MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
```

### 2. Configuration GitHub

Dans RunPod, configurez :
- **Repository:** `votre-username/votre-repo`
- **Branch:** `main` (ou votre branche)
- **Dockerfile Path:** `Dockerfile`

### 3. Build Automatique

RunPod va :
1. ✅ Cloner votre repo GitHub
2. ✅ Détecter `HF_TOKEN` dans les variables d'environnement
3. ✅ Builder l'image Docker
4. 🚀 **Automatiquement pré-cacher les modèles** si `HF_TOKEN` est présent
5. ✅ Déployer l'endpoint serverless

**Logs du build à surveiller :**
```
[BUILD] Checking for HF_TOKEN to pre-cache models...
[BUILD] HF_TOKEN found - Pre-caching models...
[WARMUP] Starting model pre-caching...
[WARMUP] ✓ Voxtral processor cached
[WARMUP] ✓ Diarization model cached
[WARMUP] ✓ Sentiment model cached
```

Si vous voyez ces messages, le pré-cache fonctionne ! 🎉

### 4. Configuration Recommandée

**GPU :**
- Type: RTX A6000 (48GB VRAM) ou supérieur
- Recommandé: A100 (80GB) pour les meilleures performances

**Workers :**
- **Min Workers:** 1 (garde une instance warm pour éviter les cold starts)
- **Max Workers:** 3-5 (selon votre charge)
- **Idle Timeout:** 30-60 secondes

**Scaling :**
- **Scale Up Delay:** 5 secondes
- **Scale Down Delay:** 60 secondes

## 📊 Temps de Démarrage

| Scénario | Temps |
|----------|-------|
| **Premier cold start** (avec pré-cache) | ~2-3 min |
| **Cold start suivants** (cache /workspace) | ~15-30s |
| **Warm instance** | <5s |

## 🔍 Vérification

### Health Check
```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "input": { "task": "health" } }'
```

Vous devriez voir :
```json
{
  "output": {
    "ok": true,
    "info": {
      "transformers_has_voxtral": true,
      "cuda_available": true,
      ...
    }
  }
}
```

### Vérifier le Cache
Dans les logs de votre pod :
```
[INIT] Loading processor...
[INIT] Processor loaded successfully  # <-- Pas de "Downloading" = cache utilisé!
```

## 🐛 Troubleshooting

### Build échoue avec erreur HuggingFace
**Problème:** `401 Unauthorized` ou `403 Forbidden`
**Solution:** Vérifiez que `HF_TOKEN` est bien configuré et valide

### Cold start trop long (>5 min)
**Problème:** Les modèles ne sont pas pré-cachés
**Solution:**
1. Vérifiez que `HF_TOKEN` est dans les variables d'environnement RunPod
2. Regardez les logs du build pour confirmer le pré-cache
3. Si le pré-cache a échoué, rebuild l'image

### Modèles téléchargés à chaque appel
**Problème:** Le cache `/workspace` n'est pas persistant
**Solution:**
1. Vérifiez que RunPod monte bien `/workspace`
2. Gardez au moins 1 worker minimum pour conserver une instance warm
3. Les instances différentes ont des caches différents (c'est normal)

### Erreur "VoxtralProcessor not found"
**Problème:** `mistral_common[audio]` n'est pas installé correctement
**Solution:**
1. Vérifiez que `requirements.txt` contient `mistral_common[audio]>=1.8.1`
2. Rebuild l'image depuis GitHub
3. Vérifiez les logs du build pour voir si l'installation a réussi

## 💡 Astuces

1. **Coût optimisé:** Utilisez min workers = 0 si vous acceptez les cold starts
2. **Performance max:** Utilisez min workers = 1 pour instance toujours warm
3. **Monitoring:** Surveillez les logs pour voir si le cache est utilisé
4. **Updates:** Simplement push sur GitHub, RunPod rebuild automatiquement

## 📞 Support

- Documentation RunPod: https://docs.runpod.io/
- Documentation Voxtral: https://huggingface.co/mistralai/Voxtral-Small-24B-2507
- Issues GitHub: Créez une issue sur votre repo
