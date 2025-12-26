# PyAnnote Modes - Guide d'utilisation

## 🎯 Vue d'ensemble

Deux nouveaux modes ont été ajoutés pour améliorer la précision de la diarisation en mode PYANNOTE_AUTO :

1. **PYANNOTE_AUTO V2** (Per-Segment Transcription) - Le plus précis mais le plus lent
2. **HYBRID_V2** (Voxtral Speaker ID + PyAnnote Correction) - Bon compromis vitesse/précision

## 📊 Comparaison des modes

| Mode | Vitesse | Précision Speaker | Précision Texte | Use Case |
|------|---------|-------------------|-----------------|----------|
| **VOXTRAL_SPEAKER_ID** | ⚡⚡⚡ Rapide | 🟡 Bonne (contexte) | 🟢 Excellente | Production standard |
| **PYANNOTE_AUTO (default)** | ⚡⚡ Moyen | ❌ Problématique | 🔴 Mauvaise | **NE PAS UTILISER** |
| **PYANNOTE_AUTO V2** | 🐌 Très lent | 🟢 Excellente (voix) | 🟢 Excellente | Précision maximale |
| **HYBRID_V2** | ⚡⚡ Moyen | 🟢 Excellente | 🟢 Excellente | **RECOMMANDÉ** pour PyAnnote |

## 🚀 Configuration

### Option 1: PYANNOTE_AUTO V2 (Per-Segment)

Active la transcription segment par segment - chaque segment PyAnnote est transcrit individuellement.

**Variable d'environnement:**
```bash
PYANNOTE_PER_SEGMENT=1
```

**Dans RunPod:**
```
Environment Variables:
PYANNOTE_PER_SEGMENT = 1
DIARIZATION_MODE = PYANNOTE_AUTO
```

**Avantages:**
- ✅ Chaque segment a son propre texte précis
- ✅ Pas de problème d'attribution de texte
- ✅ Respect parfait des changements de speaker détectés par PyAnnote

**Inconvénients:**
- ❌ **TRÈS LENT**: ~155 appels Voxtral pour un audio de 5 minutes
- ❌ Coût élevé en tokens et temps GPU
- ❌ Peut prendre 10-15 minutes pour un appel de 5 minutes

**Quand l'utiliser:**
- Transcriptions critiques où la précision absolue est requise
- Appels courts (< 2 minutes)
- Environnement de développement/test

---

### Option 2: HYBRID_V2 (Recommandé)

Combine Voxtral SPEAKER_ID (pour le texte et l'identification contextuelle) avec PyAnnote (pour les timestamps de voix).

**Variables d'environnement:**
```bash
PYANNOTE_PER_SEGMENT=0  # ou ne pas définir la variable
DIARIZATION_MODE=PYANNOTE_AUTO
```

**Fonctionnement:**
1. Voxtral transcrit TOUT l'audio avec identification des speakers (comme VOXTRAL_SPEAKER_ID)
2. PyAnnote détecte les changements de voix avec timestamps précis
3. Fusion intelligente: texte de Voxtral + validation/correction par PyAnnote

**Avantages:**
- ✅ Rapide (1 seul appel Voxtral + PyAnnote)
- ✅ Texte de qualité (Voxtral)
- ✅ Correction des erreurs de Voxtral par PyAnnote
- ✅ **Meilleur des deux mondes**

**Inconvénients:**
- 🟡 Peut avoir quelques conflits si Voxtral et PyAnnote sont en désaccord
- 🟡 Dans ce cas, PyAnnote est prioritaire

**Quand l'utiliser:**
- **Production** - C'est le mode recommandé pour PyAnnote
- Quand tu veux la reconnaissance vocale de PyAnnote sans le coût de V2
- Alternative à VOXTRAL_SPEAKER_ID avec meilleure détection vocale

---

## 🔧 Configuration complète RunPod

### Exemple 1: HYBRID_V2 (Production recommandée)

```bash
# Diarisation
DIARIZATION_MODE=PYANNOTE_AUTO
EXACT_TWO=1
MAX_SPEAKERS=2

# Mode Hybrid V2 activé automatiquement (pas besoin de PYANNOTE_PER_SEGMENT)
# PYANNOTE_PER_SEGMENT=0  # <-- Optionnel, c'est la valeur par défaut

# Autres paramètres standards
HF_TOKEN=ton_token_huggingface
MODEL_ID=mistralai/Voxtral-Small-24B-2507
ENABLE_SENTIMENT=1
```

### Exemple 2: PYANNOTE_AUTO V2 (Maximum précision)

```bash
# Diarisation
DIARIZATION_MODE=PYANNOTE_AUTO
EXACT_TWO=1
MAX_SPEAKERS=2

# Activer le mode per-segment
PYANNOTE_PER_SEGMENT=1

# Autres paramètres
HF_TOKEN=ton_token_huggingface
MODEL_ID=mistralai/Voxtral-Small-24B-2507
ENABLE_SENTIMENT=1
```

### Exemple 3: VOXTRAL_SPEAKER_ID (Fallback rapide)

```bash
# Mode classique Voxtral uniquement
DIARIZATION_MODE=VOXTRAL_SPEAKER_ID

# Autres paramètres
HF_TOKEN=ton_token_huggingface
MODEL_ID=mistralai/Voxtral-Small-24B-2507
ENABLE_SENTIMENT=1
```

---

## 📈 Temps d'exécution estimés

Pour un appel de **5 minutes** (300 secondes) :

| Mode | Temps d'exécution | Breakdown |
|------|-------------------|-----------|
| VOXTRAL_SPEAKER_ID | ~90-120s | Voxtral: 90s |
| HYBRID_V2 | ~150-180s | Voxtral: 90s + PyAnnote: 60s |
| PYANNOTE_AUTO V2 | ~900-1200s (15-20 min) | PyAnnote: 60s + Voxtral×155: 840s |

---

## 🎯 Recommandations

### Pour la production
**Utiliser HYBRID_V2** (PYANNOTE_AUTO sans PYANNOTE_PER_SEGMENT)
- Bon compromis vitesse/précision
- Correction vocale de PyAnnote
- Coût raisonnable

### Pour le développement/test
**Tester PYANNOTE_AUTO V2** sur quelques appels courts
- Vérifier la qualité maximale
- Comparer avec HYBRID_V2
- Décider si la précision supplémentaire vaut le coût

### Si problèmes avec PyAnnote
**Revenir à VOXTRAL_SPEAKER_ID**
- Mode stable et rapide
- Fonctionne toujours
- Bonne qualité générale

---

## 🐛 Dépannage

### PYANNOTE_AUTO V2 trop lent
**Solution:** Passer à HYBRID_V2 en retirant `PYANNOTE_PER_SEGMENT=1`

### HYBRID_V2 a des erreurs d'attribution
**Solution:** Vérifier les logs `[HYBRID_V2] Speaker conflict` et voir si PyAnnote corrige bien

### Tous les modes PyAnnote ont des problèmes
**Solution:** Revenir à `DIARIZATION_MODE=VOXTRAL_SPEAKER_ID`

---

## 📝 Logs à surveiller

### HYBRID_V2
```
[PYANNOTE_AUTO] Using improved hybrid: Voxtral speaker ID + PyAnnote timestamp correction
[HYBRID_V2] Starting improved hybrid mode
[HYBRID_V2] Voxtral identified X segments
[HYBRID_V2] Speaker conflict at XX.Xs: Voxtral=Agent, PyAnnote=Client - Using PyAnnote
[HYBRID_V2] Client sentiment: X (confidence: 0.XX)
```

### PYANNOTE_AUTO V2
```
[PYANNOTE_AUTO] Using per-segment transcription (slower but more accurate)
[PYANNOTE_V2] Starting per-segment transcription for X segments
[PYANNOTE_V2] Transcribing segment 1/155: Agent (2.3s)
[PYANNOTE_V2]   → 'Bonjour madame...'
[PYANNOTE_V2] Client sentiment: X (confidence: 0.XX)
```

---

## 💡 Conseils

1. **Commencer par HYBRID_V2** - C'est le meilleur compromis
2. **Tester sur quelques appels** avant de déployer en production
3. **Surveiller les logs** pour voir si PyAnnote corrige beaucoup d'erreurs de Voxtral
4. **Si budget serré** - rester sur VOXTRAL_SPEAKER_ID
5. **Si précision critique** - utiliser PYANNOTE_AUTO V2 malgré le coût

---

## 🔄 Migration depuis l'ancien PYANNOTE_AUTO

L'ancien mode PYANNOTE_AUTO (sans les corrections) **ne fonctionnait pas correctement** :
- Tous les segments Client avaient du texte vide
- Tout le texte était attribué à l'Agent
- Sentiment du Client impossible à calculer

**Migration recommandée:**
```bash
# Avant (ne fonctionnait pas)
DIARIZATION_MODE=PYANNOTE_AUTO

# Après (fonctionne correctement)
DIARIZATION_MODE=PYANNOTE_AUTO  # Active automatiquement HYBRID_V2

# Ou pour précision maximale
DIARIZATION_MODE=PYANNOTE_AUTO
PYANNOTE_PER_SEGMENT=1
```
