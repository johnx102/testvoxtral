#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, base64, tempfile, uuid, requests, json, traceback, re
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline,
    pipeline as hf_pipeline
)

# Voxtral - Requires transformers >= 4.54.0 and mistral-common[audio] >= 1.8.1
try:
    from transformers import VoxtralForConditionalGeneration  # type: ignore
    _HAS_VOXTRAL_CLASS = True
except Exception as e:
    print(f"[ERROR] VoxtralForConditionalGeneration not found: {e}")
    print(f"[ERROR] Make sure you have: transformers>=4.54.0 and mistral-common[audio]>=1.8.1")
    from transformers import AutoModel  # type: ignore
    VoxtralForConditionalGeneration = AutoModel  # type: ignore
    _HAS_VOXTRAL_CLASS = True  # Use AutoModel as fallback

from pydub import AudioSegment
from pyannote.audio import Pipeline
import runpod

# ---------------------------
# Logging minimal
# ---------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print(msg, flush=True)

# ---------------------------
# Env / Config - MODE HYBRIDE ULTRA-RAPIDE
# ---------------------------
APP_VERSION = os.environ.get("APP_VERSION", "ultra-fast-hybrid-v1.0")

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Voxtral-Small-24B-2507").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "664"))       
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "9000"))      
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Sentiment - NOUVEAU : Analyse par Voxtral
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"

# Détection des annonces/IVR - NOUVEAU
ENABLE_SINGLE_VOICE_DETECTION = os.environ.get("ENABLE_SINGLE_VOICE_DETECTION", "1") == "1"
SINGLE_VOICE_DETECTION_TIMEOUT = int(os.environ.get("SINGLE_VOICE_DETECTION_TIMEOUT", "30"))
SINGLE_VOICE_SUMMARY_TOKENS = int(os.environ.get("SINGLE_VOICE_SUMMARY_TOKENS", "48"))

# Diarization - MODES AVANCÉS
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "2"))
EXACT_TWO = os.environ.get("EXACT_TWO", "1") == "1"
MIN_SEG_DUR = float(os.environ.get("MIN_SEG_DUR", "5.0"))
MIN_SPEAKER_TIME = float(os.environ.get("MIN_SPEAKER_TIME", "8.0"))
MERGE_CONSECUTIVE = os.environ.get("MERGE_CONSECUTIVE", "1") == "1"
HYBRID_MODE = os.environ.get("HYBRID_MODE", "1") == "1"
AGGRESSIVE_MERGE = os.environ.get("AGGRESSIVE_MERGE", "1") == "1"
VOXTRAL_SPEAKER_ID = os.environ.get("VOXTRAL_SPEAKER_ID", "1") == "1"  # NOUVEAU : Speaker ID par Voxtral
PYANNOTE_AUTO = os.environ.get("PYANNOTE_AUTO", "0") == "1"  # NOUVEAU : PyAnnote auto pur
DETECT_GLOBAL_SWAP = os.environ.get("DETECT_GLOBAL_SWAP", "0") == "1"  # NOUVEAU : Détection inversion globale Agent/Client

# Transcription & résumé
STRICT_TRANSCRIPTION = os.environ.get("STRICT_TRANSCRIPTION", "1") == "1"
SMART_SUMMARY = os.environ.get("SMART_SUMMARY", "1") == "1"  
EXTRACTIVE_SUMMARY = os.environ.get("EXTRACTIVE_SUMMARY", "0") == "1"

# Libellés des rôles (après mapping)
ROLE_LABELS = [r.strip() for r in os.environ.get("ROLE_LABELS", "Agent,Client").split(",") if r.strip()]

# Globals
_processor = None
_model = None
_diarizer = None
_sentiment_clf = None
_sentiment_zero_shot = None
_alignment_model = None
_alignment_tokenizer = None
# ---------------------------
# Helpers généraux
# ---------------------------
def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path

def _b64_to_tmp(b64: str) -> str:
    raw = base64.b64decode(b64)
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(raw)
    return path

def check_gpu_memory():
    """Vérifier la mémoire GPU disponible"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        free = total - cached
        
        log(f"[GPU] Total: {total:.1f}GB | Allocated: {allocated:.1f}GB | Cached: {cached:.1f}GB | Free: {free:.1f}GB")
        
        if total < 45:
            log(f"[ERROR] GPU has only {total:.1f}GB - Voxtral Small needs ~55GB minimum")
            return False
        elif free < 10:
            log(f"[WARN] Only {free:.1f}GB free - might cause OOM")
            
        return True
    return False

# ---------------------------
# Voxtral - OPTIMISÉ POUR SMALL
# ---------------------------
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    log(f"[INIT] Loading Voxtral: {MODEL_ID}")

    try:
        proc_kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            proc_kwargs["token"] = HF_TOKEN
        log("[INIT] Loading processor...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, **proc_kwargs)
        log("[INIT] Processor loaded successfully")
    except Exception as e:
        log(f"[ERROR] Failed to load processor: {e}")
        raise

    # MÉMOIRE GPU OPTIMISÉE pour Small 24B
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    log(f"[INIT] Using dtype: {dtype}")
    
    mdl_kwargs = {
        "dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    try:
        log("[INIT] Loading model... (this may take several minutes)")
        if _HAS_VOXTRAL_CLASS:
            _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **mdl_kwargs)
        else:
            raise RuntimeError("Transformers sans VoxtralForConditionalGeneration. Utiliser transformers@main/nightly.")
        log("[INIT] Model loaded successfully")
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        raise

    try:
        p = next(_model.parameters())
        log(f"[INIT] Voxtral device={p.device}, dtype={p.dtype}")
        
        gpu_params = sum(1 for p in _model.parameters() if p.device.type == 'cuda')
        total_params = sum(1 for p in _model.parameters())
        gpu_ratio = gpu_params / total_params if total_params > 0 else 0
        log(f"[INIT] GPU params ratio: {gpu_ratio:.2f} ({gpu_params}/{total_params})")
        
        if gpu_ratio < 0.8:
            log(f"[WARN] Only {gpu_ratio:.1%} of model on GPU - expect slow performance")
            
    except Exception as e:
        log(f"[WARN] Could not check model device: {e}")

    log("[INIT] Voxtral ready.")
    return _processor, _model

def _move_to_device_no_cast(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
    return batch

def _input_len_from_batch(inputs) -> int:
    try:
        if isinstance(inputs, dict) and "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            return inputs["input_ids"].shape[1]
        elif hasattr(inputs, "input_ids") and isinstance(inputs.input_ids, torch.Tensor):
            return inputs.input_ids.shape[1]
    except Exception:
        pass
    return 0

def run_voxtral(conversation: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
    processor, model = load_voxtral()

    try:
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True)
    except (TypeError, ValueError):
        inputs = processor.apply_chat_template(conversation)

    device = _device_str()
    inputs = _move_to_device_no_cast(inputs, device)

    # Pour les longues générations (>3000 tokens), utiliser une petite température
    # pour éviter les boucles de répétition avec do_sample=False
    use_sampling = max_new_tokens > 3000

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=use_sampling,
        temperature=0.1 if use_sampling else None,  # Très faible température pour rester déterministe
        top_p=0.95 if use_sampling else None,
    )

    t0 = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    dt = round(time.time() - t0, 3)

    inp_len = _input_len_from_batch(inputs)
    decoded = processor.batch_decode(outputs[:, inp_len:], skip_special_tokens=True)
    return {"text": decoded[0] if decoded else "", "latency_s": dt}

def run_voxtral_with_timeout(conversation: List[Dict[str, Any]], max_new_tokens: int, timeout: int = 45) -> Dict[str, Any]:
    """Wrapper simplifié sans signaux Unix"""
    try:
        log(f"[VOXTRAL] Starting inference (max_tokens={max_new_tokens})")
        start_time = time.time()
        result = run_voxtral(conversation, max_new_tokens)
        duration = time.time() - start_time
        log(f"[VOXTRAL] Completed in {duration:.2f}s")
        return result
    except Exception as e:
        log(f"[ERROR] Voxtral inference failed: {e}")
        return {"text": "", "latency_s": 0}

def _build_conv_transcribe_ultra_strict(local_path: str, language: Optional[str]) -> List[Dict[str, Any]]:
    """Version ultra-stricte pour éviter les hallucinations"""
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": local_path},
            {"type": "text", "text": "lang:fr [TRANSCRIBE] Écris exactement ce qui est dit, mot pour mot, en français uniquement."}
        ],
    }]

def clean_transcription_result(text: str, duration_s: float) -> str:
    """Nettoie le résultat de transcription pour éliminer les aberrations"""
    if not text:
        return ""
    
    text = text.strip()
    
    # Filtre : Segments suspects (hallucinations communes de Voxtral)
    suspicious_phrases = [
        "so, you're making", "in fact, i have", "tell me the price", "thank you very much",
        "i think it's", "after that, it can", "and sometimes it even", "what is your age",
        "a telephone number"
    ]
    
    text_lower = text.lower()
    for phrase in suspicious_phrases:
        if phrase in text_lower and duration_s < 3.0:
            log(f"[FILTER] Suspicious phrase detected: '{text}' (duration: {duration_s:.1f}s)")
            return ""
    
    # Filtre : Mots anglais suspects dans conversation française
    english_words = ["for", "this", "week", "yes", "of", "course", "telephone", "number", "what", "age", "please", "right", "side"]
    words = text.lower().split()
    english_ratio = sum(1 for word in words if word in english_words) / max(len(words), 1)
    
    if english_ratio > 0.4 and duration_s < 4.0:
        log(f"[FILTER] Too much English detected: '{text}' (ratio: {english_ratio:.1%})")
        return ""
    
    # Filtre : Segments trop longs pour la durée audio (hallucination)
    words_count = len(text.split())
    max_expected_words = int(duration_s * 4)
    
    if words_count > max_expected_words * 1.5:
        log(f"[FILTER] Too many words for duration: {words_count} words in {duration_s:.1f}s")
        return " ".join(text.split()[:max_expected_words])
    
    return text

# ---------------------------
# Forced Alignment - Word-level timestamps (WhisperX-style)
# ---------------------------
_alignment_model = None
_alignment_tokenizer = None

def load_alignment_model():
    """
    Charge le modèle Wav2Vec2 pour forced alignment.
    Utilisé pour obtenir des timestamps précis au niveau des mots.
    """
    global _alignment_model, _alignment_tokenizer
    if _alignment_model is not None and _alignment_tokenizer is not None:
        return _alignment_model, _alignment_tokenizer

    # Modèle français optimisé pour l'alignment
    alignment_model_id = "facebook/wav2vec2-large-xlsr-53-french"
    log(f"[ALIGN] Loading alignment model: {alignment_model_id}")

    try:
        import torchaudio
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        device = _device_str()
        _alignment_tokenizer = Wav2Vec2Processor.from_pretrained(alignment_model_id)

        # Force safetensors loading to avoid torch.load security warning
        _alignment_model = Wav2Vec2ForCTC.from_pretrained(
            alignment_model_id,
            use_safetensors=True
        ).to(device)
        _alignment_model.eval()

        log(f"[ALIGN] Alignment model loaded on {device}")
        return _alignment_model, _alignment_tokenizer
    except Exception as e:
        log(f"[ALIGN] WARNING: Failed to load alignment model: {e}")
        log(f"[ALIGN] Will fall back to segment-level timestamps")
        return None, None

def align_words_to_audio(text: str, audio_path: str) -> List[Dict[str, Any]]:
    """
    Aligne le texte sur l'audio pour obtenir des timestamps au niveau des mots.
    Inspiré de WhisperX.

    Returns:
        Liste de dicts avec {word, start, end}
    """
    try:
        import torchaudio
        model, tokenizer = load_alignment_model()

        if model is None or tokenizer is None:
            log("[ALIGN] Skipping word alignment - model not available")
            return []

        # Charger l'audio
        speech, sample_rate = torchaudio.load(audio_path)

        # Resample si nécessaire (Wav2Vec2 attend 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            speech = resampler(speech)
            sample_rate = 16000

        # Mono
        if speech.shape[0] > 1:
            speech = torch.mean(speech, dim=0, keepdim=True)

        device = _device_str()
        speech = speech.to(device)

        # Tokenization
        inputs = tokenizer(speech.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inférence
        with torch.no_grad():
            logits = model(**inputs).logits

        # Décodage avec timestamps
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

        # Extraction des timestamps au niveau des caractères
        # Note: Implémentation simplifiée - WhisperX utilise CTC segmentation plus sophistiqué
        duration = speech.shape[1] / sample_rate
        words = text.split()
        word_timestamps = []

        if len(words) > 0:
            # Distribution uniforme comme approximation
            # TODO: Implémenter CTC segmentation propre si nécessaire
            time_per_word = duration / len(words)
            for i, word in enumerate(words):
                word_timestamps.append({
                    "word": word,
                    "start": i * time_per_word,
                    "end": (i + 1) * time_per_word
                })

        log(f"[ALIGN] Aligned {len(word_timestamps)} words")
        return word_timestamps

    except Exception as e:
        log(f"[ALIGN] ERROR during alignment: {e}")
        return []

def assign_speakers_to_words_iou(
    word_timestamps: List[Dict[str, Any]],
    pyannote_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Assigne les speakers aux mots en utilisant l'algorithme IoU de WhisperX.

    Args:
        word_timestamps: Liste de {word, start, end}
        pyannote_segments: Liste de {speaker, start, end}

    Returns:
        Liste de {word, start, end, speaker}
    """
    import pandas as pd
    import numpy as np

    if not word_timestamps or not pyannote_segments:
        return word_timestamps

    log(f"[IOU] Assigning speakers to {len(word_timestamps)} words using {len(pyannote_segments)} PyAnnote segments")

    # Convertir pyannote_segments en DataFrame
    diarize_df = pd.DataFrame(pyannote_segments)

    # Pour chaque mot, trouver le speaker avec la plus grande intersection
    for word in word_timestamps:
        if 'start' not in word or 'end' not in word:
            continue

        # Calculer l'intersection avec tous les segments PyAnnote
        diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])

        # Filtrer les intersections positives
        dia_tmp = diarize_df[diarize_df['intersection'] > 0]

        if len(dia_tmp) > 0:
            # Grouper par speaker et sommer les intersections
            # Le speaker avec la plus grande intersection gagne
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            word["speaker"] = speaker
        else:
            # Pas d'intersection - assigner au speaker le plus proche temporellement
            word["speaker"] = None

    assigned_count = sum(1 for w in word_timestamps if w.get("speaker"))
    log(f"[IOU] Assigned speakers to {assigned_count}/{len(word_timestamps)} words")

    return word_timestamps

# ---------------------------
# Diarization
# ---------------------------
def load_diarizer():
    global _diarizer
    if _diarizer is not None:
        return _diarizer
    log(f"[INIT] Loading diarization: {DIAR_MODEL}")
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN
    _diarizer = Pipeline.from_pretrained(DIAR_MODEL, **kwargs)
    try:
        if torch.cuda.is_available():
            try:
                _diarizer.to(torch.device("cuda"))
            except TypeError:
                _diarizer.to("cuda")
            log("[INIT] Diarizer moved to CUDA.")
        else:
            log("[INIT] CUDA not available; diarizer on CPU.")
    except Exception as e:
        log(f"[WARN] Could not move diarizer to CUDA: {e}. Keeping on CPU.")
    return _diarizer

# ---------------------------
# Sentiment
# ---------------------------
def load_sentiment():
    global _sentiment_clf, _sentiment_zero_shot
    if not ENABLE_SENTIMENT:
        return None

    device_idx = SENTIMENT_DEVICE  # -1 => CPU
    if SENTIMENT_TYPE == "zero-shot":
        if _sentiment_zero_shot is not None:
            return _sentiment_zero_shot
        log(f"[INIT] Loading zero-shot sentiment on device {device_idx}: {SENTIMENT_MODEL}")
        _sentiment_zero_shot = hf_pipeline(
            "zero-shot-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            device=device_idx,
            model_kwargs={"use_safetensors": True}
        )
        log("[INIT] Zero-shot sentiment ready.")
        return _sentiment_zero_shot
    else:
        if _sentiment_clf is not None:
            return _sentiment_clf
        log(f"[INIT] Loading classifier sentiment on device {device_idx}: {SENTIMENT_MODEL}")
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, use_safetensors=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, use_safetensors=True)
        _sentiment_clf = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=device_idx, return_all_scores=True, truncation=True
        )
        log("[INIT] Classifier sentiment ready.")
        return _sentiment_clf

def _label_fr(en_label: str) -> str:
    m = {"negative": "mauvais", "neutral": "neutre", "positive": "bon"}
    key = en_label.lower()
    if key.startswith("label_"):
        try:
            idx = int(key.split("_")[-1])
            key = ["negative", "neutral", "positive"][idx]
        except Exception:
            pass
    return m.get(key, en_label)

def classify_sentiment_with_voxtral(text: str) -> Dict[str, Any]:
    """
    Analyse de sentiment équilibrée pour conversations téléphoniques
    """
    if not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    
    text_lower = text.lower()
    
    # MOTS VRAIMENT NÉGATIFS (besoin de contexte fort)
    strong_negative = [
        "pas content du tout", "vraiment pas content", "pas contente du tout",
        "très mécontent", "très déçu", "complètement déçu",
        "annulé à cause", "annuler parce que", "se foutre la honte",
        "inadmissible", "inacceptable", "catastrophe", "scandaleux",
        # Colère explicite
        "en colère", "énervé", "furieux", "exaspéré",
        "ras-le-bol", "j'en ai marre", "c'est honteux",
        "n'importe quoi", "vous vous moquez", "c'est une blague",
        "c'est du foutage", "limite", "sérieux là",
        "c'est abusé", "vous abusez", "c'est grave"
    ]

    # MOTS MOYENNEMENT NÉGATIFS (besoin d'accumulation)
    mild_negative = [
        "pas content", "pas satisfait", "déçu", "problème",
        "difficile", "compliqué", "pas possible",
        # Frustration/Impatience
        "encore", "toujours pas", "ça fait longtemps",
        "combien de fois", "à chaque fois", "ça suffit",
        "patienter", "retard", "toujours le même",
        "ça traîne", "trop long", "pas normal"
    ]
    
    # INDICATEURS POSITIFS (seuil plus bas)
    positive_indicators = [
        "merci", "parfait", "très bien", "super", "excellent",
        "c'est bon", "d'accord", "ça marche", "pas de problème",
        "bonne journée", "au revoir", "confirmé", "réglé",
        "je vous remercie", "merci beaucoup", "c'est gentil",
        "avec plaisir", "volontiers", "pas de souci"
    ]
    
    # PHRASES QUI ANNULENT LE NÉGATIF
    neutral_phrases = [
        "c'est bon alors", "ah d'accord", "pas de problème",
        "c'est réglé", "c'est confirmé", "ça marche"
    ]
    
    # Comptage
    strong_neg_count = sum(1 for phrase in strong_negative if phrase in text_lower)
    mild_neg_count = sum(1 for word in mild_negative if word in text_lower)
    positive_count = sum(1 for word in positive_indicators if word in text_lower)
    neutral_found = any(phrase in text_lower for phrase in neutral_phrases)
    
    # LOGIQUE DE DÉCISION AJUSTÉE
    
    # 1. Fort négatif = négatif
    if strong_neg_count >= 1:
        log(f"[SENTIMENT] Strong negative: {strong_neg_count} phrases")
        return {
            "label_en": "negative",
            "label_fr": "mauvais",
            "confidence": 0.90,
            "scores": {"negative": 0.90, "neutral": 0.08, "positive": 0.02}
        }
    
    # 2. Beaucoup de positifs = positif (SEUIL PLUS CONSERVATEUR)
    if positive_count >= 3 and mild_neg_count == 0:
        log(f"[SENTIMENT] Clear positive: {positive_count} indicators")
        return {
            "label_en": "positive",
            "label_fr": "bon",
            "confidence": 0.85,
            "scores": {"negative": 0.05, "neutral": 0.10, "positive": 0.85}
        }
    
    # 3. Phrases neutres qui annulent le négatif
    if neutral_found and mild_neg_count <= 1:
        log(f"[SENTIMENT] Neutral phrases override")
        return {
            "label_en": "neutral",
            "label_fr": "neutre",
            "confidence": 0.80,
            "scores": {"negative": 0.10, "neutral": 0.80, "positive": 0.10}
        }
    
    # 4. Accumulation de négatifs moyens
    if mild_neg_count >= 3:
        log(f"[SENTIMENT] Multiple mild negatives: {mild_neg_count}")
        return {
            "label_en": "negative",
            "label_fr": "mauvais",
            "confidence": 0.75,
            "scores": {"negative": 0.75, "neutral": 0.20, "positive": 0.05}
        }
    
    # 5. Appel court et poli = probablement positif
    word_count = len(text.split())
    if word_count < 100 and positive_count >= 1 and mild_neg_count == 0:
        log(f"[SENTIMENT] Short polite call: positive")
        return {
            "label_en": "positive",
            "label_fr": "bon",
            "confidence": 0.70,
            "scores": {"negative": 0.10, "neutral": 0.20, "positive": 0.70}
        }
    
    # 6. Voxtral pour les cas ambigus
    instruction = (
        "Analyse le sentiment de cette conversation téléphonique professionnelle.\n"
        "Réponds par UN SEUL MOT : satisfaisant, neutre, ou insatisfaisant\n\n"
        "Critères :\n"
        "- satisfaisant : client content, remerciements, problème résolu, confirmation positive\n"
        "- insatisfaisant : client mécontent, frustré, problème non résolu\n"
        "- neutre : simple échange d'information\n\n"
        "NOTE : La plupart des appels courts et polis sont satisfaisants.\n\n"
        f"Conversation : {text[:1500]}"
    )
    
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": instruction}]
    }]
    
    try:
        result = run_voxtral_with_timeout(conversation, max_new_tokens=16, timeout=60)
        response = (result.get("text") or "").strip().lower()
        
        log(f"[SENTIMENT] Voxtral: '{response}' (pos:{positive_count} neg:{mild_neg_count})")
        
        if "insatisfaisant" in response:
            # Override si BEAUCOUP de positifs (seuil augmenté)
            if positive_count >= 4:
                return {
                    "label_en": "neutral",
                    "label_fr": "neutre",
                    "confidence": 0.65,
                    "scores": {"negative": 0.25, "neutral": 0.65, "positive": 0.10}
                }
            return {
                "label_en": "negative",
                "label_fr": "mauvais",
                "confidence": 0.75,
                "scores": {"negative": 0.75, "neutral": 0.20, "positive": 0.05}
            }
        elif "satisfaisant" in response and "insatisfaisant" not in response:
            return {
                "label_en": "positive",
                "label_fr": "bon",
                "confidence": 0.80,
                "scores": {"negative": 0.05, "neutral": 0.15, "positive": 0.80}
            }
        else:
            # Neutre avec tendance positive si présence de mots positifs
            if positive_count >= 1:
                return {
                    "label_en": "positive",
                    "label_fr": "bon",
                    "confidence": 0.65,
                    "scores": {"negative": 0.10, "neutral": 0.25, "positive": 0.65}
                }
            return {
                "label_en": "neutral",
                "label_fr": "neutre",
                "confidence": 0.75,
                "scores": {"negative": 0.15, "neutral": 0.75, "positive": 0.10}
            }
            
    except Exception as e:
        log(f"[SENTIMENT] Error: {e}")
        # Par défaut neutre/positif pour les appels courts
        if word_count < 100 and positive_count > 0:
            return {
                "label_en": "positive",
                "label_fr": "bon",
                "confidence": 0.60,
                "scores": {"negative": 0.10, "neutral": 0.30, "positive": 0.60}
            }
        return {
            "label_en": "neutral",
            "label_fr": "neutre",
            "confidence": 0.60,
            "scores": {"negative": 0.20, "neutral": 0.60, "positive": 0.20}
        }


def remove_repetitive_loops(text: str, max_repetitions: int = 5) -> str:
    """
    Détecte et supprime les boucles de répétition dans le texte généré par Voxtral.

    Cas typiques:
    - Mot répété avec espaces: "hello hello hello..."
    - Pattern collé: "michel.michel.michel..." (sans espaces)
    - Phrases complètes: "je vais voir ça je vais voir ça je vais voir ça"
    """
    if not text or len(text) < 20:
        return text

    # Cas 1: Détection de patterns courts répétés (ex: "michel.michel.michel...")
    # Chercher un pattern de 3-100 caractères répété consécutivement (étendu pour phrases)
    for pattern_len in range(3, min(101, len(text) // 3)):  # Étendu de 20 à 100 chars
        if len(text) < pattern_len * max_repetitions:
            continue

        # Extraire le pattern potentiel depuis la fin du texte
        # (la répétition est souvent à la fin)
        pattern = text[-pattern_len:]

        # Compter combien de fois ce pattern se répète à la fin
        repetitions = 0
        pos = len(text)
        while pos >= pattern_len:
            if text[pos - pattern_len:pos] == pattern:
                repetitions += 1
                pos -= pattern_len
            else:
                break

        if repetitions > max_repetitions:
            # Tronquer au premier pattern
            truncate_pos = pos
            log(f"[CLEANUP] Detected pattern loop: '{pattern[:30]}...' repeated {repetitions} times, truncating from {len(text)} to {truncate_pos} chars")
            return text[:truncate_pos]

    # Cas 2: Mots séparés par espaces
    words = text.split()
    if len(words) < 10:
        return text

    cleaned_words = []
    repetition_count = 1
    last_word = None

    for word in words:
        if word == last_word:
            repetition_count += 1
            if repetition_count > max_repetitions:
                log(f"[CLEANUP] Detected word loop: '{word}' repeated {repetition_count} times, truncating")
                break
        else:
            repetition_count = 1
            last_word = word
        cleaned_words.append(word)

    cleaned_text = " ".join(cleaned_words)

    if len(cleaned_text) < len(text) * 0.8:
        log(f"[CLEANUP] Removed repetitive content: {len(text)} -> {len(cleaned_text)} chars")

    return cleaned_text


def validate_sentiment_coherence(text: str, sentiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valide et corrige le sentiment si incohérent avec le contenu
    """
    if not sentiment or not text:
        return sentiment
    
    # Phrases qui indiquent TOUJOURS un problème
    strong_negative_phrases = [
        "pas content", "pas contente",
        "annulé", "annuler à cause",
        "se foutre la honte",
        "pas satisfait", "pas satisfaite",
        "je veux parler", "parler avec le responsable",
        "ce n'est pas normal",
        "c'est inadmissible",
        # Mécontentement explicite
        "ça ne va pas du tout", "c'est la dernière fois",
        "je change de", "je vais voir ailleurs",
        "un concurrent", "chez un concurrent",
        "je vais me plaindre", "porter plainte",
        "service client nul", "nul", "zéro",
        "catastrophique", "lamentable",
        # Menaces/Escalade
        "avocat", "mon avocat",
        "résilier", "résiliation",
        "remboursement immédiat", "je veux un remboursement",
        # Expressions de colère
        "j'en ai assez", "j'en peux plus",
        "c'est toujours pareil", "à chaque fois c'est pareil",
        "vous êtes nuls", "incompétent"
    ]
    
    text_lower = text.lower()
    has_strong_negative = any(phrase in text_lower for phrase in strong_negative_phrases)
    
    # Si on trouve une phrase fortement négative mais sentiment non négatif
    if has_strong_negative and sentiment.get("label_fr") != "mauvais":
        log(f"[SENTIMENT_VALIDATION] Overriding sentiment to negative due to strong negative phrases")
        return {
            "label_en": "negative",
            "label_fr": "mauvais", 
            "confidence": 0.95,
            "scores": {"negative": 0.95, "neutral": 0.04, "positive": 0.01}
        }
    
    return sentiment
    
def classify_sentiment(text: str) -> Dict[str, Any]:
    """
    Interface de sentiment - utilise Voxtral avec validation
    """
    if not ENABLE_SENTIMENT or not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    
    # Analyse Voxtral
    sentiment = classify_sentiment_with_voxtral(text)
    
    # Validation et correction si nécessaire
    sentiment = validate_sentiment_coherence(text, sentiment)
    
    return sentiment

def analyze_sentiment_by_speaker(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyse le sentiment pour chaque speaker séparément.
    FOCUS: Le client est généralement le premier speaker ou celui qui parle le moins.
    """
    if not ENABLE_SENTIMENT or not segments:
        return {}

    # Grouper les paroles par speaker
    speaker_texts = {}
    speaker_durations = {}

    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if not speaker or not text:
            continue

        if speaker not in speaker_texts:
            speaker_texts[speaker] = []
            speaker_durations[speaker] = 0.0

        speaker_texts[speaker].append(text)
        speaker_durations[speaker] += (seg.get("end", 0) - seg.get("start", 0))

    # Analyser le sentiment pour chaque speaker
    speaker_sentiments = {}
    for speaker, texts in speaker_texts.items():
        combined_text = " ".join(texts)
        sentiment = classify_sentiment(combined_text)
        speaker_sentiments[speaker] = sentiment
        log(f"[SENTIMENT_BY_SPEAKER] {speaker}: {sentiment.get('label_fr')} (confidence: {sentiment.get('confidence', 0):.2f})")

    return speaker_sentiments


def get_client_sentiment(speaker_sentiments: Dict[str, Dict[str, Any]], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Identifie le sentiment du CLIENT (pas de l'agent).
    Heuristique: Le client est généralement le premier speaker ou celui qui parle le moins.
    """
    if not speaker_sentiments:
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    # Compter le temps de parole par speaker
    speaker_durations = {}
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker:
            duration = seg.get("end", 0) - seg.get("start", 0)
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration

    # Le client parle généralement MOINS que l'agent
    if speaker_durations:
        client_speaker = min(speaker_durations.items(), key=lambda x: x[1])[0]
        log(f"[CLIENT_DETECTION] Client identifié: {client_speaker} (temps de parole: {speaker_durations[client_speaker]:.1f}s)")
        return speaker_sentiments.get(client_speaker, {"label_en": None, "label_fr": None, "confidence": None, "scores": None})

    # Fallback: premier speaker
    first_speaker = segments[0].get("speaker") if segments else None
    if first_speaker and first_speaker in speaker_sentiments:
        return speaker_sentiments[first_speaker]

    return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}


def aggregate_mood(weighted: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    total = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    total_w = 0.0
    for s, w in weighted:
        if not s or not s.get("scores"):
            continue
        for k in total.keys():
            total[k] += s["scores"].get(k, 0.0) * w
        total_w += w
    if total_w <= 0:
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    ssum = sum(total.values())
    norm = {k: (v/ssum if ssum > 0 else 0.0) for k, v in total.items()}
    label_en = max(norm.items(), key=lambda kv: kv[1])[0]
    return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": norm[label_en], "scores": norm}

# ---------------------------
# RÉSUMÉS OPTIMISÉS
# ---------------------------
def generate_natural_summary(full_transcript: str, language: Optional[str] = None) -> str:
    """Génère un résumé naturel et utile sans structure artificielle."""
    if not full_transcript.strip():
        return "Conversation vide."
    
    lang_prefix = f"lang:{language} " if language else ""
    instruction = (
        f"{lang_prefix}Résume cette conversation en 1-2 phrases simples et claires. "
        "Dis juste l'essentiel : qui appelle pourquoi, et ce qui va se passer. "
        "Sois direct et naturel, sans format particulier."
    )
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{instruction}\n\nConversation:\n{full_transcript}"}
        ]
    }]
    
    try:
        result = run_voxtral_with_timeout(conversation, max_new_tokens=72, timeout=25)
        summary = (result.get("text") or "").strip()
        
        summary = clean_generated_summary(summary)
        
        if not summary or len(summary) < 10:
            log("[WARN] Generated summary too short, using extractive fallback")
            return create_extractive_summary(full_transcript)
        
        return summary
        
    except Exception as e:
        log(f"[ERROR] Natural summary generation failed: {e}")
        return create_extractive_summary(full_transcript)

def clean_generated_summary(summary: str) -> str:
    """Nettoie et valide les résumés générés par Voxtral."""
    if not summary:
        return ""
    
    unwanted_starts = [
        "voici un résumé", "résumé de la conversation", "cette conversation",
        "dans cette conversation", "le résumé", "il s'agit"
    ]
    
    summary_lower = summary.lower().strip()
    for start in unwanted_starts:
        if summary_lower.startswith(start):
            sentences = re.split(r'[.!?]+', summary)
            for sentence in sentences[1:]:
                cleaned = sentence.strip()
                if len(cleaned) > 15 and not any(cleaned.lower().startswith(u) for u in unwanted_starts):
                    summary = cleaned
                    break
    
    patterns_to_remove = [
        r'décision[/:].*?étape[s]?\s*:', 
        r'prochaine[s]?\s*étape[s]?\s*:',
        r'action[s]?\s*à\s*prendre\s*:',
        r'conclusion\s*:',
        r'format\s*attendu\s*:'
    ]
    
    for pattern in patterns_to_remove:
        summary = re.sub(pattern, '', summary, flags=re.IGNORECASE).strip()
    
    summary = re.sub(r'\.{2,}', '.', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    
    artificial_indicators = [
        "format attendu", "décision/prochaine étape", "action à prendre",
        "structure suivante", "points suivants"
    ]
    
    if any(indicator in summary.lower() for indicator in artificial_indicators):
        return ""
    
    return summary

def create_extractive_summary(transcript: str) -> str:
    """Résumé extractif amélioré basé sur l'analyse du contenu."""
    if not transcript:
        return "Conversation indisponible."
    
    lines = [l.strip() for l in transcript.split('\n') if l.strip() and ':' in l]
    
    if not lines:
        return "Conversation très courte."
    
    summary_parts = []
    
    # Identifie le motif de l'appel (client)
    client_lines = [l for l in lines if l.lower().startswith('client:')]
    for line in client_lines:
        text = line.replace('Client:', '').strip().lower()
        
        if any(kw in text for kw in ['rendez-vous', 'rdv', 'prendre', 'avancer', 'reporter']):
            summary_parts.append("Appel pour un rendez-vous")
            break
        elif any(kw in text for kw in ['information', 'renseignement', 'savoir']):
            summary_parts.append("Demande d'information")
            break
        elif any(kw in text for kw in ['problème', 'souci', 'bug']):
            summary_parts.append("Signalement d'un problème")
            break
    
    # Identifie la réponse de l'agent
    agent_lines = [l for l in lines if l.lower().startswith('agent:')]
    for line in agent_lines:
        text = line.replace('Agent:', '').strip().lower()
        
        if any(kw in text for kw in ['je vais', 'on va', 'nous allons']):
            action = line.replace('Agent:', '').strip()
            if len(action) > 10:
                summary_parts.append(f"L'agent va {action.lower()}")
                break
        elif any(kw in text for kw in ['pas possible', 'impossible', 'désolé']):
            summary_parts.append("L'agent ne peut pas satisfaire la demande")
            break
        elif any(kw in text for kw in ['disponible', 'libre', 'possible']):
            summary_parts.append("Créneau trouvé")
            break
    
    if len(summary_parts) < 2:
        substantive_lines = []
        for line in lines:
            text = line.split(':', 1)[1].strip() if ':' in line else line
            if (len(text) > 15 and 
                not any(pol in text.lower() for pol in ['bonjour', 'au revoir', 'merci', 'bonne journée'])):
                substantive_lines.append(text)
        
        for text in substantive_lines[:2]:
            if len(summary_parts) < 3:
                summary_parts.append(text)
    
    if not summary_parts:
        return "Conversation brève sans motif identifié."
    
    return ". ".join(summary_parts) + "."

def select_best_summary_approach(transcript: str) -> str:
    """Sélectionne la meilleure approche de résumé selon le contenu."""
    lines = transcript.split('\n')
    total_words = len(transcript.split())

    # Compter les segments avec speaker (lignes contenant ":")
    speaker_lines = [l for l in lines if ':' in l and l.strip()]

    # Détection des annonces/IVR
    if len(lines) == 1 and "System:" in transcript:
        # C'est probablement une annonce
        content = transcript.replace("System:", "").strip()
        if len(content) > 20:
            return f"Annonce automatique: {content[:100]}{'...' if len(content) > 100 else ''}"
        else:
            return f"Message système: {content}"

    if total_words < 20:
        return "Conversation très brève."

    # Utiliser le résumé génératif si conversation substantielle (3+ échanges ou 30+ mots)
    if total_words > 30 and len(speaker_lines) >= 3:
        generative_summary = generate_natural_summary(transcript)

        if (generative_summary and
            len(generative_summary) > 15 and
            not any(bad in generative_summary.lower() for bad in ['format', 'structure', 'décision/'])):
            return generative_summary

    return create_extractive_summary(transcript)

# ---------------------------
# Post-traitements segments
# ---------------------------
def _enforce_max_two_speakers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    speakers = list({s["speaker"] for s in segments})
    if len(speakers) <= 2:
        return segments

    dur = {}
    cent = {}
    for s in segments:
        d = float(s["end"]) - float(s["start"])
        spk = s["speaker"]
        mid = (float(s["start"]) + float(s["end"])) / 2.0
        dur[spk] = dur.get(spk, 0.0) + d
        sm, ds = cent.get(spk, (0.0, 0.0))
        cent[spk] = (sm + mid * d, ds + d)

    top2 = sorted(dur.keys(), key=lambda k: dur[k], reverse=True)[:2]
    centroids = {}
    for spk in top2:
        sm, ds = cent[spk]
        centroids[spk] = (sm / ds) if ds > 0 else 0.0

    def nearest(mid):
        best, bestd = None, 1e18
        for spk, c in centroids.items():
            dd = abs(mid - c)
            if dd < bestd:
                best, bestd = spk, dd
        return best

    for s in segments:
        if s["speaker"] not in top2:
            mid = (float(s["start"]) + float(s["end"])) / 2.0
            s["speaker"] = nearest(mid)

    return segments

def _map_roles(segments: List[Dict[str, Any]]):
    seen = []
    for s in segments:
        spk = s["speaker"]
        if spk not in seen:
            seen.append(spk)
    mapping = {}
    for i, spk in enumerate(seen):
        if i < len(ROLE_LABELS):
            mapping[spk] = ROLE_LABELS[i]
    for s in segments:
        s["speaker"] = mapping.get(s["speaker"], s["speaker"])

def merge_consecutive_segments(segments: List[Dict[str, Any]], max_gap: float = 2.0) -> List[Dict[str, Any]]:
    """
    Fusionne les segments consécutifs du même speaker avec un gap faible.
    Version moins agressive que ultra_aggressive_merge.
    """
    if not segments:
        return segments

    merged = []
    current = segments[0].copy()

    log(f"[MERGE] Starting with {len(segments)} segments")

    for next_seg in segments[1:]:
        same_speaker = current["speaker"] == next_seg["speaker"]
        gap = float(next_seg["start"]) - float(current["end"])

        # Fusion conservative : seulement même speaker ET gap faible
        if same_speaker and gap <= max_gap:
            # Fusionner
            current["end"] = next_seg["end"]
            current_text = current.get("text", "").strip()
            next_text = next_seg.get("text", "").strip()
            if current_text and next_text:
                current["text"] = f"{current_text} {next_text}"
            elif next_text:
                current["text"] = next_text
        else:
            # Ne peut pas fusionner
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    log(f"[MERGE] Result: {len(segments)} → {len(merged)} segments")
    return merged

def optimize_diarization_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Optimise les segments de diarization avec fusion ultra-agressive."""
    log(f"[OPTIMIZE] Input: {len(segments)} segments")

    # Filtrer les segments trop courts
    filtered = []
    for seg in segments:
        duration = float(seg["end"]) - float(seg["start"])
        if duration >= MIN_SEG_DUR:
            filtered.append(seg)
        else:
            log(f"[OPTIMIZE] Filtered short segment: {duration:.1f}s < {MIN_SEG_DUR}s")

    log(f"[OPTIMIZE] After filtering: {len(filtered)} segments")

    # Fusion ULTRA-AGRESSIVE
    if AGGRESSIVE_MERGE and filtered:
        filtered = ultra_aggressive_merge(filtered, max_gap=3.0)
    elif MERGE_CONSECUTIVE and filtered:
        filtered = merge_consecutive_segments(filtered, max_gap=2.0)
    
    # Supprimer les speakers avec trop peu de temps total
    if MIN_SPEAKER_TIME > 0:
        speaker_times = {}
        for seg in filtered:
            speaker = seg["speaker"]
            duration = float(seg["end"]) - float(seg["start"])
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
        
        valid_speakers = {s for s, t in speaker_times.items() if t >= MIN_SPEAKER_TIME}
        if valid_speakers:
            before_count = len(filtered)
            filtered = [seg for seg in filtered if seg["speaker"] in valid_speakers]
            log(f"[OPTIMIZE] Kept speakers: {valid_speakers}")
            log(f"[OPTIMIZE] After speaker filtering: {before_count} → {len(filtered)} segments")
    
    log(f"[OPTIMIZE] Final: {len(segments)} → {len(filtered)} segments")
    return filtered

# ---------------------------
# DÉTECTION ET TRAITEMENT DES ANNONCES/IVR
# ---------------------------

def detect_single_voice_content(wav_path: str, language: Optional[str]) -> Dict[str, Any]:
    """
    Détecte si l'enregistrement contient une annonce/IVR (une seule voix)
    et génère un résumé adapté.
    """
    log("[SINGLE_VOICE] Detecting single voice content...")
    
    # Instruction spécialisée pour détecter le type de contenu
    instruction = (
        f"lang:{language or 'fr'} "
        "Analyse ce contenu audio et détermine s'il s'agit de :\n"
        "1) Une conversation entre deux personnes (Agent + Client)\n"
        "2) Une annonce automatique/IVR (une seule voix)\n"
        "3) Un message vocal laissé par un client\n\n"
        "Si c'est une annonce/IVR, résume le contenu en 1-2 phrases.\n"
        "Si c'est une conversation, réponds 'CONVERSATION'.\n"
        "Si c'est un message vocal, réponds 'MESSAGE_VOCAL'."
    )
    
    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": instruction}
        ]
    }]
    
    try:
        result = run_voxtral_with_timeout(conversation, max_new_tokens=64, timeout=SINGLE_VOICE_DETECTION_TIMEOUT)
        response = (result.get("text") or "").strip().lower()
        
        log(f"[SINGLE_VOICE] Detection result: '{response}'")
        
        if "conversation" in response:
            return {"type": "conversation", "summary": None}
        elif "message_vocal" in response:
            return {"type": "voicemail", "summary": response}
        else:
            # Probablement une annonce/IVR
            return {"type": "announcement", "summary": response}
            
    except Exception as e:
        log(f"[SINGLE_VOICE] Detection failed: {e}")
        return {"type": "unknown", "summary": None}

def transcribe_single_voice_content(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    Mode spécialisé pour les enregistrements à une seule voix (annonces/IVR).
    """
    log("[SINGLE_VOICE] Processing single voice content...")
    
    # Durée max
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        log(f"[SINGLE_VOICE] Audio duration: {est_dur:.1f}s")
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s). Increase MAX_DURATION_S or send shorter file."}
    except Exception as e:
        log(f"[SINGLE_VOICE] Could not check duration: {e}")
        est_dur = 180.0  # Durée par défaut
    
    # Transcription simple sans diarization
    conv_simple = _build_conv_transcribe_ultra_strict(wav_path, language or "fr")
    out_simple = run_voxtral_with_timeout(conv_simple, max_new_tokens=max_new_tokens, timeout=60)
    full_text = (out_simple.get("text") or "").strip()
    
    if not full_text:
        return {"error": "Empty transcription for single voice content"}
    
    # Créer un segment unique
    segments = [{
        "speaker": "System",
        "start": 0.0,
        "end": est_dur,
        "text": full_text,
        "mood": None
    }]
    
    full_transcript = f"System: {full_text}"
    result = {"segments": segments, "transcript": full_transcript}
    
    # Résumé spécialisé pour les annonces
    if with_summary:
        log("[SINGLE_VOICE] Generating announcement summary...")
        summary_instruction = (
            f"lang:{language or 'fr'} "
            "Résume cette annonce automatique en 1-2 phrases claires. "
            "Indique le type d'annonce (accueil, information, etc.) et le message principal."
        )
        
        summary_conv = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"{summary_instruction}\n\nContenu: {full_text}"}
            ]
        }]
        
        try:
            summary_result = run_voxtral_with_timeout(summary_conv, max_new_tokens=SINGLE_VOICE_SUMMARY_TOKENS, timeout=20)
            result["summary"] = (summary_result.get("text") or "").strip()
        except Exception as e:
            log(f"[SINGLE_VOICE] Summary generation failed: {e}")
            result["summary"] = f"Annonce automatique: {full_text[:100]}..."
    
    # Sentiment neutre pour les annonces
    if ENABLE_SENTIMENT:
        result["mood_overall"] = {
            "label_en": "neutral",
            "label_fr": "neutre",
            "confidence": 0.95,
            "scores": {"negative": 0.0, "neutral": 0.95, "positive": 0.05}
        }
        result["mood_by_speaker"] = {"System": result["mood_overall"]}
    
    log("[SINGLE_VOICE] Single voice processing completed")
    return result

# ---------------------------
# MODES DE DIARIZATION AVANCÉS
# ---------------------------

def diarize_with_voxtral_speaker_id(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    MODE VOXTRAL SPEAKER ID : 
    Demande directement à Voxtral d'identifier les speakers par le contexte conversationnel
    """
    log(f"[VOXTRAL_ID] Starting Voxtral-based speaker identification: language={language}")
    
    # Durée max
    try:
        log("[VOXTRAL_ID] Checking audio duration...")
        import soundfile as sf
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        log(f"[VOXTRAL_ID] Audio duration: {est_dur:.1f}s")
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s). Increase MAX_DURATION_S or send shorter file."}
    except Exception as e:
        log(f"[VOXTRAL_ID] Could not check duration: {e}")

    # INSTRUCTION SPÉCIALISÉE POUR IDENTIFICATION DES SPEAKERS
    instruction = (
        f"lang:{language or 'fr'} "
        "Transcris cette conversation téléphonique en identifiant clairement qui parle. "
        "Format requis:\n"
        "Agent: [ce que dit l'agent/professionnel]\n"
        "Client: [ce que dit le client/appelant]\n\n"
        "Indications pour identifier les speakers:\n"
        "- Agent = celui qui répond, dit 'bonjour', 'cabinet', 'je vais voir', 'on vous rappelle'\n"
        "- Client = celui qui appelle, dit 'je voudrais', 'ma femme', 'est-ce que je peux'\n\n"
        "Transcris mot à mot et sépare clairement chaque prise de parole."
    )
    
    # TRANSCRIPTION AVEC IDENTIFICATION DES SPEAKERS EN UNE PASSE
    log("[VOXTRAL_ID] Starting transcription with speaker identification...")
    conv_speaker_id = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": instruction}
        ]
    }]
    
    # Calcul automatique des tokens basé sur la durée (évite les hallucinations)
    # Formule: ~15-18 tokens par seconde selon la densité de parole
    speaker_tokens = int(est_dur * 18)  # Légèrement augmenté de 15 à 18 pour marge
    log(f"[VOXTRAL_ID] Using {speaker_tokens} tokens (estimated from {est_dur:.1f}s audio)")
    out_speaker_id = run_voxtral_with_timeout(conv_speaker_id, max_new_tokens=speaker_tokens, timeout=120)
    speaker_transcript = (out_speaker_id.get("text") or "").strip()

    if not speaker_transcript:
        log("[VOXTRAL_ID] Empty speaker identification result, falling back to hybrid mode")
        return diarize_then_transcribe_hybrid(wav_path, language, max_new_tokens, with_summary)

    # Nettoyer les boucles de répétition (bug Voxtral)
    original_len = len(speaker_transcript)
    speaker_transcript = remove_repetitive_loops(speaker_transcript, max_repetitions=5)

    # Vérification de cohérence : ratio chars/seconde anormalement élevé ?
    chars_per_sec = len(speaker_transcript) / max(est_dur, 1)
    if chars_per_sec > 50:  # Normal = 15-30 chars/sec pour du français parlé
        log(f"[VOXTRAL_ID] WARNING: Suspicious chars/sec ratio: {chars_per_sec:.1f} (expected ~15-30). Possible repetition bug.")

    if original_len != len(speaker_transcript):
        log(f"[VOXTRAL_ID] Cleaned repetitive loops: {original_len} -> {len(speaker_transcript)} chars")

    log(f"[VOXTRAL_ID] Speaker identification completed: {len(speaker_transcript)} chars in {out_speaker_id.get('latency_s', 0):.1f}s")

    # PARSING DU RÉSULTAT AVEC SPEAKERS IDENTIFIÉS
    log("[VOXTRAL_ID] Parsing speaker-identified transcript...")
    segments = parse_speaker_identified_transcript(speaker_transcript, est_dur)
    
    if not segments:
        log("[VOXTRAL_ID] Failed to parse speaker transcript, falling back to hybrid mode")
        return diarize_then_transcribe_hybrid(wav_path, language, max_new_tokens, with_summary)
    
    log(f"[VOXTRAL_ID] Parsed {len(segments)} segments from speaker-identified transcript")

    # POST-TRAITEMENT STANDARD
    segments = _enforce_max_two_speakers(segments)
    segments = detect_and_fix_speaker_inversion(segments)  # AVANT _map_roles
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript}
    
    # Résumé
    if with_summary:
        log("[VOXTRAL_ID] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript)
    
    # Sentiment par speaker (analyse améliorée)
    if ENABLE_SENTIMENT:
        log("[VOXTRAL_ID] Computing sentiment analysis by speaker...")

        # Analyse le sentiment pour chaque speaker séparément
        speaker_sentiments = analyze_sentiment_by_speaker(segments)

        if speaker_sentiments:
            # Attribuer le sentiment spécifique à chaque segment
            for seg in segments:
                speaker = seg.get("speaker")
                if speaker and speaker in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[speaker]

            # Mood overall (moyenne pondérée de tous les speakers)
            weighted_moods = []
            for seg in segments:
                if seg.get("text") and seg.get("mood"):
                    duration = float(seg["end"]) - float(seg["start"])
                    weighted_moods.append((seg["mood"], duration))
            result["mood_overall"] = aggregate_mood(weighted_moods) if weighted_moods else None

            # Mood par speaker (déjà calculé)
            result["mood_by_speaker"] = speaker_sentiments

            # NOUVEAU: Sentiment du client identifié
            client_mood = get_client_sentiment(speaker_sentiments, segments)
            result["mood_client"] = client_mood

            # Protection maximale contre None
            confidence_raw = client_mood.get('confidence')
            if confidence_raw is None:
                confidence = 0.0
                log(f"[VOXTRAL_ID] WARNING: Client sentiment confidence is None, using 0.0")
            else:
                confidence = float(confidence_raw)

            label_fr = client_mood.get('label_fr') or 'inconnu'
            log(f"[VOXTRAL_ID] Client sentiment: {label_fr} (confidence: {confidence:.2f})")

    log("[VOXTRAL_ID] Voxtral speaker identification completed successfully")
    return result

def merge_micro_segments(segments: List[Dict[str, Any]], max_duration: float = 2.0, max_gap: float = 1.0) -> List[Dict[str, Any]]:
    """
    Fusionne les micro-segments consécutifs du même speaker.
    Utile pour éviter les lignes multiples quand quelqu'un épelle des chiffres ou lettres.

    Args:
        segments: Liste des segments à fusionner
        max_duration: Durée max d'un segment pour être considéré comme "micro" (secondes)
        max_gap: Gap max entre deux segments pour les fusionner (secondes)

    Returns:
        Liste des segments fusionnés
    """
    if not segments:
        return segments

    log(f"[MERGE_MICRO] Input: {len(segments)} segments")

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i].copy()

        # Fusionner les segments consécutifs du même speaker s'ils sont très courts
        while i + 1 < len(segments):
            next_seg = segments[i + 1]
            duration_current = current["end"] - current["start"]
            duration_next = next_seg["end"] - next_seg["start"]
            gap = next_seg["start"] - current["end"]

            # Fusionner si:
            # - Même speaker
            # - Au moins un des deux segments est court (< max_duration)
            # - Le gap entre eux est petit (< max_gap)
            if (current["speaker"] == next_seg["speaker"] and
                (duration_current < max_duration or duration_next < max_duration) and
                gap < max_gap):

                # Fusionner le texte
                current["text"] = current["text"] + " " + next_seg["text"]
                current["end"] = next_seg["end"]

                log(f"[MERGE_MICRO] Merged: '{current['text'][:50]}...' ({duration_current:.1f}s + {duration_next:.1f}s)")

                i += 1
            else:
                break

        merged.append(current)
        i += 1

    log(f"[MERGE_MICRO] Output: {len(merged)} segments (reduced by {len(segments) - len(merged)})")
    return merged

def parse_speaker_identified_transcript(transcript: str, total_duration: float) -> List[Dict[str, Any]]:
    """
    Parse le transcript avec speakers identifiés par Voxtral
    Supporte deux formats:
    1. Multi-lignes: chaque ligne = "Speaker: text"
    2. Inline: tout sur une ligne avec "Agent: ... Client: ... Agent: ..."
    """
    if not transcript:
        return []

    segments = []
    current_time = 0.0

    # D'abord essayer de parser ligne par ligne
    lines = transcript.split('\n')
    log(f"[PARSE] Processing {len(lines)} lines from speaker transcript")

    valid_lines = []

    # Si une seule ligne, utiliser le parsing inline
    if len(lines) == 1 or (len(lines) == 2 and not lines[1].strip()):
        # Parsing inline: chercher tous les "Agent:" et "Client:"
        text = lines[0].strip()

        # Regex pour trouver tous les patterns "Agent:" ou "Client:" suivis de texte
        import re
        pattern = r'(Agent|Client):\s*([^:]+?)(?=\s*(?:Agent:|Client:|$))'
        matches = re.findall(pattern, text, re.IGNORECASE)

        log(f"[PARSE] Inline mode: found {len(matches)} speaker segments")

        # Mapper Agent/Client vers SPEAKER_00/SPEAKER_01 pour que _map_roles fonctionne
        speaker_mapping = {}
        speaker_counter = 0

        for speaker_raw, text_part in matches:
            speaker_normalized = speaker_raw.capitalize()

            # Créer le mapping vers SPEAKER_XX
            if speaker_normalized not in speaker_mapping:
                speaker_mapping[speaker_normalized] = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1

            speaker = speaker_mapping[speaker_normalized]
            text_part = text_part.strip()

            if text_part:
                # Nettoyer les répétitions
                text_part = remove_repetitive_loops(text_part, max_repetitions=3)
                if text_part:
                    valid_lines.append((speaker, text_part))
    else:
        # Parsing multi-lignes (mode original)
        speaker_mapping = {}
        speaker_counter = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Chercher les patterns Speaker: text
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker_part = parts[0].strip()
                    text_part = parts[1].strip()

                    # Normaliser les noms de speakers vers Agent/Client
                    if any(word in speaker_part.lower() for word in ['agent', 'professionnel', 'cabinet', 'secrétaire']):
                        speaker_normalized = "Agent"
                    elif any(word in speaker_part.lower() for word in ['client', 'appelant', 'patient']):
                        speaker_normalized = "Client"
                    elif speaker_part.lower() in ['agent', 'client']:
                        speaker_normalized = speaker_part.capitalize()
                    else:
                        # Speaker non identifié, essayer de deviner par le contenu
                        if any(word in text_part.lower() for word in ['cabinet', 'bonjour', 'je vais voir', 'on vous rappelle']):
                            speaker_normalized = "Agent"
                        elif any(word in text_part.lower() for word in ['je voudrais', 'ma femme', 'est-ce que je peux']):
                            speaker_normalized = "Client"
                        else:
                            speaker_normalized = "Agent"  # Défaut

                    # Mapper vers SPEAKER_XX pour que _map_roles fonctionne
                    if speaker_normalized not in speaker_mapping:
                        speaker_mapping[speaker_normalized] = f"SPEAKER_{speaker_counter:02d}"
                        speaker_counter += 1

                    speaker = speaker_mapping[speaker_normalized]

                    if text_part:  # Seulement si il y a du texte
                        # Nettoyer les répétitions dans chaque segment aussi
                        text_part = remove_repetitive_loops(text_part, max_repetitions=3)
                        if text_part:  # Vérifier qu'il reste du texte après nettoyage
                            valid_lines.append((speaker, text_part))
    
    if not valid_lines:
        log("[PARSE] No valid speaker lines found")
        return []
    
    # Créer les segments avec timestamps approximatifs
    total_words = sum(len(text.split()) for _, text in valid_lines)

    for i, (speaker, text) in enumerate(valid_lines):
        # Calculer la durée basée sur le nombre de mots
        word_count = len(text.split())
        segment_duration = (word_count / total_words) * total_duration if total_words > 0 else total_duration / len(valid_lines)

        start_time = current_time
        end_time = min(current_time + segment_duration, total_duration)

        segments.append({
            "speaker": speaker,
            "start": start_time,
            "end": end_time,
            "text": text,
            "mood": None
        })

        current_time = end_time

        log(f"[PARSE] Segment {i+1}: {speaker} ({start_time:.1f}s-{end_time:.1f}s) '{text[:30]}...'")

    # Fusionner les micro-segments consécutifs du même speaker
    segments = merge_micro_segments(segments, max_duration=2.0, max_gap=1.0)

    return segments

def apply_pyannote_per_segment_transcription(wav_path: str, raw_segments: List[Dict], language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    PYANNOTE AUTO V2: Transcrit chaque segment PyAnnote individuellement
    Plus lent mais beaucoup plus précis car chaque segment a son propre texte
    """
    import soundfile as sf
    import numpy as np
    from pydub import AudioSegment
    import tempfile

    log(f"[PYANNOTE_V2] Starting per-segment transcription for {len(raw_segments)} segments")

    # Charger l'audio complet
    audio_data, sample_rate = sf.read(wav_path)

    # Post-traitement: enforce max 2 speakers, fix inversions, map roles
    raw_segments = _enforce_max_two_speakers(raw_segments)
    raw_segments = detect_and_fix_speaker_inversion(raw_segments)
    _map_roles(raw_segments)

    # Fusionner les segments très courts du même speaker
    merged_segments = []
    i = 0
    while i < len(raw_segments):
        current = raw_segments[i].copy()

        # Fusionner les segments consécutifs du même speaker s'ils sont courts
        while i + 1 < len(raw_segments):
            next_seg = raw_segments[i + 1]
            duration_current = current["end"] - current["start"]
            duration_next = next_seg["end"] - next_seg["start"]
            gap = next_seg["start"] - current["end"]

            # Fusionner si même speaker ET (segment court OU gap très petit)
            if (current["speaker"] == next_seg["speaker"] and
                ((duration_current < 2.0 or duration_next < 2.0) and gap < 1.0)):
                current["end"] = next_seg["end"]
                i += 1
            else:
                break

        merged_segments.append(current)
        i += 1

    log(f"[PYANNOTE_V2] After merging short segments: {len(merged_segments)} segments")

    # Transcrire chaque segment individuellement
    transcribed_segments = []

    for idx, seg in enumerate(merged_segments):
        start_time = seg["start"]
        end_time = seg["end"]
        speaker = seg["speaker"]
        duration = end_time - start_time

        # Ignorer les segments trop courts (< 0.3s)
        if duration < 0.3:
            log(f"[PYANNOTE_V2] Skipping very short segment {idx+1}/{len(merged_segments)}: {duration:.1f}s")
            continue

        # Extraire le segment audio
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_audio = audio_data[start_sample:end_sample]

        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, segment_audio, sample_rate)

        try:
            # Transcrire ce segment uniquement
            log(f"[PYANNOTE_V2] Transcribing segment {idx+1}/{len(merged_segments)}: {speaker} ({duration:.1f}s)")

            # Construction de la conversation pour Voxtral
            instruction = "Transcribe."
            if language:
                instruction = f"Transcribe in {language}."

            conv = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": tmp_path},
                    {"type": "text", "text": instruction}
                ]
            }]

            # Adapter max_new_tokens à la durée du segment (environ 3 mots par seconde)
            segment_max_tokens = min(int(duration * 5), 200)  # Max 200 tokens par segment

            result = run_voxtral_with_timeout(conv, max_new_tokens=segment_max_tokens, timeout=30)
            text = result.get("text", "").strip()

            if text:
                transcribed_segments.append({
                    "speaker": speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "mood": None
                })
                log(f"[PYANNOTE_V2]   → '{text[:50]}...'")
            else:
                log(f"[PYANNOTE_V2]   → (empty transcription)")

        except Exception as e:
            log(f"[PYANNOTE_V2] Error transcribing segment {idx+1}: {e}")
            # Ajouter un segment vide en cas d'erreur
            transcribed_segments.append({
                "speaker": speaker,
                "start": start_time,
                "end": end_time,
                "text": "",
                "mood": None
            })
        finally:
            # Nettoyer le fichier temporaire
            try:
                os.remove(tmp_path)
            except:
                pass

    log(f"[PYANNOTE_V2] Transcription completed: {len(transcribed_segments)} segments with text")

    # Construire le transcript complet
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in transcribed_segments if s.get("text"))

    result = {
        "segments": transcribed_segments,
        "transcript": full_transcript
    }

    # Résumé si demandé
    if with_summary:
        result["summary"] = select_best_summary_approach(full_transcript)

    # Analyse de sentiment si activé
    if ENABLE_SENTIMENT:
        log("[PYANNOTE_V2] Computing sentiment analysis by speaker...")
        speaker_sentiments = analyze_sentiment_by_speaker(transcribed_segments)

        if speaker_sentiments:
            # Attribuer le sentiment à chaque segment
            for seg in transcribed_segments:
                speaker = seg.get("speaker")
                if speaker and speaker in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[speaker]

            # Mood overall
            weighted_moods = []
            for seg in transcribed_segments:
                if seg.get("text") and seg.get("mood"):
                    duration = float(seg["end"]) - float(seg["start"])
                    weighted_moods.append((seg["mood"], duration))
            result["mood_overall"] = aggregate_mood(weighted_moods) if weighted_moods else None

            result["mood_by_speaker"] = speaker_sentiments

            # Sentiment du client
            client_mood = get_client_sentiment(speaker_sentiments, transcribed_segments)
            result["mood_client"] = client_mood

            confidence_raw = client_mood.get('confidence')
            if confidence_raw is None:
                confidence = 0.0
                log(f"[PYANNOTE_V2] WARNING: Client sentiment confidence is None, using 0.0")
            else:
                confidence = float(confidence_raw)

            label_fr = client_mood.get('label_fr') or 'inconnu'
            log(f"[PYANNOTE_V2] Client sentiment: {label_fr} (confidence: {confidence:.2f})")

    return result

def diarize_with_pyannote_auto(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    MODE PYANNOTE AUTO PUR : 
    Laisse PyAnnote décider automatiquement du nombre de speakers
    """
    log(f"[PYANNOTE_AUTO] Starting PyAnnote automatic speaker detection")
    
    # Durée max
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s)."}
    except Exception as e:
        log(f"[PYANNOTE_AUTO] Could not check duration: {e}")

    # DIARIZATION AVEC NOMBRE DE SPEAKERS FORCÉ
    log("[PYANNOTE_AUTO] Running automatic diarization...")
    dia = load_diarizer()

    # FORCER num_speakers pour éviter la détection de la musique d'attente
    diarization_kwargs = {}
    if EXACT_TWO or MAX_SPEAKERS == 2:
        diarization_kwargs["num_speakers"] = 2
        log(f"[PYANNOTE_AUTO] Forcing exactly 2 speakers (EXACT_TWO={EXACT_TWO}, MAX_SPEAKERS={MAX_SPEAKERS})")

    diarization = dia(wav_path, **diarization_kwargs)
    
    # Collecter tous les segments
    raw_segments = []
    speakers_found = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end)
        }
        raw_segments.append(segment)
        speakers_found.add(speaker)
    
    log(f"[PYANNOTE_AUTO] Auto-detected {len(speakers_found)} speakers: {speakers_found}")
    log(f"[PYANNOTE_AUTO] Generated {len(raw_segments)} raw segments")
    
    # Si trop de speakers détectés, fusionner les moins importants
    if len(speakers_found) > 3:
        log(f"[PYANNOTE_AUTO] Too many speakers detected ({len(speakers_found)}), merging...")
        # Garder seulement les 2 speakers avec le plus de temps de parole
        speaker_times = {}
        for seg in raw_segments:
            duration = seg["end"] - seg["start"]
            speaker_times[seg["speaker"]] = speaker_times.get(seg["speaker"], 0) + duration
        
        top_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)[:2]
        main_speakers = [s[0] for s in top_speakers]
        
        log(f"[PYANNOTE_AUTO] Keeping main speakers: {main_speakers}")
        
        # Réassigner les speakers mineurs vers les principaux
        for seg in raw_segments:
            if seg["speaker"] not in main_speakers:
                # Trouver le speaker principal le plus proche temporellement
                seg_mid = (seg["start"] + seg["end"]) / 2
                best_speaker = main_speakers[0]  # Défaut
                
                for main_seg in raw_segments:
                    if main_seg["speaker"] in main_speakers:
                        main_mid = (main_seg["start"] + main_seg["end"]) / 2
                        if abs(main_mid - seg_mid) < 5.0:  # Dans les 5 secondes
                            best_speaker = main_seg["speaker"]
                            break
                
                seg["speaker"] = best_speaker
    
    # CHOIX: Transcription segment par segment (plus lent mais plus précis)
    # Ou mode hybride classique (plus rapide mais moins précis)
    USE_PER_SEGMENT_TRANSCRIPTION = os.environ.get("PYANNOTE_PER_SEGMENT", "0") == "1"

    if USE_PER_SEGMENT_TRANSCRIPTION:
        log("[PYANNOTE_AUTO] Using per-segment transcription (slower but more accurate)")
        return apply_pyannote_per_segment_transcription(wav_path, raw_segments, language, max_new_tokens, with_summary)
    else:
        # Nouveau mode ALIGNED: WhisperX-style avec forced alignment + IoU mapping
        log("[PYANNOTE_AUTO] Using WhisperX-style alignment: Voxtral transcription + Wav2Vec2 alignment + PyAnnote diarization")
        return apply_pyannote_aligned(wav_path, raw_segments, language, max_new_tokens, with_summary)

def apply_improved_hybrid_mode(wav_path: str, pyannote_segments: List[Dict], language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    MODE HYBRID AMÉLIORÉ:
    1. Utilise Voxtral SPEAKER_ID pour transcrire avec identification des speakers
    2. Utilise les timestamps PyAnnote pour corriger/valider l'attribution
    3. Combine le meilleur des deux approches
    """
    log("[HYBRID_V2] Starting improved hybrid mode")
    log("[HYBRID_V2] Step 1: Voxtral speaker identification")

    # Étape 1: Transcrire avec Voxtral SPEAKER_ID
    # Utiliser la MÊME instruction que VOXTRAL_SPEAKER_ID (qui fonctionne bien)
    instruction = (
        f"lang:{language or 'fr'} "
        "Transcris cette conversation téléphonique en identifiant clairement qui parle. "
        "Format requis:\n"
        "Agent: [ce que dit l'agent/professionnel]\n"
        "Client: [ce que dit le client/appelant]\n\n"
        "Indications pour identifier les speakers:\n"
        "- Agent = celui qui répond, dit 'bonjour', 'cabinet', 'je vais voir', 'on vous rappelle'\n"
        "- Client = celui qui appelle, dit 'je voudrais', 'ma femme', 'est-ce que je peux'\n\n"
        "Transcris mot à mot et sépare clairement chaque prise de parole."
    )

    conv = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": instruction}
        ]
    }]

    voxtral_result = run_voxtral_with_timeout(conv, max_new_tokens=max_new_tokens, timeout=120)
    transcript = voxtral_result.get("text", "").strip()

    if not transcript:
        log("[HYBRID_V2] Empty Voxtral transcription")
        return {"error": "Empty transcription", "segments": [], "transcript": ""}

    # Étape 2: Parser la transcription Voxtral
    log("[HYBRID_V2] Step 2: Parsing Voxtral speaker-identified segments")
    voxtral_segments = parse_speaker_identified_transcript(transcript, total_duration=0)

    if not voxtral_segments:
        log("[HYBRID_V2] Failed to parse speaker segments, using raw transcript")
        return {"error": "Parsing failed", "segments": [], "transcript": transcript}

    log(f"[HYBRID_V2] Voxtral identified {len(voxtral_segments)} segments")

    # Étape 3: Fusionner les segments Voxtral (texte) avec PyAnnote (timestamps)
    log("[HYBRID_V2] Step 3: Merging Voxtral text with PyAnnote timestamps")

    # Préparer les segments PyAnnote
    pyannote_segments = _enforce_max_two_speakers(pyannote_segments)
    pyannote_segments = detect_and_fix_speaker_inversion(pyannote_segments)
    _map_roles(pyannote_segments)

    # Fusionner les segments PyAnnote consécutifs du même speaker pour obtenir les "tours de parole"
    pyannote_turns = []
    if pyannote_segments:
        current_turn = {
            "speaker": pyannote_segments[0]["speaker"],
            "start": pyannote_segments[0]["start"],
            "end": pyannote_segments[0]["end"]
        }

        for i in range(1, len(pyannote_segments)):
            seg = pyannote_segments[i]
            if seg["speaker"] == current_turn["speaker"]:
                # Même speaker, étendre le tour
                current_turn["end"] = seg["end"]
            else:
                # Nouveau speaker, sauver le tour actuel
                pyannote_turns.append(current_turn)
                current_turn = {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"]
                }

        # Ajouter le dernier tour
        pyannote_turns.append(current_turn)

    log(f"[HYBRID_V2] Voxtral: {len(voxtral_segments)} text segments")
    log(f"[HYBRID_V2] PyAnnote: {len(pyannote_turns)} speaking turns")

    # Mapper chaque segment Voxtral à un tour PyAnnote basé sur le speaker et l'ordre
    corrected_segments = []
    voxtral_idx_by_speaker = {"Agent": 0, "Client": 0}

    for turn in pyannote_turns:
        speaker = turn["speaker"]

        # Trouver le prochain segment Voxtral pour ce speaker
        text = ""
        voxtral_idx = voxtral_idx_by_speaker.get(speaker, 0)

        # Chercher le prochain segment Voxtral de ce speaker
        while voxtral_idx < len(voxtral_segments):
            v_seg = voxtral_segments[voxtral_idx]
            if v_seg["speaker"] == speaker:
                text = v_seg["text"]
                voxtral_idx_by_speaker[speaker] = voxtral_idx + 1
                break
            voxtral_idx += 1

        # Créer le segment fusionné
        corrected_segments.append({
            "speaker": speaker,
            "start": turn["start"],  # Timestamp PyAnnote
            "end": turn["end"],      # Timestamp PyAnnote
            "text": text,            # Texte Voxtral
            "mood": None
        })

    # Statistiques
    for speaker in ["Agent", "Client"]:
        used = voxtral_idx_by_speaker.get(speaker, 0)
        total = sum(1 for seg in voxtral_segments if seg["speaker"] == speaker)
        log(f"[HYBRID_V2] {speaker}: used {used}/{total} Voxtral texts")

    log(f"[HYBRID_V2] Created {len(corrected_segments)} corrected segments")

    # Construire le résultat
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in corrected_segments if s.get("text"))

    result = {
        "segments": corrected_segments,
        "transcript": full_transcript
    }

    # Résumé
    if with_summary:
        result["summary"] = select_best_summary_approach(full_transcript)

    # Analyse de sentiment
    if ENABLE_SENTIMENT:
        log("[HYBRID_V2] Computing sentiment analysis by speaker...")
        speaker_sentiments = analyze_sentiment_by_speaker(corrected_segments)

        if speaker_sentiments:
            for seg in corrected_segments:
                speaker = seg.get("speaker")
                if speaker and speaker in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[speaker]

            weighted_moods = []
            for seg in corrected_segments:
                if seg.get("text") and seg.get("mood"):
                    duration = float(seg["end"]) - float(seg["start"])
                    weighted_moods.append((seg["mood"], duration))
            result["mood_overall"] = aggregate_mood(weighted_moods) if weighted_moods else None

            result["mood_by_speaker"] = speaker_sentiments

            client_mood = get_client_sentiment(speaker_sentiments, corrected_segments)
            result["mood_client"] = client_mood

            confidence_raw = client_mood.get('confidence')
            if confidence_raw is None:
                confidence = 0.0
                log(f"[HYBRID_V2] WARNING: Client sentiment confidence is None, using 0.0")
            else:
                confidence = float(confidence_raw)

            label_fr = client_mood.get('label_fr') or 'inconnu'
            log(f"[HYBRID_V2] Client sentiment: {label_fr} (confidence: {confidence:.2f})")

    return result

def apply_pyannote_aligned(wav_path: str, pyannote_segments: List[Dict], language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    MODE PYANNOTE_ALIGNED (WhisperX-style):
    1. Voxtral transcrit l'audio (texte complet, pas de speaker ID)
    2. Forced alignment pour obtenir word-level timestamps
    3. PyAnnote détecte les speakers avec timestamps précis
    4. IoU mapping pour assigner speakers aux mots
    5. Regrouper les mots en segments par speaker
    """
    log("[PYANNOTE_ALIGNED] Starting WhisperX-style alignment mode")
    log("[PYANNOTE_ALIGNED] Step 1: Voxtral transcription (no speaker ID)")

    # Étape 1: Transcrire avec Voxtral SANS speaker ID
    instruction = f"lang:{language or 'fr'} Transcris cette conversation téléphonique mot à mot."

    conv = [{
        "role": "user",
        "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": instruction}
        ]
    }]

    voxtral_result = run_voxtral_with_timeout(conv, max_new_tokens=max_new_tokens, timeout=120)
    transcript = voxtral_result.get("text", "").strip()

    if not transcript:
        log("[PYANNOTE_ALIGNED] Empty Voxtral transcription")
        return {"error": "Empty transcription", "segments": [], "transcript": ""}

    log(f"[PYANNOTE_ALIGNED] Voxtral transcript: {len(transcript)} characters")

    # Étape 2: Forced alignment pour word-level timestamps
    log("[PYANNOTE_ALIGNED] Step 2: Forced alignment for word-level timestamps")
    word_timestamps = align_words_to_audio(transcript, wav_path)

    if not word_timestamps:
        log("[PYANNOTE_ALIGNED] WARNING: Alignment failed, falling back to VOXTRAL_SPEAKER_ID")
        # Fallback vers le mode classique
        return diarize_with_voxtral_speaker_id(wav_path, language, max_new_tokens, with_summary)

    log(f"[PYANNOTE_ALIGNED] Aligned {len(word_timestamps)} words")

    # Étape 3: Préparer les segments PyAnnote
    log("[PYANNOTE_ALIGNED] Step 3: Preparing PyAnnote speaker segments")
    pyannote_segments = _enforce_max_two_speakers(pyannote_segments)
    pyannote_segments = detect_and_fix_speaker_inversion(pyannote_segments)
    _map_roles(pyannote_segments)

    log(f"[PYANNOTE_ALIGNED] PyAnnote: {len(pyannote_segments)} speaker segments")

    # Étape 4: IoU mapping - Assigner speakers aux mots
    log("[PYANNOTE_ALIGNED] Step 4: IoU-based speaker assignment")
    words_with_speakers = assign_speakers_to_words_iou(word_timestamps, pyannote_segments)

    # Étape 5: Regrouper les mots en segments par speaker
    log("[PYANNOTE_ALIGNED] Step 5: Grouping words into speaker segments")
    segments = []
    current_segment = None

    for word in words_with_speakers:
        speaker = word.get("speaker")
        if not speaker:
            continue

        if current_segment is None or current_segment["speaker"] != speaker:
            # Nouveau segment
            if current_segment is not None:
                segments.append(current_segment)

            current_segment = {
                "speaker": speaker,
                "start": word["start"],
                "end": word["end"],
                "text": word["word"],
                "mood": None
            }
        else:
            # Étendre le segment actuel
            current_segment["end"] = word["end"]
            current_segment["text"] += " " + word["word"]

    # Ajouter le dernier segment
    if current_segment is not None:
        segments.append(current_segment)

    log(f"[PYANNOTE_ALIGNED] Created {len(segments)} speaker segments")

    # Construire le résultat
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))

    result = {
        "segments": segments,
        "transcript": full_transcript
    }

    # Résumé
    if with_summary:
        result["summary"] = select_best_summary_approach(full_transcript)

    # Analyse de sentiment
    if ENABLE_SENTIMENT:
        log("[PYANNOTE_ALIGNED] Computing sentiment analysis by speaker...")
        speaker_sentiments = analyze_sentiment_by_speaker(segments)

        if speaker_sentiments:
            for seg in segments:
                speaker = seg.get("speaker")
                if speaker and speaker in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[speaker]

            weighted_moods = []
            for seg in segments:
                if seg.get("text") and seg.get("mood"):
                    duration = float(seg["end"]) - float(seg["start"])
                    weighted_moods.append((seg["mood"], duration))
            result["mood_overall"] = aggregate_mood(weighted_moods) if weighted_moods else None

            result["mood_by_speaker"] = speaker_sentiments

            client_mood = get_client_sentiment(speaker_sentiments, segments)
            result["mood_client"] = client_mood

            confidence_raw = client_mood.get('confidence')
            if confidence_raw is None:
                confidence = 0.0
                log(f"[PYANNOTE_ALIGNED] WARNING: Client sentiment confidence is None, using 0.0")
            else:
                confidence = float(confidence_raw)

            label_fr = client_mood.get('label_fr') or 'inconnu'
            log(f"[PYANNOTE_ALIGNED] Client sentiment: {label_fr} (confidence: {confidence:.2f})")

    return result

def ultra_aggressive_merge(segments: List[Dict[str, Any]], max_gap: float = 3.0) -> List[Dict[str, Any]]:
    """
    Fusion ultra-agressive pour créer de gros blocs cohérents et réduire les erreurs.
    """
    if not segments:
        return segments
    
    merged = []
    current = segments[0].copy()
    
    log(f"[ULTRA_MERGE] Starting with {len(segments)} segments")
    
    for next_seg in segments[1:]:
        same_speaker = current["speaker"] == next_seg["speaker"]
        gap = float(next_seg["start"]) - float(current["end"])
        
        # Fusion plus agressive : même speaker OU gap très petit
        should_merge = (same_speaker and gap <= max_gap) or (gap <= 0.5)
        
        if should_merge:
            # Fusionner
            current["end"] = next_seg["end"]
            
            if same_speaker:
                current_text = current.get("text", "").strip()
                next_text = next_seg.get("text", "").strip()
                if current_text and next_text:
                    current["text"] = f"{current_text} {next_text}"
                elif next_text:
                    current["text"] = next_text
            else:
                log(f"[ULTRA_MERGE] Merging different speakers due to tiny gap: {gap:.2f}s")
                current_dur = float(current["end"]) - float(current["start"])  
                next_dur = float(next_seg["end"]) - float(next_seg["start"])
                if next_dur > current_dur:
                    current["speaker"] = next_seg["speaker"]
        else:
            merged.append(current)
            current = next_seg.copy()
    
    merged.append(current)
    
    log(f"[ULTRA_MERGE] Result: {len(segments)} → {len(merged)} segments")
    return merged
def apply_hybrid_workflow_with_segments(wav_path: str, diar_segments: List[Dict], language: Optional[str], max_new_tokens: int, with_summary: bool, skip_ultra_merge: bool = False):
    """Applique le workflow hybride avec des segments de diarization fournis"""
    # Transcription globale
    conv_global = _build_conv_transcribe_ultra_strict(wav_path, language or "fr")
    out_global = run_voxtral_with_timeout(conv_global, max_new_tokens=max_new_tokens, timeout=60)
    full_text = (out_global.get("text") or "").strip()

    if not full_text:
        return {"error": "Empty global transcription"}

    # Optimisation des segments (skip pour PYANNOTE_AUTO si demandé)
    if skip_ultra_merge:
        log("[HYBRID] Skipping ultra_merge for better PyAnnote accuracy")
        optimized_segments = diar_segments  # Garder les segments PyAnnote bruts
    else:
        optimized_segments = optimize_diarization_segments(diar_segments)
    
    # Attribution du texte (réutilise la logique existante)
    sentences = smart_sentence_split(full_text)
    segments = []
    
    if optimized_segments:
        total_seg_duration = sum(seg["end"] - seg["start"] for seg in optimized_segments)
        sentence_index = 0
        
        for seg in optimized_segments:
            seg_duration = seg["end"] - seg["start"]
            sentence_proportion = seg_duration / total_seg_duration if total_seg_duration > 0 else 1.0/len(optimized_segments)
            sentences_for_segment = max(1, int(len(sentences) * sentence_proportion))
            sentences_for_segment = min(sentences_for_segment, len(sentences) - sentence_index)
            
            if sentences_for_segment > 0:
                seg_sentences = sentences[sentence_index:sentence_index + sentences_for_segment]
                seg_text = " ".join(seg_sentences).strip()
                sentence_index += sentences_for_segment
            else:
                seg_text = ""
            
            segments.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg_text,
                "mood": None
            })
    
    # Post-traitement standard
    segments = _enforce_max_two_speakers(segments)
    segments = detect_and_fix_speaker_inversion(segments)  # AVANT _map_roles
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript}
    
    if with_summary:
        result["summary"] = select_best_summary_approach(full_transcript)
    
    if ENABLE_SENTIMENT:
        log("[PYANNOTE] Computing sentiment analysis by speaker...")

        # Analyse le sentiment pour chaque speaker séparément
        speaker_sentiments = analyze_sentiment_by_speaker(segments)

        if speaker_sentiments:
            # Attribuer le sentiment spécifique à chaque segment
            for seg in segments:
                speaker = seg.get("speaker")
                if speaker and speaker in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[speaker]

            # Mood overall (moyenne pondérée de tous les speakers)
            weighted_moods = []
            for seg in segments:
                if seg.get("text") and seg.get("mood"):
                    duration = float(seg["end"]) - float(seg["start"])
                    weighted_moods.append((seg["mood"], duration))
            result["mood_overall"] = aggregate_mood(weighted_moods) if weighted_moods else None

            # Mood par speaker (déjà calculé)
            result["mood_by_speaker"] = speaker_sentiments

            # NOUVEAU: Sentiment du client identifié
            client_mood = get_client_sentiment(speaker_sentiments, segments)
            result["mood_client"] = client_mood

            if LOG_LEVEL == "DEBUG":
                log(f"[PYANNOTE] DEBUG: client_mood = {client_mood}")
                log(f"[PYANNOTE] DEBUG: speaker_sentiments keys = {list(speaker_sentiments.keys())}")

            # Protection maximale contre None
            confidence_raw = client_mood.get('confidence')
            if confidence_raw is None:
                confidence = 0.0
                log(f"[PYANNOTE] WARNING: Client sentiment confidence is None, using 0.0")
            else:
                confidence = float(confidence_raw)

            label_fr = client_mood.get('label_fr') or 'inconnu'
            log(f"[PYANNOTE] Client sentiment: {label_fr} (confidence: {confidence:.2f})")
    
    return result

# ---------------------------
# Post-traitements segments
# ---------------------------
def intelligent_dialogue_correction(text: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-correction intelligente du dialogue basée sur les patterns de conversation.
    """
    if not segments or len(segments) < 2:
        return segments
    
    log("[CORRECTION] Starting intelligent dialogue correction...")
    
    # Analyse du texte global pour identifier les tours de parole réels
    full_text = " ".join(seg.get("text", "") for seg in segments).strip()
    
    # Patterns qui indiquent un changement de locuteur
    speaker_change_patterns = [
        r'\b(oui|non|d\'accord|très bien|parfait|c\'est ça|exactement)\b\s+',
        r'\b(bonjour|bonsoir|au revoir|merci|à bientôt)\b',
        r'\b(alors|bon|eh bien|donc|en fait)\b\s+',
        r'[.!?]\s+(oui|non|d\'accord|très bien)',
    ]
    
    # Tenter de redécouper le texte de manière plus intelligente
    sentences = smart_sentence_split(full_text)
    
    # Si on a trop de segments pour peu de phrases, c'est probablement sur-segmenté
    if len(segments) > len(sentences) * 1.5:
        log(f"[CORRECTION] Over-segmentation detected: {len(segments)} segments for {len(sentences)} sentences")
        
        # Mode de récupération : créer de gros blocs alternés
        corrected_segments = []
        sentences_per_segment = max(2, len(sentences) // 4)  # 4 segments max
        
        current_speaker = segments[0]["speaker"]
        sentence_idx = 0
        segment_idx = 0
        
        while sentence_idx < len(sentences):
            # Calculer la durée proportionnelle
            proportion = sentences_per_segment / len(sentences)
            start_time = segment_idx * (segments[-1]["end"] / 4)
            end_time = min((segment_idx + 1) * (segments[-1]["end"] / 4), segments[-1]["end"])
            
            # Prendre les phrases pour ce segment
            seg_sentences = sentences[sentence_idx:sentence_idx + sentences_per_segment]
            seg_text = " ".join(seg_sentences)
            
            corrected_segments.append({
                "speaker": current_speaker,
                "start": start_time,
                "end": end_time,
                "text": seg_text,
                "mood": None
            })
            
            # Alterner les speakers
            if current_speaker == "Agent":
                current_speaker = "Client"
            else:
                current_speaker = "Agent"
                
            sentence_idx += sentences_per_segment
            segment_idx += 1
        
        log(f"[CORRECTION] Created {len(corrected_segments)} corrected segments")
        return corrected_segments
    
    return segments

def smart_sentence_split(text: str) -> List[str]:
    """
    Découpe intelligent du texte en phrases respectant le dialogue naturel.
    """
    if not text:
        return []
    
    # Marqueurs de fin de phrase étendus pour les dialogues
    sentence_endings = re.compile(r'[.!?]+\s+')
    
    # Cas spéciaux pour les dialogues téléphoniques
    # Ajouter des breaks sur certains mots de transition
    transition_words = ['oui', 'non', 'd\'accord', 'très bien', 'parfait', 'bonjour', 'au revoir', 'allô']
    
    sentences = []
    
    # Découpage basique sur la ponctuation
    basic_sentences = sentence_endings.split(text)
    
    for sentence in basic_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Si la phrase est très longue, essayer de la couper sur les transitions
        if len(sentence.split()) > 15:
            words = sentence.split()
            current_phrase = []
            
            for i, word in enumerate(words):
                current_phrase.append(word)
                
                # Couper si on trouve un mot de transition et qu'on a déjà quelques mots
                if (word.lower().rstrip('.,!?') in transition_words and 
                    len(current_phrase) > 3 and 
                    i < len(words) - 1):
                    
                    sentences.append(" ".join(current_phrase))
                    current_phrase = []
                    
            # Ajouter ce qui reste
            if current_phrase:
                sentences.append(" ".join(current_phrase))
        else:
            sentences.append(sentence)
    
    # Nettoyer et filtrer les phrases vides
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 2:
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

def post_correct_speaker_attribution(text: str, speaker: str, has_previous_speaker: bool) -> str:
    """
    Post-correction de l'attribution basée sur les patterns de dialogue.
    """
    if not text:
        return text
    
    text_lower = text.lower().strip()
    
    # Patterns qui indiquent probablement le CLIENT (celui qui appelle)
    client_patterns = [
        'bonjour madame', 'bonjour monsieur', 'je me permettais', 'je voulais', 'je voudrais',
        'ma femme', 'mon mari', 'pour ma', 'pour mon', 'est-ce que', 'pourriez-vous',
        'serait-il possible', 'j\'aimerais', 'nous avons besoin', 'dans la matinée'
    ]
    
    # Patterns qui indiquent probablement l'AGENT (professionnel qui reçoit)
    agent_patterns = [
        'cabinet', 'bonjour', 'je vais vous proposer', 'le plus tôt que je puisse',
        'par contre', 'malheureusement', 'c\'est que les', 'nous n\'avons',
        'je peux vous dire', 'alors', 'effectivement', 'bien sûr'
    ]
    
    # Corriger l'attribution si patterns évidents
    client_score = sum(1 for pattern in client_patterns if pattern in text_lower)
    agent_score = sum(1 for pattern in agent_patterns if pattern in text_lower)
    
    # Cas spéciaux de correction
    if client_score > agent_score and client_score >= 2:
        # Fort indice que c'est le client, mais attribué à l'agent
        if speaker == "Agent":
            log(f"[CORRECTION] Likely client text attributed to Agent: '{text[:30]}...'")
            return text  # Pour l'instant on garde tel quel, mais on log
    elif agent_score > client_score and agent_score >= 2:
        # Fort indice que c'est l'agent, mais attribué au client  
        if speaker == "Client":
            log(f"[CORRECTION] Likely agent text attributed to Client: '{text[:30]}...'")
            return text
    
    # Nettoyage des artefacts de découpage
    text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
    text = re.sub(r'([.!?])\s*([.!?])', r'\1', text)  # Supprimer ponctuation double
    
    return text.strip()

def detect_local_speaker_errors(segments: List[Dict[str, Any]]) -> int:
    """
    Détecte et corrige les erreurs LOCALES de speaker (segments isolés mal attribués).
    Ces erreurs locales sont souvent la SOURCE des inversions globales.

    Retourne le nombre de corrections effectuées.
    """
    if len(segments) < 3:
        return 0

    corrections = 0

    for i in range(1, len(segments) - 1):
        current = segments[i]
        prev = segments[i - 1]
        next_seg = segments[i + 1]

        # Cas suspect : segment court isolé entre 2 segments du même autre speaker
        current_speaker = current.get("speaker")
        prev_speaker = prev.get("speaker")
        next_speaker = next_seg.get("speaker")

        if prev_speaker == next_speaker and current_speaker != prev_speaker:
            duration = float(current["end"]) - float(current["start"])

            # Si le segment est court (< 3s), c'est probablement une erreur
            if duration < 3.0:
                log(f"[LOCAL_ERROR] Segment {i} ({duration:.1f}s) isolated as '{current_speaker}' between '{prev_speaker}' - CORRECTING")
                current["speaker"] = prev_speaker
                corrections += 1

    if corrections > 0:
        log(f"[LOCAL_ERROR] Corrected {corrections} local speaker errors")

    return corrections


def verify_roles_with_llm(segments: List[Dict[str, Any]]) -> bool:
    """
    Utilise Voxtral pour vérifier si les rôles Agent/Client sont corrects.
    Analyse les premiers segments pour détecter une inversion.

    Retourne True si une inversion est détectée.
    """
    if not segments:
        return False

    # Construire un texte représentatif des premiers échanges
    conversation_sample = ""
    for i, seg in enumerate(segments[:5], 1):  # 5 premiers segments
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "")
        conversation_sample += f"{i}. {speaker}: {text}\n"

    # Prompt pour Voxtral
    verification_prompt = f"""Contexte: Cabinet médical/dentaire. L'Agent est le secrétariat qui RÉPOND au téléphone. Le Client est la personne qui APPELLE.

Conversation:
{conversation_sample}

Question: Les labels Agent/Client sont-ils INVERSÉS? Réponds UNIQUEMENT par "OUI" ou "NON"."""

    try:
        conv = [{
            "role": "user",
            "content": [{"type": "text", "text": verification_prompt}]
        }]

        result = run_voxtral_with_timeout(conv, max_new_tokens=10, timeout=15)
        response = result.get("text", "").strip().upper()

        log(f"[LLM_VERIFY] Response: {response}")

        return "OUI" in response

    except Exception as e:
        log(f"[LLM_VERIFY] Error: {e}")
        return False


def detect_global_speaker_swap(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Détection MULTI-NIVEAUX des inversions Agent/Client.
    DÉSACTIVÉ PAR DÉFAUT - Activer avec DETECT_GLOBAL_SWAP=1

    STRATÉGIE À 5 NIVEAUX:
    1. Vérifier le premier locuteur (Agent répond toujours en premier dans un cabinet)
    2. Analyser le temps de parole relatif (Agent parle généralement 30-50%)
    3. Détecter et corriger les erreurs LOCALES (segments isolés mal attribués)
    4. Analyse sémantique par mots-clés
    5. Vérification par Voxtral LLM si doute

    Les erreurs locales sont souvent la SOURCE des inversions globales.
    """
    if not DETECT_GLOBAL_SWAP or not segments or len(segments) < 2:
        return segments

    log("[GLOBAL_SWAP] Multi-level inversion detection started...")

    inversion_score = 0

    # NIVEAU 1 : Vérification du premier locuteur (SIGNAL FORT)
    first_speaker = segments[0].get("speaker")
    first_text = segments[0].get("text", "").lower()

    # Dans un cabinet, l'Agent répond TOUJOURS en premier
    if first_speaker == "Client":
        log(f"[GLOBAL_SWAP] ⚠️ First speaker is Client (should be Agent for cabinet)")
        inversion_score += 50

    # Bonus si le premier dit "cabinet" ou "secrétariat" (= Agent certain)
    if first_speaker == "Agent" and any(kw in first_text for kw in ['cabinet', 'secrétariat']):
        log(f"[GLOBAL_SWAP] ✅ First speaker is Agent with cabinet greeting")
        inversion_score -= 20  # Renforce la confiance (score négatif = pas d'inversion)

    # NIVEAU 2 : Analyse du temps de parole
    agent_time = sum(float(s["end"]) - float(s["start"]) for s in segments if s.get("speaker") == "Agent")
    client_time = sum(float(s["end"]) - float(s["start"]) for s in segments if s.get("speaker") == "Client")
    total_time = agent_time + client_time

    if total_time > 0:
        agent_ratio = agent_time / total_time
        log(f"[GLOBAL_SWAP] Speaking time - Agent: {agent_ratio:.1%}, Client: {(1-agent_ratio):.1%}")

        # Dans un cabinet, l'Agent parle généralement 30-50% du temps
        # Si "Agent" parle 70%+, c'est suspect (probablement le vrai client)
        if agent_ratio > 0.70:
            log(f"[GLOBAL_SWAP] ⚠️ Agent speaks too much ({agent_ratio:.1%}) - suspicious")
            inversion_score += 25
        # Si "Agent" parle < 20%, aussi suspect (trop peu pour un secrétariat)
        elif agent_ratio < 0.20:
            log(f"[GLOBAL_SWAP] ⚠️ Agent speaks too little ({agent_ratio:.1%}) - suspicious")
            inversion_score += 15

    # NIVEAU 3 : Correction des erreurs LOCALES (segments isolés)
    # CES ERREURS CAUSENT SOUVENT LES INVERSIONS GLOBALES
    local_corrections = detect_local_speaker_errors(segments)
    if local_corrections > 0:
        log(f"[GLOBAL_SWAP] Corrected {local_corrections} local errors - recalculating...")
        # Recalculer le score après correction
        inversion_score = max(0, inversion_score - 10 * local_corrections)

    # NIVEAU 4 : Analyse sémantique par mots-clés
    agent_keywords = [
        'cabinet', 'secrétariat', 'je vais voir', 'je vous propose',
        'on vous rappelle', 'vous avez mon numéro', 'notre confrère',
        'c\'est noté', 'je vérifie', 'agenda', 'consultation'
    ]

    client_keywords = [
        'ma femme', 'mon mari', 'mon fils', 'ma fille', 'mes enfants',
        'je voudrais', 'est-ce que je peux', 'pour moi', 'j\'ai appelé',
        'rendez-vous pour moi', 'mon problème', 'j\'ai mal'
    ]

    agent_says_agent = 0
    agent_says_client = 0
    client_says_agent = 0
    client_says_client = 0

    for seg in segments:
        speaker = seg.get("speaker")
        text = (seg.get("text", "") or "").lower()

        agent_count = sum(1 for kw in agent_keywords if kw in text)
        client_count = sum(1 for kw in client_keywords if kw in text)

        if speaker == "Agent":
            agent_says_agent += agent_count
            agent_says_client += client_count
        elif speaker == "Client":
            client_says_agent += agent_count
            client_says_client += client_count

    agent_coherence = agent_says_agent - agent_says_client
    client_coherence = client_says_client - client_says_agent

    log(f"[GLOBAL_SWAP] Keyword coherence - Agent: {agent_coherence}, Client: {client_coherence}")

    # Les deux négatifs = inversion probable
    if agent_coherence < 0 and client_coherence < 0:
        log(f"[GLOBAL_SWAP] ⚠️ Both coherence scores negative - strong inversion signal")
        inversion_score += 40

    # Asymétrie forte
    if (client_says_agent >= 2 and agent_says_agent == 0) or \
       (agent_says_client >= 2 and client_says_client == 0):
        log(f"[GLOBAL_SWAP] ⚠️ Strong asymmetry detected")
        inversion_score += 30

    # NIVEAU 5 : Vérification par Voxtral LLM si doute
    if 40 <= inversion_score < 80:
        log(f"[GLOBAL_SWAP] Score in doubt zone ({inversion_score}) - asking LLM for verification")
        llm_confirms_swap = verify_roles_with_llm(segments[:5])
        if llm_confirms_swap:
            log(f"[GLOBAL_SWAP] ⚠️ LLM confirms inversion")
            inversion_score += 40
        else:
            log(f"[GLOBAL_SWAP] ✅ LLM says roles are correct")
            inversion_score -= 20

    # DÉCISION FINALE
    log(f"[GLOBAL_SWAP] Final inversion score: {inversion_score}")

    if inversion_score >= 70:
        log("[GLOBAL_SWAP] ⚠️ GLOBAL INVERSION DETECTED - Swapping all Agent ↔ Client labels")
        for seg in segments:
            if seg["speaker"] == "Agent":
                seg["speaker"] = "Client"
            elif seg["speaker"] == "Client":
                seg["speaker"] = "Agent"
    else:
        log("[GLOBAL_SWAP] ✅ No global inversion detected")

    return segments

def detect_and_fix_speaker_inversion(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Correction MINIMALE et CONSERVATRICE des erreurs de diarisation.

    PHILOSOPHIE:
    - La reconnaissance vocale PyAnnote est LA VÉRITÉ DE BASE
    - On corrige UNIQUEMENT les segments suspects isolés
    - PAS de swap global - seulement correction de segments individuels

    Cas corrigés:
    1. Segments courts (<2s) isolés entre 2 segments du même autre speaker
    2. Segments avec forte incohérence sémantique évidente
    3. Inversion globale Agent/Client (si DETECT_GLOBAL_SWAP=1)
    """
    if not segments or len(segments) < 3:
        return segments

    # ÉTAPE 1 : Détection globale (si activée)
    segments = detect_global_speaker_swap(segments)

    # ÉTAPE 2 : Corrections locales
    log("[SPEAKER_FIX] Checking for isolated segment errors...")

    corrections_made = 0

    # Phrases TRÈS caractéristiques qui indiquent FORTEMENT un rôle
    agent_strong_phrases = [
        "cabinet", "secrétariat", "je prends note", "je vous mets en ligne",
        "c'est noté", "je vérifie dans l'agenda", "vous êtes madame", "votre nom"
    ]

    client_strong_phrases = [
        "ma femme", "mon mari", "mon fils", "ma fille", "mes enfants",
        "rendez-vous pour moi", "j'appelle pour prendre"
    ]

    for i in range(1, len(segments) - 1):
        current = segments[i]
        prev = segments[i - 1]
        next_seg = segments[i + 1]

        current_speaker = current.get("speaker")
        prev_speaker = prev.get("speaker")
        next_speaker = next_seg.get("speaker")
        text = (current.get("text", "") or "").lower()

        # CAS 1: Segment court isolé entre 2 segments du même autre speaker
        duration = current.get("end", 0) - current.get("start", 0)
        is_isolated = (prev_speaker == next_speaker and prev_speaker != current_speaker)
        is_very_short = duration < 1.5  # Très court = probablement erreur
        has_few_words = len(text.split()) < 4

        if is_isolated and (is_very_short or has_few_words):
            # Vérifier conflit sémantique FORT
            has_strong_semantic_conflict = False

            current_says_agent_phrase = any(phrase in text for phrase in agent_strong_phrases)
            current_says_client_phrase = any(phrase in text for phrase in client_strong_phrases)

            if current_says_agent_phrase or current_says_client_phrase:
                # Vérifier si le speaker environnant dit le contraire
                prev_text = (prev.get("text", "") or "").lower()
                next_text = (next_seg.get("text", "") or "").lower()
                surrounding_text = prev_text + " " + next_text

                prev_says_agent = any(phrase in surrounding_text for phrase in agent_strong_phrases)
                prev_says_client = any(phrase in surrounding_text for phrase in client_strong_phrases)

                # Conflit = current dit "ma femme" mais prev/next dit "cabinet"
                if (current_says_client_phrase and prev_says_agent) or \
                   (current_says_agent_phrase and prev_says_client):
                    has_strong_semantic_conflict = True

            # Correction si segment très court isolé OU conflit sémantique fort
            if is_very_short or has_strong_semantic_conflict:
                log(f"[SPEAKER_FIX] Correcting isolated segment [{i}]: '{text[:40]}' ({duration:.1f}s)")
                log(f"[SPEAKER_FIX]   Reason: {'semantic_conflict' if has_strong_semantic_conflict else 'very_short_isolated'}")
                log(f"[SPEAKER_FIX]   Changing {current_speaker} → {prev_speaker}")
                current["speaker"] = prev_speaker
                corrections_made += 1

    if corrections_made > 0:
        log(f"[SPEAKER_FIX] ✅ Corrected {corrections_made} isolated segment(s)")
    else:
        log("[SPEAKER_FIX] ✅ No suspicious segments, keeping PyAnnote attribution")

    return segments


def improve_diarization_quality(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Améliore la qualité de la diarization en post-traitement.
    """
    if not segments:
        return segments

    improved_segments = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        speaker = seg["speaker"]

        # Si le segment est très court et répète le speaker précédent/suivant,
        # il pourrait s'agir d'une erreur de segmentation
        if len(text.split()) < 3:
            # Regarder le contexte
            prev_speaker = segments[i-1]["speaker"] if i > 0 else None
            next_speaker = segments[i+1]["speaker"] if i < len(segments)-1 else None

            # Si entouré du même speaker, c'est probablement une erreur de segmentation
            if prev_speaker == next_speaker and prev_speaker != speaker:
                log(f"[QUALITY] Potential segmentation error: short '{text}' between {prev_speaker} segments")

        improved_segments.append(seg)

    return improved_segments
def diarize_then_transcribe_hybrid(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    """
    MODE HYBRIDE ULTRA-RAPIDE : 
    1. Transcription globale complète (1 seul appel Voxtral)
    2. Diarization pour les timestamps des speakers
    3. Attribution du texte selon les timestamps
    """
    log(f"[HYBRID] Starting ultra-fast hybrid mode: language={language}, summary={with_summary}")
    
    # Durée max
    try:
        log("[HYBRID] Checking audio duration...")
        import soundfile as sf
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        log(f"[HYBRID] Audio duration: {est_dur:.1f}s")
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s). Increase MAX_DURATION_S or send shorter file."}
    except Exception as e:
        log(f"[HYBRID] Could not check duration: {e}")

    # ÉTAPE 1: TRANSCRIPTION GLOBALE (1 SEUL APPEL VOXTRAL)
    log("[HYBRID] Step 1: Global transcription...")
    conv_global = _build_conv_transcribe_ultra_strict(wav_path, language or "fr")
    global_tokens = min(max_new_tokens, int(est_dur * 8))  # ~8 tokens par seconde max
    out_global = run_voxtral_with_timeout(conv_global, max_new_tokens=global_tokens, timeout=60)
    full_text = (out_global.get("text") or "").strip()
    
    if not full_text:
        log("[HYBRID] Empty global transcription, fallback to segment mode")
        return diarize_then_transcribe_fallback(wav_path, language, max_new_tokens, with_summary)
    
    log(f"[HYBRID] Global transcription: {len(full_text)} chars in {out_global.get('latency_s', 0):.1f}s")

    # ÉTAPE 2: DIARIZATION POUR LES TIMESTAMPS
    log("[HYBRID] Step 2: Diarization for speaker timestamps...")
    dia = load_diarizer()
    try:
        if EXACT_TWO:
            diarization = dia(wav_path, num_speakers=2, min_speakers=2, max_speakers=2)
        else:
            diarization = dia(wav_path, min_speakers=1, max_speakers=MAX_SPEAKERS)
    except (TypeError, ValueError):
        if EXACT_TWO:
            diarization = dia(wav_path, num_speakers=2)
        else:
            diarization = dia(wav_path, num_speakers=MAX_SPEAKERS)
    
    # Collecter les segments de diarization
    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end)
        }
        diar_segments.append(segment)
    
    log(f"[HYBRID] Diarization found {len(diar_segments)} raw segments")

    # ÉTAPE 3: OPTIMISATION DES SEGMENTS
    optimized_segments = optimize_diarization_segments(diar_segments)
    log(f"[HYBRID] After optimization: {len(optimized_segments)} segments")

    # ÉTAPE 4: ATTRIBUTION INTELLIGENTE DU TEXTE SELON LES TIMESTAMPS
    log("[HYBRID] Step 3: Smart text attribution to speakers...")
    segments = []
    
    if not optimized_segments:
        segments = [{
            "speaker": "Agent",
            "start": 0.0,
            "end": total_duration,
            "text": full_text,
            "mood": None
        }]
    else:
        # Nouvelle approche : attribution par phrases complètes
        sentences = smart_sentence_split(full_text)
        log(f"[HYBRID] Split into {len(sentences)} sentences for attribution")
        
        total_seg_duration = sum(seg["end"] - seg["start"] for seg in optimized_segments)
        sentence_index = 0
        
        for i, seg in enumerate(optimized_segments):
            seg_duration = seg["end"] - seg["start"]
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Calculer combien de phrases attribuer à ce segment
            sentence_proportion = seg_duration / total_seg_duration if total_seg_duration > 0 else 1.0/len(optimized_segments)
            sentences_for_segment = max(1, int(len(sentences) * sentence_proportion))
            
            # Ajuster pour ne pas dépasser
            sentences_for_segment = min(sentences_for_segment, len(sentences) - sentence_index)
            
            # Récupérer les phrases pour ce segment
            if sentences_for_segment > 0:
                seg_sentences = sentences[sentence_index:sentence_index + sentences_for_segment]
                seg_text = " ".join(seg_sentences).strip()
                sentence_index += sentences_for_segment
            else:
                seg_text = ""
            
            # Post-correction basée sur les patterns de dialogue
            seg_text = post_correct_speaker_attribution(seg_text, seg["speaker"], i > 0)
            
            segments.append({
                "speaker": seg["speaker"],
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "mood": None
            })
            
            log(f"[HYBRID] Segment {i+1}/{len(optimized_segments)}: '{seg_text[:50]}...' → {seg['speaker']}")
        
        # Attribution des phrases restantes au dernier segment si nécessaire
        if sentence_index < len(sentences):
            remaining_sentences = sentences[sentence_index:]
            if segments and remaining_sentences:
                additional_text = " ".join(remaining_sentences)
                segments[-1]["text"] += " " + additional_text
                log(f"[HYBRID] Added {len(remaining_sentences)} remaining sentences to last segment")
    
    log("[HYBRID] Smart text attribution completed")
    
    # ÉTAPE 5: POST-TRAITEMENT ET CORRECTION INTELLIGENTE
    log("[HYBRID] Step 4: Intelligent dialogue correction...")
    segments = intelligent_dialogue_correction(full_text, segments)
    segments = improve_diarization_quality(segments)
    segments = _enforce_max_two_speakers(segments)
    segments = detect_and_fix_speaker_inversion(segments)  # AVANT _map_roles
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    
    result = {"segments": segments, "transcript": full_transcript}
    
    # Résumé
    if with_summary:
        log("[HYBRID] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript)
    
    # Sentiment par speaker (analyse améliorée)
    if ENABLE_SENTIMENT:
        log("[HYBRID] Computing sentiment analysis by speaker...")

        # Analyse le sentiment pour chaque speaker séparément
        speaker_sentiments = analyze_sentiment_by_speaker(segments)

        if speaker_sentiments:
            # Attribuer le sentiment spécifique à chaque segment
            for seg in segments:
                speaker = seg.get("speaker")
                if speaker and speaker in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[speaker]

            # Mood overall (moyenne pondérée de tous les speakers)
            weighted_moods = []
            for seg in segments:
                if seg.get("text") and seg.get("mood"):
                    duration = float(seg["end"]) - float(seg["start"])
                    weighted_moods.append((seg["mood"], duration))
            result["mood_overall"] = aggregate_mood(weighted_moods) if weighted_moods else None

            # Mood par speaker (déjà calculé)
            result["mood_by_speaker"] = speaker_sentiments

            # NOUVEAU: Sentiment du client identifié
            client_mood = get_client_sentiment(speaker_sentiments, segments)
            result["mood_client"] = client_mood

            # Protection maximale contre None
            confidence_raw = client_mood.get('confidence')
            if confidence_raw is None:
                confidence = 0.0
                log(f"[HYBRID] WARNING: Client sentiment confidence is None, using 0.0")
            else:
                confidence = float(confidence_raw)

            label_fr = client_mood.get('label_fr') or 'inconnu'
            log(f"[HYBRID] Client sentiment: {label_fr} (confidence: {confidence:.2f})")

        log("[HYBRID] Sentiment analysis completed")
    else:
        log("[HYBRID] Sentiment analysis skipped")

    log(f"[HYBRID] Completed: 1 Voxtral call instead of {len(diar_segments)}+ calls")
    return result

def diarize_then_transcribe_fallback(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    """Mode fallback segment-par-segment (ultra-optimisé) si le mode hybride échoue."""
    log("[FALLBACK] Using segment-by-segment mode with ultra-aggressive filtering")
    
    # Diarization
    dia = load_diarizer()
    try:
        if EXACT_TWO:
            diarization = dia(wav_path, num_speakers=2)
        else:
            diarization = dia(wav_path, min_speakers=1, max_speakers=MAX_SPEAKERS)
    except (TypeError, ValueError):
        diarization = dia(wav_path, num_speakers=MAX_SPEAKERS)
    
    # Collecter et optimiser segments
    raw_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segment = {
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end)
        }
        raw_segments.append(segment)
    
    # ULTRA-AGRESSIF : garde seulement les 4-6 plus longs segments
    raw_segments.sort(key=lambda x: x["end"] - x["start"], reverse=True)
    max_segments = min(6, len(raw_segments))
    optimized_segments = raw_segments[:max_segments]
    optimized_segments.sort(key=lambda x: x["start"])  # Retrier par temps
    
    log(f"[FALLBACK] Ultra-aggressive: {len(raw_segments)} → {max_segments} segments")
    
    # Transcription segment par segment
    audio = AudioSegment.from_wav(wav_path)
    segments = []
    
    for i, seg in enumerate(optimized_segments):
        start_s = seg["start"]
        end_s = seg["end"]
        dur_s = end_s - start_s
        speaker = seg["speaker"]
        
        # Extraction audio
        seg_audio = audio[int(start_s * 1000): int(end_s * 1000)]
        tmp = os.path.join(tempfile.gettempdir(), f"seg_{speaker}_{int(start_s*1000)}.wav")
        seg_audio.export(tmp, format="wav")
        
        # Transcription
        conv = _build_conv_transcribe_ultra_strict(tmp, "fr")
        tokens = max(32, min(96, int(dur_s * 8)))
        out = run_voxtral_with_timeout(conv, max_new_tokens=tokens, timeout=30)
        text = (out.get("text") or "").strip()
        
        segments.append({
            "speaker": speaker,
            "start": start_s,
            "end": end_s,
            "text": text,
            "mood": classify_sentiment(text) if ENABLE_SENTIMENT and text else None
        })
        
        try:
            os.remove(tmp)
        except Exception:
            pass
    
    # Post-traitement
    segments = _enforce_max_two_speakers(segments)
    segments = detect_and_fix_speaker_inversion(segments)  # AVANT _map_roles
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript}
    
    if with_summary:
        result["summary"] = select_best_summary_approach(full_transcript)
    
    return result

# ---------------------------
# Health
# ---------------------------
def health():
    info = {
        "app_version": APP_VERSION,
        "python": os.popen("python -V").read().strip(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": _device_str(),
        "transformers_has_voxtral": _HAS_VOXTRAL_CLASS,
        "hf_token_present": bool(HF_TOKEN),
        "model_id": MODEL_ID,
        "diar_model": DIAR_MODEL,
        "sentiment_analysis": "Voxtral-based" if ENABLE_SENTIMENT else "Disabled",
        "max_speakers": MAX_SPEAKERS,
        "exact_two": EXACT_TWO,
        "strict_transcription": STRICT_TRANSCRIPTION,
        "smart_summary": SMART_SUMMARY,
        "extractive_summary": EXTRACTIVE_SUMMARY,
        "min_seg_dur": MIN_SEG_DUR,
        "min_speaker_time": MIN_SPEAKER_TIME,
        "merge_consecutive": MERGE_CONSECUTIVE,
        "role_labels": ROLE_LABELS,
        "hybrid_mode": HYBRID_MODE,
        "voxtral_speaker_id": VOXTRAL_SPEAKER_ID,  # NOUVEAU
        "pyannote_auto": PYANNOTE_AUTO,  # NOUVEAU
        "aggressive_merge": AGGRESSIVE_MERGE,
        "single_voice_detection": ENABLE_SINGLE_VOICE_DETECTION,  # NOUVEAU
        "single_voice_detection_timeout": SINGLE_VOICE_DETECTION_TIMEOUT,  # NOUVEAU
        "single_voice_summary_tokens": SINGLE_VOICE_SUMMARY_TOKENS,  # NOUVEAU
        "diarization_modes": {  # NOUVEAU : résumé des modes disponibles
            "voxtral_speaker_id": VOXTRAL_SPEAKER_ID,
            "pyannote_auto": PYANNOTE_AUTO, 
            "hybrid_mode": HYBRID_MODE,
            "fallback_mode": not (VOXTRAL_SPEAKER_ID or PYANNOTE_AUTO or HYBRID_MODE),
            "single_voice_detection": ENABLE_SINGLE_VOICE_DETECTION
        },
        "optimizations": {
            "current_mode": "Voxtral Speaker ID" if VOXTRAL_SPEAKER_ID else "PyAnnote Auto" if PYANNOTE_AUTO else "Hybrid" if HYBRID_MODE else "Fallback",
            "speaker_identification": "Context-based by Voxtral" if VOXTRAL_SPEAKER_ID else "Voice-based by PyAnnote",
            "expected_quality": "High (contextual)" if VOXTRAL_SPEAKER_ID else "Medium-High (automatic)" if PYANNOTE_AUTO else "Medium (hybrid)",
            "expected_speed": "Fast (1 Voxtral call)" if VOXTRAL_SPEAKER_ID else "Fast (minimal processing)" if PYANNOTE_AUTO else "Fast (1 Voxtral + PyAnnote)",
        }
    }
    
    try:
        if _model is not None:
            p = next(_model.parameters())
            info["voxtral_device"] = str(p.device)
            info["voxtral_dtype"] = str(p.dtype)
        else:
            info["voxtral_device"] = None
            info["voxtral_dtype"] = None
    except Exception:
        info["voxtral_device"] = "unknown"
        info["voxtral_dtype"] = "unknown"

    errors = {}
    try:
        load_voxtral()
    except Exception as e:
        errors["voxtral_load"] = f"{type(e).__name__}: {e}"
    try:
        load_diarizer()
    except Exception as e:
        errors["diarizer_load"] = f"{type(e).__name__}: {e}"
    
    return {"ok": len(errors) == 0, "info": info, "errors": errors}

# ---------------------------
# Handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        log(f"[HANDLER] New job received: {job.get('input', {}).get('task', 'unknown')}")
        inp = job.get("input", {}) or {}

        if inp.get("ping"):
            log("[HANDLER] Ping request")
            return {"pong": True}

        if inp.get("task") == "health":
            log("[HANDLER] Health check request")
            return health()

        task = (inp.get("task") or "transcribe").lower()
        language = inp.get("language") or None
        max_new_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))
        with_summary = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))

        log(f"[HANDLER] Task: {task}, language: {language}, max_tokens: {max_new_tokens}, summary: {with_summary}")

        local_path, cleanup = None, False
        if inp.get("audio_url"):
            local_path = _download_to_tmp(inp["audio_url"]); cleanup = True
            log(f"[HANDLER] Downloaded audio from URL: {inp['audio_url']}")
        elif inp.get("audio_b64"):
            local_path = _b64_to_tmp(inp["audio_b64"]); cleanup = True
            log("[HANDLER] Decoded audio from base64")
        elif inp.get("file_path"):
            local_path = inp["file_path"]
            log(f"[HANDLER] Using file path: {local_path}")
        elif task not in ("health",):
            log("[HANDLER] ERROR: No audio input provided")
            return {"error": "Provide 'audio_url', 'audio_b64' or 'file_path'."}

        log(f"[HANDLER] Processing audio file: {local_path}")

        if task in ("transcribe_diarized", "diarized", "diarize"):
            log("[HANDLER] Starting diarized transcription...")

            # NOUVEAU : Détection précoce du type de contenu (si activée)
            if ENABLE_SINGLE_VOICE_DETECTION:
                content_type = detect_single_voice_content(local_path, language)
                
                if content_type["type"] == "announcement":
                    log("[HANDLER] Detected announcement/IVR, using single voice mode")
                    out = transcribe_single_voice_content(local_path, language, max_new_tokens, with_summary)
                elif content_type["type"] == "voicemail":
                    log("[HANDLER] Detected voicemail, using single voice mode")
                    out = transcribe_single_voice_content(local_path, language, max_new_tokens, with_summary)
                else:
                    # Mode normal pour les conversations (PIPELINE EXISTANT INCHANGÉ)
                    if VOXTRAL_SPEAKER_ID:
                        log("[HANDLER] Using Voxtral Speaker Identification mode")
                        out = diarize_with_voxtral_speaker_id(local_path, language, max_new_tokens, with_summary)
                    elif PYANNOTE_AUTO:
                        log("[HANDLER] Using PyAnnote Auto mode")
                        out = diarize_with_pyannote_auto(local_path, language, max_new_tokens, with_summary)
                    elif HYBRID_MODE:
                        log("[HANDLER] Using Hybrid mode (PyAnnote + Voxtral)")
                        out = diarize_then_transcribe_hybrid(local_path, language, max_new_tokens, with_summary)
                    else:
                        log("[HANDLER] Using Fallback segment mode")
                        out = diarize_then_transcribe_fallback(local_path, language, max_new_tokens, with_summary)
            else:
                # Détection désactivée - utiliser le pipeline existant (COMPORTEMENT ORIGINAL)
                if VOXTRAL_SPEAKER_ID:
                    log("[HANDLER] Using Voxtral Speaker Identification mode")
                    out = diarize_with_voxtral_speaker_id(local_path, language, max_new_tokens, with_summary)
                elif PYANNOTE_AUTO:
                    log("[HANDLER] Using PyAnnote Auto mode")
                    out = diarize_with_pyannote_auto(local_path, language, max_new_tokens, with_summary)
                elif HYBRID_MODE:
                    log("[HANDLER] Using Hybrid mode (PyAnnote + Voxtral)")
                    out = diarize_then_transcribe_hybrid(local_path, language, max_new_tokens, with_summary)
                else:
                    log("[HANDLER] Using Fallback segment mode")
                    out = diarize_then_transcribe_fallback(local_path, language, max_new_tokens, with_summary)
                
            if "error" in out:
                log(f"[HANDLER] Error in diarized transcription: {out['error']}")
                return out
            log("[HANDLER] Diarized transcription completed successfully")
            return {"task": "transcribe_diarized", **out}
        elif task in ("summary", "summarize"):
            log("[HANDLER] Starting summary task...")
            try:
                temp_result = diarize_then_transcribe_hybrid(local_path, language, max_new_tokens, False)
                if "error" in temp_result:
                    return temp_result
                
                transcript = temp_result.get("transcript", "")
                summary = select_best_summary_approach(transcript) if transcript else "Audio indisponible."
                
                log("[HANDLER] Summary task completed")
                return {"task": "summary", "text": summary, "latency_s": 0}
            except Exception as e:
                log(f"[ERROR] Smart summary failed: {e}")
                # Fallback mode résumé audio direct
                instruction = "Résume cette conversation en 1-2 phrases simples et utiles."
                conv = [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": local_path},
                        {"type": "text", "text": instruction}
                    ]
                }]
                out = run_voxtral_with_timeout(conv, max_new_tokens=72, timeout=25)
                log("[HANDLER] Fallback summary completed")
                return {"task": "summary", **out}
        else:
            log("[HANDLER] Starting simple transcription...")
            conv = _build_conv_transcribe_ultra_strict(local_path, language)
            out = run_voxtral_with_timeout(conv, max_new_tokens=min(max_new_tokens, 64), timeout=30)
            log("[HANDLER] Simple transcription completed")
            return {"task": "transcribe", **out}

    except Exception as e:
        log(f"[HANDLER] CRITICAL ERROR: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        try:
            if 'cleanup' in locals() and cleanup and local_path and os.path.exists(local_path):
                os.remove(local_path)
                log("[HANDLER] Cleanup completed")
        except Exception:
            pass

# Preload conditionnel et sûr
try:
    log("[INIT] Starting conditional preload...")
    
    if not check_gpu_memory():
        log("[CRITICAL] Insufficient GPU memory for Voxtral Small - consider using Mini")
        log("[INIT] Skipping preload due to memory issues")
    else:
        log("[INIT] Preloading Voxtral...")
        load_voxtral()

        # Lazy loading PyAnnote : charger seulement si modes qui l'utilisent
        if VOXTRAL_SPEAKER_ID and not (PYANNOTE_AUTO or HYBRID_MODE):
            log("[INIT] Skipping PyAnnote preload (Voxtral Speaker ID mode only)")
        else:
            log("[INIT] Preloading diarizer...")
            load_diarizer()

        log("[INIT] Preload completed successfully")
        
except Exception as e:
    log(f"[WARN] Preload failed - will load on first request: {e}")
    _processor = None
    _model = None
    _diarizer = None
    _sentiment_clf = None
    _sentiment_zero_shot = None

runpod.serverless.start({"handler": handler})
