#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, base64, tempfile, uuid, requests, json, traceback, re
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline,
    pipeline as hf_pipeline
)

# Voxtral class (transformers @ main / nightly)
try:
    from transformers import VoxtralForConditionalGeneration  # type: ignore
    _HAS_VOXTRAL_CLASS = True
except Exception:
    VoxtralForConditionalGeneration = None  # type: ignore
    _HAS_VOXTRAL_CLASS = False

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
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "900"))      
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Sentiment - NOUVEAU : Analyse par Voxtral ou modèle dédié
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"
SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment").strip()
SENTIMENT_TYPE = os.environ.get("SENTIMENT_TYPE", "classifier").strip()  # "classifier" ou "zero-shot"
SENTIMENT_DEVICE = int(os.environ.get("SENTIMENT_DEVICE", "-1"))  # -1 => CPU

# Diarization - MODES AVANCÉS
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "2"))
EXACT_TWO = os.environ.get("EXACT_TWO", "1") == "1"   
MIN_SEG_DUR = float(os.environ.get("MIN_SEG_DUR", "5.0"))         
MIN_SPEAKER_TIME = float(os.environ.get("MIN_SPEAKER_TIME", "8.0")) 
MERGE_CONSECUTIVE = os.environ.get("MERGE_CONSECUTIVE", "1") == "1"  
HYBRID_MODE = os.environ.get("HYBRID_MODE", "1") == "1"
AGGRESSIVE_MERGE = os.environ.get("AGGRESSIVE_MERGE", "1") == "1"
VOXTRAL_SPEAKER_ID = os.environ.get("VOXTRAL_SPEAKER_ID", "1") == "0"  # NOUVEAU : Speaker ID par Voxtral
PYANNOTE_AUTO = os.environ.get("PYANNOTE_AUTO", "0") == "1"  # NOUVEAU : PyAnnote auto pur

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
        proc_kwargs = {}
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

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
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
        "inadmissible", "inacceptable", "catastrophe", "scandaleux"
    ]
    
    # MOTS MOYENNEMENT NÉGATIFS (besoin d'accumulation)
    mild_negative = [
        "pas content", "pas satisfait", "déçu", "problème", 
        "difficile", "compliqué", "pas possible"
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
    
    # 2. Beaucoup de positifs = positif (SEUIL ABAISSÉ)
    if positive_count >= 2 and mild_neg_count == 0:
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
            # Override si trop de positifs
            if positive_count >= 2:
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
        "c'est inadmissible"
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
    
def classify_sentiment_with_model(text: str) -> Dict[str, Any]:
    """
    Analyse de sentiment avec modèle dédié (plus rapide et précis pour le français)
    """
    if not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    try:
        clf = load_sentiment()
        if clf is None:
            return classify_sentiment_with_voxtral(text)

        result = clf(text)
        if SENTIMENT_TYPE == "zero-shot":
            # Zero-shot retourne une liste de scores
            scores = {item["label"]: item["score"] for item in result[0]}
        else:
            # Classifier retourne scores pour toutes les classes
            scores = result[0]

        # Normaliser les labels
        label_mapping = {
            "1 star": "negative", "2 stars": "negative", "3 stars": "neutral",
            "4 stars": "positive", "5 stars": "positive",
            "NEGATIVE": "negative", "NEUTRAL": "neutral", "POSITIVE": "positive"
        }

        normalized_scores = {}
        for label, score in scores.items():
            norm_label = label_mapping.get(label.upper(), label.lower())
            normalized_scores[norm_label] = normalized_scores.get(norm_label, 0) + score

        best_label = max(normalized_scores.items(), key=lambda x: x[1])
        return {
            "label_en": best_label[0],
            "label_fr": _label_fr(best_label[0]),
            "confidence": best_label[1],
            "scores": normalized_scores
        }
    except Exception as e:
        log(f"[SENTIMENT_MODEL] Error: {e}, falling back to Voxtral")
        return classify_sentiment_with_voxtral(text)

def classify_sentiment(text: str) -> Dict[str, Any]:
    """
    Interface de sentiment - utilise modèle dédié ou Voxtral avec validation
    """
    if not ENABLE_SENTIMENT or not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    # Essayer d'abord le modèle dédié
    sentiment = classify_sentiment_with_model(text)

    # Validation et correction si nécessaire
    sentiment = validate_sentiment_coherence(text, sentiment)

    return sentiment

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
        f"{lang_prefix}Résume cette conversation téléphonique professionnelle en 1-2 phrases naturelles et concises. "
        "Indique : qui appelle (client ou autre), pourquoi (motif de l'appel), et l'issue ou la prochaine action. "
        "Sois précis et utilise un langage simple, comme si tu racontais brièvement ce qui s'est passé."
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
    
    if total_words < 20:
        return "Conversation très brève."
    
    if total_words > 30 and len(lines) > 3:
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

    # Résoudre les chevauchements
    filtered = resolve_overlaps(filtered)
    log(f"[OPTIMIZE] After overlap resolution: {len(filtered)} segments")

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
    
    # Ajuster les tokens selon la durée
    speaker_tokens = min(max_new_tokens, int(est_dur * 10))  # Un peu plus pour les labels
    out_speaker_id = run_voxtral_with_timeout(conv_speaker_id, max_new_tokens=speaker_tokens, timeout=90)
    speaker_transcript = (out_speaker_id.get("text") or "").strip()
    
    if not speaker_transcript:
        log("[VOXTRAL_ID] Empty speaker identification result, falling back to hybrid mode")
        return diarize_then_transcribe_hybrid(wav_path, language, max_new_tokens, with_summary)
    
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
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript}
    
    # Résumé
    if with_summary:
        log("[VOXTRAL_ID] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript)
    
    # Sentiment global rapide
    if ENABLE_SENTIMENT:
        log("[VOXTRAL_ID] Computing sentiment analysis...")
        global_mood = classify_sentiment(full_transcript) if full_transcript else None
        
        if global_mood and global_mood.get("scores"):
            # Attribution du sentiment global à tous les segments
            for seg in segments:
                if seg.get("text"):
                    seg["mood"] = global_mood
            
            # Mood overall et par speaker
            weighted_moods = [(global_mood, float(seg["end"]) - float(seg["start"])) for seg in segments if seg.get("text")]
            result["mood_overall"] = aggregate_mood(weighted_moods)
            
            per_role = {}
            for seg in segments:
                d = float(seg["end"]) - float(seg["start"])
                r = seg["speaker"]
                if seg.get("text"):
                    per_role.setdefault(r, []).append((global_mood, d))
            result["mood_by_speaker"] = {r: aggregate_mood(lst) for r, lst in per_role.items()}

    log("[VOXTRAL_ID] Voxtral speaker identification completed successfully")
    return result

def parse_speaker_identified_transcript(transcript: str, total_duration: float) -> List[Dict[str, Any]]:
    """
    Parse le transcript avec speakers identifiés par Voxtral
    """
    if not transcript:
        return []
    
    segments = []
    lines = transcript.split('\n')
    current_time = 0.0
    
    log(f"[PARSE] Processing {len(lines)} lines from speaker transcript")
    
    valid_lines = []
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
                
                # Normaliser les noms de speakers
                if any(word in speaker_part.lower() for word in ['agent', 'professionnel', 'cabinet', 'secrétaire']):
                    speaker = "Agent"
                elif any(word in speaker_part.lower() for word in ['client', 'appelant', 'patient']):
                    speaker = "Client"
                elif speaker_part.lower() in ['agent', 'client']:
                    speaker = speaker_part.capitalize()
                else:
                    # Speaker non identifié, essayer de deviner par le contenu
                    if any(word in text_part.lower() for word in ['cabinet', 'bonjour', 'je vais voir', 'on vous rappelle']):
                        speaker = "Agent"
                    elif any(word in text_part.lower() for word in ['je voudrais', 'ma femme', 'est-ce que je peux']):
                        speaker = "Client"
                    else:
                        speaker = "Agent"  # Défaut
                
                if text_part:  # Seulement si il y a du texte
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
    
    return segments

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

    # DIARIZATION AUTOMATIQUE (SANS PARAMÈTRES)
    log("[PYANNOTE_AUTO] Running automatic diarization...")
    dia = load_diarizer()
    diarization = dia(wav_path)  # Laisse PyAnnote décider complètement
    
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
    
    # Appliquer le mode hybride avec les segments PyAnnote auto
    return apply_hybrid_workflow_with_segments(wav_path, raw_segments, language, max_new_tokens, with_summary)

def resolve_overlaps(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Résoudre les chevauchements de segments pour éviter les inversions de locuteurs.
    Garde le segment avec la durée la plus longue en cas de chevauchement.
    """
    if not segments:
        return segments

    # Trier par temps de début
    segments = sorted(segments, key=lambda s: s["start"])
    resolved = []

    for seg in segments:
        if not resolved:
            resolved.append(seg)
            continue

        last = resolved[-1]
        if seg["start"] < last["end"]:  # Chevauchement
            # Calculer le degré de chevauchement
            overlap_start = max(last["start"], seg["start"])
            overlap_end = min(last["end"], seg["end"])
            overlap_duration = max(0, overlap_end - overlap_start)

            last_dur = last["end"] - last["start"]
            seg_dur = seg["end"] - seg["start"]

            # Si chevauchement mineur (< 50% de la durée du segment le plus court), ajuster les limites
            if overlap_duration < 0.5 * min(last_dur, seg_dur):
                # Étendre le dernier segment pour inclure le nouveau
                last["end"] = max(last["end"], seg["end"])
                log(f"[OVERLAP] Minor overlap resolved by extending segment")
            else:
                # Chevauchement significatif : garder le segment le plus long
                if seg_dur > last_dur:
                    resolved[-1] = seg
                    log(f"[OVERLAP] Major overlap: replaced with longer segment")
                else:
                    log(f"[OVERLAP] Major overlap: kept existing longer segment")
        else:
            resolved.append(seg)

    return resolved

def ultra_aggressive_merge(segments: List[Dict[str, Any]], max_gap: float = 3.0) -> List[Dict[str, Any]]:
    """
    Fusion ultra-agressive pour créer de gros blocs cohérents et réduire les erreurs.
    Évite la fusion de chevauchements entre locuteurs différents.
    """
    if not segments:
        return segments

    merged = []
    current = segments[0].copy()

    log(f"[ULTRA_MERGE] Starting with {len(segments)} segments")

    for next_seg in segments[1:]:
        same_speaker = current["speaker"] == next_seg["speaker"]
        gap = float(next_seg["start"]) - float(current["end"])

        # Fusion seulement si même speaker et gap positif ou très petit
        should_merge = same_speaker and gap <= max_gap

        if should_merge:
            # Fusionner
            current["end"] = next_seg["end"]

            current_text = current.get("text", "").strip()
            next_text = next_seg.get("text", "").strip()
            if current_text and next_text:
                current["text"] = f"{current_text} {next_text}"
            elif next_text:
                current["text"] = next_text
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    log(f"[ULTRA_MERGE] Result: {len(segments)} → {len(merged)} segments")
    return merged
def apply_hybrid_workflow_with_segments(wav_path: str, diar_segments: List[Dict], language: Optional[str], max_new_tokens: int, with_summary: bool):
    """Applique le workflow hybride avec des segments de diarization fournis"""
    # Transcription globale
    conv_global = _build_conv_transcribe_ultra_strict(wav_path, language or "fr")
    out_global = run_voxtral_with_timeout(conv_global, max_new_tokens=max_new_tokens, timeout=60)
    full_text = (out_global.get("text") or "").strip()
    
    if not full_text:
        return {"error": "Empty global transcription"}
    
    # Optimisation des segments
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
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript}
    
    if with_summary:
        result["summary"] = select_best_summary_approach(full_transcript)
    
    if ENABLE_SENTIMENT:
        global_mood = classify_sentiment(full_transcript) if full_transcript else None
        if global_mood:
            for seg in segments:
                if seg.get("text"):
                    seg["mood"] = global_mood
            
            weighted_moods = [(global_mood, float(seg["end"]) - float(seg["start"])) for seg in segments if seg.get("text")]
            result["mood_overall"] = aggregate_mood(weighted_moods)
            
            per_role = {}
            for seg in segments:
                d = float(seg["end"]) - float(seg["start"])
                r = seg["speaker"]
                if seg.get("text"):
                    per_role.setdefault(r, []).append((global_mood, d))
            result["mood_by_speaker"] = {r: aggregate_mood(lst) for r, lst in per_role.items()}
    
    return result

# ---------------------------
# MODE HYBRIDE ULTRA-RAPIDE (fonction principale existante)
# ---------------------------
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
        should_merge = (same_speaker and gap <= max_gap) or (gap <= 0.5)  # Fusionne même différents speakers si gap < 0.5s
        
        if should_merge:
            # Fusionner
            current["end"] = next_seg["end"]
            
            # Si même speaker, fusionner normalement
            if same_speaker:
                current_text = current.get("text", "").strip()
                next_text = next_seg.get("text", "").strip()
                if current_text and next_text:
                    current["text"] = f"{current_text} {next_text}"
                elif next_text:
                    current["text"] = next_text
            else:
                # Différents speakers mais gap tiny = probablement erreur PyAnnote
                log(f"[ULTRA_MERGE] Merging different speakers due to tiny gap: {gap:.2f}s")
                # Garde le speaker du segment le plus long
                current_dur = float(current["end"]) - float(current["start"])  
                next_dur = float(next_seg["end"]) - float(next_seg["start"])
                if next_dur > current_dur:
                    current["speaker"] = next_seg["speaker"]
                    
        else:
            # Ne peut pas fusionner
            merged.append(current)
            current = next_seg.copy()
    
    merged.append(current)
    
    log(f"[ULTRA_MERGE] Result: {len(segments)} → {len(merged)} segments ({100*(1-len(merged)/len(segments)):.1f}% reduction)")
    return merged

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

def post_correct_speaker_attribution(text: str, speaker: str, has_previous_speaker: bool) -> str:
    """Post-correction améliorée de l'attribution basée sur les patterns de dialogue."""
    if not text:
        return text
    
    text_lower = text.lower().strip()
    
    # Patterns CLIENT plus spécifiques
    client_patterns = [
        ('je voudrais', 3), ('ma femme', 3), ('mon mari', 3), ('j\'ai appelé', 2),
        ('est-ce que je pourrais', 3), ('où est-ce que', 2), ('quel autre centre', 3),
        ('les rendez-vous ne conviennent pas', 4), ('je me permettais', 3)
    ]
    
    # Patterns AGENT plus spécifiques  
    agent_patterns = [
        ('cabinet', 2), ('bonjour', 1), ('je vais voir ça', 3), ('dès que j\'ai', 2),
        ('on vous rappelle', 3), ('vous avez mon numéro', 3), ('notre confrère', 2),
        ('c\'est pour quoi faire', 2), ('allez ok', 2)
    ]
    
    # Calcul des scores pondérés
    client_score = sum(weight for pattern, weight in client_patterns if pattern in text_lower)
    agent_score = sum(weight for pattern, weight in agent_patterns if pattern in text_lower)
    
    # Détection d'erreurs flagrantes
    if client_score >= 3 and speaker == "Agent":
        log(f"[CORRECTION] Strong client pattern in agent text: '{text[:40]}...'")
    elif agent_score >= 3 and speaker == "Client":
        log(f"[CORRECTION] Strong agent pattern in client text: '{text[:40]}...'")
    
    return re.sub(r'\s+', ' ', text).strip()

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
    _map_roles(segments)
    
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    
    result = {"segments": segments, "transcript": full_transcript}
    
    # Résumé
    if with_summary:
        log("[HYBRID] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript)
    
    # Sentiment global (optimisé pour la vitesse en mode hybride)
    if ENABLE_SENTIMENT:
        log("[HYBRID] Computing sentiment analysis...")
        # Calcul du sentiment global sur le texte complet (plus rapide qu'individual)
        global_mood = classify_sentiment(full_text) if full_text else None
        
        # Attribution du même sentiment à tous les segments (approximation rapide)
        for seg in segments:
            if seg.get("text"):
                seg["mood"] = global_mood
        
        # Mood overall basé sur la durée des segments
        if global_mood and global_mood.get("scores"):
            weighted_moods = [(global_mood, float(seg["end"]) - float(seg["start"])) for seg in segments if seg.get("text")]
            result["mood_overall"] = aggregate_mood(weighted_moods)
            
            # Mood par speaker (approximation basée sur le global mood)  
            per_role = {}
            for seg in segments:
                d = float(seg["end"]) - float(seg["start"])
                r = seg["speaker"]
                if seg.get("text"):
                    per_role.setdefault(r, []).append((global_mood, d))
            result["mood_by_speaker"] = {r: aggregate_mood(lst) for r, lst in per_role.items()}
        
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
        "sentiment_analysis": f"Model: {SENTIMENT_MODEL} ({SENTIMENT_TYPE})" if ENABLE_SENTIMENT else "Disabled",
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
        "diarization_modes": {  # NOUVEAU : résumé des modes disponibles
            "voxtral_speaker_id": VOXTRAL_SPEAKER_ID,
            "pyannote_auto": PYANNOTE_AUTO, 
            "hybrid_mode": HYBRID_MODE,
            "fallback_mode": not (VOXTRAL_SPEAKER_ID or PYANNOTE_AUTO or HYBRID_MODE)
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
            
            # NOUVEAU : Sélection du mode de diarization
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
        
        log("[INIT] Preloading diarizer...")
        load_diarizer()

        if ENABLE_SENTIMENT:
            log("[INIT] Preloading sentiment model...")
            load_sentiment()

        log("[INIT] Preload completed successfully")
        
except Exception as e:
    log(f"[WARN] Preload failed - will load on first request: {e}")
    _processor = None
    _model = None
    _diarizer = None
    _sentiment_clf = None
    _sentiment_zero_shot = None

runpod.serverless.start({"handler": handler})
