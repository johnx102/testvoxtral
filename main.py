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
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))       
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "300"))      
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Sentiment (CPU par défaut)
SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli").strip()
SENTIMENT_TYPE = os.environ.get("SENTIMENT_TYPE", "zero-shot").strip().lower()
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"
SENTIMENT_DEVICE = int(os.environ.get("SENTIMENT_DEVICE", "-1"))  # -1 = CPU

# Diarization - MODE HYBRIDE ULTRA-RAPIDE
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "2"))
EXACT_TWO = os.environ.get("EXACT_TWO", "1") == "1"   
MIN_SEG_DUR = float(os.environ.get("MIN_SEG_DUR", "8.0"))         # ULTRA-AGRESSIF
MIN_SPEAKER_TIME = float(os.environ.get("MIN_SPEAKER_TIME", "3.0")) # ULTRA-AGRESSIF
MERGE_CONSECUTIVE = os.environ.get("MERGE_CONSECUTIVE", "1") == "1"  
HYBRID_MODE = os.environ.get("HYBRID_MODE", "1") == "1"  # NOUVEAU : transcription globale + diarization

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

def classify_sentiment(text: str) -> Dict[str, Any]:
    if not ENABLE_SENTIMENT or not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    if SENTIMENT_TYPE == "zero-shot":
        clf = load_sentiment()
        res = clf(text.strip(), candidate_labels=["positive", "neutral", "negative"], multi_label=False)
        scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for lbl, sc in zip(res["labels"], res["scores"]):
            scores[lbl.lower()] = float(sc)
        label_en = res["labels"][0].lower()
        return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": scores[label_en], "scores": scores}
    else:
        clf = load_sentiment()
        res = clf(text.strip())[0]
        canonical = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for r in res:
            lbl = r["label"].lower()
            if lbl.startswith("label_"):
                try:
                    idx = int(lbl.split("_")[-1])
                    lbl = ["negative", "neutral", "positive"][idx]
                except Exception:
                    pass
            if lbl in canonical:
                canonical[lbl] = float(r["score"])
        label_en = max(canonical.items(), key=lambda kv: kv[1])[0]
        return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": canonical[label_en], "scores": canonical}

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
    """Optimise les segments de diarization pour réduire le nombre d'appels Voxtral."""
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
    
    # Fusionner les segments consécutifs si activé
    if MERGE_CONSECUTIVE and filtered:
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

def merge_consecutive_segments(segments: List[Dict[str, Any]], max_gap: float = 1.0) -> List[Dict[str, Any]]:
    """Fusionne les segments consécutifs du même speaker pour réduire le nombre d'appels Voxtral."""
    if not segments:
        return segments
    
    merged = []
    current = segments[0].copy()
    
    log(f"[MERGE] Starting with {len(segments)} segments")
    
    for next_seg in segments[1:]:
        same_speaker = current["speaker"] == next_seg["speaker"]
        gap = float(next_seg["start"]) - float(current["end"])
        small_gap = gap <= max_gap
        
        if same_speaker and small_gap:
            current["end"] = next_seg["end"]
            current_text = current.get("text", "").strip()
            next_text = next_seg.get("text", "").strip()
            
            if current_text and next_text:
                current["text"] = f"{current_text} {next_text}"
            elif next_text:
                current["text"] = next_text
                
            log(f"[MERGE] Merged segments: {current['speaker']} gap={gap:.1f}s")
        else:
            merged.append(current)
            current = next_seg.copy()
    
    merged.append(current)
    
    log(f"[MERGE] Result: {len(segments)} → {len(merged)} segments ({100*(1-len(merged)/len(segments)):.1f}% reduction)")
    return merged

# ---------------------------
# MODE HYBRIDE ULTRA-RAPIDE
# ---------------------------
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

    # ÉTAPE 4: ATTRIBUTION DU TEXTE SELON LES TIMESTAMPS
    log("[HYBRID] Step 3: Attributing text to speakers...")
    segments = []
    words = full_text.split()
    total_duration = est_dur
    
    if not optimized_segments:
        segments = [{
            "speaker": "Agent",
            "start": 0.0,
            "end": total_duration,
            "text": full_text,
            "mood": None  # Skip sentiment in hybrid mode for speed
        }]
    else:
        total_seg_duration = sum(seg["end"] - seg["start"] for seg in optimized_segments)
        word_index = 0
        
        log(f"[HYBRID] Processing {len(optimized_segments)} segments for text attribution...")
        
        for i, seg in enumerate(optimized_segments):
            seg_duration = seg["end"] - seg["start"]
            
            # Attribution proportionnelle du texte (plus simple et rapide)
            word_proportion = seg_duration / total_seg_duration if total_seg_duration > 0 else 1.0/len(optimized_segments)
            words_for_segment = max(1, int(len(words) * word_proportion))
            
            # Assurer qu'on ne dépasse pas la fin des mots
            words_for_segment = min(words_for_segment, len(words) - word_index)
            
            if words_for_segment > 0:
                seg_words = words[word_index:word_index + words_for_segment]
                seg_text = " ".join(seg_words)
                word_index += words_for_segment
            else:
                seg_text = ""
            
            # SKIP sentiment in hybrid mode for speed - calculate at the end instead
            segments.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg_text,
                "mood": None  # Will be calculated globally at the end if needed
            })
            
            log(f"[HYBRID] Segment {i+1}/{len(optimized_segments)}: {len(seg_text)} chars assigned to {seg['speaker']}")
        
        # Attribution des mots restants au dernier segment si nécessaire
        if word_index < len(words):
            remaining_words = words[word_index:]
            if segments and remaining_words:
                segments[-1]["text"] += " " + " ".join(remaining_words)
                log(f"[HYBRID] Added {len(remaining_words)} remaining words to last segment")
    
    log("[HYBRID] Text attribution completed")
    
    # ÉTAPE 5: POST-TRAITEMENT STANDARD
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
        "sentiment_model": SENTIMENT_MODEL if ENABLE_SENTIMENT else None,
        "sentiment_type": SENTIMENT_TYPE,
        "sentiment_device": "cpu" if SENTIMENT_DEVICE == -1 else f"cuda:{SENTIMENT_DEVICE}",
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
        "optimizations": {
            "hybrid_transcription": HYBRID_MODE,
            "global_vs_segments": "1 Voxtral call instead of 10-50+",
            "segment_filtering": f"Ignore segments < {MIN_SEG_DUR}s",
            "speaker_filtering": f"Ignore speakers < {MIN_SPEAKER_TIME}s total",
            "consecutive_merging": MERGE_CONSECUTIVE,
            "expected_speedup": "10-20x faster" if HYBRID_MODE else "2-3x faster"
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
    if ENABLE_SENTIMENT:
        try:
            load_sentiment()
        except Exception as e:
            errors["sentiment_load"] = f"{type(e).__name__}: {e}"
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
            
            # Choisir le mode selon la configuration
            if HYBRID_MODE:
                out = diarize_then_transcribe_hybrid(local_path, language, max_new_tokens, with_summary)
            else:
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
            log("[INIT] Preloading sentiment...")
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
