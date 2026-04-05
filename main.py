#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Voxtral Serverless Worker — RunPod
# Pipeline : Transcription + Diarization + Résumé + Humeur
# Mode     : Stéréo canaux (gauche=Client, droite=Agent) + fallback Voxtral Speaker ID
# Quant    : bitsandbytes INT4 (~12-14 GB VRAM)
# =============================================================================

import os, time, base64, tempfile, uuid, requests, traceback, re, gc
from typing import Optional, List, Dict, Any, Tuple

import torch

print("=" * 70)
print(f"[STARTUP] PyTorch {torch.__version__}")
print(f"[STARTUP] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[STARTUP] CUDA version: {torch.version.cuda}")
    print(f"[STARTUP] cuDNN version: {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"[STARTUP] GPU {i}: {props.name} — {props.total_memory / 1e9:.1f} GB")
    try:
        t = torch.zeros(1).cuda()
        print(f"[STARTUP] CUDA test: OK ({t.device})")
        del t
    except Exception as e:
        print(f"[STARTUP] CUDA test: FAILED — {e}")
else:
    print("[STARTUP] WARNING: CUDA not available, running on CPU!")
print("=" * 70)

from transformers import AutoProcessor

try:
    from transformers import VoxtralForConditionalGeneration
    _HAS_VOXTRAL_CLASS = True
except Exception as e:
    print(f"[ERROR] VoxtralForConditionalGeneration not found: {e}")
    from transformers import AutoModel
    VoxtralForConditionalGeneration = AutoModel
    _HAS_VOXTRAL_CLASS = True

from pydub import AudioSegment
import runpod

# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print(msg, flush=True)

# ---------------------------
# Configuration
# ---------------------------
APP_VERSION          = os.environ.get("APP_VERSION", "voxtral-stereo-v4.1")
MODEL_ID             = os.environ.get("MODEL_ID", "mistralai/Voxtral-Small-24B-2507").strip()
HF_TOKEN             = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS       = int(os.environ.get("MAX_NEW_TOKENS", "664"))
MAX_DURATION_S       = int(os.environ.get("MAX_DURATION_S", "3600"))
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Diarization stéréo : utilise les canaux L/R du fichier WAV stéréo
# Mapping canal → rôle dépend de call_direction (détecté via préfixe in-/out-)
#   Entrant : ch0(left/rx) = Client, ch1(right/tx) = Agent
#   Sortant : ch0(left/rx) = Agent,  ch1(right/tx) = Client
# Fallback automatique vers Voxtral Speaker ID si audio mono
STEREO_DIARIZATION = os.environ.get("STEREO_DIARIZATION", "1") == "1"

# Whisper pour la transcription stéréo (plus fiable que Voxtral pour la transcription pure)
# Voxtral reste utilisé pour : résumé, sentiment, fallback mono (Speaker ID)
USE_WHISPER_STEREO = os.environ.get("USE_WHISPER_STEREO", "0") == "1"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE", "float16")

# Sentiment
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"

# Détection voix unique (annonces/IVR/messagerie)
ENABLE_SINGLE_VOICE_DETECTION  = os.environ.get("ENABLE_SINGLE_VOICE_DETECTION", "1") == "1"
SINGLE_VOICE_DETECTION_TIMEOUT = int(os.environ.get("SINGLE_VOICE_DETECTION_TIMEOUT", "30"))
SINGLE_VOICE_SUMMARY_TOKENS    = int(os.environ.get("SINGLE_VOICE_SUMMARY_TOKENS", "48"))

# Détection musique d'attente
ENABLE_HOLD_MUSIC_DETECTION  = os.environ.get("ENABLE_HOLD_MUSIC_DETECTION", "1") == "1"
HOLD_MUSIC_SPEECH_RATIO_HARD = float(os.environ.get("HOLD_MUSIC_SPEECH_RATIO_HARD", "0.03"))
HOLD_MUSIC_SPEECH_RATIO_SOFT = float(os.environ.get("HOLD_MUSIC_SPEECH_RATIO_SOFT", "0.08"))
HOLD_MUSIC_MIN_DURATION      = float(os.environ.get("HOLD_MUSIC_MIN_DURATION", "30.0"))

# Rôles speakers
ROLE_LABELS = [r.strip() for r in os.environ.get("ROLE_LABELS", "Agent,Client").split(",") if r.strip()]

# Relecture LLM des attributions (désactivé par défaut)
DETECT_GLOBAL_SWAP = os.environ.get("DETECT_GLOBAL_SWAP", "0") == "1"

# Quantization
QUANT_MODE = os.environ.get("QUANT_MODE", "torchao").lower()

_processor = None
_model     = None

# =============================================================================
# HELPERS
# =============================================================================
def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _normalize_audio(path: str) -> str:
    """
    Normalise l'audio pour Voxtral :
    - Resample 8kHz → 16kHz (téléphonie Asterisk standard)
    - Convertit en mono si stéréo
    - Normalise le volume (évite les audios trop faibles)
    Retourne le chemin du fichier normalisé (ou l'original si déjà OK).
    """
    try:
        audio = AudioSegment.from_file(path)
        needs_change = False

        if audio.frame_rate < 16000:
            audio = audio.set_frame_rate(16000)
            needs_change = True
        if audio.channels > 1:
            audio = audio.set_channels(1)
            needs_change = True

        if not needs_change:
            return path  # déjà bon, pas de conversion inutile

        out_path = path.rsplit(".", 1)[0] + "_16k.wav"
        audio.export(out_path, format="wav")
        log(f"[AUDIO] Normalized: {audio.frame_rate}Hz mono → {out_path}")
        # Supprimer l'original si c'est un fichier temporaire
        try:
            os.remove(path)
        except Exception:
            pass
        return out_path
    except Exception as e:
        log(f"[AUDIO] Normalization failed ({e}), using original")
        return path

def _normalize_audio_keepstereo(path: str) -> str:
    """
    Comme _normalize_audio mais CONSERVE le stéréo pour la diarization par canaux.
    Utilisé uniquement quand STEREO_DIARIZATION=1.
    """
    try:
        audio = AudioSegment.from_file(path)
        needs_change = False
        if audio.frame_rate < 16000:
            audio = audio.set_frame_rate(16000)
            needs_change = True
        if not needs_change:
            return path
        out_path = path.rsplit(".", 1)[0] + "_16k.wav"
        audio.export(out_path, format="wav")
        log(f"[AUDIO] Resampled: {audio.channels}ch {audio.frame_rate}Hz → 16kHz {out_path}")
        try:
            os.remove(path)
        except Exception:
            pass
        return out_path
    except Exception as e:
        log(f"[AUDIO] Normalization failed ({e}), using original")
        return path

def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    # Si stéréo diarization activé : garder le stéréo
    if STEREO_DIARIZATION:
        return _normalize_audio_keepstereo(path)
    return _normalize_audio(path)

def _b64_to_tmp(b64: str) -> str:
    raw  = base64.b64decode(b64)
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(raw)
    if STEREO_DIARIZATION:
        return _normalize_audio_keepstereo(path)
    return _normalize_audio(path)

def _validate_audio(path: str) -> Tuple[bool, str, float]:
    """Valide le fichier audio avant toute inférence. Retourne (ok, error_msg, duration_s)."""
    try:
        import soundfile as sf
        info = sf.info(path)
        duration = info.frames / float(info.samplerate or 1)
        if duration < 0.5:
            return False, f"Audio trop court ({duration:.2f}s)", duration
        if duration > MAX_DURATION_S:
            return False, f"Audio trop long ({duration:.1f}s > {MAX_DURATION_S}s)", duration
        return True, "", duration
    except Exception as e:
        return False, f"Format audio non reconnu: {e}", 0.0

def check_gpu_memory():
    if torch.cuda.is_available():
        total    = torch.cuda.get_device_properties(0).total_memory / 1e9
        cached   = torch.cuda.memory_reserved(0) / 1e9
        free     = total - cached
        min_vram = 12 if QUANT_MODE != "none" else 45
        log(f"[GPU] Total: {total:.1f}GB | Cached: {cached:.1f}GB | Free: {free:.1f}GB")
        if total < min_vram:
            log(f"[ERROR] GPU {total:.1f}GB < minimum {min_vram}GB")
            return False
        if free < 5:
            log(f"[WARN] Only {free:.1f}GB free — OOM risk")
        return True
    return False

def _gpu_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# =============================================================================
# CHARGEMENT VOXTRAL
# =============================================================================
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model
    log(f"[INIT] Loading Voxtral: {MODEL_ID} [QUANT_MODE={QUANT_MODE}]")
    proc_kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        proc_kwargs["token"] = HF_TOKEN
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **proc_kwargs)
    log("[INIT] Processor loaded")
    mdl_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN
    if QUANT_MODE in ("torchao", "bnb4") and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            mdl_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            log("[INIT] INT4 BnB config ready")
        except ImportError:
            mdl_kwargs["dtype"] = torch.bfloat16
    elif QUANT_MODE == "bnb8" and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            mdl_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            mdl_kwargs["dtype"] = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        mdl_kwargs["dtype"] = dtype
    _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **mdl_kwargs)
    log("[INIT] Model loaded successfully")
    try:
        p            = next(_model.parameters())
        gpu_params   = sum(1 for p in _model.parameters() if p.device.type == "cuda")
        total_params = sum(1 for p in _model.parameters())
        ratio        = gpu_params / total_params if total_params > 0 else 0
        log(f"[INIT] Voxtral device={p.device}, dtype={p.dtype}, GPU ratio={ratio:.2f}")
        if ratio < 0.95:
            log(f"[WARN] Only {ratio:.1%} of model on GPU")
    except Exception as e:
        log(f"[WARN] Could not check model device: {e}")
    log("[INIT] Voxtral ready.")
    return _processor, _model

# =============================================================================
# INFÉRENCE VOXTRAL
# =============================================================================
def _move_to_device(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
    return batch

def _input_len(inputs) -> int:
    try:
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return inputs["input_ids"].shape[1]
        if hasattr(inputs, "input_ids"):
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

    inputs = _move_to_device(inputs, _device_str())
    use_sampling = max_new_tokens > 3000
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=use_sampling,
        temperature=0.1 if use_sampling else None,
        top_p=0.95 if use_sampling else None,
    )
    t0 = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    dt = round(time.time() - t0, 3)
    inp_len = _input_len(inputs)
    decoded = processor.batch_decode(outputs[:, inp_len:], skip_special_tokens=True)
    result_text = decoded[0] if decoded else ""
    del outputs, inputs
    _gpu_clear()
    log("[VOXTRAL] GPU memory cleared")
    return {"text": result_text, "latency_s": dt}

def run_voxtral_with_timeout(conversation: List[Dict[str, Any]], max_new_tokens: int, timeout: int = 45) -> Dict[str, Any]:
    try:
        log(f"[VOXTRAL] Starting inference (max_tokens={max_new_tokens})")
        result = run_voxtral(conversation, max_new_tokens)
        log(f"[VOXTRAL] Completed in {result['latency_s']:.2f}s")
        return result
    except Exception as e:
        log(f"[ERROR] Voxtral inference failed: {e}")
        try:
            _gpu_clear()
            log("[VOXTRAL] GPU memory cleared after error")
        except Exception:
            pass
        return {"text": "", "latency_s": 0}

# =============================================================================
# SENTIMENT
# =============================================================================
def _label_fr(en_label: str) -> str:
    m   = {"negative": "mauvais", "neutral": "neutre", "positive": "bon"}
    key = en_label.lower()
    if key.startswith("label_"):
        try:
            idx = int(key.split("_")[-1])
            key = ["negative", "neutral", "positive"][idx]
        except Exception:
            pass
    return m.get(key, en_label)

def classify_sentiment_with_voxtral(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    text_lower      = text.lower()
    strong_negative = ["pas content du tout", "vraiment pas content", "très mécontent", "très déçu",
                       "annulé à cause", "inadmissible", "inacceptable", "catastrophe", "scandaleux",
                       "en colère", "énervé", "furieux", "exaspéré", "ras-le-bol", "j'en ai marre",
                       "c'est honteux", "n'importe quoi", "vous vous moquez", "c'est une blague",
                       "c'est du foutage", "c'est abusé", "vous abusez", "c'est grave"]
    mild_negative   = ["pas content", "pas satisfait", "déçu", "problème", "difficile", "compliqué",
                       "encore", "toujours pas", "ça fait longtemps", "combien de fois", "à chaque fois",
                       "ça suffit", "patienter", "retard", "toujours le même", "ça traîne", "trop long"]
    positive_indicators = ["merci", "parfait", "très bien", "super", "excellent", "c'est bon", "d'accord",
                           "ça marche", "pas de problème", "bonne journée", "confirmé", "réglé",
                           "je vous remercie", "merci beaucoup", "c'est gentil", "avec plaisir", "volontiers",
                           "au revoir", "à bientôt", "entendu", "c'est noté", "bonne fin de journée",
                           "bonne soirée", "à tout à l'heure", "je vous souhaite"]
    neutral_phrases = ["c'est bon alors", "ah d'accord", "pas de problème", "c'est réglé", "ça marche"]
    strong_neg_count = sum(1 for p in strong_negative if p in text_lower)
    mild_neg_count   = sum(1 for w in mild_negative if w in text_lower)
    positive_count   = sum(1 for w in positive_indicators if w in text_lower)
    neutral_found    = any(p in text_lower for p in neutral_phrases)
    if strong_neg_count >= 1:
        return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.90,
                "scores": {"negative": 0.90, "neutral": 0.08, "positive": 0.02}}
    if positive_count >= 2 and mild_neg_count == 0:
        return {"label_en": "positive", "label_fr": "bon", "confidence": 0.85,
                "scores": {"negative": 0.05, "neutral": 0.10, "positive": 0.85}}
    if neutral_found and mild_neg_count <= 1:
        return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.80,
                "scores": {"negative": 0.10, "neutral": 0.80, "positive": 0.10}}
    if mild_neg_count >= 3:
        return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.75,
                "scores": {"negative": 0.75, "neutral": 0.20, "positive": 0.05}}
    word_count = len(text.split())
    if word_count < 100 and positive_count >= 1 and mild_neg_count == 0:
        return {"label_en": "positive", "label_fr": "bon", "confidence": 0.70,
                "scores": {"negative": 0.10, "neutral": 0.20, "positive": 0.70}}
    instruction = ("Analyse le ressenti global de cette conversation téléphonique professionnelle.\n"
                   "Réponds par UN SEUL MOT : satisfaisant, neutre, ou insatisfaisant\n\n"
                   "GUIDE :\n"
                   "- satisfaisant : échange poli, demande traitée, conversation cordiale (cas le plus fréquent)\n"
                   "- neutre : très bref, ton froid, pas assez d'éléments pour juger\n"
                   "- insatisfaisant : plainte, mécontentement, frustration\n\n"
                   f"Conversation : {text[:1500]}")
    conv = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    try:
        result   = run_voxtral_with_timeout(conv, max_new_tokens=16, timeout=60)
        response = (result.get("text") or "").strip().lower()
        log(f"[SENTIMENT] Voxtral: '{response}' (pos:{positive_count} neg:{mild_neg_count})")
        if "insatisfaisant" in response:
            if positive_count >= 4:
                return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.65,
                        "scores": {"negative": 0.25, "neutral": 0.65, "positive": 0.10}}
            return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.75,
                    "scores": {"negative": 0.75, "neutral": 0.20, "positive": 0.05}}
        elif "satisfaisant" in response:
            return {"label_en": "positive", "label_fr": "bon", "confidence": 0.80,
                    "scores": {"negative": 0.05, "neutral": 0.15, "positive": 0.80}}
        else:
            if positive_count >= 1:
                return {"label_en": "positive", "label_fr": "bon", "confidence": 0.65,
                        "scores": {"negative": 0.10, "neutral": 0.25, "positive": 0.65}}
            return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.75,
                    "scores": {"negative": 0.15, "neutral": 0.75, "positive": 0.10}}
    except Exception as e:
        log(f"[SENTIMENT] Error: {e}")
        if word_count < 100 and positive_count > 0:
            return {"label_en": "positive", "label_fr": "bon", "confidence": 0.60,
                    "scores": {"negative": 0.10, "neutral": 0.30, "positive": 0.60}}
        return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.60,
                "scores": {"negative": 0.20, "neutral": 0.60, "positive": 0.20}}

def validate_sentiment_coherence(text: str, sentiment: Dict[str, Any]) -> Dict[str, Any]:
    if not sentiment or not text:
        return sentiment
    strong_negative_phrases = [
        "pas content", "pas contente", "annulé", "pas satisfait", "ce n'est pas normal",
        "inadmissible", "ça ne va pas du tout", "je vais voir ailleurs", "je vais me plaindre",
        "porter plainte", "résilier", "j'en ai assez", "vous êtes nuls", "incompétent"
    ]
    if any(p in text.lower() for p in strong_negative_phrases) and sentiment.get("label_fr") != "mauvais":
        log("[SENTIMENT_VALIDATION] Overriding sentiment to negative")
        return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.95,
                "scores": {"negative": 0.95, "neutral": 0.04, "positive": 0.01}}
    return sentiment

def classify_sentiment(text: str) -> Dict[str, Any]:
    if not ENABLE_SENTIMENT or not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    return validate_sentiment_coherence(text, classify_sentiment_with_voxtral(text))

def analyze_sentiment_by_speaker(segments: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not ENABLE_SENTIMENT or not segments:
        return {}
    speaker_texts: Dict[str, List[str]] = {}
    for seg in segments:
        speaker = seg.get("speaker")
        text    = seg.get("text", "").strip()
        if not speaker or not text:
            continue
        speaker_texts.setdefault(speaker, []).append(text)
    sentiments = {}
    for speaker, texts in speaker_texts.items():
        combined = " ".join(texts)
        s = classify_sentiment(combined)
        sentiments[speaker] = s
        log(f"[SENTIMENT_BY_SPEAKER] {speaker}: {s.get('label_fr')} (confidence: {s.get('confidence', 0):.2f})")
    return sentiments

def get_client_sentiment(speaker_sentiments: Dict[str, Dict[str, Any]], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Priorité 1 : label 'Client' direct. Priorité 2 : ROLE_LABELS[1]. Priorité 3 : fallback durée."""
    if not speaker_sentiments:
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    if "Client" in speaker_sentiments:
        log("[CLIENT_DETECTION] Client identifié: Client (label direct)")
        return speaker_sentiments["Client"]

    if len(ROLE_LABELS) >= 2 and ROLE_LABELS[1] in speaker_sentiments:
        label = ROLE_LABELS[1]
        log(f"[CLIENT_DETECTION] Client identifié: {label} (ROLE_LABELS[1])")
        return speaker_sentiments[label]

    # Fallback : speaker qui parle le plus (hors Agent)
    speaker_durations: Dict[str, float] = {}
    for seg in segments:
        sp = seg.get("speaker")
        if sp:
            speaker_durations[sp] = speaker_durations.get(sp, 0.0) + (seg.get("end", 0) - seg.get("start", 0))
    if speaker_durations:
        agent_label = ROLE_LABELS[0] if ROLE_LABELS else "Agent"
        candidates  = {k: v for k, v in speaker_durations.items() if k not in (agent_label, "Agent")}
        if candidates:
            client_speaker = max(candidates, key=lambda k: candidates[k])
        else:
            sorted_sp = sorted(speaker_durations, key=lambda k: speaker_durations[k], reverse=True)
            client_speaker = sorted_sp[1] if len(sorted_sp) > 1 else sorted_sp[0]
        log(f"[CLIENT_DETECTION] Client identifié par durée: {client_speaker} ({speaker_durations[client_speaker]:.1f}s)")
        return speaker_sentiments.get(client_speaker, {"label_en": None, "label_fr": None, "confidence": None, "scores": None})

    first = segments[0].get("speaker") if segments else None
    if first and first in speaker_sentiments:
        return speaker_sentiments[first]
    return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

def aggregate_mood(weighted: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    total = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    total_w = 0.0
    for s, w in weighted:
        if not s or not s.get("scores"):
            continue
        for k in total:
            total[k] += s["scores"].get(k, 0.0) * w
        total_w += w
    if total_w <= 0:
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    ssum = sum(total.values())
    norm = {k: (v / ssum if ssum > 0 else 0.0) for k, v in total.items()}
    label_en = max(norm, key=lambda k: norm[k])
    return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": norm[label_en], "scores": norm}

# =============================================================================
# RÉSUMÉ
# =============================================================================
def calculate_summary_tokens(duration_seconds: float, transcript_length: int) -> int:
    if duration_seconds <= 120:   base_tokens = 72
    elif duration_seconds <= 300: base_tokens = 96
    elif duration_seconds <= 600: base_tokens = 128
    elif duration_seconds <= 900: base_tokens = 160
    else:                         base_tokens = 200
    if transcript_length > 10000: base_tokens = max(base_tokens, 160)
    if transcript_length > 15000: base_tokens = max(base_tokens, 200)
    return min(base_tokens, 256)

def generate_natural_summary(full_transcript: str, language: Optional[str] = None, duration_seconds: float = 0) -> str:
    if not full_transcript.strip():
        return "Conversation vide."
    summary_tokens = calculate_summary_tokens(duration_seconds, len(full_transcript))
    log(f"[SUMMARY] Using {summary_tokens} tokens (duration={duration_seconds:.1f}s, transcript={len(full_transcript)} chars)")
    lang_prefix = f"lang:{language} " if language else ""
    instruction = (
        f"{lang_prefix}Résume cette conversation en 1-2 phrases simples et claires. "
        "Dis juste l'essentiel : qui appelle pourquoi, et ce qui va se passer. "
        "Sois direct et naturel, sans format particulier. Ne cite pas le texte, résume-le."
    )
    conv = [{"role": "user", "content": [{"type": "text", "text": f"{instruction}\n\nConversation:\n{full_transcript}"}]}]
    try:
        result  = run_voxtral_with_timeout(conv, max_new_tokens=summary_tokens, timeout=30)
        summary = clean_generated_summary((result.get("text") or "").strip())
        if not summary or len(summary) < 10:
            return create_extractive_summary(full_transcript)
        return summary
    except Exception as e:
        log(f"[ERROR] Summary generation failed: {e}")
        return create_extractive_summary(full_transcript)

def clean_generated_summary(summary: str) -> str:
    if not summary:
        return ""
    # Nettoyer les artefacts markdown et labels de parties
    summary = re.sub(r'#{1,3}\s*', '', summary)  # ### headers
    summary = re.sub(r'\*{1,3}', '', summary)     # **bold** / ***italic***
    summary = re.sub(r'(?i)partie\s*\d+\s*[-–—:]*\s*', '', summary)  # "Partie 1 —"
    summary = re.sub(r'(?i)r[ée]sum[ée]\s*[-–—:]*\s*', '', summary)  # "Résumé :"
    summary = re.sub(r'(?i)sentiment\s*[-–—:]*\s*', '', summary)      # "Sentiment :"
    summary = summary.strip(' \t\n-–—:')

    unwanted_starts = ["voici un résumé", "résumé de la conversation", "cette conversation",
                       "dans cette conversation", "le résumé", "il s'agit"]
    summary_lower   = summary.lower().strip()
    for start in unwanted_starts:
        if summary_lower.startswith(start):
            for sentence in re.split(r'[.!?]+', summary)[1:]:
                cleaned = sentence.strip()
                if len(cleaned) > 15 and not any(cleaned.lower().startswith(u) for u in unwanted_starts):
                    summary = cleaned; break
    for pattern in [r'décision[/:].*?étape[s]?\s*:', r'prochaine[s]?\s*étape[s]?\s*:',
                    r'action[s]?\s*à\s*prendre\s*:', r'conclusion\s*:', r'format\s*attendu\s*:']:
        summary = re.sub(pattern, '', summary, flags=re.IGNORECASE).strip()
    summary = re.sub(r'\.{2,}', '.', summary)
    summary = re.sub(r'\s+', ' ', summary).strip()
    if any(i in summary.lower() for i in ["format attendu", "décision/prochaine étape", "points suivants"]):
        return ""
    return summary

def create_extractive_summary(transcript: str) -> str:
    if not transcript:
        return "Conversation indisponible."
    lines  = [l.strip() for l in transcript.split('\n') if l.strip() and ':' in l]
    if not lines:
        return "Conversation très courte."
    parts  = []
    client_lines = [l for l in lines if l.lower().startswith('client:')]
    for line in client_lines:
        text = line.replace('Client:', '').strip().lower()
        if any(kw in text for kw in ['rendez-vous', 'rdv', 'prendre', 'avancer', 'reporter']):
            parts.append("Appel pour un rendez-vous"); break
        elif any(kw in text for kw in ['information', 'renseignement', 'savoir']):
            parts.append("Demande d'information"); break
        elif any(kw in text for kw in ['problème', 'souci', 'bug']):
            parts.append("Signalement d'un problème"); break
    agent_lines = [l for l in lines if l.lower().startswith('agent:')]
    for line in agent_lines:
        text = line.replace('Agent:', '').strip().lower()
        if any(kw in text for kw in ['je vais', 'on va', 'nous allons']):
            parts.append(f"L'agent va {line.replace('Agent:', '').strip().lower()}"); break
        elif any(kw in text for kw in ['pas possible', 'impossible', 'désolé']):
            parts.append("L'agent ne peut pas satisfaire la demande"); break
    if not parts:
        for line in lines[:2]:
            text = line.split(':', 1)[1].strip() if ':' in line else line
            if len(text) > 15 and not any(pol in text.lower() for pol in ['bonjour', 'au revoir', 'merci']):
                parts.append(text)
    return ". ".join(parts) + "." if parts else "Conversation brève sans motif identifié."

def select_best_summary_approach(transcript: str, duration_seconds: float = 0) -> str:
    lines       = transcript.split('\n')
    total_words = len(transcript.split())
    speaker_lines = [l for l in lines if ':' in l and l.strip()]
    if len(lines) == 1 and "System:" in transcript:
        content = transcript.replace("System:", "").strip()
        return f"Annonce automatique: {content[:100]}{'...' if len(content) > 100 else ''}"
    if total_words < 20:
        return "Conversation très brève."
    if total_words > 30 and len(speaker_lines) >= 3:
        g = generate_natural_summary(transcript, duration_seconds=duration_seconds)
        if g and len(g) > 15 and not any(bad in g.lower() for bad in ['format', 'structure', 'décision/']):
            return g
    return create_extractive_summary(transcript)

# =============================================================================
# UTILITAIRES SEGMENTS
# =============================================================================
def _enforce_max_two_speakers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    speakers = list({s["speaker"] for s in segments})
    if len(speakers) <= 2:
        return segments
    dur = {}
    cent = {}
    for s in segments:
        d   = float(s["end"]) - float(s["start"])
        spk = s["speaker"]
        mid = (float(s["start"]) + float(s["end"])) / 2.0
        dur[spk]  = dur.get(spk, 0.0) + d
        sm, ds    = cent.get(spk, (0.0, 0.0))
        cent[spk] = (sm + mid * d, ds + d)
    top2 = sorted(dur, key=lambda k: dur[k], reverse=True)[:2]
    centroids = {spk: (cent[spk][0] / cent[spk][1]) for spk in top2 if cent[spk][1] > 0}
    def nearest(mid):
        return min(centroids, key=lambda spk: abs(mid - centroids[spk]))
    for s in segments:
        if s["speaker"] not in top2:
            s["speaker"] = nearest((float(s["start"]) + float(s["end"])) / 2.0)
    return segments

def _map_roles(segments: List[Dict[str, Any]]):
    seen = []
    for s in segments:
        if s["speaker"] not in seen:
            seen.append(s["speaker"])
    mapping = {spk: ROLE_LABELS[i] for i, spk in enumerate(seen) if i < len(ROLE_LABELS)}
    for s in segments:
        s["speaker"] = mapping.get(s["speaker"], s["speaker"])

# =============================================================================
# DÉTECTION MUSIQUE D'ATTENTE
# =============================================================================
def detect_hold_music(audio_path: str) -> Dict[str, Any]:
    import soundfile as sf
    import numpy as np
    t0     = time.time()
    result = {"is_hold_music": False, "speech_ratio": 1.0, "duration": 0.0, "reason": "", "detection_time": 0.0}
    try:
        y, sr = sf.read(audio_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        duration = len(y) / sr
        result["duration"] = round(duration, 2)
        if duration < HOLD_MUSIC_MIN_DURATION:
            result["reason"] = f"too_short ({duration:.1f}s)"
            result["detection_time"] = round(time.time() - t0, 3)
            return result
        frame_length = int(0.030 * sr)
        hop_length   = int(0.010 * sr)
        n_frames     = 1 + (len(y) - frame_length) // hop_length
        rms = np.array([np.sqrt(np.mean(y[i*hop_length:i*hop_length+frame_length]**2)) for i in range(n_frames)], dtype=np.float32)
        peak = np.max(np.abs(y))
        if peak < 1e-6:
            result.update({"is_hold_music": True, "speech_ratio": 0.0, "reason": "silent_audio"})
            result["detection_time"] = round(time.time() - t0, 3)
            return result
        rms_norm        = rms / peak
        speech_threshold = float(np.median(rms_norm)) + 1.5 * float(np.std(rms_norm))
        speech_mask      = rms_norm > speech_threshold
        speech_ratio     = float(np.sum(speech_mask)) / len(rms_norm)
        result["speech_ratio"] = round(speech_ratio, 4)
        chunk_sec    = 5.0
        chunk_frames = max(1, int(chunk_sec * sr / hop_length))
        n_chunks     = max(1, len(rms_norm) // chunk_frames)
        chunk_is_speech = [float(np.mean(speech_mask[ci*chunk_frames:min((ci+1)*chunk_frames, len(rms_norm))])) >= 0.05
                           for ci in range(n_chunks)]
        speech_blocks = []
        in_block = False
        block_start = 0
        for ci, is_sp in enumerate(chunk_is_speech):
            if is_sp and not in_block:
                block_start = ci; in_block = True
            elif not is_sp and in_block:
                s = max(0.0, block_start * chunk_sec - 2.0)
                e = min(duration, ci * chunk_sec + 2.0)
                if e - s >= 3.0:
                    speech_blocks.append((round(s, 2), round(e, 2)))
                in_block = False
        if in_block:
            s = max(0.0, block_start * chunk_sec - 2.0)
            if duration - s >= 3.0:
                speech_blocks.append((round(s, 2), round(duration, 2)))
        if len(speech_blocks) > 1:
            merged = [list(speech_blocks[0])]
            for sb in speech_blocks[1:]:
                if sb[0] - merged[-1][1] <= 10.0:
                    merged[-1][1] = sb[1]
                else:
                    merged.append(list(sb))
            speech_blocks = [(round(s, 2), round(e, 2)) for s, e in merged]
        has_speech_start = any(s < duration * 0.30 for s, _ in speech_blocks)
        has_speech_end   = any(e > duration * 0.70 for _, e in speech_blocks)
        total_speech_sec = sum(e - s for s, e in speech_blocks)
        total_gap_sec    = duration - total_speech_sec
        log(f"[HOLD_MUSIC] Analysis: duration={duration:.1f}s, speech_ratio={speech_ratio:.4f}, "
            f"blocks={speech_blocks}, speech_sec={total_speech_sec:.1f}s, gap_sec={total_gap_sec:.1f}s")
        if has_speech_start and has_speech_end:
            result["reason"] = "conversation_with_gaps"
            if total_gap_sec > 30 and speech_blocks:
                result["speech_blocks"] = speech_blocks
            result["detection_time"] = round(time.time() - t0, 3)
            return result
        if has_speech_start or has_speech_end:
            result["reason"] = "partial_speech"
            if total_gap_sec > 15:
                result["speech_blocks"] = speech_blocks
            result["detection_time"] = round(time.time() - t0, 3)
            return result
        if speech_ratio < HOLD_MUSIC_SPEECH_RATIO_HARD:
            result.update({"is_hold_music": True, "reason": f"hard_threshold (ratio={speech_ratio:.4f})"})
            result["detection_time"] = round(time.time() - t0, 3)
            return result
        if speech_ratio < HOLD_MUSIC_SPEECH_RATIO_SOFT and duration > 60.0:
            result.update({"is_hold_music": True, "reason": f"soft_threshold (ratio={speech_ratio:.4f})"})
            result["detection_time"] = round(time.time() - t0, 3)
            return result
        if total_gap_sec > 30 and speech_blocks:
            result["reason"] = "normal_with_trimmable_gaps"
            result["speech_blocks"] = speech_blocks
        else:
            result["reason"] = "normal_speech"
        result["detection_time"] = round(time.time() - t0, 3)
        return result
    except Exception as e:
        log(f"[HOLD_MUSIC] Error: {e} — skipping")
        result["reason"] = f"error: {e}"
        result["detection_time"] = round(time.time() - t0, 3)
        return result

def build_hold_music_response(hold_music_result: Dict[str, Any], with_summary: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "segments": [], "transcript": "", "language": "fr",
        "audio_duration": hold_music_result.get("duration", 0.0),
        "hold_music_detected": True,
        "hold_music_speech_ratio": hold_music_result.get("speech_ratio", 0.0),
        "hold_music_reason": hold_music_result.get("reason", ""),
        "hold_music_detection_time": hold_music_result.get("detection_time", 0.0),
        "mood_overall": "neutral", "mood_by_speaker": {},
    }
    if with_summary:
        out["summary"] = "Aucune conversation détectée - musique d'attente uniquement."
    return out

def trim_audio_to_speech_blocks(local_path: str, hold_music_result: Dict[str, Any], cleanup: bool) -> Tuple[str, bool]:
    speech_blocks = hold_music_result.get("speech_blocks")
    if not speech_blocks:
        return local_path, cleanup
    try:
        audio        = AudioSegment.from_file(local_path)
        original_sec = len(audio) / 1000.0
        parts = [audio[int(s * 1000):int(e * 1000)] for s, e in speech_blocks]
        if not parts:
            return local_path, cleanup
        trimmed     = parts[0]
        for part in parts[1:]:
            trimmed += part
        trimmed_sec = len(trimmed) / 1000.0
        removed_sec = original_sec - trimmed_sec
        if removed_sec < 15:
            log(f"[TRIM] Only {removed_sec:.0f}s to remove, keeping original")
            return local_path, cleanup
        trimmed_path = local_path.rsplit(".", 1)[0] + "_trimmed.wav"
        trimmed.export(trimmed_path, format="wav")
        log(f"[TRIM] {original_sec:.1f}s → {trimmed_sec:.1f}s (removed {removed_sec:.0f}s of music)")
        if cleanup:
            os.remove(local_path)
        return trimmed_path, True
    except Exception as e:
        log(f"[TRIM] Failed: {e}")
        return local_path, cleanup

# =============================================================================
# DÉTECTION VOIX UNIQUE (annonces / IVR / messagerie)
# =============================================================================
def detect_single_voice_content(wav_path: str, language: Optional[str]) -> Dict[str, Any]:
    log("[SINGLE_VOICE] Detecting single voice content...")
    instruction = (
        f"lang:{language or 'fr'} "
        "Analyse ce contenu audio et détermine s'il s'agit de :\n"
        "1) Une conversation entre deux personnes → réponds 'CONVERSATION'\n"
        "2) Une annonce automatique/IVR → résume le contenu en 1-2 phrases\n"
        "3) Un message vocal → réponds 'MESSAGE_VOCAL'"
    )
    conv = [{"role": "user", "content": [
        {"type": "audio", "path": wav_path},
        {"type": "text", "text": instruction}
    ]}]
    try:
        result   = run_voxtral_with_timeout(conv, max_new_tokens=64, timeout=SINGLE_VOICE_DETECTION_TIMEOUT)
        response = (result.get("text") or "").strip().lower()
        log(f"[SINGLE_VOICE] Detection result: '{response}'")
        if "conversation" in response:
            return {"type": "conversation", "summary": None}
        elif "message_vocal" in response:
            return {"type": "voicemail", "summary": response}
        else:
            return {"type": "announcement", "summary": response}
    except Exception as e:
        log(f"[SINGLE_VOICE] Detection failed: {e}")
        return {"type": "unknown", "summary": None}

def transcribe_single_voice_content(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    log("[SINGLE_VOICE] Processing single voice content...")
    ok, err, est_dur = _validate_audio(wav_path)
    if not ok:
        return {"error": err}
    # Limiter les tokens à la durée réelle (single voice = court, pas besoin de 2500 tokens)
    sv_tokens = max(64, min(int(est_dur * 8), 512))
    sv_timeout = min(int(est_dur * 0.7) + 30, 120)
    log(f"[SINGLE_VOICE] Transcribing: {est_dur:.1f}s → tokens={sv_tokens}, timeout={sv_timeout}s")
    conv = [{"role": "user", "content": [
        {"type": "audio", "path": wav_path},
        {"type": "text", "text": f"lang:{language or 'fr'} [TRANSCRIBE] Écris exactement ce qui est dit, mot pour mot."}
    ]}]
    out       = run_voxtral_with_timeout(conv, max_new_tokens=sv_tokens, timeout=sv_timeout)
    full_text = (out.get("text") or "").strip()
    if not full_text:
        return {"error": "Empty transcription for single voice content"}

    # Nettoyage anti-hallucination
    full_text = remove_repetitive_loops(full_text, max_repetitions=3)
    full_text = re.sub(r'(\.\s*){5,}', '', full_text).strip()
    full_text = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold)\]\s*', '', full_text, flags=re.IGNORECASE).strip()
    if not full_text or len(full_text.split()) < 3:
        log("[SINGLE_VOICE] Empty after cleanup — likely silence/music")
        return {
            "segments": [], "transcript": "",
            "summary": "Aucune conversation détectée - silence ou musique d'attente.",
            "hold_music_detected": True,
            "mood_overall": {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.95,
                             "scores": {"negative": 0.0, "neutral": 0.95, "positive": 0.05}},
        }

    # ── Détection d'une vraie conversation dans le transcript ────────────────
    # Cas typique : appel sortant → répondeur → puis la personne rappelle ou décroche
    # Si le texte contient des échanges typiques d'une vraie conversation,
    # on repasse en diarization complète au lieu de garder "System"
    conversation_markers = [
        "allô ?", "allo ?", "oui, bonjour", "bonjour,", "c'est noté",
        "pas de souci", "à tout à l'heure", "au revoir", "je vous remercie",
        "d'accord", "pas de problème", "merci beaucoup"
    ]
    text_lower   = full_text.lower()
    marker_count = sum(1 for m in conversation_markers if m in text_lower)

    # Si on détecte 3+ marqueurs de vraie conversation → diarization complète
    if marker_count >= 3:
        log(f"[SINGLE_VOICE] Detected real conversation within voicemail ({marker_count} markers) → switching to full diarization")
        return None  # Signal au caller de basculer en mode diarization

    # Vraie voix unique (annonce/IVR/répondeur sans réponse)
    # On utilise "Agent" au lieu de "System" pour compatibilité interface
    segments        = [{"speaker": "Agent", "start": 0.0, "end": est_dur, "text": full_text, "mood": None}]
    full_transcript = f"Agent: {full_text}"
    result          = {"segments": segments, "transcript": full_transcript}

    if with_summary:
        summary_conv = [{"role": "user", "content": [{"type": "text", "text":
            f"lang:{language or 'fr'} Résume ce message vocal ou cette annonce en 1-2 phrases claires.\n\nContenu: {full_text}"
        }]}]
        try:
            sr = run_voxtral_with_timeout(summary_conv, max_new_tokens=SINGLE_VOICE_SUMMARY_TOKENS, timeout=20)
            result["summary"] = (sr.get("text") or "").strip()
        except Exception:
            result["summary"] = f"Message vocal: {full_text[:150]}..."

    if ENABLE_SENTIMENT:
        neutral = {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.95,
                   "scores": {"negative": 0.0, "neutral": 0.95, "positive": 0.05}}
        result["mood_overall"]    = neutral
        result["mood_by_speaker"] = {"Agent": neutral}
        result["mood_client"]     = neutral
    return result

# =============================================================================
# PIPELINE PRINCIPAL — VOXTRAL SPEAKER ID
# =============================================================================
def find_conversation_start(ch0_vad: List[Tuple[float, float]],
                             ch1_vad: List[Tuple[float, float]],
                             proximity_s: float = 5.0) -> float:
    """
    Trouve le début de la vraie conversation = premier moment où LES DEUX canaux
    ont de la parole (simultanément ou à moins de proximity_s l'un de l'autre).
    Tout ce qui est avant = IVR / musique d'attente / annonce.
    Retourne le timestamp de début de conversation (0.0 si pas de phase IVR détectée).
    """
    if not ch0_vad or not ch1_vad:
        return 0.0

    for b0 in ch0_vad:
        for b1 in ch1_vad:
            # Chevauchement direct
            if min(b0[1], b1[1]) > max(b0[0], b1[0]):
                start = min(b0[0], b1[0])
                if start > 3.0:  # au moins 3s d'IVR pour que ça vaille le coup de couper
                    log(f"[IVR_DETECT] Conversation starts at {start:.1f}s (overlap detected)")
                    return start
                return 0.0
            # Proches (un finit, l'autre commence dans les proximity_s secondes)
            gap = max(b1[0] - b0[1], b0[0] - b1[1])
            if 0 < gap < proximity_s:
                start = min(b0[0], b1[0])
                if start > 3.0:
                    log(f"[IVR_DETECT] Conversation starts at {start:.1f}s (gap={gap:.1f}s)")
                    return start
                return 0.0
    return 0.0

def remove_duplicate_segments(segments: List[Dict[str, Any]], similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Supprime les segments dont le texte est quasi-identique à un segment précédent.
    Détecte les annonces d'attente / IVR qui se répètent en boucle.
    """
    if len(segments) <= 1:
        return segments

    def text_similarity(a: str, b: str) -> float:
        """Similarité basique par mots communs."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / max(len(words_a), len(words_b))

    seen_texts = []
    cleaned = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        # Vérifier si ce texte est trop similaire à un texte déjà vu
        is_duplicate = False
        for seen in seen_texts:
            if text_similarity(text, seen) >= similarity_threshold:
                log(f"[DEDUP] Removed duplicate segment: '{text[:60]}...'")
                is_duplicate = True
                break
        if not is_duplicate:
            cleaned.append(seg)
            seen_texts.append(text)

    if len(cleaned) < len(segments):
        log(f"[DEDUP] Removed {len(segments) - len(cleaned)} duplicate segments")
    return cleaned

def remove_repetitive_loops(text: str, max_repetitions: int = 5) -> str:
    if not text or len(text) < 20:
        return text
    for pattern_len in range(3, 21):
        if len(text) < pattern_len * max_repetitions:
            continue
        pattern    = text[-pattern_len:]
        repetitions = 0
        pos         = len(text)
        while pos >= pattern_len:
            if text[pos - pattern_len:pos] == pattern:
                repetitions += 1; pos -= pattern_len
            else:
                break
        if repetitions > max_repetitions:
            truncate_pos = pos + pattern_len * 2
            log(f"[CLEANUP] Pattern loop '{pattern[:15]}...' ×{repetitions}, truncating")
            return text[:min(truncate_pos, len(text))]
    words = text.split()
    if len(words) < 10:
        return text
    cleaned = []
    rep_count = 1
    last_word = None
    for word in words:
        if word == last_word:
            rep_count += 1
            if rep_count > max_repetitions:
                break
        else:
            rep_count = 1; last_word = word
        cleaned.append(word)
    return " ".join(cleaned)

def merge_micro_segments(segments: List[Dict[str, Any]], max_duration: float = 2.0, max_gap: float = 1.0) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    merged = []
    i = 0
    while i < len(segments):
        current = segments[i].copy()
        while i + 1 < len(segments):
            nxt = segments[i + 1]
            if (current["speaker"] == nxt["speaker"] and
                (current["end"] - current["start"] < max_duration or nxt["end"] - nxt["start"] < max_duration) and
                nxt["start"] - current["end"] < max_gap):
                current["text"] = current["text"] + " " + nxt["text"]
                current["end"]  = nxt["end"]
                i += 1
            else:
                break
        merged.append(current)
        i += 1
    log(f"[MERGE_MICRO] {len(segments)} → {len(merged)} segments")
    return merged

def _attach_sentiment(segments, speaker_sentiments):
    """Attache le sentiment aux segments et retourne mood_overall, mood_by_speaker, mood_client."""
    for seg in segments:
        sp = seg.get("speaker")
        if sp and sp in speaker_sentiments:
            seg["mood"] = speaker_sentiments[sp]
    weighted     = [(seg["mood"], float(seg["end"]) - float(seg["start"]))
                    for seg in segments if seg.get("text") and seg.get("mood")]
    mood_overall = aggregate_mood(weighted) if weighted else None
    client_mood  = get_client_sentiment(speaker_sentiments, segments)
    label_fr     = client_mood.get("label_fr") or "inconnu"
    conf         = float(client_mood.get("confidence") or 0.0)
    log(f"[SENTIMENT] Client: {label_fr} (confidence: {conf:.2f})")
    return mood_overall, speaker_sentiments, client_mood

# =============================================================================
# DIARIZATION STÉRÉO PAR CANAUX
# =============================================================================
def _to_mono_tmp(wav_path: str) -> str:
    """Convertit un fichier stéréo en mono temporaire pour fallback Voxtral Speaker ID."""
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        if info.channels <= 1:
            return wav_path
        audio = AudioSegment.from_file(wav_path).set_channels(1)
        out   = os.path.join(tempfile.gettempdir(), f"mono_fb_{uuid.uuid4().hex}.wav")
        audio.export(out, format="wav")
        log(f"[MONO_FB] Converted stereo → mono for fallback: {out}")
        return out
    except Exception as e:
        log(f"[MONO_FB] Conversion failed: {e}")
        return wav_path

def _extract_mono_channel(wav_path: str, channel: int) -> str:
    """Extrait un canal d'un fichier stéréo en mono temporaire."""
    import soundfile as sf
    data, sr = sf.read(wav_path)
    if data.ndim == 1:
        return wav_path
    mono     = data[:, channel]
    out_path = os.path.join(tempfile.gettempdir(), f"ch{channel}_{uuid.uuid4().hex}.wav")
    sf.write(out_path, mono, sr)
    return out_path

def _vad_speech_blocks(wav_path: str, frame_ms: int = 20, min_speech_ms: int = 150,
                       padding_ms: int = 100) -> List[Tuple[float, float]]:
    """
    VAD légère basée sur l'énergie RMS — aucune dépendance supplémentaire.
    Retourne une liste de (start_sec, end_sec) des blocs de parole.
    """
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    frame_size    = int(sr * frame_ms / 1000)
    padding_frames = int(padding_ms / frame_ms)
    min_frames    = int(min_speech_ms / frame_ms)

    if len(data) < frame_size:
        return [(0.0, len(data) / sr)]

    n_frames = len(data) // frame_size
    rms = np.array([
        np.sqrt(np.mean(data[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ], dtype=np.float32)

    # Seuil adaptatif : médiane + 0.6 * std (robuste au bruit de fond)
    peak = np.max(rms)
    if peak < 1e-6:
        return []  # canal silencieux
    rms_norm      = rms / peak
    threshold     = float(np.percentile(rms_norm, 25)) + 0.6 * float(np.std(rms_norm))
    threshold     = min(threshold, 0.15)  # cap pour ne pas être trop restrictif
    speech_frames = rms_norm > threshold

    # Padding : étendre les blocs de parole de quelques frames
    padded = speech_frames.copy()
    for i in range(len(speech_frames)):
        if speech_frames[i]:
            lo = max(0, i - padding_frames)
            hi = min(len(speech_frames), i + padding_frames + 1)
            padded[lo:hi] = True
    speech_frames = padded

    # Extraire les blocs
    blocks    = []
    in_speech = False
    start_f   = 0
    for i, is_sp in enumerate(speech_frames):
        if is_sp and not in_speech:
            start_f  = i
            in_speech = True
        elif not is_sp and in_speech:
            if i - start_f >= min_frames:
                blocks.append((round(start_f * frame_ms / 1000, 3),
                               round(i * frame_ms / 1000, 3)))
            in_speech = False
    if in_speech and n_frames - start_f >= min_frames:
        blocks.append((round(start_f * frame_ms / 1000, 3),
                       round(n_frames * frame_ms / 1000, 3)))

    # Fusionner les blocs séparés par moins de 2s (forme des tours de parole cohérents)
    if not blocks:
        return []
    merged = [list(blocks[0])]
    for s, e in blocks[1:]:
        if s - merged[-1][1] < 2.0:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # Filtrer les micro-blocs < 300ms (crosstalk/bruit)
    merged = [(s, e) for s, e in merged if e - s >= 0.3]

    log(f"[VAD] {len(merged)} speech turns detected in {wav_path.split('/')[-1]}")
    return [(round(s, 3), round(e, 3)) for s, e in merged]

def _extract_speech_only(wav_path: str, vad_blocks: List[Tuple[float, float]]) -> Optional[str]:
    """
    Extrait UNIQUEMENT les portions de parole (blocs VAD) d'un fichier audio.
    Retourne le chemin du fichier nettoyé (sans musique/silence).
    """
    if not vad_blocks:
        return None
    try:
        audio = AudioSegment.from_file(wav_path)
        parts = [audio[int(s * 1000):int(e * 1000)] for s, e in vad_blocks]
        if not parts:
            return None
        clean = parts[0]
        for p in parts[1:]:
            # 200ms de silence entre les blocs pour que Voxtral ne colle pas les mots
            clean += AudioSegment.silent(duration=200, frame_rate=audio.frame_rate) + p
        out_path = wav_path.rsplit(".", 1)[0] + "_speech.wav"
        clean.export(out_path, format="wav")
        clean_dur = len(clean) / 1000.0
        orig_dur  = len(audio) / 1000.0
        log(f"[EXTRACT_SPEECH] {orig_dur:.1f}s → {clean_dur:.1f}s (removed {orig_dur - clean_dur:.1f}s silence/music)")
        return out_path
    except Exception as e:
        log(f"[EXTRACT_SPEECH] Failed: {e}")
        return None

def _transcribe_channel_with_vad(wav_path: str, speaker: str, language: Optional[str],
                                  est_dur: float) -> List[Dict[str, Any]]:
    """
    Transcrit un canal mono avec VAD :
    1. VAD → liste de blocs de parole (start, end) avec timestamps originaux
    2. Extraction audio nettoyé (que les blocs de parole, sans musique/silence)
    3. Transcription Voxtral sur l'audio nettoyé uniquement
    4. Distribution du texte sur les blocs VAD par proportion de durée
    """
    # ── VAD ──────────────────────────────────────────────────────────────────
    vad_blocks = _vad_speech_blocks(wav_path)
    if not vad_blocks:
        log(f"[STEREO] Channel {speaker} is silent (VAD found 0 blocks) — skipping")
        return []

    total_speech_dur = sum(e - s for s, e in vad_blocks)
    log(f"[STEREO] {speaker} VAD: {len(vad_blocks)} blocks, {total_speech_dur:.1f}s speech / {est_dur:.1f}s total")

    # ── Extraire seulement les parties parlées ───────────────────────────────
    # Voxtral ne voit QUE la parole, pas la musique d'attente
    clean_path = _extract_speech_only(wav_path, vad_blocks)
    transcribe_path = clean_path if clean_path else wav_path
    clean_cleanup = clean_path is not None

    try:
        # ── Transcription Voxtral sur l'audio nettoyé ────────────────────────
        # Tokens basés sur la durée de PAROLE, pas la durée totale
        tokens  = max(256, min(int(total_speech_dur * 10), 12000))
        timeout = min(int(total_speech_dur * 0.7) + 60, 600)
        instruction = (f"lang:{language or 'fr'} "
                       "[TRANSCRIBE] Transcris exactement ce qui est dit, mot pour mot. "
                       "Une seule personne parle. Conserve les hésitations (euh, bah, alors). "
                       "N'invente rien. N'écris JAMAIS [Musique] [Silence] [Attente].")
        conv = [{"role": "user", "content": [
            {"type": "audio", "path": transcribe_path},
            {"type": "text",  "text": instruction}
        ]}]
        out  = run_voxtral_with_timeout(conv, max_new_tokens=tokens, timeout=timeout)
        text = (out.get("text") or "").strip()

        # Protection anti-hallucination (basée sur la durée de parole, pas totale)
        if text and total_speech_dur > 0 and len(text) / total_speech_dur > 35:
            log(f"[STEREO] WARNING: {speaker} {len(text)/total_speech_dur:.1f} chars/sec > 35 → hallucination, canal ignoré")
            return []

        text = remove_repetitive_loops(text, max_repetitions=5)
        text = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold|BIP|Bip|bip|BEEP)\]\s*',
                      '', text, flags=re.IGNORECASE).strip()
        # Nettoyer les labels speaker que Voxtral pourrait ajouter
        text = re.sub(r'^(?:Agent|Client|Speaker\s*\d*)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

        if not text or len(text.split()) < 2:
            log(f"[STEREO] Channel {speaker} empty after cleanup — skipping")
            return []

        log(f"[STEREO] {speaker}: {len(text)} chars transcribed from {total_speech_dur:.1f}s speech")
    finally:
        if clean_cleanup and clean_path and os.path.exists(clean_path):
            try:
                os.remove(clean_path)
            except Exception:
                pass

    # ── Distribution du texte sur les blocs VAD originaux ────────────────────
    # Chaque bloc VAD reçoit une portion de texte proportionnelle à sa durée
    words = text.split()
    if not words:
        return []

    total_words = len(words)
    segments    = []
    word_idx    = 0

    for i, (b_start, b_end) in enumerate(vad_blocks):
        block_dur = b_end - b_start
        if i == len(vad_blocks) - 1:
            # Dernier bloc : tous les mots restants
            block_words = words[word_idx:]
        else:
            # Proportionnel à la durée du bloc
            n_words = max(1, round(total_words * block_dur / total_speech_dur))
            n_words = min(n_words, total_words - word_idx)
            block_words = words[word_idx:word_idx + n_words]
            word_idx += n_words

        if block_words:
            segments.append({
                "speaker": speaker,
                "start":   round(b_start, 3),
                "end":     round(b_end, 3),
                "text":    " ".join(block_words),
                "mood":    None,
            })

    log(f"[STEREO] {speaker}: {len(segments)} segments from {total_words} words / {len(vad_blocks)} VAD blocks")
    return segments

def _merge_stereo_segments(agent_segs: List[Dict], client_segs: List[Dict]) -> List[Dict]:
    """
    Fusionne les segments Agent et Client en les triant par timestamp réel.
    Fusionne les segments consécutifs du même speaker (même si un micro-segment
    de l'autre speaker s'intercale) pour former des tours de parole lisibles.
    """
    all_segs = sorted(agent_segs + client_segs, key=lambda s: s["start"])

    if not all_segs:
        return []

    # Passe 1 : fusionner les segments consécutifs du même speaker (gap < 3s)
    merged = [all_segs[0].copy()]
    for seg in all_segs[1:]:
        prev = merged[-1]
        if (seg["speaker"] == prev["speaker"] and
                seg["start"] - prev["end"] < 3.0):
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            prev["end"]  = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    # Passe 2 : absorber les micro-segments (< 1s) isolés entre deux segments du même speaker
    if len(merged) >= 3:
        cleaned = [merged[0]]
        i = 1
        while i < len(merged) - 1:
            curr = merged[i]
            prev = cleaned[-1]
            nxt  = merged[i + 1]
            curr_dur = curr["end"] - curr["start"]
            # Si le segment courant est très court ET les segments autour sont du même speaker
            if (curr_dur < 1.0 and prev["speaker"] == nxt["speaker"]
                    and prev["speaker"] != curr["speaker"]):
                # Absorber le micro-segment dans le speaker dominant
                prev["text"] = prev["text"].rstrip() + " " + curr["text"].lstrip()
                prev["end"]  = max(prev["end"], curr["end"])
                log(f"[STEREO] Absorbed micro-segment ({curr_dur:.1f}s {curr['speaker']}) into {prev['speaker']}")
                i += 1
                continue
            cleaned.append(curr)
            i += 1
        if i < len(merged):
            cleaned.append(merged[i])
        merged = cleaned

    # Passe 3 : re-fusionner les segments consécutifs du même speaker (après absorption)
    final = [merged[0]]
    for seg in merged[1:]:
        prev = final[-1]
        if prev["speaker"] == seg["speaker"]:
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            prev["end"]  = max(prev["end"], seg["end"])
        else:
            final.append(seg)

    log(f"[STEREO] Merged: {len(all_segs)} → {len(final)} segments (sorted by timestamp)")
    return final

def _vad_overlap(blocks: List[Tuple[float, float]], start: float, end: float) -> float:
    """Calcule la durée de chevauchement entre un segment [start, end] et des blocs VAD."""
    total = 0.0
    for bs, be in blocks:
        ov_start = max(start, bs)
        ov_end   = min(end, be)
        if ov_end > ov_start:
            total += ov_end - ov_start
    return total


def _build_turn_timeline_from_channels(ch0_vad: List[Tuple[float, float]],
                                        ch1_vad: List[Tuple[float, float]],
                                        left_role: str, right_role: str,
                                        max_turn_s: float = 25.0, merge_gap_s: float = 1.5) -> List[Dict[str, Any]]:
    """
    Construit la timeline de tours directement depuis les VAD par canal.
    Chaque bloc VAD = un événement de parole avec le bon speaker.
    Fusionne, absorbe les micro-segments, et découpe les tours longs.
    """
    # Créer les événements bruts depuis les VAD par canal
    raw = []
    for s, e in ch0_vad:
        raw.append({"speaker": left_role, "start": s, "end": e})
    for s, e in ch1_vad:
        raw.append({"speaker": right_role, "start": s, "end": e})
    raw.sort(key=lambda t: t["start"])

    if not raw:
        return []

    # Passe 1 : fusionner les segments consécutifs du même speaker (gap < merge_gap_s)
    merged = [raw[0].copy()]
    for seg in raw[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"] and seg["start"] - prev["end"] < merge_gap_s:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            merged.append(seg.copy())

    # Passe 2 : absorber les micro-segments (< 1.5s) isolés entre 2 du même speaker
    if len(merged) >= 3:
        cleaned = [merged[0]]
        i = 1
        while i < len(merged) - 1:
            curr = merged[i]
            prev = cleaned[-1]
            nxt  = merged[i + 1]
            curr_dur = curr["end"] - curr["start"]
            if curr_dur < 1.5 and prev["speaker"] == nxt["speaker"] and prev["speaker"] != curr["speaker"]:
                prev["end"] = max(prev["end"], curr["end"])
                i += 1
                continue
            cleaned.append(curr)
            i += 1
        if i < len(merged):
            cleaned.append(merged[i])
        merged = cleaned

    # Passe 3 : re-fusionner
    final = [merged[0]]
    for seg in merged[1:]:
        prev = final[-1]
        if prev["speaker"] == seg["speaker"]:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            final.append(seg)

    # Passe 4 : découper les tours > max_turn_s
    split = []
    for turn in final:
        dur = turn["end"] - turn["start"]
        if dur <= max_turn_s:
            split.append(turn)
        else:
            t = turn["start"]
            while t < turn["end"]:
                chunk_end = min(t + max_turn_s, turn["end"])
                split.append({"speaker": turn["speaker"], "start": round(t, 3), "end": round(chunk_end, 3)})
                t = chunk_end

    log(f"[TURNS] Built {len(split)} turns from {len(raw)} raw VAD blocks")
    return split


# =============================================================================
# WHISPER — Transcription rapide et fiable (optionnel, stéréo uniquement)
# =============================================================================
_whisper_model = None

def _load_whisper():
    """Charge le modèle FasterWhisper (lazy loading, 1 seule fois)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        log(f"[WHISPER] Loading model: {WHISPER_MODEL_SIZE} (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        log("[WHISPER] Model loaded successfully")
        return _whisper_model
    except ImportError:
        log("[WHISPER] ERROR: faster-whisper not installed. Run: pip install faster-whisper")
        return None
    except Exception as e:
        log(f"[WHISPER] ERROR loading model: {e}")
        return None


def _transcribe_turn_whisper(audio_path: str, language: str = "fr") -> str:
    """
    Transcrit un segment audio avec FasterWhisper.
    Plus rapide et fiable que Voxtral pour la transcription pure.
    Retourne le texte brut (pas de speaker labels).
    """
    model = _load_whisper()
    if model is None:
        return ""
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language or "fr",
            beam_size=5,
            vad_filter=False,  # Pas de VAD Whisper — notre pipeline gère déjà le découpage
        )
        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)
        result = " ".join(texts).strip()

        # Nettoyage : tags [Musique], [Silence], etc.
        result = re.sub(r'\[.*?\]', '', result).strip()

        # Filtre hallucinations connues de Whisper (sous-titres, crédits vidéo)
        # Ces phrases apparaissent quand Whisper reçoit du silence ou audio très faible
        # Filtre hallucinations Whisper : uniquement les phrases EXACTES connues
        # (pas de mots isolés qui risquent de matcher du vrai contenu)
        WHISPER_HALLUCINATIONS = [
            "sous-titrage st'",
            "sous-titres réalisés par",
            "sous titres réalisés par",
            "sous-titrage société",
            "amara.org",
            "merci d'avoir regardé",
            "merci d'avoir écouté cette vidéo",
            "abonnez-vous à la chaîne",
            "n'oubliez pas de vous abonner",
        ]
        result_lower = result.lower()
        # Ne filtrer que si le texte est COURT (< 30 mots) et matche une hallucination
        # Les longs textes avec un faux positif ne doivent pas être supprimés
        if len(result.split()) < 30 and any(h in result_lower for h in WHISPER_HALLUCINATIONS):
            log(f"[WHISPER] Hallucination filtered: '{result[:60]}'")
            return ""

        return result
    except Exception as e:
        log(f"[WHISPER] Transcription error: {e}")
        return ""


def _build_turns_by_energy(wav_path: str, left_role: str, right_role: str,
                            window_ms: int = 500, merge_gap_s: float = 2.5) -> List[Dict[str, Any]]:
    """
    Attribution speaker par comparaison d'énergie entre canaux stéréo.

    1. VAD sur mono → trouve TOUTE la parole (y compris voix douce)
    2. Détection début conversation (après IVR/attente)
    3. Fenêtres de window_ms ms : RMS ch0 vs ch1 → canal dominant = speaker
    4. Fusion fenêtres consécutives même speaker (gap < merge_gap_s)
    5. Absorption des micro-segments isolés (< 0.8s entre 2 du même speaker)
    6. Filtrage segments < 0.5s (bruit/artefact)
    7. Découpe tours > 25s

    Avantage vs VAD indépendant : le crosstalk ne trompe plus l'attribution
    car le vrai speaker est TOUJOURS plus fort sur son canal.
    """
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim == 1:
        return []

    ch0 = data[:, 0].copy()  # left = remote
    ch1 = data[:, 1].copy()  # right = local
    duration = len(ch0) / sr

    # --- Normalisation des canaux si déséquilibrés ---
    # Si un canal est beaucoup plus faible (micro faible, ligne basse),
    # on normalise les deux au même RMS moyen pour que la comparaison
    # d'énergie par fenêtre fonctionne correctement.
    rms_ch0_global = float(np.sqrt(np.mean(ch0 ** 2)))
    rms_ch1_global = float(np.sqrt(np.mean(ch1 ** 2)))
    if rms_ch0_global > 0 and rms_ch1_global > 0:
        ratio_global = max(rms_ch0_global, rms_ch1_global) / min(rms_ch0_global, rms_ch1_global)
        if ratio_global > 2.0:
            # Normaliser le canal faible pour matcher le canal fort
            target_rms = max(rms_ch0_global, rms_ch1_global)
            if rms_ch0_global < rms_ch1_global:
                ch0 = ch0 * (target_rms / rms_ch0_global)
                log(f"[STEREO_ENERGY] Channel imbalance detected (ratio={ratio_global:.1f}x) — normalized ch0 (was {rms_ch0_global:.6f}, now {target_rms:.6f})")
            else:
                ch1 = ch1 * (target_rms / rms_ch1_global)
                log(f"[STEREO_ENERGY] Channel imbalance detected (ratio={ratio_global:.1f}x) — normalized ch1 (was {rms_ch1_global:.6f}, now {target_rms:.6f})")

    # --- VAD sur mono pour trouver TOUTE la parole ---
    mono = (ch0 + ch1) / 2.0
    mono_tmp = os.path.join(tempfile.gettempdir(), f"mono_energy_{uuid.uuid4().hex}.wav")
    sf.write(mono_tmp, mono, sr)
    vad_blocks = _vad_speech_blocks(mono_tmp)

    # --- VAD par canal pour détecter début conversation (IVR) ---
    ch0_tmp = os.path.join(tempfile.gettempdir(), f"ch0_energy_{uuid.uuid4().hex}.wav")
    ch1_tmp = os.path.join(tempfile.gettempdir(), f"ch1_energy_{uuid.uuid4().hex}.wav")
    sf.write(ch0_tmp, ch0, sr)
    sf.write(ch1_tmp, ch1, sr)
    ch0_vad = _vad_speech_blocks(ch0_tmp)
    ch1_vad = _vad_speech_blocks(ch1_tmp)

    for p in [mono_tmp, ch0_tmp, ch1_tmp]:
        try:
            os.remove(p)
        except Exception:
            pass

    if not vad_blocks:
        return []

    # --- Détecter début de conversation (après IVR/musique) ---
    conv_start = find_conversation_start(ch0_vad, ch1_vad)
    if conv_start > 0:
        vad_blocks = [(max(s, conv_start), e) for s, e in vad_blocks if e > conv_start]
        vad_blocks = [(s, e) for s, e in vad_blocks if e > s]
        log(f"[STEREO_ENERGY] Conversation starts at {conv_start:.1f}s — trimmed IVR")

    # --- Fenêtrage et comparaison d'énergie ch0 vs ch1 ---
    window_size = int(sr * window_ms / 1000)
    raw_windows = []

    for block_start, block_end in vad_blocks:
        s_frame = int(block_start * sr)
        e_frame = min(int(block_end * sr), len(ch0))

        pos = s_frame
        while pos + window_size // 2 <= e_frame:
            end_pos = min(pos + window_size, e_frame)
            win_ch0 = ch0[pos:end_pos]
            win_ch1 = ch1[pos:end_pos]

            rms0 = float(np.sqrt(np.mean(win_ch0 ** 2))) if len(win_ch0) > 0 else 0.0
            rms1 = float(np.sqrt(np.mean(win_ch1 ** 2))) if len(win_ch1) > 0 else 0.0

            # Canal avec plus d'énergie = speaker actif (crosstalk est toujours plus faible)
            speaker = left_role if rms0 > rms1 else right_role

            raw_windows.append({
                "speaker": speaker,
                "start": round(pos / sr, 3),
                "end": round(end_pos / sr, 3),
            })
            pos = end_pos

    if not raw_windows:
        return []

    # --- Passe 1 : Fusion fenêtres consécutives même speaker ---
    merged = [raw_windows[0].copy()]
    for win in raw_windows[1:]:
        prev = merged[-1]
        if win["speaker"] == prev["speaker"] and win["start"] - prev["end"] < merge_gap_s:
            prev["end"] = win["end"]
        else:
            merged.append(win.copy())

    # --- Passe 2 : Absorber les micro-segments isolés (< 0.8s) entre 2 du même speaker ---
    # Évite la fragmentation due au crosstalk ponctuel
    if len(merged) >= 3:
        cleaned = [merged[0]]
        i = 1
        while i < len(merged) - 1:
            curr = merged[i]
            prev = cleaned[-1]
            nxt  = merged[i + 1]
            curr_dur = curr["end"] - curr["start"]
            if (curr_dur < 0.8 and prev["speaker"] == nxt["speaker"]
                    and prev["speaker"] != curr["speaker"]):
                # Absorber dans le speaker dominant
                prev["end"] = max(prev["end"], curr["end"])
                i += 1
                continue
            cleaned.append(curr)
            i += 1
        if i < len(merged):
            cleaned.append(merged[i])
        merged = cleaned

    # --- Passe 3 : Re-fusionner après absorption ---
    if len(merged) >= 2:
        refused = [merged[0]]
        for seg in merged[1:]:
            prev = refused[-1]
            if prev["speaker"] == seg["speaker"]:
                prev["end"] = seg["end"]
            else:
                refused.append(seg)
        merged = refused

    # --- Filtrer segments < 0.5s (bruit/artefact — pas assez pour Voxtral) ---
    merged = [t for t in merged if t["end"] - t["start"] >= 0.5]

    # --- Passe 4 : Supprimer les tours dans les gaps du VAD mono ---
    # Le VAD mono détecte TOUTE la parole (les 2 speakers mixés).
    # Les zones SANS parole mono = silence pur ou musique d'attente → supprimer les tours.
    # C'est la méthode la plus fiable : pas de ratio d'énergie, pas de faux positifs.
    hold_regions = []
    if len(vad_blocks) >= 2:
        # Les gaps entre les blocs VAD mono = zones sans parole
        for j in range(len(vad_blocks) - 1):
            gap_start = vad_blocks[j][1]
            gap_end = vad_blocks[j + 1][0]
            gap_dur = gap_end - gap_start
            if gap_dur >= 10.0:  # gap de 10s+ = probable musique/silence
                hold_regions.append((gap_start, gap_end))
    # Gap au début (avant le premier bloc VAD)
    if vad_blocks and vad_blocks[0][0] > 5.0:
        hold_regions.insert(0, (0.0, vad_blocks[0][0]))

    # Supprimer les tours qui tombent dans les hold regions
    if hold_regions:
        total_hold = sum(e - s for s, e in hold_regions)
        log(f"[STEREO_ENERGY] Hold music detected: {len(hold_regions)} region(s), {total_hold:.0f}s total")
        for hs, he in hold_regions:
            log(f"[STEREO_ENERGY]   Hold: {hs:.1f}s - {he:.1f}s ({he - hs:.0f}s)")

        def is_in_hold(turn_start, turn_end):
            """Vérifie si un tour est majoritairement dans une zone de hold."""
            turn_dur = turn_end - turn_start
            if turn_dur <= 0:
                return False
            overlap = 0.0
            for hs, he in hold_regions:
                ov_start = max(turn_start, hs)
                ov_end = min(turn_end, he)
                if ov_end > ov_start:
                    overlap += ov_end - ov_start
            return overlap / turn_dur > 0.5  # >50% dans le hold → supprimer

        before_count = len(merged)
        merged = [t for t in merged if not is_in_hold(t["start"], t["end"])]
        removed = before_count - len(merged)
        if removed > 0:
            log(f"[STEREO_ENERGY] Removed {removed} turns in hold regions")

    # --- Découper tours > 25s ---
    final = []
    for turn in merged:
        dur = turn["end"] - turn["start"]
        if dur <= 25.0:
            final.append(turn)
        else:
            t = turn["start"]
            while t < turn["end"]:
                chunk_end = min(t + 25.0, turn["end"])
                final.append({"speaker": turn["speaker"], "start": round(t, 3), "end": round(chunk_end, 3)})
                t = chunk_end

    log(f"[STEREO_ENERGY] Built {len(final)} turns from {len(raw_windows)} windows ({len(vad_blocks)} VAD blocks)")
    return final


def _extract_turn_audio_mono(mono_path: str, start: float, end: float) -> Optional[str]:
    """Extrait un segment audio depuis le fichier MONO (pas depuis un canal isolé)."""
    try:
        import soundfile as sf
        data, sr = sf.read(mono_path, dtype="float32")
        if data.ndim > 1:
            import numpy as np
            data = np.mean(data, axis=1)
        start_frame = int(start * sr)
        end_frame   = min(int(end * sr), len(data))
        segment     = data[start_frame:end_frame]
        if len(segment) < sr * 0.3:
            return None
        out_path = os.path.join(tempfile.gettempdir(), f"turn_{uuid.uuid4().hex}.wav")
        sf.write(out_path, segment, sr)
        return out_path
    except Exception as e:
        log(f"[EXTRACT_TURN] Failed: {e}")
        return None


def _extract_turn_audio_channel(wav_path: str, start: float, end: float, channel: int) -> Optional[str]:
    """Extrait un segment audio d'un canal spécifique d'un fichier stéréo."""
    try:
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            ch_data = data
        else:
            ch_data = data[:, channel]
        s_frame = int(start * sr)
        e_frame = min(int(end * sr), len(ch_data))
        segment = ch_data[s_frame:e_frame]
        if len(segment) < sr * 0.3:
            return None
        out_path = os.path.join(tempfile.gettempdir(), f"turn_ch{channel}_{uuid.uuid4().hex}.wav")
        sf.write(out_path, segment, sr)
        return out_path
    except Exception as e:
        log(f"[EXTRACT_TURN_CH] Failed: {e}")
        return None

def _transcribe_turn(turn_audio_path: str, language: Optional[str], duration: float, from_channel: bool = False) -> str:
    """Transcrit un segment audio court (un tour de parole). Retourne le texte brut."""
    tokens  = max(64, min(int(duration * 10), 2000))
    timeout = min(int(duration * 0.8) + 30, 120)
    if from_channel:
        # Transcription depuis un canal isolé : focus sur la voix principale
        instruction = (f"lang:{language or 'fr'} "
                       "[TRANSCRIBE] Transcris UNIQUEMENT la voix principale (la plus forte). "
                       "Ignore les voix en arrière-plan ou les échos. "
                       "Ignore les annonces d'attente, musique d'attente et messages IVR automatiques. "
                       "Conserve les hésitations (euh, bah, alors). "
                       "Si tu n'entends que du silence ou de la musique, retourne une chaîne vide. "
                       "N'invente rien. N'écris JAMAIS [Musique] [Silence] [Attente].")
    else:
        instruction = (f"lang:{language or 'fr'} "
                       "[TRANSCRIBE] Transcris exactement ce qui est dit, mot pour mot. "
                       "Ignore les annonces d'attente, musique d'attente et messages IVR automatiques. "
                       "Conserve les hésitations (euh, bah, alors). "
                       "Si tu n'entends que du silence ou de la musique, retourne une chaîne vide. "
                       "N'invente rien. N'écris JAMAIS [Musique] [Silence] [Attente].")
    conv = [{"role": "user", "content": [
        {"type": "audio", "path": turn_audio_path},
        {"type": "text",  "text": instruction}
    ]}]
    out  = run_voxtral_with_timeout(conv, max_new_tokens=tokens, timeout=timeout)
    text = (out.get("text") or "").strip()
    text = remove_repetitive_loops(text, max_repetitions=3)
    text = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold|BIP|Bip|BEEP)\]\s*',
                  '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^(?:Agent|Client|Speaker\s*\d*)\s*:\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'(\.\s*){4,}', '', text).strip()
    text = re.sub(r'\*{1,3}', '', text).strip()  # nettoyer le markdown bold **
    return text


def _generate_summary_and_sentiment(transcript: str, language: Optional[str],
                                     duration: float) -> Dict[str, Any]:
    """
    1 seul appel Voxtral pour résumé + sentiment combinés.
    Retourne {"summary": ..., "mood_label": ..., "mood_confidence": ...}
    """
    if not transcript or len(transcript.split()) < 10:
        return {"summary": "Conversation très brève.", "mood_label": "neutre", "mood_confidence": 0.8}

    summary_tokens = calculate_summary_tokens(duration, len(transcript))
    # Cap total : summary + sentiment = summary_tokens + 30
    total_tokens = summary_tokens + 30

    prompt = (
        f"lang:{language or 'fr'} "
        "Analyse cette conversation téléphonique et réponds en DEUX parties séparées par ---\n\n"
        "PARTIE 1 — RÉSUMÉ :\n"
        "Résume en 1-2 phrases simples : qui appelle, pourquoi, et ce qui est décidé.\n\n"
        "PARTIE 2 — SENTIMENT :\n"
        "Écris UN SEUL MOT pour le ressenti global de la conversation :\n"
        "- bon : échange poli et cordial, demande traitée, conversation normale\n"
        "- neutre : échange très bref, ton froid ou distant, pas assez d'éléments\n"
        "- mauvais : mécontentement, plainte, frustration, problème non résolu\n"
        "Note : la plupart des appels professionnels polis sont 'bon', pas 'neutre'.\n\n"
        "---\n\n"
        f"Conversation :\n{transcript[:3000]}\n\n"
        "Réponds maintenant (résumé puis --- puis sentiment) :"
    )

    conv = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    try:
        out = run_voxtral_with_timeout(conv, max_new_tokens=total_tokens, timeout=30)
        response = (out.get("text") or "").strip()

        # Parser la réponse — nettoyer les labels que Voxtral peut inclure
        parts = response.split("---")
        raw_summary = parts[0].strip() if parts else ""
        # Supprimer les labels "Partie 1", "RÉSUMÉ", "**...**" etc.
        raw_summary = re.sub(r'\*{1,2}(?:Partie\s*\d+\s*[-–—:]*\s*)?R[ée]sum[ée]\s*[-–—:]*\s*\*{0,2}\s*:?\s*', '', raw_summary, flags=re.IGNORECASE).strip()
        raw_summary = re.sub(r'^(?:Partie\s*\d+\s*[-–—:]*\s*)?R[ée]sum[ée]\s*[-–—:]*\s*', '', raw_summary, flags=re.IGNORECASE).strip()
        summary = clean_generated_summary(raw_summary)
        mood_label = "neutre"
        # Chercher le mood dans la partie 2 si séparateur présent, sinon dans toute la réponse
        mood_raw = parts[1].strip().lower() if len(parts) >= 2 else response.lower()
        if "mauvais" in mood_raw or "insatisf" in mood_raw or "négatif" in mood_raw:
            mood_label = "mauvais"
        elif "bon" in mood_raw or "satisf" in mood_raw or "positif" in mood_raw or "cordial" in mood_raw:
            mood_label = "bon"

        if not summary or len(summary) < 10:
            summary = create_extractive_summary(transcript)

        return {"summary": summary, "mood_label": mood_label, "mood_confidence": 0.75}
    except Exception as e:
        log(f"[SUMMARY_SENTIMENT] Error: {e}")
        return {
            "summary": create_extractive_summary(transcript),
            "mood_label": "neutre",
            "mood_confidence": 0.5
        }


def diarize_with_stereo_channels(wav_path: str, language: Optional[str], max_new_tokens: int,
                                  with_summary: bool, call_direction: str = "unknown",
                                  ch0_path: str = None, ch1_path: str = None):
    """
    MODE STÉRÉO v2 — Attribution par comparaison d'énergie entre canaux :
    1. VAD sur MONO → capte TOUTE la parole (y compris voix douce)
    2. Fenêtres 300ms : compare RMS ch0 vs ch1 → canal dominant = speaker (crosstalk-proof)
    3. Extraction de chaque tour depuis SON CANAL (jamais mono, même pour les tours courts)
    4. Transcription Voxtral par tour
    5. Summary + sentiment combinés (1 seul appel Voxtral)
    Fallback → Voxtral Speaker ID si mono ou canaux vides.
    """
    log(f"[STEREO] Starting energy-based stereo diarization: direction={call_direction}")
    ok, err, est_dur = _validate_audio(wav_path)
    if not ok:
        return {"error": err}

    import soundfile as sf
    info = sf.info(wav_path)
    if info.channels < 2:
        log(f"[STEREO] Audio is mono ({info.channels}ch) → fallback Voxtral Speaker ID")
        return diarize_with_voxtral_speaker_id(wav_path, language, max_new_tokens, with_summary, call_direction)

    log(f"[STEREO] Audio: {info.channels}ch {info.samplerate}Hz {est_dur:.1f}s")

    # ── Mapping canal → rôle ────────────────────────────────────────────────
    # Asterisk MixMonitor : ch0(left/read) = audio DISTANT, ch1(right/write) = audio LOCAL
    left_role  = "Client"   # ch0 = partie distante (patient/appelant)
    right_role = "Agent"    # ch1 = partie locale (cabinet/standard)
    log(f"[STEREO] Channel mapping: ch0={left_role} (remote), ch1={right_role} (local)")

    # ── Étape 1-2 : Construire les tours par comparaison d'énergie ──────────
    # _build_turns_by_energy gère aussi la détection IVR/attente en interne
    turns = _build_turns_by_energy(wav_path, left_role, right_role)
    if not turns:
        log("[STEREO] No turns → fallback Voxtral Speaker ID")
        mono_fb = _to_mono_tmp(wav_path)
        result = diarize_with_voxtral_speaker_id(mono_fb, language, max_new_tokens, with_summary, call_direction)
        if mono_fb != wav_path and os.path.exists(mono_fb): os.remove(mono_fb)
        return result

    speakers_found = set(t["speaker"] for t in turns)
    if len(speakers_found) < 2:
        log(f"[STEREO] Only {speakers_found} detected → fallback Voxtral Speaker ID")
        mono_fb = _to_mono_tmp(wav_path)
        result = diarize_with_voxtral_speaker_id(mono_fb, language, max_new_tokens, with_summary, call_direction)
        if mono_fb != wav_path and os.path.exists(mono_fb): os.remove(mono_fb)
        return result

    turn_counts = {s: sum(1 for t in turns if t["speaker"] == s) for s in speakers_found}
    log(f"[STEREO] Timeline: {len(turns)} turns ({turn_counts})")

    # ── Étape 3 : Transcrire chaque canal EN ENTIER ─────────────────────────
    # Au lieu de transcrire tour par tour (perte de contexte), on envoie
    # le canal complet (nettoyé du silence) à Voxtral en un seul appel.
    # Voxtral a le contexte complet → meilleure reconnaissance des mots.
    # Puis on distribue le texte sur les blocs VAD par proportion de durée.
    channel_map = {left_role: 0, right_role: 1}
    segments = []

    for role in [left_role, right_role]:
        channel = channel_map[role]
        # Collecter tous les blocs VAD de ce speaker
        role_turns = [(t["start"], t["end"]) for t in turns if t["speaker"] == role]
        if not role_turns:
            continue

        total_speech_dur = sum(e - s for s, e in role_turns)
        log(f"[STEREO] {role}: {len(role_turns)} turns, {total_speech_dur:.1f}s speech → transcribing channel {channel} in full")

        # Extraire uniquement les portions de parole du canal (sans silence)
        # et les concaténer en un seul audio
        try:
            audio = AudioSegment.from_file(wav_path)
            if audio.channels > 1:
                channels = audio.split_to_mono()
                ch_audio = channels[channel]
            else:
                ch_audio = audio

            parts = []
            for s, e in role_turns:
                part = ch_audio[int(s * 1000):int(e * 1000)]
                if len(part) > 0:
                    parts.append(part)
                    # 200ms de silence entre les blocs pour que Voxtral respire
                    parts.append(AudioSegment.silent(duration=200, frame_rate=ch_audio.frame_rate))

            if not parts:
                continue

            clean_audio = parts[0]
            for p in parts[1:]:
                clean_audio += p

            clean_dur = len(clean_audio) / 1000.0
            clean_path = os.path.join(tempfile.gettempdir(), f"ch{channel}_full_{uuid.uuid4().hex}.wav")
            clean_audio.export(clean_path, format="wav")
            log(f"[STEREO] {role}: extracted {clean_dur:.1f}s speech audio")
        except Exception as e_extract:
            log(f"[STEREO] {role}: extraction failed: {e_extract}")
            continue

        # Transcrire le canal complet en un seul appel Voxtral
        try:
            tokens = max(256, min(int(total_speech_dur * 10), 12000))
            timeout = min(int(total_speech_dur * 0.7) + 60, 600)

            if USE_WHISPER_STEREO:
                full_text = _transcribe_turn_whisper(clean_path, language or "fr")
            else:
                full_text = _transcribe_turn(clean_path, language, total_speech_dur, from_channel=True)
        finally:
            if os.path.exists(clean_path):
                os.remove(clean_path)

        if not full_text or len(full_text.split()) < 2:
            log(f"[STEREO] {role}: empty transcription, skipping")
            continue

        # Nettoyage
        full_text = remove_repetitive_loops(full_text, max_repetitions=3)
        full_text = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold)\]\s*', '', full_text, flags=re.IGNORECASE).strip()

        log(f"[STEREO] {role}: {len(full_text)} chars transcribed")

        # Distribuer le texte sur les blocs VAD par proportion de durée
        words = full_text.split()
        total_words = len(words)
        word_idx = 0

        for j, (b_start, b_end) in enumerate(role_turns):
            block_dur = b_end - b_start
            if j == len(role_turns) - 1:
                # Dernier bloc : tous les mots restants
                block_words = words[word_idx:]
            else:
                n_words = max(1, round(total_words * block_dur / total_speech_dur))
                n_words = min(n_words, total_words - word_idx)
                block_words = words[word_idx:word_idx + n_words]
                word_idx += n_words

            if block_words:
                segments.append({
                    "speaker": role,
                    "start": round(b_start, 2),
                    "end": round(b_end, 2),
                    "text": " ".join(block_words),
                    "mood": None,
                })

    if not segments:
        log("[STEREO] All channels empty → fallback Voxtral Speaker ID")
        mono_fb = _to_mono_tmp(wav_path)
        result = diarize_with_voxtral_speaker_id(mono_fb, language, max_new_tokens, with_summary, call_direction)
        if mono_fb != wav_path and os.path.exists(mono_fb): os.remove(mono_fb)
        return result

    # Trier par timestamp (les segments des deux canaux sont intercalés)
    segments.sort(key=lambda s: s["start"])

    # Supprimer les segments dupliqués (annonces en boucle)
    segments = remove_duplicate_segments(segments)
    if not segments:
        return {"segments": [], "transcript": "", "summary": "Aucune conversation détectée.", "diarization_mode": "stereo_energy"}

    # Fusionner segments consécutifs même speaker
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if prev["speaker"] == seg["speaker"]:
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            prev["end"] = seg["end"]
        else:
            merged.append(seg)

    log(f"[STEREO] Final: {len(merged)} segments")

    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in merged if s.get("text"))
    result = {
        "segments": merged,
        "transcript": full_transcript,
        "diarization_mode": "stereo_energy_whisper" if USE_WHISPER_STEREO else "stereo_energy",
        "audio_duration": est_dur,
    }

    # ── Étape 6 : Summary (même fonction que mono — résultats meilleurs) ────
    if with_summary:
        log("[STEREO] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript, duration_seconds=est_dur)

    # ── Étape 7 : Sentiment (même analyse que mono — Voxtral sur le texte) ──
    if ENABLE_SENTIMENT:
        log("[STEREO] Analyzing sentiment by speaker...")
        speaker_sentiments = analyze_sentiment_by_speaker(merged)
        if speaker_sentiments:
            mood_overall, mood_by_speaker, client_mood = _attach_sentiment(merged, speaker_sentiments)
            result["mood_overall"]    = mood_overall
            result["mood_by_speaker"] = mood_by_speaker
            result["mood_client"]     = client_mood

    log("[STEREO] Energy-based stereo diarization completed successfully")
    return result

# =============================================================================
# PIPELINE PRINCIPAL — VOXTRAL SPEAKER ID (fallback / mono)
# =============================================================================
def parse_speaker_identified_transcript(transcript: str, total_duration: float) -> List[Dict[str, Any]]:
    if not transcript:
        return []
    segments     = []
    current_time = 0.0
    lines        = transcript.split('\n')
    valid_lines  = []
    agent_variants  = {"agent", "secrétaire", "secretaire", "médecin", "medecin", "docteur", "praticien"}
    client_variants = {"client", "patient", "appelant"}
    if len(lines) == 1 or (len(lines) == 2 and not lines[1].strip()):
        text    = lines[0].strip()
        pattern = r'(Agent|Client|Secrétaire|Secretaire|Patient|Médecin|Medecin|Docteur|Praticien|Appelant|AGENT|CLIENT):\s*([^:]+?)(?=\s*(?:Agent:|Client:|Secrétaire:|Secretaire:|Patient:|Médecin:|Medecin:|Docteur:|Praticien:|Appelant:|AGENT:|CLIENT:|$))'
        matches = re.findall(pattern, text, re.IGNORECASE)
        log(f"[PARSE] Inline mode: found {len(matches)} speaker segments")
        speaker_mapping: Dict[str, str] = {}
        speaker_counter = 0
        for speaker_raw, text_part in matches:
            sn_lower = speaker_raw.lower().strip()
            if sn_lower in agent_variants:
                sn = "Agent"
            elif sn_lower in client_variants:
                sn = "Client"
            else:
                sn = speaker_raw.capitalize()
            if sn not in speaker_mapping:
                speaker_mapping[sn] = f"SPEAKER_{speaker_counter:02d}"; speaker_counter += 1
            tp = remove_repetitive_loops(text_part.strip(), max_repetitions=3)
            if tp:
                valid_lines.append((speaker_mapping[sn], tp))
    else:
        speaker_mapping = {}
        speaker_counter = 0
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            speaker_part = parts[0].strip()
            text_part    = parts[1].strip()
            if any(w in speaker_part.lower() for w in ['agent', 'professionnel', 'cabinet', 'secrétaire']):
                sn = "Agent"
            elif any(w in speaker_part.lower() for w in ['client', 'appelant', 'patient']):
                sn = "Client"
            elif speaker_part.lower() in ['agent', 'client']:
                sn = speaker_part.capitalize()
            else:
                if any(w in text_part.lower() for w in ['cabinet', 'je vais voir', 'on vous rappelle']):
                    sn = "Agent"
                elif any(w in text_part.lower() for w in ['je voudrais', 'est-ce que je peux']):
                    sn = "Client"
                else:
                    sn = "Agent"
            if sn not in speaker_mapping:
                speaker_mapping[sn] = f"SPEAKER_{speaker_counter:02d}"; speaker_counter += 1
            tp = remove_repetitive_loops(text_part, max_repetitions=3)
            if tp:
                valid_lines.append((speaker_mapping[sn], tp))
    if not valid_lines:
        return []
    total_words = sum(len(text.split()) for _, text in valid_lines)
    for i, (speaker, text) in enumerate(valid_lines):
        word_count       = len(text.split())
        segment_duration = (word_count / total_words) * total_duration if total_words > 0 else total_duration / len(valid_lines)
        start_time       = current_time
        end_time         = min(current_time + segment_duration, total_duration)
        segments.append({"speaker": speaker, "start": start_time, "end": end_time, "text": text, "mood": None})
        current_time = end_time
        log(f"[PARSE] Segment {i+1}: {speaker} ({start_time:.1f}s-{end_time:.1f}s) '{text[:30]}...'")
    return merge_micro_segments(segments)

def llm_verify_and_fix_attributions(segments: List[Dict[str, Any]]) -> int:
    if not DETECT_GLOBAL_SWAP or not segments or len(segments) < 2:
        return 0
    log("[LLM_REVIEW] Starting intelligent transcript review...")
    transcript_lines = [f"{i+1}. {s.get('speaker', '?')}: {s.get('text', '').strip()}" for i, s in enumerate(segments)]
    if len(segments) > 30:
        transcript_lines = transcript_lines[:15] + [f"... ({len(segments)-30} segments omis) ..."] + transcript_lines[-15:]
    transcript_text = "\n".join(transcript_lines)
    review_prompt = f"""Contexte: Cabinet médical/dentaire. L'Agent est le SECRÉTARIAT qui RÉPOND. Le Client est la PERSONNE qui APPELLE.

Transcript:
{transcript_text}

Mission: Identifie UNIQUEMENT les segments ÉVIDEMMENT mal attribués.
Format: "3:Client" ou "7:Agent" — un par ligne.
Si AUCUNE erreur évidente, réponds: "OK"
"""
    try:
        conv   = [{"role": "user", "content": [{"type": "text", "text": review_prompt}]}]
        result = run_voxtral_with_timeout(conv, max_new_tokens=max(100, len(segments) * 5 + 20), timeout=30)
        response = result.get("text", "").strip()
        log(f"[LLM_REVIEW] Response: {response}")
        if response.upper() == "OK" or not response:
            return 0
        corrections = 0
        for line in response.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            try:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                idx    = int(parts[0].strip()) - 1
                new_sp = parts[1].strip()
                if 0 <= idx < len(segments) and new_sp in ["Agent", "Client"]:
                    old_sp = segments[idx]["speaker"]
                    if old_sp != new_sp:
                        log(f"[LLM_REVIEW] Correcting segment {idx+1}: {old_sp} → {new_sp}")
                        segments[idx]["speaker"] = new_sp
                        corrections += 1
            except (ValueError, IndexError):
                continue
        log(f"[LLM_REVIEW] {'✅ Corrected' if corrections else '✅ No corrections'} ({corrections} attributions)")
        return corrections
    except Exception as e:
        log(f"[LLM_REVIEW] Error: {e}")
        return 0

def detect_and_fix_speaker_inversion(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments or len(segments) < 2:
        return segments
    llm_verify_and_fix_attributions(segments)
    return segments

def diarize_with_voxtral_speaker_id(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool, call_direction: str = "unknown"):
    """
    call_direction : "inbound"  → client appelle, agent décroche en premier
                     "outbound" → agent appelle, client décroche en premier
                     "unknown"  → Voxtral décide sur le contenu sémantique uniquement
    """
    log(f"[VOXTRAL_ID] Starting speaker identification: language={language}, direction={call_direction}")
    ok, err, est_dur = _validate_audio(wav_path)
    if not ok:
        return {"error": err}
    speaker_tokens = max(512, min(int(est_dur * 10), 12000))
    infer_timeout  = min(int(est_dur * 0.7) + 60, 600)
    log(f"[VOXTRAL_ID] Audio: {est_dur:.1f}s → tokens={speaker_tokens}, timeout={infer_timeout}s")
    if call_direction == "inbound":
        direction_context = (
            "CONTEXTE : Appel ENTRANT — le CLIENT a appelé le cabinet.\n"
            "L'AGENT est celui qui gère le cabinet et a répondu à cet appel.\n"
            "Le CLIENT est celui qui a appelé pour obtenir quelque chose.\n\n"
        )
    elif call_direction == "outbound":
        direction_context = (
            "CONTEXTE : Appel SORTANT — l'AGENT a appelé le client.\n"
            "L'AGENT est celui qui travaille au cabinet et a passé cet appel.\n"
            "Le CLIENT est la personne qui a été appelée.\n\n"
        )
    else:
        direction_context = (
            "CONTEXTE : Direction de l'appel inconnue.\n"
            "Identifie Agent et Client UNIQUEMENT sur ce qu'ils disent, "
            "pas sur l'ordre de parole.\n\n"
        )
    instruction = (
        f"lang:{language or 'fr'} "
        "Tu transcris un appel téléphonique professionnel. "
        "Il y a exactement deux interlocuteurs : l'Agent et le Client.\n\n"
        + direction_context +
        "COMMENT IDENTIFIER L'AGENT (secrétariat / cabinet / professionnel de santé) :\n"
        "• Nomme le cabinet ou service : 'cabinet dentaire Dupont', 'docteur Martin', 'centre médical'\n"
        "• Gère l'agenda : 'j'ai de la place le...', 'je vous propose le...', 'je note', 'c'est noté'\n"
        "• Questions pro : 'c'est de la part de qui ?', 'votre date de naissance ?', 'votre numéro ?'\n"
        "• Formules pro : 'ne quittez pas', 'je vais regarder', 'je vous rappelle', 'bonne journée'\n"
        "• Confirme : 'entendu', 'c'est bien noté', 'à lundi donc', 'rendez-vous confirmé'\n\n"
        "COMMENT IDENTIFIER LE CLIENT (patient / appelant / particulier) :\n"
        "• Expose un besoin : 'je vous appelle pour', 'je voudrais', 'j'aurais besoin', 'c'est possible ?'\n"
        "• Demande ou modifie un RDV : 'prendre rendez-vous', 'annuler', 'reporter', 'confirmer'\n"
        "• Donne ses infos personnelles : son nom, date de naissance, numéro de téléphone, adresse\n"
        "• Parle de proches : 'ma femme', 'mon mari', 'mon fils', 'ma fille', 'mon enfant'\n"
        "• Décrit un problème médical : 'j'ai mal', 'ça fait X jours', 'j'ai un souci avec'\n\n"
        "FORMAT OBLIGATOIRE — une ligne par prise de parole :\n"
        "Agent: Bonjour, cabinet dentaire Dupont, j'écoute.\n"
        "Client: Bonjour, je voudrais prendre rendez-vous pour un détartrage.\n"
        "Agent: Bien sûr, vous êtes disponible quand ?\n"
        "Client: La semaine prochaine si possible.\n\n"
        "RÈGLES STRICTES :\n"
        "• Identifie les rôles sur CE QUE LES GENS DISENT, pas sur l'ordre de parole\n"
        "• Une seule prise de parole par ligne — ne fusionne JAMAIS deux voix\n"
        "• Transcris MOT POUR MOT, sans paraphraser ni corriger\n"
        "• Conserve les hésitations : 'euh', 'bah', 'alors', 'donc'\n"
        "• N'écris JAMAIS [Musique] [Silence] [Attente] [Sonnerie] [Pause]"
    )
    conv = [{"role": "user", "content": [
        {"type": "audio", "path": wav_path},
        {"type": "text", "text": instruction}
    ]}]
    log("[VOXTRAL_ID] Starting transcription with speaker identification...")
    out = run_voxtral_with_timeout(conv, max_new_tokens=speaker_tokens, timeout=infer_timeout)
    speaker_transcript = (out.get("text") or "").strip()
    if not speaker_transcript:
        log("[VOXTRAL_ID] Empty result, returning empty")
        return {"segments": [], "transcript": "", "summary": "Aucune conversation détectée."}
    original_len       = len(speaker_transcript)
    speaker_transcript = remove_repetitive_loops(speaker_transcript, max_repetitions=5)
    chars_per_sec      = len(speaker_transcript) / max(est_dur, 1)
    if chars_per_sec > 50:
        log(f"[VOXTRAL_ID] WARNING: Suspicious chars/sec ratio: {chars_per_sec:.1f}")
    if original_len != len(speaker_transcript):
        log(f"[VOXTRAL_ID] Cleaned repetitive loops: {original_len} → {len(speaker_transcript)} chars")
    speaker_transcript = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold)\]\s*', '', speaker_transcript, flags=re.IGNORECASE).strip()
    log(f"[VOXTRAL_ID] Completed: {len(speaker_transcript)} chars in {out.get('latency_s', 0):.1f}s")
    log(f"[VOXTRAL_ID] Raw (first 500): {speaker_transcript[:500]}")
    if len(speaker_transcript) < 10:
        return {"segments": [], "transcript": "", "summary": "Aucune conversation détectée (musique/silence uniquement)."}
    segments = parse_speaker_identified_transcript(speaker_transcript, est_dur)
    if not segments:
        log("[VOXTRAL_ID] No speaker format found, retrying with forced prompt...")
        forced = (
            f"lang:{language or 'fr'} "
            "Tu DOIS séparer les speakers. Voici le transcript brut, reformate-le:\n\n"
            f"{speaker_transcript}\n\n"
            "REFORMATE en séparant OBLIGATOIREMENT Agent et Client:\n"
            "Agent: [sa phrase]\nClient: [sa phrase]\n...\n"
            "L'Agent répond/travaille. Le Client appelle/demande."
        )
        retry_tokens = min(max_new_tokens, int(len(speaker_transcript) * 1.5))
        out_retry    = run_voxtral_with_timeout([{"role": "user", "content": [{"type": "text", "text": forced}]}],
                                                max_new_tokens=retry_tokens, timeout=60)
        retry_transcript = (out_retry.get("text") or "").strip()
        if retry_transcript:
            segments = parse_speaker_identified_transcript(retry_transcript, est_dur)
        if not segments:
            segments = [{"speaker": "Agent", "start": 0.0, "end": est_dur, "text": speaker_transcript.strip(), "mood": None}]
    log(f"[VOXTRAL_ID] Parsed {len(segments)} segments")
    segments = _enforce_max_two_speakers(segments)
    _map_roles(segments)
    segments = detect_and_fix_speaker_inversion(segments)
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript, "diarization_mode": "voxtral_speaker_id"}
    if with_summary:
        log("[VOXTRAL_ID] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript, duration_seconds=est_dur)
    if ENABLE_SENTIMENT:
        log("[VOXTRAL_ID] Computing sentiment by speaker...")
        speaker_sentiments = analyze_sentiment_by_speaker(segments)
        if speaker_sentiments:
            for seg in segments:
                sp = seg.get("speaker")
                if sp and sp in speaker_sentiments:
                    seg["mood"] = speaker_sentiments[sp]
            weighted = [(seg["mood"], float(seg["end"]) - float(seg["start"]))
                        for seg in segments if seg.get("text") and seg.get("mood")]
            result["mood_overall"]    = aggregate_mood(weighted) if weighted else None
            result["mood_by_speaker"] = speaker_sentiments
            client_mood               = get_client_sentiment(speaker_sentiments, segments)
            result["mood_client"]     = client_mood
            label_fr = client_mood.get("label_fr") or "inconnu"
            conf     = float(client_mood.get("confidence") or 0.0)
            log(f"[VOXTRAL_ID] Client sentiment: {label_fr} (confidence: {conf:.2f})")
    log("[VOXTRAL_ID] Voxtral speaker identification completed successfully")
    return result

# =============================================================================
# HEALTH CHECK
# =============================================================================
def health():
    info = {
        "app_version": APP_VERSION, "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "device": _device_str(),
        "quant_mode": QUANT_MODE, "model_id": MODEL_ID,
        "sentiment": "Voxtral-based" if ENABLE_SENTIMENT else "Disabled",
        "single_voice_detection": ENABLE_SINGLE_VOICE_DETECTION,
        "hold_music_detection": ENABLE_HOLD_MUSIC_DETECTION,
        "detect_global_swap": DETECT_GLOBAL_SWAP, "role_labels": ROLE_LABELS,
        "stereo_diarization": STEREO_DIARIZATION,
        "diarization_mode": "stereo_channels (fallback: voxtral_speaker_id)" if STEREO_DIARIZATION else "voxtral_speaker_id",
    }
    try:
        if _model is not None:
            p = next(_model.parameters())
            info["voxtral_device"] = str(p.device)
            info["voxtral_dtype"]  = str(p.dtype)
    except Exception:
        pass
    errors = {}
    try:
        load_voxtral()
    except Exception as e:
        errors["voxtral_load"] = f"{type(e).__name__}: {e}"
    return {"ok": len(errors) == 0, "info": info, "errors": errors}

# =============================================================================
# HANDLER RUNPOD
# =============================================================================
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input", {}) or {}
    log(f"[HANDLER] New job received: {inp.get('task', 'unknown')}")
    if inp.get("ping"):
        return {"pong": True}
    if inp.get("task") == "health":
        return health()
    task           = (inp.get("task") or "transcribe_diarized").lower()
    language       = inp.get("language") or None
    max_new_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))
    with_summary   = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))
    call_direction = (inp.get("call_direction") or "").lower().strip()
    force_mono     = bool(inp.get("force_mono", False))
    if call_direction not in ("inbound", "outbound"):
        audio_url = inp.get("audio_url") or inp.get("file_path") or ""
        filename  = audio_url.split("/")[-1].lower()
        if filename.startswith("in-") or filename.startswith("in_"):
            call_direction = "inbound"
        elif filename.startswith("out-") or filename.startswith("out_"):
            call_direction = "outbound"
        else:
            call_direction = "unknown"
    log(f"[HANDLER] Task: {task}, language: {language}, direction: {call_direction}, max_tokens: {max_new_tokens}, summary: {with_summary}, force_mono: {force_mono}")
    local_path, cleanup = None, False
    is_stereo    = False
    mono_path    = None
    mono_cleanup = False
    ch0_path     = None
    ch1_path     = None
    try:
        if inp.get("audio_url"):
            url = inp["audio_url"].strip()
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.path or parsed.path in ('', '/'):
                return {"error": f"audio_url invalide ou tronquée: '{url}'"}
            local_path = _download_to_tmp(url); cleanup = True
            log(f"[HANDLER] Downloaded audio from URL: {url}")
        elif inp.get("audio_b64"):
            local_path = _b64_to_tmp(inp["audio_b64"]); cleanup = True
        elif inp.get("file_path"):
            local_path = inp["file_path"]
        else:
            return {"error": "Provide 'audio_url', 'audio_b64' or 'file_path'."}

        ok, err, _ = _validate_audio(local_path)
        if not ok:
            return {"error": err}
        log(f"[HANDLER] Processing audio: {local_path}")

        # ── Détection stéréo et extraction canaux ─────────────────────────────
        mono_path = local_path

        if STEREO_DIARIZATION:
            import soundfile as sf
            info = sf.info(local_path)
            if info.channels > 1:
                is_stereo = True
                # Extraire les canaux une seule fois (réutilisés pour hold music + diarization)
                ch0_path = _extract_mono_channel(local_path, 0)
                ch1_path = _extract_mono_channel(local_path, 1)
                # Version mono pour single voice detection
                audio_mono = AudioSegment.from_file(local_path).set_channels(1)
                mono_path  = local_path.rsplit(".", 1)[0] + "_mono_detect.wav"
                audio_mono.export(mono_path, format="wav")
                mono_cleanup = True
                log(f"[HANDLER] Stereo detected: extracted ch0, ch1, mono")

        # ── Détection musique d'attente ───────────────────────────────────────
        if ENABLE_HOLD_MUSIC_DETECTION:
            if is_stereo and ch0_path and ch1_path:
                # Stéréo : analyser CHAQUE canal séparément
                hold_ch0 = detect_hold_music(ch0_path)
                hold_ch1 = detect_hold_music(ch1_path)
                log(f"[HOLD_MUSIC] CH0(left): hold={hold_ch0['is_hold_music']}, ratio={hold_ch0['speech_ratio']:.4f}")
                log(f"[HOLD_MUSIC] CH1(right): hold={hold_ch1['is_hold_music']}, ratio={hold_ch1['speech_ratio']:.4f}")
                # Hold music seulement si LES DEUX canaux sont vides/musique
                if hold_ch0["is_hold_music"] and hold_ch1["is_hold_music"]:
                    log("[HANDLER] Hold music on BOTH channels — skipping inference")
                    hold_result = {
                        "is_hold_music": True,
                        "speech_ratio": max(hold_ch0["speech_ratio"], hold_ch1["speech_ratio"]),
                        "duration": max(hold_ch0.get("duration", 0), hold_ch1.get("duration", 0)),
                        "reason": f"both_channels (L={hold_ch0['reason']}, R={hold_ch1['reason']})",
                        "detection_time": hold_ch0.get("detection_time", 0) + hold_ch1.get("detection_time", 0),
                    }
                    for p in [ch0_path, ch1_path]:
                        if p and os.path.exists(p): os.remove(p)
                    if mono_cleanup and os.path.exists(mono_path): os.remove(mono_path)
                    return {"task": task, **build_hold_music_response(hold_result, with_summary)}
            else:
                # Mono : détection classique
                hold_result = detect_hold_music(mono_path)
                if hold_result["is_hold_music"]:
                    log("[HANDLER] Hold music detected — skipping inference")
                    return {"task": task, **build_hold_music_response(hold_result, with_summary)}
                if hold_result.get("speech_blocks"):
                    local_path, cleanup = trim_audio_to_speech_blocks(local_path, hold_result, cleanup)
                    # IMPORTANT : mono_path doit suivre local_path après trim
                    mono_path = local_path

        # ── Détection voix unique ────────────────────────────────────────────
        if ENABLE_SINGLE_VOICE_DETECTION:
            if is_stereo:
                # STÉRÉO : heuristique VAD (pas d'appel LLM — on sait déjà via les canaux)
                # Si un canal a < 2s de parole → c'est single voice (IVR/répondeur/annonce)
                ch0_vad_quick = _vad_speech_blocks(ch0_path) if ch0_path else []
                ch1_vad_quick = _vad_speech_blocks(ch1_path) if ch1_path else []
                ch0_speech_quick = sum(e - s for s, e in ch0_vad_quick)
                ch1_speech_quick = sum(e - s for s, e in ch1_vad_quick)
                is_single_voice = (ch0_speech_quick < 5.0 or ch1_speech_quick < 5.0)

                if is_single_voice:
                    active_speech = max(ch0_speech_quick, ch1_speech_quick)
                    log(f"[HANDLER] VAD heuristic: single voice detected (ch0={ch0_speech_quick:.1f}s, ch1={ch1_speech_quick:.1f}s)")
                    out = transcribe_single_voice_content(mono_path, language, max_new_tokens, with_summary)
                    if out is None:
                        log("[HANDLER] Real conversation found → switching to full diarization")
                    elif "error" not in out:
                        for p in [ch0_path, ch1_path]:
                            if p and os.path.exists(p): os.remove(p)
                        if mono_cleanup and os.path.exists(mono_path): os.remove(mono_path)
                        return {"task": task, **out}
                else:
                    log(f"[HANDLER] VAD heuristic: conversation (ch0={ch0_speech_quick:.1f}s, ch1={ch1_speech_quick:.1f}s)")
            else:
                # MONO : pas de détection single voice — Voxtral Speaker ID gère tout
                # Les répondeurs/IVR seront transcrits normalement (1 speaker = Agent)
                # Seule la musique d'attente est skippée (hold music detection RMS en amont)
                log("[HANDLER] Mono audio — skipping single voice detection, direct to Voxtral Speaker ID")

        # Cleanup mono detection
        if mono_cleanup and os.path.exists(mono_path):
            os.remove(mono_path)

        # ── Pipeline principal ────────────────────────────────────────────────
        if force_mono:
            # A/B test : forcer le pipeline mono même sur du stéréo
            log("[HANDLER] FORCE_MONO: Converting to mono + Voxtral Speaker ID")
            mono_fb = _to_mono_tmp(local_path)
            out = diarize_with_voxtral_speaker_id(mono_fb, language, max_new_tokens, with_summary, call_direction)
            if mono_fb != local_path and os.path.exists(mono_fb): os.remove(mono_fb)
        elif STEREO_DIARIZATION:
            log("[HANDLER] Using Stereo Channel Diarization (fallback: Voxtral Speaker ID if mono)")
            out = diarize_with_stereo_channels(local_path, language, max_new_tokens, with_summary, call_direction,
                                               ch0_path=ch0_path, ch1_path=ch1_path)
        else:
            log("[HANDLER] Using Voxtral Speaker Identification mode")
            out = diarize_with_voxtral_speaker_id(local_path, language, max_new_tokens, with_summary, call_direction)

        if "error" in out:
            return out
        log("[HANDLER] Transcription completed successfully")
        return {"task": task, **out}

    except Exception as e:
        log(f"[HANDLER] CRITICAL ERROR: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        try:
            _gpu_clear()
        except Exception:
            pass
        # Cleanup fichiers temporaires (canaux stéréo + mono detection)
        for p in [ch0_path, ch1_path]:
            try:
                if p and p != local_path and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        if mono_cleanup:
            try:
                if mono_path and mono_path != local_path and os.path.exists(mono_path):
                    os.remove(mono_path)
            except Exception:
                pass
        try:
            if cleanup and local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

# =============================================================================
# INITIALISATION
# =============================================================================
try:
    log("[INIT] Starting conditional preload...")
    log(f"[INIT] QUANT_MODE={QUANT_MODE} | APP_VERSION={APP_VERSION}")
    if not check_gpu_memory():
        log(f"[CRITICAL] Insufficient GPU memory — skipping preload")
    else:
        log("[INIT] Preloading Voxtral...")
        load_voxtral()
        log("[INIT] Preload completed successfully")
    log(f"[INIT] Hold music detection: {'ENABLED' if ENABLE_HOLD_MUSIC_DETECTION else 'DISABLED'}")
    log(f"[INIT] Single voice detection: {'ENABLED' if ENABLE_SINGLE_VOICE_DETECTION else 'DISABLED'}")
    log(f"[INIT] Sentiment analysis: {'ENABLED' if ENABLE_SENTIMENT else 'DISABLED'}")
    log(f"[INIT] Stereo diarization: {'ENABLED' if STEREO_DIARIZATION else 'DISABLED (Voxtral Speaker ID)'}")
    log(f"[INIT] Whisper stereo transcription: {'ENABLED (' + WHISPER_MODEL_SIZE + ')' if USE_WHISPER_STEREO else 'DISABLED (using Voxtral)'}")
    if USE_WHISPER_STEREO:
        log("[INIT] Preloading Whisper...")
        _load_whisper()
    log(f"[INIT] LLM review (DETECT_GLOBAL_SWAP): {'ENABLED' if DETECT_GLOBAL_SWAP else 'DISABLED'}")
except Exception as e:
    log(f"[WARN] Preload failed — will load on first request: {e}")
    _processor = None
    _model     = None

runpod.serverless.start({"handler": handler})
