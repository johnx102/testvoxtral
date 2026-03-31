#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Voxtral Serverless Worker — RunPod
# Pipeline : Transcription + Diarization + Résumé + Humeur
# Modes   : Stéréo (channel split) | Mono (Voxtral Speaker ID)
# Quant   : bitsandbytes INT4 (~12-14 GB VRAM)
# =============================================================================

import os, time, base64, tempfile, uuid, requests, traceback, re, gc
from typing import Optional, List, Dict, Any, Tuple

import torch

# =============================================================================
# DIAGNOSTIC GPU/CUDA AU DÉMARRAGE
# =============================================================================
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
APP_VERSION        = os.environ.get("APP_VERSION", "voxtral-stereo-v4.0")
MODEL_ID           = os.environ.get("MODEL_ID", "mistralai/Voxtral-Small-24B-2507").strip()
HF_TOKEN           = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS     = int(os.environ.get("MAX_NEW_TOKENS", "664"))
MAX_DURATION_S     = int(os.environ.get("MAX_DURATION_S", "3600"))
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Sentiment
ENABLE_SENTIMENT   = os.environ.get("ENABLE_SENTIMENT", "1") == "1"

# Détection voix unique (annonces/IVR/messagerie)
ENABLE_SINGLE_VOICE_DETECTION  = os.environ.get("ENABLE_SINGLE_VOICE_DETECTION", "1") == "1"
SINGLE_VOICE_DETECTION_TIMEOUT = int(os.environ.get("SINGLE_VOICE_DETECTION_TIMEOUT", "30"))
SINGLE_VOICE_SUMMARY_TOKENS    = int(os.environ.get("SINGLE_VOICE_SUMMARY_TOKENS", "48"))

# Détection musique d'attente
ENABLE_HOLD_MUSIC_DETECTION    = os.environ.get("ENABLE_HOLD_MUSIC_DETECTION", "1") == "1"
HOLD_MUSIC_SPEECH_RATIO_HARD   = float(os.environ.get("HOLD_MUSIC_SPEECH_RATIO_HARD", "0.03"))
HOLD_MUSIC_SPEECH_RATIO_SOFT   = float(os.environ.get("HOLD_MUSIC_SPEECH_RATIO_SOFT", "0.08"))
HOLD_MUSIC_MIN_DURATION        = float(os.environ.get("HOLD_MUSIC_MIN_DURATION", "30.0"))

# Rôles speakers
ROLE_LABELS = [r.strip() for r in os.environ.get("ROLE_LABELS", "Agent,Client").split(",") if r.strip()]

# Relecture LLM des attributions (désactivé par défaut — Voxtral se trompe parfois)
DETECT_GLOBAL_SWAP = os.environ.get("DETECT_GLOBAL_SWAP", "0") == "1"

# Quantization : "torchao" = bitsandbytes INT4 (12-14 GB VRAM), "none" = bfloat16 (48 GB)
QUANT_MODE = os.environ.get("QUANT_MODE", "torchao").lower()

# Stéréo : Asterisk MixMonitor enregistre left=rx (audio entrant), right=tx (audio sortant)
# Pour un appel entrant : left = Caller (Client), right = Agent
# Pour un appel sortant : left = Agent (qui a composé), right = Client (qui a décroché)
STEREO_ENABLED = os.environ.get("STEREO_ENABLED", "1") == "1"
# Seuil min de parole (secondes) pour considérer qu'un canal contient de la voix
STEREO_MIN_SPEECH_S = float(os.environ.get("STEREO_MIN_SPEECH_S", "2.0"))
# Pause minimum entre deux tours de parole (secondes)
STEREO_VAD_MIN_SILENCE = float(os.environ.get("STEREO_VAD_MIN_SILENCE", "0.4"))
# Durée minimum d'un segment de parole (secondes)
STEREO_VAD_MIN_SPEECH = float(os.environ.get("STEREO_VAD_MIN_SPEECH", "0.25"))

# Globals
_processor = None
_model     = None

# =============================================================================
# HELPERS
# =============================================================================
def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _normalize_audio(path: str) -> str:
    """
    Normalise l'audio pour le traitement :
    - Resample < 16kHz → 16kHz
    - NE convertit PAS en mono (stéréo préservé pour channel split)
    Retourne le chemin du fichier normalisé (ou l'original si déjà OK).
    """
    try:
        audio = AudioSegment.from_file(path)
        if audio.frame_rate < 16000:
            audio = audio.set_frame_rate(16000)
            out_path = path.rsplit(".", 1)[0] + "_16k.wav"
            audio.export(out_path, format="wav")
            log(f"[AUDIO] Resampled: {audio.frame_rate}Hz → 16kHz ({audio.channels}ch)")
            try:
                os.remove(path)
            except Exception:
                pass
            return out_path
        return path
    except Exception as e:
        log(f"[AUDIO] Normalization failed ({e}), using original")
        return path

def _to_mono(path: str) -> str:
    """Convertit en mono si stéréo. Pour les fonctions qui nécessitent du mono."""
    try:
        audio = AudioSegment.from_file(path)
        if audio.channels <= 1:
            return path
        mono = audio.set_channels(1)
        out = path.rsplit(".", 1)[0] + "_mono.wav"
        mono.export(out, format="wav")
        return out
    except Exception:
        return path

def _is_stereo(path: str) -> bool:
    """Vérifie si le fichier audio est stéréo."""
    try:
        audio = AudioSegment.from_file(path)
        return audio.channels >= 2
    except Exception:
        return False

def _split_stereo_channels(path: str) -> Tuple[str, str]:
    """Sépare un fichier stéréo en 2 fichiers mono (left, right)."""
    audio = AudioSegment.from_file(path)
    channels = audio.split_to_mono()
    base = path.rsplit(".", 1)[0]
    left_path = f"{base}_ch0.wav"
    right_path = f"{base}_ch1.wav"
    channels[0].export(left_path, format="wav")
    channels[1].export(right_path, format="wav")
    log(f"[STEREO] Split: left={left_path}, right={right_path}")
    return left_path, right_path

def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return _normalize_audio(path)

def _b64_to_tmp(b64: str) -> str:
    raw = base64.b64decode(b64)
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(raw)
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
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        cached    = torch.cuda.memory_reserved(0) / 1e9
        free      = total - cached
        min_vram  = 12 if QUANT_MODE != "none" else 45
        log(f"[GPU] Total: {total:.1f}GB | Cached: {cached:.1f}GB | Free: {free:.1f}GB")
        if total < min_vram:
            log(f"[ERROR] GPU {total:.1f}GB < minimum {min_vram}GB (QUANT_MODE={QUANT_MODE})")
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
# CHARGEMENT VOXTRAL — bitsandbytes INT4
# =============================================================================
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    log(f"[INIT] Loading Voxtral: {MODEL_ID} [QUANT_MODE={QUANT_MODE}]")

    proc_kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        proc_kwargs["token"] = HF_TOKEN
    log("[INIT] Loading processor...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **proc_kwargs)
    log("[INIT] Processor loaded")

    mdl_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    if QUANT_MODE in ("torchao", "bnb4") and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            log("[INIT] Configuring bitsandbytes INT4...")
            mdl_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            log("[INIT] INT4 BnB config ready — model will load in 12-14 GB")
        except ImportError:
            log("[WARN] bitsandbytes not installed — falling back to bfloat16")
            mdl_kwargs["dtype"] = torch.bfloat16
    elif QUANT_MODE == "bnb8" and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            mdl_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            log("[INIT] INT8 BnB config ready — model will load in ~24 GB")
        except ImportError:
            mdl_kwargs["dtype"] = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        mdl_kwargs["dtype"] = dtype
        log(f"[INIT] Using dtype: {dtype} (no quantization)")

    log(f"[INIT] Loading model... (this may take several minutes) [QUANT_MODE={QUANT_MODE}]")
    _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **mdl_kwargs)
    log("[INIT] Model loaded successfully")

    try:
        p = next(_model.parameters())
        log(f"[INIT] Voxtral device={p.device}, dtype={p.dtype}")
        gpu_params   = sum(1 for p in _model.parameters() if p.device.type == "cuda")
        total_params = sum(1 for p in _model.parameters())
        ratio = gpu_params / total_params if total_params > 0 else 0
        log(f"[INIT] GPU params ratio: {ratio:.2f} ({gpu_params}/{total_params})")
        if ratio < 0.95:
            log(f"[WARN] Only {ratio:.1%} of model on GPU — expect slow performance")
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
    if not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    text_lower = text.lower()

    strong_negative = [
        "pas content du tout", "vraiment pas content", "très mécontent", "très déçu",
        "annulé à cause", "inadmissible", "inacceptable", "catastrophe", "scandaleux",
        "en colère", "énervé", "furieux", "exaspéré", "ras-le-bol", "j'en ai marre",
        "c'est honteux", "n'importe quoi", "vous vous moquez", "c'est une blague",
        "c'est du foutage", "c'est abusé", "vous abusez", "c'est grave"
    ]
    mild_negative = [
        "pas content", "pas satisfait", "déçu", "problème", "difficile", "compliqué",
        "encore", "toujours pas", "ça fait longtemps", "combien de fois", "à chaque fois",
        "ça suffit", "patienter", "retard", "toujours le même", "ça traîne", "trop long"
    ]
    positive_indicators = [
        "merci", "parfait", "très bien", "super", "excellent", "c'est bon", "d'accord",
        "ça marche", "pas de problème", "bonne journée", "confirmé", "réglé",
        "je vous remercie", "merci beaucoup", "c'est gentil", "avec plaisir", "volontiers"
    ]
    neutral_phrases = ["c'est bon alors", "ah d'accord", "pas de problème", "c'est réglé", "ça marche"]

    strong_neg_count = sum(1 for p in strong_negative if p in text_lower)
    mild_neg_count   = sum(1 for w in mild_negative if w in text_lower)
    positive_count   = sum(1 for w in positive_indicators if w in text_lower)
    neutral_found    = any(p in text_lower for p in neutral_phrases)

    if strong_neg_count >= 1:
        return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.90,
                "scores": {"negative": 0.90, "neutral": 0.08, "positive": 0.02}}
    if positive_count >= 3 and mild_neg_count == 0:
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

    # Voxtral pour les cas ambigus
    instruction = (
        "Analyse le sentiment de cette conversation téléphonique professionnelle.\n"
        "Réponds par UN SEUL MOT : satisfaisant, neutre, ou insatisfaisant\n\n"
        "NOTE : La plupart des appels courts et polis sont satisfaisants.\n\n"
        f"Conversation : {text[:1500]}"
    )
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
    if not speaker_sentiments:
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    if "Client" in speaker_sentiments:
        return speaker_sentiments["Client"]
    if len(ROLE_LABELS) >= 2 and ROLE_LABELS[1] in speaker_sentiments:
        return speaker_sentiments[ROLE_LABELS[1]]
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
        return speaker_sentiments.get(client_speaker, {"label_en": None, "label_fr": None, "confidence": None, "scores": None})
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

def _apply_sentiment_to_result(result: Dict[str, Any], segments: List[Dict[str, Any]]):
    """Applique l'analyse de sentiment sur les segments et enrichit le résultat."""
    if not ENABLE_SENTIMENT:
        return
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
        log(f"[SENTIMENT] Client sentiment: {label_fr} (confidence: {conf:.2f})")

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
    unwanted_starts = ["voici un résumé", "résumé de la conversation", "cette conversation",
                       "dans cette conversation", "le résumé", "il s'agit"]
    summary_lower = summary.lower().strip()
    for start in unwanted_starts:
        if summary_lower.startswith(start):
            for sentence in re.split(r'[.!?]+', summary)[1:]:
                cleaned = sentence.strip()
                if len(cleaned) > 15 and not any(cleaned.lower().startswith(u) for u in unwanted_starts):
                    summary = cleaned
                    break
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
# UTILITAIRES SEGMENTS (MONO)
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
    """Détecte la musique d'attente. Fonctionne sur mono ou stéréo (mixe en mono)."""
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

def detect_hold_music_per_channel(left_path: str, right_path: str) -> Dict[str, Any]:
    """
    Détection musique d'attente par canal pour stéréo.
    Si les DEUX canaux sont vides/musique → hold music.
    Si UN canal a de la parole → pas hold music (conversation one-sided possible).
    Retourne aussi speech_blocks par canal pour le trim.
    """
    left_result = detect_hold_music(left_path)
    right_result = detect_hold_music(right_path)

    log(f"[HOLD_MUSIC_STEREO] Left: hold={left_result['is_hold_music']}, ratio={left_result['speech_ratio']:.4f}, reason={left_result['reason']}")
    log(f"[HOLD_MUSIC_STEREO] Right: hold={right_result['is_hold_music']}, ratio={right_result['speech_ratio']:.4f}, reason={right_result['reason']}")

    # Les deux canaux sont de la musique/silence → hold music global
    if left_result["is_hold_music"] and right_result["is_hold_music"]:
        return {
            "is_hold_music": True,
            "speech_ratio": max(left_result["speech_ratio"], right_result["speech_ratio"]),
            "duration": max(left_result.get("duration", 0), right_result.get("duration", 0)),
            "reason": f"both_channels_hold (left={left_result['reason']}, right={right_result['reason']})",
            "detection_time": left_result.get("detection_time", 0) + right_result.get("detection_time", 0),
        }

    # Au moins un canal a de la parole → pas hold music
    # On retourne les speech_blocks pour un éventuel trim
    combined = {
        "is_hold_music": False,
        "speech_ratio": max(left_result["speech_ratio"], right_result["speech_ratio"]),
        "duration": max(left_result.get("duration", 0), right_result.get("duration", 0)),
        "reason": "stereo_has_speech",
        "detection_time": left_result.get("detection_time", 0) + right_result.get("detection_time", 0),
        "left_speech_blocks": left_result.get("speech_blocks"),
        "right_speech_blocks": right_result.get("speech_blocks"),
    }
    return combined

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
# DÉTECTION VOIX UNIQUE (annonces / IVR / messagerie) — MONO ONLY
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
    conv = [{"role": "user", "content": [
        {"type": "audio", "path": wav_path},
        {"type": "text", "text": f"lang:{language or 'fr'} [TRANSCRIBE] Écris exactement ce qui est dit, mot pour mot."}
    ]}]
    out       = run_voxtral_with_timeout(conv, max_new_tokens=max_new_tokens, timeout=60)
    full_text = (out.get("text") or "").strip()
    if not full_text:
        return {"error": "Empty transcription for single voice content"}

    # Détection d'une vraie conversation dans le transcript
    conversation_markers = [
        "allô ?", "allo ?", "oui, bonjour", "bonjour,", "c'est noté",
        "pas de souci", "à tout à l'heure", "au revoir", "je vous remercie",
        "d'accord", "pas de problème", "merci beaucoup"
    ]
    text_lower   = full_text.lower()
    marker_count = sum(1 for m in conversation_markers if m in text_lower)
    if marker_count >= 3:
        log(f"[SINGLE_VOICE] Detected real conversation within voicemail ({marker_count} markers) → switching to full diarization")
        return None  # Signal au caller de basculer en mode diarization

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
# PIPELINE STÉRÉO — Channel Split + VAD + Transcription séparée
# =============================================================================

def _detect_speech_turns(audio_path: str) -> List[Tuple[float, float]]:
    """
    VAD basée sur l'énergie pour détecter les tours de parole sur un canal mono.
    Retourne une liste de (start_s, end_s) pour chaque segment de parole.
    """
    import soundfile as sf
    import numpy as np

    y, sr = sf.read(audio_path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    duration = len(y) / sr
    if duration < 0.5:
        return []

    frame_length = int(0.030 * sr)
    hop_length   = int(0.010 * sr)
    n_frames     = max(1, 1 + (len(y) - frame_length) // hop_length)

    rms = np.array([np.sqrt(np.mean(y[i*hop_length:i*hop_length+frame_length]**2))
                    for i in range(n_frames)], dtype=np.float32)

    peak = np.max(np.abs(y))
    if peak < 1e-6:
        return []

    rms_norm = rms / peak
    # Seuil adaptatif : médiane + 1.5 * écart-type, minimum 0.015
    threshold = max(float(np.median(rms_norm)) + 1.5 * float(np.std(rms_norm)), 0.015)
    speech_mask = rms_norm > threshold

    frames_per_sec = sr / hop_length
    min_speech_frames = int(STEREO_VAD_MIN_SPEECH * frames_per_sec)
    min_silence_frames = int(STEREO_VAD_MIN_SILENCE * frames_per_sec)

    # Trouver les segments de parole contigus
    turns = []
    in_speech = False
    speech_start = 0

    for i in range(len(speech_mask)):
        if speech_mask[i] and not in_speech:
            speech_start = i
            in_speech = True
        elif not speech_mask[i] and in_speech:
            if i - speech_start >= min_speech_frames:
                turns.append((speech_start / frames_per_sec, i / frames_per_sec))
            in_speech = False

    if in_speech and (len(speech_mask) - speech_start) >= min_speech_frames:
        turns.append((speech_start / frames_per_sec, len(speech_mask) / frames_per_sec))

    # Fusionner les turns séparés par moins de min_silence
    if len(turns) > 1:
        merged = [list(turns[0])]
        for start, end in turns[1:]:
            if start - merged[-1][1] < STEREO_VAD_MIN_SILENCE:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        turns = [(round(s, 3), round(e, 3)) for s, e in merged]
    else:
        turns = [(round(s, 3), round(e, 3)) for s, e in turns]

    return turns


def _transcribe_channel(channel_path: str, language: Optional[str]) -> str:
    """Transcrit un canal mono avec Voxtral (transcription simple, pas de diarization)."""
    ok, err, dur = _validate_audio(channel_path)
    if not ok or dur < 0.5:
        return ""

    tokens = max(256, min(int(dur * 8), 8000))
    timeout = min(int(dur * 0.7) + 60, 600)

    instruction = (
        f"lang:{language or 'fr'} "
        "[TRANSCRIBE] Transcris exactement ce qui est dit, mot pour mot. "
        "Une seule personne parle sur cet enregistrement. "
        "Conserve les hésitations (euh, bah, alors, donc). "
        "N'écris JAMAIS [Musique] [Silence] [Attente] [Sonnerie] [Pause]."
    )

    conv = [{"role": "user", "content": [
        {"type": "audio", "path": channel_path},
        {"type": "text", "text": instruction}
    ]}]

    out = run_voxtral_with_timeout(conv, max_new_tokens=tokens, timeout=timeout)
    text = (out.get("text") or "").strip()
    text = remove_repetitive_loops(text, max_repetitions=5)
    text = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold|Sonnerie|Pause)\]\s*', '', text, flags=re.IGNORECASE).strip()
    # Nettoyer les éventuels labels de speaker que Voxtral pourrait ajouter malgré la consigne
    text = re.sub(r'^(?:Agent|Client|Speaker\s*\d*)\s*:\s*', '', text, flags=re.IGNORECASE).strip()

    log(f"[STEREO_TRANSCRIBE] Channel {channel_path}: {len(text)} chars, {dur:.1f}s, latency={out.get('latency_s', 0):.1f}s")
    return text


def _distribute_text_to_turns(turns: List[Tuple[float, float]], full_text: str, role: str) -> List[Dict[str, Any]]:
    """
    Distribue le texte transcrit sur les tours de parole détectés par le VAD.
    Répartition proportionnelle à la durée de chaque turn.
    """
    if not turns or not full_text.strip():
        return []

    words = full_text.split()
    if not words:
        return []

    total_duration = sum(e - s for s, e in turns)
    if total_duration <= 0:
        return []

    segments = []
    word_idx = 0
    total_words = len(words)

    for i, (start, end) in enumerate(turns):
        turn_duration = end - start
        if i == len(turns) - 1:
            # Dernier turn : tous les mots restants
            turn_words = words[word_idx:]
        else:
            # Proportionnel à la durée
            word_count = max(1, round(total_words * turn_duration / total_duration))
            turn_words = words[word_idx:word_idx + word_count]
            word_idx += word_count

        if turn_words:
            segments.append({
                "speaker": role,
                "start": round(start, 2),
                "end": round(end, 2),
                "text": " ".join(turn_words),
                "mood": None
            })

    return segments


def transcribe_stereo_pipeline(
    left_path: str,
    right_path: str,
    language: Optional[str],
    max_new_tokens: int,
    with_summary: bool,
    call_direction: str = "unknown"
) -> Dict[str, Any]:
    """
    Pipeline stéréo : chaque canal = un speaker.
    1. VAD par canal → tours de parole
    2. Détection voix unique / IVR / répondeur
    3. Transcription de chaque canal
    4. Interleaving des segments par timestamp
    5. Résumé + Sentiment
    """
    # ── Mapping canal → rôle ────────────────────────────────────────────────
    # Asterisk MixMonitor: left=rx (audio reçu par le canal), right=tx (audio émis)
    # Appel entrant : left = Caller (Client), right = Agent
    # Appel sortant : left = Agent (a composé), right = Client (a décroché)
    if call_direction == "outbound":
        left_role = "Agent"
        right_role = "Client"
    else:  # inbound ou unknown → left=Client, right=Agent
        left_role = "Client"
        right_role = "Agent"

    log(f"[STEREO] Pipeline: direction={call_direction}, left={left_role}, right={right_role}")

    # ── VAD par canal ───────────────────────────────────────────────────────
    left_turns = _detect_speech_turns(left_path)
    right_turns = _detect_speech_turns(right_path)

    left_speech_s = sum(e - s for s, e in left_turns)
    right_speech_s = sum(e - s for s, e in right_turns)

    _, _, left_dur = _validate_audio(left_path)
    _, _, right_dur = _validate_audio(right_path)
    total_duration = max(left_dur, right_dur)

    log(f"[STEREO] VAD: left={len(left_turns)} turns ({left_speech_s:.1f}s speech), "
        f"right={len(right_turns)} turns ({right_speech_s:.1f}s speech)")

    # ── Détection voix unique / répondeur / IVR ────────────────────────────
    # Si un canal est quasiment vide → voix unique sur l'autre canal
    left_has_speech = left_speech_s >= STEREO_MIN_SPEECH_S
    right_has_speech = right_speech_s >= STEREO_MIN_SPEECH_S

    if not left_has_speech and not right_has_speech:
        log("[STEREO] Both channels silent — no conversation")
        result = {"segments": [], "transcript": "", "audio_duration": total_duration}
        if with_summary:
            result["summary"] = "Aucune conversation détectée."
        return result

    if not left_has_speech or not right_has_speech:
        # Un seul canal a de la parole → IVR / répondeur / annonce
        active_path = right_path if not left_has_speech else left_path
        active_role = right_role if not left_has_speech else left_role
        silent_role = left_role if not left_has_speech else right_role

        log(f"[STEREO] Single voice detected on {active_role} channel (other channel={silent_role} silent)")

        if ENABLE_SINGLE_VOICE_DETECTION:
            # Tenter la transcription voix unique
            sv_result = transcribe_single_voice_content(active_path, language, max_new_tokens, with_summary)

            if sv_result is None:
                # Signal: vraie conversation détectée dans le contenu → transcrire les 2 canaux quand même
                log("[STEREO] Single voice content contains conversation markers → transcribing both channels")
            elif "error" not in sv_result:
                # Remplacer le label "Agent" par le bon rôle
                for seg in sv_result.get("segments", []):
                    seg["speaker"] = active_role
                sv_result["transcript"] = sv_result.get("transcript", "").replace("Agent:", f"{active_role}:", 1)
                sv_result["audio_duration"] = total_duration
                return sv_result

    # ── Transcription de chaque canal ──────────────────────────────────────
    log("[STEREO] Transcribing left channel...")
    left_text = _transcribe_channel(left_path, language) if left_has_speech else ""

    log("[STEREO] Transcribing right channel...")
    right_text = _transcribe_channel(right_path, language) if right_has_speech else ""

    if not left_text and not right_text:
        result = {"segments": [], "transcript": "", "audio_duration": total_duration}
        if with_summary:
            result["summary"] = "Aucune parole détectée."
        return result

    # ── Distribution du texte sur les tours VAD ────────────────────────────
    left_segments = _distribute_text_to_turns(left_turns, left_text, left_role)
    right_segments = _distribute_text_to_turns(right_turns, right_text, right_role)

    # ── Interleaving par timestamp ────────────────────────────────────────
    all_segments = left_segments + right_segments
    all_segments.sort(key=lambda s: s["start"])

    # Fusionner les segments adjacents du même speaker (gap < 0.8s)
    merged = []
    for seg in all_segments:
        if merged and merged[-1]["speaker"] == seg["speaker"] and seg["start"] - merged[-1]["end"] < 0.8:
            merged[-1]["text"] += " " + seg["text"]
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    log(f"[STEREO] Final: {len(merged)} segments ({len(left_segments)} left + {len(right_segments)} right → merged)")

    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in merged if s.get("text"))
    result = {
        "segments": merged,
        "transcript": full_transcript,
        "audio_duration": total_duration,
        "stereo_mode": True,
        "channel_mapping": {"left": left_role, "right": right_role},
    }

    # ── Résumé ─────────────────────────────────────────────────────────────
    if with_summary:
        log("[STEREO] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript, duration_seconds=total_duration)

    # ── Sentiment ──────────────────────────────────────────────────────────
    _apply_sentiment_to_result(result, merged)

    log("[STEREO] Pipeline completed successfully")
    return result


# =============================================================================
# PIPELINE MONO — VOXTRAL SPEAKER ID (existant)
# =============================================================================
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

def parse_speaker_identified_transcript(transcript: str, total_duration: float) -> List[Dict[str, Any]]:
    if not transcript:
        return []
    segments     = []
    current_time = 0.0
    lines        = transcript.split('\n')
    valid_lines  = []
    if len(lines) == 1 or (len(lines) == 2 and not lines[1].strip()):
        text    = lines[0].strip()
        pattern = r'(Agent|Client|Secrétaire|Secretaire|Patient|Médecin|Medecin|Docteur|Praticien|Appelant|AGENT|CLIENT):\s*([^:]+?)(?=\s*(?:Agent:|Client:|Secrétaire:|Secretaire:|Patient:|Médecin:|Medecin:|Docteur:|Praticien:|Appelant:|AGENT:|CLIENT:|$))'
        matches = re.findall(pattern, text, re.IGNORECASE)
        log(f"[PARSE] Inline mode: found {len(matches)} speaker segments")
        speaker_mapping: Dict[str, str] = {}
        speaker_counter = 0
        agent_variants  = {"agent", "secrétaire", "secretaire", "médecin", "medecin", "docteur", "praticien"}
        client_variants = {"client", "patient", "appelant"}
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
        log(f"[LLM_REVIEW] {'Corrected' if corrections else 'No corrections'} ({corrections} attributions)")
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
    """Pipeline mono : Voxtral identifie les speakers dans un fichier mono."""
    log(f"[VOXTRAL_ID] Starting speaker identification: language={language}, direction={call_direction}")
    ok, err, est_dur = _validate_audio(wav_path)
    if not ok:
        return {"error": err}

    speaker_tokens = max(512, min(int(est_dur * 10), 12000))
    infer_timeout = min(int(est_dur * 0.7) + 60, 600)
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
        return {"segments": [], "transcript": "", "summary": "Aucune conversation détectée."}

    original_len       = len(speaker_transcript)
    speaker_transcript = remove_repetitive_loops(speaker_transcript, max_repetitions=5)
    if original_len != len(speaker_transcript):
        log(f"[VOXTRAL_ID] Cleaned repetitive loops: {original_len} → {len(speaker_transcript)} chars")
    speaker_transcript = re.sub(r'\[(?:Musique|Silence|Music|Silent|Attente|Hold)\]\s*', '', speaker_transcript, flags=re.IGNORECASE).strip()
    log(f"[VOXTRAL_ID] Completed: {len(speaker_transcript)} chars in {out.get('latency_s', 0):.1f}s")

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

    segments = _enforce_max_two_speakers(segments)
    _map_roles(segments)
    segments = detect_and_fix_speaker_inversion(segments)

    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))
    result = {"segments": segments, "transcript": full_transcript, "audio_duration": est_dur}

    if with_summary:
        log("[VOXTRAL_ID] Generating summary...")
        result["summary"] = select_best_summary_approach(full_transcript, duration_seconds=est_dur)

    _apply_sentiment_to_result(result, segments)

    log("[VOXTRAL_ID] Voxtral speaker identification completed successfully")
    return result

# =============================================================================
# HEALTH CHECK
# =============================================================================
def health():
    info = {
        "app_version": APP_VERSION,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": _device_str(),
        "quant_mode": QUANT_MODE,
        "model_id": MODEL_ID,
        "sentiment": "Voxtral-based" if ENABLE_SENTIMENT else "Disabled",
        "single_voice_detection": ENABLE_SINGLE_VOICE_DETECTION,
        "hold_music_detection": ENABLE_HOLD_MUSIC_DETECTION,
        "detect_global_swap": DETECT_GLOBAL_SWAP,
        "stereo_enabled": STEREO_ENABLED,
        "role_labels": ROLE_LABELS,
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

    task         = (inp.get("task") or "transcribe_diarized").lower()
    language     = inp.get("language") or None
    max_new_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))
    with_summary   = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))

    # ── Détection direction de l'appel ───────────────────────────────────────
    call_direction = (inp.get("call_direction") or "").lower().strip()
    if call_direction not in ("inbound", "outbound"):
        audio_url = inp.get("audio_url") or inp.get("file_path") or ""
        filename  = audio_url.split("/")[-1].lower()
        if filename.startswith("in-") or filename.startswith("in_"):
            call_direction = "inbound"
        elif filename.startswith("out-") or filename.startswith("out_"):
            call_direction = "outbound"
        else:
            call_direction = "unknown"
    log(f"[HANDLER] Task: {task}, language: {language}, direction: {call_direction}, max_tokens: {max_new_tokens}, summary: {with_summary}")

    local_path, cleanup = None, False
    stereo_cleanup_paths = []  # Fichiers temporaires stéréo à nettoyer

    try:
        # ── Récupération de l'audio ──────────────────────────────────────────
        if inp.get("audio_url"):
            url = inp["audio_url"].strip()
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.path or parsed.path in ('', '/'):
                return {"error": f"audio_url invalide ou tronquée: '{url}'"}
            local_path = _download_to_tmp(url); cleanup = True
        elif inp.get("audio_b64"):
            local_path = _b64_to_tmp(inp["audio_b64"]); cleanup = True
        elif inp.get("file_path"):
            local_path = inp["file_path"]
        else:
            return {"error": "Provide 'audio_url', 'audio_b64' or 'file_path'."}

        # ── Validation audio ─────────────────────────────────────────────────
        ok, err, _ = _validate_audio(local_path)
        if not ok:
            return {"error": err}

        # ── Détection stéréo ──────────────────────────────────────────────────
        is_stereo = STEREO_ENABLED and _is_stereo(local_path)
        log(f"[HANDLER] Audio: {local_path}, stereo={is_stereo}")

        # ==================================================================
        # PIPELINE STÉRÉO
        # ==================================================================
        if is_stereo:
            log("[HANDLER] === STEREO PIPELINE ===")

            # 1. Détection musique d'attente par canal
            left_path, right_path = _split_stereo_channels(local_path)
            stereo_cleanup_paths.extend([left_path, right_path])

            if ENABLE_HOLD_MUSIC_DETECTION:
                hold_result = detect_hold_music_per_channel(left_path, right_path)
                if hold_result["is_hold_music"]:
                    log("[HANDLER] Hold music detected on both channels — skipping inference")
                    return {"task": task, **build_hold_music_response(hold_result, with_summary)}

            # 2. Pipeline stéréo complet
            out = transcribe_stereo_pipeline(
                left_path, right_path, language, max_new_tokens,
                with_summary, call_direction
            )
            if "error" in out:
                return out
            return {"task": task, **out}

        # ==================================================================
        # PIPELINE MONO (existant)
        # ==================================================================
        log("[HANDLER] === MONO PIPELINE ===")

        # Convertir en mono si nécessaire (normalement déjà mono ici)
        mono_path = _to_mono(local_path)
        if mono_path != local_path:
            stereo_cleanup_paths.append(mono_path)
            local_path = mono_path

        # 1. Détection musique d'attente
        if ENABLE_HOLD_MUSIC_DETECTION:
            hold_result = detect_hold_music(local_path)
            if hold_result["is_hold_music"]:
                log("[HANDLER] Hold music detected — skipping inference")
                return {"task": task, **build_hold_music_response(hold_result, with_summary)}
            if hold_result.get("speech_blocks"):
                local_path, cleanup = trim_audio_to_speech_blocks(local_path, hold_result, cleanup)

        # 2. Détection voix unique
        if ENABLE_SINGLE_VOICE_DETECTION:
            content_type = detect_single_voice_content(local_path, language)
            if content_type["type"] in ("announcement", "voicemail"):
                log(f"[HANDLER] Detected {content_type['type']}, trying single voice mode...")
                out = transcribe_single_voice_content(local_path, language, max_new_tokens, with_summary)
                if out is None:
                    log("[HANDLER] Real conversation found in voicemail → switching to full diarization")
                elif "error" not in out:
                    return {"task": task, **out}

        # 3. Pipeline mono : Voxtral Speaker ID
        log("[HANDLER] Using Voxtral Speaker Identification mode (mono)")
        out = diarize_with_voxtral_speaker_id(local_path, language, max_new_tokens, with_summary, call_direction)
        if "error" in out:
            return out
        return {"task": task, **out}

    except Exception as e:
        log(f"[HANDLER] CRITICAL ERROR: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        try:
            _gpu_clear()
        except Exception:
            pass
        # Nettoyage fichiers temporaires
        for p in stereo_cleanup_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
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
    log(f"[INIT] STEREO_ENABLED={STEREO_ENABLED}")
    if not check_gpu_memory():
        log(f"[CRITICAL] Insufficient GPU memory (QUANT_MODE={QUANT_MODE}) — skipping preload")
    else:
        log("[INIT] Preloading Voxtral...")
        load_voxtral()
        log("[INIT] Preload completed successfully")

    log(f"[INIT] Hold music detection: {'ENABLED' if ENABLE_HOLD_MUSIC_DETECTION else 'DISABLED'}")
    log(f"[INIT] Single voice detection: {'ENABLED' if ENABLE_SINGLE_VOICE_DETECTION else 'DISABLED'}")
    log(f"[INIT] Sentiment analysis: {'ENABLED' if ENABLE_SENTIMENT else 'DISABLED'}")
    log(f"[INIT] Stereo pipeline: {'ENABLED' if STEREO_ENABLED else 'DISABLED'}")
    log(f"[INIT] LLM review (DETECT_GLOBAL_SWAP): {'ENABLED' if DETECT_GLOBAL_SWAP else 'DISABLED'}")

except Exception as e:
    log(f"[WARN] Preload failed — will load on first request: {e}")
    _processor = None
    _model     = None

runpod.serverless.start({"handler": handler})
