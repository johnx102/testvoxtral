import re
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod Serverless — Voxtral + PyAnnote (strict 2 speakers) + résumé concis + humeur
Base: dernière version fournie par l'utilisateur (fonctionnelle), avec ces améliorations :
- Diarization stricte à 2 locuteurs via pyannote (exact_two par défaut).
- Transcription STRICTE par segment avec le mode transcription natif de Voxtral (pas de chat), donc pas d'interprétation.
- Résumé concis (2–3 phrases) construit de façon extractive à partir du transcript (sans LLM), pour éviter les dérapages.
- Conserve le pipeline d'humeur existant.
"""

import os, time, base64, tempfile, uuid, requests, json, traceback, io, re
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline,
    pipeline as hf_pipeline
)

# Voxtral class (requires transformers@main)
try:
    from transformers import VoxtralForConditionalGeneration  # type: ignore
    _HAS_VOXTRAL_CLASS = True
except Exception:
    VoxtralForConditionalGeneration = None  # type: ignore
    _HAS_VOXTRAL_CLASS = False

from pydub import AudioSegment
from pyannote.audio import Pipeline
import runpod
import soundfile as sf
import numpy as np

# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print(msg, flush=True)

# ---------------------------
# Env / Config
# ---------------------------
APP_VERSION = os.environ.get("APP_VERSION", "strict-2spk-2025-08-25")

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-3B-2507").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "2400"))
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Sentiment (CPU by default to avoid NVRTC issues)
SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli").strip()
SENTIMENT_TYPE = os.environ.get("SENTIMENT_TYPE", "zero-shot").strip().lower()  # "zero-shot" or "classifier"
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"
SENTIMENT_DEVICE = int(os.environ.get("SENTIMENT_DEVICE", "-1"))  # -1 = CPU

# Diarization speaker limit
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "2"))  # hard cap in diarization call
EXACT_TWO = os.environ.get("EXACT_TWO", "1") == "1"       # force exactly 2 by défaut

# Behavior toggles
STRICT_TRANSCRIPTION = os.environ.get("STRICT_TRANSCRIPTION", "1") == "1"
ROLE_LABELS = [r.strip() for r in os.environ.get("ROLE_LABELS", "Agent,Client").split(",") if r.strip()]

# Globals
_processor = None
_model = None
_diarizer = None
_sentiment_clf = None
_sentiment_zero_shot = None

# ---------------------------
# Helpers
# ---------------------------
def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

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

# ---------------------------
# Voxtral
# ---------------------------
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    dtype = _select_dtype()
    log(f"[INIT] Loading Voxtral: {MODEL_ID} torch_dtype={dtype}")

    proc_kwargs = {}
    if HF_TOKEN:
        proc_kwargs["token"] = HF_TOKEN
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **proc_kwargs)

    mdl_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    if _HAS_VOXTRAL_CLASS:
        _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **mdl_kwargs)
    else:
        raise RuntimeError("Transformers build without VoxtralForConditionalGeneration. Update transformers@main.")

    # Fallback: ensure CUDA
    try:
        if torch.cuda.is_available():
            _model.to(torch.device("cuda"))
    except Exception as e:
        log(f"[WARN] Could not move Voxtral to CUDA explicitly: {e}")

    # Log device/dtype
    try:
        p = next(_model.parameters())
        log(f"[INIT] Voxtral device={p.device}, dtype={p.dtype}")
    except Exception:
        pass

    log("[INIT] Voxtral ready.")
    return _processor, _model

def run_voxtral_chat(conversation, max_new_tokens: int) -> Dict[str, Any]:
    """Ancien mode chat (génératif). Gardé pour résumés optionnels si besoin."""
    processor, model = load_voxtral()
    inputs = processor.apply_chat_template(conversation)
    dtype = _select_dtype()
    device = _device_str()
    inputs = inputs.to(device, dtype=dtype)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=TEMPERATURE, top_p=TOP_P, do_sample=(TEMPERATURE>0.0))
    start = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": decoded[0] if decoded else "", "latency_s": round(elapsed, 3)}

def vox_strict_transcribe_file(wav_path: str, language: Optional[str]) -> str:
    """
    Utilise le mode transcription natif de Voxtral pour un chunk audio.
    Zéro sampling, pas d'interprétation. Retourne une chaîne brute.
    """
    processor, model = load_voxtral()
    # lire audio
    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    # certains conteneurs peuvent avoir d'autres SR; Voxtral peut gérer, sinon pydub peut rééchantillonner en amont
    inputs = processor.apply_transcription_request(
        language=language or "auto",
        audio=data,
        model_id=MODEL_ID,
        sampling_rate=sr,
    )
    # déplacer sur device
    todev = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            todev[k] = v.to(_device_str(), dtype=_select_dtype())
        else:
            todev[k] = v
    with torch.no_grad():
        out = _model.generate(
            **todev,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )
    # décoder uniquement la continuation
    start_idx = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    txt = _processor.batch_decode(out[:, start_idx:], skip_special_tokens=True)[0]
    return (txt or "").strip()

# ---------------------------
# Diarization (limit to max 2 speakers)
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
# Sentiment (CPU-friendly by default)
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
# Extractive concise summary (2–3 sentences)
# ---------------------------
FILLERS = {"bonjour","bonsoir","au revoir","bonne journée","bonne soirée","bon week-end","merci","merci beaucoup","je vous en prie"}

def _clean_t(t: str) -> str:
    return re.sub(r'\s+', ' ', (t or '').strip())

def _sig_utterances(segments: List[Dict[str, Any]]):
    return [s for s in segments if _clean_t(s.get("text","")).lower() not in FILLERS and len(_clean_t(s.get("text","")))>=3]

def concise_summary_from_segments(segments: List[Dict[str, Any]]) -> str:
    segs = _sig_utterances(segments)
    if not segs:
        return "Résumé indisponible."
    # Intent (client)
    intent = next((s for s in segs if s["speaker"].lower().startswith("client")), None)
    # First meaningful agent reply
    reply = next((s for s in segs if s["speaker"].lower().startswith("agent")), None)
    # Outcome (last meaningful)
    outcome = next((s for s in reversed(segs) if _clean_t(s.get("text")) and s["speaker"].lower().startswith(("agent","client"))), None)

    parts = []
    if intent:
        parts.append(f"Le client dit : « { _clean_t(intent['text']) } ».")
    if reply and (not outcome or reply is not outcome):
        parts.append(f"L'agent répond/propose : « { _clean_t(reply['text']) } ».")
    if outcome:
        parts.append(f"Conclusion : « { _clean_t(outcome['text']) } ».")

    # cap to 3 sentences max; if only one, keep it.
    return " ".join(parts[:3])

# ---------------------------
# Enforce MAX 2 speakers post-process (safety net)
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

# ---------------------------
# Optional: map speakers to role labels
# ---------------------------
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

# ---------------------------
# Core flow
# ---------------------------
def diarize_then_transcribe(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    # Duration guard
    try:
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s). Increase MAX_DURATION_S or send shorter file."}
    except Exception:
        pass

    dia = load_diarizer()

    # Call diarizer with constraints to keep exactly 2 speakers when requested
    try:
        if EXACT_TWO:
            diarization = dia(wav_path, num_speakers=min(2, MAX_SPEAKERS))
        else:
            diarization = dia(wav_path, min_speakers=1, max_speakers=min(2, MAX_SPEAKERS))
    except TypeError:
        # Different signatures across versions
        if EXACT_TWO:
            diarization = dia(wav_path, min_speakers=min(2, MAX_SPEAKERS), max_speakers=min(2, MAX_SPEAKERS))
        else:
            diarization = dia(wav_path, num_speakers=min(2, MAX_SPEAKERS))

    audio = AudioSegment.from_wav(wav_path)

    segments = []
    transcript_parts = []

    weighted_moods = []
    mood_by_speaker_weights = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_s = float(turn.start)
        end_s = float(turn.end)
        dur_s = max(0.0, end_s - start_s)

        # exporter segment vers wav temporaire
        seg_audio = audio[int(start_s * 1000): int(end_s * 1000)]
        tmp = os.path.join(tempfile.gettempdir(), f"seg_{speaker}_{int(start_s*1000)}.wav")
        seg_audio.export(tmp, format="wav")

        # *** TRANSCRIPTION STRICTE ***
        try:
            text = vox_strict_transcribe_file(tmp, language)
        except Exception as e:
            text = ""

        mood = classify_sentiment(text) if ENABLE_SENTIMENT else {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

        seg = {"speaker": speaker, "start": start_s, "end": end_s, "text": text, "mood": mood}
        segments.append(seg)

        if text:
            transcript_parts.append(f"{speaker}: {text}")

        if ENABLE_SENTIMENT and dur_s > 0:
            weighted_moods.append((mood, dur_s))
            mood_by_speaker_weights.setdefault(speaker, []).append((mood, dur_s))

        try:
            os.remove(tmp)
        except Exception:
            pass

    # Safety net: enforce max 2 speakers
    segments = _enforce_max_two_speakers(segments)

    # Optional role mapping
    _map_roles(segments)

    # Rebuild transcript with mapped speaker labels
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))

    result = {"segments": segments, "transcript": full_transcript}

    if with_summary:
        result["summary"] = concise_summary_from_segments(segments)

    if ENABLE_SENTIMENT:
        result["mood_overall"] = aggregate_mood(weighted_moods)
        # Aggregate by (mapped) speaker label after enforcement
        per_role = {}
        for seg in segments:
            d = float(seg["end"]) - float(seg["start"])
            m = seg.get("mood")
            r = seg["speaker"]
            if not m or not m.get("scores"):
                continue
            per_role.setdefault(r, []).append((m, d))
        result["mood_by_speaker"] = {r: aggregate_mood(lst) for r, lst in per_role.items()}

    return result

# ---------------------------
# Health / diagnostics
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
        "role_labels": ROLE_LABELS,
    }
    try:
        if _model is not None:
            p = next(_model.parameters())
            info["voxtral_device"] = str(p.device)
            info["voxtral_dtype"]  = str(p.dtype)
        else:
            info["voxtral_device"] = None
            info["voxtral_dtype"]  = None
    except Exception:
        info["voxtral_device"] = "unknown"
        info["voxtral_dtype"]  = "unknown"

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
        inp = job.get("input", {}) or {}

        if inp.get("ping"):
            return {"pong": True}

        if inp.get("task") == "health":
            return health()

        task = (inp.get("task") or "transcribe").lower()
        language = inp.get("language") or None
        max_new_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))
        with_summary = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))

        local_path, cleanup = None, False
        if inp.get("audio_url"):
            local_path = _download_to_tmp(inp["audio_url"]); cleanup = True
        elif inp.get("audio_b64"):
            local_path = _b64_to_tmp(inp["audio_b64"]); cleanup = True
        elif inp.get("file_path"):
            local_path = inp["file_path"]
        elif task not in ("health",):
            return {"error": "Provide 'audio_url', 'audio_b64' or 'file_path'."}

        if task in ("transcribe_diarized", "diarized", "diarize"):
            out = diarize_then_transcribe(local_path, language, max_new_tokens, with_summary)
            if "error" in out:
                return out
            return {"task": "transcribe_diarized", **out}
        elif task in ("summary", "summarize"):
            # Résumé concis basé sur transcript global (transcription du fichier complet, sans diarization)
            # Pour rester simple et robuste, on applique la transcription stricte une fois sur tout le fichier.
            try:
                text_full = vox_strict_transcribe_file(local_path, language)
            except Exception as e:
                text_full = ""
            seg = [{"speaker": "Global", "start": 0.0, "end": 0.0, "text": text_full}]
            return {"task": "summary", "summary": concise_summary_from_segments(seg), "transcript": text_full}
        else:
            # Transcription stricte du fichier complet sans diarization
            text_full = vox_strict_transcribe_file(local_path, language)
            return {"task": "transcribe", "transcript": text_full}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        try:
            if 'cleanup' in locals() and cleanup and local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

# Warm load
try:
    load_voxtral()
    load_diarizer()
    if ENABLE_SENTIMENT:
        load_sentiment()
except Exception as e:
    log(f"[WARN] Deferred load: {e}")

runpod.serverless.start({"handler": handler})
