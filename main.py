import os, time, base64, tempfile, uuid, requests, json, traceback
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline as hf_pipeline
)

# Try Voxtral-specific class
try:
    from transformers import VoxtralForConditionalGeneration  # type: ignore
    _HAS_VOXTRAL_CLASS = True
except Exception:
    VoxtralForConditionalGeneration = None  # type: ignore
    _HAS_VOXTRAL_CLASS = False

from pydub import AudioSegment
from pyannote.audio import Pipeline

import runpod

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print(msg, flush=True)

# Env
APP_VERSION = os.environ.get("APP_VERSION", "unknown")

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-3B-2507").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "1200"))
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli").strip()
SENTIMENT_TYPE = os.environ.get("SENTIMENT_TYPE", "zero-shot").strip().lower()  # "zero-shot" or "classifier"
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"
SENTIMENT_DEVICE = int(os.environ.get("SENTIMENT_DEVICE", "-1"))  # -1 = CPU (default to avoid NVRTC issues)

# Globals
_processor = None
_model = None
_diarizer = None
_sentiment_clf = None       # TextClassificationPipeline (if classifier)
_sentiment_zero_shot = None # zero-shot pipeline

def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _device_idx() -> int:
    return 0 if torch.cuda.is_available() else -1

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

# Voxtral
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    dtype = _select_dtype()
    log(f"[INIT] Loading Voxtral: {MODEL_ID} dtype={dtype}")

    proc_kwargs = {}
    if HF_TOKEN:
        proc_kwargs["token"] = HF_TOKEN
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **proc_kwargs)

    mdl_kwargs = {"dtype": dtype}
    if torch.cuda.is_available():
        mdl_kwargs["device_map"] = {"": 0}
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    if _HAS_VOXTRAL_CLASS:
        _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **mdl_kwargs)
    else:
        raise RuntimeError("Transformers build without VoxtralForConditionalGeneration. Update transformers@main.")

    log("[INIT] Voxtral ready.")
    return _processor, _model

def _build_conv_transcribe(local_path: str, language: Optional[str]) -> List[Dict[str, Any]]:
    lang_prefix = f"lang:{language} " if language else ""
    instruction = f"{lang_prefix}[TRANSCRIBE]"
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": local_path},
            {"type": "text", "text": instruction},
        ],
    }]

def _build_conv_summary(local_path: str, language: Optional[str], max_sentences: Optional[int], style: Optional[str]) -> List[Dict[str, Any]]:
    parts = []
    if language:
        parts.append(f"lang:{language}")
    if style:
        parts.append(style.strip())
    if max_sentences:
        parts.append(f"Résumé en {max_sentences} phrases.")
    else:
        parts.append("Fais un résumé concis et structuré.")
    instruction = " ".join(parts)
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": local_path},
            {"type": "text", "text": instruction},
        ],
    }]

def run_voxtral(conversation: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
    processor, model = load_voxtral()
    inputs = processor.apply_chat_template(conversation)
    dtype = _select_dtype()
    device = _device_str()
    inputs = inputs.to(device, dtype=dtype)

    gen_kwargs = dict(max_new_tokens=max_new_tokens, temperature=TEMPERATURE, top_p=TOP_P)
    start = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": decoded[0] if decoded else "", "latency_s": round(elapsed, 3)}

# Diarization
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

# Sentiment
def load_sentiment():
    global _sentiment_clf, _sentiment_zero_shot
    if not ENABLE_SENTIMENT:
        return None

    # Force CPU by default to avoid NVRTC / TorchScript issues.
    device_idx = SENTIMENT_DEVICE

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

# Core flow
def diarize_then_transcribe(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s). Increase MAX_DURATION_S or send shorter file."}
    except Exception:
        pass

    dia = load_diarizer()
    diarization = dia(wav_path)
    audio = AudioSegment.from_wav(wav_path)

    segments, transcript_parts = [], []
    weighted_moods, mood_by_speaker_weights = [], {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_s, end_s = float(turn.start), float(turn.end)
        dur_s = max(0.0, end_s - start_s)
        seg_audio = audio[int(start_s*1000): int(end_s*1000)]
        tmp = os.path.join(tempfile.gettempdir(), f"seg_{speaker}_{int(start_s*1000)}.wav")
        seg_audio.export(tmp, format="wav")

        conv = _build_conv_transcribe(tmp, language)
        out = run_voxtral(conv, max_new_tokens=max_new_tokens)
        text = (out.get("text") or "").strip()
        mood = classify_sentiment(text)

        segments.append({"speaker": speaker, "start": start_s, "end": end_s, "text": text, "mood": mood})
        if text:
            transcript_parts.append(f"{speaker}: {text}")
        if dur_s > 0:
            weighted_moods.append((mood, dur_s))
            mood_by_speaker_weights.setdefault(speaker, []).append((mood, dur_s))

        try:
            os.remove(tmp)
        except Exception:
            pass

    result = {"segments": segments, "transcript": "\n".join(transcript_parts)}
    if with_summary:
        conv_sum = _build_conv_summary(wav_path, language, max_sentences=None, style="Résumé clair en français, à puces si pertinent.")
        s = run_voxtral(conv_sum, max_new_tokens=max_new_tokens)
        result["summary"] = s.get("text", "")

    result["mood_overall"] = aggregate_mood(weighted_moods)
    result["mood_by_speaker"] = {spk: aggregate_mood(lst) for spk, lst in mood_by_speaker_weights.items()}
    return result

# Health / diagnostics
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
        "sentiment_device": "cpu" if SENTIMENT_DEVICE == -1 else f"cuda:{SENTIMENT_DEVICE}"
    }
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

# Handler
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
            conv = _build_conv_summary(local_path, language, inp.get("max_sentences"), inp.get("style"))
            out = run_voxtral(conv, max_new_tokens=max_new_tokens)
            return {"task": "summary", **out}
        else:
            conv = _build_conv_transcribe(local_path, language)
            out = run_voxtral(conv, max_new_tokens=max_new_tokens)
            return {"task": "transcribe", **out}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        try:
            if 'cleanup' in locals() and cleanup and local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

# Warm
try:
    load_voxtral()
    load_diarizer()
    if ENABLE_SENTIMENT:
        load_sentiment()
except Exception as e:
    log(f"[WARN] Deferred load: {e}")

runpod.serverless.start({"handler": handler})
