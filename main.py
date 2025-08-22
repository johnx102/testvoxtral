import os
import time
import base64
import tempfile
import uuid
import requests
from typing import Optional, List, Dict, Any

import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor

from pydub import AudioSegment
from pyannote.audio import Pipeline

import runpod

# ---------------------------
# Configuration via env vars
# ---------------------------
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-3B-2507").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "1200"))
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Globals
_processor = None
_model = None
_diarizer = None

# ---------------------------
# Utils
# ---------------------------
def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        # Ampere (8.x) or newer => bfloat16 is OK
        if major >= 8:
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32

def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    suffix = ".wav"
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}{suffix}")
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
# Voxtral Loader/Runner
# ---------------------------
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    dtype = _select_dtype()
    print(f"[INIT] Loading Voxtral model={MODEL_ID} dtype={dtype}", flush=True)

    # Processor
    _processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN or None)

    # Model
    kwargs = {"dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    print("[INIT] Voxtral ready.", flush=True)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device, dtype=dtype)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P
    )
    start = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    # Decode only the newly generated part (skip the prompt)
    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {"text": decoded[0] if decoded else "", "latency_s": round(elapsed, 3)}

# ---------------------------
# Diarization
# ---------------------------
def load_diarizer():
    global _diarizer
    if _diarizer is not None:
        return _diarizer
    print(f"[INIT] Loading diarization pipeline: {DIAR_MODEL}", flush=True)
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN
    _diarizer = Pipeline.from_pretrained(DIAR_MODEL, **kwargs)
    if torch.cuda.is_available():
        _diarizer.to("cuda")
    print("[INIT] Diarizer ready.", flush=True)
    return _diarizer

def diarize_then_transcribe(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    # Optional safety: reject very long audio to avoid timeouts
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

    # Load full audio once
    audio = AudioSegment.from_wav(wav_path)

    segments = []
    transcript_parts = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_s = float(turn.start)
        end_s = float(turn.end)
        seg_audio = audio[int(start_s * 1000): int(end_s * 1000)]

        tmp = os.path.join(tempfile.gettempdir(), f"seg_{speaker}_{int(start_s*1000)}.wav")
        seg_audio.export(tmp, format="wav")

        conv = _build_conv_transcribe(tmp, language)
        out = run_voxtral(conv, max_new_tokens=max_new_tokens)
        text = (out.get("text") or "").strip()

        segments.append({
            "speaker": speaker,
            "start": start_s,
            "end": end_s,
            "text": text
        })
        transcript_parts.append(f"{speaker}: {text}")

        try:
            os.remove(tmp)
        except Exception:
            pass

    full_transcript = "\n".join(transcript_parts)

    result = {
        "segments": segments,
        "transcript": full_transcript
    }

    if with_summary:
        conv_sum = _build_conv_summary(wav_path, language, max_sentences=None, style="Résumé clair en français, à puces si pertinent.")
        s = run_voxtral(conv_sum, max_new_tokens=max_new_tokens)
        result["summary"] = s.get("text", "")

    return result

# ---------------------------
# RunPod Handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input", {}) or {}

    if inp.get("ping"):
        return {"pong": True, "model_id": MODEL_ID, "diar_model": DIAR_MODEL}

    task = (inp.get("task") or "transcribe").lower()
    language = inp.get("language") or None
    max_new_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))

    # For diarized task
    with_summary = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))

    # Input audio
    local_path = None
    cleanup = False
    try:
        if inp.get("audio_url"):
            local_path = _download_to_tmp(inp["audio_url"])
            cleanup = True
        elif inp.get("audio_b64"):
            local_path = _b64_to_tmp(inp["audio_b64"])
            cleanup = True
        elif inp.get("file_path"):
            local_path = inp["file_path"]
        else:
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

        else:  # "transcribe"
            conv = _build_conv_transcribe(local_path, language)
            out = run_voxtral(conv, max_new_tokens=max_new_tokens)
            return {"task": "transcribe", **out}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if cleanup and local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

# Warm models at cold start
try:
    load_voxtral()
    load_diarizer()
except Exception as e:
    print(f"[WARN] Deferred load: {e}", flush=True)

runpod.serverless.start({"handler": handler})
