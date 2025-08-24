#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod serverless — Voxtral (transcribe mode) + PyAnnote diarization + résumé concis + humeur
- Transcript STRICT: génération greedy (do_sample=False), construit uniquement depuis l'audio de chaque segment diarizé.
- Diarization: pyannote/speaker-diarization-3.1 (HF_TOKEN requis), limité à 2 locuteurs (Agent/Client).
- Résumé: 2–3 phrases, fidèle, généré par Voxtral en mode texte (greedy).
- Humeur: zero-shot (positive / neutral / negative) overall + par locuteur.
"""

import os
import io
import sys
import time
import json
import base64
import logging
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import soundfile as sf

# =========================
#  Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("voxrun")

# =========================
#  Env
# =========================
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
VOXTRAL_MODEL_ID = os.getenv("VOXTRAL_MODEL_ID", "mistralai/Voxtral-Mini-3B-2507")
DIAR_MODEL_ID = os.getenv("DIAR_MODEL_ID", "pyannote/speaker-diarization-3.1")
SENTIMENT_MODEL_ID = os.getenv("SENTIMENT_MODEL_ID", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "2"))  # limit to 2 as requested
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Make TF32 optional but not forced (pyannote warns when disabled)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# =========================
#  Transformers (Voxtral)
# =========================
from transformers import (
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
    pipeline as hf_pipeline,
)

log.info("[INIT] Loading Voxtral: %s dtype=%s device=%s", VOXTRAL_MODEL_ID, DTYPE, DEVICE)
_vox_processor = VoxtralProcessor.from_pretrained(VOXTRAL_MODEL_ID)
_vox_model = VoxtralForConditionalGeneration.from_pretrained(
    VOXTRAL_MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
)
_vox_model.eval()

# =========================
#  Sentiment
# =========================
log.info("[INIT] Loading zero-shot sentiment on device %s: %s", ("0" if DEVICE=="cuda" else "-1"), SENTIMENT_MODEL_ID)
try:
    _clf = hf_pipeline(
        "zero-shot-classification",
        model=SENTIMENT_MODEL_ID,
        device=0 if DEVICE=="cuda" else -1,
    )
    log.info("[INIT] Zero-shot sentiment ready.")
except Exception as e:
    _clf = None
    log.warning("[INIT] Sentiment disabled: %s", e)

# =========================
#  Diarization (pyannote)
# =========================
from pyannote.audio import Pipeline as PyanPipeline

_diarizer = None
def load_diarizer():
    global _diarizer
    if _diarizer is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN manquant pour pyannote.")
        log.info("[INIT] Loading diarizer: %s", DIAR_MODEL_ID)
        _diarizer = PyanPipeline.from_pretrained(DIAR_MODEL_ID, use_auth_token=HF_TOKEN)
        _diarizer.to(torch.device(DEVICE))  # IMPORTANT: torch.device
        log.info("[INIT] Diarizer moved to %s.", DEVICE.upper())
    return _diarizer

# =========================
#  Utils
# =========================
def _load_audio_from_event(event_input: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """Return mono float32 waveform and sample_rate"""
    if "audio_b64" in event_input and event_input["audio_b64"]:
        byts = base64.b64decode(event_input["audio_b64"])
        data, sr = sf.read(io.BytesIO(byts), dtype="float32", always_2d=False)
    elif "audio_url" in event_input and event_input["audio_url"]:
        import requests
        url = event_input["audio_url"]
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
    elif "file_path" in event_input and event_input["file_path"]:
        data, sr = sf.read(event_input["file_path"], dtype="float32", always_2d=False)
    else:
        raise ValueError("Provide 'audio_url', 'audio_b64' or 'file_path'.")
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Resample to 16k for safety
    target_sr = 16000
    if sr != target_sr:
        import resampy
        data = resampy.resample(data, sr, target_sr)
        sr = target_sr
    return data, sr

def _vox_transcribe_chunk(wave_mono_f32: np.ndarray, sr: int, language: str | None) -> str:
    """Strict transcription mode: greedy decode (no sampling/beams=1)."""
    inputs = _vox_processor.apply_transcription_request(
        language=language or "auto",
        audio=wave_mono_f32,
        model_id=VOXTRAL_MODEL_ID,
        sampling_rate=sr,
    )
    inputs = {k: (v.to(DEVICE, dtype=DTYPE) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = _vox_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )
    text = _vox_processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return text.strip()

def _summarize_concise(transcript_text: str, max_new_tokens: int = 160) -> str:
    """2–3 sentences, faithful, greedy"""
    if not transcript_text.strip():
        return ""
    prompt = (
        "Tu es un assistant qui résume fidèlement un appel téléphonique en français. "
        "Rédige un résumé concis (2–3 phrases), sans listes ni citations, et n’invente rien.\n\n"
        f"Transcription:\n{transcript_text}\n\nRésumé:"
    )
    tok = _vox_processor.tokenizer
    encoded = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = _vox_model.generate(
            input_ids=encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    gen = tok.batch_decode(out[:, encoded.shape[1]:], skip_special_tokens=True)[0]
    return gen.strip()

def _classify_mood_fr(text: str) -> Dict[str, Any]:
    if not _clf or not text.strip():
        return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.0, "scores": {"negative":0.0,"neutral":1.0,"positive":0.0}}
    res = _clf(text, candidate_labels=["positive","neutral","negative"], multi_label=False)
    if isinstance(res, list):
        res = res[0]
    labels = res["labels"]
    scores = res["scores"]
    mapping = {"positive":"bon","neutral":"neutre","negative":"mauvais"}
    d = dict(zip(labels, scores))
    best = labels[0]
    return {
        "label_en": best,
        "label_fr": mapping.get(best, best),
        "confidence": float(d.get(best, 0.0)),
        "scores": {k: float(d.get(k, 0.0)) for k in ["negative","neutral","positive"]},
    }

# =========================
#  Core pipeline
# =========================
def transcribe_diarized_strict(event_input: Dict[str, Any]) -> Dict[str, Any]:
    language = event_input.get("language") or None
    with_summary = bool(event_input.get("with_summary", True))
    max_new_tokens = int(event_input.get("max_new_tokens", 160))

    wav, sr = _load_audio_from_event(event_input)

    # Diarize
    diar = load_diarizer()
    diarization = diar({"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sr})

    # Convert diarization to segments list [{"start":..., "end":..., "speaker":"SPEAKER_00"}...]
    segments = []
    try:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
    except Exception:
        segments = list(diarization.get("segments", []))

    segments.sort(key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))

    # Limit speakers to MAX_SPEAKERS (2): keep first encountered
    order = []
    for s in segments:
        sp = s["speaker"]
        if sp not in order:
            order.append(sp)
        if len(order) >= MAX_SPEAKERS:
            break
    if not order:
        order = ["SPEAKER_00","SPEAKER_01"]
    while len(order) < MAX_SPEAKERS:
        order.append(f"SPEAKER_{len(order):02d}")

    # Map to Agent/Client
    role_for = {order[0]: "Agent", order[1]: "Client"} if len(order) >= 2 else {order[0]: "Agent"}
    default_role = "Agent"

    # Transcribe each segment (strict)
    PAD = 0.15
    MIN_DUR = 0.6

    out_segments: List[Dict[str, Any]] = []
    lines: List[str] = []

    n = len(wav)
    for seg in segments:
        spk = seg["speaker"]
        if spk not in role_for:
            continue
        start = float(seg["start"]); end = float(seg["end"])
        dur = end - start
        if dur < MIN_DUR:
            continue
        s = max(0.0, start - PAD)
        e = min(n / sr, end + PAD)
        s_idx = int(s * sr); e_idx = int(e * sr)
        chunk = wav[s_idx:e_idx]
        txt = _vox_transcribe_chunk(chunk, sr, language)
        role = role_for.get(spk, default_role)
        if txt:
            out_segments.append({"start": start, "end": end, "speaker": role, "text": txt})
            lines.append(f"{role}: {txt}")

    transcript = "\n".join(lines).strip()
    summary = _summarize_concise(transcript, max_new_tokens=max_new_tokens) if with_summary else ""

    # Mood
    mood_overall = _classify_mood_fr(transcript)
    mood_by_speaker = {}
    by_role_text = {}
    for seg in out_segments:
        by_role_text.setdefault(seg["speaker"], []).append(seg["text"])
    for role, texts in by_role_text.items():
        mood_by_speaker[role] = _classify_mood_fr(" ".join(texts))

    return {
        "task": "transcribe_diarized",
        "segments": out_segments,
        "transcript": transcript,
        "summary": summary,
        "mood_overall": mood_overall,
        "mood_by_speaker": mood_by_speaker,
    }

# =========================
#  RunPod handler
# =========================
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        inp = event.get("input") or {}
        task = (inp.get("task") or "transcribe_diarized").strip().lower()

        if task in ("health","status","ping"):
            info = {
                "ok": True,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device": DEVICE,
                "model_id": VOXTRAL_MODEL_ID,
                "diar_model": DIAR_MODEL_ID,
                "sentiment_model": SENTIMENT_MODEL_ID,
                "hf_token_present": bool(HF_TOKEN),
                "transformers_has_voxtral": True,
            }
            return {"status": "OK", "output": info, "executionTime": int((time.time() - t0) * 1000)}

        if task == "transcribe_diarized":
            out = transcribe_diarized_strict(inp)
            return {"status": "COMPLETED", "output": out, "executionTime": int((time.time() - t0) * 1000)}

        return {"status": "FAILED", "error": f"Unknown task '{task}'", "executionTime": int((time.time() - t0) * 1000)}
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        log.error("Handler error: %s", trace)
        return {"status": "FAILED", "error": f"{type(e).__name__}: {e}", "output": {"trace": trace}, "executionTime": int((time.time() - t0) * 1000)}

# For local testing
if __name__ == "__main__":
    ex = {"input": {"task": "health"}}
    print(json.dumps(handler(ex), ensure_ascii=False, indent=2))
