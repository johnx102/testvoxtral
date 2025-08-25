#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod — Voxtral (transcribe strict) + PyAnnote diarization stricte (exactement 2 locuteurs) + résumé concis + humeur.
- Transcript strict (greedy, sans sampling), construit uniquement à partir des segments diarizés.
- Diarization PyAnnote 3.1 forcée à 2 locuteurs (min_speakers=2, max_speakers=2) → rôles Agent/Client.
- Résumé concis (2–3 phrases) fidèle au transcript.
- Humeur globale + par locuteur (positive / neutral / negative).
"""

import os, io, time, json, base64
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
import soundfile as sf

# --------- ENV ---------
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
VOXTRAL_MODEL_ID = os.getenv("VOXTRAL_MODEL_ID", "mistralai/Voxtral-Mini-3B-2507")
DIAR_MODEL_ID = os.getenv("DIAR_MODEL_ID", "pyannote/speaker-diarization-3.1")
SENTIMENT_MODEL_ID = os.getenv("SENTIMENT_MODEL_ID", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "2400"))  # 40 minutes hard cap
MAX_SPEAKERS = 2  # strict

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# --------- Transformers (Voxtral) ---------
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor, pipeline as hf_pipeline

_vox_processor = VoxtralProcessor.from_pretrained(VOXTRAL_MODEL_ID)
_vox_model = VoxtralForConditionalGeneration.from_pretrained(VOXTRAL_MODEL_ID, torch_dtype=DTYPE)
_vox_model.to(DEVICE).eval()

# --------- Sentiment ---------
try:
    _clf = hf_pipeline("zero-shot-classification", model=SENTIMENT_MODEL_ID, device=0 if DEVICE=="cuda" else -1)
except Exception:
    _clf = None

# --------- Diarization (pyannote) ---------
from pyannote.audio import Pipeline as PyanPipeline
_diarizer = None
def load_diarizer():
    global _diarizer
    if _diarizer is None:
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN manquant pour pyannote.")
        _diarizer = PyanPipeline.from_pretrained(DIAR_MODEL_ID, use_auth_token=HF_TOKEN)
        _diarizer.to(torch.device(DEVICE))
    return _diarizer

# --------- Utils ---------
def _resample_ta(wave: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    import torchaudio.functional as AF
    wav_t = torch.from_numpy(wave).float().unsqueeze(0)  # (1, n)
    with torch.no_grad():
        out = AF.resample(wav_t, orig_freq=sr, new_freq=target_sr)
    return out.squeeze(0).numpy(), target_sr

def _load_audio_from_event(event_input: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    if "audio_b64" in event_input and event_input["audio_b64"]:
        byts = base64.b64decode(event_input["audio_b64"])
        data, sr = sf.read(io.BytesIO(byts), dtype="float32", always_2d=False)
    elif "audio_url" in event_input and event_input["audio_url"]:
        import requests
        r = requests.get(event_input["audio_url"], timeout=90)
        r.raise_for_status()
        data, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
    elif "file_path" in event_input and event_input["file_path"]:
        data, sr = sf.read(event_input["file_path"], dtype="float32", always_2d=False)
    else:
        raise ValueError("Provide 'audio_url', 'audio_b64' or 'file_path'.")

    if data.ndim == 2:
        data = data.mean(axis=1)

    # hard cap
    max_n = int(MAX_AUDIO_SECONDS * sr)
    if data.shape[0] > max_n:
        data = data[:max_n]

    # resample to 16k
    if sr != 16000:
        data, sr = _resample_ta(data, sr, 16000)

    return data, sr

def _vox_transcribe_chunk(wave_mono_f32: np.ndarray, sr: int, language: str | None) -> str:
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
    if not transcript_text.strip():
        return ""
    tok = _vox_processor.tokenizer
    prompt = (
        "Tu résumes fidèlement un appel en français. "
        "Écris 2–3 phrases concises, sans listes ni citations, sans ajout d'informations.\n\n"
        f"Transcription:\n{transcript_text}\n\nRésumé:"
    )
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
    return tok.batch_decode(out[:, encoded.shape[1]:], skip_special_tokens=True)[0].strip()

def _classify_mood_fr(text: str) -> Dict[str, Any]:
    if not _clf or not text.strip():
        return {"label_en":"neutral","label_fr":"neutre","confidence":0.0,"scores":{"negative":0.0,"neutral":1.0,"positive":0.0}}
    res = _clf(text, candidate_labels=["positive","neutral","negative"], multi_label=False)
    if isinstance(res, list):
        res = res[0]
    labels, scores = res["labels"], res["scores"]
    mapping = {"positive":"bon","neutral":"neutre","negative":"mauvais"}
    best = labels[0]
    return {
        "label_en": best,
        "label_fr": mapping.get(best, best),
        "confidence": float(scores[labels.index(best)]),
        "scores": {k: float(scores[labels.index(k)]) for k in ["negative","neutral","positive"] if k in labels},
    }

# --------- Core ---------
def transcribe_diarized_strict(event_input: Dict[str, Any]) -> Dict[str, Any]:
    language = event_input.get("language") or None
    with_summary = bool(event_input.get("with_summary", True))
    max_new_tokens = int(event_input.get("max_new_tokens", 160))

    wav, sr = _load_audio_from_event(event_input)

    # diarization strict à 2 speakers
    diar = load_diarizer()
    diarization = diar({"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sr,
                        "min_speakers": MAX_SPEAKERS, "max_speakers": MAX_SPEAKERS})

    segments: List[Dict[str, Any]] = []
    try:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": float(segment.start), "end": float(segment.end), "speaker": str(speaker)})
    except Exception:
        segments = list(diarization.get("segments", []))

    segments.sort(key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))

    # garder l'ordre d'apparition pour mapper Agent/Client
    order: List[str] = []
    for s in segments:
        sp = s["speaker"]
        if sp not in order:
            order.append(sp)
        if len(order) >= MAX_SPEAKERS:
            break
    # sécurité si pipeline renvoie moins de 2 labels
    while len(order) < MAX_SPEAKERS:
        order.append(f"SPEAKER_{len(order):02d}")

    role_for = {order[0]: "Agent", order[1]: "Client"}

    # Transcription strict sur chaque segment
    PAD, MIN_DUR = 0.10, 0.60
    n = len(wav)
    out_segments: List[Dict[str, Any]] = []
    lines: List[str] = []

    for seg in segments:
        spk = seg["speaker"]
        if spk not in role_for:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        if end - start < MIN_DUR:
            continue
        s = max(0.0, start - PAD)
        e = min(n / sr, end + PAD)
        s_idx, e_idx = int(s * sr), int(e * sr)
        chunk = wav[s_idx:e_idx]
        txt = _vox_transcribe_chunk(chunk, sr, language)
        role = role_for[spk]
        if txt:
            out_segments.append({"start": start, "end": end, "speaker": role, "text": txt})
            lines.append(f"{role}: {txt}")

    transcript = "\n".join(lines).strip()
    summary = _summarize_concise(transcript, max_new_tokens=max_new_tokens) if with_summary else ""

    # Humeur
    mood_overall = _classify_mood_fr(transcript)
    mood_by_speaker: Dict[str, Any] = {}
    texts_by_role: Dict[str, List[str]] = {}
    for s in out_segments:
        texts_by_role.setdefault(s["speaker"], []).append(s["text"])
    for role, txts in texts_by_role.items():
        mood_by_speaker[role] = _classify_mood_fr(" ".join(txts))

    return {
        "task": "transcribe_diarized",
        "segments": out_segments,
        "transcript": transcript,
        "summary": summary,
        "mood_overall": mood_overall,
        "mood_by_speaker": mood_by_speaker,
    }

# --------- RunPod handler ---------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    try:
        inp = event.get("input") or {}
        task = (inp.get("task") or "transcribe_diarized").strip().lower()

        if task in ("health", "status", "ping"):
            info = {
                "ok": True,
                "python": "%s" % (tuple(map(int, torch.__version__.split('+')[0].split('.'))) and __import__('sys').version.split()[0]),
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
        return {"status": "FAILED", "error": f"{type(e).__name__}: {e}", "output": {"trace": trace}, "executionTime": int((time.time() - t0) * 1000)}

# local test
if __name__ == "__main__":
    print(json.dumps(handler({"input": {"task": "health"}}), ensure_ascii=False, indent=2))
