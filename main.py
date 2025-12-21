#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RunPod Serverless - Voxtral + (optional) PyAnnote diarization
- Loads Voxtral once at startup (cached by HF)
- Accepts jobs with: {audio_url, language, max_new_tokens, task, with_summary}
Tasks:
- transcribe: transcription simple
- transcribe_diarized: diarization (if available) + transcription (still global transcript for now)
Notes:
- VoxtralProcessor expects BOTH text and audio. If you only pass input_ids, the model won't condition on audio.
- For generation, we decode only the generated continuation (exclude the prompt tokens).
"""

import os
import io
import time
import json
import tempfile
import logging
from typing import Any, Dict, Optional, Tuple

import requests

# RunPod
import runpod

# Audio I/O
import soundfile as sf
import numpy as np
import torch

LOG = logging.getLogger("voxtral_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------- Global models (loaded once) ----------
VOXTRAL_MODEL_ID = os.getenv("VOXTRAL_MODEL_ID", "mistralai/Voxtral-Small-24B-2507")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Enable TF32 for speed (safe for transcription workloads)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

processor = None
model = None
diarizer = None


def gpu_mem_log(prefix: str = "[GPU]") -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    LOG.info(f"{prefix} Free: {free/1024**3:.1f}GB / Total: {total/1024**3:.1f}GB")


def ensure_hf_token_present() -> None:
    # On RunPod, you usually inject HF_TOKEN in environment variables.
    if not os.getenv("HF_TOKEN"):
        LOG.warning("[INIT] ⚠ HF_TOKEN absent. Si le modèle est privé, le chargement échouera.")
    else:
        LOG.info("[INIT] ✓ HF_TOKEN présent (auth HuggingFace OK)")


def download_audio(audio_url: str, timeout: int = 120) -> str:
    LOG.info(f"[DOWNLOAD] Téléchargement: {audio_url}")
    r = requests.get(audio_url, timeout=timeout)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(r.content)
    LOG.info(f"[DOWNLOAD] ✓ Audio téléchargé: {path}")
    return path


def load_audio_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    """Return mono float32 waveform and sampling rate 16000."""
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if sr != 16000:
        LOG.info(f"[AUDIO] Resampling {sr}Hz -> 16000Hz")
        # fast CPU resample (librosa is heavier). Use scipy if available, else simple polyphase via torchaudio.
        try:
            import torchaudio
            t = torch.from_numpy(wav).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav16 = resampler(t).squeeze(0).cpu().numpy().astype(np.float32)
            return wav16, 16000
        except Exception:
            # fallback: linear interpolation
            duration = wav.shape[0] / sr
            t_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
            t_new = np.linspace(0.0, duration, num=int(duration * 16000), endpoint=False)
            wav16 = np.interp(t_new, t_old, wav).astype(np.float32)
            return wav16, 16000

    return wav, sr


def load_voxtral() -> None:
    global processor, model
    if processor is not None and model is not None:
        return

    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    LOG.info(f"[VOXTRAL] Chargement du modèle: {VOXTRAL_MODEL_ID}")
    LOG.info("[VOXTRAL] Chargement du processor (AutoProcessor)...")
    processor = AutoProcessor.from_pretrained(VOXTRAL_MODEL_ID, trust_remote_code=True)

    LOG.info("[VOXTRAL] Chargement du modèle (VoxtralForConditionalGeneration)...")
    t0 = time.time()

    # Use device_map="auto" to spread on GPU (and maybe CPU) without OOM.
    # torch_dtype bfloat16 is good on A100/H100; fallback float16.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = VoxtralForConditionalGeneration.from_pretrained(
        VOXTRAL_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    LOG.info(f"[VOXTRAL] ✓ Modèle chargé ({time.time()-t0:.1f}s)")
    gpu_mem_log()


def load_diarizer() -> None:
    global diarizer
    if diarizer is not None:
        return
    LOG.info("[DIARIZER] Chargement: pyannote/speaker-diarization-3.1")
    from pyannote.audio import Pipeline

    # Newer pyannote uses "use_auth_token"
    hf_token = os.getenv("HF_TOKEN")
    diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    if torch.cuda.is_available():
        diarizer.to(torch.device("cuda"))
        LOG.info("[DIARIZER] Pipeline déplacée sur cuda")
    LOG.info("[DIARIZER] ✓ Diarizer chargé")


def perform_diarization(audio_path: str) -> Tuple[str, float]:
    """Return RTTM string and elapsed seconds."""
    if diarizer is None:
        raise RuntimeError("Diarizer non chargé")
    t0 = time.time()
    diar = diarizer(audio_path)
    rttm_buf = io.StringIO()
    diar.write_rttm(rttm_buf)
    return rttm_buf.getvalue(), time.time() - t0


def transcribe_with_voxtral(audio_path: str, language: str = "fr", max_new_tokens: int = 1800) -> str:
    """
    Correct Voxtral usage:
    - Build an instruction text (not chat-template dependent)
    - Call processor(text=..., audio=..., sampling_rate=16000, return_tensors="pt")
    - Generate
    - Decode ONLY continuation tokens (exclude prompt tokens)
    """
    load_voxtral()

    wav, sr = load_audio_mono_16k(audio_path)
    LOG.info(f"[VOXTRAL] Audio prêt: {wav.shape[0]/sr:.1f}s @ {sr}Hz")

    # Instruction: keep it simple and stable across tokenizer versions
    if language.lower().startswith("fr"):
        prompt = "Transcris fidèlement cet audio en français. Réponds uniquement avec la transcription."
    else:
        prompt = f"Transcribe this audio in {language}. Reply with the transcription only."

    LOG.info("[VOXTRAL] Préparation des inputs (processor(text, audio))...")
    inputs = processor(
        text=prompt,
        audio=wav,
        sampling_rate=sr,
        return_tensors="pt",
    )

    # Move tensors to GPU/device_map aware placement:
    # If model is sharded via device_map, keeping inputs on CUDA is fine.
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(DEVICE)

    # Determine prompt length for slicing
    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else None

    LOG.info(f"[VOXTRAL] generate (max_new_tokens={max_new_tokens})...")
    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Slice generated tokens to remove prompt tokens (critical!)
    if prompt_len is not None and generated.shape[1] > prompt_len:
        gen_only = generated[:, prompt_len:]
    else:
        gen_only = generated

    text_out = processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
    elapsed = time.time() - t0
    LOG.info(f"[VOXTRAL] ✓ Transcription terminée ({elapsed:.1f}s, {len(text_out)} chars)")

    # Basic cleanup: remove weird empty lines
    text_out = "\n".join([line.rstrip() for line in text_out.splitlines()]).strip()
    return text_out


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod job format:
      {"input": {"audio_url": "...", "language": "fr", "max_new_tokens": 2500, "task": "transcribe_diarized", "with_summary": true}}
    """
    t0 = time.time()
    inp = job.get("input", {}) or {}

    audio_url = inp.get("audio_url")
    if not audio_url:
        return {"error": "Missing audio_url"}

    language = inp.get("language", "fr")
    task = inp.get("task", "transcribe")
    max_new_tokens = int(inp.get("max_new_tokens", 1800))
    want_diar = (task == "transcribe_diarized") and bool(inp.get("diarization", True))
    with_summary = bool(inp.get("with_summary", False))

    LOG.info(f"[HANDLER] Tâche: {task}, diarization={want_diar}, langue: {language}, tokens: {max_new_tokens}")

    audio_path = None
    diar_rttm = None
    diar_elapsed = None
    try:
        audio_path = download_audio(audio_url)
        wav, sr = sf.read(audio_path, always_2d=False)
        duration_sec = (wav.shape[0] / sr) if hasattr(wav, "shape") else None
        LOG.info(f"[HANDLER] Durée audio: {duration_sec:.1f}s" if duration_sec else "[HANDLER] Durée audio: ?")

        if want_diar:
            try:
                load_diarizer()
                LOG.info("[HANDLER] Diarization: start")
                diar_rttm, diar_elapsed = perform_diarization(audio_path)
                LOG.info(f"[HANDLER] Diarization: done ({diar_elapsed:.2f}s)")
            except Exception as e:
                LOG.warning(f"[HANDLER] Diarization indisponible -> fallback transcription simple: {e}")
                diar_rttm, diar_elapsed = None, None

        transcript = transcribe_with_voxtral(audio_path, language=language, max_new_tokens=max_new_tokens)

        result: Dict[str, Any] = {
            "task": task,
            "language": language,
            "duration_sec": float(duration_sec) if duration_sec else None,
            "elapsed_sec": round(time.time() - t0, 2),
            "diarization_rttm": diar_rttm,
            "diarization_elapsed_sec": diar_elapsed,
            "transcript": transcript,
        }

        if with_summary:
            # Simple summary: do it on CPU with a lightweight heuristic to avoid extra model calls.
            # (You can later plug an LLM summarizer if desired.)
            # Here: first 600 chars as "preview"
            preview = transcript.replace("\n", " ").strip()
            result["summary"] = preview[:600] + ("…" if len(preview) > 600 else "")

        return result

    except Exception as e:
        LOG.exception("[HANDLER] Erreur traitement")
        return {"error": str(e)}

    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def preload() -> None:
    LOG.info("[INIT] Pré-chargement des modèles...")
    ensure_hf_token_present()
    gpu_mem_log()
    try:
        load_voxtral()
        LOG.info("[INIT] ✓ Voxtral pré-chargé")
    except Exception:
        LOG.exception("[INIT] ⚠ Échec pré-chargement Voxtral")

    # Diarizer is optional; preload only if user wants it most of the time
    if os.getenv("PRELOAD_DIARIZER", "1") == "1":
        try:
            load_diarizer()
            LOG.info("[INIT] ✓ Diarizer pré-chargé")
        except Exception as e:
            LOG.warning(f"[INIT] ⚠ Échec pré-chargement Diarizer: {e}")

    LOG.info("[INIT] ✓ Initialisation terminée")


preload()
runpod.serverless.start({"handler": handler})
