#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import tempfile
import logging
from typing import Any, Dict, Optional

import requests
import numpy as np
import soundfile as sf
import librosa
import runpod

LOG = logging.getLogger("serverless")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

VOXTRAL_REPO = os.getenv("VOXTRAL_REPO", "mistralai/Voxtral-Small-24B-2507")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

_voxtral_model = None
_voxtral_processor = None
_diarizer = None


def _gpu_mem_log(tag: str = "GPU"):
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            LOG.info(f"[{tag}] Free: {free/1024**3:.1f}GB / Total: {total/1024**3:.1f}GB")
    except Exception:
        pass


def configure_hf_auth():
    if os.getenv("HF_TOKEN"):
        LOG.info("[INIT] ✓ HF_TOKEN présent (auth HuggingFace OK)")
    else:
        LOG.warning("[INIT] ⚠ HF_TOKEN absent (modèles gated = échec possible)")


def load_voxtral_model():
    global _voxtral_model, _voxtral_processor
    if _voxtral_model is not None and _voxtral_processor is not None:
        return _voxtral_model, _voxtral_processor

    LOG.info(f"[VOXTRAL] Chargement du modèle: {VOXTRAL_REPO}")
    t0 = time.time()

    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    import torch

    LOG.info("[VOXTRAL] Chargement du processor (AutoProcessor)...")
    processor = AutoProcessor.from_pretrained(VOXTRAL_REPO)

    LOG.info("[VOXTRAL] Chargement du modèle (VoxtralForConditionalGeneration)...")
    model = VoxtralForConditionalGeneration.from_pretrained(
        VOXTRAL_REPO,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.eval()

    _voxtral_model, _voxtral_processor = model, processor
    LOG.info(f"[VOXTRAL] ✓ Modèle chargé ({time.time()-t0:.1f}s)")
    _gpu_mem_log("GPU")
    return model, processor


def _ensure_torchaudio_compat():
    # pyannote.audio may call torchaudio.set_audio_backend on import.
    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            def _noop(*args, **kwargs):
                return None
            torchaudio.set_audio_backend = _noop
            LOG.info("[DIARIZER] Monkeypatch: torchaudio.set_audio_backend ajouté (no-op)")
    except Exception as e:
        LOG.warning(f"[DIARIZER] torchaudio non importable: {repr(e)}")


def load_diarizer(hf_token: Optional[str] = None):
    global _diarizer
    if _diarizer is not None:
        return _diarizer

    LOG.info(f"[DIARIZER] Chargement: {DIARIZATION_MODEL}")
    _ensure_torchaudio_compat()

    from pyannote.audio import Pipeline

    try:
        diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    except TypeError:
        diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

    _diarizer = diarizer
    LOG.info("[DIARIZER] ✓ Diarizer chargé")
    return diarizer


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


def load_audio_mono(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr


def transcribe_with_voxtral(audio_path: str, language: str = "fr", max_new_tokens: int = 2500) -> str:
    model, processor = load_voxtral_model()

    audio, sr = load_audio_mono(audio_path)
    if sr != 16000:
        LOG.info(f"[VOXTRAL] Resampling {sr}Hz -> 16000Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    duration = len(audio) / float(sr)
    LOG.info(f"[VOXTRAL] Audio prêt: {duration:.1f}s @ {sr}Hz")

    prompt = (
        f"Transcris fidèlement cet enregistrement audio en {language}. "
        "Ne rajoute rien. Retourne uniquement le texte."
    )

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "array": audio, "sampling_rate": sr},
            {"type": "text", "text": prompt},
        ]}
    ]

    LOG.info("[VOXTRAL] Préparation des inputs (apply_chat_template)...")
    # IMPORTANT: do NOT pass add_generation_prompt (unsupported by MistralCommonTokenizer in your stack)
    input_ids = processor.apply_chat_template(
        conversation,
        return_tensors="pt",
    )

    import torch
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    LOG.info(f"[VOXTRAL] Génération (max_new_tokens={max_new_tokens})...")
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()
    return text


def summarize_text(text: str) -> str:
    if not text:
        return ""
    if len(text) <= 400:
        return text
    return text[:400].rstrip() + "…"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input", {}) or {}
    task = inp.get("task", "transcribe")
    audio_url = inp.get("audio_url")
    language = inp.get("language", "fr")
    max_new_tokens = int(inp.get("max_new_tokens", 2500))
    with_summary = bool(inp.get("with_summary", False))

    LOG.info(f"[HANDLER] Tâche: {task}, langue: {language}, tokens: {max_new_tokens}")

    if not audio_url:
        return {"error": "audio_url manquant"}

    audio_path = None
    try:
        audio_path = download_audio(audio_url)
        audio, sr = load_audio_mono(audio_path)
        duration = len(audio) / float(sr)
        LOG.info(f"[HANDLER] Durée audio: {duration:.1f}s")

        hf_token = os.getenv("HF_TOKEN")

        diarized = None
        if task == "transcribe_diarized":
            try:
                LOG.info("[HANDLER] Mode diarisation activé")
                diarizer = load_diarizer(hf_token=hf_token)
                diarized = diarizer(audio_path)
            except Exception as e:
                LOG.warning(f"[HANDLER] Diarization indisponible -> fallback transcription simple: {repr(e)}")

        transcript = transcribe_with_voxtral(audio_path, language=language, max_new_tokens=max_new_tokens)

        result: Dict[str, Any] = {
            "task": task,
            "language": language,
            "duration_sec": duration,
            "transcript": transcript,
        }

        if diarized is not None:
            try:
                result["diarization_rttm"] = diarized.to_rttm()
            except Exception:
                result["diarization"] = str(diarized)

        if with_summary:
            result["summary"] = summarize_text(transcript)

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


def _preload():
    LOG.info("[INIT] Pré-chargement des modèles...")
    configure_hf_auth()
    _gpu_mem_log("GPU")

    try:
        load_voxtral_model()
        LOG.info("[INIT] ✓ Voxtral pré-chargé")
    except Exception as e:
        LOG.warning(f"[INIT] ⚠ Échec pré-chargement Voxtral: {repr(e)}")

    try:
        load_diarizer(hf_token=os.getenv("HF_TOKEN"))
        LOG.info("[INIT] ✓ Diarizer pré-chargé")
    except Exception as e:
        LOG.warning(f"[INIT] ⚠ Échec pré-chargement Diarizer: {repr(e)}")

    LOG.info("[INIT] ✓ Initialisation terminée")


if __name__ == "__main__":
    _preload()
    runpod.serverless.start({"handler": handler})
