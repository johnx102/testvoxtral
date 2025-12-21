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

# knobs
ENABLE_DIARIZATION_DEFAULT = os.getenv("ENABLE_DIARIZATION_DEFAULT", "1") == "1"
DIARIZATION_DEVICE = os.getenv("DIARIZATION_DEVICE", "cuda")  # "cuda" or "cpu"
DIARIZATION_BATCH_SIZE = int(os.getenv("DIARIZATION_BATCH_SIZE", "32"))  # pyannote speed/VRAM tradeoff
DIARIZATION_SKIP_FOR_SEC_OVER = float(os.getenv("DIARIZATION_SKIP_FOR_SEC_OVER", "0"))  # 0 = never skip

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

    # speed knobs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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
    """Load diarizer once. Move it to GPU explicitly (pyannote is often CPU by default)."""
    global _diarizer
    if _diarizer is not None:
        return _diarizer

    LOG.info(f"[DIARIZER] Chargement: {DIARIZATION_MODEL}")
    _ensure_torchaudio_compat()

    from pyannote.audio import Pipeline
    import torch

    t0 = time.time()
    try:
        diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
    except TypeError:
        diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)

    # Move to device for SPEED (this is the big one)
    device = torch.device("cuda") if (DIARIZATION_DEVICE == "cuda" and torch.cuda.is_available()) else torch.device("cpu")
    try:
        diarizer.to(device)
        LOG.info(f"[DIARIZER] Pipeline déplacée sur {device}")
    except Exception as e:
        LOG.warning(f"[DIARIZER] Impossible de déplacer sur {device}: {repr(e)}")

    # Some versions accept batch_size on call, some via attribute
    try:
        diarizer.instantiate({"segmentation": {"batch_size": DIARIZATION_BATCH_SIZE}})
        LOG.info(f"[DIARIZER] batch_size segmentation = {DIARIZATION_BATCH_SIZE}")
    except Exception:
        pass

    _diarizer = diarizer
    LOG.info(f"[DIARIZER] ✓ Diarizer chargé ({time.time()-t0:.1f}s)")
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


def _ensure_wav_16k(path: str) -> str:
    """Return a path to a 16kHz mono wav. If conversion is needed, returns a temp file path."""
    audio, sr = load_audio_mono(path)
    if sr == 16000:
        return path
    LOG.info(f"[AUDIO] Resampling {sr}Hz -> 16000Hz")
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    fd, tmp16 = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp16, audio, 16000)
    return tmp16


def transcribe_with_voxtral(audio_path: str, language: str = "fr", max_new_tokens: int = 1200) -> str:
    """Voxtral transcription. Note: large max_new_tokens can massively increase latency."""
    model, processor = load_voxtral_model()

    t0 = time.time()
    audio_path_for_model = _ensure_wav_16k(audio_path)

    # duration log
    audio, sr = load_audio_mono(audio_path_for_model)
    duration = len(audio) / float(sr)
    LOG.info(f"[VOXTRAL] Audio prêt: {duration:.1f}s @ {sr}Hz")

    prompt = (
        f"Transcris fidèlement cet enregistrement audio en {language}. "
        "Ne rajoute rien. Retourne uniquement le texte."
    )

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "path": audio_path_for_model},
            {"type": "text", "text": prompt},
        ]}
    ]

    LOG.info("[VOXTRAL] apply_chat_template...")
    input_ids = processor.apply_chat_template(conversation, return_tensors="pt")

    import torch
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    LOG.info(f"[VOXTRAL] generate (max_new_tokens={max_new_tokens})...")
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()

    if audio_path_for_model != audio_path:
        try:
            os.remove(audio_path_for_model)
        except Exception:
            pass

    LOG.info(f"[VOXTRAL] ✓ Transcription terminée ({time.time()-t0:.1f}s, {len(text)} chars)")
    return text


def summarize_text(text: str) -> str:
    # placeholder (keep cheap). You can swap later for an LLM call.
    if not text:
        return ""
    if len(text) <= 500:
        return text
    return text[:500].rstrip() + "…"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input", {}) or {}
    task = inp.get("task", "transcribe")
    audio_url = inp.get("audio_url")
    language = inp.get("language", "fr")

    # Important: 2500 tokens can be VERY slow; default to 1200 unless user insists
    max_new_tokens = int(inp.get("max_new_tokens", 1200))
    with_summary = bool(inp.get("with_summary", False))
    diarization_enabled = bool(inp.get("diarization", task == "transcribe_diarized")) and ENABLE_DIARIZATION_DEFAULT

    LOG.info(f"[HANDLER] Tâche: {task}, diarization={diarization_enabled}, langue: {language}, tokens: {max_new_tokens}")

    if not audio_url:
        return {"error": "audio_url manquant"}

    audio_path = None
    t_job = time.time()
    try:
        audio_path = download_audio(audio_url)
        audio, sr = load_audio_mono(audio_path)
        duration = len(audio) / float(sr)
        LOG.info(f"[HANDLER] Durée audio: {duration:.1f}s")

        hf_token = os.getenv("HF_TOKEN")

        diarization_info = None
        if diarization_enabled:
            if DIARIZATION_SKIP_FOR_SEC_OVER and duration > DIARIZATION_SKIP_FOR_SEC_OVER:
                LOG.warning(f"[HANDLER] Diarization skip: durée {duration:.1f}s > {DIARIZATION_SKIP_FOR_SEC_OVER}s")
            else:
                try:
                    LOG.info("[HANDLER] Diarization: start")
                    t_d = time.time()
                    diarizer = load_diarizer(hf_token=hf_token)
                    diarized = diarizer(audio_path)  # heavy step
                    diarization_info = {
                        "rttm": diarized.to_rttm(),
                        "elapsed_sec": round(time.time() - t_d, 2),
                    }
                    LOG.info(f"[HANDLER] Diarization: done ({diarization_info['elapsed_sec']}s)")
                except Exception as e:
                    LOG.warning(f"[HANDLER] Diarization indisponible -> fallback transcription simple: {repr(e)}")

        transcript = transcribe_with_voxtral(audio_path, language=language, max_new_tokens=max_new_tokens)

        result: Dict[str, Any] = {
            "task": task,
            "language": language,
            "duration_sec": duration,
            "transcript": transcript,
            "elapsed_sec": round(time.time() - t_job, 2),
        }

        if diarization_info:
            result["diarization_rttm"] = diarization_info["rttm"]
            result["diarization_elapsed_sec"] = diarization_info["elapsed_sec"]

        if with_summary:
            result["summary"] = summarize_text(transcript)

        LOG.info(f"[HANDLER] ✓ Job terminé ({result['elapsed_sec']}s)")
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
