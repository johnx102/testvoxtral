#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Whisper + Mistral Small 3.1 — RunPod Serverless Worker
# Pipeline : Whisper (transcription + diarisation stéréo) + Mistral (résumé + sentiment + correction)
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
print("=" * 70)

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
APP_VERSION        = os.environ.get("APP_VERSION", "whisper-mistral-v5.0")
HF_TOKEN           = os.environ.get("HF_TOKEN", "").strip()
MAX_DURATION_S     = int(os.environ.get("MAX_DURATION_S", "3600"))
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Whisper
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v2")
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE", "float16")

# Mistral Small 3.1 (texte only — résumé, sentiment, correction)
# Forcé en dur — ne pas utiliser d'env var pour éviter les overrides
LLM_MODEL_ID       = "mistralai/Mistral-Nemo-Instruct-2407"
QUANT_MODE         = os.environ.get("QUANT_MODE", "bnb4").lower()

# Sentiment
ENABLE_SENTIMENT   = os.environ.get("ENABLE_SENTIMENT", "1") == "1"

# Détection musique d'attente
ENABLE_HOLD_MUSIC_DETECTION = os.environ.get("ENABLE_HOLD_MUSIC_DETECTION", "1") == "1"
HOLD_MUSIC_SPEECH_RATIO_HARD = float(os.environ.get("HOLD_MUSIC_SPEECH_RATIO_HARD", "0.03"))
HOLD_MUSIC_SPEECH_RATIO_SOFT = float(os.environ.get("HOLD_MUSIC_SPEECH_RATIO_SOFT", "0.08"))
HOLD_MUSIC_MIN_DURATION      = float(os.environ.get("HOLD_MUSIC_MIN_DURATION", "30.0"))

# Globals
_whisper_model = None
_llm_tokenizer = None
_llm_model     = None


# =============================================================================
# HELPERS
# =============================================================================
def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _gpu_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path

def _validate_audio(path: str) -> Tuple[bool, str, float]:
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

def _extract_mono_channel(wav_path: str, channel: int) -> str:
    import soundfile as sf
    data, sr = sf.read(wav_path)
    if data.ndim == 1:
        return wav_path
    mono = data[:, channel]
    out_path = os.path.join(tempfile.gettempdir(), f"ch{channel}_{uuid.uuid4().hex}.wav")
    sf.write(out_path, mono, sr)
    return out_path


# =============================================================================
# WHISPER — Transcription
# =============================================================================
def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        log(f"[WHISPER] Loading model: {WHISPER_MODEL_SIZE} (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})")
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        log("[WHISPER] Model loaded successfully")
        return _whisper_model
    except Exception as e:
        log(f"[WHISPER] ERROR loading model: {e}")
        return None


def _transcribe_channel_whisper(wav_path: str, channel: int, speaker: str, language: str = "fr") -> List[Dict]:
    """Transcrit un canal avec FasterWhisper + word_timestamps. Regroupe par pauses."""
    ch_path = _extract_mono_channel(wav_path, channel)
    try:
        model = _load_whisper()
        if model is None:
            return []

        segments_raw, info = model.transcribe(ch_path, language=language, beam_size=5, word_timestamps=True, vad_filter=False)

        all_words = []
        for seg in segments_raw:
            if seg.words:
                for w in seg.words:
                    all_words.append({"word": w.word, "start": w.start, "end": w.end})

        if not all_words:
            return []

        # Regrouper par pauses > 0.8s
        result_segments = []
        current = {"speaker": speaker, "text": "", "start": all_words[0]["start"], "end": all_words[0]["end"], "mood": None}

        for word in all_words:
            if current["end"] is not None and word["start"] - current["end"] > 0.8:
                if current["text"].strip():
                    result_segments.append({
                        "speaker": speaker, "start": round(current["start"], 2),
                        "end": round(current["end"], 2), "text": current["text"].strip(), "mood": None,
                    })
                current = {"speaker": speaker, "text": "", "start": word["start"], "end": None, "mood": None}
            current["text"] += word["word"]
            current["end"] = word["end"]

        if current["text"].strip():
            result_segments.append({
                "speaker": speaker, "start": round(current["start"], 2),
                "end": round(current["end"], 2), "text": current["text"].strip(), "mood": None,
            })

        # Filtrer hallucinations Whisper
        WHISPER_HALLUCINATIONS = ["sous-titrage st'", "sous-titres réalisés par", "amara.org",
                                  "merci d'avoir regardé", "abonnez-vous à la chaîne"]
        result_segments = [s for s in result_segments
                          if not (len(s["text"].split()) < 30 and any(h in s["text"].lower() for h in WHISPER_HALLUCINATIONS))]

        log(f"[WHISPER] {speaker} (ch{channel}): {len(result_segments)} phrases, {sum(len(s['text']) for s in result_segments)} chars")
        return result_segments
    finally:
        if ch_path and ch_path != wav_path and os.path.exists(ch_path):
            os.remove(ch_path)


def _transcribe_mono_whisper(wav_path: str, language: str = "fr") -> str:
    """Transcrit un fichier mono complet. Retourne le texte brut."""
    model = _load_whisper()
    if model is None:
        return ""
    try:
        segments_raw, info = model.transcribe(wav_path, language=language, beam_size=5, vad_filter=False)
        texts = []
        for seg in segments_raw:
            text = seg.text.strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()
    except Exception as e:
        log(f"[WHISPER] Mono transcription error: {e}")
        return ""


# =============================================================================
# MISTRAL SMALL 3.1 — LLM texte (résumé, sentiment, correction)
# =============================================================================
def _load_llm():
    global _llm_tokenizer, _llm_model
    if _llm_tokenizer is not None and _llm_model is not None:
        return _llm_tokenizer, _llm_model

    log(f"[LLM] Loading Mistral Small 3.1: {LLM_MODEL_ID} [QUANT_MODE={QUANT_MODE}]")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok_kwargs = {"trust_remote_code": True}
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, **tok_kwargs)
    log("[LLM] Tokenizer loaded")

    mdl_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, "trust_remote_code": True}
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    # Mode de quantization configurable via QUANT_MODE env var :
    #   "bf16"  → bfloat16 natif (le plus rapide, nécessite ~24GB VRAM pour Nemo 12B)
    #   "bnb8"  → INT8 BitsAndBytes (~12GB VRAM, ~3× plus lent en inférence)
    #   "bnb4"  → INT4 NF4 (~7GB VRAM, qualité un peu dégradée)
    quant = QUANT_MODE.lower()
    if torch.cuda.is_available() and quant in ("bnb8", "int8"):
        from transformers import BitsAndBytesConfig
        mdl_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        log("[LLM] INT8 BnB config ready")
    elif torch.cuda.is_available() and quant in ("bnb4", "int4", "nf4"):
        from transformers import BitsAndBytesConfig
        mdl_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        log("[LLM] INT4 NF4 BnB config ready")
    else:
        mdl_kwargs["torch_dtype"] = torch.bfloat16
        # Flash Attention 2 si disponible (gain 20-40% sur long contexte)
        try:
            import flash_attn  # noqa: F401
            mdl_kwargs["attn_implementation"] = "flash_attention_2"
            log("[LLM] bfloat16 + flash_attention_2")
        except ImportError:
            log("[LLM] bfloat16 (no flash-attn)")

    _llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, **mdl_kwargs)
    log("[LLM] Model loaded successfully")
    return _llm_tokenizer, _llm_model


def run_llm(prompt: str, max_new_tokens: int = 256) -> str:
    """Envoie un prompt texte à Mistral Small 3.1 et retourne la réponse."""
    tokenizer, model = _load_llm()
    try:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = time.time()
        if isinstance(inputs, torch.Tensor):
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None)
            inp_len = inputs.shape[1]
        else:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None)
            inp_len = inputs["input_ids"].shape[1]

        decoded = tokenizer.decode(outputs[0][inp_len:], skip_special_tokens=True).strip()
        dt = round(time.time() - t0, 2)
        log(f"[LLM] Completed in {dt}s ({len(decoded)} chars)")

        del outputs, inputs
        _gpu_clear()
        return decoded
    except Exception as e:
        log(f"[LLM] Inference error: {e}")
        _gpu_clear()
        return ""


# =============================================================================
# CORRECTION DU TRANSCRIPT
# =============================================================================
ENABLE_TRANSCRIPT_CORRECTION = os.environ.get("ENABLE_TRANSCRIPT_CORRECTION", "1") == "1"

def correct_transcript(transcript: str) -> str:
    """Corrige les erreurs de reconnaissance Whisper via Mistral.
    Préserve STRICTEMENT le format ligne par ligne 'Speaker: texte'.
    Retourne le transcript corrigé ou l'original en cas d'échec."""
    if not transcript or len(transcript.split()) < 5:
        return transcript

    original_lines = [l for l in transcript.split("\n") if l.strip()]
    n_lines = len(original_lines)

    prompt = (
        "Tu es un correcteur de transcription téléphonique française. "
        "Corrige UNIQUEMENT les erreurs de reconnaissance vocale évidentes en utilisant le contexte "
        "(ex: cabinet dentaire → 'désertage'/'détertraite' = 'détartrage', 'centre d'entraînement' = 'centre dentaire', noms propres mal orthographiés, etc.).\n"
        "RÈGLES STRICTES :\n"
        f"- Garde EXACTEMENT {n_lines} lignes, dans le même ordre\n"
        "- Garde le préfixe 'Agent:' ou 'Client:' au début de chaque ligne\n"
        "- Ne reformule pas, ne résume pas, ne supprime rien\n"
        "- Change uniquement les mots manifestement mal transcrits\n"
        "- Garde la ponctuation et les hésitations naturelles\n\n"
        f"Transcript original :\n{transcript[:4000]}\n\n"
        "Transcript corrigé (même nombre de lignes, même format) :"
    )

    # Budget tokens large : approx 1 token/char pour français
    max_tok = min(1024, max(256, len(transcript) // 2))
    corrected = run_llm(prompt, max_new_tokens=max_tok)

    if not corrected:
        log("[CORRECT] LLM returned empty → keeping original")
        return transcript

    corrected_lines = [l.strip() for l in corrected.split("\n") if l.strip()]
    # Garder uniquement les lignes qui commencent par Agent: ou Client:
    corrected_lines = [l for l in corrected_lines if l.startswith("Agent:") or l.startswith("Client:")]

    if len(corrected_lines) != n_lines:
        log(f"[CORRECT] Line count mismatch ({len(corrected_lines)} vs {n_lines}) → keeping original")
        return transcript

    # Vérifier que les speakers correspondent
    for i, (orig, corr) in enumerate(zip(original_lines, corrected_lines)):
        orig_sp = orig.split(":", 1)[0]
        corr_sp = corr.split(":", 1)[0]
        if orig_sp != corr_sp:
            log(f"[CORRECT] Speaker mismatch at line {i} → keeping original")
            return transcript

    log(f"[CORRECT] Transcript corrected ({n_lines} lines)")
    return "\n".join(corrected_lines)


# =============================================================================
# RÉSUMÉ
# =============================================================================
def generate_summary(transcript: str, duration_seconds: float = 0) -> str:
    if not transcript or len(transcript.split()) < 10:
        return "Conversation très brève."

    if duration_seconds <= 180:   max_tokens = 70
    elif duration_seconds <= 600: max_tokens = 90
    else:                         max_tokens = 110

    prompt = (
        "Résume cette conversation téléphonique en 1 ou 2 phrases courtes (max 40 mots au total). "
        "Dis uniquement l'essentiel : qui appelle, pour quoi, et la conclusion. "
        "Pas d'introduction, pas de reformulation inutile, va droit au but.\n\n"
        f"Conversation:\n{transcript[:3000]}\n\nRésumé court:"
    )

    summary = run_llm(prompt, max_new_tokens=max_tokens)
    if not summary or len(summary) < 10:
        # Fallback extractif
        lines = [l.strip() for l in transcript.split('\n') if l.strip() and ':' in l]
        if lines:
            return lines[0].split(':', 1)[1].strip()[:200]
        return "Conversation brève."
    return summary


# =============================================================================
# SENTIMENT
# =============================================================================
def analyze_sentiment(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.5}

    prompt = (
        "Analyse le ressenti de cette conversation téléphonique.\n"
        "Réponds par UN SEUL MOT : bon, neutre, ou mauvais\n"
        "- bon : échange poli et cordial, demande traitée\n"
        "- neutre : échange bref, ton froid ou distant\n"
        "- mauvais : mécontentement, plainte, frustration\n\n"
        f"Conversation : {text[:1500]}"
    )

    response = run_llm(prompt, max_new_tokens=16).strip().lower()
    log(f"[SENTIMENT] LLM response: '{response}'")

    if "mauvais" in response or "insatisf" in response or "négatif" in response:
        return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.75,
                "scores": {"negative": 0.75, "neutral": 0.20, "positive": 0.05}}
    elif "bon" in response or "satisf" in response or "positif" in response:
        return {"label_en": "positive", "label_fr": "bon", "confidence": 0.80,
                "scores": {"negative": 0.05, "neutral": 0.15, "positive": 0.80}}
    else:
        return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.70,
                "scores": {"negative": 0.15, "neutral": 0.70, "positive": 0.15}}


# =============================================================================
# HOLD MUSIC DETECTION (RMS-based, pas de LLM)
# =============================================================================
def detect_hold_music(audio_path: str) -> Dict[str, Any]:
    import soundfile as sf
    import numpy as np
    result = {"is_hold_music": False, "speech_ratio": 1.0, "duration": 0.0}
    try:
        y, sr = sf.read(audio_path, dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        duration = len(y) / sr
        result["duration"] = round(duration, 2)
        if duration < HOLD_MUSIC_MIN_DURATION:
            return result
        frame_length = int(0.030 * sr)
        hop_length = int(0.010 * sr)
        n_frames = 1 + (len(y) - frame_length) // hop_length
        rms = np.array([np.sqrt(np.mean(y[i*hop_length:i*hop_length+frame_length]**2)) for i in range(n_frames)])
        peak = np.max(np.abs(y))
        if peak < 1e-6:
            result.update({"is_hold_music": True, "speech_ratio": 0.0})
            return result
        rms_norm = rms / peak
        speech_threshold = float(np.median(rms_norm)) + 1.5 * float(np.std(rms_norm))
        speech_mask = rms_norm > speech_threshold
        speech_ratio = float(np.sum(speech_mask)) / len(rms_norm)
        result["speech_ratio"] = round(speech_ratio, 4)
        if speech_ratio < HOLD_MUSIC_SPEECH_RATIO_HARD:
            result["is_hold_music"] = True
        elif speech_ratio < HOLD_MUSIC_SPEECH_RATIO_SOFT and duration > 60.0:
            result["is_hold_music"] = True
        return result
    except Exception as e:
        log(f"[HOLD_MUSIC] Error: {e}")
        return result


# =============================================================================
# DEDUPLICATION
# =============================================================================
def remove_duplicate_segments(segments: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
    if len(segments) <= 1:
        return segments
    def text_similarity(a, b):
        if not a or not b: return 0.0
        words_a, words_b = set(a.lower().split()), set(b.lower().split())
        if not words_a or not words_b: return 0.0
        return len(words_a & words_b) / max(len(words_a), len(words_b))

    seen, cleaned = [], []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text: continue
        if any(text_similarity(text, s) >= similarity_threshold for s in seen):
            continue
        cleaned.append(seg)
        seen.append(text)
    if len(cleaned) < len(segments):
        log(f"[DEDUP] Removed {len(segments) - len(cleaned)} duplicate segments")
    return cleaned


# =============================================================================
# HANDLER PRINCIPAL
# =============================================================================
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    inp = job.get("input", {}) or {}
    log(f"[HANDLER] New job received: {inp.get('task', 'unknown')}")

    if inp.get("ping"):
        return {"pong": True}
    if inp.get("task") == "health":
        return {"ok": True, "app_version": APP_VERSION, "whisper": WHISPER_MODEL_SIZE, "llm": LLM_MODEL_ID}

    language       = inp.get("language") or "fr"
    with_summary   = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))

    # Déterminer la direction de l'appel
    call_direction = (inp.get("call_direction") or "").lower().strip()
    if call_direction not in ("inbound", "outbound"):
        audio_url = inp.get("audio_url") or ""
        filename = audio_url.split("/")[-1].lower()
        if filename.startswith("in-") or filename.startswith("in_"):
            call_direction = "inbound"
        elif filename.startswith("out-") or filename.startswith("out_"):
            call_direction = "outbound"
        else:
            call_direction = "unknown"

    log(f"[HANDLER] language={language}, direction={call_direction}, summary={with_summary}")

    local_path, cleanup = None, False

    try:
        # Télécharger l'audio
        if inp.get("audio_url"):
            local_path = _download_to_tmp(inp["audio_url"].strip())
            cleanup = True
        elif inp.get("file_path"):
            local_path = inp["file_path"]
        else:
            return {"error": "Provide 'audio_url' or 'file_path'."}

        ok, err, est_dur = _validate_audio(local_path)
        if not ok:
            return {"error": err}

        log(f"[HANDLER] Processing: {local_path} ({est_dur:.1f}s)")

        # Vérifier stéréo
        import soundfile as sf
        info = sf.info(local_path)
        is_stereo = info.channels >= 2

        # Hold music detection (mono mix)
        if ENABLE_HOLD_MUSIC_DETECTION:
            if is_stereo:
                # Vérifier chaque canal
                ch0_path = _extract_mono_channel(local_path, 0)
                ch1_path = _extract_mono_channel(local_path, 1)
                hold_ch0 = detect_hold_music(ch0_path)
                hold_ch1 = detect_hold_music(ch1_path)
                log(f"[HOLD_MUSIC] CH0: hold={hold_ch0['is_hold_music']}, ratio={hold_ch0['speech_ratio']:.4f}")
                log(f"[HOLD_MUSIC] CH1: hold={hold_ch1['is_hold_music']}, ratio={hold_ch1['speech_ratio']:.4f}")
                for p in [ch0_path, ch1_path]:
                    if p != local_path and os.path.exists(p): os.remove(p)
                if hold_ch0["is_hold_music"] and hold_ch1["is_hold_music"]:
                    return {"segments": [], "transcript": "", "summary": "Musique d'attente uniquement.",
                            "hold_music_detected": True, "audio_duration": est_dur}
            else:
                hold = detect_hold_music(local_path)
                if hold["is_hold_music"]:
                    return {"segments": [], "transcript": "", "summary": "Musique d'attente uniquement.",
                            "hold_music_detected": True, "audio_duration": est_dur}

        # ── TRANSCRIPTION ──────────────────────────────────────────────────
        if is_stereo:
            log(f"[HANDLER] Stereo audio → Whisper per-channel with word timestamps")

            # Skip les canaux détectés comme musique d'attente (pattern RMS, pas juste le ratio)
            if ENABLE_HOLD_MUSIC_DETECTION and hold_ch0.get("is_hold_music"):
                log(f"[HANDLER] CH0 (Client) is hold music → skipping Whisper")
                client_segs = []
            else:
                client_segs = _transcribe_channel_whisper(local_path, 0, "Client", language)

            if ENABLE_HOLD_MUSIC_DETECTION and hold_ch1.get("is_hold_music"):
                log(f"[HANDLER] CH1 (Agent) is hold music → skipping Whisper")
                agent_segs = []
            else:
                agent_segs = _transcribe_channel_whisper(local_path, 1, "Agent", language)

            segments = client_segs + agent_segs
            segments.sort(key=lambda s: s["start"])

            if not segments:
                # Fallback mono
                log("[HANDLER] No stereo segments → fallback mono")
                audio_mono = AudioSegment.from_file(local_path).set_channels(1)
                mono_path = os.path.join(tempfile.gettempdir(), f"mono_{uuid.uuid4().hex}.wav")
                audio_mono.export(mono_path, format="wav")
                text = _transcribe_mono_whisper(mono_path, language)
                os.remove(mono_path)
                if text:
                    segments = [{"speaker": "Agent", "start": 0.0, "end": est_dur, "text": text, "mood": None}]

            diarization_mode = "stereo_whisper_words"
        else:
            log(f"[HANDLER] Mono audio → Whisper transcription")
            text = _transcribe_mono_whisper(local_path, language)
            if text:
                segments = [{"speaker": "Agent", "start": 0.0, "end": est_dur, "text": text, "mood": None}]
            else:
                segments = []
            diarization_mode = "mono_whisper"

        if not segments:
            return {"segments": [], "transcript": "", "summary": "Aucune conversation détectée.",
                    "diarization_mode": diarization_mode, "audio_duration": est_dur}

        # Dedup
        segments = remove_duplicate_segments(segments)

        # Fusionner segments consécutifs même speaker
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if prev["speaker"] == seg["speaker"]:
                prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
                prev["end"] = seg["end"]
            else:
                merged.append(seg)

        log(f"[HANDLER] Final: {len(merged)} segments")

        full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in merged if s.get("text"))

        # ── CORRECTION DU TRANSCRIPT ───────────────────────────────────────
        if ENABLE_TRANSCRIPT_CORRECTION and full_transcript:
            log("[HANDLER] Correcting transcript...")
            corrected_transcript = correct_transcript(full_transcript)
            if corrected_transcript != full_transcript:
                # Réappliquer les corrections aux segments (ligne à ligne, même ordre)
                corrected_lines = [l.strip() for l in corrected_transcript.split("\n") if l.strip()]
                seg_with_text = [s for s in merged if s.get("text")]
                if len(corrected_lines) == len(seg_with_text):
                    for seg, line in zip(seg_with_text, corrected_lines):
                        if ":" in line:
                            seg["text"] = line.split(":", 1)[1].strip()
                full_transcript = corrected_transcript

        result = {
            "segments": merged,
            "transcript": full_transcript,
            "diarization_mode": diarization_mode,
            "audio_duration": est_dur,
        }

        # ── RÉSUMÉ ─────────────────────────────────────────────────────────
        if with_summary:
            log("[HANDLER] Generating summary...")
            result["summary"] = generate_summary(full_transcript, duration_seconds=est_dur)

        # ── SENTIMENT ──────────────────────────────────────────────────────
        if ENABLE_SENTIMENT:
            log("[HANDLER] Analyzing sentiment...")
            # Sentiment par speaker
            mood_by_speaker = {}
            for speaker in set(s["speaker"] for s in merged):
                speaker_text = " ".join(s["text"] for s in merged if s["speaker"] == speaker)
                mood_by_speaker[speaker] = analyze_sentiment(speaker_text)
                log(f"[SENTIMENT] {speaker}: {mood_by_speaker[speaker].get('label_fr', '?')}")

            # Attacher aux segments
            for seg in merged:
                sp = seg.get("speaker")
                if sp in mood_by_speaker:
                    seg["mood"] = mood_by_speaker[sp]

            # Client mood
            client_mood = mood_by_speaker.get("Client", mood_by_speaker.get("Agent"))
            result["mood_overall"] = client_mood
            result["mood_by_speaker"] = mood_by_speaker
            result["mood_client"] = client_mood

        log("[HANDLER] Transcription completed successfully")
        return {"task": "transcribe_diarized", **result}

    except Exception as e:
        log(f"[HANDLER] CRITICAL ERROR: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        _gpu_clear()
        if cleanup and local_path and os.path.exists(local_path):
            try: os.remove(local_path)
            except: pass


# =============================================================================
# INITIALISATION
# =============================================================================
try:
    log("[INIT] Starting preload...")
    log(f"[INIT] QUANT_MODE={QUANT_MODE} | APP_VERSION={APP_VERSION}")

    log("[INIT] Loading LLM (Mistral Small 3.1)...")
    _load_llm()

    log("[INIT] Loading Whisper...")
    _load_whisper()

    log(f"[INIT] Hold music detection: {'ENABLED' if ENABLE_HOLD_MUSIC_DETECTION else 'DISABLED'}")
    log(f"[INIT] Sentiment analysis: {'ENABLED' if ENABLE_SENTIMENT else 'DISABLED'}")
    log(f"[INIT] Whisper model: {WHISPER_MODEL_SIZE}")
    log(f"[INIT] LLM model: {LLM_MODEL_ID}")
    log("[INIT] Ready.")
except Exception as e:
    log(f"[WARN] Preload failed: {e}")

runpod.serverless.start({"handler": handler})
