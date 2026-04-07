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

# LLM texte (résumé, sentiment, correction)
# Forcé en dur — ne pas utiliser d'env var pour éviter les overrides
LLM_MODEL_ID       = "mistralai/Ministral-8B-Instruct-2410"
_RAW_QUANT_MODE    = os.environ.get("QUANT_MODE", "bf16").lower().strip()
# Valeurs valides uniquement — tout le reste (torchao, auto, empty, ...) → bf16 par défaut
_VALID_QUANT_MODES = {"bnb4", "int4", "nf4", "bnb8", "int8", "bf16", "bfloat16"}
if _RAW_QUANT_MODE not in _VALID_QUANT_MODES:
    print(f"[CONFIG] QUANT_MODE='{_RAW_QUANT_MODE}' non reconnu → fallback 'bf16'", flush=True)
    QUANT_MODE = "bf16"
else:
    QUANT_MODE = _RAW_QUANT_MODE

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
def _whisper_is_cached() -> bool:
    """Vérifie si le modèle Whisper CTranslate2 est dans le cache HF."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST
        repo_id = f"Systran/faster-whisper-{WHISPER_MODEL_SIZE}"
        path = try_to_load_from_cache(repo_id, "model.bin")
        return path is not None and path is not _CACHED_NO_EXIST
    except Exception:
        return False


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        log(f"[WHISPER] Loading model: {WHISPER_MODEL_SIZE} (device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})")
        # local_files_only=True si cache présent → skip HEAD requests HF (~2-3s)
        local_only = _whisper_is_cached()
        if local_only:
            log("[WHISPER] Cache détecté → local_files_only")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
            local_files_only=local_only,
        )
        log("[WHISPER] Model loaded successfully")
        return _whisper_model
    except Exception as e:
        log(f"[WHISPER] ERROR loading model: {e}")
        return None


def _compress_internal_repetitions(text: str) -> str:
    """Compresse les répétitions consécutives DANS un texte (boucles Whisper).
    Cherche des séquences de 1 à 10 mots qui se répètent ≥ 2 fois immédiatement
    après et garde une seule occurrence.

    Exemples :
    - "abc abc abc def" → "abc def"
    - "il faut qu'on l'accueille et il faut qu'on l'accueille et ... × 19" → "il faut qu'on l'accueille et"
    - "Monsieur le ministre. Monsieur le ministre. Monsieur le ministre." → "Monsieur le ministre."
    """
    if not text:
        return text
    words = text.split()
    if len(words) < 4:
        return text

    # Itérer par taille de séquence décroissante (10 → 1)
    # Décroissant pour attraper d'abord les longues séquences puis les courtes
    for seq_len in range(10, 0, -1):
        if len(words) < 2 * seq_len:
            continue
        new_words = []
        i = 0
        while i < len(words):
            seq = words[i:i + seq_len]
            if len(seq) < seq_len:
                new_words.extend(seq)
                break
            # Compter combien de fois cette séquence se répète immédiatement après
            j = i + seq_len
            count = 1
            while j + seq_len <= len(words) and words[j:j + seq_len] == seq:
                count += 1
                j += seq_len
            if count >= 2:
                # Garder UNE seule occurrence et avancer
                new_words.extend(seq)
                i = j
            else:
                new_words.append(words[i])
                i += 1
        words = new_words

    return " ".join(words)


def _detect_hold_regions_in_channel(
    audio_np,
    sr: int,
    window_s: float = 5.0,
    hop_s: float = 2.5,
    min_region_s: float = 10.0,
    speech_ratio_threshold: float = 0.03,
) -> List[Tuple[float, float]]:
    """Détecte les zones (start_s, end_s) de musique d'attente / silence prolongé.

    Critères stricts pour éviter les faux positifs sur les pauses naturelles d'une conversation :
    - Fenêtre 5s, hop 2.5s
    - Speech ratio < 3% (très strict, vraie absence de parole)
    - Durée minimale après fusion : 10s (une pause normale dans un dialogue dure 2-5s)

    Une pause normale d'écoute dans une conversation NE doit PAS être considérée comme hold.
    """
    import numpy as np
    win = int(window_s * sr)
    hop = int(hop_s * sr)
    if len(audio_np) < win:
        return []

    frame_len = int(0.030 * sr)
    frame_hop = int(0.010 * sr)

    # Énergie absolue de l'audio entier (référence pour distinguer silence vs hold music)
    global_peak = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
    if global_peak < 1e-6:
        return [(0.0, len(audio_np) / sr)]

    regions_raw: List[Tuple[float, float]] = []
    for i in range(0, len(audio_np) - win + 1, hop):
        chunk = audio_np[i:i + win]
        n_frames = 1 + (len(chunk) - frame_len) // frame_hop
        if n_frames < 10:
            continue
        rms = np.array([np.sqrt(np.mean(chunk[j*frame_hop:j*frame_hop+frame_len]**2)) for j in range(n_frames)])
        local_peak = float(np.max(np.abs(chunk)))

        # Seuil critique : on compare le pic local au pic global du canal
        # Si le pic local < 5% du pic global → silence/musique très calme
        if local_peak < 0.05 * global_peak:
            is_hold = True
        else:
            rms_norm = rms / local_peak
            threshold = float(np.median(rms_norm)) + 1.5 * float(np.std(rms_norm))
            speech_ratio = float(np.sum(rms_norm > threshold)) / len(rms_norm)
            is_hold = speech_ratio < speech_ratio_threshold

        if is_hold:
            regions_raw.append((i / sr, (i + win) / sr))

    if not regions_raw:
        return []

    # Fusionner les régions adjacentes (< 1s d'écart)
    merged: List[Tuple[float, float]] = []
    for start, end in regions_raw:
        if merged and start - merged[-1][1] < 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Filtrer : ne garder que les régions ≥ min_region_s
    filtered = [(s, e) for s, e in merged if (e - s) >= min_region_s]
    return filtered


def _mute_audio_regions(audio_np, sr: int, regions: List[Tuple[float, float]]):
    """Met à zéro les samples dans les régions indiquées. Modifie le tableau en place."""
    import numpy as np
    if not regions:
        return audio_np
    audio_np = audio_np.copy()
    for start, end in regions:
        s = max(0, int(start * sr))
        e = min(len(audio_np), int(end * sr))
        if e > s:
            audio_np[s:e] = 0.0
    return audio_np


def _transcribe_channel_whisper(wav_path: str, channel: int, speaker: str, language: str = "fr",
                                  mute_regions: Optional[List[Tuple[float, float]]] = None) -> List[Dict]:
    """Transcrit un canal avec FasterWhisper + word_timestamps. Regroupe par pauses.
    Si mute_regions est fourni, les zones correspondantes sont mises à zéro avant transcription
    (utile pour skipper ce que dit l'agent pendant que le client est en hold music)."""
    # Si on doit muter des zones, on génère un fichier WAV temporaire masqué
    if mute_regions:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            mono = data[:, channel].astype(np.float32)
        else:
            mono = data.astype(np.float32)
        mono = _mute_audio_regions(mono, sr, mute_regions)
        ch_path = os.path.join(tempfile.gettempdir(), f"muted_ch{channel}_{uuid.uuid4().hex}.wav")
        sf.write(ch_path, mono, sr)
        total_muted = sum(e - s for s, e in mute_regions)
        log(f"[WHISPER] {speaker} (ch{channel}): masked {total_muted:.1f}s of cross-channel hold music")
    else:
        ch_path = _extract_mono_channel(wav_path, channel)

    try:
        model = _load_whisper()
        if model is None:
            return []

        # vad_filter=True : saute les silences (majeure cause d'hallucinations Whisper)
        # min_silence_duration_ms=1000 : ne coupe que les silences ≥ 1s (préserve les pauses naturelles)
        # speech_pad_ms=400 : marge de 400ms autour de chaque zone de parole
        # condition_on_previous_text=False : évite les boucles d'hallucination
        # temperature=0 : déterministe, moins de variations créatives
        segments_raw, info = model.transcribe(
            ch_path,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 400,
                "threshold": 0.35,
            },
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            # ANTI-HALLUCINATION : si Whisper produit du texte alors qu'il y a > 3s
            # de silence VAD-détecté entre, c'est probablement une hallucination → ignorer
            # (3s = compromis entre filtrer les boucles et garder les vraies pauses entre phrases)
            hallucination_silence_threshold=3.0,
        )

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

        def _finalize_current(cur):
            """Crée un segment à partir du buffer current, après compression des répétitions."""
            raw_text = cur["text"].strip()
            if not raw_text:
                return None
            compressed = _compress_internal_repetitions(raw_text)
            return {
                "speaker": speaker,
                "start": round(cur["start"], 2),
                "end": round(cur["end"], 2),
                "text": compressed,
                "mood": None,
            }

        for word in all_words:
            if current["end"] is not None and word["start"] - current["end"] > 0.8:
                seg = _finalize_current(current)
                if seg:
                    result_segments.append(seg)
                current = {"speaker": speaker, "text": "", "start": word["start"], "end": None, "mood": None}
            current["text"] += word["word"]
            current["end"] = word["end"]

        seg = _finalize_current(current)
        if seg:
            result_segments.append(seg)

        # Filtrer hallucinations Whisper (phrases fréquemment inventées)
        WHISPER_HALLUCINATIONS = [
            "sous-titrage", "sous-titres réalisés", "amara.org",
            "merci d'avoir regardé", "abonnez-vous", "like et abonne",
            "sous-titres fait par", "sous-titrage société radio-canada",
            "www.", ".com", ".fr/", "©",
            "merci de votre attention", "n'hésitez pas à",
            "musique entraînante", "musique douce", "[musique]", "[rires]",
            "thank you for watching", "please subscribe",
            # Hallucinations fréquentes Whisper français sur du bruit / arabe / accent
            "monsieur le ministre", "ministre de l'éducation",
            "premier ministre", "président de la république",
            "le journal de", "vingt heures", "france info", "france inter",
        ]
        def _is_hallucination(text: str) -> bool:
            t = text.lower().strip()
            if not t:
                return True
            # 0) Segment composé uniquement de ponctuation/espaces (faster-whisper
            # marque les hallucinations détectées par hallucination_silence_threshold
            # avec un texte type "..." ou "... ...")
            if not re.search(r"[a-zàâäéèêëïîôöùûüÿç0-9]", t):
                return True
            # 1) Marqueur d'hallucination connue (toute taille)
            if any(h in t for h in WHISPER_HALLUCINATIONS):
                return True
            words = t.split()
            # 1bis) Tous les "mots" ne sont que des points/ponctuation
            content_words = [w for w in words if re.search(r"[a-zàâäéèêëïîôöùûüÿç0-9]", w)]
            if len(content_words) == 0:
                return True
            if len(words) >= 6:
                from collections import Counter
                c = Counter(words)
                most_common_count = c.most_common(1)[0][1]
                # 2) Boucle de mot unique : >55% des mots sont identiques
                if most_common_count / len(words) > 0.55:
                    return True
                # 3) Vocabulaire pauvre : ratio mots uniques < 35%
                unique_ratio = len(set(words)) / len(words)
                if len(words) >= 10 and unique_ratio < 0.35:
                    return True
            return False

        before_filter = len(result_segments)
        result_segments = [s for s in result_segments if not _is_hallucination(s["text"])]
        if before_filter != len(result_segments):
            log(f"[WHISPER] Filtered {before_filter - len(result_segments)} hallucinated segments on {speaker}")

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
def _llm_is_cached() -> bool:
    """Vérifie si le modèle LLM est déjà dans le cache HF local.
    Si oui, on charge avec local_files_only=True pour éviter les HEAD HF (~5-8s)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import _CACHED_NO_EXIST
        path = try_to_load_from_cache(LLM_MODEL_ID, "config.json")
        return path is not None and path is not _CACHED_NO_EXIST
    except Exception:
        return False


def _load_llm():
    global _llm_tokenizer, _llm_model
    if _llm_tokenizer is not None and _llm_model is not None:
        return _llm_tokenizer, _llm_model

    log(f"[LLM] Loading {LLM_MODEL_ID} [QUANT_MODE={QUANT_MODE}]")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_local_only = _llm_is_cached()
    if use_local_only:
        log("[LLM] Cache détecté → local_files_only=True (skip HF HEAD requests)")

    tok_kwargs = {"trust_remote_code": True, "local_files_only": use_local_only}
    if HF_TOKEN:
        tok_kwargs["token"] = HF_TOKEN
    _llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, **tok_kwargs)
    log("[LLM] Tokenizer loaded")

    mdl_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "local_files_only": use_local_only,
    }
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

CORRECT_CHUNK_SIZE = int(os.environ.get("CORRECT_CHUNK_SIZE", "10"))


def _correct_chunk(chunk_lines: List[str]) -> Optional[List[str]]:
    """Corrige un petit bloc de lignes 'Speaker: texte'.
    Retourne la liste corrigée si valide, None si rejet (appelant garde l'original)."""
    n = len(chunk_lines)
    chunk_text = "\n".join(chunk_lines)

    prompt = (
        "Tu es un correcteur de transcription téléphonique française d'un cabinet dentaire. "
        "Corrige UNIQUEMENT les erreurs de reconnaissance vocale évidentes avec le contexte "
        "(ex: 'désertage'/'détertraite' → 'détartrage', 'centre d'entraînement' → 'centre dentaire', "
        "noms propres mal orthographiés, mots qui n'ont pas de sens dans le contexte).\n"
        "RÈGLES STRICTES :\n"
        f"- Tu DOIS rendre EXACTEMENT {n} lignes, dans le même ordre\n"
        "- Chaque ligne DOIT commencer par 'Agent:' ou 'Client:' comme dans l'original\n"
        "- NE FUSIONNE PAS des lignes, NE SÉPARE PAS des lignes\n"
        "- Ne reformule pas, ne résume pas, ne supprime rien\n"
        "- Change uniquement les mots mal transcrits\n"
        "- Garde ponctuation et hésitations\n\n"
        f"Transcript ({n} lignes) :\n{chunk_text}\n\n"
        f"Transcript corrigé (EXACTEMENT {n} lignes, même format) :"
    )

    # Budget : longueur chunk + 20% marge
    max_tok = min(1500, max(256, len(chunk_text) // 2 + 128))
    corrected = run_llm(prompt, max_new_tokens=max_tok)
    if not corrected:
        return None

    cand = [l.strip() for l in corrected.split("\n") if l.strip()]
    cand = [l for l in cand if l.startswith("Agent:") or l.startswith("Client:")]

    if len(cand) != n:
        return None

    # Les speakers doivent matcher ligne à ligne
    for orig, corr in zip(chunk_lines, cand):
        if orig.split(":", 1)[0] != corr.split(":", 1)[0]:
            return None

    return cand


def correct_transcript(transcript: str) -> str:
    """Corrige les erreurs de reconnaissance Whisper via Mistral, par chunks.
    Si un chunk est rejeté, on garde l'original pour ce chunk uniquement.
    Retourne le transcript corrigé (potentiellement partiellement)."""
    if not transcript or len(transcript.split()) < 5:
        return transcript

    original_lines = [l for l in transcript.split("\n") if l.strip()]
    n_lines = len(original_lines)

    if n_lines <= CORRECT_CHUNK_SIZE:
        # Cas simple : un seul chunk
        cand = _correct_chunk(original_lines)
        if cand is None:
            log(f"[CORRECT] Chunk unique rejeté → keeping original ({n_lines} lines)")
            return transcript
        log(f"[CORRECT] Transcript corrected ({n_lines} lines, 1 chunk)")
        return "\n".join(cand)

    # Cas multi-chunks
    result_lines: List[str] = []
    n_chunks = (n_lines + CORRECT_CHUNK_SIZE - 1) // CORRECT_CHUNK_SIZE
    n_ok = 0
    n_ko = 0
    for i in range(0, n_lines, CORRECT_CHUNK_SIZE):
        chunk = original_lines[i:i + CORRECT_CHUNK_SIZE]
        cand = _correct_chunk(chunk)
        if cand is not None:
            result_lines.extend(cand)
            n_ok += 1
        else:
            # Garder l'original pour ce chunk
            result_lines.extend(chunk)
            n_ko += 1

    log(f"[CORRECT] Transcript corrected ({n_lines} lines, {n_ok}/{n_chunks} chunks ok, {n_ko} fallback)")
    return "\n".join(result_lines)


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
def remove_duplicate_segments(segments: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
    """Supprime les segments dupliqués (boucles Whisper) en respectant 2 règles :
    - Ne dédupplique JAMAIS entre speakers différents (Agent et Client peuvent dire les mêmes mots)
    - N'agit que sur des segments d'au moins 4 mots (sinon trop de faux positifs sur 'oui oui')
    - Seuil de similarité 0.85 (au lieu de 0.7) pour être plus strict
    """
    if len(segments) <= 1:
        return segments
    def text_similarity(a, b):
        if not a or not b: return 0.0
        words_a, words_b = set(a.lower().split()), set(b.lower().split())
        if not words_a or not words_b: return 0.0
        return len(words_a & words_b) / max(len(words_a), len(words_b))

    # Indexer par speaker : on dédupplique au sein d'un même speaker uniquement
    seen_by_speaker: Dict[str, List[str]] = {}
    cleaned = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "")
        # Ne dédupplique que les segments significatifs (≥ 4 mots)
        if len(text.split()) < 4:
            cleaned.append(seg)
            continue
        seen_list = seen_by_speaker.setdefault(speaker, [])
        if any(text_similarity(text, s) >= similarity_threshold for s in seen_list):
            continue
        cleaned.append(seg)
        seen_list.append(text)
    if len(cleaned) < len(segments):
        log(f"[DEDUP] Removed {len(segments) - len(cleaned)} duplicate segments (within-speaker only)")
    return cleaned


# =============================================================================
# HELPERS RÉPONSE
# =============================================================================
HOLD_MUSIC_LABEL = "Annonce musicale / musique d'attente"


def _is_repetitive_announcement(text: str, duration: float) -> bool:
    """Détecte si un texte ressemble à une annonce IVR répétée (ex: 'Veuillez patienter' × 10).
    Conditions strictes pour éviter les faux positifs sur conversations naturelles :
    - Durée minimum : 2 min (les vraies annonces durent ≥ 2-3 min)
    - Au moins 50 mots de texte
    - ET (ratio mots uniques < 20% OU n-gram de 8 mots répété ≥ 4 fois)
    """
    if not text or duration < 120:  # Pas d'annonce pour les appels < 2 min
        return False

    # Nettoyer : retirer "..." et ponctuations
    clean = re.sub(r"[.\!?,;:\-\"\'\(\)]+", " ", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    words = clean.split()
    if len(words) < 50:
        return False

    unique_ratio = len(set(words)) / len(words)

    # Critère 1 : ratio mots uniques très bas (< 20%)
    if unique_ratio < 0.20:
        return True

    # Critère 2 : n-gram de 8 mots répété ≥ 4 fois (signature claire d'annonce IVR bouclée)
    if len(words) >= 40:
        from collections import Counter
        ngrams = [" ".join(words[i:i+8]) for i in range(len(words) - 7)]
        most_common = Counter(ngrams).most_common(1)
        if most_common and most_common[0][1] >= 4:
            return True

    # Critère 3 : durée très longue (>5 min) avec très peu de vocabulaire
    if duration >= 300 and len(set(words)) < 60:
        return True

    return False


def _build_hold_music_response(duration: float, diarization_mode: str) -> Dict[str, Any]:
    """Réponse standard quand tout l'audio est détecté comme musique d'attente.
    Fournit un transcript et un résumé explicites (pas vides) pour que l'UI affiche
    quelque chose de cohérent."""
    neutral_mood = {
        "label_en": "neutral", "label_fr": "neutre", "confidence": 0.5,
        "scores": {"negative": 0.2, "neutral": 0.6, "positive": 0.2},
    }
    segment = {
        "speaker": "System",
        "start": 0.0,
        "end": round(duration, 2),
        "text": HOLD_MUSIC_LABEL,
        "mood": neutral_mood,
    }
    return {
        "task": "transcribe_diarized",
        "segments": [segment],
        "transcript": f"System: {HOLD_MUSIC_LABEL}",
        "summary": HOLD_MUSIC_LABEL,
        "diarization_mode": diarization_mode,
        "audio_duration": round(duration, 2),
        "hold_music_detected": True,
        "mood_overall": neutral_mood,
        "mood_client": neutral_mood,
        "mood_by_speaker": {"System": neutral_mood},
    }


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
                    return _build_hold_music_response(est_dur, "stereo_hold_music")
            else:
                hold = detect_hold_music(local_path)
                if hold["is_hold_music"]:
                    return _build_hold_music_response(est_dur, "mono_hold_music")

        # ── TRANSCRIPTION ──────────────────────────────────────────────────
        if is_stereo:
            log(f"[HANDLER] Stereo audio → Whisper per-channel with word timestamps")

            # Détection des zones de hold music par canal (pour muter cross-canal)
            hold_regions_ch0: List[Tuple[float, float]] = []
            hold_regions_ch1: List[Tuple[float, float]] = []
            hold_coverage_ch0 = 0.0
            hold_coverage_ch1 = 0.0
            if ENABLE_HOLD_MUSIC_DETECTION:
                try:
                    import soundfile as sf
                    data, sr_full = sf.read(local_path)
                    if data.ndim > 1:
                        hold_regions_ch0 = _detect_hold_regions_in_channel(data[:, 0].astype("float32"), sr_full)
                        hold_regions_ch1 = _detect_hold_regions_in_channel(data[:, 1].astype("float32"), sr_full)
                        total_ch0 = sum(e - s for s, e in hold_regions_ch0)
                        total_ch1 = sum(e - s for s, e in hold_regions_ch1)
                        hold_coverage_ch0 = total_ch0 / est_dur if est_dur > 0 else 0
                        hold_coverage_ch1 = total_ch1 / est_dur if est_dur > 0 else 0
                        if hold_regions_ch0:
                            log(f"[HOLD_REGIONS] CH0 (Client): {len(hold_regions_ch0)} zones, {total_ch0:.1f}s ({hold_coverage_ch0*100:.0f}%)")
                        if hold_regions_ch1:
                            log(f"[HOLD_REGIONS] CH1 (Agent): {len(hold_regions_ch1)} zones, {total_ch1:.1f}s ({hold_coverage_ch1*100:.0f}%)")
                    del data
                except Exception as e:
                    log(f"[HOLD_REGIONS] Error: {e}")

            # Short-circuit : si un canal est 100% hold ET l'autre >80% hold → annonce/attente
            both_mostly_hold = (hold_coverage_ch0 >= 0.80 and hold_coverage_ch1 >= 0.80)
            one_full_hold_other_mostly = (
                (hold_ch0.get("is_hold_music") and hold_coverage_ch1 >= 0.75) or
                (hold_ch1.get("is_hold_music") and hold_coverage_ch0 >= 0.75)
            )
            if both_mostly_hold or one_full_hold_other_mostly:
                log(f"[HANDLER] Audio dominé par hold/silence (ch0={hold_coverage_ch0*100:.0f}%, ch1={hold_coverage_ch1*100:.0f}%) → hold music response")
                return _build_hold_music_response(est_dur, "stereo_hold_dominant")

            # Décider si on applique le cross-mute :
            # - Si UN seul canal est globalement hold music → cross-mute OK (cas typique : client en attente)
            # - Si LES DEUX ont des hold regions sans qu'aucun soit globalement hold → conversation
            #   normale avec pauses, on ne mute PAS (sinon on coupe des paroles légitimes)
            apply_cross_mute = (
                hold_ch0.get("is_hold_music") or hold_ch1.get("is_hold_music")
                or hold_coverage_ch0 >= 0.50 or hold_coverage_ch1 >= 0.50
            )
            if not apply_cross_mute:
                if hold_regions_ch0 or hold_regions_ch1:
                    log(f"[HANDLER] Conversation bidirectionnelle (ch0={hold_coverage_ch0*100:.0f}%, ch1={hold_coverage_ch1*100:.0f}%) → cross-mute désactivé")
                hold_regions_ch0 = []
                hold_regions_ch1 = []

            # Skip un canal UNIQUEMENT s'il est quasi-entièrement hold music (≥ 95%).
            # Sinon on le transcrit quand même : Whisper a son propre VAD qui sautera
            # les zones de musique/silence, et on garde les vraies paroles s'il y en a.
            FULL_HOLD_THRESHOLD = 0.95

            if ENABLE_HOLD_MUSIC_DETECTION and hold_coverage_ch0 >= FULL_HOLD_THRESHOLD:
                log(f"[HANDLER] CH0 (Client) is dominantly hold music ({hold_coverage_ch0*100:.0f}%) → skipping Whisper")
                client_segs = []
            else:
                # Pendant que l'agent est en hold music, muter ch0 (pas de conv)
                client_segs = _transcribe_channel_whisper(
                    local_path, 0, "Client", language,
                    mute_regions=hold_regions_ch1 or None,
                )

            if ENABLE_HOLD_MUSIC_DETECTION and hold_coverage_ch1 >= FULL_HOLD_THRESHOLD:
                log(f"[HANDLER] CH1 (Agent) is dominantly hold music ({hold_coverage_ch1*100:.0f}%) → skipping Whisper")
                agent_segs = []
            else:
                # Pendant que le client est en hold music, muter ch1 (agent ne parle pas au client)
                agent_segs = _transcribe_channel_whisper(
                    local_path, 1, "Agent", language,
                    mute_regions=hold_regions_ch0 or None,
                )

            segments = client_segs + agent_segs
            segments.sort(key=lambda s: s["start"])

            if not segments:
                # Si on a des hold regions significatives, c'est probablement une annonce/attente
                # → ne PAS fallback sur mono (qui re-transcrirait l'annonce)
                if hold_coverage_ch0 >= 0.5 or hold_coverage_ch1 >= 0.5 or \
                   hold_ch0.get("is_hold_music") or hold_ch1.get("is_hold_music"):
                    log("[HANDLER] No stereo segments + hold detected → hold music response")
                    return _build_hold_music_response(est_dur, "stereo_no_segments_hold")

                # Sinon, vrai fallback mono (cas où la transcription stéréo a raté sans raison)
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
            # Rien de transcriptible → probablement uniquement de la musique/silence
            log("[HANDLER] No segments detected → treating as hold music / silence")
            return _build_hold_music_response(est_dur, diarization_mode + "_empty")

        # Dedup
        segments = remove_duplicate_segments(segments)

        # Fusionner segments consécutifs même speaker, MAIS uniquement s'ils sont
        # proches dans le temps (gap < 2.5s). Préserve les pauses naturelles et
        # évite de fusionner toute une longue suite de phrases en un seul gros bloc.
        MERGE_MAX_GAP_S = 2.5
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            gap = float(seg.get("start", 0)) - float(prev.get("end", 0))
            if prev["speaker"] == seg["speaker"] and gap < MERGE_MAX_GAP_S:
                prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
                prev["end"] = seg["end"]
            else:
                merged.append(seg)

        # Compresser les répétitions internes dans chaque segment mergé
        # (rattrape les boucles créées par la fusion de segments adjacents identiques)
        for seg in merged:
            if seg.get("text"):
                before = seg["text"]
                seg["text"] = _compress_internal_repetitions(before)
                if seg["text"] != before:
                    log(f"[COMPRESS] {seg.get('speaker')}: {len(before)} → {len(seg['text'])} chars")

        # Filtrer les segments vides ou ne contenant que de la ponctuation
        # (faster-whisper avec hallucination_silence_threshold marque les hallucinations
        # par un texte "..." qu'il faut éliminer définitivement)
        before_clean = len(merged)
        merged = [
            s for s in merged
            if s.get("text", "").strip()
            and re.search(r"[a-zàâäéèêëïîôöùûüÿç0-9]", s["text"], re.IGNORECASE)
        ]
        if len(merged) != before_clean:
            log(f"[CLEAN] Removed {before_clean - len(merged)} empty/dots-only segments")

        log(f"[HANDLER] Final: {len(merged)} segments")

        # Détection d'annonce IVR répétée — UNIQUEMENT si monologue (1 seul speaker)
        # Une vraie conversation a 2 speakers, ce n'est pas une annonce.
        speakers_present = set(s.get("speaker") for s in merged if s.get("text"))
        if len(speakers_present) <= 1:
            full_text_for_check = " ".join(s.get("text", "") for s in merged)
            if _is_repetitive_announcement(full_text_for_check, est_dur):
                log(f"[HANDLER] Repetitive announcement detected ({len(full_text_for_check)} chars, monologue) → hold music response")
                return _build_hold_music_response(est_dur, "stereo_repetitive_announcement")

        full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in merged if s.get("text"))

        # ── CORRECTION DU TRANSCRIPT ───────────────────────────────────────
        # On ne corrige que si le transcript vaut la peine :
        # - au moins 3 segments OU 200 chars (sinon c'est court, peu d'erreurs probables)
        n_chars_transcript = len(full_transcript)
        n_segs_text = sum(1 for s in merged if s.get("text"))
        if ENABLE_TRANSCRIPT_CORRECTION and full_transcript and (n_segs_text >= 3 or n_chars_transcript >= 200):
            log(f"[HANDLER] Correcting transcript ({n_segs_text} segs, {n_chars_transcript} chars)...")
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
    import threading
    log("[INIT] Starting preload (LLM + Whisper en parallèle)...")
    log(f"[INIT] QUANT_MODE={QUANT_MODE} | APP_VERSION={APP_VERSION}")

    _init_t0 = time.time()
    _init_errors = {}

    def _init_llm_thread():
        try:
            _load_llm()
        except Exception as e:
            _init_errors["llm"] = e
            log(f"[INIT] LLM load error: {type(e).__name__}: {e}")

    def _init_whisper_thread():
        try:
            _load_whisper()
        except Exception as e:
            _init_errors["whisper"] = e
            log(f"[INIT] Whisper load error: {type(e).__name__}: {e}")

    t_llm = threading.Thread(target=_init_llm_thread, name="init-llm")
    t_whisper = threading.Thread(target=_init_whisper_thread, name="init-whisper")
    t_llm.start()
    t_whisper.start()
    t_llm.join()
    t_whisper.join()

    _init_dt = round(time.time() - _init_t0, 2)
    log(f"[INIT] Parallel load completed in {_init_dt}s (errors={list(_init_errors.keys()) or 'none'})")
    log(f"[INIT] Hold music detection: {'ENABLED' if ENABLE_HOLD_MUSIC_DETECTION else 'DISABLED'}")
    log(f"[INIT] Sentiment analysis: {'ENABLED' if ENABLE_SENTIMENT else 'DISABLED'}")
    log(f"[INIT] Whisper model: {WHISPER_MODEL_SIZE}")
    log(f"[INIT] LLM model: {LLM_MODEL_ID}")
    log("[INIT] Ready.")
except Exception as e:
    log(f"[WARN] Preload failed: {e}")

runpod.serverless.start({"handler": handler})
