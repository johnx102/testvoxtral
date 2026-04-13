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

# Debug Whisper : log les segments bruts AVANT tous nos filtres post-process.
# Désactivé par défaut. Activer avec DEBUG_WHISPER_RAW=1 si besoin.
DEBUG_WHISPER_RAW = os.environ.get("DEBUG_WHISPER_RAW", "0") == "1"

# Audio enhancement : normalise + compresse dynamiquement chaque canal avant Whisper.
# Booste les voix faibles (ex: client loin du combiné) sans amplifier le bruit.
ENABLE_AUDIO_ENHANCEMENT = os.environ.get("ENABLE_AUDIO_ENHANCEMENT", "1") == "1"

# Whisper : seul modèle utilisé en production
#
# bofenghuang/whisper-large-v3-french-distil-dec16 :
# - Distillé (16 décodeurs au lieu de 32) → ~2× plus rapide que large-v3
# - Fine-tuné sur 5 datasets français (~97k samples)
# - WER 7.18% sur CommonVoice FR (vs ~10% pour large-v3 standard)
# - WER 3.57% sur Multilingual LibriSpeech FR
# - Réduit les hallucinations sur les transcriptions longues
WHISPER_REPO_ID    = "bofenghuang/whisper-large-v3-french-distil-dec16"
WHISPER_LOCAL_DIR  = "/app/.cache/whisper-french-distil-dec16"
WHISPER_CT2_PATH   = WHISPER_LOCAL_DIR + "/ctranslate2"
WHISPER_DEVICE     = os.environ.get("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE    = os.environ.get("WHISPER_COMPUTE", "float16")

# Initial prompt Whisper : sert de "vocabulaire de biais" pour orienter le décodage.
# Quand Whisper hésite entre 2 mots qui sonnent pareils, il privilégie ceux du prompt.
# Couvre les domaines : téléphonie pro, médical, dentaire, commercial/vente.
# Override possible via env var WHISPER_INITIAL_PROMPT pour les clients spécialisés.
WHISPER_DEFAULT_PROMPT = (
    "Conversation téléphonique professionnelle en français entre un client et un agent. "
    "Cabinet médical ou dentaire, secrétaire, accueil, prise de rendez-vous. "
    "Vocabulaire courant : bonjour, madame, monsieur, docteur, médecin, patient, "
    "rendez-vous, urgence, douleur, dent, dent de sagesse, prothèse dentaire, "
    "détartrage, ordonnance, mutuelle, devis, facture, commande, livraison, tarif, "
    "paiement, client, assistante, joignable, rappeler, message, répondeur, "
    "merci, au revoir, bonne journée, à bientôt."
)
WHISPER_INITIAL_PROMPT = os.environ.get("WHISPER_INITIAL_PROMPT", "").strip() or WHISPER_DEFAULT_PROMPT

# LLM texte (résumé, sentiment, correction)
# Forcé en dur — ne pas utiliser d'env var pour éviter les overrides RunPod
LLM_MODEL_ID       = "Qwen/Qwen2.5-14B-Instruct"
# INT4 NF4 : ~8GB VRAM + 3GB Whisper = 11GB → large marge dans 24GB
# INT4 est souvent PLUS RAPIDE que INT8 en génération (moins de mémoire à lire
# par layer → meilleur débit mémoire). Qualité légèrement inférieure à INT8
# mais toujours très supérieure au Ministral 8B.
QUANT_MODE         = "bnb4"

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

def _enhance_audio_for_whisper(wav_path: str) -> str:
    """Normalise et compresse dynamiquement un fichier WAV mono pour aider Whisper.

    Pourquoi : sur les enregistrements téléphoniques, le client peut parler loin
    du combiné au début (voix faible) puis se rapprocher (voix forte). Whisper
    rate alors les premières phrases. La compression dynamique réduit l'écart
    entre passages calmes et forts.

    Compromis validé sur audio Marc/Schreiber 145s :
    - compression 4:1 threshold -20dB : booste les voix faibles SANS amplifier
      excessivement le bruit (qui produirait des hallucinations)
    - normalize headroom 0.5 : pic à -0.5 dB
    - PAS de gain supplémentaire (testé +3dB → trop d'hallucinations)

    Si pydub n'est pas dispo ou erreur → retourne le fichier original.
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize, compress_dynamic_range
        audio = AudioSegment.from_wav(wav_path)
        # Compression modérée : booste les voix faibles sans saturer le bruit
        audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        # Normalisation : ramène le pic à -0.5 dB
        audio = normalize(audio, headroom=0.5)
        out_path = wav_path.replace(".wav", "_enhanced.wav")
        if out_path == wav_path:
            out_path = wav_path + ".enhanced.wav"
        audio.export(out_path, format="wav")
        return out_path
    except Exception as e:
        log(f"[ENHANCE] Audio enhancement failed: {e} (using original)")
        return wav_path


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
# WHISPER — Transcription (bofenghuang french distillé)
# =============================================================================
def _ensure_whisper_model() -> str:
    """S'assure que le modèle bofenghuang français est présent localement.
    Télécharge le sous-dossier ctranslate2/ si absent. Retourne le chemin local."""
    if os.path.exists(os.path.join(WHISPER_CT2_PATH, "model.bin")):
        log(f"[WHISPER] Model cached at {WHISPER_CT2_PATH}")
        return WHISPER_CT2_PATH

    from huggingface_hub import snapshot_download
    log(f"[WHISPER] Downloading {WHISPER_REPO_ID} (ctranslate2 subfolder)...")
    snapshot_download(
        repo_id=WHISPER_REPO_ID,
        local_dir=WHISPER_LOCAL_DIR,
        allow_patterns="ctranslate2/*",
    )
    log(f"[WHISPER] Model ready at {WHISPER_CT2_PATH}")
    return WHISPER_CT2_PATH


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        ct2_path = _ensure_whisper_model()
        _whisper_model = WhisperModel(
            ct2_path,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        log(f"[WHISPER] Model loaded successfully ({WHISPER_REPO_ID})")
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

    # Vectoriser le calcul RMS par fenêtre avec torch (GPU si dispo).
    # L'ancienne boucle Python sur N fenêtres × M frames prenait ~10-30s
    # sur un audio de 20 min. La version torch prend ~0.5-2s.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        audio_t = torch.from_numpy(np.ascontiguousarray(audio_np, dtype=np.float32)).to(device)
        # Pré-calculer le RMS par micro-frame (30ms, hop 10ms) sur tout l'audio
        micro_frames = audio_t.unfold(0, frame_len, frame_hop)  # (n_micro, frame_len)
        micro_rms = micro_frames.pow(2).mean(dim=1).sqrt()  # (n_micro,)
        # Pour chaque fenêtre macro (5s, hop 2.5s), extraire les micro-RMS correspondants
        n_micro_per_win = 1 + (win - frame_len) // frame_hop
        n_micro_per_hop = hop // frame_hop
        total_micro = micro_rms.shape[0]
        del micro_frames, audio_t
    except Exception:
        # Fallback numpy si torch échoue
        device = None
        micro_rms = None

    regions_raw: List[Tuple[float, float]] = []
    for idx, i in enumerate(range(0, len(audio_np) - win + 1, hop)):
        if device and micro_rms is not None:
            # Chemin rapide : extraire les micro-RMS pré-calculés pour cette fenêtre
            micro_start = idx * n_micro_per_hop
            micro_end = min(micro_start + n_micro_per_win, total_micro)
            if micro_end - micro_start < 10:
                continue
            rms_slice = micro_rms[micro_start:micro_end]
            chunk_t = torch.from_numpy(audio_np[i:i + win]).to(device)
            local_peak = float(chunk_t.abs().max())
            del chunk_t
        else:
            # Fallback numpy
            chunk = audio_np[i:i + win]
            n_frames_np = 1 + (len(chunk) - frame_len) // frame_hop
            if n_frames_np < 10:
                continue
            rms_slice = torch.tensor([np.sqrt(np.mean(chunk[j*frame_hop:j*frame_hop+frame_len]**2)) for j in range(n_frames_np)])
            local_peak = float(np.max(np.abs(chunk)))

        if local_peak < 0.05 * global_peak:
            is_hold = True
        else:
            rms_norm = rms_slice / local_peak
            threshold = float(rms_norm.median()) + 1.5 * float(rms_norm.std())
            speech_ratio = float((rms_norm > threshold).sum()) / len(rms_norm)
            is_hold = speech_ratio < speech_ratio_threshold

        if is_hold:
            regions_raw.append((i / sr, (i + win) / sr))

    if device and micro_rms is not None:
        del micro_rms
        if device == "cuda":
            torch.cuda.empty_cache()

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


def _detect_loop_regions(
    audio_np,
    sr: int,
    min_loop_s: float = 15.0,
    max_loop_s: float = 90.0,
    global_threshold: float = 0.95,
    per_second_threshold: float = 0.96,
    min_region_s: float = 10.0,
    silence_energy_ratio: float = 0.10,
    min_active_seconds: int = 25,
    min_coverage_ratio: float = 0.40,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Détecte les RÉGIONS temporelles où un canal contient une annonce IVR ou
    musique en boucle, via auto-similarité spectrale.

    Méthode (PyTorch CUDA si dispo, fallback numpy) :
      1. STFT magnitude (frames 50ms, hop 25ms)
      2. Réduction à 24 bandes log-fréquence
      3. Aggrégation par seconde → 1 vecteur log-spectral / seconde (normalisé L2)
      4. Self-similarity matrix cosinus (matmul GPU = quasi-instantané)
      5. Recherche de la meilleure période k (15s..90s) sur diagonales décalées
      6. Pour chaque seconde t, marquage "in-loop" si sim(t, t±k) ≥ seuil
      7. Fusion en régions contiguës ≥ min_region_s

    Returns (regions, best_period_s, global_loop_score)
    """
    import numpy as np

    if audio_np is None or len(audio_np) < int(sr * min_loop_s * 2.5):
        return ([], 0.0, 0.0)

    peak = float(np.max(np.abs(audio_np)))
    if peak < 1e-4:
        return ([], 0.0, 0.0)

    # Utiliser PyTorch + CUDA si disponible (10-50× plus rapide que numpy sur
    # les grosses FFT et matmul). Fallback numpy si CUDA indisponible.
    use_torch = torch.cuda.is_available()

    frame_len = int(0.050 * sr)
    hop = int(0.025 * sr)
    n_fft = 1024 if sr >= 16000 else 512
    n_frames = 1 + (len(audio_np) - frame_len) // hop
    if n_frames < 80:
        return ([], 0.0, 0.0)

    try:
        if use_torch:
            # ── PyTorch CUDA path ──
            device = "cuda"
            audio_t = torch.from_numpy(np.ascontiguousarray(audio_np, dtype=np.float32)).to(device)
            window_t = torch.hann_window(frame_len, device=device)

            # STFT via torch.stft (retourne complex tensor)
            stft_out = torch.stft(
                audio_t, n_fft=n_fft, hop_length=hop, win_length=frame_len,
                window=window_t, return_complex=True, center=False,
            )
            spec = stft_out.abs().T  # (n_frames_stft, n_fft//2+1)
            n_frames = spec.shape[0]

            # Réduction en 24 bandes log-fréquence
            n_bins = spec.shape[1]
            n_bands = 24
            freqs = torch.linspace(0, sr / 2, n_bins, device=device)
            f_max = max(sr / 2 - 1, 200)
            log_edges = torch.from_numpy(np.geomspace(80, f_max, n_bands + 1).astype(np.float32)).to(device)
            bands = torch.zeros((n_frames, n_bands), device=device)
            for i in range(n_bands):
                mask = (freqs >= log_edges[i]) & (freqs < log_edges[i + 1])
                if mask.any():
                    bands[:, i] = spec[:, mask].mean(dim=1)
            log_bands = torch.log1p(bands)

            # Aggrégation par seconde
            frames_per_sec = int(round(1.0 / 0.025))
            n_sec = n_frames // frames_per_sec
            if n_sec < int(min_loop_s * 2.5):
                return ([], 0.0, 0.0)
            sec_vectors = log_bands[: n_sec * frames_per_sec].reshape(n_sec, frames_per_sec, n_bands).mean(dim=1)

            # Énergie par seconde (pour filtrage silence)
            samples_per_sec = sr
            sec_energy = torch.zeros(n_sec, device=device)
            for t in range(n_sec):
                s_idx = t * samples_per_sec
                e_idx = min(s_idx + samples_per_sec, len(audio_t))
                if e_idx > s_idx:
                    chunk = audio_t[s_idx:e_idx]
                    sec_energy[t] = torch.sqrt(torch.mean(chunk * chunk))
            energy_peak = float(sec_energy.max())
            if energy_peak < 1e-6:
                return ([], 0.0, 0.0)
            active_mask_t = sec_energy >= (silence_energy_ratio * energy_peak)
            n_active = int(active_mask_t.sum())
            if n_active < min_active_seconds:
                return ([], 0.0, 0.0)

            # Normalisation L2 + self-similarity (matmul GPU = quasi-instantané)
            norms = sec_vectors.norm(dim=1, keepdim=True) + 1e-9
            sec_vectors = sec_vectors / norms
            sim = sec_vectors @ sec_vectors.T  # (n_sec, n_sec) — GPU matmul

            # Convertir vers numpy pour la boucle de recherche (petite, pas de bottleneck)
            sim = sim.cpu().numpy()
            active_mask = active_mask_t.cpu().numpy()
            del audio_t, stft_out, spec, bands, log_bands, sec_vectors, sec_energy, active_mask_t
            torch.cuda.empty_cache()
        else:
            # ── Numpy fallback path ──
            window = np.hanning(frame_len).astype(np.float32)
            audio_f = np.ascontiguousarray(audio_np, dtype=np.float32)
            frames = np.lib.stride_tricks.as_strided(
                audio_f,
                shape=(n_frames, frame_len),
                strides=(audio_f.strides[0] * hop, audio_f.strides[0]),
            )
            frames_w = frames * window
            spec = np.abs(np.fft.rfft(frames_w, n=n_fft, axis=1))

            n_bins = spec.shape[1]
            n_bands = 24
            freqs = np.linspace(0, sr / 2, n_bins)
            f_max = max(sr / 2 - 1, 200)
            log_edges = np.geomspace(80, f_max, n_bands + 1)
            bands = np.zeros((n_frames, n_bands), dtype=np.float32)
            for i in range(n_bands):
                mask = (freqs >= log_edges[i]) & (freqs < log_edges[i + 1])
                if mask.any():
                    bands[:, i] = spec[:, mask].mean(axis=1)
            log_bands = np.log1p(bands)

            frames_per_sec = int(round(1.0 / 0.025))
            n_sec = n_frames // frames_per_sec
            if n_sec < int(min_loop_s * 2.5):
                return ([], 0.0, 0.0)
            sec_vectors = log_bands[: n_sec * frames_per_sec].reshape(n_sec, frames_per_sec, n_bands).mean(axis=1)

            sec_energy = np.zeros(n_sec, dtype=np.float32)
            samples_per_sec = sr
            for t in range(n_sec):
                s_idx = t * samples_per_sec
                e_idx = min(s_idx + samples_per_sec, len(audio_np))
                if e_idx > s_idx:
                    chunk = audio_np[s_idx:e_idx]
                    sec_energy[t] = float(np.sqrt(np.mean(chunk * chunk)))
            energy_peak = float(np.max(sec_energy)) if len(sec_energy) > 0 else 0.0
            if energy_peak < 1e-6:
                return ([], 0.0, 0.0)
            active_mask = sec_energy >= (silence_energy_ratio * energy_peak)
            n_active = int(np.sum(active_mask))
            if n_active < min_active_seconds:
                return ([], 0.0, 0.0)

            norms = np.linalg.norm(sec_vectors, axis=1, keepdims=True) + 1e-9
            sec_vectors = sec_vectors / norms
            sim = sec_vectors @ sec_vectors.T
    except Exception:
        return ([], 0.0, 0.0)

    # Recherche meilleure période k (CPU, petite boucle ~75 itérations)
    min_k = max(int(min_loop_s), 5)
    max_k = min(int(max_loop_s), n_sec - 5)
    if max_k <= min_k:
        return ([], 0.0, 0.0)

    best_score = 0.0
    best_k = 0
    for k in range(min_k, max_k + 1):
        valid = active_mask[: n_sec - k] & active_mask[k:]
        n_valid = int(np.sum(valid))
        if n_valid < 8:
            continue
        diag = np.diagonal(sim, offset=k)
        score = float(np.mean(diag[valid]))
        if score > best_score:
            best_score = score
            best_k = k

    # Si même la meilleure diagonale est faible → pas de boucle
    if best_score < global_threshold or best_k == 0:
        return ([], float(best_k), float(best_score))

    # Pour chaque seconde ACTIVE, vérifier si elle est dans une zone qui boucle.
    # Les secondes silencieuses ne sont jamais "in-loop" (même si elles ressemblent
    # à d'autres secondes silencieuses, ce n'est pas une vraie boucle d'annonce).
    in_loop = np.zeros(n_sec, dtype=bool)
    for t in range(n_sec):
        if not active_mask[t]:
            continue
        candidates = []
        for offset in (best_k, -best_k, 2 * best_k, -2 * best_k):
            tt = t + offset
            if 0 <= tt < n_sec and active_mask[tt]:
                candidates.append(sim[t, tt])
        if candidates and max(candidates) >= per_second_threshold:
            in_loop[t] = True

    # Fusion en régions (autoriser de petits trous ≤ 2s à l'intérieur d'une zone)
    regions: List[Tuple[float, float]] = []
    t = 0
    while t < n_sec:
        if in_loop[t]:
            start = t
            gap = 0
            end = t
            while t < n_sec:
                if in_loop[t]:
                    end = t
                    gap = 0
                else:
                    gap += 1
                    if gap > 2:
                        break
                t += 1
            regions.append((float(start), float(end + 1)))  # +1 pour inclure la dernière sec
        else:
            t += 1

    # Filtrer les régions trop courtes
    regions = [(s, e) for s, e in regions if (e - s) >= min_region_s]

    # Garde-fou final : la couverture totale des régions détectées doit être ≥
    # min_coverage_ratio de la durée du canal. Une vraie annonce IVR couvre
    # typiquement 60-100% du canal (le client est en attente longtemps), tandis
    # qu'un faux positif sur conversation normale ne couvre qu'une petite partie
    # (juste quelques secondes où le motif spectral est par hasard cohérent).
    if regions:
        total_coverage = sum(e - s for s, e in regions)
        channel_duration = n_sec  # 1 vector par seconde
        coverage_ratio = total_coverage / channel_duration if channel_duration > 0 else 0
        if coverage_ratio < min_coverage_ratio:
            # Couverture trop faible → c'est très probablement un faux positif
            return ([], float(best_k), float(best_score))

    return (regions, float(best_k), float(best_score))


def _transcribe_channel_whisper(wav_path: str, channel: int, speaker: str, language: str = "fr",
                                  mute_regions: Optional[List[Tuple[float, float]]] = None,
                                  initial_prompt: Optional[str] = None) -> List[Dict]:
    """Transcrit un canal avec FasterWhisper + word_timestamps. Regroupe par pauses.
    Si mute_regions est fourni, les zones correspondantes sont mises à zéro avant transcription
    (utile pour skipper ce que dit l'agent pendant que le client est en hold music).
    Si initial_prompt est fourni, override le prompt par défaut WHISPER_INITIAL_PROMPT.

    Optimisation : quand les mute_regions couvrent > 50% de l'audio, au lieu de
    mettre les échantillons à zéro et d'envoyer le fichier complet (Whisper + VAD
    doivent parcourir tout le fichier), on EXTRAIT uniquement les portions non-mutées
    et on les concatène en un WAV court. Les timestamps sont remappés vers l'audio
    original après transcription. Gain typique : 10× sur un appel de 20 min dont
    18 min sont mutées.
    """
    # Variables pour le remapping de timestamps (utilisées plus bas si on fait
    # l'extraction optimisée). Par défaut : pas de remapping.
    timestamp_remap_regions = None  # list of (original_start, short_start, duration)

    if mute_regions:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            mono = data[:, channel].astype(np.float32)
        else:
            mono = data.astype(np.float32)
        total_duration = len(mono) / sr
        total_muted = sum(min(e, total_duration) - max(s, 0) for s, e in mute_regions)
        mute_ratio = total_muted / total_duration if total_duration > 0 else 0

        if mute_ratio >= 0.50 and total_muted > 30:
            # ── Optimisation : extraire UNIQUEMENT les portions non-mutées ──
            # Calculer les régions à GARDER (inverse des mute_regions)
            mute_sorted = sorted(mute_regions, key=lambda r: r[0])
            keep_regions: List[Tuple[float, float]] = []
            prev_end = 0.0
            for ms, me in mute_sorted:
                ms = max(ms, 0.0)
                me = min(me, total_duration)
                if ms > prev_end + 0.01:
                    keep_regions.append((prev_end, ms))
                prev_end = max(prev_end, me)
            if prev_end < total_duration - 0.01:
                keep_regions.append((prev_end, total_duration))

            # Filtrer les keep regions trop courtes (< 5s) : ce sont des micro-trous
            # entre les zones mute, contenant du bruit/silence mais pas de vraie
            # speech. Les envoyer à Whisper génère des hallucinations et des
            # timestamps incohérents quand ils sont concaténés.
            MIN_KEEP_REGION_S = 5.0
            keep_regions = [(s, e) for s, e in keep_regions if (e - s) >= MIN_KEEP_REGION_S]

            if keep_regions:
                # Extraire et concaténer les portions à garder
                chunks = []
                timestamp_remap_regions = []
                short_offset = 0.0
                for ks, ke in keep_regions:
                    s_idx = max(0, int(ks * sr))
                    e_idx = min(len(mono), int(ke * sr))
                    if e_idx > s_idx:
                        chunk = mono[s_idx:e_idx]
                        chunk_dur = len(chunk) / sr
                        timestamp_remap_regions.append((ks, short_offset, chunk_dur))
                        chunks.append(chunk)
                        short_offset += chunk_dur

                if chunks:
                    short_audio = np.concatenate(chunks)
                    ch_path = os.path.join(tempfile.gettempdir(), f"trimmed_ch{channel}_{uuid.uuid4().hex}.wav")
                    sf.write(ch_path, short_audio, sr)
                    kept_dur = len(short_audio) / sr
                    log(f"[WHISPER] {speaker} (ch{channel}): extracted {kept_dur:.1f}s from {total_duration:.1f}s "
                        f"({len(keep_regions)} regions, skipped {total_muted:.1f}s muted)")
                else:
                    # Fallback : pas de chunk exploitable → fichier vide
                    ch_path = _extract_mono_channel(wav_path, channel)
                    timestamp_remap_regions = None
            else:
                # Tout est muté → rien à transcrire
                return []
        else:
            # Mute faible (< 50%) : comportement existant (zéro les samples)
            mono = _mute_audio_regions(mono, sr, mute_regions)
            ch_path = os.path.join(tempfile.gettempdir(), f"muted_ch{channel}_{uuid.uuid4().hex}.wav")
            sf.write(ch_path, mono, sr)
            log(f"[WHISPER] {speaker} (ch{channel}): masked {total_muted:.1f}s of cross-channel hold music")
    else:
        ch_path = _extract_mono_channel(wav_path, channel)

    # Audio enhancement : normalise + compresse pour booster les voix faibles
    # (utile quand le client est loin du combiné au début de l'appel)
    enhanced_path = None
    if ENABLE_AUDIO_ENHANCEMENT:
        enhanced_path = _enhance_audio_for_whisper(ch_path)
        if enhanced_path != ch_path:
            ch_path_for_whisper = enhanced_path
        else:
            ch_path_for_whisper = ch_path
    else:
        ch_path_for_whisper = ch_path

    try:
        model = _load_whisper()
        if model is None:
            return []

        # Config Whisper de production (anti-hallucination)
        # initial_prompt : oriente le décodage vers le vocabulaire métier (téléphonie
        # pro, médical, dentaire, commercial). Quand Whisper hésite entre 2 mots
        # phonétiquement proches, il privilégie ceux du prompt.
        prompt_to_use = initial_prompt or WHISPER_INITIAL_PROMPT
        segments_raw, info = model.transcribe(
            ch_path_for_whisper,
            language=language,
            beam_size=5,
            word_timestamps=True,
            initial_prompt=prompt_to_use,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 1000,
                "speech_pad_ms": 500,
                "threshold": 0.30,
            },
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.45,
            # 4s = compromis pour ne pas rejeter les bribes isolées en début/fin
            hallucination_silence_threshold=4.0,
        )

        # Si DEBUG_WHISPER_RAW : matérialiser le generator pour pouvoir logger
        # le texte brut de Whisper avant nos filtres post-process
        if DEBUG_WHISPER_RAW:
            segments_raw = list(segments_raw)
            log(f"[DEBUG_WHISPER_RAW] {speaker} (ch{channel}) — {len(segments_raw)} segment(s) bruts:")
            for idx, seg in enumerate(segments_raw):
                text = (seg.text or '').strip()
                log(f"[DEBUG_WHISPER_RAW]   #{idx} [{seg.start:.2f}s → {seg.end:.2f}s] {text!r}")

        all_words = []
        for seg in segments_raw:
            if seg.words:
                for w in seg.words:
                    all_words.append({"word": w.word, "start": w.start, "end": w.end})

        if not all_words:
            if DEBUG_WHISPER_RAW:
                log(f"[DEBUG_WHISPER_RAW] {speaker} (ch{channel}) — aucun mot extrait, return []")
            return []

        # ── Remapping des timestamps si on a extrait un WAV court ──
        # Les timestamps de Whisper sont relatifs au WAV court. On doit les
        # convertir vers la timeline de l'audio original.
        if timestamp_remap_regions:
            def _remap_ts(t_short: float) -> float:
                """Convertit un timestamp du WAV court vers l'audio original."""
                for orig_start, short_start, dur in timestamp_remap_regions:
                    if short_start <= t_short < short_start + dur + 0.1:
                        return orig_start + (t_short - short_start)
                # Fallback : dernier region
                if timestamp_remap_regions:
                    last = timestamp_remap_regions[-1]
                    return last[0] + (t_short - last[1])
                return t_short

            for w in all_words:
                w["start"] = _remap_ts(w["start"])
                w["end"] = _remap_ts(w["end"])

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
        if DEBUG_WHISPER_RAW:
            log(f"[DEBUG_WHISPER_RAW] {speaker} (ch{channel}) — segments après regroupement par pauses ({before_filter}):")
            for idx, s in enumerate(result_segments):
                log(f"[DEBUG_WHISPER_RAW]   #{idx} [{s['start']}s → {s['end']}s] {s['text']!r}")
        # Logger les segments rejetés par _is_hallucination
        if DEBUG_WHISPER_RAW:
            for s in result_segments:
                if _is_hallucination(s["text"]):
                    log(f"[DEBUG_WHISPER_RAW]   ❌ REJETÉ par _is_hallucination [{s['start']}s] {s['text']!r}")
        result_segments = [s for s in result_segments if not _is_hallucination(s["text"])]
        if before_filter != len(result_segments):
            log(f"[WHISPER] Filtered {before_filter - len(result_segments)} hallucinated segments on {speaker}")

        # Filtrer les répétitions de segments courts (≤ 4 mots) qui apparaissent
        # plusieurs fois sur le même canal — ce sont des hallucinations classiques
        # de Whisper qui s'accroche à un mot et le répète (ex: "Christophe" × 6).
        # On garde la PREMIÈRE occurrence (peut être légitime), on supprime les suivantes.
        # Exception : les fillers communs (oui, non, ok, d'accord...) sont gardés.
        FILLERS = {
            "oui", "non", "ok", "d'accord", "voilà", "ah", "euh", "hum",
            "merci", "bonjour", "au revoir", "bonne journée", "bien sûr",
            "exactement", "tout à fait", "absolument", "d'accord.", "oui.",
            "non.", "merci.", "au revoir.", "bonjour.", "ok.", "voilà.",
        }
        from collections import Counter
        short_text_counts: Counter = Counter()
        for s in result_segments:
            t = (s.get("text") or "").strip().lower()
            words = t.split()
            if 1 <= len(words) <= 4:
                # Ignorer les fillers (peuvent légitimement se répéter)
                if t in FILLERS or all(w.strip(".,!?") in FILLERS for w in words):
                    continue
                short_text_counts[t] += 1

        # Hallucination forte (≥5 répétitions du même texte court) → tout supprimer
        # car c'est manifestement une hallucination Whisper qui s'accroche à un mot
        strong_hallucinations = {text for text, count in short_text_counts.items() if count >= 5}
        # Hallucination modérée (3-4 répétitions) → garder la 1re occurrence (peut être légitime)
        moderate_repeated = {text for text, count in short_text_counts.items() if 3 <= count < 5}

        if strong_hallucinations or moderate_repeated:
            seen_moderate: set = set()
            new_segments = []
            removed = 0
            for s in result_segments:
                t = (s.get("text") or "").strip().lower()
                if t in strong_hallucinations:
                    removed += 1
                    continue  # supprimer TOUTES les occurrences
                if t in moderate_repeated:
                    if t in seen_moderate:
                        removed += 1
                        continue
                    seen_moderate.add(t)
                new_segments.append(s)
            if removed > 0:
                log(f"[WHISPER] Filtered {removed} repeated short segments on {speaker} "
                    f"(strong: {sorted(strong_hallucinations)}, moderate: {sorted(moderate_repeated)})")
            result_segments = new_segments

        log(f"[WHISPER] {speaker} (ch{channel}): {len(result_segments)} phrases, {sum(len(s['text']) for s in result_segments)} chars")
        return result_segments
    finally:
        if ch_path and ch_path != wav_path and os.path.exists(ch_path):
            os.remove(ch_path)
        if enhanced_path and enhanced_path != ch_path and enhanced_path != wav_path \
                and os.path.exists(enhanced_path):
            os.remove(enhanced_path)


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
# Skip la correction LLM si le transcript dépasse cette taille (en chars).
# Pourquoi : sur les longs transcripts, le LLM corrige peu (Whisper a plus de
# contexte → meilleur résultat) et le coût en temps est élevé (~10s/1000 chars).
CORRECT_MAX_CHARS = int(os.environ.get("CORRECT_MAX_CHARS", "4000"))

CORRECT_CHUNK_SIZE = int(os.environ.get("CORRECT_CHUNK_SIZE", "30"))


def correct_transcript(transcript: str) -> str:
    """Corrige les erreurs phonétiques Whisper via LLM en mode DIFF.

    Fonctionnement :
      1. Le LLM lit le transcript ENTIER pour comprendre le contexte
      2. Il identifie les mots incorrects (homonymes phonétiques)
      3. Il retourne UNIQUEMENT les corrections (pas le texte complet)
      4. On applique les corrections en Python

    Avantage : au lieu de régénérer 1800 chars (~480 tokens, ~42s), le LLM
    n'émet que les corrections (~20-50 tokens, ~3-5s). Si rien à corriger, il
    répond "OK" et on garde l'original tel quel.
    """
    if not transcript or len(transcript.split()) < 5:
        return transcript

    original_lines = [l for l in transcript.split("\n") if l.strip()]
    n_lines = len(original_lines)

    # Numéroter les lignes pour que le LLM puisse référencer
    numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(original_lines))

    prompt = (
        "Tu lis cette transcription d'un appel téléphonique (cabinet dentaire/médical) "
        "et tu identifies les erreurs phonétiques de Whisper : des mots remplacés par "
        "un homonyme incorrect qui n'a pas de sens dans le contexte.\n\n"
        "Exemples d'erreurs courantes : 'centre d'entraînement' → 'centre dentaire', "
        "'désertage' → 'détartrage', 'y a pas de passe' → 'y a pas de place', "
        "'dent sage' → 'dent de sagesse', 'géniable' → 'joignable'.\n\n"
        "RÈGLES :\n"
        "- Lis TOUT le transcript pour comprendre le contexte avant de corriger\n"
        "- Ne corrige QUE les mots dont tu es SÛR qu'ils sont faux\n"
        "- Laisse les noms propres, les hésitations ('euh', 'ben'), la ponctuation\n"
        "- Ne reformule JAMAIS, ne change que le(s) mot(s) erroné(s)\n\n"
        "FORMAT DE RÉPONSE :\n"
        "- Si des corrections sont nécessaires, une par ligne : numéro_ligne: ancien → nouveau\n"
        "- Si RIEN à corriger, réponds uniquement : OK\n\n"
        "Exemples de réponse :\n"
        "3: centre d'entraînement → centre dentaire\n"
        "7: y a pas de passe → y a pas de place\n\n"
        f"Transcript ({n_lines} lignes) :\n{numbered}\n\n"
        "Corrections :"
    )

    corrected_text = run_llm(prompt, max_new_tokens=256)
    if not corrected_text:
        return transcript

    corrected_text = corrected_text.strip()

    # Si le LLM dit OK ou rien à corriger → garder l'original
    if corrected_text.upper() in ("OK", "AUCUNE CORRECTION", "RAS", "RIEN"):
        log(f"[CORRECT] LLM: aucune correction nécessaire ({n_lines} lines)")
        return transcript

    # Parser et appliquer les corrections
    n_applied = 0
    for line in corrected_text.split("\n"):
        line = line.strip()
        if not line or "→" not in line:
            continue
        try:
            # Format attendu : "3: ancien texte → nouveau texte"
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            line_num_str = parts[0].strip()
            # Extraire le numéro de ligne (peut contenir des espaces ou préfixes)
            line_num = int("".join(c for c in line_num_str if c.isdigit()))
            if line_num < 1 or line_num > n_lines:
                continue
            correction = parts[1].strip()
            arrow_parts = correction.split("→")
            if len(arrow_parts) != 2:
                continue
            old_text = arrow_parts[0].strip()
            new_text = arrow_parts[1].strip()
            if not old_text or not new_text:
                continue
            # Appliquer la correction sur la ligne correspondante
            idx = line_num - 1
            if old_text in original_lines[idx]:
                original_lines[idx] = original_lines[idx].replace(old_text, new_text, 1)
                n_applied += 1
                log(f"[CORRECT]   L{line_num}: '{old_text}' → '{new_text}'")
        except (ValueError, IndexError):
            continue

    log(f"[CORRECT] Transcript corrected: {n_applied} correction(s) applied ({n_lines} lines)")
    return "\n".join(original_lines)


# =============================================================================
# RÉSUMÉ
# =============================================================================
def generate_summary(transcript: str, duration_seconds: float = 0) -> str:
    if not transcript or len(transcript.split()) < 10:
        return "Conversation très brève."

    if duration_seconds <= 180:   max_tokens = 70
    elif duration_seconds <= 600: max_tokens = 90
    else:                         max_tokens = 110

    # Stratégie de troncature :
    #  - Appel ≤ 15 min : on passe TOUT le transcript au LLM (Ministral 8B a 128k
    #    de contexte, il avale large). C'est le cas typique (>95% des appels).
    #  - Appel > 15 min : on garde la tête (2500 chars = qui appelle, pour quoi)
    #    et la queue (2500 chars = conclusion, accord, RDV pris). Le milieu est
    #    sacrifié car c'est généralement la partie la moins informative pour un
    #    résumé court de 1-2 phrases.
    if duration_seconds <= 900:
        transcript_for_summary = transcript
    else:
        head = transcript[:2500]
        tail = transcript[-2500:]
        transcript_for_summary = f"{head}\n[... milieu de conversation omis ...]\n{tail}"

    prompt = (
        "Résume cette conversation téléphonique en 1 ou 2 phrases courtes (max 40 mots au total). "
        "Dis uniquement l'essentiel : qui appelle, pour quoi, et la conclusion. "
        "Pas d'introduction, pas de reformulation inutile, va droit au but.\n\n"
        f"Conversation:\n{transcript_for_summary}\n\nRésumé court:"
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

    # Prompt qui distingue le TON (attitude) du SUJET (problème évoqué).
    # Un client qui appelle pour une urgence/douleur reste "neutre" si l'échange
    # est poli. "mauvais" est réservé aux clients EXPLICITEMENT mécontents du
    # service. "bon" couvre les échanges réussis et conviviaux (pas seulement les
    # remerciements effusifs).
    prompt = (
        "Tu analyses le TON ÉMOTIONNEL d'une conversation téléphonique entre un client et un agent d'un cabinet dentaire.\n"
        "Concentre-toi sur l'ATTITUDE du client envers l'agent et l'ISSUE de l'appel, PAS sur le sujet médical.\n\n"
        "Réponds par UN SEUL MOT : bon, neutre, ou mauvais\n\n"
        "Définitions :\n"
        "- bon : le client repart SATISFAIT (sa demande a été traitée OU il manifeste de la chaleur, "
        "remerciements appuyés, ton convivial, complicité avec l'agent, soulagement). Au moins UN signe positif clair.\n"
        "- neutre : conversation factuelle/polie sans émotion marquée, OU demande non aboutie sans énervement, "
        "OU échange très bref. C'est le cas par défaut quand rien ne penche vraiment d'un côté.\n"
        "- mauvais : le client montre du MÉCONTENTEMENT, de l'AGACEMENT, de l'IMPATIENCE marquée, "
        "ou se PLAINT du service, de l'attente, ou de l'agent. Il peut aussi être AGRESSIF, INSULTER ou MENACER.\n\n"
        "⚠ RÈGLE 1 : un client qui appelle pour un PROBLÈME MÉDICAL (douleur, urgence, dent cassée, 'ça fait mal', "
        "'j'ai mal') reste 'neutre' si la conversation est polie. Exprimer une douleur physique ou un besoin "
        "urgent N'EST PAS être mécontent — c'est le MOTIF de l'appel, pas une plainte contre le service.\n\n"
        "⚠ RÈGLE 2 : un client poli dont la demande est SATISFAITE (rdv pris, info reçue, problème résolu) "
        "et qui remercie en partant peut être 'bon', surtout s'il y a un ton chaleureux ou plusieurs remerciements.\n\n"
        "Exemples 'bon' :\n"
        "- Client obtient son rdv + 'Super, merci beaucoup, à bientôt !' → bon\n"
        "- 'Ah, c'est très gentil, merci à vous, bonne journée !' (ton chaleureux) → bon\n"
        "- 'Parfait, vous me sauvez' + remerciements → bon\n"
        "- Client soulagé d'avoir une réponse + remerciements appuyés → bon\n"
        "- Échange convivial avec rires ou complicité → bon\n\n"
        "Exemples 'neutre' :\n"
        "- 'J'ai mal aux dents, c'est urgent' + agent répond + 'Merci, au revoir' → neutre\n"
        "- 'Ça me fait mal' + rdv confirmé + 'OK merci beaucoup, à tout à l'heure' → neutre\n"
        "- 'Bonjour, je voudrais un rdv' + 'D'accord, c'est noté' + 'OK merci' → neutre\n"
        "- 'Vous n'avez rien avant 3 semaines ? D'accord, je rappellerai' (déçu mais poli) → neutre\n"
        "- Échange court 'Bonjour - Au revoir' → neutre\n\n"
        "Exemples 'mauvais' :\n"
        "- 'Vous m'avez fait attendre 30 min, c'est inadmissible !' → mauvais\n"
        "- 'Ça fait 3 fois que j'appelle, c'est scandaleux' → mauvais\n"
        "- 'Vous êtes incompétents' → mauvais\n"
        "- Client agacé : 'Bon c'est pas possible là, je perds mon temps' → mauvais\n"
        "- Client impatient/irrité tout au long de l'appel, soupirs, ton sec → mauvais\n\n"
        f"Conversation à analyser :\n{text[:1500]}\n\n"
        "Ton (un seul mot) :"
    )

    response = run_llm(prompt, max_new_tokens=16).strip().lower()
    log(f"[SENTIMENT] LLM response: '{response}'")

    # Note : on est strict sur "mauvais" — il faut que le mot soit clairement présent
    # (pas juste "négatif" qui peut s'appliquer au sujet)
    if "mauvais" in response:
        return {"label_en": "negative", "label_fr": "mauvais", "confidence": 0.75,
                "scores": {"negative": 0.75, "neutral": 0.20, "positive": 0.05}}
    elif "bon" in response and "bonne" not in response:
        return {"label_en": "positive", "label_fr": "bon", "confidence": 0.80,
                "scores": {"negative": 0.05, "neutral": 0.15, "positive": 0.80}}
    else:
        # Par défaut : neutre (cas le plus fréquent en téléphonie pro)
        return {"label_en": "neutral", "label_fr": "neutre", "confidence": 0.70,
                "scores": {"negative": 0.15, "neutral": 0.70, "positive": 0.15}}


# =============================================================================
# HOLD MUSIC DETECTION (RMS-based, pas de LLM)
# =============================================================================
def detect_hold_music(audio_path_or_np, sr: int = 0) -> Dict[str, Any]:
    """Détecte si un audio est de la musique d'attente.
    Accepte soit un chemin de fichier (str) soit un numpy array mono (float32).
    Utilise torch.unfold (GPU si dispo) pour le calcul RMS vectorisé."""
    import soundfile as sf
    import numpy as np
    result = {"is_hold_music": False, "speech_ratio": 1.0, "duration": 0.0}
    try:
        if isinstance(audio_path_or_np, str):
            y, sr = sf.read(audio_path_or_np, dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        else:
            y = audio_path_or_np
        duration = len(y) / sr
        result["duration"] = round(duration, 2)
        if duration < HOLD_MUSIC_MIN_DURATION:
            return result
        frame_length = int(0.030 * sr)
        hop_length = int(0.010 * sr)

        # RMS vectorisé via torch (GPU ou CPU, pas de boucle Python)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        y_t = torch.from_numpy(y).to(device)
        peak = float(y_t.abs().max())
        if peak < 1e-6:
            result.update({"is_hold_music": True, "speech_ratio": 0.0})
            return result
        frames = y_t.unfold(0, frame_length, hop_length)  # (n_frames, frame_length)
        rms = frames.pow(2).mean(dim=1).sqrt()  # (n_frames,)
        rms_norm = rms / peak
        speech_threshold = float(rms_norm.median()) + 1.5 * float(rms_norm.std())
        speech_mask = rms_norm > speech_threshold
        speech_ratio = float(speech_mask.sum()) / len(rms_norm)
        del y_t, frames, rms, rms_norm, speech_mask
        if device == "cuda":
            torch.cuda.empty_cache()

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
ON_HOLD_LABEL = "Appel mis en attente, sans réponse"


def _segments_look_like_ivr_loop(segments: List[Dict]) -> bool:
    """Détecte si une liste de segments d'un canal correspond à une annonce IVR
    répétée en boucle (cabinet dentaire d'accueil typique).

    Critères convergents :
    - Au moins 5 segments générés
    - Marqueurs typiques d'annonce dans au moins 50% des segments
      (ex: "Bienvenue au", "Notre cabinet", "Pour les premières", "Vous pouvez également",
      "Nous pratiquons", "veuillez patienter", "votre appel a été", "Merci de patienter")
    - OU répétition forte des mêmes phrases (ratio mots uniques < 35%)
    """
    if not segments or len(segments) < 5:
        return False

    IVR_MARKERS = [
        "bienvenue",
        "veuillez patienter",
        "veuillez rester en ligne",
        "votre appel a été",
        "votre appel est",
        "écourter agréablement",
        "nous pratiquons",
        "notre cabinet",
        "nos horaires",
        "nous vous invitons",
        "rendez-vous en ligne",
        "doctolib",
        "soins dentaires",
        "implantologie",
        "parodontologie",
        "rendez-vous d'urgence",
        "premières consultations",
        "merci de patienter",
        "tous nos conseillers",
        "tous nos collaborateurs",
        "veuillez ne pas quitter",
        "nous allons vous répondre",
        "écoutez attentivement",
        "tapez 1",
        "tapez 2",
    ]

    n_total = len(segments)
    n_ivr = 0
    all_words = []
    for s in segments:
        text = (s.get("text") or "").lower()
        if any(marker in text for marker in IVR_MARKERS):
            n_ivr += 1
        # Pour le calcul du ratio mots uniques
        cleaned = re.sub(r"[^\w\s]", " ", text)
        all_words.extend(cleaned.split())

    ivr_ratio = n_ivr / n_total

    # Critère 1 : >= 50% des segments contiennent un marqueur IVR
    if ivr_ratio >= 0.5:
        return True

    # Critère 2 : vocabulaire très pauvre (< 35% mots uniques) sur >= 30 mots
    if len(all_words) >= 30:
        unique_ratio = len(set(all_words)) / len(all_words)
        if unique_ratio < 0.35 and ivr_ratio >= 0.3:
            return True

    return False


def _build_on_hold_response(duration: float, diarization_mode: str) -> Dict[str, Any]:
    """Réponse standard quand l'appel a été mis en attente sans jamais être traité
    (annonce IVR en boucle côté agent, client n'a parlé à personne)."""
    neutral_mood = {
        "label_en": "neutral", "label_fr": "neutre", "confidence": 0.5,
        "scores": {"negative": 0.2, "neutral": 0.6, "positive": 0.2},
    }
    segment = {
        "speaker": "System",
        "start": 0.0,
        "end": round(duration, 2),
        "text": ON_HOLD_LABEL,
        "mood": neutral_mood,
    }
    return {
        "task": "transcribe_diarized",
        "segments": [segment],
        "transcript": f"System: {ON_HOLD_LABEL}",
        "summary": ON_HOLD_LABEL,
        "diarization_mode": diarization_mode,
        "audio_duration": round(duration, 2),
        "on_hold_detected": True,
        "mood_overall": neutral_mood,
        "mood_client": neutral_mood,
        "mood_by_speaker": {"System": neutral_mood},
    }


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
        return {"ok": True, "app_version": APP_VERSION, "whisper": WHISPER_REPO_ID, "llm": LLM_MODEL_ID}

    language       = inp.get("language") or "fr"
    with_summary   = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))
    # Override possible du prompt Whisper depuis le job (pour clients spécialisés)
    job_initial_prompt = (inp.get("initial_prompt") or "").strip() or None

    # Déterminer la direction de l'appel
    # Règle : un fichier dont le nom commence par "out" est sortant, tous les
    # autres sont entrants. On vérifie aussi si "out-" ou "out_" apparaît n'importe
    # où dans le filename (ex: q-701-out-xxx.wav).
    call_direction = (inp.get("call_direction") or "").lower().strip()
    if call_direction not in ("inbound", "outbound"):
        audio_url = inp.get("audio_url") or inp.get("file_path") or ""
        filename = audio_url.split("/")[-1].lower()
        # Log de debug pour comprendre les filenames qu'on reçoit
        log(f"[DIRECTION] audio_url={audio_url[:200]} filename={filename}")
        # Détection souple : "out-" ou "out_" au début OU dans le nom
        if (
            filename.startswith("out-")
            or filename.startswith("out_")
            or filename.startswith("out.")
            or filename.startswith("out ")
            or "/out-" in audio_url.lower()
            or "/out_" in audio_url.lower()
            or "-out-" in filename
            or "_out_" in filename
            or "-out." in filename
            or "_out." in filename
        ):
            call_direction = "outbound"
        else:
            call_direction = "inbound"

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

        # ── Mapping canal → speaker selon la direction ──────────────────────
        # Convention empirique observée sur les enregistrements FreePBX :
        #   - Appels ENTRANTS  : CH0 = client (appelant), CH1 = agent local
        #   - Appels SORTANTS  : CH0 = agent local (appelant), CH1 = distant
        # On définit le mapping ici une fois pour toutes — le reste du code
        # utilise ch0_speaker / ch1_speaker.
        if call_direction == "outbound":
            ch0_speaker = "Agent"
            ch1_speaker = "Client"
        else:
            ch0_speaker = "Client"
            ch1_speaker = "Agent"
        log(f"[HANDLER] Channel mapping: CH0={ch0_speaker}, CH1={ch1_speaker} (direction={call_direction})")

        # ── LECTURE UNIQUE du fichier audio ────────────────────────────────
        # On lit le fichier stéréo UNE SEULE FOIS et on passe les numpy arrays
        # à toutes les fonctions de détection. Ça élimine 5 relectures de
        # fichier qui prenaient jusqu'à 60s sur des workers avec un disque lent.
        import soundfile as sf
        import numpy as np
        stereo_data = None
        sr_full = 0
        ch0_np = None
        ch1_np = None
        if is_stereo:
            stereo_data, sr_full = sf.read(local_path, dtype="float32")
            if stereo_data.ndim > 1:
                ch0_np = stereo_data[:, 0].copy()
                ch1_np = stereo_data[:, 1].copy()
            else:
                ch0_np = stereo_data.copy()
                ch1_np = stereo_data.copy()
            del stereo_data

        # Hold music detection
        if ENABLE_HOLD_MUSIC_DETECTION:
            if is_stereo:
                hold_ch0 = detect_hold_music(ch0_np, sr_full)
                hold_ch1 = detect_hold_music(ch1_np, sr_full)
                log(f"[HOLD_MUSIC] CH0: hold={hold_ch0['is_hold_music']}, ratio={hold_ch0['speech_ratio']:.4f}")
                log(f"[HOLD_MUSIC] CH1: hold={hold_ch1['is_hold_music']}, ratio={hold_ch1['speech_ratio']:.4f}")
                if hold_ch0["is_hold_music"] and hold_ch1["is_hold_music"]:
                    return _build_hold_music_response(est_dur, "stereo_hold_music")
            else:
                hold = detect_hold_music(local_path)
                if hold["is_hold_music"]:
                    return _build_hold_music_response(est_dur, "mono_hold_music")

        # ── TRANSCRIPTION ──────────────────────────────────────────────────
        if is_stereo:
            log(f"[HANDLER] Stereo audio → Whisper per-channel with word timestamps")

            # Détection des zones de hold music par canal (réutilise les arrays déjà en mémoire)
            hold_regions_ch0: List[Tuple[float, float]] = []
            hold_regions_ch1: List[Tuple[float, float]] = []
            hold_coverage_ch0 = 0.0
            hold_coverage_ch1 = 0.0
            if ENABLE_HOLD_MUSIC_DETECTION:
                try:
                    hold_regions_ch0 = _detect_hold_regions_in_channel(ch0_np, sr_full)
                    hold_regions_ch1 = _detect_hold_regions_in_channel(ch1_np, sr_full)
                    total_ch0 = sum(e - s for s, e in hold_regions_ch0)
                    total_ch1 = sum(e - s for s, e in hold_regions_ch1)
                    hold_coverage_ch0 = total_ch0 / est_dur if est_dur > 0 else 0
                    hold_coverage_ch1 = total_ch1 / est_dur if est_dur > 0 else 0
                    if hold_regions_ch0:
                        log(f"[HOLD_REGIONS] CH0 (Client): {len(hold_regions_ch0)} zones, {total_ch0:.1f}s ({hold_coverage_ch0*100:.0f}%)")
                    if hold_regions_ch1:
                        log(f"[HOLD_REGIONS] CH1 (Agent): {len(hold_regions_ch1)} zones, {total_ch1:.1f}s ({hold_coverage_ch1*100:.0f}%)")
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

            # ── Détection spectrale des régions IVR (réutilise les arrays) ──
            ivr_regions_ch0: List[Tuple[float, float]] = []
            ivr_regions_ch1: List[Tuple[float, float]] = []
            try:
                ivr_regions_ch0, p0, s0 = _detect_loop_regions(ch0_np, sr_full)
                ivr_regions_ch1, p1, s1 = _detect_loop_regions(ch1_np, sr_full)
                cov0 = sum(e - s for s, e in ivr_regions_ch0)
                cov1 = sum(e - s for s, e in ivr_regions_ch1)
                log(f"[IVR_LOOP] CH0 (Client): {len(ivr_regions_ch0)} zones, {cov0:.1f}s, period={p0:.0f}s, score={s0:.3f}")
                log(f"[IVR_LOOP] CH1 (Agent): {len(ivr_regions_ch1)} zones, {cov1:.1f}s, period={p1:.0f}s, score={s1:.3f}")
            except Exception as e:
                log(f"[IVR_LOOP] Error: {e}")

            # Libérer les arrays bruts (plus besoin, Whisper lit le fichier directement)
            del ch0_np, ch1_np

            # Fusionner les régions IVR avec les hold_regions existantes (mêmes effets) :
            # toute zone marquée comme IVR ou hold/silence sera mutée pour cross-mute.
            def _merge_regions(a, b):
                merged = sorted(list(a) + list(b))
                out: List[Tuple[float, float]] = []
                for s, e in merged:
                    if out and s <= out[-1][1] + 0.5:
                        out[-1] = (out[-1][0], max(out[-1][1], e))
                    else:
                        out.append((s, e))
                return out

            mute_ch0 = _merge_regions(hold_regions_ch0, ivr_regions_ch0)
            mute_ch1 = _merge_regions(hold_regions_ch1, ivr_regions_ch1)
            mute_coverage_ch0 = sum(e - s for s, e in mute_ch0) / est_dur if est_dur > 0 else 0
            mute_coverage_ch1 = sum(e - s for s, e in mute_ch1) / est_dur if est_dur > 0 else 0

            # Si LES DEUX canaux sont quasi-entièrement dans des zones non-conversationnelles
            # (hold/silence/IVR), l'appel n'a jamais été réellement pris en charge.
            if mute_coverage_ch0 >= 0.90 and mute_coverage_ch1 >= 0.90:
                log(f"[HANDLER] Both channels fully non-conversational (ch0={mute_coverage_ch0*100:.0f}%, ch1={mute_coverage_ch1*100:.0f}%) → on_hold response")
                return _build_on_hold_response(est_dur, "stereo_dual_non_conv")

            # Remplacer les hold_regions originales par les régions fusionnées :
            # le code de transcription en dessous va utiliser hold_regions_ch{0,1}
            # comme cross-mute. Du coup l'agent ne sera pas transcrit sur les zones
            # IVR du client (et inversement), ET les bruits ambiants ne seront pas
            # capturés en face d'une annonce IVR.
            hold_regions_ch0 = mute_ch0
            hold_regions_ch1 = mute_ch1
            hold_coverage_ch0 = mute_coverage_ch0
            hold_coverage_ch1 = mute_coverage_ch1
            # Forcer l'activation du cross-mute si on a détecté des zones IVR
            if ivr_regions_ch0 or ivr_regions_ch1:
                log(f"[HANDLER] IVR regions detected → cross-mute forced")

            # Décider si on applique le cross-mute :
            # - Si la détection IVR a trouvé des zones d'annonce → TOUJOURS muter
            # - Si un canal a ≥ 50% de hold_coverage → muter
            # - Si un canal est classé "hold music" MAIS que les DEUX canaux ont
            #   du speech détecté (ratio > 5%) → c'est une conversation bidirectionnelle
            #   réelle, ne PAS muter (le hold_music est un faux positif sur un
            #   agent qui parle doucement / micro faible).
            has_ivr_regions = bool(ivr_regions_ch0 or ivr_regions_ch1)
            both_have_speech = (
                hold_ch0.get("speech_ratio", 0) > 0.05
                and hold_ch1.get("speech_ratio", 0) > 0.05
            )
            hold_flag_reliable = (
                (hold_ch0.get("is_hold_music") or hold_ch1.get("is_hold_music"))
                and not both_have_speech
            )
            # hold_coverage ≥ 50% déclenche le cross-mute SAUF si les deux
            # canaux ont du speech (> 5%). Dans une vraie conversation, l'agent
            # écoute le client → son canal est silencieux → faux "hold regions".
            # Seule exception : hold_coverage ≥ 80% (quasi-certain hold réel,
            # même avec un peu de speech résiduel sur l'autre canal).
            coverage_trigger = (
                (hold_coverage_ch0 >= 0.50 or hold_coverage_ch1 >= 0.50)
                and (not both_have_speech or hold_coverage_ch0 >= 0.80 or hold_coverage_ch1 >= 0.80)
            )
            apply_cross_mute = (
                has_ivr_regions
                or hold_flag_reliable
                or coverage_trigger
            )
            if not apply_cross_mute:
                if hold_regions_ch0 or hold_regions_ch1:
                    log(f"[HANDLER] Conversation bidirectionnelle (ch0={hold_coverage_ch0*100:.0f}%, ch1={hold_coverage_ch1*100:.0f}%) → cross-mute désactivé")
                hold_regions_ch0 = []
                hold_regions_ch1 = []

            # mute_regions passées à chaque canal :
            #   - cross-mute : régions hold/IVR de l'AUTRE canal (ne pas transcrire
            #     les bruits ambiants pendant que l'autre est en attente)
            #   - self-mute IVR : régions IVR de CE canal (ne pas transcrire l'annonce
            #     elle-même quand le canal joue une annonce en boucle)
            ch0_mute = _merge_regions(hold_regions_ch1, ivr_regions_ch0)
            ch1_mute = _merge_regions(hold_regions_ch0, ivr_regions_ch1)

            # Skip un canal UNIQUEMENT s'il est quasi-entièrement hold music (≥ 95%).
            # Sinon on le transcrit quand même : Whisper a son propre VAD qui sautera
            # les zones de musique/silence, et on garde les vraies paroles s'il y en a.
            FULL_HOLD_THRESHOLD = 0.95

            if ENABLE_HOLD_MUSIC_DETECTION and hold_coverage_ch0 >= FULL_HOLD_THRESHOLD:
                log(f"[HANDLER] CH0 ({ch0_speaker}) is dominantly hold music ({hold_coverage_ch0*100:.0f}%) → skipping Whisper")
                ch0_segs = []
            else:
                ch0_segs = _transcribe_channel_whisper(
                    local_path, 0, ch0_speaker, language,
                    mute_regions=ch0_mute or None,
                    initial_prompt=job_initial_prompt,
                )

            if ENABLE_HOLD_MUSIC_DETECTION and hold_coverage_ch1 >= FULL_HOLD_THRESHOLD:
                log(f"[HANDLER] CH1 ({ch1_speaker}) is dominantly hold music ({hold_coverage_ch1*100:.0f}%) → skipping Whisper")
                ch1_segs = []
            else:
                ch1_segs = _transcribe_channel_whisper(
                    local_path, 1, ch1_speaker, language,
                    mute_regions=ch1_mute or None,
                    initial_prompt=job_initial_prompt,
                )

            segments = ch0_segs + ch1_segs
            segments.sort(key=lambda s: s["start"])

            # Post-check : si un canal était dominé par une boucle IVR (≥95%) ET
            # que le total de speech transcrite côté de l'autre canal est négligeable
            # (< 5s cumulés), c'est en réalité un appel passé entièrement en attente.
            # Les quelques fragments transcrits sont des hallucinations Whisper sur
            # les rares fenêtres non mutées en début/fin d'appel.
            ivr_dominated = (
                (ivr_regions_ch0 and sum(e - s for s, e in ivr_regions_ch0) / est_dur >= 0.95) or
                (ivr_regions_ch1 and sum(e - s for s, e in ivr_regions_ch1) / est_dur >= 0.95)
            )
            if ivr_dominated and segments:
                total_speech = sum(max(0.0, s.get("end", 0.0) - s.get("start", 0.0)) for s in segments)
                if total_speech < 5.0:
                    log(f"[HANDLER] IVR dominated (≥95%) + résiduel négligeable ({total_speech:.1f}s de speech) → on_hold response")
                    return _build_on_hold_response(est_dur, "stereo_ivr_dominated_residual")

            # ── Interleaving post-hoc : corriger les timestamps compressés ──
            # Quand un canal a très peu de speech après un long mute (ex: client
            # après 11 min de hold music), Whisper compresse tous les word timestamps
            # dans un intervalle court → 1 gros segment contenant TOUTE la conversation
            # de ce speaker, alors que l'autre speaker a des segments étalés sur une
            # plage bien plus large.
            #
            # Détection : un speaker a ≤ 2 segments contenant ≥ 3 phrases, et l'autre
            # speaker a ≥ 3 segments étalés sur une plage plus large.
            # Action : redistribuer les phrases du speaker compressé dans les "trous"
            # entre les segments de l'autre speaker.
            #
            # Sur les conversations normales, les 2 speakers ont des segments bien
            # répartis → la condition ne se déclenche JAMAIS.
            if len(segments) >= 4:
                import re as _re
                speakers = set(s["speaker"] for s in segments)
                for spk in speakers:
                    spk_segs = [s for s in segments if s["speaker"] == spk]
                    other_segs = [s for s in segments if s["speaker"] != spk]
                    if len(other_segs) < 3:
                        continue
                    # Critère principal : le speaker a tous ses segments dans
                    # une plage temporelle COURTE alors que l'autre speaker
                    # s'étale sur une plage ≥ 3× plus large.
                    other_segs.sort(key=lambda s: s["start"])
                    other_range = other_segs[-1]["end"] - other_segs[0]["start"]
                    spk_range = max(s["end"] for s in spk_segs) - min(s["start"] for s in spk_segs)
                    if other_range < spk_range * 3:
                        continue  # pas de compression évidente
                    # Compter les phrases dans les segments de ce speaker
                    all_text = " ".join(s.get("text", "") for s in spk_segs)
                    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', all_text) if s.strip()]
                    if len(sentences) < 3:
                        continue

                    # OK : ce speaker a ses timestamps compressés. On redistribue
                    # ses phrases dans les trous entre les segments de l'autre speaker.
                    slots = []
                    # Slot avant le premier other_seg
                    conv_start = min(s["start"] for s in spk_segs + other_segs)
                    if conv_start < other_segs[0]["start"]:
                        slots.append((conv_start, other_segs[0]["start"]))
                    # Slots entre les segments de l'autre speaker
                    for i in range(len(other_segs) - 1):
                        gap_start = other_segs[i]["end"]
                        gap_end = other_segs[i + 1]["start"]
                        if gap_end - gap_start > 0.3:
                            slots.append((gap_start, gap_end))
                    # Slot après le dernier other_seg
                    conv_end = max(s["end"] for s in spk_segs + other_segs)
                    if conv_end > other_segs[-1]["end"]:
                        slots.append((other_segs[-1]["end"], conv_end))

                    if not slots:
                        continue

                    # Distribuer les phrases dans les slots
                    new_spk_segs = []
                    n_slots = len(slots)
                    for idx, (slot_start, slot_end) in enumerate(slots):
                        if idx < n_slots - 1:
                            if idx < len(sentences):
                                phrase = sentences[idx]
                            else:
                                break
                        else:
                            phrase = " ".join(sentences[idx:])
                            if not phrase:
                                break
                        new_spk_segs.append({
                            "speaker": spk,
                            "start": round(slot_start, 2),
                            "end": round(slot_end, 2),
                            "text": phrase,
                            "mood": None,
                        })

                    if new_spk_segs:
                        # Remplacer les segments compressés par les nouveaux
                        segments = other_segs + new_spk_segs
                        segments.sort(key=lambda s: s["start"])
                        log(f"[INTERLEAVE] Redistributed {spk} ({len(spk_segs)} segs, "
                            f"{len(sentences)} phrases) into {len(new_spk_segs)} sub-segments "
                            f"across {n_slots} slots (other={len(other_segs)} segs, "
                            f"range={other_range:.1f}s vs compressed={spk_range:.1f}s)")
                        break  # un seul speaker peut être compressé par appel

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
        # - au moins 3 segments OU 200 chars (sinon c'est trop court, peu d'erreurs probables)
        # - PAS plus que CORRECT_MAX_CHARS (sinon coût LLM trop élevé pour bénéfice faible)
        n_chars_transcript = len(full_transcript)
        n_segs_text = sum(1 for s in merged if s.get("text"))
        if not ENABLE_TRANSCRIPT_CORRECTION or not full_transcript:
            pass
        elif n_chars_transcript > CORRECT_MAX_CHARS:
            log(f"[HANDLER] Correction skippée (transcript trop long : {n_chars_transcript} chars > {CORRECT_MAX_CHARS} max)")
            corrected_transcript = full_transcript  # pas de correction
        elif n_segs_text >= 3 or n_chars_transcript >= 200:
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
            mood_by_speaker = {}
            neutral_default = {
                "label_en": "neutral", "label_fr": "neutre", "confidence": 0.7,
                "scores": {"negative": 0.15, "neutral": 0.7, "positive": 0.15},
            }

            # Passer le transcript COMPLET (Client + Agent interleaved) pour que
            # le LLM ait le contexte de la conversation (question/réponse, ton de
            # l'échange, résolution du problème). Le Qwen 14B est assez intelligent
            # pour distinguer le ton du client vs les réponses de l'agent.
            full_text = "\n".join(f"{s['speaker']}: {s['text']}" for s in merged)
            if full_text.strip():
                mood_by_speaker["Client"] = analyze_sentiment(full_text)
                log(f"[SENTIMENT] Client: {mood_by_speaker['Client'].get('label_fr', '?')}")
            else:
                mood_by_speaker["Client"] = neutral_default

            # Agent = toujours neutre. Un agent fait son travail, il n'a pas
            # de "sentiment" au sens métier. Analyser son texte créait des
            # faux positifs (agent qui dit "c'est pas possible" → "mauvais").
            mood_by_speaker["Agent"] = neutral_default
            log(f"[SENTIMENT] Agent: neutre (forcé)")

            # Attacher aux segments
            for seg in merged:
                sp = seg.get("speaker")
                if sp in mood_by_speaker:
                    seg["mood"] = mood_by_speaker[sp]

            # Client mood = mood global de l'appel
            client_mood = mood_by_speaker.get("Client", neutral_default)
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
    log(f"[INIT] Whisper model: {WHISPER_REPO_ID}")
    log(f"[INIT] LLM model: {LLM_MODEL_ID}")
    log("[INIT] Ready.")
except Exception as e:
    log(f"[WARN] Preload failed: {e}")

runpod.serverless.start({"handler": handler})
