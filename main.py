#!/usr/bin/env python3
"""
Whisper Serverless Worker - Transcription + Diarisation + Résumé + Sentiment
Utilise faster-whisper (large-v3) + Pyannote
Version finale 2025-12-21
"""

import os
import sys
import json
import time
import logging
import gc
import warnings
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import requests
import runpod

# Configuration des warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DURATION = int(os.getenv("MAX_DURATION_S", "9000"))
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Cache global des modèles
whisper_model = None
diarizer = None


def log_gpu_memory():
    """Affiche l'utilisation mémoire GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[GPU] {torch.cuda.get_device_name(0)} - Utilisé: {allocated:.1f}GB / {total:.1f}GB")


def cleanup_gpu():
    """Nettoie la mémoire GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def load_whisper_model():
    """Charge le modèle Whisper"""
    global whisper_model
    
    if whisper_model is not None:
        return whisper_model
    
    try:
        from faster_whisper import WhisperModel
        
        logger.info(f"[WHISPER] Chargement du modèle: {WHISPER_MODEL}")
        logger.info(f"[WHISPER] Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")
        
        start_time = time.time()
        whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            download_root=os.getenv("WHISPER_CACHE", "/app/.cache/whisper")
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[WHISPER] ✓ Modèle chargé en {elapsed:.1f}s")
        log_gpu_memory()
        
        return whisper_model
        
    except Exception as e:
        logger.error(f"[WHISPER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_diarizer():
    """Charge le pipeline de diarisation PyAnnote"""
    global diarizer
    
    if diarizer is not None:
        return diarizer
    
    try:
        from pyannote.audio import Pipeline
        
        logger.info(f"[DIARIZER] Chargement: {DIARIZATION_MODEL}")
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            logger.warning("[DIARIZER] ⚠ HF_TOKEN non défini - diarisation désactivée")
            return None
        
        # Essayer les deux APIs (pyannote change souvent)
        try:
            diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
        except TypeError:
            try:
                diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
            except Exception:
                diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL)
        
        diarizer.to(torch.device(DEVICE))
        logger.info("[DIARIZER] ✓ Diarizer chargé")
        log_gpu_memory()
        
        return diarizer
        
    except Exception as e:
        logger.error(f"[DIARIZER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def transcribe_audio(audio_path: str, language: str = None) -> Tuple[str, List[Dict]]:
    """
    Transcrit un fichier audio avec Whisper
    Retourne (texte_complet, segments)
    """
    model = load_whisper_model()
    if model is None:
        raise Exception("Impossible de charger Whisper")
    
    try:
        logger.info(f"[WHISPER] Transcription: {audio_path}")
        
        start_time = time.time()
        
        # Transcription avec Whisper
        segments_gen, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )
        
        # Collecter les segments
        segments = []
        full_text_parts = []
        
        for segment in segments_gen:
            segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            full_text_parts.append(segment.text.strip())
        
        full_text = " ".join(full_text_parts)
        
        elapsed = time.time() - start_time
        detected_lang = info.language if hasattr(info, 'language') else language
        
        logger.info(f"[WHISPER] ✓ Transcription terminée en {elapsed:.1f}s")
        logger.info(f"[WHISPER] Langue détectée: {detected_lang}, {len(segments)} segments, {len(full_text)} caractères")
        
        return full_text, segments
        
    except Exception as e:
        logger.error(f"[WHISPER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return "", []


def perform_diarization(audio_path: str) -> List[Dict]:
    """Effectue la diarisation sur un fichier audio"""
    diarizer_pipeline = load_diarizer()
    
    if diarizer_pipeline is None:
        logger.warning("[DIARIZER] Diarisation non disponible")
        return []
    
    try:
        logger.info("[DIARIZER] Analyse en cours...")
        start_time = time.time()
        
        diarization = diarizer_pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
                "duration": round(turn.end - turn.start, 2)
            })
        
        elapsed = time.time() - start_time
        logger.info(f"[DIARIZER] ✓ {len(segments)} segments détectés en {elapsed:.1f}s")
        
        return segments
        
    except Exception as e:
        logger.error(f"[DIARIZER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return []


def merge_transcription_with_diarization(
    transcription_segments: List[Dict],
    diarization_segments: List[Dict]
) -> List[Dict]:
    """
    Fusionne les segments de transcription avec la diarisation
    Assigne chaque segment de texte au speaker correspondant
    """
    if not diarization_segments:
        return [{"speaker": "SPEAKER_00", **seg} for seg in transcription_segments]
    
    result = []
    
    for trans_seg in transcription_segments:
        trans_mid = (trans_seg["start"] + trans_seg["end"]) / 2
        
        # Trouver le speaker pour ce segment
        best_speaker = "UNKNOWN"
        best_overlap = 0
        
        for diar_seg in diarization_segments:
            # Calculer l'overlap
            overlap_start = max(trans_seg["start"], diar_seg["start"])
            overlap_end = min(trans_seg["end"], diar_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg["speaker"]
        
        result.append({
            "speaker": best_speaker,
            "start": trans_seg["start"],
            "end": trans_seg["end"],
            "text": trans_seg["text"]
        })
    
    return result


def generate_summary(text: str, language: str = "fr") -> str:
    """Génère un résumé simple de la transcription"""
    if not text or len(text) < 100:
        return text
    
    try:
        # Résumé simple : premières et dernières phrases
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        
        if len(sentences) <= 4:
            return text
        
        # Prendre les 2 premières et 2 dernières phrases
        summary_parts = sentences[:2] + ["..."] + sentences[-2:]
        summary = ". ".join(summary_parts)
        
        return f"Résumé: {summary}"
        
    except Exception as e:
        logger.error(f"[SUMMARY] Erreur: {e}")
        return ""


def analyze_sentiment(text: str, language: str = "fr") -> Dict[str, Any]:
    """Analyse le sentiment de la transcription"""
    if not text:
        return {"sentiment": "neutre", "score": 0.5}
    
    try:
        text_lower = text.lower()
        
        # Dictionnaires de mots pour analyse basique
        positive_words_fr = [
            "merci", "parfait", "excellent", "super", "génial", "bien", 
            "content", "satisfait", "formidable", "bravo", "ok", "d'accord",
            "agréable", "heureux", "plaisir", "bonne", "bon", "réussi"
        ]
        negative_words_fr = [
            "problème", "erreur", "mal", "mauvais", "insatisfait", "déçu",
            "terrible", "horrible", "nul", "pire", "difficile", "compliqué",
            "impossible", "échec", "plainte", "mécontent", "colère", "énervé"
        ]
        
        positive_count = sum(1 for word in positive_words_fr if word in text_lower)
        negative_count = sum(1 for word in negative_words_fr if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            sentiment = "neutre"
            score = 0.5
        elif positive_count > negative_count:
            sentiment = "positif"
            score = 0.5 + (positive_count / (total * 2))
        elif negative_count > positive_count:
            sentiment = "négatif"
            score = 0.5 - (negative_count / (total * 2))
        else:
            sentiment = "neutre"
            score = 0.5
        
        # Déterminer satisfaction client
        if score > 0.6:
            satisfaction = "satisfait"
        elif score < 0.4:
            satisfaction = "insatisfait"
        else:
            satisfaction = "neutre"
        
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "mots_positifs": positive_count,
            "mots_negatifs": negative_count,
            "satisfaction_client": satisfaction
        }
        
    except Exception as e:
        logger.error(f"[SENTIMENT] Erreur: {e}")
        return {"sentiment": "neutre", "score": 0.5}


def download_audio(url: str) -> str:
    """Télécharge un fichier audio depuis une URL"""
    try:
        logger.info(f"[DOWNLOAD] Téléchargement: {url}")
        
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        # Déterminer l'extension
        ext = ".wav"
        content_type = response.headers.get("content-type", "").lower()
        if "mp3" in content_type or url.lower().endswith(".mp3"):
            ext = ".mp3"
        elif "ogg" in content_type or url.lower().endswith(".ogg"):
            ext = ".ogg"
        elif "m4a" in content_type or url.lower().endswith(".m4a"):
            ext = ".m4a"
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        logger.info(f"[DOWNLOAD] ✓ Téléchargé: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"[DOWNLOAD] ✗ Erreur: {e}")
        return ""


def get_audio_duration(audio_path: str) -> float:
    """Retourne la durée d'un fichier audio en secondes"""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except:
        try:
            audio, sr = sf.read(audio_path)
            return len(audio) / sr
        except:
            return 0


def process_audio_request(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Traite une demande complète de transcription"""
    temp_files = []
    
    try:
        # Extraire les paramètres
        task = job_input.get("task", "transcribe")
        audio_url = job_input.get("audio_url", "")
        language = job_input.get("language")  # None = auto-detect
        include_summary = job_input.get("summary", job_input.get("with_summary", False))
        include_sentiment = job_input.get("sentiment", False)
        include_diarization = task == "transcribe_diarized"
        
        logger.info(f"[HANDLER] Tâche: {task}")
        logger.info(f"[HANDLER] Langue: {language or 'auto'}")
        logger.info(f"[HANDLER] Options: diarization={include_diarization}, summary={include_summary}, sentiment={include_sentiment}")
        
        if not audio_url:
            return {"error": "URL audio manquante"}
        
        # Téléchargement
        audio_path = download_audio(audio_url)
        if not audio_path:
            return {"error": "Échec du téléchargement audio"}
        temp_files.append(audio_path)
        
        # Vérification durée
        duration = get_audio_duration(audio_path)
        logger.info(f"[HANDLER] Durée audio: {duration:.1f}s")
        
        if duration > MAX_DURATION:
            return {"error": f"Audio trop long (max {MAX_DURATION}s)"}
        
        if duration == 0:
            return {"error": "Impossible de lire le fichier audio"}
        
        # Résultat
        result = {
            "task": task,
            "language": language or "auto",
            "duration": round(duration, 2)
        }
        
        # Transcription
        full_text, whisper_segments = transcribe_audio(audio_path, language)
        
        if not full_text:
            return {"error": "Échec de la transcription"}
        
        # Diarisation si demandée
        if include_diarization:
            logger.info("[HANDLER] Mode diarisation activé")
            diar_segments = perform_diarization(audio_path)
            
            # Fusionner transcription + diarisation
            merged_segments = merge_transcription_with_diarization(whisper_segments, diar_segments)
            result["transcriptions"] = merged_segments
            
            # Regrouper par speaker pour le texte complet
            speakers_text = {}
            for seg in merged_segments:
                speaker = seg["speaker"]
                if speaker not in speakers_text:
                    speakers_text[speaker] = []
                speakers_text[speaker].append(seg["text"])
            
            result["speakers"] = {k: " ".join(v) for k, v in speakers_text.items()}
        else:
            result["transcription"] = full_text
            result["segments"] = whisper_segments
        
        # Résumé
        if include_summary:
            result["summary"] = generate_summary(full_text, language or "fr")
        
        # Sentiment
        if include_sentiment:
            result["sentiment"] = analyze_sentiment(full_text, language or "fr")
        
        logger.info("[HANDLER] ✓ Traitement terminé avec succès")
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
        
    finally:
        # Nettoyage des fichiers temporaires
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        cleanup_gpu()


def handler(job):
    """Handler principal pour RunPod serverless"""
    try:
        job_id = job.get("id", "unknown")
        logger.info(f"[HANDLER] === Nouveau job: {job_id} ===")
        
        job_input = job.get("input", {})
        
        if not job_input:
            return {"error": "Pas d'input fourni"}
        
        result = process_audio_request(job_input)
        
        result_size = len(json.dumps(result, ensure_ascii=False))
        logger.info(f"[HANDLER] ✓ Job terminé: {result_size} caractères")
        
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def initialize():
    """Initialise les modèles au démarrage"""
    logger.info("=== Whisper Serverless Worker ===")
    logger.info(f"Modèle Whisper: {WHISPER_MODEL}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Durée max: {MAX_DURATION}s")
    
    log_gpu_memory()
    
    # Pré-charger Whisper
    logger.info("[INIT] Chargement de Whisper...")
    if load_whisper_model():
        logger.info("[INIT] ✓ Whisper prêt")
    else:
        logger.error("[INIT] ✗ Échec chargement Whisper")
    
    # Pré-charger Pyannote
    logger.info("[INIT] Chargement de Pyannote...")
    if load_diarizer():
        logger.info("[INIT] ✓ Pyannote prêt")
    else:
        logger.warning("[INIT] ⚠ Pyannote non disponible (HF_TOKEN manquant?)")
    
    log_gpu_memory()
    logger.info("[INIT] ✓ Initialisation terminée")


if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})
