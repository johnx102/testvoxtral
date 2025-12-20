#!/usr/bin/env python3
"""
Voxtral Serverless Worker - Service de transcription avec diarisation
Optimisé pour Runpod avec versions stables des transformers
"""

import os
import json
import time
import logging
import gc
import warnings
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import requests
from pathlib import Path

import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoProcessor, MistralForConditionalGeneration
from pyannote.audio import Pipeline
import runpod

# Configuration des warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")

# Suppression des warnings torchcodec spécifiquement
import logging
torchcodec_logger = logging.getLogger("pyannote.audio.core.io")
torchcodec_logger.setLevel(logging.ERROR)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration globale
VOXTRAL_MODEL = "mistralai/Voxtral-Small-24B-2507"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DURATION = int(os.getenv("MAX_DURATION_S", "9000"))

# Variables globales pour le cache des modèles
voxtral_model = None
voxtral_processor = None
diarizer = None

def log_gpu_memory():
    """Affiche l'utilisation mémoire GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        logger.info(f"[GPU] Total: {total:.1f}GB | Allocated: {allocated:.1f}GB | Cached: {cached:.1f}GB | Free: {free:.1f}GB")

def cleanup_gpu():
    """Nettoie la mémoire GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_voxtral_model() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Charge le modèle Voxtral et son processor avec gestion d'erreurs robuste
    """
    global voxtral_model, voxtral_processor
    
    if voxtral_model is not None and voxtral_processor is not None:
        return voxtral_model, voxtral_processor
    
    try:
        logger.info(f"[VOXTRAL] Chargement du modèle: {VOXTRAL_MODEL}")
        
        # Chargement du processor avec gestion HF token
        hf_token = os.getenv("HF_TOKEN")
        logger.info("[VOXTRAL] Chargement du processor...")
        
        try:
            if hf_token:
                voxtral_processor = AutoProcessor.from_pretrained(
                    VOXTRAL_MODEL,
                    token=hf_token,
                    trust_remote_code=True
                )
            else:
                voxtral_processor = AutoProcessor.from_pretrained(
                    VOXTRAL_MODEL,
                    trust_remote_code=True
                )
            logger.info("[VOXTRAL] ✓ Processor chargé avec succès")
        except Exception as e:
            logger.error(f"[VOXTRAL] ✗ Erreur processor: {e}")
            return None, None
        
        # Chargement du modèle
        logger.info("[VOXTRAL] Chargement du modèle...")
        try:
            if hf_token:
                voxtral_model = MistralForConditionalGeneration.from_pretrained(
                    VOXTRAL_MODEL,
                    token=hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                voxtral_model = MistralForConditionalGeneration.from_pretrained(
                    VOXTRAL_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            logger.info("[VOXTRAL] ✓ Modèle chargé avec succès")
            log_gpu_memory()
            return voxtral_model, voxtral_processor
            
        except Exception as e:
            logger.error(f"[VOXTRAL] ✗ Erreur modèle: {e}")
            voxtral_processor = None
            return None, None
            
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur générale: {e}")
        return None, None

def load_diarizer() -> Optional[Pipeline]:
    """
    Charge le pipeline de diarisation PyAnnote
    """
    global diarizer
    
    if diarizer is not None:
        return diarizer
    
    try:
        logger.info(f"[DIARIZER] Chargement: {DIARIZATION_MODEL}")
        hf_token = os.getenv("HF_TOKEN")
        
        if hf_token:
            diarizer = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                use_auth_token=hf_token
            )
        else:
            diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL)
            
        diarizer.to(torch.device(DEVICE))
        logger.info("[DIARIZER] ✓ Diarizer chargé avec succès")
        return diarizer
        
    except Exception as e:
        logger.error(f"[DIARIZER] ✗ Erreur: {e}")
        return None

def transcribe_with_voxtral(audio_path: str, max_tokens: int = 2500) -> str:
    """
    Transcrit un fichier audio avec Voxtral
    """
    try:
        model, processor = load_voxtral_model()
        if model is None or processor is None:
            raise Exception("Impossible de charger Voxtral")
        
        logger.info(f"[VOXTRAL] Transcription (max_tokens={max_tokens})")
        
        # Chargement de l'audio
        audio, sample_rate = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Conversion au sample rate attendu (16kHz généralement)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Préparation des inputs
        inputs = processor(
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Déplacement vers GPU si disponible
        if DEVICE == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Génération
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Décodage
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"[VOXTRAL] ✓ Transcription terminée ({len(transcription)} caractères)")
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur transcription: {e}")
        return ""

def perform_diarization(audio_path: str) -> List[Dict]:
    """
    Effectue la diarisation sur un fichier audio
    """
    try:
        diarizer_pipeline = load_diarizer()
        if diarizer_pipeline is None:
            raise Exception("Impossible de charger le diarizer")
        
        logger.info("[DIARIZER] Analyse de la diarisation...")
        diarization = diarizer_pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
                "duration": round(turn.end - turn.start, 2)
            })
        
        logger.info(f"[DIARIZER] ✓ {len(segments)} segments détectés")
        return segments
        
    except Exception as e:
        logger.error(f"[DIARIZER] ✗ Erreur: {e}")
        return []

def extract_audio_segment(audio_path: str, start_time: float, end_time: float) -> str:
    """
    Extrait un segment audio entre start_time et end_time
    """
    try:
        # Lecture de l'audio complet
        audio, sample_rate = sf.read(audio_path)
        
        # Calcul des indices de début et fin
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        # Extraction du segment
        segment = audio[start_idx:end_idx]
        
        # Sauvegarde temporaire
        temp_path = f"/tmp/segment_{int(start_time)}_{int(end_time)}.wav"
        sf.write(temp_path, segment, sample_rate)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"[EXTRACT] Erreur extraction segment: {e}")
        return ""

def analyze_sentiment(text: str) -> str:
    """
    Analyse basique du sentiment (peut être étendue avec un modèle dédié)
    """
    if not text:
        return "neutre"
    
    # Mots-clés positifs et négatifs (français)
    positive_words = ["merci", "bien", "parfait", "excellent", "content", "satisfait", "ok"]
    negative_words = ["problème", "erreur", "mal", "mauvais", "pas bien", "insatisfait", "déçu"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positif"
    elif negative_count > positive_count:
        return "négatif"
    else:
        return "neutre"

def create_summary(transcription: str) -> str:
    """
    Crée un résumé basique de la transcription
    """
    if not transcription:
        return "Aucune transcription disponible"
    
    # Résumé simple par troncature intelligente
    sentences = transcription.split('.')
    if len(sentences) <= 3:
        return transcription
    
    # Prendre les premières et dernières phrases
    summary_sentences = sentences[:2] + sentences[-1:]
    summary = '. '.join(s.strip() for s in summary_sentences if s.strip())
    
    return f"Résumé: {summary}"

def download_audio(url: str) -> str:
    """
    Télécharge un fichier audio depuis une URL
    """
    try:
        logger.info(f"[DOWNLOAD] Téléchargement: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Création d'un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        logger.info(f"[DOWNLOAD] ✓ Audio téléchargé: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"[DOWNLOAD] ✗ Erreur: {e}")
        return ""

def process_audio_request(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite une demande de transcription audio
    """
    temp_files = []
    
    try:
        # Extraction des paramètres
        task = job_input.get("task", "transcribe")
        audio_url = job_input.get("audio_url", "")
        language = job_input.get("language", "fr")
        max_tokens = job_input.get("max_tokens", 2500)
        include_summary = job_input.get("summary", False)
        include_sentiment = job_input.get("sentiment", False)
        
        logger.info(f"[HANDLER] Tâche: {task}, langue: {language}, tokens: {max_tokens}")
        
        if not audio_url:
            return {"error": "URL audio manquante"}
        
        # Téléchargement de l'audio
        audio_path = download_audio(audio_url)
        if not audio_path:
            return {"error": "Échec du téléchargement audio"}
        temp_files.append(audio_path)
        
        # Vérification de la durée
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        logger.info(f"[HANDLER] Durée audio: {duration:.1f}s")
        
        if duration > MAX_DURATION:
            return {"error": f"Audio trop long (max {MAX_DURATION}s)"}
        
        result = {"task": task, "language": language, "duration": round(duration, 2)}
        
        # Traitement selon la tâche demandée
        if task in ["transcribe", "transcribe_diarized"]:
            if task == "transcribe_diarized":
                # Diarisation + transcription par segment
                logger.info("[HANDLER] Mode diarisation activé")
                segments = perform_diarization(audio_path)
                
                if segments:
                    transcriptions = []
                    for segment in segments:
                        # Extraction du segment audio
                        segment_path = extract_audio_segment(
                            audio_path, segment["start"], segment["end"]
                        )
                        if segment_path:
                            temp_files.append(segment_path)
                            
                            # Transcription du segment
                            segment_text = transcribe_with_voxtral(segment_path, 500)
                            
                            transcriptions.append({
                                "speaker": segment["speaker"],
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment_text
                            })
                    
                    result["transcriptions"] = transcriptions
                    
                    # Transcription complète concatenée
                    full_transcription = " ".join(t["text"] for t in transcriptions if t["text"])
                else:
                    # Fallback: transcription sans diarisation
                    logger.info("[HANDLER] Fallback: transcription simple")
                    full_transcription = transcribe_with_voxtral(audio_path, max_tokens)
                    result["transcriptions"] = [{"speaker": "UNKNOWN", "text": full_transcription}]
            else:
                # Transcription simple
                full_transcription = transcribe_with_voxtral(audio_path, max_tokens)
                result["transcription"] = full_transcription
            
            # Ajout du résumé si demandé
            if include_summary and 'transcriptions' in result:
                full_text = " ".join(t["text"] for t in result["transcriptions"] if t["text"])
                result["summary"] = create_summary(full_text)
            elif include_summary and 'transcription' in result:
                result["summary"] = create_summary(result["transcription"])
            
            # Ajout du sentiment si demandé
            if include_sentiment and 'transcriptions' in result:
                full_text = " ".join(t["text"] for t in result["transcriptions"] if t["text"])
                result["sentiment"] = analyze_sentiment(full_text)
            elif include_sentiment and 'transcription' in result:
                result["sentiment"] = analyze_sentiment(result["transcription"])
        
        logger.info("[HANDLER] ✓ Traitement terminé avec succès")
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Erreur: {e}")
        return {"error": str(e)}
        
    finally:
        # Nettoyage des fichiers temporaires
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        
        cleanup_gpu()

def initialize_models():
    """
    Initialise les modèles au démarrage (optionnel)
    """
    try:
        logger.info("[INIT] Pré-chargement des modèles...")
        log_gpu_memory()
        
        # Pré-chargement Voxtral
        model, processor = load_voxtral_model()
        if model and processor:
            logger.info("[INIT] ✓ Voxtral pré-chargé")
        else:
            logger.warning("[INIT] ⚠ Échec pré-chargement Voxtral")
        
        # Pré-chargement diarizer
        diarizer_pipeline = load_diarizer()
        if diarizer_pipeline:
            logger.info("[INIT] ✓ Diarizer pré-chargé")
        else:
            logger.warning("[INIT] ⚠ Échec pré-chargement Diarizer")
        
        log_gpu_memory()
        logger.info("[INIT] ✓ Initialisation terminée")
        
    except Exception as e:
        logger.error(f"[INIT] ✗ Erreur initialisation: {e}")

# Point d'entrée principal pour RunPod
def handler(job):
    """
    Handler principal pour RunPod serverless
    """
    job_input = job.get("input", {})
    return process_audio_request(job_input)

if __name__ == "__main__":
    # Test local ou démarrage RunPod
    if os.getenv("RUNPOD_DEBUG"):
        # Mode debug local
        test_input = {
            "task": "transcribe_diarized",
            "audio_url": "https://example.com/test.wav",
            "language": "fr",
            "max_tokens": 1000,
            "summary": True,
            "sentiment": True
        }
        result = process_audio_request(test_input)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Mode production RunPod
        initialize_models()
        runpod.serverless.start({"handler": handler})
