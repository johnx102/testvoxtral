#!/usr/bin/env python3
"""
Voxtral Serverless Worker - Transcription + Diarisation + Résumé + Sentiment
Utilise vLLM pour servir Voxtral + Pyannote pour diarisation
Version finale 2025-12-21
"""

import os
import json
import time
import logging
import gc
import warnings
import base64
from typing import Optional, Dict, Any, List
import tempfile
import requests
from pathlib import Path

import torch
import soundfile as sf
import librosa
import numpy as np
import runpod
from openai import OpenAI

# Configuration des warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")

# Suppression des warnings pyannote
logging.getLogger("pyannote.audio.core.io").setLevel(logging.ERROR)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
VLLM_PORT = os.getenv("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DURATION = int(os.getenv("MAX_DURATION_S", "9000"))
MAX_RETRIES_VLLM = 60  # 5 minutes pour que vLLM démarre

# Cache global
diarizer = None
openai_client = None


def wait_for_vllm():
    """Vérifie que vLLM est prêt (devrait déjà l'être via start.sh)"""
    global openai_client
    
    logger.info(f"[VLLM] Connexion au serveur vLLM sur {VLLM_BASE_URL}...")
    
    for i in range(30):  # 30 secondes max
        try:
            response = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"[VLLM] ✓ Serveur prêt. Modèles: {[m['id'] for m in models.get('data', [])]}")
                
                # Créer le client OpenAI
                openai_client = OpenAI(
                    api_key="EMPTY",  # vLLM n'a pas besoin de clé
                    base_url=VLLM_BASE_URL
                )
                return True
        except Exception as e:
            logger.debug(f"[VLLM] Attente... {e}")
        time.sleep(1)
    
    logger.error("[VLLM] ✗ vLLM n'est pas accessible")
    return False


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
            logger.error("[DIARIZER] ✗ HF_TOKEN non défini")
            return None
        
        # Essayer les deux APIs
        try:
            diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, token=hf_token)
        except TypeError:
            diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
        
        diarizer.to(torch.device(DEVICE))
        logger.info("[DIARIZER] ✓ Diarizer chargé")
        return diarizer
        
    except Exception as e:
        logger.error(f"[DIARIZER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def audio_to_base64(audio_path: str) -> str:
    """Convertit un fichier audio en base64"""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def transcribe_audio(audio_path: str, language: str = "fr") -> str:
    """Transcrit un audio avec Voxtral via vLLM"""
    global openai_client
    
    if openai_client is None:
        raise Exception("vLLM client non initialisé")
    
    try:
        logger.info(f"[VOXTRAL] Transcription: {audio_path}")
        
        # Lire l'audio et le convertir en base64
        audio_base64 = audio_to_base64(audio_path)
        
        # Utiliser l'API de transcription vLLM
        # Note: vLLM expose une API compatible OpenAI
        from mistral_common.protocol.transcription.request import TranscriptionRequest
        from mistral_common.protocol.instruct.messages import RawAudio
        from mistral_common.audio import Audio
        
        # Charger l'audio avec mistral_common
        audio = Audio.from_file(audio_path, strict=False)
        raw_audio = RawAudio.from_audio(audio)
        
        # Créer la requête de transcription
        models = openai_client.models.list()
        model_id = models.data[0].id
        
        req = TranscriptionRequest(
            model=model_id,
            audio=raw_audio,
            language=language,
            temperature=0.0
        ).to_openai(exclude=("top_p", "seed"))
        
        response = openai_client.audio.transcriptions.create(**req)
        
        transcription = response.text if hasattr(response, 'text') else str(response)
        logger.info(f"[VOXTRAL] ✓ Transcription: {len(transcription)} caractères")
        
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur transcription: {e}")
        import traceback
        traceback.print_exc()
        return ""


def generate_summary(text: str, language: str = "fr") -> str:
    """Génère un résumé avec Voxtral"""
    global openai_client
    
    if openai_client is None or not text:
        return ""
    
    try:
        logger.info("[VOXTRAL] Génération du résumé...")
        
        models = openai_client.models.list()
        model_id = models.data[0].id
        
        prompt = f"""Résume cette conversation téléphonique en français de manière concise et structurée.
Identifie les points clés, les décisions prises et les actions à suivre.

Transcription:
{text}

Résumé:"""
        
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        summary = response.choices[0].message.content
        logger.info(f"[VOXTRAL] ✓ Résumé généré: {len(summary)} caractères")
        return summary.strip()
        
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur résumé: {e}")
        return ""


def analyze_sentiment(text: str, language: str = "fr") -> Dict[str, Any]:
    """Analyse le sentiment avec Voxtral"""
    global openai_client
    
    if openai_client is None or not text:
        return {"sentiment": "neutre", "score": 0.5}
    
    try:
        logger.info("[VOXTRAL] Analyse du sentiment...")
        
        models = openai_client.models.list()
        model_id = models.data[0].id
        
        prompt = f"""Analyse le sentiment de cette conversation téléphonique.
Réponds UNIQUEMENT avec un JSON valide contenant:
- "sentiment": "positif", "négatif" ou "neutre"
- "score": un nombre entre 0 et 1 (0=très négatif, 1=très positif)
- "emotions": liste des émotions détectées
- "satisfaction_client": "satisfait", "insatisfait" ou "neutre"

Transcription:
{text}

JSON:"""
        
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parser le JSON
        try:
            # Nettoyer le texte
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text)
            logger.info(f"[VOXTRAL] ✓ Sentiment: {result.get('sentiment', 'neutre')}")
            return result
        except json.JSONDecodeError:
            # Fallback si pas de JSON valide
            if "positif" in result_text.lower():
                return {"sentiment": "positif", "score": 0.7}
            elif "négatif" in result_text.lower():
                return {"sentiment": "négatif", "score": 0.3}
            else:
                return {"sentiment": "neutre", "score": 0.5}
        
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur sentiment: {e}")
        return {"sentiment": "neutre", "score": 0.5}


def perform_diarization(audio_path: str) -> List[Dict]:
    """Effectue la diarisation sur un fichier audio"""
    try:
        diarizer_pipeline = load_diarizer()
        if diarizer_pipeline is None:
            return []
        
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
    """Extrait un segment audio"""
    try:
        audio, sample_rate = sf.read(audio_path)
        
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        segment = audio[start_idx:end_idx]
        
        temp_path = f"/tmp/segment_{int(start_time*1000)}_{int(end_time*1000)}.wav"
        sf.write(temp_path, segment, sample_rate)
        
        return temp_path
    except Exception as e:
        logger.error(f"[EXTRACT] Erreur: {e}")
        return ""


def download_audio(url: str) -> str:
    """Télécharge un fichier audio depuis une URL"""
    try:
        logger.info(f"[DOWNLOAD] Téléchargement: {url}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
        # Déterminer l'extension
        ext = ".wav"
        if ".mp3" in url.lower():
            ext = ".mp3"
        elif ".ogg" in url.lower():
            ext = ".ogg"
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
        
        # Convertir en WAV si nécessaire
        if ext != ".wav":
            wav_path = temp_path.replace(ext, ".wav")
            audio, sr = librosa.load(temp_path, sr=16000)
            sf.write(wav_path, audio, sr)
            os.unlink(temp_path)
            temp_path = wav_path
        
        logger.info(f"[DOWNLOAD] ✓ Audio téléchargé: {temp_path}")
        return temp_path
        
    except Exception as e:
        logger.error(f"[DOWNLOAD] ✗ Erreur: {e}")
        return ""


def cleanup_gpu():
    """Nettoie la mémoire GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def process_audio_request(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Traite une demande complète"""
    temp_files = []
    
    try:
        # Paramètres
        task = job_input.get("task", "transcribe")
        audio_url = job_input.get("audio_url", "")
        language = job_input.get("language", "fr")
        include_summary = job_input.get("summary", job_input.get("with_summary", False))
        include_sentiment = job_input.get("sentiment", False)
        include_diarization = task == "transcribe_diarized"
        
        logger.info(f"[HANDLER] Tâche: {task}, Langue: {language}")
        logger.info(f"[HANDLER] Options: summary={include_summary}, sentiment={include_sentiment}, diarization={include_diarization}")
        
        if not audio_url:
            return {"error": "URL audio manquante"}
        
        # Téléchargement
        audio_path = download_audio(audio_url)
        if not audio_path:
            return {"error": "Échec du téléchargement audio"}
        temp_files.append(audio_path)
        
        # Vérification durée
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        logger.info(f"[HANDLER] Durée audio: {duration:.1f}s")
        
        if duration > MAX_DURATION:
            return {"error": f"Audio trop long (max {MAX_DURATION}s)"}
        
        result = {
            "task": task,
            "language": language,
            "duration": round(duration, 2)
        }
        
        # Diarisation si demandée
        if include_diarization:
            logger.info("[HANDLER] Mode diarisation activé")
            segments = perform_diarization(audio_path)
            
            if segments:
                transcriptions = []
                for segment in segments:
                    segment_path = extract_audio_segment(
                        audio_path, segment["start"], segment["end"]
                    )
                    if segment_path:
                        temp_files.append(segment_path)
                        segment_text = transcribe_audio(segment_path, language)
                        
                        transcriptions.append({
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment_text
                        })
                
                result["transcriptions"] = transcriptions
                full_text = " ".join(t["text"] for t in transcriptions if t["text"])
            else:
                logger.info("[HANDLER] Diarisation échouée, fallback transcription simple")
                full_text = transcribe_audio(audio_path, language)
                result["transcriptions"] = [{"speaker": "UNKNOWN", "text": full_text}]
        else:
            # Transcription simple
            full_text = transcribe_audio(audio_path, language)
            result["transcription"] = full_text
        
        # Résumé
        if include_summary and full_text:
            result["summary"] = generate_summary(full_text, language)
        
        # Sentiment
        if include_sentiment and full_text:
            result["sentiment"] = analyze_sentiment(full_text, language)
        
        logger.info("[HANDLER] ✓ Traitement terminé")
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
        
    finally:
        # Nettoyage
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
        logger.info(f"[HANDLER] Nouveau job: {job.get('id', 'unknown')}")
        job_input = job.get("input", {})
        
        if not job_input:
            return {"error": "Pas d'input fourni"}
        
        result = process_audio_request(job_input)
        logger.info(f"[HANDLER] ✓ Job terminé: {len(str(result))} caractères")
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def main():
    """Point d'entrée principal"""
    logger.info("=== Voxtral Serverless Worker ===")
    
    # Attendre que vLLM soit prêt
    if not wait_for_vllm():
        logger.error("vLLM n'a pas démarré. Arrêt du worker.")
        return
    
    # Pré-charger le diarizer
    load_diarizer()
    
    # Démarrer le worker RunPod
    logger.info("[INIT] ✓ Démarrage du worker RunPod")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
