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
import signal
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
import tempfile
import requests
from pathlib import Path

import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM
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
MODEL_DOWNLOAD_TIMEOUT = 3600  # 60 minutes pour le téléchargement (modèle 45GB)

@contextmanager
def timeout(duration):
    """Context manager pour timeout sur le téléchargement de modèles"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Opération timeout après {duration} secondes")
    
    # Configurer le signal d'alarme
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
@contextmanager
def timeout(duration):
    """Context manager pour timeout sur le téléchargement de modèles"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Opération timeout après {duration} secondes")
    
    # Configurer le signal d'alarme
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        # Nettoyer
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def warm_model_cache():
    """Pré-télécharge le modèle Voxtral pour accélérer le démarrage"""
    from huggingface_hub import snapshot_download, scan_cache_dir
    
    logger.info("=== Voxtral Cache Warming ===")
    
    # Vérifier le cache existant
    try:
        cache_info = scan_cache_dir()
        cached_repos = [repo.repo_id for repo in cache_info.repos]
        
        if VOXTRAL_MODEL in cached_repos:
            logger.info(f"✅ Modèle {VOXTRAL_MODEL} déjà en cache")
            for repo in cache_info.repos:
                if repo.repo_id == VOXTRAL_MODEL:
                    logger.info(f"   Taille: {repo.size_on_disk_str}")
            return True
        else:
            logger.info(f"📥 Modèle {VOXTRAL_MODEL} non trouvé en cache, téléchargement...")
    except Exception as e:
        logger.warning(f"Impossible de vérifier le cache: {e}")
    
    # Télécharger le modèle
    try:
        hf_token = os.getenv("HF_TOKEN")
        start_time = time.time()
        
        logger.info("🚀 Début du téléchargement...")
        
        snapshot_path = snapshot_download(
            repo_id=VOXTRAL_MODEL,
            token=hf_token,
            cache_dir=None,
            resume_download=True,
            local_files_only=False
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Téléchargement terminé en {elapsed:.1f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement: {e}")
        return False
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
        # Chargement du processor Voxtral - approche simplifiée sans arguments problématiques
        logger.info("[VOXTRAL] Chargement du processor...")
        
        try:
            # Approche directe sans passer par AutoProcessor qui ajoute des kwargs problématiques
            from transformers.models.voxtral import VoxtralProcessor
            
            # Chargement sans les kwargs qui posent problème
            voxtral_processor = VoxtralProcessor.from_pretrained(VOXTRAL_MODEL)
            logger.info("[VOXTRAL] ✓ VoxtralProcessor chargé directement")
            
        except Exception as e:
            logger.warning(f"[VOXTRAL] ⚠ VoxtralProcessor direct échoué: {e}")
            try:
                # Fallback: chargement manuel des composants
                from transformers import AutoTokenizer, AutoFeatureExtractor
                
                logger.info("[VOXTRAL] Chargement des composants individuels...")
                
                if hf_token:
                    tokenizer = AutoTokenizer.from_pretrained(VOXTRAL_MODEL, token=hf_token)
                    feature_extractor = AutoFeatureExtractor.from_pretrained(VOXTRAL_MODEL, token=hf_token)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(VOXTRAL_MODEL)
                    feature_extractor = AutoFeatureExtractor.from_pretrained(VOXTRAL_MODEL)
                
                # Création manuelle d'un objet processor
                class VoxtralProcessorManual:
                    def __init__(self, tokenizer, feature_extractor):
                        self.tokenizer = tokenizer
                        self.feature_extractor = feature_extractor
                    
                    def __call__(self, audio=None, text=None, sampling_rate=16000, return_tensors="pt"):
                        result = {}
                        if audio is not None:
                            # Process audio through feature extractor
                            audio_features = self.feature_extractor(
                                audio, 
                                sampling_rate=sampling_rate, 
                                return_tensors=return_tensors
                            )
                            result.update(audio_features)
                        if text is not None:
                            # Process text through tokenizer
                            text_features = self.tokenizer(
                                text, 
                                return_tensors=return_tensors,
                                padding=True,
                                truncation=True
                            )
                            result.update(text_features)
                        return result
                    
                    def batch_decode(self, *args, **kwargs):
                        return self.tokenizer.batch_decode(*args, **kwargs)
                
                voxtral_processor = VoxtralProcessorManual(tokenizer, feature_extractor)
                logger.info("[VOXTRAL] ✓ Processor manuel créé")
                
            except Exception as e2:
                logger.error(f"[VOXTRAL] ✗ Erreur processor manuel: {e2}")
                return None, None
        
        # Chargement du modèle avec approche simplifiée
        logger.info("[VOXTRAL] Chargement du modèle...")
        try:
            # Vérification de la disponibilité du modèle dans le cache
            from huggingface_hub import scan_cache_dir, try_to_load_from_cache
            
            # Essayer de charger depuis le cache d'abord
            cache_info = None
            try:
                cache_info = scan_cache_dir()
                cached_repo = None
                for repo in cache_info.repos:
                    if VOXTRAL_MODEL.split('/')[-1] in repo.repo_id:
                        cached_repo = repo
                        break
                
                if cached_repo:
                    logger.info(f"[VOXTRAL] Modèle trouvé dans le cache: {cached_repo.size_on_disk_str}")
                else:
                    logger.info("[VOXTRAL] Aucun cache trouvé, téléchargement nécessaire")
            except Exception as cache_err:
                logger.warning(f"[VOXTRAL] ⚠ Impossible de vérifier le cache: {cache_err}")
            
            # Chargement du vrai modèle Voxtral avec approche directe et timeout
            logger.info(f"[VOXTRAL] Chargement du modèle Voxtral (timeout: {MODEL_DOWNLOAD_TIMEOUT}s)...")
            
            try:
                # Première tentative: import direct de VoxtralForConditionalGeneration
                from transformers.models.voxtral import VoxtralForConditionalGeneration
                
                with timeout(MODEL_DOWNLOAD_TIMEOUT):
                    # Ajout de logs détaillés pour le suivi du téléchargement
                    logger.info("[VOXTRAL] Début du téléchargement/chargement du modèle (45GB)...")
                    start_time = time.time()
                    
                    if hf_token:
                        voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
                            VOXTRAL_MODEL,
                            token=hf_token,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            resume_download=True,
                            force_download=False,
                            local_files_only=False  # Permettre le téléchargement si nécessaire
                        )
                    else:
                        voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
                            VOXTRAL_MODEL,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            resume_download=True,
                            force_download=False,
                            local_files_only=False  # Permettre le téléchargement si nécessaire
                        )
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"[VOXTRAL] ✓ Modèle chargé en {elapsed_time:.1f}s")
                
                logger.info("[VOXTRAL] ✓ VoxtralForConditionalGeneration chargé")
                
            except (ImportError, TimeoutError) as e:
                if isinstance(e, TimeoutError):
                    logger.error(f"[VOXTRAL] ✗ Timeout lors du chargement VoxtralForConditionalGeneration: {e}")
                else:
                    logger.warning("[VOXTRAL] ⚠ VoxtralForConditionalGeneration non disponible, tentative AutoModel...")
                
                # Deuxième tentative: utilisation d'AutoModel avec trust_remote_code et timeout
                try:
                    logger.info("[VOXTRAL] Tentative de fallback avec AutoModelForCausalLM...")
                    start_time_fallback = time.time()
                    
                    with timeout(MODEL_DOWNLOAD_TIMEOUT):
                        if hf_token:
                            voxtral_model = AutoModelForCausalLM.from_pretrained(
                                VOXTRAL_MODEL,
                                token=hf_token,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                resume_download=True,
                                force_download=False,
                                local_files_only=False
                            )
                        else:
                            voxtral_model = AutoModelForCausalLM.from_pretrained(
                                VOXTRAL_MODEL,
                                torch_dtype=torch.bfloat16,
                                device_map="auto",
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                resume_download=True,
                                force_download=False,
                                local_files_only=False
                            )
                    
                    elapsed_time_fallback = time.time() - start_time_fallback
                    logger.info(f"[VOXTRAL] ✓ AutoModel chargé en {elapsed_time_fallback:.1f}s")
                    
                    logger.info("[VOXTRAL] ✓ AutoModelForCausalLM avec trust_remote_code chargé")
                    
                except TimeoutError as timeout_err:
                    logger.error(f"[VOXTRAL] ✗ Timeout lors du chargement AutoModel: {timeout_err}")
                    raise
            
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
    Charge le pipeline de diarisation PyAnnote avec gestion des versions d'API
    """
    global diarizer
    
    if diarizer is not None:
        return diarizer
    
    try:
        logger.info(f"[DIARIZER] Chargement: {DIARIZATION_MODEL}")
        hf_token = os.getenv("HF_TOKEN")
        
        try:
            # Première tentative avec token (nouvelle API)
            if hf_token:
                diarizer = Pipeline.from_pretrained(
                    DIARIZATION_MODEL,
                    token=hf_token
                )
            else:
                diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL)
        except Exception as e1:
            logger.warning(f"[DIARIZER] ⚠ token échoué: {e1}")
            try:
                # Deuxième tentative avec use_auth_token (ancienne API)
                if hf_token:
                    diarizer = Pipeline.from_pretrained(
                        DIARIZATION_MODEL,
                        use_auth_token=hf_token
                    )
                else:
                    diarizer = Pipeline.from_pretrained(DIARIZATION_MODEL)
            except Exception as e2:
                logger.warning(f"[DIARIZER] ⚠ use_auth_token échoué: {e2}")
                # Troisième tentative sans authentification
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
        logger.info(f"[VOXTRAL] Début transcription: {audio_path}")
        model, processor = load_voxtral_model()
        if model is None or processor is None:
            raise Exception("Impossible de charger Voxtral")
        
        logger.info(f"[VOXTRAL] Transcription (max_tokens={max_tokens})")
        
        # Chargement de l'audio
        logger.info("[VOXTRAL] Lecture du fichier audio...")
        audio, sample_rate = sf.read(audio_path)
        logger.info(f"[VOXTRAL] Audio: {len(audio)} samples @ {sample_rate}Hz")
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            logger.info("[VOXTRAL] Audio converti en mono")
        
        # Conversion au sample rate attendu (16kHz généralement)
        if sample_rate != 16000:
            logger.info(f"[VOXTRAL] Resampling {sample_rate}Hz -> 16000Hz")
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Préparation des inputs avec le processor
        logger.info("[VOXTRAL] Préparation des inputs...")
        inputs = processor(
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        logger.info(f"[VOXTRAL] Inputs préparés: {list(inputs.keys())}")
        
        # Déplacement vers GPU si disponible
        if DEVICE == "cuda":
            logger.info("[VOXTRAL] Déplacement vers GPU...")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Génération avec Voxtral
        logger.info("[VOXTRAL] Génération en cours...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else 2
            )
        
        logger.info("[VOXTRAL] Décodage...")
        # Décodage de la transcription
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"[VOXTRAL] ✓ Transcription terminée ({len(transcription)} caractères)")
        logger.info(f"[VOXTRAL] Aperçu: {transcription[:100]}...")
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur transcription: {e}")
        import traceback
        traceback.print_exc()
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
    try:
        logger.info(f"[HANDLER] Nouveau job reçu: {job}")
        job_input = job.get("input", {})
        
        if not job_input:
            logger.error("[HANDLER] ✗ Pas d'input dans le job")
            return {"error": "Pas d'input fourni"}
        
        result = process_audio_request(job_input)
        logger.info(f"[HANDLER] ✓ Job terminé. Résultat: {len(str(result))} caractères")
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Erreur dans handler: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Erreur handler: {str(e)}"}

if __name__ == "__main__":
    import sys
    
    # Support pour cache warming
    if len(sys.argv) > 1 and sys.argv[1] == "warm_cache":
        success = warm_model_cache()
        sys.exit(0 if success else 1)
    
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
