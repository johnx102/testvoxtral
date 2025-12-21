#!/usr/bin/env python3
"""
Voxtral Serverless Worker - Service de transcription avec diarisation
Optimisé pour Runpod avec transformers dev (support Voxtral)
Version corrigée 2025-12-21 v7
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
import runpod

# Configuration des warnings AVANT les imports problématiques
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

# Suppression des warnings torchcodec
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
MODEL_DOWNLOAD_TIMEOUT = 3600  # 60 minutes pour le téléchargement
MAX_DURATION = int(os.getenv("MAX_DURATION_S", "9000"))

# Variables globales pour le cache des modèles
voxtral_model = None
voxtral_processor = None
diarizer = None


@contextmanager
def timeout(duration):
    """Context manager pour timeout sur le téléchargement de modèles"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Opération timeout après {duration} secondes")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def warm_model_cache():
    """Pré-télécharge le modèle Voxtral pour accélérer le démarrage"""
    from huggingface_hub import snapshot_download, scan_cache_dir
    
    logger.info("=== Voxtral Cache Warming ===")
    
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
    Charge le modèle Voxtral et son processor
    Utilise VoxtralForConditionalGeneration directement (pas AutoModel)
    """
    global voxtral_model, voxtral_processor
    
    if voxtral_model is not None and voxtral_processor is not None:
        return voxtral_model, voxtral_processor
    
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        logger.info(f"[VOXTRAL] Chargement du modèle: {VOXTRAL_MODEL}")
        
        # ============================================
        # CHARGEMENT MANUEL DES COMPOSANTS DU PROCESSOR
        # ============================================
        logger.info("[VOXTRAL] Chargement manuel des composants du processor...")
        
        from transformers import AutoFeatureExtractor
        from huggingface_hub import hf_hub_download
        
        # 1. Charger le feature extractor (pour l'audio)
        logger.info("[VOXTRAL] Chargement du feature extractor...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            VOXTRAL_MODEL,
            token=hf_token,
            trust_remote_code=True
        )
        logger.info("[VOXTRAL] ✓ Feature extractor chargé")
        
        # 2. Charger le tokenizer MistralCommon directement
        logger.info("[VOXTRAL] Chargement du tokenizer MistralCommon...")
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            
            tokenizer_path = hf_hub_download(
                repo_id=VOXTRAL_MODEL,
                filename="tekken.json",
                token=hf_token
            )
            
            mistral_tokenizer = MistralTokenizer.from_file(tokenizer_path)
            logger.info("[VOXTRAL] ✓ MistralTokenizer chargé depuis tekken.json")
            
        except Exception as e:
            logger.warning(f"[VOXTRAL] ⚠ MistralTokenizer direct échoué: {e}")
            from transformers import AutoTokenizer
            mistral_tokenizer = AutoTokenizer.from_pretrained(
                VOXTRAL_MODEL,
                token=hf_token,
                trust_remote_code=True,
                use_fast=False
            )
            logger.info("[VOXTRAL] ✓ Tokenizer chargé via AutoTokenizer fallback")
        
        # 3. Créer un processor wrapper
        class VoxtralProcessorWrapper:
            """Wrapper pour combiner feature_extractor et tokenizer"""
            def __init__(self, feature_extractor, tokenizer):
                self.feature_extractor = feature_extractor
                self.tokenizer = tokenizer
                self._is_mistral_tokenizer = hasattr(tokenizer, 'instruct_tokenizer')
            
            def __call__(self, audio=None, text=None, sampling_rate=16000, return_tensors="pt", **kwargs):
                result = {}
                
                if audio is not None:
                    audio_inputs = self.feature_extractor(
                        audio,
                        sampling_rate=sampling_rate,
                        return_tensors=return_tensors,
                        **kwargs
                    )
                    result.update(audio_inputs)
                
                if text is not None:
                    if self._is_mistral_tokenizer:
                        from mistral_common.protocol.instruct.messages import UserMessage
                        from mistral_common.protocol.instruct.request import ChatCompletionRequest
                        
                        tokenized = self.tokenizer.encode_chat_completion(
                            ChatCompletionRequest(messages=[UserMessage(content=text)])
                        )
                        result["input_ids"] = torch.tensor([tokenized.tokens])
                    else:
                        text_inputs = self.tokenizer(
                            text,
                            return_tensors=return_tensors,
                            padding=True,
                            truncation=True
                        )
                        result.update(text_inputs)
                
                return result
            
            def batch_decode(self, token_ids, skip_special_tokens=True, **kwargs):
                if self._is_mistral_tokenizer:
                    results = []
                    for ids in token_ids:
                        if hasattr(ids, 'tolist'):
                            ids = ids.tolist()
                        text = self.tokenizer.decode(ids)
                        results.append(text)
                    return results
                else:
                    return self.tokenizer.batch_decode(
                        token_ids,
                        skip_special_tokens=skip_special_tokens,
                        **kwargs
                    )
            
            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                return self.batch_decode([token_ids], skip_special_tokens=skip_special_tokens, **kwargs)[0]
        
        voxtral_processor = VoxtralProcessorWrapper(feature_extractor, mistral_tokenizer)
        logger.info("[VOXTRAL] ✓ Processor wrapper créé")
        
        # ============================================
        # CHARGEMENT DU MODÈLE avec VoxtralForConditionalGeneration
        # ============================================
        logger.info("[VOXTRAL] Chargement du modèle avec VoxtralForConditionalGeneration...")
        logger.info(f"[VOXTRAL] Timeout configuré: {MODEL_DOWNLOAD_TIMEOUT}s")
        
        start_time = time.time()
        
        # Import direct de la classe Voxtral
        from transformers import VoxtralForConditionalGeneration
        
        with timeout(MODEL_DOWNLOAD_TIMEOUT):
            voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
                VOXTRAL_MODEL,
                token=hf_token,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"[VOXTRAL] ✓ Modèle chargé en {elapsed_time:.1f}s")
        
        log_gpu_memory()
        return voxtral_model, voxtral_processor
            
    except Exception as e:
        logger.error(f"[VOXTRAL] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_diarizer():
    """
    Charge le pipeline de diarisation PyAnnote
    Gère les deux APIs : 'token' (3.3+) et 'use_auth_token' (anciennes versions)
    """
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
        
        # Essayer d'abord avec 'token' (pyannote 3.3+)
        try:
            diarizer = Pipeline.from_pretrained(
                DIARIZATION_MODEL,
                token=hf_token
            )
            logger.info("[DIARIZER] ✓ Chargé avec 'token' (API moderne)")
        except TypeError as e:
            if "token" in str(e):
                # Fallback vers 'use_auth_token' (anciennes versions)
                logger.info("[DIARIZER] Fallback vers 'use_auth_token'...")
                diarizer = Pipeline.from_pretrained(
                    DIARIZATION_MODEL,
                    use_auth_token=hf_token
                )
                logger.info("[DIARIZER] ✓ Chargé avec 'use_auth_token' (API legacy)")
            else:
                raise
        
        diarizer.to(torch.device(DEVICE))
        logger.info("[DIARIZER] ✓ Diarizer chargé avec succès")
        return diarizer
        
    except Exception as e:
        logger.error(f"[DIARIZER] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Conversion au sample rate attendu (16kHz)
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
            inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Génération avec Voxtral
        logger.info("[VOXTRAL] Génération en cours...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        logger.info("[VOXTRAL] Décodage...")
        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        logger.info(f"[VOXTRAL] ✓ Transcription terminée ({len(transcription)} caractères)")
        if len(transcription) > 100:
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
        import traceback
        traceback.print_exc()
        return []


def extract_audio_segment(audio_path: str, start_time: float, end_time: float) -> str:
    """
    Extrait un segment audio entre start_time et end_time
    """
    try:
        audio, sample_rate = sf.read(audio_path)
        
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        segment = audio[start_idx:end_idx]
        
        temp_path = f"/tmp/segment_{int(start_time*1000)}_{int(end_time*1000)}.wav"
        sf.write(temp_path, segment, sample_rate)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"[EXTRACT] Erreur extraction segment: {e}")
        return ""


def analyze_sentiment(text: str) -> str:
    """
    Analyse basique du sentiment
    """
    if not text:
        return "neutre"
    
    positive_words = ["merci", "bien", "parfait", "excellent", "content", "satisfait", "ok", "super", "génial"]
    negative_words = ["problème", "erreur", "mal", "mauvais", "pas bien", "insatisfait", "déçu", "terrible"]
    
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
    
    sentences = transcription.split('.')
    if len(sentences) <= 3:
        return transcription
    
    summary_sentences = sentences[:2] + sentences[-1:]
    summary = '. '.join(s.strip() for s in summary_sentences if s.strip())
    
    return f"Résumé: {summary}"


def download_audio(url: str) -> str:
    """
    Télécharge un fichier audio depuis une URL
    """
    try:
        logger.info(f"[DOWNLOAD] Téléchargement: {url}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        
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
        task = job_input.get("task", "transcribe")
        audio_url = job_input.get("audio_url", "")
        language = job_input.get("language", "fr")
        max_tokens = job_input.get("max_tokens", job_input.get("max_new_tokens", 2500))
        include_summary = job_input.get("summary", job_input.get("with_summary", False))
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
                            segment_text = transcribe_with_voxtral(segment_path, 500)
                            
                            transcriptions.append({
                                "speaker": segment["speaker"],
                                "start": segment["start"],
                                "end": segment["end"],
                                "text": segment_text
                            })
                    
                    result["transcriptions"] = transcriptions
                    full_transcription = " ".join(t["text"] for t in transcriptions if t["text"])
                else:
                    logger.info("[HANDLER] Fallback: transcription simple")
                    full_transcription = transcribe_with_voxtral(audio_path, max_tokens)
                    result["transcriptions"] = [{"speaker": "UNKNOWN", "text": full_transcription}]
            else:
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
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
        
    finally:
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
        
        cleanup_gpu()


def initialize_models():
    """
    Initialise les modèles au démarrage
    """
    try:
        logger.info("[INIT] Pré-chargement des modèles...")
        
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            try:
                login(token=hf_token, add_to_git_credential=False)
                logger.info("[INIT] ✓ Authentification HuggingFace configurée")
            except Exception as e:
                logger.warning(f"[INIT] ⚠ Échec configuration auth HF: {e}")
        else:
            logger.warning("[INIT] ⚠ HF_TOKEN non défini - accès limité aux modèles")
        
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
        import traceback
        traceback.print_exc()


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
