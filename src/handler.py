#!/usr/bin/env python3
"""
WhisperX Serverless Worker - Transcription + Diarisation + Résumé + Sentiment
Basé sur WhisperX (https://github.com/m-bain/whisperX)
Compatible RunPod Serverless
"""

import os
import sys
import gc
import time
import json
import logging
import tempfile
import warnings
from typing import Optional, Dict, Any, List

import torch
import requests
import runpod

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
MAX_DURATION = int(os.getenv("MAX_DURATION_S", "9000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# Global model cache
whisperx_model = None
align_model = None
align_metadata = None
diarize_model = None
summarizer = None


def log_gpu():
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"[GPU] {torch.cuda.get_device_name(0)} - {allocated:.1f}GB / {total:.1f}GB")


def cleanup():
    """Clean GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_whisperx():
    """Load WhisperX model"""
    global whisperx_model
    
    if whisperx_model is not None:
        return whisperx_model
    
    try:
        import whisperx
        
        logger.info(f"[WHISPERX] Loading model: {WHISPER_MODEL}")
        logger.info(f"[WHISPERX] Device: {DEVICE}, Compute: {COMPUTE_TYPE}")
        
        start = time.time()
        whisperx_model = whisperx.load_model(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None  # Auto-detect
        )
        
        logger.info(f"[WHISPERX] ✓ Loaded in {time.time() - start:.1f}s")
        log_gpu()
        
        return whisperx_model
        
    except Exception as e:
        logger.error(f"[WHISPERX] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_align_model(language: str):
    """Load alignment model for word-level timestamps"""
    global align_model, align_metadata
    
    try:
        import whisperx
        
        logger.info(f"[ALIGN] Loading model for: {language}")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=DEVICE
        )
        logger.info("[ALIGN] ✓ Loaded")
        return align_model, align_metadata
        
    except Exception as e:
        logger.warning(f"[ALIGN] ⚠ Not available for {language}: {e}")
        return None, None


def load_diarize_model(hf_token: str):
    """Load diarization model"""
    global diarize_model
    
    if diarize_model is not None:
        return diarize_model
    
    if not hf_token:
        logger.warning("[DIARIZE] ⚠ No HF_TOKEN - diarization disabled")
        return None
    
    try:
        import whisperx
        
        logger.info("[DIARIZE] Loading pyannote pipeline...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=DEVICE
        )
        logger.info("[DIARIZE] ✓ Loaded")
        return diarize_model
        
    except Exception as e:
        logger.error(f"[DIARIZE] ✗ Error: {e}")
        return None


def load_summarizer():
    """Load summarization model (lightweight)"""
    global summarizer
    
    if summarizer is not None:
        return summarizer
    
    try:
        from transformers import pipeline
        
        logger.info("[SUMMARY] Loading summarization model...")
        
        # Use a lightweight multilingual model
        summarizer = pipeline(
            "summarization",
            model="facebook/mbart-large-50",
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        
        logger.info("[SUMMARY] ✓ Loaded")
        return summarizer
        
    except Exception as e:
        logger.warning(f"[SUMMARY] ⚠ Model not loaded: {e}")
        return None


def download_audio(url: str) -> Optional[str]:
    """Download audio file from URL"""
    try:
        logger.info(f"[DOWNLOAD] {url}")
        
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        # Determine extension
        ext = ".wav"
        if ".mp3" in url.lower():
            ext = ".mp3"
        elif ".m4a" in url.lower():
            ext = ".m4a"
        elif ".ogg" in url.lower():
            ext = ".ogg"
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            path = f.name
        
        logger.info(f"[DOWNLOAD] ✓ {path}")
        return path
        
    except Exception as e:
        logger.error(f"[DOWNLOAD] ✗ {e}")
        return None


def transcribe(audio_path: str, language: str = None, batch_size: int = None) -> Dict:
    """Transcribe audio with WhisperX"""
    import whisperx
    
    model = load_whisperx()
    if model is None:
        raise Exception("Failed to load WhisperX")
    
    bs = batch_size or BATCH_SIZE
    
    logger.info(f"[TRANSCRIBE] Starting (batch_size={bs})...")
    start = time.time()
    
    # Load audio
    audio = whisperx.load_audio(audio_path)
    
    # Transcribe
    result = model.transcribe(
        audio,
        batch_size=bs,
        language=language
    )
    
    elapsed = time.time() - start
    detected_lang = result.get("language", language)
    
    logger.info(f"[TRANSCRIBE] ✓ Done in {elapsed:.1f}s, language: {detected_lang}")
    
    return result, audio, detected_lang


def align_transcript(result: Dict, audio, language: str) -> Dict:
    """Align transcript for word-level timestamps"""
    import whisperx
    
    global align_model, align_metadata
    
    if align_model is None:
        align_model, align_metadata = load_align_model(language)
    
    if align_model is None:
        logger.warning("[ALIGN] ⚠ Skipping alignment")
        return result
    
    logger.info("[ALIGN] Aligning...")
    start = time.time()
    
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )
    
    logger.info(f"[ALIGN] ✓ Done in {time.time() - start:.1f}s")
    return result


def diarize(audio_path: str, result: Dict, hf_token: str, 
            min_speakers: int = None, max_speakers: int = None) -> Dict:
    """Add speaker diarization"""
    import whisperx
    
    diarize_pipeline = load_diarize_model(hf_token)
    
    if diarize_pipeline is None:
        return result
    
    logger.info("[DIARIZE] Running...")
    start = time.time()
    
    # Run diarization
    diarize_segments = diarize_pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )
    
    # Assign speakers to segments
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    logger.info(f"[DIARIZE] ✓ Done in {time.time() - start:.1f}s")
    return result


def generate_summary(text: str, language: str = "fr") -> str:
    """Generate summary using simple extraction"""
    if not text or len(text) < 200:
        return text
    
    try:
        # Simple extractive summary
        sentences = []
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            text = text.replace(sep, '.|.')
        
        parts = [s.strip() for s in text.split('.|.') if s.strip() and len(s.strip()) > 10]
        
        if len(parts) <= 5:
            return " ".join(parts)
        
        # Take first 2, middle 1, last 2 sentences
        summary_parts = parts[:2]
        if len(parts) > 4:
            summary_parts.append(parts[len(parts)//2])
        summary_parts.extend(parts[-2:])
        
        summary = ". ".join(summary_parts)
        if not summary.endswith('.'):
            summary += '.'
        
        return f"Résumé: {summary}"
        
    except Exception as e:
        logger.error(f"[SUMMARY] Error: {e}")
        return ""


def analyze_sentiment(text: str, language: str = "fr") -> Dict:
    """Analyze sentiment"""
    if not text:
        return {"sentiment": "neutre", "score": 0.5}
    
    text_lower = text.lower()
    
    # French sentiment words
    positive = [
        "merci", "parfait", "excellent", "super", "génial", "bien", "bon", "bonne",
        "content", "satisfait", "formidable", "bravo", "accord", "oui", "agréable",
        "heureux", "plaisir", "réussi", "succès", "efficace", "rapide"
    ]
    negative = [
        "problème", "erreur", "mal", "mauvais", "insatisfait", "déçu", "terrible",
        "horrible", "nul", "pire", "difficile", "compliqué", "impossible", "échec",
        "plainte", "mécontent", "colère", "énervé", "attendre", "long", "incompétent"
    ]
    
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)
    
    total = pos_count + neg_count
    
    if total == 0:
        sentiment, score = "neutre", 0.5
    elif pos_count > neg_count:
        sentiment = "positif"
        score = 0.5 + min(0.5, pos_count / (total * 2))
    elif neg_count > pos_count:
        sentiment = "négatif"
        score = 0.5 - min(0.5, neg_count / (total * 2))
    else:
        sentiment, score = "neutre", 0.5
    
    if score > 0.6:
        satisfaction = "satisfait"
    elif score < 0.4:
        satisfaction = "insatisfait"
    else:
        satisfaction = "neutre"
    
    return {
        "sentiment": sentiment,
        "score": round(score, 2),
        "satisfaction_client": satisfaction,
        "details": {"positif": pos_count, "negatif": neg_count}
    }


def format_transcriptions(segments: List[Dict]) -> List[Dict]:
    """Format segments for output"""
    result = []
    
    for seg in segments:
        formatted = {
            "start": round(seg.get("start", 0), 2),
            "end": round(seg.get("end", 0), 2),
            "text": seg.get("text", "").strip()
        }
        
        if "speaker" in seg:
            formatted["speaker"] = seg["speaker"]
        
        if "words" in seg:
            formatted["words"] = [
                {
                    "word": w.get("word", ""),
                    "start": round(w.get("start", 0), 2),
                    "end": round(w.get("end", 0), 2),
                    **({"speaker": w["speaker"]} if "speaker" in w else {})
                }
                for w in seg["words"]
            ]
        
        result.append(formatted)
    
    return result


def process_request(job_input: Dict) -> Dict:
    """Process a transcription request"""
    temp_files = []
    
    try:
        # Extract parameters - support both old and new API
        audio_url = job_input.get("audio_url") or job_input.get("audio_file")
        language = job_input.get("language")
        batch_size = job_input.get("batch_size", BATCH_SIZE)
        
        # Task detection
        task = job_input.get("task", "transcribe")
        do_align = job_input.get("align_output", True)
        do_diarize = job_input.get("diarization", task == "transcribe_diarized")
        do_summary = job_input.get("summary", job_input.get("with_summary", False))
        do_sentiment = job_input.get("sentiment", False)
        
        # Diarization params
        hf_token = job_input.get("huggingface_access_token") or os.getenv("HF_TOKEN")
        min_speakers = job_input.get("min_speakers")
        max_speakers = job_input.get("max_speakers")
        
        logger.info(f"[HANDLER] Task: {task}")
        logger.info(f"[HANDLER] Options: align={do_align}, diarize={do_diarize}, summary={do_summary}, sentiment={do_sentiment}")
        
        if not audio_url:
            return {"error": "Missing audio_url or audio_file"}
        
        # Download audio
        audio_path = download_audio(audio_url)
        if not audio_path:
            return {"error": "Failed to download audio"}
        temp_files.append(audio_path)
        
        # Transcribe
        result, audio, detected_lang = transcribe(audio_path, language, batch_size)
        
        # Align if requested
        if do_align and result.get("segments"):
            result = align_transcript(result, audio, detected_lang)
        
        # Diarize if requested
        if do_diarize:
            result = diarize(audio_path, result, hf_token, min_speakers, max_speakers)
        
        # Format output
        segments = result.get("segments", [])
        formatted_segments = format_transcriptions(segments)
        
        # Build full text
        full_text = " ".join(s.get("text", "") for s in segments).strip()
        
        # Prepare response
        response = {
            "segments" if do_diarize else "transcription": formatted_segments if do_diarize else full_text,
            "detected_language": detected_lang
        }
        
        if do_diarize:
            response["transcriptions"] = formatted_segments  # Also add in old format
        
        # Add summary
        if do_summary:
            response["summary"] = generate_summary(full_text, detected_lang or "fr")
        
        # Add sentiment
        if do_sentiment:
            response["sentiment"] = analyze_sentiment(full_text, detected_lang or "fr")
        
        logger.info("[HANDLER] ✓ Processing complete")
        return response
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
        
    finally:
        # Cleanup temp files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass
        cleanup()


def handler(job):
    """RunPod handler"""
    try:
        job_id = job.get("id", "unknown")
        logger.info(f"[HANDLER] === Job: {job_id} ===")
        
        job_input = job.get("input", {})
        if not job_input:
            return {"error": "No input provided"}
        
        result = process_request(job_input)
        
        logger.info(f"[HANDLER] ✓ Job complete: {len(json.dumps(result, ensure_ascii=False))} chars")
        return result
        
    except Exception as e:
        logger.error(f"[HANDLER] ✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def initialize():
    """Initialize models"""
    logger.info("=== WhisperX Serverless Worker ===")
    logger.info(f"Model: {WHISPER_MODEL}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Compute: {COMPUTE_TYPE}")
    
    log_gpu()
    
    # Pre-load WhisperX
    logger.info("[INIT] Loading WhisperX...")
    if load_whisperx():
        logger.info("[INIT] ✓ WhisperX ready")
    else:
        logger.error("[INIT] ✗ WhisperX failed")
    
    log_gpu()
    logger.info("[INIT] ✓ Initialization complete")


if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})
