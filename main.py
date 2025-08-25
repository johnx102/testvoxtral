#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, base64, tempfile, uuid, requests, json, traceback, re, signal
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline,
    pipeline as hf_pipeline
)

# Voxtral class (transformers @ main / nightly)
try:
    from transformers import VoxtralForConditionalGeneration  # type: ignore
    _HAS_VOXTRAL_CLASS = True
except Exception:
    VoxtralForConditionalGeneration = None  # type: ignore
    _HAS_VOXTRAL_CLASS = False

from pydub import AudioSegment
from pyannote.audio import Pipeline
import runpod

# ---------------------------
# Logging minimal
# ---------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
def log(msg: str):
    if LOG_LEVEL in ("DEBUG", "INFO"):
        print(msg, flush=True)

# ---------------------------
# Env / Config - ULTRA-STRICT OPTIMISÉ
# ---------------------------
APP_VERSION = os.environ.get("APP_VERSION", "ultra-strict-2025-08-25")

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Voxtral-Small-24B-2507").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))       # ULTRA-STRICT : 128→64
MAX_DURATION_S = int(os.environ.get("MAX_DURATION_S", "300"))      # OPTIMISÉ : 1200→300
DIAR_MODEL = os.environ.get("DIAR_MODEL", "pyannote/speaker-diarization-3.1").strip()
WITH_SUMMARY_DEFAULT = os.environ.get("WITH_SUMMARY_DEFAULT", "1") == "1"

# Sentiment (CPU par défaut)
SENTIMENT_MODEL = os.environ.get("SENTIMENT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli").strip()
SENTIMENT_TYPE = os.environ.get("SENTIMENT_TYPE", "zero-shot").strip().lower()  # "zero-shot" | "classifier"
ENABLE_SENTIMENT = os.environ.get("ENABLE_SENTIMENT", "1") == "1"
SENTIMENT_DEVICE = int(os.environ.get("SENTIMENT_DEVICE", "-1"))  # -1 = CPU

# Diarization speaker limit
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "2"))
EXACT_TWO = os.environ.get("EXACT_TWO", "1") == "1"   # force exactement 2 si possible
MIN_SEG_DUR = float(os.environ.get("MIN_SEG_DUR", "1.0"))         # ULTRA-STRICT : 0.8→1.0

# Transcription & résumé
STRICT_TRANSCRIPTION = os.environ.get("STRICT_TRANSCRIPTION", "1") == "1"
# Résumé par défaut: génératif intelligent
CONCISE_SUMMARY = os.environ.get("CONCISE_SUMMARY", "1") == "1"
EXTRACTIVE_SUMMARY = os.environ.get("EXTRACTIVE_SUMMARY", "0") == "1"

# Libellés des rôles (après mapping)
ROLE_LABELS = [r.strip() for r in os.environ.get("ROLE_LABELS", "Agent,Client").split(",") if r.strip()]

# Globals
_processor = None
_model = None
_diarizer = None
_sentiment_clf = None
_sentiment_zero_shot = None

# ---------------------------
# Helpers généraux
# ---------------------------
def _device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _download_to_tmp(url: str) -> str:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path

def _b64_to_tmp(b64: str) -> str:
    raw = base64.b64decode(b64)
    path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4().hex}.wav")
    with open(path, "wb") as f:
        f.write(raw)
    return path

def check_gpu_memory():
    """Vérifier la mémoire GPU disponible"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        allocated = torch.cuda.memory_allocated(0) / 1e9
        cached = torch.cuda.memory_reserved(0) / 1e9
        free = total - cached
        
        log(f"[GPU] Total: {total:.1f}GB | Allocated: {allocated:.1f}GB | Cached: {cached:.1f}GB | Free: {free:.1f}GB")
        
        if total < 45:  # Moins de 45GB = problème pour Small
            log(f"[ERROR] GPU has only {total:.1f}GB - Voxtral Small needs ~55GB minimum")
            return False
        elif free < 10:  # Moins de 10GB libres
            log(f"[WARN] Only {free:.1f}GB free - might cause OOM")
            
        return True
    return False

# ---------------------------
# Voxtral - OPTIMISÉ POUR SMALL
# ---------------------------
def load_voxtral():
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    log(f"[INIT] Loading Voxtral: {MODEL_ID}")

    proc_kwargs = {}
    if HF_TOKEN:
        proc_kwargs["token"] = HF_TOKEN
    _processor = AutoProcessor.from_pretrained(MODEL_ID, **proc_kwargs)

    # MÉMOIRE GPU OPTIMISÉE pour Small 24B
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    log(f"[INIT] Using dtype: {dtype}")
    
    mdl_kwargs = {
        "torch_dtype": dtype,          # CRUCIAL : force dtype optimal
        "device_map": "auto",
        "low_cpu_mem_usage": True,     # NOUVEAU : économise RAM CPU
        "max_memory": {0: "50GB"}      # NOUVEAU : limite explicite GPU 0
    }
    if HF_TOKEN:
        mdl_kwargs["token"] = HF_TOKEN

    if _HAS_VOXTRAL_CLASS:
        _model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, **mdl_kwargs)
    else:
        raise RuntimeError("Transformers sans VoxtralForConditionalGeneration. Utiliser transformers@main/nightly.")

    # PAS de .to() explicite - laisse device_map gérer
    log("[INIT] Skipping explicit .to() - letting device_map handle placement")

    try:
        p = next(_model.parameters())
        log(f"[INIT] Voxtral device={p.device}, dtype={p.dtype}")
        
        # Vérification que le modèle est majoritairement sur GPU
        gpu_params = sum(1 for p in _model.parameters() if p.device.type == 'cuda')
        total_params = sum(1 for p in _model.parameters())
        gpu_ratio = gpu_params / total_params if total_params > 0 else 0
        log(f"[INIT] GPU params ratio: {gpu_ratio:.2f} ({gpu_params}/{total_params})")
        
        if gpu_ratio < 0.8:
            log(f"[WARN] Only {gpu_ratio:.1%} of model on GPU - expect slow performance")
            
    except Exception as e:
        log(f"[WARN] Could not check model device: {e}")

    log("[INIT] Voxtral ready.")
    return _processor, _model

def _move_to_device_no_cast(batch, device: str):
    # Déplace uniquement sur device, sans modifier dtype
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
    return batch

def _input_len_from_batch(inputs) -> int:
    try:
        if isinstance(inputs, dict) and "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            return inputs["input_ids"].shape[1]
        elif hasattr(inputs, "input_ids") and isinstance(inputs.input_ids, torch.Tensor):
            return inputs.input_ids.shape[1]
    except Exception:
        pass
    return 0

def run_voxtral(conversation: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
    processor, model = load_voxtral()
    
    # Voxtral ne supporte pas add_generation_prompt - essai avec, puis sans
    try:
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True)
    except (TypeError, ValueError):  # Capture ValueError aussi !
        inputs = processor.apply_chat_template(conversation)

    device = _device_str()
    inputs = _move_to_device_no_cast(inputs, device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,     # déterministe pour la transcription
        temperature=None,
        top_p=None,
    )

    t0 = time.time()
    outputs = model.generate(**inputs, **gen_kwargs)
    dt = round(time.time() - t0, 3)

    inp_len = _input_len_from_batch(inputs)
    decoded = processor.batch_decode(outputs[:, inp_len:], skip_special_tokens=True)
    return {"text": decoded[0] if decoded else "", "latency_s": dt}

def run_voxtral_with_timeout(conversation: List[Dict[str, Any]], max_new_tokens: int, timeout: int = 45) -> Dict[str, Any]:
    """
    Wrapper avec timeout pour éviter les requêtes infinies
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Voxtral inference timed out after {timeout}s")
    
    # Setup timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = run_voxtral(conversation, max_new_tokens)
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError as e:
        log(f"[TIMEOUT] {e}")
        return {"text": "", "latency_s": timeout}
    except Exception as e:
        signal.alarm(0)
        log(f"[ERROR] Voxtral inference failed: {e}")
        return {"text": "", "latency_s": 0}
    finally:
        signal.signal(signal.SIGALRM, old_handler)

def _build_conv_transcribe(local_path: str, language: Optional[str]) -> List[Dict[str, Any]]:
    # Force le français si pas spécifié explicitement
    if not language:
        language = "fr"  # NOUVEAU : force français par défaut
    
    lang_prefix = f"lang:{language} " if language else "lang:fr "
    
    if STRICT_TRANSCRIPTION:
        constraints = (
            "TRANSCRIPTION TEXTUELLE EXACTE UNIQUEMENT. "
            "Reproduis chaque mot exactement comme prononcé. "
            "AUCUNE traduction, AUCUN ajout, AUCUNE correction. "
            "Si c'est du français, reste en français. "
            "Si inaudible: [inaudible]. "
            "Pas de ponctuation excessive. "
            "ZÉRO interprétation."
        )
        instruction = f"{lang_prefix}[TRANSCRIBE] {constraints}"
    else:
        instruction = f"{lang_prefix}[TRANSCRIBE] Mot à mot, langue originale uniquement."
    
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": local_path},
            {"type": "text", "text": instruction},
        ],
    }]

def _build_conv_transcribe_ultra_strict(local_path: str, language: Optional[str]) -> List[Dict[str, Any]]:
    """Version ultra-stricte pour éviter les hallucinations"""
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": local_path},
            {"type": "text", "text": "lang:fr [TRANSCRIBE] Écris exactement ce qui est dit, mot pour mot, en français uniquement."}
        ],
    }]

def clean_transcription_result(text: str, duration_s: float) -> str:
    """
    Nettoie le résultat de transcription pour éliminer les aberrations
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Filtre 1: Segments suspects (hallucinations communes de Voxtral)
    suspicious_phrases = [
        "so, you're making",
        "in fact, i have",
        "tell me the price",
        "thank you very much",
        "i think it's",
        "after that, it can",
        "and sometimes it even",
        "what is your age",
        "a telephone number"
    ]
    
    text_lower = text.lower()
    for phrase in suspicious_phrases:
        if phrase in text_lower and duration_s < 3.0:  # Phrases longues dans segments courts = suspect
            log(f"[FILTER] Suspicious phrase detected: '{text}' (duration: {duration_s:.1f}s)")
            return ""  # Vide plutôt qu'hallucination
    
    # Filtre 2: Mots anglais suspects dans conversation française
    english_words = ["for", "this", "week", "yes", "of", "course", "telephone", "number", "what", "age", "please", "right", "side"]
    words = text.lower().split()
    english_ratio = sum(1 for word in words if word in english_words) / max(len(words), 1)
    
    if english_ratio > 0.4 and duration_s < 4.0:  # Plus de 40% anglais dans segment court
        log(f"[FILTER] Too much English detected: '{text}' (ratio: {english_ratio:.1%})")
        return ""
    
    # Filtre 3: Segments trop longs pour la durée audio (hallucination)
    words_count = len(text.split())
    max_expected_words = int(duration_s * 4)  # ~4 mots/seconde max
    
    if words_count > max_expected_words * 1.5:  # 50% de marge
        log(f"[FILTER] Too many words for duration: {words_count} words in {duration_s:.1f}s")
        return " ".join(text.split()[:max_expected_words])  # Tronquer
    
    return text

def _build_conv_summary(local_path: str, language: Optional[str], max_sentences: Optional[int], style: Optional[str]) -> List[Dict[str, Any]]:
    parts = []
    if language:
        parts.append(f"lang:{language}")
    parts.append("Résumé factuel, concis, sans invention.")
    if max_sentences:
        parts.append(f"Résumé en {max_sentences} phrases.")
    if style:
        parts.append(style)
    instruction = " ".join(parts)
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "path": local_path},
            {"type": "text", "text": instruction},
        ],
    }]

# ---------------------------
# Diarization
# ---------------------------
def load_diarizer():
    global _diarizer
    if _diarizer is not None:
        return _diarizer
    log(f"[INIT] Loading diarization: {DIAR_MODEL}")
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN
    _diarizer = Pipeline.from_pretrained(DIAR_MODEL, **kwargs)
    try:
        if torch.cuda.is_available():
            try:
                _diarizer.to(torch.device("cuda"))
            except TypeError:
                _diarizer.to("cuda")
            log("[INIT] Diarizer moved to CUDA.")
        else:
            log("[INIT] CUDA not available; diarizer on CPU.")
    except Exception as e:
        log(f"[WARN] Could not move diarizer to CUDA: {e}. Keeping on CPU.")
    return _diarizer

# ---------------------------
# Sentiment
# ---------------------------
def load_sentiment():
    global _sentiment_clf, _sentiment_zero_shot
    if not ENABLE_SENTIMENT:
        return None

    device_idx = SENTIMENT_DEVICE  # -1 => CPU
    if SENTIMENT_TYPE == "zero-shot":
        if _sentiment_zero_shot is not None:
            return _sentiment_zero_shot
        log(f"[INIT] Loading zero-shot sentiment on device {device_idx}: {SENTIMENT_MODEL}")
        _sentiment_zero_shot = hf_pipeline(
            "zero-shot-classification",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            device=device_idx,
            model_kwargs={"use_safetensors": True}
        )
        log("[INIT] Zero-shot sentiment ready.")
        return _sentiment_zero_shot
    else:
        if _sentiment_clf is not None:
            return _sentiment_clf
        log(f"[INIT] Loading classifier sentiment on device {device_idx}: {SENTIMENT_MODEL}")
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, use_safetensors=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, use_safetensors=True)
        _sentiment_clf = TextClassificationPipeline(
            model=mdl, tokenizer=tok, device=device_idx, return_all_scores=True, truncation=True
        )
        log("[INIT] Classifier sentiment ready.")
        return _sentiment_clf

def _label_fr(en_label: str) -> str:
    m = {"negative": "mauvais", "neutral": "neutre", "positive": "bon"}
    key = en_label.lower()
    if key.startswith("label_"):
        try:
            idx = int(key.split("_")[-1])
            key = ["negative", "neutral", "positive"][idx]
        except Exception:
            pass
    return m.get(key, en_label)

def classify_sentiment(text: str) -> Dict[str, Any]:
    if not ENABLE_SENTIMENT or not text.strip():
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

    if SENTIMENT_TYPE == "zero-shot":
        clf = load_sentiment()
        res = clf(text.strip(), candidate_labels=["positive", "neutral", "negative"], multi_label=False)
        scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for lbl, sc in zip(res["labels"], res["scores"]):
            scores[lbl.lower()] = float(sc)
        label_en = res["labels"][0].lower()
        return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": scores[label_en], "scores": scores}
    else:
        clf = load_sentiment()
        res = clf(text.strip())[0]
        canonical = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
        for r in res:
            lbl = r["label"].lower()
            if lbl.startswith("label_"):
                try:
                    idx = int(lbl.split("_")[-1])
                    lbl = ["negative", "neutral", "positive"][idx]
                except Exception:
                    pass
            if lbl in canonical:
                canonical[lbl] = float(r["score"])
        label_en = max(canonical.items(), key=lambda kv: kv[1])[0]
        return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": canonical[label_en], "scores": canonical}

def aggregate_mood(weighted: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    total = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    total_w = 0.0
    for s, w in weighted:
        if not s or not s.get("scores"):
            continue
        for k in total.keys():
            total[k] += s["scores"].get(k, 0.0) * w
        total_w += w
    if total_w <= 0:
        return {"label_en": None, "label_fr": None, "confidence": None, "scores": None}
    ssum = sum(total.values())
    norm = {k: (v/ssum if ssum > 0 else 0.0) for k, v in total.items()}
    label_en = max(norm.items(), key=lambda kv: kv[1])[0]
    return {"label_en": label_en, "label_fr": _label_fr(label_en), "confidence": norm[label_en], "scores": norm}

# ---------------------------
# Résumés à partir des segments
# ---------------------------
def _is_filler(text: str) -> bool:
    t = (text or "").lower().strip()
    fillers = [
        "bonjour", "bonsoir", "au revoir", "bonne soirée", "bon week-end",
        "merci", "merci beaucoup", "je vous en prie"
    ]
    return any(t == f or t.startswith(f + ".") for f in fillers)

def extractive_summary_from_segments(segments, max_bullets=6) -> str:
    bullets = []
    # demande client
    for s in segments:
        if s["speaker"].lower().startswith("client") and not _is_filler(s["text"]):
            if any(k in s["text"].lower() for k in ["rendez-vous", "rdv", "je veux", "je voudrais", "savoir", "prise", "demande", "est-ce que"]):
                bullets.append(f"**Demande du client** : « {s['text']} »")
                break
    # proposition agent
    for s in segments:
        if s["speaker"].lower().startswith("agent") and not _is_filler(s["text"]):
            if any(k in s["text"].lower() for k in ["je vais", "je transmets", "je m'en occupe", "je peux vous proposer", "on peut"]):
                bullets.append(f"**Réponse de l'agent** : « {s['text']} »")
                break
    # confirmation client
    for s in segments:
        if s["speaker"].lower().startswith("client") and not _is_filler(s["text"]):
            if any(k in s["text"].lower() for k in ["oui", "d'accord", "ok", "parfait", "très bien"]):
                bullets.append(f"**Confirmation** : « {s['text']} »")
                break
    # remerciements
    if any("merci" in (s["text"] or "").lower() for s in segments):
        bullets.append("**Remerciements échangés.**")
    # clôture
    if any("au revoir" in (s["text"] or "").lower() for s in segments):
        bullets.append("**Fin d'appel.**")

    if not bullets:
        for s in segments:
            if not _is_filler(s["text"]):
                bullets.append(f"{s['speaker']} : « {s['text']} »")
            if len(bullets) >= max_bullets:
                break

    return "- " + "\n- ".join(bullets[:max_bullets]) if bullets else "Résumé indisponible."

def concise_summary_from_segments(segments, max_lines=6) -> str:
    """
    Résumé court, fidèle, à partir du transcript uniquement.
    On extrait: motif/demande, réponse/action, rendez-vous/dates/heures, infos clés.
    """
    text_all = " ".join([s.get("text","") for s in segments if s.get("text")]).strip()
    if not text_all:
        return "Résumé indisponible."

    def find_first(patterns, speaker_prefix=None):
        for s in segments:
            if speaker_prefix and not s["speaker"].lower().startswith(speaker_prefix):
                continue
            t = (s.get("text") or "").lower()
            if any(p in t for p in patterns):
                return s["text"]
        return None

    demande = find_first(["rendez-vous", "rdv", "prise de", "je veux", "je voudrais", "est-ce que", "savoir", "disponibil"], "client")
    motif   = find_first(["pour", "motif", "c'est pour", "afin de"], "client")
    action  = find_first(["je vais", "je transmets", "on peut", "je peux vous", "je propose"], "agent")
    horaire = find_first([r"^(\d{1,2} ?h ?\d{0,2})", "à ", "le "], None)  # repère grossier

    lines = []
    if demande: lines.append(f"Demande: {demande}")
    if motif and motif != demande: lines.append(f"Motif: {motif}")
    if action: lines.append(f"Action de l'agent: {action}")

    # Tentative d'extraction simple de date/heure/adresse/numéro
    candidates = []
    for s in segments:
        t = s.get("text","")
        if not t: continue
        if re.search(r"\b\d{1,2}\s?h(?:\s?\d{2})?\b", t.lower()):
            candidates.append(("Horaire", t))
        if re.search(r"\b(\d{1,2}/\d{1,2}|\d{1,2}\s\w+)\b", t.lower()):
            candidates.append(("Date", t))
        if re.search(r"\b(0[1-9]|[1-9]\d)\s?\d{2}\s?\d{2}\s?\d{2}\b", t.replace("-", " ")):  # FR (très) approximatif
            candidates.append(("Téléphone", t))
        if any(k in t.lower() for k in ["rue", "boulevard", "avenue", "bd ", "av "]):
            candidates.append(("Adresse", t))

    # déduplique en gardant les premières occurrences parlantes
    seen = set()
    for k, v in candidates:
        if k not in seen and len(lines) < max_lines:
            lines.append(f"{k}: {v}")
            seen.add(k)

    # si rien trouvé, on prend les 2-3 interventions les plus longues (non-filler)
    if len(lines) < 2:
        chunks = [s["text"] for s in sorted(segments, key=lambda x: len(x.get("text","")), reverse=True)
                  if s.get("text") and not _is_filler(s["text"])]
        for c in chunks[:max_lines-len(lines)]:
            lines.append(c)

    return "\n".join(lines[:max_lines]) if lines else "Résumé indisponible."

# ---------------------------
# NOUVEAUX RÉSUMÉS GÉNÉRATIFS INTELLIGENTS
# ---------------------------

def generate_smart_summary_from_transcript(full_transcript: str, language: Optional[str] = None) -> str:
    """
    Utilise Voxtral pour générer un résumé intelligent à partir du transcript complet.
    Plus efficace et cohérent que l'extraction basique.
    """
    if not full_transcript.strip():
        return "Résumé indisponible."
    
    # Instructions très précises pour un résumé professionnel
    lang_prefix = f"lang:{language} " if language else ""
    instruction = (
        f"{lang_prefix}Résume cette conversation téléphonique en 2-3 phrases courtes et professionnelles. "
        "Format attendu: 'Le client [demande/action]. L'agent [réponse]. [Résultat final].' "
        "Sois factuel, concis, sans détails superflus. Focus sur l'essentiel business."
    )
    
    # Conversation pour Voxtral avec le transcript en mode texte
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{instruction}\n\nConversation:\n{full_transcript}"}
        ]
    }]
    
    try:
        result = run_voxtral_with_timeout(conversation, max_new_tokens=96, timeout=30)  # Court pour résumé concis
        summary = (result.get("text") or "").strip()
        
        # Validation basique de qualité
        if len(summary) < 15 or "résumé" in summary.lower() or summary.startswith("Je"):
            log("[WARN] Generated summary seems poor, falling back to extractive")
            return fallback_extractive_summary(full_transcript)
        
        return summary
        
    except Exception as e:
        log(f"[WARN] Generative summary failed: {e}")
        return fallback_extractive_summary(full_transcript)

def fallback_extractive_summary(transcript: str) -> str:
    """
    Résumé extractif simple et robuste en cas d'échec du génératif.
    Plus intelligent que la version précédente.
    """
    lines = [l.strip() for l in transcript.split('\n') if l.strip() and len(l.strip()) > 5]
    
    if not lines:
        return "Conversation très courte."
    
    # Recherche patterns métier améliorés
    client_demand = None
    agent_response = None
    
    # Mots-clés pour identifier les demandes importantes
    demand_keywords = ["voulais", "voudrais", "besoin", "demande", "rdv", "rendez-vous", 
                      "disponib", "possible", "consultation", "docteur", "savoir", "question"]
    
    # Mots-clés pour identifier les réponses importantes  
    response_keywords = ["non", "oui", "pas", "disponib", "possible", "je vais", "désolé", 
                        "malheureusement", "aujourd'hui", "demain"]
    
    for line in lines:
        if line.startswith("Client:"):
            client_text = line.replace("Client:", "").strip()
            # Score la ligne selon pertinence business
            score = len(client_text) + sum(5 for kw in demand_keywords if kw in client_text.lower())
            if score > 20 and not client_demand:  # Seuil de pertinence
                client_demand = client_text
                
        elif line.startswith("Agent:"):
            agent_text = line.replace("Agent:", "").strip()
            # Éviter les simples politesses
            if len(agent_text) > 8 and not agent_text.lower().strip() in ["bonjour", "bonsoir", "au revoir"]:
                score = len(agent_text) + sum(3 for kw in response_keywords if kw in agent_text.lower())
                if score > 15 and not agent_response:
                    agent_response = agent_text
    
    # Construction résumé structuré
    parts = []
    if client_demand:
        parts.append(f"Demande: {client_demand}")
    if agent_response:
        parts.append(f"Réponse: {agent_response}")
    
    # Fallback si patterns échouent
    if not parts:
        # Prendre les 2 segments les plus substantiels
        substantial_lines = [l for l in lines if len(l) > 20]
        parts = substantial_lines[:2] if substantial_lines else lines[:2]
    
    return " | ".join(parts) if parts else "Conversation courte."

# ---------------------------
# Post-traitements segments
# ---------------------------
def _enforce_max_two_speakers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    speakers = list({s["speaker"] for s in segments})
    if len(speakers) <= 2:
        return segments

    dur = {}
    cent = {}
    for s in segments:
        d = float(s["end"]) - float(s["start"])
        spk = s["speaker"]
        mid = (float(s["start"]) + float(s["end"])) / 2.0
        dur[spk] = dur.get(spk, 0.0) + d
        sm, ds = cent.get(spk, (0.0, 0.0))
        cent[spk] = (sm + mid * d, ds + d)

    top2 = sorted(dur.keys(), key=lambda k: dur[k], reverse=True)[:2]
    centroids = {}
    for spk in top2:
        sm, ds = cent[spk]
        centroids[spk] = (sm / ds) if ds > 0 else 0.0

    def nearest(mid):
        best, bestd = None, 1e18
        for spk, c in centroids.items():
            dd = abs(mid - c)
            if dd < bestd:
                best, bestd = spk, dd
        return best

    for s in segments:
        if s["speaker"] not in top2:
            mid = (float(s["start"]) + float(s["end"])) / 2.0
            s["speaker"] = nearest(mid)

    return segments

def _map_roles(segments: List[Dict[str, Any]]):
    seen = []
    for s in segments:
        spk = s["speaker"]
        if spk not in seen:
            seen.append(spk)
    mapping = {}
    for i, spk in enumerate(seen):
        if i < len(ROLE_LABELS):
            mapping[spk] = ROLE_LABELS[i]
    for s in segments:
        s["speaker"] = mapping.get(s["speaker"], s["speaker"])

# ---------------------------
# Core flow - ULTRA-STRICT
# ---------------------------
def diarize_then_transcribe(wav_path: str, language: Optional[str], max_new_tokens: int, with_summary: bool):
    # Durée max
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        est_dur = info.frames / float(info.samplerate or 1)
        if est_dur > MAX_DURATION_S:
            return {"error": f"Audio too long ({est_dur:.1f}s). Increase MAX_DURATION_S or send shorter file."}
    except Exception:
        pass

    dia = load_diarizer()
    try:
        if EXACT_TWO:
            diarization = dia(wav_path, num_speakers=min(2, MAX_SPEAKERS))
        else:
            diarization = dia(wav_path, min_speakers=1, max_speakers=min(2, MAX_SPEAKERS))
    except TypeError:
        if EXACT_TWO:
            diarization = dia(wav_path, min_speakers=min(2, MAX_SPEAKERS), max_speakers=min(2, MAX_SPEAKERS))
        else:
            diarization = dia(wav_path, num_speakers=min(2, MAX_SPEAKERS))

    audio = AudioSegment.from_wav(wav_path)

    segments = []
    transcript_parts = []
    weighted_moods = []
    mood_by_speaker_weights = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_s = float(turn.start)
        end_s = float(turn.end)
        dur_s = max(0.0, end_s - start_s)
        
        # ULTRA-STRICT : Ignore segments très courts
        if dur_s < MIN_SEG_DUR:
            continue
        
        # VERSION ULTRA-STRICTE pour éviter hallucinations
        if dur_s < 2.0:  # Segments courts
            max_tokens_segment = 24   # RÉDUIT : 32→24
            timeout_segment = 15
        else:
            max_tokens_segment = 48   # RÉDUIT : 64→48  
            timeout_segment = 25      # RÉDUIT : 30→25

        seg_audio = audio[int(start_s * 1000): int(end_s * 1000)]
        tmp = os.path.join(tempfile.gettempdir(), f"seg_{speaker}_{int(start_s*1000)}.wav")
        seg_audio.export(tmp, format="wav")

        # Essai 1 : Version ultra-stricte
        conv = _build_conv_transcribe_ultra_strict(tmp, "fr")
        out = run_voxtral_with_timeout(conv, max_new_tokens=max_tokens_segment, timeout=timeout_segment)
        text = (out.get("text") or "").strip()

        # Fallback seulement si vide ET segment > 1.5s
        if not text and dur_s >= 1.5:
            log(f"[FALLBACK] Empty transcription for {dur_s:.1f}s segment")
            conv2 = [{
                "role": "user",
                "content": [
                    {"type": "audio", "path": tmp},
                    {"type": "text", "text": "lang:fr [TRANSCRIBE]"}  # Minimal prompt
                ],
            }]
            out2 = run_voxtral_with_timeout(conv2, max_new_tokens=max_tokens_segment, timeout=timeout_segment)
            text = (out2.get("text") or "").strip()

        # NOUVEAU : Filtre post-transcription pour éliminer aberrations
        text = clean_transcription_result(text, dur_s)
        
        # Logging performance
        if out.get("latency_s", 0) > 20:
            log(f"[PERF] Slow transcription: {dur_s:.1f}s audio took {out.get('latency_s', 0):.1f}s")

        mood = classify_sentiment(text) if (ENABLE_SENTIMENT and text) else {"label_en": None, "label_fr": None, "confidence": None, "scores": None}

        seg = {"speaker": speaker, "start": start_s, "end": end_s, "text": text, "mood": mood}
        segments.append(seg)

        if text:
            transcript_parts.append(f"{speaker}: {text}")
            if ENABLE_SENTIMENT and dur_s > 0:
                weighted_moods.append((mood, dur_s))
                mood_by_speaker_weights.setdefault(speaker, []).append((mood, dur_s))

        try:
            os.remove(tmp)
        except Exception:
            pass

    # Safety net 2 speakers + mapping rôles
    segments = _enforce_max_two_speakers(segments)
    _map_roles(segments)

    # Transcript final
    full_transcript = "\n".join(f"{s['speaker']}: {s['text']}" for s in segments if s.get("text"))

    result = {"segments": segments, "transcript": full_transcript}

    # RÉSUMÉ GÉNÉRATIF AMÉLIORÉ
    if with_summary:
        if CONCISE_SUMMARY:
            # NOUVEAU : Résumé génératif intelligent avec Voxtral
            result["summary"] = generate_smart_summary_from_transcript(full_transcript, language)
        elif EXTRACTIVE_SUMMARY:
            # Ancien résumé extractif (conservé en option)
            result["summary"] = extractive_summary_from_segments(segments) if any(s.get("text") for s in segments) else "Résumé indisponible."
        else:
            # Mode résumé audio direct (votre version originale)
            conv_sum = _build_conv_summary(wav_path, language, max_sentences=3, style="Résumé professionnel concis en français.")
            s = run_voxtral_with_timeout(conv_sum, max_new_tokens=96, timeout=30)  # Réduit pour résumé concis
            result["summary"] = s.get("text", "") or "Résumé indisponible."

    # Humeurs
    if ENABLE_SENTIMENT:
        result["mood_overall"] = aggregate_mood(weighted_moods)
        per_role = {}
        for seg in segments:
            d = float(seg["end"]) - float(seg["start"])
            m = seg.get("mood")
            r = seg["speaker"]
            if not m or not m.get("scores"):
                continue
            per_role.setdefault(r, []).append((m, d))
        result["mood_by_speaker"] = {r: aggregate_mood(lst) for r, lst in per_role.items()}

    return result

# ---------------------------
# Health
# ---------------------------
def health():
    info = {
        "app_version": APP_VERSION,
        "python": os.popen("python -V").read().strip(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": _device_str(),
        "transformers_has_voxtral": _HAS_VOXTRAL_CLASS,
        "hf_token_present": bool(HF_TOKEN),
        "model_id": MODEL_ID,
        "diar_model": DIAR_MODEL,
        "sentiment_model": SENTIMENT_MODEL if ENABLE_SENTIMENT else None,
        "sentiment_type": SENTIMENT_TYPE,
        "sentiment_device": "cpu" if SENTIMENT_DEVICE == -1 else f"cuda:{SENTIMENT_DEVICE}",
        "max_speakers": MAX_SPEAKERS,
        "exact_two": EXACT_TWO,
        "strict_transcription": STRICT_TRANSCRIPTION,
        "concise_summary": CONCISE_SUMMARY,
        "extractive_summary": EXTRACTIVE_SUMMARY,
        "min_seg_dur": MIN_SEG_DUR,
        "role_labels": ROLE_LABELS,
    }
    try:
        if _model is not None:
            p = next(_model.parameters())
            info["voxtral_device"] = str(p.device)
            info["voxtral_dtype"]  = str(p.dtype)
        else:
            info["voxtral_device"] = None
            info["voxtral_dtype"]  = None
    except Exception:
        info["voxtral_device"] = "unknown"
        info["voxtral_dtype"]  = "unknown"

    errors = {}
    try:
        load_voxtral()
    except Exception as e:
        errors["voxtral_load"] = f"{type(e).__name__}: {e}"
    try:
        load_diarizer()
    except Exception as e:
        errors["diarizer_load"] = f"{type(e).__name__}: {e}"
    if ENABLE_SENTIMENT:
        try:
            load_sentiment()
        except Exception as e:
            errors["sentiment_load"] = f"{type(e).__name__}: {e}"
    return {"ok": len(errors) == 0, "info": info, "errors": errors}

# ---------------------------
# Handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input", {}) or {}

        if inp.get("ping"):
            return {"pong": True}

        if inp.get("task") == "health":
            return health()

        task = (inp.get("task") or "transcribe").lower()
        language = inp.get("language") or None
        max_new_tokens = int(inp.get("max_new_tokens", MAX_NEW_TOKENS))
        with_summary = bool(inp.get("with_summary", WITH_SUMMARY_DEFAULT))

        local_path, cleanup = None, False
        if inp.get("audio_url"):
            local_path = _download_to_tmp(inp["audio_url"]); cleanup = True
        elif inp.get("audio_b64"):
            local_path = _b64_to_tmp(inp["audio_b64"]); cleanup = True
        elif inp.get("file_path"):
            local_path = inp["file_path"]
        elif task not in ("health",):
            return {"error": "Provide 'audio_url', 'audio_b64' or 'file_path'."}

        if task in ("transcribe_diarized", "diarized", "diarize"):
            out = diarize_then_transcribe(local_path, language, max_new_tokens, with_summary)
            if "error" in out:
                return out
            return {"task": "transcribe_diarized", **out}
        elif task in ("summary", "summarize"):
            conv = _build_conv_summary(local_path, language, inp.get("max_sentences"), inp.get("style"))
            out = run_voxtral_with_timeout(conv, max_new_tokens=96, timeout=30)
            return {"task": "summary", **out}
        else:
            conv = _build_conv_transcribe(local_path, language)
            out = run_voxtral_with_timeout(conv, max_new_tokens=min(max_new_tokens, 64), timeout=30)
            return {"task": "transcribe", **out}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)}
    finally:
        try:
            if 'cleanup' in locals() and cleanup and local_path and os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

# Preload (chaud) avec vérification mémoire
try:
    if not check_gpu_memory():
        log("[CRITICAL] Insufficient GPU memory for Voxtral Small - consider using Mini")
    load_voxtral()
    load_diarizer()
    if ENABLE_SENTIMENT:
        load_sentiment()
except Exception as e:
    log(f"[WARN] Deferred load: {e}")

runpod.serverless.start({"handler": handler})
