#!/usr/bin/env python3
"""
Script de diagnostic pour les problèmes de téléchargement de modèles
"""

import os
import time
import psutil
import requests
from huggingface_hub import HfApi, scan_cache_dir, list_repo_files
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

VOXTRAL_MODEL = "mistralai/Voxtral-Small-24B-2507"

def check_system_resources():
    """Vérification des ressources système"""
    logger.info("=== Diagnostic Système ===")
    
    # RAM
    ram = psutil.virtual_memory()
    logger.info(f"RAM Total: {ram.total / (1024**3):.1f} GB")
    logger.info(f"RAM Disponible: {ram.available / (1024**3):.1f} GB")
    logger.info(f"RAM Utilisée: {ram.percent:.1f}%")
    
    # Disque
    disk = psutil.disk_usage('/')
    logger.info(f"Disque Total: {disk.total / (1024**3):.1f} GB")
    logger.info(f"Disque Libre: {disk.free / (1024**3):.1f} GB")
    logger.info(f"Disque Utilisé: {(disk.total - disk.free) / disk.total * 100:.1f}%")
    
    # GPU (si disponible)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU Disponibles: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {props.name} - {memory_gb:.1f} GB")
        else:
            logger.warning("Aucun GPU CUDA disponible")
    except ImportError:
        logger.warning("PyTorch non disponible")

def check_network_connectivity():
    """Test de connectivité réseau"""
    logger.info("=== Test de Connectivité ===")
    
    urls_to_test = [
        "https://huggingface.co",
        "https://huggingface.co/api/models/mistralai/Voxtral-Small-24B-2507",
        "https://pypi.org"
    ]
    
    for url in urls_to_test:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=10)
            elapsed = time.time() - start_time
            logger.info(f"✅ {url}: {response.status_code} ({elapsed:.2f}s)")
        except Exception as e:
            logger.error(f"❌ {url}: {e}")

def check_hf_authentication():
    """Vérification de l'authentification HuggingFace"""
    logger.info("=== Authentification HuggingFace ===")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("⚠️ HF_TOKEN non défini")
        return False
    
    try:
        api = HfApi(token=hf_token)
        user = api.whoami()
        logger.info(f"✅ Connecté en tant que: {user.get('name', 'N/A')}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur d'authentification: {e}")
        return False

def check_model_accessibility():
    """Vérification de l'accessibilité du modèle"""
    logger.info("=== Accessibilité du Modèle ===")
    
    try:
        hf_token = os.getenv("HF_TOKEN")
        api = HfApi(token=hf_token)
        
        # Informations du modèle
        model_info = api.model_info(VOXTRAL_MODEL, token=hf_token)
        logger.info(f"✅ Modèle accessible: {model_info.modelId}")
        logger.info(f"   Dernière modification: {model_info.lastModified}")
        logger.info(f"   Taille estimée: {model_info.safetensors}")
        
        # Liste des fichiers
        files = list_repo_files(VOXTRAL_MODEL, token=hf_token)
        model_files = [f for f in files if f.endswith('.safetensors')]
        logger.info(f"   Fichiers modèle: {len(model_files)} fichiers .safetensors")
        
        return True
    except Exception as e:
        logger.error(f"❌ Impossible d'accéder au modèle: {e}")
        return False

def check_cache_status():
    """Vérification du statut du cache"""
    logger.info("=== Statut du Cache ===")
    
    try:
        cache_info = scan_cache_dir()
        logger.info(f"Cache HuggingFace: {cache_info.size_on_disk_str}")
        
        cached_models = [repo.repo_id for repo in cache_info.repos]
        if VOXTRAL_MODEL in cached_models:
            for repo in cache_info.repos:
                if repo.repo_id == VOXTRAL_MODEL:
                    logger.info(f"✅ {VOXTRAL_MODEL} en cache: {repo.size_on_disk_str}")
                    logger.info(f"   Révisions: {len(repo.revisions)}")
        else:
            logger.info(f"ℹ️ {VOXTRAL_MODEL} non en cache")
            
    except Exception as e:
        logger.error(f"❌ Erreur de cache: {e}")

def main():
    """Diagnostic complet"""
    logger.info("🔍 Diagnostic Voxtral - Analyse des problèmes de téléchargement")
    logger.info("=" * 70)
    
    check_system_resources()
    print()
    
    check_network_connectivity()
    print()
    
    auth_ok = check_hf_authentication()
    print()
    
    if auth_ok:
        check_model_accessibility()
        print()
    
    check_cache_status()
    print()
    
    logger.info("=== Recommandations ===")
    
    # Vérifications importantes
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    if ram.available / (1024**3) < 50:
        logger.warning("⚠️ RAM faible (<50GB) - risque d'OOM lors du chargement")
    
    if disk.free / (1024**3) < 100:
        logger.warning("⚠️ Espace disque faible (<100GB) - risque d'échec de téléchargement")
    
    if not os.getenv("HF_TOKEN"):
        logger.warning("⚠️ Définir HF_TOKEN pour l'accès au modèle Voxtral")
    
    logger.info("✅ Diagnostic terminé")

if __name__ == "__main__":
    main()