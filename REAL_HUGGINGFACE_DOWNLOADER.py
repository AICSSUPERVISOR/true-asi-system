#!/usr/bin/env python3
"""
TRUE ASI - REAL HUGGINGFACE MODEL DOWNLOADER
============================================
Downloads ACTUAL model weights from HuggingFace Hub.
NO MOCK DATA. NO SIMULATIONS. 100% REAL.

Total: 100+ verified models
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional

# Configuration
DOWNLOAD_DIR = Path("/home/ubuntu/ASI_MODELS")
MAX_RETRIES = 3

# HuggingFace models with VERIFIED IDs - 100+ models
VERIFIED_MODELS = [
    # ============================================
    # TIER 1: META LLAMA (Foundation)
    # ============================================
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "size_gb": 140, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-3.1-70B-Instruct", "size_gb": 140, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-3.1-8B-Instruct", "size_gb": 16, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-3-70B-Instruct", "size_gb": 140, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-3-8B-Instruct", "size_gb": 16, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-2-70b-hf", "size_gb": 140, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-2-13b-hf", "size_gb": 26, "priority": 1, "category": "foundation"},
    {"id": "meta-llama/Llama-2-7b-hf", "size_gb": 14, "priority": 1, "category": "foundation"},
    
    # ============================================
    # TIER 2: MISTRAL/MIXTRAL
    # ============================================
    {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "size_gb": 282, "priority": 2, "category": "foundation"},
    {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "size_gb": 94, "priority": 2, "category": "foundation"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "size_gb": 14, "priority": 2, "category": "foundation"},
    {"id": "mistralai/Mistral-Large-Instruct-2407", "size_gb": 246, "priority": 2, "category": "foundation"},
    {"id": "mistralai/Mistral-Nemo-Instruct-2407", "size_gb": 24, "priority": 2, "category": "foundation"},
    
    # ============================================
    # TIER 3: QWEN
    # ============================================
    {"id": "Qwen/Qwen2.5-72B-Instruct", "size_gb": 144, "priority": 3, "category": "foundation"},
    {"id": "Qwen/Qwen2.5-32B-Instruct", "size_gb": 64, "priority": 3, "category": "foundation"},
    {"id": "Qwen/Qwen2.5-14B-Instruct", "size_gb": 28, "priority": 3, "category": "foundation"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "size_gb": 14, "priority": 3, "category": "foundation"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "size_gb": 6, "priority": 3, "category": "foundation"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size_gb": 3, "priority": 3, "category": "foundation"},
    {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "size_gb": 64, "priority": 3, "category": "code"},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "size_gb": 14, "priority": 3, "category": "code"},
    {"id": "Qwen/QwQ-32B-Preview", "size_gb": 64, "priority": 3, "category": "reasoning"},
    
    # ============================================
    # TIER 4: DEEPSEEK
    # ============================================
    {"id": "deepseek-ai/DeepSeek-V3", "size_gb": 1342, "priority": 4, "category": "foundation"},
    {"id": "deepseek-ai/DeepSeek-R1", "size_gb": 1342, "priority": 4, "category": "reasoning"},
    {"id": "deepseek-ai/DeepSeek-V2.5", "size_gb": 472, "priority": 4, "category": "foundation"},
    {"id": "deepseek-ai/deepseek-coder-33b-instruct", "size_gb": 66, "priority": 4, "category": "code"},
    {"id": "deepseek-ai/deepseek-coder-6.7b-instruct", "size_gb": 14, "priority": 4, "category": "code"},
    {"id": "deepseek-ai/deepseek-coder-1.3b-instruct", "size_gb": 2.6, "priority": 4, "category": "code"},
    {"id": "deepseek-ai/deepseek-math-7b-instruct", "size_gb": 14, "priority": 4, "category": "math"},
    
    # ============================================
    # TIER 5: GOOGLE GEMMA
    # ============================================
    {"id": "google/gemma-2-27b-it", "size_gb": 54, "priority": 5, "category": "foundation"},
    {"id": "google/gemma-2-9b-it", "size_gb": 18, "priority": 5, "category": "foundation"},
    {"id": "google/gemma-2-2b-it", "size_gb": 4, "priority": 5, "category": "foundation"},
    {"id": "google/codegemma-7b-it", "size_gb": 14, "priority": 5, "category": "code"},
    {"id": "google/codegemma-2b", "size_gb": 4, "priority": 5, "category": "code"},
    
    # ============================================
    # TIER 6: CODE MODELS
    # ============================================
    {"id": "meta-llama/CodeLlama-70b-Instruct-hf", "size_gb": 140, "priority": 6, "category": "code"},
    {"id": "meta-llama/CodeLlama-34b-Instruct-hf", "size_gb": 68, "priority": 6, "category": "code"},
    {"id": "meta-llama/CodeLlama-13b-Instruct-hf", "size_gb": 26, "priority": 6, "category": "code"},
    {"id": "meta-llama/CodeLlama-7b-Instruct-hf", "size_gb": 14, "priority": 6, "category": "code"},
    {"id": "bigcode/starcoder2-15b", "size_gb": 30, "priority": 6, "category": "code"},
    {"id": "bigcode/starcoder2-7b", "size_gb": 14, "priority": 6, "category": "code"},
    {"id": "bigcode/starcoder2-3b", "size_gb": 6, "priority": 6, "category": "code"},
    {"id": "WizardLM/WizardCoder-33B-V1.1", "size_gb": 68, "priority": 6, "category": "code"},
    {"id": "WizardLM/WizardCoder-15B-V1.0", "size_gb": 30, "priority": 6, "category": "code"},
    {"id": "Phind/Phind-CodeLlama-34B-v2", "size_gb": 68, "priority": 6, "category": "code"},
    {"id": "mistralai/Codestral-22B-v0.1", "size_gb": 44, "priority": 6, "category": "code"},
    {"id": "Salesforce/codegen2-16B", "size_gb": 32, "priority": 6, "category": "code"},
    {"id": "Salesforce/codegen2-7B", "size_gb": 14, "priority": 6, "category": "code"},
    {"id": "Salesforce/codegen25-7b-instruct", "size_gb": 14, "priority": 6, "category": "code"},
    
    # ============================================
    # TIER 7: MATH MODELS
    # ============================================
    {"id": "WizardLM/WizardMath-70B-V1.0", "size_gb": 140, "priority": 7, "category": "math"},
    {"id": "WizardLM/WizardMath-13B-V1.0", "size_gb": 26, "priority": 7, "category": "math"},
    {"id": "WizardLM/WizardMath-7B-V1.0", "size_gb": 14, "priority": 7, "category": "math"},
    {"id": "EleutherAI/llemma_34b", "size_gb": 68, "priority": 7, "category": "math"},
    {"id": "EleutherAI/llemma_7b", "size_gb": 14, "priority": 7, "category": "math"},
    {"id": "TIGER-Lab/MAmmoTH-70B", "size_gb": 140, "priority": 7, "category": "math"},
    {"id": "TIGER-Lab/MAmmoTH-13B", "size_gb": 26, "priority": 7, "category": "math"},
    {"id": "TIGER-Lab/MAmmoTH-7B", "size_gb": 14, "priority": 7, "category": "math"},
    
    # ============================================
    # TIER 8: MULTIMODAL / VISION
    # ============================================
    {"id": "llava-hf/llava-v1.6-34b-hf", "size_gb": 68, "priority": 8, "category": "multimodal"},
    {"id": "llava-hf/llava-v1.6-mistral-7b-hf", "size_gb": 14, "priority": 8, "category": "multimodal"},
    {"id": "llava-hf/llava-1.5-13b-hf", "size_gb": 26, "priority": 8, "category": "multimodal"},
    {"id": "llava-hf/llava-1.5-7b-hf", "size_gb": 14, "priority": 8, "category": "multimodal"},
    {"id": "Salesforce/blip2-opt-6.7b", "size_gb": 14, "priority": 8, "category": "multimodal"},
    {"id": "Salesforce/blip2-opt-2.7b", "size_gb": 6, "priority": 8, "category": "multimodal"},
    {"id": "microsoft/kosmos-2-patch14-224", "size_gb": 4, "priority": 8, "category": "multimodal"},
    {"id": "Qwen/Qwen2-VL-72B-Instruct", "size_gb": 144, "priority": 8, "category": "multimodal"},
    {"id": "Qwen/Qwen2-VL-7B-Instruct", "size_gb": 14, "priority": 8, "category": "multimodal"},
    
    # ============================================
    # TIER 9: EMBEDDING MODELS
    # ============================================
    {"id": "BAAI/bge-large-en-v1.5", "size_gb": 0.67, "priority": 9, "category": "embedding"},
    {"id": "BAAI/bge-base-en-v1.5", "size_gb": 0.22, "priority": 9, "category": "embedding"},
    {"id": "BAAI/bge-small-en-v1.5", "size_gb": 0.07, "priority": 9, "category": "embedding"},
    {"id": "BAAI/bge-m3", "size_gb": 1.1, "priority": 9, "category": "embedding"},
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "size_gb": 0.09, "priority": 9, "category": "embedding"},
    {"id": "sentence-transformers/all-mpnet-base-v2", "size_gb": 0.44, "priority": 9, "category": "embedding"},
    {"id": "intfloat/e5-large-v2", "size_gb": 0.67, "priority": 9, "category": "embedding"},
    {"id": "intfloat/e5-base-v2", "size_gb": 0.22, "priority": 9, "category": "embedding"},
    {"id": "thenlper/gte-large", "size_gb": 0.67, "priority": 9, "category": "embedding"},
    {"id": "thenlper/gte-base", "size_gb": 0.22, "priority": 9, "category": "embedding"},
    
    # ============================================
    # TIER 10: WHISPER (AUDIO)
    # ============================================
    {"id": "openai/whisper-large-v3", "size_gb": 3.1, "priority": 10, "category": "audio"},
    {"id": "openai/whisper-large-v2", "size_gb": 3.1, "priority": 10, "category": "audio"},
    {"id": "openai/whisper-medium", "size_gb": 1.54, "priority": 10, "category": "audio"},
    {"id": "openai/whisper-small", "size_gb": 0.49, "priority": 10, "category": "audio"},
    {"id": "openai/whisper-base", "size_gb": 0.15, "priority": 10, "category": "audio"},
    {"id": "openai/whisper-tiny", "size_gb": 0.08, "priority": 10, "category": "audio"},
    
    # ============================================
    # TIER 11: STABLE DIFFUSION (IMAGE)
    # ============================================
    {"id": "stabilityai/stable-diffusion-xl-base-1.0", "size_gb": 13.2, "priority": 11, "category": "image"},
    {"id": "stabilityai/stable-diffusion-2-1", "size_gb": 1.73, "priority": 11, "category": "image"},
    {"id": "runwayml/stable-diffusion-v1-5", "size_gb": 1.73, "priority": 11, "category": "image"},
    {"id": "stabilityai/stable-video-diffusion-img2vid-xt", "size_gb": 2.8, "priority": 11, "category": "video"},
    {"id": "stabilityai/sdxl-turbo", "size_gb": 13.2, "priority": 11, "category": "image"},
    {"id": "black-forest-labs/FLUX.1-dev", "size_gb": 24, "priority": 11, "category": "image"},
    {"id": "black-forest-labs/FLUX.1-schnell", "size_gb": 24, "priority": 11, "category": "image"},
    
    # ============================================
    # TIER 12: FALCON
    # ============================================
    {"id": "tiiuae/falcon-180B", "size_gb": 360, "priority": 12, "category": "foundation"},
    {"id": "tiiuae/falcon-40b-instruct", "size_gb": 80, "priority": 12, "category": "foundation"},
    {"id": "tiiuae/falcon-7b-instruct", "size_gb": 14, "priority": 12, "category": "foundation"},
    {"id": "tiiuae/falcon-11B", "size_gb": 22, "priority": 12, "category": "foundation"},
    
    # ============================================
    # TIER 13: MICROSOFT PHI
    # ============================================
    {"id": "microsoft/phi-3-medium-4k-instruct", "size_gb": 28, "priority": 13, "category": "foundation"},
    {"id": "microsoft/phi-3-small-8k-instruct", "size_gb": 14, "priority": 13, "category": "foundation"},
    {"id": "microsoft/phi-3-mini-4k-instruct", "size_gb": 7.6, "priority": 13, "category": "foundation"},
    {"id": "microsoft/Phi-3.5-mini-instruct", "size_gb": 7.6, "priority": 13, "category": "foundation"},
    {"id": "microsoft/Phi-3.5-MoE-instruct", "size_gb": 84, "priority": 13, "category": "foundation"},
    {"id": "microsoft/Orca-2-13b", "size_gb": 26, "priority": 13, "category": "foundation"},
    {"id": "microsoft/Orca-2-7b", "size_gb": 14, "priority": 13, "category": "foundation"},
    
    # ============================================
    # TIER 14: OTHER FOUNDATION MODELS
    # ============================================
    {"id": "mosaicml/mpt-30b-instruct", "size_gb": 60, "priority": 14, "category": "foundation"},
    {"id": "mosaicml/mpt-7b-instruct", "size_gb": 14, "priority": 14, "category": "foundation"},
    {"id": "stabilityai/stablelm-zephyr-3b", "size_gb": 6, "priority": 14, "category": "foundation"},
    {"id": "stabilityai/stablelm-2-12b", "size_gb": 24, "priority": 14, "category": "foundation"},
    {"id": "THUDM/chatglm3-6b", "size_gb": 12, "priority": 14, "category": "foundation"},
    {"id": "THUDM/glm-4-9b-chat", "size_gb": 18, "priority": 14, "category": "foundation"},
    {"id": "01-ai/Yi-34B", "size_gb": 68, "priority": 14, "category": "foundation"},
    {"id": "01-ai/Yi-9B", "size_gb": 18, "priority": 14, "category": "foundation"},
    {"id": "01-ai/Yi-6B", "size_gb": 12, "priority": 14, "category": "foundation"},
    {"id": "01-ai/Yi-1.5-34B-Chat", "size_gb": 68, "priority": 14, "category": "foundation"},
    {"id": "01-ai/Yi-1.5-9B-Chat", "size_gb": 18, "priority": 14, "category": "foundation"},
    {"id": "internlm/internlm2-20b", "size_gb": 40, "priority": 14, "category": "foundation"},
    {"id": "internlm/internlm2-7b", "size_gb": 14, "priority": 14, "category": "foundation"},
    {"id": "internlm/internlm2_5-7b-chat", "size_gb": 14, "priority": 14, "category": "foundation"},
    {"id": "openchat/openchat-3.5-0106", "size_gb": 14, "priority": 14, "category": "foundation"},
    {"id": "openchat/openchat-3.6-8b-20240522", "size_gb": 16, "priority": 14, "category": "foundation"},
    {"id": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO", "size_gb": 94, "priority": 14, "category": "foundation"},
    {"id": "NousResearch/Hermes-3-Llama-3.1-8B", "size_gb": 16, "priority": 14, "category": "foundation"},
    {"id": "teknium/OpenHermes-2.5-Mistral-7B", "size_gb": 14, "priority": 14, "category": "foundation"},
    {"id": "cognitivecomputations/dolphin-2.9.1-llama-3-70b", "size_gb": 140, "priority": 14, "category": "foundation"},
    
    # ============================================
    # TIER 15: SMALL/EFFICIENT MODELS
    # ============================================
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size_gb": 2.2, "priority": 15, "category": "foundation"},
    {"id": "HuggingFaceTB/SmolLM-1.7B-Instruct", "size_gb": 3.4, "priority": 15, "category": "foundation"},
    {"id": "HuggingFaceTB/SmolLM-360M-Instruct", "size_gb": 0.72, "priority": 15, "category": "foundation"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "size_gb": 3.4, "priority": 15, "category": "foundation"},
    {"id": "EleutherAI/gpt-j-6b", "size_gb": 12, "priority": 15, "category": "foundation"},
    {"id": "EleutherAI/pythia-12b", "size_gb": 24, "priority": 15, "category": "foundation"},
    {"id": "EleutherAI/pythia-6.9b", "size_gb": 14, "priority": 15, "category": "foundation"},
    {"id": "EleutherAI/pythia-2.8b", "size_gb": 5.6, "priority": 15, "category": "foundation"},
    {"id": "EleutherAI/pythia-1.4b", "size_gb": 2.8, "priority": 15, "category": "foundation"},
    {"id": "bigscience/bloom-7b1", "size_gb": 14, "priority": 15, "category": "foundation"},
    {"id": "bigscience/bloom-3b", "size_gb": 6, "priority": 15, "category": "foundation"},
    {"id": "bigscience/bloom-1b7", "size_gb": 3.4, "priority": 15, "category": "foundation"},
    
    # ============================================
    # TIER 16: DOMAIN-SPECIFIC
    # ============================================
    {"id": "microsoft/BioGPT", "size_gb": 3, "priority": 16, "category": "medical"},
    {"id": "stanford-crfm/BioMedLM", "size_gb": 5.4, "priority": 16, "category": "medical"},
    {"id": "allenai/scibert_scivocab_uncased", "size_gb": 0.22, "priority": 16, "category": "science"},
    {"id": "ProsusAI/finbert", "size_gb": 0.22, "priority": 16, "category": "finance"},
    {"id": "nlpaueb/legal-bert-base-uncased", "size_gb": 0.22, "priority": 16, "category": "legal"},
    {"id": "emilyalsentzer/Bio_ClinicalBERT", "size_gb": 0.22, "priority": 16, "category": "medical"},
    {"id": "dmis-lab/biobert-v1.1", "size_gb": 0.22, "priority": 16, "category": "medical"},
]

def install_huggingface_hub():
    """Install huggingface_hub if not present"""
    try:
        import huggingface_hub
        print("✓ huggingface_hub already installed")
        return True
    except ImportError:
        print("Installing huggingface_hub...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        if result.returncode == 0:
            print("✓ huggingface_hub installed")
            return True
        else:
            print("✗ Failed to install huggingface_hub")
            return False

def download_model(model_id: str, output_dir: Path, max_retries: int = 3) -> bool:
    """Download a model from HuggingFace Hub"""
    from huggingface_hub import snapshot_download, HfApi
    
    model_dir = output_dir / model_id.replace("/", "_")
    
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"  ✓ Already downloaded: {model_id}")
        return True
    
    for attempt in range(max_retries):
        try:
            print(f"  Downloading {model_id} (attempt {attempt + 1}/{max_retries})...")
            
            snapshot_download(
                repo_id=model_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4
            )
            
            print(f"  ✓ Downloaded: {model_id}")
            return True
            
        except Exception as e:
            print(f"  ✗ Error downloading {model_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    
    return False

def get_download_stats(download_dir: Path):
    """Get current download statistics"""
    if not download_dir.exists():
        return {"downloaded": 0, "total_size_gb": 0}
    
    total_size = 0
    count = 0
    for model_dir in download_dir.iterdir():
        if model_dir.is_dir():
            count += 1
            for f in model_dir.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
    
    return {
        "downloaded": count,
        "total_size_gb": total_size / (1024**3)
    }

def main():
    print("=" * 80)
    print("TRUE ASI - REAL HUGGINGFACE MODEL DOWNLOADER")
    print("=" * 80)
    print(f"\nTotal models to download: {len(VERIFIED_MODELS)}")
    print(f"Total size: {sum(m['size_gb'] for m in VERIFIED_MODELS):.2f} GB")
    print(f"Download directory: {DOWNLOAD_DIR}")
    print()
    
    # Install dependencies
    if not install_huggingface_hub():
        print("Cannot proceed without huggingface_hub")
        return
    
    # Create download directory
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort by priority
    models = sorted(VERIFIED_MODELS, key=lambda x: (x['priority'], -x['size_gb']))
    
    # Download each model
    successful = 0
    failed = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model['id']} ({model['size_gb']:.1f} GB) [{model['category']}]")
        
        if download_model(model['id'], DOWNLOAD_DIR):
            successful += 1
        else:
            failed.append(model['id'])
        
        # Print progress
        stats = get_download_stats(DOWNLOAD_DIR)
        print(f"  Progress: {stats['downloaded']} models, {stats['total_size_gb']:.2f} GB downloaded")
    
    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Successful: {successful}/{len(models)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed models:")
        for m in failed:
            print(f"  - {m}")
    
    stats = get_download_stats(DOWNLOAD_DIR)
    print(f"\nTotal downloaded: {stats['total_size_gb']:.2f} GB")

if __name__ == "__main__":
    main()
