#!/usr/bin/env python3.11
"""
PERSISTENT LLM DOWNLOADER FOR EC2
Run this on your persistent EC2 instance (NOT in Manus sandbox)

Usage:
    nohup python3.11 PERSISTENT_LLM_DOWNLOADER.py > llm_download.log 2>&1 &
    
Monitor:
    tail -f llm_download.log
    cat llm_download_progress.json
"""

import os
import json
import time
import subprocess
from datetime import datetime
from huggingface_hub import snapshot_download, list_repo_files
import boto3

# Configuration
S3_BUCKET = "asi-knowledge-base-898982995956"
S3_PREFIX = "FULL_WEIGHTED_MODELS/"
AWS_REGION = "us-east-1"
DOWNLOAD_DIR = "/tmp/llm_downloads"
PROGRESS_FILE = "llm_download_progress.json"
LOG_FILE = "llm_download.log"

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Complete list of 48 PUBLIC models (no gated models)
MODELS_TO_DOWNLOAD = [
    # Mistral (2 models) - ~105 GB
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    
    # Qwen 2.5 (5 models) - ~45 GB
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    
    # Google Gemma (3 models) - ~37 GB
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "google/codegemma-7b-it",
    
    # BigCode StarCoder (2 models) - ~21 GB
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-7b",
    
    # Salesforce CodeGen (3 models) - ~25 GB
    "Salesforce/codegen25-7b-multi",
    "Salesforce/codegen25-7b-instruct",
    "Salesforce/codegen-16B-multi",
    
    # DeepSeek (2 models) - ~29 GB
    "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
    
    # Microsoft (3 models) - ~27 GB
    "microsoft/phi-3-medium-4k-instruct",
    "microsoft/phi-3-small-8k-instruct",
    "microsoft/Orca-2-13b",
    
    # Falcon (2 models) - ~20 GB
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    
    # MosaicML MPT (2 models) - ~14 GB
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-30b-instruct",
    
    # StabilityAI (2 models) - ~14 GB
    "stabilityai/stablelm-3b-4e1t",
    "stabilityai/stablelm-zephyr-3b",
    
    # 01.AI Yi (2 models) - ~40 GB
    "01-ai/Yi-34B-Chat",
    "01-ai/Yi-6B-Chat",
    
    # THUDM ChatGLM (1 model) - ~12 GB
    "THUDM/chatglm3-6b",
    
    # Baichuan (2 models) - ~20 GB
    "baichuan-inc/Baichuan2-13B-Chat",
    "baichuan-inc/Baichuan2-7B-Chat",
    
    # InternLM (2 models) - ~27 GB
    "internlm/internlm2-20b",
    "internlm/internlm2-7b",
    
    # OpenChat (1 model) - ~7 GB
    "openchat/openchat-3.5-0106",
    
    # WizardLM (2 models) - ~28 GB
    "WizardLM/WizardCoder-15B-V1.0",
    "WizardLM/WizardMath-13B-V1.0",
    
    # TinyLlama (1 model) - ~1.1 GB
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # SmolLM (2 models) - ~2 GB
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    
    # EleutherAI (4 models) - ~30 GB
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/llemma-7b",
    
    # BLOOM (3 models) - ~12 GB
    "bigscience/bloom-7b1",
    "bigscience/bloom-3b",
    "bigscience/bloom-1b7",
]

def log(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, "a") as f:
        f.write(log_msg + "\n")

def load_progress():
    """Load download progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "current": None, "total": len(MODELS_TO_DOWNLOAD)}

def save_progress(progress):
    """Save download progress to file"""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

def upload_to_s3(local_path, model_name):
    """Upload model to S3"""
    log(f"Uploading {model_name} to S3...")
    s3_key = f"{S3_PREFIX}{model_name}/"
    
    try:
        # Upload all files in the model directory
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_path)
                s3_file_key = f"{s3_key}{relative_path}"
                
                s3_client.upload_file(local_file, S3_BUCKET, s3_file_key)
                log(f"  Uploaded: {relative_path}")
        
        log(f"✅ Successfully uploaded {model_name} to S3")
        return True
    except Exception as e:
        log(f"❌ Failed to upload {model_name} to S3: {str(e)}")
        return False

def download_model(model_name):
    """Download a single model from Hugging Face"""
    log(f"Starting download: {model_name}")
    
    try:
        # Create download directory
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        model_path = os.path.join(DOWNLOAD_DIR, model_name.replace("/", "_"))
        
        # Download model
        log(f"Downloading {model_name} from Hugging Face...")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        
        log(f"✅ Downloaded {model_name}")
        
        # Upload to S3
        if upload_to_s3(model_path, model_name):
            # Clean up local files
            subprocess.run(["rm", "-rf", model_path], check=True)
            log(f"🗑️ Cleaned up local files for {model_name}")
            return True
        else:
            return False
            
    except Exception as e:
        log(f"❌ Failed to download {model_name}: {str(e)}")
        return False

def main():
    """Main download loop"""
    log("=" * 80)
    log("PERSISTENT LLM DOWNLOADER STARTED")
    log("=" * 80)
    log(f"Total models to download: {len(MODELS_TO_DOWNLOAD)}")
    log(f"S3 Bucket: {S3_BUCKET}")
    log(f"S3 Prefix: {S3_PREFIX}")
    log("")
    
    # Load progress
    progress = load_progress()
    log(f"Progress loaded: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
    
    # Download each model
    for i, model_name in enumerate(MODELS_TO_DOWNLOAD, 1):
        # Skip if already completed
        if model_name in progress['completed']:
            log(f"[{i}/{len(MODELS_TO_DOWNLOAD)}] Skipping {model_name} (already completed)")
            continue
        
        # Update current model
        progress['current'] = model_name
        save_progress(progress)
        
        log(f"[{i}/{len(MODELS_TO_DOWNLOAD)}] Processing {model_name}")
        
        # Download and upload
        if download_model(model_name):
            progress['completed'].append(model_name)
            log(f"✅ [{i}/{len(MODELS_TO_DOWNLOAD)}] Completed {model_name}")
        else:
            progress['failed'].append(model_name)
            log(f"❌ [{i}/{len(MODELS_TO_DOWNLOAD)}] Failed {model_name}")
        
        # Save progress
        progress['current'] = None
        save_progress(progress)
        
        # Log summary
        log(f"Progress: {len(progress['completed'])}/{len(MODELS_TO_DOWNLOAD)} completed, {len(progress['failed'])} failed")
        log("")
    
    # Final summary
    log("=" * 80)
    log("DOWNLOAD COMPLETE")
    log("=" * 80)
    log(f"Total completed: {len(progress['completed'])}/{len(MODELS_TO_DOWNLOAD)}")
    log(f"Total failed: {len(progress['failed'])}/{len(MODELS_TO_DOWNLOAD)}")
    
    if progress['failed']:
        log("\nFailed models:")
        for model in progress['failed']:
            log(f"  - {model}")
    
    log("\n✅ All downloads processed!")

if __name__ == "__main__":
    main()
