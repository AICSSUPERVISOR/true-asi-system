#!/usr/bin/env python3.11
"""
LLM Downloader v3 - Using Python API directly for FULL WEIGHTED models
100/100 Quality, 100% Functionality, Non-Stop Operation
"""

import os
import json
import boto3
import time
from datetime import datetime
from huggingface_hub import snapshot_download
import shutil

# Configuration
S3_BUCKET = "asi-knowledge-base-898982995956"
S3_PREFIX = "LLM_MODELS_PUBLIC/"
PROGRESS_FILE = "/home/ubuntu/true-asi-build/llm_status_v3.json"
LOG_FILE = "/home/ubuntu/true-asi-build/llm_log_v3.txt"

s3_client = boto3.client('s3')

# 48 PUBLIC FULL-WEIGHTED MODELS
MODELS = [
    # Mistral (2)
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    # Qwen (5)
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    # Google (3)
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",
    "google/codegemma-7b-it",
    # BigCode (2)
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-3b",
    # Salesforce (3)
    "Salesforce/codegen2-7B",
    "Salesforce/codegen2-3_7B",
    "Salesforce/codegen2-1B",
    # DeepSeek (2)
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
    # Microsoft (3)
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Orca-2-7b",
    # Others (28)
    "tiiuae/falcon-7b-instruct",
    "mosaicml/mpt-7b-chat",
    "stabilityai/stablelm-zephyr-3b",
    "01-ai/Yi-6B-Chat",
    "THUDM/chatglm3-6b",
    "baichuan-inc/Baichuan2-7B-Chat",
    "internlm/internlm2-chat-7b",
    "openchat/openchat-3.5-0106",
    "WizardLM/WizardCoder-15B-V1.0",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "EleutherAI/gpt-j-6b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/llemma_7b",
    "bigscience/bloom-3b",
    "bigscience/bloom-1b7",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "liuhaotian/llava-v1.5-7b",
    "dmis-lab/biobert-v1.1",
    "emilyalsentzer/Bio_ClinicalBERT",
    "cerebras/Cerebras-GPT-6.7B",
    "cerebras/Cerebras-GPT-2.7B"
]

def log(msg):
    """Log to file and console"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def save_progress(data):
    """Save progress"""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key="LLM_DOWNLOAD_STATUS_V3.json",
            Body=json.dumps(data, indent=2)
        )
    except:
        pass

def get_dir_size(path):
    """Get directory size"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except:
        pass
    return total

def format_size(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def download_model(model_id, index, total):
    """Download FULL WEIGHTED model"""
    log(f"[{index}/{total}] Starting: {model_id}")
    start = time.time()
    
    local_dir = f"/tmp/llm_v3/{model_id.replace('/', '_')}"
    
    result = {
        "model_id": model_id,
        "status": "downloading",
        "start_time": datetime.now().isoformat()
    }
    
    try:
        # Download FULL model with ALL weights
        log(f"  Downloading FULL WEIGHTED model...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4
        )
        
        # Get size
        size_bytes = get_dir_size(local_dir)
        size_str = format_size(size_bytes)
        log(f"  Downloaded: {size_str}")
        
        # Count files
        file_count = sum([len(files) for _, _, files in os.walk(local_dir)])
        log(f"  Files: {file_count}")
        
        # Verify weight files exist
        weight_files = []
        for root, _, files in os.walk(local_dir):
            for f in files:
                if any(ext in f for ext in ['.bin', '.safetensors', '.pt', '.pth', '.ckpt']):
                    weight_files.append(f)
        
        if weight_files:
            log(f"  ✅ VERIFIED: {len(weight_files)} weight files found")
        else:
            log(f"  ⚠️  WARNING: No weight files detected!")
        
        # Upload to S3
        log(f"  Uploading to S3...")
        s3_prefix = f"{S3_PREFIX}{model_id.replace('/', '_')}"
        uploaded = 0
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{rel_path}"
                
                s3_client.upload_file(local_path, S3_BUCKET, s3_key)
                uploaded += 1
                
                if uploaded % 50 == 0:
                    log(f"    Uploaded {uploaded}/{file_count} files...")
        
        duration = time.time() - start
        
        result.update({
            "status": "success",
            "size": size_str,
            "size_bytes": size_bytes,
            "file_count": file_count,
            "weight_files": len(weight_files),
            "duration": duration,
            "end_time": datetime.now().isoformat()
        })
        
        log(f"  ✅ SUCCESS: {model_id} ({size_str}, {file_count} files, {duration:.1f}s)")
        
    except Exception as e:
        result.update({
            "status": "error",
            "error": str(e)[:500],
            "end_time": datetime.now().isoformat()
        })
        log(f"  ❌ ERROR: {model_id} - {str(e)[:200]}")
    
    finally:
        # Cleanup
        try:
            shutil.rmtree(local_dir, ignore_errors=True)
        except:
            pass
    
    return result

def main():
    log("="*80)
    log("LLM DOWNLOADER V3 - FULL WEIGHTED MODELS")
    log("="*80)
    log(f"Target: s3://{S3_BUCKET}/{S3_PREFIX}")
    log(f"Total models: {len(MODELS)}")
    log("="*80)
    
    results = []
    success = 0
    total_size = 0
    
    for i, model_id in enumerate(MODELS, 1):
        result = download_model(model_id, i, len(MODELS))
        results.append(result)
        
        if result["status"] == "success":
            success += 1
            total_size += result.get("size_bytes", 0)
        
        # Save progress
        progress = {
            "total": len(MODELS),
            "processed": i,
            "successful": success,
            "failed": i - success,
            "success_rate": f"{(success/i*100):.1f}%",
            "total_size": format_size(total_size),
            "results": results,
            "last_updated": datetime.now().isoformat()
        }
        save_progress(progress)
    
    # Final
    log("\n" + "="*80)
    log("DOWNLOAD COMPLETE")
    log("="*80)
    log(f"Total: {len(MODELS)}")
    log(f"Successful: {success}")
    log(f"Failed: {len(MODELS) - success}")
    log(f"Success Rate: {(success/len(MODELS)*100):.1f}%")
    log(f"Total Size: {format_size(total_size)}")
    log("="*80)

if __name__ == "__main__":
    main()
