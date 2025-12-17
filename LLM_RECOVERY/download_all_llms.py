#!/usr/bin/env python3
"""
LLM RECOVERY SYSTEM - Download all models and push to GitHub
Based on recovered S3 audit showing 600+ models were stored

This script:
1. Downloads models from Hugging Face
2. Saves model configs and small files to GitHub
3. Creates download scripts for large model weights
"""

import os
import json
import subprocess
from datetime import datetime

# Complete LLM catalog based on S3 audit and recovered scripts
FULL_LLM_CATALOG = {
    # Models found in S3 audit (largest files)
    "s3_recovered_models": [
        "xai-org/grok-2",
        "Salesforce/codegen-16B-mono",
        "WizardLM/WizardCoder-15B-V1.0",
        "facebook/incoder-6B",
        "EleutherAI/gpt-j-6b",
        "meta-llama/Llama-3.1-70B-Instruct",
    ],
    
    # Foundational Models (from llm_download_system.py)
    "meta_llama": [
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3-70B-Instruct",
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-7b-hf",
    ],
    
    "mistral": [
        "mistralai/Mistral-Large-Instruct-2407",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Codestral-22B-v0.1",
    ],
    
    "google": [
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-2b-it",
        "google/codegemma-7b-it",
        "google/codegemma-2b",
    ],
    
    "qwen": [
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen-VL-Chat",
    ],
    
    "deepseek": [
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "deepseek-ai/deepseek-coder-33b-instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "deepseek-ai/deepseek-math-7b-instruct",
    ],
    
    "code_models": [
        "meta-llama/CodeLlama-70b-Instruct-hf",
        "meta-llama/CodeLlama-34b-Instruct-hf",
        "meta-llama/CodeLlama-13b-Instruct-hf",
        "meta-llama/CodeLlama-7b-Instruct-hf",
        "bigcode/starcoder2-15b",
        "bigcode/starcoder2-7b",
        "bigcode/starcoder2-3b",
        "Salesforce/codegen2-16B",
        "Salesforce/codegen2-7B",
        "Salesforce/codegen2-3_7B",
        "Salesforce/codegen2-1B",
        "WizardLM/WizardCoder-33B-V1.1",
        "WizardLM/WizardCoder-15B-V1.0",
        "Phind/Phind-CodeLlama-34B-v2",
    ],
    
    "math_reasoning": [
        "EleutherAI/llemma_34b",
        "EleutherAI/llemma_7b",
        "TIGER-Lab/MAmmoTH-70B",
        "TIGER-Lab/MAmmoTH-13B",
        "TIGER-Lab/MAmmoTH-7B",
        "WizardLM/WizardMath-70B-V1.0",
        "WizardLM/WizardMath-13B-V1.0",
        "WizardLM/WizardMath-7B-V1.0",
        "GAIR/Abel-70B",
        "GAIR/Abel-13B",
        "GAIR/Abel-7B",
    ],
    
    "instruction_tuned": [
        "lmsys/vicuna-33b-v1.3",
        "lmsys/vicuna-13b-v1.5",
        "lmsys/vicuna-7b-v1.5",
        "timdettmers/guanaco-65b",
        "timdettmers/guanaco-33b",
        "timdettmers/guanaco-13b",
        "timdettmers/guanaco-7b",
        "microsoft/Orca-2-13b",
        "microsoft/Orca-2-7b",
        "Open-Orca/Mistral-7B-OpenOrca",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "NousResearch/Nous-Hermes-Llama2-70b",
        "NousResearch/Nous-Hermes-Llama2-13b",
        "WizardLM/WizardLM-70B-V1.0",
        "WizardLM/WizardLM-13B-V1.2",
        "WizardLM/WizardLM-7B-V1.0",
    ],
    
    "chat_optimized": [
        "THUDM/chatglm3-6b",
        "baichuan-inc/Baichuan2-13B-Chat",
        "baichuan-inc/Baichuan2-7B-Chat",
        "internlm/internlm2-chat-20b",
        "internlm/internlm2-chat-7b",
        "01-ai/Yi-34B-Chat",
        "01-ai/Yi-6B-Chat",
        "tiiuae/falcon-180B",
        "tiiuae/falcon-40b-instruct",
        "tiiuae/falcon-7b-instruct",
        "mosaicml/mpt-30b-chat",
        "mosaicml/mpt-7b-chat",
        "stabilityai/stablelm-tuned-alpha-7b",
        "openchat/openchat-3.5-0106",
    ],
    
    "small_efficient": [
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3-small-8k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/phi-2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "stabilityai/stablelm-zephyr-3b",
    ],
    
    "multimodal": [
        "liuhaotian/llava-v1.6-34b",
        "liuhaotian/llava-v1.5-13b",
        "liuhaotian/llava-v1.5-7b",
        "Salesforce/blip2-opt-6.7b",
        "Salesforce/blip2-opt-2.7b",
    ],
    
    "embedding": [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/e5-large-v2",
        "intfloat/e5-base-v2",
        "intfloat/e5-small-v2",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-small-en-v1.5",
    ],
    
    "medical": [
        "microsoft/BioGPT-Large",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "dmis-lab/biobert-v1.1",
        "emilyalsentzer/Bio_ClinicalBERT",
        "UFNLP/gatortron-base",
    ],
    
    "multilingual": [
        "bigscience/bloom-7b1",
        "bigscience/bloom-3b",
        "bigscience/bloom-1b7",
        "bigscience/bloom-560m",
        "inceptionai/jais-13b-chat",
    ],
}

def count_all_models():
    """Count total models in catalog"""
    total = 0
    for category, models in FULL_LLM_CATALOG.items():
        total += len(models)
    return total

def download_model_config(model_id, output_dir):
    """Download only config files (not weights) for a model"""
    try:
        safe_name = model_id.replace('/', '_')
        model_dir = os.path.join(output_dir, safe_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Download only config files
        cmd = [
            "huggingface-cli", "download",
            model_id,
            "--include", "*.json", "*.txt", "*.md", "tokenizer*",
            "--exclude", "*.bin", "*.safetensors", "*.pth", "*.gguf", "*.h5",
            "--local-dir", model_dir,
            "--local-dir-use-symlinks", "False"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return {"model_id": model_id, "status": "success", "path": model_dir}
        else:
            return {"model_id": model_id, "status": "failed", "error": result.stderr[:200]}
    except Exception as e:
        return {"model_id": model_id, "status": "error", "error": str(e)[:200]}

def create_download_script(model_id, output_file):
    """Create a shell script to download full model weights"""
    script = f"""#!/bin/bash
# Download script for {model_id}
# Generated by LLM Recovery System

echo "Downloading {model_id}..."
huggingface-cli download {model_id} --local-dir ./models/{model_id.replace('/', '_')}
echo "Download complete!"
"""
    with open(output_file, 'w') as f:
        f.write(script)
    os.chmod(output_file, 0o755)

def main():
    print("=" * 80)
    print("LLM RECOVERY SYSTEM")
    print("=" * 80)
    
    total_models = count_all_models()
    print(f"Total models in catalog: {total_models}")
    
    # Create output directories
    output_dir = "/home/ubuntu/github_push/LLM_RECOVERY"
    configs_dir = os.path.join(output_dir, "model_configs")
    scripts_dir = os.path.join(output_dir, "download_scripts")
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    
    # Install huggingface-cli
    print("Installing Hugging Face CLI...")
    subprocess.run(["pip3", "install", "--quiet", "huggingface-hub[cli]"])
    
    # Save catalog
    catalog_file = os.path.join(output_dir, "FULL_LLM_CATALOG.json")
    with open(catalog_file, 'w') as f:
        json.dump(FULL_LLM_CATALOG, f, indent=2)
    print(f"Saved catalog to {catalog_file}")
    
    # Process each category
    results = []
    processed = 0
    
    for category, models in FULL_LLM_CATALOG.items():
        print(f"\n=== Processing {category} ({len(models)} models) ===")
        
        for model_id in models:
            processed += 1
            print(f"[{processed}/{total_models}] {model_id}")
            
            # Create download script for full weights
            script_file = os.path.join(scripts_dir, f"{model_id.replace('/', '_')}.sh")
            create_download_script(model_id, script_file)
            
            # Download config files only (small files for GitHub)
            result = download_model_config(model_id, configs_dir)
            result["category"] = category
            results.append(result)
            
            # Save progress
            with open(os.path.join(output_dir, "download_progress.json"), 'w') as f:
                json.dump({
                    "total": total_models,
                    "processed": processed,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
    
    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    print("\n" + "=" * 80)
    print("LLM RECOVERY COMPLETE")
    print("=" * 80)
    print(f"Total models: {total_models}")
    print(f"Configs downloaded: {success}")
    print(f"Download scripts created: {total_models}")
    print("=" * 80)
    
    # Save final report
    with open(os.path.join(output_dir, "RECOVERY_REPORT.json"), 'w') as f:
        json.dump({
            "total_models": total_models,
            "configs_downloaded": success,
            "download_scripts_created": total_models,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nAll files saved to {output_dir}")
    print("Run download scripts to get full model weights")

if __name__ == "__main__":
    main()
