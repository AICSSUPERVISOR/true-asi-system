#!/bin/bash
# ============================================================
# ASI MODELS DOWNLOAD SCRIPT FOR RUNPOD
# Downloads all 4 priority models for 85%+ ARC-AGI performance
# ============================================================

set -e

echo "=============================================="
echo "ASI MODELS DOWNLOAD - TARGET: 85%+ ACCURACY"
echo "=============================================="
echo ""

# Configuration
MODELS_DIR="/workspace/models"
CACHE_DIR="/workspace/cache"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"

# Create directories
mkdir -p "$MODELS_DIR"
mkdir -p "$CACHE_DIR"

echo "Models directory: $MODELS_DIR"
echo "Cache directory: $CACHE_DIR"
echo ""

# Install dependencies
echo "=== Installing dependencies ==="
pip install -q huggingface_hub transformers torch accelerate bitsandbytes
echo "✅ Dependencies installed"
echo ""

# ============================================================
# MODEL 1: MARC-8B (MIT Test-Time Training)
# Expected: 62.8% ARC-AGI-1 (State-of-the-art)
# ============================================================
echo "=== Downloading MARC-8B (MIT TTT) ==="
echo "Model: ekinakyurek/marc-8B-finetuned-llama3"
echo "Size: ~16GB"
echo "Expected accuracy: 62.8%"
echo ""

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

model_id = "ekinakyurek/marc-8B-finetuned-llama3"
local_dir = "/workspace/models/marc-8B"

print(f"Downloading {model_id}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ MARC-8B downloaded to {local_dir}")
except Exception as e:
    print(f"⚠️ Error downloading MARC-8B: {e}")
    print("Trying alternative method...")
    os.system(f"huggingface-cli download {model_id} --local-dir {local_dir}")
EOF

echo ""

# ============================================================
# MODEL 2: Qwen3-8B
# Expected: 45% ARC-AGI-1 (Strong reasoning)
# ============================================================
echo "=== Downloading Qwen3-8B ==="
echo "Model: Qwen/Qwen3-8B"
echo "Size: ~16GB"
echo "Expected accuracy: 45%"
echo ""

python3 << 'EOF'
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen3-8B"
local_dir = "/workspace/models/qwen3-8B"

print(f"Downloading {model_id}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ Qwen3-8B downloaded to {local_dir}")
except Exception as e:
    print(f"⚠️ Error: {e}")
EOF

echo ""

# ============================================================
# MODEL 3: DeepSeek-Coder-V2-Lite-Instruct
# Expected: 40% ARC-AGI-1 (Code generation)
# ============================================================
echo "=== Downloading DeepSeek-Coder-V2 ==="
echo "Model: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
echo "Size: ~32GB"
echo "Expected accuracy: 40%"
echo ""

python3 << 'EOF'
from huggingface_hub import snapshot_download

model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
local_dir = "/workspace/models/deepseek-coder-v2"

print(f"Downloading {model_id}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ DeepSeek-Coder-V2 downloaded to {local_dir}")
except Exception as e:
    print(f"⚠️ Error: {e}")
EOF

echo ""

# ============================================================
# MODEL 4: NVARC (NVIDIA ARC-AGI specialized)
# Expected: 55% ARC-AGI-1 (Specialized for ARC)
# ============================================================
echo "=== Downloading NVARC ==="
echo "Model: nvidia/ARC-AGI-Llama-3.1-8B-Instruct"
echo "Size: ~16GB"
echo "Expected accuracy: 55%"
echo ""

python3 << 'EOF'
from huggingface_hub import snapshot_download

# Try multiple possible NVIDIA ARC model names
model_ids = [
    "nvidia/ARC-AGI-Llama-3.1-8B-Instruct",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "nvidia/Mistral-NeMo-12B-Instruct"
]

local_dir = "/workspace/models/nvarc"

for model_id in model_ids:
    print(f"Trying {model_id}...")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ NVARC downloaded to {local_dir}")
        break
    except Exception as e:
        print(f"⚠️ {model_id} not available: {e}")
        continue
EOF

echo ""

# ============================================================
# VERIFICATION
# ============================================================
echo "=== Verifying Downloads ==="
echo ""

for model in marc-8B qwen3-8B deepseek-coder-v2 nvarc; do
    if [ -d "/workspace/models/$model" ]; then
        size=$(du -sh "/workspace/models/$model" 2>/dev/null | cut -f1)
        echo "✅ $model: $size"
    else
        echo "❌ $model: NOT FOUND"
    fi
done

echo ""
echo "=== Disk Usage ==="
df -h /workspace

echo ""
echo "=============================================="
echo "MODEL DOWNLOAD COMPLETE"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Run: python3 run_arc_evaluation.py"
echo "2. Monitor: tail -f evaluation.log"
echo "3. Results: cat arc_agi_results.json"
