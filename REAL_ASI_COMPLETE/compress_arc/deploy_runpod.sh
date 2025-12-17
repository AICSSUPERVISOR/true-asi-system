#!/bin/bash
# CompressARC Training on Runpod
# GPU: RTX 4070 or better (8GB+ VRAM)
# Time: ~20 minutes per task

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm wandb

# Clone ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI

# Run evaluation
python evaluate_compress_arc.py \
    --data_dir ARC-AGI/data/evaluation \
    --output_file compress_arc_results.json \
    --ttt_epochs 1000 \
    --hidden_dim 64 \
    --num_layers 4

# Expected results:
# - ~40-50% on ARC-AGI-1
# - ~20 minutes per task
# - Total: ~130 hours for full evaluation (use parallel GPUs)
