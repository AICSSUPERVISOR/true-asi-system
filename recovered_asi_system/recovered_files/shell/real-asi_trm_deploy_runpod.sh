#!/bin/bash
# TRM Training on Runpod
# GPU: NVIDIA L40S or RTX 4090 (24GB VRAM sufficient)

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets wandb

# Clone TRM repository (if available)
# git clone https://github.com/alexia-jolicoeur-martineau/trm

# Download ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI

# Run training
python train_trm.py \
    --data_dir ARC-AGI/data/training \
    --output_dir ./trm_checkpoints \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --latent_dim 128 \
    --max_steps 16 \
    --recursive_updates 8

# Evaluate
python evaluate_trm.py \
    --model_path ./trm_checkpoints/best.pt \
    --data_dir ARC-AGI/data/evaluation \
    --output_file results.json
