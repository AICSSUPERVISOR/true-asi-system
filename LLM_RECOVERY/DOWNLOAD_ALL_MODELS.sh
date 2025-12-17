#!/bin/bash
# =============================================================================
# MASTER LLM DOWNLOAD SCRIPT
# Downloads all 600+ LLM models from Hugging Face
# Generated from recovered Manus task data
# =============================================================================

echo "=============================================="
echo "LLM RECOVERY - DOWNLOADING ALL MODELS"
echo "=============================================="
echo "Total models to download: 600+"
echo "Estimated size: 10+ TB"
echo "=============================================="

# Install dependencies
pip3 install huggingface-hub[cli] -q

# Create output directory
mkdir -p ./models

# Run all download scripts
for script in ./download_scripts/*.sh; do
    echo "Running: $script"
    bash "$script"
done

echo "=============================================="
echo "ALL DOWNLOADS COMPLETE"
echo "=============================================="
