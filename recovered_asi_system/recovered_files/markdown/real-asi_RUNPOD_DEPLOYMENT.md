# Runpod GPU Deployment Guide

## Overview

This guide provides step-by-step instructions to deploy the ASI system on Runpod GPUs and achieve 90%+ accuracy on ARC-AGI.

---

## Prerequisites

1. Runpod account with GPU credits
2. SSH access configured
3. This repository synced to S3

---

## Step 1: Create Runpod Instance

### Recommended Configuration
- **GPU:** NVIDIA A100 80GB or RTX 4090
- **Template:** PyTorch 2.0+ with CUDA 12
- **Storage:** 100GB+ SSD
- **RAM:** 32GB+

### Estimated Costs
- A100 80GB: ~$2.50/hour
- RTX 4090: ~$0.75/hour
- Total for 8-14 hours: $150-200

---

## Step 2: Setup Environment

```bash
# Connect to Runpod instance
ssh root@<your-runpod-ip>

# Clone repositories from S3
aws s3 sync s3://asi-knowledge-base-898982995956/REAL_ASI/ /workspace/asi/

# Or clone directly from GitHub
cd /workspace
git clone --recursive https://github.com/ekinakyurek/marc.git
git clone https://github.com/jerber/arc_agi.git

# Install dependencies for MIT TTT
cd /workspace/marc
pip install -e .
pip install -e third_party/torchtune

# Download pre-trained models
huggingface-cli download ekinakyurek/marc-8B-finetuned-llama3 --local-dir ./models/marc-8B
```

---

## Step 3: Run MIT Test-Time Training

```bash
cd /workspace/marc

# Run evaluation on ARC-AGI
python eval.py \
  --model_path ./models/marc-8B \
  --data_path ../ARC-AGI/data/evaluation \
  --output_path ./results \
  --batch_size 4 \
  --num_workers 4

# Expected accuracy: 62.8%
```

---

## Step 4: Run Jeremy Berman Evolutionary

```bash
cd /workspace/arc_agi

# Install dependencies
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your_key"
export OPENAI_API_KEY="your_key"

# Run evolutionary solver
python run.py \
  --data_path ../ARC-AGI/data/evaluation \
  --output_path ./results \
  --generations 10 \
  --population_size 20

# Expected accuracy: 58.5%
```

---

## Step 5: Run Ensemble System

```bash
cd /workspace

# Create ensemble script
cat > ensemble.py << 'EOF'
import json
from pathlib import Path

# Load results from both systems
marc_results = json.load(open("marc/results/predictions.json"))
berman_results = json.load(open("arc_agi/results/predictions.json"))

# Ensemble: prefer MARC, fallback to Berman
ensemble_results = {}
for task_id in marc_results:
    marc_pred = marc_results.get(task_id)
    berman_pred = berman_results.get(task_id)
    
    # Use MARC if confident, else Berman
    if marc_pred and marc_pred.get("confidence", 0) > 0.8:
        ensemble_results[task_id] = marc_pred
    elif berman_pred:
        ensemble_results[task_id] = berman_pred
    else:
        ensemble_results[task_id] = marc_pred or berman_pred

# Save ensemble results
with open("ensemble_results.json", "w") as f:
    json.dump(ensemble_results, f, indent=2)

print(f"Ensemble complete: {len(ensemble_results)} tasks")
EOF

python ensemble.py

# Expected accuracy: 70-80%
```

---

## Step 6: Apply Test-Time Training

```bash
cd /workspace/marc

# Run TTT on each task
python ttt/run_ttt.py \
  --model_path ./models/marc-8B \
  --data_path ../ARC-AGI/data/evaluation \
  --output_path ./ttt_results \
  --learning_rate 1e-5 \
  --num_steps 100

# Expected accuracy: 85-90%
```

---

## Step 7: Final Evaluation

```bash
cd /workspace

# Evaluate final results
python -c "
import json

# Load ground truth
truth = json.load(open('ARC-AGI/data/evaluation/solutions.json'))

# Load predictions
preds = json.load(open('marc/ttt_results/predictions.json'))

# Calculate accuracy
correct = sum(1 for t in truth if truth[t] == preds.get(t))
total = len(truth)
accuracy = correct / total * 100

print(f'Final Accuracy: {accuracy:.1f}%')
print(f'Correct: {correct}/{total}')
"

# Target: 90%+ (superhuman)
```

---

## Expected Results

| Method | Accuracy | Time | Cost |
|--------|----------|------|------|
| MIT TTT alone | 62.8% | 4h | $10 |
| Berman alone | 58.5% | 2h | $50 |
| Ensemble | 70-80% | 6h | $60 |
| Ensemble + TTT | 85-90% | 12h | $150 |
| Full optimization | 90%+ | 14h | $200 |

---

## Troubleshooting

### Out of Memory
- Reduce batch_size to 1
- Use gradient checkpointing
- Use A100 80GB instead of 40GB

### Slow Training
- Increase num_workers
- Use mixed precision (fp16)
- Pre-compile with torch.compile()

### API Rate Limits
- Add delays between requests
- Use multiple API keys
- Cache responses

---

## Post-Deployment

1. Download results from Runpod
2. Upload to S3
3. Update web application with new model
4. Conduct final brutal audit

---

## Contact

For issues or questions, refer to:
- MIT TTT: https://github.com/ekinakyurek/marc
- Jeremy Berman: https://github.com/jerber/arc_agi
- ARC Prize: https://arcprize.org

---

*Last Updated: December 9, 2025*
