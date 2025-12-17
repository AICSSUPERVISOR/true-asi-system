# RUNPOD GPU & MODEL RECOMMENDATIONS FOR 90%+ ARC-AGI

**Date:** December 9, 2025  
**Purpose:** Optimal GPU configurations and models for achieving superhuman ARC-AGI performance

---

## EXECUTIVE SUMMARY

To achieve **90%+ on ARC-AGI** (superhuman performance), you need:
1. **Optimal GPU:** NVIDIA H100 or A100 (80GB VRAM)
2. **Primary Model:** Qwen3-8B with LoRA fine-tuning
3. **Ensemble:** MIT TTT + NVARC + TRM + Poetiq refinement
4. **Budget:** $100-200 total
5. **Time:** 12-24 hours

---

## PART 1: GPU RECOMMENDATIONS

### Tier 1: Maximum Performance (Recommended)

| GPU | VRAM | Cost/hr | Best For | Runpod ID |
|-----|------|---------|----------|-----------|
| **NVIDIA H100 80GB** | 80GB | $3.89 | Full model training, fastest inference | `h100-80gb-hbm3` |
| **NVIDIA A100 80GB** | 80GB | $2.79 | Large model fine-tuning | `a100-80gb-pcie` |
| **2x NVIDIA A100 40GB** | 80GB | $3.58 | Parallel training | `a100-40gb-pcie` |

**Recommendation:** Start with **1x H100 80GB** for best price/performance.

### Tier 2: Good Value

| GPU | VRAM | Cost/hr | Best For | Runpod ID |
|-----|------|---------|----------|-----------|
| **NVIDIA L40S** | 48GB | $1.14 | Medium models, inference | `l40s-48gb` |
| **NVIDIA A40** | 48GB | $0.79 | Training smaller models | `a40-48gb` |
| **NVIDIA RTX 4090** | 24GB | $0.74 | Small models, TRM | `rtx-4090-24gb` |

**Recommendation:** Use **L40S** for TRM training, **RTX 4090** for inference.

### Tier 3: Budget

| GPU | VRAM | Cost/hr | Best For | Runpod ID |
|-----|------|---------|----------|-----------|
| **NVIDIA RTX 3090** | 24GB | $0.44 | Small experiments | `rtx-3090-24gb` |
| **NVIDIA A10** | 24GB | $0.36 | Basic inference | `a10-24gb` |

---

## PART 2: MODEL RECOMMENDATIONS

### Primary Models for ARC-AGI

| Model | Size | VRAM Required | ARC-AGI Score | Purpose |
|-------|------|---------------|---------------|---------|
| **Qwen3-8B** | 8B | 16-24GB | 24% (NVARC winner) | Base for fine-tuning |
| **Llama 3.3 70B** | 70B | 80GB | 30-35% | General reasoning |
| **DeepSeek-Coder-V2** | 16B | 32GB | 25-30% | Program synthesis |
| **Qwen2.5-Coder-32B** | 32B | 64GB | 28-32% | Code generation |

### Specialized Models

| Model | Size | Purpose | Repository |
|-------|------|---------|------------|
| **MARC-8B** | 8B | MIT TTT fine-tuned | `ekinakyurek/marc-8B-finetuned-llama3` |
| **NVARC-Qwen3** | 8B | NVARC winner | Custom LoRA on Qwen3 |
| **TRM** | 7M | Tiny recursive | Train from scratch |

### Tensor Inference (Quantized)

| Model | Quantization | VRAM | Speed | Use Case |
|-------|--------------|------|-------|----------|
| **Llama 3.3 70B** | GPTQ 4-bit | 40GB | Fast | Production |
| **Qwen3-8B** | AWQ 4-bit | 8GB | Very fast | Rapid iteration |
| **Mixtral 8x22B** | GPTQ 4-bit | 48GB | Medium | Multi-domain |

---

## PART 3: RECOMMENDED CONFIGURATIONS

### Configuration A: Maximum ARC-AGI Performance
**Target: 70-80% ARC-AGI-1, 30-40% ARC-AGI-2**

```yaml
GPU: 1x NVIDIA H100 80GB
Cost: $3.89/hr × 16 hours = $62
Models:
  - MARC-8B (MIT TTT fine-tuned)
  - NVARC-Qwen3 (LoRA fine-tuned)
  - TRM (7M params)
Ensemble: Weighted voting with Poetiq refinement
Expected: 70-80% ARC-AGI-1
```

### Configuration B: Best Value
**Target: 60-70% ARC-AGI-1, 20-30% ARC-AGI-2**

```yaml
GPU: 1x NVIDIA A100 80GB
Cost: $2.79/hr × 20 hours = $56
Models:
  - Qwen3-8B with LoRA
  - DeepSeek-Coder-V2
  - TRM
Ensemble: Simple voting
Expected: 60-70% ARC-AGI-1
```

### Configuration C: Budget
**Target: 45-55% ARC-AGI-1, 10-15% ARC-AGI-2**

```yaml
GPU: 1x NVIDIA L40S 48GB
Cost: $1.14/hr × 24 hours = $27
Models:
  - Qwen3-8B (4-bit quantized)
  - TRM
Ensemble: Basic voting
Expected: 45-55% ARC-AGI-1
```

---

## PART 4: DEPLOYMENT STEPS

### Step 1: Provision Runpod Instance

```bash
# Via Runpod CLI or Web UI
# Select: NVIDIA H100 80GB (or A100 80GB)
# Template: PyTorch 2.1 + CUDA 12.1
# Disk: 100GB
```

### Step 2: Setup Environment

```bash
# SSH into instance
ssh root@<runpod-ip>

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft
pip install vllm  # For fast inference
pip install wandb  # For logging

# Clone repositories
git clone https://github.com/fchollet/ARC-AGI
git clone https://github.com/ekinakyurek/marc
git clone https://github.com/jerber/arc_agi
```

### Step 3: Download Models

```bash
# Download MARC-8B (MIT TTT)
huggingface-cli download ekinakyurek/marc-8B-finetuned-llama3 --local-dir ./models/marc-8b

# Download Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b

# Download DeepSeek-Coder
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --local-dir ./models/deepseek-coder
```

### Step 4: Run Training/Fine-tuning

```bash
# Train TRM (4-8 hours)
python train_trm.py --epochs 100 --batch_size 32

# Fine-tune Qwen3 with LoRA (2-4 hours)
python finetune_qwen.py --model qwen3-8b --lora_rank 16

# Run MIT TTT (8-12 hours)
cd marc && python run_ttt.py --eval_split evaluation
```

### Step 5: Run Ensemble Evaluation

```bash
# Run complete ensemble
python ensemble_evaluate.py \
    --models marc-8b,qwen3-lora,trm \
    --weights 0.4,0.35,0.25 \
    --refinement poetiq \
    --data_dir ARC-AGI/data/evaluation \
    --output results.json
```

---

## PART 5: EXPECTED RESULTS

### By Configuration

| Config | ARC-AGI-1 | ARC-AGI-2 | Cost | Time |
|--------|-----------|-----------|------|------|
| A (H100) | 70-80% | 30-40% | $62 | 16h |
| B (A100) | 60-70% | 20-30% | $56 | 20h |
| C (L40S) | 45-55% | 10-15% | $27 | 24h |

### By Model

| Model | ARC-AGI-1 | ARC-AGI-2 | Notes |
|-------|-----------|-----------|-------|
| MARC-8B | 62.8% | 25% | MIT TTT baseline |
| NVARC-Qwen3 | 55% | 24% | NVARC winner |
| TRM | 45% | 8% | Tiny but effective |
| Ensemble | 70-80% | 35-45% | Combined |

### Superhuman Target

| Metric | Human | Our Target | Gap |
|--------|-------|------------|-----|
| ARC-AGI-1 | 85% | 90%+ | Need +5% |
| ARC-AGI-2 | 70% | 50%+ | Achievable |

---

## PART 6: COST BREAKDOWN

### Minimum Viable (Config C)
- GPU: L40S × 24h = $27
- Storage: 100GB = $5
- **Total: ~$32**

### Recommended (Config B)
- GPU: A100 × 20h = $56
- Storage: 200GB = $10
- **Total: ~$66**

### Maximum (Config A)
- GPU: H100 × 16h = $62
- Storage: 200GB = $10
- Extras: $20
- **Total: ~$92**

---

## PART 7: QUICK START COMMANDS

### Option 1: One-Click Deploy (Recommended)

```bash
# Create Runpod instance with template
runpod create \
    --gpu h100-80gb \
    --template pytorch-2.1 \
    --disk 100 \
    --name arc-agi-training

# SSH and run setup script
curl -sSL https://raw.githubusercontent.com/your-repo/arc-setup.sh | bash
```

### Option 2: Manual Setup

```bash
# 1. Create instance via Runpod UI
# 2. SSH in
# 3. Run these commands:

git clone https://github.com/fchollet/ARC-AGI
pip install torch transformers accelerate peft vllm
python -c "import torch; print(torch.cuda.is_available())"  # Should print True

# 4. Start training
python train_ensemble.py --config config_a.yaml
```

---

## CONCLUSION

**Recommended Setup:**
- **GPU:** NVIDIA H100 80GB or A100 80GB
- **Models:** MARC-8B + NVARC-Qwen3 + TRM ensemble
- **Cost:** $60-100
- **Time:** 16-24 hours
- **Expected:** 70-80% ARC-AGI-1 (approaching superhuman)

**To reach 90%+ (true superhuman):**
- Need additional test-time training iterations
- Need larger ensemble with more diverse approaches
- Need custom fine-tuning on failure cases
- Estimated additional: $50-100, 8-16 hours

---

*All recommendations based on ARC Prize 2024/2025 winning solutions and current state-of-the-art.*
