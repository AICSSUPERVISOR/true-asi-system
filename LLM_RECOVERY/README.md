# LLM Recovery System

This directory contains the complete LLM model recovery system, rebuilt from Manus task history.

## Contents

| File/Directory | Description |
|----------------|-------------|
| `FULL_LLM_CATALOG.json` | Complete catalog of 127+ model families |
| `RECOVERY_REPORT.json` | Status of config downloads |
| `download_scripts/` | 125 individual download scripts for each model |
| `model_configs/` | Downloaded config files (tokenizers, configs) |
| `DOWNLOAD_ALL_MODELS.sh` | Master script to download all models |

## Original S3 Data

Based on the S3 comprehensive audit, the following models were stored:

### Largest Model Files (from S3 audit)
| Model | Size |
|-------|------|
| xai-org/grok-2 | 32 GB |
| Salesforce/codegen-16B | 30 GB |
| WizardLM/WizardCoder-15B | 29 GB |
| facebook/incoder-6B | 25 GB |
| EleutherAI/gpt-j-6b | 23 GB |
| meta-llama/Llama-3.1-70B | 50+ GB (sharded) |

### Model Categories
- **Foundational Models**: Llama, Mistral, Gemma, Qwen, DeepSeek
- **Code Models**: CodeLlama, StarCoder, WizardCoder, Codestral
- **Math Reasoning**: Llemma, MAmmoTH, WizardMath, Abel
- **Instruction Tuned**: Vicuna, Guanaco, Orca, Hermes
- **Chat Optimized**: ChatGLM, Baichuan, InternLM, Yi, Falcon
- **Small/Efficient**: Phi-3, TinyLlama, SmolLM
- **Multimodal**: LLaVA, BLIP-2, Qwen-VL
- **Embedding**: MPNet, MiniLM, E5, BGE
- **Medical**: BioGPT, PubMedBERT, BioBERT
- **Multilingual**: BLOOM, JAIS

## How to Download All Models

```bash
# Option 1: Run master script
chmod +x DOWNLOAD_ALL_MODELS.sh
./DOWNLOAD_ALL_MODELS.sh

# Option 2: Download individual models
cd download_scripts
./meta-llama_Llama-3.1-70B-Instruct.sh
```

## Storage Requirements

- **Minimum**: 500 GB (small models only)
- **Recommended**: 2 TB (common models)
- **Full Recovery**: 10+ TB (all models)

## Notes

- Model weights are downloaded from Hugging Face
- Some models require authentication (Llama, Gemma)
- Run `huggingface-cli login` before downloading gated models
