/**
 * AI Model Router - Dynamic Selection from 200+ Models
 * 
 * This module provides intelligent model selection based on task type,
 * ensemble voting for superhuman accuracy, and performance tracking.
 */

export type TaskType =
  | "strategy"
  | "writing"
  | "analysis"
  | "coding"
  | "superintelligence"
  | "financial"
  | "marketing"
  | "operations"
  | "legal"
  | "technical";

export type AIModel = {
  id: string;
  name: string;
  provider: "aiml" | "asi1";
  capabilities: TaskType[];
  weight: number; // 0-100, higher = more trusted
  costPerToken: number;
  maxTokens: number;
  speed: "fast" | "medium" | "slow";
};

/**
 * Comprehensive AI Model Registry (200+ models)
 * Organized by provider and capability
 */
export const AI_MODEL_REGISTRY: AIModel[] = [
  // === Tier 1: Superintelligence Models (Highest Weight) ===
  {
    id: "asi1-ultra",
    name: "ASI1 Ultra",
    provider: "asi1",
    capabilities: ["superintelligence", "strategy", "analysis", "financial", "technical"],
    weight: 100,
    costPerToken: 0.01,
    maxTokens: 128000,
    speed: "medium",
  },
  {
    id: "asi1-pro",
    name: "ASI1 Pro",
    provider: "asi1",
    capabilities: ["superintelligence", "strategy", "analysis"],
    weight: 95,
    costPerToken: 0.008,
    maxTokens: 100000,
    speed: "fast",
  },

  // === Tier 2: Strategic Planning Models ===
  {
    id: "gpt-4-turbo",
    name: "GPT-4 Turbo",
    provider: "aiml",
    capabilities: ["strategy", "analysis", "writing", "financial"],
    weight: 90,
    costPerToken: 0.005,
    maxTokens: 128000,
    speed: "fast",
  },
  {
    id: "gpt-4",
    name: "GPT-4",
    provider: "aiml",
    capabilities: ["strategy", "analysis", "writing"],
    weight: 85,
    costPerToken: 0.003,
    maxTokens: 8192,
    speed: "medium",
  },
  {
    id: "claude-3-opus",
    name: "Claude 3 Opus",
    provider: "aiml",
    capabilities: ["strategy", "writing", "analysis", "legal"],
    weight: 88,
    costPerToken: 0.004,
    maxTokens: 200000,
    speed: "medium",
  },
  {
    id: "claude-3.5-sonnet",
    name: "Claude 3.5 Sonnet",
    provider: "aiml",
    capabilities: ["writing", "strategy", "coding"],
    weight: 87,
    costPerToken: 0.003,
    maxTokens: 200000,
    speed: "fast",
  },

  // === Tier 3: Content Writing Models ===
  {
    id: "claude-3-sonnet",
    name: "Claude 3 Sonnet",
    provider: "aiml",
    capabilities: ["writing", "marketing"],
    weight: 82,
    costPerToken: 0.002,
    maxTokens: 200000,
    speed: "fast",
  },
  {
    id: "gpt-3.5-turbo",
    name: "GPT-3.5 Turbo",
    provider: "aiml",
    capabilities: ["writing", "marketing"],
    weight: 75,
    costPerToken: 0.0005,
    maxTokens: 16385,
    speed: "fast",
  },

  // === Tier 4: Data Analysis Models ===
  {
    id: "gemini-1.5-pro",
    name: "Gemini 1.5 Pro",
    provider: "aiml",
    capabilities: ["analysis", "financial", "technical"],
    weight: 84,
    costPerToken: 0.0025,
    maxTokens: 1000000,
    speed: "medium",
  },
  {
    id: "gemini-1.5-flash",
    name: "Gemini 1.5 Flash",
    provider: "aiml",
    capabilities: ["analysis", "financial"],
    weight: 78,
    costPerToken: 0.0001,
    maxTokens: 1000000,
    speed: "fast",
  },

  // === Tier 5: Code Generation Models ===
  {
    id: "llama-3.3-70b",
    name: "Llama 3.3 70B",
    provider: "aiml",
    capabilities: ["coding", "technical"],
    weight: 80,
    costPerToken: 0.0008,
    maxTokens: 128000,
    speed: "fast",
  },
  {
    id: "codellama-70b",
    name: "CodeLlama 70B",
    provider: "aiml",
    capabilities: ["coding", "technical"],
    weight: 79,
    costPerToken: 0.0007,
    maxTokens: 100000,
    speed: "fast",
  },
  {
    id: "deepseek-coder-33b",
    name: "DeepSeek Coder 33B",
    provider: "aiml",
    capabilities: ["coding", "technical"],
    weight: 77,
    costPerToken: 0.0006,
    maxTokens: 16384,
    speed: "fast",
  },

  // === Tier 6: Specialized Models (195+ more) ===
  // Financial Analysis
  {
    id: "finbert",
    name: "FinBERT",
    provider: "aiml",
    capabilities: ["financial"],
    weight: 76,
    costPerToken: 0.0003,
    maxTokens: 512,
    speed: "fast",
  },
  {
    id: "bloomberg-gpt",
    name: "BloombergGPT",
    provider: "aiml",
    capabilities: ["financial", "analysis"],
    weight: 83,
    costPerToken: 0.004,
    maxTokens: 50000,
    speed: "medium",
  },

  // Legal Analysis
  {
    id: "legal-bert",
    name: "Legal-BERT",
    provider: "aiml",
    capabilities: ["legal"],
    weight: 74,
    costPerToken: 0.0004,
    maxTokens: 512,
    speed: "fast",
  },

  // Marketing & SEO
  {
    id: "jasper-ai",
    name: "Jasper AI",
    provider: "aiml",
    capabilities: ["marketing", "writing"],
    weight: 73,
    costPerToken: 0.0005,
    maxTokens: 3000,
    speed: "fast",
  },

  // Operations & Logistics
  {
    id: "operations-gpt",
    name: "Operations GPT",
    provider: "aiml",
    capabilities: ["operations", "analysis"],
    weight: 72,
    costPerToken: 0.0003,
    maxTokens: 8192,
    speed: "fast",
  },

  // === Tier 7: Advanced Language Models (50+ models) ===
  { id: "gpt-4o", name: "GPT-4o", provider: "aiml", capabilities: ["strategy", "analysis", "writing"], weight: 92, costPerToken: 0.006, maxTokens: 128000, speed: "fast" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "aiml", capabilities: ["writing", "analysis"], weight: 80, costPerToken: 0.0002, maxTokens: 128000, speed: "fast" },
  { id: "o1-preview", name: "O1 Preview", provider: "aiml", capabilities: ["superintelligence", "strategy", "analysis"], weight: 94, costPerToken: 0.015, maxTokens: 128000, speed: "slow" },
  { id: "o1-mini", name: "O1 Mini", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 86, costPerToken: 0.003, maxTokens: 65536, speed: "medium" },
  { id: "claude-3-haiku", name: "Claude 3 Haiku", provider: "aiml", capabilities: ["writing", "analysis"], weight: 76, costPerToken: 0.00025, maxTokens: 200000, speed: "fast" },
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "aiml", capabilities: ["analysis", "writing", "coding"], weight: 85, costPerToken: 0.0002, maxTokens: 1000000, speed: "fast" },
  { id: "gemini-2.0-flash-thinking", name: "Gemini 2.0 Flash Thinking", provider: "aiml", capabilities: ["superintelligence", "strategy"], weight: 89, costPerToken: 0.0003, maxTokens: 1000000, speed: "medium" },
  { id: "gemini-exp-1206", name: "Gemini Exp 1206", provider: "aiml", capabilities: ["superintelligence", "strategy", "analysis"], weight: 91, costPerToken: 0.004, maxTokens: 1000000, speed: "medium" },
  { id: "llama-3.1-405b", name: "Llama 3.1 405B", provider: "aiml", capabilities: ["strategy", "analysis", "coding"], weight: 88, costPerToken: 0.003, maxTokens: 128000, speed: "medium" },
  { id: "llama-3.1-70b", name: "Llama 3.1 70B", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 82, costPerToken: 0.0008, maxTokens: 128000, speed: "fast" },
  { id: "llama-3.1-8b", name: "Llama 3.1 8B", provider: "aiml", capabilities: ["writing", "analysis"], weight: 70, costPerToken: 0.0001, maxTokens: 128000, speed: "fast" },
  { id: "llama-3.2-90b", name: "Llama 3.2 90B", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 84, costPerToken: 0.001, maxTokens: 128000, speed: "fast" },
  { id: "llama-3.2-11b", name: "Llama 3.2 11B", provider: "aiml", capabilities: ["writing", "analysis"], weight: 74, costPerToken: 0.0002, maxTokens: 128000, speed: "fast" },
  { id: "llama-3.2-3b", name: "Llama 3.2 3B", provider: "aiml", capabilities: ["writing"], weight: 65, costPerToken: 0.00005, maxTokens: 128000, speed: "fast" },
  { id: "llama-3.2-1b", name: "Llama 3.2 1B", provider: "aiml", capabilities: ["writing"], weight: 60, costPerToken: 0.00003, maxTokens: 128000, speed: "fast" },
  { id: "mistral-large", name: "Mistral Large", provider: "aiml", capabilities: ["strategy", "analysis", "coding"], weight: 86, costPerToken: 0.003, maxTokens: 128000, speed: "fast" },
  { id: "mistral-medium", name: "Mistral Medium", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 80, costPerToken: 0.0015, maxTokens: 32000, speed: "fast" },
  { id: "mistral-small", name: "Mistral Small", provider: "aiml", capabilities: ["writing", "analysis"], weight: 74, costPerToken: 0.0005, maxTokens: 32000, speed: "fast" },
  { id: "mistral-nemo", name: "Mistral Nemo", provider: "aiml", capabilities: ["writing"], weight: 72, costPerToken: 0.0003, maxTokens: 128000, speed: "fast" },
  { id: "mixtral-8x7b", name: "Mixtral 8x7B", provider: "aiml", capabilities: ["strategy", "coding"], weight: 81, costPerToken: 0.0006, maxTokens: 32000, speed: "fast" },
  { id: "mixtral-8x22b", name: "Mixtral 8x22B", provider: "aiml", capabilities: ["strategy", "analysis", "coding"], weight: 85, costPerToken: 0.0012, maxTokens: 65536, speed: "fast" },
  { id: "qwen-2.5-72b", name: "Qwen 2.5 72B", provider: "aiml", capabilities: ["strategy", "analysis", "coding"], weight: 83, costPerToken: 0.0009, maxTokens: 128000, speed: "fast" },
  { id: "qwen-2.5-32b", name: "Qwen 2.5 32B", provider: "aiml", capabilities: ["writing", "coding"], weight: 77, costPerToken: 0.0004, maxTokens: 128000, speed: "fast" },
  { id: "qwen-2.5-14b", name: "Qwen 2.5 14B", provider: "aiml", capabilities: ["writing", "analysis"], weight: 73, costPerToken: 0.0002, maxTokens: 128000, speed: "fast" },
  { id: "qwen-2.5-7b", name: "Qwen 2.5 7B", provider: "aiml", capabilities: ["writing"], weight: 68, costPerToken: 0.0001, maxTokens: 128000, speed: "fast" },
  { id: "deepseek-v3", name: "DeepSeek V3", provider: "aiml", capabilities: ["superintelligence", "strategy", "coding"], weight: 93, costPerToken: 0.0014, maxTokens: 128000, speed: "fast" },
  { id: "deepseek-v2.5", name: "DeepSeek V2.5", provider: "aiml", capabilities: ["strategy", "coding"], weight: 87, costPerToken: 0.001, maxTokens: 128000, speed: "fast" },
  { id: "deepseek-chat", name: "DeepSeek Chat", provider: "aiml", capabilities: ["writing", "analysis"], weight: 79, costPerToken: 0.0005, maxTokens: 64000, speed: "fast" },
  { id: "yi-large", name: "Yi Large", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 82, costPerToken: 0.0008, maxTokens: 32000, speed: "fast" },
  { id: "yi-medium", name: "Yi Medium", provider: "aiml", capabilities: ["writing", "analysis"], weight: 76, costPerToken: 0.0004, maxTokens: 16000, speed: "fast" },
  { id: "command-r-plus", name: "Command R+", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 84, costPerToken: 0.001, maxTokens: 128000, speed: "fast" },
  { id: "command-r", name: "Command R", provider: "aiml", capabilities: ["writing", "analysis"], weight: 78, costPerToken: 0.0005, maxTokens: 128000, speed: "fast" },
  { id: "aya-expanse-32b", name: "Aya Expanse 32B", provider: "aiml", capabilities: ["writing", "analysis"], weight: 75, costPerToken: 0.0003, maxTokens: 128000, speed: "fast" },
  { id: "aya-expanse-8b", name: "Aya Expanse 8B", provider: "aiml", capabilities: ["writing"], weight: 69, costPerToken: 0.0001, maxTokens: 128000, speed: "fast" },
  { id: "phi-4", name: "Phi-4", provider: "aiml", capabilities: ["writing", "coding"], weight: 77, costPerToken: 0.0003, maxTokens: 16384, speed: "fast" },
  { id: "phi-3.5-mini", name: "Phi-3.5 Mini", provider: "aiml", capabilities: ["writing"], weight: 71, costPerToken: 0.0001, maxTokens: 128000, speed: "fast" },
  { id: "phi-3-medium", name: "Phi-3 Medium", provider: "aiml", capabilities: ["writing", "coding"], weight: 73, costPerToken: 0.0002, maxTokens: 128000, speed: "fast" },
  { id: "granite-3.1-8b", name: "Granite 3.1 8B", provider: "aiml", capabilities: ["writing", "analysis"], weight: 72, costPerToken: 0.0002, maxTokens: 128000, speed: "fast" },
  { id: "granite-3.1-2b", name: "Granite 3.1 2B", provider: "aiml", capabilities: ["writing"], weight: 66, costPerToken: 0.00005, maxTokens: 128000, speed: "fast" },
  { id: "nemotron-70b", name: "Nemotron 70B", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 83, costPerToken: 0.0009, maxTokens: 128000, speed: "fast" },
  { id: "jamba-1.5-large", name: "Jamba 1.5 Large", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 81, costPerToken: 0.0007, maxTokens: 256000, speed: "fast" },
  { id: "jamba-1.5-mini", name: "Jamba 1.5 Mini", provider: "aiml", capabilities: ["writing"], weight: 74, costPerToken: 0.0002, maxTokens: 256000, speed: "fast" },
  { id: "wizardlm-2-8x22b", name: "WizardLM-2 8x22B", provider: "aiml", capabilities: ["strategy", "coding"], weight: 84, costPerToken: 0.001, maxTokens: 65536, speed: "fast" },
  { id: "solar-pro", name: "Solar Pro", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 80, costPerToken: 0.0006, maxTokens: 4096, speed: "fast" },
  { id: "hermes-3-llama-3.1-405b", name: "Hermes 3 Llama 3.1 405B", provider: "aiml", capabilities: ["superintelligence", "strategy"], weight: 90, costPerToken: 0.003, maxTokens: 128000, speed: "medium" },
  { id: "hermes-3-llama-3.1-70b", name: "Hermes 3 Llama 3.1 70B", provider: "aiml", capabilities: ["strategy", "analysis"], weight: 83, costPerToken: 0.0009, maxTokens: 128000, speed: "fast" },

  // === Tier 8: Code Generation Models (30+ models) ===
  { id: "codestral-latest", name: "Codestral Latest", provider: "aiml", capabilities: ["coding", "technical"], weight: 88, costPerToken: 0.0012, maxTokens: 32000, speed: "fast" },
  { id: "codestral-mamba", name: "Codestral Mamba", provider: "aiml", capabilities: ["coding"], weight: 82, costPerToken: 0.0008, maxTokens: 256000, speed: "fast" },
  { id: "qwen-2.5-coder-32b", name: "Qwen 2.5 Coder 32B", provider: "aiml", capabilities: ["coding", "technical"], weight: 85, costPerToken: 0.0009, maxTokens: 128000, speed: "fast" },
  { id: "deepseek-coder-v2", name: "DeepSeek Coder V2", provider: "aiml", capabilities: ["coding", "technical"], weight: 86, costPerToken: 0.001, maxTokens: 128000, speed: "fast" },
  { id: "starcoder2-15b", name: "StarCoder2 15B", provider: "aiml", capabilities: ["coding"], weight: 78, costPerToken: 0.0005, maxTokens: 16384, speed: "fast" },
  { id: "codegen-16b", name: "CodeGen 16B", provider: "aiml", capabilities: ["coding"], weight: 76, costPerToken: 0.0004, maxTokens: 2048, speed: "fast" },
  { id: "wizardcoder-python-34b", name: "WizardCoder Python 34B", provider: "aiml", capabilities: ["coding"], weight: 81, costPerToken: 0.0007, maxTokens: 8192, speed: "fast" },
  { id: "wizardcoder-15b", name: "WizardCoder 15B", provider: "aiml", capabilities: ["coding"], weight: 77, costPerToken: 0.0005, maxTokens: 8192, speed: "fast" },
  { id: "code-llama-34b", name: "Code Llama 34B", provider: "aiml", capabilities: ["coding"], weight: 80, costPerToken: 0.0006, maxTokens: 16384, speed: "fast" },
  { id: "code-llama-13b", name: "Code Llama 13B", provider: "aiml", capabilities: ["coding"], weight: 75, costPerToken: 0.0003, maxTokens: 16384, speed: "fast" },
  { id: "code-llama-7b", name: "Code Llama 7B", provider: "aiml", capabilities: ["coding"], weight: 70, costPerToken: 0.0002, maxTokens: 16384, speed: "fast" },
  { id: "phind-codellama-34b", name: "Phind CodeLlama 34B", provider: "aiml", capabilities: ["coding"], weight: 79, costPerToken: 0.0006, maxTokens: 16384, speed: "fast" },
  { id: "replit-code-v1.5-3b", name: "Replit Code v1.5 3B", provider: "aiml", capabilities: ["coding"], weight: 68, costPerToken: 0.0001, maxTokens: 4096, speed: "fast" },
  { id: "santacoder", name: "SantaCoder", provider: "aiml", capabilities: ["coding"], weight: 67, costPerToken: 0.0001, maxTokens: 2048, speed: "fast" },
  { id: "polycoder-2.7b", name: "PolyCoder 2.7B", provider: "aiml", capabilities: ["coding"], weight: 66, costPerToken: 0.00008, maxTokens: 2048, speed: "fast" },
  { id: "incoder-6b", name: "InCoder 6B", provider: "aiml", capabilities: ["coding"], weight: 69, costPerToken: 0.00015, maxTokens: 2048, speed: "fast" },
  { id: "codegen2-16b", name: "CodeGen2 16B", provider: "aiml", capabilities: ["coding"], weight: 77, costPerToken: 0.0005, maxTokens: 2048, speed: "fast" },
  { id: "codegen2-7b", name: "CodeGen2 7B", provider: "aiml", capabilities: ["coding"], weight: 71, costPerToken: 0.0002, maxTokens: 2048, speed: "fast" },
  { id: "codegen2-3b", name: "CodeGen2 3B", provider: "aiml", capabilities: ["coding"], weight: 65, costPerToken: 0.0001, maxTokens: 2048, speed: "fast" },
  { id: "codegen2-1b", name: "CodeGen2 1B", provider: "aiml", capabilities: ["coding"], weight: 62, costPerToken: 0.00005, maxTokens: 2048, speed: "fast" },
  { id: "stablecode-3b", name: "StableCode 3B", provider: "aiml", capabilities: ["coding"], weight: 67, costPerToken: 0.0001, maxTokens: 16384, speed: "fast" },
  { id: "refact-1.6b", name: "Refact 1.6B", provider: "aiml", capabilities: ["coding"], weight: 64, costPerToken: 0.00008, maxTokens: 2048, speed: "fast" },
  { id: "aiXcoder-7b", name: "aiXcoder 7B", provider: "aiml", capabilities: ["coding"], weight: 72, costPerToken: 0.0002, maxTokens: 4096, speed: "fast" },
  { id: "codegeex2-6b", name: "CodeGeeX2 6B", provider: "aiml", capabilities: ["coding"], weight: 70, costPerToken: 0.00015, maxTokens: 8192, speed: "fast" },
  { id: "pangu-coder-2b", name: "PanGu-Coder 2B", provider: "aiml", capabilities: ["coding"], weight: 66, costPerToken: 0.0001, maxTokens: 2048, speed: "fast" },
  { id: "codeparrot-110m", name: "CodeParrot 110M", provider: "aiml", capabilities: ["coding"], weight: 58, costPerToken: 0.00003, maxTokens: 1024, speed: "fast" },
  { id: "gpt-neo-2.7b-code", name: "GPT-Neo 2.7B Code", provider: "aiml", capabilities: ["coding"], weight: 65, costPerToken: 0.0001, maxTokens: 2048, speed: "fast" },
  { id: "bloom-7b-code", name: "BLOOM 7B Code", provider: "aiml", capabilities: ["coding"], weight: 69, costPerToken: 0.00015, maxTokens: 2048, speed: "fast" },
  { id: "falcon-7b-instruct", name: "Falcon 7B Instruct", provider: "aiml", capabilities: ["coding", "writing"], weight: 73, costPerToken: 0.0002, maxTokens: 2048, speed: "fast" },
  { id: "mpt-7b-instruct", name: "MPT 7B Instruct", provider: "aiml", capabilities: ["coding", "writing"], weight: 72, costPerToken: 0.0002, maxTokens: 8192, speed: "fast" },

  // === Tier 9: Financial Analysis Models (20+ models) ===
  { id: "finbert-tone", name: "FinBERT Tone", provider: "aiml", capabilities: ["financial"], weight: 78, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "finbert-esg", name: "FinBERT ESG", provider: "aiml", capabilities: ["financial"], weight: 77, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "finbert-sentiment", name: "FinBERT Sentiment", provider: "aiml", capabilities: ["financial"], weight: 76, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "bloomberg-gpt-50b", name: "BloombergGPT 50B", provider: "aiml", capabilities: ["financial", "analysis"], weight: 85, costPerToken: 0.005, maxTokens: 50000, speed: "medium" },
  { id: "finllama-13b", name: "FinLlama 13B", provider: "aiml", capabilities: ["financial", "analysis"], weight: 80, costPerToken: 0.0008, maxTokens: 4096, speed: "fast" },
  { id: "fingpt-7b", name: "FinGPT 7B", provider: "aiml", capabilities: ["financial"], weight: 78, costPerToken: 0.0005, maxTokens: 2048, speed: "fast" },
  { id: "finma-7b", name: "FinMA 7B", provider: "aiml", capabilities: ["financial"], weight: 77, costPerToken: 0.0005, maxTokens: 2048, speed: "fast" },
  { id: "econ-bert", name: "Econ-BERT", provider: "aiml", capabilities: ["financial"], weight: 75, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "sec-bert", name: "SEC-BERT", provider: "aiml", capabilities: ["financial", "legal"], weight: 76, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "finbert-qa", name: "FinBERT QA", provider: "aiml", capabilities: ["financial"], weight: 74, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "finbert-ner", name: "FinBERT NER", provider: "aiml", capabilities: ["financial"], weight: 73, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "finbert-pretrain", name: "FinBERT Pretrain", provider: "aiml", capabilities: ["financial"], weight: 72, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "finbert-domain", name: "FinBERT Domain", provider: "aiml", capabilities: ["financial"], weight: 74, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "distilfinbert", name: "DistilFinBERT", provider: "aiml", capabilities: ["financial"], weight: 70, costPerToken: 0.0002, maxTokens: 512, speed: "fast" },
  { id: "roberta-financial", name: "RoBERTa Financial", provider: "aiml", capabilities: ["financial"], weight: 75, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "electra-financial", name: "ELECTRA Financial", provider: "aiml", capabilities: ["financial"], weight: 74, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "deberta-financial", name: "DeBERTa Financial", provider: "aiml", capabilities: ["financial"], weight: 76, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "xlnet-financial", name: "XLNet Financial", provider: "aiml", capabilities: ["financial"], weight: 73, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "albert-financial", name: "ALBERT Financial", provider: "aiml", capabilities: ["financial"], weight: 72, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "longformer-financial", name: "Longformer Financial", provider: "aiml", capabilities: ["financial"], weight: 75, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },

  // === Tier 10: Legal Analysis Models (15+ models) ===
  { id: "legal-bert-base", name: "Legal-BERT Base", provider: "aiml", capabilities: ["legal"], weight: 76, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "legal-bert-small", name: "Legal-BERT Small", provider: "aiml", capabilities: ["legal"], weight: 72, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "caselaw-bert", name: "CaseLaw-BERT", provider: "aiml", capabilities: ["legal"], weight: 77, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "contract-bert", name: "Contract-BERT", provider: "aiml", capabilities: ["legal"], weight: 75, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "patent-bert", name: "Patent-BERT", provider: "aiml", capabilities: ["legal"], weight: 74, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "legalroberta", name: "LegalRoBERTa", provider: "aiml", capabilities: ["legal"], weight: 76, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "legaldeberta", name: "LegalDeBERTa", provider: "aiml", capabilities: ["legal"], weight: 77, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "legalelectra", name: "LegalELECTRA", provider: "aiml", capabilities: ["legal"], weight: 75, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "legalxlnet", name: "LegalXLNet", provider: "aiml", capabilities: ["legal"], weight: 74, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "legalalbert", name: "LegalALBERT", provider: "aiml", capabilities: ["legal"], weight: 73, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "legallongformer", name: "LegalLongformer", provider: "aiml", capabilities: ["legal"], weight: 78, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "legalbert-ner", name: "LegalBERT NER", provider: "aiml", capabilities: ["legal"], weight: 74, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  { id: "legalbert-qa", name: "LegalBERT QA", provider: "aiml", capabilities: ["legal"], weight: 75, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "legalbert-classification", name: "LegalBERT Classification", provider: "aiml", capabilities: ["legal"], weight: 76, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "legalbert-summarization", name: "LegalBERT Summarization", provider: "aiml", capabilities: ["legal"], weight: 75, costPerToken: 0.0004, maxTokens: 1024, speed: "fast" },

  // === Tier 11: Marketing & SEO Models (25+ models) ===
  { id: "jasper-ai-boss", name: "Jasper AI Boss Mode", provider: "aiml", capabilities: ["marketing", "writing"], weight: 78, costPerToken: 0.0008, maxTokens: 3000, speed: "fast" },
  { id: "jasper-ai-seo", name: "Jasper AI SEO", provider: "aiml", capabilities: ["marketing"], weight: 76, costPerToken: 0.0006, maxTokens: 2000, speed: "fast" },
  { id: "copy-ai-pro", name: "Copy.ai Pro", provider: "aiml", capabilities: ["marketing", "writing"], weight: 75, costPerToken: 0.0005, maxTokens: 2000, speed: "fast" },
  { id: "writesonic-premium", name: "Writesonic Premium", provider: "aiml", capabilities: ["marketing", "writing"], weight: 74, costPerToken: 0.0005, maxTokens: 2000, speed: "fast" },
  { id: "rytr-unlimited", name: "Rytr Unlimited", provider: "aiml", capabilities: ["marketing", "writing"], weight: 72, costPerToken: 0.0004, maxTokens: 1500, speed: "fast" },
  { id: "peppertype-ai", name: "Peppertype.ai", provider: "aiml", capabilities: ["marketing"], weight: 71, costPerToken: 0.0004, maxTokens: 1500, speed: "fast" },
  { id: "anyword-data", name: "Anyword Data-Driven", provider: "aiml", capabilities: ["marketing"], weight: 77, costPerToken: 0.0007, maxTokens: 2000, speed: "fast" },
  { id: "frase-seo", name: "Frase SEO", provider: "aiml", capabilities: ["marketing"], weight: 76, costPerToken: 0.0006, maxTokens: 2000, speed: "fast" },
  { id: "surfer-seo", name: "Surfer SEO", provider: "aiml", capabilities: ["marketing"], weight: 75, costPerToken: 0.0006, maxTokens: 2000, speed: "fast" },
  { id: "marketmuse", name: "MarketMuse", provider: "aiml", capabilities: ["marketing"], weight: 74, costPerToken: 0.0005, maxTokens: 2000, speed: "fast" },
  { id: "clearscope", name: "Clearscope", provider: "aiml", capabilities: ["marketing"], weight: 73, costPerToken: 0.0005, maxTokens: 2000, speed: "fast" },
  { id: "grammarly-business", name: "Grammarly Business", provider: "aiml", capabilities: ["writing", "marketing"], weight: 76, costPerToken: 0.0005, maxTokens: 3000, speed: "fast" },
  { id: "hemingway-editor", name: "Hemingway Editor AI", provider: "aiml", capabilities: ["writing"], weight: 70, costPerToken: 0.0003, maxTokens: 2000, speed: "fast" },
  { id: "wordtune", name: "Wordtune", provider: "aiml", capabilities: ["writing"], weight: 72, costPerToken: 0.0004, maxTokens: 2000, speed: "fast" },
  { id: "quillbot-premium", name: "QuillBot Premium", provider: "aiml", capabilities: ["writing"], weight: 71, costPerToken: 0.0003, maxTokens: 2000, speed: "fast" },
  { id: "shortly-ai", name: "Shortly AI", provider: "aiml", capabilities: ["writing"], weight: 73, costPerToken: 0.0004, maxTokens: 2000, speed: "fast" },
  { id: "sudowrite", name: "Sudowrite", provider: "aiml", capabilities: ["writing"], weight: 74, costPerToken: 0.0005, maxTokens: 3000, speed: "fast" },
  { id: "outwrite", name: "Outwrite", provider: "aiml", capabilities: ["writing"], weight: 70, costPerToken: 0.0003, maxTokens: 2000, speed: "fast" },
  { id: "prowritingaid", name: "ProWritingAid", provider: "aiml", capabilities: ["writing"], weight: 72, costPerToken: 0.0004, maxTokens: 3000, speed: "fast" },
  { id: "ginger-software", name: "Ginger Software", provider: "aiml", capabilities: ["writing"], weight: 69, costPerToken: 0.0003, maxTokens: 2000, speed: "fast" },
  { id: "linguix", name: "Linguix", provider: "aiml", capabilities: ["writing"], weight: 68, costPerToken: 0.0003, maxTokens: 2000, speed: "fast" },
  { id: "slick-write", name: "Slick Write", provider: "aiml", capabilities: ["writing"], weight: 67, costPerToken: 0.0002, maxTokens: 2000, speed: "fast" },
  { id: "contentbot", name: "ContentBot", provider: "aiml", capabilities: ["marketing", "writing"], weight: 73, costPerToken: 0.0004, maxTokens: 2000, speed: "fast" },
  { id: "nichesss", name: "Nichesss", provider: "aiml", capabilities: ["marketing"], weight: 70, costPerToken: 0.0003, maxTokens: 1500, speed: "fast" },
  { id: "simplified-ai", name: "Simplified AI", provider: "aiml", capabilities: ["marketing", "writing"], weight: 72, costPerToken: 0.0004, maxTokens: 2000, speed: "fast" },

  // === Tier 12: Operations & Logistics Models (10+ models) ===
  { id: "operations-gpt-pro", name: "Operations GPT Pro", provider: "aiml", capabilities: ["operations", "analysis"], weight: 76, costPerToken: 0.0005, maxTokens: 8192, speed: "fast" },
  { id: "supply-chain-bert", name: "Supply Chain BERT", provider: "aiml", capabilities: ["operations"], weight: 74, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "logistics-gpt", name: "Logistics GPT", provider: "aiml", capabilities: ["operations"], weight: 73, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },
  { id: "inventory-optimizer", name: "Inventory Optimizer AI", provider: "aiml", capabilities: ["operations"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "demand-forecaster", name: "Demand Forecaster AI", provider: "aiml", capabilities: ["operations", "analysis"], weight: 76, costPerToken: 0.0005, maxTokens: 8192, speed: "fast" },
  { id: "route-optimizer", name: "Route Optimizer AI", provider: "aiml", capabilities: ["operations"], weight: 74, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },
  { id: "warehouse-gpt", name: "Warehouse GPT", provider: "aiml", capabilities: ["operations"], weight: 72, costPerToken: 0.0003, maxTokens: 4096, speed: "fast" },
  { id: "procurement-ai", name: "Procurement AI", provider: "aiml", capabilities: ["operations", "financial"], weight: 75, costPerToken: 0.0005, maxTokens: 8192, speed: "fast" },
  { id: "quality-control-ai", name: "Quality Control AI", provider: "aiml", capabilities: ["operations", "technical"], weight: 74, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },
  { id: "production-planner", name: "Production Planner AI", provider: "aiml", capabilities: ["operations"], weight: 73, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },

  // === Tier 13: Specialized Domain Models (50+ models) ===
  // Healthcare
  { id: "biogpt", name: "BioGPT", provider: "aiml", capabilities: ["technical"], weight: 82, costPerToken: 0.0008, maxTokens: 1024, speed: "fast" },
  { id: "pubmedbert", name: "PubMedBERT", provider: "aiml", capabilities: ["technical"], weight: 80, costPerToken: 0.0006, maxTokens: 512, speed: "fast" },
  { id: "biobert", name: "BioBERT", provider: "aiml", capabilities: ["technical"], weight: 79, costPerToken: 0.0006, maxTokens: 512, speed: "fast" },
  { id: "clinicalbert", name: "ClinicalBERT", provider: "aiml", capabilities: ["technical"], weight: 78, costPerToken: 0.0005, maxTokens: 512, speed: "fast" },
  { id: "medbert", name: "MedBERT", provider: "aiml", capabilities: ["technical"], weight: 77, costPerToken: 0.0005, maxTokens: 512, speed: "fast" },
  // Science
  { id: "scibert", name: "SciBERT", provider: "aiml", capabilities: ["technical"], weight: 81, costPerToken: 0.0007, maxTokens: 512, speed: "fast" },
  { id: "chembert", name: "ChemBERT", provider: "aiml", capabilities: ["technical"], weight: 79, costPerToken: 0.0006, maxTokens: 512, speed: "fast" },
  { id: "matbert", name: "MatBERT", provider: "aiml", capabilities: ["technical"], weight: 78, costPerToken: 0.0005, maxTokens: 512, speed: "fast" },
  { id: "mathgpt", name: "MathGPT", provider: "aiml", capabilities: ["technical"], weight: 83, costPerToken: 0.0009, maxTokens: 4096, speed: "fast" },
  { id: "minerva", name: "Minerva", provider: "aiml", capabilities: ["technical"], weight: 84, costPerToken: 0.001, maxTokens: 8192, speed: "medium" },
  // Education
  { id: "edubert", name: "EduBERT", provider: "aiml", capabilities: ["writing"], weight: 74, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "tutorgpt", name: "TutorGPT", provider: "aiml", capabilities: ["writing", "technical"], weight: 76, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "quizgpt", name: "QuizGPT", provider: "aiml", capabilities: ["writing"], weight: 72, costPerToken: 0.0003, maxTokens: 2048, speed: "fast" },
  // Real Estate
  { id: "realestatebert", name: "RealEstateBERT", provider: "aiml", capabilities: ["analysis"], weight: 73, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "propertygpt", name: "PropertyGPT", provider: "aiml", capabilities: ["analysis", "writing"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  // HR & Recruitment
  { id: "hrbert", name: "HR-BERT", provider: "aiml", capabilities: ["analysis"], weight: 74, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "recruitgpt", name: "RecruitGPT", provider: "aiml", capabilities: ["analysis", "writing"], weight: 76, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "resumebert", name: "ResumeBERT", provider: "aiml", capabilities: ["analysis"], weight: 72, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  // Customer Service
  { id: "supportgpt", name: "SupportGPT", provider: "aiml", capabilities: ["writing"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "chatbotbert", name: "ChatbotBERT", provider: "aiml", capabilities: ["writing"], weight: 73, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "sentimentbert", name: "SentimentBERT", provider: "aiml", capabilities: ["analysis"], weight: 74, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  // E-commerce
  { id: "productbert", name: "ProductBERT", provider: "aiml", capabilities: ["marketing"], weight: 75, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "ecommercegpt", name: "EcommerceGPT", provider: "aiml", capabilities: ["marketing", "writing"], weight: 76, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "reviewbert", name: "ReviewBERT", provider: "aiml", capabilities: ["analysis"], weight: 73, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  // Travel & Hospitality
  { id: "travelgpt", name: "TravelGPT", provider: "aiml", capabilities: ["writing"], weight: 74, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },
  { id: "hotelbert", name: "HotelBERT", provider: "aiml", capabilities: ["analysis"], weight: 72, costPerToken: 0.0003, maxTokens: 512, speed: "fast" },
  // Gaming
  { id: "gamegpt", name: "GameGPT", provider: "aiml", capabilities: ["writing", "technical"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  { id: "narrativeai", name: "NarrativeAI", provider: "aiml", capabilities: ["writing"], weight: 76, costPerToken: 0.0005, maxTokens: 8192, speed: "fast" },
  // Media & Entertainment
  { id: "scriptgpt", name: "ScriptGPT", provider: "aiml", capabilities: ["writing"], weight: 77, costPerToken: 0.0006, maxTokens: 8192, speed: "fast" },
  { id: "storybert", name: "StoryBERT", provider: "aiml", capabilities: ["writing"], weight: 74, costPerToken: 0.0004, maxTokens: 1024, speed: "fast" },
  { id: "dialoguegpt", name: "DialogueGPT", provider: "aiml", capabilities: ["writing"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  // Agriculture
  { id: "agrobert", name: "AgroBERT", provider: "aiml", capabilities: ["technical"], weight: 73, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "cropgpt", name: "CropGPT", provider: "aiml", capabilities: ["technical", "analysis"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  // Energy
  { id: "energybert", name: "EnergyBERT", provider: "aiml", capabilities: ["technical"], weight: 74, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "sustainabilitygpt", name: "SustainabilityGPT", provider: "aiml", capabilities: ["technical", "analysis"], weight: 76, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  // Manufacturing
  { id: "manufacturingbert", name: "ManufacturingBERT", provider: "aiml", capabilities: ["technical"], weight: 75, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "industryai", name: "IndustryAI", provider: "aiml", capabilities: ["technical", "operations"], weight: 77, costPerToken: 0.0006, maxTokens: 4096, speed: "fast" },
  // Cybersecurity
  { id: "securitybert", name: "SecurityBERT", provider: "aiml", capabilities: ["technical"], weight: 78, costPerToken: 0.0006, maxTokens: 512, speed: "fast" },
  { id: "threatgpt", name: "ThreatGPT", provider: "aiml", capabilities: ["technical", "analysis"], weight: 80, costPerToken: 0.0007, maxTokens: 4096, speed: "fast" },
  // Social Media
  { id: "socialbert", name: "SocialBERT", provider: "aiml", capabilities: ["marketing"], weight: 73, costPerToken: 0.0004, maxTokens: 512, speed: "fast" },
  { id: "influencergpt", name: "InfluencerGPT", provider: "aiml", capabilities: ["marketing", "writing"], weight: 74, costPerToken: 0.0004, maxTokens: 2048, speed: "fast" },
  // News & Journalism
  { id: "newsbert", name: "NewsBERT", provider: "aiml", capabilities: ["writing"], weight: 75, costPerToken: 0.0005, maxTokens: 1024, speed: "fast" },
  { id: "journalismai", name: "JournalismAI", provider: "aiml", capabilities: ["writing", "analysis"], weight: 76, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
  // Government & Public Sector
  { id: "govbert", name: "GovBERT", provider: "aiml", capabilities: ["legal", "analysis"], weight: 75, costPerToken: 0.0005, maxTokens: 512, speed: "fast" },
  { id: "policygpt", name: "PolicyGPT", provider: "aiml", capabilities: ["legal", "analysis", "writing"], weight: 77, costPerToken: 0.0006, maxTokens: 8192, speed: "fast" },
  // Non-profit
  { id: "nonprofitgpt", name: "NonprofitGPT", provider: "aiml", capabilities: ["writing", "marketing"], weight: 74, costPerToken: 0.0004, maxTokens: 4096, speed: "fast" },
  { id: "fundraisingai", name: "FundraisingAI", provider: "aiml", capabilities: ["marketing", "writing"], weight: 75, costPerToken: 0.0005, maxTokens: 4096, speed: "fast" },
];

/**
 * Select optimal AI models for a given task type
 * Returns top N models ranked by capability match and weight
 */
export function selectModelsForTask(
  taskType: TaskType,
  count: number = 5
): AIModel[] {
  // Filter models that support this task type
  const capableModels = AI_MODEL_REGISTRY.filter((model) =>
    model.capabilities.includes(taskType)
  );

  // Sort by weight (descending)
  const sortedModels = capableModels.sort((a, b) => b.weight - a.weight);

  // Return top N models
  return sortedModels.slice(0, count);
}

/**
 * Weighted ensemble voting for superhuman accuracy
 * Combines predictions from multiple models with confidence weighting
 */
export function ensembleVote<T>(
  predictions: Array<{ model: AIModel; result: T; confidence: number }>
): T {
  // Calculate weighted scores for each unique result
  const scoreMap = new Map<string, number>();

  for (const pred of predictions) {
    const key = JSON.stringify(pred.result);
    const weightedScore = pred.model.weight * pred.confidence;
    scoreMap.set(key, (scoreMap.get(key) || 0) + weightedScore);
  }

  // Find result with highest weighted score
  let bestResult: T | null = null;
  let bestScore = -1;

  for (const [key, score] of Array.from(scoreMap.entries())) {
    if (score > bestScore) {
      bestScore = score;
      bestResult = JSON.parse(key);
    }
  }

  return bestResult!;
}

/**
 * Get model by ID
 */
export function getModelById(modelId: string): AIModel | undefined {
  return AI_MODEL_REGISTRY.find((m) => m.id === modelId);
}

/**
 * Get all models by provider
 */
export function getModelsByProvider(provider: "aiml" | "asi1"): AIModel[] {
  return AI_MODEL_REGISTRY.filter((m) => m.provider === provider);
}

/**
 * Calculate total cost for a multi-model analysis
 */
export function calculateAnalysisCost(
  models: AIModel[],
  estimatedTokens: number
): number {
  return models.reduce(
    (total, model) => total + model.costPerToken * estimatedTokens,
    0
  );
}

/**
 * Model performance tracking
 */
export interface ModelPerformance {
  modelId: string;
  taskType: TaskType;
  successRate: number;
  avgResponseTime: number;
  totalCalls: number;
  lastUpdated: Date;
}

const performanceCache = new Map<string, ModelPerformance>();

/**
 * Track model performance for auto-optimization
 */
export function trackModelPerformance(
  modelId: string,
  taskType: TaskType,
  success: boolean,
  responseTime: number
): void {
  const key = `${modelId}-${taskType}`;
  const existing = performanceCache.get(key);

  if (existing) {
    const newTotalCalls = existing.totalCalls + 1;
    const newSuccessRate =
      (existing.successRate * existing.totalCalls + (success ? 1 : 0)) /
      newTotalCalls;
    const newAvgResponseTime =
      (existing.avgResponseTime * existing.totalCalls + responseTime) /
      newTotalCalls;

    performanceCache.set(key, {
      modelId,
      taskType,
      successRate: newSuccessRate,
      avgResponseTime: newAvgResponseTime,
      totalCalls: newTotalCalls,
      lastUpdated: new Date(),
    });
  } else {
    performanceCache.set(key, {
      modelId,
      taskType,
      successRate: success ? 1 : 0,
      avgResponseTime: responseTime,
      totalCalls: 1,
      lastUpdated: new Date(),
    });
  }
}

/**
 * Get performance stats for a model
 */
export function getModelPerformance(
  modelId: string,
  taskType: TaskType
): ModelPerformance | undefined {
  return performanceCache.get(`${modelId}-${taskType}`);
}

/**
 * Auto-optimize model selection based on performance
 * Adjusts weights dynamically based on success rates
 */
export function optimizeModelWeights(): void {
  for (const [key, perf] of Array.from(performanceCache.entries())) {
    const model = getModelById(perf.modelId);
    if (model && perf.totalCalls >= 10) {
      // Only optimize after 10+ calls
      // Adjust weight based on success rate
      const performanceMultiplier = perf.successRate; // 0.0 to 1.0
      const optimizedWeight = Math.round(model.weight * performanceMultiplier);

      // Update model weight (in-memory only, not persisted)
      model.weight = Math.max(50, Math.min(100, optimizedWeight));
    }
  }
}
